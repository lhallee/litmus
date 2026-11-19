"""Score texts against the reference bank, and choose the bank (variant and size) on validation.

A text's reading against the bank (`calibrate.Readings`): against the bank vectors of each origin, the mean and the
standard deviation of its same-origin pair probabilities; the mean of its `TOP_K` highest cosines to any bank
vector; and its token count. The three means normalized are the three-way reading, and one minus the human share the
two-class reading (ai or mixed against human). `calibrate.readings_from_pair_logits` is the NumPy definition this
GPU code matches.

Two bank variants are compared: (a) the `bank` split, whole held-out clusters, and (b) `bank_draw`, an even draw from
every training cluster, document-disjoint from training. Each is tried at 1,024 to 16,384 texts per origin (a smaller
bank is a prefix of a larger one's random order) and at its full size; the choice is the smallest bank whose
validation PR AUC of the two-class reading is within `--tolerance` of the best, since the browser build ships the bank.
"""

from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch

from pathlib import Path
from typing import Any
from safetensors.numpy import save_file

from .calibrate import TOP_K, Readings, not_human_score
from .embed import read_columns
from .features import FeatureSource, feature_source
from .head import PairHead, pair_logits
from .strata import ORIGINS, write_json, write_summary
from .train import load_labels_and_splits


BANK_SIZES = (1024, 2048, 4096, 8192, 16_384)
VARIANTS = {"a": "bank", "b": "bank_draw"}
ROW_CHUNK = 4096
VECTOR_CHUNK = 65_536
SCORED_SPLITS = ("validation", "test", "calibration")


@torch.no_grad()
def head_vectors(head: PairHead, source: FeatureSource, ids: np.ndarray, device: torch.device) -> torch.Tensor:
    """Each text's unit query and key, in float32 on the GPU: (m,) to (m, 2w)."""
    # ids: (m,)
    head.eval()
    parts = []
    for begin in range(0, len(ids), VECTOR_CHUNK):
        features = torch.from_numpy(source.gather(ids[begin : begin + VECTOR_CHUNK])).to(device)  # (c, f)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            parts.append(head(features).float())  # (c, 2w)

    return torch.cat(parts)  # (m, 2w)


@torch.no_grad()
def bank_readings(vectors: torch.Tensor, banks: list[torch.Tensor], scale: float, bias: float, tokens: np.ndarray,
                  top_k: int = TOP_K) -> Readings:
    """Readings of each text against the three origins' banks: (n, 2w), three (k_o, 2w) to Readings of n texts."""
    # vectors: (n, 2w); banks: (k_o, 2w) for each of three origins; tokens: (n,)
    whole = torch.cat(banks)  # (k, 2w)
    whole_unit = whole / whole.norm(dim=1, keepdim=True)  # (k, 2w)
    mean, spread, closeness = [], [], []
    for begin in range(0, len(vectors), ROW_CHUNK):
        rows = vectors[begin : begin + ROW_CHUNK].float()  # (c, 2w)
        probabilities = [torch.sigmoid(pair_logits(rows, bank, scale, bias)) for bank in banks]  # three (c, k_o)
        mean.append(torch.stack([values.mean(1) for values in probabilities], dim=1))  # (c, 3)
        spread.append(torch.stack([values.std(1, unbiased=False) for values in probabilities], dim=1))  # (c, 3)
        cosines = (rows / rows.norm(dim=1, keepdim=True)) @ whole_unit.T  # (c, k)
        closeness.append(cosines.topk(top_k, dim=1).values.mean(1))  # (c,)

    return Readings(torch.cat(mean).double().cpu().numpy(), torch.cat(spread).double().cpu().numpy(),
                    torch.cat(closeness).double().cpu().numpy(), np.asarray(tokens))


def bank_members(split: np.ndarray, labels: np.ndarray, present: np.ndarray, variant: str, seed: int) -> list[np.ndarray]:
    """Each origin's bank ids in a fixed random order, so a bank of size m is the first m: three (k_o,)."""
    # split, labels, present: (n,)
    rng = np.random.default_rng(seed)

    return [rng.permutation(np.flatnonzero((split == VARIANTS[variant]) & (labels == origin) & present))
            for origin in range(len(ORIGINS))]  # three (k_o,)


def two_class_metrics(readings: Readings, labels: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    """PR AUC, ROC AUC and balanced MCC of the two-class reading, ai or mixed against human."""
    # labels: (n,)
    from foundry.metrics import balanced_mcc
    from foundry.metrics.ranking import binary_ranking_metrics

    score, positive = not_human_score(readings.mean), (labels != 0).astype(np.int64)  # (n,), (n,)
    ranking = binary_ranking_metrics(positive, score)
    mcc = balanced_mcc(positive, score, threshold=threshold)

    return {"pr_auc": ranking["average_precision"], "roc_auc": ranking["auroc"], "balanced_mcc": mcc["balanced_mcc"],
            "threshold": mcc["threshold"]}


def choose_bank(trials: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    """The smallest bank within `tolerance` of the best validation PR AUC; ties go to the higher PR AUC."""
    best = max(trial["pr_auc"] for trial in trials)
    eligible = [trial for trial in trials if trial["pr_auc"] >= best - tolerance]

    return min(eligible, key=lambda trial: (trial["size"], -trial["pr_auc"]))


def chosen_head_name(train_dir: Path) -> str:
    """The directory name under `heads/` of the head the train or search step chose on validation."""
    return json.loads((train_dir / "selection.json").read_text(encoding="utf-8"))["head"]["name"]


DRIFT_PAIRS = (("packed_a", "packed_b"), ("packed_a", "float32"), ("recomputed", "stored"))


def score_drift(head: PairHead, drift_paths: list[Path], banks: list[torch.Tensor], device: torch.device) -> dict[str, Any]:
    """How far the two-class reading moves between two embeddings of the same texts, for each pair a pilot file holds.

    `drift.npz` holds two bfloat16 packings and float32; `reuse_drift.npz` the first corpus's stored vectors and their
    recomputation, and `reuse_drift_aligned.npz` the held-out pairs through the pilot's alignment. Each key is
    `<file>:<pair>`. `reuse_score_check` applies the vector rule to the reading: a reuse pair fails when its mean
    absolute change exceeds twice that of two packings, or the size of its mean signed change exceeds the larger of
    two packings' own and half their mean absolute change (a direction the head would learn as an origin shortcut).
    Where the file also holds float32, "two packings" is the larger of that pair and bfloat16 against float32.
    """
    # banks: (k_o, 2w) for each of three origins
    report: dict[str, Any] = {}
    for path in drift_paths:
        arrays = np.load(path)
        scores = {}
        for name in {name for pair in DRIFT_PAIRS for name in pair} & set(arrays.files):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vectors = head(torch.from_numpy(arrays[name]).to(device)).float()  # (m, 2w)
            readings = bank_readings(vectors, banks, head.scale.item(), head.bias.item(),
                                     np.zeros(len(vectors), dtype=np.int64))
            scores[name] = not_human_score(readings.mean)  # (m,)
        for first, second in DRIFT_PAIRS:
            if first in scores and second in scores:
                change = scores[first] - scores[second]  # (m,)
                report[f"{path.stem}:{first}_vs_{second}"] = {"mean_abs": float(np.abs(change).mean()),
                                                              "mean_signed": float(change.mean()),
                                                              "p99_abs": float(np.quantile(np.abs(change), 0.99)),
                                                              "max_abs": float(np.abs(change).max())}
    packings = report.get("drift:packed_a_vs_packed_b")
    arithmetic = report.get("drift:packed_a_vs_float32")
    if packings is not None and arithmetic is not None:
        # The yardstick is how far bfloat16 arithmetic moves a reading: two packings when packed; padded, two
        # batches agree exactly, so the larger of that and bfloat16 against float32 (as the embed pilot's gate).
        packings = {key: max(abs(packings[key]), abs(arithmetic[key])) for key in ("mean_abs", "mean_signed")}
        report["reuse_score_yardstick"] = packings
    if packings is not None:
        report["reuse_score_check"] = {
            key: {"passes": value["mean_abs"] <= 2 * packings["mean_abs"]
                  and abs(value["mean_signed"]) <= max(abs(packings["mean_signed"]), 0.5 * packings["mean_abs"])}
            for key, value in report.items() if key.endswith("recomputed_vs_stored")}

    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True, help="the train step's output (selection.json, heads/)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head", help="a head's directory name under heads/; default the one selection.json chose")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="skip the choice: use this variant")
    parser.add_argument("--size", type=int, help="skip the choice: texts per origin (0 for the whole variant)")
    parser.add_argument("--tolerance", type=float, default=0.002, help="PR AUC a smaller bank may give up")
    parser.add_argument("--drift", type=Path, nargs="*", default=[],
                        help="drift.npz and reuse_drift.npz from the embed pilot of the head's encoder")
    parser.add_argument("--seed", type=int, default=0)


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    device = torch.device("cuda")
    head_name = args.head or chosen_head_name(args.train_dir)
    record = json.loads((args.train_dir / "heads" / head_name / "done.json").read_text(encoding="utf-8"))
    head = PairHead.load(args.train_dir / "heads" / head_name / "head").to(device).eval()
    source = feature_source(record["features"], args.embedding_root, record["seed"])
    labels, split = load_labels_and_splits(args.corpus, args.split)
    tokens = read_columns(args.corpus, ["id", "tokens"]).sort_by("id")["tokens"].to_numpy()  # (n,)
    present = source.embedded(len(labels))  # (n,)
    scale, bias = head.scale.item(), head.bias.item()

    validation_ids = np.flatnonzero((split == "validation") & present)  # (n_v,)
    validation_vectors = head_vectors(head, source, validation_ids, device)  # (n_v, 2w)
    trials = []
    variants = [args.variant] if args.variant else sorted(VARIANTS)
    for variant in variants:
        members = bank_members(split, labels, present, variant, args.seed)
        vectors = [head_vectors(head, source, ids, device) for ids in members]  # three (k_o, 2w)
        largest = min(len(ids) for ids in members)
        sizes = [args.size or largest] if args.size is not None else [*(s for s in BANK_SIZES if s < largest), largest]
        for size in sizes:
            readings = bank_readings(validation_vectors, [bank[:size] for bank in vectors], scale, bias, tokens[validation_ids])
            trials.append({"variant": variant, "size": size, **two_class_metrics(readings, labels[validation_ids])})
    chosen = choose_bank(trials, args.tolerance)

    members = bank_members(split, labels, present, chosen["variant"], args.seed)
    bank_ids = [ids[: chosen["size"]] for ids in members]
    banks = [head_vectors(head, source, ids, device) for ids in bank_ids]  # three (size, 2w)
    args.out.mkdir(parents=True, exist_ok=True)
    save_file({name: bank.cpu().numpy().astype(np.float16) for name, bank in zip(ORIGINS, banks, strict=True)},
              str(args.out / "bank.safetensors"))
    np.savez(args.out / "bank_ids.npz", **{name: ids for name, ids in zip(ORIGINS, bank_ids, strict=True)})
    for name in SCORED_SPLITS:
        ids = np.flatnonzero((split == name) & present)  # (m,)
        readings = bank_readings(head_vectors(head, source, ids, device), banks, scale, bias, tokens[ids])
        readings.save(args.out / f"readings_{name}.npz", ids=ids, labels=labels[ids])
    drift = score_drift(head, args.drift, banks, device) if args.drift else None
    report = {"head": head_name, "features": record["features"], "trials": trials, "chosen": chosen,
              "bank_bytes_float16": int(sum(bank.numel() for bank in banks) * 2), "score_drift": drift}
    write_json(args.out / "bank_selection.json", report)
    write_summary(args.out, "bank", {**report, "seconds": time.perf_counter() - started})
