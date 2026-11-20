"""Proof of concept: can a ternary local-attention student, distilled as designed, get close to its teacher?

    python -m litmus_student poc --corpus CORPUS_DIR --embeddings modernbert_base_mean_variance.npy \\
        --checker CHECKER_DIR --out DIR --experiment-id YYYY-MM-DD_slug [--minutes 60] [--rows 1500000]

It needs no new corpus or teacher, so it answers early whether the ternary student can work. It reads the first
corpus (`lhallee/many-generator-detection-corpus`, `data/{train,validation}-*.parquet`, two
origins, 0 human and 1 ai), its stored ModernBERT-base embeddings (one float16 array in `id` order), and the released
pair head of `lhallee/ai-vs-human-checker` (a snapshot folder with `pairhead.json`, `pairhead.safetensors`,
`bank.safetensors` and its own code, `fphunt/pairhead.py`, imported from the folder as the checker itself does).

1. **Tokenize** a class-balanced draw of `--rows` training passages and every validation passage into the token
   store of `data.py`, in the standard reading (SPEC.md, section 1), with `--workers` processes; the rate is
   reported. Passages over a chunk (8,192 tokens) are dropped and counted.
2. **Teacher files**: the released head run over the stored embeddings of those rows gives every row's teacher
   query and key; the manifest maps store rows to embedding rows, so the 16 GB array is read in place.
3. **Train three arms on the same batches** for the same steps, one after another inside each step: the ternary
   student at the smallest size of the sweep, the ternary student at the middle size, and the full-precision twin of
   the smallest (`StudentModel(quantized=False)`: float weights and activations, the same architecture). The ternary
   arms use the BitNet schedule of `train.py`; the twin a single linear decay from `--twin-lr` (1e-3), since
   BitNet's high peak is a quantization recipe and would handicap a float model. The run is fitted to `--minutes`:
   ten timed steps and one evaluation set the number of steps, so the learning rate schedules end with the box.
4. **Read** every `--eval-minutes`: each arm and the teacher on all validation passages through a bank of
   `--bank-per-origin` training passages of each origin, the detector score being the card's (coverage of the ai
   bank less coverage of the human bank): PR AUC, ROC AUC and balanced MCC. The teacher is also read through its
   released bank, which should reproduce the card's validation numbers (PR AUC 0.990, ROC AUC 0.989, MCC 0.898).

**The verdict**, by rules fixed before the run, on validation PR AUC at the end, the
teacher read through the same bank rows as the students:
- `go`: the best ternary arm is within 0.02 of the teacher and within 0.01 of the full-precision twin;
- `ternary is the problem`: the twin is within 0.02 of the teacher and the small ternary arm, its quantized twin,
  is not within 0.01 of it;
- `the student is the problem`: the twin itself is more than 0.02 short of the teacher (capacity, local
  attention, or the distillation);
- `inconclusive` otherwise. Each arm's curve is also marked `rising` when its PR AUC gained more than 0.002 over the
  last quarter of the run (at least the last ten minutes), since a short arm still climbing is another finding.

Writes `poc_report.json` and prints the table; logs the curves to W&B (project `litmus`, job type `poc`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from torch import Tensor, nn

from foundry.logging import Run
from foundry.metrics import balanced_mcc
from foundry.metrics.ranking import binary_ranking_metrics
from foundry.training import clip_grad_norm, set_seed
from .contract import reference
from .data import ClusterSampler, TeacherFiles, TokenStore, cluster_stratified_rows, prefetched, write_store
from .dense import TokenTeacher, load_teacher_encoder
from .distill import LossWeights, Projections
from .model import StudentModel
from .train import bank_coverage, learning_rate, model_vectors, parameter_groups, step_maker, train_step, weight_decay


PROJECT = "litmus"
ORIGINS = (0, 1)  # the first corpus: human, ai
SPLITS = ["train", "validation"]
CARD = {"pr_auc": 0.990, "roc_auc": 0.989, "mcc": 0.898}  # lhallee/ai-vs-human-checker README, validation, 10,000 passages
CARD_TOLERANCE = 0.005
GO_TEACHER_GAP = 0.02
GO_TWIN_GAP = 0.01
RISING_GAIN = 0.002
PROBE_STEPS = 10
STORE = "store"
TEACHER = "teacher"
REPORT = "poc_report.json"


# ---------------------------------------------------------------- 1. tokenization


def split_files(corpus: Path, split: str) -> list[Path]:
    files = sorted((corpus / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {corpus / 'data'}/{split}-*.parquet")

    return files


def selected_rows(corpus: Path, rows: int, generator: np.random.Generator) -> dict[str, np.ndarray]:
    """A class-balanced uniform draw of `rows` training passages and every validation passage, ascending by id:
    `id`, `label`, `cluster`, `split` (codes into SPLITS), each (n,)."""
    parts = []
    for code, split in enumerate(SPLITS):
        table = pq.read_table(split_files(corpus, split), columns=["id", "label", "cluster"])
        ids, labels, clusters = (table[name].to_numpy() for name in ("id", "label", "cluster"))  # (m,) each
        keep = np.arange(len(ids))  # (m,)
        if split == "train":
            keep = np.concatenate([generator.permutation(np.flatnonzero(labels == origin))[: rows // len(ORIGINS)] for origin in ORIGINS])
        parts.append((ids[keep], labels[keep], clusters[keep], np.full(len(keep), code)))
    ids, labels, clusters, splits = (np.concatenate(values) for values in zip(*parts, strict=True))
    order = np.argsort(ids, kind="stable")  # (n,)
    if (np.diff(ids[order]) <= 0).any():
        raise ValueError("passage ids repeat across the corpus's splits")

    return {"id": ids[order].astype(np.int64), "label": labels[order].astype(np.int64), "cluster": clusters[order].astype(np.int64),
            "split": splits[order].astype(np.int64)}  # (n,) each


_WANTED = np.zeros(0, dtype=np.int64)
_TOKENIZER: Any = None


def _start_worker(wanted: np.ndarray, tokenizer_path: Path | None) -> None:
    # wanted: (n,) the drawn passage ids
    global _WANTED, _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # one process per core already
    _WANTED = wanted
    _TOKENIZER = reference.load_tokenizer(tokenizer_path)


def _tokenize_group(task: tuple[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row group's wanted passages, standard-read and tokenized: ids (k,), lengths (k,), tokens (sum,) uint16."""
    path, group = task
    table = pq.ParquetFile(path).read_row_group(group, columns=["id", "text"])
    ids = table["id"].to_numpy()  # (r,)
    keep = np.flatnonzero(np.isin(ids, _WANTED))  # (k,)
    texts = [reference.standard_reading(text) for text in pc.take(table["text"], keep).to_pylist()]
    encodings = _TOKENIZER.encode_batch(texts, add_special_tokens=False)
    lengths = np.asarray([len(encoding.ids) for encoding in encodings], dtype=np.int64)  # (k,)
    tokens = np.fromiter((token for encoding in encodings for token in encoding.ids), dtype=np.uint16, count=int(lengths.sum()))  # (sum,)

    return ids[keep].astype(np.int64), lengths, tokens  # (k,), (k,), (sum,)


def tokenize(corpus: Path, out: Path, rows: int, workers: int, chunk_tokens: int, tokenizer_path: Path | None,
             generator: np.random.Generator) -> dict[str, Any]:
    """The pre-tokenized store of the drawn passages (`data.py`'s format) with `corpus_ids.npy`; returns the rate."""
    started = time.perf_counter()
    chosen = selected_rows(corpus, rows, generator)
    tasks = [(str(path), group) for split in SPLITS for path in split_files(corpus, split)
             for group in range(pq.ParquetFile(path).num_row_groups)]
    texts: dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_start_worker, initargs=(chosen["id"], tokenizer_path)) as pool:
        for ids, lengths, tokens in pool.map(_tokenize_group, tasks, chunksize=1):
            for passage, piece in zip(ids.tolist(), np.split(tokens, np.cumsum(lengths)[:-1]), strict=True):
                texts[passage] = piece
    seconds = time.perf_counter() - started
    lengths = np.asarray([len(texts.get(int(passage), ())) for passage in chosen["id"]], dtype=np.int64)  # (n,)
    keep = (lengths > 0) & (lengths <= chunk_tokens)  # (n,)
    if len(texts) != len(chosen["id"]):
        raise ValueError(f"{len(chosen['id']) - len(texts)} drawn passages were not found in the parquet files")
    write_store(out, [texts[int(passage)] for passage in chosen["id"][keep]], chosen["label"][keep], chosen["split"][keep],
                chosen["cluster"][keep], SPLITS)
    np.save(out / "corpus_ids.npy", chosen["id"][keep])
    total = int(lengths[keep].sum())

    return {"passages": int(keep.sum()), "dropped_empty": int((lengths == 0).sum()), "dropped_over_chunk": int((lengths > chunk_tokens).sum()),
            "tokens": total, "seconds": seconds, "tokens_per_second": total / seconds, "passages_per_second": len(texts) / seconds,
            "workers": workers}


# ---------------------------------------------------------------- 2. teacher files


def load_pair_head(checker: Path) -> nn.Module:
    """The released pair head, by its own code in the snapshot folder (never copied here)."""
    if str(checker) not in sys.path:
        sys.path.insert(0, str(checker))
    pairhead = importlib.import_module("fphunt.pairhead")

    return pairhead.PairHead.load(checker).eval()


@torch.no_grad()
def teacher_files(store_directory: Path, embeddings_path: Path, checker: Path, out: Path, device: torch.device,
                  batch_rows: int = 65536) -> dict[str, float]:
    """The manifest `TeacherFiles` reads: the stored embeddings in place, and the released head's query and key of
    every store row; returns the head's scale and bias."""
    corpus_ids = np.load(store_directory / "corpus_ids.npy")  # (n,) ascending
    head = load_pair_head(checker).to(device)
    embeddings = np.load(embeddings_path, mmap_mode="r")  # (N, 2D)
    out.mkdir(parents=True, exist_ok=True)
    width = int(head.config["output_width"])  # 2w_T
    query_key = np.lib.format.open_memmap(out / "query_key.npy", mode="w+", dtype=np.float16, shape=(len(corpus_ids), width))  # (n, 2w_T)
    for first in range(0, len(corpus_ids), batch_rows):
        rows = corpus_ids[first : first + batch_rows]  # (r,)
        vectors = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)  # (r, 2D)
        query_key[first : first + len(rows)] = head(vectors).half().cpu().numpy()
    query_key.flush()
    np.save(out / "present.npy", np.ones(len(corpus_ids), dtype=bool))
    np.save(out / "embedding_rows.npy", corpus_ids)
    scale, bias = float(head.scale), float(head.bias)
    manifest = {"head": {"scale": scale, "bias": bias, "checker": str(checker)}, "embeddings": [{"path": str(embeddings_path.resolve())}],
                "embedding_rows": "embedding_rows.npy", "query_key": {"path": "query_key.npy", "present": "present.npy"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    return {"scale": scale, "bias": bias}


# ---------------------------------------------------------------- 3. arms


@dataclass
class Arm:
    name: str
    config_name: str
    model: StudentModel
    projections: Projections
    optimizer: torch.optim.Optimizer
    trainable: nn.Module
    first_peak: float
    second_peak: float
    decay_switches: bool  # BitNet's weight decay drops to zero at half the run; the twin keeps it
    token_weight: float = 0.0  # the per-token loss's weight (`dense.py`); 0 trains on the per-text and pair losses alone
    seconds: list[float] = field(default_factory=list)  # each step's time
    curve: list[dict[str, float]] = field(default_factory=list)


def make_arm(name: str, config_name: str, quantized: bool, arguments: argparse.Namespace, teacher: TeacherFiles,
             device: torch.device, token_weight: float = 0.0, token_width: int = 0, integer_attention: bool = False) -> Arm:
    config = reference.named_config(config_name)
    model = StudentModel(config, quantized, integer_attention).to(device)
    projections = Projections(2 * config.width, teacher.embedding_width, config.head_out, teacher.query_key.shape[1] // 2,
                              token_widths=(config.width, token_width) if token_weight else None).to(device)
    trainable = nn.ModuleDict({"model": model, "projections": projections})
    optimizer = torch.optim.AdamW(parameter_groups(trainable, arguments.weight_decay), lr=arguments.lr, betas=(0.9, 0.95),
                                  fused=device.type == "cuda")
    peaks = (arguments.lr, arguments.lr_second) if quantized else (arguments.twin_lr, arguments.twin_lr)

    return Arm(name, config_name, model, projections, optimizer, trainable, *peaks, decay_switches=quantized, token_weight=token_weight)


def arm_step(arm: Arm, step: Any, index: int, total: int, arguments: argparse.Namespace, teacher: TeacherFiles,
             weights: LossWeights, device: torch.device, token_targets: list[Tensor] | None = None) -> dict[str, float]:
    for group in arm.optimizer.param_groups:
        group["lr"] = learning_rate(index, total, arguments.warmup, arm.first_peak, arm.second_peak)
        decay = weight_decay(index, total, arguments.weight_decay) if arm.decay_switches else arguments.weight_decay
        group["weight_decay"] = decay if group["decays"] else 0.0
    started = time.perf_counter()
    dense = token_targets if arm.token_weight else None
    losses = train_step(arm.model, arm.projections, step, teacher, weights, device, dense, arm.token_weight)
    if not math.isfinite(losses["total"]):
        raise FloatingPointError(f"{arm.name}, step {index}: the loss is {losses['total']}")
    clip_grad_norm(arm.trainable, arguments.clip)
    arm.optimizer.step()
    arm.optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    arm.seconds.append(time.perf_counter() - started)

    return losses


# ---------------------------------------------------------------- 4. reading


def detection(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """PR AUC, ROC AUC and balanced MCC (at its best threshold, as the card chose its threshold on validation)."""
    # labels, score: (n,)
    ranking = binary_ranking_metrics(labels, score)

    return {"pr_auc": float(ranking["average_precision"]), "roc_auc": float(ranking["auroc"]),
            "mcc": float(balanced_mcc(labels, score)["balanced_mcc"])}


def card_score(vectors: Tensor, bank: Tensor, bank_labels: Tensor, scale: float, bias: float) -> np.ndarray:
    """The card's detector: coverage of the ai bank less coverage of the human bank, (n, 2w) to (n,)."""
    # vectors: (n, 2w); bank: (m, 2w); bank_labels: (m,)
    coverage = bank_coverage(vectors, bank, bank_labels, scale, bias, len(ORIGINS))  # (n, 2)

    return (coverage[:, 1] - coverage[:, 0]).float().cpu().numpy()  # (n,)


class Reading:
    """Validation passages and a bank of training passages, fixed for the run, and the teacher's numbers on them."""

    def __init__(self, store: TokenStore, teacher: TeacherFiles, checker: Path, arguments: argparse.Namespace,
                 generator: np.random.Generator, device: torch.device) -> None:
        self.store, self.device, self.micro_tokens = store, device, arguments.micro_tokens
        self.validation = np.flatnonzero(store.split_mask("validation"))  # (v,)
        self.bank = np.sort(cluster_stratified_rows(store, "train", arguments.bank_per_origin, generator, ORIGINS))  # (m,)
        self.labels = store.label[self.validation].astype(np.int64)  # (v,)
        self.bank_labels = torch.from_numpy(store.label[self.bank].astype(np.int64)).to(device)  # (m,)

        def vectors(rows: np.ndarray) -> Tensor:
            # rows: (r,)
            return torch.from_numpy(np.asarray(teacher.query_key[rows], dtype=np.float32)).to(device)  # (r, 2w_T)

        validation_vectors = vectors(self.validation)  # (v, 2w_T)
        self.teacher = detection(self.labels, card_score(validation_vectors, vectors(self.bank), self.bank_labels, teacher.scale, teacher.bias))
        self.teacher_released: dict[str, float] | None = None
        if (checker / "bank.safetensors").exists():
            from safetensors.torch import load_file

            released = load_file(str(checker / "bank.safetensors"))
            bank = torch.cat([released["human"], released["ai"]]).float().to(device)  # (2k, 2w_T)
            labels = torch.cat([torch.zeros(len(released["human"])), torch.ones(len(released["ai"]))]).long().to(device)  # (2k,)
            self.teacher_released = detection(self.labels, card_score(validation_vectors, bank, labels, teacher.scale, teacher.bias))

    def __call__(self, arm: Arm) -> dict[str, float]:
        model = arm.model
        validation = model_vectors(model, self.store, self.validation, self.micro_tokens, self.device)  # (v, 2w)
        bank = model_vectors(model, self.store, self.bank, self.micro_tokens, self.device)  # (m, 2w)

        scale, bias = float(model.head.scale.detach()), float(model.head.bias.detach())

        return detection(self.labels, card_score(validation, bank, self.bank_labels, scale, bias))


# ---------------------------------------------------------------- the verdict


def rising(curve: Sequence[dict[str, float]]) -> bool:
    """PR AUC gained more than RISING_GAIN over the last quarter of the run, and at least its last ten minutes."""
    if len(curve) < 2:
        return False
    end = curve[-1]["minutes"]
    since = end - max(10.0, 0.25 * end)
    earlier = min(curve[:-1], key=lambda point: abs(point["minutes"] - since))

    return curve[-1]["pr_auc"] - earlier["pr_auc"] > RISING_GAIN


def verdict(final: dict[str, float], teacher: float) -> dict[str, Any]:
    """The rules fixed before the run, on validation PR AUC: `final` maps arm names to theirs."""
    ternary = {name: value for name, value in final.items() if name.startswith("ternary")}
    best = max(ternary, key=ternary.__getitem__)
    twin, small = final["twin_small"], final["ternary_small"]
    gaps = {"best_ternary": best, "teacher_minus_best_ternary": teacher - ternary[best], "twin_minus_best_ternary": twin - ternary[best],
            "teacher_minus_twin": teacher - twin, "twin_minus_small_ternary": twin - small}
    if gaps["teacher_minus_best_ternary"] <= GO_TEACHER_GAP and gaps["twin_minus_best_ternary"] <= GO_TWIN_GAP:
        call = "go"
    elif gaps["teacher_minus_twin"] > GO_TEACHER_GAP:
        call = "the student is the problem"
    elif gaps["twin_minus_small_ternary"] > GO_TWIN_GAP:
        call = "ternary is the problem"
    else:
        call = "inconclusive"

    return {"call": call, **gaps}


# ---------------------------------------------------------------- the run


def planned_steps(arguments: argparse.Namespace, elapsed: float, step_seconds: float, eval_seconds: float) -> int:
    """Steps that fit the rest of the box, with an evaluation every `--eval-minutes` and one at the end."""
    remaining = arguments.minutes * 60.0 - elapsed
    evaluations = math.ceil(remaining / (arguments.eval_minutes * 60.0)) + 1
    budget = remaining - evaluations * eval_seconds

    return max(PROBE_STEPS + 1, PROBE_STEPS + int(budget / step_seconds))


def table(report: dict[str, Any]) -> str:
    lines = ["| Reader | PR AUC | ROC AUC | MCC | Tokens/s (train) | Rising |", "| --- | --- | --- | --- | --- | --- |"]
    for name, arm in report["arms"].items():
        last = arm["curve"][-1]
        lines.append(f"| {name} ({arm['config']}) | {last['pr_auc']:.4f} | {last['roc_auc']:.4f} | {last['mcc']:.4f} | "
                     f"{arm['tokens_per_second']:,.0f} | {'yes' if arm['rising'] else 'no'} |")
    teacher = report["teacher"]
    lines.append(f"| teacher, same bank | {teacher['same_bank']['pr_auc']:.4f} | {teacher['same_bank']['roc_auc']:.4f} | {teacher['same_bank']['mcc']:.4f} | | |")
    if teacher["released_bank"] is not None:
        released = teacher["released_bank"]
        lines.append(f"| teacher, released bank | {released['pr_auc']:.4f} | {released['roc_auc']:.4f} | {released['mcc']:.4f} | | |")
    lines.append(f"| card (validation) | {CARD['pr_auc']:.3f} | {CARD['roc_auc']:.3f} | {CARD['mcc']:.3f} | | |")

    return "\n".join(lines) + f"\n\nVerdict: {report['verdict']['call']}"


def poc(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    set_seed(arguments.seed)
    device = torch.device(arguments.device)
    generator = np.random.default_rng(arguments.seed)
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    configs = {name: reference.named_config(name) for name in (arguments.small, arguments.middle)}
    if len({config.block_size for config in configs.values()}) != 1 or len({config.chunk_tokens for config in configs.values()}) != 1:
        raise ValueError("the arms share their batches, so their block sizes and chunks must agree")
    chunk_tokens, block_size = configs[arguments.small].chunk_tokens, configs[arguments.small].block_size
    report: dict[str, Any] = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}}
    if not (out / STORE / "meta.json").exists():
        report["tokenization"] = tokenize(arguments.corpus, out / STORE, arguments.rows, arguments.workers, chunk_tokens, arguments.tokenizer, generator)
        print(f"tokenized: {json.dumps(report['tokenization'])}", file=sys.stderr)
    if not (out / TEACHER / "manifest.json").exists():
        report["teacher_head"] = teacher_files(out / STORE, arguments.embeddings, arguments.checker, out / TEACHER, device)
    store = TokenStore(out / STORE)
    teacher = TeacherFiles.load(out / TEACHER, np.flatnonzero(store.split_mask("train")), generator)
    reading = Reading(store, teacher, arguments.checker, arguments, generator, device)
    sampler = ClusterSampler(store, "train", chunk_tokens, teacher.present, origins=ORIGINS)
    token_teacher: TokenTeacher | None = None
    if arguments.dense_weight:
        # Whether a target at every token teaches the same student more per text: the same ternary student twice on
        # the same batches, with and without it.
        encoder = load_teacher_encoder(device)
        token_teacher = TokenTeacher(encoder, encoder.config.hidden_size).to(device)
        arms = [make_arm("ternary_small", arguments.small, True, arguments, teacher, device),
                make_arm("ternary_small_dense", arguments.small, True, arguments, teacher, device, arguments.dense_weight, token_teacher.width)]
    elif arguments.integer_attention:
        # The third question: what does attention whose two products are integer products cost? The same ternary
        # student twice on the same batches, with the specified float attention and with the integer prototype.
        arms = [make_arm("ternary_small", arguments.small, True, arguments, teacher, device),
                make_arm("ternary_small_integer", arguments.small, True, arguments, teacher, device, integer_attention=True)]
    else:
        arms = [make_arm("ternary_small", arguments.small, True, arguments, teacher, device),
                make_arm("ternary_middle", arguments.middle, True, arguments, teacher, device),
                make_arm("twin_small", arguments.small, False, arguments, teacher, device)]
    teacher_seconds: list[float] = []
    weights = LossWeights()
    run_config = {**report["arguments"], "teacher_scale": teacher.scale, "teacher_bias": teacher.bias, "experiment_id": arguments.experiment_id}
    box_started = time.perf_counter()
    total = arguments.steps or 10**9  # set from the probe when the box sets it
    tokens: list[int] = []
    kind = "poc_dense" if token_teacher is not None else "poc_integer_attention" if arguments.integer_attention else "poc"
    # One W&B folder per kind: a comparison run in the proof of concept's folder would resume the proof of concept's run.
    with Run(out / ("wandb" if kind == "poc" else f"wandb_{kind}"), run_config, project=PROJECT, group=arguments.experiment_id, job_type=kind,
             name=f"{arguments.experiment_id}_{kind}", pilot=arguments.pilot) as run:

        def evaluate(index: int) -> float:
            evaluation_started = time.perf_counter()
            minutes = (time.perf_counter() - box_started) / 60.0
            logged = {}
            for arm in arms:
                point = {"step": index, "minutes": minutes, **reading(arm)}
                arm.curve.append(point)
                logged.update({f"{arm.name}/{metric}": point[metric] for metric in ("pr_auc", "roc_auc", "mcc")})
            run.log(logged, step=index)
            print(json.dumps({"step": index, "minutes": round(minutes, 2), **{key: round(value, 4) for key, value in logged.items()}}), file=sys.stderr)
            return time.perf_counter() - evaluation_started

        make = step_maker(sampler, store, teacher, arguments, block_size, generator)
        last_evaluation = time.perf_counter()
        trained = -1  # the last step every arm took
        for index, step in enumerate(prefetched(make, 10**9, depth=3, pin=device.type == "cuda")):
            if index >= total:
                break
            trained = index
            token_targets = None
            if token_teacher is not None:
                teacher_started = time.perf_counter()
                token_targets = [token_teacher.targets(batch.to(device, non_blocking=True)) for batch in step.micro_batches]
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                teacher_seconds.append(time.perf_counter() - teacher_started)
            losses = {arm.name: arm_step(arm, step, index, total, arguments, teacher, weights, device, token_targets)["total"] for arm in arms}
            tokens.append(step.tokens)
            if index % 50 == 0:
                run.log({f"{name}/loss": value for name, value in losses.items()}, step=index)
            if index + 1 == PROBE_STEPS and not arguments.steps:
                eval_seconds = evaluate(index)
                last_evaluation = time.perf_counter()
                step_seconds = float(np.median([sum(times) for times in zip(*(arm.seconds[2:] for arm in arms), strict=True)]))
                step_seconds += float(np.median(teacher_seconds[2:])) if teacher_seconds else 0.0
                total = planned_steps(arguments, time.perf_counter() - box_started, step_seconds, eval_seconds)
                print(f"probe: {step_seconds:.2f} s a step for all arms, {eval_seconds:.0f} s an evaluation: {total} steps", file=sys.stderr)
            elif arguments.eval_steps and (index + 1) % arguments.eval_steps == 0:
                evaluate(index)
            elif not arguments.eval_steps and time.perf_counter() - last_evaluation >= arguments.eval_minutes * 60.0:
                evaluate(index)
                last_evaluation = time.perf_counter()
            if not arguments.steps and time.perf_counter() - box_started > arguments.minutes * 60.0:
                break
        if trained < 0:
            raise RuntimeError("the box ended before the first step")
        if not arms[0].curve or arms[0].curve[-1]["step"] != trained:
            evaluate(trained)
        run.finish(summary={f"{arm.name}/final_pr_auc": arm.curve[-1]["pr_auc"] for arm in arms})

    steps_run = len(tokens)
    tokens_per_step = float(np.mean(tokens)) if tokens else 0.0
    report["steps"] = steps_run
    report["texts_per_step"] = arguments.texts_per_step
    report["tokens_per_step"] = tokens_per_step
    report["arms"] = {arm.name: {"config": arm.config_name, "quantized": arm.model.quantized,
                                 "parameters": sum(parameter.numel() for parameter in arm.model.parameters()),
                                 "tokens_per_second": tokens_per_step / float(np.median(arm.seconds[2:] or arm.seconds)),
                                 "rising": rising(arm.curve), "curve": arm.curve} for arm in arms}
    report["teacher"] = {"same_bank": reading.teacher, "released_bank": reading.teacher_released, "card": CARD,
                         "card_reproduced": None if reading.teacher_released is None else
                         all(abs(reading.teacher_released[key] - CARD[key]) <= CARD_TOLERANCE for key in CARD),
                         "validation_passages": len(reading.validation), "bank_per_origin": arguments.bank_per_origin}
    final = {arm.name: arm.curve[-1]["pr_auc"] for arm in arms}
    if arguments.integer_attention and token_teacher is None:
        report["verdict"] = {"call": "integer attention comparison",
                             "integer_minus_float": final["ternary_small_integer"] - final["ternary_small"],
                             "teacher_minus_integer": reading.teacher["pr_auc"] - final["ternary_small_integer"]}
    elif token_teacher is None:
        report["verdict"] = verdict(final, reading.teacher["pr_auc"])
    else:
        seconds = float(np.median(teacher_seconds[2:] or teacher_seconds))
        report["token_teacher"] = {"weight": arguments.dense_weight, "seconds_per_step": seconds, "tokens_per_second": tokens_per_step / seconds}
        report["verdict"] = {"call": "dense comparison", "dense_minus_plain": final["ternary_small_dense"] - final["ternary_small"],
                             "teacher_minus_dense": reading.teacher["pr_auc"] - final["ternary_small_dense"]}
    report["minutes"] = (time.perf_counter() - started) / 60.0
    (out / arguments.report).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(table(report), file=sys.stderr)

    return report


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m litmus_student poc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="the first corpus's folder, holding data/{train,validation}-*.parquet")
    parser.add_argument("--embeddings", type=Path, required=True, help="embeddings/modernbert_base_mean_variance.npy")
    parser.add_argument("--checker", type=Path, required=True, help="the snapshot folder of lhallee/ai-vs-human-checker")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True, help="YYYY-MM-DD_slug, the W&B group")
    parser.add_argument("--minutes", type=float, default=60.0, help="the training box, evaluations included")
    parser.add_argument("--rows", type=int, default=1_500_000, help="training passages drawn, half of each origin")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="tokenization processes")
    parser.add_argument("--tokenizer", type=Path, default=None, help="a local tokenizer.json; default: the pinned Hub revision")
    parser.add_argument("--small", default="s192_l4", help="the smallest size of the sweep, and its twin's")
    parser.add_argument("--middle", default="s256_l6", help="the middle size")
    parser.add_argument("--texts-per-step", type=int, default=1024)
    parser.add_argument("--micro-tokens", type=int, default=65536)
    parser.add_argument("--pack-multiple", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--lr-second", type=float, default=1e-3)
    parser.add_argument("--twin-lr", type=float, default=1e-3, help="the full-precision twin's peak, decaying linearly")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=0.0,
                        help="above 0, the arms are the smallest ternary student without and with the per-token loss at this weight")
    parser.add_argument("--integer-attention", action="store_true",
                        help="the arms are the smallest ternary student with the specified attention and with the integer prototype")
    parser.add_argument("--report", default=REPORT, help="the report's file name in --out")
    parser.add_argument("--bank-per-origin", type=int, default=8192)
    parser.add_argument("--eval-minutes", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=0, help="a fixed number of steps instead of the box's (tests)")
    parser.add_argument("--eval-steps", type=int, default=0, help="evaluate every this many steps instead of by the clock (tests)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot", action="store_true", help="no W&B run")
    parser.set_defaults(cut_ends=0.0, cut_starts=0.0)  # train.step_maker's options, none of which the proof of concept uses

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    poc(parser().parse_args(argv))
