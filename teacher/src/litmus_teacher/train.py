"""Train pair heads on precomputed embeddings held on the GPU, compare encoders, and fit the baselines.

Each configuration (input features, output width, seed) is one W&B run through `foundry.logging.Run`, project
`litmus`, grouped under the experiment id. Each step draws two origin-balanced batches of `--batch-size` texts
(a third per origin, uniform within an origin) and scores every pair of them; the targets are same-origin. AdamW at
`--learning-rate` after 500 steps of linear warmup, weight decay 0.01, dropout 0.2, widths 1,024 and 8,192 as the
first checker: `HeadSettings`' defaults, the anchor `search` starts from, which also tunes every one of them.
Validation pair loss, on the same 16 pairs of balanced 3,072-text validation batches for every run (so runs of any
batch size or seed compare), is read every 500 steps; the checkpoint is the lowest, and training stops after 10
readings without a new one.

Sampling is uniform within an origin rather than uniform over clusters: the corpus was balanced by format and length on purpose, and e-means clusters here are topical regions whose uniform
sampling would undo that balance; the split, not the sampler, is what keeps regions apart.

The teacher reads ModernBERT-base. ModernBERT-large, mmBERT-base and concatenations are comparison arms embedded on a
subsample: with `--common-rows`, every feature set trains and validates on the rows all of them embedded, so the
arms are read against base on the same texts. The encoder is chosen by the mean of the best validation pair loss over
seeds, and the head by the lowest seed of that encoder. `--control` adds the same recipe on random vectors; `--logistic` fits a multinomial logistic
regression on each encoder's standardized embeddings and saves its probabilities for validation, test and calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import numpy as np
import pyarrow.parquet as pq
import torch

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .embed import read_columns
from .features import FeatureSource, feature_source
from .head import PairHead, same_origin_loss
from .strata import ORIGINS, write_json, write_summary


WARMUP_STEPS = 500
EVAL_EVERY = 500
PATIENCE = 10
LOG_EVERY = 100
VALIDATION_PAIRS = 16
VALIDATION_BATCH = 3072
VALIDATION_SEED = 10_000
MIN_LR_RATIO = 0.1  # a cosine schedule's floor, as a fraction of the peak
PRIOR_BIAS = math.log(0.5)  # the same-origin log-odds of a balanced batch: a third of pairs share an origin
STATISTICS_ROWS = 262_144
PILOT_ROWS = 200_000
LOGISTIC_STEPS = 3_000
LOGISTIC_EVAL_EVERY = 250
LOGISTIC_BATCH = 8_192
EVALUATED_SPLITS = ("validation", "test", "calibration")


@dataclass(frozen=True)
class HeadSettings:
    features: str
    output_width: int
    seed: int
    widths: tuple[int, ...] = (1024, 8192)
    dropout: float = 0.2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 3072  # per side, a third per origin
    max_steps: int = 40_000  # also a cosine schedule's horizon
    warmup_steps: int = WARMUP_STEPS
    schedule: str = "constant"  # after warmup: "constant", or "cosine" down to MIN_LR_RATIO at max_steps
    clamp: float = 10.0  # standardized inputs are clamped at this many standard deviations
    scale_init: float = 10.0
    learn_scale: bool = True
    bias_init: float = 0.0
    label_smoothing: float = 0.0
    eval_every: int = EVAL_EVERY
    validation_batch: int = VALIDATION_BATCH
    tag: str = ""  # names a searched configuration in place of the output width

    @property
    def name(self) -> str:
        return f"{self.features.replace('+', '_')}_{self.tag or f'w{self.output_width}'}_s{self.seed}"


def load_labels_and_splits(corpus: Path, split_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Each id's origin label and split name, in id order: (n,) int64 and (n,) object."""
    labels = read_columns(corpus, ["id", "label"]).sort_by("id")
    split = pq.read_table(split_path, columns=["id", "split"]).sort_by("id")
    assert np.array_equal(labels["id"].to_numpy(), split["id"].to_numpy()), "the split covers every corpus id"

    return labels["label"].to_numpy().astype(np.int64), split["split"].to_numpy(zero_copy_only=False)  # (n,), (n,)


def input_statistics(features: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of each input column over a sample of training rows: (n, f) to (f,), (f,)."""
    # features: (n, f)
    sample = torch.randperm(len(features), generator=generator, device=features.device)[:STATISTICS_ROWS]  # (s,)
    values = features[sample].double()  # (s, f)

    return values.mean(0).float(), values.std(0).float()  # (f,), (f,)


def balanced_draw(by_origin: list[torch.Tensor], size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of `size` rows, a third from each origin, uniform within one: rows (b,) and labels (b,)."""
    # by_origin: (n_o,) row positions for each of three origins
    if size % len(by_origin):
        raise ValueError("the batch size must divide by the number of origins")
    share = size // len(by_origin)
    rows, labels = [], []
    for origin, members in enumerate(by_origin):
        pick = torch.randint(len(members), (share,), generator=generator, device=members.device)  # (b / 3,)
        rows.append(members[pick])
        labels.append(torch.full((share,), origin, device=members.device))

    return torch.cat(rows), torch.cat(labels)  # (b,), (b,)


def origin_members(labels: torch.Tensor) -> list[torch.Tensor]:
    """The row positions of each origin: (n,) to three (n_o,) tensors."""
    # labels: (n,)
    members = [torch.nonzero(labels == origin).squeeze(1) for origin in range(len(ORIGINS))]
    for origin, rows in zip(ORIGINS, members, strict=True):
        if not len(rows):
            raise ValueError(f"no rows of origin {origin}")

    return members  # (n_o,) each


def validation_pairs(labels: torch.Tensor, batch: int = VALIDATION_BATCH,
                     count: int = VALIDATION_PAIRS) -> list[tuple[torch.Tensor, ...]]:
    """The fixed pairs of balanced validation batches every run is read on, whatever its batch size or seed."""
    # labels: (n_v,)
    by_origin = origin_members(labels)
    generator = torch.Generator(device=labels.device).manual_seed(VALIDATION_SEED)
    pairs = []
    for _ in range(count):
        rows_a, labels_a = balanced_draw(by_origin, batch, generator)  # (b,), (b,)
        rows_b, labels_b = balanced_draw(by_origin, batch, generator)  # (b,), (b,)
        pairs.append((rows_a, labels_a, rows_b, labels_b))

    return pairs  # count tuples of four (b,)


def pair_loss(head: PairHead, features_a: torch.Tensor, labels_a: torch.Tensor, features_b: torch.Tensor,
              labels_b: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """The pair loss of two batches, every text of one against every text of the other: (b, f), (b,) twice to ()."""
    # features_a, features_b: (b, f); labels_a, labels_b: (b,)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=features_a.is_cuda):
        vectors_a, vectors_b = head(features_a), head(features_b)  # (b, 2w) each

    return same_origin_loss(head.pair_logits(vectors_a, vectors_b), labels_a, labels_b, smoothing)  # ()


@torch.no_grad()
def validation_loss(head: PairHead, features: torch.Tensor, pairs: list[tuple[torch.Tensor, ...]]) -> float:
    """Mean pair loss, on hard targets, over the fixed validation batch pairs."""
    # features: (n_v, f); pairs: (b,) rows and labels of batch a, then of batch b
    head.eval()
    losses = [pair_loss(head, features[rows_a], labels_a, features[rows_b], labels_b)
              for rows_a, labels_a, rows_b, labels_b in pairs]
    head.train()

    return float(torch.stack(losses).mean())


def build_head(settings: HeadSettings, input_width: int, mean: torch.Tensor, scale: torch.Tensor,
               device: torch.device) -> PairHead:
    """A fresh head of these settings, standardizing with the training rows' column mean and scale: (f,), (f,)."""
    # mean, scale: (f,)
    head = PairHead(input_width, settings.widths, settings.output_width, settings.dropout, settings.clamp,
                    settings.scale_init, settings.learn_scale, settings.bias_init).to(device)
    head.set_input_statistics(mean, scale)

    return head


def learning_rate_factor(settings: HeadSettings, step: int) -> float:
    """The multiple of the peak learning rate at a step: linear warmup, then constant or cosine to MIN_LR_RATIO."""
    if step < settings.warmup_steps:
        return (step + 1) / settings.warmup_steps
    if settings.schedule == "constant":
        return 1.0
    if settings.schedule != "cosine":
        raise ValueError(f"unknown schedule {settings.schedule!r}")
    progress = min(1.0, (step - settings.warmup_steps) / max(1, settings.max_steps - settings.warmup_steps))

    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))


def optimizer_for(settings: HeadSettings, head: PairHead) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW at the settings' peak rate and decay, and its schedule."""
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(settings, step))


def load_split_features(source: FeatureSource, ids: np.ndarray, device: torch.device) -> torch.Tensor:
    """The rows' input vectors on the GPU in float16: (m,) to (m, f)."""
    # ids: (m,)
    return torch.from_numpy(source.gather(ids)).to(device)  # (m, f)


@dataclass(frozen=True)
class SplitTensors:
    """One feature set's training and validation rows on the GPU, loaded once and shared by every seed."""

    source: FeatureSource
    train_ids: np.ndarray  # (n_t,)
    train_features: torch.Tensor  # (n_t, f) float16
    train_labels: torch.Tensor  # (n_t,)
    validation_ids: np.ndarray  # (n_v,)
    validation_features: torch.Tensor  # (n_v, f) float16
    validation_labels: torch.Tensor  # (n_v,)


def load_split_tensors(source: FeatureSource, labels: np.ndarray, split: np.ndarray, pilot: bool,
                       device: torch.device, rows: np.ndarray | None = None) -> SplitTensors:
    """The training and validation rows this feature set has embedded, within `rows` when given; a pilot keeps 200,000."""
    # labels, split: (n,); rows: (n,) bool or None
    present = source.embedded(len(labels)) & (True if rows is None else rows)  # (n,)
    train_ids = np.flatnonzero((split == "train") & present)  # (n_t,)
    validation_ids = np.flatnonzero((split == "validation") & present)  # (n_v,)
    if pilot:
        train_ids = np.sort(np.random.default_rng(0).permutation(train_ids)[:PILOT_ROWS])

    return SplitTensors(
        source, train_ids, load_split_features(source, train_ids, device), torch.from_numpy(labels[train_ids]).to(device),
        validation_ids, load_split_features(source, validation_ids, device),
        torch.from_numpy(labels[validation_ids]).to(device),
    )


def train_head(settings: HeadSettings, rows: SplitTensors, directory: Path, experiment_id: str, pilot: bool,
               device: torch.device) -> dict[str, Any]:
    """Train one configuration to its best validation pair loss and save that checkpoint under `directory/head`."""
    from foundry.logging import Run

    source = rows.source
    train_ids, validation_ids = rows.train_ids, rows.validation_ids  # (n_t,), (n_v,)
    train_features, train_labels = rows.train_features, rows.train_labels  # (n_t, f), (n_t,)
    validation_features, validation_labels = rows.validation_features, rows.validation_labels  # (n_v, f), (n_v,)
    torch.manual_seed(settings.seed)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    by_origin, pairs = origin_members(train_labels), validation_pairs(validation_labels, settings.validation_batch)
    head = build_head(settings, source.width, *input_statistics(train_features, generator), device)
    optimizer, schedule = optimizer_for(settings, head)
    checkpoint = directory / "last.pt"
    start, best, stale = 0, float("inf"), 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        generator.set_state(state["generator"])
        start, best, stale = state["step"], state["best"], state["stale"]

    config = {**asdict(settings), "name": settings.name, "source": source.name, "train_rows": int(len(train_ids)),
              "validation_rows": int(len(validation_ids)), "experiment_id": experiment_id}
    best_step = start
    with Run(directory / "wandb", config, project="litmus", group=experiment_id, job_type="train",
             name=settings.name, pilot=pilot or None) as run:
        head.train()
        for step in run.batches(range(start, settings.max_steps), items=2 * settings.batch_size, item_unit="texts"):
            rows_a, labels_a = balanced_draw(by_origin, settings.batch_size, generator)  # (b,), (b,)
            rows_b, labels_b = balanced_draw(by_origin, settings.batch_size, generator)  # (b,), (b,)
            loss = pair_loss(head, train_features[rows_a], labels_a, train_features[rows_b], labels_b,
                             settings.label_smoothing)  # ()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % LOG_EVERY == 0:
                run.log({"train/loss": float(loss), "train/lr": schedule.get_last_lr()[0],
                         "train/scale": head.scale.item()}, step=step)
            if (step + 1) % settings.eval_every == 0:
                current = validation_loss(head, validation_features, pairs)
                run.log({"val/pair_loss": current, "val/best_pair_loss": min(best, current)}, step=step)
                if current < best:
                    best, stale, best_step = current, 0, step + 1
                    head.save(directory / "head")
                else:
                    stale += 1
                torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(),
                            "schedule": schedule.state_dict(), "generator": generator.get_state(), "step": step + 1,
                            "best": best, "stale": stale}, checkpoint)
                if stale >= PATIENCE:
                    break
        if not (directory / "head").exists():  # a pilot can end before the first reading
            head.save(directory / "head")
            best = validation_loss(head, validation_features, pairs)
        run.finish(summary={"val/best_pair_loss": best})
    record = {**config, "best_validation_pair_loss": best, "best_step": best_step, "pilot": pilot}
    if not pilot:
        write_json(directory / "done.json", record)

    return record


def fit_logistic(rows: SplitTensors, labels: np.ndarray, split: np.ndarray, directory: Path, seed: int,
                 device: torch.device) -> dict[str, Any]:
    """Multinomial logistic regression on standardized, clamped embeddings; saves split probabilities."""
    # labels, split: (n,)
    from foundry.layers.cross_entropy import cross_entropy

    source, features, train_labels = rows.source, rows.train_features, rows.train_labels  # (n_t, f), (n_t,)
    validation_features, validation_labels = rows.validation_features, rows.validation_labels  # (n_v, f), (n_v,)
    present = source.embedded(len(labels))  # (n,)
    generator = torch.Generator(device=device).manual_seed(seed)
    mean, scale = input_statistics(features, generator)  # (f,), (f,)
    scale = scale.clamp(min=1e-6)

    def standardized(values: torch.Tensor) -> torch.Tensor:
        # values: (b, f)
        return ((values.float() - mean) / scale).clamp(-10, 10)  # (b, f)

    model = torch.nn.Linear(source.width, len(ORIGINS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    by_origin = origin_members(train_labels)

    best, best_state = float("inf"), None
    for step in range(LOGISTIC_STEPS):
        drawn, batch_labels = balanced_draw(by_origin, LOGISTIC_BATCH - LOGISTIC_BATCH % 3, generator)
        loss = cross_entropy(model(standardized(features[drawn])), batch_labels)  # ()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % LOGISTIC_EVAL_EVERY == 0:
            with torch.no_grad():
                current = float(cross_entropy(model(standardized(validation_features)), validation_labels))
            if current < best:
                best, best_state = current, {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    outputs = {}
    for name in EVALUATED_SPLITS:
        ids = np.flatnonzero((split == name) & present)  # (m,)
        with torch.no_grad():
            probabilities = torch.softmax(model(standardized(load_split_features(source, ids, device))), dim=1)  # (m, 3)
        outputs[f"{name}_ids"] = ids
        outputs[f"{name}_probabilities"] = probabilities.cpu().numpy()
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "predictions.npz", **outputs)
    record = {"features": source.name, "best_validation_log_loss": best, "train_rows": int(len(rows.train_ids))}
    write_json(directory / "done.json", record)

    return record


def choose(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The encoder with the lowest mean best validation pair loss over its seeds, then that encoder's best seed."""
    by_features: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["features"] != "random":
            by_features.setdefault(f"{record['features']}|w{record['output_width']}", []).append(record)
    means = {key: float(np.mean([item["best_validation_pair_loss"] for item in items])) for key, items in by_features.items()}
    chosen_key = min(means, key=means.get)
    best = min(by_features[chosen_key], key=lambda item: item["best_validation_pair_loss"])

    return {"mean_validation_pair_loss": means, "chosen": chosen_key, "head": best}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="split.parquet from the split step")
    parser.add_argument("--embedding-root", type=Path, required=True, help="holds one directory per encoder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True, help="YYYY-MM-DD_slug, the W&B group")
    parser.add_argument("--features", nargs="+", default=["modernbert_base"],
                        help="encoder keys, or a+b for a concatenation; comparison arms: modernbert_large, mmbert_base")
    parser.add_argument("--common-rows", action="store_true",
                        help="train and validate every feature set on the rows all of them embedded (the arm "
                        "comparison); use its own --out, since heads are cached by name")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-widths", type=int, nargs="+", default=[2048], help="2w: query plus key")
    parser.add_argument("--batch-size", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=40_000)
    parser.add_argument("--control", action="store_true", help="also train the head on random vectors")
    parser.add_argument("--logistic", action="store_true", help="also fit the logistic regression baseline")
    parser.add_argument("--baselines-only", action="store_true",
                        help="train no head of --features: only --control and --logistic (the search trains the teacher)")
    parser.add_argument("--pilot", action="store_true", help="200,000 training rows and a timed window per run")


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    device = torch.device("cuda")
    labels, split = load_labels_and_splits(args.corpus, args.split)
    configurations = [HeadSettings(features, width, seed, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                   max_steps=args.max_steps)
                      for features in args.features for width in args.output_widths for seed in args.seeds
                      if not args.baselines_only]
    if args.control:
        configurations.append(HeadSettings("random", args.output_widths[0], 0, batch_size=args.batch_size,
                                           learning_rate=args.learning_rate, max_steps=args.max_steps))
    common = None  # (n,) bool: rows every requested feature set embedded
    if args.common_rows:
        common = np.ones(len(labels), dtype=bool)
        for features in args.features:
            common &= feature_source(features, args.embedding_root).embedded(len(labels))
    loaded: list[SplitTensors] = []  # the one feature set on the GPU, reused by every run that reads it

    def tensors_for(source: FeatureSource) -> SplitTensors:
        if not loaded or loaded[0].source != source:
            loaded.clear()
            torch.cuda.empty_cache()
            loaded.append(load_split_tensors(source, labels, split, args.pilot, device, common))
        return loaded[0]

    records, baselines = [], []
    for settings in configurations:
        directory = args.out / "heads" / settings.name
        if (directory / "done.json").exists() and not args.pilot:
            records.append(json.loads((directory / "done.json").read_text(encoding="utf-8")))
            continue
        source = feature_source(settings.features, args.embedding_root, settings.seed)
        records.append(train_head(settings, tensors_for(source), directory, args.experiment_id, args.pilot, device))
        logistic_dir = args.out / "logistic" / settings.features.replace("+", "_")
        if args.logistic and not args.pilot and settings.features != "random" and not (logistic_dir / "done.json").exists():
            baselines.append(fit_logistic(tensors_for(source), labels, split, logistic_dir, 0, device))
    for features in args.features if args.logistic and not args.pilot else []:
        logistic_dir = args.out / "logistic" / features.replace("+", "_")
        if (logistic_dir / "done.json").exists() and not any(item["features"] == features for item in baselines):
            baselines.append(json.loads((logistic_dir / "done.json").read_text(encoding="utf-8")))
        elif not (logistic_dir / "done.json").exists():
            baselines.append(fit_logistic(tensors_for(feature_source(features, args.embedding_root)), labels, split,
                                          logistic_dir, 0, device))
    selection = choose(records) if any(record["features"] != "random" for record in records) else {}
    if not args.pilot:
        write_json(args.out / "selection.json", selection)
    write_summary(args.out, "train" + (" --pilot" if args.pilot else ""), {
        "experiment_id": args.experiment_id, "runs": records, "baselines": baselines, "selection": selection,
        "common_rows": None if common is None else int(common.sum()), "seconds": time.perf_counter() - started})
