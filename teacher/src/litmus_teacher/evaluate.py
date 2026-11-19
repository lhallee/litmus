"""The `calibrate` step: fit the calibration table on the calibration split, then score validation and test.

Discrimination (`metrics.csv`): PR AUC (average precision), ROC AUC and balanced MCC, always all three, for the
two-class task (ai or mixed against human), human against ai alone, each origin against the rest, and their macro
mean over the three origins; overall and per length bin and per stratum column; for the pair head's raw reading, its
calibrated probabilities, the logistic regression baseline and the random-embedding control. Balanced MCC's threshold
is chosen on validation and carried to test. The uncertainty's own scorecard (`uncertainty.csv`) comes from
`calibrate.uncertainty_rows`. Test is read once: a second read needs `--allow-test-reread`, which the summary records.
"""

from __future__ import annotations

import argparse
import json
import time
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pathlib import Path
from typing import Any

from .calibrate import (
    Readings,
    Verdicts,
    apply_calibration,
    fit_calibration,
    save_table,
    three_way,
    uncertainty_rows,
    write_rows,
)
from .embed import parquet_files, read_columns
from .strata import AI, HUMAN, MIXED, ORIGINS, length_bin_names, length_bins, write_json, write_summary


DEFAULT_STRATA = ("format", "language", "domain", "generator", "era")
TASKS = ("binary", "human_vs_ai", "human_ovr", "ai_ovr", "mixed_ovr")


def task_scores(probabilities: np.ndarray, labels: np.ndarray,
                two_class: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per task: score, positive and the rows that count, from three-way probabilities (n, 3) and labels (n,)."""
    # probabilities: (n, 3); labels: (n,); two_class: (n,) or None
    binary = 1.0 - probabilities[:, HUMAN] if two_class is None else two_class  # (n,)
    everyone = np.ones(len(labels), dtype=bool)  # (n,)

    return {  # (n,), (n,), (n,) per task
        "binary": (binary, labels != HUMAN, everyone),
        "human_vs_ai": (binary, labels == AI, labels != MIXED),
        "human_ovr": (probabilities[:, HUMAN], labels == HUMAN, everyone),
        "ai_ovr": (probabilities[:, AI], labels == AI, everyone),
        "mixed_ovr": (probabilities[:, MIXED], labels == MIXED, everyone),
    }


def discrimination(score: np.ndarray, positive: np.ndarray, threshold: float | None) -> dict[str, float | None]:
    """PR AUC, ROC AUC and balanced MCC, all None when a class is missing: (n,), (n,) to a row."""
    # score, positive: (n,)
    from foundry.metrics import balanced_mcc
    from foundry.metrics.ranking import binary_ranking_metrics

    positive = positive.astype(np.int64)  # (n,)
    if positive.min(initial=1) == positive.max(initial=0):
        return {"pr_auc": None, "roc_auc": None, "balanced_mcc": None, "threshold": threshold}
    ranking = binary_ranking_metrics(positive, score)
    mcc = balanced_mcc(positive, score, threshold=threshold)

    return {"pr_auc": ranking["average_precision"], "roc_auc": ranking["auroc"], "balanced_mcc": mcc["balanced_mcc"],
            "threshold": mcc["threshold"]}


def model_rows(model: str, split: str, probabilities: np.ndarray, labels: np.ndarray, strata: dict[str, np.ndarray],
               thresholds: dict[str, float], two_class: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Discrimination rows of one model on one split, overall and per stratum; fills `thresholds` on validation."""
    # probabilities: (n, 3); labels: (n,); strata: (n,) each; two_class: (n,) or None
    rows = []
    for kind, values in strata.items():
        for value in np.unique(values.astype(str)):
            members = values.astype(str) == value  # (n,)
            macro: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "balanced_mcc": []}
            for task, (score, positive, counted) in task_scores(probabilities, labels, two_class).items():
                chosen = members & counted
                if not chosen.any():
                    continue
                key = f"{model}|{task}"
                carried = thresholds.get(key) if (split != "validation" or kind != "all") else None
                metrics = discrimination(score[chosen], positive[chosen], carried)
                if split == "validation" and kind == "all":
                    thresholds[key] = metrics["threshold"]
                rows.append({"model": model, "split": split, "task": task, "stratum_kind": kind, "stratum": value,
                             "n": int(chosen.sum()), "n_positive": int(positive[chosen].sum()), **metrics})
                if task.endswith("_ovr"):
                    for name in macro:
                        if metrics[name] is not None:
                            macro[name].append(metrics[name])
            if all(len(values_) == len(ORIGINS) for values_ in macro.values()):
                rows.append({"model": model, "split": split, "task": "three_way_macro", "stratum_kind": kind,
                             "stratum": value, "n": int(members.sum()), "n_positive": None,
                             **{name: float(np.mean(values_)) for name, values_ in macro.items()}, "threshold": None})

    return rows


def strata_of(columns: dict[str, np.ndarray], ids: np.ndarray, tokens: np.ndarray) -> dict[str, np.ndarray]:
    """The stratum arrays of the given ids: `all`, `length_bin` and every requested column."""
    # columns: (n_all,) each; ids: (n,); tokens: (n_all,)
    names = length_bin_names()

    return {"all": np.full(len(ids), "all", dtype=object),  # (n,) each
            "length_bin": np.array([names[value] for value in length_bins(tokens[ids])], dtype=object),
            **{name: values[ids] for name, values in columns.items()}}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True, help="the bank step's output for the chosen head")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logistic", type=Path, nargs="*", default=[], help="predictions.npz of logistic baselines")
    parser.add_argument("--control-bank-dir", type=Path, help="the bank step's output for the random-embedding head")
    parser.add_argument("--strata", default=",".join(DEFAULT_STRATA), help="corpus columns to report by")
    parser.add_argument("--min-rows", type=int, default=200, help="calibration rows a length group needs per cell")
    parser.add_argument("--allow-test-reread", action="store_true")


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    strata_columns = [name for name in args.strata.split(",") if name]
    available = set(pq.read_schema(parquet_files(args.corpus)[0]).names)
    missing = [name for name in strata_columns if name not in available]
    if missing:
        raise ValueError(f"the corpus has no column {missing}; pass --strata from {sorted(available)}")
    marker = args.out / "test_read.json"
    if marker.exists() and not args.allow_test_reread:
        raise RuntimeError(f"test was already read ({marker}); rerun with --allow-test-reread, which is recorded")

    metadata = read_columns(args.corpus, ["id", "label", "tokens", *strata_columns]).sort_by("id")
    labels = metadata["label"].to_numpy().astype(np.int64)  # (n,)
    tokens = metadata["tokens"].to_numpy()  # (n,)
    columns = {name: pc.fill_null(metadata[name].cast("string"), "").to_numpy(zero_copy_only=False).astype(object)
               for name in strata_columns}
    calibration, extra = Readings.load(args.bank_dir / "readings_calibration.npz")
    assert np.array_equal(extra["labels"], labels[extra["ids"]]), "readings and corpus disagree on labels"
    table = fit_calibration(calibration, extra["labels"], min_rows=args.min_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    save_table(table, args.out / "calibration.json")

    thresholds: dict[str, float] = {}
    metric_rows, uncertainty = [], []
    for split in ("validation", "test"):
        readings, extra = Readings.load(args.bank_dir / f"readings_{split}.npz")
        ids, split_labels = extra["ids"], extra["labels"]
        strata = strata_of(columns, ids, tokens)
        verdicts: Verdicts = apply_calibration(table, readings)
        metric_rows += model_rows("pair_head", split, three_way(readings.mean), split_labels, strata, thresholds)
        metric_rows += model_rows("pair_head_calibrated", split, verdicts.three_way, split_labels, strata, thresholds,
                                  two_class=verdicts.probability)
        uncertainty += uncertainty_rows(verdicts, split_labels, {name: columns[name][ids] for name in strata_columns}, split)
        for path in args.logistic:
            predictions = np.load(path)
            model = f"logistic_{path.parent.name}"
            metric_rows += model_rows(model, split, predictions[f"{split}_probabilities"], labels[predictions[f"{split}_ids"]],
                                      strata_of(columns, predictions[f"{split}_ids"], tokens), thresholds)
        if args.control_bank_dir is not None:
            control, control_extra = Readings.load(args.control_bank_dir / f"readings_{split}.npz")
            metric_rows += model_rows("random_control", split, three_way(control.mean), control_extra["labels"],
                                      strata_of(columns, control_extra["ids"], tokens), thresholds)
    for path in (args.out / "metrics.csv", args.out / "uncertainty.csv"):
        path.unlink(missing_ok=True)
    write_rows(metric_rows, args.out / "metrics.csv")
    write_rows(uncertainty, args.out / "uncertainty.csv")
    write_json(args.out / "thresholds.json", thresholds)
    earlier = json.loads(marker.read_text(encoding="utf-8"))["reads"] if marker.exists() else []
    reads = [*earlier, time.strftime("%Y-%m-%dT%H:%M:%S")]
    write_json(marker, {"reads": reads})

    def headline(split: str, model: str, task: str) -> dict[str, Any]:
        return next(row for row in metric_rows if row["split"] == split and row["model"] == model
                    and row["task"] == task and row["stratum_kind"] == "all")

    write_summary(args.out, "calibrate", {
        "test_reads": len(reads), "calibration_rows": table["calibration_rows"],
        "groups": {name: [entry["bins"] for entry in table["conformal"][name]] for name in ORIGINS},
        "headline": {split: {task: headline(split, "pair_head", task) for task in ("binary", "three_way_macro")}
                     for split in ("validation", "test")},
        "coverage_all": [row for row in uncertainty if row["stratum_kind"] == "all" and row["metric"] in
                         ("coverage", "human_false_positive_rate", "mean_set_size", "expected_calibration_error")],
        "table_bytes": (args.out / "calibration.json").stat().st_size, "seconds": time.perf_counter() - started,
    })
