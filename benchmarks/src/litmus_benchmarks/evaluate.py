"""Metrics with paired bootstrap intervals, per benchmark, task and stratum, written as one tidy table.

A score is higher for text more likely written by a model. A task compares positive origins against negative ones:
`ai_vs_human` leaves mixed text out, `any_ai_vs_human` counts it positive, and `mixed_vs_human` reads mixed text alone.
Every stratum reuses the task's replicate indices for all detectors, so the intervals of two detectors' difference
come from the same draws (a paired bootstrap).

Strata: `domain`, `attack` and `length` compare positives and negatives of the same stratum, since RAID attacks and
lengths apply to human text too; `generator` compares one generator's positives against every negative of the task,
since human text has no generator.
"""

from __future__ import annotations

import zlib
import numpy as np
import pandas as pd

from collections.abc import Callable, Mapping
from typing import Any
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from foundry.eval.bootstrap import stratified_bootstrap_indices
from foundry.metrics.binary import balanced_mcc
from foundry.metrics.ranking import weighted_roc_auc


Scorer = Callable[[list[str]], np.ndarray]
TASKS = {
    "ai_vs_human": (("ai",), ("human",)),
    "any_ai_vs_human": (("ai", "mixed"), ("human",)),
    "mixed_vs_human": (("mixed",), ("human",)),
}
FPR_TARGETS = {"tpr_at_fpr_1": 0.01, "tpr_at_fpr_5": 0.05}
METRICS = ("pr_auc", "roc_auc", "balanced_mcc", *FPR_TARGETS)
LENGTH_EDGES = (0, 50, 150, 400, 1000, np.inf)
LENGTH_LABELS = ("1-50", "51-150", "151-400", "401-1000", ">1000")
SAME_STRATUM = ("domain", "attack", "length")
MIN_PER_CLASS = 20
TEXT_FIELDS = ["row", "origin", "generator", "domain", "attack", "words"]
METRIC_COLUMNS = ("benchmark", "task", "stratum", "subset", "detector", "metric", "value", "low", "high", "positives",
                  "negatives", "replicates")


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """True positive rate at the threshold that flags at most `fpr` of the negatives (score strictly above it).

    The threshold is the negative score with `floor(fpr * n_neg)` negatives above it, so ties at the threshold are never
    flagged and the realized false positive rate never exceeds the target, as RAID's evaluation requires.
    """
    # labels, scores: (n,)
    negatives = np.sort(scores[labels == 0])  # (n_neg,)
    allowed = int(np.floor(fpr * len(negatives)))
    threshold = negatives[len(negatives) - allowed - 1] if allowed < len(negatives) else -np.inf
    return float(np.mean(scores[labels == 1] > threshold))


def point_metrics(labels: np.ndarray, scores: np.ndarray, *, mcc_subsamples: int = 10) -> dict[str, float]:
    """PR AUC (average precision), ROC AUC, balanced MCC and TPR at 1% and 5% FPR of one detector on one set."""
    # labels, scores: (n,)
    metrics = {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(weighted_roc_auc(scores, labels, np.ones((len(labels), 1)))[0]),
        "balanced_mcc": balanced_mcc(labels, scores, subsamples=mcc_subsamples)["balanced_mcc"],
    }
    metrics.update({name: tpr_at_fpr(labels, scores, fpr) for name, fpr in FPR_TARGETS.items()})
    return metrics


def bootstrap_metrics(
    labels: np.ndarray, scores: np.ndarray, replicates: list[np.ndarray], *, mcc_subsamples: int = 10,  # (n,), (n,), r x (n,)
) -> dict[str, np.ndarray]:
    """Every metric on every replicate; ROC AUC in one weighted pass over the ranking, the rest replicate by replicate."""
    # labels, scores: (n,); replicates: r index arrays, each (n,)
    weights = np.stack([np.bincount(indices, minlength=len(labels)) for indices in replicates], axis=1)  # (n, r)
    values = {name: np.empty(len(replicates)) for name in METRICS}
    values["roc_auc"] = weighted_roc_auc(scores, labels, weights.astype(np.float64))  # (r,)
    for column, indices in enumerate(replicates):
        drawn_labels, drawn_scores = labels[indices], scores[indices]  # (n,), (n,)
        values["pr_auc"][column] = average_precision_score(drawn_labels, drawn_scores)
        values["balanced_mcc"][column] = balanced_mcc(drawn_labels, drawn_scores, subsamples=mcc_subsamples)["balanced_mcc"]
        for name, fpr in FPR_TARGETS.items():
            values[name][column] = tpr_at_fpr(drawn_labels, drawn_scores, fpr)
    return values  # each metric (r,)


def length_bins(words: pd.Series) -> pd.Series:
    """Each text's length bin as a string, "" for a text of no words (outside every bin) on every pandas version."""
    bins = pd.cut(words, bins=list(LENGTH_EDGES), labels=list(LENGTH_LABELS), right=True)
    return bins.astype(object).where(bins.notna(), "").astype(str)


def subsets(frame: pd.DataFrame, positive: tuple[str, ...]) -> list[tuple[str, str, np.ndarray]]:
    """(stratum, subset, row mask) for the whole task and every stratum value with enough of both classes."""
    is_positive = frame["origin"].isin(positive).to_numpy()  # (n,)
    masks = [("all", "all", np.ones(len(frame), dtype=bool))]
    for stratum in ("domain", "generator", "attack", "length"):
        values = length_bins(frame["words"]) if stratum == "length" else frame[stratum].astype(str)
        for value in sorted(values[is_positive].unique()):
            if value in ("", "nan", "none", "None"):
                continue
            in_value = (values == value).to_numpy()  # (n,)
            mask = in_value if stratum in SAME_STRATUM else (in_value & is_positive) | ~is_positive  # (n,)
            if (mask & is_positive).sum() >= MIN_PER_CLASS and (mask & ~is_positive).sum() >= MIN_PER_CLASS:
                masks.append((stratum, value, mask))
    return masks  # each mask (n,)


def evaluate_scores(
    scores: pd.DataFrame, *, replicates: int = 1000, strata_replicates: int = 200, seed: int = 0, mcc_subsamples: int = 10,
) -> pd.DataFrame:
    """The tidy metric table (`METRIC_COLUMNS`) from per-text scores.

    `scores` holds one row per text and detector: benchmark, row, origin, generator, domain, attack, words, detector,
    score. A detector missing a score for any text of a benchmark is an error, never a silent drop.
    """
    if not np.isfinite(scores["score"]).all():
        raise ValueError("every score must be finite")
    # Each detector scored the same texts in the same order, so a text is its position within (benchmark, detector).
    scores = scores.assign(text_index=scores.groupby(["benchmark", "detector"]).cumcount())
    records: list[dict[str, object]] = []
    for benchmark, block in scores.groupby("benchmark", sort=True):
        detectors = sorted(block["detector"].unique())
        meta = block[block["detector"] == detectors[0]].set_index("text_index")[TEXT_FIELDS]
        for detector in detectors[1:]:
            other = block[block["detector"] == detector].set_index("text_index")[TEXT_FIELDS]
            if not other.equals(meta):
                raise ValueError(f"{benchmark}: {detector} scored different texts than {detectors[0]}")
        wide = meta.join(block.pivot(index="text_index", columns="detector", values="score")).reset_index(drop=True)
        for task, (positive, negative) in TASKS.items():
            frame = wide[wide["origin"].isin(positive + negative)].reset_index(drop=True)
            labels = frame["origin"].isin(positive).to_numpy().astype(np.int64)  # (n,)
            if labels.sum() < MIN_PER_CLASS or (1 - labels).sum() < MIN_PER_CLASS:
                continue
            for stratum, subset, mask in tqdm(subsets(frame, positive), desc=f"{benchmark} {task}", mininterval=10):
                sub_labels = labels[mask]  # (m,)
                count = replicates if stratum == "all" else strata_replicates
                rng = np.random.default_rng([seed, zlib.crc32(f"{benchmark}/{task}/{stratum}/{subset}".encode())])
                draws = [stratified_bootstrap_indices(sub_labels, rng) for _ in range(count)]  # count arrays of (m,)
                for detector in detectors:
                    sub_scores = frame[detector].to_numpy(dtype=np.float64)[mask]  # (m,)
                    point = point_metrics(sub_labels, sub_scores, mcc_subsamples=mcc_subsamples)
                    spread = bootstrap_metrics(sub_labels, sub_scores, draws, mcc_subsamples=mcc_subsamples)
                    for metric in METRICS:
                        low, high = np.percentile(spread[metric], [2.5, 97.5])
                        records.append({"benchmark": benchmark, "task": task, "stratum": stratum, "subset": subset,
                                        "detector": detector, "metric": metric, "value": point[metric],
                                        "low": float(low), "high": float(high), "positives": int(sub_labels.sum()),
                                        "negatives": int((1 - sub_labels).sum()), "replicates": count})
    return pd.DataFrame(records, columns=list(METRIC_COLUMNS))


def raid_leaderboard_metrics(results: dict[str, Any], detector: str) -> pd.DataFrame:
    """The metric table rows of a RAID leaderboard `results.json`, the only way to score RAID's hidden test labels.

    The overall rows come from `score_agg` and the attack rows from `scores` aggregated over every domain, model,
    decoding and repetition setting. RAID's evaluator gives no intervals, so `low` and `high` are empty.
    """
    rows = []

    def add(stratum: str, subset: str, score: dict[str, Any]) -> None:
        values = {"roc_auc": score.get("auroc")}
        for name, key in (("tpr_at_fpr_5", "0.05"), ("tpr_at_fpr_1", "0.01")):
            values[name] = ((score.get("accuracy") or {}).get(key) or {}).get("accuracy")
        for metric, value in values.items():
            if value is not None:
                rows.append({"benchmark": "raid", "task": "ai_vs_human", "stratum": stratum, "subset": subset,
                             "detector": detector, "metric": metric, "value": float(value), "low": np.nan, "high": np.nan,
                             "positives": np.nan, "negatives": np.nan, "replicates": 0})

    add("all", "all", results["score_agg"]["all"])
    add("all", "no attack", results["score_agg"]["no_adversarial"])
    for score in results["scores"]:
        if all(score[key] == "all" for key in ("domain", "model", "decoding", "repetition_penalty")) and score["attack"] not in ("all", "none"):
            add("attack", score["attack"], score)
    return pd.DataFrame(rows, columns=list(METRIC_COLUMNS))


def score_texts(texts: pd.DataFrame, scorers: Mapping[str, Scorer], *, batch_size: int = 64) -> pd.DataFrame:
    """Per-text scores of every detector, in batches, as the long table `evaluate_scores` reads."""
    frames = []
    for detector, scorer in scorers.items():
        values = []
        for start in tqdm(range(0, len(texts), batch_size), desc=detector, mininterval=10):
            batch = texts["text"].iloc[start:start + batch_size].tolist()
            batch_scores = np.asarray(scorer(batch), dtype=np.float64)  # (b,)
            if batch_scores.shape != (len(batch),):
                raise ValueError(f"{detector} returned shape {batch_scores.shape} for {len(batch)} texts")
            values.append(batch_scores)
        frames.append(texts.drop(columns=["text"]).assign(detector=detector, score=np.concatenate(values)))
    return pd.concat(frames, ignore_index=True)
