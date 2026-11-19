"""Binary ranking definitions shared by Oracle and retrieval evaluation.

Scores may be arbitrary finite real values; they need not be probabilities. Unknown
targets must be excluded by the caller's explicit observation mask before this function.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def binary_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """AP, trapezoidal PR area, and AUROC, with null for an unsupported denominator.

    AP and PR area require at least one positive. AUROC requires both classes. Empty inputs
    return three nulls. Equal scores form threshold groups; ROC ties count one half.
    """
    # labels, scores: (n,), one entry per observed row
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)  # (n,), (n,)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("binary rankings require aligned one-dimensional labels and scores")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("binary rankings require observed binary labels and finite scores")

    metrics: dict[str, float | None] = {"average_precision": None, "pr_auc_trapezoidal": None, "auroc": None}
    positive = labels == 1  # (n,)
    if positive.any():
        # One point per distinct score threshold, plus recall zero at precision one.
        precision, recall, _ = precision_recall_curve(labels, scores)  # (n_thresholds + 1,) each
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        metrics["pr_auc_trapezoidal"] = float(auc(recall, precision))
    if positive.any() and (~positive).any():
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    return metrics


def weighted_roc_auc(scores: np.ndarray, truth: np.ndarray, W: np.ndarray) -> np.ndarray:
    """ROC AUC under each column of row weights, ties averaged: one pass over the ranking serves every bootstrap replicate,
    where `W` holds a replicate's row weights per column (a resample's multiplicities, or a cluster's weights).

    Each positive is credited with the weight of the negatives below it and half the weight of the negatives it ties, so a
    column of ones gives `roc_auc_score`. Promoted 2026-11-01 from the attribute audits of pli_experimental_v2,
    ec_experimental_v3 and solubility_binary_v2.
    """
    # scores: (n); truth: (n) in {0, 1}; W: (n, r)
    order = np.argsort(scores, kind="stable")  # (n)
    s = scores[order]  # (n)
    t = truth[order].astype(bool)  # (n)
    Wo = W[order]  # (n, r)
    negative = Wo * ~t[:, None]  # (n, r)
    # Ties: every row of a tied block sees the block's negatives at half weight and the negatives below the block in full.
    _, start, counts = np.unique(s, return_index=True, return_counts=True)  # (u), (u), (u)
    block_start = np.repeat(start, counts)  # (n)
    block_end = block_start + np.repeat(counts, counts)  # (n)
    cumulative = np.vstack([np.zeros((1, W.shape[1])), np.cumsum(negative, axis=0)])  # (n + 1, r)
    credit = cumulative[block_start] + 0.5 * (cumulative[block_end] - cumulative[block_start])  # (n, r)
    positive_weight = (Wo * t[:, None]).sum(axis=0)  # (r)
    negative_weight = negative.sum(axis=0)  # (r)
    return (credit * (Wo * t[:, None])).sum(axis=0) / (positive_weight * negative_weight)  # (r)
