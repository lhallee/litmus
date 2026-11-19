"""Protein-centric Fmax and average precision of multi-label scores held as sparse entries.

An evaluation over thousands of terms is mostly unscored pairs: a model or a baseline scores some terms of each protein,
and every other term the protein could carry ties below them. These read only the scored entries and the counts of what
was not scored, so they cost the entries, never the proteins times the terms, and give the numbers of the dense
computation in which every unscored candidate scores below every scored one.

The dense forms are production_suite's `go_release_audit.fmax` and `average_precision`, which Oracle One also imports; a
consumer moves here when it switches, and `tests/tier1_unit/test_metrics_multilabel.py` holds these equal to dense
references of the same rules.

Shapes: `e` scored entries, `n` proteins, `t` terms, `h` thresholds.
"""

from __future__ import annotations

import numpy as np


# CAFA's protein-centric Fmax is the best F1 over thresholds from 0.01 to 1 in steps of 0.01 (Radivojac et al. 2013).
FMAX_THRESHOLDS = np.linspace(0.01, 1.0, 100)  # (h,)


def protein_fmax(
    protein: np.ndarray, score: np.ndarray, true: np.ndarray, positives: np.ndarray, thresholds: np.ndarray = FMAX_THRESHOLDS,
) -> tuple[float, float]:
    """CAFA's protein-centric Fmax and the threshold that reaches it.

    `protein` (e,) is each scored entry's protein row, `score` (e,) its score and `true` (e,) whether the protein carries the
    term; `positives` (n,) counts each protein's true terms, scored or not. At a threshold, precision is the mean, over the
    proteins with an entry at or above it, of the share of those entries that are true; recall is the mean, over the proteins
    with a positive, of the share of their positives among those entries. Returns (0.0, nan) when no threshold keeps an entry.
    """
    # protein, score, true: (e,); positives: (n,); thresholds: (h,)
    protein = np.asarray(protein, dtype=np.int64)  # (e,)
    score = np.asarray(score, dtype=np.float64)  # (e,)
    true = np.asarray(true, dtype=bool)  # (e,)
    positives = np.asarray(positives, dtype=np.int64)  # (n,)
    if not protein.shape == score.shape == true.shape or protein.ndim != 1 or positives.ndim != 1:
        raise ValueError(f"protein, score and true must be vectors of one length, got {protein.shape}, {score.shape}, {true.shape}")
    if not np.isfinite(score).all():
        raise ValueError("every score must be finite")
    n = len(positives)
    if len(protein) and (protein.min() < 0 or protein.max() >= n):
        raise ValueError(f"protein rows must index the {n} proteins of positives")
    hits_possible = np.bincount(protein[true], minlength=n)  # (n,)
    if (hits_possible > positives).any():
        raise ValueError("a protein has more true entries than positives")
    annotated = positives > 0  # (n,)
    if not annotated.any():
        raise ValueError("no protein has a positive, so recall is undefined")
    best, best_threshold = 0.0, float("nan")
    for threshold in np.asarray(thresholds, dtype=np.float64):
        chosen = score >= threshold  # (e,)
        predicted = np.bincount(protein[chosen], minlength=n)  # (n,)
        answered = predicted > 0  # (n,)
        if not answered.any():
            continue
        hits = np.bincount(protein[chosen & true], minlength=n)  # (n,)
        precision = float(np.mean(hits[answered] / predicted[answered]))
        recall = float(np.mean(hits[annotated] / positives[annotated]))
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best:
                best, best_threshold = f1, float(threshold)
    return best, best_threshold


def ranked_average_precision(score: np.ndarray, true: np.ndarray, candidates: int, positives: int) -> float:
    """Average precision of a ranking of `candidates` items, `positives` of them true, given only its scored items.

    `score` (e,) and `true` (e,) are the scored items. The other `candidates - e` items tie below every score and hold the
    `positives - true.sum()` true items left. Ties are handled by threshold, as scikit-learn computes it, so this equals the
    average precision of the dense ranking with every unscored item below every scored one. nan when no item is true.
    """
    # score, true: (e,)
    score = np.asarray(score, dtype=np.float64)  # (e,)
    true = np.asarray(true, dtype=bool)  # (e,)
    if score.shape != true.shape or score.ndim != 1:
        raise ValueError(f"score and true must be vectors of one length, got {score.shape} and {true.shape}")
    if not np.isfinite(score).all():
        raise ValueError("every score must be finite")
    scored_true = int(true.sum())
    if not len(score) <= candidates or not scored_true <= positives or positives - scored_true > candidates - len(score):
        raise ValueError(f"{len(score)} scored items with {scored_true} true do not fit {candidates} candidates with {positives} true")
    if positives == 0:
        return float("nan")
    total = 0.0
    if len(score):
        order = np.argsort(-score, kind="stable")  # (e,)
        ranked_score, ranked_true = score[order], true[order]  # (e,), (e,)
        last_of_group = np.concatenate([np.flatnonzero(np.diff(ranked_score) != 0), [len(score) - 1]])  # (g,) end of each tie group
        true_positives = np.cumsum(ranked_true)[last_of_group]  # (g,)
        precision = true_positives / (last_of_group + 1)  # (g,)
        recall = true_positives / positives  # (g,)
        total = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    left = positives - scored_true
    if left:  # the unscored items are one tie group, reached at full recall with every candidate counted
        total += left / positives * (positives / candidates)
    return total


def grouped_average_precision(score: np.ndarray, items: np.ndarray, true: np.ndarray) -> float:
    """Average precision of a ranking made of tie groups: group `g` holds `items[g]` items, `true[g]` of them true, at `score[g]`.

    Groups of one score are one tie group, handled by threshold as scikit-learn handles ties, so a score that depends on the term
    alone, such as a train prior, has its micro average precision over every (protein, term) pair from per-term counts. nan when
    no item is true.
    """
    # score, items, true: (g,)
    score = np.asarray(score, dtype=np.float64)  # (g,)
    items = np.asarray(items, dtype=np.int64)  # (g,)
    true = np.asarray(true, dtype=np.int64)  # (g,)
    if not score.shape == items.shape == true.shape or score.ndim != 1:
        raise ValueError(f"score, items and true must be vectors of one length, got {score.shape}, {items.shape}, {true.shape}")
    if not np.isfinite(score).all() or (true < 0).any() or (true > items).any():
        raise ValueError("every score must be finite and every group hold from 0 to its items true")
    positives = int(true.sum())
    if positives == 0:
        return float("nan")
    distinct, group = np.unique(-score, return_inverse=True)  # (u,), (g,): tie groups, highest score first
    counted = np.bincount(group, weights=items, minlength=len(distinct))  # (u,)
    found = np.bincount(group, weights=true, minlength=len(distinct))  # (u,)
    precision = np.cumsum(found) / np.maximum(np.cumsum(counted), 1)  # (u,)
    return float(np.sum(found / positives * precision))


def term_average_precisions(
    term: np.ndarray, score: np.ndarray, true: np.ndarray, candidates: np.ndarray, positives: np.ndarray,
) -> np.ndarray:
    """Each term's average precision across the proteins it can be scored on, from the scored entries alone.

    `term` (e,) is each entry's term row, with `score` (e,) and `true` (e,); `candidates` (t,) counts the proteins each term
    ranks and `positives` (t,) the true ones among them. A term with no entry is one tie group, its average precision its
    positive rate. Returns (t,), nan where a term has no positive.
    """
    # term, score, true: (e,); candidates, positives: (t,)
    term = np.asarray(term, dtype=np.int64)  # (e,)
    score = np.asarray(score, dtype=np.float64)  # (e,)
    true = np.asarray(true, dtype=bool)  # (e,)
    candidates = np.asarray(candidates, dtype=np.int64)  # (t,)
    positives = np.asarray(positives, dtype=np.int64)  # (t,)
    if not term.shape == score.shape == true.shape or term.ndim != 1 or candidates.shape != positives.shape:
        raise ValueError("term, score and true must be vectors of one length, and candidates and positives of another")
    if len(term) and (term.min() < 0 or term.max() >= len(candidates)):
        raise ValueError(f"term rows must index the {len(candidates)} terms of candidates")
    order = np.argsort(term, kind="stable")  # (e,)
    bounds = np.searchsorted(term[order], np.arange(len(candidates) + 1))  # (t + 1,)
    precisions = np.full(len(candidates), np.nan)  # (t,)
    for row in np.flatnonzero(positives > 0):
        chosen = order[bounds[row]:bounds[row + 1]]  # (e_t,)
        precisions[row] = ranked_average_precision(score[chosen], true[chosen], int(candidates[row]), int(positives[row]))
    return precisions  # (t,)
