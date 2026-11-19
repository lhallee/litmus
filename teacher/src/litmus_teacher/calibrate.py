"""Calibration and uncertainty for any scorer of Litmus readings: fit on the `calibration` split, apply anywhere.

Model-agnostic and NumPy only: the teacher's bank scores and the student's both arrive as `Readings`, and the
fitted table is plain JSON (`CALIBRATION.md` documents the layout the Rust engine reads). Fitting needs NumPy;
`uncertainty_rows` also reads `foundry.metrics` (NumPy only) for its expected calibration error.

Public functions:

    readings_from_pair_logits(logits_by_origin: Sequence[ndarray (n, k_o)], cosines: ndarray (n, k), tokens: ndarray (n,),
                              top_k: int = TOP_K) -> Readings
    vector_cosines(vectors: ndarray (n, 2w), bank: ndarray (k, 2w)) -> ndarray (n, k)
    three_way(mean: ndarray (n, 3)) -> ndarray (n, 3)
    not_human_score(mean: ndarray (n, 3)) -> ndarray (n,)
    merge_bins(counts: ndarray (bins,) or (bins, c), min_rows: int) -> list[list[int]]
    fit_calibration(readings: Readings, labels: ndarray (n,), *, min_rows: int = MIN_CELL_ROWS) -> dict[str, Any]
    save_table(table: dict[str, Any], path: Path) -> None;  load_table(path: Path) -> dict[str, Any]
    apply_calibration(table: dict[str, Any], readings: Readings) -> Verdicts
    section_uncertainty(table: dict[str, Any], sections: Readings, level: float = 0.05) -> SectionUncertainty
    merge_p_values(p_values: ndarray (C, k)) -> ndarray (k,)
    combine_chunks(table: dict[str, Any], verdicts: Verdicts, level: float = 0.05) -> ChunkCombination
    chunk_uncertainty(table: dict[str, Any], chunks: Readings, level: float = 0.05) -> ChunkCombination
    uncertainty_rows(verdicts: Verdicts, labels: ndarray (n,), strata: dict[str, ndarray (n,)], split: str) -> list[dict[str, Any]]
    write_rows(rows: list[dict[str, Any]], path: Path) -> None

What a reading of one text is (`Readings`): against the bank vectors of each origin, the mean and the population
standard deviation of its same-origin pair probabilities (three each), the mean of its `TOP_K` highest cosines to
any bank vector (closeness), and its token count (ModernBERT tokens without special tokens, the corpus `tokens`).

The methods, each fitted on the calibration split alone, per length group:

- **Conformal sets (Mondrian split conformal, by true origin and length group).** The conformity of hypothesis `y`
  is the text's normalized reading `r_y = mean_y / sum(mean)`, which needs no fitting, so the calibration rows stay
  exchangeable with new ones and the guarantee is exact. For hypothesis `y`, the p-value is
  `(1 + #{calibration texts of origin y in the group with r_y <= r}) / (n + 1)`; `y` is ruled out at level `a` when
  `p <= a`. Because the calibration texts of each origin are scored alone, a human text like the calibration set is
  called not human (its human p-value at or under `a`) with probability at most `a`: the false positive rate on
  human text is bounded per length group. The table stores, for each grid level `u`, the order statistic
  `t_u = r_(floor(u (n + 1)))`, so `p <= u` exactly when `r < t_u`; the reported p-value is the smallest grid level
  `u` with `r < t_u`, an upper bound that is exact at 5% and 10%.
- **Probability of not human, with an interval (inductive Venn-Abers).** The two-class score `1 - r_human` is cut
  at the calibration quantiles into at most `VENN_BLOCKS` blocks; for a text in block `g`, isotonic regression of the
  labels with the text added once as human and once as not human gives `p0 <= p1`, and the probability is
  `p1 / (1 - p0 + p1)`. Venn predictors are valid (well calibrated in the Venn sense) for any score, with no
  parametric shape, and the interval's width is the evidence the calibration set holds near that score. Weak
  evidence from a short text shows as a probability near the base rate and a set of more than one origin, since
  both are fitted per length group.
- **Three-way probabilities.** `softmax(slope * log r + bias)` per length group, fitted by Newton's method on the
  calibration rows' log loss (temperature with a per-origin offset); a reading, not a set.
- **The mechanism's own signals.** The spread of the pair probabilities against each origin's bank, as a percentile
  of the calibration texts of the same length group; and closeness to the bank as a conformal p-value against the
  calibration texts of the group (all origins), flagged out of reference at `OUT_OF_REFERENCE_LEVEL`.
- **A long document read in chunks** (`combine_chunks`). Each chunk is calibrated as a text of its own; the
  document's p-values merge the chunks' by twice their mean (valid under any dependence, `merge_p_values`), its
  probabilities are the chunks' means with an interval widened by their disagreement, and a mixed-document signal
  fires when one chunk rules out human and another rules out both ai and mixed, each at `level / C` (Bonferroni,
  so a wholly human or wholly ai document fires it with probability at most `level`).

A length group is a run of consecutive length bins merged until every cell it needs holds `min_rows` calibration
texts (`merge_bins`): walk the bins from the shortest, close a group once every count reaches `min_rows`, and fold a
short last group into the one before. Bins with no calibration rows join their neighbors, so every length maps to
a group. A fit with too few rows in total raises; nothing is merged silently, since the table records every group.
"""

from __future__ import annotations

import csv
import json
import numpy as np

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .strata import AI, HUMAN, LENGTH_EDGES, MIXED, ORIGINS, length_bin_names, length_bins


FORMAT = "litmus-calibration"
VERSION = 1
LEVELS = (0.05, 0.10)  # the set levels every reading reports
# p-value grid: `p <= u` is exact at each of these; 1.0 is implicit (every p-value is at most 1).
GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.2, 0.25, 0.3,
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SPREAD_LEVELS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
OUT_OF_REFERENCE_LEVEL = 0.01
MIN_CELL_ROWS = 200
VENN_BLOCKS = 32
VENN_BLOCK_ROWS = 50  # fewest calibration rows per Venn-Abers block
TOP_K = 16
RELIABILITY_BINS = 10
SIGNIFICANT_DIGITS = 7
LOG_FLOOR = 1e-12
MERGE_FACTOR = 2.0  # twice the mean of p-values is a p-value under any dependence (merge_p_values)


@dataclass(frozen=True)
class Readings:
    """What a scorer reports for each of `n` texts."""

    mean: np.ndarray  # (n, 3) mean same-origin probability against the human, ai and mixed bank
    spread: np.ndarray  # (n, 3) population standard deviation of those probabilities
    closeness: np.ndarray  # (n,) mean of the TOP_K highest cosines to any bank vector
    tokens: np.ndarray  # (n,) ModernBERT tokens without special tokens

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if self.mean.shape != (n, 3) or self.spread.shape != (n, 3) or self.closeness.shape != (n,):
            raise ValueError("readings need mean (n, 3), spread (n, 3), closeness (n,) and tokens (n,)")
        if not (np.isfinite(self.mean).all() and np.isfinite(self.spread).all() and np.isfinite(self.closeness).all()):
            raise ValueError("readings must be finite")
        if (self.mean < 0).any() or (self.mean.sum(1) <= 0).any():
            raise ValueError("mean probabilities are non-negative and not all zero")

    def subset(self, rows: np.ndarray) -> Readings:
        # rows: (m,) indices or (n,) bool
        return Readings(self.mean[rows], self.spread[rows], self.closeness[rows], self.tokens[rows])

    def save(self, path: Path, **extra: np.ndarray) -> None:
        # extra: (n,) each, such as ids and labels
        np.savez(path, mean=self.mean, spread=self.spread, closeness=self.closeness, tokens=self.tokens, **extra)

    @classmethod
    def load(cls, path: Path) -> tuple[Readings, dict[str, np.ndarray]]:
        arrays = dict(np.load(path))
        readings = cls(arrays.pop("mean"), arrays.pop("spread"), arrays.pop("closeness"), arrays.pop("tokens"))

        return readings, arrays  # readings of n texts, extra arrays (n,) each


@dataclass(frozen=True)
class Verdicts:
    """A calibrated reading of each of `n` texts."""

    length_bin: np.ndarray  # (n,)
    p_values: np.ndarray  # (n, 3) conformal p-value of human, ai, mixed (grid upper bounds)
    sets: dict[float, np.ndarray]  # level -> (n, 3) bool, the origins not ruled out
    probability: np.ndarray  # (n,) probability the text is not human (Venn-Abers)
    interval: np.ndarray  # (n, 2) the Venn-Abers interval [p0, p1]
    three_way: np.ndarray  # (n, 3) calibrated probability of each origin
    spread_percentile: np.ndarray  # (n, 3) of the spread against each bank, among calibration texts
    closeness_p_value: np.ndarray  # (n,) conformal p-value of the closeness
    out_of_reference: np.ndarray  # (n,) bool

    def not_human(self, level: float) -> np.ndarray:
        """Texts called not human at `level`: their human p-value is at or under it, (n,) bool."""
        return ~self.sets[level][:, HUMAN]  # (n,)


@dataclass(frozen=True)
class SectionUncertainty:
    """A long document's reading across its sections."""

    probability: float  # token-weighted mean of the sections' probability of not human
    disagreement: float  # token-weighted standard deviation of those probabilities, 0 to 0.5
    lowest: float
    highest: float
    called_not_human: int  # sections whose human p-value is at or under the level
    called_human: int  # sections whose set is {human}
    conflict: bool  # one section is called not human and another rules out both ai and mixed


@dataclass(frozen=True)
class ChunkCombination:
    """A long document's verdict from the verdicts of its chunks, each read as a text of its own (`combine_chunks`)."""

    chunks: int  # C
    p_values: np.ndarray  # (3,) merged conformal p-values of human, ai, mixed
    sets: dict[float, np.ndarray]  # level -> (3,) bool, the origins the merged p-values do not rule out
    probability: float  # mean of the chunks' probabilities of not human
    interval: np.ndarray  # (2,) mean Venn-Abers interval, widened by the chunks' disagreement
    disagreement: float  # population standard deviation of the chunks' probabilities, 0 to 0.5
    lowest: float
    highest: float
    three_way: np.ndarray  # (3,) mean of the chunks' calibrated three-way probabilities
    closeness_p_value: float  # merged closeness p-value
    out_of_reference: bool
    chunk_level: float  # level / C, the Bonferroni level of the mixed-document signal
    model_chunks: int  # chunks whose human p-value is at or under chunk_level
    human_chunks: int  # chunks whose ai and mixed p-values are both at or under chunk_level
    mixed_document: bool  # one chunk is a model chunk and a different chunk a human chunk

    def not_human(self, level: float) -> bool:
        return not bool(self.sets[level][HUMAN])


def sigmoid(logits: np.ndarray) -> np.ndarray:
    # logits: (...)
    return 0.5 * (1.0 + np.tanh(0.5 * logits))  # (...) stable for any finite logit


def vector_cosines(vectors: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Cosine of each text's query-and-key vector to each bank vector: (n, 2w), (k, 2w) to (n, k)."""
    # vectors: (n, 2w); bank: (k, 2w)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # (n, 2w)
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)  # (k, 2w)

    return vectors @ bank.T  # (n, k)


def readings_from_pair_logits(logits_by_origin: Sequence[np.ndarray], cosines: np.ndarray, tokens: np.ndarray,
                              top_k: int = TOP_K) -> Readings:
    """Readings from pair logits against each origin's bank (three (n, k_o)) and cosines to the whole bank (n, k)."""
    # logits_by_origin: (n, k_o) for each of three origins; cosines: (n, k); tokens: (n,)
    if len(logits_by_origin) != len(ORIGINS):
        raise ValueError("one logit matrix per origin: human, ai, mixed")
    probabilities = [sigmoid(np.asarray(logits, dtype=np.float64)) for logits in logits_by_origin]  # three (n, k_o)
    mean = np.stack([values.mean(1) for values in probabilities], axis=1)  # (n, 3)
    spread = np.stack([values.std(1) for values in probabilities], axis=1)  # (n, 3)
    top = -np.sort(-np.asarray(cosines, dtype=np.float64), axis=1)[:, :top_k]  # (n, top_k)

    return Readings(mean, spread, top.mean(1), np.asarray(tokens))


def three_way(mean: np.ndarray) -> np.ndarray:
    """The normalized three-way reading: (n, 3) to (n, 3), rows summing to one."""
    # mean: (n, 3)
    return mean / mean.sum(1, keepdims=True)  # (n, 3)


def not_human_score(mean: np.ndarray) -> np.ndarray:
    """The two-class reading, ai or mixed against human: (n, 3) to (n,)."""
    # mean: (n, 3)
    return 1.0 - three_way(mean)[:, HUMAN]  # (n,)


def merge_bins(counts: np.ndarray, min_rows: int) -> list[list[int]]:
    """Consecutive length bins merged so every group holds `min_rows` in every column: (bins,) or (bins, c)."""
    # counts: (bins,) or (bins, c)
    counts = np.asarray(counts).reshape(len(counts), -1)  # (bins, c)
    if (counts.sum(0) < min_rows).any():
        raise ValueError(f"fewer than {min_rows} calibration rows in some cell even over all lengths: {counts.sum(0).tolist()}")
    groups: list[list[int]] = []
    current: list[int] = []
    total = np.zeros(counts.shape[1], dtype=np.int64)  # (c,)
    for index, row in enumerate(counts):
        current.append(index)
        total += row
        if (total >= min_rows).all():
            groups.append(current)
            current, total = [], np.zeros_like(total)
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    return groups


def group_of_bins(groups: list[list[int]], bins: int = len(LENGTH_EDGES)) -> np.ndarray:
    """The group of each length bin: (bins,)."""
    lookup = np.full(bins, -1, dtype=np.int64)  # (bins,)
    for index, members in enumerate(groups):
        lookup[members] = index
    assert (lookup >= 0).all(), "every length bin belongs to a group"

    return lookup  # (bins,)


def grid_thresholds(scores: np.ndarray) -> list[float | None]:
    """`t_u = scores_(floor(u (n + 1)))` for each grid level, None where that rank is 0 (nothing is ruled out)."""
    # scores: (n,)
    ordered = np.sort(scores)  # (n,)
    thresholds: list[float | None] = []
    for level in GRID:
        rank = int(np.floor(level * (len(ordered) + 1)))
        thresholds.append(None if rank == 0 else float(ordered[rank - 1]))

    return thresholds


def grid_p_values(scores: np.ndarray, thresholds: Sequence[float | None]) -> np.ndarray:
    """The smallest grid level `u` with `score < t_u`, else 1: (m,) to (m,)."""
    # scores: (m,)
    p_values = np.ones(len(scores))  # (m,)
    for level, threshold in reversed(list(zip(GRID, thresholds, strict=True))):
        if threshold is not None:
            p_values = np.where(scores < threshold, level, p_values)

    return p_values  # (m,)


def isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted nondecreasing least-squares fit by pool-adjacent-violators: (g,), (g,) to (g,)."""
    # values, weights: (g,)
    sums: list[float] = []
    masses: list[float] = []
    sizes: list[int] = []
    for value, weight in zip(values, weights, strict=True):
        sums.append(value * weight)
        masses.append(weight)
        sizes.append(1)
        while len(sums) > 1 and sums[-2] / masses[-2] >= sums[-1] / masses[-1]:
            total, mass, size = sums.pop(), masses.pop(), sizes.pop()
            sums[-1] += total
            masses[-1] += mass
            sizes[-1] += size

    return np.repeat([total / mass for total, mass in zip(sums, masses, strict=True)], sizes)  # (g,)


def venn_abers_blocks(scores: np.ndarray, positive: np.ndarray) -> dict[str, list[float]]:
    """Block edges and the Venn-Abers p0, p1 of a text falling in each block: (n,), (n,) to a table entry."""
    # scores, positive: (n,)
    blocks = max(1, min(VENN_BLOCKS, len(scores) // VENN_BLOCK_ROWS))
    edges = np.unique(np.quantile(scores, np.arange(1, blocks) / blocks))  # (g - 1,)
    index = np.searchsorted(edges, scores, side="right")  # (n,)
    count = np.bincount(index, minlength=len(edges) + 1).astype(np.float64)  # (g,)
    hits = np.bincount(index, weights=positive.astype(np.float64), minlength=len(edges) + 1)  # (g,)
    p0, p1 = np.zeros(len(count)), np.zeros(len(count))  # (g,) each
    for block in range(len(count)):
        for label, out in ((0.0, p0), (1.0, p1)):
            augmented_count, augmented_hits = count.copy(), hits.copy()  # (g,) each
            augmented_count[block] += 1
            augmented_hits[block] += label
            filled = augmented_count > 0
            fitted = isotonic(augmented_hits[filled] / augmented_count[filled], augmented_count[filled])
            out[block] = fitted[np.flatnonzero(filled).tolist().index(block)]

    return {"edges": edges.tolist(), "p0": p0.tolist(), "p1": p1.tolist()}


def fit_three_way(log_readings: np.ndarray, labels: np.ndarray) -> tuple[float, list[float]]:
    """Slope and per-origin offsets of `softmax(slope * log r + bias)` by Newton's method: (n, 3), (n,) to a, b."""
    # log_readings: (n, 3); labels: (n,)
    n = len(labels)
    target = np.eye(len(ORIGINS))[labels]  # (n, 3)
    jacobian = np.zeros((n, 3, 3))  # (n, 3 logits, 3 parameters): slope, bias of ai, bias of mixed
    jacobian[:, :, 0] = log_readings
    jacobian[:, AI, 1] = 1.0
    jacobian[:, MIXED, 2] = 1.0
    theta = np.array([1.0, 0.0, 0.0])  # (3,)

    def loss_of(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        # parameters: (3,)
        logits = jacobian @ parameters  # (n, 3)
        logits -= logits.max(1, keepdims=True)
        log_probabilities = logits - np.log(np.exp(logits).sum(1, keepdims=True))  # (n, 3)
        return float(-(target * log_probabilities).sum() / n), np.exp(log_probabilities)  # (), (n, 3)

    loss, probabilities = loss_of(theta)
    for _ in range(100):
        gradient = np.einsum("nca,nc->a", jacobian, probabilities - target) / n  # (3,)
        covariance = np.einsum("nc,cd->ncd", probabilities, np.eye(3)) - np.einsum("nc,nd->ncd", probabilities, probabilities)  # (n, 3, 3)
        hessian = np.einsum("nca,ncd,ndb->ab", jacobian, covariance, jacobian) / n + 1e-9 * np.eye(3)  # (3, 3)
        step = np.linalg.solve(hessian, gradient)  # (3,)
        size = 1.0
        while True:
            candidate_loss, candidate_probabilities = loss_of(theta - size * step)
            if candidate_loss <= loss or size < 1e-6:
                break
            size /= 2
        theta, loss, probabilities = theta - size * step, candidate_loss, candidate_probabilities
        if np.abs(size * step).max() < 1e-10:
            break
    if theta[0] <= 0:
        raise ValueError(f"the readings rank origins backward in a length group (slope {theta[0]:.3g})")

    return float(theta[0]), [0.0, float(theta[1]), float(theta[2])]


def _rounded(value: Any) -> Any:
    """Round every float to SIGNIFICANT_DIGITS, so the table the engine reads is the table Python applies."""
    if isinstance(value, float):
        return float(f"{value:.{SIGNIFICANT_DIGITS}g}")
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}

    return value


def fit_calibration(readings: Readings, labels: np.ndarray, *, min_rows: int = MIN_CELL_ROWS) -> dict[str, Any]:
    """The calibration table from the calibration split's readings and labels (0 human, 1 ai, 2 mixed)."""
    # labels: (n,)
    labels = np.asarray(labels).astype(np.int64)  # (n,)
    if labels.shape != readings.tokens.shape or not np.isin(labels, range(len(ORIGINS))).all():
        raise ValueError("one label of 0, 1 or 2 per reading")
    bins = length_bins(readings.tokens)  # (n,)
    count = np.zeros((len(LENGTH_EDGES), len(ORIGINS)), dtype=np.int64)  # (bins, 3)
    np.add.at(count, (bins, labels), 1)
    ratio = three_way(readings.mean)  # (n, 3)
    score = 1.0 - ratio[:, HUMAN]  # (n,)

    conformal = {}
    for origin, name in enumerate(ORIGINS):
        groups = merge_bins(count[:, origin], min_rows)
        entries = []
        for members in groups:
            rows = (labels == origin) & np.isin(bins, members)
            entries.append({"bins": members, "n": int(rows.sum()), "thresholds": grid_thresholds(ratio[rows, origin])})
        conformal[name] = entries

    venn = []
    two_class = np.stack([count[:, HUMAN], count[:, AI] + count[:, MIXED]], axis=1)  # (bins, 2)
    for members in merge_bins(two_class, min_rows):
        rows = np.isin(bins, members)
        venn.append({"bins": members, "n": int(rows.sum()), **venn_abers_blocks(score[rows], labels[rows] != HUMAN)})

    softmax = []
    log_ratio = np.log(np.maximum(ratio, LOG_FLOOR))  # (n, 3)
    for members in merge_bins(count, min_rows):
        rows = np.isin(bins, members)
        slope, bias = fit_three_way(log_ratio[rows], labels[rows])
        softmax.append({"bins": members, "n": int(rows.sum()), "slope": slope, "bias": bias})

    spread, closeness = [], []
    for members in merge_bins(count.sum(1), min_rows):
        rows = np.isin(bins, members)
        spread.append({"bins": members, "n": int(rows.sum()),
                       **{name: np.quantile(readings.spread[rows, origin], SPREAD_LEVELS).tolist()
                          for origin, name in enumerate(ORIGINS)}})
        closeness.append({"bins": members, "n": int(rows.sum()), "thresholds": grid_thresholds(readings.closeness[rows])})

    table = {
        "format": FORMAT, "version": VERSION, "origins": list(ORIGINS), "length_edges": list(LENGTH_EDGES),
        "length_bins": length_bin_names(), "tokens": "ModernBERT tokens of the text without special tokens",
        "levels": list(LEVELS), "grid": list(GRID), "spread_levels": list(SPREAD_LEVELS), "top_k": TOP_K,
        "out_of_reference_level": OUT_OF_REFERENCE_LEVEL, "min_rows": min_rows, "calibration_rows": int(len(labels)),
        "conformity": "r_y = mean_y / (mean_human + mean_ai + mean_mixed)",
        "conformal": conformal, "probability": venn, "three_way": softmax, "spread": spread, "closeness": closeness,
    }

    return _rounded(table)


def save_table(table: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, separators=(",", ":")) + "\n", encoding="utf-8")


def load_table(path: Path) -> dict[str, Any]:
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("format") != FORMAT or table.get("version") != VERSION:
        raise ValueError(f"{path} is not a {FORMAT} table of version {VERSION}")

    return table


def _per_group(entries: list[dict[str, Any]], bins: np.ndarray) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Each group's entry with the rows whose length bin it holds."""
    # bins: (n,)
    lookup = group_of_bins([entry["bins"] for entry in entries], len(LENGTH_EDGES))  # (bins,)

    return [(entry, lookup[bins] == index) for index, entry in enumerate(entries)]  # (n,) masks


def apply_calibration(table: dict[str, Any], readings: Readings) -> Verdicts:
    """Every calibrated output for the readings, from the table alone."""
    if table["length_edges"] != list(LENGTH_EDGES) or table["grid"] != list(GRID):
        raise ValueError("the table was fitted with other length bins or another p-value grid")
    n = len(readings.tokens)
    bins = length_bins(readings.tokens)  # (n,)
    ratio = three_way(readings.mean)  # (n, 3)
    score = 1.0 - ratio[:, HUMAN]  # (n,)

    p_values = np.ones((n, len(ORIGINS)))  # (n, 3)
    for origin, name in enumerate(ORIGINS):
        for entry, rows in _per_group(table["conformal"][name], bins):
            p_values[rows, origin] = grid_p_values(ratio[rows, origin], entry["thresholds"])
    sets = {level: p_values > level for level in table["levels"]}

    probability, interval = np.zeros(n), np.zeros((n, 2))  # (n,), (n, 2)
    for entry, rows in _per_group(table["probability"], bins):
        block = np.searchsorted(np.asarray(entry["edges"]), score[rows], side="right")  # (m,)
        p0, p1 = np.asarray(entry["p0"])[block], np.asarray(entry["p1"])[block]  # (m,) each
        probability[rows] = p1 / (1.0 - p0 + p1)
        interval[rows] = np.stack([p0, p1], axis=1)

    calibrated = np.zeros((n, len(ORIGINS)))  # (n, 3)
    log_ratio = np.log(np.maximum(ratio, LOG_FLOOR))  # (n, 3)
    for entry, rows in _per_group(table["three_way"], bins):
        logits = entry["slope"] * log_ratio[rows] + np.asarray(entry["bias"])  # (m, 3)
        logits -= logits.max(1, keepdims=True)
        calibrated[rows] = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

    spread_percentile = np.zeros((n, len(ORIGINS)))  # (n, 3)
    for entry, rows in _per_group(table["spread"], bins):
        for origin, name in enumerate(ORIGINS):
            spread_percentile[rows, origin] = np.interp(readings.spread[rows, origin], entry[name], table["spread_levels"])

    closeness_p = np.ones(n)  # (n,)
    for entry, rows in _per_group(table["closeness"], bins):
        closeness_p[rows] = grid_p_values(readings.closeness[rows], entry["thresholds"])

    return Verdicts(bins, p_values, sets, probability, interval, calibrated, spread_percentile, closeness_p,
                    closeness_p <= table["out_of_reference_level"])


def section_uncertainty(table: dict[str, Any], sections: Readings, level: float = 0.05) -> SectionUncertainty:
    """A document's uncertainty from its sections' readings (the engine reads each section from the same pass)."""
    if level not in table["levels"]:
        raise ValueError(f"level {level} is not one the table carries: {table['levels']}")
    verdicts = apply_calibration(table, sections)
    weights = sections.tokens.astype(np.float64) / max(1, sections.tokens.sum())  # (s,)
    mean = float((weights * verdicts.probability).sum())
    disagreement = float(np.sqrt((weights * (verdicts.probability - mean) ** 2).sum()))
    members = verdicts.sets[level]  # (s, 3)
    not_human = ~members[:, HUMAN]  # (s,)
    human_only = members[:, HUMAN] & ~members[:, AI] & ~members[:, MIXED]  # (s,)
    model_ruled_out = ~members[:, AI] & ~members[:, MIXED]  # (s,)
    # Two different sections: one called not human, another ruling out both kinds of model text.
    first, second = np.flatnonzero(not_human), np.flatnonzero(model_ruled_out)
    conflict = bool(len(first) and len(second) and not (len(first) == len(second) == 1 and first[0] == second[0]))

    return SectionUncertainty(mean, disagreement, float(verdicts.probability.min()), float(verdicts.probability.max()),
                              int(not_human.sum()), int(human_only.sum()), conflict)


def merge_p_values(p_values: np.ndarray) -> np.ndarray:
    """Valid p-values of `C` chunks of one document merged into one, under any dependence: (C, k) to (k,).

    One chunk keeps its own. Two or more give `min(1, 2 * mean)`: twice the arithmetic mean of p-values is a p-value
    whatever their joint distribution, and no smaller factor is enough in general (Ruschendorf 1982; Meng 1993;
    Vovk and Wang, "Combining p-values via averaging", Biometrika 107(4), 2020, arXiv 1212.4966). Chunks of one
    document resemble each other, so no rule that assumes independence (Fisher's, a product) may be used.
    """
    # p_values: (C, k)
    if len(p_values) == 1:
        return p_values[0].copy()  # (k,)

    return np.minimum(1.0, MERGE_FACTOR * p_values.mean(axis=0))  # (k,)


def combine_chunks(table: dict[str, Any], verdicts: Verdicts, level: float = 0.05) -> ChunkCombination:
    """A document's verdict from its `C` chunks' verdicts (CALIBRATION.md, Long documents); `level` is the
    mixed-document signal's family level, `C = 1` returns that chunk's own values."""
    if level not in table["levels"]:
        raise ValueError(f"level {level} is not one the table carries: {table['levels']}")
    chunks = len(verdicts.probability)
    if chunks == 0:
        raise ValueError("a document has at least one chunk")
    p_values = merge_p_values(verdicts.p_values)  # (3,)
    sets = {float(each): p_values > each for each in table["levels"]}
    probability = float(verdicts.probability.mean())
    disagreement = float(verdicts.probability.std())
    low, high = verdicts.interval.mean(axis=0)  # the mean of p0 and of p1
    interval = np.clip([min(low, probability - disagreement), max(high, probability + disagreement)], 0.0, 1.0)  # (2,)
    closeness_p = float(merge_p_values(verdicts.closeness_p_value[:, None])[0])
    chunk_level = level / chunks
    model = verdicts.p_values[:, HUMAN] <= chunk_level  # (C,)
    human = (verdicts.p_values[:, AI] <= chunk_level) & (verdicts.p_values[:, MIXED] <= chunk_level)  # (C,)
    first, second = np.flatnonzero(model), np.flatnonzero(human)
    mixed_document = bool(len(first) and len(second) and not (len(first) == len(second) == 1 and first[0] == second[0]))

    return ChunkCombination(chunks, p_values, sets, probability, interval, disagreement, float(verdicts.probability.min()),
                            float(verdicts.probability.max()), verdicts.three_way.mean(axis=0), closeness_p,
                            closeness_p <= table["out_of_reference_level"], chunk_level, int(model.sum()), int(human.sum()),
                            mixed_document)


def chunk_uncertainty(table: dict[str, Any], chunks: Readings, level: float = 0.05) -> ChunkCombination:
    """`combine_chunks` of the chunks' readings, each calibrated as a text of its own length."""
    return combine_chunks(table, apply_calibration(table, chunks), level)


def _reliability(probability: np.ndarray, positive: np.ndarray) -> list[tuple[int, float, float, int]]:
    """Equal-mass bins of the probability: (bin, mean probability, observed frequency, rows)."""
    # probability, positive: (n,)
    order = np.argsort(probability, kind="stable")  # (n,)
    rows = []
    for index, members in enumerate(np.array_split(order, RELIABILITY_BINS)):
        if len(members):
            rows.append((index, float(probability[members].mean()), float(positive[members].mean()), len(members)))

    return rows


def uncertainty_rows(verdicts: Verdicts, labels: np.ndarray, strata: dict[str, np.ndarray], split: str) -> list[dict[str, Any]]:
    """The uncertainty's own scorecard, tidy: coverage, set size, false positive rate on human text, calibration.

    One row per (stratum kind, stratum, metric, level or reliability bin). `strata` maps a kind (`format`,
    `language`, `generator`, `domain`, ...) to each text's value; `all` and `length_bin` are added here.
    """
    # labels: (n,); strata: (n,) each
    from foundry.metrics import expected_calibration_error

    labels = np.asarray(labels).astype(np.int64)  # (n,)
    positive = labels != HUMAN  # (n,)
    names = length_bin_names()
    kinds = {"all": np.full(len(labels), "all", dtype=object),
             "length_bin": np.array([names[value] for value in verdicts.length_bin], dtype=object), **strata}
    rows: list[dict[str, Any]] = []

    def add(kind: str, value: str, metric: str, number: float, count: int, level: float | str = "", bin_: int | str = "") -> None:
        rows.append({"split": split, "stratum_kind": kind, "stratum": value, "metric": metric, "level": level,
                     "bin": bin_, "value": number, "n": count})

    for kind, values in kinds.items():
        for value in np.unique(values.astype(str)):
            members = values.astype(str) == value  # (n,)
            count = int(members.sum())
            truth = labels[members]
            for level, contained in verdicts.sets.items():
                held = contained[members]  # (m, 3)
                size = held.sum(1)  # (m,)
                add(kind, value, "coverage", float(held[np.arange(count), truth].mean()), count, level)
                add(kind, value, "mean_set_size", float(size.mean()), count, level)
                add(kind, value, "empty_set_share", float((size == 0).mean()), count, level)
                add(kind, value, "multi_origin_share", float((size > 1).mean()), count, level)
                human = truth == HUMAN
                if human.any():
                    add(kind, value, "human_false_positive_rate", float((~held[human, HUMAN]).mean()), int(human.sum()), level)
                if (~human).any():
                    add(kind, value, "not_human_recall", float((~held[~human, HUMAN]).mean()), int((~human).sum()), level)
            probability = verdicts.probability[members]
            add(kind, value, "expected_calibration_error",
                float(expected_calibration_error(positive[members].astype(np.int64), probability, RELIABILITY_BINS)), count)
            add(kind, value, "brier", float(((probability - positive[members]) ** 2).mean()), count)
            add(kind, value, "mean_interval_width", float(np.diff(verdicts.interval[members], axis=1).mean()), count)
            add(kind, value, "out_of_reference_share", float(verdicts.out_of_reference[members].mean()), count)
            if kind in ("all", "length_bin"):
                for index, mean, observed, size in _reliability(probability, positive[members]):
                    add(kind, value, "reliability_mean_probability", mean, size, bin_=index)
                    add(kind, value, "reliability_observed_frequency", observed, size, bin_=index)

    return rows


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Rows of one schema to CSV, appending to an existing file of the same columns."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
