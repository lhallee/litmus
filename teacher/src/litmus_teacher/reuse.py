"""Reuse the first corpus's ModernBERT-base vectors where it embedded the same text the same way.

`lhallee/many-generator-detection-corpus` holds `embeddings/modernbert_base_mean_variance.npy`, float16
(5,171,342, 1,536): the mean then the variance of ModernBERT-base's last states over a passage's tokens, special
tokens included, at up to 4,096 tokens, made in bfloat16. That is this package's rule for the same encoder, so a row
whose text is byte-identical (`text_unchanged`) and short enough that the first corpus never truncated it
(`tokens + 2 <= 4,096`) may copy its vector from row `first_corpus_id` instead of being computed.

The pilot recomputes `CHECK_ROWS` reusable rows and compares them with the stored vectors. The verdict has three
outcomes:

- `safe`: the stored vectors pass as they are, and are copied as they are.
- `aligned`: they fail, but a per-column affine map fitted from stored to recomputed vectors on half the pairs makes
  the other half pass. Every copied row then goes through that map, which is recorded.
- `refused`: they fail even aligned; nothing is copied and every row is computed.

A comparison passes when all three rules hold. Reused rows come only from the first corpus, which holds no mixed text,
so anything a head can see that tells reused from computed rows is correlated with origin and would be a shortcut:

- **No visible shift.** A linear reader learns the direction of the mean paired difference on half the texts, in
  standardized units, and scores the other half's two versions unpaired. The ROC AUC of stored against recomputed
  must be at most `AUC_CEILING` (0.6). A uniform offset of 0.05 standard deviations a column reads about 0.92 at
  1,536 columns.
- **No visible change of spread.** The same reader on the squared standardized columns, also at most 0.6. A
  per-column affine alignment sets every column's mean difference to zero on the rows it was fitted on, so the
  first reader alone could not see a difference the alignment leaves behind. This one sees a change of scale or a
  non-affine distortion of a column, which the alignment cannot remove.
- **No extra noise.** For each of the mean and variance halves, the 99th percentile of the paired `1 - cosine` must
  be at most `NOISE_FACTOR` (2) times the same quantity between two of this package's own bfloat16 packings of the
  same texts.

The alignment is chosen among four forms by validation inside the fitting half: shift only, or shift and scale, either
for all lengths or separately per length group (`ALIGNMENT_LENGTH_EDGES`). The chosen form, and how much a per-length
fit helps, says whether the offset depends on token count. `bank --drift` then reads the same pairs, raw and aligned,
through the trained head, and reports how far the two-class reading moves.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REUSE_ENCODER = "modernbert_base"
FIRST_CORPUS_CONTEXT = 4096  # positions, special tokens included
REUSED_FILE = "reused.npy"
REUSE_MARKER = "reuse.done"
ALIGNMENT_FILE = "reuse_alignment.npz"
REUSE_COLUMNS = ("first_corpus_id", "text_unchanged")
CHECK_ROWS = 2000
AUC_CEILING = 0.6
NOISE_FACTOR = 2.0
COPY_CHUNK = 65_536
ALIGNMENT_LENGTH_EDGES = (0, 256, 1024)  # token edges of the per-length alignment groups; the last is open
ALIGNMENT_FORMS = (("shift", False), ("affine", False), ("shift", True), ("affine", True))
MIN_GROUP_ROWS = 50  # a per-length form is tried only when every group holds this many rows on each side
OUTCOMES = ("safe", "aligned", "refused")

DriftFunction = Callable[[np.ndarray, np.ndarray, int], dict[str, dict[str, float]]]


def reusable_rows(metadata: pa.Table, selected: np.ndarray, max_text_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """Selected rows whose vector can be copied, and the first corpus's row of each: two (r,) int64."""
    # selected: (n,) bool, indexed by id
    ids = metadata["id"].to_numpy()  # (n,)
    first = pc.fill_null(metadata["first_corpus_id"], -1).to_numpy().astype(np.int64)  # (n,)
    unchanged = pc.fill_null(metadata["text_unchanged"], False).to_numpy(zero_copy_only=False).astype(bool)  # (n,)
    tokens = metadata["tokens"].to_numpy()  # (n,)
    keep = selected[ids] & unchanged & (first >= 0) & (tokens + 2 <= FIRST_CORPUS_CONTEXT) & (tokens <= max_text_tokens)

    return ids[keep], first[keep]  # (r,), (r,)


def length_groups(tokens: np.ndarray) -> np.ndarray:
    """The alignment length group of each text: (r,) to (r,) in [0, len(ALIGNMENT_LENGTH_EDGES))."""
    # tokens: (r,)
    return np.searchsorted(np.asarray(ALIGNMENT_LENGTH_EDGES[1:]), tokens, side="right")  # (r,)


@dataclass(frozen=True)
class Alignment:
    """A per-column affine map from stored to recomputed vectors, one row of parameters per length group."""

    form: str  # "shift" or "affine"
    per_length: bool
    scale: np.ndarray  # (g, 2d)
    shift: np.ndarray  # (g, 2d)

    def apply(self, values: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Aligned float16 vectors, variances kept non-negative: (r, 2d), (r,) to (r, 2d)."""
        # values: (r, 2d); tokens: (r,)
        group = length_groups(tokens) if self.per_length else np.zeros(len(values), dtype=np.int64)  # (r,)
        aligned = values.astype(np.float32) * self.scale[group] + self.shift[group]  # (r, 2d)
        width = values.shape[1] // 2  # d
        aligned[:, width:] = np.maximum(aligned[:, width:], 0.0)
        out = aligned.astype(np.float16)  # (r, 2d)
        if not np.isfinite(out).all():
            raise FloatingPointError("an aligned vector left float16's range")

        return out  # (r, 2d)

    def save(self, path: Path) -> None:
        np.savez(path, form=self.form, per_length=self.per_length, scale=self.scale, shift=self.shift)

    @classmethod
    def load(cls, path: Path) -> Alignment:
        arrays = np.load(path)

        return cls(str(arrays["form"]), bool(arrays["per_length"]), arrays["scale"], arrays["shift"])

    def summary(self) -> dict[str, Any]:
        width = self.scale.shape[1] // 2
        halves = {"mean": slice(0, width), "variance": slice(width, 2 * width)}
        return {"form": self.form, "per_length": self.per_length, "length_edges": list(ALIGNMENT_LENGTH_EDGES),
                **{f"{name}_mean_abs_shift": float(np.abs(self.shift[:, columns]).mean()) for name, columns in halves.items()},
                **{f"{name}_mean_abs_scale_minus_1": float(np.abs(self.scale[:, columns] - 1).mean())
                   for name, columns in halves.items()}}


def fit_form(stored: np.ndarray, recomputed: np.ndarray, tokens: np.ndarray, form: str, per_length: bool) -> Alignment:
    """Least-squares per-column map `recomputed ~ scale * stored + shift`, scale fixed at 1 for `shift`."""
    # stored, recomputed: (m, 2d); tokens: (m,)
    stored, recomputed = stored.astype(np.float64), recomputed.astype(np.float64)  # (m, 2d) each
    groups = length_groups(tokens) if per_length else np.zeros(len(stored), dtype=np.int64)  # (m,)
    count = len(ALIGNMENT_LENGTH_EDGES) if per_length else 1
    scale = np.ones((count, stored.shape[1]))  # (g, 2d)
    shift = np.zeros((count, stored.shape[1]))  # (g, 2d)
    for group in range(count):
        rows = groups == group  # (m,)
        if not rows.any():
            continue
        x, y = stored[rows], recomputed[rows]  # (m_g, 2d) each
        if form == "affine":
            centered = x - x.mean(0)  # (m_g, 2d)
            variance = (centered**2).mean(0)  # (2d,)
            covariance = (centered * (y - y.mean(0))).mean(0)  # (2d,)
            scale[group] = np.where(variance > 0, covariance / np.where(variance > 0, variance, 1), 1.0)
        shift[group] = y.mean(0) - scale[group] * x.mean(0)

    return Alignment(form, per_length, scale.astype(np.float32), shift.astype(np.float32))


def standardized_error(aligned: np.ndarray, recomputed: np.ndarray, spread: np.ndarray) -> float:
    """Mean squared error per column in units of each column's spread: (m, 2d), (m, 2d), (2d,) to ()."""
    # aligned, recomputed: (m, 2d); spread: (2d,)
    return float((((aligned.astype(np.float64) - recomputed) / spread) ** 2).mean())


def fit_alignment(stored: np.ndarray, recomputed: np.ndarray, tokens: np.ndarray, seed: int) -> tuple[Alignment, dict[str, Any]]:
    """Choose the alignment form by validation inside these pairs, then fit it on all of them."""
    # stored, recomputed: (m, 2d); tokens: (m,)
    order = np.random.default_rng(seed).permutation(len(stored))  # (m,)
    learn, check = order[: len(order) // 2], order[len(order) // 2 :]  # (m / 2,) each
    spread = recomputed.astype(np.float64).std(0) + 1e-9  # (2d,)
    errors: dict[str, float] = {}
    for form, per_length in ALIGNMENT_FORMS:
        name = f"{form}{'_per_length' if per_length else ''}"
        if per_length and any(np.bincount(length_groups(tokens[side]), minlength=len(ALIGNMENT_LENGTH_EDGES)).min()
                              < MIN_GROUP_ROWS for side in (learn, check)):
            continue  # recorded as absent from `errors`: too few rows in some length group
        candidate = fit_form(stored[learn], recomputed[learn], tokens[learn], form, per_length)
        errors[name] = standardized_error(candidate.apply(stored[check], tokens[check]), recomputed[check], spread)
    errors["none"] = standardized_error(stored[check], recomputed[check], spread)
    chosen = min((name for name in errors if name != "none"), key=errors.get)
    form, per_length = chosen.removesuffix("_per_length"), chosen.endswith("_per_length")
    best_global = min(errors[name] for name in ("shift", "affine"))
    best_per_length = min((errors[name] for name in ("shift_per_length", "affine_per_length") if name in errors),
                          default=None)
    report = {"validation_error": errors, "chosen": chosen,
              # Below 1 when fitting each length group separately helps: the offset depends on token count.
              "per_length_error_ratio": None if best_per_length is None else best_per_length / best_global}

    return fit_form(stored, recomputed, tokens, form, per_length), report


def copy_reused(values: np.ndarray, tokens: np.ndarray, source_path: Path, ids: np.ndarray, first_ids: np.ndarray,
                row_tokens: np.ndarray, alignment: Alignment | None = None) -> None:
    """Copy the first corpus's vectors, aligned when given, into rows `ids` of the store, with their token counts."""
    # values: (n, 2d) memmap; tokens: (n,) memmap; ids, first_ids, row_tokens: (r,)
    source = np.load(source_path, mmap_mode="r")  # (n_first, 2d)
    if source.ndim != 2 or source.shape[1] != values.shape[1] or source.dtype != np.float16:
        raise ValueError(f"{source_path} is {source.dtype} {source.shape}; the store holds float16 rows of {values.shape[1]}")
    if len(first_ids) and (first_ids.min() < 0 or first_ids.max() >= len(source)):
        raise ValueError("a first_corpus_id falls outside the first corpus's embeddings")
    order = np.argsort(first_ids, kind="stable")  # (r,) reads in source order
    for begin in range(0, len(order), COPY_CHUNK):
        rows = order[begin : begin + COPY_CHUNK]  # (c,)
        block = np.asarray(source[first_ids[rows]])  # (c, 2d)
        if not np.isfinite(block).all():
            raise FloatingPointError("a stored first-corpus vector is not finite")
        if alignment is not None:
            block = alignment.apply(block, row_tokens[rows])  # (c, 2d)
        values[ids[rows]] = block
        tokens[ids[rows]] = row_tokens[rows].astype(np.int32)
    values.flush()
    tokens.flush()


def paired_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """ROC AUC of scores of positives against negatives, by ranks: (p,), (q,) to ()."""
    # positive: (p,); negative: (q,)
    scores = np.concatenate([positive, negative])  # (p + q,)
    ranks = np.empty(len(scores))  # (p + q,)
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)

    return float((ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def shift_auc(stored: np.ndarray, recomputed: np.ndarray, seed: int, squared: bool = False) -> float:
    """How well a linear reader tells stored from recomputed vectors of the same texts, on held-out texts.

    With `squared`, the reader sees each standardized column squared: a change of spread instead of a shift.
    """
    # stored, recomputed: (m, 2d), row i of each the same text
    stored, recomputed = stored.astype(np.float64), recomputed.astype(np.float64)  # (m, 2d) each
    both = np.concatenate([stored, recomputed])  # (2m, 2d)
    center, scale = both.mean(0), both.std(0) + 1e-9  # (2d,) each
    stored, recomputed = (stored - center) / scale, (recomputed - center) / scale  # (m, 2d) each
    if squared:
        stored, recomputed = stored**2, recomputed**2
    order = np.random.default_rng(seed).permutation(len(stored))  # (m,)
    learn, held = order[: len(order) // 2], order[len(order) // 2 :]  # (m / 2,) each
    direction = (stored[learn] - recomputed[learn]).mean(0)  # (2d,)

    return paired_auc(stored[held] @ direction, recomputed[held] @ direction)


def compare(stored: np.ndarray, recomputed: np.ndarray, packing_drift: dict[str, dict[str, float]],
            drift: DriftFunction, seed: int) -> dict[str, Any]:
    """The three rules on one set of pairs: (m, 2d) each, against the pilot's two-packing drift."""
    # stored, recomputed: (m, 2d)
    width = stored.shape[1] // 2
    measured = drift(recomputed, stored, width)
    aucs = {"shift_auc": shift_auc(stored, recomputed, seed), "spread_auc": shift_auc(stored, recomputed, seed, squared=True)}
    reasons = [f"{name} {value:.3f} > {AUC_CEILING}" for name, value in aucs.items() if value > AUC_CEILING]
    for half in ("mean", "variance"):
        key = f"{half}_one_minus_cosine"
        ceiling = NOISE_FACTOR * max(packing_drift[key]["p99"], 1e-7)
        if measured[key]["p99"] > ceiling:
            reasons.append(f"{half} half: p99 of 1 - cosine {measured[key]['p99']:.2e} > {ceiling:.2e}")

    return {"passes": not reasons, "reasons": reasons, **aucs, "drift": measured, "rows": int(len(stored))}


def decide_reuse(stored: np.ndarray, recomputed: np.ndarray, tokens: np.ndarray,
                 packing_drift: dict[str, dict[str, float]], drift: DriftFunction,
                 seed: int) -> tuple[dict[str, Any], Alignment | None, np.ndarray | None]:
    """`safe`, `aligned` or `refused`, with the evidence; the alignment fitted, and the pairs held out to judge it."""
    # stored, recomputed: (m, 2d); tokens: (m,)
    raw = compare(stored, recomputed, packing_drift, drift, seed)
    decision: dict[str, Any] = {"outcome": "safe", "raw": raw, "aligned": None, "alignment": None,
                                "auc_ceiling": AUC_CEILING, "noise_factor": NOISE_FACTOR}
    if raw["passes"]:
        return decision, None, None  # no alignment, no held pairs (m / 2,)
    order = np.random.default_rng(seed + 1).permutation(len(stored))  # (m,)
    fit, held = order[: len(order) // 2], order[len(order) // 2 :]  # (m / 2,) each
    alignment, fit_report = fit_alignment(stored[fit], recomputed[fit], tokens[fit], seed)
    aligned = compare(alignment.apply(stored[held], tokens[held]), recomputed[held], packing_drift, drift, seed)
    decision.update({"outcome": "aligned" if aligned["passes"] else "refused", "aligned": aligned,
                     "alignment": {**alignment.summary(), **fit_report}})

    return decision, alignment, held  # alignment and held (m / 2,) pairs, whatever the outcome


def vector_drift(first: np.ndarray, second: np.ndarray, width: int) -> dict[str, dict[str, float]]:
    """1 - cosine, relative L2 and mean absolute difference between two embeddings of the same texts, per half."""
    # first, second: (m, 2d)
    report: dict[str, dict[str, float]] = {}
    for half, columns in (("mean", slice(0, width)), ("variance", slice(width, 2 * width))):
        a, b = first[:, columns].astype(np.float64), second[:, columns].astype(np.float64)  # (m, d) each
        cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))  # (m,)
        relative = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)  # (m,)
        absolute = np.abs(a - b).mean(axis=1)  # (m,) mean absolute difference per text
        for name, values in (("one_minus_cosine", 1 - cosine), ("relative_l2", relative), ("mean_abs_difference", absolute)):
            report[f"{half}_{name}"] = {"median": float(np.median(values)), "p99": float(np.quantile(values, 0.99)),
                                        "max": float(values.max())}

    return report
