"""Stage 6, caps, balance and the token budget.

The first corpus's rows all stay: they are counted in every cap, cell and report below, but never cut, and
where a cut must choose they come first. Benchmark rows are neither capped nor balanced. Every other row is
cut in this order, each cut choosing rows by the hash of their text, so it is random and a rerun makes the
same one:

Caps: a (source, generator) group of model text keeps at most `ai_cap` rows (`legacy_ai_cap` for a generator
of 2023 or earlier), a (source, domain) group of human text `human_cap`, and a (source, mix) group of mixed
text `mixed_cap`.

Balance: in each cell of (format, language group, length bin) the larger of human and model text is cut to
`balance_ratio` times the smaller. The cut goes domain by domain toward the smaller class's domain counts,
then fills what is left at random, so the classes match in domain as far as the cell allows. A cell where one
class is missing loses the other. Mixed text keeps at most as many rows in a cell as each class kept there.

Token budget: the new rows of each length bin may hold `bin_token_shares[bin]` of `new_token_budget` tokens.
A bin over its share keeps the same fraction of the new rows of every cell and origin in it, so the budget
leaves the balance as it found it. A bin under its share leaves the rest unspent.

Long documents (`long_eval`, 8,193 to 131,072 tokens) are evaluation rows apart from all of the above: in each
long bin, human and model text keep as many rows as the thinner of the two has, at most `long_eval_cap` over
the number of long bins, chosen domain by domain; mixed text keeps at most as many.

Length bins double from 8 tokens: [8, 16), [16, 32), ..., [4,096, 8,192].
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from dataclasses import dataclass
from typing import TypedDict

from .config import LENGTH_BIN_LABELS, LENGTH_BINS, LONG_BIN_LABELS, LONG_BINS, MAX_TOKENS, MIN_TOKENS, ORIGIN_LABELS, BuildConfig, language_group
from .generators import is_legacy


BALANCE_REASONS = ("keep", "cap", "balance", "mixed_balance", "token_budget", "long_eval_cap")
HUMAN, AI, MIXED = ORIGIN_LABELS["human"], ORIGIN_LABELS["ai"], ORIGIN_LABELS["mixed"]
CLASS_NAMES = {HUMAN: "human", AI: "ai", MIXED: "mixed"}


class Cell(TypedDict):
    """One cell of (format, language group, length bin) in the balance report: rows by origin before and after."""

    format: str
    language_group: str
    tokens: str
    before: dict[str, int]
    after: dict[str, int]
    status: str  # ok, thin, missing_human, missing_ai or mixed_only


class BinBudget(TypedDict):
    """One length bin's spend of the new-token budget."""

    tokens: str
    budget: int
    new_tokens_before: int
    new_tokens_after: int
    first_corpus_tokens: int
    kept_share: float


class LongBin(TypedDict):
    """One long bin's evaluation documents by origin, before and after the long-document cap."""

    tokens: str
    before: dict[str, int]
    after: dict[str, int]


@dataclass(frozen=True)
class Rows:
    """The columns caps and balance read, one entry a row of the corpus so far."""

    label: np.ndarray  # (n,) int8
    source: pa.Array
    generator: pa.Array
    domain: pa.Array
    mix: pa.Array
    format: pa.Array
    language: pa.Array
    tokens: np.ndarray  # (n,) int32
    benchmark: np.ndarray  # (n,) bool
    first: np.ndarray  # (n,) bool, a first-corpus row
    long_eval: np.ndarray  # (n,) bool
    text_key: np.ndarray  # (n,) uint64, the text's hash

    @property
    def rank(self) -> np.ndarray:
        """The order cuts keep rows in: first-corpus rows, then the rest, each by text hash."""
        return (self.text_key >> np.uint64(1)) | (np.uint64(1) << np.uint64(63)) * (~self.first).astype(np.uint64)  # (n,)


def codes_of(array: pa.Array) -> tuple[np.ndarray, list[str]]:
    """Integer codes of a string column, and the strings they stand for."""
    encoded = pc.dictionary_encode(array).combine_chunks() if isinstance(array, pa.ChunkedArray) else pc.dictionary_encode(array)

    return encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64), encoded.dictionary.to_pylist()  # (n,)


def length_bins(tokens: np.ndarray) -> np.ndarray:
    """The length bin of each training row (rows out of range are clipped to the end bins)."""
    # tokens: (n,)
    return np.clip(np.floor(np.log2(np.maximum(tokens, 1))).astype(np.int64) - 3, 0, LENGTH_BINS - 1)  # (n,)


def long_bins(tokens: np.ndarray) -> np.ndarray:
    # tokens: (n,)
    return np.clip(np.floor(np.log2(np.maximum(tokens, 1))).astype(np.int64) - 13, 0, LONG_BINS - 1)  # (n,)


def lowest_ranked(rows: np.ndarray, rank: np.ndarray, count: int) -> np.ndarray:
    """The `count` rows of lowest rank."""
    # rows: (m,); rank: (n,)
    if count >= len(rows):
        return rows  # (m,)
    return rows[np.argsort(rank[rows], kind="stable")[: max(count, 0)]]  # (count,)


def cap_codes(rows: Rows, config: BuildConfig) -> np.ndarray:
    """Code 1 (`cap`) for every new training row past its group's cap; first-corpus rows count but stay."""
    generator_codes, generators = codes_of(rows.generator)
    source_codes, _ = codes_of(rows.source)
    domain_codes, _ = codes_of(rows.domain)
    mix_codes, _ = codes_of(rows.mix)
    legacy = np.array([is_legacy(name) for name in generators], dtype=bool)[generator_codes]  # (n,)
    second = np.where(rows.label == AI, generator_codes, np.where(rows.label == HUMAN, domain_codes, mix_codes))  # (n,)
    caps = np.where(
        rows.label == AI, np.where(legacy, config.legacy_ai_cap, config.ai_cap), np.where(rows.label == HUMAN, config.human_cap, config.mixed_cap)
    )  # (n,)
    capped = ~rows.benchmark & ~rows.long_eval  # (n,)
    order = np.lexsort((rows.rank, second, source_codes, rows.label, ~capped))  # (n,)
    group = np.stack([(~capped).astype(np.int64), rows.label.astype(np.int64), source_codes, second])[:, order]  # (4, n)
    starts = np.r_[True, (group[:, 1:] != group[:, :-1]).any(axis=0)]  # (n,)
    start_positions = np.flatnonzero(starts)  # (g,)
    group_index = np.cumsum(starts) - 1  # (n,)
    position = np.arange(len(order)) - start_positions[group_index]  # (n,)
    codes = np.zeros(len(order), dtype=np.uint8)  # (n,)
    codes[order] = np.where(position < caps[order], 0, 1)
    codes[~capped | rows.first] = 0

    return codes  # (n,)


def matched_selection(large: np.ndarray, small: np.ndarray, quota: int, domains: np.ndarray, rank: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """`quota` rows of the larger class: its `fixed` rows (which always stay, even past the quota), then domain by
    domain toward the smaller class's domain counts, then at random."""
    # large: (l,); small: (s,); domains, rank: (n,); fixed: (n,) bool
    kept = large[fixed[large]]  # (f,)
    free = large[~fixed[large]]  # (l - f,)
    room = quota - len(kept)
    if room <= 0 or len(small) == 0:
        return kept  # (f,)
    small_domains, small_counts = np.unique(domains[small], return_counts=True)  # (d,), (d,)
    chosen: list[np.ndarray] = []
    for domain, count in zip(small_domains, small_counts, strict=True):
        chosen.append(lowest_ranked(free[domains[free] == domain], rank, int(count * room // len(small))))
    taken = np.concatenate(chosen) if chosen else free[:0]  # (t,)
    filler = lowest_ranked(free[~np.isin(free, taken)], rank, room - len(taken))  # (room - t,)

    return np.concatenate([kept, taken, filler])  # (quota,)


def cell_status(human: int, model: int, config: BuildConfig) -> str:
    if not human and not model:
        return "mixed_only"
    if not human or not model:
        return "missing_human" if not human else "missing_ai"
    if min(human, model) < config.thin_rows or min(human, model) < config.thin_share * max(human, model):
        return "thin"

    return "ok"


def cell_keys(rows: Rows) -> tuple[np.ndarray, list[str], list[str]]:
    """Each row's cell of (format, language group, length bin) as one integer, and the format and group names."""
    format_codes, formats = codes_of(rows.format)
    language_codes, languages = codes_of(rows.language)
    groups_of_languages = [language_group(language) for language in languages]
    group_names = sorted(set(groups_of_languages))
    group_codes = np.array([group_names.index(group) for group in groups_of_languages], dtype=np.int64)[language_codes]  # (n,)

    return (format_codes * len(group_names) + group_codes) * LENGTH_BINS + length_bins(rows.tokens), formats, group_names  # (n,)


def groups(keys: np.ndarray, members: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """The members grouped by key, in key order."""
    # keys: (n,); members: (c,)
    order = np.argsort(keys[members], kind="stable")  # (c,)
    sorted_keys = keys[members][order]  # (c,)
    boundaries = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])  # (k + 1,)

    return [(int(sorted_keys[first]), members[order[first:last]]) for first, last in zip(boundaries[:-1], boundaries[1:], strict=True)]  # k of (m_k,)


def balance_codes(rows: Rows, kept: np.ndarray, config: BuildConfig) -> tuple[np.ndarray, list[Cell]]:
    """Code 2 (`balance`) or 3 (`mixed_balance`) for rows a cell cuts, and a line per cell for the report."""
    # kept: (n,) rows still in after the caps
    keys, formats, group_names = cell_keys(rows)  # (n,)
    domain_codes, _ = codes_of(rows.domain)
    rank = rows.rank  # (n,)
    codes = np.zeros(len(rows.label), dtype=np.uint8)  # (n,)
    cells: list[Cell] = []
    for key, members in groups(keys, np.flatnonzero(kept & ~rows.benchmark & ~rows.long_eval)):
        human = members[rows.label[members] == HUMAN]
        model = members[rows.label[members] == AI]
        mixed = members[rows.label[members] == MIXED]
        smaller, larger = (human, model) if len(human) <= len(model) else (model, human)
        quota = min(len(larger), int(len(smaller) * config.balance_ratio))
        kept_larger = matched_selection(larger, smaller, quota, domain_codes, rank, rows.first) if quota < len(larger) else larger
        codes[np.setdiff1d(larger, kept_larger)] = 2
        per_class = min(len(smaller), len(kept_larger))
        kept_mixed = np.union1d(lowest_ranked(mixed, rank, per_class), mixed[rows.first[mixed]])
        codes[np.setdiff1d(mixed, kept_mixed)] = 3
        human_after = len(human) if larger is model else len(kept_larger)
        model_after = len(model) if larger is human else len(kept_larger)
        bin_index = key % LENGTH_BINS
        cells.append(
            {
                "format": formats[key // LENGTH_BINS // len(group_names)],
                "language_group": group_names[key // LENGTH_BINS % len(group_names)],
                "tokens": LENGTH_BIN_LABELS[bin_index],
                "before": {"human": len(human), "ai": len(model), "mixed": len(mixed)},
                "after": {"human": human_after, "ai": model_after, "mixed": len(kept_mixed)},
                "status": cell_status(len(human), len(model), config),
            }
        )

    return codes, cells  # (n,), cells


def budget_codes(rows: Rows, kept: np.ndarray, config: BuildConfig) -> tuple[np.ndarray, list[BinBudget]]:
    """Code 4 (`token_budget`) for new rows past their bin's share of the new-token budget, and a line per bin."""
    # kept: (n,) rows still in after caps and balance
    assert len(config.bin_token_shares) == LENGTH_BINS, "one budget share a length bin"
    keys, _, _ = cell_keys(rows)  # (n,)
    bins = keys % LENGTH_BINS  # (n,)
    training = kept & ~rows.benchmark & ~rows.long_eval  # (n,)
    new = training & ~rows.first  # (n,)
    tokens = rows.tokens.astype(np.int64)  # (n,)
    rank = rows.rank  # (n,)
    codes = np.zeros(len(rows.label), dtype=np.uint8)  # (n,)
    lines: list[BinBudget] = []
    for bin_index in range(LENGTH_BINS):
        in_bin = bins == bin_index  # (n,)
        budget = int(config.bin_token_shares[bin_index] * config.new_token_budget)
        spent = int(tokens[new & in_bin].sum())
        share = min(1.0, budget / spent) if spent else 1.0
        if share < 1.0:
            for _, members in groups(keys * len(CLASS_NAMES) + rows.label, np.flatnonzero(new & in_bin)):
                codes[np.setdiff1d(members, lowest_ranked(members, rank, int(share * len(members))))] = 4
        lines.append(
            {
                "tokens": LENGTH_BIN_LABELS[bin_index],
                "budget": budget,
                "new_tokens_before": spent,
                "new_tokens_after": int(tokens[new & in_bin & (codes == 0)].sum()),
                "first_corpus_tokens": int(tokens[training & rows.first & in_bin].sum()),
                "kept_share": round(share, 4),
            }
        )

    return codes, lines  # (n,), lines


def long_eval_codes(rows: Rows, config: BuildConfig) -> tuple[np.ndarray, list[LongBin]]:
    """Code 5 (`long_eval_cap`) for long documents past their bin's matched count, and a line per long bin."""
    domain_codes, _ = codes_of(rows.domain)
    rank = rows.rank  # (n,)
    bins = long_bins(rows.tokens)  # (n,)
    per_bin = config.long_eval_cap // LONG_BINS
    codes = np.zeros(len(rows.label), dtype=np.uint8)  # (n,)
    lines: list[LongBin] = []
    candidates = rows.long_eval & ~rows.benchmark  # (n,)
    for bin_index in range(LONG_BINS):
        members = np.flatnonzero(candidates & (bins == bin_index))  # (m,)
        human = members[rows.label[members] == HUMAN]
        model = members[rows.label[members] == AI]
        mixed = members[rows.label[members] == MIXED]
        matched = min(len(human), len(model), per_bin)
        kept_human = matched_selection(human, model, matched, domain_codes, rank, rows.first)
        kept_model = matched_selection(model, human, matched, domain_codes, rank, rows.first)
        kept_mixed = np.union1d(lowest_ranked(mixed, rank, min(matched, per_bin)), mixed[rows.first[mixed]])
        for every, kept in ((human, kept_human), (model, kept_model), (mixed, kept_mixed)):
            codes[np.setdiff1d(every, kept)] = 5
        lines.append(
            {
                "tokens": LONG_BIN_LABELS[bin_index],
                "before": {"human": len(human), "ai": len(model), "mixed": len(mixed)},
                "after": {"human": len(kept_human), "ai": len(kept_model), "mixed": len(kept_mixed)},
            }
        )

    return codes, lines  # (n,), lines


def caps_and_balance(rows: Rows, config: BuildConfig) -> tuple[np.ndarray, dict[str, list[Cell] | list[BinBudget] | list[LongBin]]]:
    """The drop code of every row (`BALANCE_REASONS`), and the report's lines per cell, budget bin and long bin."""
    unbinned = ~rows.benchmark & ~rows.long_eval & ~rows.first & ((rows.tokens < MIN_TOKENS) | (rows.tokens > MAX_TOKENS))  # (n,)
    assert not unbinned.any(), f"{int(unbinned.sum())} new training rows outside 8 to 8,192 tokens reached balance"
    codes = cap_codes(rows, config)  # (n,)
    balance, cells = balance_codes(rows, codes == 0, config)  # (n,)
    codes = np.where(codes == 0, balance, codes).astype(np.uint8)  # (n,)
    budget, bins = budget_codes(rows, codes == 0, config)  # (n,)
    codes = np.where(codes == 0, budget, codes).astype(np.uint8)  # (n,)
    long_codes, long_lines = long_eval_codes(rows, config)  # (n,)
    codes = np.where(rows.long_eval, long_codes, codes).astype(np.uint8)  # (n,)
    assert not (codes[rows.first | rows.benchmark] > 0).any(), "a first-corpus or benchmark row was cut"

    return codes, {"cells": cells, "token_budget": bins, "long_eval": long_lines}  # (n,), lines
