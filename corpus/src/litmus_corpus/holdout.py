"""Stage 7, benchmark hold-out: benchmarks are for evaluation only, so no benchmark text reaches training.

A benchmark row keeps `benchmark` set and never joins another set. A new training row that shares a document,
a text or an opening of 20 words with a benchmark row is dropped. A first-corpus row is not dropped but marked:
it gets the benchmark's name in `benchmark` and leaves training, so the first corpus's rows stay verbatim and
countable. A first-corpus row is a benchmark's when its source is one the benchmark catalog holds out (set at
read), when strata linked a benchmark text to it by content, or when it shares a document, text or opening
with a benchmark row here. Rows of a benchmark whose labels are hidden (`unlabeled`) hold copies out and are
then dropped themselves.

It runs before caps and balance (the stage numbers are not the run order), so the
balance the corpus reports is the balance it ships.
"""

from __future__ import annotations

import numpy as np

from .config import ORIGIN_LABELS, UNLABELED


HOLDOUT_REASONS = ("keep", "benchmark_doc", "benchmark_text", "benchmark_opening", "benchmark_unlabeled")
MATCH_KINDS = ("doc", "text", "opening")  # in the order a match is taken; the code of kind k is k + 1
NO_MARK = 0


def key_owners(keys: np.ndarray, owner_keys: np.ndarray, owner_ids: np.ndarray) -> np.ndarray:
    """For each key, the id of an owner holding it (the first in owner order), or `NO_MARK`."""
    # keys: (n,); owner_keys, owner_ids: (m,)
    if len(owner_keys) == 0:
        return np.full(len(keys), NO_MARK, dtype=np.int32)  # (n,)
    unique_keys, first = np.unique(owner_keys, return_index=True)  # (u,), (u,)
    positions = np.minimum(np.searchsorted(unique_keys, keys), len(unique_keys) - 1)  # (n,)
    found = unique_keys[positions] == keys  # (n,)

    return np.where(found, owner_ids[first][positions], NO_MARK).astype(np.int32)  # (n,)


def holdout_decision(
    docs: np.ndarray,
    text_keys: np.ndarray,
    opening_keys: np.ndarray,
    benchmark_ids: np.ndarray,
    labels: np.ndarray,
    first: np.ndarray,
    linked_docs: np.ndarray,
    linked_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """The hold-out drop code of every row (`HOLDOUT_REASONS`), and the benchmark id each first-corpus row is marked
    with (`NO_MARK` for none).

    `benchmark_ids` numbers each row's benchmark (0 for none); `first` marks the first corpus's rows; `linked_docs`
    and `linked_ids` are strata's links of benchmark texts to first-corpus documents. Also returns, by way of
    marking (`source`, `link`, `doc`, `text`, `opening`), the benchmark id of every first-corpus row so marked.
    """
    # docs, text_keys, opening_keys, benchmark_ids, labels, first: (n,); linked_docs, linked_ids: (l,)
    codes = np.zeros(len(docs), dtype=np.uint8)  # (n,)
    marks = np.full(len(docs), NO_MARK, dtype=np.int32)  # (n,)
    benchmark = benchmark_ids != NO_MARK  # (n,)
    ways: dict[str, np.ndarray] = {"source": benchmark_ids[benchmark & first]}

    linked = key_owners(docs, linked_docs, linked_ids)  # (n,)
    by_link = first & ~benchmark & (linked != NO_MARK)  # (n,)
    marks[by_link] = linked[by_link]
    ways["link"] = linked[by_link]

    for kind, keys in zip(MATCH_KINDS, (docs, text_keys, opening_keys), strict=True):
        owners = key_owners(keys, keys[benchmark], benchmark_ids[benchmark])  # (n,)
        open_rows = ~benchmark & (codes == 0) & (marks == NO_MARK) & (owners != NO_MARK)  # (n,)
        marks[open_rows & first] = owners[open_rows & first]
        ways[kind] = owners[open_rows & first]
        codes[open_rows & ~first] = MATCH_KINDS.index(kind) + 1
    codes[benchmark & (labels == ORIGIN_LABELS[UNLABELED])] = len(HOLDOUT_REASONS) - 1

    return codes, marks, ways  # (n,), (n,), way to (k,)
