from __future__ import annotations

import numpy as np
import pytest

from litmus_corpus.checks import date_reason, degenerate_reason, duplicate_codes, global_duplicate_codes, opening
from litmus_corpus.holdout import HOLDOUT_REASONS, holdout_decision


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("   \n ", "empty"),
        ("12345 678 + 9", "no_letters"),
        ("}", "no_letters"),
        ("aaaaaaaaaa", "no_letters"),
        ("I love you so much. " * 200, "repetition"),
        ("\n".join(["same line again here"] * 6 + ["other"]), "repetition"),
        ("A normal sentence with words. Another one follows it.", None),
        ("def f(x):\n    return x + 1\n", None),
    ],
)
def test_degenerate(text: str, reason: str | None) -> None:
    assert degenerate_reason(text) == reason


def test_date_applies_to_human_text_only() -> None:
    assert date_reason(0, None) == "human_undated"
    assert date_reason(0, "2022-01-01") == "human_not_before_2022"
    assert date_reason(0, "2021-12-31") is None
    assert date_reason(1, None) is None


def test_opening_is_the_first_twenty_words() -> None:
    words = " ".join(f"Word{index}," for index in range(30))
    assert opening(words) == " ".join(f"word{index}" for index in range(20))
    assert len(opening("字" * 500)) == 160


def test_duplicates_keep_the_first_in_order_and_drop_keys_under_two_origins() -> None:
    keys = np.array([7, 7, 7, 9, 9, 11, 7], dtype=np.uint64)  # (7,)
    labels = np.array([1, 1, 1, 0, 1, 0, 1], dtype=np.int8)  # (7,)
    docs = np.array([1, 2, 3, 4, 5, 6, 2], dtype=np.uint64)  # (7,) the last row is a passage of the kept row's document
    order = np.array([2, 0, 1, 0, 1, 0, 3])  # (7,)

    codes = duplicate_codes(keys, labels, docs, order, copy_code=1, cross_code=3)

    assert codes.tolist() == [1, 0, 1, 3, 3, 0, 0]


def test_global_duplicates_leave_benchmark_rows_and_prefer_the_first_corpus() -> None:
    text_keys = np.array([1, 1, 2, 3], dtype=np.uint64)
    opening_keys = np.array([10, 10, 20, 20], dtype=np.uint64)
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    docs = np.arange(4, dtype=np.uint64)
    benchmark = np.array([False, True, False, False])

    codes = global_duplicate_codes(text_keys, opening_keys, labels, docs, benchmark, np.zeros(4, dtype=bool))

    assert codes[1] == 0 and codes[0] == 0  # the benchmark copy is not in the pool
    assert sorted(codes[2:].tolist()) == [0, 2]  # one of the two openings is a copy

    keys = np.arange(200, dtype=np.uint64) // 2  # 100 pairs of copies
    first = np.arange(200) % 2 == 1  # the second of each pair is a first-corpus row
    codes = global_duplicate_codes(keys, keys + np.uint64(1000), np.ones(200, dtype=np.int8), np.arange(200, dtype=np.uint64), np.zeros(200, dtype=bool), first)
    assert (codes[first] == 0).all() and (codes[~first] == 1).all()


def test_holdout_drops_new_rows_and_marks_first_corpus_rows_that_share_a_key_with_a_benchmark() -> None:
    # rows: 0 benchmark B; 1 hidden-label benchmark H; 2 shares B's doc; 3 shares B's text; 4 shares B's opening;
    # 5 first-corpus row sharing B's text; 6 first-corpus row linked to H by strata; 7 first-corpus row of B's source;
    # 8 unrelated.
    docs = np.array([1, 2, 1, 4, 5, 6, 7, 8, 9], dtype=np.uint64)
    text_keys = np.array([10, 20, 30, 10, 50, 10, 70, 80, 90], dtype=np.uint64)
    opening_keys = np.array([100, 200, 300, 400, 100, 600, 700, 800, 900], dtype=np.uint64)
    names = ("", "B", "H")
    benchmark_ids = np.array([1, 2, 0, 0, 0, 0, 0, 1, 0], dtype=np.int32)
    labels = np.array([1, 3, 1, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    first = np.array([False, False, False, False, False, True, True, True, False])

    codes, marks, ways = holdout_decision(
        docs, text_keys, opening_keys, benchmark_ids, labels, first, np.array([7], dtype=np.uint64), np.array([2], dtype=np.int32)
    )

    assert [HOLDOUT_REASONS[code] for code in codes] == [
        "keep", "benchmark_unlabeled", "benchmark_doc", "benchmark_text", "benchmark_opening", "keep", "keep", "keep", "keep"
    ]
    assert [names[mark] for mark in marks] == ["", "", "", "", "", "B", "H", "", ""]
    assert {way: [names[index] for index in ids] for way, ids in ways.items() if len(ids)} == {"source": ["B"], "link": ["H"], "text": ["B"]}
