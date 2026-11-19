from __future__ import annotations

import numpy as np
import pyarrow as pa

from litmus_corpus.balance import BALANCE_REASONS, Rows, caps_and_balance
from litmus_corpus.config import LENGTH_BINS, BuildConfig


def rows_of(labels: list[int], generators: list[str], domains: list[str], formats: list[str], tokens: list[int],
            benchmark: list[bool] | None = None, first: list[bool] | None = None, long_eval: list[bool] | None = None) -> Rows:
    count = len(labels)
    return Rows(
        label=np.array(labels, dtype=np.int8),
        source=pa.array(["s"] * count),
        generator=pa.array(generators),
        domain=pa.array(domains),
        mix=pa.array(["edit" if label == 2 else "" for label in labels]),
        format=pa.array(formats),
        language=pa.array(["en"] * count),
        tokens=np.array(tokens, dtype=np.int32),
        benchmark=np.array(benchmark or [False] * count),
        first=np.array(first or [False] * count),
        long_eval=np.array(long_eval or [False] * count),
        text_key=np.random.default_rng(0).permutation(count).astype(np.uint64),
    )


def test_caps_are_lower_for_legacy_generators_and_keep_first_corpus_rows() -> None:
    labels = [1] * 30 + [1] * 30 + [0] * 60
    generators = ["gpt-4o"] * 30 + ["gpt-2"] * 30 + ["human"] * 60
    first = [False] * 20 + [True] * 10 + [False] * 90
    rows = rows_of(labels, generators, ["web"] * 120, ["plain"] * 120, [100] * 120, first=first)
    config = BuildConfig(ai_cap=20, legacy_ai_cap=5, human_cap=1000, balance_ratio=10.0)

    codes, _ = caps_and_balance(rows, config)

    kept = codes == 0
    assert kept[:30].sum() == 20 and kept[20:30].all()  # the first corpus's ten rows fill the cap first
    assert kept[30:60].sum() == 5 and kept[60:].sum() == 60


def test_balance_cuts_the_larger_class_per_cell_matching_domains() -> None:
    # Cell plain/100 tokens: 10 human (6 news, 4 web), 40 model (20 news, 20 web), 8 mixed.
    labels = [0] * 10 + [1] * 40 + [2] * 8
    domains = ["news"] * 6 + ["web"] * 4 + ["news"] * 20 + ["web"] * 20 + ["web"] * 8
    generators = ["human"] * 10 + ["gpt-4o"] * 40 + ["gpt-4o"] * 8
    # Cell code/100 tokens: model only.
    labels += [1] * 5
    domains += ["web"] * 5
    generators += ["gpt-4o"] * 5
    count = len(labels)
    rows = rows_of(labels, generators, domains, ["plain"] * 58 + ["code"] * 5, [100] * count, benchmark=[False] * (count - 1) + [True])

    codes, lines = caps_and_balance(rows, BuildConfig())

    kept = codes == 0
    model_kept = np.flatnonzero(kept[10:50]) + 10
    assert len(model_kept) == 10
    assert sorted(domains[index] for index in model_kept) == ["news"] * 6 + ["web"] * 4
    assert kept[50:58].sum() == 8 and (codes[50:58] == 0).all()  # mixed up to the per-class count, 10
    assert kept[58:62].sum() == 0 and kept[62]  # the model-only cell loses its rows; the benchmark row stays
    by_format = {cell["format"]: cell for cell in lines["cells"]}
    assert by_format["plain"]["after"] == {"human": 10, "ai": 10, "mixed": 8}
    assert by_format["code"]["status"] == "missing_human"


def test_first_corpus_rows_stay_past_the_balance_quota() -> None:
    labels = [0] * 5 + [1] * 20
    first = [False] * 5 + [True] * 8 + [False] * 12
    rows = rows_of(labels, ["human"] * 5 + ["gpt-4o"] * 20, ["web"] * 25, ["plain"] * 25, [100] * 25, first=first)

    codes, _ = caps_and_balance(rows, BuildConfig())

    assert (codes[5:13] == 0).all() and (codes[13:] == BALANCE_REASONS.index("balance")).all()


def test_token_budget_keeps_the_same_share_of_every_cell_in_a_bin() -> None:
    # Bin 64-127 tokens: 40 human and 40 model rows of 100 tokens (8,000 tokens) in two formats; bin 8-15: 4 rows.
    labels = ([0] * 20 + [1] * 20) * 2 + [0, 1, 0, 1]
    formats = ["plain"] * 40 + ["list"] * 40 + ["plain"] * 4
    tokens = [100] * 80 + [10] * 4
    shares = tuple(0.5 if index == 3 else 0.05 for index in range(LENGTH_BINS))  # 4,000 tokens to bin 3, 500 to bin 0
    rows = rows_of(labels, ["human" if label == 0 else "gpt-4o" for label in labels], ["web"] * 84, formats, tokens)

    codes, lines = caps_and_balance(rows, BuildConfig(new_token_budget=8_000, bin_token_shares=shares))

    kept = codes == 0
    for start in (0, 20, 40, 60):
        assert kept[start : start + 20].sum() == 10  # half of every (cell, origin)
    assert kept[80:].all()
    budget = {line["tokens"]: line for line in lines["token_budget"]}
    assert budget["64-127"]["new_tokens_before"] == 8_000 and budget["64-127"]["new_tokens_after"] == 4_000
    assert (codes[~kept] == BALANCE_REASONS.index("token_budget")).all()


def test_long_documents_are_matched_by_origin_and_capped() -> None:
    # Bin 16,384-32,767: 30 human and 3 model documents; bin 65,536-131,072: 10 human, 0 model.
    labels = [0] * 30 + [1] * 3 + [0] * 10 + [1] * 4
    tokens = [20_000] * 33 + [100_000] * 10 + [100] * 4
    long_eval = [True] * 43 + [False] * 4
    labels_count = len(labels)
    rows = rows_of(labels, ["human" if label == 0 else "gpt-4o" for label in labels], ["web"] * labels_count, ["plain"] * labels_count, tokens,
                   long_eval=long_eval)

    codes, lines = caps_and_balance(rows, BuildConfig(long_eval_cap=8))

    kept = codes == 0
    assert kept[:30].sum() == 2 and kept[30:33].sum() == 2  # matched at min(30, 3, 8 / 4 bins)
    assert not kept[33:43].any()  # no model document to match
    by_bin = {line["tokens"]: line for line in lines["long_eval"]}
    assert by_bin["16384-32767"]["before"] == {"human": 30, "ai": 3, "mixed": 0}
    assert by_bin["16384-32767"]["after"] == {"human": 2, "ai": 2, "mixed": 0}
    assert all(cell["tokens"] == "64-127" for cell in lines["cells"])  # long documents are in no balance cell
