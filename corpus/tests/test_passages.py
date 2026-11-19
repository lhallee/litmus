from __future__ import annotations

import numpy as np
import pyarrow as pa

from litmus_corpus.config import LENGTH_BINS, MAX_TOKENS, MIN_TOKENS, BuildConfig
from litmus_corpus.passages import passage_spans, passages_table
from litmus_corpus.records import STANDARD_SCHEMA, STRATA_SCHEMA
from litmus_corpus.storage import Report
from litmus_corpus.strata import LinkPlan, strata_table
from litmus_corpus.tokenization import token_counts

from conftest import text_table


def test_spans_are_seeded_disjoint_and_never_the_whole() -> None:
    cut_tokens = list(range(0, 4000, 10))  # a start every 10 tokens
    spans = passage_spans(4000, cut_tokens, seed=12345, max_passages=4)

    assert spans == passage_spans(4000, cut_tokens, seed=12345, max_passages=4)
    assert spans != passage_spans(4000, cut_tokens, seed=54321, max_passages=4)
    assert 1 <= len(spans) <= 4
    boundaries = [*cut_tokens, 4000]
    ranges = sorted((boundaries[first], boundaries[end]) for first, end in spans)
    assert all(low < high for low, high in ranges)
    assert all(previous[1] <= current[0] for previous, current in zip(ranges, ranges[1:], strict=False))
    assert all(MIN_TOKENS <= high - low <= 2000 for low, high in ranges)


def test_short_text_gives_no_cut() -> None:
    assert passage_spans(12, [0, 6], seed=1, max_passages=4) == []


def test_targets_spread_evenly_in_the_logarithm() -> None:
    cut_tokens = list(range(0, MAX_TOKENS * 2, 4))
    lengths = []
    for seed in range(300):
        boundaries = [*cut_tokens, MAX_TOKENS * 2]
        lengths.extend(boundaries[end] - boundaries[first] for first, end in passage_spans(MAX_TOKENS * 2, cut_tokens, seed, 4))
    bins = np.bincount(np.floor(np.log2(lengths)).astype(int) - 3, minlength=LENGTH_BINS)  # (10,)

    assert MAX_TOKENS == 8_192 and LENGTH_BINS == 10
    assert len(lengths) > 600
    assert bins.min() > 0.3 * bins.mean()  # every doubling of length is drawn
    assert max(lengths) <= MAX_TOKENS


def test_passages_cut_at_boundaries_with_exact_tokens(tokenizer_path: str) -> None:
    sentences = [f"Sentence number {index} says something new." for index in range(200)]
    prose = " ".join(sentences)
    code = "\n".join(f"value_{index} = compute({index}) + offset" for index in range(150))  # under 8,192 fixture tokens
    strata, _ = strata_table(text_table([prose, code, "Tiny."], STANDARD_SCHEMA), tokenizer_path, None, LinkPlan(None, frozenset()), Report())
    report = Report(rows_in=3)

    passages = passages_table(strata, tokenizer_path, BuildConfig(), report)

    rows = passages.to_pylist()
    whole = [row for row in rows if row["whole"]]
    cut = [row for row in rows if not row["whole"]]
    assert [row["text"] for row in whole] == [prose, code]  # "Tiny." is under 8 tokens
    assert report.dropped["under_8_tokens"]["ai"] == 1
    assert cut and all(row["tokens"] == count for row, count in zip(cut, token_counts([row["text"] for row in cut], tokenizer_path), strict=True))
    assert not any(row["long_eval"] for row in rows)
    for row in cut:
        source = prose if row["doc"] == whole[0]["doc"] else code
        assert row["text"] in source
        if source is prose:
            assert row["text"].startswith("Sentence") and row["text"].endswith(".")
        else:
            assert all(line in code.split("\n") for line in row["text"].split("\n"))
    assert {row["doc"] for row in cut} <= {row["doc"] for row in whole}


def strata_of(texts: list[str], tokens: list[int], **columns: list[object]) -> pa.Table:
    """Strata rows with given token counts and a cut start at every sentence, its token index spread evenly."""
    cut_chars = [[0, *(index + 2 for index in range(len(text)) if text.startswith(". ", index))] for text in texts]
    cut_tokens = [[round(count * char / len(text)) for char in chars] for text, count, chars in zip(texts, tokens, cut_chars, strict=True)]
    docs = [index + 1 for index in range(len(texts))]

    return text_table(texts, STRATA_SCHEMA, format=["plain"] * len(texts), doc=docs, tokens=tokens, cut_chars=cut_chars, cut_tokens=cut_tokens, **columns)


def test_long_documents_are_whole_only_for_evaluation(tokenizer_path: str) -> None:
    text = "Words of a long document go on. " * 400
    strata = strata_of([text, text, text], [20_000, 200_000, 5_000])
    report = Report(rows_in=3)

    rows = passages_table(strata, tokenizer_path, BuildConfig(), report).to_pylist()

    whole = [(row["doc"], row["long_eval"]) for row in rows if row["whole"]]
    assert whole == [(1, True), (3, False)]  # 20,000 tokens is a long_eval row; 200,000 gives no whole row
    assert all(not row["long_eval"] and row["tokens"] <= MAX_TOKENS for row in rows if not row["whole"])
    assert {row["doc"] for row in rows if not row["whole"]} >= {1, 2}  # a long document's passages share its doc
    assert report.counts["long_documents"] == {"ai|16384-32767": 1, "ai|over_131072": 1}


def test_first_corpus_and_benchmark_rows_are_not_cut(tokenizer_path: str) -> None:
    text = "A first corpus passage that is long enough to cut. " * 40
    strata = strata_of([text, text], [600, 600], first_corpus_id=[3, None], first_doc=[11, None], first_whole=[False, None], benchmark=["", "B"])

    rows = passages_table(strata, tokenizer_path, BuildConfig(), Report(rows_in=2)).to_pylist()

    assert [(row["first_corpus_id"], row["whole"], row["benchmark"]) for row in rows] == [(3, False, ""), (None, True, "B")]
