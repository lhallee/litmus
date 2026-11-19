from __future__ import annotations

import pytest

from pathlib import Path
from litmus_corpus import language
from litmus_corpus.formats import text_format
from litmus_corpus.linking import index_shard, save_index
from litmus_corpus.records import STANDARD_SCHEMA
from litmus_corpus.storage import Report, write_parquet
from litmus_corpus.strata import LinkPlan, doc_hash, strata_table
from litmus_corpus.tokenization import cut_starts, token_counts

from conftest import text_table


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A plain paragraph. Another sentence.", "plain"),
        ("<think>\nhmm\n</think>\n\nThe answer.", "chat"),
        ("User: hello\nAssistant: hi\nUser: bye", "chat"),
        ("Here is code:\n```python\ndef f(x):\n    return x * 2\n\nprint(f(2))\n```", "code"),
        ("import os\nfrom pathlib import Path\nx = 1;\nif (x) {\n}", "code"),
        ("| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |", "table"),
        ("Let $x$ be real. Then $x^2 \\geq 0$ and \\frac{1}{2} holds.", "math"),
        ("- apples\n- pears\n- plums", "list"),
        ("# Title\n\nSome **bold** words and a [link](http://example.com).", "markdown"),
    ],
)
def test_formats(text: str, expected: str) -> None:
    assert text_format(text) == expected


def test_cut_starts_take_sentences_in_prose_and_lines_in_code() -> None:
    prose = "One sentence. Two sentence!\nA new line? 末尾。次の文。"
    starts = cut_starts(prose, "plain")
    assert [prose[start:start + 3] for start in starts] == ["One", "Two", "A n", "末尾。", "次の文"]

    code = "x = 1. y = 2\nreturn x"
    assert cut_starts(code, "code") == [0, code.index("return")]


class FakeModel:
    """Stands in for fastText: French when the text says bonjour, else English, else nothing."""

    class Core:
        @staticmethod
        def predict(line: str, k: int, threshold: float, errors: str) -> list[tuple[float, str]]:
            assert "\n" not in line
            if "bonjour" in line:
                return [(0.95, "__label__fr")]
            if "zzz" in line:
                return [(0.2, "__label__en")]
            return [(0.9, "__label__en")]

    f = Core()


NO_LINKS = LinkPlan(None, frozenset())


def test_strata_detects_language_only_for_multi_sources(tokenizer_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(language, "language_model", lambda path: FakeModel())
    texts = ["bonjour tout le monde.\nencore", "The quick brown fox. It jumps.", "zzz zzz", "bonjour but catalog says en"]
    table = text_table(
        texts, STANDARD_SCHEMA, label=[0, 1, 1, 0], generator=["human", "gpt-4o", "gpt-4o", "human"],
        language=["multi", "multi", "multi", "en"], date=["2019-01-01", None, None, "2019-01-01"],
    )

    strata, links = strata_table(table, tokenizer_path, "fake/path", NO_LINKS, Report())

    assert links.num_rows == 0
    assert strata.column("language").to_pylist() == ["fr", "en", "und", "en"]
    assert strata.column("doc").to_pylist()[1] == doc_hash(texts[1])
    assert strata.column("tokens").to_pylist() == token_counts(texts, tokenizer_path)
    chars = strata.column("cut_chars").to_pylist()[1]
    cut_tokens = strata.column("cut_tokens").to_pylist()[1]
    assert chars == [0, texts[1].index("It")]
    assert cut_tokens[0] == 0 and 0 < cut_tokens[1] < strata.column("tokens").to_pylist()[1]


PASSAGE = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu. A second sentence closes it."


def test_raw_and_benchmark_texts_link_to_the_first_corpus_passage_they_hold(tokenizer_path: str, tmp_path: Path) -> None:
    first = text_table([PASSAGE], STANDARD_SCHEMA, source=["reread"], first_corpus_id=[1], first_doc=[999], first_whole=[False])
    write_parquet(first, tmp_path / "first.parquet")
    assert index_shard(tmp_path / "first.parquet", tmp_path / "pieces" / "first.parquet") == 1
    save_index([tmp_path / "pieces" / "first.parquet"], tmp_path / "index")
    texts = [PASSAGE, "Intro.\n" + PASSAGE, PASSAGE + " More words follow here." * 20, "Intro.\n" + PASSAGE, "Nothing in common with any passage at all here."]
    table = text_table(texts, STANDARD_SCHEMA, source=["first_source", "reread", "reread", "bench_source", "reread"], benchmark=["", "", "", "B", ""],
                       first_corpus_id=[1, None, None, None, None], first_doc=[999, None, None, None, None], first_whole=[False, None, None, None, None])
    report = Report()

    strata, links = strata_table(table, tokenizer_path, None, LinkPlan(str(tmp_path / "index"), frozenset({"reread"})), report)

    assert strata.column("text").to_pylist() == [texts[0], texts[2], texts[3], texts[4]]  # the raw copy of the passage is dropped
    assert strata.column("doc").to_pylist() == [999, 999, doc_hash(texts[3]), doc_hash(texts[4])]
    assert report.dropped["represented_in_first_corpus"]["ai"] == 1
    assert links.to_pylist() == [{"first_doc": 999, "benchmark": "B"}]
    assert report.counts["first_corpus_link"] == {"ai|raw_linked_dropped": 1, "ai|raw_linked_kept": 1, "ai|benchmark_linked": 1, "ai|raw_unlinked": 1}
