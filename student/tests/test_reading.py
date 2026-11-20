"""The canonical reading (reference.canonical_reading), a store built in it, and training's mid-sentence cuts
(data.Cutter): what each removes, what each leaves alone, and that reading a text twice changes nothing more."""

from __future__ import annotations

import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pathlib import Path
from typing import Any
from litmus_student.contract import reference
from litmus_student.data import Cutter, TokenStore, tokenize_reading, write_rows


CASES = {
    "## Summary\n\nThe **key idea** is *simple*: use `pip`—it’s “fast”…  Really.\n\n- one\n- two\n1. first\n> quoted\n\n---\n\n"
    "| a | b |\n|---|---|\n| 1 | 2 |\n\n```py\nx = **y**  # “keep”\n```":
        "Summary\n\nThe key idea is simple: use pip - it's \"fast\"... Really.\n\none two 1. first quoted\n\n\na\tb\n1\t2\n\n"
        "x = **y**  # \"keep\"",  # a list joins like any prose; code keeps its marks but not its curly quotes
    "- buy some milk\n- walk the dog\n\n## Title\nBody text here now.": "buy some milk walk the dog\n\nTitle Body text here now.",
    "Intro:\n```\npip install numpy\n```\nMore text follows here.": "Intro: pip install numpy More text follows here.",
    "It was late; we left.\nThey had to\nreturn home from the $5 show -> early.": "It was late; we left. They had to return home from the $5 show -> early.",
    "Dear team,\nI wanted to follow up on the report we\ndiscussed last week.  It is ready.\n\nThanks,\nAnna":
        "Dear team, I wanted to follow up on the report we discussed last week. It is ready.\n\nThanks,\nAnna",
    "Tom &amp; Jerry &#8212; a classic&nbsp;show. See &lt;here&gt;.": "Tom & Jerry - a classic show. See <here>.",
    "A man -- a plan -- a canal. Score 3-2 and 1–2.": "A man - a plan - a canal. Score 3-2 and 1 - 2.",
    "[the docs](https://x.y) and ![a cat](cat.png) and <https://z.w>": "the docs and a cat and https://z.w",
    "---": "---",  # nothing with a letter or digit would be left, so the standard reading stays
    # A run of hyphens and a flag in a code span are not dashes.
    "-----Original Message-----\nFrom: Lisa Hesse\nSent: Monday": "-----Original Message----- From: Lisa Hesse Sent: Monday",
    "Pass `--verbose` to see more of the log.": "Pass --verbose to see more of the log.",
    # What one pass uncovers, the next reads: a code span's contents as prose, escaped marks as marks, a bullet's bullet.
    "`` Hello there,'' he said. `` Come in.''": " Hello there,'' he said. Come in.''",
    "Apologies, I rarely write. \\_\\_\\_\\_\\_ Surrounded by flames.": "Apologies, I rarely write. _ Surrounded by flames.",
    "- - - forwarded by sara on monday - - -": "forwarded by sara on monday - - -",
    # Stacked block marks go in one pass, however many.
    "+ + + + + + + + + + + +    suppose that the matrix is full rank .": "suppose that the matrix is full rank .",
    "# # # # # # # # Title of the piece\n\nIt starts here.": "Title of the piece\n\nIt starts here.",
    # Classes fixed so another implementation agrees: a non-ASCII letter is a word character, U+2028 breaks a line.
    "Привет*мир* и всё, *really*.": "Привет*мир* и всё, really.",
    "First line here and its second line&#8232;and a third.": "First line here and its second line and a third.",
    # Headings lose every closing run; a code span closes on a run of its own length; link text holds no bracket.
    "# # Title # #\n\nBody.": "Title\n\nBody.",
    "``code ` here`` and `x`": "code ` here and x",
    "[see [1]](http://a.b) and [ok](http://c.d \"t\")": "[see [1]](http://a.b) and ok",
}
CODE = [
    "def f(**kwargs):\n    return 2 ** 3  # power\nprint(f())",
    "for (int i = 0; i < n; i++) {\n  total += x[i];\n}",
    "x = a--b;\ny = \"it’s\";",
    "SELECT name\nFROM users\nWHERE id = 1;",
    # A comment beside code keeps its mark, where a Markdown heading or bullet would lose it.
    "# load the data\ndf = pd.read_csv(path)\n# plot it\n# and show it\ndf.plot()",
    "/**\n * Adds two numbers.\n * @param a the first\n */\nint add(int a, int b) {",
]


@pytest.mark.parametrize("text", CASES)
def test_canonical_reading_cases(text: str) -> None:
    assert reference.canonical_reading(text) == CASES[text]


@pytest.mark.parametrize("text", CODE)
def test_code_stays_as_written(text: str) -> None:
    """Code keeps its lines and marks; only its typography is typed, as everywhere."""
    standard = reference.standard_reading(text)
    out = reference.canonical_reading(text)
    assert out.split("\n") == [reference.DASH.sub(" - ", line.translate(reference.TYPED)) for line in standard.split("\n")]


def test_raw_and_rendered_markdown_read_alike() -> None:
    raw = "## Steps\n\n1. Install it:\n\n```\npip install litmus\n```\n\n2. Run **one** check.\n\n> Tip: it reads *every* format."
    copied = "Steps\n\n1. Install it:\n\npip install litmus\n\n2. Run one check.\n\nTip: it reads every format."
    assert reference.canonical_reading(raw) == reference.canonical_reading(copied)


@pytest.mark.parametrize("unit", ["`", "[a](", "[a](u \"", "**a ", "__a ", "~~a ", "<http:a", "# ", "+ ", "> - [ ] ", " ",
                                  " " * 999 + "=", "|" + " " * 999 + "-"])
def test_reading_is_linear_on_adversarial_lines(unit: str) -> None:
    """A line of 200,000 characters of unclosed marks reads in well under the minutes a quadratic rule would take
    (each reads in under 0.2 s on a laptop; the bound leaves room for a loaded machine)."""
    line = unit * (200_000 // len(unit))
    started = time.perf_counter()
    reference.canonical_reading(line + " end of the line")
    assert time.perf_counter() - started < 10.0


@pytest.mark.parametrize("text", [*CASES, *CODE, "Plain prose with nothing to change at all.", "Line one\nline two\n\n* a\n* b"])
def test_reading_twice_changes_nothing(text: str) -> None:
    once = reference.canonical_reading(text)
    assert reference.canonical_reading(once) == once


def test_readings_by_name(tmp_path: Path) -> None:
    text = "**Bold** “quote”"
    assert reference.read_as(text, "standard") == reference.standard_reading(text)
    assert reference.read_as(text, reference.CANONICAL_READING) == 'Bold "quote"'
    with pytest.raises(ValueError, match="unknown reading"):
        reference.read_as(text, "other")
    (tmp_path / "config.json").write_text('{"text_reading": "canonical-1"}', encoding="utf-8")
    assert reference.release_reading(tmp_path) == "canonical-1"
    # A release's `reading` holds its chunk and window sizes, not a reading's name.
    (tmp_path / "config.json").write_text('{"reading": {"chunk_tokens": 96, "window_width": 32, "window_stride": 16}}', encoding="utf-8")
    assert reference.release_reading(tmp_path) == "standard"
    (tmp_path / "config.json").write_text('{"text_reading": "canonical-9"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown here"):
        reference.release_reading(tmp_path)


def test_cutter_cuts_only_at_word_starts_near_the_ends() -> None:
    ids = np.arange(200, dtype=np.int64)  # (200,) each id its own place, so a kept piece says where it was cut
    word_start = ids % 3 == 0  # (V,) every third token starts a word
    cutter = Cutter(word_start, end=1.0, start=1.0, generator=np.random.default_rng(0))
    reach = min(Cutter.SPAN, len(ids) // 4)
    for _ in range(200):
        out = cutter.cut(ids)
        first, stop = int(out[0]), int(out[-1]) + 1
        assert word_start[first] and 0 < first <= reach  # the start cut lands on a word start in the first quarter
        assert word_start[stop] and stop >= len(ids) - reach  # the end cut lands before a word start in the last quarter
    short = np.asarray([2, 7, 5], dtype=np.int64)
    assert np.array_equal(cutter.cut(short), short)
    never = Cutter(word_start, end=0.0, start=0.0, generator=np.random.default_rng(0))
    assert np.array_equal(never.cut(ids), ids)
    again = [Cutter(word_start, 0.5, 0.5, np.random.default_rng(3)).cut(ids).tolist() for _ in range(2)]
    assert again[0] == again[1]  # a seed gives the same cuts


def test_cutter_word_starts_from_the_tokenizer(tokenizer: Any) -> None:
    cutter = Cutter.from_tokenizer(tokenizer, 1.0, 0.0, np.random.default_rng(0))
    ids = reference.token_ids(tokenizer, " ".join(["The river runs past the old mill and on to the sea."] * 4))
    out = cutter.cut(ids)
    assert len(out) < len(ids)
    assert tokenizer.decode(ids[len(out):].tolist()).startswith(" ")  # cut between words


def test_store_in_the_canonical_reading(tmp_path: Path, tokenizer: Any) -> None:
    texts = ["**Hello** there.", "A second, longer text about rivers\nand ledgers.", "日本語", "code = 1\n    x"]
    counts = [len(reference.token_ids(tokenizer, text)) for text in texts]  # the standard counts the corpus holds
    table = pa.table({"id": [2, 0, 3, 1], "text": texts, "label": pa.array([0, 1, 2, 0], pa.int8()),
                      "split": ["train", "train", "validation", "train"], "cluster": [5, 5, 6, 7], "tokens": counts,
                      "long_eval": [False, False, True, False]})
    pq.write_table(table, tmp_path / "corpus.parquet")
    tokenize_reading(tmp_path / "corpus.parquet", tmp_path / "store", None, 2, reference.CANONICAL_READING, workers=2)
    write_rows(tmp_path / "corpus.parquet", tmp_path / "store", None, reference.CANONICAL_READING)
    store = TokenStore(tmp_path / "store")
    assert store.reading == reference.CANONICAL_READING
    for corpus_id, text in zip([2, 0, 3, 1], texts, strict=True):
        assert store.ids(corpus_id).tolist() == reference.token_ids(tokenizer, text, reference.CANONICAL_READING).tolist()
        assert store.length[corpus_id] == len(store.ids(corpus_id))
    assert store.length[2] < counts[0]  # the bold marks are gone
