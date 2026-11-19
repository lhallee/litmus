from __future__ import annotations

import time
import pyarrow as pa
import pytest

from litmus_corpus.config import BuildConfig
from litmus_corpus.records import READ_SCHEMA
from litmus_corpus.standardize import GroupShares, count_reading_changes, standard_reading, standardize_table
from litmus_corpus.storage import Report
from litmus_corpus.tells import clean_text, fix_mojibake, frequent_lines, remove_tells, truncation_lengths

from conftest import text_table


def read(text: str) -> str:
    return standard_reading(pa.array([text])).to_pylist()[0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Café", "Café"),
        ("one\r\ntwo\rthree", "one\ntwo\nthree"),
        ("a\u200bb\ufeffc\x07d\u00ade", "abcde"),
        ("keep\u200djoiner\u200cnon", "keep\u200djoiner\u200cnon"),
        ("line   \t\nnext \u00a0", "line\nnext"),
        ("\n\n  \nstart", "start"),
        ("a\n\n\n\n\n\nb", "a\n\n\nb"),
        ("    indented code\n\tkept", "    indented code\n\tkept"),
    ],
)
def test_standard_reading(raw: str, expected: str) -> None:
    assert read(raw) == expected


def clean(text: str, prompt: str | None = None, cut_length: tuple[int | None, int | None] = (None, None)) -> tuple[str | None, Report]:
    report = Report()

    return clean_text(text, prompt, "ai", frozenset(), frozenset(), cut_length, report), report


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("None", "null_string"),
        ("[deleted]", "null_string"),
        ("Broken \ufffd byte", "replacement_character"),
        ("Error: rate limit exceeded, retry later", "api_error"),
        ("I'm sorry, but I can't help with that request.", "refusal"),
        ("Hello NAME_1, how are you?", "anonymized"),
        ("Dear team, I am writing from [Company Name] about the order.", "template_slot"),
    ],
)
def test_dropping_rules(text: str, reason: str) -> None:
    cleaned, report = clean(text)

    assert cleaned is None
    assert report.dropped[reason]["ai"] == 1


def test_long_text_opening_with_sorry_is_not_a_refusal() -> None:
    text = "I'm sorry to hear about your loss. " + "We remember the good days together. " * 30
    cleaned, report = clean(text)

    assert cleaned == text
    assert not report.dropped


@pytest.mark.parametrize(
    ("text", "expected", "rule"),
    [
        ("Itâ€™s fine", "It’s fine", "mojibake"),
        ("first line\\nsecond line\\nthird", "first line\nsecond line\nthird", "export_escapes"),
        ("Tom &amp; Jerry said &quot;hi&quot;", 'Tom & Jerry said "hi"', "html_entities"),
        ("one<br>two<br />three", "one\ntwo\nthree", "html_breaks"),
        ("Answer: The sky is blue.", "The sky is blue.", "response_label"),
        ("The story ends here.\n[truncated]", "The story ends here.", "truncation_marker"),
        ("Best regards,\n[Your Name]\n", "Best regards,\n", "template_line"),
    ],
)
def test_rewriting_rules(text: str, expected: str, rule: str) -> None:
    cleaned, report = clean(text)

    assert cleaned == expected
    assert report.changed[rule]["ai"] == 1
    assert report.examples[rule]


def test_long_whitespace_run_is_read_in_linear_time() -> None:
    """A 200,000-space run inside a line: a pattern that restarts at every space of it takes minutes."""
    text = "def f():" + " " * 200_000 + "return 1\nThe reply goes on to its end."
    started = time.perf_counter()
    cleaned, _ = clean(text)
    count_reading_changes(text, "ai", Report())
    elapsed = time.perf_counter() - started

    assert cleaned == text
    assert elapsed < 5


@pytest.mark.parametrize("ending", ["\n[truncated]", "  \n\n(continued) \n", "<truncated>", " [...]"])
def test_truncation_marker_with_surrounding_whitespace(ending: str) -> None:
    cleaned, report = clean("The story ends here." + ending)

    assert cleaned == "The story ends here."
    assert report.changed["truncation_marker"]["ai"] == 1


def test_html_in_code_is_left_alone() -> None:
    text = "<p>Tom &amp; Jerry &lt;3</p>"
    assert clean(text)[0] == text


def test_prompt_echo_is_removed() -> None:
    cleaned, report = clean("Write a poem about rain please. Rain falls softly on the roof.", prompt="Write a poem about rain please.")

    assert cleaned == "Rain falls softly on the roof."
    assert report.changed["prompt_echo"]["ai"] == 1


def test_gutenberg_frame_is_cut() -> None:
    text = "Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nThe book itself.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicence"
    cleaned, report = clean(text)

    assert cleaned == "The book itself.\n"
    assert report.changed["gutenberg_frame"]["ai"] == 1


def test_fixed_length_cut_drops_the_cut_sentence() -> None:
    text = "First sentence is whole. Second sentence is whole. The third was cut in the mid"
    cleaned, report = clean(text, cut_length=(len(text), None))

    assert cleaned == "First sentence is whole. Second sentence is whole."
    assert report.changed["fixed_length_cut"]["ai"] == 1

    fragment, report = clean("Only a fragment that was cut in the middle of", cut_length=(None, 10))
    assert fragment is None
    assert report.dropped["truncated_fragment"]["ai"] == 1


def test_unfinished_ending_is_cut_back_in_prose_only() -> None:
    finished = "The first point is made well here, with enough words to carry most of the reply on its own."
    text = finished + "\n\nThe second idea follows, and it continues as opposed"
    cleaned, report = clean(text)
    assert cleaned == finished
    assert report.changed["unfinished_ending"]["ai"] == 1

    for kept in (
        "Steps:\n- use fresh basil leaves for the garnish of the dish",
        "lol i dont think that is how any of this works but ok",  # nothing to cut back to: left whole
        "```\ncode\n```\nand then the text runs on without any end at all here",
        "A finished sentence. Another finished one, ending properly as it should.",
    ):
        assert clean(kept)[0] == kept


def test_truncation_length_is_found_at_the_top_of_a_shard() -> None:
    texts = [f"text {index} " + "x" * (20 + index) for index in range(60)] + ["y" * 300] * 15
    origins = ["human"] * len(texts)

    assert truncation_lengths(texts, origins)["human"][0] == 300
    assert truncation_lengths(texts[:60], origins[:60])["human"] == (None, None)


def test_repeated_header_is_found_and_removed() -> None:
    texts = [f"Generated essay\nBody number {index}." for index in range(30)] + [f"Other {index}\nBody." for index in range(5)]
    assert frequent_lines([text.partition("\n")[0] for text in texts]) == frozenset({"Generated essay"})

    report = Report()
    cleaned = remove_tells(texts, [None] * len(texts), ["ai"] * len(texts), [False] * len(texts), report)

    assert cleaned[0] == "Body number 0."
    assert cleaned[-1] == "Other 4\nBody."
    assert report.changed["repeated_header"]["ai"] == 30


def test_benchmark_text_keeps_its_tells() -> None:
    report = Report()
    assert remove_tells(["Answer: kept"], [None], ["ai"], [True], report) == ["Answer: kept"]


def test_mojibake_leaves_clean_text_alone() -> None:
    assert fix_mojibake("Ångström and café") == "Ångström and café"


def test_standardize_table_precaps_and_rereads_rewritten_text() -> None:
    texts = [f"Answer:   \nreply {index}  " for index in range(1000)]
    config = BuildConfig(ai_cap=100)
    shares = GroupShares.from_counts({"ai|gpt-4o": 1000}, config)
    report = Report(rows_in=1000)

    table = standardize_table(text_table(texts, READ_SCHEMA), shares, report)

    assert 60 < table.num_rows < 140  # a tenth kept by hash
    assert report.dropped["precap"]["ai"] == 1000 - table.num_rows
    assert all(text.startswith("reply ") and text == text.strip() for text in table.column("text").to_pylist())


@pytest.mark.parametrize(
    ("decomposed", "composed"),
    [
        ("Café", "Café"),  # e and a combining acute
        ("각", "각"),  # Hangul jamo: kiyeok, a, kiyeok, as one syllable
        ("Ångström", "Ångström"),  # A and a combining ring above
    ],
)
def test_standard_reading_composes_decomposed_text(decomposed: str, composed: str) -> None:
    assert read(decomposed) == composed
    assert read(composed) == composed


def test_first_corpus_rows_keep_their_text_and_say_whether_it_changed() -> None:
    texts = ["Answer: kept as the first corpus has it.", "Café changed by the reading."]
    table = text_table(texts, READ_SCHEMA, first_corpus_id=[1, 2], first_doc=[11, 12], first_whole=[True, True])
    report = Report(rows_in=2)

    standardized = standardize_table(table, GroupShares({}), report)

    assert standardized.column("text").to_pylist() == [texts[0], "Café changed by the reading."]
    assert standardized.column("text_unchanged").to_pylist() == [True, False]
    assert report.changed["reading_nfc"]["ai"] == 1 and "response_label" not in report.changed
    assert report.counts["first_corpus"] == {"rows": 2, "text_unchanged": 1}
