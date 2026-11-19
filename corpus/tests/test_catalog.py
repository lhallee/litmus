from __future__ import annotations

import pytest

from pathlib import Path
from litmus_corpus.catalog import FIRST_CORPUS, CatalogError, load_catalog
from litmus_corpus.generators import era, is_legacy, maker
from litmus_corpus.text_paths import PathError, parse_alternatives, parse_path, texts_at, generator_at

from conftest import catalog_line, spec_of, write_benchmarks, write_catalog


def test_flat_human_source_takes_latest_year_of_its_evidence() -> None:
    spec = spec_of(name="c4", origin="human", text_column="text", date_evidence="C4, crawled April 2019 from pages of 2013 to 2019")

    assert spec.source_date == "2019-12-31"
    assert spec.origins() == frozenset({"human"})


def test_human_source_without_a_date_is_read_and_its_rows_left_to_the_checks() -> None:
    spec = spec_of(name="undated", origin="human", text_column="text")

    assert spec.source_date is None and spec.date_column is None


def test_by_column_needs_a_label_map_and_star_maps_the_rest() -> None:
    spec = spec_of(name="raid", origin="by_column", text_column="generation", label_column="model", label_map='{"human": "human", "*": "ai"}',
                   generator_column="model", date_evidence="2016", row_filter='{"attack": "none"}')
    assert spec.origins() == frozenset({"human", "ai"})
    assert spec.row_filter == {"attack": ["none"]} and spec.needed_columns() == ["generation", "model", "attack"]

    with pytest.raises(CatalogError, match="raid"):
        spec_of(name="raid", origin="by_column", text_column="generation", generator_column="model")


def test_several_text_expressions_are_each_a_text_mapped_by_label_map() -> None:
    spec = spec_of(name="hc3", origin="by_column", text_column="human_answers[] | chatgpt_answers[]",
                   label_map='{"human_answers[]": "human", "chatgpt_answers": "ai"}', generator="gpt-3.5-turbo (via the API)", date_evidence="2020")
    assert [path.column for path in spec.text_paths] == ["human_answers", "chatgpt_answers"]
    assert [spec.column_origin(path) for path in spec.text_paths] == ["human", "ai"]
    assert spec.generator == "gpt-3.5-turbo"

    with pytest.raises(CatalogError, match="label_map entry"):
        spec_of(name="hc3", origin="by_column", text_column="a | b", label_map='{"a": "human"}', generator="x", date_evidence="2020")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"origin": "ai", "text_column": "text"}, "generator"),
        ({"origin": "robot", "text_column": "text"}, "origin"),
        ({"origin": "ai", "text_column": "text", "generator": "gpt-4", "revision": ""}, "revision"),
        ({"origin": "ai", "text_column": "text", "generator": "gpt-4", "priority": "high"}, "priority"),
        ({"origin": "ai", "text_column": "see card", "generator": "gpt-4"}, "cannot read"),
        ({"origin": "human", "text_column": "text", "label_column": "label"}, "takes no label_column"),
    ],
)
def test_unusable_lines_name_the_source(values: dict[str, str], message: str) -> None:
    with pytest.raises(CatalogError, match=f"bad_source.*{message}|{message}.*bad_source|source bad_source"):
        spec_of(name="bad_source", **values)


def test_path_expressions() -> None:
    path = parse_path("messages[].content (role=assistant)")
    assert path.column == "messages" and path.keep == ("role", "assistant") and not path.is_plain
    row = {"messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A.", "reasoning_content": "think"}]}
    found = texts_at(row, path)
    assert [item.text for item in found] == ["<think>\nthink\n</think>\n\nA."]
    assert found[0].prompt == "Q?"

    nested = texts_at({"meta": '{"answer": {"text": "deep"}}'}, parse_path("meta.answer.text (JSON)"))
    assert [item.text for item in nested] == ["deep"]

    texts, models = parse_alternatives("completions[].response"), parse_alternatives("completions[].model")
    row = {"completions": [{"response": "one", "model": "gpt-4o"}, {"response": "two", "model": "claude-3-opus"}]}
    pairs = [(item.text, generator_at(row, item, texts[0], models[0])) for item in texts_at(row, texts[0])]
    assert pairs == [("one", "gpt-4o"), ("two", "claude-3-opus")]

    assert parse_path("llama-3.1-70b_edits").column == "llama-3.1-70b_edits"
    assert parse_path("title + body").columns == ("title", "body")
    for unreadable in ("Command R+", "<model>.content", "see card", "row_json"):
        with pytest.raises(PathError):
            parse_path(unreadable)


def test_benchmarks_are_held_out_and_disagreements_listed(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "sources.csv",
        [
            catalog_line(name="first", origin="ai", text_column="text", generator="gpt-4o", priority="1"),
            catalog_line(name="later", origin="ai", text_column="text", generator="gpt-4o", priority="3"),
            catalog_line(name="bench", repo="bench/repo", split="test", origin="ai", text_column="text", generator="gpt-4o", in_first_corpus="yes"),
            catalog_line(name="flagged_only", repo="other/repo", origin="ai", text_column="text", generator="gpt-4o", benchmark="Elsewhere"),
            catalog_line(name="no_derivs", origin="ai", text_column="text", generator="gpt-4o", license="cc-by-nc-nd-4.0"),
            catalog_line(name="sharealike", origin="ai", text_column="text", generator="gpt-4o", license="cc-by-nc-sa-4.0"),
            catalog_line(name="wildai", repo="pangram/WildAI", origin="ai", text_column="text", generator="x"),
            catalog_line(name="stands", origin="ai", text_column="text", generator="gpt-4o", in_first_corpus="yes", notes="raw re-read: no"),
            catalog_line(name="broken", origin="ai", text_column="text"),
        ],
    )
    benchmarks = write_benchmarks(
        tmp_path / "benchmarks.csv",
        [
            {"name": "BenchOne", "repo": "bench/repo", "split": "test", "label_map": '{"*": "ai"}'},
            {"name": "Unflagged", "repo": "u/v", "split": "test", "label_map": '{"*": "human"}'},
        ],
    )

    catalog_read = load_catalog(catalog, benchmarks, max_priority=1)

    assert [spec.name for spec in catalog_read.specs] == ["first", "sharealike", "BenchOne", "Unflagged", FIRST_CORPUS]
    assert catalog_read.specs[2].benchmark == "BenchOne" and catalog_read.specs[2].origin == "ai"
    assert catalog_read.held_out_first_sources == {"bench": "BenchOne"}
    assert catalog_read.specs[-1].held_out_first == {"bench": "BenchOne"} and catalog_read.specs[-1].verbatim
    assert set(catalog_read.excluded) == {"bench", "flagged_only", "no_derivs", "wildai", "stands"}
    assert set(catalog_read.problems) == {"broken"}
    assert any("flagged_only" in line for line in catalog_read.disagreements)
    assert any("Unflagged" in line for line in catalog_read.disagreements)


def test_benchmarks_are_read_from_the_benchmark_catalog(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "sources.csv", [catalog_line(name="first", origin="ai", text_column="text", generator="gpt-4o")])
    benchmarks = write_benchmarks(
        tmp_path / "benchmarks.csv",
        [
            {"name": "hidden", "repo": "x/y", "split": "test", "text_column": "generation"},
            {"name": "labelled", "repo": "x/z", "config": "file:test.csv", "split": "train", "label_column": "label",
             "label_map": '{"1": "human", "0": "ai"}', "language": "en,cs,de", "row_filter": '{"attack": "none"}'},
            {"name": "columns", "repo": "x/c", "split": "train", "text_column": "human_answers,chatgpt_answers",
             "label_map": '{"human_answers": "human", "chatgpt_answers": "ai"}'},
            {"name": "polished", "repo": "x/p", "split": "train", "label_map": '{"*": "mixed"}', "language": "code: python,java"},
        ],
    )

    specs = {spec.name: spec for spec in load_catalog(catalog, benchmarks, max_priority=1, first_corpus=False).specs}

    assert specs["hidden"].origin == "unlabeled" and specs["hidden"].benchmark == "hidden"
    assert specs["labelled"].origins() == frozenset({"human", "ai"}) and specs["labelled"].generator == "unknown"
    assert specs["labelled"].language == "multi" and specs["labelled"].data_file == "test.csv"
    assert specs["labelled"].row_filter == {"attack": ["none"]}
    assert "attack" in specs["labelled"].needed_columns()
    assert [path.column for path in specs["columns"].text_paths] == ["human_answers", "chatgpt_answers"] and specs["columns"].origin == "by_column"
    assert specs["polished"].origin == "mixed" and specs["polished"].language == "code"


def test_an_unreadable_benchmark_stops_the_build_unless_skipped_by_name(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "sources.csv", [catalog_line(name="first", origin="ai", text_column="text", generator="gpt-4o")])
    benchmarks = write_benchmarks(tmp_path / "benchmarks.csv", [{"name": "gated", "repo": "x/g", "split": "test", "text_column": ""}])

    with pytest.raises(CatalogError, match="gated"):
        load_catalog(catalog, benchmarks, max_priority=1)
    skipped = load_catalog(catalog, benchmarks, max_priority=1, skip_benchmarks=("gated",), first_corpus=False)
    assert [spec.name for spec in skipped.specs] == ["first"] and skipped.skipped_benchmarks == ["gated"]
    with pytest.raises(CatalogError, match="typo"):
        load_catalog(catalog, benchmarks, max_priority=1, skip_benchmarks=("typo",))


def test_a_benchmark_name_may_not_repeat_a_source_name(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "sources.csv", [catalog_line(name="same", origin="ai", text_column="text", generator="gpt-4o")])
    benchmarks = write_benchmarks(tmp_path / "benchmarks.csv", [{"name": "same", "repo": "x/y", "split": "test"}])

    with pytest.raises(CatalogError, match="same"):
        load_catalog(catalog, benchmarks, max_priority=1)


@pytest.mark.parametrize(
    ("generator", "origin", "expected"),
    [("human", "human", "human"), ("gpt-4o", "ai", "recent"), ("gpt-2", "ai", "legacy"), ("mystery", "ai", "unknown"), ("", "mixed", "unknown")],
)
def test_era(generator: str, origin: str, expected: str) -> None:
    assert era(generator, origin) == expected


@pytest.mark.parametrize(
    ("generator", "provider", "legacy"),
    [
        ("gpt-4o-2024-08-06", "openai", False),
        ("gpt-3.5-turbo-0613", "openai", True),
        ("ChatGPT", "openai", True),
        ("meta-llama/Llama-2-70b-chat-hf", "meta", True),
        ("Meta-Llama-3.1-8B-Instruct", "meta", False),
        ("claude-3-5-sonnet-20240620", "anthropic", False),
        ("claude-instant-1", "anthropic", True),
        ("Qwen2.5-72B-Instruct", "alibaba", False),
        ("deepseek-r1", "deepseek", False),
        ("mistral-7b-instruct-v0.1", "mistral", True),
        ("o3-mini", "openai", False),
        ("mystery-model", "unknown", True),
    ],
)
def test_generator_makers(generator: str, provider: str, legacy: bool) -> None:
    assert maker(generator).provider == provider
    assert is_legacy(generator) is legacy
