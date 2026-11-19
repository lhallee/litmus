"""The whole build on small sources, a benchmark and a first corpus, in one process, from memory instead of the Hub."""

from __future__ import annotations

import csv
import time
import numpy as np
import pyarrow.parquet as pq
import pytest

from collections.abc import Callable
from pathlib import Path
from litmus_corpus.config import FIRST_CORPUS_REPO, MAX_TOKENS, BuildConfig
from litmus_corpus.pipeline import Build, BuildOptions, run_parallel
from litmus_corpus.records import CORPUS_SCHEMA

from conftest import catalog_line, write_benchmarks, write_catalog


VOCABULARY = [f"{consonant}{vowel}{ending}" for consonant in "bdfgklmnprst" for vowel in "aeiou" for ending in ("n", "r", "ss", "")]


def essay(seed: int, words: str) -> str:
    sentences = [f"In part {seed}.{index} the {words} moved {index % 7} steps toward the river and rested." for index in range(4 + seed % 30)]

    return " ".join(sentences)


def long_document(seed: int) -> str:
    """About 14,000 characters of varied sentences: 8,193 to 16,383 tokens of the fixture tokenizer."""
    generator = np.random.default_rng(seed)
    sentences = [" ".join(generator.choice(VOCABULARY, size=int(generator.integers(8, 16)))).capitalize() + "." for _ in range(300)]

    return " ".join(sentences)


def first_row(identifier: int, text: str, label: int, source: str, whole: bool = False) -> dict[str, object]:
    return {"id": identifier, "text": text, "label": label, "source": source, "generator": "human" if label == 0 else "gpt-4o",
            "domain": "essay", "doc": 10_000 + identifier, "whole": whole}


PASSAGE = ". ".join(essay(50, "robot poet").split(". ")[:3]) + "."  # the first corpus's trimmed passage of model essay 50


@pytest.fixture
def build_inputs(tmp_path: Path, fake_streams: Callable, tokenizer_path: str, monkeypatch: pytest.MonkeyPatch) -> BuildOptions:
    monkeypatch.setattr("litmus_corpus.pipeline.ensure_tokenizer", lambda: Path(tokenizer_path))
    human = [{"text": essay(index, "old farmer"), "year": 2015} for index in range(80)]
    human.append({"text": essay(999, "new farmer"), "year": 2023})  # fails its date
    human.append({"text": long_document(1), "year": 2015})
    model = [{"text": essay(index, "robot poet"), "model": "gpt-4o" if index % 2 else "llama-2-7b-chat"} for index in range(80)]
    model.append({"text": human[0]["text"], "model": "gpt-4o"})  # a copy of a human text: both go
    model.append({"text": long_document(2), "model": "gpt-4o"})
    chats = [
        {"messages": [{"role": "user", "content": f"Question {index}?"}, {"role": "assistant", "content": essay(index, "chat helper")}]}
        for index in range(20)
    ]
    benchmark = [{"text": essay(index, "robot poet"), "label": 1} for index in range(3)]  # copies three model texts
    hidden = [{"generation": human[1]["text"]}, {"generation": "A hidden benchmark text that nothing else holds at all."}]
    first = {
        "train": [
            first_row(1, PASSAGE, 1, "model_essays"),
            first_row(2, essay(5, "old farmer"), 0, "human_essays", whole=True),  # also read new: the first corpus's copy stays
            first_row(3, essay(1, "robot poet"), 1, "model_essays", whole=True),  # FixtureBench's text: linked, leaves training
        ],
        "validation": [first_row(4, essay(200, "old farmer"), 0, "human_essays", whole=True)],
        "test": [first_row(5, essay(2, "lost poet"), 1, "bench", whole=True)],  # from a held-out source: FixtureBench's
        "unused": [first_row(6, essay(201, "old farmer"), 0, "human_essays", whole=True)],
    }
    fake_streams(
        {"fixture/human": human, "fixture/model": model, "fixture/chats": chats, "fixture/bench": benchmark, "fixture/hidden": hidden}
        | {f"{FIRST_CORPUS_REPO}/{split}": rows for split, rows in first.items()}
    )
    catalog = write_catalog(
        tmp_path / "sources.csv",
        [
            catalog_line(name="human_essays", repo="fixture/human", origin="human", text_column="text", date_column="year", domain="essay",
                         in_first_corpus="yes", notes="raw re-read: yes"),
            catalog_line(name="model_essays", repo="fixture/model", origin="ai", text_column="text", generator_column="model", domain="essay",
                         in_first_corpus="yes", notes="raw re-read: yes"),
            catalog_line(name="chats", repo="fixture/chats", origin="ai", text_column="messages", generator="gpt-3.5-turbo", domain="chat"),
            catalog_line(name="gated", repo="fixture/gated", origin="ai", text_column="text", generator="gpt-4o"),
            catalog_line(name="bench", repo="fixture/bench", split="test", origin="by_column", text_column="text", label_column="label",
                         label_map='{"0": "human", "1": "ai"}', generator="unknown-model", date_evidence="2020", priority="5", in_first_corpus="yes"),
        ],
    )
    benchmarks = write_benchmarks(
        tmp_path / "benchmarks.csv",
        [
            {"name": "FixtureBench", "repo": "fixture/bench", "split": "test", "label_column": "label", "label_map": '{"0": "human", "1": "ai"}'},
            {"name": "HiddenBench", "repo": "fixture/hidden", "split": "test", "text_column": "generation"},  # labels hidden, no source line
        ],
    )

    return BuildOptions(catalog=str(catalog), benchmarks=str(benchmarks), priority=1, limit_per_source=None)


def slow_identity(value: int) -> int:
    time.sleep(0.02 * (5 - value))  # later tasks finish first
    return value


def test_parallel_results_come_back_in_task_order() -> None:
    assert run_parallel(slow_identity, [(value,) for value in range(5)], workers=3, description="order") == list(range(5))


def test_an_unreadable_benchmark_stops_the_build(tmp_path: Path, fake_streams: Callable, tokenizer_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litmus_corpus.pipeline.ensure_tokenizer", lambda: Path(tokenizer_path))
    fake_streams({"fixture/model": [{"text": essay(index, "robot poet")} for index in range(5)]})
    catalog = write_catalog(tmp_path / "sources.csv", [catalog_line(name="model", repo="fixture/model", origin="ai", text_column="text", generator="gpt-4o")])
    benchmarks = write_benchmarks(tmp_path / "benchmarks.csv", [{"name": "Gone", "repo": "fixture/gone", "split": "test", "label_map": '{"*": "ai"}'}])
    options = BuildOptions(catalog=str(catalog), benchmarks=str(benchmarks), priority=1, limit_per_source=None, first_corpus=False)

    with pytest.raises(RuntimeError, match="benchmarks could not be read.*Gone"):
        Build(options, tmp_path / "out", workers=1, config=BuildConfig()).run()


def test_build_end_to_end_and_resume(tmp_path: Path, build_inputs: BuildOptions) -> None:
    out = tmp_path / "out"
    config = BuildConfig(shard_rows=40)

    report = Build(build_inputs, out, workers=1, config=config).run()

    corpus = out / "corpus"
    shards = sorted((corpus / "data").glob("*.parquet"))
    assert pq.read_table(shards[0]).schema.equals(CORPUS_SCHEMA)
    rows = [row for path in shards for row in pq.read_table(path).to_pylist()]
    assert [row["id"] for row in rows] == list(range(len(rows))) == list(range(report["rows"]))
    training = [row for row in rows if not row["benchmark"] and not row["long_eval"]]
    benchmark = [row for row in rows if row["benchmark"]]
    long_eval = [row for row in rows if row["long_eval"]]

    # Lengths: no training row past 8,192 tokens; one long document per origin, kept whole for evaluation.
    assert all(8 <= row["tokens"] <= MAX_TOKENS for row in training)
    assert sorted(row["label"] for row in long_eval) == [0, 1] and all(row["whole"] and row["tokens"] > MAX_TOKENS for row in long_eval)
    long_docs = {row["doc"] for row in long_eval}
    assert any(row["doc"] in long_docs for row in training)  # its passages train, under its doc

    # Benchmarks: evaluation only, hidden labels never written, nothing of theirs in training.
    by_benchmark = {}
    for row in benchmark:
        by_benchmark.setdefault(row["benchmark"], []).append(row)
    assert set(by_benchmark) == {"FixtureBench"} and len(by_benchmark["FixtureBench"]) == 5  # 3 read, 2 first-corpus rows marked
    assert not any(row["text"].startswith("In part 1.0 the old farmer") for row in rows)  # held out by the hidden benchmark
    benchmark_docs = {row["doc"] for row in benchmark}
    assert not any(row["doc"] in benchmark_docs for row in training)

    # The first corpus: verbatim rows with their ids, preferred over a new copy, its raw source linked by doc.
    first = {row["first_corpus_id"]: row for row in rows if row["first_corpus_id"] is not None}
    assert set(first) == {1, 2, 3, 4, 5, 6}
    assert first[3]["benchmark"] == "FixtureBench" and first[5]["benchmark"] == "FixtureBench"
    assert all(row["text_unchanged"] for row in first.values())
    assert sum(row["text"] == essay(5, "old farmer") for row in rows) == 1
    raw_fifty = [row for row in rows if row["whole"] and row["first_corpus_id"] is None and row["text"] == essay(50, "robot poet")]
    assert raw_fifty and raw_fifty[0]["doc"] == first[1]["doc"] == 10_001

    # Checks and balance.
    assert not any("new farmer" in row["text"] for row in rows)
    assert not any(row["text"].startswith("In part 0.0 the old farmer") for row in rows)
    texts = [row["text"] for row in training]
    assert len(texts) == len(set(texts))
    assert {row["whole"] for row in training} == {True, False}
    assert {"Question 0?"} & set(texts) == set()  # user turns of an undated chat source are dropped
    human = sum(row["label"] == 0 for row in training)
    model = sum(row["label"] == 1 for row in training)
    assert human > 0 and model > 0 and abs(human - model) <= len(first)
    assert {row["era"] for row in training} <= {"human", "recent", "legacy"}

    # The report and sources.csv.
    assert report["failed_sources"]["gated"]["stage"] == "open"
    # Row 5 is FixtureBench's by its source; row 3 by strata's link of FixtureBench's text to it.
    assert report["first_corpus"]["benchmark_rows"] == {"source": {"FixtureBench": 1}, "link": {"FixtureBench": 1}, "doc": {}, "text": {}, "opening": {}}
    assert report["first_corpus"]["links"]["ai|raw_linked_kept"] == 1
    assert report["long_documents"] == {"human|8193-16383": 1, "ai|8193-16383": 1}
    assert [line["after"] for line in report["long_eval"] if line["tokens"] == "8193-16383"] == [{"human": 1, "ai": 1, "mixed": 0}]
    assert sum(line["new_tokens_after"] for line in report["token_budget"]) == report["written"]["human_new_training_tokens"] + report["written"][
        "ai_new_training_tokens"] + report["written"]["mixed_new_training_tokens"]
    assert "bench" in report["catalog"]["excluded"] and "HiddenBench: a split of the benchmark catalog with no flagged line in sources.csv" in report["catalog"]["disagreements"]
    with (corpus / "sources.csv").open(encoding="utf-8") as handle:
        sources = {line["name"]: line for line in csv.DictReader(handle)}
    assert int(sources["human_essays"]["human"]) == sum(row["label"] == 0 and row["source"] == "human_essays" for row in rows)
    assert int(sources["human_essays"]["first_corpus"]) == 3
    assert sorted(path.name for path in (corpus / "samples").glob("*.txt"))
    stages = report["stages"]
    assert stages["checks_keys"]["total"]["dropped"]["human_not_before_2022"]["human"] >= 1
    assert "text_under_two_origins" in stages["checks"]["total"]["dropped"]
    assert stages["holdout"]["total"]["dropped"]

    again = Build(build_inputs, out, workers=1, config=config).run()
    assert again["rows"] == report["rows"]

    with pytest.raises(RuntimeError, match="other options"):
        Build(build_inputs, out, workers=1, config=BuildConfig(shard_rows=41)).run()
