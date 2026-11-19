from __future__ import annotations

import datetime as dt
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collections.abc import Callable
from pathlib import Path
from litmus_corpus import read
from litmus_corpus.adapters import SchemaMismatch, generic_adapter, parse_date, register, ADAPTERS, adapter_for
from litmus_corpus.catalog import first_corpus_spec
from litmus_corpus.config import BuildConfig
from litmus_corpus.read import ReadTask, entry_tasks, part_report_path, plan_source, read_limits, read_part
from litmus_corpus.storage import Report, read_json

from conftest import spec_of


def test_flat_text_with_a_label_column() -> None:
    spec = spec_of(name="raid", origin="by_column", text_column="generation", label_column="model",
                   label_map='{"human": "human", "*": "ai"}', generator_column="model", date_evidence="2019")
    batch = pa.table({"generation": ["a person wrote this", "a model wrote this", None], "model": ["human", "gpt-4o", "gpt2"]})
    report = Report()

    records = generic_adapter(batch, spec, report)

    assert records.column("text").to_pylist() == ["a person wrote this", "a model wrote this"]
    assert records.column("label").to_pylist() == [0, 1]
    assert records.column("generator").to_pylist() == ["human", "gpt-4o"]
    assert records.column("provider").to_pylist() == ["human", "openai"]
    assert records.column("date").to_pylist() == ["2019-12-31", "2019-12-31"]
    assert report.dropped["null_text"]["ai"] == 1


def test_row_filter_keeps_matching_rows() -> None:
    line = {"name": "raid", "origin": "by_column", "text_column": "generation", "label_column": "model", "label_map": '{"human": "human", "*": "ai"}',
            "generator_column": "model", "date_evidence": "2019", "row_filter": '{"attack": "none", "repeat": [true]}'}
    spec = spec_of(**line)
    batch = pa.table({"generation": ["kept", "attacked", "unrepeated"], "model": ["gpt2", "gpt2", "gpt2"], "attack": ["none", "homoglyph", "none"],
                      "repeat": [True, True, False]})
    report = Report()

    records = generic_adapter(batch, spec, report)

    assert records.column("text").to_pylist() == ["kept"]
    assert report.dropped["row_filter"]["any"] == 2


def test_label_outside_the_map_fails_with_the_source_name() -> None:
    spec = spec_of(name="strict", origin="by_column", text_column="text", label_column="label", label_map='{"0": "human", "1": "ai"}',
                   generator="gpt-4", date_evidence="2019")

    with pytest.raises(SchemaMismatch, match="strict"):
        generic_adapter(pa.table({"text": ["x"], "label": [7]}), spec, Report())


def test_two_list_columns_each_with_its_origin() -> None:
    spec = spec_of(name="hc3", origin="by_column", text_column="human_answers[] | chatgpt_answers",
                   label_map='{"human_answers": "human", "chatgpt_answers": "ai"}', generator="gpt-3.5-turbo", date_evidence="2020")
    batch = pa.table({"human_answers": [["h1", "h2"], []], "chatgpt_answers": [["m1"], ["m2"]]})

    records = generic_adapter(batch, spec, Report())

    assert list(zip(records.column("text").to_pylist(), records.column("label").to_pylist(), strict=True)) == [("h1", 0), ("h2", 0), ("m1", 1), ("m2", 1)]


def test_joined_columns_make_one_text_and_struct_fields_date_it() -> None:
    spec = spec_of(name="forum", origin="human", text_column="title + answers[].text", date_column="meta.year")
    batch = pa.table(
        {
            "title": ["Why is the sky blue?", "Empty"],
            "answers": [[{"text": "Scattering."}, {"text": "Rayleigh, mostly."}], []],
            "meta": [{"year": 2016}, {"year": 2023}],
        }
    )

    records = generic_adapter(batch, spec, Report())

    assert records.column("text").to_pylist() == ["Why is the sky blue?\n\nScattering.\n\nRayleigh, mostly.", "Empty"]
    assert records.column("date").to_pylist() == ["2016-12-31", "2023-12-31"]


def test_extra_generator_expressions_pair_by_position() -> None:
    spec = spec_of(name="arena_like", origin="ai", text_column="conversation_a[].content (role=assistant)", generator_column="model_a | model_b")
    batch = pa.table({"conversation_a": [conversation(("user", "Hi?"), ("assistant", "From A."))], "model_a": ["gpt-4-0613"], "model_b": ["claude-2.1"]})

    assert generic_adapter(batch, spec, Report()).column("generator").to_pylist() == ["gpt-4-0613"]


def test_paired_generators_come_from_the_same_element() -> None:
    spec = spec_of(name="arena", origin="ai", text_column="conversation_a[].content (role=assistant) | conversation_b[].content (role=assistant)",
                   generator_column="model_a | model_b")
    batch = pa.table(
        {
            "conversation_a": [conversation(("user", "Hi?"), ("assistant", "From A."))],
            "conversation_b": [conversation(("user", "Hi?"), ("assistant", "From B."))],
            "model_a": ["gpt-4-0613"],
            "model_b": ["claude-2.1"],
        }
    )

    records = generic_adapter(batch, spec, Report())

    assert records.column("text").to_pylist() == ["From A.", "From B."]
    assert records.column("generator").to_pylist() == ["gpt-4-0613", "claude-2.1"]
    assert records.column("prompt").to_pylist() == ["Hi?", "Hi?"]


def conversation(*turns: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in turns]


def test_conversation_keeps_assistant_turns_and_drops_recent_user_turns() -> None:
    spec = spec_of(name="chats", origin="ai", text_column="messages", generator_column="model", date_column="when")
    batch = pa.table(
        {
            "messages": [
                conversation(("system", "be nice"), ("user", "Old question?"), ("assistant", "Old answer.")),
                conversation(("user", "New question?"), ("assistant", "New answer.")),
            ],
            "model": ["gpt-4", "gpt-4o"],
            "when": ["2020-05-01", "2024-01-02"],
        }
    )
    report = Report()

    records = generic_adapter(batch, spec, report)

    assert records.column("text").to_pylist() == ["Old question?", "Old answer.", "New answer."]
    assert records.column("label").to_pylist() == [0, 1, 1]
    assert records.column("prompt").to_pylist() == [None, "Old question?", "New question?"]
    assert report.dropped["user_turn_not_before_2022"]["human"] == 1
    assert report.dropped["system_turn"]["ai"] == 1


def test_reasoning_field_is_kept_as_a_block() -> None:
    spec = spec_of(name="thinker", origin="ai", text_column="messages", generator="deepseek-r1")
    batch = pa.table({"messages": [[{"role": "assistant", "content": "Answer.", "reasoning_content": "Let me think."}]]})

    records = generic_adapter(batch, spec, Report())

    assert records.column("text").to_pylist() == ["<think>\nLet me think.\n</think>\n\nAnswer."]


def test_unknown_role_and_unknown_shape_fail_loudly() -> None:
    spec = spec_of(name="odd", origin="ai", text_column="messages", generator="gpt-4")
    with pytest.raises(SchemaMismatch, match="odd"):
        generic_adapter(pa.table({"messages": [conversation(("narrator", "x"))]}), spec, Report())
    with pytest.raises(SchemaMismatch, match="odd"):
        generic_adapter(pa.table({"messages": [1, 2]}), spec, Report())
    with pytest.raises(SchemaMismatch, match="odd"):
        generic_adapter(pa.table({"other": ["x"]}), spec, Report())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019-05-04T10:00:00Z", "2019-05-04"),
        ("2019", "2019-12-31"),
        ("2020-02", "2020-02-29"),
        ("12/31/2018", "2018-12-31"),
        (2017, "2017-12-31"),
        (1_500_000_000, "2017-07-14"),
        (dt.datetime(2021, 3, 1), "2021-03-01"),
        ("no date here", None),
        (None, None),
    ],
)
def test_parse_date(value: object, expected: str | None) -> None:
    assert parse_date(value) == expected


def test_registered_adapter_replaces_the_generic_one() -> None:
    spec = spec_of(name="registered_fixture", origin="ai", text_column="text", generator="gpt-4")

    @register("registered_fixture")
    def only_long(batch: pa.Table, spec: object, report: Report) -> pa.Table:
        return generic_adapter(batch.filter(pa.compute.greater(pa.compute.utf8_length(batch.column("text")), 3)), spec, report)

    try:
        assert adapter_for(spec) is only_long
        assert only_long(pa.table({"text": ["abc", "abcdef"]}), spec, Report()).column("text").to_pylist() == ["abcdef"]
    finally:
        del ADAPTERS["registered_fixture"]


def test_read_part_writes_shards_then_its_report_and_respects_the_limit(tmp_path: Path, fake_streams: Callable) -> None:
    spec = spec_of(name="flat", repo="fixture/flat", origin="ai", text_column="text", generator="gpt-4o", split="train|test")
    fake_streams({"fixture/flat": [{"text": f"text number {index}"} for index in range(50)]})
    task = ReadTask(spec, split_index=1, part=0, parts=2, limit=10, directory=tmp_path / "flat")

    payload = read_part(task, BuildConfig(shard_rows=4, read_batch_rows=3))

    shards = sorted(task.directory.glob("s1-p000-*.parquet"))
    assert task.split == "test"
    assert payload["rows_in"] == 10
    assert [pq.read_table(path).num_rows for path in shards] == [6, 4]
    assert read_json(part_report_path(task))["shards"] == [path.name for path in shards]


def test_a_source_that_cannot_be_opened_or_misses_columns_fails_alone(tmp_path: Path, fake_streams: Callable) -> None:
    gated = spec_of(name="gated", repo="fixture/gated", origin="ai", text_column="text", generator="gpt-4o")
    renamed = spec_of(name="renamed", repo="fixture/renamed", origin="ai", text_column="text", generator="gpt-4o")
    fake_streams({"fixture/renamed": [{"body": "the column moved"}]})

    assert plan_source(gated, None, BuildConfig())["stage"] == "open"
    task = ReadTask(renamed, split_index=0, part=0, parts=1, limit=None, directory=tmp_path / "renamed")
    failure = read_part(task, BuildConfig())
    assert failure["stage"] == "first rows" and "renamed" in failure["failed"]
    assert read_json(tmp_path / "renamed" / "failed.json")["stage"] == "first rows"
    assert not part_report_path(task).exists()


class RateLimited(Exception):
    """Stands in for `HfHubHTTPError` with a 429 response."""

    class Response:
        status_code = 429

    response = Response()


def test_rate_limits_are_retried_and_other_failures_are_not(monkeypatch: pytest.MonkeyPatch, fake_streams: Callable, tmp_path: Path) -> None:
    waits: list[float] = []
    monkeypatch.setattr("litmus_corpus.read.pause", waits.append)
    fake_streams({"fixture/flaky": [{"text": f"text number {index}"} for index in range(5)]})
    real_open = read.open_stream
    calls = {"count": 0}

    def flaky_open(spec: object, split: str) -> object:
        calls["count"] += 1
        if calls["count"] <= 2:
            raise RateLimited("429 Client Error: Too Many Requests")
        return real_open(spec, split)

    monkeypatch.setattr(read, "open_stream", flaky_open)
    spec = spec_of(name="flaky", repo="fixture/flaky", origin="ai", text_column="text", generator="gpt-4o")
    payload = read_part(ReadTask(spec, 0, 0, 1, None, tmp_path / "flaky"), BuildConfig())

    assert payload["rows_in"] == 5 and waits == [30, 60]
    assert read.is_transient(RuntimeError("wrapped")) is False
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = RateLimited()
    assert read.is_transient(wrapped)


def test_wildcard_configs_and_splits_expand_into_entries(monkeypatch: pytest.MonkeyPatch, fake_streams: Callable, tmp_path: Path) -> None:
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda repo, revision: ["en", "en.noclean", "de", "fr"])
    monkeypatch.setattr(datasets, "get_dataset_split_names", lambda repo, config, revision: ["train", "a_think", "b_think", "b_plain"])
    fake_streams({"fixture/many": [{"text": f"row {index}"} for index in range(4)]})
    spec = spec_of(name="many", repo="fixture/many", origin="ai", text_column="text", generator="gpt-4o", config="[!e]*|en", split="*think")

    plan = plan_source(spec, 40, BuildConfig())
    tasks = entry_tasks(spec, plan, tmp_path)

    assert [(entry["config"], entry["split"]) for entry in plan["parts"]] == [
        (config, split) for config in ("en", "de", "fr") for split in ("a_think", "b_think")
    ]
    assert [task.prefix for task in tasks] == [f"s{index}-p000" for index in range(6)]
    assert tasks[2].spec.config == "de" and tasks[2].split == "a_think" and tasks[2].limit == 7


def test_messages_kept_as_json_text_are_read() -> None:
    spec = spec_of(name="json_turns", origin="ai", text_column="messages (JSON)", generator="claude-sonnet-4-6")
    batch = pa.table({"messages": [['{"role": "system", "content": "Be brief."}', '{"role": "assistant", "content": "Done."}']]})

    assert generic_adapter(batch, spec, Report()).column("text").to_pylist() == ["Done."]


def test_read_limits_share_the_budget_without_letting_one_source_flood() -> None:
    config = BuildConfig(new_token_budget=1_000_000, read_multiple=1.0, catalog_bytes_per_token=1.0, read_origin_shares=(0.5, 0.5, 0.0), human_cap=10**9)
    huge = spec_of(name="huge", origin="human", text_column="text", rows="1000000", bytes="100000000", date_evidence="2019")
    small = spec_of(name="small", origin="human", text_column="text", rows="1000", bytes="100000", date_evidence="2019")
    books = spec_of(name="books", origin="human", text_column="text", rows="100", bytes="10000000", date_evidence="2019")
    model = spec_of(name="model", origin="ai", text_column="text", generator="gpt-4o", rows="10000", bytes="1000000")
    held = spec_of(name="held", origin="ai", text_column="text", generator="gpt-4o", rows="10000", bytes="1000000", benchmark="B")

    limits = read_limits([huge, small, books, model, held], config, None)

    # Half the budget each to human and model text; within human text, the small source is read whole (100,000
    # tokens) and the other two split the rest (200,000 tokens each), a book costing its 100,000 tokens.
    assert limits == {"huge": 2000, "small": 1000, "books": 2, "model": 5000, "held": None}
    assert read_limits([huge, held], config, 7) == {"huge": 7, "held": 7}


def test_first_corpus_rows_enter_as_they_are() -> None:
    spec = first_corpus_spec({"raid": "RAID"})
    batch = pa.table(
        {
            "id": [5, 6],
            "text": ["A human passage.", "A model passage."],
            "label": [0, 1],
            "source": ["c4", "raid"],
            "generator": ["human", "gpt-4o"],
            "domain": ["web", "news"],
            "doc": pa.array([2**63 + 1, 7], pa.uint64()),
            "whole": [True, False],
        }
    )

    records = adapter_for(spec)(batch, spec, Report())

    assert records.column("first_corpus_id").to_pylist() == [5, 6]
    assert records.column("first_doc").to_pylist() == [2**63 + 1, 7]
    assert records.column("source").to_pylist() == ["c4", "raid"]
    assert records.column("benchmark").to_pylist() == ["", "RAID"]
    assert records.column("date").to_pylist() == ["2021-12-31", None]
