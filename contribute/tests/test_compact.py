"""Compaction on local shards with the Hub replaced by an in-memory publisher."""

from __future__ import annotations

import io
import json
import shutil
import pyarrow.parquet as pq
import pytest

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from litmus_contribute.compact import compact, plan_batch, read_shard
from litmus_contribute.config import Settings
from litmus_contribute.shards import (AnnotationRecord, ContainerName, ContributionRecord, LocalStore, RemovalRecord,
                                      ShardWriter, content_hash, list_shards)

from conftest import START, Clock


SETTINGS = Settings()  # windows: shards 900 s, annotations 1800 s, grace 900 s
HOUR = 3600.0


class FakePublisher:
    """The Hub as a dictionary of files with a history of commits; `corrupt` breaks what verification reads back."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.history: list[dict[str, bytes]] = []
        self.corrupt = False
        self.squashes = 0

    def list_files(self, revision: str | None = None) -> list[str]:
        return sorted(self.files if revision is None else self.history[int(revision)])

    def download(self, path: str, revision: str | None = None) -> bytes:
        if revision is None:
            return self.files[path]
        body = self.history[int(revision)][path]
        return body + b"x" if self.corrupt else body

    def commit(self, additions: dict[str, bytes], deletions: list[str], message: str) -> str:
        self.files.update(additions)
        for path in deletions:
            del self.files[path]
        self.history.append(dict(self.files))
        return str(len(self.history) - 1)

    def squash_history(self, message: str) -> None:
        self.history = [dict(self.files)]
        self.squashes += 1


class Container:
    """One web container's writer, at a shared clock."""

    def __init__(self, root: Path, name: str, clock: Clock) -> None:
        self.clock = clock
        self.writer = ShardWriter(LocalStore(root), ContainerName(name), SETTINGS.shard_window_seconds,
                                  SETTINGS.shard_max_bytes, clock=clock)
        self.count = 0

    def now_ms(self) -> int:
        return int(self.clock() * 1000)

    def contribute(self, text: str, annotation: str | None = None) -> str:
        self.count += 1
        identifier = f"{self.now_ms():012x}{self.writer.name.value[:4].encode().hex()}{self.count:016x}"[:36]
        self.writer.append(ContributionRecord(kind="contribution", id=identifier, text=text, content_hash=content_hash(text),
                                              reading={"score": len(text) / 100}, model_version="m1", annotation=annotation,
                                              terms_version="t1", received_at=self.now_ms()))
        return identifier

    def annotate(self, identifier: str, annotation: str | None) -> None:
        self.writer.append(AnnotationRecord(kind="annotation", id=identifier, annotation=annotation, at=self.now_ms()))

    def remove(self, text: str) -> None:
        self.writer.append(RemovalRecord(kind="removal", content_hash=content_hash(text), at=self.now_ms()))


def utc(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


def data_rows(publisher: FakePublisher) -> list[dict[str, Any]]:
    return [row for path in publisher.list_files() if path.startswith("data/")
            for row in pq.read_table(io.BytesIO(publisher.files[path])).to_pylist()]


def test_compaction_dedups_applies_labels_and_removals(tmp_path: Path) -> None:
    clock = Clock()
    a, b = Container(tmp_path, "a", clock), Container(tmp_path, "b", clock)
    first_alpha = a.contribute("alpha")
    a.contribute("alpha")  # a repeat in the same file: stored without its text
    beta = a.contribute("beta", annotation="ai")
    clock.advance(10)
    b.contribute("alpha")
    b.contribute("gamma")
    a.remove("gamma")
    clock.advance(600)
    b.annotate(first_alpha, "human")  # another container labels it
    b.annotate("0" * 36, "ai")  # matches nothing: an orphan
    clock.advance(1400)
    a.annotate(beta, "human")  # 2,010 s after beta: past the window, ignored

    publisher = FakePublisher()
    receipt = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR), card="# card\n")
    assert receipt.hub_commit == "0" and receipt.submissions == 5 and receipt.removed_submissions == 1
    assert (receipt.new_removals, receipt.orphan_annotations, receipt.late_annotations, receipt.torn_lines) == (1, 1, 1, 0)

    rows = {row["text"]: row for row in data_rows(publisher)}
    assert set(rows) == {"alpha", "beta"}
    assert rows["alpha"]["count"] == 3 and rows["alpha"]["id"] == first_alpha
    assert rows["alpha"]["annotation"] == "human"
    assert json.loads(rows["alpha"]["annotation_counts"]) == {"ai": 0, "human": 1, "mixed": 0}
    assert rows["alpha"]["annotated_at"] == utc(START + 610).replace(microsecond=0)
    assert rows["beta"]["annotation"] == "ai" and rows["beta"]["annotated_at"] == rows["beta"]["received_at"]
    assert json.loads(rows["alpha"]["reading"]) == {"score": 0.05}
    assert publisher.files["README.md"] == b"# card\n"
    removals = [path for path in publisher.files if path.startswith("removals/")]
    assert pq.read_table(io.BytesIO(publisher.files[removals[0]])).column("content_hash").to_pylist() == [content_hash("gamma")]
    assert list_shards(tmp_path) == []
    assert json.loads((tmp_path / "receipts" / f"{receipt.batch_id}.json").read_text(encoding="utf-8"))["rows"] == 2


def test_removal_reaches_earlier_batches_and_later_ones(tmp_path: Path) -> None:
    clock = Clock()
    web = Container(tmp_path, "w", clock)
    web.contribute("delta")
    web.contribute("epsilon")
    publisher = FakePublisher()
    compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    assert {row["text"] for row in data_rows(publisher)} == {"delta", "epsilon"}

    clock.advance(6 * HOUR)
    web.remove("delta")
    web.remove("epsilon")
    clock.advance(1)
    web.contribute("delta")  # sent again after its removal: kept out
    web.contribute("zeta")
    receipt = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    assert receipt.new_removals == 2 and receipt.deleted_from_hub  # the first batch emptied and was deleted
    assert {row["text"] for row in data_rows(publisher)} == {"zeta"}
    # the squash leaves one revision, so no earlier version of the dataset holds a removed text
    assert receipt.squashed and publisher.squashes == 1 and len(publisher.history) == 1
    assert not any(b"delta" in body or b"epsilon" in body
                   for revision in publisher.history for path, body in revision.items() if path.startswith("data/"))

    clock.advance(6 * HOUR)
    web.contribute("delta")
    later = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    assert "delta" not in {row["text"] for row in data_rows(publisher)}
    assert not later.squashed and publisher.squashes == 1  # nothing on the Hub held it, so no squash


def test_partial_removal_rewrites_a_batch(tmp_path: Path) -> None:
    clock = Clock()
    web = Container(tmp_path, "w", clock)
    web.contribute("keep me")
    web.contribute("drop me")
    publisher = FakePublisher()
    compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    clock.advance(6 * HOUR)
    web.remove("drop me")
    receipt = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    assert len(receipt.rewritten) == 1 and receipt.rows == 0
    assert {row["text"] for row in data_rows(publisher)} == {"keep me"}
    assert receipt.squashed and len(publisher.history) == 1
    assert b"drop me" not in publisher.history[0][receipt.rewritten[0]]


def test_squash_is_forced_and_skipped_on_a_dry_run(tmp_path: Path) -> None:
    clock = Clock()
    web = Container(tmp_path, "w", clock)
    web.contribute("one")
    publisher = FakePublisher()
    first = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    assert not first.squashed and publisher.squashes == 0  # no removal: the history is kept
    forced = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 6 * HOUR), squash=True)
    assert forced.squashed and forced.hub_commit is None and publisher.squashes == 1  # finishes a stopped run
    clock.advance(6 * HOUR)
    web.remove("one")
    dry = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR), dry_run=True)
    assert dry.deleted_from_hub and not dry.squashed and publisher.squashes == 1


def test_recent_shards_wait(tmp_path: Path) -> None:
    clock = Clock()
    web = Container(tmp_path, "w", clock)
    old = web.contribute("old text")
    clock.advance(HOUR)
    web.annotate(old, "mixed")  # too late for the window, but read
    recent = web.contribute("recent text")
    recent_end = (int(clock()) // 900 + 1) * 900
    now = recent_end + SETTINGS.compaction_grace_seconds + 60  # the recent shard ended before C1 but after C2
    plan = plan_batch(list_shards(tmp_path), utc(now), SETTINGS)
    assert len(plan.consumed) == 1 and len(plan.annotation_only) == 1
    publisher = FakePublisher()
    compact(LocalStore(tmp_path), publisher, SETTINGS, utc(now))
    assert {row["text"] for row in data_rows(publisher)} == {"old text"}
    assert [shard.path for shard in list_shards(tmp_path)] == [plan.annotation_only[0].path]
    assert recent


def test_failed_verification_keeps_the_shards(tmp_path: Path) -> None:
    clock = Clock()
    Container(tmp_path, "w", clock).contribute("text")
    publisher = FakePublisher()
    publisher.corrupt = True
    with pytest.raises(RuntimeError, match="differs"):
        compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    assert len(list_shards(tmp_path)) == 1


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    clock = Clock()
    Container(tmp_path, "w", clock).contribute("text")
    publisher = FakePublisher()
    receipt = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR), card="card", dry_run=True)
    assert receipt.rows == 1 and publisher.history == [] and len(list_shards(tmp_path)) == 1


def test_rerun_after_a_stop_between_push_and_delete(tmp_path: Path) -> None:
    clock = Clock()
    web = Container(tmp_path, "w", clock)
    web.contribute("one")
    web.contribute("two")
    backup = tmp_path.parent / "backup"
    shutil.copytree(tmp_path / "shards", backup)
    publisher = FakePublisher()
    first = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 5 * HOUR))
    shutil.copytree(backup, tmp_path / "shards", dirs_exist_ok=True)  # as if the run had stopped before deleting its shards
    second = compact(LocalStore(tmp_path), publisher, SETTINGS, utc(clock() + 6 * HOUR))
    assert second.batch_id == first.batch_id and second.hub_commit is None
    assert len(publisher.history) == 1 and len(data_rows(publisher)) == 2
    assert list_shards(tmp_path) == []


def test_corruption_fails_loudly(tmp_path: Path) -> None:
    clock = Clock()
    web = Container(tmp_path, "w", clock)
    web.contribute("honest text")
    [shard] = list_shards(tmp_path)
    path = tmp_path / shard.path
    record = json.loads(path.read_text(encoding="utf-8"))
    record["text"] = "tampered text"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        compact(LocalStore(tmp_path), FakePublisher(), SETTINGS, utc(clock() + 5 * HOUR))


def test_torn_last_line_is_counted_and_a_bad_middle_line_raises(tmp_path: Path) -> None:
    clock = Clock()
    web = Container(tmp_path, "w", clock)
    web.contribute("whole line")
    [shard] = list_shards(tmp_path)
    path = tmp_path / shard.path
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"contribution","id":"cut')
    records, torn = read_shard(path)
    assert len(records) == 1 and torn == 1
    path.write_text(path.read_text(encoding="utf-8") + '\n{"kind":"removal","content_hash":"0","at":0}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_shard(path)
