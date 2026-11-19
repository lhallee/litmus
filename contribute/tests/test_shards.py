"""Shard naming, rolling, repeats, and concurrent writers that never share a file."""

from __future__ import annotations

import json
import threading

from datetime import UTC, datetime
from pathlib import Path
from litmus_contribute.shards import (ContainerName, ContributionRecord, LocalStore, ShardFile, ShardWriter, VolumeStore,
                                      content_hash, list_shards, shard_path)

from conftest import START, Clock


def contribution(text: str, at_ms: int, identifier: str) -> ContributionRecord:
    return ContributionRecord(kind="contribution", id=identifier, text=text, content_hash=content_hash(text),
                              reading={"score": 0.5}, model_version="m1", annotation=None, terms_version="t1",
                              received_at=at_ms)


def test_shard_name_round_trip() -> None:
    start = datetime(2026, 11, 18, 3, 45, tzinfo=UTC)
    path = shard_path(start, 900, "ta-01ABC-1a2b3c4d", 7)
    assert path == Path("shards/20261118/20261118T034500Z_900_ta-01ABC-1a2b3c4d_0007.jsonl")
    shard = ShardFile.parse(path)
    assert (shard.window_start, shard.window_seconds, shard.writer, shard.sequence) == (start, 900, "ta-01ABC-1a2b3c4d", 7)
    assert shard.window_end == datetime(2026, 11, 18, 4, 0, tzinfo=UTC)


def test_container_name_is_lazy_and_unique(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TASK_ID", "ta-01K/odd id")
    first, second = ContainerName(), ContainerName()
    assert first._value is None
    assert first.value.startswith("ta-01K-odd-id-") and first.value == first.value
    assert first.value != second.value


def test_windows_and_size_roll(tmp_path: Path) -> None:
    clock = Clock()
    writer = ShardWriter(LocalStore(tmp_path), ContainerName("w1"), window_seconds=900, max_bytes=600, clock=clock)
    for index in range(4):
        writer.append(contribution(f"text {index} " + "y" * 150, int(clock() * 1000), f"id{index}"))
    clock.advance(900)
    writer.append(contribution("next window", int(clock() * 1000), "id9"))
    shards = list_shards(tmp_path)
    assert len({shard.window_start for shard in shards}) == 2
    assert max(shard.sequence for shard in shards) >= 1  # the size cap rolled the first window
    for shard in shards:
        assert (tmp_path / shard.path).stat().st_size <= 600
    assert sum(len((tmp_path / shard.path).read_text(encoding="utf-8").splitlines()) for shard in shards) == 5


def test_repeat_in_one_file_drops_text_and_reading(tmp_path: Path) -> None:
    clock = Clock()
    writer = ShardWriter(LocalStore(tmp_path), ContainerName("w1"), window_seconds=900, max_bytes=10**6, clock=clock)
    writer.append(contribution("same", int(clock() * 1000), "a"))
    writer.append(contribution("same", int(clock() * 1000), "b"))
    clock.advance(900)
    writer.append(contribution("same", int(clock() * 1000), "c"))
    lines = {json.loads(line)["id"]: json.loads(line) for shard in list_shards(tmp_path)
             for line in (tmp_path / shard.path).read_text(encoding="utf-8").splitlines()}
    assert lines["a"]["text"] == "same" and lines["b"]["text"] is None and lines["b"]["reading"] is None
    assert lines["c"]["text"] == "same"  # a new file starts with the text again


def test_concurrent_writers_never_share_a_file(tmp_path: Path) -> None:
    store, clock = LocalStore(tmp_path), Clock(START)
    writers = [ShardWriter(store, ContainerName(f"c{index}"), 900, 10**6, clock=clock) for index in range(8)]

    def run(writer: ShardWriter, index: int) -> None:
        for count in range(50):
            writer.append(contribution(f"writer {index} text {count}", int(clock() * 1000), f"{index}-{count}"))

    threads = [threading.Thread(target=run, args=(writer, index)) for index, writer in enumerate(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    shards = list_shards(tmp_path)
    assert len(shards) == 8
    for shard in shards:
        writer_index = shard.writer[1:]
        lines = (tmp_path / shard.path).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 50
        assert all(json.loads(line)["id"].startswith(f"{writer_index}-") for line in lines)


class CountingVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


def test_commit_only_when_dirty(tmp_path: Path) -> None:
    volume = CountingVolume()
    writer = ShardWriter(VolumeStore(tmp_path, volume), ContainerName("w"), 900, 10**6, clock=Clock())
    assert not writer.commit_if_dirty()
    writer.append(contribution("t", 0, "a"))
    assert writer.commit_if_dirty() and not writer.commit_if_dirty()
    assert volume.commits == 1
