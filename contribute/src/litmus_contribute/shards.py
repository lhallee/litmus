"""Append-only shard files on a volume, one per container and time window, safe under concurrent containers.

Modal volumes keep the last committed writer of a file and lose the rest, so no two containers ever write one file:
every name carries its writer, and a writer only appends to its own files. Layout under the store's root:

    shards/<YYYYMMDD>/<window start, UTC>_<window seconds>_<writer>_<sequence>.jsonl
    receipts/<batch id>.json      # one per compaction

Each line is one record (JSON): a contribution, an annotation of one, or a removal of a text by its content hash.
A repeat of a text already in the same file is stored without its text and reading, which compaction fills back in.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict


SHARD_NAME = re.compile(r"(?P<start>\d{8}T\d{6}Z)_(?P<seconds>\d+)_(?P<writer>[A-Za-z0-9-]+)_(?P<sequence>\d{4})\.jsonl")
WRITER = re.compile(r"[^A-Za-z0-9-]")


class ContributionRecord(TypedDict):
    kind: Literal["contribution"]
    id: str
    text: str | None  # None for a repeat of a text earlier in the same shard
    content_hash: str
    reading: dict[str, Any] | None  # None for a repeat
    model_version: str
    annotation: str | None
    terms_version: str
    received_at: int  # milliseconds since the epoch, UTC


class AnnotationRecord(TypedDict):
    kind: Literal["annotation"]
    id: str
    annotation: str | None
    at: int


class RemovalRecord(TypedDict):
    kind: Literal["removal"]
    content_hash: str
    at: int


Record = ContributionRecord | AnnotationRecord | RemovalRecord


class Volume(Protocol):
    """What a store needs of a Modal Volume."""

    def commit(self) -> None: ...

    def reload(self) -> None: ...


class ShardStore(Protocol):
    root: Path

    def commit(self) -> None: ...

    def reload(self) -> None: ...


class LocalStore:
    """A local directory: the tests' stand-in for the volume, where writes are visible at once."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


class VolumeStore:
    """A Modal Volume mounted at `root`: files are written through the mount, made durable by `commit`."""

    def __init__(self, root: Path, volume: Volume) -> None:
        self.root = root
        self.volume = volume

    def commit(self) -> None:
        self.volume.commit()

    def reload(self) -> None:
        self.volume.reload()


def content_hash(text: str) -> str:
    """SHA-256 of the text's UTF-8 bytes exactly as received."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ContainerName:
    """This container's name in its shard files and the usage table: Modal's task id when there is one, and a random
    suffix either way. It is drawn on first use, never at import, so a memory snapshot cannot fix one name for every
    container restored from it."""

    def __init__(self, fixed: str | None = None) -> None:
        self._value = fixed

    @property
    def value(self) -> str:
        if self._value is None:
            stem = WRITER.sub("-", os.environ.get("MODAL_TASK_ID") or "local")[:48]
            self._value = f"{stem}-{secrets.token_hex(4)}"
        return self._value


@dataclass(frozen=True)
class ShardFile:
    path: Path  # relative to the store's root
    window_start: datetime
    window_seconds: int
    writer: str
    sequence: int

    @property
    def window_end(self) -> datetime:
        return self.window_start + timedelta(seconds=self.window_seconds)

    @classmethod
    def parse(cls, relative: Path) -> ShardFile:
        match = SHARD_NAME.fullmatch(relative.name)
        if match is None:
            raise ValueError(f"{relative} is not a shard name")
        start = datetime.strptime(match["start"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return cls(path=relative, window_start=start, window_seconds=int(match["seconds"]), writer=match["writer"],
                   sequence=int(match["sequence"]))


def shard_path(window_start: datetime, window_seconds: int, writer: str, sequence: int) -> Path:
    stamp = window_start.strftime("%Y%m%dT%H%M%SZ")
    return Path("shards") / window_start.strftime("%Y%m%d") / f"{stamp}_{window_seconds}_{writer}_{sequence:04d}.jsonl"


def list_shards(root: Path) -> list[ShardFile]:
    """Every shard under `root`, oldest window first."""
    shards = [ShardFile.parse(path.relative_to(root)) for path in (root / "shards").glob("*/*.jsonl")]
    return sorted(shards, key=lambda shard: (shard.window_start, shard.writer, shard.sequence))


def record_line(record: Record) -> str:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


class ShardWriter:
    """Appends records to this container's shard for the current window and commits on a schedule."""

    def __init__(self, store: ShardStore, name: ContainerName, window_seconds: int, max_bytes: int,
                 clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.name = name
        self.window_seconds = window_seconds
        self.max_bytes = max_bytes
        self.clock = clock
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._window_start: datetime | None = None
        self._sequence = 0
        self._seen: set[tuple[str, str, str]] = set()  # (content hash, model version, terms version) in this file
        self._dirty = False
        self.records_written = 0

    def _target(self, now: float, size: int) -> Path:
        """The file to append `size` bytes to: a new one when the window turns or the current one would pass the cap."""
        start_seconds = int(now) // self.window_seconds * self.window_seconds
        window_start = datetime.fromtimestamp(start_seconds, tz=UTC)
        if window_start != self._window_start:
            self._window_start, self._sequence = window_start, 0
            self._open_new()
        assert self._path is not None
        full = self.store.root / self._path
        if full.exists() and full.stat().st_size + size > self.max_bytes:
            self._sequence += 1
            self._open_new()
        return self.store.root / self._path

    def _open_new(self) -> None:
        assert self._window_start is not None
        self._path = shard_path(self._window_start, self.window_seconds, self.name.value, self._sequence)
        self._seen = set()

    def append(self, record: Record) -> None:
        """Append one record (one line, written whole, the file closed after) to this container's own file."""
        with self._lock:
            line = record_line(record)
            target = self._target(self.clock(), len(line.encode("utf-8")))
            if record["kind"] == "contribution":
                key = (record["content_hash"], record["model_version"], record["terms_version"])
                if key in self._seen:
                    line = record_line({**record, "text": None, "reading": None})
                self._seen.add(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._dirty = True
            self.records_written += 1

    def commit_if_dirty(self) -> bool:
        """Commit the store if anything was written since the last commit."""
        with self._lock:
            if not self._dirty:
                return False
            self.store.commit()
            self._dirty = False
            return True
