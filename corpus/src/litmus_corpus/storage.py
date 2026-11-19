"""Parquet shards and JSON reports on disk, written so that a stopped build resumes where it stopped.

Every file is written to a temporary name and moved into place, so a file that exists is whole. A shard's
report is written after the shard, so a report on disk marks the shard done.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]  # a report as JSON holds it; its values are whatever a stage counted
PARQUET_COMPRESSION = "zstd"
MAX_ROW_GROUP_BYTES = 128 * 2**20  # text bytes a parquet row group holds at most, before its last row
ROW_OVERHEAD_BYTES = 256  # the other columns of a row, generously
EXAMPLES_PER_RULE = 3
EXAMPLE_CHARACTERS = 240


def row_group_starts(table: pa.Table) -> list[int]:
    """Row offsets that split a table into row groups of about `MAX_ROW_GROUP_BYTES` of text each.

    Rows are grouped by bytes, not by count, since one text may be 131,072 tokens and another 8; a group
    exceeds the limit by at most its last row, so readers that refuse groups over 1 GiB never see one.
    """
    if table.num_rows == 0:
        return [0]
    if "text" in table.column_names:
        sizes = np.asarray(pc.binary_length(table.column("text")).fill_null(0)).astype(np.int64) + ROW_OVERHEAD_BYTES  # (n,)
    else:
        sizes = np.full(table.num_rows, max(1, table.nbytes // table.num_rows), dtype=np.int64)  # (n,)
    before = np.cumsum(sizes) - sizes  # (n,) bytes of the rows before each row
    groups = before // MAX_ROW_GROUP_BYTES  # (n,)

    return np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]]).tolist()


def write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    starts = [*row_group_starts(table), table.num_rows]
    with pq.ParquetWriter(temporary, table.schema, compression=PARQUET_COMPRESSION) as writer:
        for first, last in zip(starts[:-1], starts[1:], strict=True):
            piece = table.slice(first, last - first)
            writer.write_table(piece, row_group_size=max(1, piece.num_rows))
    os.replace(temporary, path)


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> JsonObject:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Report:
    """Counts in and out of one stage for one shard or source, and why rows left or changed.

    `dropped` and `changed` count by reason or rule, then by origin. `examples` keeps a few texts a rule
    changed, as the part it removed, so a reader can audit the rule.
    """

    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    changed: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def drop(self, reason: str, origin: str, count: int = 1) -> None:
        if count:
            self.dropped[reason][origin] += count

    def change(self, rule: str, origin: str, removed: str) -> None:
        self.changed[rule][origin] += 1
        if len(self.examples[rule]) < EXAMPLES_PER_RULE:
            self.examples[rule].append(removed[:EXAMPLE_CHARACTERS])

    def count(self, table: str, key: str, number: int = 1) -> None:
        """A free count kept beside the drops, as `counts[table][key]`, such as rows per generator."""
        if number:
            self.counts[table][key] += number

    def merge(self, other: Report) -> None:
        self.rows_in += other.rows_in
        self.rows_out += other.rows_out
        for target, source in ((self.dropped, other.dropped), (self.changed, other.changed), (self.counts, other.counts)):
            for key, by_origin in source.items():
                for origin, number in by_origin.items():
                    target[key][origin] += number
        for rule, texts in other.examples.items():
            room = EXAMPLES_PER_RULE - len(self.examples[rule])
            self.examples[rule].extend(texts[:room])

    def to_json(self) -> JsonObject:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped": {key: dict(value) for key, value in sorted(self.dropped.items())},
            "changed": {key: dict(value) for key, value in sorted(self.changed.items())},
            "examples": {key: list(value) for key, value in sorted(self.examples.items()) if value},
            "counts": {key: dict(value) for key, value in sorted(self.counts.items())},
        }

    @classmethod
    def from_json(cls, payload: JsonObject) -> Report:
        report = cls(rows_in=payload["rows_in"], rows_out=payload["rows_out"])
        for target, key in ((report.dropped, "dropped"), (report.changed, "changed"), (report.counts, "counts")):
            for name, by_origin in payload[key].items():
                for origin, number in by_origin.items():
                    target[name][origin] = number
        for rule, texts in payload["examples"].items():
            report.examples[rule] = list(texts)

        return report

    def check_balance(self, where: str) -> None:
        """Every row in is a row out or a counted drop; a stage that splits rows into passages skips this."""
        dropped = sum(sum(by_origin.values()) for by_origin in self.dropped.values())
        assert self.rows_in == self.rows_out + dropped, f"{where}: {self.rows_in} in, {self.rows_out} out, {dropped} dropped"


class ShardWriter:
    """Batches of rows to numbered parquet shards of at most `max_rows` rows or `max_bytes` bytes."""

    def __init__(self, directory: Path, prefix: str, schema: pa.Schema, max_rows: int, max_bytes: int) -> None:
        self.directory = directory
        self.prefix = prefix
        self.schema = schema
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.pending: list[pa.Table] = []
        self.pending_rows = 0
        self.pending_bytes = 0
        self.written: list[Path] = []

    def add(self, table: pa.Table) -> None:
        assert table.schema.equals(self.schema), f"{self.prefix}: schema {table.schema} is not {self.schema}"
        if table.num_rows == 0:
            return
        self.pending.append(table)
        self.pending_rows += table.num_rows
        self.pending_bytes += table.nbytes
        if self.pending_rows >= self.max_rows or self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.directory / f"{self.prefix}-{len(self.written):05d}.parquet"
        write_parquet(pa.concat_tables(self.pending).combine_chunks(), path)
        self.written.append(path)
        self.pending, self.pending_rows, self.pending_bytes = [], 0, 0


def shard_paths(stage_directory: Path, sources: list[str]) -> list[Path]:
    """The parquet shards a stage wrote for these sources, in a fixed order."""
    paths: list[Path] = []
    for source in sorted(sources):
        paths.extend(sorted((stage_directory / source).glob("*.parquet")))

    return paths


def report_path(shard: Path) -> Path:
    return shard.with_suffix(".json")


def merged_reports(paths: list[Path]) -> dict[str, Report]:
    """Shard reports merged by source, the folder each shard sits in."""
    by_source: dict[str, Report] = {}
    for path in paths:
        by_source.setdefault(path.parent.name, Report()).merge(Report.from_json(read_json(path)))

    return by_source


def stage_report(by_source: dict[str, Report]) -> JsonObject:
    """A stage's report: totals, and each source's counts."""
    total = Report()
    for report in by_source.values():
        total.merge(report)

    return {"total": total.to_json(), "sources": {name: report.to_json() for name, report in sorted(by_source.items())}}
