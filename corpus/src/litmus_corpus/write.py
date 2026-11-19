"""Stage 8, write: the corpus shards in the README's columns, its `sources.csv`, and samples to read by eye.

`id` numbers the rows in shard order from 0. The samples are 20 rows a cell of (origin, format), the rows of
lowest text hash, one text file a cell; `long_eval` documents are left out of them, as they run to 131,072
tokens.
"""

from __future__ import annotations

import csv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from collections import defaultdict
from pathlib import Path

from .catalog import FIRST_CORPUS, SourceSpec
from .config import FIRST_CORPUS_REPO, FIRST_CORPUS_REVISION, FIRST_CORPUS_SPLITS, LABEL_ORIGINS, ORIGINS
from .generators import era
from .records import CORPUS_SCHEMA, conform
from .storage import JsonObject, write_parquet


SAMPLE_COLUMNS = ["source", "generator", "domain", "language", "tokens", "whole", "mix", "benchmark", "first_corpus_id", "text_unchanged"]


def corpus_shard(source_path: Path, target_path: Path, first_id: int) -> int:
    """One corpus shard from a balanced shard, its rows numbered from `first_id`; returns its row count."""
    table = pq.read_table(source_path)
    labels = table.column("label").to_pylist()
    assert set(labels) <= {0, 1, 2}, f"{source_path}: a row without a corpus label reached the write"
    ids = pa.array(np.arange(first_id, first_id + table.num_rows, dtype=np.int64))  # (n,)
    eras = pa.array([era(generator, LABEL_ORIGINS[label]) for generator, label in zip(table.column("generator").to_pylist(), labels, strict=True)])
    corpus = conform(table.append_column("id", ids).append_column("era", eras), CORPUS_SCHEMA)
    write_parquet(corpus, target_path)

    return corpus.num_rows


def sample_rows(shards: list[Path], per_cell: int) -> dict[tuple[str, str], list[tuple[int, Path, int]]]:
    """For each (origin, format), the `per_cell` training-length rows of lowest text hash, as (hash, shard, row)."""
    chosen: dict[tuple[str, str], list[tuple[int, Path, int]]] = defaultdict(list)
    for shard in shards:
        table = pq.read_table(shard, columns=["label", "format", "text_key", "long_eval"])
        labels = table.column("label").to_numpy()  # (n,)
        formats = np.array(table.column("format").to_pylist(), dtype=object)  # (n,)
        keys = table.column("text_key").to_numpy()  # (n,)
        short = ~table.column("long_eval").to_numpy(zero_copy_only=False)  # (n,)
        for label in np.unique(labels):
            for format_name in np.unique(formats[labels == label]):
                rows = np.flatnonzero((labels == label) & (formats == format_name) & short)  # (m,)
                lowest = rows[np.argsort(keys[rows], kind="stable")[:per_cell]]  # (k,)
                chosen[(LABEL_ORIGINS[int(label)], str(format_name))].extend((int(keys[row]), shard, int(row)) for row in lowest)

    return {cell: sorted(rows, key=lambda item: item[0])[:per_cell] for cell, rows in chosen.items()}


def write_samples(shards: list[Path], first_ids: dict[Path, int], directory: Path, per_cell: int) -> int:
    """Text files of sample rows, one a cell; returns the files written."""
    directory.mkdir(parents=True, exist_ok=True)
    cells = sample_rows(shards, per_cell)
    wanted: dict[Path, list[int]] = defaultdict(list)
    for rows in cells.values():
        for _, shard, row in rows:
            wanted[shard].append(row)
    records: dict[tuple[Path, int], JsonObject] = {}
    for shard, rows in wanted.items():
        taken = pq.read_table(shard, columns=["text", *SAMPLE_COLUMNS]).take(pa.array(rows)).to_pylist()
        records.update({(shard, row): record for row, record in zip(rows, taken, strict=True)})
    for (origin, format_name), rows in sorted(cells.items()):
        blocks: list[str] = []
        for _, shard, row in rows:
            record = records[(shard, row)]
            header = " | ".join(f"{name}={record[name]}" for name in SAMPLE_COLUMNS)
            blocks.append(f"=== id={first_ids[shard] + row} | {header} ===\n{record['text']}\n")
        (directory / f"{origin}-{format_name}.txt").write_text("\n".join(blocks), encoding="utf-8")

    return len(cells)


SOURCE_COUNTS = (*ORIGINS, "total", "first_corpus", "benchmark_rows", "long_eval")


def write_sources(shards: list[Path], specs: list[SourceSpec], read_counts: dict[str, int], path: Path) -> None:
    """`sources.csv`: each source's repository, revision and licence, rows read, and corpus rows by origin.

    A first-corpus row keeps its source's name, so a source's rows count both what was read from it now and
    what the first corpus holds of it (`first_corpus`); a source only the first corpus holds is written with
    the first corpus's repository and revision.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for shard in shards:
        table = pq.read_table(shard, columns=["source", "label", "first_corpus_id", "benchmark", "long_eval"])
        firsts = table.column("first_corpus_id").is_valid().to_pylist()
        columns = (table.column(name).to_pylist() for name in ("source", "label", "benchmark", "long_eval"))
        for (source, label, benchmark, long_eval), first in zip(zip(*columns, strict=True), firsts, strict=True):
            by_source = counts[source]
            by_source[LABEL_ORIGINS[label]] += 1
            by_source["total"] += 1
            by_source["first_corpus"] += int(first)
            by_source["benchmark_rows"] += int(bool(benchmark))
            by_source["long_eval"] += int(long_eval)
    known = {spec.name: spec for spec in specs if spec.name != FIRST_CORPUS}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "repo", "config", "split", "revision", "license", "benchmark", "rows_read", *SOURCE_COUNTS])
        for name in sorted(set(known) | set(counts)):
            spec = known.get(name)
            if spec is None:
                described = [name, FIRST_CORPUS_REPO, "", "|".join(FIRST_CORPUS_SPLITS), FIRST_CORPUS_REVISION, "", "", 0]
            else:
                described = [name, spec.repo, spec.config or "", "|".join(spec.splits), spec.revision, spec.license, spec.benchmark, read_counts.get(name, 0)]
            writer.writerow([*described, *(counts[name][column] for column in SOURCE_COUNTS)])
