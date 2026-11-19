"""Extend an earlier corpus with a later build's new rows, keeping every earlier row and its `id`.

    python -m litmus_corpus extend --earlier EARLIER/corpus/data --build BUILD --out OUT --workers N \\
        [--trained-first-source NAME ...] [--group-source NAME ...]

An earlier corpus is already embedded and read by models, so its rows keep their `id`, text and every column but
`benchmark` and `doc`; the later build (`litmus_corpus build`, run without the first corpus) holds only new sources and every
benchmark. The result in `OUT/data` is the earlier corpus's rows, then the build's new rows numbered after them:

- **Earlier rows marked anew.** A row whose `benchmark` is its own `source` and whose source is named by
  `--trained-first-source` (a first-corpus source now trained on as a benchmark family's train split) loses the
  mark if it passes the checks its build skipped for a held-out row, as `checks` runs them on a first-corpus
  training row: a degenerate text keeps its mark, and so does one whose text or opening another earlier training
  row (or an unmarked row of lower `id`) holds under another `doc`, or that holds under two origins. Then every
  earlier row that is not a benchmark row and shares a `doc`, a text or an opening of 20 words
  with a benchmark row of the build (`checks/`, before the hold-out, so hidden-label benchmarks count), or whose
  `doc` strata linked to a benchmark text (`links/`), takes that benchmark's name: it leaves training and keeps
  its `id`, as the hold-out stage marks the first corpus's rows.
- **New rows.** A build row that copies an earlier row's text, or its opening under another `doc`, is dropped:
  the earlier row stands for it, and an earlier row is never dropped for a later one (the corpus-wide check drops
  both copies of a text under two origins; here only the new copy goes). A build's benchmark row is appended only
  for a benchmark the earlier corpus does not hold.
- **Groups kept whole.** An earlier training row that stands for a dropped new row of a `--group-source` (a source
  read by `group_column`, whose rows share their group's `doc`) takes that row's `doc`, so the split keeps the
  group, and the earlier row with it, on one side: RAID rows of the first corpus join their RAID source group.

`OUT/report.json` counts what was unmarked, marked and dropped by reason, and the training rows by origin and
source, earlier against new.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from .checks import degenerate_reason, duplicate_codes, hash64, opening
from .config import LABEL_ORIGINS
from .holdout import NO_MARK, key_owners
from .records import CORPUS_SCHEMA, conform
from .storage import JsonObject, write_json, write_parquet


KEY_COLUMNS = ["id", "label", "source", "benchmark", "doc", "first_corpus_id"]
EXTEND_REASONS = ("keep", "earlier_text", "earlier_opening")
UNMARK_REASONS = ("unmarked", "degenerate", "copy_text", "copy_opening", "text_under_two_origins", "opening_under_two_origins")


@dataclass(frozen=True)
class Keys:
    """Columns and content keys of a corpus's rows, in row order: every array is (n,)."""

    ids: np.ndarray
    label: np.ndarray
    source: np.ndarray  # object
    benchmark: np.ndarray  # object
    doc: np.ndarray
    first: np.ndarray  # bool, a first-corpus row
    text_key: np.ndarray
    opening_key: np.ndarray
    degenerate: np.ndarray  # bool, a held-out first-corpus row whose text `checks` would drop; False elsewhere


def shard_keys(path: Path) -> pa.Table:
    """One corpus shard's key columns, the text and opening keys of its texts, and whether each held-out
    first-corpus row's text is degenerate (the only rows that may be unmarked, and the checks skipped them)."""
    table = pq.read_table(path, columns=[*KEY_COLUMNS, "text"])
    texts = table.column("text").to_pylist()
    held_first = [bool(benchmark) and first for benchmark, first in zip(
        table.column("benchmark").to_pylist(), table.column("first_corpus_id").is_valid().to_pylist(), strict=True)]
    keys = table.drop_columns(["text"])
    keys = keys.append_column("text_key", pa.array([hash64(text) for text in texts], type=pa.uint64()))
    keys = keys.append_column("opening_key", pa.array([hash64(opening(text)) for text in texts], type=pa.uint64()))

    return keys.append_column("degenerate", pa.array(
        [held and degenerate_reason(text) is not None for text, held in zip(texts, held_first, strict=True)], type=pa.bool_()))


def corpus_keys(paths: list[Path], workers: int) -> Keys:
    """Every row's keys, in shard order, read in `workers` processes."""
    from .pipeline import run_parallel

    table = pa.concat_tables(run_parallel(shard_keys, [(path,) for path in paths], workers, "keys"))  # type: ignore[arg-type]

    return Keys(
        ids=table.column("id").to_numpy(), label=table.column("label").to_numpy(),
        source=np.asarray(table.column("source").to_pylist(), dtype=object), benchmark=np.asarray(table.column("benchmark").to_pylist(), dtype=object),
        doc=table.column("doc").to_numpy(), first=table.column("first_corpus_id").is_valid().to_numpy(zero_copy_only=False),
        text_key=table.column("text_key").to_numpy(), opening_key=table.column("opening_key").to_numpy(),
        degenerate=table.column("degenerate").to_numpy(zero_copy_only=False),
    )


def unmark_codes(earlier: Keys, candidates: np.ndarray) -> np.ndarray:
    """Why each candidate for unmarking keeps its mark (an index into `UNMARK_REASONS`, 0 to unmark): (n,).

    The checks a held-out row skipped: a degenerate text, then the corpus-wide copy check over the earlier
    training rows and the candidates, where a training row always stands and candidates rank by `id`.
    """
    # candidates: (n,) bool
    codes = np.where(candidates & earlier.degenerate, 1, 0).astype(np.uint8)  # (n,)
    pool = np.flatnonzero((earlier.benchmark == "") | (candidates & (codes == 0)))  # (m,)
    order = np.where(candidates[pool], len(earlier.ids) + earlier.ids[pool], 0)  # (m,) training rows first
    text_codes = duplicate_codes(earlier.text_key[pool], earlier.label[pool], earlier.doc[pool], order, 2, 4)  # (m,)
    opening_codes = duplicate_codes(earlier.opening_key[pool], earlier.label[pool], earlier.doc[pool], order, 3, 5)  # (m,)
    pooled = np.where(text_codes > 0, text_codes, opening_codes)  # (m,)
    chosen = candidates[pool]  # (m,)
    codes[pool[chosen]] = pooled[chosen]

    return codes  # (n,)


def benchmark_keys(build: Path) -> tuple[pa.Table, pa.Table]:
    """The build's benchmark rows before its hold-out (doc, text_key, opening_key, benchmark), and its links of
    benchmark texts to first-corpus documents (first_doc, benchmark)."""
    checks = [path for path in sorted((build / "checks").glob("*/*.parquet"))]
    tables = [pq.read_table(path, columns=["doc", "text_key", "opening_key", "benchmark"]) for path in checks]
    rows = pa.concat_tables([table.filter(pc.not_equal(table.column("benchmark"), "")) for table in tables])
    link_paths = sorted((build / "links").glob("*/*.parquet"))
    links = pa.concat_tables([pq.read_table(path) for path in link_paths]) if link_paths else pa.table(
        {"first_doc": pa.array([], pa.uint64()), "benchmark": pa.array([], pa.string())})

    return rows, links


def marks_by_content(earlier: Keys, open_rows: np.ndarray, rows: pa.Table, links: pa.Table) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The benchmark (an index into the returned names) each open earlier row shares a doc, link, text or opening
    with, `NO_MARK` for none; and the names, as an array whose entry 0 is ''."""
    # open_rows: (n,) bool
    names = np.asarray(["", *sorted(set(rows.column("benchmark").to_pylist()) | set(links.column("benchmark").to_pylist()))], dtype=object)
    number = {name: index for index, name in enumerate(names)}
    row_ids = np.asarray([number[name] for name in rows.column("benchmark").to_pylist()], dtype=np.int32)  # (b,)
    link_ids = np.asarray([number[name] for name in links.column("benchmark").to_pylist()], dtype=np.int32)  # (l,)
    marks = np.full(len(earlier.ids), NO_MARK, dtype=np.int32)  # (n,)
    ways: dict[str, np.ndarray] = {}
    for way, keys, owner_keys, owner_ids in (
        ("doc", earlier.doc, rows.column("doc").to_numpy(), row_ids),
        ("link", earlier.doc, links.column("first_doc").to_numpy(), link_ids),
        ("text", earlier.text_key, rows.column("text_key").to_numpy(), row_ids),
        ("opening", earlier.opening_key, rows.column("opening_key").to_numpy(), row_ids),
    ):
        owners = key_owners(keys, owner_keys, owner_ids)  # (n,)
        hit = open_rows & (marks == NO_MARK) & (owners != NO_MARK)  # (n,)
        marks[hit] = owners[hit]
        ways[way] = hit

    return marks, {"names": names, **ways}  # (n,), (k + 1,) names and (n,) per way


def copy_codes(new: Keys, earlier: Keys) -> np.ndarray:
    """1 (`earlier_text`) for a new row whose text an earlier row holds, 2 (`earlier_opening`) for one whose opening
    an earlier row of another doc holds, 0 to keep: (m,)."""
    codes = np.zeros(len(new.ids), dtype=np.uint8)  # (m,)
    codes[np.isin(new.text_key, earlier.text_key)] = 1
    order = np.argsort(earlier.opening_key, kind="stable")  # (n,)
    sorted_openings = earlier.opening_key[order]  # (n,)
    first = np.searchsorted(sorted_openings, new.opening_key, side="left")  # (m,)
    last = np.searchsorted(sorted_openings, new.opening_key, side="right")  # (m,)
    for row in np.flatnonzero((last > first) & (codes == 0)):
        if (earlier.doc[order[first[row] : last[row]]] != new.doc[row]).any():
            codes[row] = 2

    return codes  # (m,)


def adopted_docs(earlier: Keys, new: Keys, codes: np.ndarray, group_sources: tuple[str, ...], training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The earlier rows' docs, where an earlier training row stands for a dropped new row of a group source (by its
    text, else its opening) the new row's group doc; and which rows took one: (n,), (n,) bool."""
    # codes: (m,) `copy_codes`; training: (n,) bool
    docs = earlier.doc.copy()  # (n,)
    adopted = np.zeros(len(docs), dtype=bool)  # (n,)
    grouped = np.isin(new.source, list(group_sources))  # (m,)
    for code, earlier_keys, new_keys in ((1, earlier.text_key, new.text_key), (2, earlier.opening_key, new.opening_key)):
        chosen = np.flatnonzero(grouped & (codes == code))  # (c,)
        owners = key_owners(earlier_keys, new_keys[chosen], (chosen + 1).astype(np.int32))  # (n,) a new row + 1, or NO_MARK (0)
        hit = training & ~adopted & (owners != NO_MARK)  # (n,)
        docs[hit] = new.doc[owners[hit] - 1]
        adopted |= hit

    return docs, adopted  # (n,), (n,)


def rewrite_earlier(path: Path, target: Path, changes: dict[int, tuple[str, int]]) -> int:
    """One earlier shard with its changed rows' new `benchmark` and `doc` values, every other column as it was;
    returns its rows."""
    table = pq.read_table(path)
    ids = table.column("id").to_numpy()  # (r,)
    benchmarks = table.column("benchmark").to_pylist()
    docs = table.column("doc").to_pylist()
    changed = [changes.get(int(row), (benchmark, doc)) for row, benchmark, doc in zip(ids, benchmarks, docs, strict=True)]
    table = table.set_column(table.schema.get_field_index("benchmark"), "benchmark", pa.array([pair[0] for pair in changed], type=pa.string()))
    table = table.set_column(table.schema.get_field_index("doc"), "doc", pa.array([pair[1] for pair in changed], type=table.schema.field("doc").type))
    write_parquet(conform(table, CORPUS_SCHEMA), target)

    return table.num_rows


def append_new(path: Path, target: Path, keep: np.ndarray, first_id: int) -> int:
    """One build shard's kept rows, numbered from `first_id`; returns their count."""
    # keep: (r,) bool
    table = pq.read_table(path).filter(pa.array(keep))
    ids = pa.array(np.arange(first_id, first_id + table.num_rows, dtype=np.int64))  # (k,)
    table = table.set_column(table.schema.get_field_index("id"), "id", ids)
    write_parquet(conform(table, CORPUS_SCHEMA), target)

    return table.num_rows


def origin_counts(label: np.ndarray, source: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, int]]:
    """Rows by origin, and by source and origin, among `mask`."""
    # label, source, mask: (n,)
    counts: dict[str, dict[str, int]] = {"origin": {}, "source": {}}
    for code, origin in LABEL_ORIGINS.items():
        chosen = mask & (label == code)  # (n,)
        if chosen.any():
            counts["origin"][origin] = int(chosen.sum())
            names, numbers = np.unique(source[chosen], return_counts=True)
            for name, number in zip(names, numbers, strict=True):
                counts["source"][f"{name}|{origin}"] = int(number)

    return counts


def extend(earlier_data: Path, build: Path, out: Path, workers: int, trained_first_sources: tuple[str, ...],
           group_sources: tuple[str, ...] = ()) -> JsonObject:
    """Write the extended corpus to `out / 'data'` and its report; returns the report."""
    from .pipeline import run_parallel

    earlier_paths = sorted(earlier_data.glob("*.parquet"))
    new_paths = sorted((build / "corpus" / "data").glob("*.parquet"))
    assert earlier_paths and new_paths, f"no shards in {earlier_data} or {build / 'corpus' / 'data'}"
    earlier = corpus_keys(earlier_paths, workers)
    assert np.array_equal(earlier.ids, np.arange(len(earlier.ids))), "the earlier corpus's ids run 0..n-1 in shard order"
    new = corpus_keys(new_paths, workers)

    # Earlier rows: unmark the family train splits now trained on, then mark by the build's benchmark content.
    held = earlier.benchmark != ""  # (n,)
    for name in trained_first_sources:
        assert ((earlier.source == name) & (earlier.benchmark == name) & earlier.first).any(), f"no first-corpus row of {name} is marked {name}"
    candidates = held & earlier.first & np.isin(earlier.source, list(trained_first_sources)) & (earlier.benchmark == earlier.source)  # (n,)
    held_back = unmark_codes(earlier, candidates)  # (n,)
    unmarked = candidates & (held_back == 0)  # (n,)
    benchmark = earlier.benchmark.copy()  # (n,)
    benchmark[unmarked] = ""
    rows, links = benchmark_keys(build)
    marks, ways = marks_by_content(earlier, benchmark == "", rows, links)
    names = ways.pop("names")
    marked = marks != NO_MARK  # (n,)
    benchmark[marked] = names[marks[marked]]

    # New rows: drop copies of earlier rows; append benchmark rows only for benchmarks the earlier corpus lacks.
    earlier_benchmarks = set(earlier.benchmark[held].tolist())
    is_benchmark = new.benchmark != ""  # (m,)
    codes = copy_codes(new, earlier)  # (m,)
    codes[is_benchmark] = 0
    keep = (codes == 0) & (~is_benchmark | ~np.isin(new.benchmark, list(earlier_benchmarks)))  # (m,)
    docs, adopted = adopted_docs(earlier, new, codes, group_sources, benchmark == "")  # (n,), (n,)

    shards = out / "data"
    counts = [pq.ParquetFile(path).metadata.num_rows for path in new_paths]
    offsets = np.r_[0, np.cumsum(counts)]  # (s + 1,)
    earlier_offsets = np.r_[0, np.cumsum([pq.ParquetFile(path).metadata.num_rows for path in earlier_paths])]  # (e + 1,)
    changed_rows = np.flatnonzero((benchmark != earlier.benchmark) | (docs != earlier.doc))  # (c,) ascending, as ids run in shard order
    bounds = np.searchsorted(changed_rows, earlier_offsets)  # (e + 1,)
    rewrites = [(path, shards / f"shard-{index:05d}.parquet",
                 {int(row): (str(benchmark[row]), int(docs[row])) for row in changed_rows[bounds[index] : bounds[index + 1]]})
                for index, path in enumerate(earlier_paths)]
    run_parallel(rewrite_earlier, rewrites, workers, "earlier")
    first_id = len(earlier.ids)
    tasks = []
    for path, start, end in zip(new_paths, offsets[:-1], offsets[1:], strict=True):
        kept = keep[start:end]  # (r,)
        if kept.any():
            tasks.append((path, shards / f"shard-{len(earlier_paths) + len(tasks):05d}.parquet", kept, first_id))
            first_id += int(kept.sum())
    written = run_parallel(append_new, tasks, workers, "new")
    assert sum(written) == int(keep.sum()), "every kept new row is written once"  # type: ignore[arg-type]

    report: JsonObject = {
        "earlier_rows": len(earlier.ids), "new_rows": int(keep.sum()), "rows": len(earlier.ids) + int(keep.sum()),
        "earlier": {
            "unmarked_trained_first_sources": {name: int((unmarked & (earlier.source == name)).sum()) for name in trained_first_sources},
            "trained_first_sources_by_check": {name: {reason: int((candidates & (earlier.source == name) & (held_back == code)).sum())
                                                      for code, reason in enumerate(UNMARK_REASONS)} for name in trained_first_sources},
            "marked_by_content": {way: int(hit.sum()) for way, hit in ways.items()},
            "marked_by_benchmark": {str(name): int(count) for name, count in zip(*np.unique(benchmark[marked], return_counts=True), strict=True)},
            "unmarked_then_marked_again": int((unmarked & marked).sum()),
            "group_docs_adopted": {str(name): int(count) for name, count in zip(*np.unique(earlier.source[adopted], return_counts=True), strict=True)},
            "training_rows_before": int((~held).sum()), "training_rows_after": int((benchmark == "").sum()),
        },
        "new": {
            "build_rows": len(new.ids),
            "dropped": {reason: int((codes == code).sum()) for code, reason in enumerate(EXTEND_REASONS) if code},
            "benchmark_rows_appended": int((keep & is_benchmark).sum()),
            "benchmark_rows_left_to_the_earlier_corpus": int((is_benchmark & ~keep).sum()),
        },
        "rows_by": {
            "earlier_training": origin_counts(earlier.label, earlier.source, benchmark == ""),
            "new_training": origin_counts(new.label, new.source, keep & ~is_benchmark),
        },
    }
    write_json(report, out / "report.json")
    tqdm.write(f"extended: {report['rows']} rows, {report['new_rows']} new")

    return report
