"""Stage 1, read: stream a source from the Hub at its pinned revision into raw records.

How much of each source is read is planned before anything is downloaded (`read_limits`), so the text that
reaches standardizing, tokenizing and cutting is a small multiple of what survives balance, never the whole
download. A catalog config or split may be a pattern (`*`, `[!e]*|e[!n]*`, `*think`, `a|b|c`), expanded
against the repository's configs and splits at its revision into (config, split) entries; the row limit is
shared evenly among them. Each entry is read in up to `max_read_parts` parts, one worker a part, each part a
slice of its files. A part writes its shards and then its report; a part whose report exists is
done, and a part without one is read again from the start, its partial shards replaced.

A rate limit (429), a server error, a timeout or a dropped connection is retried with growing waits, the part
read again from its start. A source that cannot be opened, or whose first rows do not have the columns its
catalog line names (gated, moved, renamed), fails alone: `failed.json` records why, and the build goes on
without it, unless it is a benchmark or the first corpus (`pipeline.py`). Any other failure, and a transient
one that outlasts its retries, stops the build, which resumes from the parts already done.
"""

from __future__ import annotations

import fnmatch
import math
import time
import traceback
import datasets
import httpx

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .adapters import adapter_for
from .catalog import SourceSpec
from .config import ORIGINS, BuildConfig
from .generators import is_legacy
from .records import READ_SCHEMA
from .storage import JsonObject, Report, ShardWriter, write_json


READ_SLACK = 1.25
RETRY_SECONDS = (30, 60, 120, 240, 480)  # waits after a rate limit or a server error, about 15 minutes in all
PATTERN_MARKS = ("*", "?", "[", "|")  # a catalog config or split holding one names several, as `fnmatch` patterns
UNKNOWN_BYTES_PER_ROW = 2_000  # for a catalog line without rows or bytes
UNKNOWN_ROWS = 500_000


@dataclass(frozen=True)
class ReadTask:
    spec: SourceSpec  # with one config, the one this part reads
    split_index: int
    part: int
    parts: int
    limit: int | None  # rows of the split this part reads at most
    directory: Path  # the stage folder of this source
    entry: int | None = None  # the (config, split) entry of the source's plan, when it differs from `split_index`

    @property
    def split(self) -> str:
        return self.spec.splits[self.split_index]

    @property
    def prefix(self) -> str:
        return f"s{self.split_index if self.entry is None else self.entry}-p{self.part:03d}"


class SourceFailed(RuntimeError):
    """A source that could not be opened or whose first rows did not fit its catalog line."""


FILE_BUILDERS = {".jsonl": ("json", {}), ".json": ("json", {}), ".csv": ("csv", {}), ".tsv": ("csv", {"delimiter": "\t"}),
                 ".parquet": ("parquet", {})}


def open_stream(spec: SourceSpec, split: str) -> datasets.IterableDataset:
    """One split of a source as a stream at its pinned revision. A `file:` config reads that one file at the revision
    with the builder its extension names, so a repository that also holds a loading script (HC3, AuTexTification),
    which `datasets` no longer runs, is read all the same; its one split is `train`."""
    if spec.data_file is not None:
        suffix = Path(spec.data_file).suffix.lower()
        if suffix not in FILE_BUILDERS:
            raise ValueError(f"source {spec.name}: no builder for a {suffix!r} file ({spec.data_file})")
        builder, options = FILE_BUILDERS[suffix]
        location = f"hf://datasets/{spec.repo}@{spec.revision}/{spec.data_file}"
        return datasets.load_dataset(builder, data_files={"train": location}, split=split, streaming=True, **options)

    return datasets.load_dataset(spec.repo, spec.config, split=split, revision=spec.revision, streaming=True)


def source_size(spec: SourceSpec) -> tuple[int, int]:
    """(rows, catalog bytes) of a source, filled in where the catalog has neither."""
    rows = spec.rows if spec.rows else (spec.bytes // UNKNOWN_BYTES_PER_ROW if spec.bytes else UNKNOWN_ROWS)
    size = spec.bytes if spec.bytes else rows * UNKNOWN_BYTES_PER_ROW

    return max(rows, 1), max(size, 1)


def water_fill(demands: dict[str, float], budget: float) -> dict[str, float]:
    """An equal share for every claimant, no claimant above its demand, until the budget or the demand runs out."""
    allotted = {name: 0.0 for name in demands}
    open_claims = {name: demand for name, demand in demands.items() if demand > 0}
    remaining = budget
    while open_claims and remaining > 1e-6:
        share = remaining / len(open_claims)
        satisfied = {name: demand for name, demand in open_claims.items() if demand - allotted[name] <= share}
        if not satisfied:
            for name in open_claims:
                allotted[name] += share
            return allotted
        for name, demand in satisfied.items():
            remaining -= demand - allotted[name]
            allotted[name] = demand
            del open_claims[name]

    return allotted


def read_limits(specs: list[SourceSpec], config: BuildConfig, pilot_limit: int | None) -> dict[str, int | None]:
    """Rows of each source worth reading, or None for all of them.

    The read budget is `new_token_budget * read_multiple` tokens of text, the text every later stage reads and
    tokenizes. A source's tokens are its rows times its mean text length (catalog bytes at
    `catalog_bytes_per_token`), counted whole: a 100,000-token book costs its full length to tokenize, though
    it gives at most `max_sub_passages` training rows of at most 8,192 tokens, so a source of long texts is read
    for fewer rows. The budget is split among the origins by `read_origin_shares` and, within an origin,
    water-filled across the sources that give it: every source gets an equal share or all of itself, so a 10 TB
    source and a 10 MB source both contribute and neither floods. A source of several origins claims from each
    in proportion. Unspent budget passes to the origins still short. A source of one group is further limited
    to what its pre-cap keeps (`READ_SLACK` above its group's cap). Benchmarks, the first corpus and a source of
    groups (`group_column`, sampled whole by `group_share` instead) are read
    whole, and a pilot's limit caps every source.
    """
    budget = config.new_token_budget * config.read_multiple
    claims: dict[str, dict[str, float]] = {origin: {} for origin in ORIGINS}
    sizes = {spec.name: source_size(spec) for spec in specs}
    per_row = {name: size / rows / config.catalog_bytes_per_token for name, (rows, size) in sizes.items()}
    for spec in specs:
        if spec.held_out or spec.verbatim or spec.group_column is not None:
            continue
        origins =[origin for origin in spec.origins() if origin in ORIGINS] or list(ORIGINS[:2])
        for origin in origins:
            claims[origin][spec.name] = sizes[spec.name][0] * per_row[spec.name] / len(origins)
    allotted: dict[str, float] = {}
    shares = dict(zip(ORIGINS, config.read_origin_shares, strict=True))
    remaining = budget
    for _ in range(len(ORIGINS)):
        short = {origin: claim for origin, claim in claims.items() if sum(claim.values()) > 1e-6}
        if not short or remaining <= 1e-6:
            break
        weight = sum(shares[origin] for origin in short)
        spent = 0.0
        for origin, claim in short.items():
            given = water_fill(claim, remaining * shares[origin] / weight)
            for name, amount in given.items():
                allotted[name] = allotted.get(name, 0.0) + amount
                claim[name] -= amount
            spent += sum(given.values())
        remaining -= spent
    limits: dict[str, int | None] = {}
    for spec in specs:
        if spec.held_out or spec.verbatim or spec.group_column is not None:
            # A source of groups is read whole: its `group_share` samples whole groups, where a row limit would keep
            # only the stream's first rows (RAID's files run attack by attack).
            limits[spec.name] = pilot_limit
            continue
        candidates = [math.ceil(allotted.get(spec.name, 0.0) / per_row[spec.name])]
        if spec.origin in ORIGINS and not spec.generator_paths:
            candidates.append(math.ceil(group_cap(spec, config) * config.precap_factor * READ_SLACK))
        if pilot_limit is not None:
            candidates.append(pilot_limit)
        limits[spec.name] = max(1, min(candidates))

    return limits


def group_cap(spec: SourceSpec, config: BuildConfig) -> int:
    if spec.origin == "human":
        return config.human_cap
    if spec.origin == "mixed":
        return config.mixed_cap

    return config.legacy_ai_cap if is_legacy(spec.generator or "") else config.ai_cap


def is_pattern(name: str | None) -> bool:
    return name is not None and any(mark in name for mark in PATTERN_MARKS)


def matches(name: str, pattern: str) -> bool:
    """A config or split name against a catalog pattern: shell wildcards, alternatives joined by `|`."""
    return any(fnmatch.fnmatchcase(name, alternative) for alternative in pattern.split("|"))


def source_entries(spec: SourceSpec) -> list[tuple[str | None, str]]:
    """The (config, split) pairs a source names, its wildcard configs and splits expanded at its revision."""
    configs: list[str | None] = [spec.config]
    if is_pattern(spec.config) and spec.data_file is None:
        assert spec.config is not None
        configs = [name for name in datasets.get_dataset_config_names(spec.repo, revision=spec.revision) if matches(name, spec.config)]
    entries: list[tuple[str | None, str]] = []
    for config_name in configs:
        for split in spec.splits:
            if is_pattern(split):
                names = datasets.get_dataset_split_names(spec.repo, config_name, revision=spec.revision)
                entries.extend((config_name, name) for name in names if matches(name, split))
            else:
                entries.append((config_name, split))
    if not entries:
        raise ValueError(f"source {spec.name}: no config and split of {spec.repo} match {spec.config!r} and {list(spec.splits)}")

    return entries


def plan_parts(spec: SourceSpec, limit: int | None, config: BuildConfig) -> list[JsonObject]:
    """Each (config, split) entry of a source and how many parts to read it in: one a file, at most
    `max_read_parts`, fewer for a small limit."""
    entries = source_entries(spec)
    plan: list[JsonObject] = []
    for config_name, split in entries:
        files = open_stream(replace(spec, config=config_name), split).num_shards
        count = min(files, config.max_read_parts)
        if limit is not None:
            count = min(count, max(1, math.ceil(limit / len(entries) / 100_000)))
        plan.append({"config": config_name, "split": split, "parts": max(count, 1)})

    return plan


def entry_tasks(spec: SourceSpec, plan: JsonObject, directory: Path) -> list[ReadTask]:
    """Every part of every entry of a source's read plan, each with its share of the source's row limit."""
    entries = plan["parts"]
    limit = plan["limit"]
    tasks: list[ReadTask] = []
    for index, entry in enumerate(entries):
        concrete = replace(spec, config=entry["config"], splits=(entry["split"],))
        per_part = None if limit is None else math.ceil(limit / len(entries) / entry["parts"])
        tasks.extend(ReadTask(concrete, 0, part, entry["parts"], per_part, directory, entry=index) for part in range(entry["parts"]))

    return tasks


def plan_source(spec: SourceSpec, limit: int | None, config: BuildConfig) -> JsonObject:
    """The read plan of a source, or why it cannot be read."""
    try:
        return with_retries(lambda: {"parts": plan_parts(spec, limit, config), "limit": limit}, f"plan {spec.name}")
    except Exception as error:  # noqa: BLE001 - a source that cannot be opened fails alone, by design, and is reported
        if is_transient(error):
            raise  # still rate-limited after every retry: stop, and a rerun plans it again
        return {"failed": f"{type(error).__name__}: {error}"[:2000], "stage": "open"}


def part_report_path(task: ReadTask) -> Path:
    return task.directory / f"{task.prefix}.json"


def is_transient(error: BaseException) -> bool:
    """A rate limit (429), a server error (5xx), a timeout or a dropped connection, anywhere in the error's chain."""
    seen: BaseException | None = error
    while seen is not None:
        status = getattr(getattr(seen, "response", None), "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True
        if isinstance(seen, (httpx.TimeoutException, httpx.NetworkError, ConnectionError, TimeoutError)):
            return True
        if "429" in str(seen) and "Too Many Requests" in str(seen):
            return True
        seen = seen.__cause__ or seen.__context__

    return False


def pause(seconds: float) -> None:
    time.sleep(seconds)


def with_retries(step: Callable[[], JsonObject], what: str) -> JsonObject:
    """`step()`, run again after each transient failure, waiting longer each time; other failures pass through."""
    for wait in (*RETRY_SECONDS, None):
        try:
            return step()
        except Exception as error:
            if wait is None or not is_transient(error):
                raise
            print(f"{what}: {type(error).__name__}, retrying in {wait} s", flush=True)
            pause(wait)

    raise AssertionError("unreachable: the last attempt returns or raises")


def read_part(task: ReadTask, config: BuildConfig) -> JsonObject:
    """Read one part of one split of a source to shards, again from the start after a transient failure; returns
    its report, or the reason the source failed."""
    return with_retries(lambda: read_part_once(task, config), f"read {task.spec.name} {task.prefix}")


def read_part_once(task: ReadTask, config: BuildConfig) -> JsonObject:
    spec = task.spec
    for stale in task.directory.glob(f"{task.prefix}-*.parquet"):
        stale.unlink()
    adapter = adapter_for(spec)
    writer = ShardWriter(task.directory, task.prefix, READ_SCHEMA, config.shard_rows, config.shard_bytes)
    report = Report()
    rows = 0
    batches = 0
    try:
        stream = open_stream(spec, task.split)
        if task.parts > 1:
            stream = stream.shard(num_shards=task.parts, index=task.part)
        for batch in stream.with_format("arrow").iter(batch_size=config.read_batch_rows):
            if task.limit is not None:
                batch = batch.slice(0, task.limit - rows)
            rows += batch.num_rows
            records = adapter(batch, spec, report)
            batches += 1
            report.rows_in += batch.num_rows
            report.rows_out += records.num_rows
            writer.add(records)
            if task.limit is not None and rows >= task.limit:
                break
    except Exception as error:
        if batches == 0 and not is_transient(error):
            failure = {"failed": f"{type(error).__name__}: {error}"[:2000], "stage": "first rows", "trace": traceback.format_exc()[-2000:]}
            write_json(failure, task.directory / "failed.json")
            return failure
        error.add_note(f"while reading source {spec.name}, split {task.split}, part {task.part} of {task.parts}")
        raise
    writer.flush()
    payload = report.to_json()
    payload["shards"] = [path.name for path in writer.written]
    write_json(payload, part_report_path(task))

    return payload
