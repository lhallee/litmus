"""Compaction: closed shards to one deduplicated parquet batch on a private Hub dataset, removals applied everywhere,
then the shards deleted from the volume once the push is verified.

Which shards a run reads (C1 = now - grace; A the annotation window, W the shard window):

- contributions come from shards whose window ended by C2 = C1 - A - W, which are then deleted;
- annotations and removals come from every shard ended by C1, since an annotation of a text stored by C2 arrives by
  C2 + A and so lies in a shard ended by C1. Shards between C2 and C1 stay for the next run.

The Hub dataset (the record; the volume is a working copy):

    data/<YYYY>/<batch id>.parquet      one per run, rows deduplicated within it (COLUMNS)
    removals/<batch id>.parquet         content hashes removed in that run, never their text
    README.md                           the dataset card, written once

A removed hash is dropped from every batch built after it, and every Hub data file holding it is rewritten (or
deleted when it empties) in the same commit that adds the batch. Older Hub revisions would still hold the text, so a
run that rewrites or deletes a data file then squashes the dataset's history to one commit
(`HfApi.super_squash_history`), after the push is verified: the terms promise that a removal reaches earlier versions
of the private dataset at the compilation that applies it, within 30 days of the request. The dataset is
private and never published, so the squash loses only the working history; the receipts on the volume keep the record
of each run. A run that stopped at the squash is finished by a rerun with `squash=True`.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .shards import AnnotationRecord, ContributionRecord, Record, ShardFile, ShardStore, content_hash, list_shards


SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("content_hash", pa.string()),
    ("reading", pa.string()),  # the engine's reading, JSON
    ("model_version", pa.string()),
    ("annotation", pa.string()),  # the plurality of the submissions' labels; null when none or tied
    ("annotated_at", pa.timestamp("ms", tz="UTC")),
    ("received_at", pa.timestamp("ms", tz="UTC")),
    ("terms_version", pa.string()),
    ("count", pa.int32()),  # submissions of this text under this model and terms version
    ("annotation_counts", pa.string()),  # JSON {"ai": n, "human": n, "mixed": n}: label agreement, for review
])
REMOVALS_SCHEMA = pa.schema([("content_hash", pa.string()), ("requested_at", pa.timestamp("ms", tz="UTC"))])


class Publisher(Protocol):
    def list_files(self, revision: str | None = None) -> list[str]: ...

    def download(self, path: str, revision: str | None = None) -> bytes: ...

    def commit(self, additions: dict[str, bytes], deletions: list[str], message: str) -> str: ...

    def squash_history(self, message: str) -> None: ...


class HubPublisher:
    """A private Hugging Face dataset; `huggingface_hub` reads `HF_TOKEN` from the environment."""

    def __init__(self, repo_id: str) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi()
        self.exists = self.api.repo_exists(repo_id, repo_type="dataset")
        if self.exists and not self.api.repo_info(repo_id, repo_type="dataset").private:
            raise RuntimeError(f"{repo_id} is public: contributions go to a private dataset only")

    def list_files(self, revision: str | None = None) -> list[str]:
        if not self.exists:
            return []
        return self.api.list_repo_files(self.repo_id, repo_type="dataset", revision=revision)

    def download(self, path: str, revision: str | None = None) -> bytes:
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory() as directory:
            local = hf_hub_download(self.repo_id, path, repo_type="dataset", revision=revision, cache_dir=directory)
            return Path(local).read_bytes()

    def commit(self, additions: dict[str, bytes], deletions: list[str], message: str) -> str:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        if not self.exists:
            self.api.create_repo(self.repo_id, repo_type="dataset", private=True)
            self.exists = True
        operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=body) for path, body in additions.items()]
        operations += [CommitOperationDelete(path_in_repo=path) for path in deletions]
        return self.api.create_commit(self.repo_id, operations=operations, commit_message=message, repo_type="dataset").oid

    def squash_history(self, message: str) -> None:
        """Replace the main branch's history with one commit of its current files; not reversible."""
        self.api.super_squash_history(self.repo_id, repo_type="dataset", commit_message=message)


@dataclass(frozen=True)
class Plan:
    consumed: list[ShardFile]  # contributions read, files deleted after the push
    annotation_only: list[ShardFile]  # annotations and removals read, files kept
    cutoff: datetime  # C1


@dataclass
class Receipt:
    batch_id: str | None
    plan_cutoff: str
    consumed: list[str]
    annotation_only: list[str]
    submissions: int = 0
    rows: int = 0
    removed_submissions: int = 0
    new_removals: int = 0
    orphan_annotations: int = 0
    late_annotations: int = 0
    torn_lines: int = 0
    rewritten: list[str] = field(default_factory=list)
    deleted_from_hub: list[str] = field(default_factory=list)
    hub_commit: str | None = None  # the verified commit; a squash after it replaces its id
    squashed: bool = False
    dry_run: bool = False


def plan_batch(shards: list[ShardFile], now: datetime, settings: Settings) -> Plan:
    cutoff = now - timedelta(seconds=settings.compaction_grace_seconds)
    consumed_by = cutoff - timedelta(seconds=settings.annotation_window_seconds + settings.shard_window_seconds)
    consumed = [shard for shard in shards if shard.window_end <= consumed_by]
    annotation_only = [shard for shard in shards if consumed_by < shard.window_end <= cutoff]
    return Plan(consumed=consumed, annotation_only=annotation_only, cutoff=cutoff)


def read_shard(path: Path) -> tuple[list[Record], int]:
    """A shard's records and whether its last line was torn (a write cut off before its newline); any other bad line
    is corruption and raises."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    torn = 0 if lines[-1] == "" else 1
    records: list[Record] = []
    for number, line in enumerate(lines[:-1], start=1):
        record = json.loads(line)
        if record.get("kind") not in ("contribution", "annotation", "removal"):
            raise ValueError(f"{path}:{number}: unknown record kind")
        records.append(record)
    return records, torn


def batch_id(consumed: list[ShardFile]) -> str:
    names = "\n".join(sorted(shard.path.as_posix() for shard in consumed))
    return f"{consumed[0].window_start:%Y%m%dT%H%M%SZ}_{hashlib.sha256(names.encode()).hexdigest()[:12]}"


def final_annotations(contributions: list[ContributionRecord], events: Iterable[AnnotationRecord],
                      window_ms: int) -> tuple[dict[str, tuple[str | None, int | None]], int, int]:
    """Each contribution's label after its events within the window (the latest wins), with the counts of events that
    match no contribution here (orphans) and of events past the window (late)."""
    received = {record["id"]: record["received_at"] for record in contributions}
    finals: dict[str, tuple[str | None, int | None]] = {
        record["id"]: (record["annotation"], record["received_at"] if record["annotation"] else None) for record in contributions}
    latest: dict[str, int] = {}
    orphans = late = 0
    for event in sorted(events, key=lambda event: (event["at"], event["id"])):
        if event["id"] not in received:
            orphans += 1
        elif event["at"] - received[event["id"]] > window_ms:
            late += 1
        elif event["at"] >= latest.get(event["id"], -1):
            latest[event["id"]] = event["at"]
            finals[event["id"]] = (event["annotation"], event["at"] if event["annotation"] else None)
    return finals, orphans, late


def collapse(contributions: list[ContributionRecord], finals: dict[str, tuple[str | None, int | None]]) -> list[dict[str, Any]]:
    """One row per (content hash, model version, terms version), counting its submissions."""
    groups: dict[tuple[str, str, str], list[ContributionRecord]] = {}
    for record in sorted(contributions, key=lambda record: (record["received_at"], record["id"])):
        groups.setdefault((record["content_hash"], record["model_version"], record["terms_version"]), []).append(record)
    rows = []
    for (digest, model_version, terms_version), submissions in groups.items():
        first = next((record for record in submissions if record["text"] is not None), None)
        assert first is not None, f"{digest}: every submission is a repeat, but a repeat follows its text in one shard"
        assert first["reading"] is not None
        if content_hash(first["text"]) != digest:
            raise ValueError(f"{first['id']}: the stored text does not match its content hash")
        labels = Counter(finals[record["id"]][0] for record in submissions if finals[record["id"]][0] is not None)
        ranked = labels.most_common()
        plurality = ranked[0][0] if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]) else None
        times = [finals[record["id"]][1] for record in submissions if finals[record["id"]][0] is not None]
        rows.append({
            "id": first["id"], "text": first["text"], "content_hash": digest,
            "reading": json.dumps(first["reading"], separators=(",", ":"), sort_keys=True, allow_nan=False),
            "model_version": model_version, "annotation": plurality,
            "annotated_at": max(time for time in times if time is not None) if times else None,
            "received_at": submissions[0]["received_at"], "terms_version": terms_version, "count": len(submissions),
            "annotation_counts": json.dumps({label: labels.get(label, 0) for label in ("ai", "human", "mixed")}, sort_keys=True),
        })
    return rows


def to_parquet(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue()


def utc_time(ms: int | None) -> datetime | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, tz=UTC)


def rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    ordered = sorted(rows, key=lambda row: (row["received_at"], row["id"]))
    columns = {name: [row[name] for row in ordered] for name in SCHEMA.names}
    columns["annotated_at"] = [utc_time(value) for value in columns["annotated_at"]]
    columns["received_at"] = [utc_time(value) for value in columns["received_at"]]
    return pa.table(columns, schema=SCHEMA)


def hub_removals(publisher: Publisher, files: list[str]) -> set[str]:
    return {digest for path in files if path.startswith("removals/") and path.endswith(".parquet")
            for digest in pq.read_table(io.BytesIO(publisher.download(path))).column("content_hash").to_pylist()}


def scrub_hub(publisher: Publisher, files: list[str], removed: set[str]) -> tuple[dict[str, bytes], list[str]]:
    """Every Hub data file holding a removed hash, rewritten without those rows, or listed for deletion when empty."""
    rewritten, deleted = {}, []
    for path in sorted(path for path in files if path.startswith("data/") and path.endswith(".parquet")):
        table = pq.read_table(io.BytesIO(publisher.download(path)))
        keep = [digest not in removed for digest in table.column("content_hash").to_pylist()]
        if all(keep):
            continue
        if any(keep):
            rewritten[path] = to_parquet(table.filter(pa.array(keep)))
        else:
            deleted.append(path)
    return rewritten, deleted


def compact(store: ShardStore, publisher: Publisher, settings: Settings, now: datetime, card: str | None = None,
            dry_run: bool = False, squash: bool = False) -> Receipt:
    """One compaction run. The history is squashed when the run rewrites or deletes a Hub data file, or when `squash`
    asks for it (to finish a run that stopped at the squash)."""
    store.reload()
    plan = plan_batch(list_shards(store.root), now, settings)
    receipt = Receipt(batch_id=batch_id(plan.consumed) if plan.consumed else None, plan_cutoff=plan.cutoff.isoformat(),
                      consumed=[shard.path.as_posix() for shard in plan.consumed],
                      annotation_only=[shard.path.as_posix() for shard in plan.annotation_only], dry_run=dry_run)
    contributions: list[ContributionRecord] = []
    annotations: list[AnnotationRecord] = []
    removals: dict[str, int] = {}
    for shard in [*plan.consumed, *plan.annotation_only]:
        records, torn = read_shard(store.root / shard.path)
        receipt.torn_lines += torn
        for record in records:
            if record["kind"] == "contribution" and shard in plan.consumed:
                contributions.append(record)
            elif record["kind"] == "annotation":
                annotations.append(record)
            elif record["kind"] == "removal":
                removals[record["content_hash"]] = min(record["at"], removals.get(record["content_hash"], record["at"]))

    files = publisher.list_files()
    published_removals = hub_removals(publisher, files)
    new_removals = {digest: at for digest, at in removals.items() if digest not in published_removals}
    removed = published_removals | set(new_removals)
    receipt.new_removals = len(new_removals)
    receipt.submissions = len(contributions)
    kept = [record for record in contributions if record["content_hash"] not in removed]
    receipt.removed_submissions = len(contributions) - len(kept)
    finals, receipt.orphan_annotations, receipt.late_annotations = final_annotations(
        kept, annotations, settings.annotation_window_seconds * 1000)
    table = rows_table(collapse(kept, finals))
    receipt.rows = table.num_rows

    additions: dict[str, bytes] = {}
    if receipt.batch_id is not None and table.num_rows:
        batch_path = f"data/{receipt.batch_id[:4]}/{receipt.batch_id}.parquet"
        if batch_path in files:  # a run that pushed and stopped before deleting its shards; removals rewrite it below
            remote = pq.read_table(io.BytesIO(publisher.download(batch_path)), columns=["id", "content_hash"])
            remote_ids = {row_id for row_id, digest in zip(remote.column("id").to_pylist(), remote.column("content_hash").to_pylist(),
                                                           strict=True) if digest not in removed}
            if remote_ids != set(table.column("id").to_pylist()):
                raise RuntimeError(f"{batch_path} is on the Hub with other rows than these shards give")
        else:
            additions[batch_path] = to_parquet(table)
    run_id = receipt.batch_id or f"{now:%Y%m%dT%H%M%SZ}_removals"
    if new_removals:
        removal_rows = sorted(new_removals.items())
        additions[f"removals/{run_id}.parquet"] = to_parquet(pa.table({
            "content_hash": [digest for digest, _ in removal_rows],
            "requested_at": [datetime.fromtimestamp(at / 1000, tz=UTC) for _, at in removal_rows]}, schema=REMOVALS_SCHEMA))
    rewritten, deleted = scrub_hub(publisher, files, set(new_removals)) if new_removals else ({}, [])
    additions.update(rewritten)
    receipt.rewritten, receipt.deleted_from_hub = sorted(rewritten), deleted
    if card is not None and "README.md" not in files:
        additions["README.md"] = card.encode("utf-8")

    if dry_run:
        return receipt
    if additions or deleted:
        commit = publisher.commit(additions, deleted, f"Litmus contributions: {run_id}")
        for path, body in additions.items():
            if publisher.download(path, revision=commit) != body:
                raise RuntimeError(f"{path} on the Hub at {commit} differs from what was pushed")
        still_there = set(deleted) & set(publisher.list_files(revision=commit))
        if still_there:
            raise RuntimeError(f"{sorted(still_there)} are still on the Hub at {commit}")
        receipt.hub_commit = commit
    if squash or rewritten or deleted:  # earlier revisions hold texts removed now: they go before the shards do
        publisher.squash_history(f"Litmus contributions: history squashed after {run_id}")
        receipt.squashed = True
    for shard in plan.consumed:
        (store.root / shard.path).unlink()
    if receipt.batch_id is not None or new_removals:
        receipts = store.root / "receipts"
        receipts.mkdir(exist_ok=True)
        (receipts / f"{run_id}.json").write_text(json.dumps(asdict(receipt), indent=2), encoding="utf-8")
    store.commit()
    return receipt
