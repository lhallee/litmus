"""The build: every stage in order over a process pool, each resuming from the shards it already finished.

    out/read/<source>/          stage 1, raw records (a plan, and a report a part; `failed.json` if it failed)
    out/standardize/<source>/   stage 2
    out/index/                  the first corpus's fingerprints, for linking (`linking.py`)
    out/strata/<source>/        stage 3
    out/links/<source>/         stage 3, benchmark texts linked to first-corpus documents
    out/passages/<source>/      stage 4
    out/checks_keys/<source>/   stage 5, per-row checks and keys
    out/checks/<source>/        stage 5, after the corpus-wide copies are dropped
    out/holdout/<source>/       stage 7, run before 6 so the reported balance is the shipped one
    out/balance/<source>/       stage 6
    out/corpus/                 stage 8: data/, sources.csv, report.json, samples/

Each stage folder holds a `report.json` of counts in and out by reason, and `done.json` once finished. A
shard is done when its report exists, so a stopped build reruns only unfinished shards. An output folder
holds one build: its options are recorded in `build.json`, and a resume with other options fails.
"""

from __future__ import annotations

import faulthandler
import json
import multiprocessing
import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm

from .adapters import GROUP_SEPARATOR
from .balance import BALANCE_REASONS, Rows, caps_and_balance
from .catalog import BENCHMARK_GENERATOR, FIRST_CORPUS, MULTI_LANGUAGE, Catalog, SourceSpec, load_catalog
from .checks import GLOBAL_REASONS, filter_shard, global_duplicate_codes, keyed_shard
from .config import BuildConfig
from .generators import UNKNOWN, maker
from .holdout import HOLDOUT_REASONS, NO_MARK, holdout_decision
from .language import ensure_language_model, file_sha256
from .linking import index_shard, save_index
from .passages import passages_shard
from .publish import publish
from .read import entry_tasks, part_report_path, plan_source, read_limits, read_part
from .standardize import GroupShares, standardize_shard
from .storage import JsonObject, Report, merged_reports, read_json, report_path, shard_paths, stage_report, write_json
from .strata import LinkPlan, strata_shard
from .tokenization import ensure_tokenizer
from .write import corpus_shard, write_samples, write_sources


@dataclass(frozen=True)
class BuildOptions:
    catalog: str
    benchmarks: str
    priority: int
    limit_per_source: int | None
    skip_benchmarks: tuple[str, ...] = ()  # benchmarks the build may not read, each named on purpose
    first_corpus: bool = True
    hub_repo: str | None = None  # a private dataset repository to upload the corpus to
    family_train: bool = False  # train on benchmark families' train and validation lines, their tests held out
    first_index: str | None = None  # an earlier build's `index/`, to link raw re-reads when the first corpus is not read
    earlier_build: str | None = None  # an earlier build's folder: the training sources it read are not read again


@dataclass(frozen=True)
class Decision:
    """A corpus-wide stage's drop code for every row, what it records, and benchmark names it gives rows."""

    codes: np.ndarray  # (n,) uint8
    record: JsonObject
    marks: np.ndarray | None = None  # (n,) int32, an index into `names`, 0 for none
    names: tuple[str, ...] = ()


STUCK_SECONDS = 300


def worker_setup() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def watched(function: Callable[..., object], *task: object) -> object:
    """`function(*task)`; a call still running after `STUCK_SECONDS` prints its stack to the log, and again each
    interval after, so a stage that stops advancing names the line it is on."""
    faulthandler.dump_traceback_later(STUCK_SECONDS, repeat=True)
    try:
        return function(*task)
    finally:
        faulthandler.cancel_dump_traceback_later()


Task = tuple[object, ...]  # the arguments of one call of a stage's function


def run_parallel(function: Callable[..., object], tasks: list[Task], workers: int, description: str) -> list[object]:
    """`function(*task)` for every task, in `workers` processes (inline for one), returned in task order; the first
    failure stops the build."""
    if not tasks:
        return []
    if workers == 1:
        return [function(*task) for task in tqdm(tasks, desc=description, mininterval=10)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=context, initializer=worker_setup) as pool:
        futures = [pool.submit(watched, function, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=description, mininterval=10):
            future.result()  # raises the first failure as it comes

    return [future.result() for future in futures]


def earlier_sources(build: Path) -> list[str]:
    """The sources an earlier build read whole, which its corpus already samples: every source of its read report but
    the ones that failed, sorted."""
    report = read_json(build / "read" / "report.json")

    return sorted(set(report["sources"]) - set(report["failed"]))


def benchmark_column(table: pa.Table) -> np.ndarray:
    return (table.column("benchmark").to_numpy(zero_copy_only=False) != "").astype(bool)  # (n,)


def first_column(table: pa.Table) -> np.ndarray:
    return table.column("first_corpus_id").is_valid().to_numpy(zero_copy_only=False)  # (n,)


class Build:
    """One corpus build in one output folder."""

    def __init__(self, options: BuildOptions, out: Path, workers: int, config: BuildConfig) -> None:
        self.options = options
        self.out = out
        self.workers = workers
        self.config = config
        self.catalog: Catalog | None = None
        self.specs: list[SourceSpec] = []
        self.failed: dict[str, JsonObject] = {}
        self.timings: dict[str, dict[str, float]] = {}
        self.read_earlier: list[str] = []

    def stage(self, name: str) -> Path:
        return self.out / name

    @property
    def sources(self) -> list[str]:
        """The sources the stages after reading carry: every source that did not fail."""
        return [spec.name for spec in self.specs if spec.name not in self.failed]

    def record_options(self) -> None:
        path = self.out / "build.json"
        catalogs = {name: file_sha256(Path(location)) for name, location in (("catalog", self.options.catalog), ("benchmarks", self.options.benchmarks))}
        recorded = json.loads(json.dumps({"options": asdict(self.options), "config": asdict(self.config), "sha256": catalogs}))
        if path.exists() and read_json(path) != recorded:
            raise RuntimeError(f"{self.out} holds a build with other options; use a new folder")
        write_json(recorded, path)

    def run(self) -> JsonObject:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.out.mkdir(parents=True, exist_ok=True)
        self.catalog = load_catalog(
            Path(self.options.catalog), Path(self.options.benchmarks), self.options.priority, self.options.skip_benchmarks, self.options.first_corpus,
            self.options.family_train,
        )
        self.specs = self.catalog.specs
        if self.options.earlier_build is not None:
            self.read_earlier = earlier_sources(Path(self.options.earlier_build))
            self.specs = [spec for spec in self.specs if spec.held_out or spec.verbatim or spec.name not in self.read_earlier]
        self.record_options()
        tokenizer_path = str(ensure_tokenizer())
        needs_language = any(spec.language == MULTI_LANGUAGE for spec in self.specs)
        language_path = str(ensure_language_model(self.out / "models")) if needs_language else None
        self.timed("read", self.read)
        self.failed = self.failed_sources()
        if FIRST_CORPUS in self.failed:
            raise RuntimeError(f"the first corpus could not be read, and the corpus is built on it: {self.failed[FIRST_CORPUS]['failed']}")
        unread = {spec.name: self.failed[spec.name]["failed"] for spec in self.specs if spec.held_out and spec.name in self.failed}
        if unread:
            raise RuntimeError(f"benchmarks could not be read, so nothing would hold their texts out of training; fix them or skip each by name: {unread}")
        shares = {name: GroupShares.from_counts(dict(report.counts.get("group", {})), self.config) for name, report in self.read_reports().items()}
        self.timed("standardize", lambda: self.map_stage("read", "standardize", standardize_shard, lambda path: (shares[path.parent.name],)))
        self.timed("index", self.index)
        plan = self.link_plan()
        links = self.stage("links")
        self.timed(
            "strata",
            lambda: self.map_stage("standardize", "strata", strata_shard, lambda path: (tokenizer_path, language_path, plan, links / path.parent.name / path.name)),
        )
        self.timed("passages", lambda: self.map_stage("strata", "passages", passages_shard, lambda path: (tokenizer_path, self.config)))
        self.timed("checks_keys", lambda: self.map_stage("passages", "checks_keys", keyed_shard, lambda path: ()))
        self.timed("checks", lambda: self.global_stage("checks_keys", "checks", self.duplicate_codes, GLOBAL_REASONS))
        self.timed("holdout", lambda: self.global_stage("checks", "holdout", self.benchmark_codes, HOLDOUT_REASONS))
        self.timed("balance", lambda: self.global_stage("holdout", "balance", self.balance_codes, BALANCE_REASONS))
        self.timed("write", self.write)
        if self.options.hub_repo is not None:
            revision = publish(self.stage("corpus"), self.options.hub_repo)
            write_json({"repo": self.options.hub_repo, "revision": revision}, self.stage("corpus") / "hub.json")

        return read_json(self.stage("corpus") / "report.json")

    def timed(self, name: str, step: Callable[[], object]) -> None:
        done = self.stage(name) / "done.json"
        if done.exists():
            self.timings[name] = read_json(done)
            return
        start = time.perf_counter()
        step()
        seconds = time.perf_counter() - start
        report = read_json(self.stage(name) / "report.json") if (self.stage(name) / "report.json").exists() else {}
        rows_in = report.get("total", {}).get("rows_in", 0)
        self.timings[name] = {"seconds": round(seconds, 2), "rows_in": rows_in, "rows_per_second": round(rows_in / max(seconds, 1e-9), 1)}
        write_json(self.timings[name], done)

    def read(self) -> None:
        stage = self.stage("read")
        limits = read_limits(self.specs, self.config, self.options.limit_per_source)
        unplanned = [spec for spec in self.specs if not (stage / spec.name / "plan.json").exists()]
        planned = run_parallel(plan_source, [(spec, limits[spec.name], self.config) for spec in unplanned], self.workers, "plan")
        for spec, plan in zip(unplanned, planned, strict=True):
            write_json(plan, stage / spec.name / "plan.json")
        tasks: list[Task] = []
        for spec in self.specs:
            plan = read_json(stage / spec.name / "plan.json")
            if "failed" in plan or (stage / spec.name / "failed.json").exists():
                continue
            tasks.extend((task, self.config) for task in entry_tasks(spec, plan, stage / spec.name) if not part_report_path(task).exists())
        run_parallel(read_part, tasks, self.workers, "read")
        payload = stage_report(self.read_reports())
        payload["limits"] = limits
        payload["failed"] = self.failed_sources()
        write_json(payload, stage / "report.json")

    def failed_sources(self) -> dict[str, JsonObject]:
        """Sources that could not be opened or whose first rows did not fit, to why."""
        stage = self.stage("read")
        failed: dict[str, JsonObject] = {}
        for spec in self.specs:
            plan = read_json(stage / spec.name / "plan.json")
            if "failed" in plan:
                failed[spec.name] = plan
            elif (stage / spec.name / "failed.json").exists():
                failed[spec.name] = read_json(stage / spec.name / "failed.json")

        return failed

    def read_reports(self) -> dict[str, Report]:
        stage = self.stage("read")
        paths = [path for name in self.sources for path in sorted((stage / name).glob("s[0-9]*-p[0-9][0-9][0-9].json"))]

        return merged_reports(paths)

    def map_stage(self, source_stage: str, target_stage: str, function: Callable[..., JsonObject], extra: Callable[[Path], Task]) -> None:
        """`function(input shard, output shard, *extra(input shard))` for every unfinished shard, largest first."""
        inputs = shard_paths(self.stage(source_stage), self.sources)
        target = self.stage(target_stage)
        outputs = [target / path.parent.name / path.name for path in inputs]
        tasks = [(path, output, *extra(path)) for path, output in zip(inputs, outputs, strict=True) if not report_path(output).exists()]
        tasks.sort(key=lambda task: -task[0].stat().st_size)
        run_parallel(function, tasks, self.workers, target_stage)
        write_json(stage_report(merged_reports([report_path(output) for output in outputs])), target / "report.json")

    def index(self) -> None:
        """The first corpus's fingerprint index, from its standardized shards."""
        target = self.stage("index")
        inputs = shard_paths(self.stage("standardize"), [FIRST_CORPUS] if FIRST_CORPUS in self.sources else [])
        pieces = [target / "pieces" / path.name for path in inputs]
        run_parallel(index_shard, [(path, piece) for path, piece in zip(inputs, pieces, strict=True) if not piece.exists()], self.workers, "index")
        size = save_index(pieces, target)
        write_json({"total": {"rows_in": size}, "fingerprints": size}, target / "report.json")

    def link_plan(self) -> LinkPlan:
        reread = frozenset(spec.name for spec in self.specs if spec.in_first_corpus and spec.reread and spec.name in self.sources)
        if self.options.first_index is not None:
            assert FIRST_CORPUS not in self.sources, "an earlier build's index stands in for the first corpus, never beside it"
            assert (Path(self.options.first_index) / "fingerprints.npy").exists(), f"no index in {self.options.first_index}"
            return LinkPlan(self.options.first_index, reread)
        has_index = FIRST_CORPUS in self.sources and (self.stage("index") / "fingerprints.npy").exists()

        return LinkPlan(str(self.stage("index")) if has_index else None, reread)

    def global_stage(self, source_stage: str, target_stage: str, decide: Callable[[list[Path]], Decision], reasons: tuple[str, ...]) -> None:
        """A drop code for every row from the whole corpus at once, then each shard filtered by its codes."""
        inputs = shard_paths(self.stage(source_stage), self.sources)
        target = self.stage(target_stage)
        codes_path = target / "codes.npy"
        if not codes_path.exists():
            target.mkdir(parents=True, exist_ok=True)
            decision = decide(inputs)
            marks = decision.marks if decision.marks is not None else np.zeros(0, dtype=np.int32)
            np.save(target / "marks.npy", marks)
            write_json({**decision.record, "names": list(decision.names)}, target / "decision.json")
            np.save(target / "codes.tmp.npy", decision.codes)
            os.replace(target / "codes.tmp.npy", codes_path)
        codes = np.load(codes_path)  # (n,)
        marks = np.load(target / "marks.npy")  # (n,) or (0,)
        names = tuple(read_json(target / "decision.json")["names"])
        counts = [pq.ParquetFile(path).metadata.num_rows for path in inputs]
        offsets = np.r_[0, np.cumsum(counts)]  # (s + 1,)
        assert offsets[-1] == len(codes), f"{target_stage}: {len(codes)} codes for {offsets[-1]} rows"
        outputs = [target / path.parent.name / path.name for path in inputs]
        tasks = [
            (path, output, codes[start:end], reasons, marks[start:end] if len(marks) else None, names)
            for path, output, start, end in zip(inputs, outputs, offsets[:-1], offsets[1:], strict=True)
            if not report_path(output).exists()
        ]
        run_parallel(filter_shard, tasks, self.workers, target_stage)
        payload = stage_report(merged_reports([report_path(output) for output in outputs]))
        payload["decision"] = read_json(target / "decision.json")
        write_json(payload, target / "report.json")

    @staticmethod
    def columns(inputs: list[Path], names: list[str]) -> pa.Table:
        assert inputs, "no shards reached a corpus-wide stage"

        return pa.concat_tables([pq.read_table(path, columns=names) for path in inputs])

    def duplicate_codes(self, inputs: list[Path]) -> Decision:
        table = self.columns(inputs, ["text_key", "opening_key", "label", "doc", "benchmark", "first_corpus_id"])
        codes = global_duplicate_codes(
            table.column("text_key").to_numpy(),
            table.column("opening_key").to_numpy(),
            table.column("label").to_numpy(),
            table.column("doc").to_numpy(),
            benchmark_column(table),
            first_column(table),
        )  # (n,)

        return Decision(codes, {"rows": int(len(codes)), "dropped": int((codes > 0).sum())})

    def benchmark_codes(self, inputs: list[Path]) -> Decision:
        table = self.columns(inputs, ["doc", "text_key", "opening_key", "benchmark", "label", "first_corpus_id"])
        link_paths = [path for path in shard_paths(self.stage("links"), self.sources)]
        links = pa.concat_tables([pq.read_table(path) for path in link_paths]) if link_paths else pa.table({"first_doc": pa.array([], pa.uint64()), "benchmark": pa.array([], pa.string())})
        benchmarks = table.column("benchmark").to_pylist()
        linked_names = links.column("benchmark").to_pylist()
        names = ("", *sorted({name for name in [*benchmarks, *linked_names] if name}))
        number = {name: index for index, name in enumerate(names)}
        codes, marks, ways = holdout_decision(
            table.column("doc").to_numpy(),
            table.column("text_key").to_numpy(),
            table.column("opening_key").to_numpy(),
            np.array([number[name] for name in benchmarks], dtype=np.int32),
            table.column("label").to_numpy(),
            first_column(table),
            links.column("first_doc").to_numpy(),
            np.array([number[name] for name in linked_names], dtype=np.int32),
        )  # (n,), (n,)
        marked = {way: {names[int(key)]: int(count) for key, count in zip(*np.unique(ids, return_counts=True), strict=True)} for way, ids in ways.items()}
        record = {
            "rows": int(len(codes)),
            "benchmark_rows": int(benchmark_column(table).sum()),
            "dropped": int((codes > 0).sum()),
            "linked_benchmark_texts": len(linked_names),
            "first_corpus_benchmark_rows": marked,  # by how each was found: source, link, doc, text, opening
            "first_corpus_marked": int((marks != NO_MARK).sum()),
        }

        return Decision(codes, record, marks, names)

    def balance_codes(self, inputs: list[Path]) -> Decision:
        names = ["label", "source", "generator", "domain", "mix", "format", "language", "tokens", "benchmark", "first_corpus_id", "long_eval", "text_key"]
        table = self.columns(inputs, names)
        rows = Rows(
            label=table.column("label").to_numpy(),
            source=table.column("source").combine_chunks(),
            generator=table.column("generator").combine_chunks(),
            domain=table.column("domain").combine_chunks(),
            mix=table.column("mix").combine_chunks(),
            format=table.column("format").combine_chunks(),
            language=table.column("language").combine_chunks(),
            tokens=table.column("tokens").to_numpy(),
            benchmark=benchmark_column(table),
            first=first_column(table),
            long_eval=table.column("long_eval").to_numpy(zero_copy_only=False),
            text_key=table.column("text_key").to_numpy(),
        )
        codes, lines = caps_and_balance(rows, self.config)
        flagged = [cell for cell in lines["cells"] if cell["status"] != "ok"]

        return Decision(codes, {**lines, "flagged_cells": flagged})

    def write(self) -> None:
        corpus = self.stage("corpus")
        inputs = shard_paths(self.stage("balance"), self.sources)
        counts = [pq.ParquetFile(path).metadata.num_rows for path in inputs]
        first_ids = dict(zip(inputs, np.r_[0, np.cumsum(counts)][:-1].tolist(), strict=True))
        nonempty = [path for path, count in zip(inputs, counts, strict=True) if count]
        outputs = [corpus / "data" / f"shard-{index:05d}.parquet" for index in range(len(nonempty))]
        tasks = [(path, output, first_ids[path]) for path, output in zip(nonempty, outputs, strict=True) if not output.exists()]
        run_parallel(corpus_shard, tasks, self.workers, "write")
        read_reports = self.read_reports()
        write_sources(outputs, self.specs, {name: report.rows_in for name, report in read_reports.items()}, corpus / "sources.csv")
        samples = write_samples(nonempty, first_ids, corpus / "samples", self.config.samples_per_cell) if nonempty else 0
        stages = {name: read_json(self.stage(name) / "report.json") for name in ("read", "standardize", "index", "strata", "passages", "checks_keys", "checks", "holdout", "balance")}
        generators = {spec.generator for spec in self.specs if spec.generator}
        for report in read_reports.values():
            generators.update(key.partition(GROUP_SEPARATOR)[2] for key in report.counts.get("group", {}))
        assert self.catalog is not None
        report = {
            "build": read_json(self.out / "build.json"),
            "rows": int(sum(counts)),
            "shards": len(outputs),
            "sample_files": samples,
            "written": self.written_counts(outputs),
            "first_corpus": {
                "standardize": stages["standardize"]["sources"].get(FIRST_CORPUS, {}),
                "links": stages["strata"]["total"].get("counts", {}).get("first_corpus_link", {}),
                "benchmark_rows": stages["holdout"]["decision"]["first_corpus_benchmark_rows"],
                "dropped_by_checks": {name: stages[name]["total"].get("counts", {}).get("first_corpus_dropped", {}) for name in ("checks_keys", "checks")},
            },
            "long_documents": stages["passages"]["total"].get("counts", {}).get("long_documents", {}),
            "token_budget": stages["balance"]["decision"]["token_budget"],
            "long_eval": stages["balance"]["decision"]["long_eval"],
            "unknown_generators": sorted(name for name in generators if name not in ("", "human", BENCHMARK_GENERATOR) and maker(name) == UNKNOWN),
            "catalog": {
                "problems": self.catalog.problems,
                "excluded": self.catalog.excluded,
                "held_out_first_sources": self.catalog.held_out_first_sources,
                "skipped_benchmarks": self.catalog.skipped_benchmarks,
                "disagreements": self.catalog.disagreements,
                "families": self.catalog.families,
                "read_by_the_earlier_build": [name for name in self.read_earlier if name in {spec.name for spec in self.catalog.specs}],
            },
            "failed_sources": self.failed,
            "timings": self.timings,
            "stages": stages,
        }
        write_json(report, corpus / "report.json")

    @staticmethod
    def written_counts(outputs: list[Path]) -> JsonObject:
        """Corpus rows and tokens by origin, and how many are first-corpus, benchmark, `long_eval` and unchanged rows."""
        totals: dict[str, int] = {}
        for path in outputs:
            table = pq.read_table(path, columns=["label", "tokens", "first_corpus_id", "benchmark", "long_eval", "text_unchanged"])
            labels = table.column("label").to_numpy()  # (n,)
            tokens = table.column("tokens").to_numpy().astype(np.int64)  # (n,)
            flags = {
                "first_corpus": first_column(table),
                "benchmark": benchmark_column(table),
                "long_eval": table.column("long_eval").to_numpy(zero_copy_only=False),
                "text_unchanged": table.column("text_unchanged").to_numpy(zero_copy_only=False),
            }
            new_training = ~flags["first_corpus"] & ~flags["benchmark"] & ~flags["long_eval"]  # (n,)
            for origin, label in (("human", 0), ("ai", 1), ("mixed", 2)):
                chosen = labels == label  # (n,)
                updates = {f"{origin}_rows": int(chosen.sum()), f"{origin}_new_training_tokens": int(tokens[chosen & new_training].sum())}
                updates |= {f"{origin}_{flag}": int((chosen & values).sum()) for flag, values in flags.items()}
                for key, value in updates.items():
                    totals[key] = totals.get(key, 0) + value

        return totals
