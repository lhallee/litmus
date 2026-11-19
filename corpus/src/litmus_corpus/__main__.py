"""The Litmus corpus commands.

    python -m litmus_corpus build --out <folder> --workers N [--priority 1] [--limit-per-source K]
        [--new-token-budget 3e9] [--hub-repo <owner/name>] [--family-train] [--first-index <earlier build>/index]
        [--earlier-build <earlier build>]
    python -m litmus_corpus extend --earlier <corpus>/data --build <folder> --out <folder> --workers N
        [--trained-first-source NAME ...] [--group-source NAME ...]
    python -m litmus_corpus catalog check [--priority 3] [--out check.json]
"""

from __future__ import annotations

import argparse
import json
import sys

from dataclasses import replace
from pathlib import Path

from .catalog_check import check_catalog
from .config import BuildConfig
from .extend import extend
from .pipeline import Build, BuildOptions
from .storage import write_json


CORPUS_FOLDER = Path(__file__).resolve().parents[2]
CHECK_THREADS = 8


def add_catalogs(command: argparse.ArgumentParser) -> None:
    command.add_argument("--catalog", type=Path, default=CORPUS_FOLDER / "sources.csv")
    command.add_argument("--benchmarks", type=Path, default=CORPUS_FOLDER.parent / "benchmarks" / "catalog.csv")


def parser() -> argparse.ArgumentParser:
    commands = argparse.ArgumentParser(prog="litmus_corpus", description="Build the Litmus corpus from the source catalog.")
    subcommands = commands.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="run every stage, resuming from finished shards")
    add_catalogs(build)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--workers", type=int, required=True)
    build.add_argument("--priority", type=int, default=1, help="read sources of this priority or lower, and every benchmark")
    build.add_argument("--limit-per-source", type=int, default=None, help="rows read from each source at most, for a small trial run")
    build.add_argument("--new-token-budget", type=float, default=BuildConfig.new_token_budget, help="tokens of new training rows at most")
    build.add_argument("--no-first-corpus", action="store_true", help="leave the first corpus out (tests and trial runs only)")
    build.add_argument("--hub-repo", default=None, help="a private dataset repository to upload the corpus to; a public one is refused")
    build.add_argument(
        "--skip-benchmark", action="append", default=[], metavar="NAME",
        help="a benchmark the build cannot read (no text column, gated); named on purpose, recorded in the report",
    )
    build.add_argument("--family-train", action="store_true",
                       help="train on benchmark families' train and validation lines, their test lines read as benchmarks")
    build.add_argument("--first-index", type=Path, default=None,
                       help="an earlier build's index/ folder, to link raw re-reads to the first corpus when it is not read")
    build.add_argument("--earlier-build", type=Path, default=None,
                       help="an earlier build's folder: the training sources it read are left out, as its corpus already samples them")
    extending = subcommands.add_parser("extend", help="an earlier corpus's rows with their ids, then a later build's new rows")
    extending.add_argument("--earlier", type=Path, required=True, help="the earlier corpus's data/ folder")
    extending.add_argument("--build", type=Path, required=True, help="the later build's folder, built without the first corpus")
    extending.add_argument("--out", type=Path, required=True)
    extending.add_argument("--workers", type=int, required=True)
    extending.add_argument("--trained-first-source", action="append", default=[], metavar="NAME",
                           help="a first-corpus source held out in the earlier corpus and trained on now")
    extending.add_argument("--group-source", action="append", default=[], metavar="NAME",
                           help="a source read by group_column; an earlier row standing for one of its dropped rows takes its group doc")
    catalog = subcommands.add_parser("catalog", help="work on the source catalog")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    check = catalog_commands.add_parser("check", help="verify revisions, columns and row counts against the Hub")
    add_catalogs(check)
    check.add_argument("--priority", type=int, default=3)
    check.add_argument("--out", type=Path, default=Path("catalog_check.json"))
    check.add_argument("--threads", type=int, default=CHECK_THREADS)

    return commands


def run_build(parsed: argparse.Namespace) -> None:
    if parsed.workers < 1:
        raise SystemExit("--workers must be 1 or more")
    if parsed.new_token_budget <= 0:
        raise SystemExit("--new-token-budget must be positive")
    options = BuildOptions(
        catalog=str(parsed.catalog.resolve()),
        benchmarks=str(parsed.benchmarks.resolve()),
        priority=parsed.priority,
        limit_per_source=parsed.limit_per_source,
        skip_benchmarks=tuple(sorted(parsed.skip_benchmark)),
        first_corpus=not parsed.no_first_corpus,
        hub_repo=parsed.hub_repo,
        family_train=parsed.family_train,
        first_index=None if parsed.first_index is None else str(parsed.first_index.resolve()),
        earlier_build=None if parsed.earlier_build is None else str(parsed.earlier_build.resolve()),
    )
    config = replace(BuildConfig(), new_token_budget=parsed.new_token_budget)
    report = Build(options, parsed.out, parsed.workers, config).run()
    print(json.dumps({"rows": report["rows"], "written": report["written"], "timings": report["timings"]}, indent=1))


def run_check(parsed: argparse.Namespace) -> None:
    payload = check_catalog(parsed.catalog, parsed.benchmarks, parsed.priority, parsed.threads)
    write_json(payload, parsed.out)
    for source in payload["results"]:
        for problem in source["problems"]:
            print(f"{source['name']}: {problem}")
    print(f"{payload['with_problems']} of {payload['sources']} sources with problems; {len(payload['catalog_problems'])} catalog lines unusable; written to {parsed.out}")


def main(arguments: list[str]) -> None:
    parsed = parser().parse_args(arguments)
    if parsed.command == "build":
        run_build(parsed)
    elif parsed.command == "extend":
        report = extend(parsed.earlier, parsed.build, parsed.out, parsed.workers, tuple(parsed.trained_first_source),
                        tuple(parsed.group_source))
        print(json.dumps({key: report[key] for key in ("earlier_rows", "new_rows", "rows")}, indent=1))
    else:
        assert parsed.command == "catalog" and parsed.catalog_command == "check"
        run_check(parsed)


if __name__ == "__main__":
    main(sys.argv[1:])
