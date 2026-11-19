"""Command line: score catalog benchmarks, compute the metric table, measure single-core cost, draw the figures.

    python -m litmus_benchmarks score --benchmarks mage beemo --detectors Binoculars "Desklib v1.01" --max-rows 2000 --out scores.csv
    python -m litmus_benchmarks metrics --scores scores.csv --out metrics.csv
    python -m litmus_benchmarks cost --benchmark mage --detector e5-small-lora --texts 200 --out costs.csv
    python -m litmus_benchmarks gpu-use --documents /work/sets/text/human /work/sets/text/ai --out /work/gpu_use/gpu_use.json
    python -m litmus_benchmarks architectures --out architectures.csv
    python -m litmus_benchmarks results --scores BASELINES LITMUS --evaluation evaluation.json --costs costs.csv \\
        --model-version VERSION --work WORK --site ../site/assets --docs ../docs/images

Rerun detectors are open, so the figures read every detector not named Litmus or ai-vs-human-checker as open.
"""

from __future__ import annotations

import argparse
import pandas as pd

from dataclasses import asdict
from pathlib import Path

from . import report
from .catalog import CatalogEntry, find_entry, load_benchmark, read_catalog, sample_texts
from .detectors import build_scorer, measure_cpu_cost
from .estimate import architectures
from .evaluate import evaluate_scores, score_texts
from .gpu_use import DEFAULT_DETECTORS, PILOT_DOCUMENTS, measure, work


HERE = Path(__file__).resolve().parents[2]
COST_POOL = 200
COST_WORDS = (100, 300)  # short enough that no detector truncates (512 tokens), so every detector reads every word


def cost_texts(entry: CatalogEntry, *, seed: int) -> list[str]:
    """The fixed cost set: `COST_POOL` texts of the benchmark with `COST_WORDS` words, sampled with `seed`; a smaller
    count takes the first of these, so a slow detector's sample is a subset of the others'."""
    texts = load_benchmark(entry)
    texts = texts[texts["words"].between(*COST_WORDS)]
    return sample_texts(texts, COST_POOL, seed)["text"].tolist()


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="litmus_benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="score catalog benchmarks with named detectors")
    score.add_argument("--catalog", type=Path, default=HERE / "catalog.csv")
    score.add_argument("--benchmarks", nargs="+", required=True)
    score.add_argument("--detectors", nargs="+", required=True)
    score.add_argument("--max-rows", type=int, default=2000)
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--device", default="cuda")
    score.add_argument("--out", type=Path, required=True)
    metrics = commands.add_parser("metrics", help="tidy metric table with bootstrap intervals from per-text scores")
    metrics.add_argument("--scores", type=Path, required=True)
    metrics.add_argument("--replicates", type=int, default=1000)
    metrics.add_argument("--strata-replicates", type=int, default=200)
    metrics.add_argument("--seed", type=int, default=0)
    metrics.add_argument("--out", type=Path, required=True)
    cost = commands.add_parser("cost", help="single-thread CPU cost of one detector; run once per detector, fresh process")
    cost.add_argument("--catalog", type=Path, default=HERE / "catalog.csv")
    cost.add_argument("--benchmark", required=True)
    cost.add_argument("--detector", required=True)
    cost.add_argument("--texts", type=int, default=COST_POOL, help=f"the first N of the fixed {COST_POOL}-text set")
    cost.add_argument("--seed", type=int, default=0)
    cost.add_argument("--out", type=Path, required=True)
    gpu = commands.add_parser("gpu-use", help="GPU utilization and memory of named detectors reading long documents, one JSON")
    gpu.add_argument("--documents", type=Path, nargs="+", required=True, help="folders of .txt documents, read in this order")
    gpu.add_argument("--detectors", nargs="+", default=list(DEFAULT_DETECTORS))
    gpu.add_argument("--out", type=Path, required=True)
    gpu.add_argument("--interval", type=float, default=0.25, help="seconds between nvidia-smi polls (a poll itself takes some tens of ms)")
    gpu.add_argument("--idle-seconds", type=float, default=10.0, help="the idle baseline, taken first with nothing of ours running")
    gpu.add_argument("--max-documents", type=int, help="read only the first N documents")
    gpu.add_argument("--pilot", action="store_true", help=f"{PILOT_DOCUMENTS} documents, a 2 s baseline; the JSON says pilot")
    gpu.add_argument("--gpu", type=int, default=0, help="the GPU index to poll and run on")
    gpu.add_argument("--description", default="", help="what the documents are, for the JSON (the draw's chosen.json is read as well when found)")
    gpu.add_argument("--load-timeout", type=float, default=3600.0, help="seconds a detector may take to load")
    gpu.add_argument("--read-timeout", type=float, default=7200.0, help="seconds a detector may take to read the documents")
    worker = commands.add_parser("gpu-use-worker", help="one detector's process for gpu-use; not run by hand")
    worker.add_argument("--detector", required=True)
    worker.add_argument("--documents", type=Path, nargs="+", required=True)
    worker.add_argument("--max-documents", type=int)
    worker.add_argument("--batch-size", type=int, required=True)
    worker.add_argument("--device", default="cuda:0")
    report.add_arguments(commands.add_parser("results", help="results.json, the Markdown tables and every figure, from score files"), HERE)
    architectures = commands.add_parser("architectures", help="architecture and size of measured and leaderboard models")
    architectures.add_argument("--out", type=Path, default=HERE / "architectures.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.command == "score":
        entries = read_catalog(args.catalog)
        texts = pd.concat([load_benchmark(find_entry(entries, name), max_rows=args.max_rows, seed=args.seed)
                           for name in args.benchmarks], ignore_index=True)
        scorers = {name: build_scorer(name, args.device) for name in args.detectors}
        score_texts(texts, scorers, batch_size=args.batch_size).to_csv(args.out, index=False)
    elif args.command == "metrics":
        table = evaluate_scores(pd.read_csv(args.scores, keep_default_na=False), replicates=args.replicates,
                                strata_replicates=args.strata_replicates, seed=args.seed)
        table.to_csv(args.out, index=False)
    elif args.command == "cost":
        texts = cost_texts(find_entry(read_catalog(args.catalog), args.benchmark), seed=args.seed)[:args.texts]
        measured = pd.DataFrame([asdict(measure_cpu_cost(args.detector, build_scorer(args.detector, "cpu"), texts))])
        measured.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
    elif args.command == "gpu-use":
        limit = args.max_documents if args.max_documents is not None else (PILOT_DOCUMENTS if args.pilot else None)
        measure(folders=args.documents, detectors=args.detectors, out=args.out, interval=args.interval,
                idle_seconds=2.0 if args.pilot else args.idle_seconds, limit=limit, description=args.description, gpu=args.gpu,
                device=f"cuda:{args.gpu}", load_timeout=args.load_timeout, read_timeout=args.read_timeout, pilot=args.pilot)
    elif args.command == "gpu-use-worker":
        work(args.detector, args.documents, args.max_documents, args.batch_size, args.device)
    elif args.command == "architectures":
        architectures().to_csv(args.out, index=False, lineterminator="\n")
    else:
        for path in report.build(args):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
