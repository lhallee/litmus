"""Stages `score`, `cost`, and `results`: Litmus on every benchmark, its single-core cost, and the comparison.

`score` reads each benchmark of `release/baselines.toml` with the release's engine (the `litmus_engine` wheel built
from this checkout, reading the release folder: the program as a user runs it) and writes `scores_<benchmark>__litmus.csv`.
It needs no GPU. The texts are the rows the stored baseline scores were made from: a benchmark is sampled with the same
seed and size, and the stage fails unless Litmus read exactly the rows the baselines read. `cost` measures the engine on
one core. `results` runs the benchmarks package's `results` command, which writes `site/assets/results.json`, the
figures, and `results/results.md`. The baselines were scored once, on the machine of their time; `release/RELEASING.md`
says where their scores are kept.
"""

from __future__ import annotations

import csv
import json
import os
import tomllib

from pathlib import Path

from . import receipts
from .commands import StageError, run
from .hub import baselines_path, read_baselines
from .stages import Context, Stage


DETECTOR_IN_PACKAGE = "Litmus engine"
DETECTOR_IN_RESULTS = "Litmus"
COST_BENCHMARK = "mage"
COST_TEXTS = 200
RESULTS_MADE = "results_made.json"  # results.json as the `results` stage wrote it, without the release block


def benchmarks_environment(context: Context, **extra: str) -> dict[str, str]:
    """The environment the benchmarks package runs in: its source and the vendored `foundry` library on the path."""
    paths = [str(context.repository / "benchmarks" / "src")]
    vendored = context.repository / "training"
    if vendored.is_dir():
        paths.append(str(vendored))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([*paths, *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])])
    environment["PYTHONUTF8"] = "1"
    environment.update(extra)
    return environment


def require_engine(context: Context, environment: dict[str, str]) -> None:
    try:
        run([context.python, "-c", "import litmus_engine"], context.work, environment, capture=True)
    except StageError as error:
        raise StageError("the Python module litmus_engine is not installed: build the wheel from this checkout (`bash engine/litmus-py/build_wheel.sh . <folder>` "
                         "for Linux x86_64, or `maturin develop --release` in engine/litmus-py elsewhere) and `pip install` it\n" + str(error)) from error


def read_rows(path: Path) -> list[tuple[str, str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [(row["row"], row["origin"], row["words"]) for row in csv.DictReader(stream)]


def relabel(source: Path, target: Path) -> None:
    """Write `source` with its detector named as the results name Litmus."""
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [{**row, "detector": DETECTOR_IN_RESULTS} for row in reader]
    assert fields is not None
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def same_rows(context: Context, name: str, litmus: Path) -> None:
    """Fail unless Litmus read the rows of benchmark `name` that every stored baseline read."""
    read = read_rows(litmus)
    for baseline in sorted((context.work / "baselines" / "scores").glob(f"scores_{name}__*.csv")):
        if read_rows(baseline) != read:
            raise StageError(f"Litmus read different rows of {name} than {baseline.name}: the benchmark's data, the seed, or the sample size is not the baselines'. "
                             "Score the baselines again with the same settings (release/baselines.toml), or fix the settings")


def score(context: Context) -> None:
    settings = read_baselines(baselines_path(context))["scores"]
    environment = benchmarks_environment(context, LITMUS_ENGINE_RELEASE=str(context.work / "model"))
    require_engine(context, environment)
    folder = context.work / "scores" / "litmus"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "weights.txt"
    weights = receipts.file_digest(context.work / "model" / "model.safetensors")
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() != weights:
        for stale in folder.glob("scores_*__litmus.csv"):
            stale.unlink()
    marker.write_text(weights + "\n", encoding="utf-8", newline="\n")
    for name in settings["benchmarks"]:
        final = folder / f"scores_{name}__litmus.csv"
        if final.is_file():
            continue
        part = folder / f"scores_{name}__litmus.part.csv"
        part.unlink(missing_ok=True)
        run([context.python, "-m", "litmus_benchmarks", "score", "--benchmarks", name, "--detectors", DETECTOR_IN_PACKAGE, "--max-rows", str(settings["max_rows"]),
             "--seed", str(settings["seed"]), "--device", "cpu", "--out", str(part)], context.repository / "benchmarks", environment)
        relabel(part, final)
        part.unlink()
        same_rows(context, name, final)
    context.note("benchmarks", list(settings["benchmarks"]))


def score_outputs(context: Context) -> list[Path]:
    return [context.work / "scores" / "litmus"]


def cost(context: Context) -> None:
    # One thread for every library, as benchmarks/cost_chain.sh measured the other detectors (the command checks it).
    environment = benchmarks_environment(context, LITMUS_ENGINE_RELEASE=str(context.work / "model"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                         TOKENIZERS_PARALLELISM="false")
    require_engine(context, environment)
    folder = context.work / "costs"
    folder.mkdir(exist_ok=True)
    target = folder / "costs_litmus.csv"
    target.unlink(missing_ok=True)  # the cost command appends
    run([context.python, "-m", "litmus_benchmarks", "cost", "--benchmark", COST_BENCHMARK, "--detector", DETECTOR_IN_PACKAGE, "--texts", str(COST_TEXTS), "--out", str(target)],
        context.repository / "benchmarks", environment)
    with target.open(newline="", encoding="utf-8") as stream:
        ours = {row["machine"] for row in csv.DictReader(stream)}
    theirs = set()
    for path in sorted((context.work / "baselines" / "costs").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            theirs |= {row["machine"] for row in csv.DictReader(stream)}
    context.note("machine", sorted(ours))
    context.note("baseline_machines", sorted(theirs))
    context.note("same_machine_as_baselines", ours <= theirs)
    if not ours <= theirs:
        print(f"note: Litmus's cost was measured on {sorted(ours)}, the baselines' on {sorted(theirs)}; the cost table says so and compares across machines", flush=True)


def cost_outputs(context: Context) -> list[Path]:
    return [context.work / "costs" / "costs_litmus.csv"]


def results(context: Context) -> None:
    settings = read_baselines(baselines_path(context))
    record = tomllib.loads((context.work / "record" / "record.toml").read_text(encoding="utf-8"))
    environment = benchmarks_environment(context)
    costs = sorted((context.work / "baselines" / "costs").glob("*.csv")) + [context.work / "costs" / "costs_litmus.csv"]
    command = [context.python, "-m", "litmus_benchmarks", "results",
               "--scores", str(context.work / "baselines" / "scores"), str(context.work / "scores" / "litmus"),
               "--evaluation", str(context.work / "record" / "evaluation_test.json"),
               "--costs", *map(str, costs),
               "--model-version", context.plan.version, "--date", context.plan.date, "--workers", str(context.options.workers),
               "--work", str(context.work / "results"), "--site", str(context.repository / "site" / "assets"), "--docs", str(context.repository / "docs" / "images"),
               "--litmus-trained-on", *record["benchmarks"]["litmus_trained_on"]]
    filters = record["benchmarks"].get("row_filter", [])
    if filters:
        command += ["--row-filter", *filters]
    assert settings["scores"]["benchmarks"], "baselines.toml lists no benchmarks"
    run(command, context.repository / "benchmarks", environment)
    # The `docs` stage writes the release block into results.json after this stage, and holds that file in its own
    # receipt. This stage's receipt holds what it made, the file without that block, so filling the documents does not
    # make the results read as stale.
    made = json.loads((context.repository / "site" / "assets" / "results.json").read_text(encoding="utf-8"))
    made.pop("release", None)
    (context.work / "results" / RESULTS_MADE).write_text(json.dumps(made, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def results_outputs(context: Context) -> list[Path]:
    repository = context.repository
    return [context.work / "results" / RESULTS_MADE, repository / "site" / "assets" / "figures", context.work / "results" / "results.md",
            *sorted((repository / "docs" / "images").glob("benchmarks_*.png"))]


STAGES = (
    Stage("score", ("fetch",), "Litmus on every benchmark, on the rows the baselines read", score, score_outputs),
    Stage("cost", ("fetch",), "Litmus's cost on one core, for the cost table", cost, cost_outputs),
    Stage("results", ("record", "score", "cost"), "results.json, the figures, and the results tables", results, results_outputs),
)
