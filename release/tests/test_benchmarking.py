"""A release that reads every text as an earlier one did takes that release's scores and cost; any other is refused."""

from __future__ import annotations

import copy
import json
import shutil
import pytest

from pathlib import Path
from litmus_release import benchmarking, receipts, stages
from litmus_release.commands import StageError
from litmus_release.stages import Context

from .conftest import CONFIG


BASELINES = """
[scores]
repo = "owner/baselines"
repo_type = "dataset"
revision = "{revision}"
path = "scores"
max_rows = 3
seed = 0
benchmarks = ["mage"]
[costs]
files = ["costs/costs_small.csv"]
"""
SCORES = "row,origin,words,detector,score\n0,human,12,{detector},0.1\n1,ai,30,{detector},0.9\n"
COST = "detector,machine,tokens_per_second\nLitmus engine,Neoverse / Linux,23000\n"
BETWEEN_SOURCES = {"level": 0.9, "log_odds": 1.4}


def receipt(work: Path, stage: str, outputs: list[Path]) -> None:
    receipts.write(work, receipts.Receipt(stage=stage, status="done", started="2026-01-01T00:00:00+00:00", seconds=1.0, platform="Linux aarch64", python="3.12",
                                          upstream={}, outputs={f"work:{path.relative_to(work).as_posix()}": receipts.digest(path) for path in outputs}))


@pytest.fixture
def earlier(tmp_path: Path, context: Context) -> Path:
    """The work folder of an earlier release of the context's model, with its scores and cost, and the context set up
    as a release whose calibration table has gained the shift between sources."""
    source = tmp_path / "earlier"
    shutil.copytree(context.work / "model", source / "model")
    (source / "scores" / "litmus").mkdir(parents=True)
    (source / "scores" / "litmus" / "scores_mage__litmus.csv").write_text(SCORES.format(detector="Litmus"), encoding="utf-8", newline="\n")
    (source / "scores" / "litmus" / "weights.txt").write_text("weights\n", encoding="utf-8", newline="\n")
    (source / "costs").mkdir()
    (source / "costs" / "costs_litmus.csv").write_text(COST, encoding="utf-8", newline="\n")
    receipt(source, "score", [source / "scores" / "litmus"])
    receipt(source, "cost", [source / "costs" / "costs_litmus.csv"])

    widened = copy.deepcopy(CONFIG)
    widened["calibration"]["between_sources"] = BETWEEN_SOURCES
    (context.work / "model" / "config.json").write_text(json.dumps(widened, indent=1), encoding="utf-8")
    (context.work / "baselines" / "scores").mkdir(parents=True)
    (context.work / "baselines" / "scores" / "scores_mage__other.csv").write_text(SCORES.format(detector="Other"), encoding="utf-8", newline="\n")
    (context.work / "baselines" / "costs").mkdir()
    (context.work / "baselines" / "costs" / "costs_small.csv").write_text("detector,machine,tokens_per_second\nOther,Xeon / Linux,300\n", encoding="utf-8", newline="\n")
    context.options.baselines = tmp_path / "baselines.toml"
    context.options.baselines.write_text(BASELINES.format(revision="b" * 40), encoding="utf-8")
    receipt(context.work, "fetch", [context.work / "model"])
    return source


def stage(name: str) -> stages.Stage:
    return next(found for found in benchmarking.STAGES if found.name == name)


def test_a_release_that_reads_the_same_adopts_the_scores_and_the_cost(context: Context, earlier: Path) -> None:
    stages.adopt_stage(context, stage("score"), earlier)
    stages.adopt_stage(context, stage("cost"), earlier)
    assert (context.work / "scores" / "litmus" / "scores_mage__litmus.csv").read_text(encoding="utf-8") == SCORES.format(detector="Litmus")
    assert (context.work / "costs" / "costs_litmus.csv").read_text(encoding="utf-8") == COST
    by_name = {found.name: found for found in benchmarking.STAGES}
    assert stages.state(context, by_name["score"])[0] == "done" and stages.state(context, by_name["cost"])[0] == "done"
    scored, costed = receipts.read(context.work, "score"), receipts.read(context.work, "cost")
    assert scored.details == {"adopted_from": str(earlier), "made_at": "2026-01-01T00:00:00+00:00", "made_on": "Linux aarch64", "benchmarks": ["mage"]}
    assert costed.details["machine"] == ["Neoverse / Linux"] and costed.details["same_machine_as_baselines"] is False


def test_the_shift_between_sources_is_the_only_difference_a_score_survives(context: Context, earlier: Path) -> None:
    model = context.work / "model"
    assert benchmarking.reading_digest(model) == benchmarking.reading_digest(earlier / "model")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    config["calibration"]["calibration_rows"] += 1
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert benchmarking.reading_digest(model) != benchmarking.reading_digest(earlier / "model")
    with pytest.raises(StageError, match="reads differently"):
        stages.adopt_stage(context, stage("score"), earlier)
    assert receipts.read(context.work, "score") is None and not (context.work / "scores").exists()


@pytest.mark.parametrize("name", ["model.safetensors", "tokenizer.json"])
def test_other_weights_or_another_tokenizer_are_refused(context: Context, earlier: Path, name: str) -> None:
    (earlier / "model" / name).write_bytes(b"another release")
    for adopted in ("score", "cost"):
        with pytest.raises(StageError, match="other weights, another tokenizer"):
            stages.adopt_stage(context, stage(adopted), earlier)


def test_scores_changed_since_their_stage_ran_are_refused(context: Context, earlier: Path) -> None:
    (earlier / "scores" / "litmus" / "scores_mage__litmus.csv").write_text(SCORES.format(detector="Edited"), encoding="utf-8", newline="\n")
    with pytest.raises(StageError, match="is not what its score stage left"):
        stages.adopt_stage(context, stage("score"), earlier)


def test_scores_of_other_rows_than_the_baselines_read_are_refused(context: Context, earlier: Path) -> None:
    (context.work / "baselines" / "scores" / "scores_mage__other.csv").write_text("row,origin,words,detector,score\n7,human,12,Other,0.1\n", encoding="utf-8", newline="\n")
    with pytest.raises(StageError, match="read different rows of mage"):
        stages.adopt_stage(context, stage("score"), earlier)
    assert receipts.read(context.work, "score") is None


def test_a_missing_benchmark_an_unfinished_stage_and_the_work_folder_itself_are_refused(context: Context, earlier: Path) -> None:
    with pytest.raises(StageError, match="own work folder"):
        stages.adopt_stage(context, stage("score"), context.work)
    context.options.baselines.write_text(BASELINES.format(revision="b" * 40).replace('["mage"]', '["mage", "hc3"]'), encoding="utf-8")
    with pytest.raises(StageError, match="no scores of hc3"):
        stages.adopt_stage(context, stage("score"), earlier)
    (earlier / "receipts" / "cost.json").unlink()
    with pytest.raises(StageError, match="no finished cost stage"):
        stages.adopt_stage(context, stage("cost"), earlier)
