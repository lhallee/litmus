from __future__ import annotations

import pytest

from pathlib import Path
from litmus_release import receipts, stages
from litmus_release.commands import StageError
from litmus_release.stages import Context, Stage, Waiting


def make(context: Context, log: list[str]) -> tuple[Stage, ...]:
    """Three stages in a chain; each writes one file and logs that it ran."""
    def writer(name: str, content: str = "x"):
        def run(running: Context) -> None:
            log.append(name)
            (running.work / f"{name}.txt").write_text(content + str(len(log)), encoding="utf-8")
        return run

    def outputs(name: str):
        return lambda running: [running.work / f"{name}.txt"]

    return (Stage("a", (), "first", writer("a"), outputs("a")),
            Stage("b", ("a",), "second", writer("b"), outputs("b")),
            Stage("c", ("b",), "third", writer("c"), outputs("c")))


def test_stages_run_in_order_and_a_second_run_skips_what_is_done(context) -> None:
    log: list[str] = []
    chain = make(context, log)
    assert stages.execute(context, chain, list(chain)) == ["a", "b", "c"]
    assert stages.execute(context, chain, list(chain)) == []
    assert log == ["a", "b", "c"]
    assert [row[1] for row in stages.report(context, chain)] == ["done", "done", "done"]
    assert receipts.read(context.work, "b").upstream["a"] == receipts.outputs_digest(receipts.read(context.work, "a"))


def test_forcing_a_stage_makes_what_follows_it_stale_and_runs_it_again(context) -> None:
    log: list[str] = []
    chain = make(context, log)
    stages.execute(context, chain, list(chain))
    context.options.force = True
    stages.execute(context, chain, [chain[0]])
    context.options.force = False
    assert [row[1] for row in stages.report(context, chain)] == ["done", "stale", "stale"]
    assert "a changed" in stages.report(context, chain)[1][2]
    assert stages.execute(context, chain, list(chain)) == ["b", "c"]


def test_an_edited_output_makes_its_stage_stale(context) -> None:
    log: list[str] = []
    chain = make(context, log)
    stages.execute(context, chain, list(chain))
    (context.work / "b.txt").write_text("changed by hand", encoding="utf-8")
    report = stages.report(context, chain)
    assert report[1][1] == "stale" and "changed after this stage ran" in report[1][2]
    assert report[2][1] == "stale" and "b is stale" in report[2][2]


def test_a_stage_that_fails_leaves_no_receipt_and_the_next_does_not_run(context) -> None:
    log: list[str] = []
    chain = list(make(context, log))

    def fail(running: Context) -> None:
        raise StageError("the data is missing: do this")

    chain[1] = Stage("b", ("a",), "second", fail, chain[1].outputs)
    with pytest.raises(StageError, match="do this"):
        stages.execute(context, tuple(chain), list(chain))
    assert receipts.read(context.work, "b") is None and log == ["a"]


def test_a_stage_that_leaves_less_than_it_promised_fails(context) -> None:
    stage = Stage("a", (), "first", lambda running: None, lambda running: [running.work / "never.txt"])
    with pytest.raises(StageError, match="without writing"):
        stages.execute(context, (stage,), [stage])


def test_a_stage_needs_the_stages_before_it_to_be_done(context) -> None:
    chain = make(context, [])
    with pytest.raises(StageError, match="needs a, which is pending"):
        stages.execute(context, chain, [chain[1]])


def test_a_stage_for_another_platform_waits_and_says_how_to_bring_its_result_back(context) -> None:
    adopted: list[Path] = []

    def adopt(running: Context, source: Path) -> None:
        adopted.append(source)
        (running.work / "w.txt").write_text("from elsewhere", encoding="utf-8")

    stage = Stage("w", (), "windows only", lambda running: None, lambda running: [running.work / "w.txt"], platform="windows", adopt=adopt)
    with pytest.raises(Waiting, match="runs on windows, and this is linux.*adopt w"):
        stages.execute(context, (stage,), [stage])
    stages.adopt_stage(context, stage, context.work)
    assert adopted == [context.work] and stages.state(context, stage)[0] == "done"
    assert receipts.read(context.work, "w").details["adopted_from"] == str(context.work)


def test_a_gated_stage_needs_publish_and_a_dry_run_leaves_no_receipt(context) -> None:
    sent: list[str] = []
    stage = Stage("p", (), "publishes", lambda running: sent.append("dry" if running.options.dry_run else "sent"), lambda running: [running.work / "plan.json"], gated=True)
    with pytest.raises(StageError, match="--confirm"):
        stages.execute(context, (stage,), [stage])
    context.options.dry_run = True
    stages.execute(context, (stage,), [stage])
    assert sent == ["dry"] and receipts.read(context.work, "p") is None


def test_an_input_that_changes_makes_a_stage_stale(context) -> None:
    holder = {"value": "one"}
    stage = Stage("a", (), "first", lambda running: (running.work / "a.txt").write_text("x", encoding="utf-8"), lambda running: [running.work / "a.txt"],
                  inputs=lambda running: {"setting": holder["value"]})
    stages.execute(context, (stage,), [stage])
    assert stages.state(context, stage)[0] == "done"
    holder["value"] = "two"
    assert stages.state(context, stage) == ("stale", "the input setting changed after this stage ran")


def test_unknown_stage_names_are_refused_and_ranges_follow_registry_order(context) -> None:
    chain = make(context, [])
    assert [stage.name for stage in stages.order(chain, None, "b", "c")] == ["b", "c"]
    assert [stage.name for stage in stages.order(chain, ["c", "a"])] == ["a", "c"]
    with pytest.raises(StageError, match="no stage named"):
        stages.order(chain, ["nope"])
