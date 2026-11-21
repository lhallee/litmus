"""The resumable engine: stages in order, each idempotent, each leaving a receipt, none run twice without cause.

A stage is a name, the stages it needs, a function that does the work, and a function that lists what it leaves. The
engine runs the chosen stages in registry order, skips a stage whose receipt is current, stops at the first failure,
and says what to do. A stage that needs another platform is not run here; it says which command to run there and how
to bring the result back (`adopt`).
"""

from __future__ import annotations

import platform
import sys
import time

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import receipts
from .commands import StageError
from .plan import Plan


class Waiting(StageError):
    """A stage that cannot run on this machine; the message names where it can."""


@dataclass
class Options:
    """What a run may use that is not part of the release: processes, a flag for each kind of remote effect."""

    workers: int = 1
    publish: bool = False
    dry_run: bool = False
    ad_clips: Path | None = None
    baselines: Path | None = None
    allow_dirty: bool = False
    force: bool = False
    # `pages` only: publish the site although the release does not hold the macOS programs it links yet.
    ahead_of_macos: bool = False


@dataclass
class Context:
    """What a stage reads: the plan, the work folder, the checkout it edits, the options, and a note pad that
    becomes the receipt's details."""

    plan: Plan
    work: Path
    repository: Path
    options: Options = field(default_factory=Options)
    system: str = field(default_factory=lambda: platform.system().lower())
    python: str = field(default_factory=lambda: sys.executable)
    details: dict[str, Any] = field(default_factory=dict)

    def note(self, key: str, value: Any) -> None:
        self.details[key] = value

    def reference(self, path: Path) -> str:
        """A path as a receipt writes it: relative to the work folder or the checkout, so a copy of either still reads."""
        for prefix, base in (("work", self.work), ("repo", self.repository)):
            if path.is_relative_to(base):
                return f"{prefix}:{path.relative_to(base).as_posix()}"
        return path.as_posix()

    def locate(self, reference: str) -> Path:
        prefix, separator, rest = reference.partition(":")
        if separator and prefix in ("work", "repo"):
            return (self.work if prefix == "work" else self.repository) / rest
        return Path(reference)


@dataclass(frozen=True)
class Stage:
    """One step of a release."""

    name: str
    needs: tuple[str, ...]
    summary: str
    run: Callable[[Context], None]
    outputs: Callable[[Context], list[Path]]
    platform: str | None = None
    required: bool = True
    gated: bool = False
    # Digests of what the stage reads that no stage writes (a settings file, the checkout's version); a change makes it stale.
    inputs: Callable[[Context], dict[str, str]] = field(default=lambda context: {})
    # Takes what another machine or a person produced (a folder or a file), checks it, and places it where run would.
    adopt: Callable[[Context, Path], None] | None = None


def state(context: Context, stage: Stage, known: dict[str, Stage] | None = None) -> tuple[str, str]:
    """`done`, `stale` (with why), or `pending`, from the stage's receipt. With the stages by name, a stage whose upstream is
    not done is stale too, however far back."""
    receipt = receipts.read(context.work, stage.name)
    if receipt is None or receipt.status != "done":
        return "pending", ""
    for need in stage.needs:
        if known is not None and state(context, known[need], known)[0] != "done":
            return "stale", f"{need} is {state(context, known[need], known)[0]}"
        upstream = receipts.read(context.work, need)
        if upstream is None:
            return "stale", f"{need} has no receipt"
        if receipt.upstream.get(need) != receipts.outputs_digest(upstream):
            return "stale", f"{need} changed after this stage ran"
    for name, wanted in stage.inputs(context).items():
        if receipt.upstream.get(f"input:{name}") != wanted:
            return "stale", f"the input {name} changed after this stage ran"
    for reference, recorded in receipt.outputs.items():
        path = context.locate(reference)
        if not path.exists():
            return "stale", f"{reference} is gone"
        if receipts.digest(path) != recorded:
            return "stale", f"{reference} changed after this stage ran"
    return "done", ""


def order(stages: tuple[Stage, ...], names: list[str] | None = None, start: str | None = None, until: str | None = None) -> list[Stage]:
    """The stages to consider, in registry order: those named, or from `start` to `until`."""
    known = [stage.name for stage in stages]
    for name in [*(names or []), *(item for item in (start, until) if item)]:
        if name not in known:
            raise StageError(f"no stage named {name!r}; the stages are {', '.join(known)}")
    if names:
        return [stage for stage in stages if stage.name in names]
    first = known.index(start) if start else 0
    last = known.index(until) if until else len(stages) - 1
    return list(stages[first:last + 1])


def check_needs(context: Context, stage: Stage, finished: set[str], by_name: dict[str, Stage]) -> None:
    for need in stage.needs:
        if need in finished:
            continue
        found, why = state(context, by_name[need], by_name)
        if found != "done":
            raise StageError(f"stage {stage.name} needs {need}, which is {found}{f' ({why})' if why else ''}: "
                             f"run `python release/cut.py run --work {context.work} --only {need}` first, or run from {need}")


def finish(context: Context, stage: Stage, stamp: str, started: float) -> None:
    """Check what the stage left and write its receipt; nothing is written if it left less than it promised."""
    produced = {}
    for path in stage.outputs(context):
        if not path.exists():
            raise StageError(f"stage {stage.name} finished without writing {path}")
        produced[context.reference(path)] = receipts.digest(path)
    system, python = receipts.environment()
    upstream = {}
    for need in stage.needs:
        needed = receipts.read(context.work, need)
        assert needed is not None, f"{need} has no receipt although {stage.name} ran after it"
        upstream[need] = receipts.outputs_digest(needed)
    upstream.update({f"input:{name}": digest for name, digest in stage.inputs(context).items()})
    receipts.write(context.work, receipts.Receipt(stage=stage.name, status="done", started=stamp, seconds=round(time.monotonic() - started, 1),
                                                  platform=system, python=python, upstream=upstream, outputs=produced, details=dict(context.details)))


def run_stage(context: Context, stage: Stage) -> None:
    """Run one stage and write its receipt."""
    context.details = {}
    started = time.monotonic()
    stamp = receipts.now()
    stage.run(context)
    finish(context, stage, stamp, started)


def adopt_stage(context: Context, stage: Stage, source: Path) -> None:
    """Take `source` as the stage's result, and write the receipt as if the stage had run here."""
    if stage.adopt is None:
        raise StageError(f"stage {stage.name} cannot adopt a result; run it")
    context.details = {"adopted_from": str(source)}
    started = time.monotonic()
    stamp = receipts.now()
    stage.adopt(context, source)
    finish(context, stage, stamp, started)


def execute(context: Context, stages: tuple[Stage, ...], selected: list[Stage]) -> list[str]:
    """Run `selected` in order. Returns the names of the stages that ran; raises at the first stage that cannot."""
    by_name = {stage.name: stage for stage in stages}
    finished = {stage.name for stage in stages if state(context, stage, by_name)[0] == "done"} - {stage.name for stage in selected if context.options.force}
    ran = []
    for stage in selected:
        found, why = state(context, stage, by_name)
        if found == "done" and not context.options.force:
            print(f"{stage.name}: done, skipped", flush=True)
            continue
        check_needs(context, stage, finished, by_name)
        if stage.gated and context.options.dry_run:
            print(f"{stage.name}: dry run", flush=True)
            stage.run(context)
            continue
        if stage.gated and not context.options.publish:
            raise StageError(f"stage {stage.name} changes something outside this machine; run it with `python release/cut.py {stage.name} --work <work folder> --confirm`, after `--dry-run` shows what it will do")
        if stage.platform and stage.platform != context.system:
            then = f"; then `python release/cut.py adopt {stage.name} --work {context.work} --from <that machine's work folder>` here" if stage.adopt else ""
            raise Waiting(f"stage {stage.name} runs on {stage.platform}, and this is {context.system}: copy the work folder {context.work} there (without build/ and publish/; its receipts are what the stage checks), "
                          f"then run `python release/cut.py run --work <the copy> --only {stage.name}` on that machine from a checkout of the release commit{then}")
        print(f"{stage.name}: running ({stage.summary}){f' because {why}' if why else ''}", flush=True)
        run_stage(context, stage)
        finished.add(stage.name)
        ran.append(stage.name)
        print(f"{stage.name}: done", flush=True)
    return ran


def report(context: Context, stages: tuple[Stage, ...]) -> list[tuple[str, str, str]]:
    """Each stage's name, its state, and a note (why it is stale, where it runs, when it ran)."""
    rows = []
    by_name = {stage.name: stage for stage in stages}
    for stage in stages:
        found, why = state(context, stage, by_name)
        receipt = receipts.read(context.work, stage.name)
        note = why or (f"{receipt.started} on {receipt.platform}" if receipt and found == "done" else "")
        if found != "done" and stage.platform:
            note = f"runs on {stage.platform}" + (f"; {note}" if note else "")
        if stage.gated:
            note = ("changes remote state; " + note).strip("; ")
        if not stage.required:
            note = ("optional; " + note).strip("; ")
        rows.append((stage.name, found, note))
    return rows
