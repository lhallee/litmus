"""The command line of the release driver (`python release/cut.py`)."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from . import bundle as bundling
from . import plan as planning
from . import registry, slots, stages
from .commands import StageError
from .data import documents
from .documents import markdown_files, read_text, unfinished, values
from .documents import targets_path
from .plan import PlanError
from .versions import MANIFESTS, problems, apply


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_PATH = "release"
DEFAULT_RECORD_PATH = "record"
PUBLISHING = ("publish", "pages", "macos")


def add_work(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work", type=Path, required=True, help="the work folder of this release, outside the repository")
    parser.add_argument("--repository", type=Path, default=REPOSITORY, help="the checkout the release edits and builds (default: this one)")


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cut.py", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the stages in order, skipping what is done; the first call writes the plan")
    add_work(run)
    run.add_argument("--model", help="the model to release: OWNER/NAME@<40-character commit>, or local:FOLDER for a trial (never published); needed once")
    run.add_argument("--version", help="the release version, MAJOR.MINOR.PATCH; needed once")
    run.add_argument("--release-path", default=DEFAULT_RELEASE_PATH, help="the release folder inside the model repository")
    run.add_argument("--record-path", default=DEFAULT_RECORD_PATH, help="the record folder inside the model repository")
    run.add_argument("--from", dest="start", help="the first stage to consider")
    run.add_argument("--until", default="assemble", help="the last stage to consider (default assemble; the publishing stages have their own commands)")
    run.add_argument("--only", nargs="+", help="only these stages")
    run.add_argument("--force", action="store_true", help="run the chosen stages again even if they are done")
    run.add_argument("--workers", type=int, default=1, help="processes for the metric table")
    run.add_argument("--allow-dirty", action="store_true", help="build with uncommitted changes in engine/, app/, or site/ (a trial; never published)")
    run.add_argument("--baselines", type=Path, help="a baselines file other than release/baselines.toml")
    run.add_argument("--ad-clips", type=Path, help="footage of the Linux app, for the optional ad stage")

    for name in ("status",):
        status = commands.add_parser(name, help="which stages are done, stale, or waiting")
        add_work(status)

    adopt = commands.add_parser("adopt", help="take a stage's result made on another machine or by hand")
    adopt.add_argument("stage")
    add_work(adopt)
    adopt.add_argument("--from", dest="source", type=Path, required=True)

    for name in PUBLISHING:
        publish = commands.add_parser(name, help=f"the {name} stage: changes remote state, so it needs --confirm; --dry-run prints what it would do")
        add_work(publish)
        mode = publish.add_mutually_exclusive_group(required=True)
        mode.add_argument("--dry-run", action="store_true")
        mode.add_argument("--confirm", action="store_true", help="do it: the person's explicit word is required first")

    bundle = commands.add_parser("bundle", help="gather a run's outputs into the release folder and record folder to upload with the model")
    bundle.add_argument("--release", type=Path, required=True, help="the exported release folder")
    bundle.add_argument("--out", type=Path, required=True, help="a new folder")
    bundle.add_argument("--record", type=Path, required=True, help="the filled record.toml")
    bundle.add_argument("--evaluation-test", type=Path, required=True, help="the test evaluation's folder (evaluation.json and evaluation.csv)")
    bundle.add_argument("--evaluation-validation", type=Path, help="the validation evaluation's folder")
    bundle.add_argument("--teacher", type=Path, help="the teacher's calibration folder (metrics.csv, uncertainty.csv)")
    bundle.add_argument("--file", nargs="*", default=[], help="more record files as name=path, such as corpus_report.json=...")

    template = commands.add_parser("record-template", help="write record.toml with an empty entry for every key the documents read")
    template.add_argument("--repository", type=Path, default=REPOSITORY)
    template.add_argument("--out", type=Path)

    slot_command = commands.add_parser("slots", help="list or check the slots of the documents")
    slot_command.add_argument("action", choices=("list", "check"))
    add_work(slot_command)

    version = commands.add_parser("version", help="check or set the version in every manifest of a checkout, without a work folder")
    version.add_argument("action", choices=("check", "set"))
    version.add_argument("version")
    version.add_argument("--repository", type=Path, default=REPOSITORY)
    return parser.parse_args(argv)


def make_context(arguments: argparse.Namespace, options: stages.Options) -> stages.Context:
    work = arguments.work.resolve()
    repository = arguments.repository.resolve()
    if repository == work or work.is_relative_to(repository):
        raise PlanError(f"{work} is inside the repository; the work folder stays outside it")
    if getattr(arguments, "model", None):
        if not arguments.version:
            raise PlanError("--model goes with --version")
        plan = planning.create(work, model=arguments.model, version=arguments.version, release_path=arguments.release_path, record_path=arguments.record_path)
    else:
        plan = planning.load(work)
    return stages.Context(plan=plan, work=work, repository=repository, options=options)


def command_run(arguments: argparse.Namespace) -> int:
    options = stages.Options(workers=arguments.workers, force=arguments.force, allow_dirty=arguments.allow_dirty, baselines=arguments.baselines, ad_clips=arguments.ad_clips)
    context = make_context(arguments, options)
    registry_stages = registry.build()
    selected = stages.order(registry_stages, arguments.only, arguments.start, None if arguments.only else arguments.until)
    stages.execute(context, registry_stages, selected)
    return 0


def command_status(arguments: argparse.Namespace) -> int:
    context = make_context(arguments, stages.Options())
    print(f"{context.plan.model} as version {context.plan.version}, work folder {context.work}")
    for name, found, note in stages.report(context, registry.build()):
        print(f"  {name:<14} {found:<8} {note}")
    return 0


def command_adopt(arguments: argparse.Namespace) -> int:
    context = make_context(arguments, stages.Options())
    registry_stages = registry.build()
    chosen = stages.order(registry_stages, [arguments.stage])[0]
    stages.check_needs(context, chosen, set(), {stage.name: stage for stage in registry_stages})
    stages.adopt_stage(context, chosen, arguments.source.resolve())
    print(f"{chosen.name}: adopted from {arguments.source}")
    return 0


def command_publishing(arguments: argparse.Namespace) -> int:
    options = stages.Options(publish=arguments.confirm, dry_run=arguments.dry_run)
    context = make_context(arguments, options)
    registry_stages = registry.build()
    stages.execute(context, registry_stages, stages.order(registry_stages, [arguments.command]))
    return 0


def command_bundle(arguments: argparse.Namespace) -> int:
    extra = {}
    for item in arguments.file:
        name, separator, path = item.partition("=")
        if not separator:
            raise StageError(f"--file takes name=path, not {item!r}")
        extra[name] = Path(path)
    written = bundling.bundle(arguments.release, arguments.out, record=arguments.record, evaluation_test=arguments.evaluation_test,
                              evaluation_validation=arguments.evaluation_validation, teacher=arguments.teacher, extra=extra)
    for path in written:
        print(path)
    print("upload it to the model's Hub repository, release/ and record/ at the repository's root, with `hf upload` or HfApi.upload_folder, and pin the commit it makes")
    return 0


def command_template(arguments: argparse.Namespace) -> int:
    text = bundling.template(arguments.repository.resolve())
    if arguments.out:
        arguments.out.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0


def command_slots(arguments: argparse.Namespace) -> int:
    context = make_context(arguments, stages.Options())
    available = documents(context, targets_path(context))
    renderer = values(context, available)
    bad = 0
    for path in markdown_files(context.repository):
        text = read_text(path)
        if "<!--litmus:" not in text:
            continue
        name = path.relative_to(context.repository).as_posix()
        for slot in slots.find(text):
            try:
                shown = renderer.render(slot.key, slot.block)
                state = "ok" if text[slot.start:slot.end].strip("\n") == shown.strip("\n") else "stale"
            except (LookupError, ValueError) as error:
                shown, state = str(error), "MISSING"
            bad += state != "ok"
            if arguments.action == "list" or state != "ok":
                print(f"{name}: {slot.key}: {state}: {shown.splitlines()[0][:100] if shown else ''}")
    left = unfinished(context)
    print(f"{bad} slots do not render or are stale; {len(left)} unfinished items")
    return 1 if arguments.action == "check" and (bad or left) else 0


def command_version(arguments: argparse.Namespace) -> int:
    repository = arguments.repository.resolve()
    if arguments.action == "set":
        changed = apply(repository, arguments.version)
        print("changed: " + (", ".join(path.relative_to(repository).as_posix() for path in changed) or "nothing"))
    found = problems(repository, arguments.version)
    print("\n".join(found) if found else f"every manifest states {arguments.version} ({len(MANIFESTS)} manifests)")
    return 1 if found else 0


COMMANDS = {"run": command_run, "status": command_status, "adopt": command_adopt, "bundle": command_bundle, "record-template": command_template,
            "slots": command_slots, "version": command_version, **{name: command_publishing for name in PUBLISHING}}


def main(argv: list[str] | None = None) -> int:
    arguments = parse(argv)
    try:
        return COMMANDS[arguments.command](arguments)
    except stages.Waiting as waiting:
        print(f"waiting: {waiting}", file=sys.stderr)
        return 3
    except StageError as error:
        print(f"failed: {error}", file=sys.stderr)
        return 1
