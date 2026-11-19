"""Stage `fetch`: the model's release folder and its training record from the Hub at the pinned commit, and the stored
baseline scores the comparison reuses.

The model repository holds the release folder (`model.safetensors`, `config.json`, `tokenizer.json`) and a record
folder (`record.toml` and the evaluation, teacher, and corpus files the documents read; `release/RELEASING.md`
lists them) at paths the plan names. The baselines repository holds one score file per benchmark and detector, and
the cost files, at the commit `release/baselines.toml` pins. Both are fetched at commits, never at branches.
"""

from __future__ import annotations

import json
import re
import shutil
import tomllib

from pathlib import Path

from . import receipts
from .commands import StageError
from .plan import REVISION
from .stages import Context, Stage


RELEASE_FILES = ("model.safetensors", "config.json", "tokenizer.json")
REQUIRED_RECORD = ("record.toml", "evaluation_test.json")
BASELINES = "baselines.toml"
SCORE_FILE = re.compile(r"^scores_(?P<benchmark>[a-z0-9_]+?)__(?P<detector>[a-z0-9_]+)\.csv$")


def download(repo: str, repo_type: str, revision: str, patterns: list[str], destination: Path) -> Path:
    """Download the files of a Hub repository that match `patterns` at a commit; returns the folder."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, repo_type=repo_type, revision=revision, allow_patterns=patterns, local_dir=destination))


def read_baselines(path: Path) -> dict:
    """The baselines settings, checked: a pinned commit, the benchmarks, the sample size and seed."""
    settings = tomllib.loads(path.read_text(encoding="utf-8"))
    scores = settings["scores"]
    if not REVISION.match(scores["revision"]):
        raise StageError(f"{path} pins no commit of the baselines repository (revision = {scores['revision']!r}): upload the stored baseline scores "
                         "to a private Hub repository and pin its commit there (release/RELEASING.md, Stored inputs)")
    for key in ("repo", "repo_type", "path", "max_rows", "seed", "benchmarks"):
        if key not in scores:
            raise StageError(f"{path}: [scores] lacks {key}")
    if not settings["costs"]["files"]:
        raise StageError(f"{path}: [costs] lists no cost files")
    return settings


def baselines_path(context: Context) -> Path:
    return context.options.baselines or context.repository / "release" / BASELINES


def copy_folder(source: Path, destination: Path, names: tuple[str, ...] | None = None) -> list[Path]:
    """Replace `destination` with the files of `source` (only `names` when given); returns what was copied."""
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    copied = []
    for item in sorted(source.iterdir()):
        if item.is_file() and (names is None or item.name in names):
            copied.append(Path(shutil.copy2(item, destination / item.name)))
    return copied


def verify_release_folder(context: Context, folder: Path) -> dict:
    """The release folder's files and format, and every hash it states, through `tools/verify_model.py`."""
    missing = [name for name in RELEASE_FILES if not (folder / name).is_file()]
    if missing:
        raise StageError(f"{context.plan.repo}@{context.plan.revision[:8]} lacks {', '.join(missing)} under {context.plan.release_path!r}; "
                         f"pass --release-path if the files sit elsewhere in the repository")
    from .building import load_tool

    problems = load_tool(context.repository, "verify_model").verify(folder, [])
    if problems:
        raise StageError("the release folder does not match the hashes it states:\n" + "\n".join(problems))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "litmus-student":
        raise StageError(f"{folder / 'config.json'} is format {config.get('format')!r}, not litmus-student")
    return config


def fetch(context: Context) -> None:
    plan, work = context.plan, context.work
    scratch = work / "hub" / "model"
    release_patterns = [f"{plan.release_path.strip('./') + '/' if plan.release_path.strip('./') else ''}{name}" for name in RELEASE_FILES]
    record_folder = plan.record_path.strip("./")
    record_patterns = [f"{record_folder}/*" if record_folder else "record.toml"]
    root = Path(plan.revision) if plan.local else download(plan.repo, "model", plan.revision, [*release_patterns, *record_patterns], scratch)
    release = root / plan.release_path.strip("./") if plan.release_path.strip("./") else root
    record = root / record_folder if record_folder else root
    config = verify_release_folder(context, release)
    absent = [name for name in REQUIRED_RECORD if not (record / name).is_file()]
    if absent:
        raise StageError(f"the record folder {plan.record_path!r} of {plan.model} lacks {', '.join(absent)}: assemble the record with "
                         f"`python release/cut.py bundle` where the training outputs are, and upload it (release/RELEASING.md, The record)")
    copy_folder(release, work / "model", RELEASE_FILES)
    copy_folder(record, work / "record")
    context.note("weights_sha256", receipts.file_digest(work / "model" / "model.safetensors"))
    context.note("model_version_in_config", (config.get("provenance") or {}).get("version"))

    settings = read_baselines(baselines_path(context))
    scores, costs = settings["scores"], settings["costs"]
    base = scores["path"].strip("./")
    files = [f"{base}/scores_{name}__*.csv" for name in scores["benchmarks"]] + [entry for entry in costs["files"]]
    stored = download(scores["repo"], scores["repo_type"], scores["revision"], files, work / "hub" / "baselines")
    present = {name: [path for path in (stored / base).glob(f"scores_{name}__*.csv")] for name in scores["benchmarks"]}
    empty = [name for name, found in present.items() if not found]
    if empty:
        raise StageError(f"the baselines repository has no stored scores for {', '.join(empty)} at {scores['revision'][:8]}: score the baselines on a "
                         "GPU machine you name (launch/benchmark_chain.sh) and upload them first")
    flat = work / "baselines" / "scores"
    shutil.rmtree(flat, ignore_errors=True)
    flat.mkdir(parents=True)
    for found in present.values():
        for path in found:
            shutil.copy2(path, flat / path.name)
    cost_folder = work / "baselines" / "costs"
    shutil.rmtree(cost_folder, ignore_errors=True)
    cost_folder.mkdir(parents=True)
    for entry in costs["files"]:
        shutil.copy2(stored / entry, cost_folder / Path(entry).name)
    context.note("baselines", f"{scores['repo']}@{scores['revision']}")


def outputs(context: Context) -> list[Path]:
    return [context.work / "model", context.work / "record", context.work / "baselines" / "scores", context.work / "baselines" / "costs"]


def inputs(context: Context) -> dict[str, str]:
    return {"baselines.toml": receipts.file_digest(baselines_path(context))}


STAGES = (Stage("fetch", (), "the release folder, its record, and the stored baselines from the Hub", fetch, outputs, inputs=inputs),)
