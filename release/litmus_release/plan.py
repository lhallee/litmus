"""The plan: which model is released as which version, written once into the work folder.

A work folder holds one plan. Every later command reads it, so a stage cannot run against another model or version by
a slip of an argument; a command that names a different model or version than the plan's fails.
"""

from __future__ import annotations

import datetime
import json
import re

from dataclasses import asdict, dataclass
from pathlib import Path


PLAN_FILE = "plan.json"
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
# A model read from a folder instead of the Hub: for a trial on this machine, never published.
LOCAL = "local"


class PlanError(SystemExit):
    """A plan or a command line that cannot be accepted."""


@dataclass(frozen=True)
class Plan:
    """What is released: the model's Hub repository and pinned commit, the version, and where the files sit in it."""

    version: str
    repo: str
    revision: str
    release_path: str
    record_path: str
    date: str

    @property
    def tag(self) -> str:
        return f"v{self.version}"

    @property
    def local(self) -> bool:
        return self.repo == LOCAL

    @property
    def model(self) -> str:
        return f"local:{self.revision}" if self.local else f"{self.repo}@{self.revision}"


def parse_model(text: str) -> tuple[str, str]:
    """`owner/name@commit` as (repository, commit); a branch or a short hash is refused, since it can move. `local:FOLDER`
    names a folder that holds the release and record folders, for a trial."""
    if text.startswith("local:"):
        folder = Path(text[len("local:"):]).resolve()
        if not folder.is_dir():
            raise PlanError(f"{folder} is not a folder")
        return LOCAL, folder.as_posix()
    repo, separator, revision = text.partition("@")
    if not separator or not REPOSITORY_ID.match(repo):
        raise PlanError(f"--model is `owner/name@<commit>`, not {text!r}")
    if not REVISION.match(revision):
        raise PlanError(f"--model pins a full 40-character commit, not {revision!r}: a branch or a tag can move under a release "
                        "(find the commit with `huggingface_hub.HfApi().model_info(repo).sha`)")
    return repo, revision


def create(work: Path, *, model: str, version: str, release_path: str, record_path: str, today: datetime.date | None = None) -> Plan:
    """Write the plan of a new work folder, or return the plan already there if it says the same thing."""
    if not VERSION.match(version):
        raise PlanError(f"--version is MAJOR.MINOR.PATCH such as 1.0.0, not {version!r}")
    repo, revision = parse_model(model)
    date = (today or datetime.datetime.now(datetime.UTC).date()).isoformat()
    existing = work / PLAN_FILE
    if existing.is_file():
        plan = load(work)
        if (plan.version, plan.repo, plan.revision, plan.release_path, plan.record_path) != (version, repo, revision, release_path, record_path):
            raise PlanError(f"{work} already holds the plan {plan.model} as version {plan.version}; use another work folder for another release")
        return plan
    plan = Plan(version=version, repo=repo, revision=revision, release_path=release_path, record_path=record_path, date=date)
    work.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps(asdict(plan), indent=1) + "\n", encoding="utf-8", newline="\n")
    return plan


def load(work: Path) -> Plan:
    path = work / PLAN_FILE
    if not path.is_file():
        raise PlanError(f"{work} holds no {PLAN_FILE}: start the release with `python release/cut.py run --model OWNER/NAME@COMMIT --version X.Y.Z --work {work}`")
    return Plan(**json.loads(path.read_text(encoding="utf-8")))
