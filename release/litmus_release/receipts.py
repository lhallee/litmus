"""Receipts: what each stage ran, with what, and what it left, so a later run can tell done from stale.

A receipt is `receipts/<stage>.json` in the work folder. A stage counts as done when its receipt says so, the
stages it needs left the same outputs, and its own outputs still hash as they did.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


RECEIPTS = "receipts"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(path: Path) -> str:
    """The SHA-256 of a file, or of a folder's file names and contents; a missing path is an error."""
    if path.is_file():
        return file_digest(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    outer = hashlib.sha256()
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        outer.update(member.relative_to(path).as_posix().encode() + b"\0" + file_digest(member).encode())
    return outer.hexdigest()


@dataclass
class Receipt:
    """One stage's record: when and where it ran, what it started from, and what it wrote."""

    stage: str
    status: str
    started: str
    seconds: float
    platform: str
    python: str
    upstream: dict[str, str]
    outputs: dict[str, str]
    details: dict[str, Any] = field(default_factory=dict)


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def write(work: Path, receipt: Receipt) -> Path:
    folder = work / RECEIPTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{receipt.stage}.json"
    path.write_text(json.dumps(asdict(receipt), indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def read(work: Path, stage: str) -> Receipt | None:
    path = work / RECEIPTS / f"{stage}.json"
    return Receipt(**json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def outputs_digest(receipt: Receipt) -> str:
    """One digest of a receipt's outputs, which the stages downstream record as their `upstream`."""
    return hashlib.sha256(json.dumps(receipt.outputs, sort_keys=True).encode()).hexdigest()


def environment() -> tuple[str, str]:
    return f"{platform.system()} {platform.machine()}", sys.version.split()[0]
