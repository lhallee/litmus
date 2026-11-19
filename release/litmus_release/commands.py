"""Running programs for a stage: one place, so a failure names the command and a test can replace it."""

from __future__ import annotations

import subprocess

from collections.abc import Mapping
from pathlib import Path


class StageError(SystemExit):
    """A stage that cannot finish; its message says what happened and what to do next."""


def run(command: list[str], cwd: Path, environment: Mapping[str, str] | None = None, capture: bool = False) -> str:
    """Run a command to its end and fail loudly on a nonzero exit. Output streams to the terminal unless `capture`,
    which returns the standard output."""
    finished = subprocess.run(command, cwd=cwd, env=dict(environment) if environment is not None else None, check=False,
                              capture_output=capture, text=True, encoding="utf-8")
    if finished.returncode != 0:
        detail = f":\n{finished.stderr.strip()}" if capture and finished.stderr else ""
        raise StageError(f"`{' '.join(command)}` in {cwd} exited with {finished.returncode}{detail}")
    return finished.stdout if capture else ""
