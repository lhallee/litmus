"""`python -m litmus_student <command> ...`: `poc` (the proof of concept, `poc.py`) and `search` (the distillation's
Bayesian search, `search.py`)."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {"poc": "litmus_student.poc", "search": "litmus_student.search"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: python -m litmus_student {{{','.join(COMMANDS)}}} ...")
    importlib.import_module(COMMANDS[sys.argv[1]]).main(sys.argv[2:])


if __name__ == "__main__":
    main()
