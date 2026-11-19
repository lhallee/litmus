"""`python -m litmus_teacher embed|split|search|train|bank|calibrate|export ...`: one command per step."""

from __future__ import annotations

import argparse
import importlib


STEPS = {
    "embed": "embed",
    "split": "split",
    "search": "search",
    "train": "train",
    "bank": "bank",
    "calibrate": "evaluate",
    "export": "export",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m litmus_teacher", description=__doc__)
    commands = parser.add_subparsers(dest="step", required=True)
    for step, module_name in STEPS.items():
        module = importlib.import_module(f".{module_name}", __package__)
        module.add_arguments(commands.add_parser(step, help=(module.__doc__ or "").splitlines()[0]))
    args = parser.parse_args()
    importlib.import_module(f".{STEPS[args.step]}", __package__).main(args)


if __name__ == "__main__":
    main()
