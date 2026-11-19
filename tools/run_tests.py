"""Run every test suite of this repository, or the ones named, and report each one's outcome and time.

    python tools/run_tests.py                      # every suite below
    python tools/run_tests.py security engine app  # the named suites
    python tools/run_tests.py --list               # what each suite proves

A suite whose tool is missing (cargo, or pytest and the package's dependencies) fails with the reason; nothing is
skipped quietly. The build check (`build`) rebuilds the command line and the WebAssembly module twice, so it is the
slowest by far and runs only when named or with `--all`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# tools/ is this script's folder, so its sibling imports by name.
import verify_build

from dataclasses import dataclass
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Suite:
    """One test suite: where it runs, its command, and what passing it shows."""

    name: str
    folder: str
    command: tuple[str, ...]
    proves: str
    python_path: tuple[str, ...] = ()


PYTHON = sys.executable
SUITES = (
    Suite("security", ".", (PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tools/security/tests"),
          "Every network, process, clipboard, registry, input, file-writing, storage, and dynamic-code use in the source "
          "is declared in tools/security/capabilities.json, `unsafe` counts match, the Rust dependencies match their lock files "
          "and licenses, and the model verifier catches a tampered file"),
    Suite("engine", "engine", ("cargo", "test", "--release", "--locked", "-p", "litmus-engine", "-p", "litmus-files", "-p", "litmus-cli"),
          "The engine's kernels agree across SIMD backends and with the scalar path, the release format, chunks, windows, "
          "cells, and the strip; file reading, walking, and the cache; the command line"),
    Suite("app", "app", ("cargo", "test", "--release", "--locked"),
          "The app's wording, strip geometry, heat-map cells, history, settings, sorting, one-instance lock, and icon"),
    Suite("corpus", "corpus", (PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"), "The corpus pipeline on fixtures: reading, standardizing, strata, checks, balance, writing"),
    Suite("teacher", "teacher", (PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"), "The teacher: splits, the pair head, the bank, calibration, evaluation", ("src", "../training", "../corpus/src")),
    Suite("student", "student", (PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"),
          "The student: the NumPy reference, export, distillation, integer attention, and parity with the reference",
          ("src", ".", "../training", "../teacher/src", "../corpus/src")),
    Suite("benchmarks", "benchmarks", (PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"), "The benchmark catalog, scoring, metrics, and figures", ("src", "../training")),
    Suite("contribute", "contribute", (PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"), "The contribute server: requests, privacy, rate limits, shards, legal documents", ("src", "../training")),
    Suite("release", "release", (PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"),
          "The release driver: stage receipts and staleness, version setting, every number's slot and its data, the record bundle, "
          "the release assets, and the gates and dry runs of the publishing stages"),
    Suite("build", ".", (PYTHON, "tools/verify_build.py"),
          "Two builds of the command line, from copies at different paths, and two of the WebAssembly module, at the fixed "
          "path, are byte-identical"),
)
DEFAULT = ("security", "engine", "app", "corpus", "teacher", "student", "benchmarks", "contribute")


def run(suite: Suite) -> tuple[bool, float, str]:
    """Run one suite; return whether it passed, its seconds, and the last lines of its output on failure."""
    environment = dict(os.environ)
    command = suite.command
    if command[0] == "cargo":
        # rustup installs cargo in ~/.cargo/bin, which a shell may not have on PATH. Windows looks a program up on
        # this process's PATH, not the child's, so cargo is named by its full path; the child's PATH lets cargo find
        # rustc and rustup's proxies.
        cargo = verify_build.cargo_folder()
        if cargo is None:
            return False, 0.0, "cannot start cargo: install Rust with rustup (https://rustup.rs)"
        environment["PATH"] = f"{cargo}{os.pathsep}{environment.get('PATH', '')}"
        command = (str(cargo / verify_build.executable("cargo")), *command[1:])
    if suite.python_path:
        # training/ holds the vendored `foundry` library; the rest are sibling packages the suite imports.
        folder = REPOSITORY / suite.folder
        entries = [str((folder / entry).resolve()) for entry in suite.python_path]
        environment["PYTHONPATH"] = os.pathsep.join([*entries, environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    started = time.monotonic()
    try:
        finished = subprocess.run(command, cwd=REPOSITORY / suite.folder, env=environment, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        return False, time.monotonic() - started, f"cannot start {command[0]}: {error}"
    seconds = time.monotonic() - started
    tail = "\n".join((finished.stdout + finished.stderr).strip().splitlines()[-15:])
    return finished.returncode == 0, seconds, tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suites", nargs="*", help=f"suites to run, of {', '.join(suite.name for suite in SUITES)}; default: all but build")
    parser.add_argument("--all", action="store_true", help="also run the build check")
    parser.add_argument("--list", action="store_true", help="say what each suite proves and exit")
    arguments = parser.parse_args()

    if arguments.list:
        for suite in SUITES:
            print(f"{suite.name:<11} {suite.proves}")
        return 0

    unknown = sorted(set(arguments.suites) - {suite.name for suite in SUITES})
    if unknown:
        parser.error(f"no suite named {', '.join(unknown)}")
    if arguments.suites:
        names = tuple(arguments.suites)
    else:
        names = (*DEFAULT, "build") if arguments.all else DEFAULT
    failed = []
    for suite in (suite for suite in SUITES if suite.name in names):
        print(f"{suite.name}: running {' '.join(suite.command[1:] if suite.command[0] == PYTHON else suite.command)}", flush=True)
        passed, seconds, tail = run(suite)
        print(f"{suite.name}: {'passed' if passed else 'FAILED'} in {seconds:.0f} s", flush=True)
        if not passed:
            failed.append(suite.name)
            print(tail)
    print("all passed" if not failed else f"failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
