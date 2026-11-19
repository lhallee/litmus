"""Check a Litmus release folder's files against the SHA-256 hashes the release states.

A release folder holds `model.safetensors`, `config.json`, and `tokenizer.json`. `config.json` states the
tokenizer's hash (`tokenizer.sha256`) and, in a trained release, the weights' hash (`weights.sha256`).
A `SHA256SUMS` file in the folder, and any file `--sums` names (this repository's `release/SHA256SUMS`, for one),
in `sha256sum` format, are checked too. Every stated hash must match, and at least one source must state the
weights' hash, or the check fails.

    python tools/verify_model.py path/to/release
    python tools/verify_model.py path/to/release --sums release/SHA256SUMS

The weights are an opaque blob: a matching hash says the file is the one the release names, not what it computes.
The engine reads it as plain tensors (safetensors, no code), and `student/SPEC.md` says what each tensor is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from pathlib import Path


WEIGHTS = "model.safetensors"
CONFIG = "config.json"
TOKENIZER = "tokenizer.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stated_hashes(folder: Path, sums_files: list[Path]) -> dict[str, list[tuple[str, str]]]:
    """Each file's stated hashes, with where each was stated: `config.json`'s fields and each SHA256SUMS file."""
    stated: dict[str, list[tuple[str, str]]] = {}
    config = json.loads((folder / CONFIG).read_text(encoding="utf-8"))
    for file_name, field in ((TOKENIZER, "tokenizer"), (WEIGHTS, "weights")):
        digest = (config.get(field) or {}).get("sha256")
        if digest:
            stated.setdefault(file_name, []).append((digest.lower(), f"{CONFIG} {field}.sha256"))
    own = folder / "SHA256SUMS"
    for sums in ([own] if own.is_file() else []) + sums_files:
        for line in sums.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                digest, name = line.split(maxsplit=1)
                stated.setdefault(name.lstrip("*"), []).append((digest.lower(), str(sums)))
    return stated


def verify(folder: Path, sums_files: list[Path]) -> list[str]:
    """Each problem with the folder's hashes, as one line of text; empty when every stated hash matches."""
    problems = []
    stated = stated_hashes(folder, sums_files)
    if WEIGHTS not in stated:
        problems.append(f"no hash of {WEIGHTS} is stated: {CONFIG} has no weights.sha256 and no SHA256SUMS lists it")
    for file_name, claims in sorted(stated.items()):
        path = folder / file_name
        if not path.is_file():
            problems.append(f"{file_name} is stated in {claims[0][1]} but missing from the folder")
            continue
        actual = sha256(path)
        print(f"{actual}  {file_name}")
        problems += [f"{file_name}: {source} states {digest}, the file is {actual}" for digest, source in claims if digest != actual]
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="a release folder holding model.safetensors, config.json, and tokenizer.json")
    parser.add_argument("--sums", type=Path, nargs="*", default=[], help="further SHA256SUMS files to check against")
    arguments = parser.parse_args()
    problems = verify(arguments.folder, arguments.sums)
    for problem in problems:
        print(problem)
    print("every stated hash matches" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
