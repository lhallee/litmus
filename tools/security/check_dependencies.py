"""Check the Rust dependencies of the engine and the app against `tools/security/dependencies.json`.

Each `Cargo.lock` pins every crate the build may compile. The check fails when:

- a lock file holds a crate, or a version, that `dependencies.json` does not list, or the reverse, so any change to
  the dependency tree shows up as a reviewed change to that file;
- a crate cannot be taken under the licenses THIRD_PARTY.md covers as a group (COVERED) and is not named there,
  or is under a license outside NAMED;
- a lock file holds a crate whose purpose is network access (NETWORK_CRATES).

The licenses come from each crate's own `Cargo.toml` in the local Cargo registry, read by `--write`; the check itself
needs only the lock files and `dependencies.json`, so it runs offline.

    python tools/security/check_dependencies.py            # check; exit status 1 on any problem
    python tools/security/check_dependencies.py --write    # rewrite dependencies.json from the lock files and the registry
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
RECORD = REPOSITORY / "tools" / "security" / "dependencies.json"
THIRD_PARTY = REPOSITORY / "engine" / "THIRD_PARTY.md"
LOCK_FILES = ("engine/Cargo.lock", "app/Cargo.lock")

# The licenses THIRD_PARTY.md covers as a group ("MIT, Apache-2.0, Zlib, Unicode-3.0, Unlicense, 0BSD or BSD"):
# a crate that can be taken under these alone needs no line of its own.
COVERED = frozenset({
    "MIT", "MIT-0", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause", "BSD-3-Clause", "Zlib",
    "Unicode-3.0", "Unicode-DFS-2016", "Unlicense", "0BSD", "CC0-1.0",
})
# Further licenses a crate may be taken under when THIRD_PARTY.md names the crate and says why it is acceptable:
# permissive ones outside the group, the font licenses, and MPL-2.0, whose copyleft is per file and is met by
# using the crate's files unmodified, as published.
NAMED = COVERED | frozenset({"ISC", "BSL-1.0", "OFL-1.1", "Ubuntu-font-1.0", "MPL-2.0"})
# Crates whose purpose is talking to a network: HTTP clients and servers, TLS, sockets, and async runtimes' I/O.
NETWORK_CRATES = frozenset({
    "reqwest", "hyper", "hyper-util", "h2", "h3", "ureq", "isahc", "curl", "curl-sys", "attohttpc", "surf",
    "tungstenite", "tokio-tungstenite", "native-tls", "openssl", "openssl-sys", "rustls", "quinn", "tokio", "socket2",
    "websocket", "http-body", "trust-dns-resolver", "hickory-resolver",
})


def locked_crates(lock_file: Path) -> list[tuple[str, str]]:
    """The (name, version) of every crate from a registry in a `Cargo.lock`, sorted; path crates are this repository."""
    packages = tomllib.loads(lock_file.read_text(encoding="utf-8"))["package"]
    return sorted((package["name"], package["version"]) for package in packages if package.get("source", "").startswith("registry+"))


def registry_license(name: str, version: str) -> str:
    """A crate's license expression as its own `Cargo.toml` states it, from the local Cargo registry."""
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    for index in sorted((cargo_home / "registry" / "src").iterdir()):
        manifest = index / f"{name}-{version}" / "Cargo.toml"
        if manifest.is_file():
            package = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]
            if "license" in package:
                return package["license"]
            return f"see {package['license-file']}"
    raise SystemExit(f"{name} {version} is not in the local Cargo registry; run `cargo fetch --locked` first")


def satisfiable(expression: str, allowed: frozenset[str]) -> bool:
    """Whether an SPDX expression can be met using only the `allowed` licenses.

    `OR` takes any branch and `AND` needs every term; parentheses group, and the old `A/B` form means `A OR B`.
    """

    def evaluate(tokens: list[str]) -> bool:
        # Reduce parentheses first, innermost out.
        while "(" in tokens:
            close = tokens.index(")")
            open_at = max(index for index in range(close) if tokens[index] == "(")
            inner = evaluate(tokens[open_at + 1:close])
            tokens = tokens[:open_at] + ["MET" if inner else "UNMET"] + tokens[close + 1:]
        branches = " ".join(tokens).split(" OR ")
        return any(all(term.strip() in allowed | {"MET"} for term in branch.split(" AND ")) for branch in branches)

    tokens = [token.strip() for token in re.findall(r"\(|\)|[^()]+", expression.replace("/", " OR ")) if token.strip()]
    return evaluate(tokens)


def problems_with(record: dict[str, list[dict[str, str]]]) -> list[str]:
    """Each way the lock files, the record, and THIRD_PARTY.md disagree, as one line of text."""
    third_party = THIRD_PARTY.read_text(encoding="utf-8")
    problems: list[str] = []
    for lock_name in LOCK_FILES:
        locked = set(locked_crates(REPOSITORY / lock_name))
        recorded = {(crate["name"], crate["version"]): crate["license"] for crate in record[lock_name]}
        problems += [f"{lock_name}: {name} {version} is locked but not in dependencies.json" for name, version in sorted(locked - recorded.keys())]
        problems += [f"{lock_name}: {name} {version} is in dependencies.json but not locked" for name, version in sorted(recorded.keys() - locked)]
        for (name, version), expression in sorted(recorded.items()):
            if name in NETWORK_CRATES:
                problems.append(f"{lock_name}: {name} {version} is a network crate")
            if satisfiable(expression, COVERED):
                continue
            if not satisfiable(expression, NAMED):
                problems.append(f"{lock_name}: {name} {version} is under {expression!r}, which this check does not accept")
            elif not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", third_party):
                problems.append(f"{lock_name}: {name} {version} ({expression}) is not named in engine/THIRD_PARTY.md")
    return problems


def write_record() -> None:
    """Rewrite dependencies.json from the lock files and the local registry."""
    record = {
        lock_name: [{"name": name, "version": version, "license": registry_license(name, version)} for name, version in locked_crates(REPOSITORY / lock_name)]
        for lock_name in LOCK_FILES
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {RECORD.relative_to(REPOSITORY).as_posix()}: " + ", ".join(f"{len(crates)} crates in {name}" for name, crates in record.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite dependencies.json from the lock files and the local registry")
    arguments = parser.parse_args()
    if arguments.write:
        write_record()
        return 0

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    problems = problems_with(record)
    for problem in problems:
        print(problem)
    counts = ", ".join(f"{len(record[name])} crates in {name}" for name in LOCK_FILES)
    print(f"{counts}; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
