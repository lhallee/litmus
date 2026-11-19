"""Stage `version`: the release version in every place the checkout states one.

The places are the Cargo manifests and lock files of the engine and the app (the programs print the Cargo version) and
the `pyproject.toml` of each Python package. The Python wheel takes the engine's version. The model card, the site, the
manifest, and the results state the version from the plan through the documents and the build, not from a file here.
Edits keep each file's line endings and change one line.
"""

from __future__ import annotations

import json
import re
import tomllib

from dataclasses import dataclass
from pathlib import Path

from .commands import StageError, run
from .stages import Context, Stage


@dataclass(frozen=True)
class Manifest:
    """A file with a `version = "..."` line in one table."""

    path: str
    table: tuple[str, ...]


@dataclass(frozen=True)
class Lock:
    """A Cargo lock file and the packages of this repository it lists (path packages, with no source)."""

    path: str
    crates: tuple[str, ...]


MANIFESTS = (
    Manifest("engine/Cargo.toml", ("workspace", "package")),
    Manifest("app/Cargo.toml", ("package",)),
    Manifest("corpus/pyproject.toml", ("project",)),
    Manifest("teacher/pyproject.toml", ("project",)),
    Manifest("student/pyproject.toml", ("project",)),
    Manifest("benchmarks/pyproject.toml", ("project",)),
    Manifest("contribute/pyproject.toml", ("project",)),
)
# The agent plugin's manifest states the release version too (a JSON file with one "version" key).
PLUGINS = (".claude-plugin/plugin.json",)
LOCKS = (
    Lock("engine/Cargo.lock", ("litmus-cli", "litmus-engine", "litmus-files", "litmus-py")),
    Lock("app/Cargo.lock", ("litmus-app", "litmus-engine", "litmus-files")),
)


def table_span(raw: bytes, table: tuple[str, ...]) -> tuple[int, int]:
    """Where a TOML table's lines start and end in the file's bytes."""
    header = re.search(rb"(?m)^\[" + re.escape(".".join(table).encode()) + rb"\][ \t]*\r?$", raw)
    if header is None:
        raise StageError(f"no [{'.'.join(table)}] table found")
    following = re.search(rb"(?m)^\[", raw[header.end():])
    return header.end(), header.end() + (following.start() if following else len(raw) - header.end())


def stated(repository: Path, manifest: Manifest) -> str:
    value = tomllib.loads((repository / manifest.path).read_text(encoding="utf-8"))
    for name in manifest.table:
        value = value[name]
    return value["version"]


def locked(repository: Path, lock: Lock) -> dict[str, str]:
    packages = tomllib.loads((repository / lock.path).read_text(encoding="utf-8"))["package"]
    found = {package["name"]: package["version"] for package in packages if package["name"] in lock.crates and "source" not in package}
    missing = set(lock.crates) - found.keys()
    if missing:
        raise StageError(f"{lock.path} lists no path package named {', '.join(sorted(missing))}; update LOCKS in release/litmus_release/versions.py")
    return found


def problems(repository: Path, version: str) -> list[str]:
    """Each place that does not state `version`."""
    found = [f"{manifest.path} states {stated(repository, manifest)}" for manifest in MANIFESTS if stated(repository, manifest) != version]
    for plugin in PLUGINS:
        value = json.loads((repository / plugin).read_text(encoding="utf-8"))["version"]
        found += [f"{plugin} states {value}"] if value != version else []
    for lock in LOCKS:
        found += [f"{lock.path} locks {name} at {value}" for name, value in locked(repository, lock).items() if value != version]
    return found


def rewrite(path: Path, edit: tuple[re.Pattern[bytes], bytes], span: tuple[int, int] | None = None) -> bool:
    """Replace the first match of a pattern (within `span`) in a file's bytes; returns whether the file changed."""
    raw = path.read_bytes()
    start, end = span or (0, len(raw))
    changed, count = edit[0].subn(edit[1], raw[start:end], count=1)
    if count != 1:
        raise StageError(f"{path} has no line for {edit[0].pattern!r} to change")
    path.write_bytes(raw[:start] + changed + raw[end:])
    return changed != raw[start:end]


def apply(repository: Path, version: str) -> list[Path]:
    """Set `version` everywhere; returns the files that changed."""
    changed = []
    replacement = rb'\1"' + version.encode() + rb'"'
    for manifest in MANIFESTS:
        path = repository / manifest.path
        if rewrite(path, (re.compile(rb'(?m)^(version[ \t]*=[ \t]*)"[^"]*"'), replacement), table_span(path.read_bytes(), manifest.table)):
            changed.append(path)
    for plugin in PLUGINS:
        path = repository / plugin
        if rewrite(path, (re.compile(rb'("version"[ 	]*:[ 	]*)"[^"]*"'), replacement)):
            changed.append(path)
    for lock in LOCKS:
        path = repository / lock.path
        for crate in lock.crates:
            pattern = re.compile(rb'(name = "' + re.escape(crate.encode()) + rb'"\r?\nversion = )"[^"]*"')
            if rewrite(path, (pattern, replacement)) and path not in changed:
                changed.append(path)
    return changed


def set_version(context: Context) -> None:
    repository = context.repository
    changed = apply(repository, context.plan.version)
    left = problems(repository, context.plan.version)
    if left:
        raise StageError("the version is still not " + context.plan.version + ":\n" + "\n".join(left))
    if changed:
        run([context.python, "tools/security/check_dependencies.py"], repository)
    context.note("changed", [path.relative_to(repository).as_posix() for path in changed])


def outputs(context: Context) -> list[Path]:
    return [context.repository / manifest.path for manifest in MANIFESTS] + [context.repository / plugin for plugin in PLUGINS] + [context.repository / lock.path for lock in LOCKS]


STAGES = (Stage("version", (), "the release version in the Cargo and Python manifests", set_version, outputs),)
