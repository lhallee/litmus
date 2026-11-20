"""Stages `build-windows`, `build-linux`, `verify`, and `reproduce`: the programs, the site, and the browser page, built from
the release commit with the release's model, and checked.

The builds are `release/build_release.py` and `release/build_linux.sh` themselves, which build through
`tools/verify_build.py`'s functions, so what a user rebuilds with `verify_build.py --model` is what was built. These
stages add nothing to a build; they run it, and `verify` reads its records.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from . import receipts
from .commands import StageError, run
from .documents import unfinished
from .stages import Context, Stage
from .versions import problems


def load_tool(repository: Path, name: str) -> ModuleType:
    """Import `tools/<name>.py` by path, as `release/build_release.py` does."""
    module_name = f"litmus_tools_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = repository / "tools" / f"{name}.py"
    if not path.is_file():
        raise StageError(f"{path} is missing")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def head(repository: Path) -> tuple[str | None, bool | None]:
    """The checkout's commit and whether engine/, app/, or site/ differ from it; (None, None) outside git."""
    found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False)
    if found.returncode != 0:
        return None, None
    status = run(["git", "status", "--porcelain", "--", "engine", "app", "site"], repository, capture=True)
    return found.stdout.strip(), bool(status.strip())


def ready_to_build(context: Context) -> None:
    """The checkout states the version, holds no unfilled slot, and (unless allowed) matches its commit."""
    found = problems(context.repository, context.plan.version)
    if found:
        raise StageError("the checkout does not state version " + context.plan.version + ":\n" + "\n".join(found) + "\nrun the `version` stage")
    left = unfinished(context)
    if left:
        raise StageError("the documents are not finished, so the release commit is not ready:\n" + "\n".join(left[:20]) + "\nrun the `docs` stage, then commit")
    commit, dirty = head(context.repository)
    if dirty and not context.options.allow_dirty:
        raise StageError("engine/, app/, or site/ differ from the commit: commit the release (the version, the documents, the site's data), then build, "
                         "so the build's manifest names the commit a user can check out. --allow-dirty builds anyway, for a trial that is never published")
    context.note("commit", commit)
    context.note("uncommitted_changes_in_source", dirty)


MEDIA_FILES = ("litmus_wide_site.mp4", "litmus_wide_poster.png")


def build_windows(context: Context) -> None:
    ready_to_build(context)
    media = context.work / "media"
    missing = [name for name in MEDIA_FILES if not (media / name).is_file()]
    if missing:
        raise StageError(f"{media} lacks {', '.join(missing)}: copy the ad's wide cut for the web and its poster there (demo/, `node render.mjs wide`; "
                         "release/RELEASING.md, The ad), since the home page shows the film")
    command = [context.python, "release/build_release.py", "--release", str(context.work / "model"), "--out", str(context.work / "build" / "windows"),
               "--release-version", context.plan.version, "--replace", "--media", str(media)]
    run(command, context.repository)


def build_linux(context: Context) -> None:
    ready_to_build(context)
    out = context.work / "build" / "linux"
    shutil.rmtree(out, ignore_errors=True)
    run(["bash", "release/build_linux.sh", str(context.work / "model"), str(out), context.plan.version], context.repository)


def adopter(platform_name: str) -> Callable[[Context, Path], None]:
    """The adoption of a build made on another machine: a folder holding `build/<platform>`, or the build folder itself."""
    def adopt(context: Context, source: Path) -> None:
        folder = source / "build" / platform_name if (source / "build" / platform_name).is_dir() else source
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            raise StageError(f"{folder} holds no manifest.json: pass the {platform_name} machine's work folder, or its build/{platform_name} folder")
        check_manifest(context, json.loads(manifest.read_text(encoding="utf-8")), folder)
        problems_found = check_sums(folder)
        if problems_found:
            raise StageError("the adopted build does not match its SHA256SUMS:\n" + "\n".join(problems_found))
        out = context.work / "build" / platform_name
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(folder, out)

    return adopt


def check_manifest(context: Context, manifest: dict, folder: Path) -> None:
    """A build's manifest names this release's model files and version."""
    for name, entry in manifest["model"]["files"].items():
        actual = receipts.file_digest(context.work / "model" / name)
        if entry["sha256"] != actual:
            raise StageError(f"{folder}: the build embeds {name} {entry['sha256'][:12]}, and this release's is {actual[:12]}")
    if (manifest.get("release") or {}).get("version") != context.plan.version:
        raise StageError(f"{folder}: the build is of version {(manifest.get('release') or {}).get('version')}, and this release is {context.plan.version}")


def check_sums(folder: Path) -> list[str]:
    """Each file `SHA256SUMS` lists that is missing or differs."""
    found = []
    for line in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = folder / name.lstrip("*")
        if not path.is_file():
            found.append(f"{folder.name}/{name}: listed and missing")
        elif receipts.file_digest(path) != digest:
            found.append(f"{folder.name}/{name}: its hash differs from SHA256SUMS")
    return found


def verify(context: Context) -> None:
    found = []
    commit, dirty = head(context.repository)
    for platform_name in ("windows", "linux"):
        folder = context.work / "build" / platform_name
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        check_manifest(context, manifest, folder)
        found += check_sums(folder)
        built_commit = manifest["source"]["commit"]
        if built_commit != commit:
            found.append(f"{platform_name}: built at commit {built_commit}, and the checkout is at {commit}")
        if manifest["source"]["uncommitted_changes_in_source"] and not context.options.allow_dirty:
            found.append(f"{platform_name}: built with uncommitted changes in engine/, app/, or site/")
    results = json.loads((context.work / "build" / "windows" / "site" / "assets" / "results.json").read_text(encoding="utf-8"))
    empty = [key for key, value in results["release"].items() if value in (None, "")]
    if empty:
        found.append("the built site's results.json has no value for release." + ", release.".join(empty))
    if results["release"]["version"] != context.plan.version:
        found.append(f"the built site states version {results['release']['version']}")
    problems_found = load_tool(context.repository, "verify_model").verify(context.work / "model", [context.repository / "release" / "SHA256SUMS"])
    found += problems_found
    if found:
        raise StageError("the builds do not check out:\n" + "\n".join(found))
    context.note("commit", commit)


def reproduce(context: Context) -> None:
    tools = load_tool(context.repository, "verify_build")
    fixed = Path(os.environ.get("LITMUS_FIXED_ROOT", str(tools.FIXED_ROOT)))
    folder = context.work / "build" / "windows"
    output = run([context.python, "tools/verify_build.py", "--model", str(context.work / "model"), "--sums", str(folder / "SHA256SUMS"), "--twice",
                  "--fixed-root", str(fixed)], context.repository, capture=True)
    (context.work / "build" / "reproduce.txt").write_text(output, encoding="utf-8", newline="\n")
    if "MATCH: " not in output:
        raise StageError("the programs rebuilt from this source are not the ones built; the output is in build/reproduce.txt")


def windows_outputs(context: Context) -> list[Path]:
    return [context.work / "build" / "windows"]


def linux_outputs(context: Context) -> list[Path]:
    return [context.work / "build" / "linux"]


def verify_outputs(context: Context) -> list[Path]:
    return [context.work / "build" / "windows" / "SHA256SUMS", context.work / "build" / "linux" / "SHA256SUMS"]


def reproduce_outputs(context: Context) -> list[Path]:
    return [context.work / "build" / "reproduce.txt"]


STAGES = (
    Stage("build-windows", ("docs",), "the Windows programs, the site, the browser page, and the contribute page", build_windows, windows_outputs, platform="windows", adopt=adopter("windows")),
    Stage("build-linux", ("docs",), "the Linux programs, in the pinned image", build_linux, linux_outputs, platform="linux", adopt=adopter("linux")),
    Stage("verify", ("build-windows", "build-linux"), "hashes, manifests, commit, and the release block of each build", verify, verify_outputs),
    Stage("reproduce", ("build-windows",), "rebuild both Windows programs twice and compare hashes", reproduce, reproduce_outputs, platform="windows"),
)
