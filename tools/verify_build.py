"""Rebuild Litmus's programs from this checkout and compare their SHA-256 hashes with a release's, or with a second
build.

**Check a release's downloads.** Given the release's model files and its `SHA256SUMS`, with this checkout at the
release's commit:

    python tools/verify_build.py --model <folder> --sums <SHA256SUMS>
    python tools/verify_build.py --model <folder> --sums <SHA256SUMS> --twice   # also a second build from a clean state

rebuilds `litmus-app.exe` and `litmus.exe` exactly as `release/build_release.py` built the release, through the same
functions (`prepare_fixed_root`, `stage_model`, `build_programs`): from a fresh copy of `engine/` and `app/` at one
fixed path (`C:/litmus-build` on Windows, `/tmp/litmus-build` elsewhere), with the release's model files copied to
`<fixed path>/model` and built in (`--features embedded`), the pinned toolchain (`rust-toolchain.toml`), `--locked`
dependencies, and the flags in `release_rustflags`. It prints each program's hash beside the one `SHA256SUMS` lists
and says MATCH when both are the same, byte for byte. `--model` is a folder holding the release's
`model.safetensors`, `config.json`, and `tokenizer.json` (the github.io page serves them from its `app/` folder);
`--sums` defaults to `release/SHA256SUMS`. A `manifest.json` beside the sums file names the release's commit, which
the run compares with this checkout's.

**Check that a build is deterministic**, with no release (for developers):

    python tools/verify_build.py                  # the command line and the WebAssembly module, each built twice
    python tools/verify_build.py --artifacts app  # any of cli, wasm, and app (the desktop app)

- The command line (`cli`) is built from two copies of the source at different paths, so a path that leaked into
  the binary would change its hash.
- The app's builds (`wasm`, `app`) depend on where the source sits: the engine's crates are path dependencies outside
  the app's Cargo workspace, and Cargo hashes their absolute path into every symbol name, which changes the code's
  layout. They are built twice from fresh copies at the fixed path.

These builds carry no model, so they are compared with each other only; a release's files are checked with `--model`.

Needs Rust through rustup (it reads rust-toolchain.toml and installs the pinned compiler if missing; cargo is found in
`~/.cargo/bin` when it is not on PATH) and, for `wasm`, the wasm32-unknown-unknown target, which the same file names.
Each build takes 1.5 to 4 minutes on a desktop CPU, most of it link-time optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from dataclasses import dataclass, replace
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent.parent
CHECKSUMS = REPOSITORY / "release" / "SHA256SUMS"
# What a build reads: the two Cargo workspaces. The app's path dependencies reach into engine/.
SOURCE_FOLDERS = ("engine", "app")
SKIPPED = shutil.ignore_patterns("target", "dist", "__pycache__", ".pytest_cache", "node_modules")
# Fixed names the remapped paths take inside a binary, the same on every machine.
SOURCE_ROOT_NAME = "/litmus"
CARGO_HOME_NAME = "/cargo"
# Where a path-dependent artifact is built, the same on every machine of one platform.
FIXED_ROOT = Path("C:/litmus-build") if platform.system() == "Windows" else Path("/tmp/litmus-build")
# Left in the fixed root by this script and release/build_release.py; either may empty a folder that holds it.
FIXED_ROOT_MARKER = ".litmus-verify-build"
# A release's model, built into both programs from `<fixed root>/model`.
MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json")


@dataclass(frozen=True)
class Artifact:
    """One binary: how to build it, and where the build leaves it."""

    name: str
    workspace: str
    command: tuple[str, ...]
    output: str
    native: bool
    path_independent: bool
    extra_rustflags: tuple[str, ...] = ()


def executable(name: str) -> str:
    return f"{name}.exe" if platform.system() == "Windows" else name


def cargo_folder() -> Path | None:
    """The folder holding cargo: `$CARGO_HOME/bin` (rustup's default, `~/.cargo/bin`) when it holds cargo, else the
    folder of the cargo PATH finds; None when there is none."""
    rustup_bin = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "bin"
    if (rustup_bin / executable("cargo")).is_file():
        return rustup_bin
    found = shutil.which("cargo")
    return Path(found).parent if found else None


def put_cargo_on_path() -> Path:
    """Put cargo's folder first on this process's PATH, so `cargo` and rustup's proxies (which pick the pinned
    toolchain) run from it; return the folder."""
    folder = cargo_folder()
    if folder is None:
        raise SystemExit("cargo is not installed: install Rust with rustup (https://rustup.rs)")
    os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
    return folder


def artifacts() -> dict[str, Artifact]:
    """What the build check builds, without a model: the command line, the browser module, the desktop app."""
    return {
        "cli": Artifact(
            name=executable("litmus"),
            workspace="engine",
            command=("cargo", "build", "--release", "--locked", "-p", "litmus-cli"),
            output=f"release/{executable('litmus')}",
            native=True,
            path_independent=True,
        ),
        # The browser page's module as cargo writes it, before wasm-bindgen adds its JavaScript glue; the same
        # command as app/scripts/build_web.sh.
        "wasm": Artifact(
            name="litmus_app.wasm",
            workspace="app",
            command=("cargo", "rustc", "--profile", "web", "--target", "wasm32-unknown-unknown", "--lib", "--crate-type", "cdylib", "--locked"),
            output="wasm32-unknown-unknown/web/litmus_app.wasm",
            native=False,
            path_independent=False,
            extra_rustflags=("-C", "target-feature=+simd128"),
        ),
        "app": Artifact(
            name=executable("litmus-app"),
            workspace="app",
            command=("cargo", "build", "--release", "--locked"),
            output=f"release/{executable('litmus-app')}",
            native=True,
            path_independent=False,
        ),
    }


def program_artifacts() -> dict[str, Artifact]:
    """The two programs a release ships, by their file names, in the order they are built, each with the model built
    in (`--features embedded`, which reads `LITMUS_EMBED_DIR`)."""
    known = artifacts()
    embedded = ("--features", "embedded")
    return {known[key].name: replace(known[key], command=known[key].command + embedded) for key in ("cli", "app")}


def cargo_home() -> Path:
    return Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).resolve()


def release_rustflags(source_root: Path, native: bool) -> list[str]:
    """The flags that remove machine-specific bytes: the source and registry paths, and for a native Windows binary
    the link time and the PDB's path; a native Windows binary also links the C runtime statically."""
    flags = [
        f"--remap-path-prefix={source_root}={SOURCE_ROOT_NAME}",
        f"--remap-path-prefix={cargo_home()}={CARGO_HOME_NAME}",
    ]
    if native and platform.system() == "Windows":
        # The MSVC linker writes the link time into the PE header unless /Brepro, and the full PDB path unless told
        # to write its file name alone. The C runtime is linked in (crt-static), so the binaries run on a Windows
        # without the Visual C++ Redistributable (VCRUNTIME140.dll) installed.
        flags += ["-C", "link-arg=/Brepro", "-C", "link-arg=/PDBALTPATH:%_PDB%", "-C", "target-feature=+crt-static"]
    return flags


def source_date_epoch() -> str:
    """The commit time of HEAD when this is a git checkout, else 0; read by any build step that stamps a date."""
    found = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=REPOSITORY, capture_output=True, text=True, check=False)
    return found.stdout.strip() if found.returncode == 0 and found.stdout.strip() else "0"


def copy_source(origin: Path, destination: Path) -> None:
    """Copy the build's source folders under `origin`, and nothing built, into `destination`, replacing an earlier copy.

    The build output lives beside the source folders, in `destination/target`, so a kept folder reuses it.
    """
    for folder in SOURCE_FOLDERS:
        shutil.rmtree(destination / folder, ignore_errors=True)
        shutil.copytree(origin / folder, destination / folder, ignore=SKIPPED)


def prepare_fixed_root(fixed_root: Path, keep_build: bool = False) -> None:
    """Empty the fixed root (or, with `keep_build`, keep it and its build folder) and mark it as made here. A folder
    without the marker was not made by this script or release/build_release.py, and is refused."""
    if fixed_root.exists() and not (fixed_root / FIXED_ROOT_MARKER).is_file():
        raise SystemExit(f"{fixed_root} exists and was not made by tools/verify_build.py or release/build_release.py; remove it or pass another --fixed-root")
    if fixed_root.exists() and not keep_build:
        shutil.rmtree(fixed_root)
        assert not fixed_root.exists(), f"{fixed_root} could not be removed"
    fixed_root.mkdir(parents=True, exist_ok=True)
    (fixed_root / FIXED_ROOT_MARKER).write_text("made by tools/verify_build.py or release/build_release.py, which delete this folder\n", encoding="utf-8")


def stage_model(model_folder: Path, fixed_root: Path) -> Path:
    """Copy a release's three model files to `<fixed root>/model`, the one path both programs embed them from, and
    return that folder."""
    missing = [name for name in MODEL_FILES if not (model_folder / name).is_file()]
    if missing:
        raise SystemExit(f"{model_folder} lacks {', '.join(missing)}; the model folder holds {', '.join(MODEL_FILES)}")
    staged = fixed_root / "model"
    staged.mkdir(exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(model_folder / name, staged / name)
    return staged


def build(artifact: Artifact, source_root: Path, extra_environment: dict[str, str] | None = None) -> Path:
    """Build one artifact in one copy of the source and return the file it produced."""
    target_directory = source_root / "target"
    environment = dict(os.environ)
    environment.update({
        "CARGO_TARGET_DIR": str(target_directory),
        # The encoded form keeps a path with a space in it as one flag.
        "CARGO_ENCODED_RUSTFLAGS": "\x1f".join([*artifact.extra_rustflags, *release_rustflags(source_root, artifact.native)]),
        "CARGO_INCREMENTAL": "0",
        "SOURCE_DATE_EPOCH": source_date_epoch(),
        **(extra_environment or {}),
    })
    environment.pop("RUSTFLAGS", None)
    environment.pop("CARGO_BUILD_RUSTFLAGS", None)
    print(f"building {artifact.name} in {source_root}", flush=True)
    subprocess.run(artifact.command, cwd=source_root / artifact.workspace, env=environment, check=True)
    produced = target_directory / artifact.output
    assert produced.is_file(), f"the build did not write {produced}"
    return produced


def build_programs(fixed_root: Path, model: Path, out: Path) -> dict[str, Path]:
    """Build the programs a release ships in the fixed root, with the model staged by `stage_model` built in, and copy
    each into `out` under its shipped name before the next build starts. Returns the copies by name."""
    assert model == fixed_root / "model", f"the model is embedded from {fixed_root / 'model'}, the path every release uses"
    built = {}
    for name, artifact in program_artifacts().items():
        started = time.monotonic()
        produced = build(artifact, fixed_root, {"LITMUS_EMBED_DIR": str(model)})
        built[name] = out / name
        shutil.copy2(produced, built[name])
        print(f"built {name} in {time.monotonic() - started:.0f} s", flush=True)
    return built


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def published_hashes(sums: Path = CHECKSUMS) -> dict[str, str]:
    """Hashes by file name (as the line writes it, `*` dropped) from a file in `sha256sum` format; empty when absent."""
    if not sums.is_file():
        return {}
    hashes = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            hashes[name.lstrip("*")] = digest
    return hashes


def two_paths(artifact: Artifact, work: Path, snapshot: Path) -> tuple[str, str]:
    """The hashes of two builds from copies of `snapshot` at different paths and depths."""
    copies = (work / "first" / "litmus", work / "second" / "another place" / "litmus")
    for copy in copies:
        copy_source(snapshot, copy)
    first, second = (sha256(build(artifact, copy)) for copy in copies)
    return first, second


def one_path(artifact: Artifact, fixed_root: Path, work: Path, snapshot: Path) -> tuple[str, str]:
    """The hashes of two clean builds from fresh copies of `snapshot` at the same fixed path; nothing of the first
    build remains.

    Each build's artifact is copied into `work` as `first_<name>` and `second_<name>` before the next build starts.
    """
    hashes = []
    for order in ("first", "second"):
        prepare_fixed_root(fixed_root)
        copy_source(snapshot, fixed_root)
        produced = build(artifact, fixed_root)
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, work / f"{order}_{artifact.name}")
        hashes.append(sha256(produced))
    shutil.rmtree(fixed_root, ignore_errors=True)
    return hashes[0], hashes[1]


def checkout_commit() -> tuple[str, bool]:
    """This checkout's commit, and whether `engine/` or `app/` differ from it; ("", False) outside git."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, capture_output=True, text=True, check=False)
    if head.returncode != 0:
        return "", False
    changes = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_FOLDERS], cwd=REPOSITORY, capture_output=True, text=True, check=True)
    return head.stdout.strip(), bool(changes.stdout.strip())


def check_model(model: Path, published: dict[str, str]) -> list[str]:
    """Each model file against the hash the sums file lists for it, under its own name or under `site/app/` as a
    release's output lists it. Returns the mismatches."""
    problems = []
    for name in MODEL_FILES:
        listed = published.get(name) or published.get(f"site/app/{name}")
        actual = sha256(model / name)
        if listed is not None and listed != actual:
            problems.append(f"{model / name} has SHA-256 {actual}; the sums file lists {listed}")
    return problems


def verify_release(model: Path, sums: Path, fixed_root: Path, work: Path, twice: bool) -> bool:
    """Rebuild the shipped programs with the release's model, once or twice from a clean state, and compare each with
    the sums file. Prints the comparison; returns whether every build matches."""
    published = published_hashes(sums)
    names = list(program_artifacts())
    missing = [name for name in names if name not in published]
    if missing:
        raise SystemExit(f"{sums} lists no {' or '.join(missing)}; pass the SHA256SUMS of the release whose programs you downloaded")
    problems = check_model(model, published)
    if problems:
        print("The model files are not the release's, so the programs cannot match:\n" + "\n".join(problems))
        return False
    commit, changed = checkout_commit()
    manifest = sums.parent / "manifest.json"
    released = json.loads(manifest.read_text(encoding="utf-8"))["source"]["commit"] if manifest.is_file() else None
    print(f"this checkout: {commit or 'not a git checkout'}{' with changes in engine/ or app/' if changed else ''}")
    if released is not None:
        print(f"the release:   {released}{'' if released == commit else '  (a different commit: check out the release first, or the programs will differ)'}")
    # One snapshot of the source, taken now, feeds every build, so an edit made while the builds run reaches none.
    snapshot = work / "source"
    copy_source(REPOSITORY, snapshot)
    builds: list[dict[str, str]] = []
    for round_number in range(1, 3 if twice else 2):
        prepare_fixed_root(fixed_root)
        copy_source(snapshot, fixed_root)
        staged = stage_model(model, fixed_root)
        folder = work / f"build_{round_number}"
        folder.mkdir(parents=True, exist_ok=True)
        builds.append({name: sha256(path) for name, path in build_programs(fixed_root, staged, folder).items()})
    shutil.rmtree(fixed_root, ignore_errors=True)

    print()
    matched = True
    for name in names:
        print(name)
        for round_number, hashes in enumerate(builds, start=1):
            print(f"  {f'build {round_number}':<15} {hashes[name]}")
        print(f"  {sums.name:<15} {published[name]}")
        same = all(hashes[name] == published[name] for hashes in builds)
        matched &= same
        print(f"  {'MATCH' if same else 'DIFFERENT'}")
    print()
    if matched:
        print(f"MATCH: {' and '.join(names)}, built here from this source with this model, are byte for byte the files {sums} lists.")
    else:
        print(f"DIFFERENT: at least one program built here is not the file {sums} lists. Check that this checkout is the "
              "release's commit, that the toolchain is the pinned one, and that the build ran at the fixed path.")
    return matched


def verify_determinism(chosen: list[str], fixed_root: Path, work: Path) -> bool:
    """Build each chosen artifact twice and compare the two; prints the comparison and returns whether all agree."""
    # One snapshot of the source, taken now, feeds every build, so an edit made while the builds run reaches none.
    snapshot = work / "source"
    copy_source(REPOSITORY, snapshot)
    known = artifacts()
    rows = []
    for key in chosen:
        artifact = known[key]
        first, second = two_paths(artifact, work, snapshot) if artifact.path_independent else one_path(artifact, fixed_root, work, snapshot)
        where = "two different paths" if artifact.path_independent else f"one path, {fixed_root}"
        rows.append((artifact.name, where, first, second))
    print()
    for name, where, first, second in rows:
        print(f"{name} (built at {where})\n  first build   {first}\n  second build  {second}\n  {'same' if first == second else 'DIFFERENT'}")
    print("\nThese builds carry no model and are compared with each other only; to check a release's programs, pass --model and --sums.")
    return all(first == second for _, _, first, second in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, help="the release's model folder: rebuild litmus-app.exe and litmus.exe with it built in and compare with --sums")
    parser.add_argument("--sums", type=Path, default=CHECKSUMS, help=f"the release's SHA256SUMS (default {CHECKSUMS.relative_to(REPOSITORY).as_posix()})")
    parser.add_argument("--twice", action="store_true", help="with --model: build the programs a second time from a clean state as well")
    parser.add_argument("--artifacts", nargs="+", choices=("cli", "wasm", "app"), help="without --model: build these twice and compare (default cli wasm)")
    parser.add_argument("--keep", type=Path, help="build in this directory and keep every built file there")
    parser.add_argument("--fixed-root", type=Path, default=FIXED_ROOT, help=f"the one path the path-dependent builds use, emptied before each build (default {FIXED_ROOT})")
    arguments = parser.parse_args()
    if arguments.model is not None and arguments.artifacts is not None:
        parser.error("--model rebuilds a release's two programs; --artifacts is for the build check without a model")
    if arguments.model is None and arguments.twice:
        parser.error("--twice goes with --model")

    put_cargo_on_path()
    work = arguments.keep.resolve() if arguments.keep else Path(tempfile.mkdtemp(prefix="litmus_verify_"))
    work.mkdir(parents=True, exist_ok=True)
    fixed_root = arguments.fixed_root.resolve()
    if arguments.model is not None:
        passed = verify_release(arguments.model.resolve(), arguments.sums.resolve(), fixed_root, work, arguments.twice)
    else:
        passed = verify_determinism(arguments.artifacts or ["cli", "wasm"], fixed_root, work)
    if not arguments.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
