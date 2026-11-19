"""Build every file a Litmus release ships, from this checkout and one trained release folder.

    python release/build_release.py --release DIR --out DIR

Writes into `--out`, which must be absent or empty (or an earlier output of this script, with `--replace`):

- `litmus-app.exe`: the desktop app with the release built in (`--features embedded`), so the one file runs offline
  with nothing beside it. A folder chosen in Settings, `LITMUS_MODEL`, or a `release` folder beside the program still
  wins over the built-in release, for developers.
- `litmus.exe`: the command line, with the release built in the same way (`--model` and `LITMUS_MODEL` still win).
- `site/`: the GitHub Pages folder: the eight pages, the stylesheet, the script, the mark, `assets/` (screenshots,
  figures, and the data files the pages read, with this build's sizes, hashes and version written into
  `assets/results.json`), and `app/` with the WebAssembly module, its JavaScript glue, the host page, the worker, the
  icon, and the release's three files. Every link is relative, so the folder serves from
  `https://lhallee.github.io/litmus/` or from a custom domain's root. The home page's film (`assets/media/`, the ad's
  wide cut for the web and its poster) is not in the repository; `--media` names the folder that holds the two files.
- `contribute/`: the page the contribute server serves as `LITMUS_CONTRIBUTE_SITE`: the same module built with the
  `contribute` feature, no worker and no model files.
- `manifest.json`: the model's version, the commit and a digest of the source built, the toolchain and tool versions,
  the build settings, and the sizes.
- `SHA256SUMS`: the SHA-256 of every other file, in `sha256sum` format, paths relative to `--out`.

The two programs are built by `tools/verify_build.py`'s own functions (`prepare_fixed_root`, `stage_model`,
`build_programs`), the ones its `--model` check runs, so a user rebuilding a release with that check runs the same
build: from a fresh copy of `engine/` and `app/` at one fixed path (`C:/litmus-build` on Windows), with the pinned
toolchain (`rust-toolchain.toml`), `--locked` dependencies, and its path-remapping flags. The release's files are
copied to `<fixed root>/model`, so the embedded files' path is the same on every machine. The browser modules are
built at the same path with the same flags. The fixed root is deleted at the end unless `--incremental` keeps it for
the next run.

Needs Rust through rustup (the pinned compiler and the wasm32-unknown-unknown target install on first use) and
`wasm-bindgen-cli` at the version `app/Cargo.toml` pins. A clean build takes 7.3 to 8.4 minutes on an 8-core desktop CPU.

`--native-only` builds the two programs alone (`litmus` and `litmus-app` on Linux and macOS), with `manifest.json`
and `SHA256SUMS`, and needs no wasm-bindgen: the build of another platform's programs, whose site and contribute
page come from the Windows build. `release/build_linux.sh` runs it in a pinned Linux image.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib

from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
RELEASE_FILES = ("model.safetensors", "config.json", "tokenizer.json")
RELEASE_FORMAT = "litmus-student"
RELEASE_FORMAT_VERSIONS = (1, 2, 3)
SITE_PAGES = ("index.html", "use.html", "download.html", "how.html", "reading.html", "benchmarks.html", "license.html", "privacy.html")
SITE_ASSETS = ("style.css", "site.js", "litmus_mark.svg")
# The home page's film, rendered by demo/ (`node render.mjs wide`), which is too large for the repository.
SITE_MEDIA = ("litmus_wide_site.mp4", "litmus_wide_poster.png")
# The browser page's own files, before the model's.
BROWSER_PAGE_FILES = ("index.html", "worker.js", "favicon.svg", "litmus_app.js", "litmus_app_bg.wasm")
MANIFEST_FORMAT = "litmus-release-manifest"
# Marks an output folder while the build writes it; removed just before the records, so a failed run shows.
INCOMPLETE = "BUILD_INCOMPLETE"
# A link or fetch that starts at the server's root breaks under the /litmus/ prefix of a project page.
ROOT_PATHS = re.compile(
    r"""(?:\b(?:href|src|action)\s*=\s*["']/(?!/))"""
    r"""|(?:url\(\s*["']?/(?!/))"""
    r"""|(?:\b(?:fetch|URL|import|Worker)\s*\(\s*["'`]/(?!/))"""
    r"""|(?:\bfrom\s+["']/(?!/))"""
)
LINK = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"'#?]*)""")


@dataclass(frozen=True)
class Release:
    """What the build needs to know of a release folder."""

    folder: Path
    version: str
    format_version: int
    size: str
    attention: str
    output: str
    license: str
    checkpoint_step: int | None
    weights_hash_stated: bool
    # The version of the release this build belongs to (`--release-version`, which release/cut.py passes), when it names one.
    release_version: str | None = None


@dataclass(frozen=True)
class Tools:
    """The programs the build runs, and their versions."""

    cargo_bin: Path
    # None in a --native-only build, which builds no browser module.
    wasm_bindgen: Path | None
    wasm_bindgen_version: str | None


def load_tool(name: str) -> ModuleType:
    """Import `tools/<name>.py` by path; the release build reuses its build settings and checks."""
    path = REPOSITORY / "tools" / f"{name}.py"
    if not path.is_file():
        raise SystemExit(f"{path} is missing; the release build reuses it")
    spec = importlib.util.spec_from_file_location(f"litmus_tools_{name}", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: dataclasses look their module up by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(command: list[str], cwd: Path, environment: dict[str, str] | None = None, stdin: str | None = None) -> str:
    """Run a command, fail loudly on a nonzero exit, and return its standard output."""
    finished = subprocess.run(command, cwd=cwd, env=environment, input=stdin, capture_output=True, text=True, encoding="utf-8")
    if finished.returncode != 0:
        raise SystemExit(f"{' '.join(command)} exited with {finished.returncode}:\n{finished.stderr.strip()}")
    return finished.stdout


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_release(folder: Path, verify_model: ModuleType, allow_placeholder: bool, release_version: str | None = None) -> Release:
    """Check a release folder (its three files, its format, the hashes it states) and read what the build records."""
    missing = [name for name in RELEASE_FILES if not (folder / name).is_file()]
    if missing:
        raise SystemExit(f"{folder} lacks {', '.join(missing)}; a release folder holds {', '.join(RELEASE_FILES)}")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != RELEASE_FORMAT or config.get("format_version") not in RELEASE_FORMAT_VERSIONS:
        raise SystemExit(f"{folder}: format {config.get('format')!r} version {config.get('format_version')!r}, "
                         f"the engine reads {RELEASE_FORMAT} versions {RELEASE_FORMAT_VERSIONS}")
    provenance = config.get("provenance") or {}
    if provenance.get("kind") == "random" and not allow_placeholder:
        raise SystemExit(f"{folder} is the random-weights placeholder; pass --allow-placeholder to build it for a trial build")
    # The engine checks a folder's tokenizer against config.json, but never the embedded copy, so the build checks
    # every hash the release states (config.json's fields and a SHA256SUMS beside the files) before embedding.
    stated = verify_model.stated_hashes(folder, [])
    for name, claims in sorted(stated.items()):
        actual = verify_model.sha256(folder / name)
        wrong = [f"{source} states {digest}" for digest, source in claims if digest != actual]
        if wrong:
            raise SystemExit(f"{folder / name} has sha256 {actual}, but {'; '.join(wrong)}")
    if "tokenizer.json" not in stated:
        raise SystemExit(f"{folder}/config.json states no tokenizer.sha256, so the tokenizer cannot be checked")
    model_bytes = (folder / "model.safetensors").read_bytes()
    config_bytes = (folder / "config.json").read_bytes()
    # As the engine names a release (litmus-files cache::release_version): the provenance's version, else the
    # first 8 bytes of SHA-256 over the weights then the config.
    version = provenance.get("version") or f"sha256:{sha256_bytes(model_bytes + config_bytes)[:16]}"
    model = config["model"]
    return Release(
        folder=folder,
        version=version,
        format_version=config["format_version"],
        size=f"d{model['width']} L{model['depth']}",
        attention=model.get("attention", "float"),
        output=model.get("output", "float"),
        license=config.get("license", ""),
        checkpoint_step=provenance.get("step"),
        weights_hash_stated="model.safetensors" in stated,
        release_version=release_version,
    )


def find_tools(verify_build: ModuleType, app_manifest: Path, native_only: bool) -> Tools:
    """Cargo (put on PATH for this process, so rustup's proxies pick the pinned toolchain) and, unless `native_only`,
    wasm-bindgen at the version the app pins."""
    cargo_bin = verify_build.put_cargo_on_path()
    if native_only:
        return Tools(cargo_bin=cargo_bin, wasm_bindgen=None, wasm_bindgen_version=None)
    dependencies = tomllib.loads(app_manifest.read_text(encoding="utf-8"))["target"]['cfg(target_arch = "wasm32")']["dependencies"]
    pinned = str(dependencies["wasm-bindgen"]).lstrip("=")
    named = os.environ.get("WASM_BINDGEN") or shutil.which("wasm-bindgen")
    install = f"cargo install wasm-bindgen-cli --version {pinned} --locked"
    if named is None:
        raise SystemExit(f"wasm-bindgen is not installed: {install} (or set WASM_BINDGEN to one)")
    version = run([named, "--version"], REPOSITORY).split()[-1]
    if version != pinned:
        raise SystemExit(f"{named} is wasm-bindgen {version}; the app pins {pinned}: {install}")
    return Tools(cargo_bin=cargo_bin, wasm_bindgen=Path(named), wasm_bindgen_version=version)


def toolchain(fixed_root: Path) -> dict[str, str | None]:
    """The compiler, Cargo, and on Windows the MSVC tools that link, as the pinned toolchain reports them."""
    channels = {tomllib.loads((REPOSITORY / folder / "rust-toolchain.toml").read_text(encoding="utf-8"))["toolchain"]["channel"]
                for folder in ("engine", "app")}
    assert len(channels) == 1, f"engine/ and app/ pin different toolchains: {sorted(channels)}"
    details = dict(line.split(": ", 1) for line in run(["rustc", "-vV"], fixed_root / "app").splitlines() if ": " in line)
    found = {
        "channel": channels.pop(),
        "rustc": details["release"],
        "rustc_commit": details["commit-hash"],
        "llvm": details["LLVM version"],
        "host": details["host"],
        "cargo": run(["cargo", "-V"], fixed_root / "app").strip(),
        "msvc_tools": None,
    }
    # RUSTUP_TOOLCHAIN (as release/build_linux.sh sets it) overrides the files: it must name the same compiler.
    assert found["rustc"] == found["channel"], f"rustc is {found['rustc']}; rust-toolchain.toml pins {found['channel']}"
    # Windows matches environment variable names without regard to case.
    vswhere = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if platform.system() == "Windows" and vswhere.is_file():
        installation = run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"], REPOSITORY).strip()
        version_file = Path(installation) / "VC" / "Auxiliary" / "Build" / "Microsoft.VCToolsVersion.default.txt"
        if installation and version_file.is_file():
            found["msvc_tools"] = version_file.read_text(encoding="utf-8").strip()
    return found


def snapshot(verify_build: ModuleType, fixed_root: Path, release: Release) -> Path:
    """Copy the source the build reads (engine/, app/, site/) and the release's files into the fixed root, once, so
    an edit made while the build runs reaches none of it. Returns the embedded release's folder."""
    verify_build.copy_source(REPOSITORY, fixed_root)
    shutil.rmtree(fixed_root / "site", ignore_errors=True)
    shutil.copytree(REPOSITORY / "site", fixed_root / "site")
    return verify_build.stage_model(release.folder, fixed_root)


def source_digest(fixed_root: Path) -> str:
    """SHA-256 over every source file built (path and content), so a rebuild can show it started from the same tree."""
    digest = hashlib.sha256()
    for folder in ("engine", "app", "site"):
        for path in sorted((fixed_root / folder).rglob("*")):
            relative = path.relative_to(fixed_root)
            if path.is_file() and "target" not in relative.parts:
                digest.update(relative.as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def git_state() -> dict[str, str | bool | None]:
    """The commit and whether the built folders differ from it; outside a git checkout (a copied tree), the commit is
    None and the manifest says why, and the source digest alone names the source."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, capture_output=True, text=True, encoding="utf-8")
    if head.returncode != 0:
        return {"commit": None, "uncommitted_changes_in_source": None, "git": (head.stderr.strip().splitlines() or ["not a git checkout"])[0]}
    changes = run(["git", "status", "--porcelain", "--", "engine", "app", "site"], REPOSITORY).strip()
    return {"commit": head.stdout.strip(), "uncommitted_changes_in_source": bool(changes)}


def build_artifacts(verify_build: ModuleType, fixed_root: Path, model: Path, out: Path, tools: Tools) -> dict[str, str]:
    """Build the command line and the app with the release built in (`verify_build.build_programs`, the build its
    `--model` check repeats), and the browser module twice (the private page, then the contribute feature), each
    copied out or hashed before the next build can overwrite it. Returns the SHA-256 of each browser module as cargo
    wrote it, before wasm-bindgen."""
    known = verify_build.artifacts()
    module_hashes: dict[str, str] = {}
    verify_build.build_programs(fixed_root, model, out)
    if tools.wasm_bindgen is None:
        return module_hashes
    for folder, artifact in (
        (out / "site" / "app", known["wasm"]),
        (out / "contribute", replace(known["wasm"], name="litmus_app_contribute.wasm", command=known["wasm"].command + ("--features", "contribute"))),
    ):
        started = time.monotonic()
        produced = verify_build.build(artifact, fixed_root)
        folder.mkdir(parents=True, exist_ok=True)
        run([str(tools.wasm_bindgen), "--target", "web", "--no-typescript", "--out-dir", str(folder), "--out-name", "litmus_app", str(produced)], fixed_root)
        module_hashes[f"{folder.name}_module"] = sha256_file(produced)
        print(f"built {folder.relative_to(out).as_posix()}/litmus_app_bg.wasm in {time.monotonic() - started:.0f} s", flush=True)
    return module_hashes


def assemble_site(fixed_root: Path, out: Path, release: Release, media: Path) -> None:
    """The GitHub Pages folder: the pages from site/, the film from `media` into assets/media, and in app/ the host
    page, worker, icon and model files beside the module wasm-bindgen wrote there."""
    source = fixed_root / "site"
    pages = sorted(path.name for path in source.glob("*.html"))
    if pages != sorted(SITE_PAGES):
        raise SystemExit(f"site/ holds the pages {pages}, the build ships {sorted(SITE_PAGES)}; update SITE_PAGES")
    missing = [name for name in SITE_MEDIA if not (media / name).is_file()]
    if missing:
        raise SystemExit(f"{media} lacks {', '.join(missing)}: --media names the folder of the ad's wide cut for the web and its poster")
    site = out / "site"
    for name in (*SITE_PAGES, *SITE_ASSETS):
        shutil.copy2(source / name, site / name)
    shutil.copytree(source / "assets", site / "assets")
    (site / "assets" / "media").mkdir(exist_ok=True)
    for name in SITE_MEDIA:
        shutil.copy2(media / name, site / "assets" / "media" / name)
    # GitHub Pages would otherwise run Jekyll over the folder.
    (site / ".nojekyll").write_bytes(b"")
    app = site / "app"
    shutil.copy2(source / "app" / "index.html", app / "index.html")
    shutil.copy2(source / "app" / "worker.js", app / "worker.js")
    shutil.copy2(fixed_root / "app" / "assets" / "litmus_icon.svg", app / "favicon.svg")
    for name in RELEASE_FILES:
        shutil.copy2(release.folder / name, app / name)
    record_release_facts(out, release)


def record_release_facts(out: Path, release: Release) -> None:
    """Write this build's version, the programs' sizes and SHA-256, and the browser page's sizes into the built site's
    `assets/results.json`, so the download page states the files this build made. `site/` itself is left as it is."""
    path = out / "site" / "assets" / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    app = out / "site" / "app"
    results["release"].update({
        "version": release.release_version or release.version,
        "app_exe_bytes": (out / "litmus-app.exe").stat().st_size,
        "app_exe_sha256": sha256_file(out / "litmus-app.exe"),
        "cli_exe_bytes": (out / "litmus.exe").stat().st_size,
        "cli_exe_sha256": sha256_file(out / "litmus.exe"),
        "browser_model_bytes": sum((app / name).stat().st_size for name in RELEASE_FILES),
        "browser_page_bytes": sum((app / name).stat().st_size for name in BROWSER_PAGE_FILES),
    })
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def assemble_contribute(fixed_root: Path, out: Path) -> None:
    """The contribute server's page: the host page marked as the contribute build, and the icon."""
    contribute = out / "contribute"
    page = (fixed_root / "site" / "app" / "index.html").read_text(encoding="utf-8")
    private, marked = '<html lang="en" data-build="private">', '<html lang="en" data-build="contribute">'
    assert page.count(private) == 1, f"site/app/index.html opens with {private} once"
    (contribute / "index.html").write_text(page.replace(private, marked), encoding="utf-8", newline="")
    shutil.copy2(fixed_root / "app" / "assets" / "litmus_icon.svg", contribute / "favicon.svg")


def check_paths(out: Path) -> None:
    """Fail on a link or fetch from the server's root, and on a relative link in a page to a file the folder lacks."""
    problems = []
    for path in sorted(out.rglob("*")):
        if path.suffix not in (".html", ".css", ".js"):
            continue
        text = path.read_text(encoding="utf-8")
        problems += [f"{path.relative_to(out).as_posix()}: a root path, {match.group(0)!r}" for match in ROOT_PATHS.finditer(text)]
        if path.suffix != ".html":
            continue
        for target in LINK.findall(text):
            if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("//"):
                continue
            resolved = (path.parent / target).resolve()
            if not (resolved.is_file() or (resolved / "index.html").is_file()):
                problems.append(f"{path.relative_to(out).as_posix()}: links to {target}, which the folder lacks")
    if problems:
        raise SystemExit("the site would break when served:\n" + "\n".join(problems))
    for folder in (out / "site" / "app", out / "contribute"):
        names = {path.name for path in folder.iterdir()}
        expected = {"index.html", "favicon.svg", "litmus_app.js", "litmus_app_bg.wasm"}
        expected |= {"worker.js", *RELEASE_FILES} if folder.name == "app" else set()
        assert names == expected, f"{folder} holds {sorted(names)}, expected {sorted(expected)}"


def smoke_test(out: Path, release: Release, cli_name: str) -> None:
    """The command line's built-in release reads a text exactly as the release folder does, and names the version."""
    environment = {key: value for key, value in os.environ.items() if key not in ("LITMUS_MODEL", "LITMUS_TOKENIZER")}
    cli = str(out / cli_name)
    report = run([cli, "info"], out, environment)
    assert "model: built in" in report, f"litmus info does not report its built-in release:\n{report}"
    text = "The road was closed for three weeks, so she sold yesterday's bread at the far end of the detour."
    built_in = run([cli, "score", "--json", "--no-cache", "-"], out, environment, text)
    from_folder = run([cli, "score", "--json", "--no-cache", "--model", str(release.folder), "-"], out, environment, text)
    assert '"reading"' in built_in, f"litmus score --json printed no reading:\n{built_in}"
    assert built_in == from_folder, "the built-in release reads a text differently from the release folder"
    print(f"{cli_name} reads with its built-in release {release.version}, the same as the folder", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shipped_files(out: Path) -> list[Path]:
    return sorted(path for path in out.rglob("*") if path.is_file() and path.name != "SHA256SUMS")


def sizes(out: Path, programs: tuple[str, ...], native_only: bool) -> dict[str, int]:
    """Bytes of each download a user makes: the two programs, the browser page before and after the model, and
    the page's module gzipped as a server compresses it (the programs alone in a --native-only build)."""
    if native_only:
        return {name: (out / name).stat().st_size for name in programs}
    app = out / "site" / "app"
    page = BROWSER_PAGE_FILES
    return {
        "litmus-app.exe": (out / "litmus-app.exe").stat().st_size,
        "litmus.exe": (out / "litmus.exe").stat().st_size,
        "browser_page": sum((app / name).stat().st_size for name in page),
        "browser_page_gzip": sum(len(gzip.compress((app / name).read_bytes(), 9, mtime=0)) for name in page),
        "browser_model": sum((app / name).stat().st_size for name in RELEASE_FILES),
        "browser_model_gzip": sum(len(gzip.compress((app / name).read_bytes(), 9, mtime=0)) for name in RELEASE_FILES),
        "site_folder": sum(path.stat().st_size for path in (out / "site").rglob("*") if path.is_file()),
        "contribute_folder": sum(path.stat().st_size for path in (out / "contribute").rglob("*") if path.is_file()),
    }


def write_records(out: Path, release: Release, tools: Tools, toolchain_found: dict[str, str | None], source: dict[str, Any],
                  verify_build: ModuleType, fixed_root: Path, module_hashes: dict[str, str]) -> None:
    """manifest.json, then SHA256SUMS over every file including the manifest."""
    (out / INCOMPLETE).unlink()
    native_only = tools.wasm_bindgen is None
    browser = {} if native_only else {
        "rustflags_wasm": ["-C", "target-feature=+simd128", *verify_build.release_rustflags(fixed_root, False)],
        # The browser modules as cargo wrote them, before wasm-bindgen: tools/verify_build.py --artifacts wasm
        # rebuilds the private page's and compares.
        "modules_before_wasm_bindgen_sha256": dict(sorted(module_hashes.items())),
    }
    manifest = {
        "format": MANIFEST_FORMAT,
        "model": {
            "version": release.version,
            "format_version": release.format_version,
            "size": release.size,
            "attention": release.attention,
            "output": release.output,
            "license": release.license,
            "checkpoint_step": release.checkpoint_step,
            "weights_hash_stated_in_config": release.weights_hash_stated,
            "files": {name: {"bytes": (release.folder / name).stat().st_size, "sha256": sha256_file(release.folder / name)} for name in RELEASE_FILES},
        },
        "release": {"version": release.release_version},
        "source": source,
        "toolchain": {**toolchain_found, "wasm_bindgen": tools.wasm_bindgen_version, "python": platform.python_version()},
        "build": {
            "fixed_root": fixed_root.as_posix(),
            "embedded_release": (fixed_root / "model").as_posix(),
            "cargo_flags": ["--locked"],
            "rustflags_native": verify_build.release_rustflags(fixed_root, True),
            "source_date_epoch": verify_build.source_date_epoch(),
            "native_only": native_only,
            **browser,
        },
        "sizes": sizes(out, tuple(verify_build.program_artifacts()), native_only),
        "files": {path.relative_to(out).as_posix(): path.stat().st_size for path in shipped_files(out)},
    }
    # LF line ends on every system, so `sha256sum -c` reads the names whole.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    lines = [f"{sha256_file(path)}  {path.relative_to(out).as_posix()}" for path in shipped_files(out)]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def prepare_out(out: Path, replace_earlier: bool, native_only: bool) -> None:
    """An empty output folder; with `replace_earlier`, an earlier output of this script is deleted first."""
    if out.exists() and any(out.iterdir()):
        manifest = out / "manifest.json"
        finished = manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("format") == MANIFEST_FORMAT
        if not (replace_earlier and (finished or (out / INCOMPLETE).is_file())):
            raise SystemExit(f"{out} is not empty; choose an empty folder, or pass --replace for an earlier output of this script")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if not native_only:
        (out / "site" / "app").mkdir(parents=True, exist_ok=True)
        (out / "contribute").mkdir(parents=True, exist_ok=True)
    (out / INCOMPLETE).write_text("release/build_release.py is writing this folder, or stopped before it finished\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, required=True, help="the trained release folder: model.safetensors, config.json, tokenizer.json")
    parser.add_argument("--out", type=Path, required=True, help="where the shipped files go; absent or empty")
    parser.add_argument("--replace", action="store_true", help="delete an earlier output of this script in --out first")
    parser.add_argument("--fixed-root", type=Path, help="the one path the binaries are built at (default tools/verify_build.py's)")
    parser.add_argument("--incremental", action="store_true", help="keep the fixed root's build folder between runs (faster; not a clean build)")
    parser.add_argument("--release-version", help="the release's version (MAJOR.MINOR.PATCH): the site and manifest.json state it beside the model's own")
    parser.add_argument("--allow-placeholder", action="store_true", help="build a random-weights placeholder release, for a trial build")
    parser.add_argument("--native-only", action="store_true", help="build the two programs alone, with no site or contribute page (Linux, macOS)")
    parser.add_argument("--media", type=Path, help=f"the folder holding the home page's film, {' and '.join(SITE_MEDIA)} (needed unless --native-only)")
    arguments = parser.parse_args()
    if not arguments.native_only and arguments.media is None:
        parser.error("--media is needed for the site: the folder of the ad's wide cut for the web and its poster")

    started = time.monotonic()
    verify_build = load_tool("verify_build")
    verify_model = load_tool("verify_model")
    release = read_release(arguments.release.resolve(), verify_model, arguments.allow_placeholder, arguments.release_version)
    tools = find_tools(verify_build, REPOSITORY / "app" / "Cargo.toml", arguments.native_only)
    fixed_root = (arguments.fixed_root or verify_build.FIXED_ROOT).resolve()
    out = arguments.out.resolve()
    assert not out.is_relative_to(REPOSITORY), f"{out} is inside the repository; build output stays out of the tree"
    prepare_out(out, arguments.replace, arguments.native_only)
    verify_build.prepare_fixed_root(fixed_root, keep_build=arguments.incremental)
    embedded = snapshot(verify_build, fixed_root, release)
    source = {**git_state(), "digest_engine_app_site": source_digest(fixed_root)}
    toolchain_found = toolchain(fixed_root)
    print(f"building release {release.version} ({release.size}, format {release.format_version}) with Rust {toolchain_found['rustc']} at {fixed_root}", flush=True)

    module_hashes = build_artifacts(verify_build, fixed_root, embedded, out, tools)
    if not arguments.native_only:
        assemble_site(fixed_root, out, release, arguments.media.resolve())
        assemble_contribute(fixed_root, out)
        check_paths(out)
    smoke_test(out, release, verify_build.executable("litmus"))
    write_records(out, release, tools, toolchain_found, source, verify_build, fixed_root, module_hashes)
    if not arguments.incremental:
        shutil.rmtree(fixed_root)

    print()
    for name, size in sizes(out, tuple(verify_build.program_artifacts()), arguments.native_only).items():
        print(f"{name:20} {size / 1e6:8.1f} MB")
    if not release.weights_hash_stated:
        print("note: the release's config.json states no weights.sha256; SHA256SUMS and manifest.json carry the hash")
    print(f"wrote {out} in {(time.monotonic() - started) / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
