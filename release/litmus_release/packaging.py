"""Stage `assemble`: what each destination receives, laid out in the work folder, and nothing sent.

`publish/github/` holds the release's assets under the names the site and the install scripts expect, one `SHA256SUMS`
that names every attached file, and `NOTES.md`; `publish/pages/` is the `gh-pages` branch's content; `publish/hub/`
is the public model repository's content (the model's three files, the model card as its README, and the license).
`publish --dry-run` prints exactly these.
"""

from __future__ import annotations

import json
import shutil
import tomllib

from pathlib import Path

from . import receipts
from .commands import StageError
from .stages import Context, Stage


MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json")
# Each platform's programs: (the program's name in its build folder, its asset key in targets.toml `[programs.assets]`).
PLATFORM_PROGRAMS = {
    "windows": (("litmus-app.exe", "windows_app"), ("litmus.exe", "windows_cli")),
    "linux": (("litmus-app", "linux_app"), ("litmus", "linux_cli")),
}
# Platforms whose programs the `macos` stage attaches once the release is public. `targets.toml` lists them so that the
# site links their programs; `assemble` lays out none of theirs.
ATTACHED_AFTER_PUBLISHING = ("macos",)


def targets(context: Context) -> dict:
    return tomllib.loads((context.repository / "release" / "targets.toml").read_text(encoding="utf-8"))


def github_assets(context: Context) -> dict[str, Path]:
    """Asset name to source file for every file the release attaches; a listed platform with a missing program is an error."""
    settings = targets(context)
    names = settings["programs"]["assets"]
    assets: dict[str, Path] = {}
    for platform_name in settings["programs"]["platforms"]:
        if platform_name in ATTACHED_AFTER_PUBLISHING:
            continue
        if platform_name not in PLATFORM_PROGRAMS:
            raise StageError(f"targets.toml lists the platform {platform_name!r}; the release builds {', '.join(PLATFORM_PROGRAMS)} and attaches {', '.join(ATTACHED_AFTER_PUBLISHING)} after publishing")
        for program, key in PLATFORM_PROGRAMS[platform_name]:
            source = context.work / "build" / platform_name / program
            if not source.is_file():
                raise StageError(f"{source} is missing; the {platform_name} build did not write {program}")
            assets[names[key]] = source
    for name in MODEL_FILES:
        assets[name] = context.work / "model" / name
    assets["manifest.json"] = context.work / "build" / "windows" / "manifest.json"
    assets["manifest-linux.json"] = context.work / "build" / "linux" / "manifest.json"
    assets["results.json"] = context.work / "build" / "windows" / "site" / "assets" / "results.json"
    return assets


def sums_text(files: dict[str, Path]) -> str:
    """`SHA256SUMS` in `sha256sum` format: one line per attached file, named as attached."""
    return "".join(f"{receipts.file_digest(path)}  {name}\n" for name, path in sorted(files.items()))


def assemble(context: Context) -> None:
    out = context.work / "publish"
    shutil.rmtree(out, ignore_errors=True)
    github, pages, hub = out / "github", out / "pages", out / "hub"
    for folder in (github, hub):
        folder.mkdir(parents=True)
    assets = github_assets(context)
    for name, source in assets.items():
        shutil.copy2(source, github / name)
    (github / "SHA256SUMS").write_text(sums_text(assets), encoding="utf-8", newline="\n")
    notes = context.repository / "release" / "release_notes.md"
    shutil.copy2(notes, github / "NOTES.md")
    shutil.copytree(context.work / "build" / "windows" / "site", pages)
    for name in MODEL_FILES:
        shutil.copy2(context.work / "model" / name, hub / name)
    shutil.copy2(context.repository / "release" / "model_card.md", hub / "README.md")
    shutil.copy2(context.repository / "LICENSE", hub / "LICENSE")
    (out / "plan.json").write_text(json.dumps({"tag": context.plan.tag, "github": sorted(path.name for path in github.iterdir()),
                                                "hub": sorted(path.name for path in hub.iterdir())}, indent=1) + "\n", encoding="utf-8", newline="\n")
    context.note("github_assets", sorted(assets))


def outputs(context: Context) -> list[Path]:
    return [context.work / "publish" / "github", context.work / "publish" / "pages", context.work / "publish" / "hub"]


STAGES = (Stage("assemble", ("verify", "reproduce"), "lay out what the GitHub release, the Pages branch, and the Hub receive", assemble, outputs),)
