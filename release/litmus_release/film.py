"""Stage `film`: the film's README preview, from the release's media folder into `docs/images/`.

GitHub plays no video stored in a repository, so the README shows the film as a silent animated WebP that links to the
site, where the film plays with sound. `demo/render.mjs` writes it beside the wide cut's other web encodes
(`litmus_wide_preview.webp`); it reaches `<work>/media/` with them (release/RELEASING.md, The ad), and this stage copies
it to where the README reads it. A new film makes the stage stale through its digest.
"""

from __future__ import annotations

import shutil

from pathlib import Path

from . import receipts
from .commands import StageError
from .stages import Context, Stage


PREVIEW = "litmus_wide_preview.webp"


def source(context: Context) -> Path:
    return context.work / "media" / PREVIEW


def target(context: Context) -> Path:
    return context.repository / "docs" / "images" / PREVIEW


def film(context: Context) -> None:
    preview = source(context)
    if not preview.is_file():
        raise StageError(f"{preview} is missing: render the ad's wide cut (demo/, `node render.mjs wide`, or `--web-only` from its finished file) "
                         f"and copy {PREVIEW} to {preview.parent} with the site's film and poster")
    with preview.open("rb") as stream:
        head = stream.read(12)
    if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
        raise StageError(f"{preview} is not a WebP file")
    target(context).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(preview, target(context))
    context.note("preview_bytes", str(preview.stat().st_size))


def outputs(context: Context) -> list[Path]:
    return [target(context)]


def inputs(context: Context) -> dict[str, str]:
    preview = source(context)
    return {PREVIEW: receipts.file_digest(preview)} if preview.is_file() else {}


STAGES = (Stage("film", (), "the film's README preview into docs/images", film, outputs, inputs=inputs),)
