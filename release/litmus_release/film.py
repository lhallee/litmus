"""Stage `film`: the film's README preview, from the release's media folder into `docs/images/`.

GitHub plays no video stored in a repository, so the README shows the film as a silent animated WebP that links to the
site, where the film plays with sound. `demo/render.mjs` writes it beside the wide cut's other web encodes
(`litmus_wide_preview.webp`) with `litmus_wide_film.json`, which names the model the film's numbers were measured with;
they reach `<work>/media/` together (release/RELEASING.md, The ad), and this stage copies the preview to where the
README reads it. A new film makes the stage stale through its digest.

The film draws measured numbers (the speed, the memory, the readings of its texts), so it is held to the release's own
weights: a film measured with another model, or one that does not say, stops the stage. The films released with 1.0.0
and 1.1.0 were an earlier model's, and showed a speed neither release had.
"""

from __future__ import annotations

import json
import shutil

from pathlib import Path

from . import receipts
from .commands import StageError
from .stages import Context, Stage


PREVIEW = "litmus_wide_preview.webp"
RECORD = "litmus_wide_film.json"
RENDER_AGAIN = "render the ad again from this release (demo/README.md, A new release's film)"


def source(context: Context) -> Path:
    return context.work / "media" / PREVIEW


def record(context: Context) -> Path:
    return context.work / "media" / RECORD


def target(context: Context) -> Path:
    return context.repository / "docs" / "images" / PREVIEW


def check_model(context: Context) -> str:
    """The weights the film's numbers were measured with, which must be this release's."""
    stated = record(context)
    if not stated.is_file():
        raise StageError(f"{stated} is missing, so nothing says which model the film's numbers were measured with: {RENDER_AGAIN} and copy {RECORD} with the film")
    weights = context.work / "model" / "model.safetensors"
    if not weights.is_file():
        raise StageError(f"{weights} is missing: run the `fetch` stage first, since the film is held to this release's weights")
    measured = (json.loads(stated.read_text(encoding="utf-8")).get("model") or {}).get("weights_sha256")
    actual = receipts.file_digest(weights)
    if measured != actual:
        raise StageError(f"the film in {stated.parent} was measured with weights {str(measured)[:12]}, and this release's are {actual[:12]}, so its numbers are another model's: {RENDER_AGAIN}")
    return actual


def film(context: Context) -> None:
    preview = source(context)
    if not preview.is_file():
        raise StageError(f"{preview} is missing: render the ad's wide cut (demo/, `node render.mjs wide`, or `--web-only` from its finished file) "
                         f"and copy {PREVIEW} and {RECORD} to {preview.parent} with the site's film and poster")
    with preview.open("rb") as stream:
        head = stream.read(12)
    if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
        raise StageError(f"{preview} is not a WebP file")
    context.note("film_weights_sha256", check_model(context))
    target(context).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(preview, target(context))
    context.note("preview_bytes", str(preview.stat().st_size))


def outputs(context: Context) -> list[Path]:
    return [target(context)]


def inputs(context: Context) -> dict[str, str]:
    return {path.name: receipts.file_digest(path) for path in (source(context), record(context)) if path.is_file()}


STAGES = (Stage("film", ("fetch",), "the film's README preview into docs/images", film, outputs, inputs=inputs),)
