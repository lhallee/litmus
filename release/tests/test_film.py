from __future__ import annotations

import hashlib
import json
import pytest

from litmus_release import film
from litmus_release.commands import StageError
from tests.conftest import WEIGHTS


WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8X" + bytes(16)


def record(context, weights: str | None) -> None:
    """The film's record as `demo/render.mjs` writes it, naming the weights its numbers were measured with."""
    model = None if weights is None else {"version": "sha256:0", "weights_sha256": weights, "size": "d192 L4"}
    (context.work / "media" / film.RECORD).write_text(json.dumps({"model": model}), encoding="utf-8")


def test_the_preview_is_copied_to_docs_images_and_its_digest_is_an_input(context) -> None:
    media = context.work / "media"
    media.mkdir()
    (media / film.PREVIEW).write_bytes(WEBP)
    record(context, hashlib.sha256(WEIGHTS).hexdigest())
    film.film(context)
    assert (context.repository / "docs" / "images" / film.PREVIEW).read_bytes() == WEBP
    assert film.outputs(context) == [context.repository / "docs" / "images" / film.PREVIEW]
    assert sorted(film.inputs(context)) == sorted([film.PREVIEW, film.RECORD])
    assert context.details["film_weights_sha256"] == hashlib.sha256(WEIGHTS).hexdigest()


def test_a_missing_or_foreign_preview_stops_the_stage(context) -> None:
    with pytest.raises(StageError, match="render the ad's wide cut"):
        film.film(context)
    (context.work / "media").mkdir()
    (context.work / "media" / film.PREVIEW).write_bytes(b"GIF89a" + bytes(16))
    with pytest.raises(StageError, match="not a WebP"):
        film.film(context)


def test_a_film_measured_with_another_model_stops_the_stage(context) -> None:
    # The films released with 1.0.0 and 1.1.0 showed an earlier model's speed: no record, a record that names no
    # weights, and another model's weights each stop the stage, and nothing reaches docs/images.
    media = context.work / "media"
    media.mkdir()
    (media / film.PREVIEW).write_bytes(WEBP)
    with pytest.raises(StageError, match="nothing says which model"):
        film.film(context)
    for weights in (None, "0" * 64):
        record(context, weights)
        with pytest.raises(StageError, match="another model's"):
            film.film(context)
    assert not (context.repository / "docs" / "images" / film.PREVIEW).exists()
    (context.work / "model" / "model.safetensors").unlink()
    with pytest.raises(StageError, match="run the `fetch` stage first"):
        film.film(context)
