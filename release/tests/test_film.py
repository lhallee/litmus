from __future__ import annotations

import pytest

from litmus_release import film
from litmus_release.commands import StageError


WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8X" + bytes(16)


def test_the_preview_is_copied_to_docs_images_and_its_digest_is_an_input(context) -> None:
    media = context.work / "media"
    media.mkdir()
    (media / film.PREVIEW).write_bytes(WEBP)
    film.film(context)
    assert (context.repository / "docs" / "images" / film.PREVIEW).read_bytes() == WEBP
    assert film.outputs(context) == [context.repository / "docs" / "images" / film.PREVIEW]
    assert list(film.inputs(context)) == [film.PREVIEW]


def test_a_missing_or_foreign_preview_stops_the_stage(context) -> None:
    with pytest.raises(StageError, match="render the ad's wide cut"):
        film.film(context)
    (context.work / "media").mkdir()
    (context.work / "media" / film.PREVIEW).write_bytes(b"GIF89a" + bytes(16))
    with pytest.raises(StageError, match="not a WebP"):
        film.film(context)
