"""The language of a text from a `multi` source, by fastText's lid.176.

fastText's language identifier (Joulin et al. 2016) is chosen for speed and coverage: one linear layer over
hashed character n-grams, about 8,000 texts of 2,000 characters a second on one core (measured), 176
languages, ISO 639-1 labels as the catalog writes them, and the identifier CCNet, Dolma and RedPajama filter
with. Sources with one language take the catalog's, so only `multi` sources pay for it.
"""

from __future__ import annotations

import hashlib
import urllib.request
import fasttext

from functools import cache
from pathlib import Path

from .config import LANGUAGE_CHARACTERS, LANGUAGE_MODEL_SHA256, LANGUAGE_MODEL_URL, MIN_LANGUAGE_PROBABILITY


UNDETERMINED = "und"
LABEL_PREFIX = "__label__"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)

    return digest.hexdigest()


def ensure_language_model(directory: Path) -> Path:
    """The language model on disk, downloaded once and checked against its pinned hash."""
    path = directory / "lid.176.bin"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        urllib.request.urlretrieve(LANGUAGE_MODEL_URL, temporary)
        temporary.replace(path)
    found = file_sha256(path)
    if found != LANGUAGE_MODEL_SHA256:
        raise RuntimeError(f"{path} has sha256 {found}, not the pinned {LANGUAGE_MODEL_SHA256}")

    return path


@cache
def language_model(path: str) -> fasttext.FastText._FastText:
    return fasttext.load_model(path)


def detect_languages(texts: list[str], model_path: str) -> list[str]:
    """The most probable language of each text's opening, or `und` below `MIN_LANGUAGE_PROBABILITY`."""
    model = language_model(model_path)
    languages: list[str] = []
    for text in texts:
        line = " ".join(text[:LANGUAGE_CHARACTERS].split())
        # The binding's own predict breaks on NumPy 2, so the C++ predict is called directly.
        predictions = model.f.predict(line, 1, 0.0, "strict")
        if not predictions:
            languages.append(UNDETERMINED)
            continue
        probability, label = predictions[0]
        languages.append(label.removeprefix(LABEL_PREFIX) if probability >= MIN_LANGUAGE_PROBABILITY else UNDETERMINED)

    return languages
