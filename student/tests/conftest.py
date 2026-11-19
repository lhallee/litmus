"""Shared fixtures: a tiny random release, its loaded reference model, the pinned tokenizer, and a synthetic corpus
with a teacher export for the training comparisons."""

from __future__ import annotations

import json
import numpy as np
import pytest

from pathlib import Path
from typing import Any
from litmus_student.contract import reference
from litmus_student.data import write_store


SPLITS = ["bank", "train", "validation"]


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A three-origin corpus whose origins use different tokens, some validation texts past a tiny chunk, and a teacher
    export (embeddings, query and key, present rows) whose vectors cluster by origin: the store and the export."""
    generator = np.random.default_rng(0)
    counts = {"train": 240, "validation": 150, "bank": 120}
    split = np.concatenate([np.full(count, SPLITS.index(name)) for name, count in counts.items()])  # (n,)
    n = len(split)
    label = np.arange(n) % 3  # (n,)
    lengths = np.where(generator.random(n) < 0.3, generator.integers(9, 60, n), generator.integers(1, 9, n))  # (n,)
    validation = np.flatnonzero(split == SPLITS.index("validation"))  # (v,)
    lengths[validation[::15]] = generator.integers(100, 200, len(validation[::15]))  # read as two or three tiny chunks
    texts = [generator.integers(1000 * origin, 1000 * origin + 1000, length) for origin, length in zip(label, lengths, strict=True)]
    write_store(tmp_path / "store", texts, label, split, generator.integers(0, 6, n), SPLITS)

    export = tmp_path / "teacher"
    export.mkdir()
    centers = generator.standard_normal((3, 12))  # (3, 2D)
    np.save(export / "embeddings.npy", (centers[label] + 0.3 * generator.standard_normal((n, 12))).astype(np.float16))
    query_key = generator.standard_normal((3, 2, 6))[label] + 0.3 * generator.standard_normal((n, 2, 6))  # (n, 2, w_T)
    query_key /= np.linalg.norm(query_key, axis=-1, keepdims=True)
    np.save(export / "query_key.npy", query_key.reshape(n, 12).astype(np.float16))
    present = np.ones(n, dtype=bool)  # (n,)
    present[::37] = False
    np.save(export / "present.npy", present)
    manifest = {"head": {"scale": 10.0, "bias": -1.0}, "embeddings": [{"path": "embeddings.npy"}],
                "query_key": {"path": "query_key.npy", "present": "present.npy"}}
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return tmp_path / "store", export


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    return reference.load_tokenizer()


@pytest.fixture(scope="session")
def tiny_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tiny_release")
    reference.init_release(reference.named_config("tiny"), directory, seed=3, bank_per_origin=8, calibration_tokens=200)
    return directory


@pytest.fixture(scope="session")
def tiny_model(tiny_release: Path) -> Any:
    return reference.ReferenceModel.load(tiny_release)


@pytest.fixture
def generator() -> np.random.Generator:
    return np.random.default_rng(11)
