"""The input vectors a head reads: one encoder's embeddings, two concatenated, or the random-embedding control."""

from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path

from .embed import EMBEDDINGS_FILE, TOKENS_FILE


RANDOM_CHUNK = 4096  # rows per generator seed, so a row's random vector never depends on which rows are asked for
RANDOM_WIDTH = 2048


@dataclass(frozen=True)
class FeatureSource:
    """Rows of `(n, f)` input vectors in `id` order, read from embedding directories or drawn at random.

    `name` is how runs record it: an encoder key, `a+b` for a concatenation, or `random` for the control, whose
    vectors are standard normal, fixed by the row's id and the seed, and carry nothing about the text.
    """

    name: str
    directories: tuple[Path, ...]
    seed: int = 0

    @property
    def is_random(self) -> bool:
        return not self.directories

    @property
    def width(self) -> int:
        if self.is_random:
            return RANDOM_WIDTH

        return sum(np.load(directory / EMBEDDINGS_FILE, mmap_mode="r").shape[1] for directory in self.directories)

    def embedded(self, n: int) -> np.ndarray:
        """Which rows every directory has embedded: (n,) bool."""
        present = np.ones(n, dtype=bool)  # (n,)
        for directory in self.directories:
            present &= np.load(directory / TOKENS_FILE, mmap_mode="r")[:n] >= 0

        return present  # (n,)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """The vectors of the given ids, in the order given: (m,) to (m, f) float16."""
        # ids: (m,)
        if self.is_random:
            return random_rows(ids, RANDOM_WIDTH, self.seed)  # (m, f)
        order = np.argsort(ids)  # (m,) sorted reads are much faster from a memory map
        parts = [np.load(directory / EMBEDDINGS_FILE, mmap_mode="r")[ids[order]] for directory in self.directories]
        values = np.concatenate(parts, axis=1)  # (m, f) in sorted order
        out = np.empty_like(values)  # (m, f)
        out[order] = values

        return out  # (m, f)


def random_rows(ids: np.ndarray, width: int, seed: int) -> np.ndarray:
    """Standard normal rows fixed by (id, seed): (m,) to (m, width) float16."""
    # ids: (m,)
    out = np.empty((len(ids), width), dtype=np.float16)  # (m, width)
    chunks = ids // RANDOM_CHUNK  # (m,)
    for chunk in np.unique(chunks):
        generator = torch.Generator().manual_seed(seed * 1_000_003 + int(chunk))
        block = torch.randn(RANDOM_CHUNK, width, generator=generator, dtype=torch.float32)  # (RANDOM_CHUNK, width)
        members = chunks == chunk
        out[members] = block[torch.from_numpy(ids[members] % RANDOM_CHUNK)].numpy().astype(np.float16)

    return out  # (m, width)


def feature_source(spec: str, embedding_root: Path, seed: int = 0) -> FeatureSource:
    """`modernbert_large`, `mmbert_base`, `modernbert_large+mmbert_base` or `random`, read under `embedding_root/<encoder>`."""
    if spec == "random":
        return FeatureSource("random", (), seed)

    return FeatureSource(spec, tuple(embedding_root / part for part in spec.split("+")), seed)
