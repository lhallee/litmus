"""The fixed vocabulary every step shares: origins, length bins and the run summary file.

NumPy and the standard library only, so `calibrate` stays cheap to import.
"""

from __future__ import annotations

import json
import os
import tempfile
import numpy as np

from pathlib import Path
from typing import Any


ORIGINS = ("human", "ai", "mixed")  # label 0, 1, 2 in the corpus
HUMAN, AI, MIXED = 0, 1, 2
# Length bins on the ModernBERT token count of the text without special tokens (the corpus `tokens` column):
# bin i holds [LENGTH_EDGES[i], LENGTH_EDGES[i + 1]), and the last bin is open above. Doubling from 16 tokens,
# finest where scores are weakest, then one bin of 8,192 to 32,767 and one of 32,768 and more.
LENGTH_EDGES = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768)
SUMMARY_FILE = "run_summary.json"


def length_bins(tokens: np.ndarray, edges: tuple[int, ...] = LENGTH_EDGES) -> np.ndarray:
    """The length bin of each text from its token count: (n,) to (n,) int64 in [0, len(edges))."""
    # tokens: (n,)
    tokens = np.asarray(tokens)
    if (tokens < 0).any():
        raise ValueError("token counts must be non-negative")

    return np.searchsorted(np.asarray(edges[1:]), tokens, side="right").astype(np.int64)  # (n,)


def length_bin_names(edges: tuple[int, ...] = LENGTH_EDGES) -> list[str]:
    """Readable names of the bins, such as `16-31` and `32768+`."""
    names = [f"{low}-{high - 1}" for low, high in zip(edges[:-1], edges[1:], strict=True)]

    return [*names, f"{edges[-1]}+"]


def write_json(path: Path, payload: Any) -> None:
    """Replace `path` with indented JSON atomically: a reader sees the old file or the whole new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_summary(directory: Path, step: str, payload: dict[str, Any]) -> Path:
    """Write the step's `run_summary.json`: what ran, on what, and what it produced."""
    path = directory / SUMMARY_FILE
    write_json(path, {"step": step, **payload})

    return path


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot write {type(value).__name__} as JSON")
