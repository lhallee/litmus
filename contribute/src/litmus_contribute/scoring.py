"""Scoring behind a small interface: the engine's Python binding in a container, a fake in the tests.

The binding (`litmus_engine`, PyO3 over the Rust `litmus-engine`) does not exist yet. This module assumes, and the
README states, exactly this surface:

    model = litmus_engine.Model.from_path(path)   # a release directory or file
    model.version -> str                          # the release's version, stored with every row
    model.count_tokens(text: str) -> int          # tokens the engine would score
    model.score(text: str) -> dict                # the document reading, JSON-ready (student/SPEC.md, "Fields"),
                                                  # computed with the GIL released
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Protocol


JsonObject = dict[str, Any]  # a JSON object as `json.loads` returns it; the reading's fields belong to the engine


class Scorer(Protocol):
    @property
    def version(self) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def score(self, text: str) -> JsonObject: ...


class EngineScorer:
    """The engine's binding, loaded once (in a memory snapshot on Modal)."""

    def __init__(self, release: Path) -> None:
        import litmus_engine

        self.model = litmus_engine.Model.from_path(str(release))

    @property
    def version(self) -> str:
        return self.model.version

    def count_tokens(self, text: str) -> int:
        return self.model.count_tokens(text)

    def score(self, text: str) -> JsonObject:
        return self.model.score(text)


def serialize_reading(reading: JsonObject, max_bytes: int) -> str:
    """The reading as compact JSON, failing loudly when the engine returns something that is not a bounded JSON object."""
    assert isinstance(reading, dict), f"the engine's reading is a {type(reading).__name__}, not an object"
    serialized = json.dumps(reading, separators=(",", ":"), sort_keys=True, allow_nan=False)
    assert len(serialized.encode("utf-8")) <= max_bytes, f"the engine's reading is over {max_bytes} bytes"
    return serialized
