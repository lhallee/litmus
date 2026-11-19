"""An extended corpus's store starts from the earlier store's rows, copied, never recomputed."""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path
from litmus_teacher.embed import EmbeddingStore, EmbedSettings, copy_earlier


def settings(**changes: object) -> EmbedSettings:
    values = {"encoder": "modernbert_base", "attention": "sdpa", "batch_tokens": 131_072, "overlap": 1024, "max_text_tokens": 131_072,
              "compile": False} | changes
    return EmbedSettings(**values)  # type: ignore[arg-type]


def test_the_earlier_rows_are_copied_and_only_embedded_ones_count(tmp_path: Path) -> None:
    earlier = EmbeddingStore(tmp_path / "earlier", 5, settings())
    earlier.values[:] = np.arange(5 * 1536, dtype=np.float16).reshape(5, 1536) % 7
    earlier.tokens[:] = [3, -1, 9, 4, 2]  # row 1 was a benchmark row, never embedded
    earlier.values.flush()
    earlier.tokens.flush()
    store = EmbeddingStore(tmp_path / "later", 8, settings(compile=True))

    copied = copy_earlier(store, tmp_path / "earlier", settings(compile=True))

    assert copied.tolist() == [True, False, True, True, True, False, False, False]
    assert np.array_equal(store.values[:5], earlier.values) and store.tokens[:].tolist() == [3, -1, 9, 4, 2, -1, -1, -1]
    assert copy_earlier(store, tmp_path / "earlier", settings()).tolist() == copied.tolist()  # resumed: the marker stands
    with pytest.raises(ValueError, match="other settings"):
        copy_earlier(EmbeddingStore(tmp_path / "other", 8, settings(overlap=512)), tmp_path / "earlier", settings(overlap=512))
    with pytest.raises(ValueError, match="not an extension"):
        copy_earlier(EmbeddingStore(tmp_path / "small", 4, settings()), tmp_path / "earlier", settings())
