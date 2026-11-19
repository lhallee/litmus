"""The embed step's host side: slice planning and the worker's tokenize, window and pack, with a stand-in tokenizer."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pathlib import Path
from litmus_teacher.embed import EmbedSettings, Slice, TokenizedSlices, dense_row_count, plan_slices


class CharacterTokenizer:
    """One token per character, wrapped as [1] ... [2] when asked: enough to drive the worker."""

    pad_token_id = 0

    def __call__(self, texts: str | list[str], add_special_tokens: bool, return_attention_mask: bool = False) -> dict:
        def encode(text: str) -> list[int]:
            body = [3 + ord(character) % 100 for character in text]
            return [1, *body, 2] if add_special_tokens else body

        return {"input_ids": encode(texts) if isinstance(texts, str) else [encode(text) for text in texts]}


def corpus(tmp_path: Path, lengths: list[int]) -> Path:
    path = tmp_path / "corpus.parquet"
    texts = ["x" * length for length in lengths]
    pq.write_table(pa.table({"id": np.arange(len(lengths)), "text": texts, "tokens": lengths}), path, row_group_size=4)

    return path


def test_dense_ids_required() -> None:
    assert dense_row_count(np.array([2, 0, 1])) == 3
    with pytest.raises(ValueError):
        dense_row_count(np.array([0, 2]))


def test_plan_slices_cut_by_rows_and_tokens(tmp_path: Path) -> None:
    lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    path = corpus(tmp_path, lengths)
    selected = np.ones(len(lengths), dtype=bool)
    selected[1] = False
    plan = plan_slices(path, selected, slice_rows=3, slice_tokens=10_000)
    assert [(item.row_group, item.offset, item.length) for item in plan] == [(0, 0, 3), (0, 3, 1), (1, 0, 3), (1, 3, 1),
                                                                           (2, 0, 2)]
    assert plan[0].tokens == 10 + 30
    by_tokens = plan_slices(path, np.ones(len(lengths), dtype=bool), slice_rows=100, slice_tokens=60)
    assert all(item.tokens <= 60 or item.length == 1 for item in by_tokens)
    assert sum(item.length for item in by_tokens) == len(lengths)


def test_worker_batches_count_every_token(tmp_path: Path) -> None:
    lengths = [5, 70, 200, 3]
    path = corpus(tmp_path, lengths)
    selection = tmp_path / "selection.npy"
    np.save(selection, np.ones(len(lengths), dtype=bool))
    settings = EmbedSettings("modernbert_large", "flash_attention_2", batch_tokens=256, overlap=12,
                             max_text_tokens=150, compile=False)
    plan = [Slice(0, 0, 0, 4, sum(lengths))]
    dataset = TokenizedSlices(path, plan, selection, settings)
    tokenizer = CharacterTokenizer()
    texts = ["x" * length for length in lengths]
    batches = list(dataset._slice_batches(0, np.arange(4), texts, tokenizer, 1, 2, 0))
    assert batches[-1]["final"] and not any(batch["final"] for batch in batches[:-1])
    counted = np.zeros(4, dtype=np.int64)
    for batch in batches:
        segment, rows = batch["segment"].numpy(), batch["row_index"].numpy()
        np.add.at(counted, rows[segment[segment >= 0]], 1)
        assert len(batch["input_ids"]) <= settings.batch_tokens
    assert counted.tolist() == [5 + 2, 70 + 2, 150 + 2, 3 + 2]  # the 200-token text is capped at 150
    assert batches[-1]["content_tokens"].tolist() == [5, 70, 150, 3]
