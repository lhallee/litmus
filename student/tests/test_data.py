"""Packing invariants, the pre-tokenized store, cluster sampling and pre-tokenization of a Parquet corpus."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pathlib import Path
from typing import Any
from litmus_student.contract import reference
from litmus_student.data import ClusterSampler, TokenStore, cluster_stratified_rows, pack, packed_batch, tokenize_corpus, write_rows, write_store


@pytest.mark.parametrize("seed", range(5))
def test_pack_keeps_every_text_whole_in_its_blocks(seed: int) -> None:
    generator = np.random.default_rng(seed)
    W, budget = 16, 256
    lengths = np.concatenate([generator.integers(1, 9, 40), generator.integers(9, 200, 15), [300]])  # (b,)
    texts = [np.ones(n, dtype=np.int64) for n in lengths]
    plans = pack(lengths, budget, W)
    assert sorted(placement.row for plan in plans for placement in plan) == list(range(len(lengths)))
    for plan in plans:
        batch = packed_batch(plan, texts, W, W)  # asserts no overlap, long texts on block starts, short in half-blocks
        end = max(placement.start + lengths[placement.row] for placement in plan)
        assert end <= budget or len(plan) == 1 or any(lengths[placement.row] > budget for placement in plan)
        assert batch.tokens.shape[0] % W == 0 and batch.real_tokens == sum(lengths[placement.row] for placement in plan)


def test_store_round_trip_and_samplers(tmp_path: Path) -> None:
    generator = np.random.default_rng(0)
    n = 300
    texts = [generator.integers(0, 50368, generator.integers(1, 40)) for _ in range(n)]
    label = np.arange(n) % 3  # (n,)
    split = (np.arange(n) // 3) % 2  # (n,)
    cluster = generator.integers(0, 12, n)  # (n,)
    write_store(tmp_path, texts, label, split, cluster, ["train", "validation"])
    store = TokenStore(tmp_path)
    assert len(store) == n and all(np.array_equal(store.ids(row), texts[row]) for row in range(n))
    present = np.ones(n, dtype=bool)  # (n,)
    present[0] = False
    sampler = ClusterSampler(store, "train", 1000, present)
    rows = sampler.draw(90, generator)  # (b,)
    assert np.bincount(store.label[rows], minlength=3).tolist() == [30, 30, 30]
    assert (store.split[rows] == 0).all() and 0 not in rows
    chosen = cluster_stratified_rows(store, "validation", 10, generator)  # (m,)
    assert len(np.unique(chosen)) == 30 and (store.split[chosen] == 1).all()


def test_tokenize_corpus_matches_the_reference(tmp_path: Path, tokenizer: Any) -> None:
    texts = ["Hello there.", "A second, longer text about rivers and ledgers.", "日本語", "code = 1\n    x"]  # standard readings
    counts = [len(reference.token_ids(tokenizer, text)) for text in texts]
    table = pa.table({"id": [2, 0, 3, 1], "text": texts, "label": pa.array([0, 1, 2, 0], pa.int8()),
                      "split": ["train", "train", "validation", "train"], "cluster": [5, 5, 6, 7], "tokens": counts,
                      "long_eval": [False, False, True, False]})
    pq.write_table(table, tmp_path / "corpus.parquet")
    tokenize_corpus(tmp_path / "corpus.parquet", tmp_path / "store", None, batch_rows=2)
    write_rows(tmp_path / "corpus.parquet", tmp_path / "store", None)
    store = TokenStore(tmp_path / "store")
    for corpus_id, text in zip([2, 0, 3, 1], texts, strict=True):
        assert store.ids(corpus_id).tolist() == reference.token_ids(tokenizer, text).tolist()
    assert store.label.tolist() == [1, 0, 0, 2] and store.splits == ["train", "validation"]
    assert store.long_eval.tolist() == [False, False, False, True]
    # The teacher's split.parquet overrides the corpus's own columns; a row with no cluster gets -1.
    split = pa.table({"id": [0, 1, 2, 3], "split": ["bank", "train", "benchmark", "train"], "cluster": pa.array([4, 9, None, 9], pa.int64())})
    pq.write_table(split, tmp_path / "split.parquet")
    write_rows(tmp_path / "corpus.parquet", tmp_path / "store", tmp_path / "split.parquet")
    store = TokenStore(tmp_path / "store")
    assert store.splits == ["bank", "benchmark", "train"] and store.split.tolist() == [0, 2, 1, 2] and store.cluster.tolist() == [4, 9, -1, 9]
    with pytest.raises(ValueError, match="every corpus id"):
        pq.write_table(split.slice(0, 3), tmp_path / "short.parquet")
        write_rows(tmp_path / "corpus.parquet", tmp_path / "store", tmp_path / "short.parquet")
    table = table.set_column(5, "tokens", pa.array([count + 2 for count in counts]))
    pq.write_table(table, tmp_path / "corpus.parquet")
    with pytest.raises(ValueError, match="tokenized differently"):
        tokenize_corpus(tmp_path / "corpus.parquet", tmp_path / "store2", None, batch_rows=2)
