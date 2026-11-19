"""Windows, exact moment combination, and the packing that feeds the GPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from litmus_teacher.embed import pack_pieces, packed_batch, padded_batch, piece_moments, processed_tokens
from litmus_teacher.moments import combine_moments, window_count_mask, window_spans


CONTEXT, OVERLAP = 64, 12  # a small encoder: 62 content tokens a window


@pytest.mark.parametrize("content", [0, 1, 61, 62, 63, 100, 113, 500, 1031])
def test_every_state_counts_once(content: int) -> None:
    windows = window_spans(content, CONTEXT, OVERLAP)
    counted = np.zeros(content, dtype=np.int64)  # (c,)
    for window in windows:
        assert window.stop - window.start == min(content, CONTEXT - 2)
        mask = window_count_mask(window)  # (l,)
        counted[window.start : window.stop] += mask[1:-1]
    assert (counted == 1).all()
    assert sum(window.counted for window in windows) == content + 2
    assert windows[0].first and windows[-1].last
    for left, right in zip(windows[:-1], windows[1:], strict=False):
        assert left.stop - right.start >= OVERLAP


def test_windowed_moments_equal_direct() -> None:
    rng = np.random.default_rng(0)
    content, width = 500, 7
    states = rng.normal(3.0, 5.0, size=(content + 2, width))  # (c + 2, d): [start, content..., end]
    count, total, squares = np.zeros(1), np.zeros((1, width)), np.zeros((1, width))
    for window in window_spans(content, CONTEXT, OVERLAP):
        positions = np.r_[0, np.arange(window.start, window.stop) + 1, content + 1]  # state index of each piece position
        rows = states[positions[window_count_mask(window)]]  # (counted, d)
        count += len(rows)
        total += rows.sum(0)
        squares += (rows**2).sum(0)
    mean, variance = combine_moments(count, total, squares)  # (1, d) each
    np.testing.assert_allclose(mean[0], states.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(variance[0], states.var(0), rtol=1e-10, atol=1e-10)


def test_processed_tokens_matches_windows() -> None:
    contents = np.array([0, 5, 62, 63, 200, 1031])
    expected = [sum(window.stop - window.start + 2 for window in window_spans(int(c), CONTEXT, OVERLAP)) for c in contents]
    assert processed_tokens(contents, CONTEXT, OVERLAP, cap=10_000).tolist() == expected


def _pieces() -> list[tuple[int, np.ndarray, np.ndarray]]:
    pieces = []
    for row, content in enumerate([3, 70, 10]):
        for window in window_spans(content, CONTEXT, OVERLAP):
            tokens = np.r_[100, np.arange(window.start, window.stop) + 1000 * row, 101].astype(np.int64)
            pieces.append((row, tokens, window_count_mask(window)))

    return pieces  # (slice row, tokens (l,), counted (l,)) per piece


def test_packed_batch_segments_and_positions() -> None:
    pieces = _pieces()
    batch = packed_batch(pieces, pad_id=0, context=CONTEXT, pad_to=256)
    cu_seqlens = batch["cu_seqlens"].numpy()
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == 256 == len(batch["input_ids"])
    for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        assert batch["position_ids"][begin:end].tolist() == list(range(end - begin))
        assert end - begin <= CONTEXT
    segment = batch["segment"].numpy()
    assert sorted(set(segment.tolist()) - {-1}) == [0, 1, 2]
    assert np.bincount(segment[segment >= 0]).tolist() == [3 + 2, 70 + 2, 10 + 2]  # windows of row 1 share a row


def test_padded_batch_matches_packed_counts() -> None:
    batch = padded_batch(_pieces(), pad_id=0)
    segment = batch["segment"].numpy()
    assert np.bincount(segment[segment >= 0]).tolist() == [5, 72, 12]
    assert batch["attention_mask"].sum() == sum(len(piece) for _, piece, _ in _pieces())


def test_pack_pieces_respects_budget() -> None:
    lengths = [64, 64, 40, 30, 20, 10, 5]
    packed = pack_pieces(lengths, 128, packed=True)
    assert all(sum(lengths[i] for i in batch) <= 128 for batch in packed)
    padded = pack_pieces(lengths, 128, packed=False)
    assert all(max(lengths[i] for i in batch) * len(batch) <= 128 for batch in padded)
    assert sorted(i for batch in packed for i in batch) == list(range(len(lengths)))


def test_piece_moments_match_numpy() -> None:
    generator = torch.Generator().manual_seed(0)
    states = torch.randn(30, 4, generator=generator)  # (t, d)
    segment = torch.tensor([0] * 10 + [-1] * 5 + [1] * 10 + [0] * 5)  # (t,)
    count, total, squares = piece_moments(states, segment, rows=2)
    assert count.tolist() == [15, 10]
    np.testing.assert_allclose(total[0].numpy(), states[segment == 0].double().sum(0).numpy())
    np.testing.assert_allclose(squares[1].numpy(), (states[segment == 1].double() ** 2).sum(0).numpy())
