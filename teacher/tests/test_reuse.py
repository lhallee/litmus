"""Reusing the first corpus's ModernBERT-base vectors: eligibility, the copy, alignment and the three outcomes."""

from __future__ import annotations

import argparse
import json
import numpy as np
import pyarrow as pa
import pytest

from pathlib import Path
from litmus_teacher.embed import EmbeddingStore, EmbedSettings, copy_reusable, reuse_gate_of
from litmus_teacher.reuse import (
    Alignment,
    copy_reused,
    decide_reuse,
    fit_alignment,
    reusable_rows,
    shift_auc,
    vector_drift,
)


WIDTH = 2 * 768  # ModernBERT-base mean then variance
NOISE = 0.01


def metadata() -> pa.Table:
    return pa.table({
        "id": [0, 1, 2, 3, 4, 5, 6],
        "tokens": [10, 10, 10, 5000, 4094, 10, 4095],
        "first_corpus_id": pa.array([3, None, 0, 1, 2, 4, 0], pa.int64()),
        "text_unchanged": pa.array([True, True, False, True, True, None, True]),
    })


def texts(m: int = 2000, d: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise-free vectors (mean half around 0, variance half positive), token counts, and a second rng draw."""
    rng = np.random.default_rng(seed)
    base = np.concatenate([rng.normal(0.0, 1.0, (m, d)) + rng.normal(0, 1, d),
                           np.abs(rng.normal(1.0, 0.3, (m, d)))], axis=1)  # (m, 2d)
    tokens = np.exp(rng.uniform(np.log(8), np.log(4000), m)).astype(np.int64)  # (m,)

    return base, tokens, rng.normal(size=(4, m, 2 * d))  # (m, 2d), (m,), four noise draws (m, 2d)


def packing_drift(base: np.ndarray, noise: np.ndarray) -> dict[str, dict[str, float]]:
    # base: (m, 2d); noise: (4, m, 2d)
    return vector_drift(base + NOISE * noise[0], base + NOISE * noise[1], base.shape[1] // 2)


def test_reusable_rows_rules() -> None:
    selected = np.ones(7, dtype=bool)
    ids, first = reusable_rows(metadata(), selected, max_text_tokens=131_072)
    # 1 has no first-corpus row, 2 changed, 3 and 6 past the 4,096-position cap, 5 unknown
    assert ids.tolist() == [0, 4] and first.tolist() == [3, 2]
    selected[4] = False
    assert reusable_rows(metadata(), selected, 131_072)[0].tolist() == [0]


def test_copy_reused_writes_only_those_rows(tmp_path: Path) -> None:
    source = np.abs(np.random.default_rng(0).normal(size=(5, WIDTH))).astype(np.float16)
    np.save(tmp_path / "first.npy", source)
    values = np.lib.format.open_memmap(tmp_path / "values.npy", mode="w+", dtype=np.float16, shape=(7, WIDTH))
    tokens = np.lib.format.open_memmap(tmp_path / "tokens.npy", mode="w+", dtype=np.int32, shape=(7,))
    tokens[:] = -1
    copy_reused(values, tokens, tmp_path / "first.npy", np.array([0, 4]), np.array([3, 2]), np.array([10, 4094]))
    assert np.array_equal(values[0], source[3]) and np.array_equal(values[4], source[2])
    assert tokens.tolist() == [10, -1, -1, -1, 4094, -1, -1]
    alignment = Alignment("affine", False, np.full((1, WIDTH), 2.0, np.float32), np.full((1, WIDTH), 0.5, np.float32))
    copy_reused(values, tokens, tmp_path / "first.npy", np.array([1]), np.array([0]), np.array([20]), alignment)
    np.testing.assert_allclose(values[1].astype(np.float32), 2 * source[0].astype(np.float32) + 0.5, rtol=2e-3)
    with pytest.raises(ValueError):
        copy_reused(values, tokens, tmp_path / "first.npy", np.array([1]), np.array([5]), np.array([10]))
    np.save(tmp_path / "narrow.npy", source[:, :100])
    with pytest.raises(ValueError):
        copy_reused(values, tokens, tmp_path / "narrow.npy", np.array([1]), np.array([0]), np.array([10]))


def test_copy_reusable_into_store_is_resumable_and_recorded(tmp_path: Path) -> None:
    source = np.abs(np.random.default_rng(1).normal(size=(5, WIDTH))).astype(np.float16)
    np.save(tmp_path / "first.npy", source)
    settings = EmbedSettings("modernbert_base", "flash_attention_2", 131_072, 1024, 131_072, False)
    store = EmbeddingStore(tmp_path / "store", 7, settings)
    args = argparse.Namespace(reuse=tmp_path / "first.npy")
    gate = {"outcome": "safe", "copy": True, "alignment_path": None, "accepted_risk": False}
    reused = copy_reusable(args, metadata(), np.ones(7, dtype=bool), settings, store, gate)
    assert np.flatnonzero(reused).tolist() == [0, 4]
    assert np.array_equal(store.values[4], source[2]) and store.tokens[0] == 10 and store.tokens[1] == -1
    assert json.loads((tmp_path / "store" / "meta.json").read_text(encoding="utf-8"))["reuse"]["outcome"] == "safe"
    np.save(tmp_path / "first.npy", np.zeros_like(source))  # a second call must not copy again
    again = copy_reusable(args, metadata(), np.ones(7, dtype=bool), settings, store, gate)
    assert np.array_equal(again, reused) and np.array_equal(store.values[4], source[2])


def test_shift_auc_sees_an_offset_not_noise() -> None:
    rng = np.random.default_rng(2)
    stored = rng.normal(size=(2000, 64))
    noisy = stored + 0.01 * rng.normal(size=stored.shape)
    assert 0.4 < shift_auc(stored, noisy, seed=0) < 0.6
    assert shift_auc(noisy + 0.3, noisy, seed=0) > 0.9  # d-prime about 0.3 * sqrt(64) = 2.4 for a linear reader
    assert shift_auc(noisy * 1.5, noisy, seed=0, squared=True) > 0.9


def test_matching_vectors_are_safe() -> None:
    base, tokens, noise = texts()
    decision, alignment, _ = decide_reuse(base + NOISE * noise[2], base + NOISE * noise[3], tokens,
                                          packing_drift(base, noise), vector_drift, seed=0)
    assert decision["outcome"] == "safe" and alignment is None


def test_planted_affine_difference_is_aligned_away() -> None:
    base, tokens, noise = texts()
    rng = np.random.default_rng(3)
    columns = base.shape[1]
    scale, shift = rng.uniform(0.9, 1.1, columns), rng.normal(0, 0.1, columns)  # (2d,) each
    stored = (base - shift) / scale + NOISE * noise[2]  # recomputed is scale * stored + shift, plus noise
    recomputed = base + NOISE * noise[3]
    decision, alignment, held = decide_reuse(stored, recomputed, tokens, packing_drift(base, noise), vector_drift, seed=0)
    assert not decision["raw"]["passes"] and decision["raw"]["shift_auc"] > 0.9
    assert decision["outcome"] == "aligned", decision["aligned"]["reasons"]
    assert 0.4 < decision["aligned"]["shift_auc"] < 0.6 and 0.4 < decision["aligned"]["spread_auc"] < 0.6
    assert decision["alignment"]["chosen"].startswith("affine")
    aligned = alignment.apply(stored[held], tokens[held])
    assert 0.4 < shift_auc(aligned, recomputed[held], seed=1) < 0.6


def test_alignment_does_not_hide_a_non_affine_difference() -> None:
    base, tokens, noise = texts()
    columns = (base - base.mean(0)) / base.std(0)  # (m, 2d) standardized
    stored = base + 0.3 * base.std(0) * (columns**2 - 1) + NOISE * noise[2]  # same column means, a bent column
    decision, _, _ = decide_reuse(stored, base + NOISE * noise[3], tokens, packing_drift(base, noise), vector_drift, seed=0)
    assert decision["outcome"] == "refused"
    assert decision["aligned"]["reasons"]


def test_length_dependent_offset_picks_a_per_length_alignment() -> None:
    base, tokens, noise = texts()
    shift = np.where(tokens[:, None] < 256, 0.2, -0.2)  # (m, 1): the offset flips with length
    alignment, report = fit_alignment(base + shift + NOISE * noise[2], base + NOISE * noise[3], tokens, seed=0)
    assert alignment.per_length and report["per_length_error_ratio"] < 0.5


def test_alignment_round_trip_keeps_variances_non_negative(tmp_path: Path) -> None:
    alignment = Alignment("shift", False, np.ones((1, 4), np.float32), np.array([[0.0, 0.0, -5.0, -5.0]], np.float32))
    alignment.save(tmp_path / "alignment.npz")
    loaded = Alignment.load(tmp_path / "alignment.npz")
    out = loaded.apply(np.ones((3, 4), np.float16), np.array([10, 300, 2000]))
    assert (out[:, 2:] == 0).all() and (out[:, :2] == 1).all() and loaded.form == "shift"


def test_reuse_gate_outcomes(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.json"
    with pytest.raises(RuntimeError):
        reuse_gate_of(pilot, accept_risk=False)
    assert reuse_gate_of(pilot, accept_risk=True) == {"outcome": "unchecked", "copy": True, "alignment_path": None,
                                                      "accepted_risk": True}
    for outcome, copies in (("safe", True), ("aligned", True), ("refused", False)):
        check = {"outcome": outcome, "alignment_path": "a.npz" if outcome == "aligned" else None}
        pilot.write_text(json.dumps({"reuse_check": check}), encoding="utf-8")
        gate = reuse_gate_of(pilot, accept_risk=False)
        assert gate["outcome"] == outcome and gate["copy"] is copies
        assert (gate["alignment_path"] == "a.npz") is (outcome == "aligned")
