"""reference.py: quantization, packing, the file format, the standard reading, blocks, streaming and the reading."""

from __future__ import annotations

import json
import numpy as np
import pyarrow as pa
import pytest

from pathlib import Path
from typing import Any
from litmus_student.contract import reference


def test_ternarize_is_absmean_and_packing_round_trips(generator: np.random.Generator) -> None:
    W = generator.standard_normal((8, 12)).astype(np.float32)  # (o, i)
    T, gamma = reference.ternarize(W)  # (o, i), ()
    assert gamma == pytest.approx(float(np.abs(W).mean()), rel=1e-6)
    assert np.array_equal(T, np.clip(np.rint(W / gamma), -1, 1))
    packed = reference.pack_ternary(T)  # (o, i/4)
    assert packed.shape == (8, 3) and packed.dtype == np.uint8
    assert np.array_equal(reference.unpack_ternary(packed), T)
    # The code of one row, by hand: t + 1, lowest bits first.
    row = np.array([[-1, 0, 1, 1]], dtype=np.int8)
    assert reference.pack_ternary(row)[0, 0] == 0 | 1 << 2 | 2 << 4 | 2 << 6
    with pytest.raises(ValueError, match="code 3"):
        reference.unpack_ternary(np.array([[0b11]], dtype=np.uint8))


def test_activation_quantization_rounds_half_to_even_and_clamps() -> None:
    X = np.array([[127.0, 2.5, 3.5, -2.5, -0.5]], dtype=np.float32)  # max 127, so s = 1
    X_q, s = reference.quantize_activations(X, 1e-5)
    assert s[0] == np.float32(1.0)
    assert X_q.tolist() == [[127.0, 2.0, 4.0, -2.0, -0.0]]
    zeros, s_zero = reference.quantize_activations(np.zeros((1, 4), dtype=np.float32), 1e-5)
    assert (zeros == 0).all() and s_zero[0] == np.float32(127.0 / 1e-5)


def test_ternary_product_is_exact_integer_sums_then_one_scale(generator: np.random.Generator) -> None:
    T = generator.integers(-1, 2, (5, 8)).astype(np.int8)  # (o, i)
    weight = reference.Ternary.from_packed(reference.pack_ternary(T), np.array([0.25], dtype=np.float32))
    X_q = generator.integers(-128, 128, (3, 8)).astype(np.float64)  # (n, i)
    s = np.array([2.0, 4.0, 0.5], dtype=np.float32)  # (n,)
    expected = (X_q @ T.T.astype(np.float64)) * (0.25 / s.astype(np.float64))[:, None]  # (n, o)
    assert np.allclose(reference.ternary_product(X_q, s, weight), expected, rtol=1e-6)


def test_safetensors_files_read_with_the_library(tmp_path: Path, generator: np.random.Generator) -> None:
    from safetensors.numpy import load_file

    tensors = {"a": generator.integers(-127, 128, (3, 4)).astype(np.int8), "b": generator.random(5).astype(np.float32),
               "c": generator.random((2, 2)).astype(np.float16), "d": np.arange(4, dtype=np.uint8)}
    reference.write_safetensors(tmp_path / "x.safetensors", tensors)
    for loaded in (load_file(str(tmp_path / "x.safetensors")), reference.read_safetensors(tmp_path / "x.safetensors")):
        assert loaded.keys() == tensors.keys()
        for name, array in tensors.items():
            assert loaded[name].dtype == array.dtype and np.array_equal(loaded[name], array)


STANDARD_READING_CASES = [
    "Line one\r\nLine two  \r\n\r\n\r\n\r\n\r\nAfter\u200b zero\u2060width\x07 mark\t \n",
    "\n \n  \u3000\nStart\u00a0 \nend\u2003",
    "a\u200c\u200db\u00adc\ufeff",
    "x\r\ry\n\n\n\n\nz \t",
    "",
    "   ",
    "cafe\u0301 \u202aembedded\u202c",
    "tab\tinside\t\n\x0b\x0c\x85",
    "\u2028line\u2029para\u00a0\n\n\n\n",
    "Special [CLS] stays text <|endoftext|>",
]


def test_standard_reading_matches_the_corpus(tokenizer: Any) -> None:
    """The same text as the corpus's own reading, character for character, and so the same token ids."""
    from litmus_corpus.standardize import standard_reading as corpus_reading

    theirs = corpus_reading(pa.array(STANDARD_READING_CASES)).to_pylist()
    for text, corpus_text in zip(STANDARD_READING_CASES, theirs, strict=True):
        ours = reference.standard_reading(text)
        assert ours == corpus_text, repr(text)
        assert tokenizer.encode(ours, add_special_tokens=False).ids == tokenizer.encode(corpus_text, add_special_tokens=False).ids


def test_tokenizer_adds_no_special_tokens(tokenizer: Any) -> None:
    ids = reference.token_ids(tokenizer, "Hello [CLS] world\r\n")
    assert ids.tolist() == [12092, 209, 50281, 1533]
    assert tokenizer.get_vocab_size() == 50368


def test_block_spans_shift_on_odd_layers(tiny_model: Any) -> None:
    W = tiny_model.config.block_size  # 16
    assert tiny_model.block_spans(0, 0, 40) == [(0, 16), (16, 32), (32, 40)]
    assert tiny_model.block_spans(1, 0, 40) == [(0, 8), (8, 24), (24, 40)]
    assert tiny_model.block_spans(1, 0, 5) == [(0, 5)]
    assert all(stop - start <= W for start, stop in tiny_model.block_spans(1, 0, 1000))


def test_streaming_equals_whole_at_every_piece_size(tiny_model: Any, generator: np.random.Generator) -> None:
    config = tiny_model.config
    W, C = config.block_size, config.chunk_tokens  # 16, 96
    for n in (1, 7, W // 2, W // 2 + 1, W, W + 1, config.window_width + 1, C, C + 1, 2 * C + 5, 211):
        ids = generator.integers(0, config.vocab_size, n)  # (n,)
        whole, vectors = tiny_model.read_whole(ids)
        for piece in (1, 7, W - 1, W, W + 1, C - 1, n):
            streamed, streamed_vectors = tiny_model.read_streaming(ids, piece)
            for mine, theirs in zip(streamed_vectors["chunks"], vectors["chunks"], strict=True):
                assert np.abs(mine["pooled"] - theirs["pooled"]).max() <= 1e-6
                assert np.abs(mine["window_pooled"] - theirs["window_pooled"]).max() <= 1e-6
            assert [(w["start"], w["end"]) for w in streamed["windows"]] == [(w["start"], w["end"]) for w in whole["windows"]]
            assert [(c["start"], c["end"]) for c in streamed["cells"]] == [(c["start"], c["end"]) for c in whole["cells"]]
            assert abs(streamed["verdict"]["probability"] - whole["verdict"]["probability"]) <= 1e-9
            assert streamed["verdict"]["p_values"] == whole["verdict"]["p_values"] and streamed["verdict"]["sets"] == whole["verdict"]["sets"]


def test_chunk_bounds_are_equal_to_within_one_token() -> None:
    assert reference.chunk_bounds(96, 96) == [(0, 96)]
    assert reference.chunk_bounds(97, 96) == [(0, 48), (48, 97)]
    assert reference.chunk_bounds(290, 96) == [(0, 72), (72, 145), (145, 217), (217, 290)]
    for n in (1, 8191, 8192, 8193, 16384, 16385, 131072, 100_003):
        bounds = reference.chunk_bounds(n, 8192)
        lengths = [end - start for start, end in bounds]
        assert len(bounds) == -(-n // 8192) and max(lengths) <= 8192 and max(lengths) - min(lengths) <= 1
        assert bounds[0][0] == 0 and bounds[-1][1] == n and all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))


def test_windows_and_cells_of_a_chunk() -> None:
    assert reference.window_spans(20, 32, 16) == [(0, 20)]
    assert reference.window_spans(32, 32, 16) == [(0, 32)]
    assert reference.window_spans(64, 32, 16) == [(0, 32), (16, 48), (32, 64)]
    assert reference.window_spans(72, 32, 16) == [(0, 32), (16, 48), (32, 64), (40, 72)]
    assert reference.cell_spans(72, 16) == [(0, 16), (16, 32), (32, 48), (48, 64), (64, 72)]
    spans = reference.window_spans(8192, 512, 128)
    assert len(spans) == 61 and spans[-1] == (7680, 8192)
    assert reference.window_spans(1000, 512, 128) == [(0, 512), (128, 640), (256, 768), (384, 896), (488, 1000)]


def test_heat_map_by_hand() -> None:
    def window(start: int, end: int, position: float, lo: float, hi: float, unlike: bool = False) -> Any:
        return {"start": 100 + start, "end": 100 + end, "strip": {"position": position, "band_lo": lo, "band_hi": hi, "unlike_reference": unlike}}

    windows = [window(0, 32, 0.1, 0.0, 0.1), window(16, 48, 0.3, 0.0, 0.5), window(32, 64, 0.9, 0.9, 1.0), window(40, 72, 0.7, 0.5, 1.0, True)]
    cells = reference.heat_map(windows, 100, 72, 16)
    assert [(cell["start"], cell["end"], cell["windows"]) for cell in cells] == [(100, 116, 1), (116, 132, 2), (132, 148, 3), (148, 164, 2), (164, 172, 1)]
    assert cells[1]["position"] == pytest.approx(0.2) and cells[1]["spread"] == pytest.approx(0.1)
    # Cell [32, 48): window 1 holds 16 of its tokens, window 2 all 16, window 3 eight.
    assert cells[2]["position"] == pytest.approx((16 * 0.3 + 16 * 0.9 + 8 * 0.7) / 40)
    assert (cells[2]["band_lo"], cells[2]["band_hi"], cells[2]["unlike_reference"]) == (0.0, 1.0, True)
    assert not cells[0]["unlike_reference"] and cells[4]["position"] == pytest.approx(0.7)


def test_windows_from_totals_equal_pooling_their_states(tiny_model: Any, generator: np.random.Generator) -> None:
    """A window's vector, the difference of two float64 running totals, equals pooling its own states directly."""
    config = tiny_model.config
    H = (3.0 * generator.standard_normal((config.chunk_tokens, config.width))).astype(np.float32)  # (m, d)
    pool = tiny_model.chunk_pool(len(H))
    pool.add(H)
    for start, end in reference.window_spans(len(H), config.window_width, config.window_stride):
        part = H[start:end].astype(np.float64)  # (t, d)
        direct = reference.pooled_statistics(part.sum(axis=0), (part * part).sum(axis=0), end - start)  # (2d,)
        assert np.abs(pool.pooled(start, end) - direct).max() <= 1e-6 * (1 + np.abs(direct).max())
    # At the release's lengths: totals over 8,192 tokens of width 384 lose nothing a float32 vector can show.
    H = (3.0 * generator.standard_normal((8192, 384))).astype(np.float32)  # (m, d)
    pool = reference.ChunkPool(384, 8192, 512, 128)
    pool.add(H)
    worst = 0.0
    for start, end in reference.window_spans(8192, 512, 128):
        part = H[start:end].astype(np.float64)  # (t, d)
        direct = reference.pooled_statistics(part.sum(axis=0), (part * part).sum(axis=0), end - start)  # (2d,)
        worst = max(worst, float((np.abs(pool.pooled(start, end) - direct) / (1 + np.abs(direct))).max()))
    assert worst <= 1.2e-7  # one float32 rounding step at most


def test_window_context_gap_runs(tiny_model: Any, generator: np.random.Generator) -> None:
    ids = generator.integers(0, 50000, tiny_model.config.chunk_tokens)  # (m,)
    rows = reference.window_context_gap(tiny_model, ids, lambda part: tiny_model.encode(part)[0])
    assert len(rows) == len(reference.window_spans(len(ids), tiny_model.config.window_width, tiny_model.config.window_stride))
    assert all(row["pooled_gap"] >= 0.0 and 0.0 <= row["same_set"] <= 1.0 for row in rows)


def test_reading_is_the_teachers_calibration(tiny_model: Any, generator: np.random.Generator) -> None:
    """`Calibration.read` equals the teacher's `apply_calibration` on every chunk and window, and the verdict its
    `chunk_uncertainty` on the chunks."""
    from litmus_teacher import calibrate

    table = tiny_model.calibration.table
    for n in (3, 40, 150, 290):
        document, _ = tiny_model.read_whole(generator.integers(0, 50000, n))
        rows = [*document["chunks"], *document["windows"]]
        readings = calibrate.Readings(np.array([r["mean"] for r in rows]), np.array([r["spread"] for r in rows]),
                                      np.array([r["closeness"] for r in rows]), np.array([r["n_tokens"] for r in rows]))
        verdicts = calibrate.apply_calibration(table, readings)
        assert np.array_equal(verdicts.p_values, [r["p_values"] for r in rows])
        assert np.array_equal(verdicts.probability, [r["probability"] for r in rows])
        assert np.array_equal(verdicts.interval, [r["interval"] for r in rows])
        assert np.allclose(verdicts.three_way, [r["three_way"] for r in rows], rtol=0, atol=1e-15)
        assert np.allclose(verdicts.spread_percentile, [r["spread_percentile"] for r in rows], rtol=0, atol=1e-12)
        assert np.array_equal(verdicts.closeness_p_value, [r["closeness_p"] for r in rows])
        assert verdicts.out_of_reference.tolist() == [r["out_of_reference"] for r in rows]
        for level in (0.05, 0.10):
            assert verdicts.not_human(level).tolist() == [r["not_human"][reference.level_key(level)] for r in rows]
        chunks = len(document["chunks"])
        assert chunks == -(-n // tiny_model.config.chunk_tokens)
        expected = calibrate.chunk_uncertainty(table, readings.subset(np.arange(chunks)), reference.MIXED_LEVEL)
        verdict = document["verdict"]
        assert np.allclose(verdict["p_values"], expected.p_values, rtol=0, atol=1e-15)
        assert verdict["probability"] == pytest.approx(expected.probability, abs=1e-15)
        assert np.allclose(verdict["interval"], expected.interval, rtol=0, atol=1e-15)
        assert (verdict["mixed_document"], verdict["model_chunks"], verdict["human_chunks"]) == \
            (expected.mixed_document, expected.model_chunks, expected.human_chunks)
        if chunks == 1:
            only = document["chunks"][0]
            assert verdict["p_values"] == only["p_values"] and verdict["probability"] == only["probability"]
            assert verdict["strip"] == only["strip"] and verdict["sets"] == only["sets"]


def test_explanation_counts_give_the_grid_p_values(tiny_model: Any, generator: np.random.Generator) -> None:
    """The exact p-value `(1 + count) / (n + 1)` from the stored calibration scores rounds up to the reading's grid p-value."""
    grid = tiny_model.calibration.grid
    for n in (2, 9, 30, 70, 140, 300):
        document, _ = tiny_model.read_whole(generator.integers(0, 50000, n))
        for reading in document["chunks"]:
            explain = reading["explain"]
            for origin, entry in enumerate(explain["conformal"]):
                ceiling = next((level for level in grid if entry["p"] <= level), 1.0)
                assert ceiling == reading["p_values"][origin], (entry, reading["p_values"])
            ceiling = next((level for level in grid if explain["closeness"]["p"] <= level), 1.0)
            assert ceiling == reading["closeness_p"]
            cosines = [neighbor["cosine"] for neighbor in explain["nearest"]]
            assert len(cosines) == reference.NEAREST and cosines == sorted(cosines, reverse=True)
            assert set(explain["nearest"][0]) == {"index", "origin", "cosine", "probability", "tokens", *reference.RANDOM_BANK_FIELDS}


def test_bank_reading_by_hand(tiny_model: Any, generator: np.random.Generator) -> None:
    q = generator.standard_normal(tiny_model.config.head_out).astype(np.float32)
    k = generator.standard_normal(tiny_model.config.head_out).astype(np.float32)
    q, k = q / np.linalg.norm(q), k / np.linalg.norm(k)
    reading = tiny_model.bank_reading(q, k)
    logits = tiny_model.pair_scale * (tiny_model.bank_key @ q + tiny_model.bank_query @ k) / 2 + tiny_model.pair_bias  # (m,)
    p = 1 / (1 + np.exp(-logits))
    for origin in range(3):
        assert reading["mean"][origin] == pytest.approx(p[tiny_model.bank_origin == origin].mean(), abs=1e-12)
    vector = np.concatenate([q, k]).astype(np.float64)
    bank = np.concatenate([tiny_model.bank_query, tiny_model.bank_key], axis=1)
    cosines = bank @ vector / np.linalg.norm(bank, axis=1) / np.linalg.norm(vector)
    assert reading["closeness"] == pytest.approx(np.sort(cosines)[-reference.TOP_K :].mean(), abs=1e-12)


def test_litmus_strip() -> None:
    strip = reference.litmus_strip([0.2, 0.5, 0.3], ["ai"], False)
    assert strip["position"] == pytest.approx(0.65) and strip["band_lo"] == pytest.approx(0.65) and strip["band_hi"] == 1.0
    strip = reference.litmus_strip([0.7, 0.1, 0.2], ["human", "mixed"], False)
    assert strip["position"] == pytest.approx(0.2) and (strip["band_lo"], strip["band_hi"]) == (0.0, 0.5)
    strip = reference.litmus_strip([0.3, 0.3, 0.4], [], False)
    assert strip["band_lo"] == strip["band_hi"] == pytest.approx(0.5) and strip["unlike_reference"]
    assert reference.litmus_strip([0.3, 0.3, 0.4], ["ai"], True)["unlike_reference"]


def test_release_config_round_trips(tiny_release: Path) -> None:
    config = json.loads((tiny_release / "config.json").read_text(encoding="utf-8"))
    assert config["format"] == reference.FORMAT and config["calibration"]["format"] == reference.CALIBRATION_FORMAT
    assert reference.StudentConfig.from_release(config) == reference.named_config("tiny")


def test_surprise_by_hand(tiny_model: Any, generator: np.random.Generator) -> None:
    """Each token's log probability is its logit less the log-sum-exp of every logit, from the chunk's final states;
    every chunk is read as a text of its own; the reading's mean, perplexity and percentiles follow (SPEC.md, Surprise)."""
    assert tiny_model.language is not None
    ids = generator.integers(1000, 2000, 150)  # (n,) two chunks of the tiny config
    log_p = tiny_model.token_log_probabilities(ids)  # (n,)
    assert log_p.shape == (150,) and (log_p < 0.0).all()
    chunk = tiny_model.config.chunk_tokens
    bounds = reference.chunk_bounds(150, chunk)
    head, config = tiny_model.language, tiny_model.config
    rows = tiny_model.embed_weight.astype(np.float64) * tiny_model.embed_scale[:, None].astype(np.float64)  # (V, e)
    for start, end in bounds:
        H, _ = tiny_model.encode(ids[start:end])  # (c, d)
        G = reference.rms_norm(reference.bitlinear(H, head.proj_norm, head.proj, config), head.norm, config.norm_eps)  # (c, e)
        L = float(head.scale) * (G @ rows.T) + head.bias  # (c, V)
        total = np.log(np.exp(L - L.max(axis=1, keepdims=True)).sum(axis=1)) + L.max(axis=1)  # (c,)
        assert np.allclose(log_p[start:end], L[np.arange(end - start), ids[start:end]] - total, atol=1e-5)
        assert np.array_equal(tiny_model.token_log_probabilities(ids[start:end]), log_p[start:end])

    reading = tiny_model.surprise(ids)
    assert reading["n_tokens"] == 150 and reading["mean_log_probability"] == pytest.approx(log_p.mean(), abs=1e-12)
    assert reading["perplexity"] == pytest.approx(np.exp(-log_p.mean()))
    assert reading["percentile"] is not None and all(0.0 <= share <= 1.0 for share in reading["percentile"])


def test_surprise_percentile_interpolates_its_quantiles() -> None:
    quantiles = list(np.arange(reference.SURPRISE_QUANTILES, dtype=np.float64))  # 0, 1, ..., 100
    assert reference.surprise_percentile(quantiles, -1.0) == 0.0 and reference.surprise_percentile(quantiles, 101.0) == 1.0
    assert reference.surprise_percentile(quantiles, 25.5) == pytest.approx(0.255)
    values = np.concatenate([np.linspace(-3.0, -1.0, 51), np.linspace(-4.0, -2.0, 51), np.linspace(-2.5, -0.5, 51)])  # (r,)
    table = reference.surprise_table(values, np.repeat([0, 1, 2], 51), "calibration")
    assert table["split"] == "calibration" and table["texts"] == [51, 51, 51]
    assert table["quantiles"]["human"][0] == -3.0 and table["quantiles"]["human"][-1] == -1.0
    reading = reference.surprise_reading(-2.0, 10, table)
    assert reading["percentile"] == pytest.approx([0.5, 1.0, 0.25])
