"""Integer attention (SPEC.md, Integer attention): the reference's arithmetic by hand, its parity with the PyTorch
prototype in eval mode, streaming, and the release file's format 2."""

from __future__ import annotations

import json
import math
import numpy as np
import pytest
import torch

from pathlib import Path
from litmus_student.contract import reference
from litmus_student.export import model_tensors, with_attention
from litmus_student.model import ATTENTION_WEIGHT_LEVELS, StudentModel, single_text_batch


def test_integer_attention_is_integer_products_by_hand(generator: np.random.Generator) -> None:
    """Scores are integer dot products times two float32 factors, weights whole 127ths, outputs integer sums of
    weights times int8 values times one scale per block and head."""
    h, n, d_h = 2, 11, 8
    Q, K, V = (generator.standard_normal((h, n, d_h)).astype(np.float32) for _ in range(3))
    out = reference.integer_attention(Q, K, V, 1e-5)  # (h, n, d_h)
    assert out.dtype == np.float32 and reference.ATTENTION_WEIGHT_LEVELS == ATTENTION_WEIGHT_LEVELS == 127

    for head in range(h):
        s_q = np.float32(127.0) / np.abs(Q[head]).max(axis=1)  # (n,)
        s_k = np.float32(127.0) / np.abs(K[head]).max(axis=1)  # (n,)
        s_v = np.float32(127.0) / np.abs(V[head]).max()  # ()
        q = np.clip(np.rint(Q[head] * s_q[:, None]), -127, 127).astype(np.int64)  # (n, d_h)
        k = np.clip(np.rint(K[head] * s_k[:, None]), -127, 127).astype(np.int64)  # (n, d_h)
        v = np.clip(np.rint(V[head] * s_v), -127, 127).astype(np.int64)  # (n, d_h)
        for i in range(n):
            scores = np.empty(n, dtype=np.float32)  # (n,)
            x = np.float32(np.float32(1.0 / math.sqrt(d_h)) / s_q[i])
            for j in range(n):
                scores[j] = np.float32(np.float32(np.float32(int(q[i] @ k[j])) * x) * np.float32(np.float32(1.0) / s_k[j]))
            e = reference.exp_nonpositive(scores - scores.max())  # (n,)
            r = np.float32(127.0 / sum(float(value) for value in e))  # the float64 sum, then one rounding
            weights = np.rint(e * r)  # (n,)
            assert weights.min() >= 0 and weights.max() <= 127
            sums = weights.astype(np.int64) @ v  # (d_h,), exact
            t = np.float32(1.0) / (np.float32(127.0) * s_v)
            assert np.array_equal(out[head, i], sums.astype(np.float32) * t)


def test_exp_is_close_and_exact_by_its_definition() -> None:
    x = -np.linspace(0.0, 90.0, 20001, dtype=np.float32)  # (k,)
    got = reference.exp_nonpositive(x)  # (k,)
    assert got.dtype == np.float32 and got[0] == np.float32(1.0)
    want = np.exp(x.astype(np.float64))  # (k,)
    inside = x > -87.0
    assert (np.abs(got[inside] - want[inside]) <= 4e-7 * want[inside]).all()
    # One value step by step, in Python floats rounded to float32 after every operation.
    f = np.float32
    value = f(-3.3)
    shifted = f(f(value * reference.EXP_LOG2E) + reference.EXP_MAGIC)
    n = f(shifted - reference.EXP_MAGIC)
    rest = f(f(value - f(n * reference.EXP_C1)) - f(n * reference.EXP_C2))
    p = reference.EXP_POLYNOMIAL[0]
    for coefficient in reference.EXP_POLYNOMIAL[1:]:
        p = f(f(p * rest) + coefficient)
    p = f(f(f(p * f(rest * rest)) + rest) + f(1.0))
    assert reference.exp_nonpositive(np.asarray([value]))[0] == f(p * f(2.0 ** int(n)))


def test_integer_attention_is_close_to_float_attention(generator: np.random.Generator) -> None:
    """Rounding to 8 bits and 127ths moves outputs by a few percent of the values' range, no more."""
    h, n, d_h = 3, 64, 64
    Q, K, V = (generator.standard_normal((h, n, d_h)).astype(np.float32) for _ in range(3))
    S = (Q @ K.transpose(0, 2, 1)) / np.float32(8.0)  # (h, n, n)
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P /= P.sum(axis=-1, keepdims=True)
    exact = P @ V  # (h, n, d_h)
    assert np.abs(reference.integer_attention(Q, K, V, 1e-5) - exact).max() < 0.05 * np.abs(V).max()


def integer_student(name: str, seed: int) -> StudentModel:
    """A random integer-attention student with norms, gains and head statistics moved off their initial values."""
    torch.manual_seed(seed)
    model = StudentModel(reference.named_config(name), integer_attention=True)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.add_(0.1 * torch.randn_like(parameter))
        model.head.running_mean.normal_(0.0, 0.1)
        model.head.running_var.uniform_(0.5, 2.0)

    return model.eval()


def reference_of(model: StudentModel, attention: str) -> reference.ReferenceModel:
    tensors = model_tensors(model)
    bank = np.random.default_rng(0).standard_normal((30, 2, model.config.head_out))  # (m, 2, w)
    bank /= np.linalg.norm(bank, axis=-1, keepdims=True)
    tensors["bank.vectors"] = bank.reshape(30, -1).astype(np.float16)  # (m, 2w)
    tensors["bank.origin"] = np.repeat(np.arange(3), 10).astype(np.uint8)  # (m,)

    return reference.ReferenceModel(model.config, tensors, None, attention=attention)


def test_eval_attention_is_the_reference_bit_for_bit(generator: np.random.Generator) -> None:
    """Given the same queries, keys and values, the model's eval path gives the reference's bits: in a block of one
    text with padding (the whole-block scale), and in a packed block of three texts with `value_max` set per text."""
    from litmus_student.model import integer_block_attention

    h, W, d_h = 2, 16, 8
    Q, K, V = (generator.standard_normal((h, W, d_h)).astype(np.float32) for _ in range(3))
    n = 11  # one text, then padding rows of zeros that attend only to themselves
    Q[:, n:], K[:, n:], V[:, n:] = 0.0, 0.0, 0.0
    mask = torch.zeros(1, 1, W, W, dtype=torch.bool)
    mask[..., :n, :n] = True
    mask |= torch.eye(W, dtype=torch.bool)
    with torch.no_grad():
        mine = integer_block_attention(*(torch.from_numpy(U)[None] for U in (Q, K, V)), mask, 1e-5, training=False)[0].numpy()
    assert np.array_equal(mine[:, :n], reference.integer_attention(Q[:, :n], K[:, :n], V[:, :n], 1e-5))

    spans = [(0, 5), (5, 8), (8, 16)]  # three texts packed in one block
    Q, K, V = (generator.standard_normal((h, W, d_h)).astype(np.float32) * 3 for _ in range(3))
    mask = torch.zeros(1, 1, W, W, dtype=torch.bool)
    for start, stop in spans:
        mask[..., start:stop, start:stop] = True
    V_t = torch.from_numpy(V)[None]  # (1, h, W, d_h)
    value_max = torch.where(mask, V_t.abs().amax(dim=-1)[:, :, None, :], 0.0).amax(dim=-1, keepdim=True)  # (1, h, W, 1)
    with torch.no_grad():
        mine = integer_block_attention(torch.from_numpy(Q)[None], torch.from_numpy(K)[None], V_t, mask, 1e-5, False, value_max)[0].numpy()
    for start, stop in spans:
        assert np.array_equal(mine[:, start:stop], reference.integer_attention(Q[:, start:stop], K[:, start:stop], V[:, start:stop], 1e-5))


# Lengths of SPEC.md's Parity measurement. Shorter texts of random models exceed the tolerances now and then in both
# modes, more often with integer attention (SPEC.md, Parity, gives the rates): one rounding that lands on the other side
# of a half, from float sums run in another order, moves a short text's pooled vector by more than 3e-3.
@pytest.mark.parametrize(("name", "lengths"), [("tiny", (300,)), ("s192_l4", (1500,))])
def test_eval_mode_equals_reference(name: str, lengths: tuple[int, ...]) -> None:
    """The model's integer attention in eval mode and the reference agree as SPEC.md's Parity asks."""
    model = integer_student(name, seed=1)
    loaded = reference_of(model, "integer")
    generator = np.random.default_rng(2)
    for n in lengths:
        ids = generator.integers(0, model.config.vocab_size, n)  # (n,)
        H, _ = loaded.encode(ids)  # (n, d)
        pool = loaded.chunk_pool(n)
        pool.add(H)
        pooled = pool.pooled(0, n)  # (2d,)
        query, key = loaded.head(pooled)  # (w,) each
        with torch.no_grad():
            mine_pooled = model.pooled(single_text_batch(ids, model.config.block_size))  # (1, 2d)
            mine = model.head(mine_pooled)[0].numpy()  # (2w,)
        assert (np.abs(mine_pooled[0].numpy() - pooled) / (1 + np.abs(pooled))).max() <= 3e-3
        w = model.config.head_out
        assert np.abs(mine[:w] - query).max() <= 1e-3 and np.abs(mine[w:] - key).max() <= 1e-3


def test_a_float_trained_model_reads_with_integer_attention() -> None:
    """The post-training switch: the same weights, read with integer attention by the model and the reference."""
    torch.manual_seed(3)
    model = StudentModel(reference.named_config("tiny")).eval()
    with_attention(model, "integer")
    assert model.integer_attention and all(layer.integer_attention for layer in model.encoder.layers)
    loaded = reference_of(model, "integer")
    ids = np.random.default_rng(4).integers(0, model.config.vocab_size, 70)  # (n,)
    H, _ = loaded.encode(ids)  # (n, d)
    with torch.no_grad():
        mine = model.encoder(single_text_batch(ids, model.config.block_size))[:70].numpy()  # (n, d)
    # An 8-bit attention code that rounds the other way on another processor's float order moves a weight by 1/127;
    # the largest mean difference measured is 0.0033 (aarch64 GH200, 2026-11-19), under 0.003 on x86-64.
    assert np.abs(mine.mean(axis=0) - H.mean(axis=0)).max() <= 4e-3
    with_attention(model, "float")
    assert not model.integer_attention and not any(layer.integer_attention for layer in model.encoder.layers)


@pytest.fixture(scope="module")
def integer_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tiny_integer_release")
    reference.init_release(reference.named_config("tiny"), directory, seed=3, bank_per_origin=8, calibration_tokens=200, attention="integer")
    return directory


def test_integer_release_is_format_2_and_float_release_is_unchanged(integer_release: Path, tiny_release: Path) -> None:
    integer_config = json.loads((integer_release / "config.json").read_text(encoding="utf-8"))
    float_config = json.loads((tiny_release / "config.json").read_text(encoding="utf-8"))
    assert integer_config["format_version"] == reference.INTEGER_ATTENTION_FORMAT_VERSION == 2
    assert integer_config["model"]["attention"] == "integer"
    assert float_config["format_version"] == reference.FORMAT_VERSION == 1 and "attention" not in float_config["model"]
    assert reference.StudentConfig.from_release(integer_config) == reference.StudentConfig.from_release(float_config)
    assert reference.read_safetensors(integer_release / "model.safetensors").keys() == reference.read_safetensors(tiny_release / "model.safetensors").keys()
    assert reference.ReferenceModel.load(integer_release).attention == "integer"
    assert reference.ReferenceModel.load(tiny_release).attention == "float"


def test_release_attention_refuses_what_it_cannot_read(tiny_release: Path) -> None:
    config = json.loads((tiny_release / "config.json").read_text(encoding="utf-8"))
    assert reference.release_attention(config) == "float"
    with pytest.raises(ValueError, match="needs format 2"):
        reference.release_attention({**config, "model": {**config["model"], "attention": "integer"}})
    with pytest.raises(ValueError, match="names model.attention"):
        reference.release_attention({**config, "format_version": 2})
    with pytest.raises(ValueError, match="not a litmus-student release"):
        reference.release_attention({**config, "format_version": 4})
    assert reference.release_attention({**config, "format_version": 2, "model": {**config["model"], "attention": "float"}}) == "float"


def test_integer_streaming_equals_whole(integer_release: Path, generator: np.random.Generator) -> None:
    model = reference.ReferenceModel.load(integer_release)
    config = model.config
    W, C = config.block_size, config.chunk_tokens
    for n in (1, W + 1, C + 1, 211):
        ids = generator.integers(0, config.vocab_size, n)  # (n,)
        whole, vectors = model.read_whole(ids)
        for piece in (1, 7, W - 1, W + 1, n):
            streamed, streamed_vectors = model.read_streaming(ids, piece)
            for mine, theirs in zip(streamed_vectors["chunks"], vectors["chunks"], strict=True):
                assert np.abs(mine["pooled"] - theirs["pooled"]).max() <= 1e-6
            assert streamed["verdict"]["p_values"] == whole["verdict"]["p_values"]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_float_trained_checkpoint_exports_an_integer_release(tmp_path: Path) -> None:
    """`--attention integer` on a checkpoint trained with float attention: format 2, the bank and calibration read with
    integer attention, and export's own check of the reference against the model passes."""
    from litmus_student.export import export_release
    from litmus_student.train import parser, train
    from test_export import synthetic_corpus

    store, corpus, teacher = synthetic_corpus(tmp_path)
    train(parser().parse_args([
        "--config", "tiny", "--data", str(store), "--teacher-export", str(teacher), "--experiment-id", "2026-11-17_student_test",
        "--out", str(tmp_path / "run"), "--device", "cpu", "--pilot", "--pilot-steps", "150", "--steps", "150", "--lr", "4e-3",
        "--lr-second", "3e-3", "--warmup", "3", "--texts-per-step", "60", "--micro-tokens", "256", "--pack-multiple", "256",
        "--eval-every", "75", "--val-texts", "60", "--eval-per-origin", "30", "--bank-per-origin", "30", "--max-train-tokens", "96",
    ]))  # test_export's pilot run
    checkpoint = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    assert not checkpoint["integer_attention"]

    exported = export_release(tmp_path / "run" / "best.pt", store, corpus, tmp_path / "release", bank_per_origin=40, check_texts=8,
                              micro_tokens=256, device=torch.device("cpu"), attention="integer")
    config = json.loads((tmp_path / "release" / "config.json").read_text(encoding="utf-8"))
    assert exported["attention"] == "integer" and config["format_version"] == 2 and config["model"]["attention"] == "integer"
    assert config["provenance"]["attention"] == {"release": "integer", "trained": "float"}
    assert exported["parity"]["pooled"] <= 3e-3 and exported["parity"]["query_key"] <= 1e-3
    loaded = reference.ReferenceModel.load(tmp_path / "release")
    assert loaded.attention == "integer"
    reading, _ = loaded.read_whole(np.random.default_rng(1).integers(1000, 2000, 120))
    assert len(reading["chunks"]) == 2

    float_export = export_release(tmp_path / "run" / "best.pt", store, corpus, tmp_path / "float_release", bank_per_origin=20,
                                  check_texts=4, micro_tokens=256, device=torch.device("cpu"))
    float_config = json.loads((tmp_path / "float_release" / "config.json").read_text(encoding="utf-8"))
    assert float_export["attention"] == "float" and float_config["format_version"] == 1
    assert "attention" not in float_config["model"] and "attention" not in float_config["provenance"]


def test_integer_and_float_releases_of_the_same_weights_differ_a_little(integer_release: Path, tiny_release: Path) -> None:
    """The same random weights (both releases come from seed 3): the reading moves, but not far."""
    ids = np.random.default_rng(8).integers(0, 50000, 60)  # (n,)
    _, integer_vectors = reference.ReferenceModel.load(integer_release).read_whole(ids)
    _, float_vectors = reference.ReferenceModel.load(tiny_release).read_whole(ids)
    integer_pooled, float_pooled = integer_vectors["chunks"][0]["pooled"], float_vectors["chunks"][0]["pooled"]
    gap = np.abs(integer_pooled - float_pooled) / (1 + np.abs(float_pooled))
    assert 0 < gap.max() < 0.2
