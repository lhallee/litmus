"""The PyTorch student in eval mode against reference.py, and packed batches against single texts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from litmus_student.contract import reference
from litmus_student.data import pack, packed_batch
from litmus_student.export import model_tensors
from litmus_student.model import StudentModel, single_text_batch


def random_student(name: str, seed: int) -> StudentModel:
    """A random student with norms, gains and head statistics moved off their initial values, in eval mode."""
    torch.manual_seed(seed)
    model = StudentModel(reference.named_config(name))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.add_(0.1 * torch.randn_like(parameter))
        model.head.running_mean.normal_(0.0, 0.1)
        model.head.running_var.uniform_(0.5, 2.0)

    return model.eval()


def reference_of(model: StudentModel) -> object:
    tensors = model_tensors(model)
    generator = np.random.default_rng(0)
    bank = generator.standard_normal((30, 2, model.config.head_out))  # (m, 2, w)
    bank /= np.linalg.norm(bank, axis=-1, keepdims=True)
    tensors["bank.vectors"] = bank.reshape(30, -1).astype(np.float16)  # (m, 2w)
    tensors["bank.origin"] = np.repeat(np.arange(3), 10).astype(np.uint8)  # (m,)

    return reference.ReferenceModel(model.config, tensors, None)


POOLED_TOLERANCE = 3e-3  # SPEC.md, Parity: abs <= 3e-3 * (1 + abs(value))


@pytest.mark.parametrize(("name", "lengths"), [("tiny", (1, 5, 8, 9, 16, 55, 300)), ("s192_l4", (1, 128, 129, 256, 1500))])
def test_eval_mode_equals_reference(name: str, lengths: tuple[int, ...]) -> None:
    model = random_student(name, seed=1)
    loaded = reference_of(model)
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
        assert (np.abs(mine_pooled[0].numpy() - pooled) / (1 + np.abs(pooled))).max() <= POOLED_TOLERANCE
        w = model.config.head_out
        assert np.abs(mine[:w] - query).max() <= 1e-3 and np.abs(mine[w:] - key).max() <= 1e-3


def test_packed_texts_equal_texts_alone() -> None:
    """Many texts in one stream, long and short, give each text's own result: the mask and the alignment hold."""
    model = random_student("tiny", seed=4)
    W = model.config.block_size
    generator = np.random.default_rng(5)
    lengths = np.array([1, 3, 8, 9, 17, 40, 5, 2, 64, 7, 8, 30])  # (b,)
    texts = [generator.integers(0, model.config.vocab_size, n) for n in lengths]
    plans = pack(lengths, budget=128, block_size=W)
    with torch.no_grad():
        for plan in plans:
            together = model.pooled(packed_batch(plan, texts, W, W))  # (n, 2d)
            for local, placement in enumerate(plan):
                alone = model.pooled(single_text_batch(texts[placement.row], W))[0]  # (2d,)
                assert torch.allclose(together[local], alone, atol=1e-5, rtol=1e-5), placement


def test_training_forward_is_close_to_eval() -> None:
    """The straight-through forward computes the same values as the integer path, up to float rounding."""
    model = random_student("tiny", seed=6)
    batch = single_text_batch(np.random.default_rng(7).integers(0, 50000, 100), model.config.block_size)
    with torch.no_grad():
        evaluated = model.encoder(batch)[:100]  # (n, d)
        model.train()
        trained = model.encoder(batch)[:100]  # (n, d)
    assert (trained - evaluated).abs().max() < 0.05
    assert (trained - evaluated).abs().mean() < 1e-3
