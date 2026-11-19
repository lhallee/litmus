"""The per-token loss: the teacher's states land on the student's tokens, and the dense proof of concept runs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathlib import Path
from types import SimpleNamespace
from torch import nn

from litmus_student import poc as poc_module
from litmus_student.contract import reference
from litmus_student.data import pack, packed_batch
from litmus_student.dense import CLS_ID, PAD_ID, SEP_ID, TokenTeacher, length_groups, token_loss
from litmus_student.model import ATTENTION_WEIGHT_LEVELS, StudentModel, integer_block_attention
from litmus_student.poc import parser, poc

from test_poc import stand_ins


WIDTH = 6


class MarkedEncoder(nn.Module):
    """A stand-in teacher whose state at a position names its token, its column and the row's length, so a state
    gathered from the wrong place is seen."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=WIDTH)
        self.calls: list[tuple[int, int]] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        # input_ids, attention_mask: (g, l)
        rows, width = input_ids.shape
        self.calls.append((rows, width))
        assert (input_ids[:, 0] == CLS_ID).all()
        lengths = attention_mask.sum(dim=1)  # (g,)
        assert (input_ids[torch.arange(rows), lengths - 1] == SEP_ID).all()
        assert ((input_ids == PAD_ID) == (attention_mask == 0)).all()
        states = torch.zeros(rows, width, WIDTH)  # (g, l, D)
        states[..., 0] = input_ids.float()
        states[..., 1] = torch.arange(width).float()[None, :]
        states[..., 2] = lengths.float()[:, None]

        return SimpleNamespace(last_hidden_state=states)


def test_length_groups_partition_the_texts_within_the_budget() -> None:
    lengths = torch.tensor([5, 90, 7, 30, 6, 28])
    groups = length_groups(lengths, budget=64)

    assert sorted(torch.cat(groups).tolist()) == list(range(6))
    for group in groups:
        assert len(group) == 1 or len(group) * (int(lengths[group].max()) + 2) <= 64


def test_teacher_states_land_on_their_tokens() -> None:
    generator = np.random.default_rng(0)
    config = reference.named_config("tiny")
    texts = [generator.integers(5, 900, size=int(length)) for length in generator.integers(1, 60, size=40)]
    lengths = np.asarray([len(ids) for ids in texts])
    encoder = MarkedEncoder()
    teacher = TokenTeacher(encoder, WIDTH, budget=128)
    for plan in pack(lengths, 256, config.block_size):
        batch = packed_batch(plan, texts, config.block_size, config.block_size)
        states = teacher.states(batch)  # (t_v, D)
        valid = batch.segment >= 0
        assert states.shape == (int(valid.sum()), WIDTH)
        assert torch.equal(states[:, 0], batch.tokens[valid].float())  # the token's own state
        assert torch.equal(states[:, 1], batch.position[valid].float() + 1)  # one past [CLS]
        own = torch.bincount(batch.segment[valid], minlength=batch.texts)[batch.segment[valid]]  # (t_v,)
        assert torch.equal(states[:, 2], own.float() + 2)  # read in its own text's row
    assert len(encoder.calls) > 1 and all(rows == 1 or rows * width <= 128 for rows, width in encoder.calls)


def test_token_loss_is_zero_at_the_target_and_has_a_gradient() -> None:
    torch.manual_seed(0)
    states = torch.randn(10, 4, requires_grad=True)
    segment = torch.tensor([0, 0, 0, 1, 1, -1, -1, 2, 2, 2])
    projection = nn.Linear(4, WIDTH)
    targets = projection(states[segment >= 0]).detach()

    assert float(token_loss(states, segment, targets, projection)) == pytest.approx(0.0, abs=1e-6)
    loss = token_loss(states, segment, targets + 1.0, projection)
    loss.backward()
    assert float(loss) > 0.5 and states.grad is not None
    assert float(states.grad[segment < 0].abs().sum()) == 0.0 and float(states.grad[segment >= 0].abs().sum()) > 0.0


def test_integer_attention_products_are_integer_products() -> None:
    """Scores are integer dot products times the two tokens' scales, and outputs are integer sums of weights in
    127ths times int8 values times one scale per block and head."""
    torch.manual_seed(0)
    b, h, W, d_h = 2, 3, 8, 16
    Q, K, V = torch.randn(b, h, W, d_h), torch.randn(b, h, W, d_h), torch.randn(b, h, W, d_h)
    mask = torch.ones(b, 1, W, W, dtype=torch.bool)
    mask[..., -2:] = False
    mask |= torch.eye(W, dtype=torch.bool)
    out = integer_block_attention(Q, K, V, mask, 1e-5, training=False)  # (b, h, W, d_h)

    def integers(X: torch.Tensor, dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        scale = 127.0 / X.abs().amax(dim=dims, keepdim=True)
        return (X * scale).round(), scale

    (Q_i, s_q), (K_i, s_k), (V_i, s_v) = integers(Q, (-1,)), integers(K, (-1,)), integers(V, (-2, -1))
    scores = (Q_i @ K_i.transpose(-1, -2)) / (s_q * s_k.transpose(-1, -2)) / d_h**0.5
    weights = (torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) * ATTENTION_WEIGHT_LEVELS).round()
    assert float(weights[..., :-2, -2:].abs().sum()) == 0.0  # masked keys get no weight
    expected = (weights @ V_i) / (ATTENTION_WEIGHT_LEVELS * s_v)
    assert torch.allclose(out, expected, atol=1e-5)

    Q.requires_grad_(True)
    V.requires_grad_(True)
    integer_block_attention(Q, K, V, mask, 1e-5, training=True).square().sum().backward()
    assert float(Q.grad.abs().sum()) > 0 and float(V.grad.abs().sum()) > 0


def test_integer_attention_model_trains_and_reloads(tmp_path: Path) -> None:
    config = reference.named_config("tiny")
    torch.manual_seed(0)
    model = StudentModel(config, integer_attention=True)
    generator = np.random.default_rng(0)
    texts = [generator.integers(5, 900, size=int(length)) for length in generator.integers(4, 40, size=12)]
    lengths = np.asarray([len(ids) for ids in texts])
    batch = packed_batch(pack(lengths, 256, config.block_size)[0], texts, config.block_size, config.block_size)
    model.train()
    model(batch).square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.encoder.layers[0].q.parameters())
    model.eval()
    plain = StudentModel(config)
    plain.load_state_dict(model.state_dict())
    plain.eval()
    with torch.no_grad():
        assert not torch.allclose(model(batch), plain(batch), atol=1e-6)  # the same weights read differently


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_integer_attention_proof_of_concept_runs_end_to_end(tmp_path: Path) -> None:
    corpus, embeddings, checker = stand_ins(tmp_path)
    report = poc(parser().parse_args([
        "--corpus", str(corpus), "--embeddings", str(embeddings), "--checker", str(checker), "--out", str(tmp_path / "poc"),
        "--experiment-id", "2026-11-17_student_poc_test", "--rows", "300", "--workers", "2", "--small", "tiny", "--middle", "tiny",
        "--texts-per-step", "24", "--micro-tokens", "256", "--pack-multiple", "256", "--warmup", "2", "--bank-per-origin", "20",
        "--steps", "6", "--eval-steps", "3", "--device", "cpu", "--pilot", "--integer-attention", "--report", "integer_report.json",
    ]))

    assert set(report["arms"]) == {"ternary_small", "ternary_small_integer"}
    assert report["verdict"]["call"] == "integer attention comparison" and (tmp_path / "poc" / "integer_report.json").exists()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_dense_proof_of_concept_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus, embeddings, checker = stand_ins(tmp_path)
    monkeypatch.setattr(poc_module, "load_teacher_encoder", lambda device: MarkedEncoder())
    report = poc(parser().parse_args([
        "--corpus", str(corpus), "--embeddings", str(embeddings), "--checker", str(checker), "--out", str(tmp_path / "poc"),
        "--experiment-id", "2026-11-17_student_poc_test", "--rows", "300", "--workers", "2", "--small", "tiny", "--middle", "tiny",
        "--texts-per-step", "24", "--micro-tokens", "256", "--pack-multiple", "256", "--warmup", "2", "--bank-per-origin", "20",
        "--steps", "6", "--eval-steps", "3", "--device", "cpu", "--pilot", "--dense-weight", "1.0", "--report", "dense_report.json",
    ]))

    assert set(report["arms"]) == {"ternary_small", "ternary_small_dense"}
    assert report["verdict"]["call"] == "dense comparison" and report["token_teacher"]["tokens_per_second"] > 0
    assert (tmp_path / "poc" / "dense_report.json").exists() and not (tmp_path / "poc" / "poc_report.json").exists()
