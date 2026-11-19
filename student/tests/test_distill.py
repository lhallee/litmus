"""The distillation losses on synthetic vectors, and the gradient-cached step against a direct full-graph step."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pathlib import Path
from types import SimpleNamespace
from litmus_student.contract import reference
from litmus_student.data import Step, pack, packed_batch
from litmus_student.distill import LossWeights, Projections, distillation_losses, pair_logits
from litmus_student.model import StudentModel
from litmus_student.search import DEFAULT_PARAMS, SCHEDULE, loss_weights
from litmus_student.train import learning_rate, parser, scheduled_step, searched_settings, train_step


def unit_pairs(b: int, w: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    query, key = torch.randn(b, w, generator=generator), torch.randn(b, w, generator=generator)  # (b, w) each
    return torch.cat([F.normalize(query, dim=-1), F.normalize(key, dim=-1)], dim=-1)  # (b, 2w)


def losses_for(query_key: torch.Tensor, teacher_query_key: torch.Tensor, labels: torch.Tensor, scale: float = 10.0) -> dict[str, torch.Tensor]:
    # query_key: (b, 2w); teacher_query_key: (b, 2w_T); labels: (b,)
    b, w = query_key.shape[0], query_key.shape[1] // 2
    projections = Projections(8, 6, w, teacher_query_key.shape[1] // 2)
    standardized = torch.randn(b, 8, generator=torch.Generator().manual_seed(1))  # (b, 2d)
    teacher_embeddings = torch.randn(b, 6, generator=torch.Generator().manual_seed(2))  # (b, 2D)

    return distillation_losses(standardized, query_key, torch.tensor(scale), torch.tensor(0.0), labels, teacher_embeddings,
                               teacher_query_key, scale, 0.0, projections, LossWeights())  # name to ()


def test_pair_logits_are_symmetric() -> None:
    vectors = unit_pairs(7, 5, seed=0)
    logits = pair_logits(vectors, vectors, 3.0, -1.0)  # (b, b)
    assert torch.allclose(logits, logits.T)
    query, key = vectors.chunk(2, dim=-1)
    assert torch.isclose(logits[1, 2], 3.0 * 0.5 * (query[1] @ key[2] + query[2] @ key[1]) - 1.0)


def test_soft_and_rank_losses_have_no_gradient_at_the_teacher() -> None:
    teacher = unit_pairs(16, 4, seed=3)
    student = teacher.clone().requires_grad_(True)
    labels = torch.arange(16) % 3
    losses = losses_for(student, teacher, labels)
    (losses["pair_soft"] + losses["pair_rank"]).backward()
    assert student.grad is not None and student.grad.abs().max() < 1e-6
    for name in LossWeights().names():
        assert torch.isfinite(losses[name])


def test_label_loss_prefers_origin_structure() -> None:
    labels = torch.arange(30) % 3
    centers = unit_pairs(3, 4, seed=4)  # (3, 2w)
    structured = F.normalize(centers[labels] + 0.05 * unit_pairs(30, 4, seed=5), dim=-1)  # (b, 2w), close within an origin
    scrambled = unit_pairs(30, 4, seed=6)
    teacher = unit_pairs(30, 4, seed=7)
    assert losses_for(structured, teacher, labels)["pair_label"] < losses_for(scrambled, teacher, labels)["pair_label"]


def test_gradient_cached_step_equals_the_direct_step() -> None:
    """train_step's two-pass gradient equals back-propagating the same losses through one graph of every micro-batch."""
    torch.manual_seed(0)
    config = reference.named_config("tiny")
    generator = np.random.default_rng(1)
    lengths = np.array([3, 30, 9, 4, 17, 50, 2, 8, 12])  # (b,)
    texts = [generator.integers(0, 50368, n) for n in lengths]
    plans = pack(lengths, budget=64, block_size=config.block_size)
    assert len(plans) > 1, "the test needs several micro-batches"
    rows = np.asarray([placement.row for plan in plans for placement in plan])  # (b,)
    batches = [packed_batch(plan, texts, config.block_size, config.block_size) for plan in plans]
    labels = torch.from_numpy(rows % 3)
    teacher_embeddings = torch.randn(len(rows), 6)  # (b, 2D)
    teacher_query_key = unit_pairs(len(rows), 5, seed=8)  # (b, 2w_T)
    step = Step(rows, batches, labels, teacher_embeddings, teacher_query_key, int(lengths.sum()))
    teacher = SimpleNamespace(scale=12.0, bias=-0.5)

    model = StudentModel(config)
    projections = Projections(2 * config.width, 6, config.head_out, 5)
    direct_model = StudentModel(config)
    direct_model.load_state_dict(model.state_dict())
    direct_projections = Projections(2 * config.width, 6, config.head_out, 5)
    direct_projections.load_state_dict(projections.state_dict())

    train_step(model, projections, step, teacher, LossWeights(), torch.device("cpu"))

    direct_model.train()
    pooled = torch.cat([direct_model.pooled(batch) for batch in batches])  # (b, 2d)
    standardized = direct_model.head.standardize(pooled)  # (b, 2d)
    losses = distillation_losses(standardized, direct_model.head.project(standardized), direct_model.head.scale, direct_model.head.bias,
                                 labels, teacher_embeddings, teacher_query_key, 12.0, -0.5, direct_projections, LossWeights())
    losses["total"].backward()

    for (name, cached), (_, direct) in zip(model.named_parameters(), direct_model.named_parameters(), strict=True):
        assert cached.grad is not None and direct.grad is not None, name
        assert torch.allclose(cached.grad, direct.grad, rtol=1e-4, atol=1e-6), name
    for cached, direct in zip(projections.parameters(), direct_projections.parameters(), strict=True):
        assert torch.allclose(cached.grad, direct.grad, rtol=1e-4, atol=1e-6)


def test_searched_settings_replace_only_what_the_search_took(tmp_path: Path) -> None:
    best = {**DEFAULT_PARAMS, "learning_rate": 4e-3, "second_ratio": 0.5, "warmup_share": 0.1, "weight_decay": 0.02, "decay_drops": False,
            "texts_per_step": 512, "head_hidden": 1024, "head_out": 64, "head_clamp": 5.0, "log_embedding": 1.0, "temperature": 2.0}
    base = ["--config", "tiny", "--data", "d", "--teacher-export", "t", "--experiment-id", "e", "--out", "o", "--steps", "1000"]
    config = reference.named_config("tiny")
    summary = tmp_path / "summary.json"

    def read(taken: list[str]) -> tuple:
        summary.write_text(json.dumps({"recommendation": {"settings": best, "taken": taken}}), encoding="utf-8")
        arguments = parser().parse_args([*base, "--settings", str(summary)])
        return arguments, *searched_settings(arguments, config)

    arguments, kept, weights, taken = read([])
    assert kept == config and weights == LossWeights() and taken == [] and arguments.lr == 1.5e-3 and not arguments.decay_stays

    arguments, changed, weights, _ = read([name for name in DEFAULT_PARAMS if name not in SCHEDULE])  # loss and head only
    assert (changed.head_hidden, changed.head_out, changed.head_clamp) == (1024, 64, 5.0) and weights == loss_weights(best)
    assert weights.temperature == 2.0 and arguments.lr == 1.5e-3 and arguments.texts_per_step == 2048

    arguments, _, _, _ = read(list(DEFAULT_PARAMS))  # the schedule too
    assert arguments.lr == 4e-3 and arguments.lr_second == pytest.approx(2e-3) and arguments.warmup == 100
    assert arguments.weight_decay == 0.02 and arguments.decay_stays and arguments.texts_per_step == 512


def test_time_box_moves_the_schedule_with_the_clock() -> None:
    assert scheduled_step(40, 1000, elapsed=500.0, box=0.0) == 40  # no box: the step itself
    assert scheduled_step(40, 1000, elapsed=30.0, box=60.0) == 500  # half the box gone, half the schedule
    assert scheduled_step(700, 1000, elapsed=30.0, box=60.0) == 700  # steps ahead of the clock keep their place
    assert scheduled_step(40, 1000, elapsed=90.0, box=60.0) == 1000  # a closed box is the end of the schedule
    assert learning_rate(scheduled_step(40, 1000, 90.0, 60.0), 1000, 10, 1e-3, 1e-3) == 0.0
