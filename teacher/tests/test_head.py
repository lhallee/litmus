"""The pair head over three origins, its labels, its loss and its batches."""

from __future__ import annotations

import torch

from pathlib import Path
from litmus_teacher.features import random_rows
from litmus_teacher.head import PairHead, pair_logits, same_origin_loss, same_origin_targets
from litmus_teacher.train import HeadSettings, balanced_draw, choose, origin_members


def test_same_origin_targets_three_origins() -> None:
    rows, columns = torch.tensor([0, 1, 2, 2]), torch.tensor([2, 0, 1])
    expected = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    assert torch.equal(same_origin_targets(rows, columns), expected)


def test_head_outputs_unit_halves_and_symmetric_logits() -> None:
    torch.manual_seed(0)
    head = PairHead(12, (16,), 8, dropout=0.0).eval()
    vectors = head(torch.randn(5, 12))  # (5, 8)
    query, key = vectors.chunk(2, dim=-1)
    assert torch.allclose(query.norm(dim=-1), torch.ones(5)) and torch.allclose(key.norm(dim=-1), torch.ones(5))
    other = head(torch.randn(3, 12))
    assert torch.allclose(head.pair_logits(vectors, other), head.pair_logits(other, vectors).T, atol=1e-6)


def test_loss_matches_binary_cross_entropy() -> None:
    logits = torch.tensor([[2.0, -1.0], [0.5, 30.0]])
    rows, columns = torch.tensor([0, 1]), torch.tensor([0, 1])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    log_sigmoid = torch.nn.functional.logsigmoid
    expected = -(targets * log_sigmoid(logits) + (1 - targets) * log_sigmoid(-logits)).mean()
    assert torch.allclose(same_origin_loss(logits, rows, columns), expected, atol=1e-6)


def test_pair_logits_formula() -> None:
    rows, columns = torch.randn(2, 6), torch.randn(3, 6)
    expected = 4.0 * 0.5 * (rows[:, :3] @ columns[:, 3:].T + rows[:, 3:] @ columns[:, :3].T) - 1.0
    assert torch.allclose(pair_logits(rows, columns, 4.0, -1.0), expected, atol=1e-5)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    head = PairHead(6, (8,), 4, dropout=0.1)
    head.set_input_statistics(torch.arange(6.0), torch.full((6,), 2.0))
    head.save(tmp_path / "head")
    loaded = PairHead.load(tmp_path / "head").eval()
    features = torch.randn(3, 6)
    assert torch.allclose(head.eval()(features), loaded(features))


def test_balanced_draw_takes_a_third_per_origin() -> None:
    labels = torch.tensor([0] * 50 + [1] * 10 + [2] * 5)
    rows, drawn = balanced_draw(origin_members(labels), 30, torch.Generator().manual_seed(0))
    assert torch.equal(labels[rows], drawn)
    assert torch.bincount(drawn).tolist() == [10, 10, 10]


def test_choose_by_mean_over_seeds() -> None:
    records = [{"features": "a", "output_width": 8, "seed": seed, "best_validation_pair_loss": loss}
               for seed, loss in enumerate([0.30, 0.31, 0.32])]
    records += [{"features": "b", "output_width": 8, "seed": seed, "best_validation_pair_loss": loss}
                for seed, loss in enumerate([0.29, 0.40, 0.41])]
    records.append({"features": "random", "output_width": 8, "seed": 0, "best_validation_pair_loss": 0.1})
    chosen = choose(records)
    assert chosen["chosen"] == "a|w8" and chosen["head"]["seed"] == 0
    assert HeadSettings("a+b", 8, 1).name == "a_b_w8_s1"


def test_random_rows_depend_on_id_only() -> None:
    ids = torch.tensor([5, 9000, 3]).numpy()
    together = random_rows(ids, 16, seed=1)
    alone = random_rows(ids[1:2], 16, seed=1)
    assert (together[1] == alone[0]).all() and not (together[0] == together[2]).all()
