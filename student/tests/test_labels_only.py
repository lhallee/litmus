"""A labels-only run needs no teacher file; distillation still does."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from pathlib import Path
from litmus_student.data import ClusterSampler, TeacherFiles, TokenStore, teacher_targets
from litmus_student.model import PackedBatch, mask_stream
from litmus_student.train import load_checkpoint, parser, train


def arguments(store: Path, out: Path, *extra: str) -> list[str]:
    return ["--data", str(store), "--out", str(out), "--experiment-id", "2026-11-18_labels_only_test", "--config", "tiny",
            "--device", "cpu", "--steps", "4", "--warmup", "0", "--texts-per-step", "30", "--micro-tokens", "256",
            "--pack-multiple", "256", "--eval-every", "2", "--val-texts", "30", "--eval-per-origin", "20", "--bank-per-origin", "20",
            "--max-train-tokens", "96", "--pilot", "--pilot-steps", "4", *extra]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_labels_only_trains_without_the_teacher_files(tmp_path: Path, synthetic_corpus: tuple[Path, Path]) -> None:
    store, _ = synthetic_corpus

    summary = train(parser().parse_args(arguments(store, tmp_path / "run", "--labels-only")))

    assert np.isfinite(summary["val/best_loss"]) and 0.0 <= summary["val/pr_auc"] <= 1.0
    assert not any(key.startswith(("val/teacher_", "release/")) for key in summary)
    assert (tmp_path / "run" / "best.pt").exists() and json.loads((tmp_path / "run" / "pilot.json").read_text(encoding="utf-8"))["steps"] == 4


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_uniform_epochs_with_masked_language_modeling_choose_on_pr_auc(tmp_path: Path, synthetic_corpus: tuple[Path, Path]) -> None:
    """Litmus 1.1's run: uniform draws in epochs, half the texts masked for the masked language loss, validation every
    half epoch against the withheld reference bank, best.pt on validation PR AUC, stopped by patience."""
    store, _ = synthetic_corpus
    extra = ["--labels-only", "--uniform", "--mlm-share", "0.5", "--epochs", "3", "--eval-every-epochs", "0.5", "--select", "pr_auc",
             "--patience", "2", "--bank-recipe", "reference", "--bank-per-origin", "5"]
    given = arguments(store, tmp_path / "run", *extra)
    del given[given.index("--pilot") : given.index("--pilot") + 3]  # no pilot: the epochs set the steps
    summary = train(parser().parse_args(given))

    assert 0.0 <= summary["val/best_pr_auc"] <= 1.0 and summary["val/best_pr_auc"] == summary["val/pr_auc"]
    model, checkpoint = load_checkpoint(tmp_path / "run" / "best.pt", torch.device("cpu"))
    assert checkpoint["mlm"] and model.mlm is not None


def test_masking_chooses_a_share_of_texts_and_their_tokens() -> None:
    tokens = torch.arange(1, 4001)  # (t,)
    segment = torch.arange(4000) // 100  # (t,) forty texts of 100 tokens
    batch = PackedBatch(tokens, segment, torch.arange(4000) % 100, 40)
    generator = torch.Generator().manual_seed(0)
    masked, targets = mask_stream(batch, 0, 50000, 0.5, generator)
    chosen = targets >= 0  # (t,)
    texts = chosen.view(40, 100).any(dim=1)  # (40,)
    assert 10 <= int(texts.sum()) <= 30  # about half the texts
    assert 0.08 < float(chosen.view(40, 100)[texts].float().mean()) < 0.22  # about 15% of a masked text's tokens
    assert torch.equal(targets[chosen], tokens[chosen]) and torch.equal(masked.tokens[~chosen], tokens[~chosen])
    assert float((masked.tokens[chosen] == 0).float().mean()) > 0.6  # most chosen become [MASK]
    again, _ = mask_stream(batch, 0, 50000, 0.5, generator)
    assert not torch.equal(again.tokens, masked.tokens)  # drawn anew at every step


def test_distillation_still_needs_the_teacher_files(tmp_path: Path, synthetic_corpus: tuple[Path, Path]) -> None:
    store, _ = synthetic_corpus

    with pytest.raises(ValueError, match="--teacher-export"):
        train(parser().parse_args(arguments(store, tmp_path / "run")))


def test_without_a_teacher_every_row_is_drawn_and_its_targets_are_zero(synthetic_corpus: tuple[Path, Path]) -> None:
    store_path, export = synthetic_corpus
    store = TokenStore(store_path)
    absent = TeacherFiles.absent(len(store))
    loaded = TeacherFiles.load(export, np.flatnonzero(store.split_mask("train")), np.random.default_rng(0))

    with_teacher = ClusterSampler(store, "train", 96, loaded.present)
    without = ClusterSampler(store, "train", 96, absent.present)
    rows = without.draw(60, np.random.default_rng(1))  # (b,)
    labels, embeddings, query_key = teacher_targets(rows, store, absent)

    assert sum(map(len, without.members[0])) > sum(map(len, with_teacher.members[0]))  # rows the teacher missed are drawn too
    assert absent.missing and not loaded.missing
    assert labels.tolist() == store.label[rows].tolist() and not embeddings.any() and not query_key.any()
