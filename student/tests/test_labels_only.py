"""A labels-only run needs no teacher file; distillation still does."""

from __future__ import annotations

import json
import numpy as np
import pytest

from pathlib import Path
from litmus_student.data import ClusterSampler, TeacherFiles, TokenStore, teacher_targets
from litmus_student.train import parser, train


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
