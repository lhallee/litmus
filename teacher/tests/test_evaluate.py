"""Discrimination rows: all three metrics, thresholds chosen on validation and carried to test."""

from __future__ import annotations

import numpy as np

from litmus_teacher.evaluate import model_rows


def probabilities(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # labels: (n,)
    logits = rng.normal(size=(len(labels), 3)) + 2.0 * np.eye(3)[labels]  # (n, 3)
    return np.exp(logits) / np.exp(logits).sum(1, keepdims=True)  # (n, 3)


def test_thresholds_carry_from_validation_to_test() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 900)
    strata = {"all": np.full(900, "all", dtype=object), "format": np.where(np.arange(900) % 2, "a", "b").astype(object)}
    thresholds: dict[str, float] = {}
    validation = model_rows("m", "validation", probabilities(labels, rng), labels, strata, thresholds)
    test = model_rows("m", "test", probabilities(labels, rng), labels, strata, thresholds)
    assert {"m|binary", "m|human_vs_ai", "m|human_ovr", "m|ai_ovr", "m|mixed_ovr"} == set(thresholds)
    for row in test:
        if row["task"] != "three_way_macro":
            assert row["threshold"] == thresholds[f"m|{row['task']}"]
    for row in validation + test:
        assert row["pr_auc"] is not None and row["roc_auc"] is not None and row["balanced_mcc"] is not None
    assert {row["task"] for row in test} >= {"binary", "three_way_macro"}
