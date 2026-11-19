"""The distillation search on CPU: a few trials of the tiny model on the synthetic corpus, its records, and its rules."""

from __future__ import annotations

import json
import math
import pytest

from pathlib import Path
from litmus_student.distill import LossWeights
from litmus_student.search import (AGREEMENT, DEFAULT_PARAMS, MIN_PAIRED, SCHEDULE, SCHEDULE_AGREEMENT, loss_weights, parser,
                                   recommendation, search, spearman)

from test_export import synthetic_corpus


def test_default_parameters_give_the_default_weights() -> None:
    weights = loss_weights(DEFAULT_PARAMS)
    for name, value in vars(LossWeights()).items():
        assert getattr(weights, name) == pytest.approx(value)


def test_spearman() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) is None


def test_recommendation_takes_settings_only_where_rungs_agree() -> None:
    rungs = [10, 30]
    best = {**DEFAULT_PARAMS, "learning_rate": 4e-3, "temperature": 2.0}
    records = [{"number": 7, "params": best, "readings": [{"texts": 10, "pair_loss": 0.3}, {"texts": 30, "pair_loss": 0.2}]}]

    def agreement(rho: float) -> dict:
        return {"between_rungs": [{"rungs": rungs, "trials": MIN_PAIRED, "spearman": rho}]}

    weak = recommendation(records, agreement(AGREEMENT - 0.1), rungs)
    assert weak["taken"] == [] and weak["settings"] == DEFAULT_PARAMS
    middle = recommendation(records, agreement((AGREEMENT + SCHEDULE_AGREEMENT) / 2), rungs)
    assert middle["settings"]["temperature"] == 2.0 and middle["settings"]["learning_rate"] == DEFAULT_PARAMS["learning_rate"]
    strong = recommendation(records, agreement(SCHEDULE_AGREEMENT), rungs)
    assert strong["settings"]["learning_rate"] == 4e-3 and set(SCHEDULE) <= set(strong["taken"])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_search_loop_on_a_toy_model(tmp_path: Path) -> None:
    store, _, teacher = synthetic_corpus(tmp_path)
    out = tmp_path / "search"
    summary = search(parser().parse_args([
        "--data", str(store), "--teacher-export", str(teacher), "--out", str(out), "--config", "tiny", "--workers", "1",
        "--max-trials", "4", "--min-texts", "60", "--rungs", "2", "--val-texts", "30", "--bank-per-origin", "10",
        "--micro-tokens", "256", "--pack-multiple", "256", "--device", "cpu", "--max-minutes", "10",
    ]))
    trials = json.loads((out / "trials.json").read_text(encoding="utf-8"))
    assert len(trials) == 4 and trials[0]["params"] == DEFAULT_PARAMS  # the anchor first
    assert sum(summary["states"].values()) == 4 and summary["trials_at_rung"][0] >= 1
    for trial in trials:
        for reading in trial["readings"]:
            assert math.isfinite(reading["pair_loss"]) and 0.0 <= reading["pr_auc"] <= 1.0
    assert (out / "leaderboard.md").read_text(encoding="utf-8").startswith("| Trial")
    assert (out / "study.journal").exists() and "recommendation" in summary
