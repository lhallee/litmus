"""The hyperparameter search on a tiny synthetic study on the CPU: lockstep, pruning, the time box, resuming, the choice."""

from __future__ import annotations

import itertools
import json
import numpy as np
import pytest
import torch

from collections.abc import Callable
from pathlib import Path
from litmus_teacher.bank import chosen_head_name
from litmus_teacher.embed import EMBEDDINGS_FILE, TOKENS_FILE
from litmus_teacher.features import FeatureSource
from litmus_teacher.search import (
    DEFAULT_PARAMS, DEFAULT_TAG, Lane, SearchConfig, choose_teacher, full_space, lockstep_step, open_study,
    prefix_positions, retrain, run_search, settings_of,
)
from litmus_teacher.train import (
    HeadSettings, balanced_draw, build_head, input_statistics, load_split_tensors, optimizer_for, origin_members,
)


CPU = torch.device("cpu")
WIDTH = 16
TINY_SPACE = {
    "depth": {"choices": [1, 2]}, "hidden_1": {"choices": [8, 16]}, "hidden_2": {"choices": [8, 16]},
    "hidden_3": {"choices": [8]}, "output_width": {"choices": [4, 8]}, "batch_size": {"choices": [12, 24]},
    "warmup_steps": {"choices": [0, 2]}, "schedule": {"choices": ["constant", "cosine"]}, "clamp": {"choices": [5.0, 10.0]},
    "scale": {"choices": ["learned", "fixed"]}, "scale_init": {"choices": [5.0, 10.0]},
    "bias_init": {"choices": ["zero", "prior"]}, "label_smoothing": {"choices": [0.0, 0.1]},
    "learning_rate": {"low": 1e-7, "high": 3e-2, "log": True}, "weight_decay": {"low": 1e-4, "high": 1e-2, "log": True},
    "dropout": {"low": 0.0, "high": 0.2, "log": False},
}
ANCHOR = {"depth": 1, "hidden_1": 16, "output_width": 8, "batch_size": 12, "learning_rate": 1e-2, "weight_decay": 1e-3,
          "dropout": 0.0, "warmup_steps": 0, "schedule": "constant", "clamp": 10.0, "scale": "learned", "scale_init": 10.0,
          "bias_init": "zero", "label_smoothing": 0.0}
SPLITS = (("train", 540), ("validation", 120), ("test", 60), ("calibration", 60), ("bank", 120))


def ticking() -> Callable[[], float]:
    """A clock that advances one second each time it is read: the search reads it once per lockstep iteration."""
    counter = itertools.count()

    return lambda: float(next(counter))


def synthetic(root: Path) -> tuple[np.ndarray, np.ndarray, FeatureSource]:
    """900 texts of three origins around separated centers, in every split; an embedding directory under `root`."""
    rng = np.random.default_rng(0)
    n = sum(size for _, size in SPLITS)
    labels = np.arange(n) % 3  # (n,)
    centers = rng.normal(size=(3, WIDTH)) * 1.5  # (3, f)
    directory = root / "embeddings" / "synthetic"
    directory.mkdir(parents=True)
    np.save(directory / EMBEDDINGS_FILE, (centers[labels] + rng.normal(size=(n, WIDTH))).astype(np.float16))
    np.save(directory / TOKENS_FILE, np.full(n, 100, dtype=np.int32))
    split = np.concatenate([np.full(size, name, dtype=object) for name, size in SPLITS])  # (n,)

    return labels.astype(np.int64), split, FeatureSource("synthetic", (directory,))  # (n,), (n,)


def tiny(**overrides: object) -> SearchConfig:
    settings = {"features": "synthetic", "max_minutes": 1000.0, "slots": 3, "trial_steps": 12, "eval_every": 4,
                "patience": 100, "validation_batch": 12, "startup_trials": 2, "pruner_startup_trials": 1000,
                "space": TINY_SPACE, "anchors": (ANCHOR,)}

    return SearchConfig(**{**settings, **overrides})


def test_anchor_is_the_default_recipe_and_inside_the_space() -> None:
    assert settings_of(DEFAULT_PARAMS, "modernbert_base", 0, 40_000, 500) == HeadSettings("modernbert_base", 2048, 0)
    space = full_space()
    for name, value in DEFAULT_PARAMS.items():
        spec = space[name]
        assert value in spec["choices"] if "choices" in spec else spec["low"] <= value <= spec["high"], name


def test_a_prefix_of_a_balanced_draw_is_balanced() -> None:
    labels = torch.tensor([0] * 40 + [1] * 30 + [2] * 20)
    rows, drawn = balanced_draw(origin_members(labels), 24, torch.Generator().manual_seed(0))
    keep = prefix_positions(12, 24, CPU)
    assert keep.tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
    assert torch.bincount(drawn[keep]).tolist() == [4, 4, 4] and torch.equal(labels[rows[keep]], drawn[keep])
    with pytest.raises(ValueError):
        prefix_positions(24, 12, CPU)


def test_a_head_in_lockstep_trains_as_it_would_alone(tmp_path: Path) -> None:
    labels, split, source = synthetic(tmp_path)
    rows = load_split_tensors(source, labels, split, False, CPU)
    statistics = input_statistics(rows.train_features, torch.Generator().manual_seed(0))
    by_origin = origin_members(rows.train_labels)

    def lane(params: dict[str, object], init: int) -> Lane:
        settings = settings_of(params, "synthetic", 0, 10, 5)
        torch.manual_seed(init)
        head = build_head(settings, WIDTH, *statistics, CPU)
        return Lane(None, settings, head, *optimizer_for(settings, head), entered=0, started=0.0)

    alone, together = lane(ANCHOR, 1), lane(ANCHOR, 1)
    peer = lane({**ANCHOR, "learning_rate": 3e-3, "output_width": 4, "dropout": 0.1}, 2)
    first, second = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    for _ in range(5):
        lockstep_step([alone], rows.train_features, by_origin, first)
        lockstep_step([together, peer], rows.train_features, by_origin, second)
    assert alone.step == together.step == peer.step == 5
    for (name, value), other in zip(alone.head.state_dict().items(), together.head.state_dict().values(), strict=True):
        assert torch.equal(value, other), name


def test_search_trains_in_lockstep_and_reads_no_held_out_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels, split, source = synthetic(tmp_path)
    gathered: list[np.ndarray] = []
    original = FeatureSource.gather
    monkeypatch.setattr(FeatureSource, "gather", lambda self, ids: gathered.append(ids) or original(self, ids))
    rows = load_split_tensors(source, labels, split, False, CPU)
    summary = run_search(rows, tmp_path / "search", tiny(max_trials=6), CPU, clock=ticking())

    trials = json.loads((tmp_path / "search" / "trials.json").read_text(encoding="utf-8"))
    assert len(trials) == 6 and all(trial["state"] == "COMPLETE" for trial in trials)
    assert trials[0]["params"] == ANCHOR and summary["best"]["value"] == min(trial["value"] for trial in trials)
    assert all(len(trial["curve"]) == 3 for trial in trials)  # 12 steps read every 4
    spans = [(trial["user_attrs"]["entered"], trial["user_attrs"]["left"]) for trial in trials]
    assert any(a[0] < b[1] and b[0] < a[1] for a, b in itertools.combinations(spans, 2)), "trials overlap in lockstep"
    assert summary["head_steps"] == 6 * 12 and summary["iterations"] < 6 * 12
    assert (tmp_path / "search" / "leaderboard.md").read_text(encoding="utf-8").startswith("# litmus_teacher_")
    read = np.concatenate(gathered)
    assert set(split[read]) == {"train", "validation"}, "no trial reads test, calibration or a bank"


def test_the_time_box_stops_starting_trials_and_finishes_those_in_flight(tmp_path: Path) -> None:
    labels, split, source = synthetic(tmp_path)
    rows = load_split_tensors(source, labels, split, False, CPU)
    config = tiny(max_minutes=0.5, slots=2, trial_steps=20, eval_every=5)  # a box of 30 clock readings
    summary = run_search(rows, tmp_path / "search", config, CPU, clock=ticking())

    trials = json.loads((tmp_path / "search" / "trials.json").read_text(encoding="utf-8"))
    assert summary["closed_at_iteration"] == 29
    assert len(trials) == 4 and all(trial["state"] == "COMPLETE" for trial in trials)
    assert all(trial["user_attrs"]["entered"] < 29 for trial in trials)
    assert summary["iterations"] == 40 and max(trial["user_attrs"]["left"] for trial in trials) == 40


def test_the_median_pruner_stops_a_trial_that_does_not_learn(tmp_path: Path) -> None:
    labels, split, source = synthetic(tmp_path)
    rows = load_split_tensors(source, labels, split, False, CPU)
    learns = ({**ANCHOR, "learning_rate": 3e-2}, {**ANCHOR, "learning_rate": 2e-2})
    stuck = {**ANCHOR, "learning_rate": 1e-7}
    config = tiny(slots=1, max_trials=3, trial_steps=40, eval_every=4, pruner_startup_trials=2,
                  pruner_warmup_readings=1, anchors=(*learns, stuck))
    run_search(rows, tmp_path / "search", config, CPU, clock=ticking())

    trials = json.loads((tmp_path / "search" / "trials.json").read_text(encoding="utf-8"))
    assert [trial["state"] for trial in trials] == ["COMPLETE", "COMPLETE", "PRUNED"]
    assert trials[2]["user_attrs"]["end"] == "pruned" and trials[2]["user_attrs"]["readings"] < 10


def test_a_search_resumes_from_its_journal(tmp_path: Path) -> None:
    labels, split, source = synthetic(tmp_path)
    rows = load_split_tensors(source, labels, split, False, CPU)
    out = tmp_path / "search"
    run_search(rows, out, tiny(max_trials=3), CPU, clock=ticking())
    open_study(out, tiny()).ask()  # a trial the ended process left running
    summary = run_search(rows, out, tiny(max_trials=6), CPU, clock=ticking())

    trials = json.loads((out / "trials.json").read_text(encoding="utf-8"))
    assert summary["reaped"] == [3] and [trial["number"] for trial in trials] == list(range(6))
    assert [trial["state"] for trial in trials].count("FAIL") == 1 and trials[3]["state"] == "FAIL"
    assert sum(trial["params"] == ANCHOR for trial in trials) == 1, "the anchor is enqueued once"
    assert all(trial["state"] == "COMPLETE" for number, trial in enumerate(trials) if number != 3)


def test_the_choice_rule() -> None:
    def entry(loss: float, pr_auc: float, sd: float) -> dict[str, dict[str, float]]:
        return {"pair_loss": {"mean": loss, "sd": 0.001}, "pr_auc": {"mean": pr_auc, "sd": sd},
                "roc_auc": {"mean": pr_auc, "sd": sd}, "balanced_mcc": {"mean": 0.5, "sd": 0.01}}

    records = [{"tag": tag, "seed": seed, "best_validation_pair_loss": loss}
               for tag, losses in (("t0001", (0.30, 0.29, 0.31)), ("t0002", (0.32, 0.33, 0.31)), (DEFAULT_TAG, (0.4,)))
               for seed, loss in enumerate(losses)]
    near = {"t0001": entry(0.30, 0.950, 0.004), "t0002": entry(0.32, 0.953, 0.004), DEFAULT_TAG: entry(0.40, 0.90, 0.01)}
    choice = choose_teacher(near, records)
    assert choice["chosen"] == "t0001" and not choice["disagree"] and choice["head"]["seed"] == 1
    assert choice["gain_over_default"]["pair_loss"]["difference"] == pytest.approx(-0.10)
    beyond = {**near, "t0002": entry(0.32, 0.970, 0.004)}
    choice = choose_teacher(beyond, records)
    assert choice["disagree"] and choice["chosen"] == "t0002" and choice["by_pair_loss"] == "t0001"


def test_retrain_reads_the_best_through_the_bank_and_selects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    labels, split, source = synthetic(tmp_path)
    rows = load_split_tensors(source, labels, split, False, CPU)
    out = tmp_path / "search"
    config = tiny(max_trials=4)
    run_search(rows, out, config, CPU, clock=ticking())
    report = retrain(open_study(out, config), rows, labels, split, out, config, "2026-11-17_test", CPU, top=1,
                     seeds=(0, 1), bank_size=30)

    tags = list(report["summary"])
    assert DEFAULT_TAG in tags and len(tags) <= 2
    assert report["bank"]["texts_per_origin"] == 30 and len(report["records"]) == 2 * len(tags)
    for record in report["records"]:
        assert set(record["bank"]) == {"two_class", "human", "ai", "mixed"}
        assert 0.0 <= record["bank"]["two_class"]["pr_auc"] <= 1.0
    name = chosen_head_name(out)
    assert (out / "heads" / name / "head" / "pairhead.json").exists()
    assert name.startswith("synthetic_") and report["choice"]["chosen"] in tags
    assert json.loads((out / "selection.json").read_text(encoding="utf-8"))["head"]["tag"] == report["choice"]["chosen"]
