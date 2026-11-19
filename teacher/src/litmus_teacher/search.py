"""Bayesian search over the teacher head's recipe, heads trained in lockstep on shared batches, then the runs of record.

The study is Optuna's TPE, multivariate and grouped (so a setting drawn only under a condition, a hidden width past the
depth, is modeled), with a constant liar for the trials in flight and its first 12 trials random; a median pruner that waits for 8 finished trials
and 3 readings of a trial; the default recipe enqueued as trial 0, the anchor; a journal file a restarted search resumes,
failing the trials the ended process left running.

**Space** (`CATEGORICAL`, `FLOATS`): learning rate (log, 1e-4 to 3e-3), weight decay (log, 1e-4 to 0.1), dropout (0 to
0.4), depth (1 to 3 hidden layers) and each hidden width (512 to 8,192, drawn only for the layers the depth has), output
width 2w (256 to 2,048), batch per side (768 to 6,144), warmup steps, schedule (constant, or cosine to a tenth), input
clamp (3 to 30 standard deviations), the pair logit's scale (learned from, or fixed at, 5 to 20) and bias (learned from
zero, or from the balanced prior log-odds, log 1/2), and label smoothing (0 to 0.1). Smoothing is defensible here: a
mixed text holds human and machine spans, so its pair targets are the least certain, and the calibration step refits the
probability scale anyway; the objective reads hard targets, so a smoothed trial is judged on the same loss.

**Fixed, and why**: the input (ModernBERT-base's mean and variance; `train --common-rows` compares the encoders);
origin-balanced thirds and the symmetric same-origin loss, on which the bank reading is defined; the unit query and key,
GELU, LayerNorm and AdamW's betas, which the reading depends on or which matter little once the learning rate is tuned;
the step budget (`--trial-steps`, also the cosine horizon), the readings and the patience, which are the protocol, not the
recipe. The runs of record train under the same budget and protocol, so the search optimizes what is retrained.

**Objective**: a trial's lowest validation pair loss (`train.validation_pairs`: the same 16 pairs of balanced 3,072-text
batches for every trial, hard targets), reported at every reading for the pruner. Only training and validation rows are
loaded: no trial reads test, calibration or a bank.

**Lockstep**: training and validation features sit on the GPU once, in one process. Up to `--slots` heads train
together; each step draws one pair of balanced batches at the largest batch size in flight, and a head of batch size b
takes the first b/3 rows of each origin's block, itself a balanced uniform draw. Heads of any batch size so share every
batch, and TPE draws the batch size freely. A
finished or pruned head's slot is refilled at once. A trial's batch stream depends on when it entered, so a trial is
reproducible in distribution only; the retrained seeds are the record.

**Time box**: no trial starts after `--max-minutes`; trials in flight finish, at most one trial budget past the box. The
summary records the overrun, head-steps per second, and the trials a box holds at the measured rate.

**After the search** (unless `--no-retrain`): the 5 best complete configurations, and the default recipe for the
comparison, each train with seeds 0, 1 and 2 as runs of record (`train.train_head`, W&B project `litmus`), and each head
is read on validation through the `bank` split (variant a, at most 16,384 texts per origin): PR AUC, ROC AUC and balanced
MCC of the two-class reading (ai or mixed against human) and of each origin against the rest of the three-way reading.

**Choice rule, stated before any result**: P is the configuration of lowest mean validation pair loss over seeds, B the
one of highest mean two-class PR AUC. The bank disagrees when B is not P and B's mean PR AUC beats P's by more than the
larger of the two configurations' seed standard deviations of it; then B is the teacher, since the bank reading is what
calibration reads and the browser ships, and both are reported. Otherwise P is. The head of record is the chosen
configuration's seed of lowest validation pair loss; the gain over the default is the difference of means, reported with
each configuration's seed standard deviation (ddof 1).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
import numpy as np
import optuna
import torch

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from foundry.serialization.atomic import write_text_atomic
from .bank import bank_members, bank_readings, head_vectors, two_class_metrics
from .calibrate import three_way
from .features import FeatureSource, feature_source
from .head import PairHead
from .strata import ORIGINS, write_json, write_summary
from .train import (
    EVAL_EVERY, PATIENCE, PRIOR_BIAS, VALIDATION_BATCH, HeadSettings, SplitTensors, balanced_draw, build_head, input_statistics,
    load_labels_and_splits, load_split_tensors, optimizer_for, origin_members, pair_loss, train_head, validation_loss,
    validation_pairs,
)


SPACE = "litmus_teacher_2026_11_17"  # the study's name; a changed space takes a new one
STARTUP_TRIALS = 12  # trials TPE draws at random, the anchor among them, before it models the finished ones
PRUNER_STARTUP_TRIALS = 8  # finished trials before the median pruner prunes
PRUNER_WARMUP_READINGS = 3  # readings of a trial before it can be pruned
TRIAL_STEPS = 8000  # about 49 million texts drawn at the default batch, 6 or 7 passes over the training rows
SLOTS = 8
TOP_CONFIGURATIONS = 5
SEEDS = (0, 1, 2)
BANK_VARIANT = "a"
BANK_SIZE = 16_384  # texts per origin, at most
JOURNAL = "study.journal"
DEFAULT_TAG = "default"

WIDTHS = (512, 1024, 2048, 4096, 8192)
MAX_DEPTH = 3
CATEGORICAL: dict[str, tuple[Any, ...]] = {
    "depth": tuple(range(1, MAX_DEPTH + 1)),
    **{f"hidden_{layer}": WIDTHS for layer in range(1, MAX_DEPTH + 1)},
    "output_width": (256, 512, 1024, 2048),
    "batch_size": (768, 1536, 3072, 6144),
    "warmup_steps": (0, 250, 500, 1000),
    "schedule": ("constant", "cosine"),
    "clamp": (3.0, 5.0, 10.0, 30.0),
    "scale": ("learned", "fixed"),
    "scale_init": (5.0, 10.0, 20.0),
    "bias_init": ("zero", "prior"),
    "label_smoothing": (0.0, 0.05, 0.1),
}
FLOATS: dict[str, tuple[float, float, bool]] = {  # low, high, log
    "learning_rate": (1e-4, 3e-3, True),
    "weight_decay": (1e-4, 0.1, True),
    "dropout": (0.0, 0.4, False),
}
# The anchor, trial 0 of every study: `train.HeadSettings`' defaults, the first checker's recipe.
DEFAULT_PARAMS: dict[str, Any] = {
    "depth": 2, "hidden_1": 1024, "hidden_2": 8192, "output_width": 2048, "batch_size": 3072, "learning_rate": 3e-4,
    "weight_decay": 0.01, "dropout": 0.2, "warmup_steps": 500, "schedule": "constant", "clamp": 10.0, "scale": "learned",
    "scale_init": 10.0, "bias_init": "zero", "label_smoothing": 0.0,
}
SHOWN = ("depth", "output_width", "batch_size", "learning_rate", "weight_decay", "dropout", "warmup_steps", "schedule",
         "clamp", "scale", "scale_init", "bias_init", "label_smoothing")


def full_space() -> dict[str, dict[str, Any]]:
    """The whole space, a setting a distribution: `choices`, or `low` and `high` with `log`."""
    return {**{name: {"choices": list(values)} for name, values in CATEGORICAL.items()},
            **{name: {"low": low, "high": high, "log": log} for name, (low, high, log) in FLOATS.items()}}


def draw(trial: optuna.Trial, name: str, space: Mapping[str, Mapping[str, Any]]) -> Any:
    spec = space[name]
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])

    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def suggest(trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One configuration: every setting, and a hidden width for each layer the depth has."""
    params = {name: draw(trial, name, space) for name in (*FLOATS, *(name for name in CATEGORICAL
                                                                       if not name.startswith("hidden_")))}
    for layer in range(1, params["depth"] + 1):
        params[f"hidden_{layer}"] = draw(trial, f"hidden_{layer}", space)

    return params


def settings_of(params: Mapping[str, Any], features: str, seed: int, steps: int, eval_every: int, tag: str = "",
                validation_batch: int = VALIDATION_BATCH) -> HeadSettings:
    """The head settings a configuration trains with, under the protocol's step budget and readings."""
    return HeadSettings(
        features, params["output_width"], seed,
        widths=tuple(params[f"hidden_{layer}"] for layer in range(1, params["depth"] + 1)),
        dropout=params["dropout"], learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        batch_size=params["batch_size"], max_steps=steps, warmup_steps=params["warmup_steps"],
        schedule=params["schedule"], clamp=params["clamp"], scale_init=params["scale_init"],
        learn_scale=params["scale"] == "learned", bias_init={"zero": 0.0, "prior": PRIOR_BIAS}[params["bias_init"]],
        label_smoothing=params["label_smoothing"], eval_every=eval_every, validation_batch=validation_batch, tag=tag,
    )


@dataclass(frozen=True)
class SearchConfig:
    """What a search draws from and under which protocol; the defaults are the GPU run's."""

    features: str = "modernbert_base"
    max_minutes: float = 30.0
    max_trials: int | None = None  # the study's trials, failed included, after which none starts
    slots: int = SLOTS
    trial_steps: int = TRIAL_STEPS
    eval_every: int = EVAL_EVERY
    patience: int = PATIENCE
    validation_batch: int = VALIDATION_BATCH
    seed: int = 0
    startup_trials: int = STARTUP_TRIALS
    pruner_startup_trials: int = PRUNER_STARTUP_TRIALS
    pruner_warmup_readings: int = PRUNER_WARMUP_READINGS
    space: Mapping[str, Mapping[str, Any]] = field(default_factory=full_space)
    anchors: tuple[Mapping[str, Any], ...] = (DEFAULT_PARAMS,)


def open_study(root: Path, config: SearchConfig) -> optuna.Study:
    """The study on its journal. TPE's seed is offset by the trials already in it, so a resumed search does not
    redraw the random trials it drew before; Optuna 5 counts a pruned trial at its last reported value."""
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    root.mkdir(parents=True, exist_ok=True)
    journal = str(root / JOURNAL)
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=config.pruner_startup_trials,
                                         n_warmup_steps=config.pruner_warmup_readings)
    study = optuna.create_study(study_name=SPACE, storage=storage, pruner=pruner, direction="minimize",
                                load_if_exists=True)
    study.sampler = optuna.samplers.TPESampler(seed=config.seed + len(study.get_trials(deepcopy=False)),
                                               n_startup_trials=config.startup_trials, multivariate=True, group=True,
                                               constant_liar=True)

    return study


def reap(study: optuna.Study) -> list[int]:
    """Fail the trials an ended search left running (one process runs a study, so every running trial is stale)."""
    stale = [trial.number for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))]
    for number in stale:
        study.tell(number, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)

    return stale  # the summary records them


def enqueue(study: optuna.Study, params: Mapping[str, Any]) -> None:
    """Enqueue an anchor unless a trial of the study carries its params already, a failed one aside."""
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.FAIL and trial.system_attrs.get("fixed_params", trial.params) == params:
            return
    study.enqueue_trial(dict(params))


def fail(study: optuna.Study, trial: optuna.Trial, error: str) -> None:
    trial.set_user_attr("error", error[-1500:])
    study.tell(trial, state=optuna.trial.TrialState.FAIL)


@dataclass
class Lane:
    """One trial in flight: its head, optimizer and schedule, and its readings so far."""

    trial: optuna.Trial
    settings: HeadSettings
    head: PairHead
    optimizer: torch.optim.Optimizer
    schedule: torch.optim.lr_scheduler.LambdaLR
    entered: int  # the lockstep iteration it entered at
    started: float
    step: int = 0
    readings: int = 0
    best: float = math.inf
    best_step: int = 0
    stale: int = 0
    error: str | None = None


@lru_cache(maxsize=64)
def prefix_positions(batch: int, largest: int, device: torch.device) -> torch.Tensor:
    """Positions of a b-text balanced batch within a balanced draw of `largest`: the first b/3 of each origin's block."""
    share, block = batch // len(ORIGINS), largest // len(ORIGINS)
    if batch % len(ORIGINS) or largest % len(ORIGINS) or share > block:
        raise ValueError(f"a batch of {batch} is not a balanced prefix of {largest}")
    origins = torch.arange(len(ORIGINS), device=device)[:, None]  # (3, 1)

    return (origins * block + torch.arange(share, device=device)[None, :]).reshape(-1)  # (b,)


def lockstep_step(lanes: Sequence[Lane], features: torch.Tensor, by_origin: list[torch.Tensor],
                  generator: torch.Generator) -> None:
    """One optimizer step of every lane on one shared pair of balanced draws at the largest batch size in flight."""
    # features: (n_t, f); by_origin: (n_o,) row positions for each of three origins
    largest = max(lane.settings.batch_size for lane in lanes)
    rows_a, labels_a = balanced_draw(by_origin, largest, generator)  # (B,), (B,)
    rows_b, labels_b = balanced_draw(by_origin, largest, generator)  # (B,), (B,)
    features_a, features_b = features[rows_a], features[rows_b]  # (B, f) each
    for lane in lanes:
        batch = lane.settings.batch_size
        try:
            if batch == largest:
                loss = pair_loss(lane.head, features_a, labels_a, features_b, labels_b, lane.settings.label_smoothing)  # ()
            else:
                keep = prefix_positions(batch, largest, features.device)  # (b,)
                loss = pair_loss(lane.head, features_a[keep], labels_a[keep], features_b[keep], labels_b[keep],
                                 lane.settings.label_smoothing)  # ()
            lane.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lane.optimizer.step()
            lane.schedule.step()
        except torch.OutOfMemoryError as error:
            lane.error = f"out of memory at step {lane.step}: {error}"
            lane.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        lane.step += 1


def read_lane(lane: Lane, features: torch.Tensor, pairs: list[tuple[torch.Tensor, ...]], patience: int) -> str | None:
    """Read a lane's validation pair loss and report it; its end ("complete", "pruned", "diverged") or None to go on."""
    # features: (n_v, f); pairs: (b,) rows and labels of batch a, then of batch b
    value = validation_loss(lane.head, features, pairs)
    if not math.isfinite(value):
        return "diverged"
    lane.readings += 1
    lane.trial.report(value, lane.readings)
    if value < lane.best:
        lane.best, lane.best_step, lane.stale = value, lane.step, 0
    else:
        lane.stale += 1
    if lane.trial.should_prune():
        return "pruned"
    if lane.step >= lane.settings.max_steps or lane.stale >= patience:
        return "complete"

    return None


def start_lane(study: optuna.Study, config: SearchConfig, width: int, statistics: tuple[torch.Tensor, torch.Tensor],
               iteration: int, device: torch.device) -> Lane | None:
    """Ask the study for a trial and build its head; None when the head does not fit (the trial fails)."""
    # statistics: (f,) column mean and (f,) scale of the training rows
    trial = study.ask()
    params = suggest(trial, config.space)
    settings = settings_of(params, config.features, config.seed, config.trial_steps, config.eval_every,
                           f"t{trial.number:04d}", config.validation_batch)
    torch.manual_seed(config.seed * 1_000_003 + trial.number)
    try:
        head = build_head(settings, width, *statistics, device)
    except torch.OutOfMemoryError as error:
        fail(study, trial, f"out of memory building the head: {error}")
        torch.cuda.empty_cache()
        return None
    optimizer, schedule = optimizer_for(settings, head)
    trial.set_user_attr("parameters", sum(value.numel() for value in head.parameters()))

    return Lane(trial, settings, head, optimizer, schedule, iteration, time.perf_counter())


def end_lane(study: optuna.Study, lane: Lane, end: str, iteration: int) -> None:
    """Tell the study how a lane ended, with what it did as user attributes."""
    attrs = {"steps": lane.step, "readings": lane.readings, "best_step": lane.best_step, "entered": lane.entered,
             "left": iteration, "seconds": time.perf_counter() - lane.started, "end": end}
    for key, value in attrs.items():
        lane.trial.set_user_attr(key, value)
    if end == "complete":
        study.tell(lane.trial, lane.best)
    elif end == "pruned" or (end == "diverged" and lane.readings):
        study.tell(lane.trial, state=optuna.trial.TrialState.PRUNED)  # counted at its last finite reading
    else:
        fail(study, lane.trial, lane.error or "the validation pair loss was not finite at the first reading")


def has_room(study: optuna.Study, config: SearchConfig) -> bool:
    if config.max_trials is None:
        return True
    taken = study.get_trials(deepcopy=False, states=tuple(state for state in optuna.trial.TrialState
                                                          if state != optuna.trial.TrialState.WAITING))

    return len(taken) < config.max_trials


def run_search(rows: SplitTensors, out: Path, config: SearchConfig, device: torch.device,
               clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Run the study until the time box closes or `max_trials` are taken, every trial in flight finished.

    `clock` is read once before the loop and once per lockstep iteration, so a test can drive the box.
    """
    study = open_study(out, config)
    reaped = reap(study)
    for anchor in config.anchors:
        enqueue(study, anchor)
    generator = torch.Generator(device=device).manual_seed(config.seed + len(study.get_trials(deepcopy=False)))
    statistics = input_statistics(rows.train_features, generator)  # (f,), (f,)
    by_origin = origin_members(rows.train_labels)
    pairs = validation_pairs(rows.validation_labels, config.validation_batch)
    box, begun, wall = config.max_minutes * 60.0, clock(), time.perf_counter()
    lanes: list[Lane] = []
    iteration = head_steps = started = 0
    closed_at: int | None = None
    finished_steps: list[int] = []
    while True:
        if clock() - begun >= box and closed_at is None:
            closed_at = iteration
        while closed_at is None and len(lanes) < config.slots and has_room(study, config):
            lane = start_lane(study, config, rows.source.width, statistics, iteration, device)
            started += 1
            if lane is not None:
                lanes.append(lane)
        if not lanes:
            break
        lockstep_step(lanes, rows.train_features, by_origin, generator)
        head_steps += len(lanes)
        iteration += 1
        for lane in list(lanes):
            end = "failed" if lane.error else None
            if end is None and (lane.step % config.eval_every == 0 or lane.step >= lane.settings.max_steps):
                end = read_lane(lane, rows.validation_features, pairs, config.patience)
            if end is not None:
                end_lane(study, lane, end, iteration)
                lanes.remove(lane)
                finished_steps.append(lane.step)
                publish(study, out)
    publish(study, out)
    seconds = time.perf_counter() - wall
    rate = head_steps / seconds if seconds else 0.0
    mean_steps = float(np.mean(finished_steps)) if finished_steps else None
    summary = {
        "space": SPACE, "config": {key: value for key, value in config.__dict__.items() if key not in ("space", "anchors")},
        "reaped": reaped, "trials_started": started, "states": state_counts(study), "iterations": iteration,
        "closed_at_iteration": closed_at, "head_steps": head_steps, "seconds": seconds,
        "overrun_seconds": max(0.0, seconds - box), "head_steps_per_second": rate, "mean_steps_per_trial": mean_steps,
        "projected_trials_per_box": rate * box / mean_steps if mean_steps else None,
        "projected_trials_per_30_minutes": rate * 1800.0 / mean_steps if mean_steps else None,
        "best": best_trial(study),
    }
    write_json(out / "search_summary.json", summary)

    return summary


def state_counts(study: optuna.Study) -> dict[str, int]:
    trials = study.get_trials(deepcopy=False)

    return {state.name.lower(): sum(trial.state == state for trial in trials) for state in optuna.trial.TrialState}


def best_trial(study: optuna.Study) -> dict[str, Any] | None:
    done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not done:
        return None
    best = min(done, key=lambda trial: trial.value)

    return {"number": best.number, "value": best.value, "params": best.params}


def leaderboard(study: optuna.Study, top: int = 15) -> str:
    """The study's state as markdown: counts, then the best complete trials by validation pair loss."""
    counts = state_counts(study)
    done = sorted(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)), key=lambda trial: trial.value)
    lines = [f"# {SPACE}", "",
             f"Updated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. {counts['complete']} complete, {counts['pruned']} pruned, "
             f"{counts['running']} running, {counts['fail']} failed. The objective is each trial's lowest validation "
             "pair loss (lower is better); no trial reads test, calibration or a bank.", ""]
    if done:
        lines += ["| rank | trial | pair loss | steps | best step | hidden | " + " | ".join(SHOWN) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(SHOWN)]
        for rank, trial in enumerate(done[:top], start=1):
            params, attrs = trial.params, trial.user_attrs
            hidden = "x".join(str(params[f"hidden_{layer}"]) for layer in range(1, params["depth"] + 1))
            shown = " | ".join(f"{params[key]:.2e}" if key in FLOATS else str(params[key]) for key in SHOWN)
            lines.append(f"| {rank} | {trial.number} | {trial.value:.5f} | {attrs.get('steps', '')} | "
                         f"{attrs.get('best_step', '')} | {hidden} | {shown} |")
        lines += ["", "## Best trial's settings", "", "```json", json.dumps(done[0].params, indent=2, sort_keys=True), "```"]

    return "\n".join(lines) + "\n"


def publish(study: optuna.Study, root: Path) -> None:
    """Rewrite ROOT/leaderboard.md and ROOT/trials.json, every trial as plain data."""
    write_text_atomic(root / "leaderboard.md", leaderboard(study))
    write_json(root / "trials.json", [
        {"number": trial.number, "state": trial.state.name, "value": trial.value, "params": trial.params,
         "user_attrs": trial.user_attrs, "curve": [trial.intermediate_values[step] for step in sorted(trial.intermediate_values)],
         "started": trial.datetime_start.isoformat() if trial.datetime_start else None,
         "finished": trial.datetime_complete.isoformat() if trial.datetime_complete else None}
        for trial in study.get_trials(deepcopy=False)])


def candidates(study: optuna.Study, top: int, default: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The `top` best complete configurations, distinct, tagged by trial, and the default recipe tagged `default`."""
    done = sorted(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)), key=lambda trial: trial.value)
    chosen: list[tuple[str, dict[str, Any]]] = []
    for trial in done:
        if len(chosen) == top:
            break
        if all(trial.params != params for _, params in chosen):
            chosen.append((DEFAULT_TAG if trial.params == dict(default) else f"t{trial.number:04d}", trial.params))
    if all(tag != DEFAULT_TAG for tag, _ in chosen):
        chosen.append((DEFAULT_TAG, dict(default)))

    return chosen


def one_against_rest(positive: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """PR AUC, ROC AUC and balanced MCC of one score against a binary label: (n,), (n,)."""
    # positive, score: (n,)
    from foundry.metrics import balanced_mcc
    from foundry.metrics.ranking import binary_ranking_metrics

    ranking = binary_ranking_metrics(positive.astype(np.int64), score)

    return {"pr_auc": ranking["average_precision"], "roc_auc": ranking["auroc"],
            "balanced_mcc": balanced_mcc(positive.astype(np.int64), score)["balanced_mcc"]}


def bank_metrics(head: PairHead, source: FeatureSource, validation_ids: np.ndarray, validation_labels: np.ndarray,
                 bank_ids: list[np.ndarray], device: torch.device) -> dict[str, dict[str, float]]:
    """A head's validation reading through a fixed bank: the two-class metrics and each origin's against the rest."""
    # validation_ids, validation_labels: (n_v,); bank_ids: (k_o,) for each of three origins
    head.eval()
    vectors = head_vectors(head, source, validation_ids, device)  # (n_v, 2w)
    banks = [head_vectors(head, source, ids, device) for ids in bank_ids]  # three (k_o, 2w)
    readings = bank_readings(vectors, banks, head.scale.item(), head.bias.item(), np.zeros(len(validation_ids), np.int64))
    two_class = two_class_metrics(readings, validation_labels)
    shares = three_way(readings.mean)  # (n_v, 3)
    metrics = {"two_class": {key: two_class[key] for key in ("pr_auc", "roc_auc", "balanced_mcc")}}
    for origin, name in enumerate(ORIGINS):
        metrics[name] = one_against_rest(validation_labels == origin, shares[:, origin])

    return metrics


def spread(values: Sequence[float]) -> dict[str, float]:
    """Mean and seed standard deviation (ddof 1; 0 for one seed) of a metric over seeds."""
    values = np.asarray(values, dtype=np.float64)  # (s,)

    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "seeds": len(values)}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each configuration's metrics over its seeds: pair loss, the two-class bank metrics, and each origin's PR AUC."""
    tags = list(dict.fromkeys(record["tag"] for record in records))
    summary = {}
    for tag in tags:
        members = [record for record in records if record["tag"] == tag]
        summary[tag] = {"pair_loss": spread([record["best_validation_pair_loss"] for record in members]),
                        **{metric: spread([record["bank"]["two_class"][metric] for record in members])
                           for metric in ("pr_auc", "roc_auc", "balanced_mcc")},
                        **{f"{origin}_pr_auc": spread([record["bank"][origin]["pr_auc"] for record in members])
                           for origin in ORIGINS}}

    return summary


def choose_teacher(summary: Mapping[str, Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The module docstring's choice rule: P by mean pair loss unless B's bank PR AUC beats P's beyond seed noise."""
    by_loss = min(summary, key=lambda tag: summary[tag]["pair_loss"]["mean"])
    by_bank = max(summary, key=lambda tag: summary[tag]["pr_auc"]["mean"])
    margin = max(summary[by_loss]["pr_auc"]["sd"], summary[by_bank]["pr_auc"]["sd"])
    lead = summary[by_bank]["pr_auc"]["mean"] - summary[by_loss]["pr_auc"]["mean"]
    disagree = by_bank != by_loss and lead > margin
    chosen = by_bank if disagree else by_loss
    head = min((record for record in records if record["tag"] == chosen), key=lambda record: record["best_validation_pair_loss"])
    gain = None
    if DEFAULT_TAG in summary:
        gain = {metric: {"difference": summary[chosen][metric]["mean"] - summary[DEFAULT_TAG][metric]["mean"],
                         "chosen_sd": summary[chosen][metric]["sd"], "default_sd": summary[DEFAULT_TAG][metric]["sd"]}
                for metric in ("pair_loss", "pr_auc", "roc_auc", "balanced_mcc")}

    return {"by_pair_loss": by_loss, "by_bank_pr_auc": by_bank, "bank_lead": lead, "seed_margin": margin,
            "disagree": disagree, "chosen": chosen, "head": head, "gain_over_default": gain}


def retrain(study: optuna.Study, rows: SplitTensors, labels: np.ndarray, split: np.ndarray, out: Path,
            config: SearchConfig, experiment_id: str, device: torch.device, top: int = TOP_CONFIGURATIONS,
            seeds: Sequence[int] = SEEDS, bank_size: int = BANK_SIZE) -> dict[str, Any]:
    """Train the best configurations and the default with each seed as runs of record, read them through the bank on
    validation, and choose the teacher by the stated rule; writes retrain.json and the bank step's selection.json."""
    # labels, split: (n,)
    source = rows.source
    members = bank_members(split, labels, source.embedded(len(labels)), BANK_VARIANT, seed=0)
    size = min(bank_size, *(len(ids) for ids in members))
    bank_ids = [ids[:size] for ids in members]  # three (size,)
    validation_labels = labels[rows.validation_ids]  # (n_v,)
    records = []
    for tag, params in candidates(study, top, config.anchors[0]):
        for seed in seeds:
            settings = settings_of(params, config.features, seed, config.trial_steps, config.eval_every, tag,
                                   config.validation_batch)
            directory = out / "heads" / settings.name
            done = directory / "done.json"
            record = (json.loads(done.read_text(encoding="utf-8")) if done.exists()
                      else train_head(settings, rows, directory, experiment_id, False, device))
            head = PairHead.load(directory / "head").to(device)
            records.append({**record, "tag": tag, "params": params,
                            "bank": bank_metrics(head, source, rows.validation_ids, validation_labels, bank_ids, device)})
    summary = summarize(records)
    choice = choose_teacher(summary, records)
    report = {"experiment_id": experiment_id, "bank": {"variant": BANK_VARIANT, "texts_per_origin": size},
              "records": records, "summary": summary, "choice": choice}
    write_json(out / "retrain.json", report)
    write_json(out / "selection.json", {"head": choice["head"], "rule": "search.choose_teacher", "choice": {
        key: value for key, value in choice.items() if key != "head"}})

    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="split.parquet from the split step")
    parser.add_argument("--embedding-root", type=Path, required=True, help="holds one directory per encoder")
    parser.add_argument("--out", type=Path, required=True, help="the study's journal, trials.json, leaderboard.md, heads/")
    parser.add_argument("--experiment-id", required=True, help="YYYY-MM-DD_slug, the W&B group of the runs of record")
    parser.add_argument("--features", default="modernbert_base")
    parser.add_argument("--max-minutes", type=float, default=30.0, help="no trial starts after this; trials in flight finish")
    parser.add_argument("--max-trials", type=int, help="the study's trials, failed included, after which none starts")
    parser.add_argument("--slots", type=int, default=SLOTS, help="heads trained in lockstep")
    parser.add_argument("--trial-steps", type=int, default=TRIAL_STEPS, help="each trial's and each run of record's budget")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=TOP_CONFIGURATIONS, help="configurations retrained as runs of record")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--no-retrain", action="store_true", help="stop after the search")
    parser.add_argument("--retrain-only", action="store_true", help="skip the search: retrain from the journal")
    parser.add_argument("--pilot", action="store_true",
                        help="200,000 training rows under OUT/pilot, no retrain: measures the rate and projects the trials")


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    device = torch.device("cuda")
    out = args.out / "pilot" if args.pilot else args.out
    labels, split = load_labels_and_splits(args.corpus, args.split)
    source = feature_source(args.features, args.embedding_root)
    rows = load_split_tensors(source, labels, split, args.pilot, device)
    config = SearchConfig(features=args.features, max_minutes=args.max_minutes, max_trials=args.max_trials,
                          slots=args.slots, trial_steps=args.trial_steps, eval_every=args.eval_every, seed=args.seed)
    search = None if args.retrain_only else run_search(rows, out, config, device)
    retrained = None
    if not (args.no_retrain or args.pilot):
        retrained = retrain(open_study(out, config), rows, labels, split, out, config, args.experiment_id, device,
                            args.top, args.seeds)
    write_summary(out, "search" + (" --pilot" if args.pilot else ""), {
        "experiment_id": args.experiment_id, "search": search,
        "retrain": None if retrained is None else {"summary": retrained["summary"], "choice": retrained["choice"]},
        "train_rows": int(len(rows.train_ids)), "validation_rows": int(len(rows.validation_ids)),
        "seconds": time.perf_counter() - started})
