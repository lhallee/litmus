"""Bayesian search over the distillation's settings, in a 45-minute box on one GPU.

    python -m litmus_student search --data PRETOKENIZED --teacher-export TEACHER_EXPORT --out DIR \\
        [--max-minutes 45] [--workers 3] [--config s192_l4] [--origins 0,1,2]

**The study** reuses the teacher's plumbing (`litmus_teacher.search`: `draw`, `enqueue`, `fail`, `reap`) rather than
a second copy: Optuna's TPE, multivariate and grouped, with a constant liar for the trials in flight and its first `STARTUP_TRIALS` random; the
current defaults enqueued as trial 0, the anchor; one journal file that `--workers` processes share (each takes
`1 / workers` of the GPU's memory, so several small students train at once) and a restarted search resumes.

**Multi-fidelity.** The resource is texts trained. Asynchronous successive halving (`SuccessiveHalvingPruner`, rate
3) reads a trial at rungs of `--min-texts` x 3^k texts, k = 0 .. `--rungs` - 1 (32,768, 98,304 and 294,912 by
default), and stops it at a rung unless it is in the best third there. A trial's learning-rate schedule spans its
top rung, so a promoted trial continues the same schedule. No trial starts once less than a rung-0 trial's measured
time is left in the box, and trials still running at the box's end stop there, pruned at their last rung.

**What a box holds**, estimated (the search's own `summary.json` records the measured texts per second): a trial of
`s192_l4` at about 600 tokens a text costs about 20 s per 32,768 texts at 1M tokens per second; with 3 workers at
an assumed 0.5M tokens per second each, a rung-0 trial takes about 45 s with its two readings, a rung-1 promotion
about 90 s more and a rung-2 promotion about 4.5 minutes more, so 135 worker-minutes hold about 70 trials at rung 0,
about 24 at rung 1 and about 8 at rung 2. A confirmation of the best 3 at the middle size (`--confirm 3`) trains
them again, unpruned, at the top rung as `s256_l6` (`--confirm-config`) on one worker, in `--confirm-minutes` (15)
kept back from the box, and reports the rank agreement of the two sizes; with three configurations that agreement
is a sign, not a test.

**Space** (encoder depth and width are not in it: they are the size sweep, and the release rule picks the smallest
size within 0.01 PR AUC of the teacher):
- schedule: peak learning rate (log, 3e-4 to 6e-3), the second stage's peak as a share of the first (0.3 to 1),
  warmup as a share of the trial's steps (0.02, 0.05, 0.1), weight decay (log, 1e-3 to 0.3), and whether it drops to
  zero at half the run (BitNet's recipe) or stays;
- texts per step (256, 512, 1,024, 2,048);
- losses, scale-free: each of embedding, query and key, soft pairs and ranked pairs as a log ratio to the
  true-label pair loss (-4 to 1.5), so the five shares are a softmax of (those four, 0); their total weight (log,
  0.5 to 8); and the temperature (log, 0.5 to 4);
- head: hidden width (256, 512, 1,024), query and key width (64 or 128), input clamp (3, 5, 10, 30 deviations).

**Objective**: the validation pair loss against true labels, the binary cross-entropy of the student's pair logits
over every off-diagonal pair of `--val-texts` origin-balanced validation texts, minimized. Not the distillation loss,
whose size the searched weights change; not the PR AUC through a bank as the objective, though it is read beside: at
a short budget the bank's PR AUC on a few thousand texts sits near its ceiling and moves by about its own sampling
error (some 0.002 to 0.003), while the pair loss averages millions of pairs and moves smoothly with every pair's
separation, which is what the bank reading averages. The search reports the rank agreement between the two across
its own readings, so the choice is checked. No trial reads test or calibration.

**What a short search cannot tell.** A setting that wins at 300,000 texts may lose at a full run's 4 million, the
learning rate above all. The search reports, from its own trials, the rank correlation (Spearman) of the objective
between consecutive rungs. The rule for the full runs, stated before any result: the best trial at the top rung
supplies its loss and head settings when the agreement between the last two rungs is at least `AGREEMENT` (0.6, at
least `MIN_PAIRED` trials), and also its schedule settings (learning rates, warmup, weight decay, texts per step)
only when that agreement is at least `SCHEDULE_AGREEMENT` (0.8); every setting not taken keeps its default.

**Records**: trials make no W&B run. `trials.json` (every trial: state, settings, readings at each rung with seconds),
`leaderboard.md`, `summary.json` (counts, rates, rank agreement, the recommendation) and `study.journal`.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
import numpy as np
import optuna
import torch

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from litmus_teacher.search import draw, enqueue, fail, reap
from torch import nn

from foundry.serialization.atomic import write_text_atomic
from foundry.training import clip_grad_norm, set_seed
from .contract import reference
from .data import ClusterSampler, TeacherFiles, TokenStore, build_step, cluster_stratified_rows, prefetched
from .distill import LossWeights, Projections, pair_logits, same_origin
from .model import StudentModel
from .train import bank_shares, learning_rate, model_vectors, parameter_groups, train_step, weight_decay


SPACE = "litmus_student_2026_11_17"  # the study's name; a changed space takes a new one
STARTUP_TRIALS = 12
REDUCTION = 3
AGREEMENT = 0.6
SCHEDULE_AGREEMENT = 0.8
MIN_PAIRED = 5
JOURNAL = "study.journal"
LOSS_RATIOS = ("embedding", "query_key", "pair_soft", "pair_rank")  # log weights relative to pair_label
SCHEDULE = ("learning_rate", "second_ratio", "warmup_share", "weight_decay", "decay_drops", "texts_per_step")
CATEGORICAL: dict[str, tuple[Any, ...]] = {
    "texts_per_step": (256, 512, 1024, 2048),
    "warmup_share": (0.02, 0.05, 0.1),
    "decay_drops": (True, False),
    "head_hidden": (256, 512, 1024),
    "head_out": (64, 128),
    "head_clamp": (3.0, 5.0, 10.0, 30.0),
}
FLOATS: dict[str, tuple[float, float, bool]] = {  # low, high, log
    "learning_rate": (3e-4, 6e-3, True),
    "second_ratio": (0.3, 1.0, False),
    "weight_decay": (1e-3, 0.3, True),
    **{f"log_{name}": (-4.0, 1.5, False) for name in LOSS_RATIOS},
    "loss_total": (0.5, 8.0, True),
    "temperature": (0.5, 4.0, True),
}
# The anchor, trial 0: `train.py`'s defaults (warmup 100 of 2,000 steps is a share of 0.05).
DEFAULT_PARAMS: dict[str, Any] = {
    "texts_per_step": 2048, "warmup_share": 0.05, "decay_drops": True, "head_hidden": 512, "head_out": 128, "head_clamp": 10.0,
    "learning_rate": 1.5e-3, "second_ratio": 1e-3 / 1.5e-3, "weight_decay": 0.1, "log_embedding": 0.0, "log_query_key": 0.0,
    "log_pair_soft": 0.0, "log_pair_rank": math.log(0.5), "loss_total": 4.5, "temperature": 1.0,
}


def full_space() -> dict[str, dict[str, Any]]:
    return {**{name: {"choices": list(values)} for name, values in CATEGORICAL.items()},
            **{name: {"low": low, "high": high, "log": log} for name, (low, high, log) in FLOATS.items()}}


def suggest(trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: draw(trial, name, space) for name in (*FLOATS, *CATEGORICAL)}


def loss_weights(params: Mapping[str, Any]) -> LossWeights:
    """The five weights from four log ratios to the true-label pair loss and their total."""
    logits = np.asarray([params[f"log_{name}"] for name in LOSS_RATIOS] + [0.0])  # (5,)
    shares = np.exp(logits - logits.max())  # (5,)
    shares /= shares.sum()
    weights = params["loss_total"] * shares  # (5,)

    return LossWeights(*(float(value) for value in weights), temperature=float(params["temperature"]))


def rung_texts(min_texts: int, rungs: int) -> list[int]:
    return [min_texts * REDUCTION**rung for rung in range(rungs)]


def open_study(root: Path, seed: int, min_texts: int) -> optuna.Study:
    """The study on its journal, shared by the workers; TPE's seed is offset by the trials already in it."""
    import warnings

    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    root.mkdir(parents=True, exist_ok=True)
    journal = str(root / JOURNAL)
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=min_texts, reduction_factor=REDUCTION, min_early_stopping_rate=0)
    study = optuna.create_study(study_name=SPACE, storage=storage, pruner=pruner, direction="minimize", load_if_exists=True)
    study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.get_trials(deepcopy=False)), n_startup_trials=STARTUP_TRIALS,
                                               multivariate=True, group=True, constant_liar=True)

    return study


# ---------------------------------------------------------------- one trial


@dataclass
class Context:
    """What every trial of a worker reads, loaded once."""

    store: TokenStore
    teacher: TeacherFiles
    sampler: ClusterSampler
    validation: np.ndarray  # (v,) origin-balanced validation rows
    validation_labels: torch.Tensor  # (v,)
    bank: np.ndarray  # (m,)
    bank_labels: torch.Tensor  # (m,)
    origins: tuple[int, ...]
    device: torch.device


def load_context(arguments: argparse.Namespace, device: torch.device) -> Context:
    generator = np.random.default_rng(arguments.seed)
    store = TokenStore(arguments.data)
    teacher = TeacherFiles.load(arguments.teacher_export, np.flatnonzero(store.split_mask("train")), generator)
    chunk_tokens = reference.named_config(arguments.config).chunk_tokens
    origins = tuple(int(value) for value in arguments.origins.split(","))
    sampler = ClusterSampler(store, "train", chunk_tokens, teacher.present, origins=origins)
    validation = np.sort(ClusterSampler(store, "validation", chunk_tokens, teacher.present, origins=origins).draw(arguments.val_texts, generator))
    validation = np.unique(validation)  # (v,) a text drawn twice is read once
    bank = cluster_stratified_rows(store, arguments.bank_split, arguments.bank_per_origin, generator, origins)  # (m',)
    bank = np.sort(bank[store.length[bank] <= chunk_tokens])  # (m,)

    def labels(rows: np.ndarray) -> torch.Tensor:
        # rows: (r,); labels as indices into `origins`
        return torch.from_numpy(np.searchsorted(np.asarray(origins), store.label[rows].astype(np.int64))).to(device)  # (r,)

    return Context(store, teacher, sampler, validation, labels(validation), bank, labels(bank), origins, device)


@torch.no_grad()
def read(model: StudentModel, context: Context, micro_tokens: int) -> dict[str, float]:
    """The objective, the validation pair loss against true labels, and beside it the bank's PR AUC (human against
    the rest) and the mean of the bank's normalized reading."""
    from foundry.layers.cross_entropy import binary_cross_entropy_with_logits
    from foundry.metrics.ranking import binary_ranking_metrics

    vectors = model_vectors(model, context.store, context.validation, micro_tokens, context.device)  # (v, 2w)
    scale, bias = float(model.head.scale.detach()), float(model.head.bias.detach())
    logits = pair_logits(vectors, vectors, scale, bias)  # (v, v)
    off_diagonal = ~torch.eye(len(vectors), dtype=torch.bool, device=vectors.device)  # (v, v)
    loss = binary_cross_entropy_with_logits(logits, same_origin(context.validation_labels), reduction="none")[off_diagonal].mean()
    bank = model_vectors(model, context.store, context.bank, micro_tokens, context.device)  # (m, 2w)
    shares = bank_shares(vectors, bank, context.bank_labels, scale, bias, len(context.origins)).cpu().numpy()  # (v, o)
    written = (context.validation_labels.cpu().numpy() != 0).astype(np.int64)  # (v,)
    ranking = binary_ranking_metrics(written, 1.0 - shares[:, 0])

    return {"pair_loss": float(loss), "pr_auc": float(ranking["average_precision"])}


def run_trial(trial: optuna.Trial, params: Mapping[str, Any], context: Context, arguments: argparse.Namespace, deadline: float,
              config_name: str | None = None) -> list[dict[str, float]]:
    """Train one configuration up to its top rung, reading and reporting at each; raises TrialPruned when the pruner or
    the box stops it. Returns the readings."""
    set_seed(arguments.seed + trial.number)
    generator = np.random.default_rng(arguments.seed + trial.number)
    config = replace(reference.named_config(config_name or arguments.config), head_hidden=params["head_hidden"],
                     head_out=params["head_out"], head_clamp=params["head_clamp"])
    device, teacher = context.device, context.teacher
    model = StudentModel(config).to(device)
    projections = Projections(2 * config.width, teacher.embedding_width, config.head_out, teacher.query_key.shape[1] // 2).to(device)
    trainable = nn.ModuleDict({"model": model, "projections": projections})
    optimizer = torch.optim.AdamW(parameter_groups(trainable, params["weight_decay"]), lr=params["learning_rate"], betas=(0.9, 0.95),
                                  fused=device.type == "cuda")
    weights = loss_weights(params)
    rungs = rung_texts(arguments.min_texts, arguments.rungs)
    texts_per_step = int(params["texts_per_step"])
    total = math.ceil(rungs[-1] / texts_per_step)
    warmup = max(1, round(params["warmup_share"] * total))
    first, second = params["learning_rate"], params["learning_rate"] * params["second_ratio"]

    def make() -> Any:
        rows = context.sampler.draw(texts_per_step, generator)  # (b,)
        return build_step(rows, context.store, teacher, arguments.micro_tokens, config.block_size, arguments.pack_multiple)

    readings: list[dict[str, float]] = []
    started, texts, rung = time.perf_counter(), 0, 0
    for index, step in enumerate(prefetched(make, total, depth=2, pin=device.type == "cuda")):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(index, total, warmup, first, second)
            decay = weight_decay(index, total, params["weight_decay"]) if params["decay_drops"] else params["weight_decay"]
            group["weight_decay"] = decay if group["decays"] else 0.0
        losses = train_step(model, projections, step, teacher, weights, device)
        if not math.isfinite(losses["total"]):
            raise FloatingPointError(f"step {index}: the loss is {losses['total']}")
        clip_grad_norm(trainable, arguments.clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        texts += texts_per_step
        while rung < len(rungs) and texts >= rungs[rung]:  # a large step may pass two rungs; each is read
            reading = {"texts": rungs[rung], "seconds": time.perf_counter() - started, **read(model, context, arguments.micro_tokens)}
            readings.append(reading)
            trial.set_user_attr("readings", readings)
            trial.report(reading["pair_loss"], rungs[rung])
            rung += 1
            if rung < len(rungs) and (trial.should_prune() or time.time() > deadline):
                raise optuna.TrialPruned()
        if time.time() > deadline and rung < len(rungs):
            trial.set_user_attr("stopped_by_box", True)
            raise optuna.TrialPruned()

    return readings


# ---------------------------------------------------------------- workers


def work(arguments: argparse.Namespace, worker: int, deadline: float) -> None:
    """One worker: ask, train, tell, until the box has no room for a rung-0 trial."""
    device = torch.device(arguments.device)
    if device.type == "cuda" and arguments.workers > 1:
        torch.cuda.set_per_process_memory_fraction(0.95 / arguments.workers)  # the current device; `cuda` has no index to pass
    torch.backends.cuda.matmul.allow_tf32 = True
    context = load_context(arguments, device)
    study = open_study(arguments.out, arguments.seed + 1000 * worker, arguments.min_texts)
    rung_zero: list[float] = []
    space = full_space()
    while True:
        if arguments.max_trials and len(study.get_trials(deepcopy=False)) >= arguments.max_trials:
            return
        needed = float(np.median(rung_zero)) if rung_zero else 0.0
        if time.time() + needed > deadline:
            return
        trial = study.ask()
        params = suggest(trial, space)
        try:
            readings = run_trial(trial, params, context, arguments, deadline)
            study.tell(trial, readings[-1]["pair_loss"])
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except (FloatingPointError, torch.OutOfMemoryError) as error:
            fail(study, trial, repr(error))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        readings = study.trials[trial.number].user_attrs.get("readings", [])
        if readings:
            rung_zero.append(readings[0]["seconds"])


def _work_entry(arguments: argparse.Namespace, worker: int, deadline: float) -> None:
    work(arguments, worker, deadline)


# ---------------------------------------------------------------- the report


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Rank correlation of two equal-length sequences (average ranks for ties); None under 3 pairs."""
    if len(first) < 3:
        return None

    def ranks(values: Sequence[float]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)  # (n,)
        order = np.argsort(values, kind="stable")  # (n,)
        ranked = np.empty(len(values))  # (n,)
        ranked[order] = np.arange(len(values))
        for value in np.unique(values):
            tied = values == value
            ranked[tied] = ranked[tied].mean()
        return ranked  # (n,)

    a, b = ranks(first), ranks(second)
    if a.std() == 0 or b.std() == 0:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def trial_records(study: optuna.Study) -> list[dict[str, Any]]:
    return [{"number": trial.number, "state": trial.state.name, "params": trial.params, "readings": trial.user_attrs.get("readings", []),
             "stopped_by_box": trial.user_attrs.get("stopped_by_box", False), "error": trial.user_attrs.get("error")}
            for trial in study.get_trials(deepcopy=False)]


def rank_agreement(records: Sequence[Mapping[str, Any]], rungs: Sequence[int]) -> dict[str, Any]:
    """Spearman of the objective between consecutive rungs, over the trials read at both, and of the objective with
    the bank's PR AUC at each rung (negated loss against PR AUC, so agreement is positive)."""
    def at(record: Mapping[str, Any], texts: int) -> dict[str, float] | None:
        return next((reading for reading in record["readings"] if reading["texts"] == texts), None)

    between = []
    for low, high in zip(rungs, rungs[1:], strict=False):
        pairs = [(at(record, low), at(record, high)) for record in records]
        pairs = [(one["pair_loss"], two["pair_loss"]) for one, two in pairs if one and two]
        between.append({"rungs": [low, high], "trials": len(pairs),
                        "spearman": spearman([one for one, _ in pairs], [two for _, two in pairs])})
    objective = []
    for texts in rungs:
        readings = [reading for record in records if (reading := at(record, texts))]
        objective.append({"texts": texts, "trials": len(readings),
                          "spearman": spearman([-reading["pair_loss"] for reading in readings], [reading["pr_auc"] for reading in readings])})

    return {"between_rungs": between, "objective_with_pr_auc": objective}


def recommendation(records: Sequence[Mapping[str, Any]], agreement: Mapping[str, Any], rungs: Sequence[int]) -> dict[str, Any]:
    """The settings for the full runs, by the rule stated in the module's docstring."""
    top = [record for record in records if any(reading["texts"] == rungs[-1] for reading in record["readings"])]
    settings = dict(DEFAULT_PARAMS)
    if not top or not agreement["between_rungs"]:
        return {"settings": settings, "taken": [], "reason": "no trial reached the top rung"}
    best = min(top, key=lambda record: record["readings"][-1]["pair_loss"])
    last = agreement["between_rungs"][-1]
    rho = last["spearman"]
    enough = last["trials"] >= MIN_PAIRED and rho is not None
    taken = []
    if enough and rho >= AGREEMENT:
        taken += [name for name in DEFAULT_PARAMS if name not in SCHEDULE]
    if enough and rho >= SCHEDULE_AGREEMENT:
        taken += list(SCHEDULE)
    settings.update({name: best["params"][name] for name in taken})
    reason = (f"rank agreement {rho} between rungs {last['rungs']} over {last['trials']} trials; "
              f"loss and head settings need {AGREEMENT}, schedule settings {SCHEDULE_AGREEMENT}, with at least {MIN_PAIRED} trials")

    return {"settings": settings, "best_trial": best["number"], "taken": taken, "reason": reason}


def leaderboard(records: Sequence[Mapping[str, Any]], top: int = 15) -> str:
    read = [record for record in records if record["readings"]]
    ranked = sorted(read, key=lambda record: (-record["readings"][-1]["texts"], record["readings"][-1]["pair_loss"]))[:top]
    shown = ("learning_rate", "second_ratio", "texts_per_step", "weight_decay", "loss_total", "temperature", "head_hidden", "head_out", "head_clamp")
    lines = ["| Trial | State | Texts | Pair loss | PR AUC | " + " | ".join(shown) + " |", "| --- " * (5 + len(shown)) + "|"]
    for record in ranked:
        last = record["readings"][-1]
        values = [f"{record['params'].get(name, ''):.3g}" if isinstance(record["params"].get(name), float) else str(record["params"].get(name, ""))
                  for name in shown]
        lines.append(f"| {record['number']} | {record['state']} | {last['texts']:,} | {last['pair_loss']:.4f} | {last['pr_auc']:.4f} | "
                     + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def confirm(records: Sequence[Mapping[str, Any]], arguments: argparse.Namespace, rungs: Sequence[int], deadline: float) -> list[dict[str, Any]]:
    """The best `--confirm` configurations at the top rung, trained again at `--confirm-config`, unpruned."""
    top = [record for record in records if any(reading["texts"] == rungs[-1] for reading in record["readings"])]
    chosen = sorted(top, key=lambda record: record["readings"][-1]["pair_loss"])[: arguments.confirm]
    context = load_context(arguments, torch.device(arguments.device))
    confirmed = []
    for record in chosen:
        if time.time() > deadline:
            break
        readings = run_trial(optuna.trial.FixedTrial(record["params"], number=record["number"]), record["params"], context, arguments,
                             math.inf, arguments.confirm_config)
        confirmed.append({"number": record["number"], "small": record["readings"][-1], "confirmed": readings[-1]})

    return confirmed


def search(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    box = started + arguments.max_minutes * 60.0
    deadline = box - arguments.confirm_minutes * 60.0 if arguments.confirm else box
    study = open_study(arguments.out, arguments.seed, arguments.min_texts)
    stale = reap(study)
    enqueue(study, DEFAULT_PARAMS)
    if arguments.workers == 1:
        work(arguments, 0, deadline)
    else:
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=_work_entry, args=(arguments, worker, deadline)) for worker in range(arguments.workers)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed = [process.exitcode for process in processes if process.exitcode]
        if failed:
            raise RuntimeError(f"search workers exited with codes {failed}")
    study = open_study(arguments.out, arguments.seed, arguments.min_texts)
    records = trial_records(study)
    rungs = rung_texts(arguments.min_texts, arguments.rungs)
    agreement = rank_agreement(records, rungs)
    confirmed = confirm(records, arguments, rungs, box) if arguments.confirm else []
    texts = sum(reading["texts"] for record in records for reading in record["readings"][-1:])
    minutes = (time.time() - started) / 60.0
    summary = {
        "space": SPACE, "config": arguments.config, "workers": arguments.workers, "minutes": minutes, "rungs": rungs, "stale_trials": stale,
        "states": {state: sum(record["state"] == state for record in records) for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING")},
        "trials_at_rung": [sum(any(reading["texts"] == texts_at for reading in record["readings"]) for record in records) for texts_at in rungs],
        "texts_trained": texts, "texts_per_second": texts / max(1.0, minutes * 60.0),
        "rank_agreement": agreement, "recommendation": recommendation(records, agreement, rungs),
        "confirmed": confirmed, "confirm_config": arguments.confirm_config if arguments.confirm else None,
        "confirm_agreement": spearman([entry["small"]["pair_loss"] for entry in confirmed], [entry["confirmed"]["pair_loss"] for entry in confirmed]),
        "default": DEFAULT_PARAMS,
    }
    write_text_atomic(arguments.out / "trials.json", json.dumps(records, indent=1, default=str) + "\n")
    write_text_atomic(arguments.out / "leaderboard.md", leaderboard(records))
    write_text_atomic(arguments.out / "summary.json", json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({key: summary[key] for key in ("states", "trials_at_rung", "texts_per_second", "rank_agreement", "recommendation")},
                     indent=1, default=str), file=sys.stderr)

    return summary


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m litmus_student search", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="the pre-tokenized corpus of `litmus_student.data`")
    parser.add_argument("--teacher-export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", default="s192_l4", help="the size searched at, the sweep's smallest")
    parser.add_argument("--origins", default="0,1,2", help="the labels the corpus holds (0,1 for the first corpus)")
    parser.add_argument("--max-minutes", type=float, default=45.0)
    parser.add_argument("--max-trials", type=int, default=0, help="stop asking after this many trials (0: the box decides)")
    parser.add_argument("--workers", type=int, default=3, help="processes sharing the GPU, each with 1/workers of its memory")
    parser.add_argument("--min-texts", type=int, default=32_768, help="rung 0; each later rung is 3 times the one before")
    parser.add_argument("--rungs", type=int, default=3)
    parser.add_argument("--val-texts", type=int, default=4096)
    parser.add_argument("--bank-split", default="bank")
    parser.add_argument("--bank-per-origin", type=int, default=2048)
    parser.add_argument("--micro-tokens", type=int, default=65536)
    parser.add_argument("--pack-multiple", type=int, default=8192)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--confirm", type=int, default=0, help="train the best this many again at --confirm-config at the top rung")
    parser.add_argument("--confirm-config", default="s256_l6")
    parser.add_argument("--confirm-minutes", type=float, default=15.0, help="of the box, kept for the confirmation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    search(parser().parse_args(argv))
