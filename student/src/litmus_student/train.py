"""Distill the Litmus student from the frozen teacher: one size per process, several processes per GPU.

    python -m litmus_student.train --config s192_l4 --data PRETOKENIZED --teacher-export TEACHER_EXPORT \\
        --experiment-id YYYY-MM-DD_slug --out RUN_DIR [--pilot]

**The step.** Each optimizer step draws `--texts-per-step` training texts (a third per origin, texts uniform within an origin, at
most `--max-train-tokens` tokens each; with `--uniform`, uniform over all texts in epochs), packs them into micro-batches of `--micro-tokens` positions, and computes the
losses of `distill.py` over all of them at once, the pair losses over the full `b x b` matrix. The step is exact
through gradient caching (Gao et al. 2021, arXiv 2101.06983): a first pass without gradients gives every text's
pooled vector; the losses and their gradients with respect to those vectors are computed on the whole matrix; a
second pass recomputes each micro-batch with gradients and back-propagates its slice. The cost is one extra forward
pass (about a third more compute); the memory is one micro-batch's activations whatever `b` is, so the matrix can
be 2,048 or 8,192 texts wide and several sizes still share one GPU. A memory bank of detached vectors from earlier
steps would not do: its vectors come from older weights and carry no gradient, so the step would not be the gradient
of the step's loss. The teacher's targets exist for every corpus row, including each passage the corpus
cut from a longer text, so texts are never cropped: a crop would have no teacher target.

**The schedule** follows the BitNet b1.58 training tips (Ma et al., "Training Tips, Code and FAQ", section 1): a
high peak learning rate decaying linearly to zero, dropped to a lower peak at half the run, and weight decay 0.1
removed at half the run, with AdamW betas (0.9, 0.95). A fixed 1e-4, suited to probes on frozen features, does not
transfer to quantization-aware training from scratch.

**Checkpoints** by validation loss (`best.pt`); `last.pt` at every evaluation. Validation also reads the bank, each
text chunked as inference reads it (a text past `chunk_tokens` scores the mean of its chunks' shares): PR AUC
(average precision), ROC AUC and balanced MCC of `1 - share_human` for model-written (ai or mixed) against human, and
the PR AUC of human against ai, the student's beside the teacher's on the same texts and bank texts. The summary is
the chosen checkpoint's, with the release rule read off it (`release/*`): the release is the smallest student
within 0.01 PR AUC of the teacher on validation, human against ai.

**Texts** are at most `--max-train-tokens` (8,192, the chunk) long: no corpus training row is longer, and every
text then has the teacher's own target. Whole documents past a chunk (`long_eval`) are never trained on;
`evaluate.py` reads them.

**`--pilot`** runs 200 steps and one evaluation, writes `pilot.json` and prints tokens per second, step time, peak
memory and the projected hours of `--steps`; it makes no W&B run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import numpy as np
import torch

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from torch import Tensor, nn

from foundry.logging import Run
from foundry.metrics import balanced_mcc
from foundry.metrics.ranking import binary_ranking_metrics
from foundry.training import clip_grad_norm, set_seed
from .contract import reference
from .data import (BANK_RECIPES, ClusterSampler, Cutter, EpochSampler, Presenter, Step, TeacherFiles, TextStore, TokenStore,
                   build_step, cluster_stratified_rows, pack, packed_batch, prefetched, recipe_bank_rows, teacher_targets)
from .dense import token_loss
from .distill import LossWeights, Projections, distillation_losses, pair_logits
from .model import StudentConfig, StudentModel, mask_stream, pool


PROJECT = "litmus"
PILOT_STEPS = 200
LOG_EVERY = 50
RELEASE_MARGIN = 0.01  # PR AUC, human against ai, below the teacher's on validation


# ---------------------------------------------------------------- schedule


def learning_rate(step: int, total: int, warmup: int, first_peak: float, second_peak: float) -> float:
    """Linear warmup, then `peak * (1 - step / total)`, with the first peak before half the run and the second after."""
    if step < warmup:
        return first_peak * (step + 1) / warmup
    peak = first_peak if step < total // 2 else second_peak

    return peak * max(0.0, 1.0 - step / total)


def weight_decay(step: int, total: int, decay: float) -> float:
    return decay if step < total // 2 else 0.0


def scheduled_step(step: int, total: int, elapsed: float, box: float) -> int:
    """The step the schedules read. With a time box of `box` seconds it is the further of the step itself and the
    box's elapsed share of `total`, so the learning rate reaches zero when the box closes, however fast steps run."""
    if box <= 0:
        return step

    return max(step, min(total, int(total * elapsed / box)))


def parameter_groups(modules: nn.Module, decay: float) -> list[dict[str, Any]]:
    """Weight decay on matrices (latent ternary weights, the embedding, the head's and projections' weights) only."""
    matrices = [parameter for parameter in modules.parameters() if parameter.ndim >= 2]
    others = [parameter for parameter in modules.parameters() if parameter.ndim < 2]

    return [{"params": matrices, "weight_decay": decay, "decays": True}, {"params": others, "weight_decay": 0.0, "decays": False}]


# ---------------------------------------------------------------- forward helpers


def autocast(device: torch.device, enabled: bool = True) -> contextlib.AbstractContextManager:
    """bfloat16 autocast on a GPU; float32 elsewhere, or when `enabled` is false (the release's own arithmetic)."""
    return torch.autocast(device.type, dtype=torch.bfloat16) if enabled and device.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def texts_pooled(model: StudentModel, texts: list[np.ndarray], micro_tokens: int, device: torch.device, precise: bool = False) -> Tensor:
    """Pooled statistics of texts of at most `chunk_tokens` tokens in the model's current mode, in order: (m, 2d) float32."""
    # texts: (l_i,) each of m
    lengths = np.asarray([len(ids) for ids in texts], dtype=np.int64)  # (m,)
    assert lengths.max(initial=0) <= model.config.chunk_tokens, "a text longer than a chunk is read as chunks (SPEC.md, Chunks)"
    W = model.config.block_size
    pooled = torch.empty(len(texts), 2 * model.config.width, device=device)  # (m, 2d)
    for plan in pack(lengths, micro_tokens, W):
        batch = packed_batch(plan, texts, W, W).to(device)
        with autocast(device, enabled=not precise):
            values = model.pooled(batch).float()  # (n, 2d)
        pooled[[placement.row for placement in plan]] = values

    return pooled  # (m, 2d)


def model_pooled(model: StudentModel, store: TokenStore, rows: np.ndarray, micro_tokens: int, device: torch.device,
                 precise: bool = False) -> Tensor:
    """Pooled statistics of corpus rows of at most `chunk_tokens` tokens, in row order: (m, 2d) float32."""
    # rows: (m,)
    return texts_pooled(model, [store.ids(int(row)) for row in rows], micro_tokens, device, precise)  # (m, 2d)


@torch.no_grad()
def model_vectors(model: StudentModel, store: TokenStore, rows: np.ndarray, micro_tokens: int, device: torch.device,
                  precise: bool = False) -> Tensor:
    """Unit queries and keys of corpus rows in eval mode, in row order: (m, 2w) float32."""
    # rows: (m,)
    model.eval()
    return model.head(model_pooled(model, store, rows, micro_tokens, device, precise))  # (m, 2w)


def chunked(texts: list[np.ndarray], chunk_tokens: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Each text cut into its chunks, each a text of its own (SPEC.md, Chunks), and the text each chunk came from:
    k chunks, (k,)."""
    # texts: (l_i,) each of m
    pieces, owner = [], []
    for index, ids in enumerate(texts):
        for start, end in reference.chunk_bounds(len(ids), chunk_tokens):
            pieces.append(ids[start:end])
            owner.append(index)

    return pieces, np.asarray(owner, dtype=np.int64)  # k chunks, (k,)


@torch.no_grad()
def chunked_shares(model: StudentModel, texts: list[np.ndarray], bank: Tensor, bank_labels: Tensor, micro_tokens: int,
                   device: torch.device) -> Tensor:
    """Each text's bank shares read as inference reads it: the mean over its chunks of each chunk's shares (the
    uncalibrated counterpart of the verdict's pooled probabilities): (m, 3)."""
    # texts: (l_i,) each of m; bank: (b, 2w); bank_labels: (b,)
    model.eval()
    pieces, owner = chunked(texts, model.config.chunk_tokens)  # k chunks, (k,)
    vectors = model.head(texts_pooled(model, pieces, micro_tokens, device))  # (k, 2w)
    shares = bank_shares(vectors, bank, bank_labels, float(model.head.scale), float(model.head.bias))  # (k, 3)
    owner_t = torch.from_numpy(owner).to(shares.device)  # (k,)
    total = torch.zeros(len(texts), 3, device=shares.device).index_add_(0, owner_t, shares)  # (m, 3)

    return total / torch.bincount(owner_t, minlength=len(texts)).float()[:, None]  # (m, 3)


def bank_coverage(vectors: Tensor, bank: Tensor, bank_labels: Tensor, scale: float, bias: float, origins: int = 3,
                  chunk: int = 4096) -> Tensor:
    """Each text's mean same-origin probability against each origin's bank texts: (n, 2w), (m, 2w) to (n, o)."""
    # vectors: (n, 2w); bank: (m, 2w); bank_labels: (m,) in 0 .. origins - 1
    one_hot = torch.nn.functional.one_hot(bank_labels, origins).float()  # (m, o)
    assert (one_hot.sum(dim=0) > 0).all(), "the bank holds every origin"

    return torch.cat([torch.sigmoid(pair_logits(vectors[first : first + chunk], bank, scale, bias)) @ one_hot
                      for first in range(0, len(vectors), chunk)]) / one_hot.sum(dim=0)  # (n, o)


def bank_shares(vectors: Tensor, bank: Tensor, bank_labels: Tensor, scale: float, bias: float, origins: int = 3) -> Tensor:
    """Each text's normalized coverage of the bank's origins: (n, 2w), (m, 2w) to (n, o)."""
    # vectors: (n, 2w); bank: (m, 2w); bank_labels: (m,)
    coverage = bank_coverage(vectors, bank, bank_labels, scale, bias, origins)  # (n, o)

    return coverage / coverage.sum(dim=-1, keepdim=True)  # (n, o)


def detection_metrics(labels: np.ndarray, shares: np.ndarray) -> dict[str, float]:
    """PR AUC, ROC AUC and balanced MCC of `1 - share_human` for model-written (ai or mixed) against human text, and
    the PR AUC of human against ai alone (mixed left out), the release margin's metric."""
    # labels: (n,); shares: (n, 3)
    written = (labels != 0).astype(np.int64)  # (n,)
    score = 1.0 - shares[:, 0]  # (n,)
    ranking = binary_ranking_metrics(written, score)
    pure = labels != 2  # (n,)
    human_ai = binary_ranking_metrics(written[pure], score[pure])

    return {"pr_auc": float(ranking["average_precision"]), "roc_auc": float(ranking["auroc"]),
            "mcc": balanced_mcc(written, score)["balanced_mcc"], "pr_auc_human_ai": float(human_ai["average_precision"])}


def release_margin(validation: dict[str, float]) -> dict[str, float]:
    """The release rule: the smallest student within RELEASE_MARGIN PR AUC of the teacher on
    validation, human against ai."""
    gap = validation["val/teacher_pr_auc_human_ai"] - validation["val/pr_auc_human_ai"]

    return {"release/pr_auc_human_ai": validation["val/pr_auc_human_ai"], "release/teacher_pr_auc_human_ai": validation["val/teacher_pr_auc_human_ai"],
            "release/gap": gap, "release/margin": RELEASE_MARGIN, "release/within_margin": float(gap <= RELEASE_MARGIN)}


# ---------------------------------------------------------------- the step


@dataclass
class Masking:
    """A step's masked language modeling (`model.mask_stream`, `model.MaskedLanguageHead`)."""

    mask_id: int
    text_share: float  # of texts masked
    weight: float  # of the mean cross-entropy of the chosen positions, beside the pair loss
    generator: torch.Generator  # on the device; drawn on at every step, so a text drawn again is masked anew


def train_step(model: StudentModel, projections: Projections, step: Step, teacher: TeacherFiles, weights: LossWeights,
               device: torch.device, token_targets: list[Tensor] | None = None, token_weight: float = 0.0,
               masking: Masking | None = None) -> dict[str, float]:
    """One exact gradient-cached step over all of a step's texts; returns the losses. With `token_targets` (the
    teacher's state at every real token of each micro-batch, `dense.py`), the per-token loss joins each micro-batch's
    second pass at `token_weight`, averaged over the micro-batches. With `masking`, a share of the texts is masked in
    both passes (so the pair loss reads them masked too), and the masked language loss, the mean over the step's
    chosen positions, joins the second pass at `masking.weight`."""
    model.train()
    batches = [batch.to(device, non_blocking=True) for batch in step.micro_batches]
    if masking is not None:
        assert token_targets is None and model.mlm is not None, "masking needs the model's masked language head and no per-token loss"
        masked = [mask_stream(batch, masking.mask_id, model.config.vocab_size, masking.text_share, masking.generator) for batch in batches]
        batches, targets = [batch for batch, _ in masked], [target for _, target in masked]
    with torch.no_grad(), autocast(device):
        parts = [model.pooled(batch).float() for batch in batches]  # each (n_i, 2d)
    pooled = torch.cat(parts).requires_grad_(True)  # (b, 2d)
    standardized = model.head.standardize(pooled)  # (b, 2d)
    losses = distillation_losses(standardized, model.head.project(standardized), model.head.scale, model.head.bias, step.labels.to(device),
                                 step.teacher_embeddings.to(device), step.teacher_query_key.to(device), teacher.scale,
                                 teacher.bias, projections, weights)
    losses["total"].backward()
    assert pooled.grad is not None, "the losses reach every pooled vector"
    gradients = pooled.grad.split([batch.texts for batch in batches])
    if masking is not None:
        assert model.mlm is not None
        chosen = max(1, int(sum(int((target >= 0).sum()) for target in targets)))
        language_total = 0.0
        for batch, gradient, target in zip(batches, gradients, targets, strict=True):
            with autocast(device):
                states = model.encoder(batch)  # (t, d)
            part = pool(states, batch.segment, batch.texts)  # (n_i, 2d)
            language = model.mlm.loss(states, target, model.encoder.embed) / chosen  # ()
            ((part * gradient).sum() + masking.weight * language).backward()
            language_total += float(language.detach())
        reported = {name: float(value.detach()) for name, value in losses.items()}
        reported["mlm"] = language_total
        reported["total"] += masking.weight * language_total

        return reported
    if token_targets is None:
        for batch, gradient in zip(batches, gradients, strict=True):
            with autocast(device):
                part = model.pooled(batch).float()  # (n_i, 2d)
            part.backward(gradient)

        return {name: float(value.detach()) for name, value in losses.items()}
    assert projections.token is not None, "the per-token loss needs the projections' token map"
    token_total = 0.0
    for batch, gradient, targets in zip(batches, gradients, token_targets, strict=True):
        with autocast(device):
            states = model.encoder(batch)  # (t, d)
        part = pool(states, batch.segment, batch.texts)  # (n_i, 2d)
        token = token_loss(states, batch.segment, targets, projections.token) / len(batches)  # ()
        ((part * gradient).sum() + token_weight * token).backward()
        token_total += float(token.detach())
    reported = {name: float(value.detach()) for name, value in losses.items()}
    reported["token"] = token_total
    reported["total"] += token_weight * token_total

    return reported


# ---------------------------------------------------------------- evaluation


class Evaluation:
    """Fixed validation texts and bank texts, read the same way at every evaluation."""

    def __init__(self, store: TokenStore, teacher: TeacherFiles, arguments: argparse.Namespace, generator: np.random.Generator,
                 device: torch.device) -> None:
        self.store, self.teacher, self.device, self.micro_tokens = store, teacher, device, arguments.micro_tokens
        chunk_tokens = reference.named_config(arguments.config).chunk_tokens
        # The losses need one teacher target per text, so their texts are at most a chunk long.
        self.loss_rows = ClusterSampler(store, "validation", chunk_tokens, teacher.present).draw(arguments.val_texts, generator)  # (v,)
        labels, embeddings, query_key = teacher_targets(self.loss_rows, store, teacher)  # (v,), (v, 2D), (v, 2w_T)
        self.loss_labels, self.loss_embeddings, self.loss_query_key = labels.to(device), embeddings.to(device), query_key.to(device)
        # Read texts of any length, chunked as inference reads them; both the student and the teacher must hold them.
        read_rows = cluster_stratified_rows(store, "validation", arguments.eval_per_origin, generator)  # (r',)
        self.read_rows = np.sort(read_rows[teacher.present[read_rows]])  # (r,)
        # A bank text is one vector, so at most a chunk long.
        if arguments.bank_recipe == "split":
            bank_rows = cluster_stratified_rows(store, arguments.bank_split, arguments.bank_per_origin, generator)  # (m',)
        else:  # the rows export draws for the same recipe, size and seed, so the checkpoint is chosen with the release's bank
            bank_rows = recipe_bank_rows(store, arguments.bank_recipe, arguments.bank_per_origin, arguments.seed)  # (m',)
        self.bank_rows = np.sort(bank_rows[(store.length[bank_rows] <= chunk_tokens) & teacher.present[bank_rows]])  # (m,)
        self.teacher_metrics: dict[str, float] = {}
        if teacher.missing:  # a labels-only run without the teacher's files has no teacher reading
            return
        teacher_read = torch.from_numpy(np.asarray(teacher.query_key[self.read_rows], dtype=np.float32))  # (r, 2w_T)
        teacher_bank = torch.from_numpy(np.asarray(teacher.query_key[self.bank_rows], dtype=np.float32))  # (m, 2w_T)
        bank_labels = torch.from_numpy(store.label[self.bank_rows].astype(np.int64))  # (m,)
        shares = bank_shares(teacher_read, teacher_bank, bank_labels, teacher.scale, teacher.bias)  # (r, 3)
        self.teacher_metrics = detection_metrics(store.label[self.read_rows], shares.numpy())

    @torch.no_grad()
    def __call__(self, model: StudentModel, projections: Projections, weights: LossWeights) -> dict[str, float]:
        model.eval()
        pooled = model_pooled(model, self.store, self.loss_rows, self.micro_tokens, self.device)  # (v, 2d)
        standardized = model.head.standardize(pooled)  # (v, 2d)
        losses = distillation_losses(standardized, model.head.project(standardized), model.head.scale, model.head.bias,
                                     self.loss_labels, self.loss_embeddings, self.loss_query_key, self.teacher.scale,
                                     self.teacher.bias, projections, weights)
        bank = model_vectors(model, self.store, self.bank_rows, self.micro_tokens, self.device)  # (m, 2w)
        bank_labels = torch.from_numpy(self.store.label[self.bank_rows].astype(np.int64)).to(self.device)  # (m,)
        texts = [self.store.ids(int(row)) for row in self.read_rows]
        shares = chunked_shares(model, texts, bank, bank_labels, self.micro_tokens, self.device)  # (r, 3)
        metrics = {f"val/{'loss' if name == 'total' else name}": float(value) for name, value in losses.items()}
        metrics.update({f"val/{name}": value for name, value in detection_metrics(self.store.label[self.read_rows], shares.cpu().numpy()).items()})
        metrics.update({f"val/teacher_{name}": value for name, value in self.teacher_metrics.items()})

        return metrics


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path, model: StudentModel, projections: Projections, optimizer: torch.optim.Optimizer, step: int,
                    validation: dict[str, float], run: dict[str, Any] | None, arguments: argparse.Namespace) -> None:
    temporary = path.with_suffix(".partial")
    torch.save({"config": asdict(model.config), "quantized": model.quantized, "integer_attention": model.integer_attention,
                "int8_output": model.head.int8_output, "mlm": model.mlm is not None, "model": model.state_dict(), "projections": projections.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "validation": validation, "run": run,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[StudentModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = StudentModel(StudentConfig(**checkpoint["config"]), checkpoint.get("quantized", True),
                         checkpoint.get("integer_attention", False), checkpoint.get("int8_output", False),
                         checkpoint.get("mlm", False)).to(device)
    model.load_state_dict(checkpoint["model"])

    return model, checkpoint


# ---------------------------------------------------------------- the run


def step_maker(sampler: ClusterSampler | EpochSampler, store: TokenStore, teacher: TeacherFiles, arguments: argparse.Namespace,
               block_size: int, generator: np.random.Generator, presenter: Presenter | None = None) -> Callable[[], Step]:
    """Draw and pack the next step's texts; called in order on one thread. With a `presenter` (`--texts`), each text
    is presented anew at every draw; with `--cut-ends` or `--cut-starts`, each text is perhaps cut mid-sentence
    (`Cutter`); each from its own generator, so the draws stay those of the seed."""
    cutter = None
    if arguments.cut_ends or arguments.cut_starts:
        cutter = Cutter.from_tokenizer(reference.load_tokenizer(), arguments.cut_ends, arguments.cut_starts,
                                       np.random.default_rng([arguments.seed, 1]))

    def make() -> Step:
        rows = sampler.draw(arguments.texts_per_step, generator)  # (b,)
        return build_step(rows, store, teacher, arguments.micro_tokens, block_size, arguments.pack_multiple, cutter, presenter)

    return make


def searched_settings(arguments: argparse.Namespace, config: StudentConfig) -> tuple[StudentConfig, LossWeights, list[str]]:
    """The run's head and loss weights, and the schedule written into `arguments`, from the distillation search's
    `summary.json` (`--settings`). Only what the search's rule took replaces a default; returns what was taken."""
    from .search import SCHEDULE, loss_weights  # search.py imports this module

    weights = LossWeights(arguments.w_embedding, arguments.w_query_key, arguments.w_pair_soft, arguments.w_pair_rank,
                          arguments.w_pair_label, arguments.temperature)
    if arguments.settings is None:
        return config, weights, []
    recommendation = json.loads(arguments.settings.read_text(encoding="utf-8"))["recommendation"]
    settings, taken = recommendation["settings"], list(recommendation["taken"])
    if any(name not in SCHEDULE for name in taken):  # the loss and head settings are taken together
        weights = loss_weights(settings)
        config = replace(config, head_hidden=int(settings["head_hidden"]), head_out=int(settings["head_out"]),
                         head_clamp=float(settings["head_clamp"]))
    if any(name in SCHEDULE for name in taken):  # and so is the schedule
        arguments.lr = float(settings["learning_rate"])
        arguments.lr_second = arguments.lr * float(settings["second_ratio"])
        arguments.warmup = max(1, round(float(settings["warmup_share"]) * arguments.steps))
        arguments.weight_decay = float(settings["weight_decay"])
        arguments.decay_stays = not settings["decay_drops"]
        arguments.texts_per_step = int(settings["texts_per_step"])

    return config, weights, taken


def labels_only(weights: LossWeights) -> LossWeights:
    """The control with no teacher: every teacher term off and their whole weight on the true-label pair loss, so the
    same student on the same batches, head and schedule learns from the origins alone."""
    total = weights.embedding + weights.query_key + weights.pair_soft + weights.pair_rank + weights.pair_label

    return LossWeights(0.0, 0.0, 0.0, 0.0, total, weights.temperature)


def train(arguments: argparse.Namespace) -> dict[str, float]:
    set_seed(arguments.seed)
    device = torch.device(arguments.device)
    if device.type == "cuda" and arguments.memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(arguments.memory_fraction)  # the current device; `cuda` has no index to pass
    torch.backends.cuda.matmul.allow_tf32 = True
    generator = np.random.default_rng(arguments.seed)
    given_texts = arguments.texts_per_step  # given, it wins over the search's
    store = TokenStore(arguments.data)
    if arguments.teacher_export is None:
        if not arguments.labels_only:
            raise ValueError("distillation needs --teacher-export; only a --labels-only run trains without the teacher's files")
        teacher = TeacherFiles.absent(len(store))
    else:
        teacher = TeacherFiles.load(arguments.teacher_export, np.flatnonzero(store.split_mask("train")), generator)
    if teacher.query_key.shape[0] != len(store):
        raise ValueError(f"the teacher holds {teacher.query_key.shape[0]} rows and the corpus {len(store)}")
    if arguments.uniform:
        sampler: ClusterSampler | EpochSampler = EpochSampler(store, "train", arguments.max_train_tokens, teacher.present)
    else:
        sampler = ClusterSampler(store, "train", arguments.max_train_tokens, teacher.present, arguments.clusters_uniform)
    if arguments.epochs or arguments.eval_every_epochs:  # counted in steps of the given texts per step, before the schedule reads them
        if given_texts is None:
            raise ValueError("--epochs and --eval-every-epochs count steps of --texts-per-step texts; give it")
        per_epoch = len(sampler.rows if isinstance(sampler, ClusterSampler) else sampler.all_rows) / given_texts
        if arguments.epochs:
            arguments.steps = math.ceil(arguments.epochs * per_epoch)
        if arguments.eval_every_epochs:
            arguments.eval_every = max(1, round(arguments.eval_every_epochs * per_epoch))
        print(f"{per_epoch:,.0f} steps an epoch; {arguments.steps:,} steps at most, validation every {arguments.eval_every:,}", file=sys.stderr)
    config, weights, taken = searched_settings(arguments, reference.named_config(arguments.config))
    arguments.texts_per_step = given_texts or arguments.texts_per_step or 2048
    if arguments.labels_only:
        weights = labels_only(weights)
    if arguments.max_train_tokens > config.chunk_tokens:
        raise ValueError(f"--max-train-tokens {arguments.max_train_tokens} exceeds the chunk of {config.chunk_tokens} tokens, the most read as one text")
    masking = None
    if arguments.mlm_share > 0:
        mask_id = reference.load_tokenizer().token_to_id("[MASK]")
        assert mask_id is not None and mask_id < config.vocab_size, "the tokenizer has a [MASK] token inside the vocabulary"
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(int(np.random.default_rng([arguments.seed, 3]).integers(2**62)))
        masking = Masking(mask_id, arguments.mlm_share, arguments.mlm_weight, mask_generator)
    model = StudentModel(config, integer_attention=arguments.integer_attention, int8_output=arguments.int8_output,
                         mlm=masking is not None).to(device)
    if arguments.init is not None:  # a warm start: the weights of an earlier run, a new optimizer and schedule
        model.load_state_dict(load_checkpoint(arguments.init, device)[0].state_dict())
    projections = Projections(2 * config.width, teacher.embedding_width, config.head_out, teacher.query_key.shape[1] // 2).to(device)
    if arguments.compile:
        model.encoder = torch.compile(model.encoder, dynamic=True)
    trainable = nn.ModuleDict({"model": model, "projections": projections})
    optimizer = torch.optim.AdamW(parameter_groups(trainable, arguments.weight_decay), lr=arguments.lr, betas=(0.9, 0.95),
                                  fused=device.type == "cuda")
    evaluation = Evaluation(store, teacher, arguments, generator, device)
    total = arguments.pilot_steps if arguments.pilot else arguments.steps
    arguments.out.mkdir(parents=True, exist_ok=True)
    run_config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
                  "student": asdict(config), "loss_weights": asdict(weights), "settings_taken": taken,
                  "teacher_scale": teacher.scale, "teacher_bias": teacher.bias,
                  "experiment_id": arguments.experiment_id}
    presenter = None
    if arguments.texts is not None:
        if len(TextStore(arguments.texts)) != len(store):
            raise ValueError(f"the text store {arguments.texts} holds another corpus than {arguments.data}")
        presenter = Presenter(arguments.texts, arguments.synonyms, arguments.max_train_tokens, arguments.present_workers,
                              np.random.default_rng([arguments.seed, 2]))

    # The checkpoint is chosen on validation loss, or on validation PR AUC (model-written against human, the reading's
    # bank shares); with --patience, training stops once that many evaluations in a row fail to improve on it.
    key = "val/loss" if arguments.select == "loss" else "val/pr_auc"
    sign = 1.0 if arguments.select == "loss" else -1.0  # lower is better after the sign
    best = math.inf
    stale = 0
    validation: dict[str, float] = {}
    best_validation: dict[str, float] = {}
    clock: list[tuple[float, int]] = []
    try:
        with Run(arguments.out / "wandb", run_config, project=PROJECT, group=arguments.experiment_id, job_type="train",
                 name=(f"{arguments.experiment_id}_{arguments.config}{'_integer_attention' if arguments.integer_attention else ''}{'_int8_output' if arguments.int8_output else ''}"
                       f"{'_labels_only' if arguments.labels_only else ''}{'_warm' if arguments.init is not None else ''}"),
                 pilot=arguments.pilot) as run:
            progress = prefetched(step_maker(sampler, store, teacher, arguments, config.block_size, generator, presenter), total, depth=3,
                                  pin=device.type == "cuda")
            began = time.perf_counter()
            for index, step in enumerate(progress):
                scheduled = scheduled_step(index, arguments.steps, time.perf_counter() - began, 60.0 * arguments.minutes)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(scheduled, arguments.steps, arguments.warmup, arguments.lr, arguments.lr_second)
                    decay = arguments.weight_decay if arguments.decay_stays else weight_decay(scheduled, arguments.steps, arguments.weight_decay)
                    group["weight_decay"] = decay if group["decays"] else 0.0
                losses = train_step(model, projections, step, teacher, weights, device, masking=masking)
                if not math.isfinite(losses["total"]):
                    raise FloatingPointError(f"step {index}: the loss is {losses['total']}")
                gradient_norm = clip_grad_norm(trainable, arguments.clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                clock.append((time.perf_counter(), step.tokens))
                if index % LOG_EVERY == 0 or index == total - 1:
                    rate = throughput(clock)
                    run.log({**{f"train/{'loss' if name == 'total' else name}": value for name, value in losses.items()},
                             "train/lr": optimizer.param_groups[0]["lr"], "train/grad_norm": gradient_norm, "train/tokens_per_second": rate}, step=index)
                    shown = ", ".join(f"{name} {value:.4f}" for name, value in losses.items() if name in ("total", "pair_label", "mlm"))
                    print(f"step {index}: {shown}, {rate:,.0f} tokens/s", file=sys.stderr)
                last = index == total - 1 or scheduled >= arguments.steps  # the last step, or the time box closed
                if (index + 1) % arguments.eval_every == 0 or last:
                    started = time.perf_counter()
                    validation = evaluation(model, projections, weights)
                    validation["val/seconds"] = time.perf_counter() - started
                    run.log(validation, step=index)
                    save_checkpoint(arguments.out / "last.pt", model, projections, optimizer, index, validation, run.identity, arguments)
                    if sign * validation[key] < best:
                        best, stale = sign * validation[key], 0
                        best_validation = {**validation, "val/step": float(index)}
                        save_checkpoint(arguments.out / "best.pt", model, projections, optimizer, index, validation, run.identity, arguments)
                    else:
                        stale += 1
                    print(json.dumps({**validation, "stale": stale}), file=sys.stderr)
                    if arguments.patience and stale >= arguments.patience:
                        print(f"stopped: {stale} evaluations in a row without a better {key}", file=sys.stderr)
                        last = True
                if last:
                    progress.close()  # a closed time box leaves steps unbuilt
                    break
            # The summary is the chosen checkpoint's, with the release rule read off it.
            summary = {f"val/best_{arguments.select}": sign * best, **{name: value for name, value in best_validation.items() if name != "val/seconds"},
                       **({} if teacher.missing else release_margin(best_validation))}
            print(json.dumps({name: value for name, value in summary.items() if name.startswith("release/")}), file=sys.stderr)
            if arguments.pilot:
                summary.update(pilot_report(arguments, clock, validation.get("val/seconds", 0.0), device))
            run.finish(summary=summary)
    finally:
        if presenter is not None:
            presenter.close()

    return summary


def timed_window(clock: list[tuple[float, int]], skip: int = 5) -> list[tuple[float, int]]:
    """The steps after the first `skip` (compilation and warm caches), or all of a shorter run."""
    return clock[min(skip, max(0, len(clock) - 2)) :]


def throughput(clock: list[tuple[float, int]]) -> float:
    """Tokens per second over the timed window; each entry is (end time, tokens) of a step."""
    window = timed_window(clock)
    if len(window) < 2:
        return 0.0
    return sum(tokens for _, tokens in window[1:]) / (window[-1][0] - window[0][0])


def pilot_report(arguments: argparse.Namespace, clock: list[tuple[float, int]], eval_seconds: float, device: torch.device) -> dict[str, float]:
    """Tokens per second, step time, peak memory and the projected hours of the full run, written to `pilot.json`."""
    times = np.diff([moment for moment, _ in timed_window(clock)])  # (s,)
    step_seconds = float(np.median(times)) if len(times) else 0.0
    evaluations = arguments.steps // arguments.eval_every + 1
    projected = (arguments.steps * step_seconds + evaluations * eval_seconds) / 3600
    report = {"cost/tokens_per_second": throughput(clock), "cost/step_seconds": step_seconds,
              "cost/eval_seconds": eval_seconds, "cost/projected_hours": projected,
              "cost/peak_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0}
    (arguments.out / "pilot.json").write_text(json.dumps({**report, "steps": arguments.steps, "config": arguments.config}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1), file=sys.stderr)

    return report


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help=f"one of {', '.join(reference.config_names())}")
    parser.add_argument("--data", type=Path, required=True, help="the pre-tokenized corpus of `litmus_student.data`")
    parser.add_argument("--teacher-export", type=Path, default=None,
                        help="the directory of the teacher's manifest.json; a --labels-only run may leave it out and read no teacher file")
    parser.add_argument("--experiment-id", required=True, help="YYYY-MM-DD_slug, the W&B group")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--minutes", type=float, default=0.0,
                        help="a time box: the schedules follow the clock when it runs ahead of the steps, and the run ends when it closes")
    parser.add_argument("--epochs", type=float, default=0.0,
                        help="the most epochs of the training texts to run, in steps of --texts-per-step, in place of --steps")
    parser.add_argument("--texts-per-step", type=int, default=None,
                        help="given, it wins over the search's; default the search's when --settings gives it, else 2048")
    parser.add_argument("--micro-tokens", type=int, default=65536, help="positions per micro-batch, a multiple of the block size")
    parser.add_argument("--pack-multiple", type=int, default=8192, help="streams pad to a multiple of this, so shapes repeat")
    parser.add_argument("--max-train-tokens", type=int, default=8192, help="at most the config's chunk_tokens")
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--lr-second", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-every-epochs", type=float, default=0.0, help="validation every this share of an epoch, in place of --eval-every")
    parser.add_argument("--select", choices=["loss", "pr_auc"], default="loss",
                        help="the checkpoint kept as best.pt: lowest validation loss, or highest validation PR AUC")
    parser.add_argument("--patience", type=int, default=0, help="stop once this many evaluations in a row fail to improve --select; 0 never")
    parser.add_argument("--val-texts", type=int, default=4096)
    parser.add_argument("--eval-per-origin", type=int, default=3000)
    parser.add_argument("--bank-split", default="bank")
    parser.add_argument("--bank-per-origin", type=int, default=4096)
    parser.add_argument("--bank-recipe", choices=["split", *BANK_RECIPES], default="split",
                        help="the bank the validation readings use: --bank-split's texts over clusters (default), or the export "
                             "recipe of the same name (mixed_out, for a split with no `bank` set)")
    for name, default in asdict(LossWeights()).items():
        parser.add_argument(f"--{'temperature' if name == 'temperature' else 'w-' + name.replace('_', '-')}", type=float, default=default)
    parser.add_argument("--decay-stays", action="store_true", help="keep the weight decay past half the run (default: it drops to zero)")
    parser.add_argument("--settings", type=Path, default=None,
                        help="the distillation search's summary.json; what its rule took replaces the settings above")
    parser.add_argument("--init", type=Path, default=None,
                        help="a checkpoint of the same config whose model weights start this run (a warm start)")
    parser.add_argument("--labels-only", action="store_true",
                        help="the control with no teacher: only the true-label pair loss, at the same total weight")
    parser.add_argument("--clusters-uniform", action="store_true",
                        help="draw each origin's clusters uniformly instead of its texts (the default keeps the corpus's cell balance)")
    parser.add_argument("--integer-attention", action="store_true",
                        help="train with integer attention (model.integer_block_attention), SPEC.md's format 2")
    parser.add_argument("--int8-output", action="store_true",
                        help="train with the head's query and key quantized to int8 (model.StudentHead), so the loss sees the unit vectors of "
                             "the int8 codes a SPEC.md format 3 release stores and compares; export releases such a checkpoint as int8 by default")
    parser.add_argument("--cut-ends", type=float, default=0.0,
                        help="the chance a training text is cut at a word start in its last quarter (data.Cutter); the same for every origin")
    parser.add_argument("--cut-starts", type=float, default=0.0,
                        help="the chance a training text loses its tokens before a word start in its first quarter")
    parser.add_argument("--texts", type=Path, default=None,
                        help="the text store (`data --texts`): each training text presented anew at every draw (data.Presenter); "
                             "validation, test, calibration and the bank read `--data` as stored")
    parser.add_argument("--synonyms", type=Path, default=None, help="the synonym table of the presentation's swaps; none swapped without it")
    parser.add_argument("--present-workers", type=int, default=16, help="processes presenting and tokenizing training texts")
    parser.add_argument("--uniform", action="store_true",
                        help="draw training texts uniformly over all of them, in epochs (data.EpochSampler), whatever their origin")
    parser.add_argument("--mlm-share", type=float, default=0.0, help="the share of each step's texts masked for the masked language loss; 0 none")
    parser.add_argument("--mlm-weight", type=float, default=0.1, help="the masked language loss's weight beside the pair loss")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=1.0, help="of the GPU, so several sizes share one")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=PILOT_STEPS)

    return parser


def main() -> None:
    train(parser().parse_args())


if __name__ == "__main__":
    main()
