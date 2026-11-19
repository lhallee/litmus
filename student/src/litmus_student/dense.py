"""Per-token distillation: the student's token states against the frozen teacher's last hidden states.

The losses of `distill.py` give a text one target, its pooled embedding. Here every token is a target: the teacher
encoder reads the step's texts as the student trains, each as `[CLS] tokens [SEP]`, and the student's state at a
token, through a learned linear map, is held to the teacher's state at the same token (`1 - cosine` plus mean squared
error, the teacher's states standardized per column by running statistics). The teacher's pooled embedding is the
mean and variance of exactly these states, so this is the embedding loss before the pooling threw the tokens away.

The student's tokens carry no special tokens; the teacher's two are added here and their states dropped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import Tensor, nn

from .model import PackedBatch


TEACHER_NAME = "answerdotai/ModernBERT-base"
TEACHER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
CLS_ID, SEP_ID, PAD_ID = 50281, 50282, 50283
MOMENTUM = 0.01
EPS = 1e-6


def load_teacher_encoder(device: torch.device) -> nn.Module:
    """The frozen teacher encoder at its pinned revision, float32 weights (bfloat16 runs through autocast)."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(TEACHER_NAME, revision=TEACHER_REVISION, dtype=torch.float32, attn_implementation="sdpa")
    tokenizer_ids = (model.config.cls_token_id, model.config.sep_token_id, model.config.pad_token_id)
    assert tokenizer_ids == (CLS_ID, SEP_ID, PAD_ID), f"{TEACHER_NAME} has special tokens {tokenizer_ids}"

    return model.to(device).eval().requires_grad_(False)


def length_groups(lengths: Tensor, budget: int) -> list[Tensor]:
    """Texts grouped by length, each group's padded size (texts times its longest, specials included) within
    `budget` positions where more than one text is in it: (n,) to index tensors that partition 0 .. n - 1."""
    # lengths: (n,)
    order = torch.argsort(lengths).tolist()
    groups: list[list[int]] = [[]]
    for text in order:
        longest = int(lengths[text]) + 2  # ascending, so the newest text is the group's longest
        if groups[-1] and (len(groups[-1]) + 1) * longest > budget:
            groups.append([])
        groups[-1].append(text)

    return [torch.tensor(group, dtype=torch.long, device=lengths.device) for group in groups if group]


class TokenTeacher(nn.Module):
    """The teacher's standardized state at every real token of a packed batch."""

    def __init__(self, encoder: nn.Module, width: int, budget: int = 65536) -> None:
        super().__init__()
        self.encoder = encoder
        self.width = width
        self.budget = budget
        self.register_buffer("running_mean", torch.zeros(width))  # (D,)
        self.register_buffer("running_var", torch.ones(width))  # (D,)
        self.register_buffer("seeded", torch.zeros((), dtype=torch.bool))  # ()

    @torch.no_grad()
    def states(self, batch: PackedBatch) -> Tensor:
        """The teacher's last hidden state at each real token, in the stream's order: (t_v, D) float32."""
        valid = batch.segment >= 0  # (t,)
        segment, position, tokens = batch.segment[valid], batch.position[valid], batch.tokens[valid]  # (t_v,) each
        lengths = torch.bincount(segment, minlength=batch.texts)  # (n,)
        assert int(lengths.min()) > 0, "every text of a packed batch has a token"
        out = torch.empty(len(tokens), self.width, device=tokens.device)  # (t_v, D)
        group_of = torch.empty(batch.texts, dtype=torch.long, device=tokens.device)  # (n,)
        row_of = torch.empty_like(group_of)  # (n,)
        groups = length_groups(lengths, self.budget)
        for index, group in enumerate(groups):
            group_of[group] = index
            row_of[group] = torch.arange(len(group), device=tokens.device)
        token_group, token_row = group_of[segment], row_of[segment]  # (t_v,) each
        for index, group in enumerate(groups):
            sizes = lengths[group]  # (g,)
            width = int(sizes.max()) + 2
            chosen = token_group == index  # (t_v,)
            rows, columns = token_row[chosen], position[chosen] + 1  # (t_g,) each
            ids = torch.full((len(group), width), PAD_ID, dtype=torch.long, device=tokens.device)  # (g, l)
            ids[rows, columns] = tokens[chosen]
            ids[:, 0] = CLS_ID
            ids[torch.arange(len(group), device=tokens.device), sizes + 1] = SEP_ID
            mask = torch.arange(width, device=tokens.device)[None, :] <= (sizes + 1)[:, None]  # (g, l)
            with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=tokens.device.type == "cuda"):
                hidden = self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state  # (g, l, D)
            out[chosen] = hidden[rows, columns].float()

        return out  # (t_v, D)

    @torch.no_grad()
    def targets(self, batch: PackedBatch) -> Tensor:
        """`states`, each column standardized by running statistics the first batch seeds: (t_v, D) float16."""
        states = self.states(batch)  # (t_v, D)
        momentum = 1.0 if not bool(self.seeded) else MOMENTUM
        self.running_mean.lerp_(states.mean(dim=0), momentum)
        self.running_var.lerp_(states.var(dim=0, unbiased=False), momentum)
        self.seeded.fill_(True)

        return ((states - self.running_mean) / torch.sqrt(self.running_var + EPS)).half()  # (t_v, D)


def token_loss(states: Tensor, segment: Tensor, targets: Tensor, projection: nn.Module) -> Tensor:
    """`1 - cosine` plus mean squared error of the student's mapped token states against the teacher's: ()."""
    # states: (t, d); segment: (t,); targets: (t_v, D)
    predicted = projection(states[segment >= 0].float())  # (t_v, D)
    targets = targets.float()  # (t_v, D)

    return (1.0 - F.cosine_similarity(predicted, targets, dim=-1)).mean() + F.mse_loss(predicted, targets)
