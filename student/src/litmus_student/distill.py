"""Distillation losses: the student's texts against the frozen teacher's, per text and per pair of texts.

Per text:
- **embedding**: the student's standardized pooled statistics through a learned linear map against the teacher's
  standardized embedding (mean and variance of its last states): `1 - cosine` plus mean squared error.
- **query_key**: the student head's unit query and key, each through a learned linear map and renormalized, against
  the teacher head's: `1 - cosine`, query and key averaged.

Per pair, over the step's whole `b x b` matrix of pair logits (the diagonal, a text with itself, left out):
- **pair_soft**: binary cross-entropy of the student's logit against the teacher's probability, every cell.
- **pair_rank**: cross-entropy of each row's softmax against the teacher's row softmax, which keeps the teacher's
  order of which texts are nearest a text (the bank's reading is an average over such a row).
- **pair_label**: binary cross-entropy against the true same-origin label, the teacher head's own objective.

Cross-entropies are `foundry.layers.cross_entropy`'s, which keep the gradient of a confident target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataclasses import dataclass, fields
from torch import Tensor, nn

from foundry.layers.cross_entropy import binary_cross_entropy_with_logits, soft_target_cross_entropy


DIAGONAL_LOGIT = -1e4  # a self-pair's logit in the row softmax: probability zero, gradient zero, no NaN


@dataclass(frozen=True)
class LossWeights:
    embedding: float = 1.0
    query_key: float = 1.0
    pair_soft: float = 1.0
    pair_rank: float = 0.5
    pair_label: float = 1.0
    temperature: float = 1.0  # divides both matrices' logits in pair_soft and pair_rank

    def names(self) -> list[str]:
        return [field.name for field in fields(self) if field.name != "temperature"]


class Projections(nn.Module):
    """The student's vectors mapped into the teacher's spaces, for the per-text losses only; never shipped."""

    def __init__(self, pooled_width: int, teacher_width: int, head_width: int, teacher_head_width: int,
                 token_widths: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.embedding = nn.Linear(pooled_width, teacher_width)
        self.query = nn.Linear(head_width, teacher_head_width)
        self.key = nn.Linear(head_width, teacher_head_width)
        # The student's token state to the teacher's, for the per-token loss of `dense.py`: (d, D).
        self.token = nn.Linear(*token_widths) if token_widths is not None else None


def pair_logits(rows: Tensor, columns: Tensor, scale: Tensor | float, bias: Tensor | float) -> Tensor:
    """The same-origin logit of each row text and each column text, `scale * (q_r . k_c + q_c . k_r) / 2 + bias`:
    (n, 2w), (m, 2w) to (n, m)."""
    # rows: (n, 2w); columns: (m, 2w); scale, bias: ()
    query_r, key_r = rows.chunk(2, dim=-1)  # (n, w) each
    query_c, key_c = columns.chunk(2, dim=-1)  # (m, w) each

    return scale * 0.5 * (query_r @ key_c.T + key_r @ query_c.T) + bias  # (n, m)


def same_origin(labels: Tensor) -> Tensor:
    """Whether each pair of texts shares an origin: (b,) to (b, b) float."""
    # labels: (b,)
    return (labels[:, None] == labels[None, :]).float()  # (b, b)


def distillation_losses(standardized: Tensor, query_key: Tensor, student_scale: Tensor, student_bias: Tensor, labels: Tensor,
                        teacher_embeddings: Tensor, teacher_query_key: Tensor, teacher_scale: float, teacher_bias: float,
                        projections: Projections, weights: LossWeights) -> dict[str, Tensor]:
    """Each loss and their weighted sum, `total`, for one step's texts, all float32."""
    # standardized: (b, 2d); query_key: (b, 2w); student_scale, student_bias: (); labels: (b,); teacher_embeddings: (b, 2D);
    # teacher_query_key: (b, 2w_T)
    b = labels.shape[0]
    off_diagonal = ~torch.eye(b, dtype=torch.bool, device=labels.device)  # (b, b)
    losses: dict[str, Tensor] = {}

    predicted = projections.embedding(standardized)  # (b, 2D)
    cosine = F.cosine_similarity(predicted, teacher_embeddings, dim=-1)  # (b,)
    losses["embedding"] = (1.0 - cosine).mean() + F.mse_loss(predicted, teacher_embeddings)

    query, key = query_key.chunk(2, dim=-1)  # (b, w) each
    teacher_query, teacher_key = teacher_query_key.chunk(2, dim=-1)  # (b, w_T) each
    query_cosine = (F.normalize(projections.query(query), dim=-1) * teacher_query).sum(dim=-1)  # (b,)
    key_cosine = (F.normalize(projections.key(key), dim=-1) * teacher_key).sum(dim=-1)  # (b,)
    losses["query_key"] = 1.0 - 0.5 * (query_cosine + key_cosine).mean()

    student = pair_logits(query_key, query_key, student_scale, student_bias)  # (b, b)
    with torch.no_grad():
        teacher = pair_logits(teacher_query_key, teacher_query_key, teacher_scale, teacher_bias)  # (b, b)
    tau = weights.temperature
    soft = torch.sigmoid(teacher / tau)  # (b, b)
    losses["pair_soft"] = binary_cross_entropy_with_logits(student / tau, soft, reduction="none")[off_diagonal].mean()
    rank_student = (student / tau).masked_fill(~off_diagonal, DIAGONAL_LOGIT)  # (b, b)
    rank_teacher = (teacher / tau).masked_fill(~off_diagonal, DIAGONAL_LOGIT)  # (b, b)
    losses["pair_rank"] = soft_target_cross_entropy(rank_student, rank_teacher, dim=-1).mean()
    losses["pair_label"] = binary_cross_entropy_with_logits(student, same_origin(labels), reduction="none")[off_diagonal].mean()

    losses["total"] = sum(getattr(weights, name) * losses[name] for name in weights.names())

    return losses  # name to ()
