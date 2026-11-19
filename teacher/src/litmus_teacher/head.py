"""The pair head: two texts of the same origin score high together, texts of different origins low.

The first checker's head (lhallee/ai-vs-human-checker, `fphunt/pairhead.py`), over three origins. Each text's
embedding is standardized with training statistics and clamped at 10 standard deviations by default (`search` tries
others), and a multilayer perceptron maps it to a unit query and a unit key of `w` values each. A pair is scored

    s(a, b) = scale * (q_a . k_b + q_b . k_a) / 2 + bias

which is symmetric, one inner product of stored vectors, and lets a text score itself low. The scale is learned
within [1, 100] from 10 by default, or held fixed; the bias is learned. The logit reads "a and b
share an origin": human with human, ai with ai, mixed with mixed. Training compares every text of one batch with
every text of another and takes binary cross-entropy on the logits (`foundry.layers.cross_entropy`'s, which keeps
a confident pair's gradient).
"""

from __future__ import annotations

import json
import math
import torch
import torch.nn.functional as F

from pathlib import Path
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


WEIGHTS_FILE = "pairhead.safetensors"
CONFIG_FILE = "pairhead.json"
SCALE_INIT = 10.0
SCALE_MIN = 1.0
SCALE_MAX = 100.0
CLAMP_DEVIATIONS = 10.0


class PairHead(nn.Module):
    """Embeddings (b, f) to a unit query and a unit key side by side (b, 2w)."""

    def __init__(self, input_width: int, widths: tuple[int, ...], output_width: int, dropout: float,
                 clamp_deviations: float = CLAMP_DEVIATIONS, scale_init: float = SCALE_INIT, learn_scale: bool = True,
                 bias_init: float = 0.0) -> None:
        super().__init__()
        if output_width % 2:
            raise ValueError("the output is a query half and a key half, so its width is even")
        if not SCALE_MIN <= scale_init <= SCALE_MAX:
            raise ValueError(f"the scale starts within [{SCALE_MIN}, {SCALE_MAX}]")
        # The variance columns are orders of magnitude smaller than the mean columns, so each is standardized.
        self.register_buffer("input_mean", torch.zeros(input_width))  # (f,)
        self.register_buffer("input_scale", torch.ones(input_width))  # (f,)
        layers: list[nn.Module] = []
        width = input_width
        for hidden_width in widths:
            layers += [nn.Linear(width, hidden_width), nn.LayerNorm(hidden_width), nn.GELU(), nn.Dropout(dropout)]
            width = hidden_width
        layers.append(nn.Linear(width, output_width))
        self.layers = nn.Sequential(*layers)
        self.clamp_deviations = clamp_deviations
        log_scale = torch.tensor(math.log(scale_init))  # ()
        if learn_scale:
            self.log_scale = nn.Parameter(log_scale)
        else:  # a fixed temperature: saved with the weights, never trained
            self.register_buffer("log_scale", log_scale)
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))  # ()
        self.config = {"input_width": input_width, "widths": list(widths), "output_width": output_width,
                       "dropout": dropout, "clamp_deviations": clamp_deviations, "scale_init": scale_init,
                       "learn_scale": learn_scale, "bias_init": bias_init}

    def set_input_statistics(self, mean: Tensor, scale: Tensor) -> None:
        # mean, scale: (f,)
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale.clamp(min=1e-6))

    @property
    def scale(self) -> Tensor:
        return self.log_scale.float().exp().clamp(SCALE_MIN, SCALE_MAX)  # ()

    def forward(self, features: Tensor) -> Tensor:
        # features: (b, f)
        standardized = ((features.float() - self.input_mean) / self.input_scale).clamp(-self.clamp_deviations,
                                                                                      self.clamp_deviations)  # (b, f)
        query, key = self.layers(standardized).float().chunk(2, dim=-1)  # (b, w) each

        return torch.cat([F.normalize(query, dim=-1, eps=1e-8), F.normalize(key, dim=-1, eps=1e-8)], dim=-1)  # (b, 2w)

    def pair_logits(self, rows: Tensor, columns: Tensor) -> Tensor:
        """The logit that each row text and each column text share an origin: (r, 2w), (c, 2w) to (r, c)."""
        # rows: (r, 2w); columns: (c, 2w)
        return pair_logits(rows, columns, self.scale, self.bias)  # (r, c)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: value.contiguous() for name, value in self.state_dict().items()}, str(directory / WEIGHTS_FILE))
        config = {**self.config, "scale": float(self.scale.detach()), "bias": float(self.bias.detach())}
        (directory / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> PairHead:
        config = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
        model = cls(config["input_width"], tuple(config["widths"]), config["output_width"], config["dropout"],
                    config["clamp_deviations"], config["scale_init"], config["learn_scale"], config["bias_init"])
        model.load_state_dict(load_file(str(directory / WEIGHTS_FILE)))

        return model


def pair_logits(rows: Tensor, columns: Tensor, scale: Tensor | float, bias: Tensor | float) -> Tensor:
    """`scale * (q_r . k_c + q_c . k_r) / 2 + bias` for every row and column: (r, 2w), (c, 2w) to (r, c)."""
    # rows: (r, 2w); columns: (c, 2w); scale, bias: ()
    query_r, key_r = rows.float().chunk(2, dim=-1)  # (r, w) each
    query_c, key_c = columns.float().chunk(2, dim=-1)  # (c, w) each
    dot = 0.5 * (query_r @ key_c.T + key_r @ query_c.T)  # (r, c)

    return scale * dot + bias  # (r, c)


def same_origin_targets(row_labels: Tensor, column_labels: Tensor) -> Tensor:
    """1 where the two texts share an origin, of three: (r,), (c,) to (r, c) float."""
    # row_labels: (r,); column_labels: (c,)
    return (row_labels[:, None] == column_labels[None, :]).float()  # (r, c)


def same_origin_loss(logits: Tensor, row_labels: Tensor, column_labels: Tensor, smoothing: float = 0.0) -> Tensor:
    """Mean binary cross-entropy of pair logits against whether the two labels are equal: (r, c) to ().

    `smoothing` moves each target `smoothing / 2` toward one half; validation always reads hard targets.
    """
    # logits: (r, c); row_labels: (r,); column_labels: (c,)
    from foundry.layers.cross_entropy import binary_cross_entropy_with_logits

    targets = same_origin_targets(row_labels, column_labels)  # (r, c)

    return binary_cross_entropy_with_logits(logits, targets * (1.0 - smoothing) + 0.5 * smoothing)  # ()
