"""The Litmus student in PyTorch: a BitNet b1.58 encoder with block-local attention and a pair head.

Training is quantization-aware: every weight matrix is ternarized and every activation entering one is quantized to
8 bits in the forward pass, with straight-through gradients (the FAQ of BitNet b1.58, section 3, figure 3). In eval
mode the same modules compute SPEC.md's arithmetic: integer products of ternary weights and int8 activations, one
float scale per token, so `model.eval()` agrees with `reference.py` (tests/test_parity.py).

Texts are packed into one flat stream per micro-batch so that each text's attention blocks are whole blocks of the
stream: a text longer than W/2 starts at a multiple of W, and shorter texts share half-blocks of W/2, which neither
layer parity ever splits. Each layer then reshapes the stream into blocks (shifted by W/2 on odd layers) and attends
inside a block under a mask of same text and same text block; no padding token is ever attended.

    python -m litmus_student.model    # the size table of SPEC.md
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import NamedTuple
from torch import Tensor, nn

from .contract import reference


StudentConfig = reference.StudentConfig
SCALE_INIT = 10.0
SCALE_MIN = 1.0
SCALE_MAX = 100.0
STANDARDIZE_EPS = 1e-5
STANDARDIZE_MOMENTUM = 0.01
ADDS_PER_SECOND = 2e10  # one core; bitnet.cpp, arXiv 2410.16144, table 5 (SPEC.md, Sizes)
MULTIPLIES_PER_SECOND = 2e10


# ---------------------------------------------------------------- packed batches


@dataclass
class PackedBatch:
    """Texts packed into one stream of `t` positions, a multiple of the block size W."""

    tokens: Tensor  # (t,) int64; 0 at padding
    segment: Tensor  # (t,) int64; the text's index in the batch, -1 at padding
    position: Tensor  # (t,) int64; the position inside its text, 0 at padding
    texts: int  # n

    def to(self, device: torch.device, non_blocking: bool = False) -> PackedBatch:
        return PackedBatch(self.tokens.to(device, non_blocking=non_blocking), self.segment.to(device, non_blocking=non_blocking),
                           self.position.to(device, non_blocking=non_blocking), self.texts)

    @property
    def real_tokens(self) -> int:
        return int((self.segment >= 0).sum())


def single_text_batch(ids: np.ndarray, block_size: int) -> PackedBatch:
    """One text as a packed batch: (n,) ids to a stream padded to a multiple of W."""
    # ids: (n,)
    n = len(ids)
    t = max(block_size, -(-n // block_size) * block_size)
    tokens = torch.zeros(t, dtype=torch.long)  # (t,)
    tokens[:n] = torch.as_tensor(ids, dtype=torch.long)
    segment = torch.full((t,), -1, dtype=torch.long)  # (t,)
    segment[:n] = 0
    position = torch.zeros(t, dtype=torch.long)  # (t,)
    position[:n] = torch.arange(n)

    return PackedBatch(tokens, segment, position, 1)


# ---------------------------------------------------------------- quantization


def straight_through(latent: Tensor, quantized: Tensor, training: bool) -> Tensor:
    """The quantized value forward; in training, the gradient passes to `latent` unchanged."""
    # latent, quantized: (...) the same shape
    return latent + (quantized - latent).detach() if training else quantized  # (...)


class QuantizedActivations(NamedTuple):
    values: Tensor  # (..., k) dequantized, with a straight-through gradient in training
    integers: Tensor  # (..., k) the int8 values, as float32
    scale: Tensor  # (..., 1) s = 127 / absmax


def quantize_activations(X: Tensor, eps: float, training: bool, quantized: bool = True) -> QuantizedActivations:
    """Per-token absmax 8-bit quantization (SPEC.md, section 2), in float32: (..., k). With `quantized` false (the
    full-precision twin) the values pass unchanged."""
    # X: (..., k)
    if not quantized:
        return QuantizedActivations(X, X, torch.ones_like(X[..., :1]))
    X = X.float()  # (..., k)
    m = X.abs().amax(dim=-1, keepdim=True).clamp(min=eps)  # (..., 1)
    s = 127.0 / m  # (..., 1)
    X_q = (X * s).round().clamp(-128, 127)  # (..., k)

    return QuantizedActivations(straight_through(X, X_q / s, training), X_q, s)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))  # (k,)
        self.eps = eps

    def forward(self, X: Tensor) -> Tensor:
        # X: (..., k)
        X = X.float()  # (..., k)
        r = 1.0 / torch.sqrt(X.square().mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)

        return (X * r) * self.gain  # (..., k)


class TernaryLinear(nn.Module):
    """A weight matrix ternarized by its absolute mean on every forward (SPEC.md, section 2); no bias. With
    `quantized` false, a plain float matrix: the full-precision twin of the proof of concept."""

    def __init__(self, inputs: int, outputs: int, eps: float, init_scale: float = 1.0, quantized: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outputs, inputs) * (init_scale / math.sqrt(inputs)))  # (o, i)
        self.eps = eps
        self.quantized = quantized

    def ternary(self) -> tuple[Tensor, Tensor]:
        """The ternary values t (o, i) and the scale gamma (), both without gradient."""
        W = self.weight.detach().float()  # (o, i)
        gamma = W.abs().mean().clamp(min=self.eps)  # ()
        T = (W * (1.0 / gamma)).round().clamp(-1, 1)  # (o, i)

        return T, gamma  # (o, i), ()

    def forward(self, X: QuantizedActivations) -> Tensor:
        # X: (..., i)
        if not self.quantized:
            return F.linear(X.values, self.weight)  # (..., o)
        T, gamma = self.ternary()  # (o, i), ()
        if self.training:
            W = self.weight + (T * gamma - self.weight).detach()  # (o, i)
            return F.linear(X.values, W)  # (..., o)
        with torch.autocast(X.integers.device.type, enabled=False):
            acc = X.integers @ T.T  # (..., o), exact integer sums in float32

        return acc * (gamma / X.scale)  # (..., o)


class BitLinear(nn.Module):
    """RMS norm, 8-bit quantization, ternary product."""

    def __init__(self, inputs: int, outputs: int, config: StudentConfig, init_scale: float = 1.0, quantized: bool = True) -> None:
        super().__init__()
        self.norm = RMSNorm(inputs, config.norm_eps)
        self.linear = TernaryLinear(inputs, outputs, config.weight_eps, init_scale, quantized)
        self.act_eps = config.act_eps
        self.quantized = quantized

    def forward(self, X: Tensor) -> Tensor:
        # X: (..., i)
        return self.linear(quantize_activations(self.norm(X), self.act_eps, self.training, self.quantized))  # (..., o)


class ObservedBitLinear(BitLinear):
    """`BitLinear` (the same arithmetic, the same parameters) that also gives the L1 term of its int8 input codes and
    counts their zeros. `valid` (t,) marks the real tokens of the stream it reads."""

    def __init__(self, source: BitLinear) -> None:
        nn.Module.__init__(self)
        self.norm, self.linear, self.act_eps, self.quantized = source.norm, source.linear, source.act_eps, source.quantized
        self.valid: Tensor | None = None
        self.track_penalty = False
        self.penalty: Tensor | None = None  # () the sum over real tokens of mean_i |q_i| / 127
        self.recording = False
        self.zeros = self.counted = 0

    def forward(self, X: Tensor) -> Tensor:
        # X: (t, i)
        N = self.norm(X)  # (t, i)
        A = quantize_activations(N, self.act_eps, self.training, self.quantized)  # (t, i) each part
        if self.track_penalty and self.training and torch.is_grad_enabled():
            assert self.valid is not None, "the stream's real tokens are set before the forward"
            largest = N.abs().amax(dim=-1).clamp(min=self.act_eps)  # (t,)
            self.penalty = (A.values.abs().mean(dim=-1) / largest)[self.valid].sum()  # ()
        if self.recording:
            assert self.valid is not None, "the stream's real tokens are set before the forward"
            codes = A.integers[self.valid]  # (t_v, i)
            self.zeros += int((codes == 0).sum())
            self.counted += codes.numel()

        return self.linear(A)  # (t, o)


class QuantizedEmbedding(nn.Module):
    """Token rows stored int8 with one float32 scale per row (SPEC.md, section 3); float rows when not `quantized`."""

    def __init__(self, vocab_size: int, width: int, quantized: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, width))  # (V, e)
        self.quantized = quantized

    @staticmethod
    def quantize(rows: Tensor) -> tuple[Tensor, Tensor]:
        """Per-row absmax int8: (..., e) to integer values (..., e) and scales (..., 1)."""
        # rows: (..., e)
        scale = rows.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0  # (..., 1)
        return (rows / scale).round().clamp(-127, 127), scale  # (..., e), (..., 1)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (t,)
        rows = self.weight[tokens].float()  # (t, e)
        if not self.quantized:
            return rows  # (t, e)
        integers, scale = self.quantize(rows.detach())  # (t, e), (t, 1)

        return straight_through(rows, integers * scale, self.training)  # (t, e)


# ---------------------------------------------------------------- the encoder


def apply_rope(U: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate halves (SPEC.md, section 3): (t, h, d_h) with (t, 1, d_h/2) tables to (t, h, d_h)."""
    # U: (t, h, d_h); cosine, sine: (t, 1, d_h/2)
    first, second = U.chunk(2, dim=-1)  # (t, h, d_h/2) each

    return torch.cat([first * cosine - second * sine, second * cosine + first * sine], dim=-1)  # (t, h, d_h)


def block_mask(batch: PackedBatch, block_size: int, shift: int) -> Tensor:
    """Who may attend to whom inside each block of the stream shifted by `shift`: (b_n, 1, W, W) bool.

    Padding rows attend to themselves only, so softmax stays finite; their outputs are never read.
    """
    W = block_size
    segment = F.pad(batch.segment, (shift, W - shift if shift else 0), value=-1)  # (t',), t' = b_n * W
    text_block = F.pad(torch.div(batch.position + shift, W, rounding_mode="floor"), (shift, W - shift if shift else 0), value=-1)  # (t',)
    segment, text_block = segment.view(-1, W), text_block.view(-1, W)  # (b_n, W) each
    same = (segment[:, :, None] == segment[:, None, :]) & (text_block[:, :, None] == text_block[:, None, :])  # (b_n, W, W)
    same &= (segment >= 0)[:, None, :]
    same |= torch.eye(W, dtype=torch.bool, device=same.device)

    return same[:, None]  # (b_n, 1, W, W)


ATTENTION_WEIGHT_LEVELS = 127  # attention weights in 7 bits, so a pair of weight-by-int8 products fits 16 bits


def quantize_group(X: Tensor, dims: tuple[int, ...], eps: float, training: bool) -> Tensor:
    """8-bit absmax quantization with one scale over `dims`, dequantized, straight-through in training."""
    # X: (...)
    X = X.float()
    s = 127.0 / X.abs().amax(dim=dims, keepdim=True).clamp(min=eps)

    return straight_through(X, (X * s).round().clamp(-127, 127) / s, training)


def integer_block_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor, eps: float, training: bool,
                            value_max: Tensor | None = None) -> Tensor:
    """Attention inside blocks whose two products are integer products (SPEC.md, Integer attention, an optional mode
    of a release; eval mode is `integer_block_attention_eval`). Queries and keys are int8 with one scale per token
    and head, so a score is an integer dot product times two scales; the softmax stays float; attention weights are
    rounded to 7 bits and values are int8 with one scale per block and head, so the weighted sum of values is an
    integer dot product times one scale. `value_max` (b_n, h, W, 1), when given, is the largest value magnitude of
    each row's own text block (the rows it attends to), which then sets the value scale instead of the whole block."""
    # Q, K, V: (b_n, h, W, d_h); mask: (b_n, 1, W, W) bool
    if not training:
        return integer_block_attention_eval(Q, K, V, mask, eps, value_max)  # (b_n, h, W, d_h)
    with torch.autocast(Q.device.type, enabled=False):
        Q_q = quantize_group(Q, (-1,), eps, training)  # (b_n, h, W, d_h)
        K_q = quantize_group(K, (-1,), eps, training)  # (b_n, h, W, d_h)
        if value_max is None:
            V_q = quantize_group(V, (-2, -1), eps, training)  # (b_n, h, W, d_h)
        else:
            s_v = 127.0 / value_max.float().clamp(min=eps)  # (b_n, h, W, 1)
            V_q = straight_through(V.float(), (V.float() * s_v).round().clamp(-127, 127) / s_v, training)  # (b_n, h, W, d_h)
        scores = (Q_q @ K_q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])  # (b_n, h, W, W)
        P = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)  # (b_n, h, W, W)
        P_q = straight_through(P, (P * ATTENTION_WEIGHT_LEVELS).round() / ATTENTION_WEIGHT_LEVELS, training)  # (b_n, h, W, W)

        return P_q @ V_q  # (b_n, h, W, d_h)


def float32_constant(value: float, device: torch.device) -> Tensor:
    """A float32 constant as a 0-d tensor, so products with it are float32 products of float32 operands."""
    return torch.tensor(np.float32(value), dtype=torch.float32, device=device)  # ()


def exp_nonpositive(X: Tensor) -> Tensor:
    """`reference.exp_nonpositive` (SPEC.md, Integer attention): `exp(x)` for `x <= 0` in float32, one rounding per
    operation, so it gives the reference's and the engine's bits; -inf gives 2^-126 times the polynomial, so mask
    after it."""
    # X: (...) float32
    def constant(value: np.float32) -> Tensor:
        return float32_constant(value, X.device)

    magic = constant(reference.EXP_MAGIC)
    X = torch.maximum(X.float(), constant(np.float32(-87.0)))  # (...)
    shifted = X * constant(reference.EXP_LOG2E) + magic  # (...)
    n = shifted - magic  # (...)
    exponent = shifted.view(torch.int32) - magic.view(torch.int32)  # (...)
    r = (X - n * constant(reference.EXP_C1)) - n * constant(reference.EXP_C2)  # (...)
    p = torch.full_like(X, float(reference.EXP_POLYNOMIAL[0]))  # (...)
    for coefficient in reference.EXP_POLYNOMIAL[1:]:
        p = p * r + constant(coefficient)
    p = p * (r * r) + r + constant(np.float32(1.0))  # (...)
    scale = torch.bitwise_left_shift(exponent + 127, 23).view(torch.float32)  # (...)

    return p * scale  # (...)


def integer_block_attention_eval(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor, eps: float, value_max: Tensor | None = None) -> Tensor:
    """`integer_block_attention` in eval mode, as SPEC.md (Integer attention) and `reference.integer_attention` compute
    it: the scores and the weighted values are products of integers, exact in float32 (and under TF32, whose 10-bit
    mantissa holds every int8), each scaled after the sum; the weights are `round(E * float32(127 / sum E))` with the
    specified exp and a float64 sum, so they are the reference's weights. It equals the training forward's
    dequantized products up to float rounding."""
    # Q, K, V: (b_n, h, W, d_h); mask: (b_n, 1, W, W) bool; value_max: (b_n, h, W, 1) or None
    with torch.autocast(Q.device.type, enabled=False):
        def integers(X: Tensor, maximum: Tensor) -> tuple[Tensor, Tensor]:
            # X: (b_n, h, W, d_h); maximum: broadcast to X
            s = float32_constant(127.0, X.device) / maximum.float().clamp(min=eps)  # e.g. (b_n, h, W, 1)
            return (X.float() * s).round().clamp(-127, 127), s  # (b_n, h, W, d_h), e.g. (b_n, h, W, 1)

        Q_i, s_q = integers(Q, Q.float().abs().amax(dim=-1, keepdim=True))  # (b_n, h, W, d_h), (b_n, h, W, 1)
        K_i, s_k = integers(K, K.float().abs().amax(dim=-1, keepdim=True))  # (b_n, h, W, d_h), (b_n, h, W, 1)
        block_max = V.float().abs().amax(dim=(-2, -1), keepdim=True) if value_max is None else value_max  # (b_n, h, 1 or W, 1)
        V_i, s_v = integers(V, block_max)  # (b_n, h, W, d_h), (b_n, h, 1 or W, 1)
        x = float32_constant(1.0 / math.sqrt(Q.shape[-1]), Q.device) / s_q  # (b_n, h, W, 1)
        y = (float32_constant(1.0, Q.device) / s_k).transpose(-1, -2)  # (b_n, h, 1, W)
        scores = ((Q_i @ K_i.transpose(-1, -2)) * x) * y  # (b_n, h, W, W)
        top = scores.masked_fill(~mask, float("-inf")).amax(dim=-1, keepdim=True)  # (b_n, h, W, 1)
        E = exp_nonpositive((scores - top).masked_fill(~mask, 0.0)).masked_fill(~mask, 0.0)  # (b_n, h, W, W)
        r = (ATTENTION_WEIGHT_LEVELS / E.double().sum(dim=-1, keepdim=True)).float()  # (b_n, h, W, 1)
        P_i = (E * r).round()  # (b_n, h, W, W), integers in [0, 127]
        t = float32_constant(1.0, Q.device) / (float32_constant(ATTENTION_WEIGHT_LEVELS, Q.device) * s_v)  # (b_n, h, 1 or W, 1)

        return (P_i @ V_i) * t  # (b_n, h, W, d_h)


class Layer(nn.Module):
    def __init__(self, config: StudentConfig, index: int, quantized: bool = True, integer_attention: bool = False) -> None:
        super().__init__()
        d, f = config.width, config.ffn_width
        out_scale = 1.0 / math.sqrt(2 * config.depth)
        self.index, self.config, self.quantized, self.integer_attention = index, config, quantized, integer_attention
        self.attn_norm = RMSNorm(d, config.norm_eps)
        self.q = TernaryLinear(d, d, config.weight_eps, quantized=quantized)
        self.k = TernaryLinear(d, d, config.weight_eps, quantized=quantized)
        self.v = TernaryLinear(d, d, config.weight_eps, quantized=quantized)
        self.o = BitLinear(d, d, config, out_scale, quantized)
        self.up = BitLinear(d, f, config, quantized=quantized)
        self.down = BitLinear(f, d, config, out_scale, quantized)

    def forward(self, X: Tensor, batch: PackedBatch, mask: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        # X: (t, d) float32; mask: (b_n, 1, W, W); cosine, sine: (W, d_h/2)
        config = self.config
        t, h, d_h, W = X.shape[0], config.heads, config.head_width, config.block_size
        shift = config.shift(self.index)
        A = quantize_activations(self.attn_norm(X), config.act_eps, self.training, self.quantized)  # (t, d) each part
        r = torch.remainder(batch.position + shift, W)  # (t,)
        cos_r, sin_r = cosine[r][:, None, :], sine[r][:, None, :]  # (t, 1, d_h/2) each
        Q = apply_rope(self.q(A).view(t, h, d_h), cos_r, sin_r)  # (t, h, d_h)
        K = apply_rope(self.k(A).view(t, h, d_h), cos_r, sin_r)  # (t, h, d_h)
        V = self.v(A).view(t, h, d_h)  # (t, h, d_h)

        padding = (shift, W - shift) if shift else (0, 0)
        def blocks(U: Tensor) -> Tensor:
            # U: (t, h, d_h)
            U = F.pad(U.transpose(0, 2), padding).transpose(0, 2)  # (t', h, d_h)
            return U.reshape(-1, W, h, d_h).transpose(1, 2)  # (b_n, h, W, d_h)

        Q_b, K_b, V_b = blocks(Q), blocks(K), blocks(V)  # (b_n, h, W, d_h) each
        if self.integer_attention:
            # One value scale per block of each text (SPEC.md, Integer attention), not per block of the packed stream,
            # which may hold several texts and padding: each row's scale is the largest value magnitude of the rows it
            # attends to, which are exactly its text block (the mask is same text and same text block).
            visible = torch.where(mask, V_b.float().abs().amax(dim=-1)[:, :, None, :], 0.0)  # (b_n, h, W, W)
            value_max = visible.amax(dim=-1, keepdim=True).detach()  # (b_n, h, W, 1)
            O = integer_block_attention(Q_b, K_b, V_b, mask, config.act_eps, self.training, value_max)  # (b_n, h, W, d_h)
        else:
            dtype = torch.get_autocast_dtype(X.device.type) if torch.is_autocast_enabled(X.device.type) else torch.float32
            O = F.scaled_dot_product_attention(Q_b.to(dtype), K_b.to(dtype), V_b.to(dtype), attn_mask=mask)  # (b_n, h, W, d_h)
        O = O.transpose(1, 2).reshape(-1, config.width)[shift : shift + t]  # (t, d)
        X = X + self.o(O).float()  # (t, d)
        U = self.up(X)  # (t, f)
        X = X + self.down(F.relu(U).float().square()).float()  # (t, d)

        return X  # (t, d)


class StudentEncoder(nn.Module):
    """Token ids to token states (SPEC.md, section 3)."""

    def __init__(self, config: StudentConfig, quantized: bool = True, integer_attention: bool = False) -> None:
        super().__init__()
        self.config = config
        self.embed = QuantizedEmbedding(config.vocab_size, config.embed_width, quantized)
        self.proj = BitLinear(config.embed_width, config.width, config, quantized=quantized) if config.projected else None
        self.layers = nn.ModuleList(Layer(config, index, quantized, integer_attention) for index in range(config.depth))
        self.final_norm = RMSNorm(config.width, config.norm_eps)
        cosine, sine = reference.rope_tables(config.block_size, config.head_width, config.rope_base)  # (W, d_h/2) each
        self.register_buffer("cosine", torch.from_numpy(cosine), persistent=False)
        self.register_buffer("sine", torch.from_numpy(sine), persistent=False)

    def forward(self, batch: PackedBatch) -> Tensor:
        # batch: t positions
        config = self.config
        assert batch.tokens.shape[0] % config.block_size == 0, "a packed stream is a whole number of blocks"
        X = self.embed(batch.tokens)  # (t, e)
        if self.proj is not None:
            X = self.proj(X).float()  # (t, d)
        masks = {shift: block_mask(batch, config.block_size, shift) for shift in {config.shift(0), config.shift(1)}}
        for layer in self.layers:
            X = layer(X, batch, masks[config.shift(layer.index)], self.cosine, self.sine)  # (t, d)

        return self.final_norm(X)  # (t, d)


def pool(H: Tensor, segment: Tensor, texts: int) -> Tensor:
    """The mean and population variance of each text's token states, side by side: (t, d) to (n, 2d) float32."""
    # H: (t, d); segment: (t,)
    valid = segment >= 0  # (t,)
    rows, states = segment[valid], H[valid].float()  # (t_v,), (t_v, d)
    counts = torch.bincount(rows, minlength=texts).clamp(min=1).float()[:, None]  # (n, 1)
    mean = torch.zeros(texts, H.shape[1], device=H.device).index_add_(0, rows, states) / counts  # (n, d)
    centered = states - mean[rows]  # (t_v, d)
    variance = torch.zeros_like(mean).index_add_(0, rows, centered.square()) / counts  # (n, d)

    return torch.cat([mean, variance], dim=-1)  # (n, 2d)


# ---------------------------------------------------------------- the head


class HalfLinear(nn.Module):
    """A float32 linear layer whose weights are rounded to float16 on every forward, as the release stores them."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outputs, inputs) / math.sqrt(inputs))  # (o, i)
        self.bias = nn.Parameter(torch.zeros(outputs))  # (o,)

    def forward(self, X: Tensor) -> Tensor:
        # X: (..., i) float32
        W = straight_through(self.weight, self.weight.half().float(), self.training)  # (o, i)
        b = straight_through(self.bias, self.bias.half().float(), self.training)  # (o,)

        return F.linear(X, W, b)  # (..., o)


class StudentHead(nn.Module):
    """Pooled statistics (n, 2d) to a unit query and a unit key (n, 2w), with the pair logit's scale and bias.

    Each input column is standardized, by the batch in training and by running statistics (momentum 0.01, seeded by
    the first batch) in eval, as `head.input_mean` and `head.input_scale` ship, and clamped at `config.head_clamp` deviations.

    With `int8_output` the query and the key are each quantized to int8 with their own absmax scale before they are
    normalized, straight-through in training, so the model learns the vectors a release stores and compares as int8:
    a unit vector of the quantized values is the unit vector of the integers.
    """

    def __init__(self, config: StudentConfig, int8_output: bool = False) -> None:
        super().__init__()
        self.int8_output = int8_output
        self.act_eps = config.act_eps
        width = 2 * config.width
        self.register_buffer("running_mean", torch.zeros(width))  # (2d,)
        self.register_buffer("running_var", torch.ones(width))  # (2d,)
        self.register_buffer("seeded", torch.zeros((), dtype=torch.bool))  # ()
        self.fc1 = HalfLinear(width, config.head_hidden)
        self.fc2 = HalfLinear(config.head_hidden, 2 * config.head_out)
        self.log_scale = nn.Parameter(torch.tensor(math.log(SCALE_INIT)))  # ()
        self.bias = nn.Parameter(torch.zeros(()))  # ()
        self.out_width = config.head_out
        self.clamp = config.head_clamp

    def input_statistics(self) -> tuple[Tensor, Tensor]:
        """The shipped standardization: mean (2d,) and scale (2d,)."""
        return self.running_mean, torch.sqrt(self.running_var + STANDARDIZE_EPS)  # (2d,), (2d,)

    def standardize(self, pooled: Tensor) -> Tensor:
        # pooled: (n, 2d)
        if not self.training:
            mean, scale = self.input_statistics()  # (2d,) each
            return ((pooled - mean) / scale).clamp(-self.clamp, self.clamp)  # (n, 2d)
        mean, variance = pooled.mean(dim=0), pooled.var(dim=0, unbiased=False)  # (2d,) each
        with torch.no_grad():
            momentum = torch.where(self.seeded, STANDARDIZE_MOMENTUM, 1.0)  # ()
            self.running_mean.lerp_(mean.detach(), momentum)
            self.running_var.lerp_(variance.detach(), momentum)
            self.seeded.fill_(True)

        return ((pooled - mean) / torch.sqrt(variance + STANDARDIZE_EPS)).clamp(-self.clamp, self.clamp)  # (n, 2d)

    @property
    def scale(self) -> Tensor:
        return self.log_scale.exp().clamp(SCALE_MIN, SCALE_MAX)  # ()

    def project(self, Z: Tensor) -> Tensor:
        """Standardized statistics to unit queries and keys: (n, 2d) to (n, 2w), float32."""
        # Z: (n, 2d)
        with torch.autocast(Z.device.type, enabled=False):
            O = self.fc2(F.silu(self.fc1(Z.float())))  # (n, 2w)
            query, key = O.chunk(2, dim=-1)  # (n, w) each
            if self.int8_output:
                query = quantize_group(query, (-1,), self.act_eps, self.training)  # (n, w), 127ths of each row's largest
                key = quantize_group(key, (-1,), self.act_eps, self.training)  # (n, w)

            return torch.cat([F.normalize(query, dim=-1, eps=1e-8), F.normalize(key, dim=-1, eps=1e-8)], dim=-1)  # (n, 2w)

    def forward(self, pooled: Tensor) -> Tensor:
        # pooled: (n, 2d)
        return self.project(self.standardize(pooled.float()))  # (n, 2w)


class StudentModel(nn.Module):
    """The encoder and the head. `quantized` false is the full-precision twin (float weights and activations, the
    same architecture), a control for the proof of concept that no release is made from. `integer_attention` reads
    attention with `integer_block_attention`, the optional integer mode of SPEC.md (a format 2 release).
    `int8_output` quantizes the head's query and key to int8 (`StudentHead`)."""

    def __init__(self, config: StudentConfig, quantized: bool = True, integer_attention: bool = False, int8_output: bool = False) -> None:
        super().__init__()
        self.config = config
        self.quantized = quantized
        self.integer_attention = integer_attention
        self.encoder = StudentEncoder(config, quantized, integer_attention)
        self.head = StudentHead(config, int8_output)

    def pooled(self, batch: PackedBatch) -> Tensor:
        """The texts' pooled statistics: (n, 2d)."""
        return pool(self.encoder(batch), batch.segment, batch.texts)  # (n, 2d)

    def forward(self, batch: PackedBatch) -> Tensor:
        """The texts' unit queries and keys: (n, 2w)."""
        return self.head(self.pooled(batch))  # (n, 2w)


# ---------------------------------------------------------------- sizes


def estimated_cost(config: StudentConfig, bank_per_origin: int = 4096) -> dict[str, float]:
    """Adds and multiplies per token, file sizes and single-core speed at the assumed rates (SPEC.md, Sizes)."""
    d, f, e, L, W = config.width, config.ffn_width, config.embed_width, config.depth, config.block_size
    layer_weights = L * (4 * d * d + 2 * d * f)
    projection = e * d if config.projected else 0
    adds = layer_weights + projection
    multiplies = L * 2 * W * d  # scores and weighted values at full blocks
    norms = 4 * (L * (3 * d + f) + d + (e if config.projected else 0))
    head = 2 * (2 * d * config.head_hidden + config.head_hidden + config.head_hidden * 2 * config.head_out + 2 * config.head_out) + 4 * 2 * 2 * d + 8
    model_bytes = config.vocab_size * (e + 4) + adds / 4 + 4 * (4 * L + bool(projection)) + norms + head
    bank_bytes = 3 * bank_per_origin * (2 * 2 * config.head_out + 1)
    seconds_per_token = adds / ADDS_PER_SECOND + multiplies / MULTIPLIES_PER_SECOND

    return {"ternary_weights": layer_weights + projection, "adds_per_token": adds, "multiplies_per_token": multiplies,
            "model_bytes": model_bytes, "bank_bytes": bank_bytes, "tokens_per_second": 1.0 / seconds_per_token,
            "seconds_120k": 120_000 * seconds_per_token}


def main() -> None:
    print("| Config | d | L | Ternary weights | Adds per token | Multiplies per token | Model MB | Tokens/s, one core | 120,000 tokens |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in reference.config_names():
        if name == "tiny":
            continue
        config = reference.named_config(name)
        cost = estimated_cost(config)
        print(f"| `{name}` | {config.width} | {config.depth} | {cost['ternary_weights'] / 1e6:.2f}M | {cost['adds_per_token'] / 1e6:.2f}M | "
              f"{cost['multiplies_per_token'] / 1e6:.2f}M | {cost['model_bytes'] / 1e6:.1f} | {cost['tokens_per_second']:,.0f} | {cost['seconds_120k']:.0f} s |")


if __name__ == "__main__":
    main()
