"""Windows over a long text, and the exact mean and variance of its token states from per-window totals.

The rule for which token states a text's statistics count, for both encoders:

- A text is tokenized without special tokens into `c` content tokens. A text of at most `w = context - 2` content
  tokens is one piece, `[start] content [end]`, and all `c + 2` states count, special tokens included, as the
  first corpus pooled ("special tokens included").
- A longer text is cut into windows of exactly `w` content tokens, each wrapped in its own start and end tokens, so
  every window is a sequence the encoder saw in training. Window `j` starts at `j * (w - overlap)`; the last window
  is moved back to end at the text's end, so no window is short. Consecutive windows overlap by at least `overlap`
  tokens, and a token in an overlap counts once, in the window where it sits farther from the edge: the boundary
  between windows `j` and `j + 1` is the middle of their overlap. Only the first window's start token and the last
  window's end token count. A text of `c` content tokens therefore always counts `c + 2` states, as an unwindowed
  one does, and each counted content state has at least `overlap // 2` tokens of context on both sides within its
  window, except at the text's own ends.

Each window gives a count, a sum and a sum of squares over its counted states, accumulated in float64; the text's
mean is `sum / count` and its variance `squares / count - mean ** 2`, the population variance over the counted states.
In float64 the subtraction loses nothing that matters for states of order 1 to 100 (relative error near 1e-12).
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """One piece of a text: content tokens `[start, stop)`, of which `[count_start, count_stop)` count.

    `first` and `last` say whether the piece's own start and end special tokens count.
    """

    start: int
    stop: int
    count_start: int
    count_stop: int
    first: bool
    last: bool

    @property
    def counted(self) -> int:
        """States this piece adds to its text's totals, special tokens included."""
        return self.count_stop - self.count_start + int(self.first) + int(self.last)


def window_spans(content_tokens: int, context: int, overlap: int) -> list[Window]:
    """The windows of a text of `content_tokens` tokens for an encoder of `context` positions, special tokens included."""
    width = context - 2  # w: content tokens per window
    if width <= 0 or not 0 <= overlap < width:
        raise ValueError(f"need context > 2 and 0 <= overlap < context - 2; got context {context}, overlap {overlap}")
    if content_tokens < 0:
        raise ValueError("a text has a non-negative number of tokens")
    if content_tokens <= width:
        return [Window(0, content_tokens, 0, content_tokens, True, True)]

    stride = width - overlap
    starts = list(range(0, content_tokens - width, stride))
    starts.append(content_tokens - width)  # the range stops short of it, so it is never a repeat
    windows = []
    for index, start in enumerate(starts):
        stop = start + width
        count_start = 0 if index == 0 else (starts[index - 1] + width + start) // 2
        count_stop = content_tokens if index == len(starts) - 1 else (stop + starts[index + 1]) // 2
        windows.append(Window(start, stop, count_start, count_stop, index == 0, index == len(starts) - 1))
    assert sum(window.counted for window in windows) == content_tokens + 2, "every state counts exactly once"

    return windows


def window_count_mask(window: Window) -> np.ndarray:
    """Which positions of the piece `[start] content[start:stop] [end]` count: (stop - start + 2,) bool."""
    mask = np.zeros(window.stop - window.start + 2, dtype=bool)  # (l,), l = piece length with special tokens
    mask[0] = window.first
    mask[1 + window.count_start - window.start : 1 + window.count_stop - window.start] = True
    mask[-1] = window.last

    return mask  # (l,)


def combine_moments(count: np.ndarray, total: np.ndarray, squares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population variance from summed counts, sums and sums of squares: (n,), (n, d), (n, d) to (n, d) twice."""
    # count: (n,); total, squares: (n, d)
    if (count <= 0).any():
        raise ValueError("every text needs at least one counted state")
    count = count.astype(np.float64)[:, None]  # (n, 1)
    mean = total / count  # (n, d)
    variance = np.maximum(squares / count - mean**2, 0.0)  # (n, d); clamps a rounding residue below zero

    return mean, variance  # (n, d), (n, d)
