"""Panel letters for multi-panel figures, as the figure convention asks (docs/conventions/figures.md)."""

from __future__ import annotations

import numpy as np

from collections.abc import Iterable
from matplotlib.axes import Axes


LETTER_SIZE = 14


def letter_panels(axes: Iterable[Axes] | np.ndarray) -> None:
    """Letter the panels A, B, C, ... in the order given (reading order), in bold at each panel's top left, outside
    the plot area; a figure of one panel gets no letter."""
    # axes: (rows, columns) or (n,), any order of panels
    panels = list(axes.flat) if isinstance(axes, np.ndarray) else list(axes)
    if len(panels) == 1:
        return
    for index, axis in enumerate(panels):
        axis.annotate(chr(ord("A") + index), xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-38, 4),
                      textcoords="offset points", fontsize=LETTER_SIZE, fontweight="bold", va="bottom", ha="left",
                      annotation_clip=False)
