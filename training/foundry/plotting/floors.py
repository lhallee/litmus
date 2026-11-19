"""Floor bars: one panel per record, each baseline a horizontal bar colored by its role with its value or interval written
beside it, and the grid's last cell a legend. The pair floors (figures/atlas/2026-11-01_pair_floors) and the release floors
(figures/production_suite/2026-11-01_release_floors) draw with it from their data.json, whose panels look like

    {"letter": "A", "axis": "pooled ROC AUC\\nflock_ppi leakage_free", "higher_is_better": true,
     "bars": [{"name": "prior", "role": "control", "value": 0.5},
              {"name": "Pfam family-pair lookup", "role": "domain", "value": 0.681, "low": 0.658, "high": 0.704}]}

A bar with `low` and `high` is drawn to `low` with the interval as a line to `high`; one with `value` alone is drawn to it.
The roles' colors and legend labels, and their order (a panel's bars are sorted by it, top to bottom), come from the caller,
so a figure set adds a role without a change here. Each panel draws its own letter above its tick labels, since the long
baseline names on the left leave no room where `panels.letter_panels` puts one.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


INK = "#222222"
SURFACE = "#FCFCFB"

Panel = dict[str, object]


def values(panel: Panel) -> list[float]:
    """Every plotted number of the panel: each bar's value and interval ends, where present."""
    return [v for b in panel["bars"] for v in (b.get("value"), b.get("low"), b.get("high")) if v is not None]


def x_limits(panel: Panel) -> tuple[float, float]:
    """Room right of the longest bar for its label: from zero (or just below a negative value) when higher is better, else
    from a margin below the lowest value."""
    top = max(values(panel))
    if panel["higher_is_better"]:
        low = min(0.0, min(values(panel)))
        return (low - (0.015 if low < 0 else 0.0), top * 1.36 + 0.02)
    low = min(values(panel))
    return (low - 0.25 * (top - low) - 0.02, top + 0.45 * (top - low) + 0.02)


def x_span(panel: Panel) -> float:
    left, right = x_limits(panel)
    return right - left


def draw_panel(ax: Axes, panel: Panel, colors: dict[str, str]) -> None:
    """One record's floors on `ax`: its bars in the order of `colors` (the roles' order), top to bottom, each labelled with
    its value or interval; the panel's letter outside the plot area."""
    roles = list(colors)
    bars = sorted(panel["bars"], key=lambda b: roles.index(b["role"]))
    y = list(range(len(bars)))[::-1]  # (n) top to bottom in role order
    for position, bar in zip(y, bars, strict=True):
        low = bar.get("low", bar.get("value"))
        high = bar.get("high", bar.get("value"))
        ax.barh(position, low, height=0.62, color=colors[bar["role"]], edgecolor=SURFACE, linewidth=1.0)
        if "low" in bar:
            ax.plot([low, high], [position, position], color=INK, linewidth=1.6, solid_capstyle="butt")
            ax.plot([high, high], [position - 0.2, position + 0.2], color=INK, linewidth=1.2)
            label = f"{low:.3f} to {high:.3f}"
        else:
            label = f"{low:.3f}"
        ax.text(high + 0.012 * x_span(panel), position, label, va="center", ha="left", fontsize=7.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([b["name"] for b in bars], fontsize=7.2, color=INK)
    ax.set_xlabel(panel["axis"], fontsize=7.6, color=INK)
    ax.set_xlim(x_limits(panel))
    ax.tick_params(axis="x", labelsize=7, colors=INK, length=2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BBBBBB")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    ax.text(-0.02, 1.04, panel["letter"], transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, ha="right", va="bottom")


def floors_figure(panels: list[Panel], colors: dict[str, str], labels: dict[str, str], grid: tuple[int, int],
                  figsize: tuple[float, float], range_label: str, adjust: dict[str, float]) -> Figure:
    """The panels on a `grid` of axes in reading order, with the one cell left the legend: a patch per role in the order of
    `colors`, labelled from `labels`, and the interval line labelled `range_label`. `adjust` goes to `subplots_adjust`. The
    caller saves the figure, at 300 dpi on `SURFACE`."""
    rows, columns = grid
    assert len(panels) + 1 == rows * columns, (len(panels), grid)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, facecolor=SURFACE)
    for ax, panel in zip(axes.flat, panels, strict=False):
        draw_panel(ax, panel, colors)
    legend_ax = axes.flat[len(panels)]
    legend_ax.axis("off")
    handles = [Patch(facecolor=colors[role], edgecolor="none", label=labels[role]) for role in colors]
    handles.append(Line2D([0], [0], color=INK, linewidth=1.6, label=range_label))
    legend_ax.legend(handles=handles, loc="center left", fontsize=7.8, frameon=False, labelcolor=INK,
                     title="bar color is the baseline's role", title_fontsize=7.8, alignment="left")
    fig.subplots_adjust(**adjust)
    return fig
