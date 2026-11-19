"""The comparison figures, drawn from the tidy metric table, `published.csv`, per-text scores and measured CPU costs.

Every figure follows one convention: PNG at 300 dpi, no title in the image, panels lettered A, B, C, short axis
labels and minimal legends, in the site's dark and light looks on a transparent ground. Litmus is the one accent;
every other detector is a neutral gray with its own marker, the same in every figure. A number we computed is filled
and carries its bootstrap 95% interval as a line; a number quoted from a leaderboard is hollow; a detector's number
on a benchmark it trained on is faded; an estimated throughput is lighter still. Origins use the litmus scale.
"""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")

import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from foundry.plotting.panels import letter_panels
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from pathlib import Path

from .evaluate import LENGTH_LABELS, tpr_at_fpr
from .palette import (FONT_SIZE, LINE_WIDTH, METRIC_SCALE, ORIGIN_HATCHES, ORIGIN_LABELS, ORIGIN_ORDER, PREDECESSOR,
                      RERUN_MARKERS, THEMES, detector_role, lightness, short_name)
from .results import benchmark_name, in_distribution


DPI = 300
THEME_NAMES = ("dark", "light")
FIGURES = ("headline", "raid", "pareto", "strata", "origins")
METRIC_LABELS = {"roc_auc": "ROC AUC", "pr_auc": "PR AUC", "balanced_mcc": "Balanced MCC",
                 "tpr_at_fpr_1": "TPR at 1% FPR", "tpr_at_fpr_5": "TPR at 5% FPR"}
KIND_MARKERS = {"commercial": "^", "open": "o"}
REPORTER_MARKERS = {"self-reported": "^", "self-submitted": "v", "competitor": "D", "benchmark": "o", "third party": "s"}
FADED = 0.35
OWN_SPLITS = ("validation", "test")
ORIGIN_ORDER_BY_LABEL = ("human", "ai", "mixed")  # evaluation.json's labels 0, 1, 2


def style(theme: str = "light") -> dict[str, str]:
    """Set matplotlib to the theme's ink and rules on a transparent ground; returns the theme's colors."""
    colors = THEMES[theme]
    plt.rcParams.update({
        "font.size": FONT_SIZE, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "savefig.dpi": DPI,
        "text.color": colors["ink"], "axes.labelcolor": colors["ink"], "axes.edgecolor": colors["soft"],
        "xtick.color": colors["soft"], "ytick.color": colors["soft"], "xtick.labelcolor": colors["ink"],
        "ytick.labelcolor": colors["ink"], "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.transparent": True, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round", "hatch.linewidth": 0.6,
    })
    return colors


def save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", format="png", transparent=True)
    plt.close(figure)
    return path


def detector_look(detector: str, colors: Mapping[str, str]) -> tuple[str, str, float]:
    """Color, marker and size of a rerun detector: Litmus the accent, the rest grays with their own markers."""
    role = detector_role(detector, "open")
    return colors[role], RERUN_MARKERS.get(detector, "o"), 9.0 if role == "litmus" else 5.0


def detector_order(detectors: Sequence[str]) -> list[str]:
    """Other detectors first (alphabetical), the predecessor, then Litmus last: the control first, the result last."""
    rank = {PREDECESSOR: 1, "Litmus": 2}
    return sorted(set(detectors), key=lambda name: (rank.get(name, 0), name.lower()))


def draw_point(axis: Axes, x: float, y: float, detector: str, colors: Mapping[str, str], *, faded: bool = False,
               low: float | None = None, high: float | None = None, horizontal: bool = True) -> None:
    """One rerun detector's value, with its interval when given; faded where it trained on the benchmark."""
    color, marker, size = detector_look(detector, colors)
    alpha = FADED if faded else 1.0
    if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
        xs, ys = ([low, high], [y, y]) if horizontal else ([x, x], [low, high])
        axis.plot(xs, ys, color=color, alpha=alpha, linewidth=LINE_WIDTH, zorder=2)
    axis.plot([x], [y], marker=marker, markersize=size, linestyle="none", color=color, alpha=alpha,
              markeredgewidth=0.5, zorder=4 if detector == "Litmus" else 3)


def detector_handles(detectors: Sequence[str], colors: Mapping[str, str], *, faded: bool = False,
                     quoted: bool = False) -> list[Line2D]:
    handles = []
    for detector in detectors:
        color, marker, size = detector_look(detector, colors)
        handles.append(Line2D([], [], marker=marker, color=color, linestyle="none", markersize=min(size, 7.5),
                              label=short_name(detector)))
    if quoted:
        handles.append(Line2D([], [], marker="o", color=colors["open"], markerfacecolor="none", linestyle="none",
                              markersize=5, label="Leaderboard"))
    if faded:
        handles.append(Line2D([], [], marker="o", color=colors["open"], alpha=FADED, linestyle="none", markersize=5,
                              label="Trained on it"))
    return handles


def bottom_legend(figure: plt.Figure, handles: list[Line2D], *, columns: int | None = None) -> None:
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=columns or min(len(handles), 5),
                  handletextpad=0.2, columnspacing=1.0, fontsize=FONT_SIZE - 0.5)


LABEL_SIDES = {"right": ((5, 0), "left", "center"), "left": ((-5, 0), "right", "center"),
               "below": ((0, -7), "center", "top"), "above": ((0, 7), "center", "bottom"),
               "below left": ((-4, -6), "right", "top"), "below right": ((4, -6), "left", "top")}


def place_labels(axis: Axes, points: Sequence[tuple[float, float, str, str, str]], *, fontsize: float = FONT_SIZE - 1.5) -> None:
    """Label points on their chosen side (`LABEL_SIDES`; a right label near the right edge goes left), nudging
    labels apart vertically so none overlap, with a thin leader where a label moved away from its point."""
    figure = axis.figure
    texts: list[Annotation] = []
    for x, y, text, color, side in points:
        offset, horizontal, vertical = LABEL_SIDES[side]
        texts.append(axis.annotate(text, (x, y), xytext=offset, textcoords="offset points", va=vertical, ha=horizontal,
                                   fontsize=fontsize, color=color, zorder=5))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer)
    to_points = 72.0 / figure.dpi
    for text in texts:
        if text.get_ha() == "left" and text.get_window_extent(renderer).x1 > frame.x1 + 4:
            text.set_ha("right")
            text.set_position((-5, 0))
    placed = []
    for index in np.argsort([text.get_window_extent(renderer).y0 for text in texts]):
        text = texts[index]
        box = text.get_window_extent(renderer)
        shift = 0.0
        for other in placed:
            moved = box.translated(0, shift)
            if moved.x1 > other.x0 and moved.x0 < other.x1 and moved.y1 > other.y0 and moved.y0 < other.y1:
                shift += other.y1 - moved.y0 + 1
        if shift:
            dx, dy = text.get_position()
            moved_text = axis.annotate(text.get_text(), text.xy, xytext=(dx, dy + shift * to_points), textcoords="offset points",
                                       va=text.get_va(), ha=text.get_ha(), fontsize=fontsize, color=text.get_color(), zorder=5,
                                       arrowprops={"arrowstyle": "-", "color": text.get_color(), "linewidth": 0.4,
                                                   "shrinkA": 0, "shrinkB": 2})
            text.remove()
            texts[index] = moved_text
        placed.append(box.translated(0, shift))


def overall_rows(metrics: pd.DataFrame, task: str = "ai_vs_human") -> pd.DataFrame:
    return metrics[(metrics["task"] == task) & (metrics["stratum"] == "all") & ~metrics["benchmark"].isin(OWN_SPLITS)]


def order_by_litmus(overall: pd.DataFrame, metric: str) -> list[str]:
    """Benchmarks ordered by Litmus's value on `metric`, lowest first (drawn at the bottom); alphabetical without Litmus."""
    litmus = overall[overall["detector"].str.startswith("Litmus") & (overall["metric"] == metric)]
    if litmus.empty:
        return sorted(overall["benchmark"].unique())
    order = litmus.groupby("benchmark")["value"].max().sort_values().index.tolist()
    return order + sorted(set(overall["benchmark"]) - set(order))


def value_of(table: pd.DataFrame, **match: str) -> pd.Series | None:
    rows = table
    for column, wanted in match.items():
        rows = rows[rows[column] == wanted]
    return None if rows.empty else rows.iloc[0]


def headline_figure(metrics: pd.DataFrame, mapping: Mapping[str, frozenset[str]], path: Path, *, theme: str = "light",
                    task: str = "ai_vs_human",
                    panels: tuple[str, ...] = ("pr_auc", "roc_auc", "balanced_mcc", "tpr_at_fpr_1")) -> Path:
    """A panel per metric; a row per benchmark, ordered by Litmus's first metric, best at the top; every rerun detector
    with its interval, faded where it trained on the benchmark."""
    colors = style(theme)
    overall = overall_rows(metrics, task)
    benchmarks = order_by_litmus(overall, panels[0])
    detectors = detector_order(overall["detector"].unique())
    figure, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 1.5, 0.4 * len(benchmarks) + 0.9), sharey=True)
    axes = np.atleast_1d(axes)
    offsets = np.linspace(-0.3, 0.3, len(detectors)) if len(detectors) > 1 else np.zeros(1)  # (detectors,)
    any_faded = False
    for axis, metric in zip(axes, panels, strict=True):
        for row, benchmark in enumerate(benchmarks):
            if row % 2 == 0:
                axis.axhspan(row - 0.5, row + 0.5, color=colors["rule"], alpha=0.35, linewidth=0, zorder=0)
            for offset, detector in zip(offsets, detectors, strict=True):
                line = value_of(overall, benchmark=benchmark, detector=detector, metric=metric)
                if line is None:
                    continue
                faded = in_distribution(mapping, detector, benchmark) is True
                any_faded |= faded
                draw_point(axis, float(line["value"]), row + offset, detector, colors, faded=faded,
                           low=float(line["low"]), high=float(line["high"]))
        axis.set_xlabel(METRIC_LABELS[metric])
        axis.set_ylim(-0.5, len(benchmarks) - 0.5)
        axis.grid(axis="x", color=colors["rule"], linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(benchmarks)), [benchmark_name(benchmark) for benchmark in benchmarks])
    letter_panels(list(axes))
    bottom_legend(figure, detector_handles(detectors, colors, faded=any_faded), columns=4)
    return save(figure, path)


def attack_label(attack: str) -> str:
    return "no attack" if attack == "none" else attack.replace("_", " ")


def raid_figure(metrics: pd.DataFrame, attacks: pd.DataFrame, published: pd.DataFrame, mapping: Mapping[str, frozenset[str]],
                path: Path, *, theme: str = "light", benchmark: str = "raid_extra", metric: str = "tpr_at_fpr_5") -> Path:
    """A: the metric per attack on the labeled split, each attack's AI texts against every human text, hardest attack
    first, with the leaderboard's per-attack numbers of the same detectors hollow. B: per domain and overall on the
    split. C: overall, the split's rerun numbers above the leaderboard's quoted ones."""
    colors = style(theme)
    figure = plt.figure(figsize=(9.6, 6.0))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.8, 1.0], height_ratios=[1.2, 1.0], wspace=0.55, hspace=0.75)
    left_top, left_bottom, right = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[:, 1])
    ours = attacks[(attacks["benchmark"] == benchmark) & (attacks["metric"] == metric)]
    detectors = detector_order(ours["detector"].unique())
    order = ours.groupby("subset")["value"].mean().sort_values().index.tolist()
    quoted = published[(published["benchmark"] == "raid") & (published["metric"] == metric)
                       & (published["subset"].str.startswith("attack: ") | (published["subset"] == "no attack"))]
    quoted = quoted[quoted["reported_by"].str.contains("scored by the RAID leaderboard")]
    quoted = quoted.assign(attack=quoted["subset"].str.removeprefix("attack: ").replace({"no attack": "none"}))
    quoted = quoted.drop_duplicates(["detector", "attack"])
    faded_any, quoted_any = False, False
    for detector in detectors:
        color, marker, _ = detector_look(detector, colors)
        faded = in_distribution(mapping, detector, benchmark) is True
        faded_any |= faded
        rows = ours[ours["detector"] == detector].set_index("subset").reindex(order)
        left_top.plot(range(len(order)), rows["value"], color=color, alpha=FADED if faded else 1.0,
                      linewidth=LINE_WIDTH * (1.5 if detector == "Litmus" else 0.9), zorder=3 if detector == "Litmus" else 2)
        for x, (_, line) in enumerate(rows.iterrows()):
            if np.isfinite(line["value"]):
                draw_point(left_top, x, line["value"], detector, colors, faded=faded)
        theirs = quoted[quoted["detector"] == detector].set_index("attack").reindex(order)["value"].astype(float)
        if theirs.notna().any():
            quoted_any = True
            left_top.plot(range(len(order)), theirs, color=color, linewidth=0.7, linestyle=":", zorder=1)
            left_top.plot(range(len(order)), theirs, marker=marker, markersize=4.5, linestyle="none", color=color,
                          markerfacecolor="none", markeredgewidth=0.8, zorder=2)
    left_top.set_xticks(range(len(order)), [attack_label(attack) for attack in order], rotation=35, ha="right")
    left_top.set_ylabel(METRIC_LABELS[metric])
    left_top.set_ylim(0, 1.03)
    left_top.grid(axis="y", color=colors["rule"], linewidth=0.5)
    left_top.set_axisbelow(True)
    domains = metrics[(metrics["benchmark"] == benchmark) & (metrics["metric"] == metric) & (metrics["task"] == "ai_vs_human")
                      & metrics["stratum"].isin(["domain", "all"])]
    columns = sorted(set(domains.loc[domains["stratum"] == "domain", "subset"])) + ["all"]
    offsets = np.linspace(-0.3, 0.3, len(detectors)) if len(detectors) > 1 else np.zeros(1)  # (detectors,)
    for offset, detector in zip(offsets, detectors, strict=True):
        faded = in_distribution(mapping, detector, benchmark) is True
        for x, column in enumerate(columns):
            line = value_of(domains, detector=detector, subset=column)
            if line is not None:
                draw_point(left_bottom, x + offset, float(line["value"]), detector, colors, faded=faded,
                           low=float(line["low"]), high=float(line["high"]), horizontal=False)
    left_bottom.set_xticks(range(len(columns)), [column.capitalize() if column != "all" else "Both" for column in columns])
    left_bottom.set_ylabel(METRIC_LABELS[metric])
    left_bottom.set_ylim(0, 1.03)
    left_bottom.grid(axis="y", color=colors["rule"], linewidth=0.5)
    left_bottom.set_axisbelow(True)
    overall = domains[domains["stratum"] == "all"].sort_values("value")
    board = published[(published["benchmark"] == "raid") & (published["subset"] == "all attacks")
                      & (published["metric"] == metric)].sort_values("value")
    labels = []
    for row, line in enumerate(board.itertuples()):
        role = detector_role(line.detector, line.kind)
        color = colors[role] if role != "litmus" else colors["open"]
        marker = RERUN_MARKERS.get(line.detector, KIND_MARKERS.get(line.kind, "o"))
        right.plot([float(line.value)], [row], marker=marker, markersize=4.5, linestyle="none", color=color,
                   markerfacecolor="none", markeredgewidth=0.8)
        labels.append(short_name(line.detector))
    start = len(board) + 2.0  # a row for each group's label
    for row, line in enumerate(overall.itertuples()):
        faded = in_distribution(mapping, line.detector, benchmark) is True
        draw_point(right, float(line.value), start + row, line.detector, colors, faded=faded, low=float(line.low),
                   high=float(line.high))
        labels.append(short_name(line.detector))
    positions = list(range(len(board))) + [start + row for row in range(len(overall))]
    right.set_yticks(positions, labels, fontsize=FONT_SIZE - 1.5)
    right.axhline(len(board) + 0.5, color=colors["soft"], linewidth=0.6)
    for y, text in ((len(board), "RAID test, leaderboard"), (start + len(overall), "RAID extra, rerun")):
        right.text(0.5, y, text, transform=right.get_yaxis_transform(), va="center", ha="center",
                   fontsize=FONT_SIZE - 1, color=colors["soft"])
    right.set_xlabel(METRIC_LABELS[metric])
    right.set_xlim(0, 1.03)
    right.set_ylim(-0.7, start + len(overall) + 0.6)
    right.grid(axis="x", color=colors["rule"], linewidth=0.5)
    right.set_axisbelow(True)
    right.tick_params(axis="y", length=0)
    letter_panels([left_top, left_bottom, right])
    bottom_legend(figure, detector_handles(detectors, colors, faded=faded_any, quoted=quoted_any), columns=5)
    return save(figure, path)


def pareto_frontier(cost: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Indices of points no cheaper point beats, in order of cost: the best value reachable at each cost."""
    # cost, value: (n,)
    order = np.argsort(cost, kind="stable")  # (n,)
    best = -np.inf
    kept = []
    for index in order:
        if value[index] > best:
            kept.append(index)
            best = value[index]
    return np.array(kept, dtype=int)  # (k,)


def draw_frontier(axis: Axes, speed: np.ndarray, value: np.ndarray, color: str) -> None:
    """The throughput frontier: points no faster point matches in accuracy, joined as steps from fast to slow."""
    # speed, value: (n,)
    kept = pareto_frontier(-speed, value)  # (k,) fastest first, each slower point more accurate
    kept = kept[np.argsort(speed[kept])]  # (k,) slowest first
    axis.step(speed[kept], value[kept], where="pre", color=color, linestyle="--", linewidth=0.8, zorder=1)


def throughput_of(costs: pd.DataFrame, detector: str) -> float | None:
    """Measured tokens a second on the cloud core; Litmus reads its engine's."""
    name = "Litmus engine" if detector == "Litmus" else detector
    rows = costs[(costs["detector"] == name) & costs["measured"]]
    return None if rows.empty else float(rows["tokens_per_second"].iloc[0])


def pareto_figure(fair: pd.DataFrame, every: pd.DataFrame, costs: pd.DataFrame, published: pd.DataFrame, path: Path, *,
                  theme: str = "light", metric: str = "tpr_at_fpr_5") -> Path:
    """Accuracy against single-core throughput (log). A: mean PR AUC over the benchmarks no compared detector trained
    on; B: over every benchmark all scored; C: the RAID leaderboard's quoted number against measured or estimated
    throughput, Litmus's throughput a line since it has no leaderboard number. The frontier is dashed."""
    colors = style(theme)
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), gridspec_kw={"wspace": 0.42})
    for axis, table, label in ((axes[0], fair, "Mean PR AUC, unseen"), (axes[1], every, "Mean PR AUC, all")):
        speeds, values, points = [], [], []
        for line in table.itertuples():
            speed = throughput_of(costs, line.detector)
            if speed is None:
                continue
            draw_point(axis, speed, line.value, line.detector, colors, low=line.low, high=line.high, horizontal=False)
            speeds.append(speed)
            values.append(line.value)
            points.append((speed, line.value, short_name(line.detector), detector_look(line.detector, colors)[0],
                           "below left" if line.detector == "Litmus" else "right"))
        if speeds:
            draw_frontier(axis, np.array(speeds), np.array(values), colors["soft"])
        axis.set_xscale("log")
        axis.set_xlabel("Tokens a second, one core")
        axis.set_ylabel(label)
        axis.grid(color=colors["rule"], linewidth=0.5)
        axis.set_axisbelow(True)
        axis.margins(x=0.25)
        place_labels(axis, points)
    axis = axes[2]
    board = published[(published["benchmark"] == "raid") & (published["subset"] == "all attacks") & (published["metric"] == metric)]
    speeds, values, points = [], [], []
    for line in board.itertuples():
        rows = costs[costs["detector"] == line.detector]
        if rows.empty:
            continue
        measured = bool(rows["measured"].iloc[0])
        speed = float(rows["tokens_per_second"].iloc[0])
        color = (colors[detector_role(line.detector, line.kind)] if measured else colors["estimated"])
        marker = RERUN_MARKERS.get(line.detector, KIND_MARKERS.get(line.kind, "o"))
        axis.plot([speed], [float(line.value)], marker=marker, markersize=5 if measured else 4.5, markerfacecolor="none",
                  markeredgewidth=0.9 if measured else 0.7, color=color, linestyle="none", zorder=3)
        speeds.append(speed)
        values.append(float(line.value))
        points.append((speed, float(line.value), short_name(line.detector), color, "right"))
    if speeds:
        draw_frontier(axis, np.array(speeds), np.array(values), colors["soft"])
    litmus_speed = throughput_of(costs, "Litmus")
    if litmus_speed is not None:
        axis.axvline(litmus_speed, color=colors["litmus"], linewidth=1.0, linestyle="--", zorder=2)
        axis.annotate("Litmus", (litmus_speed, 0.7), xycoords=("data", "axes fraction"), xytext=(3, 0),
                      textcoords="offset points", ha="left", va="center", color=colors["litmus"], fontsize=FONT_SIZE - 1)
    axis.set_xscale("log")
    axis.set_xlabel("Tokens a second, one core")
    axis.set_ylabel(f"RAID {METRIC_LABELS[metric]}")
    axis.grid(color=colors["rule"], linewidth=0.5)
    axis.set_axisbelow(True)
    axis.margins(x=0.25)
    place_labels(axis, points)
    letter_panels(list(axes))
    handles = [Line2D([], [], marker="o", color=colors["open"], linestyle="none", markersize=5, label="Rerun"),
               Line2D([], [], marker="o", color=colors["open"], markerfacecolor="none", linestyle="none", markersize=5,
                      label="Leaderboard, measured speed"),
               Line2D([], [], marker="o", color=colors["estimated"], markerfacecolor="none", linestyle="none", markersize=5,
                      label="Leaderboard, estimated speed"),
               Line2D([], [], color=colors["soft"], linestyle="--", linewidth=0.8, label="Frontier")]
    bottom_legend(figure, handles, columns=4)
    return save(figure, path)


def strata_figure(lengths: pd.DataFrame, metrics: pd.DataFrame, mapping: Mapping[str, frozenset[str]], path: Path, *,
                  theme: str = "light", metric: str = "pr_auc") -> Path:
    """A: the metric by length bin over the pooled benchmarks, a line and band per detector. B: detector by benchmark,
    each cell the metric, a dagger where the detector trained on the benchmark."""
    colors = style(theme)
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.0, 1.6], "wspace": 0.35})
    block = lengths[lengths["metric"] == metric]
    bins = [label for label in LENGTH_LABELS if label in set(block["subset"])]
    detectors = detector_order(block["detector"].unique())
    for detector in detectors:
        color, marker, size = detector_look(detector, colors)
        rows = block[block["detector"] == detector].set_index("subset").reindex(bins)
        litmus = detector == "Litmus"
        left.fill_between(range(len(bins)), rows["low"], rows["high"], color=color, alpha=0.18 if litmus else 0.08, linewidth=0)
        left.plot(range(len(bins)), rows["value"], color=color, linewidth=LINE_WIDTH * (1.6 if litmus else 0.9),
                  marker=marker, markersize=size * 0.75, zorder=3 if litmus else 2, label=short_name(detector))
    left.set_xticks(range(len(bins)), bins)
    left.set_xlabel("Words")
    left.set_ylabel(METRIC_LABELS[metric])
    left.grid(axis="y", color=colors["rule"], linewidth=0.5)
    left.set_axisbelow(True)
    left.legend(fontsize=FONT_SIZE - 1.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1.6,
                columnspacing=0.8)
    overall = overall_rows(metrics)
    overall = overall[overall["metric"] == metric]
    grid = overall.pivot_table(index="detector", columns="benchmark", values="value")
    grid = grid.loc[[name for name in detector_order(grid.index) if name in grid.index][::-1],
                    grid.mean(axis=0).sort_values().index]
    low = np.floor(np.nanmin(grid.to_numpy()) * 10) / 10
    image = right.imshow(grid.to_numpy(), aspect="auto", cmap=METRIC_SCALE, vmin=low, vmax=1.0)
    cmap = plt.get_cmap(METRIC_SCALE)
    for (row, column), value in np.ndenumerate(grid.to_numpy()):
        if not np.isfinite(value):
            continue
        shade = cmap((value - low) / (1.0 - low))
        ink = "#111111" if lightness(to_hex(shade)) > 55 else "#F5F5F5"
        mark = "†" if in_distribution(mapping, grid.index[row], grid.columns[column]) else ""
        right.text(column, row, f"{value:.2f}{mark}", ha="center", va="center", fontsize=FONT_SIZE - 2, color=ink)
    right.set_xticks(range(grid.shape[1]), [benchmark_name(name) for name in grid.columns], rotation=40, ha="right",
                     fontsize=FONT_SIZE - 1)
    right.set_yticks(range(grid.shape[0]), [short_name(name) for name in grid.index], fontsize=FONT_SIZE - 1)
    right.spines[:].set_visible(False)
    right.tick_params(length=0)
    bar = figure.colorbar(image, ax=right, fraction=0.035, pad=0.02)
    bar.set_label(METRIC_LABELS[metric])
    bar.outline.set_visible(False)
    letter_panels([left, right])
    return save(figure, path)


def origins_figure(scores: pd.DataFrame, path: Path, *, theme: str = "light", evaluation: Mapping[str, object] | None = None,
                   detector: str = "Litmus", fpr: float = 0.01) -> Path:
    """A: the detector's probability of not human by origin, on its own held-out split when `evaluation` is given, else
    pooled over the benchmarks. B: share of mixed and AI text flagged per benchmark at the threshold flagging `fpr` of
    the benchmark's own human texts (APT-Eval's polished texts against its originals)."""
    colors = style(theme)
    rows = scores[scores["detector"] == detector]
    if rows.empty:
        raise ValueError(f"no scores of {detector}")
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.0), gridspec_kw={"width_ratios": [1.0, 1.9], "wspace": 0.25})
    if evaluation is not None:
        passages = evaluation["passages"]
        origins = np.array(ORIGIN_ORDER_BY_LABEL)[np.array([passage["label"] for passage in passages])]  # (n,)
        values = np.array([passage["probability"] for passage in passages], dtype=np.float64)  # (n,)
    else:
        origins, values = rows["origin"].to_numpy(), rows["score"].to_numpy(dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 41)
    for origin in ORIGIN_ORDER:
        chosen = values[origins == origin]
        if len(chosen):
            left.hist(chosen, bins=edges, density=True, histtype="stepfilled", alpha=0.3, color=colors[origin],
                      hatch=ORIGIN_HATCHES[origin], edgecolor=colors[origin], linewidth=LINE_WIDTH, label=ORIGIN_LABELS[origin])
    left.set_xlabel("Probability not human")
    left.set_ylabel("Density")
    left.legend(fontsize=FONT_SIZE - 1, loc="upper center")
    paired = rows.assign(benchmark=rows["benchmark"].replace({"apt_eval_human": "apt_eval"}))
    flagged = []
    for benchmark, block in paired.groupby("benchmark"):
        human = block.loc[block["origin"] == "human", "score"].to_numpy(dtype=np.float64)  # (h,)
        if len(human) < 20:
            continue
        for origin in ("mixed", "ai"):
            chosen = block.loc[block["origin"] == origin, "score"].to_numpy(dtype=np.float64)  # (c,)
            if len(chosen):
                labels = np.r_[np.zeros(len(human)), np.ones(len(chosen))]  # (h + c,)
                flagged.append((benchmark, origin, tpr_at_fpr(labels, np.r_[human, chosen], fpr)))
    table = pd.DataFrame(flagged, columns=["benchmark", "origin", "share"])
    order = table.groupby("benchmark")["share"].mean().sort_values().index.tolist()
    width = 0.4
    for position, origin in enumerate(("mixed", "ai")):
        shares = table[table["origin"] == origin].set_index("benchmark").reindex(order)["share"]
        positions = np.arange(len(order)) + (position - 0.5) * width  # (benchmarks,)
        right.bar(positions, shares, width * 0.92, color=colors[origin], alpha=0.85, hatch=ORIGIN_HATCHES[origin],
                  edgecolor=colors[origin], linewidth=0.0, label=ORIGIN_LABELS[origin])
        for x, share in zip(positions, shares, strict=True):
            if np.isfinite(share):
                right.text(x, share + 0.015, f"{share:.0%}", ha="center", va="bottom", fontsize=FONT_SIZE - 2.5,
                           color=colors["soft"])
    names = {"apt_eval": "APT-Eval"}
    right.set_xticks(range(len(order)), [names.get(name, benchmark_name(name)) for name in order], rotation=35, ha="right")
    right.set_ylabel(f"Flagged at {fpr:.0%} FPR")
    right.set_ylim(0, 1.03)
    right.grid(axis="y", color=colors["rule"], linewidth=0.5)
    right.set_axisbelow(True)
    right.legend(fontsize=FONT_SIZE - 1, loc="upper left")
    letter_panels([left, right])
    return save(figure, path)


def reporter_category(reported_by: str) -> str:
    """Who stands behind a published number: the benchmark's authors, a detector's owner who submitted predictions the
    benchmark scored, a detector's owner alone, a competitor, or an independent third party."""
    text = reported_by.lower()
    if text.startswith(("benchmark authors", "task organizers")):
        return "benchmark"
    if "submission" in text:
        return "self-submitted"
    if text.startswith("competitor"):
        return "competitor"
    if text.startswith(("vendor", "detector owner", "detector authors")):
        return "self-reported"
    if text.startswith("third party"):
        return "third party"
    raise ValueError(f"unknown reporter {reported_by!r}")


def claims_figure(published: pd.DataFrame, path: Path, *, theme: str = "light") -> Path:
    """Every published (detector, benchmark, subset, metric) measured by more than one kind of reporter, one row each,
    with the reporters' values joined: where a vendor's number and others' agree or part."""
    colors = style(theme)
    frame = published.assign(category=published["reported_by"].map(reporter_category))
    groups = [(key, block) for key, block in frame.groupby(["detector", "benchmark", "subset", "metric"])
              if block["category"].nunique() > 1]
    if not groups:
        raise ValueError("no published number was measured by more than one kind of reporter")
    figure, axis = plt.subplots(figsize=(5.2, 0.38 * len(groups) + 0.9))
    for row, ((detector, _, _, _), block) in enumerate(groups):
        color = colors[detector_role(detector, block["kind"].iloc[0])]
        values = block["value"].astype(float)
        axis.plot([values.min(), values.max()], [row, row], color=colors["rule"], linewidth=2.0, zorder=1)
        for _, line in block.iterrows():
            axis.plot([float(line["value"])], [row], marker=REPORTER_MARKERS[line["category"]], markersize=5,
                      markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.0, linestyle="none", zorder=2)
    axis.set_yticks(range(len(groups)), [f"{detector}, {benchmark} {subset}, {METRIC_LABELS.get(metric, metric)}"
                                         for (detector, benchmark, subset, metric), _ in groups], fontsize=FONT_SIZE - 1)
    axis.set_xlabel("Published value")
    shown = set(frame.loc[[index for _, block in groups for index in block.index], "category"])
    handles = [Line2D([], [], marker=marker, color=colors["soft"], markerfacecolor="none", linestyle="none", markersize=5,
                      label=category.capitalize()) for category, marker in REPORTER_MARKERS.items() if category in shown]
    axis.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=FONT_SIZE - 1)
    return save(figure, path)


def draw_site_figures(inputs: Mapping[str, object], site: Path, docs: Path | None) -> list[Path]:
    """Every figure in both themes as `site/<name>_<theme>.png`, copied to `docs/benchmarks_<name>_<theme>.png`.

    `inputs` holds the frames each figure reads: metrics, attacks, lengths, published, scores, costs, fair, every,
    mapping, evaluation.
    """
    paths = []
    for theme in THEME_NAMES:
        drawn = [
            headline_figure(inputs["metrics"], inputs["mapping"], site / f"headline_{theme}.png", theme=theme),
            raid_figure(inputs["metrics"], inputs["attacks"], inputs["published"], inputs["mapping"], site / f"raid_{theme}.png",
                        theme=theme),
            pareto_figure(inputs["fair"], inputs["every"], inputs["costs"], inputs["published"], site / f"pareto_{theme}.png",
                          theme=theme),
            strata_figure(inputs["lengths"], inputs["metrics"], inputs["mapping"], site / f"strata_{theme}.png", theme=theme),
            origins_figure(inputs["scores"], site / f"origins_{theme}.png", theme=theme, evaluation=inputs["evaluation"]),
        ]
        for path in drawn:
            paths.append(path)
            if docs is not None:
                docs.mkdir(parents=True, exist_ok=True)
                paths.append(Path(shutil.copyfile(path, docs / f"benchmarks_{path.name}")))
    return paths
