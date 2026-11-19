---
id: foundry.plotting
kind: library_module
status: active
aliases: [foundry.plotting]
---
# foundry.plotting

Plots that several projects share, one submodule each, so that importing one loads only the
libraries it draws with.

| Submodule | Holds |
|---|---|
| `regression` | A scatter of predictions against true values, with a regression line, its 95% band, and the fit's statistics |
| `panels` | `letter_panels(axes)`: bold capitals A, B, C at each panel's top left, outside the plot area, in the order given; a single panel gets none, as the [figure convention](../../docs/conventions/figures.md) asks. Promoted 2026-10-23 from the production suite's `atlas_figures.letter_panels`, whose placement it keeps; the production suite's own copies (`atlas_figures.py`, `atlas_pli_figures.py`, `fusion/probe_playground.py`) are not yet converted. Tests: `tests/tier1_unit/test_plotting_panels.py` |
| `floors` | `draw_panel(ax, panel, colors)` and `floors_figure(panels, colors, labels, grid, figsize, range_label, adjust)`: one panel per record, each baseline a horizontal bar colored by its role with its value or interval written beside it, and the grid's last cell a legend. Promoted 2026-11-01 from the two figure sets that draw with it, `figures/atlas/2026-11-01_pair_floors` and `figures/production_suite/2026-11-01_release_floors`, whose PNGs reproduced byte for byte at promotion. Tests: `tests/tier1_unit/test_plotting_floors.py` |

## `foundry.plotting.regression`

| Name | Does |
|---|---|
| `plot_predictions(predictions, labels, metrics, output_path)` | Leaves out each pair where either value is not finite, then draws the pairs, a red regression line with its 95% band, and a box with R^2 and Spearman's and Pearson's rho and their p-values. Saves the figure at 300 dpi to `output_path`, creating its directory, prints the path, and closes the figure |

`predictions` and `labels` are NumPy arrays of any shape with the same number of elements, read
flattened, pair by pair. The box takes each statistic from `metrics` where it holds `r_squared`,
`spearman_rho`, `spear_pval`, `pearson_rho`, or `pear_pval`, but computes each from the pairs
either way, so fewer than two finite pairs raise `ValueError`: none from `r2_score`, one from
`pearsonr`. The figure is left open then. seaborn bootstraps the band with a generator it seeds
from the operating system, so two calls with the same values draw slightly different bands.
The figure has no title; put its description in the surrounding caption, following the
[figure convention](../../docs/conventions/figures.md).

Promoted on 2026-09-24. These copies now import it:

- DSM's `metrics/regression_plot.py`, which its `synthyra_api/affinity_pred.py` plots with.
- egfr_binding_proxies' `metrics/regression_plot.py`, a copy of DSM's, which its own
  `synthyra_api/affinity_pred.py` plots with.

At promotion, against the copy, byte for byte, with the band's generator seeded the same on both sides: the PNG
written, what was printed, the warnings raised, each error, the files and directories left behind,
the figures left open, and NumPy's global random state. The inputs were random predictions and
labels of several shapes and dtypes, with values that are not finite, constant, tied, huge, or
tiny, too few pairs, lengths that do not match, `metrics` holding any of its keys with any value,
and paths into new, existing, and blocked directories. What changed:

- Each copy now holds `plot_predictions` alone. It also held the names it imported, such as `plt`
  and `np`, which nothing imported from it.

On 2026-10-14, removed the in-image title to follow the figure convention. Statistics,
pair filtering, confidence bands, file handling and 300 dpi output are unchanged.

Not here: Protify's `regression_ci_plot` in `visualization/ci_plots.py` draws the same figure
under a title of its own, but computes every statistic itself, sets its box without a background,
creates no directory, and prints nothing, so one copy would change one project's figures.

## `foundry.plotting.floors`

A panel is a dict with `letter`, `axis` (the x label, naming the record), `higher_is_better` and `bars`, each bar a dict with
`name`, `role` and either `value` or `low` and `high` (then the bar reaches `low` and a line marks the interval). The roles' colors
and legend labels come from the caller as dicts in the roles' order, which also sorts a panel's bars top to bottom; a bar with a
role the colors do not name raises `ValueError`. `floors_figure` asserts that the grid has exactly one cell more than there are
panels, puts the legend there, applies `adjust` through `subplots_adjust` and returns the figure for the caller to save at 300
dpi on `SURFACE`. Each panel draws its own letter above its tick labels rather than through `panels.letter_panels`, since the
baseline names on the left leave no room where that places one.

## Tests

`tests/tier1_unit/test_plotting.py`, with DSM's `tests/test_metrics.py` as the guard on the name it
imports. egfr_binding_proxies also has its own table-analysis tests under `tests/`.
