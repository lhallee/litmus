"""A regression plot of predictions against their true values, with its statistics in a box.

DSM's `metrics.regression_plot` held it, and egfr_binding_proxies copied that module whole.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections.abc import Mapping
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


__all__ = ["plot_predictions"]


def plot_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    metrics: Mapping[str, float],
    output_path: str | os.PathLike[str],
) -> None:
    """Save a scatter of predictions against labels, with a regression line and its 95% band.

    Pairs where either value is not finite are left out. The box gives R^2, and Spearman's and
    Pearson's rho with their p-values, from `metrics` under `r_squared`, `spearman_rho`,
    `spear_pval`, `pearson_rho`, and `pear_pval` where it holds them. Each is computed from the
    pairs whether or not `metrics` holds it, so fewer than two finite pairs raise `ValueError`,
    and the figure is left open. seaborn draws the band's bootstrap from a generator it seeds from
    the operating system, so two calls draw different bands. The figure is saved at 300 dpi, in a
    directory created for it, and closed, and the path it went to is printed.
    """
    # predictions: (...); labels: (...), each of n elements
    predictions, labels = predictions.flatten(), labels.flatten()  # (n,), (n,)
    mask = np.isfinite(labels) & np.isfinite(predictions)  # (n,); n_f = pairs with both finite
    labels, predictions = labels[mask], predictions[mask]  # (n_f,), (n_f,)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=labels, y=predictions, ax=ax)
    sns.regplot(x=labels, y=predictions, ci=95, ax=ax, scatter=False, line_kws={"color": "red"})

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")

    # Each default is computed before `get` looks up its key.
    r2 = metrics.get("r_squared", r2_score(labels, predictions))
    r_s = metrics.get("spearman_rho", spearmanr(labels, predictions)[0])
    p_s = metrics.get("spear_pval", spearmanr(labels, predictions)[1])
    r_p = metrics.get("pearson_rho", pearsonr(labels, predictions)[0])
    p_p = metrics.get("pear_pval", pearsonr(labels, predictions)[1])

    stats_text = (
        f"$R^2$ = {r2:.2f}\n"
        f"Spearman $\\rho$ = {r_s:.2f}  (p = {p_s:.2e})\n"
        f"Pearson $\\rho$ = {r_p:.2f}  (p = {p_p:.2e})"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=12, verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    plt.close(fig)
