from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from litmus_benchmarks.evaluate import evaluate_scores
from litmus_benchmarks.figures import (THEME_NAMES, claims_figure, draw_site_figures, headline_figure, origins_figure,
                                       own_split_lengths, pareto_figure, pareto_frontier, raid_figure, reporter_category,
                                       speed_figure, strata_figure)
from litmus_benchmarks.results import align_texts, attack_rows, length_rows, mean_pr_auc
from PIL import Image


BENCHMARKS = Path(__file__).resolve().parents[1]
DETECTORS = {"Litmus": 3.0, "ai-vs-human-checker": 2.0, "Binoculars": 1.5, "e5-small-lora": 1.0}
MAPPING = {"Litmus": frozenset(), "ai-vs-human-checker": frozenset({"mage"}), "Binoculars": frozenset(),
           "e5-small-lora": frozenset({"raid_extra"})}


@pytest.fixture(scope="module")
def scores() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    frames = []
    for benchmark in ("mage", "raid_extra", "beemo"):
        n = 300
        origins = rng.choice(["human", "ai", "mixed"], size=n, p=[0.5, 0.35, 0.15])  # (n,)
        texts = pd.DataFrame({"benchmark": benchmark, "row": np.arange(n), "origin": origins,
                              "generator": np.where(origins == "human", "human", rng.choice(["gpt4", "llama"], size=n)),
                              "domain": rng.choice(["czech", "german"], size=n),
                              "attack": rng.choice(["none", "paraphrase"], size=n), "words": rng.integers(20, 900, size=n)})
        for detector, separation in DETECTORS.items():
            shift = np.select([origins == "ai", origins == "mixed"], [separation, separation / 2], 0.0)  # (n,)
            frames.append(texts.assign(detector=detector, score=1 / (1 + np.exp(-(rng.normal(size=n) + shift - 1)))))
    return align_texts(pd.concat(frames, ignore_index=True))[0]


@pytest.fixture(scope="module")
def metrics(scores: pd.DataFrame) -> pd.DataFrame:
    by_set = scores.drop(columns=["benchmark"]).rename(columns={"text_set": "benchmark"})
    return evaluate_scores(by_set, replicates=20, strata_replicates=10, mcc_subsamples=2)


@pytest.fixture(scope="module")
def published() -> pd.DataFrame:
    return pd.read_csv(BENCHMARKS / "published.csv", keep_default_na=False)


@pytest.fixture(scope="module")
def costs() -> pd.DataFrame:
    return pd.DataFrame({"detector": ["Litmus engine", "ai-vs-human-checker", "Binoculars", "e5-small-lora",
                                      "Desklib v1.01", "TMR (Target Mining RoBERTa)"],
                         "tokens_per_second": [9000.0, 260.0, 4.3, 1600.0, 120.0, 490.0],
                         "words_per_second": [7300.0, 210.0, 3.5, 1300.0, 98.0, 400.0],
                         "measured": [True, True, True, True, True, False]})


@pytest.fixture(scope="module")
def evaluation() -> dict:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 3, size=300)
    return {"split": "validation", "passages": [{"label": int(label), "probability": float(np.clip(0.3 * label + rng.normal(0.2, 0.15), 0, 1))}
                                                for label in labels],
            "rows": [{"evaluation": "passages", "stratum_kind": "tokens", "stratum": band, "metric": "pr_auc", "value": value}
                     for band, value in (("1-128", 0.92), ("129-512", 0.99), ("2049-8192", 1.0), ("all", 0.95))]}


def assert_png_at_300_dpi(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 1000
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert tuple(round(value) for value in image.info["dpi"]) == (300, 300)
        assert image.mode == "RGBA" and image.getextrema()[3][0] == 0, "a transparent ground"


@pytest.mark.parametrize("theme", THEME_NAMES)
def test_headline_and_strata_figures(metrics: pd.DataFrame, scores: pd.DataFrame, tmp_path: Path, theme: str) -> None:
    assert_png_at_300_dpi(headline_figure(metrics, MAPPING, tmp_path / "headline.png", theme=theme))
    lengths = length_rows(scores, ["mage", "beemo"], replicates=10)
    assert set(lengths["subset"]) <= {"1-50", "51-150", "151-400", "401-1000", ">1000"}
    assert_png_at_300_dpi(strata_figure(lengths, metrics, MAPPING, tmp_path / "strata.png", theme=theme))


def test_raid_figure_with_leaderboard(metrics: pd.DataFrame, scores: pd.DataFrame, published: pd.DataFrame, tmp_path: Path) -> None:
    attacks = attack_rows(scores, "raid_extra", replicates=10)
    assert set(attacks["subset"]) == {"none", "paraphrase"}
    humans = int((scores[(scores["benchmark"] == "raid_extra") & (scores["detector"] == "Litmus")]["origin"] == "human").sum())
    assert (attacks["negatives"] == humans).all(), "each attack is read against every human text"
    assert_png_at_300_dpi(raid_figure(metrics, attacks, published, MAPPING, tmp_path / "raid.png", theme="dark"))


def test_pareto_figure_and_frontier(scores: pd.DataFrame, costs: pd.DataFrame, published: pd.DataFrame, tmp_path: Path) -> None:
    fair = mean_pr_auc(scores, ["beemo"], replicates=10)
    every = mean_pr_auc(scores, ["beemo", "mage", "raid_extra"], replicates=10)
    assert ((fair["low"] <= fair["value"] + 1e-9) & (fair["value"] <= fair["high"] + 1e-9)).all()
    assert_png_at_300_dpi(pareto_figure(fair, every, costs, published, tmp_path / "pareto.png", theme="light"))
    assert pareto_frontier(np.array([3.0, 1.0, 2.0, 4.0]), np.array([0.9, 0.5, 0.4, 0.8])).tolist() == [1, 0]


@pytest.mark.parametrize("theme", THEME_NAMES)
def test_speed_figure_with_and_without_the_own_split(scores: pd.DataFrame, costs: pd.DataFrame, evaluation: dict,
                                                     tmp_path: Path, theme: str) -> None:
    fair = mean_pr_auc(scores, ["beemo"], replicates=10)
    lengths = length_rows(scores, ["mage", "beemo"], replicates=10)
    assert_png_at_300_dpi(speed_figure(fair, costs, lengths, tmp_path / "speed.png", theme=theme, evaluation=evaluation))
    assert_png_at_300_dpi(speed_figure(fair, costs, lengths, tmp_path / "speed_public.png", theme=theme))
    assert own_split_lengths(evaluation, "pr_auc")["stratum"].tolist() == ["1-128", "129-512", "2049-8192"]
    with pytest.raises(ValueError, match="no roc_auc rows"):
        own_split_lengths(evaluation, "roc_auc")


def test_origins_figure_reads_own_split(scores: pd.DataFrame, evaluation: dict, tmp_path: Path) -> None:
    assert_png_at_300_dpi(origins_figure(scores, tmp_path / "origins.png", theme="dark", evaluation=evaluation))
    assert_png_at_300_dpi(origins_figure(scores, tmp_path / "origins_pooled.png", theme="light"))


def test_site_figures_write_both_themes_and_docs_copies(metrics: pd.DataFrame, scores: pd.DataFrame, costs: pd.DataFrame,
                                                        published: pd.DataFrame, evaluation: dict, tmp_path: Path) -> None:
    inputs = {"metrics": metrics, "attacks": attack_rows(scores, "raid_extra", replicates=10),
              "lengths": length_rows(scores, ["mage", "beemo"], replicates=10), "published": published, "scores": scores,
              "costs": costs, "fair": mean_pr_auc(scores, ["beemo"], replicates=10),
              "every": mean_pr_auc(scores, ["beemo", "mage"], replicates=10), "mapping": MAPPING, "evaluation": evaluation}
    paths = draw_site_figures(inputs, tmp_path / "site", tmp_path / "docs")
    names = {path.name for path in paths}
    for name in ("headline", "raid", "pareto", "speed", "strata", "origins"):
        for theme in THEME_NAMES:
            assert f"{name}_{theme}.png" in names and f"benchmarks_{name}_{theme}.png" in names


def test_claims_figure_reads_every_reporter(published: pd.DataFrame, tmp_path: Path) -> None:
    categories = {reporter_category(value) for value in published["reported_by"]}
    assert {"benchmark", "self-submitted", "self-reported", "competitor", "third party"} <= categories
    assert_png_at_300_dpi(claims_figure(published, tmp_path / "claims.png"))
