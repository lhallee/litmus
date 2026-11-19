from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from litmus_benchmarks.evaluate import METRICS, evaluate_scores, point_metrics, raid_leaderboard_metrics, tpr_at_fpr
from sklearn.metrics import roc_auc_score


def synthetic_scores(detectors: dict[str, float], n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Scores of several detectors on one benchmark of human, AI and mixed texts; `separation` shifts AI scores up."""
    rng = np.random.default_rng(seed)
    origins = rng.choice(["human", "ai", "mixed"], size=n, p=[0.45, 0.4, 0.15])  # (n,)
    texts = pd.DataFrame({
        "benchmark": "toy", "row": np.arange(n), "origin": origins,
        "generator": np.where(origins == "human", "human", rng.choice(["g1", "g2"], size=n)),
        "domain": rng.choice(["news", "essays"], size=n), "attack": "none",
        "words": rng.integers(10, 600, size=n),
    })
    frames = []
    for detector, separation in detectors.items():
        shift = np.select([origins == "ai", origins == "mixed"], [separation, separation / 2], 0.0)  # (n,)
        frames.append(texts.assign(detector=detector, score=rng.normal(size=n) + shift))
    return pd.concat(frames, ignore_index=True)


def test_tpr_at_fpr_counts_exactly() -> None:
    negatives = np.arange(100, dtype=float)  # five negatives (95..99) lie above the 5% threshold of 94
    positives = np.array([93.0, 94.0, 94.5, 99.5])
    labels = np.r_[np.zeros(100), np.ones(4)].astype(int)
    scores = np.r_[negatives, positives]
    assert tpr_at_fpr(labels, scores, 0.05) == pytest.approx(2 / 4)  # 94.5 and 99.5 beat 94; a tie does not
    assert tpr_at_fpr(labels, scores, 0.0) == pytest.approx(1 / 4)


def test_tpr_at_fpr_never_exceeds_target_with_ties() -> None:
    labels = np.r_[np.zeros(50), np.ones(50)].astype(int)
    scores = np.r_[np.full(50, 0.7), np.linspace(0, 1, 50)]  # every negative tied
    threshold_flags = scores[labels == 0] > 0.7
    assert threshold_flags.mean() <= 0.05
    assert tpr_at_fpr(labels, scores, 0.05) == pytest.approx(np.mean(scores[labels == 1] > 0.7))


def test_point_metrics_on_perfect_and_reversed_scores() -> None:
    labels = np.r_[np.zeros(60), np.ones(40)].astype(int)
    perfect = point_metrics(labels, labels + np.linspace(0, 0.5, 100))
    assert all(perfect[name] == pytest.approx(1.0) for name in METRICS)
    reversed_metrics = point_metrics(labels, -labels.astype(float))
    assert reversed_metrics["roc_auc"] == pytest.approx(0.0)
    assert reversed_metrics["tpr_at_fpr_1"] == pytest.approx(0.0)


def test_roc_auc_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=300)
    scores = rng.integers(0, 8, size=300).astype(float)  # heavy ties
    assert point_metrics(labels, scores)["roc_auc"] == pytest.approx(roc_auc_score(labels, scores))


def test_evaluate_scores_intervals_and_strata() -> None:
    table = evaluate_scores(synthetic_scores({"strong": 3.0, "weak": 0.5}), replicates=60, strata_replicates=30,
                            mcc_subsamples=3)
    overall = table[(table["task"] == "ai_vs_human") & (table["stratum"] == "all")]
    assert set(overall["metric"]) == set(METRICS)
    assert ((overall["low"] <= overall["value"] + 1e-9) & (overall["value"] <= overall["high"] + 1e-9)).all()
    roc = overall[overall["metric"] == "roc_auc"].set_index("detector")["value"]
    assert roc["strong"] > 0.95 > roc["weak"] > 0.55
    generator = table[(table["stratum"] == "generator") & (table["task"] == "ai_vs_human")]
    assert set(generator["subset"]) == {"g1", "g2"}
    all_negatives = overall["negatives"].iloc[0]
    assert (generator["negatives"] == all_negatives).all()  # every human text is a negative for each generator
    domains = table[(table["stratum"] == "domain") & (table["task"] == "ai_vs_human")]
    assert (domains["negatives"] < all_negatives).all()  # a domain compares within itself
    assert {"any_ai_vs_human", "mixed_vs_human"} <= set(table["task"])


def test_evaluate_scores_is_deterministic() -> None:
    scores = synthetic_scores({"a": 1.0, "b": 2.0})
    first = evaluate_scores(scores, replicates=20, strata_replicates=10, mcc_subsamples=2)
    second = evaluate_scores(scores, replicates=20, strata_replicates=10, mcc_subsamples=2)
    pd.testing.assert_frame_equal(first, second)


def raid_result(tpr5: float, tpr1: float) -> dict:
    """A RAID leaderboard results.json in miniature, with the fields `raid_leaderboard_metrics` reads."""
    def score(attack: str, scale: float, domain: str = "all") -> dict:
        return {"domain": domain, "model": "all", "decoding": "all", "repetition_penalty": "all", "attack": attack,
                "accuracy": {"0.05": {"accuracy": tpr5 * scale}, "0.01": {"accuracy": tpr1 * scale}}, "auroc": 0.99}
    return {"score_agg": {"all": score("all", 1.0), "no_adversarial": score("none", 1.0)},
            "scores": [score("paraphrase", 0.9), score("homoglyph", 0.8), score("paraphrase", 0.1, domain="news"),
                       score("all", 1.0)]}


def test_raid_leaderboard_metrics_reads_overall_and_attack_rows() -> None:
    table = raid_leaderboard_metrics(raid_result(0.98, 0.95), "Litmus")
    overall = table[(table["stratum"] == "all") & (table["subset"] == "all")].set_index("metric")["value"]
    assert overall["tpr_at_fpr_5"] == pytest.approx(0.98) and overall["tpr_at_fpr_1"] == pytest.approx(0.95)
    attacks = table[(table["stratum"] == "attack") & (table["metric"] == "tpr_at_fpr_5")].set_index("subset")["value"]
    assert attacks.to_dict() == pytest.approx({"paraphrase": 0.98 * 0.9, "homoglyph": 0.98 * 0.8})  # news row skipped


def test_evaluate_scores_refuses_mismatched_texts_and_nonfinite_scores() -> None:
    scores = synthetic_scores({"a": 1.0, "b": 2.0})
    shuffled = scores.copy()
    b_rows = shuffled.index[shuffled["detector"] == "b"]
    shuffled.loc[b_rows, "row"] = shuffled.loc[b_rows, "row"].to_numpy()[::-1]
    with pytest.raises(ValueError, match="scored different texts"):
        evaluate_scores(shuffled, replicates=5, strata_replicates=5)
    broken = scores.copy()
    broken.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_scores(broken, replicates=5, strata_replicates=5)
