from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from litmus_benchmarks.estimate import estimate_throughput
from litmus_benchmarks.results import (BenchmarkSets, align_texts, benchmark_sets, cached_metrics, catalog_filters, cost_table, filter_rows, flag_rates,
                                       in_distribution, markdown_tables, own_split_rows, parse_row_filters, results_json,
                                       scores_digest, site_rows, threshold_at_fpr, trained_on)


def texts(benchmark: str, n: int, detectors: dict[str, float], seed: int = 0, origins: tuple[str, ...] = ("human", "ai")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origin = rng.choice(origins, size=n)  # (n,)
    base = pd.DataFrame({"benchmark": benchmark, "row": np.arange(n), "origin": origin, "generator": "g",
                         "domain": rng.choice(["code", "czech", "german"], size=n), "attack": "none", "words": 100})
    return pd.concat([base.assign(detector=name, score=rng.normal(size=n) + shift * (origin != "human"))
                      for name, shift in detectors.items()], ignore_index=True)


def test_the_scores_digest_is_the_same_on_every_platform() -> None:
    """The digest names the metrics cache a work folder carries between machines, so it hashes the scores with `\\n`
    line endings whatever the platform's are."""
    import hashlib

    scores = pd.DataFrame({"benchmark": ["a", "a"], "row": [0, 1], "score": [0.25, 0.5]})
    text = "benchmark,row,score\na,0,0.25\na,1,0.5\n"
    assert scores_digest(scores, 3, 2) == hashlib.sha256((text + "(3, 2)").encode()).hexdigest()[:16]


def test_row_filters_keep_named_values_of_one_benchmark() -> None:
    scores = pd.concat([texts("raid_extra", 90, {"a": 1.0}), texts("mage", 30, {"a": 1.0})], ignore_index=True)
    kept = filter_rows(scores, parse_row_filters([]))
    assert set(kept.loc[kept["benchmark"] == "raid_extra", "domain"]) == {"czech", "german"}
    assert (kept["benchmark"] == "mage").sum() == 30
    assert len(filter_rows(scores, parse_row_filters(["raid_extra:"]))) == 120
    only_code = filter_rows(scores, parse_row_filters(["raid_extra:domain=code"]))
    assert set(only_code.loc[only_code["benchmark"] == "raid_extra", "domain"]) == {"code"}


def test_align_texts_splits_other_samples_and_drops_unscored_texts() -> None:
    same = texts("beemo", 50, {"a": 1.0, "b": 2.0})
    other = texts("beemo", 50, {"c": 1.0}, seed=9).assign(row=lambda frame: frame["row"] + 1000)
    scores = pd.concat([same, other], ignore_index=True).astype({"score": object})
    scores.loc[(scores["detector"] == "a") & (scores["row"] == 3), "score"] = ""
    aligned, notes = align_texts(scores)
    assert set(aligned.loc[aligned["detector"].isin(["a", "b"]), "text_set"]) == {"beemo"}
    assert set(aligned.loc[aligned["detector"] == "c", "text_set"]) == {"beemo#2"}
    assert (aligned.loc[aligned["detector"] == "b", "row"] != 3).all(), "the text one detector left unscored leaves the set"
    assert {note["text_set"] for note in notes} == {"beemo", "beemo#2"}


def test_metrics_do_not_depend_on_worker_count(tmp_path) -> None:
    scores, _ = align_texts(pd.concat([texts("beemo", 80, {"a": 1.0, "b": 2.0}), texts("mixset", 80, {"a": 1.0, "b": 2.0}, seed=3)],
                                      ignore_index=True))
    one = cached_metrics(scores, tmp_path / "one", replicates=10, strata_replicates=5, workers=1)
    two = cached_metrics(scores, tmp_path / "two", replicates=10, strata_replicates=5, workers=2)
    pd.testing.assert_frame_equal(one, two)
    assert set(one["text_set"]) == {"beemo", "mixset"}


def test_trained_on_reads_litmus_from_the_catalog_column_unless_given() -> None:
    catalog = pd.DataFrame({"name": ["mage", "beemo"], "in_distribution": ["yes", "no"],
                            "row_filter": ['{"domain": ["czech", "german"]}', '{"split": "test"}']})
    mapping = trained_on(catalog, None)
    assert in_distribution(mapping, "Litmus", "mage") and not in_distribution(mapping, "Litmus", "beemo")
    assert in_distribution(mapping, "Desklib v1.01", "raid_extra") and in_distribution(mapping, "RADAR", "mage") is None
    assert trained_on(catalog, [])["Litmus"] == frozenset(), "an explicit empty list overrides the catalog"
    assert catalog_filters(catalog) == {"mage": {"domain": ("czech", "german")}}, "a raw-column filter was applied at scoring"


def test_benchmark_sets_need_every_detector_and_no_training() -> None:
    scores = pd.concat([texts("beemo", 40, {"Litmus": 1.0, "e5-small-lora": 1.0}),
                        texts("raid_extra", 40, {"Litmus": 1.0, "e5-small-lora": 1.0}),
                        texts("droid", 40, {"Litmus": 1.0})], ignore_index=True)
    sets = benchmark_sets(scores, trained_on(pd.DataFrame({"name": []}), None))
    assert sets.every == ["beemo", "raid_extra"] and sets.unseen == ["beemo"]


def test_flag_rates_use_the_pooled_human_threshold() -> None:
    pool = texts("beemo", 400, {"Litmus": 3.0})
    polished = pd.DataFrame({"benchmark": "apt_eval", "row": range(100), "origin": "mixed", "generator": "g", "domain": "d",
                             "attack": "", "words": 100, "detector": "Litmus", "score": np.linspace(-3, 6, 100)})
    rates = flag_rates(pd.concat([pool, polished], ignore_index=True), ["beemo"], replicates=50).set_index("benchmark")
    human = pool.loc[pool["origin"] == "human", "score"].to_numpy()
    assert rates.loc["apt_eval", "flagged_at_fpr_1"] == pytest.approx(np.mean(polished["score"] > threshold_at_fpr(human, 0.01)))
    assert rates.loc["apt_eval", "flagged_at_own_threshold"] == pytest.approx(np.mean(polished["score"] > 0.5))
    assert np.mean(human > threshold_at_fpr(human, 0.05)) <= 0.05


def test_estimates_scale_by_non_embedding_parameters_and_refuse_far_ones() -> None:
    models = pd.DataFrame({"detector": ["base", "large", "tiny", "other"], "model_type": ["roberta", "roberta", "roberta", "t5"],
                           "non_embedding_parameters": [100, 300, 5, 100], "measured": [True, False, False, False],
                           "repo": "r", "revision": "v", "parameters": 0})
    costs = pd.DataFrame({"detector": ["base"], "words_per_second": [300.0], "tokens_per_second": [400.0]})
    estimates = estimate_throughput(models, costs).set_index("detector")
    assert estimates.loc["large", "tokens_per_second"] == pytest.approx(400.0 / 3)
    assert "larger" not in estimates.loc["tiny", "excluded"] and estimates.loc["tiny", "excluded"].startswith("20 times smaller")
    assert estimates.loc["other", "excluded"] == "no measured t5 model"


def test_results_json_keeps_release_and_marks_training(tmp_path) -> None:
    evaluation = {"split": "test", "passages": [{"label": label % 3, "probability": 0.2 + 0.3 * (label % 3) + 0.01 * (label % 7)}
                                                 for label in range(300)]}
    own = own_split_rows(evaluation, "Litmus", replicates=20)
    assert set(own["task"]) == {"ai_vs_human", "any_ai_vs_human", "mixed_vs_human"} and (own["benchmark"] == "test").all()
    metrics = own.assign(benchmark="raid_extra", detector="e5-small-lora")
    mapping = trained_on(pd.DataFrame({"name": []}), None)
    rows = site_rows(pd.concat([own, metrics], ignore_index=True), pd.DataFrame(), mapping)
    assert {row["in_distribution"] for row in rows if row["detector"] == "e5-small-lora"} == {True}
    measured = pd.DataFrame({"detector": ["Litmus engine"], "tokens": [1230], "words": [1000], "tokens_per_second": [9000.0],
                             "words_per_second": [7300.0], "ms_per_1k_words": [137.0], "machine": ["cpu"], "texts": [200],
                             "peak_rss_mb": [800.0]})
    architectures = pd.DataFrame(columns=["detector", "model_type", "non_embedding_parameters", "measured", "repo", "revision"])
    costs, excluded = cost_table(measured, architectures.astype({"measured": bool}))
    assert set(costs["detector"]) == {"Litmus engine"}
    empty = pd.DataFrame(columns=["detector", "value", "low", "high"])
    document = results_json({"release": {"version": "kept"}, "benchmarks": [{"id": "test", "name": "Test"}]}, model_name="Litmus",
                            model_version="interim", date="2026-11-18", rows=rows, costs=costs, texts_per_benchmark=300,
                            sets=BenchmarkSets([], []), fair=empty, every=empty, filters=parse_row_filters([]))
    assert document["release"] == {"version": "kept"} and document["schema"] == 1
    assert {entry["id"] for entry in document["benchmarks"]} >= {"test", "raid_extra"}
    table = markdown_tables(rows, costs, excluded, ["e5-small-lora", "Litmus"])
    assert "e5-small-lora †" in table and "Litmus engine" in table
