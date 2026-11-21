"""Calibration on synthetic scores: coverage holds on exchangeable data and visibly fails under a shift."""

from __future__ import annotations

import json
import numpy as np
import pytest

from pathlib import Path
from litmus_teacher.calibrate import (
    GRID,
    Readings,
    Verdicts,
    _held_share,
    apply_calibration,
    chunk_uncertainty,
    combine_chunks,
    fit_between_sources,
    fit_calibration,
    grid_p_values,
    grid_thresholds,
    isotonic,
    load_table,
    merge_bins,
    merge_p_values,
    readings_from_pair_logits,
    save_table,
    section_uncertainty,
    source_shifts,
    uncertainty_rows,
    vector_cosines,
    write_rows,
)
from litmus_teacher.strata import HUMAN, length_bins


def synthetic(n: int, seed: int, human_shift: float = 0.0, closeness_shift: float = 0.0) -> tuple[Readings, np.ndarray]:
    """Readings whose evidence grows with length: a short text's three means are nearly equal."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)  # (n,)
    tokens = np.exp(rng.uniform(np.log(8), np.log(20_000), n)).astype(np.int64)  # (n,)
    strength = np.clip(np.log(tokens) / np.log(20_000), 0.05, 1.0)  # (n,)
    logits = rng.normal(0.0, 1.0, (n, 3)) + 2.5 * strength[:, None] * np.eye(3)[labels]  # (n, 3)
    logits[:, 1] += human_shift * (labels == HUMAN)  # human text drifting toward the ai bank
    mean = 1 / (1 + np.exp(-logits))  # (n, 3)
    spread = rng.uniform(0.05, 0.3, (n, 3))  # (n, 3)
    closeness = rng.normal(0.6, 0.05, n) - closeness_shift  # (n,)

    return Readings(mean, spread, closeness, tokens), labels  # readings of n texts, (n,)


@pytest.fixture(scope="module")
def table() -> dict:
    return fit_calibration(*synthetic(30_000, seed=1))


def test_coverage_holds_on_exchangeable_data(table: dict) -> None:
    readings, labels = synthetic(30_000, seed=2)
    verdicts = apply_calibration(table, readings)
    for level in (0.05, 0.10):
        held = verdicts.sets[level][np.arange(len(labels)), labels]  # (n,)
        for origin in range(3):
            assert held[labels == origin].mean() >= 1 - level - 0.01
        human = labels == HUMAN
        assert verdicts.not_human(level)[human].mean() <= level + 0.01
        for group in np.unique(verdicts.length_bin):  # the bound holds per length bin too, within sampling error
            members = human & (verdicts.length_bin == group)
            if members.sum() >= 500:
                assert verdicts.not_human(level)[members].mean() <= level + 0.03


def test_coverage_fails_under_shift(table: dict) -> None:
    readings, labels = synthetic(30_000, seed=3, human_shift=3.0)
    verdicts = apply_calibration(table, readings)
    human = labels == HUMAN
    assert verdicts.not_human(0.05)[human].mean() > 0.15  # three times the 5% bound the exchangeable case keeps
    held = verdicts.sets[0.05][np.arange(len(labels)), labels]
    assert held[human].mean() < 0.85


def test_short_texts_come_back_uncertain(table: dict) -> None:
    readings, labels = synthetic(30_000, seed=4)
    verdicts = apply_calibration(table, readings)
    short, long = readings.tokens < 32, readings.tokens > 4096
    sizes = verdicts.sets[0.05].sum(1)  # (n,)
    assert sizes[short].mean() > sizes[long].mean() + 0.5
    assert np.abs(verdicts.probability[short] - 2 / 3).mean() < np.abs(verdicts.probability[long] - 2 / 3).mean()


def test_out_of_reference_flag(table: dict) -> None:
    same = apply_calibration(table, synthetic(20_000, seed=5)[0])
    shifted = apply_calibration(table, synthetic(20_000, seed=5, closeness_shift=0.2)[0])
    assert same.out_of_reference.mean() <= 0.02
    assert shifted.out_of_reference.mean() > 0.9


def test_venn_abers_interval_and_calibration(table: dict) -> None:
    readings, labels = synthetic(30_000, seed=6)
    verdicts = apply_calibration(table, readings)
    low, high = verdicts.interval[:, 0], verdicts.interval[:, 1]
    assert (low <= verdicts.probability + 1e-12).all() and (verdicts.probability <= high + 1e-12).all()
    positive = labels != HUMAN
    for group in np.unique(verdicts.length_bin):
        members = verdicts.length_bin == group
        if members.sum() >= 1000:
            assert abs(verdicts.probability[members].mean() - positive[members].mean()) < 0.03


def test_grid_p_values_exact_at_every_level() -> None:
    rng = np.random.default_rng(7)
    calibration, queries = rng.normal(size=999), rng.normal(size=5000)
    exact = (1 + (calibration[None, :] <= queries[:, None]).sum(1)) / (len(calibration) + 1)  # (q,)
    reported = grid_p_values(queries, grid_thresholds(calibration))
    assert (reported >= exact - 1e-12).all()
    for level in GRID:
        np.testing.assert_array_equal(exact <= level, reported <= level)


def test_merge_bins_rule() -> None:
    assert merge_bins(np.array([5, 300, 0, 50, 180, 400]), 200) == [[0, 1], [2, 3, 4], [5]]
    assert merge_bins(np.array([300, 10]), 200) == [[0, 1]]
    assert merge_bins(np.array([[250, 10], [250, 300]]), 200) == [[0, 1]]
    with pytest.raises(ValueError):
        merge_bins(np.array([50, 60]), 200)


def test_isotonic_is_nondecreasing_and_mass_preserving() -> None:
    values, weights = np.array([0.5, 0.2, 0.9, 0.4, 1.0]), np.array([1.0, 3.0, 1.0, 1.0, 2.0])
    fitted = isotonic(values, weights)
    assert (np.diff(fitted) >= 0).all()
    assert np.isclose((fitted * weights).sum(), (values * weights).sum())


def test_table_round_trip_is_small(table: dict, tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    save_table(table, path)
    assert path.stat().st_size < 64 * 1024
    readings, _ = synthetic(2000, seed=8)
    first, second = apply_calibration(table, readings), apply_calibration(load_table(path), readings)
    np.testing.assert_array_equal(first.p_values, second.p_values)
    np.testing.assert_array_equal(first.probability, second.probability)
    groups = [entry["bins"] for entry in table["conformal"]["human"]]
    assert sorted(b for group in groups for b in group) == list(range(len(table["length_edges"])))
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "litmus-calibration"


def test_section_uncertainty_flags_conflict(table: dict) -> None:
    human_like = [0.97, 0.02, 0.02]
    ai_like = [0.02, 0.97, 0.02]
    sections = Readings(np.array([human_like, ai_like]), np.full((2, 3), 0.1), np.full(2, 0.6), np.array([3000, 3000]))
    summary = section_uncertainty(table, sections)
    assert summary.conflict and summary.called_not_human == 1 and summary.disagreement > 0.3
    agree = Readings(np.array([ai_like, ai_like]), np.full((2, 3), 0.1), np.full(2, 0.6), np.array([3000, 3000]))
    assert not section_uncertainty(table, agree).conflict


def test_merged_p_values_are_valid_under_dependence() -> None:
    rng = np.random.default_rng(11)
    uniform = rng.random((200_000, 1))  # (n, 1)
    for chunks in (2, 4, 16):
        identical = np.repeat(uniform, chunks, axis=1)  # (n, C), perfectly dependent
        noisy = np.clip(identical + rng.normal(0.0, 0.05, identical.shape), 0.0, 1.0)  # (n, C), strongly dependent
        independent = rng.random((200_000, chunks))  # (n, C)
        for p in (identical, noisy, independent):
            merged = np.array([merge_p_values(row[:, None])[0] for row in p[:20_000]])  # (m,)
            for level in (0.05, 0.10):
                assert (merged <= level).mean() <= level + 0.005
    np.testing.assert_array_equal(merge_p_values(np.array([[0.03, 0.5, 1.0]])), [0.03, 0.5, 1.0])
    np.testing.assert_allclose(merge_p_values(np.array([[0.02, 0.9], [0.04, 0.8]])), [0.06, 1.0])


def test_one_chunk_is_its_own_verdict(table: dict) -> None:
    readings, _ = synthetic(50, seed=12)
    for row in range(50):
        chunk = readings.subset(np.array([row]))
        verdict = apply_calibration(table, chunk)
        combined = chunk_uncertainty(table, chunk)
        np.testing.assert_array_equal(combined.p_values, verdict.p_values[0])
        assert combined.probability == verdict.probability[0] and combined.disagreement == 0.0
        np.testing.assert_array_equal(combined.interval, verdict.interval[0])
        np.testing.assert_array_equal(combined.three_way, verdict.three_way[0])
        assert combined.out_of_reference == verdict.out_of_reference[0] and not combined.mixed_document


def test_chunks_combine_with_an_interval_and_a_mixed_signal(table: dict) -> None:
    human_like = [0.97, 0.02, 0.02]
    ai_like = [0.02, 0.97, 0.02]
    split = Readings(np.array([human_like, human_like, ai_like]), np.full((3, 3), 0.1), np.full(3, 0.6), np.full(3, 6000))
    combined = chunk_uncertainty(table, split)
    assert combined.mixed_document and combined.model_chunks == 1 and combined.human_chunks == 2
    assert combined.interval[0] <= max(0.0, combined.probability - combined.disagreement) + 1e-12
    assert combined.interval[1] >= min(1.0, combined.probability + combined.disagreement) - 1e-12
    agree = Readings(np.array([ai_like] * 3), np.full((3, 3), 0.1), np.full(3, 0.6), np.full(3, 6000))
    together = chunk_uncertainty(table, agree)
    assert not together.mixed_document and together.disagreement < 1e-12 and together.not_human(0.05)
    assert together.interval[1] - together.interval[0] < combined.interval[1] - combined.interval[0]


def test_mixed_signal_and_merged_calls_hold_their_level_on_whole_documents(table: dict) -> None:
    """Documents of one origin whose chunks share most of their evidence: the merged call and the signal keep the level."""
    rng = np.random.default_rng(13)
    documents, chunks = 3000, 4
    for origin in (HUMAN, 1):
        tokens = np.full(documents * chunks, 6000)  # (n,)
        shared = rng.normal(0.0, 1.0, (documents, 1, 3))  # (D, 1, 3) the document's own evidence
        strength = np.log(6000) / np.log(20_000)  # as `synthetic` draws a text of this length, so each chunk is exchangeable with it
        logits = 0.8 * shared + 0.6 * rng.normal(0.0, 1.0, (documents, chunks, 3)) + 2.5 * strength * np.eye(3)[origin]  # (D, C, 3)
        mean = (1 / (1 + np.exp(-logits))).reshape(-1, 3)  # (n, 3)
        verdicts = apply_calibration(table, Readings(mean, np.full((len(tokens), 3), 0.1), np.full(len(tokens), 0.6), tokens))
        fired, called = [], []
        for document in range(documents):
            rows = np.arange(document * chunks, (document + 1) * chunks)
            part = Verdicts(verdicts.length_bin[rows], verdicts.p_values[rows], {key: value[rows] for key, value in verdicts.sets.items()},
                            verdicts.probability[rows], verdicts.interval[rows], verdicts.three_way[rows],
                            verdicts.spread_percentile[rows], verdicts.closeness_p_value[rows], verdicts.out_of_reference[rows])
            combined = combine_chunks(table, part)
            fired.append(combined.mixed_document)
            called.append(not combined.sets[0.05][origin])
        assert np.mean(fired) <= 0.05 + 0.015 and np.mean(called) <= 0.05 + 0.015


def test_readings_from_pair_logits_definition() -> None:
    rng = np.random.default_rng(9)
    logits = [rng.normal(size=(4, size)) for size in (5, 6, 7)]
    vectors, bank = rng.normal(size=(4, 8)), rng.normal(size=(18, 8))
    readings = readings_from_pair_logits(logits, vector_cosines(vectors, bank), np.array([10, 20, 30, 40]), top_k=3)
    probabilities = 1 / (1 + np.exp(-logits[1]))
    np.testing.assert_allclose(readings.mean[:, 1], probabilities.mean(1))
    np.testing.assert_allclose(readings.spread[:, 1], probabilities.std(1))
    cosines = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)) @ (bank / np.linalg.norm(bank, axis=1, keepdims=True)).T
    np.testing.assert_allclose(readings.closeness, np.sort(cosines, axis=1)[:, -3:].mean(1))


def test_uncertainty_rows_tidy_csv(table: dict, tmp_path: Path) -> None:
    readings, labels = synthetic(5000, seed=10)
    verdicts = apply_calibration(table, readings)
    formats = np.where(np.arange(5000) % 2, "plain", "code").astype(object)
    rows = uncertainty_rows(verdicts, labels, {"format": formats}, "test")
    metrics = {row["metric"] for row in rows}
    assert {"coverage", "mean_set_size", "human_false_positive_rate", "expected_calibration_error",
            "reliability_mean_probability", "reliability_observed_frequency"} <= metrics
    assert {row["stratum_kind"] for row in rows} == {"all", "length_bin", "format"}
    write_rows(rows, tmp_path / "uncertainty.csv")
    lines = (tmp_path / "uncertainty.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(rows) + 1 and lines[0].startswith("split,stratum_kind,stratum,metric")


def test_length_bins_edges() -> None:
    assert length_bins(np.array([0, 15, 16, 8191, 8192, 32768, 10**6])).tolist() == [0, 0, 1, 9, 10, 11, 11]


def with_sources(n: int, seed: int, spread: float, sources: int = 24) -> tuple[Readings, np.ndarray, np.ndarray]:
    """Readings as `synthetic` gives them, each from one of `sources` sources; a source moves its texts' evidence
    toward human or away from it by its own amount, drawn with standard deviation `spread`."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)  # (n,)
    source = rng.integers(0, sources, n)  # (n,)
    tokens = np.exp(rng.uniform(np.log(8), np.log(20_000), n)).astype(np.int64)  # (n,)
    strength = np.clip(np.log(tokens) / np.log(20_000), 0.05, 1.0)  # (n,)
    logits = rng.normal(0.0, 1.0, (n, 3)) + 2.5 * strength[:, None] * np.eye(3)[labels]  # (n, 3)
    logits[:, HUMAN] += rng.normal(0.0, spread, sources)[source]
    mean = 1 / (1 + np.exp(-logits))  # (n, 3)
    names = np.array([f"source {index}" for index in source])  # (n,)

    return Readings(mean, rng.uniform(0.05, 0.3, (n, 3)), rng.normal(0.6, 0.05, n), tokens), labels, names  # readings of n texts, (n,) each


def test_source_shifts_definition() -> None:
    # One length group, two regions. Source a has 30 texts in region 0 and 10 in region 1, source b 30 and 30, and
    # source c is too small to say anything.
    region = np.array([0] * 30 + [1] * 10 + [0] * 30 + [1] * 30 + [0] * 5)
    sources = np.array(["a"] * 40 + ["b"] * 60 + ["c"] * 5)
    shift, variance, usable, names = source_shifts(region, np.zeros(105, dtype=np.int64), sources, 2, 1)
    assert names == ["a", "b"]
    # Against a, the others (b and c) put 35 of 65 texts in region 0, so 40 texts are expected as 21.54 and 18.46.
    expected = 40 * np.array([35, 30]) / 65
    np.testing.assert_allclose(shift[0], np.log((np.array([30, 10]) + 0.5) / (expected + 0.5)))
    np.testing.assert_allclose(variance[0], 1 / (np.array([30, 10]) + 0.5))
    assert usable.all()
    # A source the others' texts put under 5 texts in a region says nothing there.
    region = np.array([0] * 39 + [1] * 1 + [0] * 59 + [1] * 1)
    _, _, usable, _ = source_shifts(region, np.zeros(100, dtype=np.int64), np.array(["a"] * 40 + ["b"] * 60), 2, 1)
    assert usable.tolist() == [[True, False], [True, False]]


def test_sources_that_read_alike_need_almost_no_range_and_shifted_ones_a_wide_one() -> None:
    def between(spread: float) -> dict:
        readings, labels, sources = with_sources(40_000, seed=11, spread=spread)
        return fit_calibration(readings, labels, sources=sources)["between_sources"]

    # Measured at seed 11: spread 0.0 gives 0.042, 0.25 gives 0.517, 0.5 gives 0.884, 1.0 gives 1.587, and 2.0 gives
    # 1.700 (the range saturates once the shifts are large), so the steps compared are the ones before saturation.
    alike, shifted, wider = between(0.0), between(0.5), between(1.0)
    assert alike["sources"] == {"human": 24, "not_human": 24} and alike["level"] == 0.9
    assert 0.0 <= alike["log_odds"] < 0.35, alike["log_odds"]
    assert shifted["log_odds"] > alike["log_odds"] + 0.5 and wider["log_odds"] > shifted["log_odds"] + 0.3
    assert len(shifted["regions"]) == 7 and abs(sum(region["weight"] for region in shifted["regions"]) - 1.0) < 1e-6


def test_the_half_width_holds_its_level_of_pairings() -> None:
    readings, labels, sources = with_sources(40_000, seed=12, spread=1.0)
    table = fit_calibration(readings, labels)
    for level in (0.8, 0.9, 0.95):
        entry = fit_between_sources(table, readings, labels, sources, level=level)
        centers, spreads, weights = (np.array([region[key] for region in entry["regions"]]) for key in ("center", "spread", "weight"))
        assert abs(_held_share(entry["log_odds"], centers, spreads, weights) - level) < 1e-4
    assert fit_between_sources(table, readings, labels, sources, level=0.95)["log_odds"] > fit_between_sources(table, readings, labels, sources, level=0.8)["log_odds"]


def test_the_table_carries_the_range_only_when_sources_are_given() -> None:
    readings, labels, sources = with_sources(30_000, seed=13, spread=0.5)
    plain, sourced = fit_calibration(readings, labels), fit_calibration(readings, labels, sources=sources)
    assert "between_sources" not in plain
    entry = sourced.pop("between_sources")
    assert sourced == plain, "the rest of the table is the same fit"
    assert set(entry) == {"level", "log_odds", "sources", "min_texts", "min_expected", "regions"}
    with pytest.raises(ValueError, match="one label and one source per reading"):
        fit_between_sources(plain, readings, labels, sources[:-1])
    # Seven sources a side are too few to say how sources differ.
    few = np.array([f"source {index % 7}" for index in range(len(labels))])
    with pytest.raises(ValueError, match="too few human sources"):
        fit_between_sources(plain, readings, labels, few)
