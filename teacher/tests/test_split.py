"""The cluster-to-set assignment on synthetic clusters: disjoint by cluster and by doc, moated, balanced."""

from __future__ import annotations

import numpy as np
import pytest

from litmus_teacher.split import (
    FREE,
    HELD_OUT,
    MOAT,
    Rows,
    SplitConfig,
    adjacency,
    assign_holdable,
    assign_rows,
    balance_set,
    balance_train_languages,
    check_split,
    choose_clusters,
    docs_together,
    nearest_clusters,
    resolve_docs,
    round_robin,
    set_summary,
    take,
    thin_length_groups,
    training_sample,
)


def synthetic_rows(k: int = 80, per_cluster: int = 300, seed: int = 0) -> tuple[Rows, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centroids = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (k, 2) on a ring, unit norm
    cluster = np.repeat(np.arange(k), per_cluster)  # (m,)
    m = len(cluster)
    label = rng.integers(0, 3, m)  # (m,)
    doc = rng.integers(0, m // 3, m)  # (m,) about three passages a doc, spread over clusters
    rows = Rows(ids=np.arange(m), label=label, doc=doc, length_bin=rng.integers(0, 6, m),
                format=rng.integers(0, 3, m), language=rng.integers(0, 2, m), cluster=cluster)

    return rows, centroids  # rows of m, (k, 2)


@pytest.fixture(scope="module")
def assigned() -> tuple[Rows, np.ndarray, np.ndarray, list[np.ndarray], SplitConfig]:
    rows, centroids = synthetic_rows()
    config = SplitConfig(targets={"test": 300, "calibration": 400, "validation": 200, "bank": 200, "bank_draw": 150}, cap=40)
    rng = np.random.default_rng(1)
    linked = adjacency(nearest_clusters(centroids, config.neighbors))
    state = choose_clusters(rows, len(centroids), linked, config, rng)
    split = assign_rows(rows, state, config, rng)

    return rows, state, split, linked, config  # rows of m, (k,), (m,), neighbors, config


def test_invariants_hold(assigned: tuple) -> None:
    rows, state, split, linked, config = assigned
    check_split(split, rows, state, linked, config.balance_keys)  # raises on any broken invariant
    for index, name in enumerate(HELD_OUT):
        assert (state == index).any(), f"{name} received no cluster"
        assert (split == name).sum() > 0


def test_disjoint_by_cluster_and_doc(assigned: tuple) -> None:
    rows, state, split, _, _ = assigned
    names = ("train", *HELD_OUT, "bank_draw")
    for first in names:
        for second in names:
            if first < second:
                assert not np.intersect1d(rows.doc[split == first], rows.doc[split == second]).size
    train_clusters = set(rows.cluster[split == "train"].tolist())
    for name in HELD_OUT:
        assert not train_clusters & set(rows.cluster[split == name].tolist())


def test_moat_separates_held_out(assigned: tuple) -> None:
    rows, state, split, linked, _ = assigned
    for cluster in np.flatnonzero(state >= 0):
        for other in linked[cluster]:
            assert state[other] in (MOAT, state[cluster])
    assert not np.isin(split[np.isin(rows.cluster, np.flatnonzero(state == MOAT))], ("train", *HELD_OUT)).any()
    assert (state == FREE).any()


def test_held_out_sets_balanced_and_capped(assigned: tuple) -> None:
    rows, _, split, _, config = assigned
    for name in (*HELD_OUT, "bank_draw"):
        members = split == name
        for value in np.unique(rows.length_bin[members]):
            counts = np.bincount(rows.label[members & (rows.length_bin == value)], minlength=3)
            assert counts[0] == counts[1] and counts[2] <= counts[0]
        mixed = int((rows.label[members] == 2).sum())
        assert 0 < mixed <= 0.25 * config.targets[name]
    for name in HELD_OUT:
        members = split == name
        per_cell = np.bincount(rows.cluster[members] * 3 + rows.label[members])
        assert per_cell.max() <= config.cap


def test_resolve_docs_priority() -> None:
    split = np.array(["validation", "test", "bank", "calibration", "train"], dtype=object)
    doc = np.array([7, 7, 8, 8, 9])
    resolved = resolve_docs(split, doc)
    assert resolved.tolist() == ["unused", "test", "unused", "calibration", "train"]


def test_round_robin_spreads_groups() -> None:
    rng = np.random.default_rng(0)
    candidates = np.arange(100)
    groups = np.r_[np.zeros(90, dtype=np.int64), np.ones(5, dtype=np.int64), np.full(5, 2)]
    kept = round_robin(candidates, groups, 15, rng)
    assert len(kept) == 15 and len(set(kept.tolist())) == 15
    assert np.bincount(groups[kept], minlength=3).tolist() == [5, 5, 5]


def test_balance_set_equalizes_origins() -> None:
    rows, _ = synthetic_rows(k=4, per_cluster=100)
    split = np.full(len(rows), "test", dtype=object)
    balanced = balance_set(split, "test", rows, ("length_bin",), np.random.default_rng(0))
    for value in range(6):
        counts = np.bincount(rows.label[(balanced == "test") & (rows.length_bin == value)], minlength=3)
        assert counts[0] == counts[1] and counts[2] <= counts[0]
    assert (rows.label[balanced == "test"] == 2).sum() <= 0.25 * (rows.label[balanced == "test"] == 0).sum()


def test_scarce_mixed_text_does_not_empty_a_cell() -> None:
    """A length bin with no mixed text keeps its human and AI rows, balanced with each other."""
    rows, _ = synthetic_rows(k=4, per_cluster=100)
    long_bin = rows.length_bin == 5
    label = np.where(long_bin & (rows.label == 2), 0, rows.label)  # no mixed text in the longest bin
    rows = Rows(ids=rows.ids, label=label, doc=rows.doc, length_bin=rows.length_bin, format=rows.format,
                language=rows.language, cluster=rows.cluster)
    split = np.full(len(rows), "test", dtype=object)
    balanced = balance_set(split, "test", rows, ("length_bin",), np.random.default_rng(0))
    counts = np.bincount(rows.label[(balanced == "test") & long_bin], minlength=3)
    assert counts[0] == counts[1] > 0 and counts[2] == 0


def test_set_summary_counts(assigned: tuple) -> None:
    rows, _, split, _, _ = assigned
    names = {"format": np.array(["code", "plain", "chat"]), "language": np.array(["en", "fr"])}
    summary = set_summary(split, rows, names)
    for name, entry in summary.items():
        members = split == name
        assert entry["rows"] == int(members.sum())
        assert entry["origins"]["ai"] == int((members & (rows.label == 1)).sum())
        assert entry["format"]["chat"][2] == int((members & (rows.format == 2) & (rows.label == 2)).sum())


def test_bank_draw_respects_target(assigned: tuple) -> None:
    rows, _, split, _, config = assigned
    counts = np.bincount(rows.label[split == "bank_draw"], minlength=3)
    assert counts[0] == counts[1] <= config.targets["bank_draw"]
    assert 0 < counts[2] <= config.mixed_share * config.targets["bank_draw"]


def test_a_doc_moves_whole_to_its_most_common_cluster() -> None:
    # A RAID-shaped group: one source document's generations and attacks share a doc but embed apart.
    cluster = np.array([4, 4, 7, 7, 7, 2, 9, 9, 1])
    doc = np.array([10, 10, 10, 10, 10, 11, 12, 12, 13])

    together, moved = docs_together(cluster, doc)

    assert together.tolist() == [7, 7, 7, 7, 7, 2, 9, 9, 1]
    assert moved == {"docs": 4, "docs_in_several_clusters": 1, "rows_moved": 2}
    tie, _ = docs_together(np.array([5, 3]), np.array([1, 1]))
    assert tie.tolist() == [3, 3]


def test_no_held_out_bank_and_a_mixed_only_bank_draw() -> None:
    # The second build's split: no held-out-cluster bank, 120 mixed rows reserved, human and AI bank texts left to export.
    rows, centroids = synthetic_rows()
    config = SplitConfig(targets={"test": 300, "calibration": 400, "validation": 200, "bank": 0, "bank_draw": 120}, cap=40,
                         bank_draw_origins=(2,))
    rng = np.random.default_rng(1)
    linked = adjacency(nearest_clusters(centroids, config.neighbors))

    state = choose_clusters(rows, len(centroids), linked, config, rng)
    split = assign_rows(rows, state, config, rng)

    check_split(split, rows, state, linked, config.balance_keys, config.bank_draw_origins)
    assert not (state == HELD_OUT.index("bank")).any() and not (split == "bank").any()
    assert np.bincount(rows.label[split == "bank_draw"], minlength=3).tolist() == [0, 0, 120]
    assert len(np.unique(rows.cluster[split == "bank_draw"])) > 20
    thin = thin_length_groups(split, rows, floor=40)
    assert "calibration" in thin and all(min(counts) < 40 for counts in thin["calibration"].values())


def test_rows_of_other_languages_only_train() -> None:
    # Language 1 may not be held out: its rows train, or are unused where their cluster or doc is held.
    rows, centroids = synthetic_rows()
    config = SplitConfig(targets={"test": 300, "calibration": 400, "validation": 200, "bank": 0, "bank_draw": 120}, cap=40,
                         bank_draw_origins=(2,))
    linked = adjacency(nearest_clusters(centroids, config.neighbors))
    holdable = rows.language == 0
    rng = np.random.default_rng(1)
    state = choose_clusters(take(rows, holdable), len(centroids), linked, config, rng)
    split = assign_holdable(rows, holdable, state, config, rng)

    check_split(split, rows, state, linked, config.balance_keys, config.bank_draw_origins)
    others = ~holdable
    assert set(split[others]) == {"train", "unused"} and (split[others] == "train").sum() > 1000
    assert (split[others & (state[rows.cluster] != FREE)] == "unused").all()
    held_docs = np.unique(rows.doc[np.isin(split, (*HELD_OUT, "bank_draw"))])
    assert not np.isin(rows.doc[split == "train"], held_docs).any()
    assert np.bincount(rows.label[split == "test"], minlength=3)[:2].min() > 100

    # With every row holdable the split is `assign_rows`'s, draw for draw.
    every = np.ones(len(rows), dtype=bool)
    first, second = np.random.default_rng(2), np.random.default_rng(2)
    state = choose_clusters(rows, len(centroids), linked, config, np.random.default_rng(3))
    assert (assign_holdable(rows, every, state, config, first) == assign_rows(rows, state, config, second)).all()


def test_training_sample_binds_only_above_its_cap() -> None:
    assert training_sample(1000, k=10, max_points_per_centroid=None, seed=0) is None
    assert training_sample(1000, k=10, max_points_per_centroid=100, seed=0) is None
    sample = training_sample(1000, k=10, max_points_per_centroid=20, seed=0)
    assert sample is not None and len(sample) == 200 and len(np.unique(sample)) == 200
    assert (np.diff(sample) > 0).all() and sample.min() >= 0 and sample.max() < 1000
    assert sample.tolist() == training_sample(1000, k=10, max_points_per_centroid=20, seed=0).tolist()
    assert sample.tolist() != training_sample(1000, k=10, max_points_per_centroid=20, seed=1).tolist()


def test_a_language_one_origin_holds_is_cut_in_training() -> None:
    # Language 0 is level, language 1 has 10 human rows and 400 AI, language 2 has 300 human rows and no AI.
    label = np.r_[np.zeros(500), np.ones(500), np.zeros(10), np.ones(400), np.zeros(300), np.full(50, 2)].astype(np.int64)
    language = np.r_[np.zeros(1000), np.ones(410), np.full(300, 2), np.ones(50)].astype(np.int64)
    m = len(label)
    rng = np.random.default_rng(0)
    rows = Rows(ids=np.arange(m), label=label, doc=np.arange(m), length_bin=rng.integers(0, 6, m),
                format=np.zeros(m, dtype=np.int64), language=language, cluster=np.zeros(m, dtype=np.int64))
    split = np.full(m, "train", dtype=object)
    split[:40] = "test"  # held-out rows are neither counted nor cut

    balanced, cut = balance_train_languages(split, rows, 1.5, rng)

    assert cut == {1: [10, 400, 10, 15], 2: [300, 0, 0, 0]}
    training = balanced == "train"
    assert np.bincount(label[training & (language == 0)]).tolist() == [460, 500]
    assert np.bincount(label[training & (language == 1)], minlength=3).tolist() == [10, 15, 50]
    assert not (training & (language == 2)).any()
    assert (balanced[:40] == "test").all() and set(balanced[~training][40:]) == {"unused"}
    kept_bins = np.bincount(rows.length_bin[training & (language == 1) & (label == 1)], minlength=6)
    assert kept_bins.max() - kept_bins.min() <= 1
