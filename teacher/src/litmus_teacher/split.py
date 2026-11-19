"""Cut the corpus into train, validation, test, bank and calibration sets of whole e-means clusters.

**Clustering.** Spherical e-means (Bessa et al. 2026, arXiv 2611.09501), the FAISS 1.16.0 reference
(`faiss.contrib.e_means.Emeans`, MIT), on the GPU. Cosine geometry: each encoder's mean half is centered on the
eligible rows' mean and L2-normalized, encoders are concatenated with weight `1 / sqrt(#encoders)` so the joint vector
has norm 1, and `Emeans(spherical=True)` keeps its centroids on the unit sphere (each use rescales the running
average to norm 1, and its closing step rescales each cluster's exact mean), so the highest dot product is the highest
cosine. Final labels come from one `assign` pass against the final centroids, not the moving last-pass labels.
With `--max-points-per-centroid` e-means trains on a seeded random subset of that many rows a cluster, as
`faiss.Kmeans` caps its training set, and the features stay in host memory: every row is still assigned.

**k.** About `--rows-per-cluster` (2,500) eligible rows a cluster, half the first corpus's 5,050 (5.17M rows in
1,024). Four held-out sets instead of two, each drawn from a hundred or more clusters under a per-cluster cap, need
finer regions; at 2,500 rows a cluster still holds enough text that a moat of its two nearest clusters is a real
distance, which the distance report checks.

**Sets.** Clusters are given to held-out sets greedily: each round serves the set furthest from its per-origin
target, scoring a free cluster by the share of the target it fills (at most `--cap` rows per origin), times a bonus
for formats and languages the set lacks, divided by one plus the free clusters it would put in the moat, and picks
at random among the eight best. The two nearest clusters of every held-out cluster (and every cluster that counts it
among its two nearest) go to the moat (`unused`), unless they join the same set. A cluster next to another set's
cluster is never a candidate, so held-out sets are separated from training and from each other by a moat.
Unpicked clusters are training clusters.

Then, in order: a `doc` in several held-out sets stays only in the first of test, calibration, validation, bank;
each held-out cluster keeps at most `--cap` rows per origin, chosen round-robin over length bins; each held-out set is
balanced by cutting human and AI text to the smaller of the two in each `--balance-keys` cell, keeping rows round-robin
over format and language. Mixed text does not bound them: it is scarce and short (the corpus of 2026-11-17 holds 111
mixed rows over 1,024 tokens against 168,748 of each other origin), so a cell balanced on all three origins would
leave the held-out sets no long text. A cell keeps at most as many mixed rows as human ones, a set at most
`--mixed-share` (0.25) of its per-origin target, and the clusters are chosen toward that smaller mixed target. Bank variant (b), `bank_draw`, takes the same number of rows per origin round-robin from every training
cluster, balanced the same way; with `--bank-draw-origins mixed` it takes that many mixed rows alone, round-robin over
length bins, for a bank whose human and AI texts come from training rows at export. A held-out set whose target is 0
(`--targets bank=0`) takes no cluster. Every training row whose `doc` has a row in any held-out set or in `bank_draw` goes to
`unused`. Rows with `benchmark` set are `benchmark` and enter nothing.

**Languages in the held-out sets.** A language is a region of the embedding, so a held-out cluster of a rare language
and its moat take most of that language: the first split of the corpus of 2026-11-18 left training 1,097 of 22,946
German rows and gave test and calibration 11% and 12% text in languages training barely held. With
`--held-out-languages en und`, clusters are chosen and sets are filled from those languages' rows alone; a row of
another language trains, unless its cluster is held out or in the moat or its doc is held, when it is `unused`.

**Languages in training.** The corpus balances human and AI text in each language group, and every rare language
shares one group, so a rare language can be nearly all one origin (the corpus of 2026-11-18 trains on 21 human and
1,813 AI Czech texts), which teaches the language as the origin. With `--train-language-ratio r`, the larger origin
of each language in training is cut to `r` times the smaller, and the rows cut are `unused`.
"""

from __future__ import annotations

import argparse
import math
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embed import EMBEDDINGS_FILE, TOKENS_FILE, dense_row_count, read_columns
from .strata import LENGTH_EDGES, ORIGINS, length_bin_names, length_bins, write_json, write_summary


HELD_OUT = ("test", "calibration", "validation", "bank")  # also the doc priority, first wins
SPLIT_NAMES = ("train", *HELD_OUT, "bank_draw", "unused", "benchmark", "unembedded")
DEFAULT_TARGETS = {"test": 20_000, "calibration": 30_000, "validation": 10_000, "bank": 16_384, "bank_draw": 16_384}
FREE, MOAT = -1, -2
NEIGHBORS = 2
FEATURE_CHUNK_ROWS = 1 << 18  # a chunk's float32, centered, normalized and float64 copies together stay under 4 GiB
ASSIGN_CHUNK_ROWS = 1 << 20
OVERSHOOT = 1.25  # clusters are chosen toward 1.25 times the target, since capping and balancing cut rows
TOP_CANDIDATES = 8
CELL_FLOOR = 200  # a (format, language) cell under this many rows in a set counts as missing from it
LENGTH_GROUP_FLOOR = 200  # calibration texts of each origin a length group needs (reported, not enforced)
NOVELTY_WEIGHT = 0.5
MIXED = ORIGINS.index("mixed")
MIXED_SHARE = 0.25
DISTANCE_TRAIN_SAMPLE = 400_000
DISTANCE_QUERY_SAMPLE = 4_000
DISTANCE_NEIGHBORS = 8


@dataclass(frozen=True)
class Rows:
    """The eligible rows' attributes, aligned: every array is (m,), integer codes for the categories."""

    ids: np.ndarray
    label: np.ndarray
    doc: np.ndarray
    length_bin: np.ndarray
    format: np.ndarray
    language: np.ndarray
    cluster: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class SplitConfig:
    targets: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TARGETS))  # rows per origin
    cap: int = 500  # rows per held-out cluster and origin
    balance_keys: tuple[str, ...] = ("length_bin",)
    neighbors: int = NEIGHBORS
    seed: int = 0
    mixed_share: float = MIXED_SHARE  # a held-out set's mixed rows as a share of its per-origin target
    bank_draw_origins: tuple[int, ...] = (0, 1, 2)  # labels `bank_draw` takes; fewer than all: `target` rows of each, unbalanced


def share_of(amount: np.ndarray, target: np.ndarray) -> np.ndarray:
    """`amount / target`, 0 where the target is 0 (a set, or a set's origin, not wanted)."""
    return np.divide(amount, target, out=np.zeros(np.broadcast_shapes(amount.shape, target.shape)), where=target > 0)


def codes(column: pa.ChunkedArray) -> np.ndarray:
    """Integer codes of a string column, nulls as the empty string: (n,) to (n,) int64."""
    values = pc.fill_null(column, "").to_numpy(zero_copy_only=False)  # (n,)

    return np.unique(values, return_inverse=True)[1].astype(np.int64)  # (n,)


def cluster_features(embedding_dirs: list[Path], ids: np.ndarray, device: torch.device,
                     store: torch.device | None = None) -> torch.Tensor:
    """Each encoder's mean half centered, unit-normalized, weighted by 1/sqrt(#encoders) and concatenated: (m, f) bfloat16.

    Computed on `device` a chunk at a time and kept on `store` (`device` when None): 14.8 million rows of 768 are
    21 GiB, more than a 24 GB GPU holds beside e-means, so a sampled clustering keeps them in host memory.
    """
    # ids: (m,)
    store = device if store is None else store
    parts = []
    for directory in embedding_dirs:
        values = np.load(directory / EMBEDDINGS_FILE, mmap_mode="r")  # (n, 2d)
        width = values.shape[1] // 2  # d

        def chunks() -> Iterator[tuple[int, torch.Tensor]]:
            """The mean half in float32, `FEATURE_CHUNK_ROWS` rows at a time: 10 million rows at once, and their
            float64 copy for the mean, do not fit an 80 GB GPU."""
            for start in range(0, len(ids), FEATURE_CHUNK_ROWS):
                rows = np.ascontiguousarray(values[ids[start : start + FEATURE_CHUNK_ROWS], :width])  # (c, d)
                yield start, torch.from_numpy(rows).to(device).float()

        total = torch.zeros(width, dtype=torch.float64, device=device)  # (d,)
        for _, chunk in chunks():
            total += chunk.double().sum(0)
        center = (total / len(ids)).float()  # (d,)
        unit = torch.empty(len(ids), width, dtype=torch.bfloat16, device=store)  # (m, d)
        for start, chunk in chunks():
            rows = (F.normalize(chunk - center, dim=1) / math.sqrt(len(embedding_dirs))).to(torch.bfloat16)  # (c, d)
            assert torch.isfinite(rows).all(), "e-means does not check GPU input for NaN or infinity"
            unit[start : start + len(chunk)] = rows.to(store)
        parts.append(unit)
    features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)  # (m, f), f the summed widths

    return features  # (m, f)


def training_sample(rows: int, k: int, max_points_per_centroid: int | None, seed: int) -> np.ndarray | None:
    """The sorted rows e-means trains on when `max_points_per_centroid * k` is fewer than all, else None: (s,)."""
    if max_points_per_centroid is None or rows <= max_points_per_centroid * k:
        return None

    return np.sort(np.random.default_rng(seed).choice(rows, max_points_per_centroid * k, replace=False))  # (s,)


def run_emeans(features: torch.Tensor, k: int, seed: int, backend: str, device: torch.device,
               max_points_per_centroid: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Spherical e-means through the FAISS reference: centroids (k, f) on the unit sphere, labels (m,), and its recipe.

    With `max_points_per_centroid` and more rows than that many a centroid, e-means trains on a seeded random subset
    of that size, as `faiss.Kmeans` caps its training set. Every row is then assigned to its nearest final centroid,
    `ASSIGN_CHUNK_ROWS` at a time, so the features may live in host memory.
    """
    from faiss.contrib.e_means import Emeans

    # features: (m, f), on the GPU or in host memory
    model = Emeans(features.shape[1], k, nepochs=5, seed=seed, spherical=True, assignment_backend=backend, verbose=True)
    sample = training_sample(len(features), k, max_points_per_centroid, seed)
    model.train((features if sample is None else features[torch.from_numpy(sample)]).to(device))
    centroids = model.centroids.float()  # (k, f)
    norms = centroids.norm(dim=1)  # (k,)
    assert (((norms - 1).abs() < 1e-3) | (norms == 0)).all(), "spherical centroids are unit vectors, or zero if never used"
    labels = torch.cat([model.assign(features[start : start + ASSIGN_CHUNK_ROWS].to(device))[1]
                        for start in range(0, len(features), ASSIGN_CHUNK_ROWS)])  # (m,), nearest final centroid by cosine
    record = {key: value for key, value in model.recipe.items() if isinstance(value, (int, float, str, bool))}
    record.update({"n_splits": int(model.n_splits), "objective_by_pass": [float(value) for value in model.obj.cpu()],
                   "assign_backend": model.assignment_recipe["assignment_backend"],
                   "training_rows": int(len(features) if sample is None else len(sample)),
                   "max_points_per_centroid": max_points_per_centroid})

    return centroids, labels, record  # (k, f), (m,)


def nearest_clusters(centroids: np.ndarray, count: int) -> np.ndarray:
    """Each cluster's `count` nearest other clusters by centroid cosine: (k, f) to (k, count)."""
    # centroids: (k, f), unit rows
    similarity = centroids @ centroids.T  # (k, k)
    np.fill_diagonal(similarity, -np.inf)

    return np.argsort(-similarity, axis=1)[:, :count]  # (k, count)


def adjacency(neighbors: np.ndarray) -> list[np.ndarray]:
    """The symmetric neighbor relation: a cluster's nearest clusters and the clusters that count it among theirs."""
    # neighbors: (k, count)
    k = len(neighbors)
    linked: list[set[int]] = [set(row.tolist()) for row in neighbors]
    for cluster in range(k):
        for other in neighbors[cluster]:
            linked[int(other)].add(cluster)

    return [np.array(sorted(members), dtype=np.int64) for members in linked]  # (k_c,) per cluster


def docs_together(cluster: np.ndarray, doc: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Every row moved to its doc's most common cluster (the lowest on a tie), so a doc, such as a source group whose
    generations and attacks all derive from one human document, is never divided by clustering: (m,), (m,) to (m,),
    with how many docs spanned several clusters and how many rows moved."""
    # cluster, doc: (m,)
    order = np.lexsort((cluster, doc))  # (m,) by doc, then cluster
    sorted_doc, sorted_cluster = doc[order], cluster[order]  # (m,) each
    pair_start = np.r_[True, (np.diff(sorted_doc) != 0) | (np.diff(sorted_cluster) != 0)]  # (m,)
    pair_index = np.cumsum(pair_start) - 1  # (m,)
    pair_size = np.bincount(pair_index)  # (p,)
    pair_doc, pair_cluster = sorted_doc[pair_start], sorted_cluster[pair_start]  # (p,) each
    by_size = np.lexsort((pair_cluster, -pair_size, pair_doc))  # (p,) per doc: largest pair first, lowest cluster on a tie
    first_of_doc = np.r_[True, np.diff(pair_doc[by_size]) != 0]  # (p,)
    best_doc, best_cluster = pair_doc[by_size][first_of_doc], pair_cluster[by_size][first_of_doc]  # (d,) each, ascending doc
    docs, pairs_per_doc = np.unique(pair_doc, return_counts=True)  # (d,), (d,)
    together = best_cluster[np.searchsorted(best_doc, doc)]  # (m,)

    moved = {"docs": int(len(docs)), "docs_in_several_clusters": int((pairs_per_doc > 1).sum()), "rows_moved": int((together != cluster).sum())}

    return together, moved  # (m,), counts


def choose_clusters(rows: Rows, k: int, linked: list[np.ndarray], config: SplitConfig,
                    rng: np.random.Generator) -> np.ndarray:
    """The greedy cluster assignment: per cluster, the index of its held-out set in HELD_OUT, FREE or MOAT, (k,)."""
    # linked: (k_c,) neighbor ids per cluster
    sets = list(HELD_OUT)
    available = np.zeros((k, len(ORIGINS)), dtype=np.int64)  # (k, 3) rows per cluster and origin
    np.add.at(available, (rows.cluster, rows.label), 1)
    contribution = np.minimum(available, config.cap)  # (k, 3) rows a held-out cluster can give
    cells = rows.format * (rows.language.max() + 1) + rows.language  # (m,)
    cell_codes, cell_index = np.unique(cells, return_inverse=True)
    cell_rows = np.zeros((k, len(cell_codes)), dtype=np.float64)  # (k, c) rows per cluster and (format, language)
    np.add.at(cell_rows, (rows.cluster, cell_index), 1)
    scale = contribution.sum(1) / np.maximum(available.sum(1), 1)  # (k,) share of a cluster's rows it gives
    cell_rows *= scale[:, None]

    targets = np.array([[config.targets[name]] * len(ORIGINS) for name in sets], dtype=np.float64) * OVERSHOOT  # (s, 3)
    targets[:, MIXED] *= config.mixed_share
    filled = np.zeros_like(targets)  # (s, 3)
    cells_filled = np.zeros((len(sets), len(cell_codes)))  # (s, c)
    state = np.full(k, FREE, dtype=np.int64)  # (k,)
    moat_owner = np.zeros((k, len(sets)), dtype=bool)  # (k, s)
    exhausted = np.zeros(len(sets), dtype=bool)  # (s,)
    while True:
        deficit = share_of(np.maximum(targets - filled, 0), targets)  # (s, 3); a set with no target is never open
        open_sets = np.flatnonzero((deficit.sum(1) > 0) & ~exhausted)
        if not len(open_sets):
            break
        chosen_set = int(open_sets[np.argmax(deficit[open_sets].mean(1))])

        others_moat = moat_owner & ~np.eye(len(sets), dtype=bool)[chosen_set]  # (k, s)
        candidate = ((state == FREE) | ((state == MOAT) & ~others_moat.any(1)))  # (k,)
        for cluster in np.flatnonzero(candidate):
            neighbor_state = state[linked[cluster]]
            if ((neighbor_state >= 0) & (neighbor_state != chosen_set)).any():
                candidate[cluster] = False
        gain = share_of(np.minimum(contribution, np.maximum(targets[chosen_set] - filled[chosen_set], 0)),
                        targets[chosen_set]).sum(1)  # (k,)
        candidate &= gain > 0
        if not candidate.any():
            exhausted[chosen_set] = True
            continue
        missing = cells_filled[chosen_set] < CELL_FLOOR  # (c,)
        novelty = (cell_rows[:, missing].sum(1) / np.maximum(cell_rows.sum(1), 1e-9))  # (k,)
        newly_moated = np.array([(state[linked[cluster]] == FREE).sum() for cluster in range(k)])  # (k,)
        score = np.where(candidate, gain * (1 + NOVELTY_WEIGHT * novelty) / (1 + newly_moated), -np.inf)  # (k,)
        best = np.argsort(-score)[:TOP_CANDIDATES]
        best = best[np.isfinite(score[best])]
        picked = int(rng.choice(best))

        state[picked] = chosen_set
        moat_owner[picked] = False
        for other in linked[picked]:
            if state[other] in (FREE, MOAT):
                state[other] = MOAT
                moat_owner[other, chosen_set] = True
        filled[chosen_set] += contribution[picked]
        cells_filled[chosen_set] += cell_rows[picked]

    return state  # (k,)


def round_robin(candidates: np.ndarray, groups: np.ndarray, keep: int, rng: np.random.Generator) -> np.ndarray:
    """`keep` of the candidate rows, taken one per group in turn, at random within a group: (r,), (r,) to (keep,)."""
    # candidates, groups: (r,)
    if keep >= len(candidates):
        return candidates  # (r,)
    shuffle = rng.permutation(len(candidates))  # (r,)
    order = shuffle[np.argsort(groups[shuffle], kind="stable")]  # (r,) grouped, random within a group
    sorted_groups = groups[order]  # (r,)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]  # (g,)
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))  # (r,) position in its group
    turn = np.lexsort((rng.random(len(order)), rank))  # (r,) by rank, ties at random

    return candidates[order[turn[:keep]]]  # (keep,)


def resolve_docs(split: np.ndarray, doc: np.ndarray) -> np.ndarray:
    """A doc in several held-out sets stays in the first by HELD_OUT order; its rows elsewhere become `unused`: (m,) to (m,)."""
    # split, doc: (m,)
    split = split.copy()
    claimed = np.zeros(doc.max() + 1 if len(doc) else 0, dtype=bool)  # (docs,)
    for name in HELD_OUT:
        members = split == name  # (m,)
        clash = members & claimed[doc]
        split[clash] = "unused"
        claimed[doc[members & ~clash]] = True

    return split  # (m,)


def segments(indices: np.ndarray, keys: np.ndarray) -> Iterator[np.ndarray]:
    """The indices grouped by key, one array per distinct key in ascending order, by one sort: (r,), (r,)."""
    # indices, keys: (r,)
    order = np.argsort(keys, kind="stable")  # (r,)
    bounds = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]  # (g + 1,)
    for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
        yield indices[order[begin:end]]  # (r_g,)


def cap_held_out(split: np.ndarray, rows: Rows, cap: int, rng: np.random.Generator) -> np.ndarray:
    """At most `cap` rows per held-out cluster and origin, round-robin over length bins: (m,) to (m,)."""
    # split: (m,)
    split = split.copy()
    held = np.flatnonzero(np.isin(split, HELD_OUT))  # (h,)
    group = rows.cluster[held] * len(ORIGINS) + rows.label[held]  # (h,)
    for members in segments(held, group):
        if len(members) > cap:
            kept = round_robin(members, rows.length_bin[members], cap, rng)
            split[np.setdiff1d(members, kept)] = "unused"

    return split  # (m,)


def balance_set(split: np.ndarray, name: str, rows: Rows, keys: tuple[str, ...], rng: np.random.Generator,
                limit: int | None = None, mixed_share: float = MIXED_SHARE) -> np.ndarray:
    """Cut human and AI text to the smaller of the two in each cell of `keys`, keeping rows round-robin over format
    and language; a cell keeps at most as many mixed rows as human ones.

    With `limit`, every cell is then cut by the same factor so human and AI text hold at most `limit` rows each,
    which keeps the cells balanced. Mixed rows are cut the same way to `mixed_share` of that.
    """
    # split: (m,)
    split = split.copy()
    members = np.flatnonzero(split == name)  # (r,)
    if not len(members):
        return split  # (m,)
    cell = np.zeros(len(members), dtype=np.int64)  # (r,)
    for key in keys:
        values = getattr(rows, key)[members]
        cell = cell * (values.max() + 1) + values
    spread = rows.format[members] * (rows.language.max() + 1) + rows.language[members]  # (r,)
    labels = rows.label[members]  # (r,)
    cells = list(segments(np.arange(len(members)), cell))  # positions into `members`, one array per cell
    counts = np.array([[int((labels[positions] == origin).sum()) for origin in range(len(ORIGINS))]
                       for positions in cells])  # (cells, 3)
    pair = counts[:, :MIXED].min(axis=1)  # (cells,) human and AI rows kept
    if limit is not None and pair.sum() > limit:
        pair = np.floor(pair * limit / pair.sum()).astype(np.int64)
    mixed = np.minimum(counts[:, MIXED], pair)  # (cells,)
    mixed_limit = int(mixed_share * (limit if limit is not None else pair.sum()))
    if mixed.sum() > mixed_limit:
        mixed = np.floor(mixed * mixed_limit / mixed.sum()).astype(np.int64)
    keeps = np.column_stack([pair, pair, mixed])  # (cells, 3)
    for positions, keep in zip(cells, keeps, strict=True):
        for origin in range(len(ORIGINS)):
            chosen = positions[labels[positions] == origin]
            kept = round_robin(chosen, spread[chosen], int(keep[origin]), rng)
            split[members[np.setdiff1d(chosen, kept)]] = "unused"

    return split  # (m,)


def draw_bank(split: np.ndarray, rows: Rows, target: int, keys: tuple[str, ...], rng: np.random.Generator,
              mixed_share: float = MIXED_SHARE, origins: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    """Variant (b): an even share of `target` rows per origin from every training cluster, of docs no held-out set holds.

    With every origin the draw is balanced as a held-out set is; with fewer (the mixed rows alone, when the human and
    AI bank texts are drawn from training rows at export) each origin keeps `target` rows, round-robin over length bins.
    """
    # split: (m,)
    split = split.copy()
    held_docs = np.zeros(rows.doc.max() + 1, dtype=bool)  # (docs,)
    held_docs[rows.doc[np.isin(split, HELD_OUT)]] = True
    open_rows = (split == "train") & ~held_docs[rows.doc]  # (m,)
    for origin in origins:
        pool = np.flatnonzero(open_rows & (rows.label == origin))  # (p,)
        if not len(pool) or not target:
            continue
        quota = math.ceil(target / len(np.unique(rows.cluster[pool])))
        drawn = np.concatenate([round_robin(members, rows.length_bin[members], quota, rng)
                                for members in segments(pool, rows.cluster[pool])])  # (d,)
        if len(origins) < len(ORIGINS):
            drawn = round_robin(drawn, rows.length_bin[drawn], target, rng)  # (target,)
        split[drawn] = "bank_draw"
    if len(origins) < len(ORIGINS):
        return split  # (m,)

    return balance_set(split, "bank_draw", rows, keys, rng, limit=target, mixed_share=mixed_share)  # (m,)


def assign_rows(rows: Rows, state: np.ndarray, config: SplitConfig, rng: np.random.Generator) -> np.ndarray:
    """Every eligible row's split from the cluster assignment: (m,) strings."""
    # state: (k,)
    cluster_split = np.array([HELD_OUT[value] if value >= 0 else ("unused" if value == MOAT else "train")
                              for value in state], dtype=object)  # (k,)
    split = cluster_split[rows.cluster].astype(object)  # (m,)
    split = resolve_docs(split, rows.doc)
    split = cap_held_out(split, rows, config.cap, rng)
    for name in HELD_OUT:
        split = balance_set(split, name, rows, config.balance_keys, rng, limit=config.targets[name], mixed_share=config.mixed_share)
    split = draw_bank(split, rows, config.targets["bank_draw"], config.balance_keys, rng, config.mixed_share, config.bank_draw_origins)
    outside = np.zeros(rows.doc.max() + 1, dtype=bool)  # (docs,)
    outside[rows.doc[np.isin(split, (*HELD_OUT, "bank_draw"))]] = True
    split[(split == "train") & outside[rows.doc]] = "unused"

    return split  # (m,)


def take(rows: Rows, mask: np.ndarray) -> Rows:
    """The rows `mask` selects: every array (m,) to (s,)."""
    return Rows(**{name: getattr(rows, name)[mask] for name in ("ids", "label", "doc", "length_bin", "format", "language", "cluster")})


def assign_holdable(rows: Rows, holdable: np.ndarray, state: np.ndarray, config: SplitConfig,
                    rng: np.random.Generator) -> np.ndarray:
    """Every eligible row's split when only the `holdable` rows may enter a held-out set or the bank draw: (m,) strings.

    The holdable rows are split by `assign_rows`, with `state` chosen from them alone. A row that is not holdable
    trains, unless its cluster is held out or in the moat, or its doc has a row in a held-out set or the draw, when
    it is `unused`, so the moat and the doc rule hold for every row.
    """
    # holdable: (m,) bool; state: (k,)
    split = np.full(len(rows), "train", dtype=object)  # (m,)
    split[holdable] = assign_rows(take(rows, holdable), state, config, rng)
    others = ~holdable  # (m,)
    split[others & (state[rows.cluster] != FREE)] = "unused"
    outside = np.zeros(rows.doc.max() + 1, dtype=bool)  # (docs,)
    outside[rows.doc[np.isin(split, (*HELD_OUT, "bank_draw"))]] = True
    split[others & outside[rows.doc]] = "unused"

    return split  # (m,)


def balance_train_languages(split: np.ndarray, rows: Rows, ratio: float,
                            rng: np.random.Generator) -> tuple[np.ndarray, dict[int, list[int]]]:
    """In each language, cut the larger of training's human and AI text to `ratio` times the smaller, round-robin
    over length bins; the rows cut become `unused`. A language one origin holds nearly alone would teach the
    language as the origin. Returns the split and, per language code cut, human and AI rows before, then after."""
    # split: (m,)
    split = split.copy()
    cut: dict[int, list[int]] = {}
    training = np.flatnonzero((split == "train") & (rows.label != MIXED))  # (t,)
    for members in segments(training, rows.language[training]):
        labels = rows.label[members]  # (r,)
        counts = [int((labels == origin).sum()) for origin in range(MIXED)]
        larger = int(np.argmax(counts))
        keep = math.floor(ratio * min(counts))
        if counts[larger] <= keep:
            continue
        chosen = members[labels == larger]  # (r_larger,)
        kept = round_robin(chosen, rows.length_bin[chosen], keep, rng)  # (keep,)
        split[np.setdiff1d(chosen, kept)] = "unused"
        after = list(counts)
        after[larger] = keep
        cut[int(rows.language[members[0]])] = [*counts, *after]

    return split, cut  # (m,)


def check_split(split: np.ndarray, rows: Rows, state: np.ndarray, linked: list[np.ndarray],
                keys: tuple[str, ...], bank_draw_origins: tuple[int, ...] = (0, 1, 2)) -> None:
    """The invariants: disjoint by cluster and by doc, moats empty, held-out sets balanced in every cell (`bank_draw`
    too, unless it draws fewer origins than all, when it holds only those)."""
    # split: (m,); state: (k,); linked: (k_c,) neighbor ids per cluster
    used = {name: split == name for name in ("train", *HELD_OUT, "bank_draw")}
    clusters = {name: set(np.unique(rows.cluster[mask]).tolist()) for name, mask in used.items()}
    for first in ("train", *HELD_OUT):
        for second in ("train", *HELD_OUT):
            if first < second:
                assert not clusters[first] & clusters[second], f"{first} and {second} share a cluster"
    assert clusters["bank_draw"] <= set(np.flatnonzero(state == FREE).tolist()), "bank_draw comes from training clusters"
    names = list(used)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = np.intersect1d(rows.doc[used[first]], rows.doc[used[second]])
            assert not len(shared), f"{first} and {second} share {len(shared)} docs"
    moat = np.flatnonzero(state == MOAT)
    assert not np.isin(rows.cluster[np.isin(split, ("train", *HELD_OUT))], moat).any(), "a moat cluster kept rows"
    for name in HELD_OUT:
        holder = set(np.flatnonzero(state == HELD_OUT.index(name)).tolist())
        for cluster in holder:
            near = state[linked[cluster]]
            assert ((near < 0) | (near == HELD_OUT.index(name))).all(), f"a {name} cluster touches another set"
    for name in (*HELD_OUT, "bank_draw"):
        members = split == name
        if name == "bank_draw" and len(bank_draw_origins) < len(ORIGINS):
            assert np.isin(rows.label[members], bank_draw_origins).all(), "bank_draw holds an origin it does not draw"
            continue
        cell = np.zeros(len(rows), dtype=np.int64)
        for key in keys:
            values = getattr(rows, key)
            cell = cell * (values.max() + 1) + values
        for value in np.unique(cell[members]):
            counts = np.bincount(rows.label[members & (cell == value)], minlength=len(ORIGINS))
            assert counts[0] == counts[1] and counts[MIXED] <= counts[0], f"{name} is unbalanced in a {keys} cell: {counts.tolist()}"


@torch.inference_mode()
def distance_report(features: torch.Tensor, split: np.ndarray, doc: np.ndarray, rng: np.random.Generator,
                    device: torch.device | None = None) -> dict[str, Any]:
    """Median cosine of each set's rows to their nearest training row of another doc, and training's to itself."""
    # features: (m, f) unit rows, compared on `device` (theirs when None); split, doc: (m,)
    device = features.device if device is None else device
    train = np.flatnonzero(split == "train")
    reference = rng.permutation(train)[:DISTANCE_TRAIN_SAMPLE]  # (t,)
    reference_vectors = features[torch.from_numpy(reference).to(features.device)].to(device)  # (t, f)
    report: dict[str, Any] = {}
    if len(reference) < DISTANCE_NEIGHBORS:  # a smoke run whose held-out sets and moats took every cluster
        return report
    for name in ("train", *HELD_OUT, "bank_draw"):
        members = np.flatnonzero(split == name)
        if name == "train":
            members = np.setdiff1d(members, reference)
        queries = rng.permutation(members)[:DISTANCE_QUERY_SAMPLE]  # (q,)
        if not len(queries):
            continue
        similarity = features[torch.from_numpy(queries).to(features.device)].to(device).float() @ reference_vectors.float().T  # (q, t)
        top, index = similarity.topk(DISTANCE_NEIGHBORS, dim=1)  # (q, 8) each
        other_doc = torch.from_numpy(doc[reference]).to(device)[index] != torch.from_numpy(doc[queries]).to(device)[:, None]  # (q, 8)
        nearest = torch.where(other_doc, top, torch.full_like(top, -2.0)).max(dim=1).values.cpu().numpy()  # (q,)
        nearest = nearest[nearest > -2]
        report[name] = {"rows": int(len(nearest)), "median": float(np.median(nearest)),
                        "p10": float(np.quantile(nearest, 0.1)), "p90": float(np.quantile(nearest, 0.9))}

    return report


def set_summary(split: np.ndarray, rows: Rows, names: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rows per origin, and per origin in each length bin, format and language, for every set."""
    # split: (m,); names: (v,) category names per column
    present, set_code = np.unique(split.astype(str), return_inverse=True)  # (s,), (m,)
    origins = len(ORIGINS)
    by_origin = np.bincount(set_code * origins + rows.label, minlength=len(present) * origins).reshape(-1, origins)  # (s, 3)
    tables = {}
    for column, labels in (("length_bin", length_bin_names()), ("format", names["format"]), ("language", names["language"])):
        values = getattr(rows, column)  # (m,)
        width = int(values.max()) + 1
        counts = np.bincount((set_code * width + values) * origins + rows.label, minlength=len(present) * width * origins)
        tables[column] = (labels, counts.reshape(len(present), width, origins))  # (s, v, 3)
    summary = {}
    for index, name in enumerate(present):
        members = set_code == index  # (m,)
        entry = {"rows": int(by_origin[index].sum()), "clusters": int(len(np.unique(rows.cluster[members]))),
                 "origins": {origin: int(by_origin[index, number]) for number, origin in enumerate(ORIGINS)}}
        for column, (labels, counts) in tables.items():
            entry[column] = {str(labels[value]): counts[index, value].tolist()
                             for value in np.flatnonzero(counts[index].sum(1))}
        summary[str(name)] = entry

    return summary


def thin_length_groups(split: np.ndarray, rows: Rows, floor: int = LENGTH_GROUP_FLOOR) -> dict[str, dict[str, list[int]]]:
    """Per held-out set, the length bins holding rows where some origin has fewer than `floor` rows, with the counts
    per origin: the calibration table needs `floor` texts of each origin in every length group it keeps."""
    # split: (m,)
    bins = length_bin_names()
    thin: dict[str, dict[str, list[int]]] = {}
    for name in HELD_OUT:
        members = split == name  # (m,)
        if not members.any():
            continue
        counts = np.zeros((len(bins), len(ORIGINS)), dtype=np.int64)  # (b, 3)
        np.add.at(counts, (rows.length_bin[members], rows.label[members]), 1)
        short = np.flatnonzero((counts.sum(1) > 0) & (counts.min(1) < floor))
        if len(short):
            thin[name] = {str(bins[value]): counts[value].tolist() for value in short}

    return thin


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True,
                        help="embedding directories; their centered, normalized mean halves are clustered together")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-cluster", type=int, default=2500, help="k = round(eligible rows / this)")
    parser.add_argument("--cap", type=int, default=500, help="rows per held-out cluster and origin")
    parser.add_argument("--targets", default=",".join(f"{key}={value}" for key, value in DEFAULT_TARGETS.items()),
                        help="rows per origin of each set, as name=count pairs")
    parser.add_argument("--balance-keys", default="length_bin", help="comma-separated: length_bin, format, language")
    parser.add_argument("--mixed-share", type=float, default=MIXED_SHARE,
                        help="a held-out set's mixed rows as a share of its per-origin target")
    parser.add_argument("--bank-draw-origins", default=",".join(ORIGINS),
                        help="comma-separated origins `bank_draw` takes; `mixed` alone takes the bank_draw target of mixed rows, "
                             "for a bank whose human and AI texts are drawn from training rows at export")
    parser.add_argument("--allow-unembedded", action="store_true",
                        help="give rows some encoder did not embed the split `unembedded` instead of failing")
    parser.add_argument("--assignment-backend", default="triton",
                        help="e-means kernels: triton needs only PyTorch; cute needs the CUTLASS DSL packages")
    parser.add_argument("--max-points-per-centroid", type=int,
                        help="train e-means on a random subset of this many rows a cluster and keep the features in host "
                             "memory, for a GPU that cannot hold every row; unset, e-means trains on every row on the GPU")
    parser.add_argument("--held-out-languages", nargs="+",
                        help="the corpus `language` values whose rows may enter a held-out set or the bank draw; rows of "
                             "every other language only train; unset, every row may be held out")
    parser.add_argument("--train-language-ratio", type=float,
                        help="in each language, cut the larger of training's human and AI text to this many times the "
                             "smaller (to `unused`); unset, training keeps every row")
    parser.add_argument("--limit", type=int, help="split only the first this many eligible rows (a smoke run)")
    parser.add_argument("--keep-docs-together", action="store_true",
                        help="move every row to its doc's most common cluster, so no doc (a source group) is divided")
    parser.add_argument("--seed", type=int, default=0)


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if (args.out / "split.parquet").exists():
        print(f"{args.out / 'split.parquet'} exists; the split is done (delete it to redo)")
        return
    device = torch.device("cuda")
    columns = ["id", "label", "doc", "format", "language", "tokens", "benchmark"]
    metadata = read_columns(args.corpus, columns).sort_by("id")
    n = dense_row_count(metadata["id"].to_numpy())
    benchmark = pc.fill_null(metadata["benchmark"], "").to_numpy(zero_copy_only=False) != ""  # (n,)
    embedded = np.ones(n, dtype=bool)  # (n,)
    for directory in args.embeddings:
        embedded &= np.load(directory / TOKENS_FILE, mmap_mode="r")[:] >= 0
    missing = ~benchmark & ~embedded
    if missing.any() and not args.allow_unembedded:
        raise ValueError(f"{int(missing.sum())} non-benchmark rows are not embedded by every encoder given")
    eligible = np.flatnonzero(~benchmark & embedded)  # (m,)
    if args.limit is not None:
        eligible = eligible[: args.limit]
    targets = {**DEFAULT_TARGETS, **{key: int(value) for key, value in
                                     (pair.split("=") for pair in args.targets.split(",") if pair)}}
    draw_origins = tuple(ORIGINS.index(origin) for origin in args.bank_draw_origins.split(",") if origin)
    config = SplitConfig(targets, args.cap, tuple(key for key in args.balance_keys.split(",") if key), NEIGHBORS, args.seed,
                         args.mixed_share, draw_origins)

    names = {column: np.unique(pc.fill_null(metadata[column], "").to_numpy(zero_copy_only=False))
             for column in ("format", "language")}
    k = max(16, round(len(eligible) / args.rows_per_cluster))
    sampled = training_sample(len(eligible), k, args.max_points_per_centroid, args.seed) is not None
    features = cluster_features(args.embeddings, eligible, device, torch.device("cpu") if sampled else device)  # (m, f)
    centroids, labels, recipe = run_emeans(features, k, args.seed, args.assignment_backend, device, args.max_points_per_centroid)
    doc = codes(metadata["doc"].cast(pa.string()))[eligible]  # (m,)
    cluster = labels.cpu().numpy().astype(np.int64)  # (m,)
    together = {"kept": False}
    if args.keep_docs_together:
        cluster, moved = docs_together(cluster, doc)
        together = {"kept": True, **moved}
        print(f"docs together: {moved}", flush=True)
    rows = Rows(
        ids=eligible, label=metadata["label"].to_numpy()[eligible].astype(np.int64),
        doc=doc, length_bin=length_bins(metadata["tokens"].to_numpy()[eligible]),
        format=codes(metadata["format"])[eligible], language=codes(metadata["language"])[eligible],
        cluster=cluster,
    )
    if not np.isin(rows.label, range(len(ORIGINS))).all():
        raise ValueError("labels must be 0 human, 1 ai or 2 mixed")
    rng = np.random.default_rng(args.seed)
    linked = adjacency(nearest_clusters(centroids.cpu().numpy(), config.neighbors))
    holdable = np.ones(len(rows), dtype=bool)  # (m,)
    if args.held_out_languages is not None:
        holdable = np.isin(names["language"][rows.language], args.held_out_languages)
        print(f"held-out sets take {int(holdable.sum()):,} rows of {args.held_out_languages}; "
              f"{int((~holdable).sum()):,} rows of other languages only train", flush=True)
    held = take(rows, holdable)
    state = choose_clusters(held, k, linked, config, rng)
    split = assign_holdable(rows, holdable, state, config, rng)
    language_cut: dict[str, list[int]] = {}
    if args.train_language_ratio is not None:
        assert args.train_language_ratio >= 1.0, "--train-language-ratio is the larger origin over the smaller, at least 1"
        split, cut = balance_train_languages(split, rows, args.train_language_ratio, rng)
        language_cut = {str(names["language"][code]): counts for code, counts in cut.items()}
        print(f"training languages cut to {args.train_language_ratio} to 1: {len(cut)} languages, "
              f"{sum(counts[0] + counts[1] - counts[2] - counts[3] for counts in cut.values()):,} rows to unused", flush=True)
    check_split(split, rows, state, linked, config.balance_keys, config.bank_draw_origins)
    distances = distance_report(features, split, rows.doc, rng, device)

    full_split = np.full(n, "unused", dtype=object)  # (n,)
    full_split[benchmark] = "benchmark"
    full_split[~benchmark & ~embedded] = "unembedded"
    full_split[eligible] = split
    cluster = np.full(n, -1, dtype=np.int32)  # (n,)
    cluster[eligible] = rows.cluster
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "centroids.npy", centroids.cpu().numpy())
    pq.write_table(pa.table({"id": np.arange(n, dtype=np.int64), "cluster": cluster,
                             "split": pa.array(full_split.tolist(), pa.string())}), args.out / "split.parquet")
    summary = {
        "k": k, "rows_per_cluster": args.rows_per_cluster, "eligible_rows": int(len(eligible)), "emeans": recipe,
        "config": {"targets": config.targets, "cap": config.cap, "balance_keys": list(config.balance_keys),
                   "neighbors": config.neighbors, "seed": config.seed, "mixed_share": config.mixed_share,
                   "bank_draw_origins": [ORIGINS[origin] for origin in config.bank_draw_origins]},
        "thin_length_groups": thin_length_groups(split, rows),
        "train_language_ratio": args.train_language_ratio, "train_languages_cut": language_cut,
        "held_out_languages": args.held_out_languages,
        "clusters": {name: np.flatnonzero(state == index).tolist() for index, name in enumerate(HELD_OUT)},
        "moat_clusters": int((state == MOAT).sum()), "training_clusters": int((state == FREE).sum()),
        "sets": set_summary(full_split[eligible], rows, names), "distance_to_training": distances,
        "length_edges": list(LENGTH_EDGES), "embeddings": [path.as_posix() for path in args.embeddings],
        "docs_together": together,
    }
    write_json(args.out / "split_summary.json", summary)
    write_summary(args.out, "split", {**{key: summary[key] for key in ("k", "eligible_rows", "emeans", "config",
                                                                        "distance_to_training")},
                                      "rows_by_split": {name: int((full_split == name).sum()) for name in SPLIT_NAMES},
                                      "seconds": time.perf_counter() - started})
