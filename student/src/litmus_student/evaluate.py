"""Evaluate a release where training never looks: whole long documents read in chunks, documents spliced from
held-out passages of different origins, and the gap between a heat-map window and the same tokens read alone.

    python -m litmus_student.evaluate --checkpoint RUN/best.pt --release RELEASE --data PRETOKENIZED \\
        --experiment-id YYYY-MM-DD_slug --out DIR [--split validation] [--pilot]

Token states come from the checkpoint in eval mode, in float32 (the release's arithmetic; `export.py` checks that the
two agree within SPEC.md's Parity). The head, the bank, the calibration, the windows and the chunk combination are
the release's, read by `reference.py`, so every number here is a reading a user gets, from the code the engine
matches.

0. **Passages.** `--passages-per-origin` passages of each origin of `--split`, spread uniformly over clusters, each
   read as a user's text is: the same metrics as the long documents, by token band (1 to 128, 129 to 512, 513 to
   2,048, 2,049 to 8,192, and all). On `--split test` this is the release's test reading, run once.
1. **Long documents** (`long_eval` rows of `--split`, 8,193 to 131,072 tokens). Each is read in chunks and judged
   by its verdict. By number of chunks (2, 3 to 4, 5 to 8, 9 to 16, and all): PR AUC, ROC AUC and balanced MCC (at
   probability 0.5, a threshold fixed in advance) of the verdict's probability of not human, ai or mixed against
   human; coverage and mean size of the merged conformal sets at 0.05 and 0.10; the share of human documents
   called not human at each level; and how often the mixed-document signal fires, per origin.
2. **Spliced documents.** Held-out human and ai passages of `--split` (never `long_eval`) are concatenated as token
   ids into documents of one chunk with known boundaries: human then ai, ai then human, and three parts
   (human-ai-human, ai-human-ai), each passage drawn from one length band. Per cell of the heat map (one stride):
   its true origin is the passage holding most of its tokens, and the map's accuracy is how often the cell's
   position falls on that origin's side of 0.5. Each boundary is located by the cell edges that best fit the known
   order of origins (least squares of the cell positions against the anchors 0 and 1, weighted by cell tokens) and
   its error is reported in tokens. The same documents are also mapped by non-overlapping windows of the same width,
   so the two maps are compared on the same text, which measures whether the sliding window helps. All by pattern
   and passage band.
3. **Window context gap.** A window's vector is pooled from token states that saw neighbors outside it, unlike a
   standalone text of those tokens; per window of validation texts, the pooled vector's and the strip position's
   gap and whether the 0.10 sets and the side of 0.5 agree (`reference.window_context_gap`).

Writes `evaluation.json` (everything) and `evaluation.csv` (tidy rows) to `--out` and logs headline numbers to W&B
(project `litmus`, the experiment's group, job type `evaluate`).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import numpy as np
import torch

from pathlib import Path
from typing import Any
from tqdm import tqdm

from foundry.logging import Run
from foundry.metrics import balanced_mcc
from foundry.metrics.ranking import binary_ranking_metrics
from .contract import reference
from .data import TokenStore, cluster_stratified_rows, pack, packed_batch
from .model import single_text_batch


POOL_POSITIONS = 1 << 18  # stream positions one packed forward holds when many texts are read at once
BANK_ROWS = 2048  # texts whose products with the whole bank are held at once: four (rows, m) float64 arrays
READ_TEXTS = 1024  # documents `long_documents` hands to `Reader.read_many` at a time


PROJECT = "litmus"
CHUNK_BANDS = {"2": (2, 2), "3-4": (3, 4), "5-8": (5, 8), "9-16": (9, 16), "all": (2, 1 << 30)}
TOKEN_BANDS = {"1-128": (1, 128), "129-512": (129, 512), "513-2048": (513, 2048), "2049-8192": (2049, 8192), "all": (1, 1 << 30)}
PATTERNS = {"human_ai": (0, 1), "ai_human": (1, 0), "human_ai_human": (0, 1, 0), "ai_human_ai": (1, 0, 1)}
PASSAGE_BANDS = ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))  # tokens per passage in window widths, [lo, hi)
ANCHOR = {0: 0.0, 1: 1.0}  # strip anchors of human and ai


def passage_bands(window_width: int, window_stride: int, chunk_tokens: int) -> list[tuple[int, int]]:
    """Passage lengths [lo, hi) in tokens, at least one stride (so every passage holds a cell) and capped so that
    three passages fit one chunk: with the release's 512, 128 and 8,192, [128, 256), [256, 512), [512, 1,024),
    [1,024, 2,048) and [2,048, 2,731)."""
    top = chunk_tokens // 3 + 1
    bands = [(max(int(low * window_width), window_stride), min(int(high * window_width), top)) for low, high in PASSAGE_BANDS]

    return [(low, high) for low, high in bands if low < high]


class Reader:
    """A checkpoint's token states read by its release."""

    def __init__(self, checkpoint: Path, release: Path, device: torch.device) -> None:
        from .train import load_checkpoint

        self.model, _ = load_checkpoint(checkpoint, device)
        self.model.eval()
        self.release = reference.ReferenceModel.load(release)
        if self.release.config != self.model.config:
            raise ValueError(f"the release's config {self.release.config} is not the checkpoint's {self.model.config}")
        self.device = device

    @property
    def config(self) -> reference.StudentConfig:
        return self.model.config

    @torch.no_grad()
    def states(self, ids: np.ndarray) -> np.ndarray:
        """One text of at most a chunk: (n,) ids to (n, d) float32 token states, in float32 (no autocast)."""
        # ids: (n,)
        assert 0 < len(ids) <= self.config.chunk_tokens, "a text is read as chunks of at most chunk_tokens"
        H = self.model.encoder(single_text_batch(ids, self.config.block_size).to(self.device))  # (t, d)

        return H[: len(ids)].float().cpu().numpy()  # (n, d)

    def read(self, ids: np.ndarray, windows: bool = True) -> reference.DocumentReading:
        """The document's reading (SPEC.md), chunked; without the heat map's windows when `windows` is false."""
        # ids: (n,)
        states = [self.states(ids[start:end]) for start, end in reference.chunk_bounds(len(ids), self.config.chunk_tokens)]

        return self.release.read_states(states, windowed=windows)[0]

    @torch.no_grad()
    def pooled(self, chunks: list[np.ndarray], positions: int = POOL_POSITIONS) -> np.ndarray:
        """Many texts of at most a chunk each, encoded in packed forwards: (m_i,) ids each to (c, 2d) float32, the
        pooled statistics `ChunkPool` gives a whole chunk (float64 sums of the float32 token states, taken on the
        device)."""
        config = self.config
        lengths = np.asarray([len(ids) for ids in chunks], dtype=np.int64)  # (c,)
        assert len(chunks) and lengths.min() > 0 and lengths.max() <= config.chunk_tokens, "each text is one chunk with a token"
        vectors = np.zeros((len(chunks), 2 * config.width), dtype=np.float32)  # (c, 2d)
        budget = max(positions - positions % config.block_size, config.block_size)
        for placements in pack(lengths, budget, config.block_size):
            batch = packed_batch(placements, chunks, config.block_size, config.block_size).to(self.device)
            H = self.model.encoder(batch)  # (t, d)
            valid = batch.segment >= 0  # (t,)
            rows, states = batch.segment[valid], H[valid].double()  # (t_v,), (t_v, d)
            sums = torch.zeros(batch.texts, config.width, dtype=torch.float64, device=self.device)  # (n, d)
            total = sums.index_add(0, rows, states).cpu().numpy()  # (n, d)
            squares = sums.index_add(0, rows, states.square()).cpu().numpy()  # (n, d)
            for local, placement in enumerate(placements):
                vectors[placement.row] = reference.pooled_statistics(total[local], squares[local], int(lengths[placement.row]))

        return vectors  # (c, 2d)

    def read_many(self, texts: list[np.ndarray]) -> list[reference.Verdict]:
        """Every text's verdict, with no heat map: all their chunks pooled in packed forwards, then read against the
        whole bank together (`verdicts`). `read(ids, windows=False)["verdict"]` of each, to the last float32 bit of
        the token states."""
        # texts: (n_i,) ids each, none empty
        bounds = [reference.chunk_bounds(len(ids), self.config.chunk_tokens) for ids in texts]
        chunks = [ids[start:end] for ids, spans in zip(texts, bounds, strict=True) for start, end in spans]
        owner = np.repeat(np.arange(len(texts)), [len(spans) for spans in bounds])  # (c,) the text of each chunk

        return verdicts(self.release, self.pooled(chunks), [len(ids) for ids in chunks], owner, len(texts))


def bank_readings(release: reference.ReferenceModel, Q: np.ndarray, K: np.ndarray, top_k: int) -> list[reference.BankReading]:
    """`release.bank_reading` of many texts: in Int8 output every query and key against every bank row in four
    products a slice of `BANK_ROWS` texts (the integer sums are exact in float64 in any order, and the float64 steps
    after them are the reference's, element by element); a float release is read a text at a time."""
    # Q, K: (c, w) as `head` returns
    if release.output != "int8":
        return [release.bank_reading(q, k, top_k) for q, k in zip(Q, K, strict=True)]
    assert Q.dtype == np.int8 and K.dtype == np.int8, "an int8 output reads int8 codes"
    origins = [release.bank_origin == origin for origin in range(3)]  # (m,) each
    m = len(release.bank_origin)
    readings: list[reference.BankReading] = []
    for begin in range(0, len(Q), BANK_ROWS):
        codes_q, codes_k = Q[begin:begin + BANK_ROWS], K[begin:begin + BANK_ROWS]  # (r, w) each
        q64, k64 = codes_q.astype(np.float64), codes_k.astype(np.float64)  # (r, w) each
        inverse_q, inverse_k = reference.inverse_norms(codes_q)[:, None], reference.inverse_norms(codes_k)[:, None]  # (r, 1) each
        qq, qk = q64 @ release.bank_query.T, q64 @ release.bank_key.T  # (r, m) each: q . q_j, q . k_j
        kq, kk = k64 @ release.bank_query.T, k64 @ release.bank_key.T  # (r, m) each: k . q_j, k . k_j
        pair = (qk * inverse_q) * release.bank_inverse_key + (kq * inverse_k) * release.bank_inverse_query  # (r, m)
        logits = release.pair_scale * pair / 2.0 + release.pair_bias  # (r, m)
        p = 0.5 * (1.0 + np.tanh(0.5 * logits))  # (r, m)
        cosines = ((qq * inverse_q) * release.bank_inverse_query + (kk * inverse_k) * release.bank_inverse_key) / 2.0  # (r, m)
        mean = np.stack([np.ascontiguousarray(p[:, mask]).mean(axis=1) for mask in origins], axis=1)  # (r, 3)
        spread = np.stack([np.ascontiguousarray(p[:, mask]).std(axis=1) for mask in origins], axis=1)  # (r, 3)
        # The top_k highest cosines, in descending order as the reference sums them: (r, top_k).
        highest = np.sort(np.partition(cosines, m - top_k, axis=1)[:, m - top_k:], axis=1)[:, ::-1]
        closeness = np.ascontiguousarray(highest).mean(axis=1)  # (r,)
        readings.extend({"mean": mean[row].tolist(), "spread": spread[row].tolist(), "closeness": float(closeness[row])}
                        for row in range(len(codes_q)))

    return readings


def verdicts(release: reference.ReferenceModel, pooled: np.ndarray, tokens: list[int], owner: np.ndarray,
             documents: int) -> list[reference.Verdict]:
    """The verdicts of documents from their chunks' pooled vectors, chunks in document order: what
    `ReferenceModel.document` gives as `verdict`, the bank read for every chunk at once."""
    # pooled: (c, 2d) float32; tokens: (c,) each chunk's tokens; owner: (c,) each chunk's document, non-decreasing, covering 0..documents-1
    assert release.calibration is not None, "a reading needs the release's calibration table"
    assert len(pooled) == len(owner) == len(tokens) and (np.diff(owner) >= 0).all() \
        and np.array_equal(np.unique(owner), np.arange(documents)), "every document has its chunks, in order"
    heads = [release.head(vector) for vector in pooled]
    Q, K = np.stack([q for q, _ in heads]), np.stack([k for _, k in heads])  # (c, w) each
    banks = bank_readings(release, Q, K, release.calibration.top_k)
    readings = [release.calibration.read(bank, n) for bank, n in zip(banks, tokens, strict=True)]
    ends = np.searchsorted(owner, np.arange(documents), side="right")  # (documents,)
    starts = np.concatenate([[0], ends[:-1]])  # (documents,)

    return [release.calibration.verdict(readings[start:end]) for start, end in zip(starts, ends, strict=True)]


def ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """PR AUC, ROC AUC and balanced MCC at 0.5 of a probability; null where a class is missing."""
    # labels, scores: (n,)
    metrics = binary_ranking_metrics(labels, scores)
    both = len(np.unique(labels)) == 2
    mcc = balanced_mcc(labels, scores, threshold=0.5)["balanced_mcc"] if both else None

    return {"pr_auc": metrics["average_precision"], "roc_auc": metrics["auroc"], "mcc": mcc}


# ---------------------------------------------------------------- long documents


def long_documents(reader: Reader, store: TokenStore, rows: np.ndarray, name: str = "long documents") -> list[dict[str, Any]]:
    """Each document's verdict, compactly; a passage is a document of one chunk."""
    # rows: (m,)
    documents = []
    for begin in tqdm(range(0, len(rows), READ_TEXTS), desc=name, unit="batch", mininterval=30, file=sys.stderr):
        part = rows[begin:begin + READ_TEXTS]  # (b,)
        for row, verdict in zip(part, reader.read_many([store.ids(int(row)) for row in part]), strict=True):
            documents.append({"row": int(row), "label": int(store.label[row]), "tokens": int(store.length[row]), "chunks": verdict["chunks"],
                              "probability": verdict["probability"], "sets": verdict["sets"], "not_human": verdict["not_human"],
                              "mixed_document": verdict["mixed_document"], "interval": verdict["interval"],
                              "disagreement": verdict["disagreement"]})

    return documents


def long_document_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tidy rows of the long-document evaluation, by chunk band."""
    return verdict_rows(documents, "long_documents", "chunks", CHUNK_BANDS)


def passage_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tidy rows of the passage evaluation, by token band."""
    return verdict_rows(documents, "passages", "tokens", TOKEN_BANDS)


def verdict_rows(documents: list[dict[str, Any]], evaluation: str, kind: str, bands: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """Tidy rows of documents' verdicts by band of `kind` (a document's `chunks` or `tokens`)."""
    rows: list[dict[str, Any]] = []
    for band, (low, high) in bands.items():
        members = [document for document in documents if low <= document[kind] <= high]
        if not members:
            continue
        labels = np.asarray([document["label"] for document in members])  # (m,)
        written = (labels != 0).astype(np.int64)  # (m,)
        probability = np.asarray([document["probability"] for document in members])  # (m,)

        def add(metric: str, value: float | None, count: int, level: str = "") -> None:
            rows.append({"evaluation": evaluation, "stratum_kind": kind, "stratum": band, "metric": metric, "level": level,
                         "value": value, "n": count})

        for metric, value in ranking(written, probability).items():
            add(metric, value, len(members))
        add("mean_interval_width", float(np.mean([document["interval"][1] - document["interval"][0] for document in members])), len(members))
        add("mean_disagreement", float(np.mean([document["disagreement"] for document in members])), len(members))
        for level in ("0.05", "0.1"):
            held = [reference.ORIGINS[document["label"]] in document["sets"][level] for document in members]
            add("coverage", float(np.mean(held)), len(members), level)
            add("mean_set_size", float(np.mean([len(document["sets"][level]) for document in members])), len(members), level)
            human = [document for document in members if document["label"] == 0]
            if human:
                add("human_called_not_human", float(np.mean([document["not_human"][level] for document in human])), len(human), level)
        for origin, name in enumerate(reference.ORIGINS):
            of_origin = [document for document in members if document["label"] == origin]
            if of_origin:
                add(f"mixed_document_rate_{name}", float(np.mean([document["mixed_document"] for document in of_origin])), len(of_origin))

    return rows


# ---------------------------------------------------------------- spliced documents


def window_map(reader: Reader, H: np.ndarray, width: int, stride: int) -> list[reference.Cell]:
    """The heat map of one chunk's states read by windows of `width` every `stride` tokens, in cells of the release's
    stride: `stride` equal to the release's is the release's map; equal to `width`, the non-overlapping sections."""
    # H: (m, d)
    release = reader.release
    assert release.calibration is not None, "the release carries its calibration table"
    pool = reference.ChunkPool(reader.config.width, len(H), width, stride)
    pool.add(H)
    spans = reference.window_spans(len(H), width, stride)
    heads = [release.head(pool.pooled(start, end)) for start, end in spans]
    Q, K = np.stack([q for q, _ in heads]), np.stack([k for _, k in heads])  # (v, w) each
    banks = bank_readings(release, Q, K, release.calibration.top_k)  # every window against the whole bank together
    windows: list[reference.WindowReading] = []
    for (start, end), bank in zip(spans, banks, strict=True):
        reading = release.calibration.read(bank, end - start)
        windows.append({**reading, "start": start, "end": end, "chunk": 0})  # type: ignore[typeddict-item]

    return reference.heat_map(windows, 0, len(H), reader.config.window_stride)


def locate(cells: list[reference.Cell], order: tuple[int, ...]) -> list[int]:
    """The boundaries between consecutive passages of known origins: the cell edges whose step function of anchors best
    fits the cell positions, weighted by cell tokens (least squares, exhaustive)."""
    assert len(cells) >= len(order), "every passage holds at least one cell"
    weights = np.asarray([cell["end"] - cell["start"] for cell in cells], dtype=np.float64)  # (c,)
    positions = np.asarray([cell["position"] for cell in cells])  # (c,)
    prefix = [np.concatenate([[0.0], np.cumsum(values)]) for values in (weights, weights * positions, weights * positions**2)]  # 3 x (c + 1,)

    def cost(first: int, last: int, anchor: float) -> float:
        mass, linear, square = (values[last] - values[first] for values in prefix)
        return square - 2 * anchor * linear + anchor * anchor * mass

    best, chosen = np.inf, ()
    for edges in itertools.combinations(range(1, len(cells)), len(order) - 1):
        bounds = (0, *edges, len(cells))
        total = sum(cost(bounds[index], bounds[index + 1], ANCHOR[origin]) for index, origin in enumerate(order))
        if total < best:
            best, chosen = total, edges

    return [cells[edge]["start"] for edge in chosen]


def spliced_documents(reader: Reader, store: TokenStore, split: str, per_cell: int, generator: np.random.Generator) -> list[dict[str, Any]]:
    """Per spliced document: pattern, band, the sliding and the section maps' cell accuracy and boundary errors."""
    pool_rows = store.split_mask(split) & ~store.long_eval  # (n,)
    width = reader.config.window_width
    splices = []
    bands = passage_bands(width, reader.config.window_stride, reader.config.chunk_tokens)
    for (pattern, order), (low, high) in itertools.product(PATTERNS.items(), bands):
        candidates = {origin: np.flatnonzero(pool_rows & (store.label == origin) & (store.length >= low) & (store.length < high))
                      for origin in set(order)}
        if any(len(rows) < len(order) for rows in candidates.values()):
            print(f"spliced {pattern} [{low}, {high}): too few passages, skipped", file=sys.stderr)
            continue
        for _ in range(per_cell):
            parts = []
            for origin in set(order):
                drawn = iter(generator.choice(candidates[origin], order.count(origin), replace=False))
                parts.extend((index, int(next(drawn))) for index, each in enumerate(order) if each == origin)
            rows = [row for _, row in sorted(parts)]
            pieces = [store.ids(row) for row in rows]
            ids = np.concatenate(pieces)  # (n,)
            assert len(ids) <= reader.config.chunk_tokens, "a spliced document is one chunk"
            truth = np.repeat(np.asarray(order), [len(piece) for piece in pieces])  # (n,) each token's origin
            boundaries = np.cumsum([len(piece) for piece in pieces])[:-1].tolist()
            H = reader.states(ids)  # (n, d)
            splice: dict[str, Any] = {"pattern": pattern, "band": f"{low}-{high - 1}", "tokens": len(ids), "rows": rows}
            for name, stride in (("sliding", reader.config.window_stride), ("sections", width)):
                cells = window_map(reader, H, width, stride)
                labels = [int(np.bincount(truth[cell["start"] : cell["end"]], minlength=2).argmax()) for cell in cells]
                correct = [(cell["position"] >= 0.5) == bool(label) for cell, label in zip(cells, labels, strict=True)]
                located = locate(cells, order)
                splice[name] = {"cell_accuracy": float(np.mean(correct)), "cells": len(cells),
                                "boundary_errors": [abs(found - true) for found, true in zip(located, boundaries, strict=True)]}
            splices.append(splice)

    return splices


def spliced_rows(splices: list[dict[str, Any]], stride: int) -> list[dict[str, Any]]:
    """Tidy rows of the spliced documents by pattern and passage band; `boundary_within_stride` is the share of
    boundaries placed within one cell of the truth."""
    rows: list[dict[str, Any]] = []
    keys = sorted({(splice["pattern"], splice["band"]) for splice in splices}) + [("all", "all")]
    for pattern, band in keys:
        members = [splice for splice in splices if pattern in ("all", splice["pattern"]) and band in ("all", splice["band"])]
        for name in ("sliding", "sections"):
            errors = [error for splice in members for error in splice[name]["boundary_errors"]]
            for metric, value in (("cell_accuracy", float(np.mean([splice[name]["cell_accuracy"] for splice in members]))),
                                  ("boundary_error_mean", float(np.mean(errors))), ("boundary_error_median", float(np.median(errors))),
                                  ("boundary_within_stride", float(np.mean(np.asarray(errors) <= stride)))):
                rows.append({"evaluation": f"spliced_{name}", "stratum_kind": "pattern|band", "stratum": f"{pattern}|{band}",
                             "metric": metric, "level": "", "value": value, "n": len(members)})

    return rows


# ---------------------------------------------------------------- window context gap


def context_gap(reader: Reader, store: TokenStore, split: str, texts: int, generator: np.random.Generator) -> list[dict[str, Any]]:
    """`reference.window_context_gap` on validation texts long enough for several windows, each window marked
    interior when it touches neither end of its text."""
    config = reader.config
    candidates = np.flatnonzero(store.split_mask(split) & ~store.long_eval & (store.length >= 2 * config.window_width)
                                & (store.length <= config.chunk_tokens))  # (m,)
    rows = []
    for row in generator.choice(candidates, min(texts, len(candidates)), replace=False):
        ids = store.ids(int(row))
        for gap in reference.window_context_gap(reader.release, ids, reader.states):
            rows.append({**gap, "row": int(row), "interior": gap["start"] > 0 and gap["end"] < len(ids)})

    return rows


def context_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum, members in (("all", gaps), ("interior", [gap for gap in gaps if gap["interior"]]),
                             ("edge", [gap for gap in gaps if not gap["interior"]])):
        if not members:
            continue
        for metric, values in (("pooled_gap", [gap["pooled_gap"] for gap in members]), ("query_key_gap", [gap["query_key_gap"] for gap in members]),
                               ("position_gap", [gap["position_gap"] for gap in members])):
            rows.append({"evaluation": "window_context", "stratum_kind": "window", "stratum": stratum, "metric": metric + "_mean",
                         "level": "", "value": float(np.mean(values)), "n": len(members)})
            rows.append({"evaluation": "window_context", "stratum_kind": "window", "stratum": stratum, "metric": metric + "_p95",
                         "level": "", "value": float(np.quantile(values, 0.95)), "n": len(members)})
        for metric in ("same_set", "same_side"):
            rows.append({"evaluation": "window_context", "stratum_kind": "window", "stratum": stratum, "metric": metric,
                         "level": "0.1" if metric == "same_set" else "", "value": float(np.mean([gap[metric] for gap in members])), "n": len(members)})

    return rows


# ---------------------------------------------------------------- the run


def headline(rows: list[dict[str, Any]]) -> dict[str, float]:
    """The numbers logged to W&B (at most 40 keys); the CSV holds the rest."""
    wanted = {("long_documents", "all", metric, level) for metric, level in
              (("pr_auc", ""), ("roc_auc", ""), ("mcc", ""), ("coverage", "0.05"), ("coverage", "0.1"), ("human_called_not_human", "0.05"),
               ("mixed_document_rate_human", ""), ("mixed_document_rate_ai", ""), ("mixed_document_rate_mixed", ""))}
    wanted |= {("passages", "all", metric, level) for metric, level in
               (("pr_auc", ""), ("roc_auc", ""), ("mcc", ""), ("coverage", "0.1"), ("human_called_not_human", "0.1"))}
    wanted |= {(f"spliced_{name}", "all|all", metric, "") for name in ("sliding", "sections")
               for metric in ("cell_accuracy", "boundary_error_median", "boundary_within_stride")}
    wanted |= {("window_context", "interior", metric, level) for metric, level in
               (("pooled_gap_mean", ""), ("position_gap_mean", ""), ("position_gap_p95", ""), ("same_set", "0.1"), ("same_side", ""))}
    logged = {}
    for row in rows:
        if (row["evaluation"], row["stratum"], row["metric"], str(row["level"])) in wanted and row["value"] is not None:
            logged[f"{row['evaluation']}/{row['metric']}{'@' + str(row['level']) if row['level'] else ''}"] = float(row["value"])

    return logged


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(arguments.device)
    generator = np.random.default_rng(arguments.seed)
    reader = Reader(arguments.checkpoint, arguments.release, device)
    store = TokenStore(arguments.data)
    long_rows = np.flatnonzero(store.split_mask(arguments.split) & store.long_eval)  # (m,)
    if arguments.long_documents < len(long_rows):
        long_rows = np.sort(generator.choice(long_rows, arguments.long_documents, replace=False))
    report: dict[str, Any] = {"split": arguments.split, "checkpoint": str(arguments.checkpoint), "release": str(arguments.release)}
    passages = np.sort(cluster_stratified_rows(store, arguments.split, arguments.passages_per_origin, generator))  # (p,)
    report["passages"] = long_documents(reader, store, passages, "passages")
    report["long_documents"] = long_documents(reader, store, long_rows)
    report["spliced"] = spliced_documents(reader, store, arguments.split, arguments.spliced_per_cell, generator)
    report["window_context"] = context_gap(reader, store, arguments.split, arguments.context_texts, generator)
    rows = (passage_rows(report["passages"]) + long_document_rows(report["long_documents"])
            + spliced_rows(report["spliced"], reader.config.window_stride) + context_rows(report["window_context"]))
    report["rows"] = rows

    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "evaluation.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    with (arguments.out / "evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["evaluation", "stratum_kind", "stratum", "metric", "level", "value", "n"])
        writer.writeheader()
        writer.writerows(rows)
    summary = headline(rows)
    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
              "long_documents_read": len(long_rows), "spliced_read": len(report["spliced"])}
    with Run(arguments.out / "wandb", config, project=PROJECT, group=arguments.experiment_id, job_type="evaluate",
             name=f"{arguments.experiment_id}_evaluate", pilot=arguments.pilot) as run:
        run.finish(summary=summary)
    print(json.dumps(summary, indent=1), file=sys.stderr)

    return report


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True, help="the checkpoint's release, from export.py")
    parser.add_argument("--data", type=Path, required=True, help="the pre-tokenized corpus of `litmus_student.data`")
    parser.add_argument("--experiment-id", required=True, help="YYYY-MM-DD_slug, the W&B group")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--passages-per-origin", type=int, default=2000, help="passages of each origin, spread over clusters")
    parser.add_argument("--long-documents", type=int, default=6000, help="at most this many long documents")
    parser.add_argument("--spliced-per-cell", type=int, default=200, help="spliced documents per pattern and passage band")
    parser.add_argument("--context-texts", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot", action="store_true", help="no W&B run")

    return parser


def main() -> None:
    evaluate(parser().parse_args())


if __name__ == "__main__":
    main()
