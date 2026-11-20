"""Pre-tokenized corpus, step sampling, packing into block-aligned streams, teacher targets and prefetch.

On-disk format of a pre-tokenized corpus (a directory), every row indexed by its corpus `id` (`0 .. n-1`, the row
order of the teacher's `.npy` files):

| File | Holds |
| --- | --- |
| `tokens.npy` | every text's token ids back to back, uint16 (the vocabulary is 50,368 < 65,536) |
| `offsets.npy` | int64 `(n + 1,)`: text `i` is `tokens[offsets[i] : offsets[i + 1]]` |
| `rows.npz` | `label` int8, `split` int8 (codes into `meta.json`'s `splits`), `cluster` int64, `length` int64, `long_eval` bool (a whole document over 8,192 tokens, kept to evaluate the chunked reading and never trained on) |
| `meta.json` | row and token counts, split names, the tokenizer's repository, revision and sha256, the corpus path |

    python -m litmus_student.data --corpus CORPUS_PARQUET_DIR --out PRETOKENIZED --tokens-only   # while the teacher embeds
    python -m litmus_student.data --corpus CORPUS_PARQUET_DIR --out PRETOKENIZED --split SPLIT_PARQUET

Texts are tokenized exactly as the engine reads them: the corpus text (already in the standard reading) through
ModernBERT-large's tokenizer with no special tokens. The corpus's `tokens` column must equal the count for every
row, so a different tokenizer or text fails the build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from tqdm import tqdm

from .contract import reference
from .model import PackedBatch
from .presentation import load_synonyms, presented


TOKENS_FILE = "tokens.npy"
TEXTS_FILE = "texts.bin"  # a text store's UTF-8 texts back to back
TEXT_STARTS_FILE = "text_starts.npy"
TEXT_LENGTHS_FILE = "text_lengths.npy"
OFFSETS_FILE = "offsets.npy"
ROWS_FILE = "rows.npz"
META_FILE = "meta.json"
CORPUS_COLUMNS = ["id", "label", "tokens", "long_eval"]
SPLIT_COLUMNS = ["id", "split", "cluster"]


# ---------------------------------------------------------------- pre-tokenization


def parquet_files(corpus: Path) -> list[Path]:
    """One file, or every `*.parquet` under a directory, in a fixed order (the teacher's rule)."""
    if corpus.is_file():
        return [corpus]
    files = sorted(corpus.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {corpus}")

    return files


def corpus_lengths(corpus: Path) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    """The corpus's row columns, its ids, and every row's token count indexed by id: table, (n,), (n,)."""
    columns = pa.concat_tables([pq.read_table(path, columns=CORPUS_COLUMNS) for path in parquet_files(corpus)])
    ids = columns["id"].to_numpy()  # (n,)
    n = len(ids)
    if n == 0 or ids.min() != 0 or ids.max() != n - 1 or len(np.unique(ids)) != n:
        raise ValueError("corpus ids must be a permutation of 0..n-1, since they index the teacher's rows")
    length = np.zeros(n, dtype=np.int64)  # (n,)
    length[ids] = columns["tokens"].to_numpy()
    if (length <= 0).any():
        raise ValueError(f"{int((length <= 0).sum())} rows have no tokens; the student reads only texts of one token or more")

    return columns, ids, length


def write_rows(corpus: Path, out: Path, split_path: Path | None, reading: str = "standard") -> None:
    """Write `rows.npz` and `meta.json` beside the tokens. Each row's split and cluster come from the teacher's
    `split.parquet` (`id`, `split`, `cluster`; a row with no cluster, as a benchmark row, gets -1), or, with no
    `split_path`, from the corpus's own `split` and `cluster` columns. Lengths are the tokens' own, which in the
    standard reading must equal the corpus's `tokens`."""
    columns, ids, corpus_length = corpus_lengths(corpus)
    length = np.diff(np.load(out / OFFSETS_FILE))  # (n,)
    if reading == "standard" and not np.array_equal(length, corpus_length):
        raise ValueError("the tokens disagree with the corpus's `tokens`; the corpus was tokenized differently")
    n = len(ids)
    if split_path is None:
        assigned = pa.concat_tables([pq.read_table(path, columns=SPLIT_COLUMNS) for path in parquet_files(corpus)])
    else:
        assigned = pq.read_table(split_path, columns=SPLIT_COLUMNS)
    assigned_ids = assigned["id"].to_numpy()  # (n,)
    if len(assigned_ids) != n or not np.array_equal(np.sort(assigned_ids), np.arange(n)):
        raise ValueError("the split must name every corpus id exactly once")
    names = assigned["split"].to_pylist()
    split_names = sorted(set(names))
    split_code = {name: code for code, name in enumerate(split_names)}
    split = np.zeros(n, dtype=np.int8)  # (n,)
    split[assigned_ids] = [split_code[name] for name in names]
    cluster = np.zeros(n, dtype=np.int64)  # (n,)
    cluster[assigned_ids] = pc.fill_null(assigned["cluster"], -1).to_numpy()
    label = np.zeros(n, dtype=np.int8)  # (n,)
    label[ids] = columns["label"].to_numpy()
    long_eval = np.zeros(n, dtype=bool)  # (n,)
    long_eval[ids] = columns["long_eval"].to_numpy(zero_copy_only=False)
    np.savez(out / ROWS_FILE, label=label, split=split, cluster=cluster, length=length, long_eval=long_eval)
    meta = {"rows": n, "tokens": int(length.sum()), "splits": split_names, "corpus": str(corpus), "reading": reading,
            "split": None if split_path is None else str(split_path),
            "tokenizer": {"repository": reference.TOKENIZER_REPOSITORY, "revision": reference.TOKENIZER_REVISION,
                          "sha256": reference.TOKENIZER_SHA256}}
    (out / META_FILE).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


WORKER_TOKENIZER: Any = None  # a worker process's own tokenizer (`read_and_tokenize`)


def start_worker(tokenizer_path: Path | None) -> None:
    global WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # many processes, one thread each
    WORKER_TOKENIZER = reference.load_tokenizer(tokenizer_path)


def read_and_tokenize(job: tuple[np.ndarray, list[str], str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A batch's texts in a reading, tokenized: its ids (r,), token counts (r,), and tokens back to back (t,) uint16."""
    ids, texts, reading = job
    read = [reference.read_as(text, reading) for text in texts]
    encodings = WORKER_TOKENIZER.encode_batch(read, add_special_tokens=False)
    lengths = np.asarray([len(encoding.ids) for encoding in encodings], dtype=np.int64)  # (r,)
    tokens = np.fromiter((index for encoding in encodings for index in encoding.ids), dtype=np.uint16, count=int(lengths.sum()))  # (t,)
    return ids, lengths, tokens


def tokenize_reading(corpus: Path, out: Path, tokenizer_path: Path | None, batch_rows: int, reading: str, workers: int) -> None:
    """Write the tokens of `corpus` read in `reading` (`reference.READINGS`, not the standard one, which the corpus's
    texts already are): each batch read and tokenized in a worker process, at most two batches a worker in flight, then
    every row's tokens placed by its id. Lengths come from the tokens, not the corpus's `tokens` column."""
    _, _, corpus_length = corpus_lengths(corpus)
    n = len(corpus_length)
    length = np.full(n, -1, dtype=np.int64)  # (n,)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    progress = tqdm(total=n, desc=f"read {reading} and tokenize", unit="text", mininterval=30, file=sys.stderr)

    def collect(part: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        length[part[0]] = part[1]
        parts.append(part)
        progress.update(len(part[0]))

    pending: list = []

    def collect_one() -> None:
        """Collect whichever batch finishes first, so one slow batch never keeps the other workers waiting."""
        while not (done := [job for job in pending if job.ready()]):
            pending[0].wait(0.05)
        pending.remove(done[0])
        collect(done[0].get())

    # spawned, not forked: the parent's Arrow and tokenizer threads could hold a lock a forked child never sees released
    with get_context("spawn").Pool(workers, initializer=start_worker, initargs=(tokenizer_path,)) as pool:
        for path in parquet_files(corpus):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=["id", "text"]):
                pending.append(pool.apply_async(read_and_tokenize, ((batch.column("id").to_numpy(), batch.column("text").to_pylist(), reading),)))
                if len(pending) >= 2 * workers:
                    collect_one()
        while pending:
            collect_one()
    progress.close()
    if (length < 1).any():
        raise ValueError(f"{int((length < 1).sum())} rows have no tokens in the {reading} reading, or were not read")
    offsets = np.zeros(n + 1, dtype=np.int64)  # (n + 1,)
    np.cumsum(length, out=offsets[1:])
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(out / (TOKENS_FILE + ".partial"), mode="w+", dtype=np.uint16, shape=(int(offsets[-1]),))  # (N,)
    for ids, lengths, values in parts:
        starts = np.cumsum(lengths) - lengths  # (r,) each row's first token in `values`
        # every token's place in the store: its row's offset plus its place inside the row
        tokens[np.repeat(offsets[ids], lengths) + np.arange(len(values)) - np.repeat(starts, lengths)] = values
    tokens.flush()
    del tokens
    np.save(out / OFFSETS_FILE, offsets)
    os.replace(out / (TOKENS_FILE + ".partial"), out / TOKENS_FILE)


def tokenize_corpus(corpus: Path, out: Path, tokenizer_path: Path | None, batch_rows: int) -> None:
    """Write the tokens of `corpus` to `out` (`tokens.npy`, `offsets.npy`); `write_rows` writes the rest. The
    tokens need no split, so they can be written while the teacher's embedding, which the split waits for, runs."""
    # The image turns the tokenizer's threads off for the training loaders' worker processes; this step forks none, and
    # one thread read 1,750 texts a second (2.8 hours for 17.4 million texts).
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = reference.load_tokenizer(tokenizer_path)
    files = parquet_files(corpus)
    _, _, length = corpus_lengths(corpus)
    n = len(length)
    offsets = np.zeros(n + 1, dtype=np.int64)  # (n + 1,)
    np.cumsum(length, out=offsets[1:])

    out.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(out / (TOKENS_FILE + ".partial"), mode="w+", dtype=np.uint16, shape=(int(offsets[-1]),))  # (N,)
    progress = tqdm(total=n, desc="tokenize", unit="text", mininterval=30, file=sys.stderr)
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=["id", "text"]):
            batch_ids = batch.column("id").to_numpy()  # (r,)
            encodings = tokenizer.encode_batch(batch.column("text").to_pylist(), add_special_tokens=False)
            for row, encoding in zip(batch_ids, encodings, strict=True):
                row_ids = np.asarray(encoding.ids, dtype=np.uint16)  # (l,)
                if len(row_ids) != length[row]:
                    raise ValueError(f"row {row}: {len(row_ids)} tokens here, {length[row]} in the corpus's `tokens`; "
                                     "the corpus was tokenized differently")
                tokens[offsets[row] : offsets[row + 1]] = row_ids
            progress.update(len(batch_ids))
    progress.close()
    tokens.flush()
    del tokens
    np.save(out / OFFSETS_FILE, offsets)
    os.replace(out / (TOKENS_FILE + ".partial"), out / TOKENS_FILE)


class TokenStore:
    """A pre-tokenized corpus, memory-mapped."""

    def __init__(self, directory: Path) -> None:
        self.meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        if self.meta["tokenizer"]["sha256"] != reference.TOKENIZER_SHA256:
            raise ValueError(f"{directory} was tokenized with another tokenizer")
        self.tokens = np.load(directory / TOKENS_FILE, mmap_mode="r")  # (N,) uint16
        self.offsets = np.load(directory / OFFSETS_FILE)  # (n + 1,)
        rows = np.load(directory / ROWS_FILE)
        self.label, self.split, self.cluster, self.length = rows["label"], rows["split"], rows["cluster"], rows["length"]  # (n,) each
        if "long_eval" not in rows:
            raise ValueError(f"{directory} predates `long_eval`; tokenize the corpus again")
        self.long_eval = rows["long_eval"].astype(bool)  # (n,)
        self.splits: list[str] = self.meta["splits"]
        self.reading: str = self.meta.get("reading", "standard")  # how each text was read before tokenizing

    def __len__(self) -> int:
        return len(self.length)

    def ids(self, row: int) -> np.ndarray:
        return np.asarray(self.tokens[self.offsets[row] : self.offsets[row + 1]], dtype=np.int64)  # (l,)

    def split_mask(self, name: str) -> np.ndarray:
        if name not in self.splits:
            raise KeyError(f"no split {name!r}; the corpus has {self.splits}")
        return self.split == self.splits.index(name)  # (n,)


def write_store(directory: Path, texts: list[np.ndarray], label: np.ndarray, split: np.ndarray, cluster: np.ndarray,
                split_names: list[str], long_eval: np.ndarray | None = None) -> None:
    """A pre-tokenized corpus from token id arrays, for tests and small experiments."""
    # texts: (l_i,) each of n; label, split, cluster, long_eval: (n,)
    directory.mkdir(parents=True, exist_ok=True)
    length = np.asarray([len(ids) for ids in texts], dtype=np.int64)  # (n,)
    offsets = np.concatenate([[0], np.cumsum(length)]).astype(np.int64)  # (n + 1,)
    tokens = np.concatenate(texts).astype(np.uint16)  # (N,)
    np.save(directory / TOKENS_FILE, tokens)
    np.save(directory / OFFSETS_FILE, offsets)
    np.savez(directory / ROWS_FILE, label=label.astype(np.int8), split=split.astype(np.int8), cluster=cluster.astype(np.int64), length=length,
             long_eval=np.zeros(len(texts), dtype=bool) if long_eval is None else long_eval.astype(bool))
    meta = {"rows": len(texts), "tokens": int(offsets[-1]), "splits": split_names, "corpus": "synthetic",
            "tokenizer": {"repository": reference.TOKENIZER_REPOSITORY, "revision": reference.TOKENIZER_REVISION,
                          "sha256": reference.TOKENIZER_SHA256}}
    (directory / META_FILE).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- texts, presented at every draw


def write_texts(corpus: Path, out: Path, batch_rows: int = 65536) -> None:
    """Every corpus text's UTF-8 bytes back to back (`texts.bin`), with each id's start and length in bytes
    (`text_starts.npy`, `text_lengths.npy`), copied from Arrow's own buffers, so the presenter reads any text at once."""
    _, ids, _ = corpus_lengths(corpus)
    n = len(ids)
    starts = np.full(n, -1, dtype=np.int64)  # (n,)
    lengths = np.zeros(n, dtype=np.int64)  # (n,)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out / (TEXTS_FILE + ".partial"), "wb") as handle:
        for path in parquet_files(corpus):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=["id", "text"]):
                text = batch.column("text")
                if text.null_count:
                    raise ValueError(f"{path}: {text.null_count} texts are null")
                width = np.int64 if pa.types.is_large_string(text.type) else np.int32
                buffers = text.buffers()
                offsets = np.frombuffer(buffers[1], dtype=width)[text.offset : text.offset + len(text) + 1].astype(np.int64)  # (r + 1,)
                handle.write(memoryview(buffers[2])[offsets[0] : offsets[-1]])
                rows = batch.column("id").to_numpy()  # (r,)
                starts[rows] = written + offsets[:-1] - offsets[0]
                lengths[rows] = np.diff(offsets)
                written += int(offsets[-1] - offsets[0])
    if (starts < 0).any():
        raise ValueError(f"{int((starts < 0).sum())} corpus ids have no text")
    np.save(out / TEXT_STARTS_FILE, starts)
    np.save(out / TEXT_LENGTHS_FILE, lengths)
    os.replace(out / (TEXTS_FILE + ".partial"), out / TEXTS_FILE)


class TextStore:
    """The corpus's texts by id (`write_texts`), memory-mapped."""

    def __init__(self, directory: Path) -> None:
        self.data = np.memmap(directory / TEXTS_FILE, dtype=np.uint8, mode="r")  # (B,)
        self.starts = np.load(directory / TEXT_STARTS_FILE)  # (n,)
        self.lengths = np.load(directory / TEXT_LENGTHS_FILE)  # (n,)

    def __len__(self) -> int:
        return len(self.starts)

    def text(self, row: int) -> str:
        start = int(self.starts[row])
        return bytes(self.data[start : start + int(self.lengths[row])]).decode("utf-8")


PRESENTER: dict[str, Any] = {}  # a presenter process's text store, tokenizer and synonym table


def start_presenter(texts: Path, synonyms: Path | None, tokenizer_path: Path | None) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # many processes, one thread each
    PRESENTER.update(store=TextStore(texts), tokenizer=reference.load_tokenizer(tokenizer_path),
                     synonyms=None if synonyms is None else load_synonyms(synonyms))


def present_and_tokenize(job: tuple[np.ndarray, np.ndarray, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rows' texts each presented from its own seed (`presentation.presented`) and tokenized, at most `max_tokens`
    tokens each: token counts (r,) and tokens back to back (t,) uint16."""
    rows, seeds, max_tokens = job
    read = [presented(PRESENTER["store"].text(int(row)), np.random.default_rng(int(seed)), PRESENTER["synonyms"])
            for row, seed in zip(rows, seeds, strict=True)]
    encodings = PRESENTER["tokenizer"].encode_batch(read, add_special_tokens=False)
    lengths = np.asarray([min(len(encoding.ids), max_tokens) for encoding in encodings], dtype=np.int64)  # (r,)
    assert lengths.min(initial=1) > 0, "a presented text keeps a token"
    tokens = np.fromiter((index for encoding in encodings for index in encoding.ids[:max_tokens]), dtype=np.uint16, count=int(lengths.sum()))  # (t,)
    return lengths, tokens


class Presenter:
    """Training texts presented anew at every draw (`presentation.presented`, Logan, 2026-11-19: "random each epoch so
    can get different versions of each document"), in worker processes: each drawn text gets a fresh seed from
    `generator`, so a seed gives the same run and no two draws of a text share a presentation by design. Validation,
    test, calibration and the reference bank never pass through it: they read the token store as stored."""

    def __init__(self, texts: Path, synonyms: Path | None, max_tokens: int, workers: int, generator: np.random.Generator,
                 tokenizer_path: Path | None = None) -> None:
        self.max_tokens, self.workers, self.generator = max_tokens, workers, generator
        # spawned, not forked: the parent's Arrow, tokenizer and CUDA threads could hold a lock a forked child never sees released
        self.pool = get_context("spawn").Pool(workers, initializer=start_presenter, initargs=(texts, synonyms, tokenizer_path))

    def ids(self, rows: np.ndarray) -> list[np.ndarray]:
        """Each row's presented tokens, in order: (b,) to b arrays (l_i,) int64."""
        seeds = self.generator.integers(0, 2**63 - 1, size=len(rows))  # (b,)
        parts = [part for part in np.array_split(np.arange(len(rows)), 4 * self.workers) if len(part)]
        out: list[np.ndarray] = []
        for lengths, tokens in self.pool.map(present_and_tokenize, [(rows[part], seeds[part], self.max_tokens) for part in parts]):
            out.extend(np.split(tokens.astype(np.int64), np.cumsum(lengths)[:-1]))
        return out

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()


# ---------------------------------------------------------------- sampling


class ClusterSampler:
    """Draws texts class-balanced, restricted to texts of at most `max_tokens` tokens that the teacher embedded (`present`),
    never a `long_eval` document.

    By default a class's texts are drawn uniformly, as the teacher's trainer draws them: the corpus balances the
    origins inside each cell of format, language and length, and drawing a class's clusters uniformly would give each
    origin its own mix of those cells, which a model can read as the class. `clusters_uniform` draws a class's
    e-means clusters uniformly and a member at random within one instead.
    """

    def __init__(self, store: TokenStore, split: str, max_tokens: int, present: np.ndarray, clusters_uniform: bool = False,
                 origins: tuple[int, ...] = (0, 1, 2)) -> None:
        # present: (n,) bool; origins: the labels the corpus holds (the first corpus has human and ai only)
        eligible = store.split_mask(split) & (store.length <= max_tokens) & present & ~store.long_eval  # (n,)
        self.clusters_uniform = clusters_uniform
        self.members: list[list[np.ndarray]] = []  # per class, per cluster, the rows
        for origin in origins:
            rows = np.flatnonzero(eligible & (store.label == origin))  # (n_o,)
            if len(rows) == 0:
                raise ValueError(f"split {split!r} has no texts of origin {reference.ORIGINS[origin]} under {max_tokens} tokens")
            order = np.argsort(store.cluster[rows], kind="stable")  # (n_o,)
            rows = rows[order]
            boundaries = np.flatnonzero(np.diff(store.cluster[rows])) + 1  # (c_o - 1,)
            self.members.append(np.split(rows, boundaries))

    @property
    def rows(self) -> np.ndarray:
        """Every eligible row: (n_e,)."""
        return np.concatenate([np.concatenate(clusters) for clusters in self.members])  # (n_e,)

    def draw(self, texts: int, generator: np.random.Generator) -> np.ndarray:
        """`texts` row indices, an equal share of each origin (the remainder to the first): (b,)."""
        rows = []
        classes = len(self.members)
        for origin, clusters in enumerate(self.members):
            count = texts // classes + (1 if origin < texts % classes else 0)
            if self.clusters_uniform:
                chosen = generator.integers(0, len(clusters), count)  # (count,)
                rows.extend(int(generator.choice(clusters[index])) for index in chosen)
                continue
            sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.int64)  # (c_o,)
            ends = np.cumsum(sizes)  # (c_o,)
            picks = generator.integers(0, int(ends[-1]), count)  # (count,) positions among the class's texts
            owners = np.searchsorted(ends, picks, side="right")  # (count,) the cluster holding each position
            rows.extend(int(clusters[owner][pick - (ends[owner] - sizes[owner])]) for owner, pick in zip(owners, picks, strict=True))

        return np.asarray(rows, dtype=np.int64)  # (b,)


class EpochSampler:
    """Draws texts uniformly over all of a split's eligible texts (`ClusterSampler`'s), whatever their origin, in
    epochs: each epoch a fresh random order of every text, so each is drawn once an epoch (Logan, 2026-11-19: "uniform
    over all texts"; "random each epoch")."""

    def __init__(self, store: TokenStore, split: str, max_tokens: int, present: np.ndarray) -> None:
        self.all_rows = np.sort(ClusterSampler(store, split, max_tokens, present).rows)  # (n_e,)
        self.order = np.zeros(0, dtype=np.int64)
        self.position = 0
        self.epoch = -1

    def draw(self, texts: int, generator: np.random.Generator) -> np.ndarray:
        """The next `texts` rows of the epoch's order, a new order begun when it runs out: (b,)."""
        drawn = []
        while texts > 0:
            if self.position == len(self.order):
                self.order, self.position, self.epoch = generator.permutation(self.all_rows), 0, self.epoch + 1
            take = min(texts, len(self.order) - self.position)
            drawn.append(self.order[self.position : self.position + take])
            self.position += take
            texts -= take
        return np.concatenate(drawn)  # (b,)


def cluster_stratified_rows(store: TokenStore, split: str, per_origin: int, generator: np.random.Generator,
                            origins: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    """Up to `per_origin` distinct rows of each origin of a split, spread uniformly over clusters, never a `long_eval`
    document: (m,)."""
    chosen = []
    mask = store.split_mask(split) & ~store.long_eval  # (n,)
    for origin in origins:
        rows = np.flatnonzero(mask & (store.label == origin))  # (n_o,)
        if len(rows) <= per_origin:
            chosen.append(rows)
            continue
        # A random order inside each cluster, then round-robin over clusters: every cluster gives one before any gives two.
        rows = generator.permutation(rows)
        rank = np.zeros(len(rows), dtype=np.int64)  # (n_o,)
        order = np.argsort(store.cluster[rows], kind="stable")  # (n_o,)
        sorted_clusters = store.cluster[rows][order]  # (n_o,)
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_clusters)) + 1]  # (c_o,)
        rank[order] = np.arange(len(rows)) - np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
        tie = generator.random(len(rows))  # (n_o,)
        chosen.append(rows[np.lexsort((tie, rank))[:per_origin]])

    return np.concatenate(chosen)  # (m,)


# Bank recipes: the split each origin's bank texts are drawn from, human, ai and mixed in that order. `mixed_out` takes
# human and AI texts from the training rows and mixed texts from `bank_draw`, texts no training step saw. `reference`
# takes all three from the withheld `bank` set, clusters no other set holds (Litmus 1.1).
BANK_RECIPES = {"mixed_out": ("train", "train", "bank_draw"), "reference": ("bank", "bank", "bank")}


def uniform_order(count: int, seed: int, origin: int) -> np.ndarray:
    """A seeded random order of one origin's `count` texts: (count,). A bank of size N is the first N, so a larger bank
    holds every smaller one."""
    return np.random.default_rng([seed, origin, 0]).permutation(count)  # (count,)


def bank_rows_of_splits(store: TokenStore, splits: tuple[str, str, str], size: int, seed: int, recipe: str) -> np.ndarray:
    """The store rows of a bank of `size` texts per origin, human, ai and mixed in that order (the release's bank
    grouping): each origin's texts are those of its split in `splits`, never a `long_eval` document, the first `size` in
    `uniform_order` for `seed`. A split the store lacks, or too few texts of an origin in it, fails with a message that
    names the recipe."""
    rows = []
    for origin, split in enumerate(splits):
        if split not in store.splits:
            raise ValueError(f"the bank recipe {recipe!r} takes {reference.ORIGINS[origin]} texts from the {split!r} split, "
                             f"which this corpus does not have (its splits: {store.splits})")
        members = np.flatnonzero(store.split_mask(split) & ~store.long_eval & (store.label == origin))  # (m_o,)
        if len(members) < size:
            raise ValueError(f"the bank recipe {recipe!r} needs {size} {reference.ORIGINS[origin]} texts from the {split!r} split, "
                             f"which holds {len(members)}")
        rows.append(members[uniform_order(len(members), seed, origin)[:size]])

    return np.concatenate(rows)  # (3 * size,)


def recipe_bank_rows(store: TokenStore, recipe: str, size: int, seed: int) -> np.ndarray:
    """The store rows of the bank a `BANK_RECIPES` recipe gives at `size` texts per origin: (3 * size,)."""
    if recipe not in BANK_RECIPES:
        raise ValueError(f"no bank recipe {recipe!r}; known: {sorted(BANK_RECIPES)}")

    return bank_rows_of_splits(store, BANK_RECIPES[recipe], size, seed, recipe)  # (3 * size,)


# ---------------------------------------------------------------- packing


@dataclass
class Placement:
    row: int  # index into the step's rows
    start: int  # first position in the micro-batch's stream


def pack(lengths: np.ndarray, budget: int, block_size: int) -> list[list[Placement]]:
    """Place texts into micro-batches of at most `budget` positions (a text longer than the budget gets its own).

    A text longer than W/2 starts at a multiple of W, so its blocks are the stream's blocks under both shifts; texts
    of at most W/2 share half-blocks (W/2-aligned spans of W/2 positions), which no block of either shift splits.
    Long texts go first, largest first; short ones fill the free room of half-blocks, first fit, largest first.
    """
    # lengths: (b,)
    W, half = block_size, block_size // 2
    assert budget % W == 0, "the budget is a whole number of blocks"
    batches: list[list[Placement]] = []
    ends: list[int] = []  # the stream length each micro-batch has reached, in half-block units
    room: list[dict[int, int]] = []  # per micro-batch: half-block index to free positions
    order = np.argsort(-lengths, kind="stable")  # (b,)
    for row in order:
        n = int(lengths[row])
        if n > half:
            halves = -(-n // half)
            target = next((index for index, end in enumerate(ends) if (end + end % 2 + halves) * half <= budget), None)
            if target is None:
                batches.append([])
                ends.append(0)
                room.append({})
                target = len(batches) - 1
            start_half = ends[target] + ends[target] % 2
            if ends[target] % 2:
                room[target][ends[target]] = half  # the half-block skipped to reach a block boundary
            batches[target].append(Placement(int(row), start_half * half))
            ends[target] = start_half + halves
            if n % half:
                room[target][start_half + halves - 1] = half - n % half
            continue
        placed = False
        for target in range(len(batches)):
            fit = next((index for index, free in sorted(room[target].items()) if free >= n), None)
            if fit is not None:
                batches[target].append(Placement(int(row), fit * half + half - room[target][fit]))
                room[target][fit] -= n
                placed = True
                break
            if (ends[target] + 1) * half <= budget:
                batches[target].append(Placement(int(row), ends[target] * half))
                room[target][ends[target]] = half - n
                ends[target] += 1
                placed = True
                break
        if not placed:
            batches.append([Placement(int(row), 0)])
            ends.append(1)
            room.append({0: half - n})

    return batches


def packed_batch(placements: list[Placement], texts: list[np.ndarray], block_size: int, multiple: int) -> PackedBatch:
    """A micro-batch's stream, padded to a multiple of `multiple` positions (itself a multiple of W)."""
    # texts: (l_i,) each, the step's texts
    assert multiple % block_size == 0, "streams pad to whole blocks"
    end = max(placement.start + len(texts[placement.row]) for placement in placements)
    t = -(-end // multiple) * multiple
    tokens = np.zeros(t, dtype=np.int64)  # (t,)
    segment = np.full(t, -1, dtype=np.int64)  # (t,)
    position = np.zeros(t, dtype=np.int64)  # (t,)
    for local, placement in enumerate(placements):
        ids = texts[placement.row]  # (l,)
        span = slice(placement.start, placement.start + len(ids))
        assert (segment[span] == -1).all(), "texts never overlap in a stream"
        assert len(ids) <= block_size // 2 or placement.start % block_size == 0, "a long text starts on a block boundary"
        assert len(ids) > block_size // 2 or placement.start // (block_size // 2) == (placement.start + len(ids) - 1) // (block_size // 2), \
            "a short text stays inside one half-block"
        tokens[span] = ids
        segment[span] = local
        position[span] = np.arange(len(ids))

    return PackedBatch(torch.from_numpy(tokens), torch.from_numpy(segment), torch.from_numpy(position), len(placements))


# ---------------------------------------------------------------- teacher targets and steps


@dataclass
class TeacherFiles:
    """The teacher's export for distillation (`litmus_teacher.export`'s `manifest.json`), row `i` for corpus id `i`.

    The teacher's embedding is the head's input: the listed encoders' embeddings side by side, each mean then
    variance. Rows the teacher did not embed hold zeros and are marked absent in `present.npy`.
    """

    embeddings: list[np.ndarray]  # each (n, 2D_e) float16, memory-mapped
    query_key: np.ndarray  # (n, 2w_T) float16, unit query then unit key
    present: np.ndarray  # (n,) bool
    scale: float
    bias: float
    embedding_mean: np.ndarray  # (2D,) float32, over training rows
    embedding_std: np.ndarray  # (2D,) float32
    embedding_index: np.ndarray | None = None  # (n,) the embeddings' row of each corpus row, when they hold a larger corpus
    missing: bool = False  # no teacher: a labels-only run's zero targets (`absent`)

    @property
    def embedding_width(self) -> int:
        return sum(values.shape[1] for values in self.embeddings)  # 2D

    def embedding_rows(self, rows: np.ndarray) -> np.ndarray:
        """Ascending rows' teacher embeddings: (b,) to (b, 2D) float32."""
        # rows: (b,)
        stored = rows if self.embedding_index is None else self.embedding_index[rows]  # (b,) ascending, as the index is
        return np.concatenate([np.asarray(values[stored], dtype=np.float32) for values in self.embeddings], axis=1)  # (b, 2D)

    @classmethod
    def load(cls, export: Path, train_rows: np.ndarray, generator: np.random.Generator) -> TeacherFiles:
        """Memory-map the arrays the manifest names, and standardize the embedding from 100,000 training rows.

        `embedding_rows` in the manifest (optional) names an int64 array, ascending, of each corpus row's row in the
        embeddings, when those hold a larger corpus than the pre-tokenized one (the proof of concept's subsample).
        """
        # train_rows: (t,)
        manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))

        def located(name: str) -> Path:
            path = Path(name)
            return path if path.is_absolute() or path.exists() else export / path

        embeddings = [np.load(located(entry["path"]), mmap_mode="r") for entry in manifest["embeddings"]]  # each (N, 2D_e)
        query_key = np.load(located(manifest["query_key"]["path"]), mmap_mode="r")  # (n, 2w_T)
        present = np.load(located(manifest["query_key"]["present"])).astype(bool)  # (n,)
        index = np.load(located(manifest["embedding_rows"])).astype(np.int64) if "embedding_rows" in manifest else None  # (n,)
        if index is not None and (len(index) != query_key.shape[0] or (np.diff(index) <= 0).any()
                                  or any(index[-1] >= values.shape[0] for values in embeddings)):
            raise ValueError(f"{export}: `embedding_rows` must give each row an ascending row inside the embeddings")
        if (index is None and any(values.shape[0] != query_key.shape[0] for values in embeddings)) or present.shape[0] != query_key.shape[0]:
            raise ValueError(f"the teacher's files in {export} disagree on the row count")
        trained = train_rows[present[train_rows]]  # (t,)
        sample = np.sort(generator.choice(trained, min(100_000, len(trained)), replace=False))  # (s,)
        files = cls(embeddings, query_key, present, float(manifest["head"]["scale"]), float(manifest["head"]["bias"]),
                    np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), index)
        drawn = files.embedding_rows(sample)  # (s, 2D)
        files.embedding_mean, files.embedding_std = drawn.mean(axis=0), drawn.std(axis=0) + 1e-6  # (2D,) each

        return files

    @classmethod
    def absent(cls, rows: int) -> TeacherFiles:
        """No teacher, for a run from labels alone: every row present, with zero targets two columns wide that the
        labels-only loss weights by zero, so no teacher file is read or needed."""
        return cls([np.zeros((rows, 2), dtype=np.float16)], np.zeros((rows, 2), dtype=np.float16), np.ones(rows, dtype=bool), 1.0, 0.0,
                   np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), missing=True)


@dataclass
class Step:
    """One optimizer step's texts: micro-batches whose texts, in order, are `rows`."""

    rows: np.ndarray  # (b,) corpus rows, in micro-batch order
    micro_batches: list[PackedBatch]
    labels: torch.Tensor  # (b,) int64
    teacher_embeddings: torch.Tensor  # (b, 2D) float32, standardized
    teacher_query_key: torch.Tensor  # (b, 2w_T) float32
    tokens: int  # real tokens


@dataclass
class Cutter:
    """Training texts cut mid-sentence, as a text cut off by a length limit or a selection reads, at the same rate for
    every origin so that how a text ends says nothing of who wrote it: Litmus 1.0.0, trained on passages cut only at
    sentence starts, called a third of mixed texts human once their last sentence was cut off
    (experiments/tooling/quill/2026-11-19_confounders). Each text of at least `MIN_TOKENS` loses, with probability
    `end`, its tokens from a word start in its last quarter (at most `SPAN` tokens back), and with probability `start`,
    its tokens before a word start in its first quarter. A word start is a token that begins with a space or a line
    break in the byte-level vocabulary."""

    word_start: np.ndarray  # (V,) bool
    end: float
    start: float
    generator: np.random.Generator
    MIN_TOKENS = 16
    SPAN = 64

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, end: float, start: float, generator: np.random.Generator) -> Cutter:
        vocabulary = tokenizer.get_vocab()
        word_start = np.zeros(max(vocabulary.values()) + 1, dtype=bool)  # (V,)
        for token, index in vocabulary.items():
            word_start[index] = token.startswith(("Ġ", "Ċ", "ĉ"))  # a byte-level space, line feed or tab first
        return cls(word_start, end, start, generator)

    def cut(self, ids: np.ndarray) -> np.ndarray:
        """A text's ids, perhaps cut at either end: (l,) to (l',)."""
        n = len(ids)
        if n < self.MIN_TOKENS:
            return ids
        reach = min(self.SPAN, n // 4)
        begin, stop = 0, n
        if self.generator.random() < self.end:
            places = np.flatnonzero(self.word_start[ids[n - reach:n]]) + n - reach  # word starts in the last quarter
            if len(places):
                stop = int(self.generator.choice(places))
        if self.generator.random() < self.start:
            places = np.flatnonzero(self.word_start[ids[1:reach + 1]]) + 1  # word starts in the first quarter
            if len(places):
                begin = int(self.generator.choice(places))
        return ids[begin:stop]


def build_step(rows: np.ndarray, store: TokenStore, teacher: TeacherFiles, budget: int, block_size: int, multiple: int,
               cutter: Cutter | None = None, presenter: Presenter | None = None) -> Step:
    """Pack drawn rows into micro-batches and gather their targets; with a `presenter`, each text presented anew,
    and with a `cutter`, each text perhaps cut first."""
    # rows: (b,)
    texts = presenter.ids(rows) if presenter is not None else [store.ids(int(row)) for row in rows]
    if cutter is not None:
        texts = [cutter.cut(ids) for ids in texts]
    length = np.asarray([len(ids) for ids in texts], dtype=np.int64)  # (b,)
    plans = pack(length, budget, block_size)
    micro_batches = [packed_batch(plan, texts, block_size, multiple) for plan in plans]
    ordered = np.asarray([rows[placement.row] for plan in plans for placement in plan], dtype=np.int64)  # (b,)
    labels, embeddings, query_key = teacher_targets(ordered, store, teacher)  # (b,), (b, 2D), (b, 2w_T)

    return Step(ordered, micro_batches, labels, embeddings, query_key, int(length.sum()))


def teacher_targets(rows: np.ndarray, store: TokenStore, teacher: TeacherFiles) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Labels (b,), standardized teacher embeddings (b, 2D) and teacher queries and keys (b, 2w_T) of rows, in order."""
    # rows: (b,)
    assert teacher.present[rows].all(), "every distilled row has a teacher target"
    gather = np.sort(rows)  # memory maps read fastest in order
    inverse = np.searchsorted(gather, rows)  # (b,); a row drawn twice reads the same values
    embeddings = (teacher.embedding_rows(gather)[inverse] - teacher.embedding_mean) / teacher.embedding_std  # (b, 2D)
    query_key = np.asarray(teacher.query_key[gather], dtype=np.float32)[inverse]  # (b, 2w_T)

    return torch.from_numpy(store.label[rows].astype(np.int64)), torch.from_numpy(embeddings), torch.from_numpy(query_key)  # (b,), (b, 2D), (b, 2w_T)


def pinned(step: Step) -> Step:
    """The step with its micro-batches in page-locked memory, for asynchronous copies to the GPU."""
    step.micro_batches = [PackedBatch(batch.tokens.pin_memory(), batch.segment.pin_memory(), batch.position.pin_memory(), batch.texts)
                          for batch in step.micro_batches]
    return step


def prefetched(make: Callable[[], Step], count: int, depth: int, pin: bool) -> Iterator[Step]:
    """`count` steps from `make`, built on one background thread `depth` ahead, so the GPU never waits on packing or
    memory maps. One worker keeps the draws in order, so a seed gives the same steps; a worker's error is raised here."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="litmus-steps") as pool:
        ahead: deque[Future[Step]] = deque()
        for index in range(count):
            while len(ahead) < depth and index + len(ahead) < count:
                ahead.append(pool.submit(lambda: pinned(make()) if pin else make()))
            yield ahead.popleft().result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="a Parquet file or a directory of them")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None, help="a local tokenizer.json; default: the pinned Hub revision")
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--split", type=Path, default=None,
                        help="the teacher's split.parquet (id, split, cluster); default: the corpus's own split and cluster columns")
    parser.add_argument("--tokens-only", action="store_true", help="write the tokens and stop; run again with --split for the rows")
    parser.add_argument("--reading", choices=reference.READINGS, default="standard",
                        help="read each text this way before tokenizing (reference.READINGS; the corpus's texts are already standard)")
    parser.add_argument("--texts", action="store_true",
                        help="write the text store the training presenter reads (write_texts) to --out, and stop")
    parser.add_argument("--workers", type=int, default=32, help="processes reading and tokenizing, for a reading other than the standard")
    arguments = parser.parse_args()
    if arguments.texts:
        write_texts(arguments.corpus, arguments.out)
        return
    reading = arguments.reading
    if not (arguments.out / TOKENS_FILE).exists():
        if reading == "standard":
            tokenize_corpus(arguments.corpus, arguments.out, arguments.tokenizer, arguments.batch_rows)
        else:
            tokenize_reading(arguments.corpus, arguments.out, arguments.tokenizer, arguments.batch_rows, reading, arguments.workers)
    if not arguments.tokens_only:
        write_rows(arguments.corpus, arguments.out, arguments.split, reading)


if __name__ == "__main__":
    main()
