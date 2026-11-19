"""Embed the corpus with a frozen encoder: the mean and variance of its last hidden states over each text's tokens.

Output is one float16 `.npy` in `id` order, `(n, 2d)`, mean then variance (`embeddings.npy`), beside an int32
`tokens.npy` holding each row's content tokens as this encoder counts them after any cap, or -1 for a row not yet
embedded. Which states count, and how a text past the encoder's context is windowed, is `moments`'s rule.

Throughput comes from five choices:

- **Packing.** With a flash-attention implementation, a batch is one row of up to `--batch-tokens` tokens holding
  many pieces back to back, each with its own positions and its own `cu_seqlens` segment, so no position is padding
  and no piece attends to another (transformers' padding-free path, `cu_seq_lens_q/k`). Under SDPA, pieces are
  length-sorted into padded `(b, l)` batches instead. The pilot checks packed states against unpacked float32 ones.
- **Tokenization ahead of the GPU.** DataLoader workers each read whole Parquet row groups, tokenize, window, sort
  and pack, and hand over pinned tensors; the GPU loop only copies and runs.
- **bfloat16 autocast** over float32 weights. bfloat16 states depend on the batch a text shares, and the first
  checker's head was sensitive to that; the teacher accepts it for throughput (the shipped student is the
  deterministic model), and the pilot measures the drift between two packings and against float32.
- **Moments on the GPU in float64**, reduced per piece by `index_add_`, copied back without blocking, and combined
  per text on the host when its slice ends.
- **`torch.compile`**, optional (`--compile`): with it, packed batches are padded to exactly `--batch-tokens` so the
  shape never changes. Run the pilot with and without it and keep the faster.

Resumable by slice: a slice is a run of rows inside one row group, cut at `--slice-rows` rows or `--slice-tokens`
corpus tokens, and its rows are written, flushed and marked done together.

The teacher's encoder is ModernBERT-base; ModernBERT-large and mmBERT-base are comparison arms embedded on a subsample
(`--ids`) and read against it on validation. With `--reuse`, ModernBERT-base rows the first corpus embedded with the
same text copy its vectors (`reuse`), once the pilot's reuse check has passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from .moments import combine_moments, window_count_mask, window_spans
from .reuse import (
    ALIGNMENT_FILE,
    CHECK_ROWS,
    REUSE_COLUMNS,
    REUSE_ENCODER,
    REUSE_MARKER,
    REUSED_FILE,
    Alignment,
    copy_reused,
    decide_reuse,
    reusable_rows,
    vector_drift,
)
from .strata import LENGTH_EDGES, length_bins, write_json, write_summary


@dataclass(frozen=True)
class EncoderSpec:
    """A frozen encoder pinned to one revision."""

    name: str
    revision: str
    width: int  # d
    context: int  # positions, special tokens included


ENCODERS = {
    "modernbert_base": EncoderSpec("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8", 768, 8192),
    "modernbert_large": EncoderSpec("answerdotai/ModernBERT-large", "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13", 1024, 8192),
    "mmbert_base": EncoderSpec("jhu-clsp/mmBERT-base", "c5955035435e2bf121cde7f3c8863ef52ff35d82", 768, 8192),
}
PACKED_ATTENTION = ("flash_attention_2", "flash_attention_3")
EMBEDDINGS_FILE = "embeddings.npy"
TOKENS_FILE = "tokens.npy"
META_FILE = "meta.json"
DONE_DIRECTORY = "slices"
TOKENIZE_CHUNK = 256  # texts per tokenizer call inside a worker
MAX_ROW_GROUP_BYTES = 1 << 30  # each worker holds one row group's text in memory
PILOT_ROWS = 20_000
PILOT_CHECK_ROWS = 64
PILOT_DRIFT_ROWS = 1_000
PILOT_WARMUP_BATCHES = 20
# Pieces are timed in three classes by their longest piece; the projection prices each class at its own rate.
RATE_CLASSES = (512, 2048, 8192)
# The packing check: packed bfloat16 against unpacked float32 states of the same texts.
MEAN_COSINE_FLOOR = 0.999
VARIANCE_COSINE_FLOOR = 0.99


@dataclass(frozen=True)
class Slice:
    """Rows `[offset, offset + length)` of one row group, embedded and marked done together."""

    file: int
    row_group: int
    offset: int
    length: int
    tokens: int  # selected rows' corpus tokens, for planning

    @property
    def key(self) -> str:
        return f"{self.file:04d}-{self.row_group:05d}-{self.offset:08d}"


@dataclass
class EmbedSettings:
    """Everything that changes a stored vector, recorded in `meta.json` and checked on resume."""

    encoder: str
    attention: str
    batch_tokens: int
    overlap: int
    max_text_tokens: int
    compile: bool

    @property
    def spec(self) -> EncoderSpec:
        return ENCODERS[self.encoder]

    @property
    def packed(self) -> bool:
        return self.attention in PACKED_ATTENTION


@dataclass
class BatchTiming:
    """GPU time of one batch, read from CUDA events once the run ends."""

    tokens: int
    longest: int
    start: torch.cuda.Event
    stop: torch.cuda.Event


@dataclass
class SliceTotals:
    """Per-row count, sum and sum of squares of one slice, accumulated in float64 on the host."""

    count: np.ndarray  # (r,)
    total: np.ndarray  # (r, d)
    squares: np.ndarray  # (r, d)
    seen: int = 0


@dataclass
class EmbedStats:
    rows: int = 0
    slices: int = 0
    processed_tokens: int = 0
    counted_tokens: int = 0
    windowed_rows: int = 0
    truncated_rows: int = 0
    seconds: float = 0.0
    timings: list[BatchTiming] = field(default_factory=list)


def parquet_files(corpus: Path) -> list[Path]:
    """The corpus's Parquet files in a fixed order: one file, or every `*.parquet` under a directory."""
    if corpus.is_file():
        return [corpus]
    files = sorted(corpus.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {corpus}")

    return files


def read_columns(corpus: Path, columns: Sequence[str]) -> pa.Table:
    """The named columns of every row, in file and row order."""
    return pa.concat_tables([pq.read_table(path, columns=list(columns)) for path in parquet_files(corpus)])


def dense_row_count(ids: np.ndarray) -> int:
    """The row count `n` of an `id` column that must be a permutation of 0..n-1, the embeddings' row order."""
    # ids: (n,)
    n = len(ids)
    if n == 0 or ids.min() != 0 or ids.max() != n - 1 or len(np.unique(ids)) != n:
        raise ValueError("corpus ids must be a permutation of 0..n-1, since they index the embedding rows")

    return n


def plan_slices(corpus: Path, selected: np.ndarray, slice_rows: int, slice_tokens: int) -> list[Slice]:
    """Cut every row group into slices of at most `slice_rows` rows and about `slice_tokens` selected tokens."""
    # selected: (n,) bool, indexed by id
    plan: list[Slice] = []
    for file_index, path in enumerate(parquet_files(corpus)):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).total_byte_size
            if size > MAX_ROW_GROUP_BYTES:
                raise ValueError(f"{path} row group {group} holds {size / 2**30:.1f} GiB; rewrite the corpus with row "
                                 f"groups under {MAX_ROW_GROUP_BYTES / 2**30:.0f} GiB, since each worker holds one whole")
            table = parquet.read_row_group(group, columns=["id", "tokens"])
            ids = table["id"].to_numpy()  # (g,)
            tokens = np.where(selected[ids], table["tokens"].to_numpy(), 0)  # (g,)
            offset = 0
            while offset < len(ids):
                cumulative = np.cumsum(tokens[offset : offset + slice_rows])  # (s,)
                length = max(1, int(np.searchsorted(cumulative, slice_tokens, side="right")))
                length = min(length, len(cumulative))
                if selected[ids[offset : offset + length]].any():
                    plan.append(Slice(file_index, group, offset, length, int(cumulative[length - 1])))
                offset += length

    return plan


def special_token_ids(tokenizer: object) -> tuple[int, int]:
    """The start and end token the tokenizer wraps a text in, read from its own template, not assumed."""
    wrapped = tokenizer("", add_special_tokens=True)["input_ids"]
    if len(wrapped) != 2:
        raise ValueError(f"expected the empty text to encode as [start, end], got {wrapped}")

    return int(wrapped[0]), int(wrapped[1])


def load_tokenizer(spec: EncoderSpec) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.name, revision=spec.revision)


class TokenizedSlices(IterableDataset):
    """Worker side: read row groups, tokenize, window, sort and pack each slice into GPU batches."""

    def __init__(self, corpus: Path, plan: list[Slice], selection_path: Path, settings: EmbedSettings,
                 shuffle_seed: int | None = None) -> None:
        super().__init__()
        self.files = parquet_files(corpus)
        self.plan = plan
        self.selection_path = selection_path
        self.settings = settings
        self.shuffle_seed = shuffle_seed  # the pilot's second packing: same texts, other neighbors

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int | bool]]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        worker_info = get_worker_info()
        worker, workers = (0, 1) if worker_info is None else (worker_info.id, worker_info.num_workers)
        groups = sorted({(item.file, item.row_group) for item in self.plan})
        mine = set(groups[worker::workers])
        selected = np.load(self.selection_path, mmap_mode="r")  # (n,) bool
        tokenizer = load_tokenizer(self.settings.spec)
        start_id, end_id = special_token_ids(tokenizer)
        pad_id = int(tokenizer.pad_token_id)

        by_group: dict[tuple[int, int], list[tuple[int, Slice]]] = {}
        for index, item in enumerate(self.plan):
            if (item.file, item.row_group) in mine:
                by_group.setdefault((item.file, item.row_group), []).append((index, item))
        for (file_index, group), items in by_group.items():
            table = pq.ParquetFile(self.files[file_index]).read_row_group(group, columns=["id", "text"])
            ids = table["id"].to_numpy()  # (g,)
            texts = table["text"]
            for index, item in items:
                rows = np.arange(item.offset, item.offset + item.length)  # (s,)
                rows = rows[np.asarray(selected[ids[rows]])]  # (r,)
                yield from self._slice_batches(index, ids[rows], [texts[int(row)].as_py() for row in rows],
                                               tokenizer, start_id, end_id, pad_id)

    def _slice_batches(self, slice_index: int, ids: np.ndarray, texts: list[str], tokenizer: object, start_id: int,
                       end_id: int, pad_id: int) -> Iterator[dict[str, torch.Tensor | int | bool]]:
        # ids: (r,)
        settings, spec = self.settings, self.settings.spec
        content: list[np.ndarray] = []
        for begin in range(0, len(texts), TOKENIZE_CHUNK):
            encoded = tokenizer(texts[begin : begin + TOKENIZE_CHUNK], add_special_tokens=False,
                                return_attention_mask=False)["input_ids"]
            content.extend(np.asarray(row[: settings.max_text_tokens], dtype=np.int32) for row in encoded)
        content_tokens = np.array([len(row) for row in content], dtype=np.int64)  # (r,)

        pieces: list[tuple[int, np.ndarray, np.ndarray]] = []  # (slice row, token ids (l,), counted (l,))
        for row, tokens in enumerate(content):
            for window in window_spans(len(tokens), spec.context, settings.overlap):
                piece = np.concatenate([[start_id], tokens[window.start : window.stop], [end_id]]).astype(np.int64)
                pieces.append((row, piece, window_count_mask(window)))
        order = sorted(range(len(pieces)), key=lambda index: -len(pieces[index][1]))
        if self.shuffle_seed is not None:
            order = list(np.random.default_rng(self.shuffle_seed).permutation(order))

        batches = pack_pieces([len(pieces[index][1]) for index in order], settings.batch_tokens, settings.packed)
        for number, members in enumerate(batches):
            chosen = [pieces[order[index]] for index in members]
            batch = (packed_batch(chosen, pad_id, spec.context, settings.batch_tokens if settings.compile else None)
                     if settings.packed else padded_batch(chosen, pad_id))
            batch.update({"slice": slice_index, "slice_rows": len(content), "final": number == len(batches) - 1})
            if batch["final"]:
                batch["ids"] = torch.from_numpy(ids.astype(np.int64))  # (r,)
                batch["content_tokens"] = torch.from_numpy(content_tokens)  # (r,)
                batch["windows"] = sum(len(window_spans(int(c), spec.context, settings.overlap)) > 1
                                       for c in content_tokens)
            yield batch


def pack_pieces(lengths: Sequence[int], budget: int, packed: bool) -> list[list[int]]:
    """Group pieces, already in the order to keep, into batches: total length (packed) or longest times count (padded)."""
    batches: list[list[int]] = []
    current: list[int] = []
    used = longest = 0
    for index, length in enumerate(lengths):
        if length > budget:
            raise ValueError(f"a piece of {length} tokens exceeds the batch budget of {budget}")
        grown = used + length if packed else max(longest, length) * (len(current) + 1)
        if current and grown > budget:
            batches.append(current)
            current, used, longest = [], 0, 0
        current.append(index)
        used += length
        longest = max(longest, length)
    if current:
        batches.append(current)

    return batches


def batch_rows(chosen: Sequence[tuple[int, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[int]]:
    """The slice rows a batch touches, and each piece's batch-local row: windows of one text share a row."""
    # chosen: (piece tokens (l,), counted mask (l,)) with each piece's slice row
    rows = sorted({row for row, _, _ in chosen})
    local = {row: index for index, row in enumerate(rows)}

    return np.asarray(rows, dtype=np.int64), [local[row] for row, _, _ in chosen]  # (r_b,), one row per piece


def packed_batch(chosen: Sequence[tuple[int, np.ndarray, np.ndarray]], pad_id: int, context: int,
                 pad_to: int | None) -> dict[str, torch.Tensor | int | bool]:
    """Pieces back to back in one row, positions restarting per piece, and `cu_seqlens` marking each piece."""
    # chosen: (slice row, piece tokens (l,), counted mask (l,)) per piece
    rows, local = batch_rows(chosen)
    tokens = [piece for _, piece, _ in chosen]
    segment = [np.where(mask, row, -1) for (_, _, mask), row in zip(chosen, local, strict=True)]
    total = sum(len(piece) for piece in tokens)
    if pad_to is not None:
        filler = pad_to - total
        while filler > 0:  # filler pieces of at most `context` keep attention over them cheap
            size = min(filler, context)
            tokens.append(np.full(size, pad_id, dtype=np.int64))
            segment.append(np.full(size, -1, dtype=np.int64))
            filler -= size
    lengths = np.array([len(piece) for piece in tokens], dtype=np.int64)  # (p,)
    cu_seqlens = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)  # (p + 1,)
    positions = np.concatenate([np.arange(length) for length in lengths])  # (t,)

    return {
        "input_ids": torch.from_numpy(np.concatenate(tokens)),  # (t,)
        "position_ids": torch.from_numpy(positions),  # (t,)
        "cu_seqlens": torch.from_numpy(cu_seqlens),  # (p + 1,)
        "segment": torch.from_numpy(np.concatenate(segment).astype(np.int64)),  # (t,)
        "row_index": torch.from_numpy(rows),  # (r_b,)
        "longest": int(lengths.max()),
        "real_tokens": total,
    }


def padded_batch(chosen: Sequence[tuple[int, np.ndarray, np.ndarray]], pad_id: int) -> dict[str, torch.Tensor | int | bool]:
    """Pieces as padded rows `(b, l)` with an attention mask, for SDPA or eager attention."""
    # chosen: (slice row, piece tokens (l,), counted mask (l,)) per piece
    rows, local = batch_rows(chosen)
    longest = max(len(piece) for _, piece, _ in chosen)
    input_ids = np.full((len(chosen), longest), pad_id, dtype=np.int64)  # (b, l)
    attention_mask = np.zeros((len(chosen), longest), dtype=np.int64)  # (b, l)
    segment = np.full((len(chosen), longest), -1, dtype=np.int64)  # (b, l)
    for index, ((_, piece, mask), row) in enumerate(zip(chosen, local, strict=True)):
        input_ids[index, : len(piece)] = piece
        attention_mask[index, : len(piece)] = 1
        segment[index, : len(piece)] = np.where(mask, row, -1)

    return {
        "input_ids": torch.from_numpy(input_ids),  # (b, l)
        "attention_mask": torch.from_numpy(attention_mask),  # (b, l)
        "segment": torch.from_numpy(segment.reshape(-1)),  # (b * l,)
        "row_index": torch.from_numpy(rows),  # (r_b,)
        "longest": longest,
        "real_tokens": int(attention_mask.sum()),
    }


def load_encoder(settings: EmbedSettings, device: torch.device, *, attention: str | None = None) -> torch.nn.Module:
    """The frozen encoder in float32 weights (bfloat16 runs through autocast), in eval mode."""
    from transformers import AutoModel

    spec = settings.spec
    model = AutoModel.from_pretrained(spec.name, revision=spec.revision, dtype=torch.float32,
                                      attn_implementation=attention or settings.attention)
    assert model.config.hidden_size == spec.width, f"{spec.name} is {model.config.hidden_size} wide, not {spec.width}"

    return model.to(device).eval()


def encode(model: torch.nn.Module, batch: dict[str, torch.Tensor], settings: EmbedSettings,
           autocast: bool = True) -> torch.Tensor:
    """Last hidden states of a batch, flattened to one row per position: (t, d)."""
    # batch: (t,) input_ids and position_ids with (p + 1,) cu_seqlens when packed; (b, l) input_ids and attention_mask when padded
    context = settings.spec.context
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        if "cu_seqlens" in batch:
            cu_seqlens = batch["cu_seqlens"]  # (p + 1,)
            # A constant max length keeps the compiled graph from specializing on each batch's longest piece.
            states = model(input_ids=batch["input_ids"][None], position_ids=batch["position_ids"][None],
                           cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens, max_length_q=context,
                           max_length_k=context).last_hidden_state  # (1, t, d)
        else:
            states = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state  # (b, l, d)

    return states.reshape(-1, states.shape[-1])  # (t, d)


def piece_moments(states: torch.Tensor, segment: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count, sum and sum of squares per batch-local row in float64: (t, d), (t,) to (r_b,), (r_b, d), (r_b, d)."""
    # states: (t, d); segment: (t,), -1 where a position does not count
    keep = segment >= 0  # (t,)
    index = segment[keep]  # (t_c,)
    values = states[keep].double()  # (t_c, d)
    total = torch.zeros(rows, states.shape[1], dtype=torch.float64, device=states.device)  # (r_b, d)
    total.index_add_(0, index, values)
    squares = torch.zeros_like(total)  # (r_b, d)
    squares.index_add_(0, index, values.square_())
    count = torch.bincount(index, minlength=rows)  # (r_b,)

    return count, total, squares  # (r_b,), (r_b, d), (r_b, d)


class EmbeddingStore:
    """The `(n, 2d)` float16 embeddings and `(n,)` int32 token counts in `id` order, with per-slice done markers."""

    def __init__(self, directory: Path, rows: int, settings: EmbedSettings) -> None:
        self.directory = directory
        self.width = settings.spec.width  # d
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"rows": rows, **asdict(settings), "encoder_name": settings.spec.name,
                "revision": settings.spec.revision, "context": settings.spec.context,
                "layout": "mean then variance of the last hidden states", "dtype": "float16"}
        meta_path, values_path, tokens_path = directory / META_FILE, directory / EMBEDDINGS_FILE, directory / TOKENS_FILE
        if meta_path.exists():
            earlier = json.loads(meta_path.read_text(encoding="utf-8"))
            changed = {key for key in meta if key != "compile" and earlier.get(key) != meta[key]}
            if changed:
                raise ValueError(f"{directory} was embedded with other settings ({sorted(changed)}); use a new directory")
            self.values = np.lib.format.open_memmap(values_path, mode="r+")  # (n, 2d)
            self.tokens = np.lib.format.open_memmap(tokens_path, mode="r+")  # (n,)
        else:
            self.values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float16, shape=(rows, 2 * self.width))
            self.tokens = np.lib.format.open_memmap(tokens_path, mode="w+", dtype=np.int32, shape=(rows,))
            self.tokens[:] = -1
            self.tokens.flush()
            write_json(meta_path, meta)
        assert self.values.shape == (rows, 2 * self.width) and self.tokens.shape == (rows,)
        (directory / DONE_DIRECTORY).mkdir(exist_ok=True)

    def done(self) -> set[str]:
        return {path.stem for path in (self.directory / DONE_DIRECTORY).glob("*.done")}

    def write(self, item: Slice, ids: np.ndarray, mean: np.ndarray, variance: np.ndarray, tokens: np.ndarray) -> None:
        """Store one slice's rows, flush, then mark the slice done, so a crash never leaves a marked slice unwritten."""
        # ids, tokens: (r,); mean, variance: (r, d)
        values = np.concatenate([mean, variance], axis=1).astype(np.float16)  # (r, 2d)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"slice {item.key} has a statistic beyond float16's range")
        self.values[ids] = values
        self.tokens[ids] = tokens.astype(np.int32)
        self.values.flush()
        self.tokens.flush()
        (self.directory / DONE_DIRECTORY / f"{item.key}.done").write_text("", encoding="utf-8")


def run_embedding(corpus: Path, plan: list[Slice], selection_path: Path, store: EmbeddingStore, settings: EmbedSettings,
                  model: torch.nn.Module, workers: int, shuffle_seed: int | None = None) -> EmbedStats:
    """Embed every slice of `plan` not yet done, and return what it cost."""
    device = next(model.parameters()).device
    done = store.done()
    remaining = [item for item in plan if item.key not in done]
    stats = EmbedStats(slices=len(remaining))
    if not remaining:
        return stats
    loader = DataLoader(TokenizedSlices(corpus, remaining, selection_path, settings, shuffle_seed), batch_size=None,
                        num_workers=workers, pin_memory=True, prefetch_factor=4 if workers else None,
                        in_order=False)
    totals: dict[int, SliceTotals] = {}
    pending: tuple[dict[str, Any], tuple[torch.Tensor, ...], torch.cuda.Event] | None = None
    started = time.perf_counter()
    progress = tqdm(total=sum(item.tokens for item in remaining), desc=f"embed {settings.encoder}", unit="tok",
                    unit_scale=True, mininterval=30, file=sys.stderr)
    for batch in loader:
        on_device = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items() if key not in ("ids", "content_tokens")}
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        states = encode(model, on_device, settings)  # (t, d)
        moments = piece_moments(states, on_device["segment"], len(batch["row_index"]))
        stop.record()
        stats.timings.append(BatchTiming(int(on_device["segment"].numel()), batch["longest"], start, stop))
        host = tuple(value.to("cpu", non_blocking=True) for value in moments)
        copied = torch.cuda.Event()
        copied.record()
        if pending is not None:
            _accumulate(*pending, totals, remaining, store, settings, stats)
        pending = (batch, host, copied)
        progress.update(batch["real_tokens"])
    if pending is not None:
        _accumulate(*pending, totals, remaining, store, settings, stats)
    torch.cuda.synchronize()
    progress.close()
    stats.seconds = time.perf_counter() - started
    assert not totals, f"slices left open: {sorted(totals)}"

    return stats


def _accumulate(batch: dict[str, Any], host: tuple[torch.Tensor, ...], copied: torch.cuda.Event, totals: dict[int, SliceTotals],
                plan: list[Slice], store: EmbeddingStore, settings: EmbedSettings, stats: EmbedStats) -> None:
    """Add one batch's per-row totals into its slice; when the slice's last batch arrives, write the slice."""
    # host: (r_b,) count, (r_b, d) total, (r_b, d) squares, on the host
    copied.synchronize()
    count, total, squares = (value.numpy() for value in host)  # (r_b,), (r_b, d), (r_b, d)
    rows = batch["row_index"].numpy()  # (r_b,)
    slice_totals = totals.setdefault(batch["slice"], SliceTotals(
        np.zeros(batch["slice_rows"], dtype=np.int64), np.zeros((batch["slice_rows"], settings.spec.width)),
        np.zeros((batch["slice_rows"], settings.spec.width))))
    slice_totals.count[rows] += count
    slice_totals.total[rows] += total
    slice_totals.squares[rows] += squares
    stats.processed_tokens += int(batch["segment"].numel())
    stats.counted_tokens += int(count.sum())
    if not batch["final"]:
        return

    item = plan[batch["slice"]]
    ids, content_tokens = batch["ids"].numpy(), batch["content_tokens"].numpy()  # (r,), (r,)
    if not (slice_totals.count == content_tokens + 2).all():
        raise AssertionError(f"slice {item.key}: counted states differ from content tokens + 2")
    mean, variance = combine_moments(slice_totals.count, slice_totals.total, slice_totals.squares)  # (r, d) each
    store.write(item, ids, mean, variance, content_tokens)
    stats.rows += len(ids)
    stats.windowed_rows += int(batch["windows"])
    stats.truncated_rows += int((content_tokens >= settings.max_text_tokens).sum())
    del totals[batch["slice"]]


def batch_rates(timings: Sequence[BatchTiming], skip: int) -> dict[str, dict[str, float]]:
    """Processed tokens per second of GPU time in each class of longest piece, after `skip` warm-up batches."""
    torch.cuda.synchronize()
    rates: dict[str, dict[str, float]] = {}
    for low, high in zip((0, *RATE_CLASSES[:-1]), RATE_CLASSES, strict=True):
        chosen = [timing for timing in timings[skip:] if low < timing.longest <= high]
        seconds = sum(timing.start.elapsed_time(timing.stop) for timing in chosen) / 1000
        tokens = sum(timing.tokens for timing in chosen)
        rates[f"le_{high}"] = {"batches": len(chosen), "tokens": tokens, "seconds": seconds,
                               "tokens_per_second": tokens / seconds if seconds else math.nan}

    return rates


def processed_tokens(content_tokens: np.ndarray, context: int, overlap: int, cap: int) -> np.ndarray:
    """The tokens the GPU runs for each text, windows and special tokens included: (n,) to (n,)."""
    # content_tokens: (n,)
    content = np.minimum(content_tokens, cap)  # (n,)
    width, stride = context - 2, context - 2 - overlap
    windows = np.where(content <= width, 1, 1 + np.ceil((content - width) / stride))  # (n,)

    return np.where(windows == 1, content + 2, windows * context).astype(np.int64)  # (n,)


def project_hours(content_tokens: np.ndarray, rates: dict[str, dict[str, float]], settings: EmbedSettings,
                  token_ratio: float) -> dict[str, float]:
    """GPU hours to embed texts of these corpus token counts at the measured rate of each class."""
    # content_tokens: (n,), counted by the corpus's (ModernBERT) tokenizer; token_ratio rescales to this encoder's
    spec = settings.spec
    processed = processed_tokens(np.ceil(content_tokens * token_ratio).astype(np.int64), spec.context,
                                 settings.overlap, settings.max_text_tokens)  # (n,)
    longest = np.minimum(processed, spec.context)  # (n,)
    hours, total = 0.0, 0
    for low, high in zip((0, *RATE_CLASSES[:-1]), RATE_CLASSES, strict=True):
        tokens = int(processed[(longest > low) & (longest <= high)].sum())
        rate = rates[f"le_{high}"]["tokens_per_second"]
        if tokens and not math.isfinite(rate):
            raise ValueError(f"the pilot timed no batch of pieces up to {high} tokens; widen its sample")
        hours += tokens / rate / 3600 if tokens else 0.0
        total += tokens

    return {"processed_tokens": total, "gpu_hours": hours, "texts": int(len(content_tokens))}


def pilot_sample(metadata: pa.Table, selected: np.ndarray, rows: int, seed: int) -> np.ndarray:
    """Ids spread evenly over length bins (as far as each holds rows), so every rate class is timed."""
    # selected: (n,) bool, indexed by id
    ids = metadata["id"].to_numpy()  # (n,)
    bins = length_bins(metadata["tokens"].to_numpy())  # (n,)
    rng = np.random.default_rng(seed)
    pool = {int(b): rng.permutation(ids[(bins == b) & selected[ids]]) for b in np.unique(bins)}
    chosen: list[np.ndarray] = []
    share = rows // max(1, len(pool))
    for members in pool.values():
        chosen.append(members[:share])
    taken = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
    rest = np.setdiff1d(ids[selected[ids]], taken)
    extra = rng.permutation(rest)[: rows - len(taken)]

    return np.sort(np.concatenate([taken, extra]))  # (m,)


def write_pilot_corpus(corpus: Path, ids: np.ndarray, path: Path) -> np.ndarray:
    """Copy the pilot's texts into a small Parquet with local ids 0..m-1; return the corpus id of each local id."""
    # ids: (m,)
    parts = [pq.read_table(file, columns=["id", "text", "tokens"], filters=[("id", "in", ids.tolist())])
             for file in parquet_files(corpus)]
    table = pa.concat_tables(parts).sort_by("id")
    source = table["id"].to_numpy()  # (m,)
    assert np.array_equal(source, ids), "every pilot id must be found once"
    table = table.set_column(0, "id", pa.array(np.arange(len(source), dtype=np.int64)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=2048)

    return source  # (m,)


@torch.inference_mode()
def reference_moments(model: torch.nn.Module, corpus: Path, local_ids: np.ndarray, settings: EmbedSettings) -> np.ndarray:
    """Float32, unpacked, one text at a time: the statistics a text has alone, (m, 2d) float16 like the store."""
    # local_ids: (m,)
    tokenizer = load_tokenizer(settings.spec)
    table = pq.read_table(corpus, columns=["id", "text"]).sort_by("id")
    device = next(model.parameters()).device
    out = np.zeros((len(local_ids), 2 * settings.spec.width), dtype=np.float16)  # (m, 2d)
    for row, local in enumerate(local_ids):
        ids = tokenizer(table["text"][int(local)].as_py(), add_special_tokens=True, return_tensors="pt")["input_ids"]
        assert ids.shape[1] <= settings.spec.context, "the reference reads single-window texts only"
        states = model(input_ids=ids.to(device)).last_hidden_state[0].double()  # (l, d)
        out[row] = torch.cat([states.mean(0), states.var(0, unbiased=False)]).cpu().numpy().astype(np.float16)

    return out  # (m, 2d)


def selection_mask(metadata: pa.Table, ids_path: Path | None, sample_rows: int | None, include_benchmark: bool,
                   seed: int) -> np.ndarray:
    """Which ids to embed: all, or those in an id file, or a uniform sample; benchmark rows only on request."""
    ids = metadata["id"].to_numpy()  # (n,)
    selected = np.zeros(len(ids), dtype=bool)  # (n,), indexed by id
    if ids_path is not None:
        selected[np.load(ids_path)] = True
    else:
        selected[:] = True
    if not include_benchmark:
        benchmark = pc.fill_null(metadata["benchmark"], "").to_numpy(zero_copy_only=False)
        selected[ids[benchmark != ""]] = False
    if sample_rows is not None:
        chosen = np.random.default_rng(seed).permutation(np.flatnonzero(selected))[:sample_rows]
        selected[:] = False
        selected[chosen] = True

    return selected  # (n,)


def run_pilot(args: argparse.Namespace, settings: EmbedSettings, metadata: pa.Table, selected: np.ndarray) -> dict[str, Any]:
    """Time 20,000 texts, check packing against float32, measure bfloat16 drift, and project the full run."""
    # selected: (n,) bool, indexed by id
    device = torch.device("cuda")
    directory = args.out / ("pilot_compiled" if settings.compile else "pilot")
    pilot_corpus = directory / "texts.parquet"
    source = write_pilot_corpus(args.corpus, pilot_sample(metadata, selected, args.pilot_rows, args.seed), pilot_corpus)
    local_selected = np.ones(len(source), dtype=bool)  # (m,)
    selection_path = directory / "selection.npy"
    np.save(selection_path, local_selected)

    model = load_encoder(settings, device)
    if settings.compile:
        model = torch.compile(model, dynamic=False)
    plan = plan_slices(pilot_corpus, local_selected, args.slice_rows, args.slice_tokens)
    store = EmbeddingStore(directory / "packed_a", len(source), settings)
    stats = run_embedding(pilot_corpus, plan, selection_path, store, settings, model, args.workers)
    rates = batch_rates(stats.timings, PILOT_WARMUP_BATCHES)

    corpus_tokens = metadata["tokens"].to_numpy()  # (n,)
    pilot_tokens = corpus_tokens[source]  # (m,)
    token_ratio = float(store.tokens[:].sum() / max(1, np.minimum(pilot_tokens, settings.max_text_tokens).sum()))
    ids = metadata["id"].to_numpy()  # (n,)
    to_compute = selected.copy()  # (n,), indexed by id: what the full run computes rather than copies
    if args.reuse:
        to_compute[reusable_rows(metadata, selected, settings.max_text_tokens)[0]] = False
    projection = project_hours(corpus_tokens[to_compute[ids]], rates, settings, token_ratio)
    projection["reused_texts"] = int(selected.sum() - to_compute.sum())

    single = np.flatnonzero(store.tokens[:] <= settings.spec.context - 2)  # (m_1,), texts of one window
    rng = np.random.default_rng(args.seed)
    drift_ids = np.sort(rng.permutation(single)[:PILOT_DRIFT_ROWS])  # (m_d,)
    drift_selected = np.zeros(len(source), dtype=bool)
    drift_selected[drift_ids] = True
    np.save(directory / "drift_selection.npy", drift_selected)
    other = EmbeddingStore(directory / "packed_b", len(source), settings)
    run_embedding(pilot_corpus, plan_slices(pilot_corpus, drift_selected, args.slice_rows, args.slice_tokens),
                  directory / "drift_selection.npy", other, settings, model, args.workers, shuffle_seed=args.seed + 1)
    reuse_check = check_reuse(args, settings, metadata, selected, model, directory) if args.reuse else None
    del model
    torch.cuda.empty_cache()
    reference_model = load_encoder(settings, device, attention="sdpa")
    reference = reference_moments(reference_model, pilot_corpus, drift_ids, settings)  # (m_d, 2d)
    packed_a = np.asarray(store.values[drift_ids])  # (m_d, 2d)
    packed_b = np.asarray(other.values[drift_ids])  # (m_d, 2d)
    width = settings.spec.width
    np.savez(directory / "drift.npz", ids=source[drift_ids], packed_a=packed_a, packed_b=packed_b, float32=reference)
    against_reference = vector_drift(packed_a, reference, width)
    two_packings = vector_drift(packed_a, packed_b, width)
    # A leak across pieces shows as a text's states changing with the batch it shares. Packed, that is read against
    # the text alone in float32. Padded, no piece can reach another, and float32 against bfloat16 is arithmetic the
    # teacher accepts (measured on the H100: two padded batches agree to 5 decimals while the worst text sits at
    # cosine 0.9925 from float32), so the check is the two batches and the float32 drift is reported by length.
    check = against_reference if settings.packed else two_packings
    worst_mean = 1 - check["mean_one_minus_cosine"]["max"]
    worst_variance = 1 - check["variance_one_minus_cosine"]["max"]
    if worst_mean < MEAN_COSINE_FLOOR or worst_variance < VARIANCE_COSINE_FLOOR:
        raise AssertionError(f"a text's bfloat16 states depend on its batch (worst cosine: mean {worst_mean:.5f}, "
                             f"variance {worst_variance:.5f}); pieces may leak into each other")
    drift_tokens = np.asarray(store.tokens[drift_ids])  # (m_d,)
    float32_by_length = {}
    for low, high in zip((1, *RATE_CLASSES), (*RATE_CLASSES, settings.spec.context + 1), strict=True):
        members = (drift_tokens >= low) & (drift_tokens < high)  # (m_d,)
        if members.any():
            float32_by_length[f"{low}-{high - 1}"] = {"texts": int(members.sum()), **vector_drift(packed_a[members], reference[members], width)}
    if reuse_check is not None:
        # The noise rule's yardstick is how far bfloat16 arithmetic moves a vector. Packed, two packings show it.
        # Padded, two batches agree to 5 decimals, which would refuse any stored vector for being bfloat16 at all,
        # so the yardstick is the larger of that and this run's own bfloat16 against float32. The two readers of
        # the shift and the spread, which are what a head could exploit, are unchanged.
        noise = {key: {**values, "p99": max(values["p99"], against_reference[key]["p99"])} for key, values in two_packings.items()}
        reuse_check = decide_and_record_reuse(reuse_check, noise, directory, args.seed)
        reuse_check["noise_yardstick"] = {key: noise[key]["p99"] for key in ("mean_one_minus_cosine", "variance_one_minus_cosine")}

    return {
        "pilot_rows": int(len(source)), "rates": rates, "token_ratio_to_corpus": token_ratio,
        "pilot_wall_seconds": stats.seconds, "pilot_rows_per_second": stats.rows / stats.seconds,
        "projection": projection,
        "drift": {"bf16_two_packings": two_packings, "bf16_against_float32": against_reference,
                  "bf16_against_float32_by_length": float32_by_length},
        "packing_check": {"texts": int(len(drift_ids)), "against": "float32, alone" if settings.packed else "a second batch",
                          "worst_mean_cosine": worst_mean, "worst_variance_cosine": worst_variance},
        "reuse_check": reuse_check,
    }


def check_reuse(args: argparse.Namespace, settings: EmbedSettings, metadata: pa.Table, selected: np.ndarray,
                model: torch.nn.Module, directory: Path) -> dict[str, Any]:
    """Recompute CHECK_ROWS reusable rows as the full run would, and compare them with the stored vectors."""
    # selected: (n,) bool, indexed by id
    ids, first_ids = reusable_rows(metadata, selected, settings.max_text_tokens)  # (r,), (r,)
    if not len(ids):
        raise ValueError("no row is reusable: check text_unchanged and first_corpus_id")
    pick = np.random.default_rng(args.seed + 2).permutation(len(ids))[:CHECK_ROWS]  # (m,)
    pick = pick[np.argsort(ids[pick])]  # (m,) in ascending id, the order the check corpus is written in
    check_corpus = directory / "reuse_texts.parquet"
    source = write_pilot_corpus(args.corpus, ids[pick], check_corpus)  # (m,)
    every = np.ones(len(source), dtype=bool)  # (m,)
    np.save(directory / "reuse_selection.npy", every)
    store = EmbeddingStore(directory / "reuse_recomputed", len(source), settings)
    run_embedding(check_corpus, plan_slices(check_corpus, every, args.slice_rows, args.slice_tokens),
                  directory / "reuse_selection.npy", store, settings, model, args.workers)
    recomputed = np.asarray(store.values[:])  # (m, 2d)
    stored = np.asarray(np.load(args.reuse, mmap_mode="r")[first_ids[pick]])  # (m, 2d)
    tokens = np.asarray(store.tokens[:]).astype(np.int64)  # (m,)
    np.savez(directory / "reuse_drift.npz", ids=source, stored=stored, recomputed=recomputed, tokens=tokens)

    return {"rows": int(len(source)), "reusable_rows": int(len(ids)), "arrays": (stored, recomputed, tokens)}


def decide_and_record_reuse(check: dict[str, Any], packing_drift: dict[str, dict[str, float]], directory: Path,
                            seed: int) -> dict[str, Any]:
    """The pilot's reuse outcome; an aligned or refused one also saves its alignment and the held-out aligned pairs."""
    stored, recomputed, tokens = check.pop("arrays")  # (m, 2d), (m, 2d), (m,)
    decision, alignment, held = decide_reuse(stored, recomputed, tokens, packing_drift, vector_drift, seed)
    if alignment is not None:
        alignment.save(directory / ALIGNMENT_FILE)
        np.savez(directory / "reuse_drift_aligned.npz", recomputed=recomputed[held],
                 stored=alignment.apply(stored[held], tokens[held]), tokens=tokens[held])
        decision["alignment_path"] = (directory / ALIGNMENT_FILE).as_posix()

    return {**check, **decision}


def reuse_gate_of(pilot_path: Path, accept_risk: bool) -> dict[str, Any]:
    """What the full run does with reusable rows, from the pilot's outcome.

    `safe` copies as stored, `aligned` copies through the pilot's alignment, `refused` copies nothing so every row is
    computed. Without a pilot reuse check the run stops, unless `--accept-reuse-risk`, which copies as stored; so does
    accepting the risk over `refused`. The record says which happened.
    """
    check = json.loads(pilot_path.read_text(encoding="utf-8")).get("reuse_check") if pilot_path.exists() else None
    outcome = check["outcome"] if check else None
    if outcome is None and not accept_risk:
        raise RuntimeError("no reuse check in the pilot; run the pilot with --reuse, or pass --accept-reuse-risk")
    if outcome == "aligned":
        return {"outcome": outcome, "copy": True, "alignment_path": check["alignment_path"], "accepted_risk": False}
    if outcome == "safe":
        return {"outcome": outcome, "copy": True, "alignment_path": None, "accepted_risk": False}

    return {"outcome": outcome or "unchecked", "copy": accept_risk, "alignment_path": None, "accepted_risk": accept_risk}


def copy_reusable(args: argparse.Namespace, metadata: pa.Table, selected: np.ndarray, settings: EmbedSettings,
                  store: EmbeddingStore, gate: dict[str, Any]) -> np.ndarray:
    """Copy every reusable row into the store once, through the alignment the gate names; return which rows were reused.

    A marker makes it resumable, and `meta.json` records the outcome and the alignment beside the store.
    """
    # selected: (n,) bool, indexed by id
    reused = np.zeros(len(selected), dtype=bool)  # (n,)
    marker, mask_path = store.directory / REUSE_MARKER, store.directory / REUSED_FILE
    if marker.exists():
        return np.load(mask_path)  # (n,)
    alignment = Alignment.load(Path(gate["alignment_path"])) if gate["alignment_path"] else None
    ids, first_ids = reusable_rows(metadata, selected, settings.max_text_tokens)  # (r,), (r,)
    tokens_by_id = np.zeros(len(selected), dtype=np.int64)  # (n,)
    tokens_by_id[metadata["id"].to_numpy()] = metadata["tokens"].to_numpy()
    copy_reused(store.values, store.tokens, args.reuse, ids, first_ids, tokens_by_id[ids], alignment)
    reused[ids] = True
    np.save(mask_path, reused)
    record = {"rows": int(len(ids)), "source": args.reuse.as_posix(), "outcome": gate["outcome"],
              "accepted_risk": gate["accepted_risk"], "alignment": alignment.summary() if alignment else None,
              "alignment_path": gate["alignment_path"]}
    meta_path = store.directory / META_FILE
    write_json(meta_path, {**json.loads(meta_path.read_text(encoding="utf-8")), "reuse": record})
    write_json(marker, record)

    return reused  # (n,)


EARLIER_MARKER = "earlier.done"
COPY_ROWS = 1 << 20


def copy_earlier(store: EmbeddingStore, earlier: Path, settings: EmbedSettings) -> np.ndarray:
    """Copy an earlier store of the same encoder and settings into this one, row for row, for a corpus that extends the
    earlier corpus (`litmus_corpus extend`: its first rows are the earlier rows, ids and texts unchanged); return which
    rows hold a vector from it, those the earlier store had embedded. Resumable by a marker."""
    rows = store.tokens.shape[0]
    marker = store.directory / EARLIER_MARKER
    meta = json.loads((earlier / META_FILE).read_text(encoding="utf-8"))
    changed = {key for key, value in asdict(settings).items() if key != "compile" and meta.get(key) != value}
    if changed:
        raise ValueError(f"{earlier} was embedded with other settings ({sorted(changed)}); its vectors cannot be copied")
    if meta["rows"] > rows:
        raise ValueError(f"{earlier} holds {meta['rows']} rows, more than this corpus's {rows}: it is not an extension")
    values = np.load(earlier / EMBEDDINGS_FILE, mmap_mode="r")  # (n_e, 2d)
    tokens = np.load(earlier / TOKENS_FILE, mmap_mode="r")  # (n_e,)
    if not marker.exists():
        for start in tqdm(range(0, meta["rows"], COPY_ROWS), desc="copy earlier", mininterval=30, file=sys.stderr):
            end = min(start + COPY_ROWS, meta["rows"])
            store.values[start:end] = values[start:end]
            store.tokens[start:end] = tokens[start:end]
        store.values.flush()
        store.tokens.flush()
        write_json(marker, {"source": earlier.as_posix(), "rows": meta["rows"]})
    copied = np.zeros(rows, dtype=bool)  # (n,)
    copied[: meta["rows"]] = np.asarray(tokens) >= 0

    return copied  # (n,)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True, help="the corpus Parquet file or directory")
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default=REUSE_ENCODER,
                        help="the teacher is ModernBERT-base; large and mmBERT-base are comparison arms on a subsample")
    parser.add_argument("--reuse", type=Path, help="the first corpus's modernbert_base_mean_variance.npy, to copy "
                        "rows whose text it embedded unchanged (modernbert_base only)")
    parser.add_argument("--accept-reuse-risk", action="store_true",
                        help="copy rows as stored when the pilot refused reuse or never checked it; recorded")
    parser.add_argument("--out", type=Path, required=True, help="one directory per encoder")
    parser.add_argument("--attention", default="flash_attention_2",
                        help="transformers attn_implementation; flash ones pack, sdpa and eager pad")
    parser.add_argument("--batch-tokens", type=int, default=131_072, help="positions per GPU batch")
    parser.add_argument("--overlap", type=int, default=1024, help="least overlap of consecutive windows")
    parser.add_argument("--max-text-tokens", type=int, default=131_072, help="content tokens kept from a text's start")
    parser.add_argument("--compile", action="store_true", help="torch.compile the encoder over fixed-size batches")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--slice-rows", type=int, default=8192)
    parser.add_argument("--slice-tokens", type=int, default=32_000_000)
    parser.add_argument("--earlier-store", type=Path, default=None,
                        help="an earlier corpus's store of this encoder, which this corpus extends: its rows are copied, not computed")
    parser.add_argument("--ids", type=Path, help="an .npy of ids to embed; default every non-benchmark row")
    parser.add_argument("--sample-rows", type=int, help="embed a uniform random sample of this many selected rows")
    parser.add_argument("--include-benchmark", action="store_true", help="also embed rows held for a benchmark")
    parser.add_argument("--limit", type=int, help="embed only the first slices holding this many rows (a smoke run)")
    parser.add_argument("--pilot", action="store_true", help="time --pilot-rows texts, check packing and drift, project hours")
    parser.add_argument("--pilot-rows", type=int, default=PILOT_ROWS,
                        help="texts the pilot times, spread evenly over length bins; the long bins dominate its time "
                             "(20,000 rows took over 30 minutes under padded SDPA)")
    parser.add_argument("--seed", type=int, default=0)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("embedding runs on a CUDA GPU")
    settings = EmbedSettings(args.encoder, args.attention, args.batch_tokens, args.overlap, args.max_text_tokens,
                             args.compile)
    if settings.batch_tokens < settings.spec.context:
        raise ValueError("--batch-tokens must hold at least one full window")
    torch.backends.cuda.matmul.allow_tf32 = False  # the float32 reference stays float32
    if args.reuse is not None and settings.encoder != REUSE_ENCODER:
        raise ValueError(f"--reuse holds {REUSE_ENCODER} vectors; it cannot fill a {settings.encoder} store")
    if args.reuse is not None and args.earlier_store is not None:
        raise ValueError("--reuse and --earlier-store together: the earlier store's row-for-row copy would overwrite reused rows")
    metadata = read_columns(args.corpus, ["id", "tokens", "benchmark", *(REUSE_COLUMNS if args.reuse else ())])
    rows = dense_row_count(metadata["id"].to_numpy())
    selected = selection_mask(metadata, args.ids, args.sample_rows, args.include_benchmark, args.seed)  # (n,)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.pilot:
        report = run_pilot(args, settings, metadata, selected)
        write_json(args.out / ("pilot_compiled.json" if settings.compile else "pilot.json"), report)
        write_summary(args.out, "embed --pilot", {"settings": asdict(settings), **report,
                                                  "seconds": time.perf_counter() - started})
        return

    store = EmbeddingStore(args.out, rows, settings)
    reused = np.zeros(rows, dtype=bool)  # (n,), indexed by id
    reuse_gate = None
    if args.reuse is not None:
        reuse_gate = reuse_gate_of(args.out / "pilot.json", args.accept_reuse_risk)
        if reuse_gate["copy"]:
            reused = copy_reusable(args, metadata, selected, settings, store, reuse_gate)
    if args.earlier_store is not None:
        reused |= copy_earlier(store, args.earlier_store, settings)
    compute = selected & ~reused  # (n,)
    plan = plan_slices(args.corpus, compute, args.slice_rows, args.slice_tokens)
    if args.limit is not None:
        cumulative = np.cumsum([item.length for item in plan])
        plan = plan[: int(np.searchsorted(cumulative, args.limit)) + 1]
    selection_path = args.out / "selection.npy"
    np.save(selection_path, compute)
    model = load_encoder(settings, torch.device("cuda"))
    if settings.compile:
        model = torch.compile(model, dynamic=False)
    stats = run_embedding(args.corpus, plan, selection_path, store, settings, model, args.workers)
    rates = batch_rates(stats.timings, PILOT_WARMUP_BATCHES) if stats.timings else {}
    corpus_tokens = metadata["tokens"].to_numpy()  # (n,)
    ids = metadata["id"].to_numpy()  # (n,)
    embedded = np.asarray(store.tokens[:]) >= 0  # (n,), indexed by id
    write_summary(args.out, "embed", {
        "settings": asdict(settings), "encoder_name": settings.spec.name, "revision": settings.spec.revision,
        "rows": rows, "selected_rows": int(selected.sum()), "embedded_rows": int(embedded.sum()),
        "reused_rows": int(reused.sum()), "rows_to_compute": int(compute.sum()),
        "computed_rows": int((embedded & ~reused).sum()), "reuse_gate": reuse_gate,
        "slices_planned": len(plan), "slices_run": stats.slices, "rows_this_run": stats.rows,
        "processed_tokens_this_run": stats.processed_tokens, "counted_tokens_this_run": stats.counted_tokens,
        "windowed_rows_this_run": stats.windowed_rows, "truncated_rows_this_run": stats.truncated_rows,
        "seconds": time.perf_counter() - started, "embed_seconds": stats.seconds,
        "processed_tokens_per_second": stats.processed_tokens / stats.seconds if stats.seconds else None,
        "rates": rates,
        # Only meaningful for ModernBERT, whose tokenizer the corpus counts with: rows whose count differs.
        "rows_whose_tokens_differ_from_corpus": int((np.asarray(store.tokens[:])[ids[embedded[ids]]]
                                                      != np.minimum(corpus_tokens[embedded[ids]],
                                                                    settings.max_text_tokens)).sum()),
        "length_edges": list(LENGTH_EDGES),
    })
