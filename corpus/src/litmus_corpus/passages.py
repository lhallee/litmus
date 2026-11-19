"""Stage 4, passages: each text gives itself and up to `max_sub_passages` passages cut from it.

No training row is longer than 8,192 tokens, the most the model reads at once. A text of at most 8,192 tokens
gives itself as a row with `whole` true. A text of 8,193 to 131,072 tokens gives itself as a `long_eval` row
(`whole` true), kept only to evaluate the model's chunked reading of a long document; a longer one gives no
whole row. Every text also gives passages that start and end at its cut starts (sentence or line starts; line
starts only in code and tables), so no passage begins or ends inside a sentence of prose or inside a line of
code. Every row cut from a text carries the text's `doc`, so a split by `doc` keeps a long document and its
passages on one side.

Each passage's target length is drawn evenly in the logarithm from 8 tokens to half the text (or to 8,192 for
a longer text), by a generator seeded with the text's hash: both classes draw from one distribution at any
text length, and a rerun gives the same cuts. A target is placed at a random start and ends at the cut start
nearest the target; a placement that misses the target by more than a factor of two, or overlaps a passage
already placed, is tried again, three times at most. A passage's token count is its own encoding's.

A first-corpus row enters verbatim, uncut, with its own `whole`; a benchmark row stays whole as the benchmark
has it.
"""

from __future__ import annotations

import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path

from .config import LABEL_ORIGINS, LONG_BIN_LABELS, LONG_MAX_TOKENS, MAX_TOKENS, MIN_TOKENS, BuildConfig, long_bin
from .formats import cuts_at_lines_only, text_format
from .records import PASSAGE_SCHEMA, STRATA_SCHEMA, conform
from .storage import JsonObject, Report, write_json, write_parquet
from .tokenization import token_counts


PLACEMENT_ATTEMPTS = 3
MIN_TARGET_SHARE = 0.5
MAX_TARGET_FACTOR = 2.0
MIN_OWN_FORMAT_LINES = 3  # a shorter passage keeps its text's format, since a line or two says too little


def passage_spans(tokens: int, cut_tokens: list[int], seed: int, max_passages: int) -> list[tuple[int, int]]:
    """Spans (first cut start, end cut start) of the passages cut from one text, in text order.

    An end equal to `len(cut_tokens)` means the end of the text. The whole text is never a span.
    """
    starts = len(cut_tokens)
    boundaries = np.array([*cut_tokens, tokens], dtype=np.int64)  # (s + 1,)
    upper = min(MAX_TOKENS, tokens // 2) if tokens <= MAX_TOKENS else MAX_TOKENS
    if upper < MIN_TOKENS:
        return []
    generator = np.random.Generator(np.random.PCG64(seed))
    targets = np.floor(np.exp(generator.uniform(math.log(MIN_TOKENS), math.log(upper + 1), size=max_passages))).astype(np.int64)  # (k,)
    taken: list[tuple[int, int]] = []
    occupied: list[tuple[int, int]] = []
    for target in targets:
        for _ in range(PLACEMENT_ATTEMPTS):
            last_start = int(np.searchsorted(boundaries[:starts], tokens - target, side="right")) - 1
            if last_start < 0:
                break
            first = int(generator.integers(0, last_start + 1))
            end = int(np.searchsorted(boundaries, boundaries[first] + target, side="left"))
            if end - 1 > first and abs(boundaries[end - 1] - boundaries[first] - target) < abs(boundaries[end] - boundaries[first] - target):
                end -= 1
            span_tokens = int(boundaries[end] - boundaries[first])
            low, high = int(boundaries[first]), int(boundaries[end])
            fits = MIN_TOKENS <= span_tokens <= MAX_TOKENS and MIN_TARGET_SHARE * target <= span_tokens <= MAX_TARGET_FACTOR * target
            whole = first == 0 and end == starts
            overlaps = any(low < other_high and other_low < high for other_low, other_high in occupied)
            if fits and not whole and not overlaps:
                taken.append((first, end))
                occupied.append((low, high))
                break

    return sorted(taken)


def passage_text(text: str, start: int, end: int, format_name: str) -> str:
    piece = text[start:end].rstrip()

    return piece.lstrip("\n") if cuts_at_lines_only(format_name) else piece.strip()


def passages_table(table: pa.Table, tokenizer_path: str, config: BuildConfig, report: Report) -> pa.Table:
    """One row a passage: each text whole when it fits, and its cut passages."""
    rows: dict[str, list[object]] = {name: [] for name in PASSAGE_SCHEMA.names}
    pending: list[tuple[dict[str, object], str]] = []  # cut passages waiting for their token counts
    carried = [name for name in PASSAGE_SCHEMA.names if name in STRATA_SCHEMA.names and name != "text" and name != "tokens"]
    columns = {name: table.column(name).to_pylist() for name in (*carried, "text", "tokens", "cut_chars", "cut_tokens")}
    for index in range(table.num_rows):
        text = columns["text"][index]
        tokens = columns["tokens"][index]
        origin = LABEL_ORIGINS[columns["label"][index]]
        attributes = {name: columns[name][index] for name in carried}
        if attributes["first_corpus_id"] is not None:
            # A first-corpus row enters verbatim and is not cut again; it keeps its own `whole`.
            assert attributes["first_whole"] is not None, "a first-corpus row without its `whole`"
            add_row(rows, attributes, text, tokens, whole=attributes["first_whole"], long_eval=tokens > MAX_TOKENS)
            report.count("rows", f"{origin}|first_corpus{'_over_8192' if tokens > MAX_TOKENS else ''}")
            continue
        if attributes["benchmark"]:
            add_row(rows, attributes, text, tokens, whole=True, long_eval=False)  # a benchmark row stays as the benchmark has it
            report.count("rows", f"{origin}|benchmark")
            continue
        if tokens < MIN_TOKENS:
            report.drop("under_8_tokens", origin)
            continue
        if tokens <= MAX_TOKENS:
            add_row(rows, attributes, text, tokens, whole=True, long_eval=False)
            report.count("rows", f"{origin}|whole")
        elif tokens <= LONG_MAX_TOKENS:
            add_row(rows, attributes, text, tokens, whole=True, long_eval=True)
            report.count("rows", f"{origin}|long_eval")
            report.count("long_documents", f"{origin}|{LONG_BIN_LABELS[long_bin(tokens)]}")
        else:
            report.count("long_documents", f"{origin}|over_{LONG_MAX_TOKENS}")
        chars = [*columns["cut_chars"][index], len(text)]
        for first, end in passage_spans(tokens, columns["cut_tokens"][index], int(attributes["doc"]), config.max_sub_passages):
            piece = passage_text(text, chars[first], chars[end], attributes["format"])
            if piece:
                pending.append((attributes, piece))
    counts = token_counts([piece for _, piece in pending], tokenizer_path) if pending else []
    for (attributes, piece), tokens in zip(pending, counts, strict=True):
        origin = LABEL_ORIGINS[attributes["label"]]
        if not MIN_TOKENS <= tokens <= MAX_TOKENS:
            report.count("passages", f"{origin}|out_of_range")
            continue
        lines = sum(1 for line in piece.split("\n") if line.strip())
        own_format = text_format(piece) if lines >= MIN_OWN_FORMAT_LINES else attributes["format"]
        add_row(rows, {**attributes, "format": own_format}, piece, tokens, whole=False, long_eval=False)
        report.count("rows", f"{origin}|cut")
    passages = pa.table({name: pa.array(values, type=PASSAGE_SCHEMA.field(name).type) for name, values in rows.items()})

    return conform(passages, PASSAGE_SCHEMA)


def add_row(rows: dict[str, list[object]], attributes: dict[str, object], text: str, tokens: int, whole: bool, long_eval: bool) -> None:
    for name, value in attributes.items():
        rows[name].append(value)
    rows["text"].append(text)
    rows["tokens"].append(tokens)
    rows["whole"].append(whole)
    rows["long_eval"].append(long_eval)


def passages_shard(source_path: Path, target_path: Path, tokenizer_path: str, config: BuildConfig) -> JsonObject:
    """Cut one strata shard into passages; writes the shard and then its report, and returns the report."""
    table = pq.read_table(source_path)
    report = Report(rows_in=table.num_rows)
    passages = passages_table(table, tokenizer_path, config, report)
    report.rows_out = passages.num_rows
    write_parquet(passages, target_path)
    payload = report.to_json()
    write_json(payload, target_path.with_suffix(".json"))

    return payload
