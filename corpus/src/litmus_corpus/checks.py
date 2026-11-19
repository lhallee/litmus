"""Stage 5, checks: drop duplicates, empty and degenerate text, and human text that fails its date.

Per row (in each shard):

    empty                 no character but whitespace
    no_letters            no letter at all (digits, punctuation, symbols), or one character repeated
    repetition            a text of 1,000 bytes or more that zlib shrinks below 8% (a loop), or one line
                          of 10 or more characters making half or more of four or more lines
    human_undated         human text with no date
    human_not_before_2022 human text dated 2022 or later

Across the corpus, as in the first corpus: one row for each text and one for each opening of 20 words (the
first 20 words, lowercased, at most 160 characters, so a text without spaces has an opening too). The copy
kept is a first-corpus row when one holds the key, else the first in a seeded order, so no new source is
preferred; the kept row's document keeps all its rows, as a passage cut from a text's start shares its opening.
A text or opening found under two origins is dropped from both, first-corpus rows included (the report counts
those apart). Benchmark rows are checked for emptiness only and never dropped as copies; stage
7 removes what copies them.

Code, lists, URLs and markup are kept: the corpus is raw text.
"""

from __future__ import annotations

import hashlib
import re
import zlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from collections import Counter
from pathlib import Path

from .adapters import HUMAN_DATE_LIMIT
from .config import LABEL_ORIGINS
from .records import KEYED_SCHEMA, conform
from .storage import JsonObject, Report, write_json, write_parquet


OPENING_WORDS = 20
OPENING_CHARACTERS = 160
OPENING_SCAN_CHARACTERS = 4_000
WORD_PATTERN = re.compile(r"\w+")
LETTER_PATTERN = re.compile(r"[^\W\d_]")
COMPRESSION_MIN_BYTES = 1_000
COMPRESSION_SCAN_BYTES = 65_536
MIN_COMPRESSION_RATIO = 0.08
REPEATED_LINE_CHARACTERS = 10
REPEATED_LINE_SHARE = 0.5
REPEATED_LINE_MIN_LINES = 4
# Drop reasons of the corpus-wide step, by code; 0 keeps the row.
GLOBAL_REASONS = ("keep", "duplicate_text", "duplicate_opening", "text_under_two_origins", "opening_under_two_origins")
ORDER_SEED = 20261117


def hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def opening(text: str) -> str:
    """The first 20 words of a text, lowercased and space-joined, at most 160 characters."""
    words = WORD_PATTERN.findall(text[:OPENING_SCAN_CHARACTERS].lower())[:OPENING_WORDS]

    return " ".join(words)[:OPENING_CHARACTERS]


def degenerate_reason(text: str) -> str | None:
    """Why a text is empty or degenerate, or None."""
    if not text.strip():
        return "empty"
    if not LETTER_PATTERN.search(text) or len(set(text) - {" ", "\n", "\t"}) == 1:
        return "no_letters"
    encoded = text.encode("utf-8")[:COMPRESSION_SCAN_BYTES]
    if len(encoded) >= COMPRESSION_MIN_BYTES and len(zlib.compress(encoded, 1)) < MIN_COMPRESSION_RATIO * len(encoded):
        return "repetition"
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= REPEATED_LINE_MIN_LINES:
        line, count = Counter(lines).most_common(1)[0]
        if len(line) >= REPEATED_LINE_CHARACTERS and count >= REPEATED_LINE_SHARE * len(lines):
            return "repetition"

    return None


def date_reason(label: int, date: str | None) -> str | None:
    if LABEL_ORIGINS[label] != "human":
        return None
    if date is None:
        return "human_undated"
    if date >= HUMAN_DATE_LIMIT:
        return "human_not_before_2022"

    return None


def keyed_shard(source_path: Path, target_path: Path) -> JsonObject:
    """Per-row checks and keys for one passage shard; writes the kept rows with their keys, then the report."""
    table = pq.read_table(source_path)
    report = Report(rows_in=table.num_rows)
    texts = table.column("text").to_pylist()
    labels = table.column("label").to_pylist()
    dates = table.column("date").to_pylist()
    benchmarks = table.column("benchmark").to_pylist()
    first = table.column("first_corpus_id").is_valid().to_pylist()
    keep = np.zeros(table.num_rows, dtype=bool)  # (n,)
    for index, (text, label, date, benchmark, is_first) in enumerate(zip(texts, labels, dates, benchmarks, first, strict=True)):
        if benchmark:
            reason = "empty" if not text.strip() else None
        else:
            reason = degenerate_reason(text) or date_reason(label, date)
        if reason is not None:
            report.drop(reason, LABEL_ORIGINS[label])
            if is_first:
                report.count("first_corpus_dropped", reason)
        keep[index] = reason is None
    kept = table.filter(pa.array(keep))
    kept_texts = [text for text, flag in zip(texts, keep, strict=True) if flag]
    kept = kept.append_column("text_key", pa.array([hash64(text) for text in kept_texts], type=pa.uint64()))
    kept = kept.append_column("opening_key", pa.array([hash64(opening(text)) for text in kept_texts], type=pa.uint64()))
    kept = conform(kept, KEYED_SCHEMA)
    report.rows_out = kept.num_rows
    report.check_balance(str(source_path))
    write_parquet(kept, target_path)
    payload = report.to_json()
    write_json(payload, target_path.with_suffix(".json"))

    return payload


def duplicate_codes(keys: np.ndarray, labels: np.ndarray, docs: np.ndarray, order: np.ndarray, copy_code: int, cross_code: int) -> np.ndarray:
    """A drop code for each row: `cross_code` when its key holds two origins, `copy_code` for every copy but the first.

    `order` ranks rows; within a key the row of lowest rank is kept, with every row of its document: rows of one
    document (a text and a passage cut from its start, or a raw text and the first corpus's passage of it) share
    an opening by design and are not copies of each other.
    """
    # keys, labels, docs, order: (n,)
    codes = np.zeros(len(keys), dtype=np.uint8)  # (n,)
    if len(keys) == 0:
        return codes  # (0,)
    sorted_rows = np.lexsort((order, keys))  # (n,)
    sorted_keys = keys[sorted_rows]  # (n,)
    sorted_labels = labels[sorted_rows]  # (n,)
    is_first = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]  # (n,)
    starts = np.flatnonzero(is_first)  # (g,)
    lowest = np.minimum.reduceat(sorted_labels, starts)  # (g,)
    highest = np.maximum.reduceat(sorted_labels, starts)  # (g,)
    group = np.cumsum(is_first) - 1  # (n,)
    two_origins = (lowest != highest)[group]  # (n,)
    kept_doc = docs[sorted_rows][starts][group]  # (n,)
    is_copy = docs[sorted_rows] != kept_doc  # (n,)
    sorted_codes = np.where(two_origins, cross_code, np.where(is_copy, copy_code, 0)).astype(np.uint8)  # (n,)
    codes[sorted_rows] = sorted_codes

    return codes  # (n,)


def global_duplicate_codes(
    text_keys: np.ndarray, opening_keys: np.ndarray, labels: np.ndarray, docs: np.ndarray, benchmark: np.ndarray, first: np.ndarray
) -> np.ndarray:
    """The corpus-wide drop code of every row (`GLOBAL_REASONS`); benchmark rows always keep.

    The copy kept is a first-corpus row when one holds the key (its rows enter verbatim), else the first in a
    seeded order.
    """
    # text_keys, opening_keys, labels, docs, benchmark, first: (n,)
    codes = np.zeros(len(text_keys), dtype=np.uint8)  # (n,)
    pool = np.flatnonzero(~benchmark)  # (m,)
    order = np.random.default_rng(ORDER_SEED).permutation(len(pool)) + np.where(first[pool], 0, len(pool))  # (m,)
    text_codes = duplicate_codes(text_keys[pool], labels[pool], docs[pool], order, 1, 3)  # (m,)
    opening_codes = duplicate_codes(opening_keys[pool], labels[pool], docs[pool], order, 2, 4)  # (m,)
    codes[pool] = np.where(text_codes > 0, text_codes, opening_codes)

    return codes  # (n,)


def filter_shard(
    source_path: Path, target_path: Path, codes: np.ndarray, reasons: tuple[str, ...], marks: np.ndarray | None = None, names: tuple[str, ...] = ()
) -> JsonObject:
    """Keep the rows of a shard whose code is 0; each other code is counted as its reason.

    `marks`, when given, names a benchmark for some rows (an index into `names`, 0 for none): the row's
    `benchmark` becomes that name, and the marks are counted by benchmark.
    """
    # codes: (n,); marks: (n,) or None
    table = pq.read_table(source_path)
    assert len(codes) == table.num_rows, f"{source_path}: {len(codes)} codes for {table.num_rows} rows"
    report = Report(rows_in=table.num_rows)
    labels = table.column("label").to_numpy()  # (n,)
    first = table.column("first_corpus_id").is_valid().to_numpy(zero_copy_only=False)  # (n,)
    for code in np.unique(codes[codes > 0]):
        for label, number in zip(*np.unique(labels[codes == code], return_counts=True), strict=True):
            report.drop(reasons[int(code)], LABEL_ORIGINS[int(label)], int(number))
        report.count("first_corpus_dropped", reasons[int(code)], int((first & (codes == code)).sum()))
    if marks is not None:
        assert len(marks) == table.num_rows, f"{source_path}: {len(marks)} marks for {table.num_rows} rows"
        benchmarks = table.column("benchmark").to_pylist()
        for index in np.flatnonzero(marks):
            assert not benchmarks[index], f"{source_path}: row {index} is already the benchmark {benchmarks[index]}'s"
            benchmarks[index] = names[int(marks[index])]
            report.count("first_corpus_marked", benchmarks[index])
        table = table.set_column(table.schema.get_field_index("benchmark"), "benchmark", pa.array(benchmarks, type=pa.string()))
    kept = table.filter(pa.array(codes == 0))
    report.rows_out = kept.num_rows
    report.check_balance(str(source_path))
    write_parquet(kept, target_path)
    payload = report.to_json()
    write_json(payload, target_path.with_suffix(".json"))

    return payload
