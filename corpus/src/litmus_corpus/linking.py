"""Links between the first corpus's trimmed passages and the raw texts they were cut from.

The first corpus's rows enter verbatim. A source it read can also be read raw again, and a benchmark's text
can reach it through another source; either would put one document in the corpus twice under two `doc`
hashes, which could land on both sides of a split. So the two are linked by content.

The link is a fingerprint: the first 12 words (lowercased word characters) of a first-corpus passage. Every
first-corpus passage starts at a sentence or line start of the text it was cut from (the first reader cut whole
sentences, after dropping lines that were not prose), so the raw text is fingerprinted at each of its sentence
and line starts (`cut_starts`), and a raw text is linked to the passage whose fingerprint one of its starts
gives. Twelve words are long enough that unrelated texts almost never share them, and short enough to survive
the first reader's changes inside a sentence (quotes and dashes settled, emphasis marks removed), which leave
the words themselves alone.

A raw text linked to a passage takes the passage's `doc`. It is kept only where it differs materially from the
trimmed passage: formatted (any format but `plain`, which takes in code, lists, chat and reasoning), or longer
than `MATERIAL_LENGTH` times the passage. Otherwise the passage stands for it and it is dropped. A benchmark text
linked to a passage marks the passage as that benchmark's: it leaves training.
"""

from __future__ import annotations

import hashlib
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataclasses import dataclass
from functools import cache
from pathlib import Path


FINGERPRINT_WORDS = 12
FINGERPRINT_CHARACTERS = 800  # enough text after a start for 12 words
MATERIAL_LENGTH = 1.25
WORD_PATTERN = re.compile(r"\w+")
NO_MATCH = -1


def fingerprint(text: str, start: int = 0) -> int | None:
    """The hash of the 12 words from `start`, or None when fewer follow."""
    words = WORD_PATTERN.findall(text[start : start + FINGERPRINT_CHARACTERS].lower())[:FINGERPRINT_WORDS]
    if len(words) < FINGERPRINT_WORDS:
        return None

    return int.from_bytes(hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest(), "big")


def index_shard(source_path: Path, target_path: Path) -> int:
    """The fingerprint, `doc` and word count of every first-corpus row of a standardized shard; returns rows."""
    table = pq.read_table(source_path, columns=["text", "first_doc", "first_corpus_id"])
    table = table.filter(table.column("first_corpus_id").is_valid())
    fingerprints: list[int] = []
    docs: list[int] = []
    words: list[int] = []
    for text, doc in zip(table.column("text").to_pylist(), table.column("first_doc").to_pylist(), strict=True):
        found = fingerprint(text)
        if found is not None:
            fingerprints.append(found)
            docs.append(doc)
            words.append(len(text.split()))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"fingerprint": pa.array(fingerprints, pa.uint64()), "doc": pa.array(docs, pa.uint64()), "words": pa.array(words, pa.int32())}),
        target_path,
    )

    return len(fingerprints)


def save_index(pieces: list[Path], directory: Path) -> int:
    """One sorted index from the per-shard pieces, as three arrays a worker maps from disk; returns its size."""
    tables = [pq.read_table(path) for path in pieces]
    table = pa.concat_tables(tables) if tables else pa.table({"fingerprint": pa.array([], pa.uint64()), "doc": pa.array([], pa.uint64()), "words": pa.array([], pa.int32())})
    fingerprints = table.column("fingerprint").to_numpy()  # (n,)
    order = np.argsort(fingerprints, kind="stable")  # (n,)
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in (("fingerprints", fingerprints[order]), ("docs", table.column("doc").to_numpy()[order]), ("words", table.column("words").to_numpy()[order])):
        np.save(directory / f"{name}.npy", values)

    return int(len(order))


@dataclass(frozen=True)
class FirstIndex:
    fingerprints: np.ndarray  # (n,) uint64, sorted
    docs: np.ndarray  # (n,) uint64
    words: np.ndarray  # (n,) int32

    def lookup(self, queries: np.ndarray) -> np.ndarray:
        """The index row of each query fingerprint, or `NO_MATCH`."""
        # queries: (q,)
        if len(self.fingerprints) == 0 or len(queries) == 0:
            return np.full(len(queries), NO_MATCH, dtype=np.int64)  # (q,)
        positions = np.searchsorted(self.fingerprints, queries)  # (q,)
        clipped = np.minimum(positions, len(self.fingerprints) - 1)  # (q,)

        return np.where(self.fingerprints[clipped] == queries, clipped, NO_MATCH)  # (q,)


@cache
def load_index(directory: str) -> FirstIndex:
    path = Path(directory)

    return FirstIndex(*(np.load(path / f"{name}.npy", mmap_mode="r") for name in ("fingerprints", "docs", "words")))


def link_texts(texts: list[str], cut_starts: list[list[int]], index: FirstIndex) -> list[int]:
    """For each text, the index row of the first-corpus passage it holds (the one most of its starts give), or `NO_MATCH`."""
    owners: list[int] = []
    queries: list[int] = []
    for text_index, (text, starts) in enumerate(zip(texts, cut_starts, strict=True)):
        for start in starts:
            found = fingerprint(text, start)
            if found is not None:
                owners.append(text_index)
                queries.append(found)
    matches = index.lookup(np.array(queries, dtype=np.uint64))  # (q,)
    linked = [NO_MATCH] * len(texts)
    hits: dict[int, list[int]] = {}
    for owner, match in zip(owners, matches.tolist(), strict=True):
        if match != NO_MATCH:
            hits.setdefault(owner, []).append(match)
    for owner, rows in hits.items():
        linked[owner] = max(set(rows), key=rows.count)

    return linked
