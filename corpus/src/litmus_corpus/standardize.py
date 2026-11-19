"""Stage 2, standardize: one reading for every class, then the removal of marks only one class carries.

The standard reading, the same for every text, and the only rewriting the corpus does:

    NFC            Unicode normal form C
    LF             line endings to `\\n`
    controls       control characters other than tab and newline removed, and invisible format marks
                   (zero-width space, word joiner, byte-order mark, soft hyphen, direction marks); the
                   zero-width joiner and non-joiner stay, as Persian, Indic scripts and emoji need them
    trailing       whitespace at the end of every line and of the text removed, and blank lines at its start
    blank lines    runs of blank lines capped at two

Then `tells.py` removes what a dataset's making left in one class, rule by rule, each counted.

Before either, each group of a source (a generator, or the source's human or mixed text) keeps at most
`precap_factor` times its row cap of texts, drawn by a hash of the text, so later stages never read text the
caps would discard.

Benchmark rows and the first corpus's rows get the standard reading only: no pre-cap and no tell removal. A
first-corpus row whose text the reading leaves byte for byte has `text_unchanged` true, so its embedding is
reused; the report counts the others by the step that changed them (`reading_*`).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dataclasses import dataclass
from pathlib import Path

from .adapters import GROUP_SEPARATOR
from .config import LABEL_ORIGINS, ORIGIN_LABELS, BuildConfig
from .generators import is_legacy
from .records import STANDARD_SCHEMA, conform
from .storage import JsonObject, Report, write_json, write_parquet
from .tells import remove_tells


# RE2 patterns, run through Arrow. `\x{...}` names a code point; `\p{Zs}` is every space separator.
CONTROL_PATTERN = (
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\x{00AD}\x{180E}\x{200B}\x{200E}\x{200F}\x{202A}-\x{202E}"
    r"\x{2060}-\x{2064}\x{2066}-\x{206F}\x{FEFF}\x{FFF9}-\x{FFFB}]"
)
TRAILING_PATTERN = r"[ \t\p{Zs}]+(\n|$)"
TEXT_END_PATTERN = r"\s+$"
LEADING_BLANK_PATTERN = r"^(?:[ \t\p{Zs}]*\n)+"
BLANK_RUN_PATTERN = r"\n{4,}"

# The same steps as Python patterns, to say why a first-corpus text changed. Python's `re` reads `\uXXXX`.
SPACE_SEPARATORS = r" \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000"
READING_STEPS: tuple[tuple[str, re.Pattern[str] | None], ...] = (
    ("reading_nfc", None),
    ("reading_line_endings", re.compile(r"\r")),
    (
        "reading_controls",
        re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad\u180e\u200b\u200e\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb]"),
    ),
    # Each branch starts only where a whitespace run starts, so a long run is read once, not once per character.
    ("reading_trailing_whitespace", re.compile(rf"(?<![{SPACE_SEPARATORS}])[{SPACE_SEPARATORS}]+(?:\n|$)|(?<!\s)\s+$")),
    ("reading_leading_blank_lines", re.compile(rf"^[{SPACE_SEPARATORS}]*\n")),
    ("reading_blank_runs", re.compile(r"\n{4,}")),
)


def nfc(texts: pa.Array) -> pa.Array:
    """Unicode normal form C. Arrow's `utf8_normalize` returned `e` and a combining accent uncomposed under NFC
    (pyarrow 25.0.1, tested here), so Python's normalizer is used, behind its fast check."""
    return pa.array(
        [text if unicodedata.is_normalized("NFC", text) else unicodedata.normalize("NFC", text) for text in texts.to_pylist()],
        type=pa.string(),
    )


def standard_reading(texts: pa.Array) -> pa.Array:
    """The standard reading of every text; the regular expressions run vectorized (RE2 through Arrow)."""
    texts = nfc(texts)
    texts = pc.replace_substring_regex(texts, r"\r\n?", "\n")
    texts = pc.replace_substring_regex(texts, CONTROL_PATTERN, "")
    texts = pc.replace_substring_regex(texts, TRAILING_PATTERN, r"\1")
    texts = pc.replace_substring_regex(texts, TEXT_END_PATTERN, "")
    texts = pc.replace_substring_regex(texts, LEADING_BLANK_PATTERN, "")

    return pc.replace_substring_regex(texts, BLANK_RUN_PATTERN, "\n\n\n")


def text_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


@dataclass(frozen=True)
class GroupShares:
    """The share of each of a source's groups that the pre-cap keeps, by (label, generator)."""

    shares: dict[tuple[int, str], float]

    @classmethod
    def from_counts(cls, counts: dict[str, int], config: BuildConfig) -> GroupShares:
        """From a source's read report's group counts, keyed `origin|generator`."""
        shares: dict[tuple[int, str], float] = {}
        for key, total in counts.items():
            origin, _, generator = key.partition(GROUP_SEPARATOR)
            label = ORIGIN_LABELS[origin]
            if origin == "human":
                cap = config.human_cap
            elif origin == "mixed":
                cap = config.mixed_cap
            else:
                cap = config.legacy_ai_cap if is_legacy(generator) else config.ai_cap
            shares[(label, generator)] = min(1.0, config.precap_factor * cap / max(total, 1))

        return cls(shares)


def exempt_rows(table: pa.Table) -> np.ndarray:
    """Rows that keep their text as their source has it: benchmark rows and the first corpus's rows."""
    benchmark = np.asarray(pc.not_equal(table.column("benchmark"), "").to_numpy(zero_copy_only=False))  # (n,)
    first = np.asarray(pc.is_valid(table.column("first_corpus_id")).to_numpy(zero_copy_only=False))  # (n,)

    return benchmark | first  # (n,)


def precap_mask(table: pa.Table, shares: GroupShares, report: Report) -> np.ndarray:
    """Which texts the pre-cap keeps: a text is kept when its hash, as a share of 2**64, is below its group's share."""
    labels = table.column("label").to_numpy()  # (n,)
    generators = table.column("generator").to_pylist()
    texts = table.column("text").to_pylist()
    exempt = exempt_rows(table)  # (n,)
    keep = np.ones(table.num_rows, dtype=bool)  # (n,)
    for index, (label, generator, text) in enumerate(zip(labels, generators, texts, strict=True)):
        if exempt[index]:
            continue
        share = shares.shares[(int(label), generator)]
        if share < 1.0 and text_hash(text) >= share * 2**64:
            keep[index] = False
            report.drop("precap", LABEL_ORIGINS[int(label)])

    return keep  # (n,)


def count_reading_changes(original: str, origin: str, report: Report) -> None:
    """Which steps of the standard reading changed a text, each counted with an example."""
    for step, pattern in READING_STEPS:
        hit = not unicodedata.is_normalized("NFC", original) if pattern is None else pattern.search(original) is not None
        if hit:
            report.change(step, origin, repr(original[:120]))


def reread_changed(texts: list[str], cleaned: list[str | None], origins: list[str], report: Report) -> list[str | None]:
    """The standard reading again for texts a rule rewrote, so a rewrite leaves no stray whitespace."""
    positions = [index for index, (text, clean) in enumerate(zip(texts, cleaned, strict=True)) if clean is not None and clean is not text]
    if not positions:
        return cleaned
    redone = standard_reading(pa.array([cleaned[index] for index in positions], type=pa.string())).to_pylist()
    reread = list(cleaned)
    for index, text in zip(positions, redone, strict=True):
        if text.strip():
            reread[index] = text
        else:
            reread[index] = None
            report.drop("empty_after_rules", origins[index])

    return reread


def standardize_table(table: pa.Table, shares: GroupShares, report: Report) -> pa.Table:
    """A read shard's records, pre-capped, read the standard way and without their tells."""
    table = table.filter(pa.array(precap_mask(table, shares, report)))
    originals = table.column("text").to_pylist()
    texts = standard_reading(table.column("text").combine_chunks()).to_pylist()
    origins = [LABEL_ORIGINS[label] for label in table.column("label").to_pylist()]
    exempt = exempt_rows(table).tolist()
    cleaned = remove_tells(texts, table.column("prompt").to_pylist(), origins, exempt, report)
    cleaned = reread_changed(texts, cleaned, origins, report)
    first = pc.is_valid(table.column("first_corpus_id")).to_pylist()
    unchanged: list[bool] = []
    for original, text, origin, is_first in zip(originals, cleaned, origins, first, strict=True):
        unchanged.append(bool(is_first) and text == original)
        if is_first and text != original:
            count_reading_changes(original, origin, report)
    keep = pa.array([text is not None for text in cleaned])
    table = table.set_column(table.schema.get_field_index("text"), "text", pa.array(cleaned, type=pa.string()))
    table = table.set_column(table.schema.get_field_index("text_unchanged"), "text_unchanged", pa.array(unchanged, type=pa.bool_()))
    report.count("first_corpus", "rows", sum(1 for flag in first if flag))
    report.count("first_corpus", "text_unchanged", sum(unchanged))

    return conform(table.filter(keep), STANDARD_SCHEMA)


def standardize_shard(source_path: Path, target_path: Path, shares: GroupShares) -> JsonObject:
    """Standardize one read shard; writes the shard and then its report, and returns the report."""
    table = pq.read_table(source_path)
    report = Report(rows_in=table.num_rows)
    table = standardize_table(table, shares, report)
    report.rows_out = table.num_rows
    report.check_balance(str(source_path))
    write_parquet(table, target_path)
    payload = report.to_json()
    write_json(payload, target_path.with_suffix(".json"))

    return payload
