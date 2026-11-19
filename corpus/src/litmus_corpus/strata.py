"""Stage 3, strata: each text's format, language, document hash, tokens, and where it may be cut.

`language` stays the catalog's unless the catalog says `multi`, when fastText reads it (`language.py`).
`doc` is the 64-bit BLAKE2b hash of the standardized text; every passage cut from the text carries it. A
first-corpus row keeps the first corpus's `doc`, and a raw text linked to a first-corpus passage takes the
passage's (`linking.py`): such a text is dropped here when the passage stands for it. A benchmark text linked
to a first-corpus passage is written to a links file, which marks the passage as that benchmark's.
"""

from __future__ import annotations

import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

from dataclasses import dataclass
from pathlib import Path

from .catalog import MULTI_LANGUAGE
from .config import LABEL_ORIGINS
from .formats import text_format
from .language import detect_languages
from .linking import MATERIAL_LENGTH, NO_MATCH, link_texts, load_index
from .records import STRATA_SCHEMA, conform
from .storage import JsonObject, Report, write_json, write_parquet
from .tokenization import encode_with_cuts


TOKEN_BATCH_TEXTS = 1_000


@dataclass(frozen=True)
class LinkPlan:
    """What strata links to the first corpus: the index, and the sources whose raw texts the first corpus read."""

    index_directory: str | None
    reread_sources: frozenset[str]


def doc_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def link_to_first_corpus(table: pa.Table, cut_chars: list[list[int]], formats: list[str], plan: LinkPlan, report: Report) -> tuple[list[int], list[bool], pa.Table]:
    """Each row's `doc`, whether to keep it, and the benchmark rows' links to first-corpus passages."""
    texts = table.column("text").to_pylist()
    sources = table.column("source").to_pylist()
    benchmarks = table.column("benchmark").to_pylist()
    first_docs = table.column("first_doc").to_pylist()
    origins = [LABEL_ORIGINS[label] for label in table.column("label").to_pylist()]
    docs = [first if first is not None else doc_hash(text) for text, first in zip(texts, first_docs, strict=True)]
    keep = [True] * len(texts)
    candidates = [
        index for index, (source, benchmark, first) in enumerate(zip(sources, benchmarks, first_docs, strict=True))
        if first is None and (benchmark or source in plan.reread_sources)
    ]
    linked_docs: list[int] = []
    linked_benchmarks: list[str] = []
    if plan.index_directory is None or not candidates:
        return docs, keep, pa.table({"first_doc": pa.array([], pa.uint64()), "benchmark": pa.array([], pa.string())})
    index = load_index(plan.index_directory)
    matches = link_texts([texts[position] for position in candidates], [cut_chars[position] for position in candidates], index)
    for position, match in zip(candidates, matches, strict=True):
        origin = origins[position]
        if match == NO_MATCH:
            report.count("first_corpus_link", f"{origin}|{'benchmark' if benchmarks[position] else 'raw'}_unlinked")
            continue
        first_doc = int(index.docs[match])
        if benchmarks[position]:
            linked_docs.append(first_doc)
            linked_benchmarks.append(benchmarks[position])
            report.count("first_corpus_link", f"{origin}|benchmark_linked")
            continue
        docs[position] = first_doc
        material = formats[position] != "plain" or len(texts[position].split()) > MATERIAL_LENGTH * int(index.words[match])
        report.count("first_corpus_link", f"{origin}|raw_linked_{'kept' if material else 'dropped'}")
        if not material:
            keep[position] = False
            report.drop("represented_in_first_corpus", origin)

    return docs, keep, pa.table({"first_doc": pa.array(linked_docs, pa.uint64()), "benchmark": pa.array(linked_benchmarks, pa.string())})


def strata_table(table: pa.Table, tokenizer_path: str, language_model_path: str | None, plan: LinkPlan, report: Report) -> tuple[pa.Table, pa.Table]:
    """A standardized shard with its strata columns added, and the links of its benchmark rows."""
    texts = table.column("text").to_pylist()
    formats = [text_format(text) for text in texts]
    languages = table.column("language").to_pylist()
    multi = [index for index, language in enumerate(languages) if language == MULTI_LANGUAGE]
    if multi:
        assert language_model_path is not None, "a multi-language source needs the language model"
        for index, detected in zip(multi, detect_languages([texts[index] for index in multi], language_model_path), strict=True):
            languages[index] = detected
    tokens: list[int] = []
    cut_chars: list[list[int]] = []
    cut_tokens: list[list[int]] = []
    for start in range(0, len(texts), TOKEN_BATCH_TEXTS):
        batch_tokens, batch_chars, batch_cut_tokens = encode_with_cuts(
            texts[start : start + TOKEN_BATCH_TEXTS], formats[start : start + TOKEN_BATCH_TEXTS], tokenizer_path
        )
        tokens.extend(batch_tokens)
        cut_chars.extend(batch_chars)
        cut_tokens.extend(batch_cut_tokens)
    docs, keep, links = link_to_first_corpus(table, cut_chars, formats, plan, report)
    for label, format_name, language, kept in zip(table.column("label").to_pylist(), formats, languages, keep, strict=True):
        if kept:
            report.count("format", f"{LABEL_ORIGINS[label]}|{format_name}")
            report.count("language", f"{LABEL_ORIGINS[label]}|{language}")
    table = table.set_column(table.schema.get_field_index("language"), "language", pa.array(languages, type=pa.string()))
    table = table.append_column("format", pa.array(formats, type=pa.string()))
    table = table.append_column("doc", pa.array(docs, type=pa.uint64()))
    table = table.append_column("tokens", pa.array(tokens, type=pa.int32()))
    table = table.append_column("cut_chars", pa.array(cut_chars, type=pa.list_(pa.int32())))
    table = table.append_column("cut_tokens", pa.array(cut_tokens, type=pa.list_(pa.int32())))

    return conform(table.filter(pa.array(keep)), STRATA_SCHEMA), links


def strata_shard(source_path: Path, target_path: Path, tokenizer_path: str, language_model_path: str | None, plan: LinkPlan, links_path: Path) -> JsonObject:
    """Add strata to one standardized shard; writes the shard, its benchmark links, and then its report."""
    table = pq.read_table(source_path)
    report = Report(rows_in=table.num_rows)
    table, links = strata_table(table, tokenizer_path, language_model_path, plan, report)
    report.rows_out = table.num_rows
    report.check_balance(str(source_path))
    write_parquet(table, target_path)
    write_parquet(links, links_path)
    payload = report.to_json()
    write_json(payload, target_path.with_suffix(".json"))

    return payload
