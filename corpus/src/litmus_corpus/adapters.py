"""Raw records from a batch of a source's rows: one text a record, with its origin, generator and date.

Most sources need no code: the generic adapter follows the catalog's column expressions (`text_paths.py`).
A plain column is read by its Arrow type, without leaving Arrow:

    string                 one text a row
    list of strings        one text an element (HC3's answers)
    list of messages       a conversation: each assistant turn is model text, its reasoning block kept;
                           a user turn is human text only when its date is before 2022, otherwise dropped;
                           system and tool turns are dropped

A path (`messages[].content (role=assistant)`, `conversations[].info.answer_content`, `messages (JSON)`) is
followed row by row. A source whose rows need more registers its own adapter by name with `@register(name)`;
the first corpus has one. A column of another type, or one that is missing, raises `SchemaMismatch` naming
the source.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from collections.abc import Callable

from .catalog import FIRST_CORPUS, LABEL_KEY_SEPARATOR, SourceSpec, value_key
from .config import ORIGIN_LABELS
from .generators import maker
from .records import READ_SCHEMA, conform
from .storage import Report
from .text_paths import COLUMN_NAME, TextPath, generator_at, texts_at, with_reasoning


Adapter = Callable[[pa.Table, SourceSpec, Report], pa.Table]
ADAPTERS: dict[str, Adapter] = {}

HUMAN_DATE_LIMIT = "2022-01-01"  # human text must be dated before this
GROUP_SEPARATOR = "|"  # between origin and generator in a read report's group counts
ROLE_KEYS = ("role", "from", "speaker", "author")
CONTENT_KEYS = ("content", "value", "text", "message")
ASSISTANT_ROLES = frozenset({"assistant", "gpt", "bot", "model", "chatgpt", "ai", "chatbot"})
USER_ROLES = frozenset({"user", "human", "prompter", "customer"})
DROPPED_ROLES = frozenset({"system", "tool", "function", "ipython", "observation", "developer"})
ISO_DATE_PATTERN = re.compile(r"\b(19\d\d|20\d\d)(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?")
DAY_FIRST_PATTERN = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](19\d\d|20\d\d)\b")
EXPANDED_SCHEMA = pa.schema(
    [
        pa.field("parent", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("origin", pa.string()),
        pa.field("prompt", pa.string()),
        pa.field("generator", pa.string()),
    ]
)


class SchemaMismatch(ValueError):
    """A source's rows do not have the shape its catalog line says."""


def register(name: str) -> Callable[[Adapter], Adapter]:
    def add(adapter: Adapter) -> Adapter:
        assert name not in ADAPTERS, f"two adapters for {name}"
        ADAPTERS[name] = adapter
        return adapter

    return add


def adapter_for(spec: SourceSpec) -> Adapter:
    return ADAPTERS.get(spec.name, generic_adapter)


def iso_date(year: int, month: int, day: int) -> str | None:
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_date(value: object) -> str | None:
    """An ISO date from a date column's value: a date, a timestamp, an epoch, a year, or a string holding one.

    A value with a year but no month or day takes the year's last day, so a text is never dated earlier than
    its value allows. Returns None when the value holds no date.
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if 1900 <= number <= 2100:
            return f"{int(number)}-12-31"
        seconds = number / 1000 if number > 1e11 else number
        return dt.datetime.fromtimestamp(seconds, tz=dt.UTC).date().isoformat()
    text = str(value)
    day_first = DAY_FIRST_PATTERN.search(text)
    if day_first:
        first, second, year = (int(part) for part in day_first.groups())
        month, day = (first, second) if first <= 12 else (second, first)
        return iso_date(year, month, day)
    found = ISO_DATE_PATTERN.search(text)
    if found is None:
        return None
    year = int(found.group(1))
    if found.group(2) is None:
        return f"{year}-12-31"
    month = int(found.group(2))
    if not 1 <= month <= 12:
        return f"{year}-12-31"
    if found.group(3) is None:
        return (dt.date(year + month // 12, month % 12 + 1, 1) - dt.timedelta(days=1)).isoformat()

    return iso_date(year, month, int(found.group(3)))


def mapped_values(column: pa.ChunkedArray | pa.Array, mapping: Callable[[object], str | None]) -> pa.Array:
    """`mapping` of each value, computed once per distinct value."""
    distinct = pc.unique(column)
    images = pa.array([mapping(value) for value in distinct.to_pylist()], type=pa.string())
    positions = pc.index_in(column, value_set=distinct)

    return images.take(positions)


def row_origins(table: pa.Table, spec: SourceSpec, path: TextPath) -> pa.Array:
    """The origin of each row's text under one text expression: an origin, or `drop`."""
    if spec.origin != "by_column":
        return pa.array([spec.origin] * table.num_rows, type=pa.string())
    if spec.label_column is None:
        origin = spec.column_origin(path)
        assert origin is not None, f"source {spec.name}: no origin for {path.expression}"
        return pa.array([origin] * table.num_rows, type=pa.string())

    def origin_of(value: object) -> str:
        key = value_key(value)
        origin = spec.label_of(key)
        if origin is None:
            raise SchemaMismatch(f"source {spec.name}: label {key!r} is not in its label_map")
        return origin

    columns = spec.label_columns
    if len(columns) == 1:
        return mapped_values(table.column(columns[0]), origin_of)
    keys = [mapped_values(table.column(column), lambda value: value_key(value) if value is not None else "") for column in columns]

    return mapped_values(pc.binary_join_element_wise(*keys, LABEL_KEY_SEPARATOR), origin_of)


def group_hash(source: str, value: object) -> int:
    """A group's 64-bit hash, its `doc`: BLAKE2b of the source's name and the group's value."""
    return int.from_bytes(hashlib.blake2b(f"{source}|{value_key(value)}".encode(), digest_size=8).digest(), "big")


def row_groups(table: pa.Table, spec: SourceSpec) -> pa.Array:
    """Each row's group hash (uint64), or nulls for a source without a `group_column`."""
    if spec.group_column is None:
        return pa.nulls(table.num_rows, pa.uint64())
    return mapped_values(table.column(spec.group_column), lambda value: None if value is None else str(group_hash(spec.name, value))).cast(pa.uint64())


def sampled_groups(table: pa.Table, spec: SourceSpec, report: Report) -> pa.Table:
    """The rows of the groups `group_share` keeps, each group kept or left whole by its hash; the rest are counted
    as `group_sample`. A row without a group fails, since it could not be kept whole."""
    if spec.group_column is None or spec.group_share >= 1.0:
        return table
    groups = row_groups(table, spec)
    if groups.null_count:
        raise SchemaMismatch(f"source {spec.name}: {groups.null_count} rows have no {spec.group_column}")
    keep = np.asarray(groups.to_numpy(zero_copy_only=False), dtype=np.uint64) < np.uint64(int(spec.group_share * 2**64 - 1))  # (n,)
    report.drop("group_sample", "any", int((~keep).sum()))

    return table.filter(pa.array(keep))


def row_generators(table: pa.Table, spec: SourceSpec, index: int) -> pa.Array | None:
    """Each row's generator for the `index`-th text expression, or None when it must be read per text."""
    path = spec.generator_path(index)
    if path is None:
        return pa.array([spec.generator] * table.num_rows, type=pa.string())
    if path.expression == COLUMN_NAME:
        return pa.array([spec.text_paths[index].column] * table.num_rows, type=pa.string())
    if path.is_plain:
        return mapped_values(table.column(path.column), lambda value: None if value is None else str(value).strip() or None)

    return None


def row_dates(table: pa.Table, spec: SourceSpec) -> pa.Array:
    if spec.date_column is not None:
        column, *fields = spec.date_column.split(".")
        values = table.column(column).combine_chunks()
        for name in fields:
            if not pa.types.is_struct(values.type):
                raise SchemaMismatch(f"source {spec.name}: date column {spec.date_column} does not lead to a field of {values.type}")
            values = pc.struct_field(values, name)
        return mapped_values(values, parse_date)

    return pa.array([spec.source_date] * table.num_rows, type=pa.string())


def message_field(message: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in message:
            return key

    return None


def message_text(content: object) -> str | None:
    """A message's text: a string, or the text parts of a list of parts; None when a part is not text."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                return None
            parts.append(part.get("text") or "")
        return "\n".join(parts)

    return None


def conversation_texts(
    conversations: list[object], origins: list[str], dates: list[str | None], spec: SourceSpec, report: Report
) -> tuple[list[int], list[str], list[str], list[str | None]]:
    """Each kept turn of each conversation: its row, text, origin and prompt."""
    parents: list[int] = []
    texts: list[str] = []
    turn_origins: list[str] = []
    prompts: list[str | None] = []
    for row, (conversation, origin, date) in enumerate(zip(conversations, origins, dates, strict=True)):
        if isinstance(conversation, str):
            try:
                conversation = json.loads(conversation)
            except json.JSONDecodeError:
                conversation = None
        if not isinstance(conversation, list):
            report.drop("null_text", origin)
            continue
        prompt: str | None = None
        for message in conversation:
            if isinstance(message, str) and message.lstrip().startswith("{"):
                message = json.loads(message)  # a message kept as JSON text; malformed JSON fails loudly
            role_key = message_field(message, ROLE_KEYS) if isinstance(message, dict) else None
            content_key = message_field(message, CONTENT_KEYS) if isinstance(message, dict) else None
            if role_key is None or content_key is None:
                raise SchemaMismatch(f"source {spec.name}: a message has no role or content: {str(message)[:200]}")
            role = str(message[role_key]).strip().lower()
            text = message_text(message[content_key])
            if role in DROPPED_ROLES:
                report.drop(f"{role}_turn", origin)
                continue
            if role not in ASSISTANT_ROLES and role not in USER_ROLES:
                raise SchemaMismatch(f"source {spec.name}: unknown role {role!r}")
            if text is None:
                report.drop("non_text_turn", origin)
                continue
            if role in USER_ROLES:
                prompt = text
                if origin != "human" and (date is None or date >= HUMAN_DATE_LIMIT):
                    report.drop("user_turn_not_before_2022", "human")
                    continue
                parents.append(row)
                texts.append(text)
                turn_origins.append("human")
                prompts.append(None)
                continue
            parents.append(row)
            texts.append(with_reasoning(text, message))
            turn_origins.append(origin)
            prompts.append(prompt)

    return parents, texts, turn_origins, prompts


def plain_texts(table: pa.Table, spec: SourceSpec, path: TextPath, dates: pa.Array, generators: pa.Array, report: Report) -> pa.Table:
    """The texts of a plain column, by its Arrow type."""
    values = table.column(path.column).combine_chunks()
    origins = row_origins(table, spec, path)
    kind = values.type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        parents = pa.array(np.arange(table.num_rows), type=pa.int64())  # (n,)
        return pa.table(
            {"parent": parents, "text": values.cast(pa.string()), "origin": origins, "prompt": pa.nulls(table.num_rows, pa.string()), "generator": generators},
            schema=EXPANDED_SCHEMA,
        )
    is_list = pa.types.is_list(kind) or pa.types.is_large_list(kind)
    if is_list and (pa.types.is_string(kind.value_type) or pa.types.is_large_string(kind.value_type)):
        parents = pc.list_parent_indices(values).cast(pa.int64())
        texts = pc.list_flatten(values).cast(pa.string())
        return pa.table(
            {"parent": parents, "text": texts, "origin": origins.take(parents), "prompt": pa.nulls(len(texts), pa.string()), "generator": generators.take(parents)},
            schema=EXPANDED_SCHEMA,
        )
    if is_list and pa.types.is_struct(kind.value_type):
        parents, texts, turn_origins, prompts = conversation_texts(values.to_pylist(), origins.to_pylist(), dates.to_pylist(), spec, report)
        parent_array = pa.array(parents, type=pa.int64())
        return pa.table(
            {"parent": parent_array, "text": texts, "origin": turn_origins, "prompt": prompts, "generator": generators.take(parent_array)},
            schema=EXPANDED_SCHEMA,
        )

    raise SchemaMismatch(f"source {spec.name}: column {path.column} has type {kind}, not text, a list of text, or a conversation")


def path_texts(table: pa.Table, spec: SourceSpec, path: TextPath, generators: pa.Array | None, report: Report) -> pa.Table:
    """The texts a path finds, row by row, each with its generator from the same element or the row."""
    index = spec.text_paths.index(path)
    generator_path = spec.generator_path(index)
    columns = [*path.columns] + ([generator_path.column] if generators is None and generator_path is not None else [])
    rows = table.select(list(dict.fromkeys(columns))).to_pylist()
    origins = row_origins(table, spec, path).to_pylist()
    row_generator = generators.to_pylist() if generators is not None else None
    if path.is_json and len(path.steps) == 1 and path.keep is None:
        parents, texts, turn_origins, prompts = conversation_texts(
            [row.get(path.column) for row in rows], origins, row_dates(table, spec).to_pylist(), spec, report
        )
        found_generators = [row_generator[parent] if row_generator is not None else None for parent in parents]
        return pa.table({"parent": parents, "text": texts, "origin": turn_origins, "prompt": prompts, "generator": found_generators}, schema=EXPANDED_SCHEMA)
    parents_found: list[int] = []
    texts_found: list[str] = []
    origins_found: list[str] = []
    prompts_found: list[str | None] = []
    generators_found: list[str | None] = []
    for row_index, row in enumerate(rows):
        for found in texts_at(row, path):
            parents_found.append(row_index)
            texts_found.append(found.text)
            origins_found.append(origins[row_index])
            prompts_found.append(found.prompt)
            generators_found.append(row_generator[row_index] if row_generator is not None else generator_at(row, found, path, generator_path))

    return pa.table(
        {"parent": parents_found, "text": texts_found, "origin": origins_found, "prompt": prompts_found, "generator": generators_found},
        schema=EXPANDED_SCHEMA,
    )


def filtered_rows(table: pa.Table, spec: SourceSpec, report: Report) -> pa.Table:
    """The rows that match every entry of the source's `row_filter`; the rest are counted as `row_filter`."""
    if not spec.row_filter:
        return table
    keep = pa.array(np.ones(table.num_rows, dtype=bool))
    for column, values in spec.row_filter.items():
        as_text = mapped_values(table.column(column), lambda value: None if value is None else value_key(value))
        keep = pc.and_(keep, pc.is_in(as_text, value_set=pa.array(values, type=pa.string())).fill_null(False))
    report.drop("row_filter", "any", table.num_rows - int(np.asarray(keep).sum()))

    return table.filter(keep)


def check_columns(table: pa.Table, spec: SourceSpec) -> None:
    missing = [column for column in spec.needed_columns() if column not in table.column_names]
    if missing:
        raise SchemaMismatch(f"source {spec.name}: columns {missing} are not in {table.column_names}")


def generic_adapter(table: pa.Table, spec: SourceSpec, report: Report) -> pa.Table:
    """Read records from a batch of rows by the source's column expressions."""
    check_columns(table, spec)
    table = sampled_groups(filtered_rows(table, spec, report), spec, report)
    dates = row_dates(table, spec)
    pieces: list[pa.Table] = []
    for index, path in enumerate(spec.text_paths):
        generators = row_generators(table, spec, index)
        if path.is_plain and generators is not None:
            pieces.append(plain_texts(table, spec, path, dates, generators, report))
        else:
            pieces.append(path_texts(table, spec, path, generators, report))
    expanded = pa.concat_tables(pieces)

    return finish_records(expanded, dates, spec, report, row_groups(table, spec))


def finish_records(expanded: pa.Table, dates: pa.Array, spec: SourceSpec, report: Report, groups: pa.Array | None = None) -> pa.Table:
    """Records in `READ_SCHEMA` from texts with their parent rows; drops counted by reason and origin. `groups`, each
    parent row's group hash, becomes the record's `first_doc`, the `doc` every row of its group shares."""
    origins = expanded.column("origin").combine_chunks()
    texts = expanded.column("text").combine_chunks()
    parents = expanded.column("parent").combine_chunks()
    is_human = pc.equal(origins, "human")
    generator = pc.if_else(is_human, "human", expanded.column("generator").combine_chunks())
    keep = np.ones(len(texts), dtype=bool)  # (n,)
    reasons = (
        ("label_drop", pc.equal(origins, "drop")),
        ("null_text", pc.or_(pc.is_null(texts), pc.equal(pc.utf8_trim_whitespace(texts.fill_null("")), ""))),
        ("no_generator", pc.and_(pc.equal(origins, "ai"), pc.is_null(generator))),
    )
    for reason, mask in reasons:
        hit = np.asarray(mask.fill_null(False)) & keep  # (n,)
        for origin, number in zip(*np.unique(np.asarray(origins.filter(pa.array(hit))), return_counts=True), strict=True):
            report.drop(reason, str(origin), int(number))
        keep &= ~hit
    kept = pa.array(keep)
    origins = origins.filter(kept)
    generator = pc.utf8_trim_whitespace(generator.filter(kept).fill_null(""))
    count = len(origins)
    records = pa.table(
        {
            "text": texts.filter(kept),
            "label": mapped_values(origins, lambda value: str(ORIGIN_LABELS[value])).cast(pa.int8()),
            "source": pa.array([spec.name] * count, type=pa.string()),
            "generator": generator,
            "provider": mapped_values(generator, lambda value: maker(value).provider if value not in ("", "human") else value),
            "domain": pa.array([spec.domain] * count, type=pa.string()),
            "language": pa.array([spec.language] * count, type=pa.string()),
            "date": dates.take(parents.filter(kept)),
            "mix": pc.if_else(pc.equal(origins, "mixed"), spec.mix, ""),
            "benchmark": pa.array([spec.benchmark] * count, type=pa.string()),
            "first_corpus_id": pa.nulls(count, pa.int64()),
            "first_doc": pa.nulls(count, pa.uint64()) if groups is None else groups.take(parents.filter(kept)),
            "first_whole": pa.nulls(count, pa.bool_()),
            "text_unchanged": pa.array(np.zeros(count, dtype=bool)),
            "prompt": expanded.column("prompt").combine_chunks().filter(kept),
        }
    )
    count_groups(records, report)

    return conform(records, READ_SCHEMA)


def count_groups(records: pa.Table, report: Report) -> None:
    """Texts per (origin, generator) in the report, which the pre-cap sizes each group's share from."""
    origins = mapped_values(records.column("label"), lambda value: next(name for name, label in ORIGIN_LABELS.items() if label == value))
    keys = pc.binary_join_element_wise(origins, records.column("generator"), GROUP_SEPARATOR)
    for key, number in zip(*np.unique(np.asarray(keys), return_counts=True), strict=True):
        report.count("group", str(key), int(number))


@register(FIRST_CORPUS)
def first_corpus_adapter(table: pa.Table, spec: SourceSpec, report: Report) -> pa.Table:
    """The first corpus's rows as they are: its text, source, generator, domain, `doc`, `whole` and `id`.

    A row from a source the catalog holds out as a benchmark's split gets that benchmark's name in `benchmark`.
    """
    for column in ("id", "text", "label", "source", "generator", "domain", "doc", "whole"):
        if column not in table.column_names:
            raise SchemaMismatch(f"source {spec.name}: column {column} is missing")
    count = table.num_rows
    labels = table.column("label").cast(pa.int8())
    human = pc.equal(labels, ORIGIN_LABELS["human"])
    generators = pc.if_else(human, "human", table.column("generator"))
    records = pa.table(
        {
            "text": table.column("text"),
            "label": labels,
            "source": table.column("source"),
            "generator": generators,
            "provider": mapped_values(generators, lambda value: "human" if value == "human" else maker(value).provider),
            "domain": table.column("domain"),
            "language": pa.array([spec.language] * count, type=pa.string()),
            "date": pc.if_else(human, spec.source_date, pa.scalar(None, pa.string())),
            "mix": pa.array([""] * count, type=pa.string()),
            "benchmark": mapped_values(table.column("source"), lambda value: spec.held_out_first.get(str(value), "")),
            "first_corpus_id": table.column("id"),
            "first_doc": table.column("doc").cast(pa.uint64()),
            "first_whole": table.column("whole"),
            "text_unchanged": pa.array(np.zeros(count, dtype=bool)),
            "prompt": pa.nulls(count, pa.string()),
        }
    )
    count_groups(records, report)

    return conform(records, READ_SCHEMA)
