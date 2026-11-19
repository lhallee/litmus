"""The columns of each stage's shards. A stage conforms its output to its schema before it writes."""

from __future__ import annotations

import pyarrow as pa


TEXT_FIELDS = [
    pa.field("text", pa.string(), nullable=False),
    pa.field("label", pa.int8(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("generator", pa.string(), nullable=False),
    pa.field("provider", pa.string(), nullable=False),
    pa.field("domain", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
    pa.field("date", pa.string()),  # ISO date of the text, or of its source, or null
    pa.field("mix", pa.string(), nullable=False),
    pa.field("benchmark", pa.string(), nullable=False),
    pa.field("first_corpus_id", pa.int64()),  # the row's `id` in the first corpus, or null
    pa.field("first_doc", pa.uint64()),  # the first corpus's `doc` of the row, or null
    pa.field("first_whole", pa.bool_()),  # the first corpus's `whole` of the row, or null
    pa.field("text_unchanged", pa.bool_(), nullable=False),  # standardized text is byte-identical to the first corpus's
]

# Read: one raw text a row, with the prompt it answered when the source gives one (for prompt echoes).
READ_SCHEMA = pa.schema([*TEXT_FIELDS, pa.field("prompt", pa.string())])

# Standardize: the standard reading, prompts spent.
STANDARD_SCHEMA = pa.schema(TEXT_FIELDS)

# Strata: format, document hash, tokens, and where the text may be cut, as character offsets and the token
# index at each.
STRATA_SCHEMA = pa.schema(
    [
        *TEXT_FIELDS,
        pa.field("format", pa.string(), nullable=False),
        pa.field("doc", pa.uint64(), nullable=False),
        pa.field("tokens", pa.int32(), nullable=False),
        pa.field("cut_chars", pa.list_(pa.int32()), nullable=False),
        pa.field("cut_tokens", pa.list_(pa.int32()), nullable=False),
    ]
)

# Passages: one row a passage, whole texts and cuts.
PASSAGE_FIELDS = [
    *TEXT_FIELDS,
    pa.field("format", pa.string(), nullable=False),
    pa.field("doc", pa.uint64(), nullable=False),
    pa.field("tokens", pa.int32(), nullable=False),
    pa.field("whole", pa.bool_(), nullable=False),
    pa.field("long_eval", pa.bool_(), nullable=False),  # a whole document over 8,192 tokens, for evaluation only
]
PASSAGE_SCHEMA = pa.schema(PASSAGE_FIELDS)

# Checks onward: the keys duplicates and the hold-out are found by.
KEYED_SCHEMA = pa.schema(
    [*PASSAGE_FIELDS, pa.field("text_key", pa.uint64(), nullable=False), pa.field("opening_key", pa.uint64(), nullable=False)]
)

# The corpus, as the project README's row contract states it.
CORPUS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("label", pa.int8(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("generator", pa.string(), nullable=False),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("era", pa.string(), nullable=False),  # human, legacy (2023 or earlier), recent, or unknown
        pa.field("domain", pa.string(), nullable=False),
        pa.field("format", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("doc", pa.uint64(), nullable=False),
        pa.field("tokens", pa.int32(), nullable=False),
        pa.field("whole", pa.bool_(), nullable=False),
        pa.field("long_eval", pa.bool_(), nullable=False),
        pa.field("mix", pa.string(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("first_corpus_id", pa.int64()),
        pa.field("text_unchanged", pa.bool_(), nullable=False),
    ]
)


def conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """The table's columns in the schema's order and types; a missing column or a null in a non-null one fails."""
    missing = [name for name in schema.names if name not in table.column_names]
    assert not missing, f"missing columns {missing}"
    columns = []
    for schema_field in schema:
        column = table.column(schema_field.name).cast(schema_field.type)
        assert schema_field.nullable or column.null_count == 0, f"nulls in non-null column {schema_field.name}"
        columns.append(column)

    return pa.Table.from_arrays(columns, schema=schema)
