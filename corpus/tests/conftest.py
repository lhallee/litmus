"""Fixtures with no network: a small byte-level BPE tokenizer of ModernBERT's kind, catalogs, fake streams."""

from __future__ import annotations

import csv
import datasets
import pyarrow as pa
import pytest

from collections.abc import Callable
from pathlib import Path
from litmus_corpus.catalog import SOURCE_COLUMNS, SourceSpec, parse_source
from litmus_corpus.records import conform
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


TRAINING_TEXT = [
    "The quick brown fox jumps over the lazy dog. A second sentence follows here!",
    "def add(left, right):\n    return left + right\n",
    "| name | value |\n| --- | --- |\n| a | 1 |",
    "Les chats dorment au soleil. Ils sont heureux.",
    "这是一个句子。这是另一个句子。",
] * 20


@pytest.fixture(scope="session")
def tokenizer_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A byte-level BPE tokenizer with ModernBERT's normalizer and pre-tokenizer, trained on a few lines."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["[CLS]", "[SEP]"])
    tokenizer.train_from_iterator(TRAINING_TEXT, trainer)
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))

    return str(path)


def catalog_line(**values: str) -> dict[str, str]:
    """A catalog line with every column, defaults filled."""
    line = {column: "" for column in SOURCE_COLUMNS}
    line.update({"repo": "fixture/repo", "split": "train", "revision": "abc123", "domain": "web", "language": "en", "priority": "1"})
    line.update(values)

    return line


def spec_of(**values: str) -> SourceSpec:
    return parse_source(catalog_line(**values))


def write_catalog(path: Path, lines: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*SOURCE_COLUMNS, "row_filter"])
        writer.writeheader()
        writer.writerows(lines)

    return path


BENCHMARK_FIELDS = [
    "name", "repo", "config", "split", "revision", "text_column", "label_column", "label_map", "generator_column", "language", "license", "row_filter",
]


def write_benchmarks(path: Path, lines: list[dict[str, str]]) -> Path:
    """A benchmark catalog with the benchmark catalog's columns."""
    defaults = {field: "" for field in BENCHMARK_FIELDS} | {"revision": "abc123", "text_column": "text", "label_map": "{}", "language": "en"}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows([defaults | line for line in lines])

    return path


Rows = list[dict[str, object]]


def text_table(texts: list[str], schema: pa.Schema, **columns: list[object]) -> pa.Table:
    """Records of `texts` in `schema`, model text of one source by default; `columns` overrides any column."""
    count = len(texts)
    defaults: dict[str, list[object]] = {
        "text": texts,
        "label": [1] * count,
        "source": ["s"] * count,
        "generator": ["gpt-4o"] * count,
        "provider": ["openai"] * count,
        "domain": ["web"] * count,
        "language": ["en"] * count,
        "date": [None] * count,
        "mix": [""] * count,
        "benchmark": [""] * count,
        "first_corpus_id": [None] * count,
        "first_doc": [None] * count,
        "first_whole": [None] * count,
        "text_unchanged": [False] * count,
        "prompt": [None] * count,
    }
    values = defaults | columns

    return conform(pa.table({name: pa.array(values[name], type=schema.field(name).type) for name in schema.names if name in values}), schema)


@pytest.fixture
def fake_streams(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Rows]], None]:
    """Serve rows from memory in place of the Hub: `fake_streams({repo: rows})`, or `{"repo/split": rows}` for one split.

    A repo absent from the map fails to open, as a gated or moved repository does.
    """

    def install(rows_by_repo: dict[str, Rows]) -> None:
        def open_stream(spec: SourceSpec, split: str) -> datasets.IterableDataset:
            rows = rows_by_repo.get(f"{spec.repo}/{split}", rows_by_repo.get(spec.repo))
            if rows is None:
                raise FileNotFoundError(f"{spec.repo} cannot be opened")
            return datasets.Dataset.from_list(rows).to_iterable_dataset(num_shards=min(2, len(rows)))

        monkeypatch.setattr("litmus_corpus.read.open_stream", open_stream)

    return install
