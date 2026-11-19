"""The benchmark catalog (`catalog.csv`) and the loader that turns one of its rows into texts with origins.

A row names a Hub dataset split exactly (`repo`, `config`, `split`, `revision`), because the corpus builder holds every
listed split out of Litmus's training data by that key. A `config` written `file:<path>` names one file of the
repository, read whole, and its `split` is `train`, the name the datasets library gives a file loaded on its own.

Origins are `human`, `ai` and `mixed`. `label_map` maps a label value, as text, to an origin, and the key `*` maps
every value not listed; a row with no `label_column` takes its origin from `*` alone. A `text_column` naming several
columns, comma-separated, reads each listed column as its own text, with the column name as the label, which is how
paired benchmarks (HC3, Beemo) store their versions; a cell holding a list of texts, or a list of dicts of texts, gives
one row per text.
"""

from __future__ import annotations

import ast
import csv
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path


ORIGINS = ("human", "ai", "mixed")
ANY_LABEL = "*"
FILE_PREFIX = "file:"
CATALOG_COLUMNS = (
    "name", "repo", "config", "split", "revision", "rows", "text_column", "label_column", "label_map",
    "generator_column", "domain_column", "attack_column", "language", "license", "metric", "protocol",
    "labels_public", "url", "notes", "row_filter", "in_distribution",
)
TEXTS_COLUMNS = ("benchmark", "row", "text", "origin", "generator", "domain", "attack", "language", "words")


@dataclass(frozen=True)
class CatalogEntry:
    """One benchmark split as `catalog.csv` lists it."""

    name: str
    repo: str
    config: str
    split: str
    revision: str
    text_column: str
    label_column: str
    label_map: dict[str, str]
    generator_column: str
    domain_column: str
    attack_column: str
    language: str
    metric: str
    labels_public: str
    row_filter: dict[str, str | list[str]]
    in_distribution: bool  # Litmus trained on this benchmark's family (its training split), so it is not an unseen test

    @property
    def text_columns(self) -> list[str]:
        return [column.strip() for column in self.text_column.split(",") if column.strip()]

    @property
    def is_melted(self) -> bool:
        return len(self.text_columns) > 1


def read_catalog(path: Path) -> list[CatalogEntry]:
    """Every row of the catalog, checked: known columns, origins in `label_map`, a revision on every row."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CATALOG_COLUMNS:
            raise ValueError(f"{path} columns are {reader.fieldnames}, expected {list(CATALOG_COLUMNS)}")
        entries = [parse_entry(line) for line in reader]

    keys = [(entry.repo, entry.config, entry.split, json.dumps(entry.row_filter, sort_keys=True)) for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path} lists a split twice")

    return entries


def parse_entry(line: dict[str, str]) -> CatalogEntry:
    label_map = json.loads(line["label_map"] or "{}")
    unknown = sorted(set(label_map.values()) - set(ORIGINS))
    if unknown:
        raise ValueError(f"{line['name']}: label_map names unknown origins {unknown}")
    if not line["revision"].strip():
        raise ValueError(f"{line['name']}: every catalog row pins a revision")
    if line["in_distribution"].strip() not in ("yes", "no"):
        raise ValueError(f"{line['name']}: in_distribution is yes or no, not {line['in_distribution']!r}")

    return CatalogEntry(
        name=line["name"].strip(), repo=line["repo"].strip(), config=line["config"].strip(), split=line["split"].strip(),
        revision=line["revision"].strip(), text_column=line["text_column"].strip(),
        label_column=line["label_column"].strip(), label_map={str(key): value for key, value in label_map.items()},
        generator_column=line["generator_column"].strip(), domain_column=line["domain_column"].strip(),
        attack_column=line["attack_column"].strip(), language=line["language"].strip(), metric=line["metric"].strip(),
        labels_public=line["labels_public"].strip(), row_filter=json.loads(line["row_filter"] or "{}"),
        in_distribution=line["in_distribution"].strip() == "yes",
    )


def origin_of(label: object, label_map: dict[str, str]) -> str:
    """The origin a label maps to; a label the map does not name and no `*` covers is an error, never a guess."""
    key = str(label)
    if key in label_map:
        return label_map[key]
    if ANY_LABEL in label_map:
        return label_map[ANY_LABEL]
    raise KeyError(f"label {key!r} is not in the label map {label_map}")


def holds_texts(value: object) -> bool:
    """Whether a parsed literal is a list or dict whose leaves are texts, numbers or nothing, with a text among them."""
    def leaves(item: object) -> list[object]:
        if isinstance(item, dict):
            return [leaf for inner in item.values() for leaf in leaves(inner)]
        if isinstance(item, (list, tuple)):
            return [leaf for inner in item for leaf in leaves(inner)]
        return [item]

    found = leaves(value)
    return (isinstance(value, (list, tuple, dict)) and any(isinstance(leaf, str) for leaf in found)
            and all(leaf is None or isinstance(leaf, (str, bool, int, float)) for leaf in found))


def cell_texts(cell: object) -> list[str]:
    """The texts one cell holds: a string, a list of strings, or a list of dicts of strings (also stored as text)."""
    if cell is None or isinstance(cell, (bool, int, float, np.number)):
        return []  # a missing cell, or a number beside the texts of a nested cell (a score, a count)
    if isinstance(cell, bytes):  # DroidCollection stores its code as bytes
        return [cell.decode("utf-8", errors="replace")]
    if isinstance(cell, str):
        stripped = cell.strip()
        if stripped.startswith(("[", "{")):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return [cell]
            # A stored list or dict of texts, and only that: source code can also be a Python literal (`[...]`).
            return cell_texts(parsed) if holds_texts(parsed) else [cell]
        return [cell]
    if isinstance(cell, dict):
        return [text for value in cell.values() for text in cell_texts(value)]
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [text for value in cell for text in cell_texts(value)]
    raise TypeError(f"cannot read texts from a cell of type {type(cell).__name__}")


def records_to_texts(entry: CatalogEntry, table: pd.DataFrame) -> pd.DataFrame:
    """One row per text with its origin and strata, in the benchmark's own row order."""
    if not entry.text_columns:
        raise ValueError(f"{entry.name}: no text column is set (gated or unverified); fill the catalog row first")
    for column, value in entry.row_filter.items():  # a value, or a list of values to keep (as `results.catalog_filters` reads it)
        kept = [str(each) for each in value] if isinstance(value, list) else [str(value)]
        table = table[table[column].astype(str).isin(kept)]
    if entry.row_filter and table.empty:
        raise ValueError(f"{entry.name}: the row filter {entry.row_filter} keeps no row")

    def strata(column: str) -> pd.Series:
        return table[column].astype(str) if column else pd.Series([""] * len(table), index=table.index)

    generator, domain, attack = strata(entry.generator_column), strata(entry.domain_column), strata(entry.attack_column)
    rows: list[dict[str, object]] = []
    for position, index in enumerate(table.index):
        for column in entry.text_columns:
            if entry.is_melted:
                origin = origin_of(column, entry.label_map)
            elif entry.label_column:
                origin = origin_of(table.at[index, entry.label_column], entry.label_map)
            else:
                origin = origin_of("", entry.label_map)
            # A melted version (Beemo's expert edit, HC3's ChatGPT answer) is its own stratum where the row names none.
            row_generator = column if entry.is_melted and not entry.generator_column else generator[index]
            row_attack = column if entry.is_melted and not entry.attack_column else attack[index]
            for text in cell_texts(table.at[index, column]):
                rows.append({"benchmark": entry.name, "row": position, "text": text, "origin": origin,
                             "generator": "human" if origin == "human" and entry.is_melted else row_generator,
                             "domain": domain[index], "attack": row_attack, "language": entry.language,
                             "words": len(text.split())})

    texts = pd.DataFrame(rows, columns=list(TEXTS_COLUMNS))
    assert texts["origin"].isin(ORIGINS).all(), "every text has a known origin"
    return texts


def sample_texts(texts: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """A uniform sample of at most `max_rows` texts without replacement, so the benchmark's class mix is kept."""
    if max_rows is None or len(texts) <= max_rows:
        return texts.reset_index(drop=True)
    picked = np.sort(np.random.default_rng(seed).choice(len(texts), size=max_rows, replace=False))  # (max_rows,)
    return texts.iloc[picked].reset_index(drop=True)


def read_file_table(path: Path) -> pd.DataFrame:
    """A downloaded benchmark file as a table, read by its extension; a TSV's unnamed index column is dropped."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", index_col=0)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"no reader for {path.suffix} files")


def load_table(entry: CatalogEntry) -> pd.DataFrame:
    """The catalog split's raw table at its pinned revision.

    A `file:` config is downloaded as that one file and read with pandas, because some benchmark repos (MAGE, HC3,
    AuTexTification) load through scripts the datasets library no longer runs.
    """
    if entry.config.startswith(FILE_PREFIX):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(entry.repo, entry.config[len(FILE_PREFIX):], repo_type="dataset", revision=entry.revision)
        return read_file_table(Path(path))

    import datasets

    return datasets.load_dataset(entry.repo, entry.config, split=entry.split, revision=entry.revision).to_pandas()


def load_benchmark(entry: CatalogEntry, *, max_rows: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Download the catalog split at its pinned revision and return its texts (`TEXTS_COLUMNS`), sampled to `max_rows`."""
    if entry.labels_public.startswith("no"):
        raise ValueError(f"{entry.name}: test labels are hidden; score it through the benchmark's submission route")
    return sample_texts(records_to_texts(entry, load_table(entry)), max_rows, seed)


def find_entry(entries: list[CatalogEntry], name: str) -> CatalogEntry:
    matches = [entry for entry in entries if entry.name == name]
    if len(matches) != 1:
        raise KeyError(f"{len(matches)} catalog rows are named {name!r}")
    return matches[0]
