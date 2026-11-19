"""The source catalog (`sources.csv`) and the benchmark catalog, read and checked before anything is downloaded.

A line the build cannot use is a catalog problem: the source is left out and named with the reason in the
report and by `python -m litmus_corpus catalog check`, and the build goes on without it.

Conventions this reader takes from `sources.csv`:

- `origin` is `human`, `ai`, `mixed` or `by_column`. A `by_column` source names its origin per row by
  `label_column`, whose values `label_map` maps to an origin, or holds several text expressions, each mapped by
  `label_map`. A `label_map` value may be `drop`, and the key `*` maps every value not listed.
- `text_column` and `generator_column` are column expressions (`text_paths.py`); `split` may join splits with `|`.
  Generator expressions pair with text expressions by position; extra ones are unused (`model_a | model_b`
  beside `conversation_a[].content` reads `model_a`).
- `date_column` may name a field of a struct column (`metadata.year`).
- `generator` is the source's one generator; a note after it in parentheses or after a `;` is not part of the name.
- `config` of the form `file:<path>` reads that one file of the repository.
- An optional `row_filter` column holds a JSON object of column to value (or list of values).
- A source without `date_column` dates its human text by the latest year in `date_evidence`; with neither, its
  human rows are dropped as undated by the checks, and the source is still read for its other rows.
- `in_first_corpus` is `yes` or `no`; `raw re-read: no` in `notes` means the first corpus's rows stand for the
  source, and it is not read raw.
- `priority` is an integer; a build with `--priority N` reads the sources of priority N or lower.
- `label_column` may join columns with ` + ` (`model + attack`): the row's key is their values joined by `|`, and
  a `label_map` key ending in `|*` maps every key that starts with what precedes the `*`.
- An optional `group_column` names the rows derived from one document (RAID's `source_id`): every row of a group
  gets one `doc`, so a split never divides it, and an optional `group_share` keeps that share of the groups,
  chosen whole by a hash of the group.
- An optional `family_split` of the form `<family>:<train|validation|test>` places a flagged line in its benchmark
  family.

Held out of every Litmus set: every split of `benchmarks/catalog.csv` and every `sources.csv` line with a
`benchmark` flag. The benchmark catalog's splits are read, by the benchmark catalog's own columns, and written
as benchmark rows; a flagged `sources.csv` line is not read for training. The report lists where the two lists
disagree. Never read for training: a licence that forbids derivatives (cutting passages is one), and
`pangram/WildAI`, whose labels are a detector's predictions.

A build with `family_train` trains on a family's `train` and `validation` lines and reads its `test` lines as
benchmarks, held out by name and by content; a family qualifies only when its test is held out, by a `test` line
or by a benchmark of the family's name. A `train` line that is a benchmark's split qualifies only when the two
`row_filter`s select disjoint values of one column (RAID extra's code rows beside its news benchmark). Without
`family_train` every flagged line is held out as before. A training source may not take a benchmark's name.
"""

from __future__ import annotations

import csv
import json
import re

from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import FIRST_CORPUS_REPO, FIRST_CORPUS_REVISION, FIRST_CORPUS_SPLITS, ORIGINS, UNLABELED
from .text_paths import COLUMN_NAME, PathError, TextPath, parse_alternatives


SOURCE_COLUMNS = (
    "name", "repo", "config", "split", "revision", "origin", "text_column", "label_column", "label_map",
    "generator_column", "generator", "mix", "date_column", "date_evidence", "domain", "language", "kind", "rows",
    "bytes", "license", "benchmark", "in_first_corpus", "priority", "verified", "notes",
)
BENCHMARK_COLUMNS = ("name", "repo", "config", "split", "revision", "text_column", "label_column", "label_map", "generator_column", "language")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")
COLUMN_PATTERN = re.compile(r"^[\w\- ]+$")
DATE_COLUMN_PATTERN = re.compile(r"^[\w\- ]+(?:\.[\w\- ]+)*$")  # a column, or a field of a struct column (`metadata.year`)
YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20\d\d)\b")
REREAD_PATTERN = re.compile(r"raw re-read: (yes|no)", re.IGNORECASE)
EXPRESSION_SEPARATOR = re.compile(r"\s+\|\s+|\s*,\s*")  # ` | ` in `sources.csv`, `,` in the benchmark catalog
SPLIT_SEPARATOR = "|"
GENERATOR_NOTE = re.compile(r"[(;]")
LABEL_VALUES = frozenset(ORIGINS) | {"drop"}
ANY_VALUE = "*"
MULTI_LANGUAGE = "multi"
FILE_CONFIG_PREFIX = "file:"
BENCHMARK_GENERATOR = "unknown"  # a benchmark's model text that names no generator; never trained on
FIRST_CORPUS = "first_corpus"
NO_DERIVATIVES = re.compile(r"(?:^|-)nd(?:-|$)", re.IGNORECASE)  # cc-by-nd, cc-by-nc-nd
EXCLUDED_REPOS = {"pangram/WildAI": "labels are a detector's predictions, not ground truth"}
LABEL_COLUMN_SEPARATOR = " + "
LABEL_KEY_SEPARATOR = "|"
TRAINED_FAMILY_SPLITS = ("train", "validation")
TEST_FAMILY_SPLIT = "test"
FAMILY_SPLIT_PATTERN = re.compile(r"^([a-z0-9_]+):(train|validation|test)$")


class CatalogError(ValueError):
    """A catalog line the build cannot use."""


@dataclass(frozen=True)
class SourceSpec:
    """One source to read: a line of `sources.csv`, a benchmark, or the first corpus."""

    name: str
    repo: str
    config: str | None
    splits: tuple[str, ...]
    revision: str
    origin: str
    text_paths: tuple[TextPath, ...]
    label_column: str | None
    label_map: dict[str, str]
    generator_paths: tuple[TextPath, ...]  # empty when `generator` names the one generator
    generator: str | None
    mix: str
    date_column: str | None
    date_evidence: str
    domain: str
    language: str
    kind: str
    priority: int
    license: str
    benchmark: str
    rows: int | None = None
    bytes: int | None = None
    row_filter: dict[str, list[str]] = field(default_factory=dict)
    source_date: str | None = None  # the date `date_evidence` gives every text, ISO, or None
    in_first_corpus: bool = False
    reread: bool | None = None  # for a first-corpus source, whether its raw text is worth reading again
    verbatim: bool = False  # rows enter as they are: the first corpus
    held_out_first: dict[str, str] = field(default_factory=dict)  # the first corpus's held-out sources, to their benchmark
    verified: str = ""
    group_column: str | None = None  # rows of one value derive from one document and share a `doc`
    group_share: float = 1.0  # the share of groups kept, chosen whole by a hash of the group
    family: str = ""  # the benchmark family of a flagged line, from `family_split`
    family_split: str = ""  # train, validation or test within the family

    @property
    def label_columns(self) -> tuple[str, ...]:
        """The columns whose joined values name a row's origin (`model + attack` is two)."""
        return () if self.label_column is None else tuple(part.strip() for part in self.label_column.split(LABEL_COLUMN_SEPARATOR.strip()))

    def label_of(self, key: str) -> str | None:
        """The origin `label_map` gives a row's label key: the exact key, else the longest `prefix|*` pattern that
        matches, else `*`, else None."""
        if key in self.label_map:
            return self.label_map[key]
        patterns = sorted((pattern for pattern in self.label_map if pattern.endswith(LABEL_KEY_SEPARATOR + ANY_VALUE)), key=len, reverse=True)
        for pattern in patterns:
            if key.startswith(pattern[:-1]):
                return self.label_map[pattern]

        return self.label_map.get(ANY_VALUE)

    def origins(self) -> frozenset[str]:
        """The origins this source can give."""
        if self.origin == "by_column":
            return frozenset(value for value in self.label_map.values() if value != "drop")

        return frozenset({self.origin})

    def column_origin(self, path: TextPath) -> str | None:
        """The origin `label_map` gives a text expression, for a source told apart by its columns."""
        for key in (path.expression.strip(), path.column):
            if key in self.label_map:
                return self.label_map[key]

        return None

    def generator_path(self, index: int) -> TextPath | None:
        """The generator expression paired with the `index`-th text expression, or None for the constant."""
        if not self.generator_paths:
            return None
        if len(self.generator_paths) >= len(self.text_paths):
            return self.generator_paths[index]

        return self.generator_paths[0]

    def needed_columns(self) -> list[str]:
        columns = [column for path in self.text_paths for column in path.columns]
        columns += [path.column for path in self.generator_paths if path.expression != COLUMN_NAME]
        columns += list(self.label_columns)
        columns += [column.split(".")[0] for column in (self.date_column, self.group_column) if column is not None]
        columns += list(self.row_filter)

        return list(dict.fromkeys(columns))

    @property
    def data_file(self) -> str | None:
        """The one file a `file:` config names, or None."""
        return self.config.removeprefix(FILE_CONFIG_PREFIX) if self.config and self.config.startswith(FILE_CONFIG_PREFIX) else None

    @property
    def held_out(self) -> bool:
        return bool(self.benchmark)


def latest_year(evidence: str) -> int | None:
    years = [int(year) for year in YEAR_PATTERN.findall(evidence)]

    return max(years) if years else None


def optional(value: str) -> str | None:
    stripped = value.strip()

    return stripped or None


def json_object(raw: str, name: str, column: str) -> dict[str, object]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise CatalogError(f"source {name}: {column} {text!r} is not JSON") from error
    if not isinstance(parsed, dict):
        raise CatalogError(f"source {name}: {column} {text!r} is not a JSON object")

    return parsed


def value_key(value: object) -> str:
    """A cell's value as `label_map` and `row_filter` write it: `true`, `1` (for 1.0), or the text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def row_filter_of(raw: str, name: str) -> dict[str, list[str]]:
    parsed = json_object(raw, name, "row_filter")

    return {str(column): [value_key(value) for value in (values if isinstance(values, list) else [values])] for column, values in parsed.items()}


def expressions(raw: str, name: str, column: str) -> tuple[TextPath, ...]:
    text = raw.strip()
    if not text:
        return ()
    if text.lower().startswith(COLUMN_NAME):
        return (TextPath(COLUMN_NAME, (), None, False),)
    try:
        return tuple(parse_alternatives(" | ".join(EXPRESSION_SEPARATOR.split(text))))
    except PathError as error:
        raise CatalogError(f"source {name}: {column}: {error}") from error


def family_of(raw: str, name: str) -> tuple[str, str]:
    """(family, split) of a `family_split` cell, or two empty strings for an empty cell."""
    text = raw.strip()
    if not text:
        return "", ""
    found = FAMILY_SPLIT_PATTERN.match(text)
    if found is None:
        raise CatalogError(f"source {name}: family_split {text!r} is not <family>:<train|validation|test>")

    return found.group(1), found.group(2)


def generator_name(raw: str) -> str | None:
    """The source's generator without the note a catalog writer put after it."""
    return optional(GENERATOR_NOTE.split(raw, maxsplit=1)[0])


def check_labels(spec: SourceSpec) -> None:
    """The origin, label map and text expressions agree."""

    def fail(message: str) -> CatalogError:
        return CatalogError(f"source {spec.name}: {message}")

    if spec.label_column is not None and not all(COLUMN_PATTERN.match(column) for column in spec.label_columns):
        raise fail(f"label_column {spec.label_column!r} is not a column or columns joined by ' + '")
    if spec.group_column is not None and not COLUMN_PATTERN.match(spec.group_column):
        raise fail(f"group_column {spec.group_column!r} is not a column")
    if not 0.0 < spec.group_share <= 1.0:
        raise fail(f"group_share {spec.group_share} is not in (0, 1]")
    if spec.origin == "by_column":
        if not spec.label_map:
            raise fail("origin by_column needs a label_map")
        if spec.label_column is None:
            unmapped = [path.expression for path in spec.text_paths if spec.column_origin(path) is None]
            if len(spec.text_paths) < 2 or unmapped:
                raise fail(f"by_column without a label_column needs a label_map entry for each text expression; none for {unmapped}")
        unknown = set(spec.label_map.values()) - LABEL_VALUES
        if unknown:
            raise fail(f"label_map maps to {sorted(unknown)}, not an origin or drop")
    elif spec.label_map or spec.label_column is not None:
        raise fail(f"origin {spec.origin} takes no label_column or label_map")


def parse_source(line: dict[str, str]) -> SourceSpec:
    """One `sources.csv` line as a `SourceSpec`, or `CatalogError` naming the source and the fault."""
    name = line["name"].strip()
    if not NAME_PATTERN.match(name):
        raise CatalogError(f"source name {name!r} is not a plain file name")

    def fail(message: str) -> CatalogError:
        return CatalogError(f"source {name}: {message}")

    origin = line["origin"].strip()
    if origin not in (*ORIGINS, "by_column"):
        raise fail(f"origin {origin!r} is not human, ai, mixed or by_column")
    repo = line["repo"].strip()
    if not REPO_PATTERN.match(repo):
        raise fail(f"repo {repo!r} is not a dataset on the Hub")
    revision = line["revision"].strip()
    if not revision:
        raise fail("no pinned revision")
    priority_text = line["priority"].strip()
    if not re.fullmatch(r"-?\d+", priority_text):
        raise fail(f"priority {priority_text!r} is not an integer")
    language = line["language"].strip()
    if not language:
        raise fail("no language")
    text_paths = expressions(line["text_column"], name, "text_column")
    if not text_paths or text_paths[0].expression == COLUMN_NAME:
        raise fail("no text_column")
    date_column = optional(line["date_column"])
    if date_column is not None and not DATE_COLUMN_PATTERN.match(date_column):
        raise fail(f"date_column {date_column!r} is not a column")
    date_evidence = line["date_evidence"].strip()
    year = latest_year(date_evidence)
    numbers = {key: line[key].strip().replace(",", "") for key in ("rows", "bytes")}
    notes = line["notes"]
    reread = REREAD_PATTERN.search(notes)
    share_text = line.get("group_share", "").strip()
    if share_text and not re.fullmatch(r"\d*\.?\d+", share_text):
        raise fail(f"group_share {share_text!r} is not a number")
    family, family_split = family_of(line.get("family_split", ""), name)
    spec = SourceSpec(
        name=name,
        repo=repo,
        config=optional(line["config"]),
        splits=tuple(split.strip() for split in (line["split"].strip() or "train").split(SPLIT_SEPARATOR)),
        revision=revision,
        origin=origin,
        text_paths=text_paths,
        label_column=optional(line["label_column"]),
        label_map={str(key): str(value) for key, value in json_object(line["label_map"], name, "label_map").items()},
        generator_paths=expressions(line["generator_column"], name, "generator_column"),
        generator=generator_name(line["generator"]),
        mix=line["mix"].strip(),
        date_column=date_column,
        date_evidence=date_evidence,
        domain=line["domain"].strip() or "unknown",
        language=language,
        kind=line["kind"].strip(),
        priority=int(priority_text),
        license=line["license"].strip(),
        benchmark=line["benchmark"].strip(),
        rows=int(numbers["rows"]) if numbers["rows"].isdigit() else None,
        bytes=int(numbers["bytes"]) if numbers["bytes"].isdigit() else None,
        row_filter=row_filter_of(line.get("row_filter", ""), name),
        source_date=f"{year}-12-31" if year is not None else None,
        in_first_corpus=line["in_first_corpus"].strip().lower() in ("yes", "true", "1"),
        reread=None if reread is None else reread.group(1).lower() == "yes",
        verified=line["verified"].strip(),
        group_column=optional(line.get("group_column", "")),
        group_share=float(share_text) if share_text else 1.0,
        family=family,
        family_split=family_split,
    )
    check_labels(spec)
    paired = len(spec.generator_paths)
    if 1 < paired < len(spec.text_paths):
        raise fail(f"{paired} generator expressions for {len(spec.text_paths)} text expressions")
    if "ai" in spec.origins() and not spec.generator_paths and spec.generator is None:
        raise fail("model text needs a generator or a generator_column")

    return spec


def benchmark_language(text: str) -> str:
    """The benchmark catalog's language as a corpus language: one code, `multi` for several, `code` for code."""
    stripped = text.strip()
    if stripped.lower().startswith("code"):
        return "code"
    codes = [code.strip() for code in stripped.split(",") if code.strip()]

    return codes[0] if len(codes) == 1 else MULTI_LANGUAGE


def spec_from_benchmark(line: dict[str, str]) -> SourceSpec:
    """The source line of a benchmark, from the benchmark catalog's own columns.

    The origin comes from `label_column` and `label_map`; a map of `*` alone gives every row one origin; a map
    keyed by text columns gives each column its origin; with neither, the labels are hidden and the rows are
    `unlabeled`: read, keyed, used to hold copies out of training, and not written to the corpus.
    """
    name = line["name"].strip()
    if not NAME_PATTERN.match(name):
        raise CatalogError(f"benchmark name {name!r} is not a plain file name")
    text_paths = expressions(line["text_column"], name, "text_column")
    if not text_paths:
        raise CatalogError(f"benchmark {name}: no text_column, so its rows cannot be read or held out; fix the catalog or skip it by name")
    label_column = optional(line["label_column"])
    label_map = {str(key): str(value) for key, value in json_object(line["label_map"], name, "label_map").items()}
    columns = {path.column for path in text_paths} | {path.expression for path in text_paths}
    if label_column is not None or len(text_paths) > 1 and set(label_map) <= columns and label_map:
        origin = "by_column"
    elif set(label_map) == {ANY_VALUE}:
        origin, label_map = label_map[ANY_VALUE], {}
    elif not label_map:
        origin = UNLABELED
    else:
        raise CatalogError(f"benchmark {name}: label_map {label_map} fits neither a label column nor its text columns")
    generator_paths = expressions(line["generator_column"], name, "generator_column")
    spec = SourceSpec(
        name=name,
        repo=line["repo"].strip(),
        config=optional(line["config"]),
        splits=(line["split"].strip() or "test",),
        revision=line["revision"].strip(),
        origin=origin,
        text_paths=text_paths,
        label_column=label_column,
        label_map=label_map,
        generator_paths=generator_paths,
        generator=None if generator_paths else BENCHMARK_GENERATOR,
        mix="",
        date_column=None,
        date_evidence="",
        domain="benchmark",
        language=benchmark_language(line["language"]),
        kind="benchmark",
        priority=0,
        license=line.get("license", "").strip(),
        benchmark=name,
        rows=int(rows) if (rows := line.get("rows", "").strip().replace(",", "")).isdigit() else None,
        row_filter=row_filter_of(line.get("row_filter", ""), name),
    )
    if not spec.revision:
        raise CatalogError(f"benchmark {name}: no pinned revision")
    if origin not in (*ORIGINS, "by_column", UNLABELED):
        raise CatalogError(f"benchmark {name}: origin {origin!r} is not an origin")
    if origin == "by_column":
        check_labels(spec)

    return spec


def first_corpus_spec(held_out_first: dict[str, str]) -> SourceSpec:
    """The first corpus, read whole and entered verbatim: its rows keep their text, `doc` and embeddings."""
    return SourceSpec(
        name=FIRST_CORPUS,
        repo=FIRST_CORPUS_REPO,
        config=None,
        splits=FIRST_CORPUS_SPLITS,
        revision=FIRST_CORPUS_REVISION,
        origin="by_column",
        text_paths=tuple(parse_alternatives("text")),
        label_column="label",
        label_map={"0": "human", "1": "ai"},
        generator_paths=tuple(parse_alternatives("generator")),
        generator=None,
        mix="",
        date_column=None,
        date_evidence="its human text is dated before 2022 by its build (2021)",
        domain="first_corpus",
        language="en",
        kind="first_corpus",
        priority=0,
        license="per-source-terms",
        benchmark="",
        source_date="2021-12-31",
        verbatim=True,
        held_out_first=dict(held_out_first),
    )


def read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise CatalogError(f"{path} lacks columns {sorted(missing)}")

        return [{key: value or "" for key, value in line.items()} for line in reader]


@dataclass(frozen=True)
class Catalog:
    """What a build reads, and what it leaves out with the reason."""

    specs: list[SourceSpec]
    problems: dict[str, str]  # a line the build cannot use, to why
    excluded: dict[str, str]  # a usable line the build must not read for training, to why
    held_out_first_sources: dict[str, str]  # a first-corpus `source` whose rows are a benchmark's, to that benchmark
    skipped_benchmarks: list[str]
    disagreements: list[str]  # where `sources.csv` and the benchmark catalog disagree on what is held out
    families: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # a trained family's lines, by split


def disjoint_filters(first: dict[str, list[str]], second: dict[str, list[str]]) -> bool:
    """Whether two row filters select disjoint values of a column both filter on, so no row passes both."""
    return any(not set(first[column]) & set(second[column]) for column in set(first) & set(second))


def admitted_spec(line: dict[str, str], max_priority: int, benchmark_names: set[str]) -> tuple[SourceSpec | None, str]:
    """A training line as a spec, or None and why it is left out (an exclusion, or a problem prefixed `problem: `).
    A training line named like a benchmark stops the build: both would read and write under one name."""
    name = line["name"].strip()
    if line["repo"].strip() in EXCLUDED_REPOS:
        return None, EXCLUDED_REPOS[line["repo"].strip()]
    if NO_DERIVATIVES.search(line["license"].strip()):
        return None, f"licence {line['license'].strip()} forbids derivatives"
    if name in benchmark_names:
        raise CatalogError(f"source names repeat (a benchmark's name may not name a source): {name}; rename the training line")
    try:
        spec = parse_source(line)
    except CatalogError as error:
        return None, f"problem: {error}"
    if spec.in_first_corpus and spec.reread is False:
        return None, "the first corpus's rows stand for it (raw re-read: no)"
    if spec.priority > max_priority and not spec.family:
        return None, ""

    return spec, ""


def family_test_spec(line: dict[str, str]) -> SourceSpec:
    """A family's test line as a benchmark whose rows are `unlabeled`: read whole and keyed to hold their copies out
    of training, then dropped, so a test split's labels (word boundaries, generator classes) never need a map."""
    neutral = {**line, "origin": "human", "label_column": "", "label_map": "", "generator_column": "", "generator": "", "date_column": ""}

    return replace(parse_source(neutral), origin=UNLABELED, benchmark=line["name"].strip(), priority=0, group_column=None, group_share=1.0)


def load_catalog(
    sources_path: Path, benchmarks_path: Path, max_priority: int, skip_benchmarks: tuple[str, ...] = (), first_corpus: bool = True,
    family_train: bool = False,
) -> Catalog:
    """The sources a build reads: those of priority `max_priority` or lower, every benchmark, and the first corpus.

    With `family_train`, a flagged line of a benchmark family whose test is held out is read: its `train` and
    `validation` lines for training, whatever their priority, its `test` lines as benchmarks. A benchmark that
    cannot be read stops the build unless `skip_benchmarks` names it; any other line the build cannot use is a
    problem, left out and reported.
    """
    lines = read_csv(sources_path, SOURCE_COLUMNS)
    benchmark_lines = read_csv(benchmarks_path, BENCHMARK_COLUMNS)
    benchmark_names = {line["name"].strip() for line in benchmark_lines}
    unknown_skips = sorted(set(skip_benchmarks) - benchmark_names)
    if unknown_skips:
        raise CatalogError(f"skipped benchmarks {unknown_skips} are not in {benchmarks_path}")
    by_split = {(line["repo"].strip(), optional(line["config"]), line["split"].strip()): line["name"].strip() for line in benchmark_lines}
    benchmark_filters = {line["name"].strip(): row_filter_of(line.get("row_filter", ""), line["name"].strip()) for line in benchmark_lines}
    problems: dict[str, str] = {}
    excluded: dict[str, str] = {}
    held_out_first: dict[str, str] = {}
    flagged_keys: set[tuple[str, str | None, str]] = set()
    disagreements: list[str] = []
    training: list[SourceSpec] = []
    family_lines: list[tuple[str, dict[str, str]]] = []  # flagged train and validation lines, with their family
    family_tests: list[SourceSpec] = []
    tested = {name for name in benchmark_names if name not in skip_benchmarks}  # families whose test is held out
    for line in lines:
        name = line["name"].strip()
        key = (line["repo"].strip(), optional(line["config"]), line["split"].strip())
        matched = by_split.get(key)
        flag = line["benchmark"].strip()
        try:
            family, family_split = family_of(line.get("family_split", ""), name)
        except CatalogError as error:
            problems[name] = str(error)
            continue
        if family_train and family_split in TRAINED_FAMILY_SPLITS and (
            not matched or disjoint_filters(row_filter_of(line.get("row_filter", ""), name), benchmark_filters[matched])
        ):
            family_lines.append((family, line))
            flagged_keys.add(key)
            continue
        if family_train and family_split == TEST_FAMILY_SPLIT and not matched:
            try:
                family_tests.append(family_test_spec(line))
                tested.add(family)
            except CatalogError as error:
                problems[name] = str(error)
        if flag or matched:
            benchmark = matched or name
            excluded[name] = f"held out: benchmark {benchmark}"
            flagged_keys.add(key)
            if line["in_first_corpus"].strip().lower() in ("yes", "true", "1"):
                held_out_first[name] = benchmark
            if flag and not matched:
                disagreements.append(f"{name}: flagged ({flag}) in sources.csv, not a split of the benchmark catalog")
                if name in benchmark_names and key not in by_split:
                    disagreements.append(f"{name}: held out under the name of benchmark {name}, though it is not that benchmark's split")
            continue
        spec, reason = admitted_spec(line, max_priority, benchmark_names)
        if spec is not None:
            training.append(spec)
        elif reason.startswith("problem: "):
            problems[name] = reason.removeprefix("problem: ")
        elif reason:
            excluded[name] = reason
    families: dict[str, dict[str, list[str]]] = {}
    for family, line in family_lines:
        name = line["name"].strip()
        if family not in tested:
            excluded[name] = f"held out: family {family} has no test this build holds out"
            continue
        spec, reason = admitted_spec(line, max_priority, benchmark_names)
        if spec is None:
            if reason.startswith("problem: "):
                problems[name] = reason.removeprefix("problem: ")
            else:
                excluded[name] = reason or f"family {family}: left out"
            continue
        training.append(replace(spec, benchmark=""))  # the flag named the family; its rows are training rows
        families.setdefault(family, {}).setdefault(spec.family_split, []).append(name)
    for spec in family_tests:
        families.setdefault(spec.family, {}).setdefault(TEST_FAMILY_SPLIT, []).append(spec.name)
    for key, benchmark in sorted(by_split.items(), key=lambda item: item[1]):
        if key not in flagged_keys:
            disagreements.append(f"{benchmark}: a split of the benchmark catalog with no flagged line in sources.csv")
    benchmarks = [spec_from_benchmark(line) for line in benchmark_lines if line["name"].strip() not in skip_benchmarks]
    specs = training + benchmarks + family_tests + ([first_corpus_spec(held_out_first)] if first_corpus else [])
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CatalogError(f"source names repeat (a benchmark's name may not name a source): {duplicates}")

    return Catalog(specs, problems, excluded, held_out_first, sorted(skip_benchmarks), disagreements, families)
