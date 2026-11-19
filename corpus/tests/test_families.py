"""Benchmark families trained on, RAID-shaped labels and groups, and an earlier corpus extended with new rows."""

from __future__ import annotations

import csv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pathlib import Path
from litmus_corpus.adapters import generic_adapter, group_hash
from litmus_corpus.catalog import SOURCE_COLUMNS, CatalogError, load_catalog
from litmus_corpus.checks import hash64, opening
from litmus_corpus.config import UNLABELED, BuildConfig
from litmus_corpus.extend import UNMARK_REASONS, Keys, adopted_docs, copy_codes, extend, unmark_codes
from litmus_corpus.read import read_limits
from litmus_corpus.records import CORPUS_SCHEMA, KEYED_SCHEMA, conform
from litmus_corpus.storage import Report

from conftest import catalog_line, spec_of, write_benchmarks


FAMILY_COLUMNS = ["row_filter", "family_split", "in_distribution", "group_column", "group_share"]
RAID_LABELS = '{"human|paraphrase": "mixed", "human|synonym": "drop", "human|*": "human", "*": "ai"}'


def write_family_catalog(path: Path, lines: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*SOURCE_COLUMNS, *FAMILY_COLUMNS])
        writer.writeheader()
        writer.writerows([{column: "" for column in FAMILY_COLUMNS} | line for line in lines])

    return path


def raid_line(**values: str) -> dict[str, str]:
    return catalog_line(name="raid_train", repo="liamdugan/raid", config="raid", split="train", origin="by_column", text_column="generation",
                        label_column="model + attack", label_map=RAID_LABELS, generator_column="model", date_evidence="2021",
                        benchmark="RAID train", family_split="raid:train", group_column="source_id") | values


def raid_rows(sources: list[str]) -> pa.Table:
    """Each source document as RAID holds it: its human text and one generation, each unattacked and under three
    attacks."""
    rows: dict[str, list[str]] = {"generation": [], "model": [], "attack": [], "source_id": []}
    for source in sources:
        for model in ("human", "gpt4"):
            for attack in ("none", "homoglyph", "paraphrase", "synonym"):
                rows["generation"].append(f"{model} text of {source} under {attack}, long enough to read.")
                rows["model"].append(model)
                rows["attack"].append(attack)
                rows["source_id"].append(source)

    return pa.table(rows)


def test_raid_labels_follow_model_and_attack() -> None:
    spec = spec_of(**raid_line())
    report = Report()

    records = generic_adapter(raid_rows(["doc-a"]), spec, report)

    labels = dict(zip(records.column("text").to_pylist(), records.column("label").to_pylist(), strict=True))
    assert labels["human text of doc-a under none, long enough to read."] == 0
    assert labels["human text of doc-a under homoglyph, long enough to read."] == 0
    assert labels["human text of doc-a under paraphrase, long enough to read."] == 2
    assert "human text of doc-a under synonym, long enough to read." not in labels
    assert all(labels[f"gpt4 text of doc-a under {attack}, long enough to read."] == 1 for attack in ("none", "homoglyph", "paraphrase", "synonym"))
    assert report.dropped["label_drop"]["drop"] == 1
    mixed = records.filter(pa.compute.equal(records.column("label"), 2))
    assert mixed.column("mix").to_pylist() == [spec.mix]


def test_a_group_shares_one_doc_and_is_kept_or_left_whole() -> None:
    sources = [f"doc-{index}" for index in range(200)]
    spec = spec_of(**raid_line(group_share="0.3"))
    report = Report()

    records = generic_adapter(raid_rows(sources), spec, report)

    kept = {source for source in sources if any(f"of {source} under" in text for text in records.column("text").to_pylist())}
    for source in kept:
        rows = records.filter(pa.compute.match_substring(records.column("text"), f"of {source} under"))
        assert rows.num_rows == 7  # all eight rows of the group but its synonym-substituted human text
        assert set(rows.column("first_doc").to_pylist()) == {group_hash("raid_train", source)}
    assert 0.2 < len(kept) / len(sources) < 0.4
    assert report.dropped["group_sample"]["any"] == 8 * (len(sources) - len(kept))
    # Read whole: a row limit would keep only the stream's first rows, and RAID's files run attack by attack.
    plain = spec_of(name="plain", origin="ai", text_column="text", generator="gpt-4o", rows="900000")
    limits = read_limits([spec, plain], BuildConfig(new_token_budget=1e5), None)
    assert limits["raid_train"] is None and limits["plain"] is not None and limits["plain"] < 900_000
    assert spec_of(**raid_line(group_share="0.3")).group_share == 0.3
    with pytest.raises(CatalogError, match="group_share"):
        spec_of(**raid_line(group_share="1.5"))


def family_catalog(tmp_path: Path) -> tuple[Path, Path]:
    lines = [
        catalog_line(name="plain", origin="ai", text_column="text", generator="gpt-4o"),
        raid_line(in_first_corpus="yes", notes="raw re-read: yes"),
        raid_line(name="raid_extra_code", split="extra", priority="2", family_split="raid:train", row_filter='{"domain": "code"}'),
        raid_line(name="raid_extra_news", split="extra", family_split="raid:train", row_filter='{"domain": ["czech", "code"]}'),
        catalog_line(name="raid_test", repo="liamdugan/raid", config="raid_test", split="test", origin="ai", text_column="generation",
                     generator="unknown", benchmark="RAID test", family_split="raid:test"),
        catalog_line(name="mage_train", repo="x/mage", origin="by_column", text_column="text", label_column="label", label_map='{"0": "human", "*": "ai"}',
                     generator_column="model", date_evidence="2020", benchmark="MAGE train", family_split="mage:train", priority="2"),
        catalog_line(name="mage_test", repo="x/mage", split="test", origin="by_column", text_column="text", label_column="label",
                     generator_column="model", benchmark="MAGE test", family_split="mage:test", in_first_corpus="yes"),
        catalog_line(name="lone_train", repo="x/lone", origin="ai", text_column="text", generator="gpt-4o", benchmark="Lone train",
                     family_split="lone:train"),
    ]
    benchmarks = [
        {"name": "raid", "repo": "liamdugan/raid", "config": "raid_test", "split": "test", "text_column": "generation"},
        {"name": "raid_extra", "repo": "liamdugan/raid", "config": "raid", "split": "extra", "text_column": "generation",
         "row_filter": '{"domain": ["czech", "german"]}'},
    ]

    return write_family_catalog(tmp_path / "sources.csv", lines), write_benchmarks(tmp_path / "benchmarks.csv", benchmarks)


def test_without_family_training_every_flagged_line_stays_held_out(tmp_path: Path) -> None:
    sources, benchmarks = family_catalog(tmp_path)

    catalog = load_catalog(sources, benchmarks, max_priority=3, first_corpus=False)

    assert [spec.name for spec in catalog.specs] == ["plain", "raid", "raid_extra"]
    assert catalog.excluded["raid_train"] == "held out: benchmark raid_train"
    assert catalog.excluded["raid_extra_code"] == catalog.excluded["raid_extra_news"] == "held out: benchmark raid_extra"
    assert catalog.excluded["mage_train"] == "held out: benchmark mage_train" and catalog.families == {}
    assert catalog.held_out_first_sources == {"raid_train": "raid_train", "mage_test": "mage_test"}


def test_family_training_reads_train_splits_whose_test_is_held_out(tmp_path: Path) -> None:
    sources, benchmarks = family_catalog(tmp_path)

    catalog = load_catalog(sources, benchmarks, max_priority=1, first_corpus=False, family_train=True)
    specs = {spec.name: spec for spec in catalog.specs}

    # raid_train: the family's test is the benchmark raid. raid_extra_code is a split of the benchmark raid_extra, admitted
    # because their row filters select disjoint domains; raid_extra_news shares a domain with it, so it stays held out.
    assert "raid_train" in specs and "raid_extra_code" in specs and "raid_extra_news" not in specs
    # The flag named the family: a trained line's rows are training rows, read under the budget, never benchmark rows.
    assert not any(specs[name].held_out for name in ("raid_train", "raid_extra_code", "mage_train"))
    assert catalog.excluded["raid_extra_news"] == "held out: benchmark raid_extra"
    # A family read at a priority the build does not read otherwise, its test read whole as hidden-label benchmark rows.
    assert "mage_train" in specs and specs["mage_test"].origin == UNLABELED and specs["mage_test"].benchmark == "mage_test"
    assert catalog.held_out_first_sources == {"mage_test": "mage_test"}
    assert catalog.excluded["lone_train"] == "held out: family lone has no test this build holds out"
    assert catalog.families == {"raid": {"train": ["raid_train", "raid_extra_code"]}, "mage": {"train": ["mage_train"], "test": ["mage_test"]}}


def test_a_training_source_may_not_take_a_benchmarks_name(tmp_path: Path) -> None:
    sources = write_family_catalog(tmp_path / "sources.csv", [raid_line(name="raid")])
    benchmarks = write_benchmarks(tmp_path / "benchmarks.csv", [{"name": "raid", "repo": "liamdugan/raid", "config": "raid_test", "split": "test"}])

    with pytest.raises(CatalogError, match="raid"):
        load_catalog(sources, benchmarks, max_priority=1, family_train=True)
    # Held out, as without the switch, the line is only noted: it does not stand for the benchmark it is named after.
    held = load_catalog(sources, benchmarks, max_priority=1, first_corpus=False)
    assert any("not that benchmark's split" in note for note in held.disagreements)


def test_the_real_catalog_trains_raid_and_keeps_every_other_line_without_the_switch() -> None:
    folder = Path(__file__).resolve().parents[1]
    skips = ("autextification_en", "autextification_es", "editlens", "hc3")
    before = load_catalog(folder / "sources.csv", folder.parent / "benchmarks" / "catalog.csv", 1, skips, first_corpus=False)
    after = load_catalog(folder / "sources.csv", folder.parent / "benchmarks" / "catalog.csv", 1, skips, first_corpus=False, family_train=True)
    names = {spec.name for spec in after.specs} - {spec.name for spec in before.specs}

    assert "raid_train" not in {spec.name for spec in before.specs} and before.excluded["raid_train"].startswith("held out")
    assert {"raid_train", "raid_extra_code"} <= names
    assert {spec.family for spec in after.specs if spec.name in names} <= set(after.families)
    assert all(spec.family or spec.benchmark for spec in after.specs if spec.name in names)


def corpus_table(texts: list[str], first_id: int, **columns: list[object]) -> pa.Table:
    count = len(texts)
    values: dict[str, list[object]] = {
        "id": list(range(first_id, first_id + count)), "text": texts, "label": [1] * count, "source": ["s"] * count,
        "generator": ["gpt-4o"] * count, "provider": ["openai"] * count, "era": ["recent"] * count, "domain": ["web"] * count,
        "format": ["plain"] * count, "language": ["en"] * count, "doc": [hash64(text) for text in texts], "tokens": [10] * count,
        "whole": [True] * count, "long_eval": [False] * count, "mix": [""] * count, "benchmark": [""] * count, "first_corpus_id": [None] * count,
        "text_unchanged": [False] * count,
    } | columns

    return conform(pa.table({name: pa.array(values[name], type=CORPUS_SCHEMA.field(name).type) for name in CORPUS_SCHEMA.names}), CORPUS_SCHEMA)


def test_extend_keeps_earlier_ids_marks_anew_and_drops_new_copies(tmp_path: Path) -> None:
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    texts = ["an earlier model text about rivers and the sea", "a raid row the first corpus held out by its source",
             "a text a new benchmark now holds out entirely", "a human text that stays in training as it was"]
    pq.write_table(corpus_table(texts, 0, label=[1, 1, 0, 0], source=["s", "raid", "h", "h"], benchmark=["", "raid", "", ""],
                                first_corpus_id=[None, 7, None, None]), earlier / "shard-00000.parquet")
    build = tmp_path / "build"
    (build / "corpus" / "data").mkdir(parents=True)
    new_texts = ["an earlier model text about rivers and the sea", "an entirely new model text about mountains", "the benchmark's text itself"]
    pq.write_table(corpus_table(new_texts, 0, source=["n", "n", "hc3"], benchmark=["", "", "hc3"]), build / "corpus" / "data" / "shard-00000.parquet")
    checks = build / "checks" / "hc3"
    checks.mkdir(parents=True)
    benchmark_texts = [texts[2], "the benchmark's text itself"]
    keyed = {name: [None] * 2 for name in KEYED_SCHEMA.names} | {
        "text": benchmark_texts, "label": [0, 1], "source": ["hc3"] * 2, "generator": ["human", "unknown"], "provider": ["human", "unknown"],
        "domain": ["benchmark"] * 2, "language": ["en"] * 2, "mix": [""] * 2, "benchmark": ["hc3"] * 2, "text_unchanged": [False] * 2,
        "format": ["plain"] * 2, "doc": [1, 2], "tokens": [10] * 2, "whole": [True] * 2, "long_eval": [False] * 2,
        "text_key": [hash64(text) for text in benchmark_texts], "opening_key": [hash64(opening(text)) for text in benchmark_texts],
    }
    pq.write_table(conform(pa.table({name: pa.array(keyed[name], type=KEYED_SCHEMA.field(name).type) for name in KEYED_SCHEMA.names}), KEYED_SCHEMA),
                   checks / "s0-p000.parquet")

    report = extend(earlier, build, tmp_path / "out", workers=1, trained_first_sources=("raid",))

    out = pa.concat_tables([pq.read_table(path) for path in sorted((tmp_path / "out" / "data").glob("*.parquet"))])
    assert out.column("id").to_pylist() == [0, 1, 2, 3, 4, 5]
    assert out.column("text").to_pylist()[:4] == texts
    assert out.column("benchmark").to_pylist() == ["", "", "hc3", "", "", "hc3"]
    assert out.column("text").to_pylist()[4:] == ["an entirely new model text about mountains", "the benchmark's text itself"]
    assert report["new"]["dropped"] == {"earlier_text": 1, "earlier_opening": 0}
    assert report["earlier"]["unmarked_trained_first_sources"] == {"raid": 1} and report["earlier"]["marked_by_benchmark"] == {"hc3": 1}
    with pytest.raises(AssertionError, match="mage_train"):
        extend(earlier, build, tmp_path / "again", workers=1, trained_first_sources=("mage_train",))
    assert np.array_equal(pq.read_table(tmp_path / "out" / "data" / "shard-00000.parquet").column("doc").to_numpy(),
                          pq.read_table(earlier / "shard-00000.parquet").column("doc").to_numpy())


def test_a_held_out_row_is_unmarked_only_if_it_passes_the_checks_it_skipped() -> None:
    # rows: 0 training; 1 copies its text; 2 new; 3 shares row 2's opening; 4 degenerate; 5 human training;
    # 6 holds row 5's text as ai; 7 another benchmark's row, never a candidate
    keys = Keys(
        ids=np.arange(8), label=np.array([1, 1, 1, 1, 1, 0, 1, 1]),
        source=np.array(["s", "raid", "raid", "raid", "raid", "h", "raid", "x"], dtype=object),
        benchmark=np.array(["", "raid", "raid", "raid", "raid", "", "raid", "hc3"], dtype=object),
        doc=np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.uint64), first=np.array([False, True, True, True, True, False, True, True]),
        text_key=np.array([100, 100, 200, 201, 300, 400, 400, 500], dtype=np.uint64),
        opening_key=np.array([1000, 1000, 2000, 2000, 3000, 4000, 4000, 5000], dtype=np.uint64),
        degenerate=np.array([False, False, False, False, True, False, False, False]),
    )
    candidates = keys.first & (keys.benchmark == keys.source)

    codes = unmark_codes(keys, candidates)

    reasons = [UNMARK_REASONS[code] for code in codes]
    assert reasons[1:5] == ["copy_text", "unmarked", "copy_opening", "degenerate"]
    assert reasons[6] == "text_under_two_origins" and codes[[0, 5, 7]].tolist() == [0, 0, 0]


def keys_of(source: list[str], doc: list[int], text_key: list[int], opening_key: list[int], benchmark: list[str] | None = None) -> Keys:
    count = len(source)
    return Keys(ids=np.arange(count), label=np.ones(count, dtype=np.int64), source=np.array(source, dtype=object),
                benchmark=np.array(benchmark or [""] * count, dtype=object), doc=np.array(doc, dtype=np.uint64),
                first=np.ones(count, dtype=bool), text_key=np.array(text_key, dtype=np.uint64),
                opening_key=np.array(opening_key, dtype=np.uint64), degenerate=np.zeros(count, dtype=bool))


def test_an_earlier_row_standing_for_a_group_row_joins_its_group() -> None:
    # earlier: 0 copies a RAID group row's text; 1 its opening; 2 copies a row of an ungrouped source; 3 held out
    earlier = keys_of(["raid", "raid", "s", "raid"], [1, 2, 3, 4], [10, 11, 12, 13], [20, 21, 22, 23], ["", "", "", "raid"])
    new = keys_of(["raid_train", "raid_train", "n", "raid_train", "raid_train"], [900, 900, 901, 902, 903],
                  [10, 30, 12, 13, 31], [20, 21, 22, 23, 32])
    codes = copy_codes(new, earlier)

    docs, adopted = adopted_docs(earlier, new, codes, ("raid_train",), earlier.benchmark == "")

    assert codes.tolist() == [1, 2, 1, 1, 0]
    assert docs.tolist() == [900, 900, 3, 4] and adopted.tolist() == [True, True, False, False]
