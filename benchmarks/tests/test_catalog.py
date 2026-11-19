from __future__ import annotations

import json
import pandas as pd
import pytest

from pathlib import Path
from litmus_benchmarks.catalog import (FILE_PREFIX, ORIGINS, CatalogEntry, origin_of, read_catalog, records_to_texts,
                                       sample_texts)


BENCHMARKS = Path(__file__).resolve().parents[1]


def entry(**fields: object) -> CatalogEntry:
    defaults = dict(name="toy", repo="org/toy", config="default", split="test", revision="0" * 40, text_column="text",
                    label_column="label", label_map={"0": "human", "1": "ai"}, generator_column="", domain_column="",
                    attack_column="", language="en", metric="auroc", labels_public="yes", row_filter={},
                    in_distribution=False)
    return CatalogEntry(**{**defaults, **fields})


def test_the_catalog_parses_with_exact_keys() -> None:
    entries = read_catalog(BENCHMARKS / "catalog.csv")
    assert len(entries) >= 15
    for item in entries:
        assert len(item.revision) == 40, f"{item.name} pins a full commit"
        assert set(item.label_map.values()) <= set(ORIGINS)
        if item.config.startswith(FILE_PREFIX):
            assert item.split == "train", f"{item.name}: a file config's split is train"
        if item.labels_public.startswith("yes") and item.text_column:
            assert item.label_column or item.is_melted or "*" in item.label_map, f"{item.name} has no way to an origin"
    seen = {item.name for item in entries if item.in_distribution}
    assert {"raid", "mage", "droid"} <= seen and not seen & {"beemo", "mixset", "mage_ood", "raid_extra"}


def test_label_map_falls_back_to_star_and_refuses_unknown_labels() -> None:
    assert origin_of("human", {"human": "human", "*": "ai"}) == "human"
    assert origin_of("gpt4", {"human": "human", "*": "ai"}) == "ai"
    with pytest.raises(KeyError):
        origin_of("2", {"0": "human", "1": "ai"})


def test_melted_columns_explode_lists_of_dicts() -> None:
    table = pd.DataFrame({
        "human_output": ["a person wrote this"],
        "model_output": ["a model wrote this"],
        "human_edits": ["an expert edited this"],
        "gpt-4o_edits": [str([{"P1": "edit one"}, {"P2": "edit two"}, {"P3": "edit three"}])],
        "model": ["zephyr"],
    })
    beemo = entry(text_column="human_output,model_output,human_edits,gpt-4o_edits", label_column="",
                  label_map={"human_output": "human", "model_output": "ai", "human_edits": "mixed", "gpt-4o_edits": "ai"},
                  generator_column="model")
    texts = records_to_texts(beemo, table)
    assert texts["origin"].tolist() == ["human", "ai", "mixed", "ai", "ai", "ai"]
    assert texts["generator"].tolist() == ["human"] + ["zephyr"] * 5
    assert texts["attack"].tolist()[-3:] == ["gpt-4o_edits"] * 3


def test_row_filter_and_constant_origin() -> None:
    table = pd.DataFrame({"code": ["x = 1", "y = 2", "z = 3"], "split": ["train", "test", "test"]})
    texts = records_to_texts(entry(text_column="code", label_column="", label_map={"*": "mixed"}, row_filter={"split": "test"}), table)
    assert texts["text"].tolist() == ["y = 2", "z = 3"]
    assert (texts["origin"] == "mixed").all()
    assert texts["row"].tolist() == [0, 1]


def test_row_filter_keeps_any_of_a_list_and_fails_when_it_keeps_nothing() -> None:
    table = pd.DataFrame({"code": ["a = 1", "b = 2", "c = 3", "d = 4"], "domain": ["code", "czech", "german", "czech"]})
    kept = entry(text_column="code", label_column="", label_map={"*": "ai"}, row_filter={"domain": ["czech", "german"]})
    assert records_to_texts(kept, table)["text"].tolist() == ["b = 2", "c = 3", "d = 4"]
    with pytest.raises(ValueError, match="keeps no row"):
        records_to_texts(entry(text_column="code", label_column="", label_map={"*": "ai"}, row_filter={"domain": ["polish"]}), table)


def test_code_that_is_a_python_literal_stays_one_text() -> None:
    code = ["[...]", "{1: ..., 2: None}", "[1, 2, 3]", "['a', 'b']", "x = [1]"]
    texts = records_to_texts(entry(text_column="code", label_column="", label_map={"*": "ai"}), pd.DataFrame({"code": code}))
    # Only a list or dict holding texts is a stored list of texts; the fourth is one, the others are source code.
    assert texts["text"].tolist() == ["[...]", "{1: ..., 2: None}", "[1, 2, 3]", "a", "b", "x = [1]"]


def test_a_row_without_text_column_is_refused() -> None:
    with pytest.raises(ValueError, match="no text column"):
        records_to_texts(entry(text_column=""), pd.DataFrame({"text": ["a"]}))


def test_sampling_is_seeded_and_without_replacement() -> None:
    texts = pd.DataFrame({"text": [str(index) for index in range(100)]})
    first, second = sample_texts(texts, 10, seed=4), sample_texts(texts, 10, seed=4)
    assert first.equals(second)
    assert first["text"].is_unique and len(first) == 10
    assert len(sample_texts(texts, None, seed=0)) == 100


def test_label_maps_are_json_in_the_csv() -> None:
    lines = pd.read_csv(BENCHMARKS / "catalog.csv", dtype=str, keep_default_na=False)
    for value in lines["label_map"]:
        json.loads(value)
