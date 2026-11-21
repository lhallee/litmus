from __future__ import annotations

import pytest

from litmus_release.datapath import MissingValue, parse_path, resolve, spell, split_key


DOCUMENTS = {
    "config": {"bank": {"counts": [4096, 4096, 4093]}, "val/pr_auc": 0.5},
    "eval": {"rows": [{"metric": "pr_auc", "level": "", "value": 0.9}, {"metric": "coverage", "level": 0.1, "value": 0.88}, {"metric": "coverage", "level": 0.05, "value": 0.95},
                      {"metric": "boundary", "stratum": "all|all", "value": 3}]},
}


def test_path_steps_are_names_indexes_and_selectors() -> None:
    assert parse_path("config.bank.counts[1]") == ["config", "bank", "counts", 1]
    assert parse_path("eval.rows[metric=coverage,level=0.1].value") == ["eval", "rows", {"metric": "coverage", "level": "0.1"}, "value"]
    assert parse_path("eval.rows[metric=pr_auc,level=].value")[2] == {"metric": "pr_auc", "level": ""}


def test_resolve_follows_names_indexes_and_selectors() -> None:
    assert resolve(DOCUMENTS, "config.bank.counts[2]") == 4093
    assert resolve(DOCUMENTS, "config.val/pr_auc") == 0.5
    assert resolve(DOCUMENTS, "eval.rows[metric=coverage,level=0.05].value") == 0.95
    assert resolve(DOCUMENTS, "eval.rows[metric=pr_auc,level=].value") == 0.9
    assert resolve(DOCUMENTS, "eval.rows[metric=boundary,stratum=all|all].value") == 3


@pytest.mark.parametrize("path", ["nothing.x", "config.bank.missing", "config.bank.counts[9]", "eval.rows[metric=coverage].value", "eval.rows[metric=absent].value"])
def test_resolve_names_what_is_missing_and_never_guesses(path: str) -> None:
    with pytest.raises(MissingValue):
        resolve(DOCUMENTS, path)


def test_a_format_follows_the_last_as_and_never_a_selector_bar() -> None:
    assert split_key("config.bank.counts[0] as int") == ("config.bank.counts[0]", "int")
    assert split_key("eval.rows[stratum=all|all,metric=b].value as f3") == ("eval.rows[stratum=all|all,metric=b].value", "f3")
    assert split_key("eval.rows[stratum=all|all,metric=b].value") == ("eval.rows[stratum=all|all,metric=b].value", "")


def test_spell_formats() -> None:
    assert spell(4096, "int") == "4,096"
    assert spell(28_267, "sig2") == "28,000" and spell(5_540, "sig2") == "5,500" and spell(28_920.4, "sig2") == "29,000"
    assert spell(4096) == "4,096"
    assert spell(0.93456, "f3") == "0.935"
    assert spell(0.0512, "pct1") == "5.1%"
    assert spell(11_288_265, "mb1") == "11.3 MB"
    assert spell(1_790_000, "m2") == "1.79M"
    assert spell(0.1321, "ms0") == "132"
    assert spell(1.431902, "odds1") == "4.2" and spell(0.0, "odds1") == "1.0"
    assert spell(30.0, "f0") == "30"
    assert spell(["a", "b", "c"]) == "a, b, and c"
    assert spell("text") == "text"
    assert spell({"columns": ["a", "b"], "rows": [[1, 2]]}) == "| a | b |\n| --- | --- |\n| 1 | 2 |"


@pytest.mark.parametrize(("value", "spec"), [(0.5, ""), (None, "f3"), (float("nan"), "f3"), (True, ""), (0.5, "bogus")])
def test_spell_refuses_what_it_cannot_write_exactly(value: object, spec: str) -> None:
    with pytest.raises(ValueError):
        spell(value, spec)
