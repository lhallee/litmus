from __future__ import annotations

import pytest

from pathlib import Path
from litmus_release.data import documents
from litmus_release.datapath import MissingValue
from litmus_release.values import Values


def values(context, **extra) -> Values:
    available = documents(context, context.repository / "release" / "targets.toml")
    merged = {name: available[name] for name in available}
    merged.update(extra)
    return Values.with_templates(merged, context.repository / "release" / "templates.toml")


def test_a_value_and_a_template_render_from_the_data(context) -> None:
    renderer = values(context)
    assert renderer.render("model.bank.human as int", False) == "4,096"
    assert renderer.render("t:size", False) == "`s192_l4` (1.79M ternary weights)"
    assert renderer.render("release.version", False) == "1.0.0"


def test_ternary_weights_count_every_matrix_and_the_projection(context) -> None:
    # docs/training/model.md: 1.79M for s192_l4, the embedding's projection included.
    assert documents(context, context.repository / "release" / "targets.toml")["model"]["ternary_weights"] == 4 * (4 * 192 * 192 + 2 * 192 * 768) + 128 * 192


def test_a_line_that_depends_on_a_document_drops_when_the_document_is_absent(context) -> None:
    without = values(context).render("t:table", True)
    assert "Teacher" not in without and "0.935" in without
    with_teacher = values(context, teacher={"rows": [{"model": "head", "pr_auc": 0.912}]}).render("t:table", True)
    assert "| Teacher | 0.912 |" in with_teacher


def test_a_template_picks_its_text_by_whether_a_path_holds_a_value(context) -> None:
    assert values(context).render("t:contribute", False) == "The contribute site, once it is live,"
    context.repository.joinpath("release", "targets.toml").write_text(
        context.repository.joinpath("release", "targets.toml").read_text(encoding="utf-8").replace('site_url = ""', 'site_url = "https://example.org/c"'), encoding="utf-8")
    assert values(context).render("t:contribute", False) == "https://example.org/c"


def test_the_benchmark_matrix_has_every_detector_and_marks_training_on_a_family(context) -> None:
    table = values(context).render("table.benchmark_matrix as roc_auc", True)
    lines = table.splitlines()
    assert lines[0] == "| Benchmark | Litmus | Other |"
    assert lines[2] == "| MAGE | 0.812 † | 0.700 |"
    assert "Test" not in table


@pytest.mark.parametrize("key", ["nothing.here", "t:absent", "table.absent", "results.rows[benchmark=nope,detector=Litmus,task=ai_vs_human].pr_auc as f3", "model.bank.human as bogus"])
def test_a_key_the_data_cannot_fill_fails(context, key: str) -> None:
    with pytest.raises((MissingValue, ValueError)):
        values(context).render(key, False)


def test_an_empty_value_is_an_error_not_an_empty_slot(context) -> None:
    with pytest.raises(MissingValue, match="empty"):
        values(context, record={"blank": ""}).render("record.blank", False)


def test_a_stale_results_file_is_refused(context) -> None:
    path = context.repository / "site" / "assets" / "results.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"version": "1.0.0"', '"version": "0.9.0"'), encoding="utf-8")
    with pytest.raises(SystemExit, match="0.9.0"):
        documents(context, Path(context.repository / "release" / "targets.toml"))["results"]
