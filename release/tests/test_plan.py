from __future__ import annotations

import datetime
import pytest

from pathlib import Path
from litmus_release import plan as planning


MODEL = "owner/name@" + "b" * 40


def create(work: Path, **changes: str) -> planning.Plan:
    arguments = {"model": MODEL, "version": "1.2.3", "release_path": "release", "record_path": "record", **changes}
    return planning.create(work, today=datetime.date(2026, 11, 18), **arguments)


def test_a_plan_is_written_once_and_read_back(tmp_path: Path) -> None:
    made = create(tmp_path / "w")
    assert (made.version, made.tag, made.repo, made.revision, made.date) == ("1.2.3", "v1.2.3", "owner/name", "b" * 40, "2026-11-18")
    assert planning.load(tmp_path / "w") == made
    assert create(tmp_path / "w") == made


@pytest.mark.parametrize("model", ["owner/name", "owner/name@main", "owner/name@" + "b" * 12, "name@" + "b" * 40])
def test_a_model_is_pinned_to_a_full_commit(tmp_path: Path, model: str) -> None:
    with pytest.raises(planning.PlanError):
        create(tmp_path / "w", model=model)


@pytest.mark.parametrize("version", ["1", "1.0", "v1.0.0", "1.0.0-rc1"])
def test_a_version_is_three_numbers(tmp_path: Path, version: str) -> None:
    with pytest.raises(planning.PlanError, match="MAJOR.MINOR.PATCH"):
        create(tmp_path / "w", version=version)


def test_a_work_folder_refuses_another_model_or_version(tmp_path: Path) -> None:
    create(tmp_path / "w")
    with pytest.raises(planning.PlanError, match="already holds"):
        create(tmp_path / "w", version="2.0.0")
    with pytest.raises(planning.PlanError, match="already holds"):
        create(tmp_path / "w", model="owner/name@" + "c" * 40)


def test_a_folder_stands_in_for_a_model_in_a_trial_and_says_so(tmp_path: Path) -> None:
    (tmp_path / "files").mkdir()
    made = create(tmp_path / "w", model=f"local:{tmp_path / 'files'}")
    assert made.local and made.model.startswith("local:")


def test_a_work_folder_without_a_plan_says_how_to_start(tmp_path: Path) -> None:
    with pytest.raises(planning.PlanError, match="cut.py run --model"):
        planning.load(tmp_path)
