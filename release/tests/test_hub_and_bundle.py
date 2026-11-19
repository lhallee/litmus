from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
import pytest

from pathlib import Path
from litmus_release import bundle, hub, plan as planning, receipts
from litmus_release.commands import StageError
from litmus_release.stages import Context, Options
from tests.conftest import CONFIG, WEIGHTS


def release_folder(folder: Path) -> Path:
    """A release folder whose config states the tokenizer's and the weights' hashes, as an export writes it."""
    folder.mkdir(parents=True)
    tokenizer = b'{"model": {}}'
    (folder / "model.safetensors").write_bytes(WEIGHTS)
    (folder / "tokenizer.json").write_bytes(tokenizer)
    config = {**CONFIG, "tokenizer": {"sha256": hashlib.sha256(tokenizer).hexdigest()}, "weights": {"sha256": hashlib.sha256(WEIGHTS).hexdigest()}}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def record_folder(folder: Path, *, weights: str | None = None) -> Path:
    folder.mkdir(parents=True)
    sha = weights or hashlib.sha256(WEIGHTS).hexdigest()
    (folder / "record.toml").write_text(f'[identity]\nweights_sha256 = "{sha}"\n[benchmarks]\nlitmus_trained_on = []\n', encoding="utf-8")
    (folder / "evaluation_test.json").write_text('{"split": "test", "passages": []}', encoding="utf-8")
    return folder


def baselines_file(path: Path, *, revision: str = "d" * 40) -> Path:
    path.write_text(f'[scores]\nrepo = "owner/baselines"\nrepo_type = "dataset"\nrevision = "{revision}"\npath = "scores"\nmax_rows = 10\nseed = 0\nbenchmarks = ["mage"]\n'
                    '[costs]\nfiles = ["costs/costs_small.csv"]\n', encoding="utf-8")
    return path


def fetch_context(tmp_path: Path, repository: Path, *, baselines: Path) -> Context:
    trial = tmp_path / "trial"
    release_folder(trial / "release")
    record_folder(trial / "record")
    (repository / "tools").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parents[2] / "tools" / "verify_model.py", repository / "tools" / "verify_model.py")
    plan = planning.create(tmp_path / "work", model=f"local:{trial}", version="1.0.0", release_path="release", record_path="record")
    return Context(plan=plan, work=tmp_path / "work", repository=repository, options=Options(baselines=baselines), system="linux")


def fake_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[tuple]) -> None:
    def download(repo: str, repo_type: str, revision: str, patterns: list[str], destination: Path) -> Path:
        calls.append((repo, repo_type, revision, patterns))
        (destination / "scores").mkdir(parents=True)
        (destination / "costs").mkdir()
        (destination / "scores" / "scores_mage__desklib.csv").write_text("benchmark,row,origin,words,detector,score\nmage,0,ai,10,d,0.5\n", encoding="utf-8")
        (destination / "costs" / "costs_small.csv").write_text("detector,machine\nd,m\n", encoding="utf-8")
        return destination

    monkeypatch.setattr(hub, "download", download)


def test_fetch_reads_a_local_trial_folder_and_the_pinned_baselines(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    fake_download(monkeypatch, tmp_path, calls)
    context = fetch_context(tmp_path, repository, baselines=baselines_file(tmp_path / "baselines.toml"))
    hub.fetch(context)
    assert sorted(path.name for path in (context.work / "model").iterdir()) == ["config.json", "model.safetensors", "tokenizer.json"]
    assert (context.work / "record" / "record.toml").is_file() and (context.work / "baselines" / "scores" / "scores_mage__desklib.csv").is_file()
    assert calls == [("owner/baselines", "dataset", "d" * 40, ["scores/scores_mage__*.csv", "costs/costs_small.csv"])]
    assert context.details["weights_sha256"] == hashlib.sha256(WEIGHTS).hexdigest()


def test_fetch_refuses_baselines_with_no_pinned_commit_and_says_what_to_upload(tmp_path: Path, repository: Path) -> None:
    context = fetch_context(tmp_path, repository, baselines=baselines_file(tmp_path / "baselines.toml", revision=""))
    with pytest.raises(StageError, match="upload the stored baseline scores"):
        hub.fetch(context)


def test_fetch_refuses_a_release_folder_that_does_not_match_its_hashes(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_download(monkeypatch, tmp_path, [])
    context = fetch_context(tmp_path, repository, baselines=baselines_file(tmp_path / "baselines.toml"))
    (tmp_path / "trial" / "release" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(StageError, match="does not match the hashes"):
        hub.fetch(context)


def test_fetch_refuses_a_model_without_its_record(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_download(monkeypatch, tmp_path, [])
    context = fetch_context(tmp_path, repository, baselines=baselines_file(tmp_path / "baselines.toml"))
    (tmp_path / "trial" / "record" / "evaluation_test.json").unlink()
    with pytest.raises(StageError, match="evaluation_test.json"):
        hub.fetch(context)


def test_a_bundle_ties_the_record_to_the_weights_and_names_every_file(tmp_path: Path) -> None:
    release = release_folder(tmp_path / "release")
    evaluation = tmp_path / "evaluation_test"
    evaluation.mkdir()
    (evaluation / "evaluation.json").write_text("{}", encoding="utf-8")
    (evaluation / "evaluation.csv").write_text("evaluation,metric\n", encoding="utf-8")
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "metrics.csv").write_text("model\n", encoding="utf-8")
    (teacher / "uncertainty.csv").write_text("split\n", encoding="utf-8")
    extra = tmp_path / "corpus.json"
    extra.write_text("{}", encoding="utf-8")
    record = tmp_path / "record.toml"
    record.write_text("[training]\nseed = 0\n[benchmarks]\nlitmus_trained_on = []\n", encoding="utf-8")
    written = bundle.bundle(release, tmp_path / "out", record=record, evaluation_test=evaluation, evaluation_validation=None, teacher=teacher, extra={"corpus_report.json": extra})
    names = sorted(path.relative_to(tmp_path / "out").as_posix() for path in written)
    assert names == ["record/corpus_report.json", "record/evaluation_test.csv", "record/evaluation_test.json", "record/record.toml", "record/teacher_metrics.csv",
                     "record/teacher_uncertainty.csv", "release/config.json", "release/model.safetensors", "release/tokenizer.json"]
    stated = tomllib.loads((tmp_path / "out" / "record" / "record.toml").read_text(encoding="utf-8"))["identity"]["weights_sha256"]
    assert stated == receipts.file_digest(release / "model.safetensors")


def test_a_bundle_refuses_a_record_of_other_weights_and_a_used_folder(tmp_path: Path) -> None:
    release = release_folder(tmp_path / "release")
    evaluation = tmp_path / "e"
    evaluation.mkdir()
    (evaluation / "evaluation.json").write_text("{}", encoding="utf-8")
    record = tmp_path / "record.toml"
    record.write_text('[identity]\nweights_sha256 = "' + "0" * 64 + '"\n', encoding="utf-8")
    arguments = {"record": record, "evaluation_test": evaluation, "evaluation_validation": None, "teacher": None, "extra": {}}
    with pytest.raises(StageError, match="states weights"):
        bundle.bundle(release, tmp_path / "out", **arguments)
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("x", encoding="utf-8")
    with pytest.raises(StageError, match="not empty"):
        bundle.bundle(release, tmp_path / "used", **arguments)


def test_the_record_template_lists_what_the_documents_read_and_is_valid_toml(repository: Path) -> None:
    page = repository / "docs" / "page.md"
    page.write_text(page.read_text(encoding="utf-8") + "<!--litmus:record.training.seed-->[PENDING]<!--/litmus--> <!--litmus:record.splits.rows[set=train].human as int-->[PENDING]<!--/litmus-->\n",
                    encoding="utf-8", newline="\n")
    text = bundle.template(repository)
    parsed = tomllib.loads(text)
    assert parsed["training"] == {"seed": ""} and parsed["splits"]["rows"] == [{"set": "train", "human": ""}] and parsed["benchmarks"]["litmus_trained_on"] == []
    assert "# read by docs/page.md" in text
