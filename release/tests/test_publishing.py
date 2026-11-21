from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import pytest

from pathlib import Path
from litmus_release import benchmarking, packaging, publishing, stages
from litmus_release.commands import StageError
from litmus_release.stages import Stage
from tests.conftest import WEIGHTS


def build(work: Path, platform_name: str, commit: str) -> None:
    folder = work / "build" / platform_name
    (folder / "site" / "assets").mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if platform_name == "windows" else ""
    for program in ("litmus", "litmus-app"):
        (folder / f"{program}{suffix}").write_bytes(f"{platform_name} {program}".encode())
    (folder / "manifest.json").write_text(json.dumps({"source": {"commit": commit, "uncommitted_changes_in_source": False}}), encoding="utf-8")
    (folder / "site" / "assets" / "results.json").write_text("{}", encoding="utf-8")


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def test_the_release_assets_carry_the_names_the_site_and_the_install_scripts_expect(context) -> None:
    for name in ("windows", "linux"):
        build(context.work, name, "c" * 40)
    assets = packaging.github_assets(context)
    assert {"litmus-app.exe", "litmus.exe", "litmus-app-linux-x86_64", "litmus-linux-x86_64", "model.safetensors", "config.json", "tokenizer.json", "manifest.json", "manifest-linux.json",
            "results.json"} == set(assets)
    sums = packaging.sums_text(assets).splitlines()
    assert f"{hashlib.sha256(b'linux litmus').hexdigest()}  litmus-linux-x86_64" in sums
    assert f"{hashlib.sha256(WEIGHTS).hexdigest()}  model.safetensors" in sums
    assert all("  SHA256SUMS" not in line for line in sums)


def test_assemble_lays_out_the_three_destinations_and_sends_nothing(context) -> None:
    for name in ("windows", "linux"):
        build(context.work, name, "c" * 40)
    (context.repository / "release" / "release_notes.md").write_text("notes\n", encoding="utf-8")
    (context.repository / "release" / "model_card.md").write_text("card\n", encoding="utf-8")
    (context.repository / "LICENSE").write_text("license\n", encoding="utf-8")
    (context.work / "build" / "windows" / "site" / "index.html").write_text("<html></html>", encoding="utf-8")
    packaging.assemble(context)
    publish = context.work / "publish"
    assert sorted(path.name for path in (publish / "github").iterdir()) == sorted([*packaging.github_assets(context), "SHA256SUMS", "NOTES.md"])
    assert (publish / "hub" / "README.md").read_text(encoding="utf-8") == "card\n" and (publish / "pages" / "index.html").is_file()


def test_a_listed_platform_without_its_programs_is_an_error(context) -> None:
    build(context.work, "windows", "c" * 40)
    with pytest.raises(StageError, match="linux"):
        packaging.github_assets(context)


def list_platforms(context, platforms: str) -> None:
    targets = context.repository / "release" / "targets.toml"
    text, changed = re.subn(r"(?m)^platforms = .*$", f"platforms = {platforms}", targets.read_text(encoding="utf-8"))
    assert changed == 1
    targets.write_text(text, encoding="utf-8")


def test_macos_is_listed_for_the_site_and_assembled_by_no_build(context) -> None:
    for name in ("windows", "linux"):
        build(context.work, name, "c" * 40)
    before = set(packaging.github_assets(context))
    list_platforms(context, '["windows", "linux", "macos"]')
    assert set(packaging.github_assets(context)) == before and not any("macos" in name for name in before)
    list_platforms(context, '["windows", "linux", "solaris"]')
    with pytest.raises(StageError, match="solaris"):
        packaging.github_assets(context)


def test_the_macos_programs_are_attached_before_the_site_that_links_them(context, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert [stage.name for stage in publishing.stages(())] == ["publish", "macos", "pages"]
    assert {stage.name: stage.needs for stage in publishing.stages(())} == {"publish": ("assemble",), "macos": ("publish",), "pages": ("publish",)}
    programs = ["litmus-macos-arm64", "litmus-app-macos-arm64", "litmus-macos-x86_64", "litmus-app-macos-x86_64"]
    settings = publishing.targets(context)
    assert publishing.missing_macos_programs(settings, {"assets": [{"name": name} for name in programs]}) == []
    list_platforms(context, '["windows", "linux", "macos"]')
    settings = publishing.targets(context)
    assert publishing.missing_macos_programs(settings, {"assets": [{"name": name} for name in programs]}) == []
    assert publishing.missing_macos_programs(settings, {"assets": [{"name": name} for name in programs[:3]]}) == ["litmus-app-macos-x86_64"]
    assert publishing.missing_macos_programs(settings, None) == programs

    # A public release without the macOS programs: the stage refuses before it touches the branch.
    monkeypatch.setattr(publishing, "gh_json", lambda running, *arguments: {"isDraft": False, "tagName": "v1.0.0", "assets": [{"name": "litmus.exe"}]})
    with pytest.raises(StageError, match="run the `macos` stage first"):
        publishing.pages(context, ())
    assert not (context.work / "pages_repo").exists()
    context.options.dry_run = True
    publishing.pages(context, ())
    shown = capsys.readouterr().out
    assert "will refuse until the `macos` stage has attached them" in shown and "litmus-app-macos-arm64" in shown
    context.options.ahead_of_macos = True
    publishing.pages(context, ())
    assert "resolve once the `macos` stage has attached them" in capsys.readouterr().out


def test_the_macos_stage_waits_for_a_public_release(context, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publishing, "gh_json", lambda running, *arguments: {"isDraft": True, "tagName": "v1.0.0"})
    with pytest.raises(StageError, match="not public yet"):
        publishing.macos(context, ())


def repository_with_origin(context) -> str:
    root = context.repository
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.org")
    git(root, "config", "user.name", "t")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "release")
    origin = root.parent / "origin.git"
    git(root.parent, "init", "-q", "--bare", str(origin))
    git(root, "remote", "add", "origin", str(origin))
    git(root, "push", "-q", "origin", "main")
    return git(root, "rev-parse", "HEAD")


def done(context, chain: tuple[Stage, ...]) -> None:
    for stage in chain:
        context.work.joinpath(f"{stage.name}.txt").write_text(stage.name, encoding="utf-8")
        stages.run_stage(context, stage)


def test_the_gate_names_every_reason_not_to_publish(context) -> None:
    commit = repository_with_origin(context)
    for name in ("windows", "linux"):
        build(context.work, name, "c" * 40)
    (context.repository / "stray.txt").write_text("uncommitted", encoding="utf-8")
    pending = Stage("assemble", (), "a stage not done", lambda running: None, lambda running: [running.work / "assemble.txt"])
    with pytest.raises(StageError) as error:
        publishing.gate(context, (pending,))
    message = str(error.value)
    assert "stage assemble is pending" in message and "uncommitted changes" in message and f"build was made at commit {'c' * 40}" in message and commit in message
    assert "not a branch head" not in message


def test_a_dry_run_prints_what_it_would_send_and_sends_nothing(context, capsys: pytest.CaptureFixture[str]) -> None:
    (context.repository / "docs" / "page.md").write_text("No slots here.\n", encoding="utf-8", newline="\n")
    commit = repository_with_origin(context)
    for name in ("windows", "linux"):
        build(context.work, name, commit)
    (context.repository / "release" / "release_notes.md").write_text("notes\n", encoding="utf-8")
    (context.repository / "release" / "model_card.md").write_text("card\n", encoding="utf-8")
    (context.repository / "LICENSE").write_text("license\n", encoding="utf-8")
    git(context.repository, "add", "-A")
    git(context.repository, "commit", "-q", "-m", "notes")
    commit = git(context.repository, "rev-parse", "HEAD")
    git(context.repository, "push", "-q", "origin", "main")
    for name in ("windows", "linux"):
        build(context.work, name, commit)
    (context.work / "build" / "windows" / "site" / "index.html").write_text("<html></html>", encoding="utf-8")
    packaging.assemble(context)
    context.options.dry_run = True
    publishing.publish(context, ())
    shown = capsys.readouterr().out
    assert "[dry run] Hub: upload" in shown and "owner/litmus" in shown and "then tag it v1.0.0" in shown
    assert f"gh release create v1.0.0 --repo owner/litmus --draft --target {commit}" in shown
    assert "litmus-linux-x86_64" in shown and "SHA256SUMS" in shown and "--notes-file" in shown and "NOTES.md" in shown
    assert git(context.repository, "ls-remote", "--tags", "origin") == ""


def test_the_macos_stage_names_its_workflow_run_and_the_one_replacement(context, capsys: pytest.CaptureFixture[str]) -> None:
    context.options.dry_run = True
    publishing.macos(context, ())
    shown = capsys.readouterr().out
    assert "gh workflow run release-build.yml --repo owner/litmus --ref v1.0.0 -f model_url=https://github.com/owner/litmus/releases/download/v1.0.0/" in shown
    assert "litmus-macos-arm64" in shown and "REPLACE the release's SHA256SUMS" in shown


def test_litmus_scores_are_relabeled_and_must_cover_the_baselines_rows(context) -> None:
    fields = ["benchmark", "row", "origin", "generator", "domain", "attack", "language", "words", "detector", "score"]

    def write(path: Path, detector: str, rows: list[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({"benchmark": "mage", "row": row, "origin": "ai", "generator": "g", "domain": "d", "attack": "", "language": "en", "words": 10, "detector": detector, "score": 0.5} for row in rows)

    baseline = context.work / "baselines" / "scores" / "scores_mage__desklib.csv"
    write(baseline, "Desklib", [0, 1, 2])
    part = context.work / "scores" / "part.csv"
    write(part, "Litmus engine", [0, 1, 2])
    final = context.work / "scores" / "scores_mage__litmus.csv"
    benchmarking.relabel(part, final)
    assert {row["detector"] for row in csv.DictReader(final.open(encoding="utf-8"))} == {"Litmus"}
    benchmarking.same_rows(context, "mage", final)
    write(final, "Litmus", [0, 1])
    with pytest.raises(StageError, match="different rows"):
        benchmarking.same_rows(context, "mage", final)
