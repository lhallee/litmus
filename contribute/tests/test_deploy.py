"""The Modal web app's definition builds against the pinned SDK without reaching Modal: the decorators take these
settings, and a draft of the legal documents is refused unless a private trial says so."""

from __future__ import annotations

import dataclasses
import importlib
import sys
import pytest

from pathlib import Path
from litmus_contribute import legal

from conftest import PRIVATE_PAGE, write_placeholder_site


def deploy_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft: str | None) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "config.json").write_text("{}", encoding="utf-8")
    wheel = tmp_path / "litmus_engine-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    wheel.write_bytes(b"not a real wheel")
    monkeypatch.setenv("LITMUS_CONTRIBUTE_SITE", str(write_placeholder_site(tmp_path / "site")))
    monkeypatch.setenv("LITMUS_RELEASE", str(release))
    monkeypatch.setenv("LITMUS_ENGINE_WHEEL", str(wheel))
    monkeypatch.setenv("LITMUS_PRIVATE_PAGE", PRIVATE_PAGE)
    if draft is None:
        monkeypatch.delenv("LITMUS_LEGAL_DRAFT", raising=False)
    else:
        monkeypatch.setenv("LITMUS_LEGAL_DRAFT", draft)
    sys.modules.pop("litmus_contribute.deploy_web", None)


def test_web_app_defines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_environment(tmp_path, monkeypatch, draft=None)  # the documents as they stand are approved
    module = importlib.import_module("litmus_contribute.deploy_web")
    assert module.app.name == "litmus-contribute"
    assert module.MAX_CONTAINERS == 32
    assert module.Contribute is not None


def draft_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have the deployment read the documents as a draft, whatever their status in the repository."""
    def as_draft(directory: Path, private_page: str) -> legal.Legal:
        return dataclasses.replace(read(directory, private_page), status="draft")

    read = legal.load_legal
    monkeypatch.setattr(legal, "load_legal", as_draft)


def test_draft_documents_refuse_to_deploy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_environment(tmp_path, monkeypatch, draft=None)
    draft_documents(monkeypatch)
    with pytest.raises(SystemExit, match="draft"):
        importlib.import_module("litmus_contribute.deploy_web")


def test_a_private_trial_deploys_draft_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_environment(tmp_path, monkeypatch, draft="allow")
    draft_documents(monkeypatch)
    assert importlib.import_module("litmus_contribute.deploy_web").app.name == "litmus-contribute"
