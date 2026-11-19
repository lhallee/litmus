"""The legal documents: one version and status, the private page filled in, no blanks, the notice beside the consent box
saying what the terms promise, the consent label required."""

from __future__ import annotations

import shutil
import pytest

from pathlib import Path
from litmus_contribute.legal import DOCUMENTS, load_legal, read_document

from conftest import LEGAL, PRIVATE_PAGE


CONTACT = "https://github.com/lhallee/litmus/issues"


def test_documents_load_as_one_draft() -> None:
    legal = load_legal(LEGAL, PRIVATE_PAGE)
    every = legal.notice + legal.consent_label + legal.terms_of_use + legal.privacy_notice
    assert legal.status == "draft"  # only a maintainer sets approved
    assert PRIVATE_PAGE in legal.notice and "{private_page}" not in legal.notice
    assert legal.blanks() == []  # every blank is filled
    assert "open source" not in every.lower()
    assert "PolyForm Noncommercial" in legal.terms_of_use
    assert "CC BY-NC" not in every and "may be published" not in every  # texts are never published
    assert "Article" not in legal.privacy_notice and "GDPR" not in every  # one policy everywhere, no law named
    assert "designated agent" not in every.lower()  # no agent is filed, so none is claimed


def test_consent_label_links_the_documents() -> None:
    legal = load_legal(LEGAL, PRIVATE_PAGE)
    assert legal.consent_label.startswith("I agree to the [Terms of Use](/legal/terms-of-use) and the "
                                          "[Privacy Notice](/legal/privacy-notice)")
    assert "18 or older" in legal.consent_label


def test_documents_keep_their_promises() -> None:
    legal = load_legal(LEGAL, PRIVATE_PAGE)
    sensitive = "Please do not paste sensitive or personal information, yours or anyone else's"
    assert sensitive in legal.notice and sensitive in legal.privacy_notice
    assert "Please do not submit sensitive or personal information, yours or anyone else's" in legal.terms_of_use
    assert "We never publish it or release it in any dataset" in legal.notice
    assert "We never publish a Submission" in legal.terms_of_use and "Never published" in legal.privacy_notice
    for document in (legal.notice, legal.terms_of_use, legal.privacy_notice):
        assert "removal form" in document and "30 days" in document
    for document in (legal.terms_of_use, legal.privacy_notice):
        assert CONTACT in document and "Issues are public" in document
    assert "Commonwealth of Pennsylvania" in legal.terms_of_use and "courts located in Pennsylvania" in legal.terms_of_use
    assert "tick the box beside the **Check** button" in legal.terms_of_use
    # how a reading is used is the reader's decision: recommendations, never rules
    assert "We recommend not using a Reading alone" in legal.terms_of_use
    assert "We recommend not using a reading alone to accuse anyone" in legal.notice
    for document in (legal.notice, legal.terms_of_use, legal.privacy_notice):
        assert "never proof" not in document and "must not use a Reading" not in document


def test_mismatched_versions_are_refused(tmp_path: Path) -> None:
    copy = tmp_path / "legal"
    shutil.copytree(LEGAL, copy)
    terms = copy / DOCUMENTS["terms_of_use"]
    version = read_document(terms).version
    terms.write_text(terms.read_text(encoding="utf-8").replace(f"version: {version}", "version: other", 1),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_legal(copy, PRIVATE_PAGE)


def test_notice_without_consent_label_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / "legal"
    shutil.copytree(LEGAL, copy)
    notice = copy / DOCUMENTS["notice"]
    lines = notice.read_text(encoding="utf-8").splitlines(keepends=True)
    notice.write_text("".join(line for line in lines if not line.startswith("consent_label:")), encoding="utf-8")
    with pytest.raises(ValueError, match="consent_label"):
        load_legal(copy, PRIVATE_PAGE)


def test_private_page_must_be_https() -> None:
    with pytest.raises(ValueError, match="https"):
        load_legal(LEGAL, "http://example.org")
