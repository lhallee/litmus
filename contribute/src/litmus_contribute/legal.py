"""The documents a visitor is shown, read from `legal/`: the on-page notice, the Terms of Use and the Privacy Notice.

All three carry one `version`, the one every stored row records, and one `status` (`draft` until a maintainer approves them).
The notice's front matter also carries `consent_label`, the text beside the box a visitor ticks to agree before a
text can be sent. A blank still to fill is written `{{BLANK: what goes here}}`; a deployment refuses documents
that hold one. The documents link to each other by file name, which resolves in the repository, and are served with
those links pointing at the site's routes.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path


DOCUMENTS = {"notice": "NOTICE.md", "terms_of_use": "TERMS_OF_USE.md", "privacy_notice": "PRIVACY_NOTICE.md"}
PRIVATE_PAGE = "{private_page}"
FRONT_MATTER = re.compile(r"\A---\n(?P<header>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)
VERSION = re.compile(r"[A-Za-z0-9._-]{1,64}")
BLANK = re.compile(r"\{\{BLANK:[^}]*\}\}")
ROUTES = {"(TERMS_OF_USE.md)": "(/legal/terms-of-use)", "(PRIVACY_NOTICE.md)": "(/legal/privacy-notice)"}


@dataclass(frozen=True)
class Legal:
    version: str
    status: str  # `draft` or `approved`
    notice: str  # Markdown, the lines beside the consent box and the Check button
    consent_label: str  # Markdown, the label of the box a visitor ticks to agree
    terms_of_use: str
    privacy_notice: str

    def blanks(self) -> list[str]:
        """Every `{{BLANK: ...}}` left in the documents and the consent label."""
        texts = (self.notice, self.consent_label, self.terms_of_use, self.privacy_notice)
        return [blank for text in texts for blank in BLANK.findall(text)]


@dataclass(frozen=True)
class Document:
    version: str
    status: str
    body: str
    fields: dict[str, str]  # every front-matter field, version and status included


def read_document(path: Path) -> Document:
    """A document's front matter and body."""
    match = FRONT_MATTER.match(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
    if match is None:
        raise ValueError(f"{path} has no front matter between --- lines")
    fields = {}
    for line in match["header"].splitlines():
        if line.strip():
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    version, status = fields.get("version", ""), fields.get("status", "")
    if not VERSION.fullmatch(version):
        raise ValueError(f"{path}: version {version!r} is not 1 to 64 of A-Z a-z 0-9 . _ -")
    if status not in ("draft", "approved"):
        raise ValueError(f"{path}: status must be draft or approved, got {status!r}")
    return Document(version=version, status=status, body=match["body"].strip(), fields=fields)


def load_legal(directory: Path, private_page: str) -> Legal:
    """Read the three documents, which must share one version and one status, and fill in the private page."""
    if not private_page.startswith("https://"):
        raise ValueError(f"The private page's address must be https, got {private_page!r}")
    read = {field: read_document(directory / name) for field, name in DOCUMENTS.items()}
    versions = {document.version for document in read.values()}
    statuses = {document.status for document in read.values()}
    if len(versions) != 1 or len(statuses) != 1:
        raise ValueError(f"{directory}: the documents disagree on version {sorted(versions)} or status {sorted(statuses)}")
    notice = read["notice"]
    if PRIVATE_PAGE not in notice.body:
        raise ValueError(f"{directory / DOCUMENTS['notice']} never names the private page ({PRIVATE_PAGE})")
    consent_label = notice.fields.get("consent_label", "")
    if not consent_label:
        raise ValueError(f"{directory / DOCUMENTS['notice']} has no consent_label in its front matter")
    bodies = {field: as_served(document.body, private_page) for field, document in read.items()}
    return Legal(version=versions.pop(), status=statuses.pop(), consent_label=as_served(consent_label, private_page),
                 **bodies)


def as_served(body: str, private_page: str) -> str:
    """A document's body as the site serves it: the private page filled in, links to the documents' routes."""
    for file_link, route in ROUTES.items():
        body = body.replace(file_link, route)
    return body.replace(PRIVATE_PAGE, private_page)
