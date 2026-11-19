"""Shared fixtures: a placeholder build of the page, the real legal documents, a fake scorer, a settable clock and
an app over a local directory standing in for the volume."""

from __future__ import annotations

import pytest

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from litmus_contribute.config import Settings
from litmus_contribute.legal import Legal, load_legal
from litmus_contribute.shards import ContainerName, LocalStore
from litmus_contribute.site import load_site
from litmus_contribute.usage import MemoryTable, UsageLedger
from litmus_contribute.web import create_app
from starlette.applications import Starlette
from starlette.testclient import TestClient


CONTRIBUTE = Path(__file__).resolve().parents[1]
LEGAL = CONTRIBUTE / "legal"
PRIVATE_PAGE = "https://lhallee.github.io/litmus/"
START = 1_795_000_000.0  # 2026-11-18, seconds since the epoch


class FakeScorer:
    """Stands in for the engine's binding: one token a word, a fixed small reading."""

    version = "fake-model-1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def score(self, text: str) -> dict[str, Any]:
        self.calls.append(text)
        return {"n_tokens": len(text.split()), "p_values": {"human": 0.5, "ai": 0.04, "mixed": 0.3},
                "sets": {"0.05": ["human", "mixed"]}, "strip": {"position": 0.35, "band_lo": 0.0, "band_hi": 0.5}}


class Clock:
    def __init__(self, now: float = START) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Site:
    client: TestClient
    app: Starlette
    root: Path
    scorer: FakeScorer
    usage: UsageLedger
    clock: Clock
    legal: Legal


def write_placeholder_site(directory: Path) -> Path:
    """The contribute build does not exist yet: a page, a hashed script, a hashed wasm module and a stylesheet."""
    (directory / "assets").mkdir(parents=True)
    (directory / "index.html").write_text("<!doctype html><title>Litmus</title><p>Placeholder page.</p>" + "<!-- pad -->" * 200,
                                          encoding="utf-8")
    (directory / "assets" / "litmus-app-0123456789abcdef.js").write_text("console.log('litmus');\n" * 100, encoding="utf-8")
    (directory / "assets" / "litmus-app-0123456789abcdef_bg.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00" + bytes(4096))
    (directory / "style.css").write_text("body { margin: 0; }\n" * 50, encoding="utf-8")
    return directory


@pytest.fixture
def legal() -> Legal:
    return load_legal(LEGAL, PRIVATE_PAGE)


@pytest.fixture
def placeholder_site(tmp_path: Path) -> Path:
    return write_placeholder_site(tmp_path / "site")


@pytest.fixture
def make_site(tmp_path: Path, placeholder_site: Path, legal: Legal) -> Iterator[Any]:
    clients: list[TestClient] = []

    def make(settings: Settings | None = None, clock: Clock | None = None) -> Site:
        settings = settings or Settings()
        clock = clock or Clock()
        root = tmp_path / f"volume{len(clients)}"
        root.mkdir()
        name = ContainerName()
        scorer = FakeScorer()
        usage = UsageLedger(MemoryTable(), name, settings.daily_scoring_seconds, clock=clock)
        app = create_app(settings, LocalStore(root), scorer, load_site(placeholder_site), legal, usage, name, PRIVATE_PAGE,
                         clock=clock)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return Site(client=client, app=app, root=root, scorer=scorer, usage=usage, clock=clock, legal=legal)

    yield make
    for client in clients:
        client.__exit__(None, None, None)
