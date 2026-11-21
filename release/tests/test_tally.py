from __future__ import annotations

import datetime
import json
import urllib.error
import urllib.request
import pytest

from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from typing import Any
from litmus_release import tally
from litmus_release.commands import StageError


NOW = datetime.datetime(2026, 11, 21, 22, 40, 12, tzinfo=datetime.UTC)


def asset(name: str, count: int) -> dict[str, Any]:
    return {"name": name, "download_count": count}


RELEASES = [
    {"tag_name": "v1.1.0", "draft": False, "assets": [asset("litmus-app.exe", 2), asset("litmus.exe", 1), asset("litmus-app-linux-x86_64", 2), asset("model.safetensors", 8), asset("SHA256SUMS", 1)]},
    {"tag_name": "v1.2.0", "draft": True, "assets": [asset("litmus-app.exe", 40)]},
    {"tag_name": "v1.0.0", "draft": False, "assets": [asset("litmus-app.exe", 3), asset("litmus-linux-x86_64", 3)]},
]


class Answers:
    """The three services, answering from memory and keeping every request made."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, Mapping[str, str]]] = []

    def __call__(self, address: str, headers: Mapping[str, str]) -> Any:
        self.asked.append((address, headers))
        if address.startswith("https://api.github.com/repos/owner/litmus/releases"):
            return RELEASES if address.endswith("page=1") else []
        if address.startswith("https://huggingface.co/api/models/owner/litmus?"):
            return {"id": "owner/litmus", "downloads": 20, "downloadsAllTime": 28}
        if address.startswith("https://litmus.goatcounter.com/api/v0/stats/total?"):
            return {"total": 314, "total_events": 0, "total_utc": 314, "stats": []}
        raise AssertionError(f"an address the tally has no business reading: {address}")


def with_counter(repository: Path) -> None:
    targets = repository / "release" / "targets.toml"
    targets.write_text(targets.read_text(encoding="utf-8").replace('count_url = ""', 'count_url = "https://litmus.goatcounter.com/count"'), encoding="utf-8")


def test_the_tally_counts_programs_of_published_releases_and_leaves_the_models_files_out(repository: Path) -> None:
    answers = Answers()
    reading = tally.read(repository, fetch=answers, now=NOW, token=None)
    assert reading["at"] == "2026-11-21T22:40:12Z"
    # The draft is not a release yet, and the model's files and the checksums are fetched by the workflow's own jobs.
    assert reading["programs"] == {"v1.1.0": {"litmus-app.exe": 2, "litmus.exe": 1, "litmus-app-linux-x86_64": 2}, "v1.0.0": {"litmus-app.exe": 3, "litmus-linux-x86_64": 3}}
    assert reading["apps"] == 7 and reading["command_lines"] == 4
    assert reading["hub"] == {"all_time": 28, "last_30_days": 20}
    assert reading["page_openings"] is None and "names no counter" in reading["page_openings_note"]
    assert all("goatcounter" not in address for address, _ in answers.asked)
    words = tally.report(reading)
    assert "v1.1.0: 5" in words and "7 of the app, 4 of the command line" in words and "28 to date, 20 in the last 30 days" in words


def test_the_page_openings_are_read_from_the_named_counter_with_the_token(repository: Path) -> None:
    with_counter(repository)
    answers = Answers()
    reading = tally.read(repository, fetch=answers, now=NOW, token="secret-token")
    assert reading["page_openings"] == 314 and reading["page_openings_note"] == "counted at litmus.goatcounter.com"
    address, headers = next((address, headers) for address, headers in answers.asked if "goatcounter" in address)
    assert address == "https://litmus.goatcounter.com/api/v0/stats/total?start=2026-01-01T00%3A00%3A00Z&end=2026-11-21T23%3A00%3A00Z"
    assert headers == {"Authorization": "Bearer secret-token"}
    assert "secret-token" not in tally.report(reading) and "secret-token" not in json.dumps(reading)


def test_a_counter_without_its_token_is_said_and_not_guessed(repository: Path) -> None:
    with_counter(repository)
    answers = Answers()
    reading = tally.read(repository, fetch=answers, now=NOW, token="")
    assert reading["page_openings"] is None and tally.COUNTER_TOKEN in reading["page_openings_note"]
    assert "none read" in tally.report(reading)
    assert all("goatcounter" not in address for address, _ in answers.asked)


def test_each_reading_is_one_line_of_the_tally_file(repository: Path, tmp_path: Path) -> None:
    path = tmp_path / "tally.jsonl"
    for hour in (1, 2):
        tally.append(tally.read(repository, fetch=Answers(), now=NOW.replace(hour=hour), token=None), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["at"] for line in lines] == ["2026-11-21T01:40:12Z", "2026-11-21T02:40:12Z"]


def test_a_service_that_refuses_fails_with_its_address_and_never_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(StageError) as failure:
        tally.fetch_json("https://litmus.goatcounter.com/api/v0/stats/total", {"Authorization": "Bearer secret-token"})
    assert "HTTP 401" in str(failure.value) and "secret-token" not in str(failure.value)
