"""Command `tally`: how often Litmus's programs were downloaded and its browser page opened.

Each number is read from the place that already counts it, so nothing here adds a count of its own: GitHub counts the
downloads of every file of a release, the Hugging Face Hub counts the model's downloads, and the counter named in
`release/targets.toml` (`[counter] count_url`) counts the browser page's openings (`site/site.js`, `countOpening`).
GitHub's totals are cumulative, so one reading is the tally to date; `--out` appends the reading as one dated line to
a file, which gives the tally over time.

A GitHub count is every fetch of a file, by a person or by a program such as a link check, and the release workflow
fetches the model's files in every job. So the tally is of the programs, and the model's files are left out.
"""

from __future__ import annotations

import datetime
import json
import os
import tomllib
import urllib.error
import urllib.parse
import urllib.request

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from .commands import StageError
from .documents import page_counter


GITHUB_API = "https://api.github.com"
HUB_API = "https://huggingface.co/api"
# The environment variable holding the counter's API token (GoatCounter: Settings, API), read and never printed.
COUNTER_TOKEN = "LITMUS_COUNTER_TOKEN"
# A time before any counter existed, so that the counter's total is of every opening it holds.
COUNTER_START = "2026-01-01T00:00:00Z"
# The kinds of program, by the ending of their key in `[programs.assets]`.
KINDS = {"_app": "app", "_cli": "command line"}

# Reads one address with its headers and returns the JSON it answers with.
Fetch = Callable[[str, Mapping[str, str]], Any]


class HubDownloads(TypedDict):
    all_time: int
    last_30_days: int


class Reading(TypedDict):
    """One reading of every count. `programs` is release tag -> file name -> downloads; `page_openings` is none while
    no counter is named or its token is not given, and `page_openings_note` says which."""

    at: str
    programs: dict[str, dict[str, int]]
    apps: int
    command_lines: int
    hub: HubDownloads
    page_openings: int | None
    page_openings_note: str


def fetch_json(address: str, headers: Mapping[str, str]) -> Any:
    """The JSON at an address, or a failure that names the address (never its headers, which may hold a token)."""
    request = urllib.request.Request(address, headers={"User-Agent": "litmus-release-tally", **headers})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise StageError(f"{address} answered HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise StageError(f"{address} could not be reached: {error.reason}") from error


def program_names(targets: Mapping[str, Any]) -> dict[str, str]:
    """File name -> kind of program, for every program a release may hold."""
    names = {}
    for key, name in targets["programs"]["assets"].items():
        kinds = [kind for ending, kind in KINDS.items() if key.endswith(ending)]
        assert len(kinds) == 1, f"release/targets.toml: [programs.assets] {key} names neither an app nor a command line"
        names[name] = kinds[0]
    return names


def program_downloads(fetch: Fetch, repository: str, names: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Release tag -> program file -> the downloads GitHub counts, for every published release, newest first."""
    downloads: dict[str, dict[str, int]] = {}
    page = 1
    while True:
        releases = fetch(f"{GITHUB_API}/repos/{repository}/releases?per_page=100&page={page}", {"Accept": "application/vnd.github+json"})
        if not releases:
            return downloads
        for release in releases:
            if release["draft"]:
                continue
            downloads[release["tag_name"]] = {asset["name"]: asset["download_count"] for asset in release["assets"] if asset["name"] in names}
        page += 1


def hub_downloads(fetch: Fetch, model_repository: str) -> HubDownloads:
    """The model's downloads as the Hub counts them: `downloadsAllTime`, and `downloads`, its last 30 days."""
    found = fetch(f"{HUB_API}/models/{model_repository}?expand[]=downloads&expand[]=downloadsAllTime", {})
    return {"all_time": found["downloadsAllTime"], "last_30_days": found["downloads"]}


def page_openings(fetch: Fetch, count_address: str, token: str | None, now: datetime.datetime) -> tuple[int | None, str]:
    """The browser page's openings to date and a note, or none with the reason they were not read.

    The counter is a GoatCounter site; its total is `GET /api/v0/stats/total` (https://www.goatcounter.com/api.html),
    whose `total` is the visits between `start` and `end`.
    """
    if not count_address:
        return None, "not counted: release/targets.toml names no counter"
    if not token:
        return None, f"not read: the counter's API token is not in {COUNTER_TOKEN}"
    end = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    query = urllib.parse.urlencode({"start": COUNTER_START, "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")})
    found = fetch(f"{count_address.removesuffix('/count')}/api/v0/stats/total?{query}", {"Authorization": f"Bearer {token}"})
    return found["total"], f"counted at {urllib.parse.urlsplit(count_address).hostname}"


def read(repository: Path, fetch: Fetch = fetch_json, now: datetime.datetime | None = None, token: str | None = None) -> Reading:
    """Every count, read now."""
    targets = tomllib.loads((repository / "release" / "targets.toml").read_text(encoding="utf-8"))
    at = now if now is not None else datetime.datetime.now(datetime.UTC)
    names = program_names(targets)
    programs = program_downloads(fetch, targets["github"]["repo"], names)
    by_kind = {kind: sum(count for files in programs.values() for name, count in files.items() if names[name] == kind) for kind in KINDS.values()}
    openings, note = page_openings(fetch, page_counter(targets), token if token is not None else os.environ.get(COUNTER_TOKEN), at)
    return {"at": at.strftime("%Y-%m-%dT%H:%M:%SZ"), "programs": programs, "apps": by_kind["app"], "command_lines": by_kind["command line"],
            "hub": hub_downloads(fetch, targets["hub"]["model_repo"]), "page_openings": openings, "page_openings_note": note}


def report(reading: Reading) -> str:
    """The reading in words, one line per release and per count."""
    width = max((len(name) for files in reading["programs"].values() for name in files), default=0)
    lines = [f"Litmus, counted at {reading['at']}", "", "Program downloads (GitHub's count of each file of each release):"]
    for tag, files in reading["programs"].items():
        lines.append(f"  {tag}: {sum(files.values())}")
        lines.extend(f"    {name:<{width}}  {count}" for name, count in files.items())
    lines.append(f"  every release: {reading['apps']} of the app, {reading['command_lines']} of the command line")
    lines.append("")
    lines.append(f"Model downloads (Hugging Face Hub): {reading['hub']['all_time']} to date, {reading['hub']['last_30_days']} in the last 30 days")
    openings = reading["page_openings"]
    lines.append(f"Browser page openings: {openings if openings is not None else 'none read'} ({reading['page_openings_note']})")
    return "\n".join(lines)


def append(reading: Reading, path: Path) -> None:
    """Add the reading to a tally file, one JSON line per reading."""
    with path.open("a", encoding="utf-8", newline="\n") as tally:
        tally.write(json.dumps(reading, ensure_ascii=False) + "\n")
