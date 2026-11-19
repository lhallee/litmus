"""`python -m litmus_corpus catalog check`: verify the catalogs against the Hub before a build.

For every source the build would read (training sources and benchmark splits, not the first corpus), at the
catalog's pinned revision:

    revision   the repository exists and has the revision
    columns    each column the line names is a feature of its config and split (the datasets-server's first
               rows; for a `file:` config, or when the server has no rows, the first streamed row)
    rows       the split's row count, beside the catalog's (the datasets-server's size), flagged when they
               differ by more than `ROW_TOLERANCE`; a partial count (a large dataset the server converted
               only in part) is a lower bound, flagged only when it exceeds the catalog's

The Hub token, when one is set, is read from the environment by `huggingface_hub`, so gated sources are checked as
the build will read them.
"""

from __future__ import annotations

import httpx

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, get_session
from huggingface_hub.utils import build_hf_headers

from .catalog import FIRST_CORPUS, SourceSpec, load_catalog
from .read import open_stream
from .storage import JsonObject


SERVER = "https://datasets-server.huggingface.co"
DEFAULT_CONFIG = "default"
ROW_TOLERANCE = 0.01
REQUEST_SECONDS = 90.0


def server_json(endpoint: str, parameters: dict[str, str]) -> JsonObject:
    """One datasets-server response, or `{"error": ...}`."""
    try:
        response = get_session().get(f"{SERVER}/{endpoint}", params=parameters, headers=build_hf_headers(), timeout=REQUEST_SECONDS)
    except httpx.HTTPError as error:
        return {"error": f"{type(error).__name__}: {error}"}
    if response.status_code != 200:
        return {"error": f"[{response.status_code}] {response.text[:200]}"}

    return response.json()


def server_features(spec: SourceSpec, split: str) -> set[str] | None:
    """The feature names of a config and split from the datasets-server, or None when it has none."""
    if spec.data_file is not None:
        return None
    first = server_json("first-rows", {"dataset": spec.repo, "config": spec.config or DEFAULT_CONFIG, "split": split})
    if "error" in first:
        return None

    return {feature["name"] for feature in first.get("features", [])}


def streamed_features(spec: SourceSpec, split: str) -> set[str]:
    """The column names of the first streamed row, as the build reads it."""
    for row in open_stream(spec, split).take(1):
        return set(row)

    return set()


def server_rows(spec: SourceSpec) -> tuple[int | None, bool]:
    """The rows of the source's config and splits by the datasets-server, or None when it does not say, and
    whether the count is partial (the server converted only part of a large dataset: a lower bound)."""
    if spec.data_file is not None:
        return None, False
    size = server_json("size", {"dataset": spec.repo})
    if "error" in size:
        return None, False
    config = spec.config or DEFAULT_CONFIG
    parts = [entry for entry in size.get("size", {}).get("splits", []) if entry["config"] == config and entry["split"] in spec.splits]
    if len(parts) != len(spec.splits):
        return None, False

    return sum(entry["num_rows"] for entry in parts), bool(size.get("partial"))


def check_source(spec: SourceSpec, api: HfApi, features_of: Callable[[SourceSpec, str], set[str]] = streamed_features) -> JsonObject:
    """The problems of one source, and what was found."""
    found: JsonObject = {"name": spec.name, "repo": spec.repo, "config": spec.config, "splits": list(spec.splits), "problems": []}
    try:
        api.dataset_info(spec.repo, revision=spec.revision)
    except Exception as error:  # noqa: BLE001 - every Hub failure is a finding of the check, reported by name
        found["problems"].append(f"revision {spec.revision}: {type(error).__name__}: {str(error)[:200]}")
        return found
    for split in spec.splits:
        try:
            names = server_features(spec, split)
            if names is None:
                names = features_of(spec, split)
        except Exception as error:  # noqa: BLE001 - as above
            found["problems"].append(f"split {split}: cannot read its first row: {type(error).__name__}: {str(error)[:200]}")
            continue
        missing = [column for column in spec.needed_columns() if column not in names]
        if missing:
            found["problems"].append(f"split {split}: columns not found: {', '.join(missing)} (has {', '.join(sorted(names))[:300]})")
    rows, partial = server_rows(spec)
    found["rows_hub"] = rows
    found["rows_hub_partial"] = partial
    found["rows_catalog"] = spec.rows
    differs = rows is not None and spec.rows and abs(rows - spec.rows) > ROW_TOLERANCE * spec.rows
    if differs and not (partial and rows < spec.rows):
        found["problems"].append(f"rows: the Hub has {rows:,}{' at least' if partial else ''}, the catalog {spec.rows:,}")

    return found


def check_catalog(sources_path: Path, benchmarks_path: Path, max_priority: int, threads: int) -> JsonObject:
    """Every check for every source a build of `max_priority` would read, and the catalog's own problems."""
    catalog = load_catalog(sources_path, benchmarks_path, max_priority, first_corpus=False)
    specs = [spec for spec in catalog.specs if spec.name != FIRST_CORPUS]
    api = HfApi()
    with ThreadPoolExecutor(threads) as pool:
        findings = list(pool.map(lambda spec: check_source(spec, api), specs))

    return {
        "sources": len(specs),
        "with_problems": sum(1 for found in findings if found["problems"]),
        "results": findings,
        "catalog_problems": catalog.problems,
        "excluded": catalog.excluded,
        "disagreements": catalog.disagreements,
    }
