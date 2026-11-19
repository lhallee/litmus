"""The data documents every slot reads: the plan, the targets, the release folder, the record, the results, the
measurements, and the build.

Documents load on first use, so a slot that names a file no stage has written yet fails with that file's name. A
document that describes another model or version than the plan's is refused: a stale file never fills a slot.
"""

from __future__ import annotations

import csv
import json
import re
import tomllib

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .commands import StageError
from .stages import Context
from .values import comparison_counts


class StaleDocument(StageError):
    """A data document an earlier release's stage wrote: for another version, or for other weights."""


class Documents(Mapping[str, Any]):
    """Named documents, each loaded when first read."""

    def __init__(self, loaders: Mapping[str, Callable[[], Any]]) -> None:
        self.loaders = dict(loaders)
        self.cache: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        if name not in self.cache:
            self.cache[name] = self.loaders[name]()
        return self.cache[name]

    def __contains__(self, name: object) -> bool:
        return name in self.loaders

    def __iter__(self) -> Iterator[str]:
        return iter(self.loaders)

    def __len__(self) -> int:
        return len(self.loaders)


NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
RESULTS_SPLITS = {"benchmarks": "**Flag rates on APT-Eval**", "flag_rates": "**Single-core throughput**"}


def typed(cell: str) -> Any:
    """A CSV cell as a number when it is one, else the text; an empty cell stays empty."""
    if NUMBER.match(cell):
        return int(cell) if re.match(r"^-?\d+$", cell) else float(cell)
    return cell


def read_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {"rows": [{key: typed(value) for key, value in row.items()} for row in csv.DictReader(stream)]}


def read_any(path: Path) -> Any:
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"no reader for {path}")


def model_facts(folder: Path) -> dict[str, Any]:
    """What the release folder says about its model: the size, the weights, the file sizes, the bank."""
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = config["model"]
    width, depth, ffn, embed = model["width"], model["depth"], model["ffn_width"], model["embed_width"]
    files = {name: (folder / name).stat().st_size for name in ("model.safetensors", "config.json", "tokenizer.json")}
    bank = dict(zip(config["bank"]["origins"], config["bank"]["counts"], strict=True))
    return {
        "config_name": f"s{width}_l{depth}",
        "width": width,
        "depth": depth,
        "heads": model["heads"],
        "ffn_width": ffn,
        "embed_width": embed,
        # Four attention matrices and two feed-forward matrices a layer, and the embedding's projection (student/SPEC.md, Sizes).
        "ternary_weights": depth * (4 * width * width + 2 * width * ffn) + embed * width,
        "files": files,
        "files_bytes": sum(files.values()),
        "attention": model.get("attention", "float"),
        "output": model.get("output", "float"),
        "bank": bank,
        "bank_total": sum(bank.values()),
        "calibration_rows": config["calibration"]["calibration_rows"],
        "length_groups": len(config["calibration"]["length_bins"]),
    }


def release_facts(context: Context, targets: Mapping[str, Any]) -> dict[str, Any]:
    plan = context.plan
    repo = targets["github"]["repo"]
    base = f"https://github.com/{repo}/releases/download/{plan.tag}/"
    return {
        "version": plan.version,
        "tag": plan.tag,
        "date": plan.date,
        "model_repo": plan.repo,
        "model_revision": plan.revision,
        "download_base": base,
        "latest_base": f"https://github.com/{repo}/releases/latest/download/",
        "page": f"https://github.com/{repo}/releases/tag/{plan.tag}",
    }


def split_results_markdown(text: str) -> dict[str, str]:
    """`results.md` in its three parts: the per-benchmark tables, the flag rates, and the throughput table."""
    first = text.index(RESULTS_SPLITS["benchmarks"])
    second = text.index(RESULTS_SPLITS["flag_rates"])
    return {"benchmarks": text[:first].strip("\n"), "flag_rates": text[first:second].strip("\n"), "throughput": text[second:].strip("\n")}


def checked(context: Context, name: str, document: Mapping[str, Any], *, version: tuple[str, ...] | None = None, weights: tuple[str, ...] | None = None) -> Mapping[str, Any]:
    """The document, if it describes this plan's version and this release's weights where it says so."""
    plan = context.plan
    if version is not None:
        value: Any = document
        for key in version:
            value = value[key]
        if value != plan.version:
            raise StaleDocument(f"{name} is for version {value}, and this release is {plan.version}: rerun the stage that wrote it")
    if weights is not None:
        value = document
        for key in weights:
            value = value[key]
        actual = json.loads((context.work / "receipts" / "fetch.json").read_text(encoding="utf-8"))["details"]["weights_sha256"]
        if value != actual:
            raise StaleDocument(f"{name} was made with weights {str(value)[:12]}, and this release's are {actual[:12]}: rerun the stage that wrote it")
    return document


def documents(context: Context, targets_path: Path) -> Documents:
    """Every document a slot may read, for this release."""
    work, repository = context.work, context.repository
    targets = tomllib.loads(targets_path.read_text(encoding="utf-8"))
    loaders: dict[str, Callable[[], Any]] = {
        "plan": lambda: {"version": context.plan.version, "tag": context.plan.tag, "date": context.plan.date, "repo": context.plan.repo,
                         "revision": context.plan.revision, "model": context.plan.model},
        "release": lambda: release_facts(context, targets),
        "target": lambda: targets,
    }
    if (work / "model" / "config.json").is_file():
        loaders["config"] = lambda: json.loads((work / "model" / "config.json").read_text(encoding="utf-8"))
        loaders["model"] = lambda: model_facts(work / "model")
    record = work / "record"
    if record.is_dir():
        for path in sorted(record.iterdir()):
            if path.suffix in (".toml", ".json", ".csv") and path.is_file():
                # A document is named by its file's stem; a JSON file beside a CSV of the same stem is `<stem>_full`.
                twin = path.suffix == ".json" and path.with_suffix(".csv").is_file()
                loaders[f"{path.stem}_full" if twin else path.stem] = lambda path=path: read_any(path)
    results = repository / "site" / "assets" / "results.json"
    if results.is_file():
        loaders["results"] = lambda: checked(context, "site/assets/results.json", json.loads(results.read_text(encoding="utf-8")), version=("model", "version"))
        loaders["counts"] = lambda: comparison_counts(loaders["results"]())
    links = repository / "site" / "assets" / "links.json"
    if links.is_file():
        loaders["links"] = lambda: json.loads(links.read_text(encoding="utf-8"))
    markdown = work / "results" / "results.md"
    if markdown.is_file():
        loaders["results_md"] = lambda: split_results_markdown(markdown.read_text(encoding="utf-8"))
    for name, path in (("bench", work / "measure" / "bench.json"), ("browser", work / "measure" / "browser.json"), ("build", work / "build" / "windows" / "manifest.json"),
                       ("linux_build", work / "build" / "linux" / "manifest.json")):
        if path.is_file():
            loaders[name] = lambda name=name, path=path: checked(context, name, json.loads(path.read_text(encoding="utf-8")),
                                                                   weights=("weights_sha256",) if name in ("bench", "browser") else None)
    return Documents(loaders)
