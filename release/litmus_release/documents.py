"""Stages `record` and `docs`: every number of every published document, filled from the release's data.

`record` checks that the training record holds what the documents and the comparison read. `docs` writes the results'
release block and the site's links, fills each slot of each Markdown document (`slots.py`), writes each fallback number
of the site's pages from the same data, writes the hashes of the model's files, and refuses to finish while anything
is left unfilled. `gate` is that refusal, and publishing runs it again.
"""

from __future__ import annotations

import json
import os
import re
import tomllib

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import receipts, slots
from .commands import StageError
from .data import Documents, StaleDocument, documents
from .datapath import MissingValue, resolve, spell
from .stages import Context, Stage
from .values import Values


SKIPPED = {".git", "target", "node_modules", "dist", "__pycache__", ".pytest_cache"}
FALLBACK = re.compile(r'(?P<open><(?P<tag>span|dd|td|strong|b|em)\b[^>]*\bdata-value="(?P<key>[^"]+)"[^>]*>)(?P<body>[^<]*)(?P<close></(?P=tag)>)')
FORMAT_ATTRIBUTE = re.compile(r'data-format="([^"]+)"')
# The site script's formats (site/site.js FORMATS), by their names.
SITE_FORMATS = {"mb": "mb1", "int": "int", "num": "f3", "f1": "f1", "pct": "pct1", "text": "text"}
RELEASE_BLOCK = ("version", "date", "app_tokens_per_second", "browser_tokens_per_second", "speed_document_tokens",
                 "app_document_seconds", "browser_document_seconds", "speed_machine")


def markdown_files(repository: Path) -> list[Path]:
    """Every Markdown file of the checkout, in path order."""
    return sorted(path for path in repository.rglob("*.md") if not SKIPPED.intersection(path.relative_to(repository).parts))


def write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def values(context: Context, available: Documents | None = None) -> Values:
    return Values.with_templates(available if available is not None else documents(context, targets_path(context)), context.repository / "release" / "templates.toml")


def targets_path(context: Context) -> Path:
    return context.repository / "release" / "targets.toml"


def read_exact(path: Path) -> str:
    """A file's text with its line ends as they are (`Path.read_text` has no `newline` before Python 3.13)."""
    with path.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def read_text(path: Path) -> str:
    text = read_exact(path)
    if "\r\n" in text and "<!--litmus:" in text:
        raise StageError(f"{path} has CRLF line ends; the release writes LF in files that hold slots (convert it, `git ls-files --eol`)")
    return text


def problems_of_keys(context: Context, available: Documents, only_present: bool) -> list[str]:
    """Every slot whose key does not render now, as `file: key: why`. With `only_present`, a slot that names a document
    that does not exist yet, or one an earlier release wrote, is skipped (the stage that writes it has not run)."""
    renderer = values(context, available)
    found = []
    for path in markdown_files(context.repository):
        text = read_text(path)
        if "<!--litmus:" not in text:
            continue
        for slot in slots.find(text):
            try:
                renderer.render(slot.key, slot.block)
            except StaleDocument:
                if only_present:  # a document an earlier release wrote, which this release's own stage has not yet rewritten
                    continue
                raise
            except (MissingValue, ValueError, KeyError) as error:
                if only_present and isinstance(error, MissingValue) and "no data document named" in str(error):
                    continue
                found.append(f"{path.relative_to(context.repository).as_posix()}: {slot.key}: {error}")
    return found


def check_record(context: Context) -> None:
    record = tomllib.loads((context.work / "record" / "record.toml").read_text(encoding="utf-8"))
    actual = receipts.file_digest(context.work / "model" / "model.safetensors")
    stated = (record.get("identity") or {}).get("weights_sha256")
    if stated != actual:
        raise StageError(f"record.toml states weights {str(stated)[:12]}, and the release's are {actual[:12]}: the record is for another model; "
                         "rebuild the record where the run's outputs are (`python release/cut.py bundle`)")
    benchmarks = record.get("benchmarks") or {}
    if not isinstance(benchmarks.get("litmus_trained_on"), list):
        raise StageError("record.toml [benchmarks] needs `litmus_trained_on`, the list of benchmarks this model trained on (an empty list if none)")
    found = problems_of_keys(context, documents(context, targets_path(context)), only_present=True)
    if found:
        raise StageError("the documents ask the record for what it does not hold:\n" + "\n".join(found[:40]) +
                         "\nFill these in record.toml (`python release/cut.py record-template` lists every key the documents read) and upload the record again")
    summary = context.work / "data"
    summary.mkdir(exist_ok=True)
    (summary / "record_check.json").write_text(json.dumps({"weights_sha256": actual, "sections": sorted(record)}, indent=1) + "\n", encoding="utf-8", newline="\n")


def record_outputs(context: Context) -> list[Path]:
    return [context.work / "data" / "record_check.json"]


def release_block(available: Documents, previous: Mapping[str, Any]) -> dict[str, Any]:
    """The results' release block: the version and date, and the speeds the measurements gave; the build adds the
    programs' sizes and hashes to its own copy."""
    bench = available["bench"]
    document = bench["document"]
    block = dict(previous)
    block.update({
        "version": available["plan"]["version"],
        "date": available["plan"]["date"],
        "app_tokens_per_second": round(document["text_tokens_per_second"]),
        "speed_document_tokens": document["tokens"],
        "app_document_seconds": round(document["seconds"], 1),
        "speed_machine": bench["machine"]["name"],
    })
    if "browser" in available:
        browser = available["browser"]
        if browser["document_tokens"] != document["tokens"]:
            raise StageError(f"the browser was measured on {browser['document_tokens']} tokens and the program on {document['tokens']}: measure the same document")
        block.update({"browser_tokens_per_second": round(browser["tokens_per_second"]), "browser_document_seconds": round(browser["seconds"], 1)})
    return block


def write_site_data(context: Context, available: Documents) -> list[Path]:
    """`site/assets/results.json`'s release block and `site/assets/links.json`, from the data."""
    assets = context.repository / "site" / "assets"
    results_path = assets / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["release"] = release_block(available, results.get("release", {}))
    write_atomically(results_path, json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    targets = available["target"]
    release = available["release"]
    platforms = targets["programs"]["platforms"]
    links = {
        "schema": 1,
        "training_guide": targets["github"]["training_guide"],
        "contribute_site": targets["contribute"]["site_url"],
        "model_on_hugging_face": f"https://huggingface.co/{targets['hub']['model_repo']}",
        # macOS programs are attached after publishing, so the site links them only once targets.toml lists the platform.
        "download_macos_app": release["download_base"] + targets["programs"]["assets"]["macos_arm64_app"] if "macos" in platforms else "",
        "download_linux_app": release["download_base"] + targets["programs"]["assets"]["linux_app"] if "linux" in platforms else "",
    }
    write_atomically(assets / "links.json", json.dumps(links, indent=1, ensure_ascii=False) + "\n")
    return [results_path, assets / "links.json"]


def fill_site_fallbacks(context: Context, available: Documents) -> list[Path]:
    """Write each `data-value` element's text from its value, so a page read without the script shows the release's numbers.
    An element whose value does not exist yet (the build writes the programs' sizes) keeps its wording, which must hold no digit."""
    changed = []
    site = context.repository / "site"
    site_documents = {"results": available["results"], "links": available["links"]}
    for page in sorted(site.glob("*.html")):
        text = read_exact(page)

        def replace(match: re.Match[str]) -> str:
            key = match.group("key")
            attribute = FORMAT_ATTRIBUTE.search(match.group("open"))
            name = attribute.group(1) if attribute else "text"
            try:
                value = resolve(site_documents, key)
            except MissingValue:
                value = None
            if value in (None, ""):
                if re.search(r"\d", match.group("body")):
                    raise StageError(f"{page.name}: {key} holds a typed number ({match.group('body')!r}) and the data has no value for it")
                return match.group(0)
            return match.group("open") + spell(value, SITE_FORMATS.get(name, "text")) + match.group("close")

        updated = FALLBACK.sub(replace, text)
        if updated != text:
            write_atomically(page, updated)
            changed.append(page)
    return changed


def write_model_sums(context: Context) -> Path:
    """`release/SHA256SUMS`: the model's three files, which the release workflow checks the downloaded model against. The
    programs' hashes belong to the release's own `SHA256SUMS`, since a program cannot be listed in the commit it is built from."""
    names = ("model.safetensors", "config.json", "tokenizer.json")
    lines = [f"{receipts.file_digest(context.work / 'model' / name)}  {name}" for name in names]
    path = context.repository / "release" / "SHA256SUMS"
    write_atomically(path, "\n".join(lines) + "\n")
    return path


def fill_markdown(context: Context, available: Documents) -> list[Path]:
    renderer = values(context, available)
    changed = []
    for path in markdown_files(context.repository):
        text = read_text(path)
        if "<!--litmus:" not in text:
            continue
        filled = slots.fill(text, renderer.render)
        if filled != text:
            write_atomically(path, filled)
        changed.append(path)
    return changed


def unfinished(context: Context) -> list[str]:
    """What stops a publication: a slot not filled, a `[PENDING...]` outside a slot, and a slot whose text is not what the data renders now."""
    available = documents(context, targets_path(context))
    renderer = values(context, available)
    found = []
    for path in markdown_files(context.repository):
        text = read_text(path)
        name = path.relative_to(context.repository).as_posix()
        found += [f"{name}: {item}" for item in slots.unfinished(text)]
        if "<!--litmus:" in text:
            found += [f"{name}: slot {key!r} is stale" for key in slots.stale(text, renderer.render)]
    for page in sorted((context.repository / "site").glob("*.html")):
        found += [f"site/{page.name}: {item}" for item in slots.STRAY.findall(page.read_text(encoding="utf-8"))]
    return found


def fill(context: Context) -> None:
    available = documents(context, targets_path(context))
    written = write_site_data(context, available)
    available = documents(context, targets_path(context))
    written += fill_site_fallbacks(context, available)
    written.append(write_model_sums(context))
    written += fill_markdown(context, documents(context, targets_path(context)))
    left = unfinished(context)
    if left:
        raise StageError("the documents still hold unfinished slots:\n" + "\n".join(left[:60]))
    context.note("filled", sorted(path.relative_to(context.repository).as_posix() for path in written))


def outputs(context: Context) -> list[Path]:
    repository = context.repository
    markdown = [path for path in markdown_files(repository) if "<!--litmus:" in read_exact(path)]
    return [*markdown, repository / "site" / "assets" / "results.json", repository / "site" / "assets" / "links.json", repository / "release" / "SHA256SUMS"]


STAGES = (
    Stage("record", ("fetch",), "check the training record against what the documents read", check_record, record_outputs),
    Stage("docs", ("record", "results", "bench", "browser"), "fill every number of every document and the site's data", fill, outputs),
)
