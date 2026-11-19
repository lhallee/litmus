"""The repository's own documents and settings, checked against the driver: a slot that names a source nothing makes would only fail on release day."""

from __future__ import annotations

import re
import tomllib

from pathlib import Path
from litmus_release import cli, registry, slots
from litmus_release.data import documents as load
from litmus_release.datapath import parse_path, split_key
from litmus_release.documents import FALLBACK, markdown_files, read_text
from litmus_release.values import FIELD, TABLES, Values, comparison_counts
from tests.conftest import RESULTS


REPOSITORY = Path(__file__).resolve().parents[2]
# The data documents a slot may name: the driver's own, and the files of the record folder the documents read (release/RELEASING.md).
DOCUMENTS = {"plan", "release", "target", "config", "model", "record", "results", "results_md", "counts", "links", "bench", "browser", "build", "linux_build",
             "evaluation_test", "evaluation_test_full", "evaluation_validation", "teacher_metrics", "teacher_uncertainty"}


def all_slots() -> list[tuple[str, str]]:
    found = []
    for path in markdown_files(REPOSITORY):
        text = read_text(path)
        if "<!--litmus:" in text:
            found += [(path.relative_to(REPOSITORY).as_posix(), slot.key) for slot in slots.find(text)]
    return found


def test_every_slot_names_a_template_a_table_or_a_known_document() -> None:
    templates = tomllib.loads((REPOSITORY / "release" / "templates.toml").read_text(encoding="utf-8"))
    for where, key in all_slots():
        name, _ = split_key(key)
        if name.startswith("t:"):
            assert name[2:] in templates, f"{where}: {key} names no template"
        elif name.startswith("table."):
            assert name[len("table."):] in TABLES, f"{where}: {key} names no table"
        else:
            assert parse_path(name)[0] in DOCUMENTS, f"{where}: {key} names no known document"


def test_every_template_field_names_a_known_document() -> None:
    templates = tomllib.loads((REPOSITORY / "release" / "templates.toml").read_text(encoding="utf-8"))
    Values.with_templates({}, REPOSITORY / "release" / "templates.toml")
    for name, entry in templates.items():
        pieces = [entry] if isinstance(entry, str) else [entry["then"], entry["else"]]
        paths = [split_key(field)[0] for text in pieces for field in FIELD.findall(text)]
        paths += re.findall(r"^@if (\S+) ", "\n".join(pieces), flags=re.MULTILINE)
        paths += [entry["if"]] if isinstance(entry, dict) else []
        for path in paths:
            assert parse_path(path)[0] in DOCUMENTS, f"template {name}: {path} names no known document"


def test_the_driver_lists_its_stages_in_a_working_order() -> None:
    names = [stage.name for stage in registry.build()]
    assert names[:3] == ["fetch", "record", "version"] and names.index("docs") < names.index("build-windows") < names.index("assemble") < names.index("publish")


def test_the_site_reads_only_data_the_driver_writes() -> None:
    written = {"results", "links"}
    for page in sorted((REPOSITORY / "site").glob("*.html")):
        for match in FALLBACK.finditer(page.read_text(encoding="utf-8")):
            assert match.group("key").split(".")[0] in written, f"{page.name}: {match.group('key')}"


def test_the_targets_and_baselines_files_hold_what_the_driver_reads() -> None:
    targets = tomllib.loads((REPOSITORY / "release" / "targets.toml").read_text(encoding="utf-8"))
    assert {"github", "hub", "contribute", "programs", "bench", "citation"} <= targets.keys()
    assert {"windows_app", "windows_cli", "linux_app", "linux_cli", "macos_arm64_app", "macos_arm64_cli", "macos_x86_64_app", "macos_x86_64_cli"} == targets["programs"]["assets"].keys()
    baselines = tomllib.loads((REPOSITORY / "release" / "baselines.toml").read_text(encoding="utf-8"))
    assert baselines["scores"]["benchmarks"] and baselines["costs"]["files"]


def test_the_command_line_parses_every_command() -> None:
    assert cli.parse(["run", "--work", "w", "--model", "o/n@" + "a" * 40, "--version", "1.0.0"]).command == "run"
    assert cli.parse(["publish", "--work", "w", "--dry-run"]).dry_run
    assert cli.parse(["version", "check", "1.0.0"]).action == "check"


def fabricate(keys: list[str]) -> dict:
    """Data documents shaped as the keys read them, with made-up leaves: it proves each key's grammar and format, not a real schema."""
    leaves = {"int": 1234, "f0": 30.0, "f1": 1.5, "f2": 1.25, "f3": 0.5, "pct0": 0.25, "pct1": 0.25, "mb1": 5_000_000, "m2": 1_790_000, "ms0": 0.1, "": "text"}
    documents: dict[str, dict] = {}
    for key in keys:
        name, spec = split_key(key)
        steps = parse_path(name)
        leaf = leaves.get(spec, "text")
        node: dict | list = documents.setdefault(str(steps[0]), {})
        for position, step in enumerate(steps[1:], start=1):
            last = position == len(steps) - 1
            following = steps[position + 1] if not last else None
            container: dict | list = [] if isinstance(following, (int, dict)) else {}
            if isinstance(step, str):
                assert isinstance(node, dict), key
                if last:
                    node.setdefault(step, leaf)
                else:
                    if not isinstance(node.get(step), (dict, list)):
                        node[step] = container
                    node = node[step]
            elif isinstance(step, int):
                assert isinstance(node, list), key
                while len(node) <= step:
                    node.append({} if not last else leaf)
                node = node[step] if not last else node
            else:
                assert isinstance(node, list), key
                row = next((item for item in node if all(str(item.get(field)) == value for field, value in step.items())), None)
                if row is None:
                    row = dict(step)
                    node.append(row)
                node = row
    return documents


def test_every_slot_of_the_repository_renders_from_data_shaped_as_its_key_reads_it(context) -> None:
    keys = [key for _, key in all_slots() if not key.startswith(("t:", "table."))]
    templates = tomllib.loads((REPOSITORY / "release" / "templates.toml").read_text(encoding="utf-8"))
    for entry in templates.values():
        pieces = [entry] if isinstance(entry, str) else [entry["then"], entry["else"]]
        keys += [field for text in pieces for field in FIELD.findall(text)]
        keys += re.findall(r"^@if (\S+) ", "\n".join(pieces), flags=re.MULTILINE)
    made = fabricate(keys)
    available = load(context, REPOSITORY / "release" / "targets.toml")
    merged = {name: available[name] for name in available if name not in ("results", "counts")}
    results = dict(RESULTS)
    for name in ("rows", "costs"):
        results[name] = [dict(row) for row in results[name]]
        for row in made.get("results", {}).get(name, []):
            selector = {field: value for field, value in row.items() if isinstance(value, str) and value != "text"}
            held = next((item for item in results[name] if all(str(item.get(field)) == value for field, value in selector.items())), None)
            if held is None:
                results[name].append(row)
            else:
                held.update({field: value for field, value in row.items() if field not in held})
    results["release"] = {**results["release"], **made.get("results", {}).get("release", {})}
    results["texts_per_benchmark"] = 3000
    merged["results"] = results
    merged["counts"] = comparison_counts(results)
    merged["results_md"] = {"benchmarks": "table", "flag_rates": "table", "throughput": "table"}
    for name in ("record", "bench", "browser", "evaluation_test", "teacher_metrics", "teacher_uncertainty"):
        merged[name] = made.get(name, {})
    merged["bench"] = {**merged["bench"], "machine": {"name": "n", "processor": "p", "system": "s", "backend": "avx2"}}
    renderer = Values.with_templates(merged, REPOSITORY / "release" / "templates.toml")
    failures = []
    for where, key in all_slots():
        try:
            assert renderer.render(key, key.startswith(("t:table", "table."))).strip()
        except Exception as error:  # noqa: BLE001 - every kind of failure is reported with its slot
            failures.append(f"{where}: {key}: {error}")
    assert not failures, "\n".join(failures[:10])
