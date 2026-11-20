"""What fills a slot: a value from a data document, a sentence or table from `release/templates.toml`, or a table
the code lays out from the results.

A key is one of:

- `document.path as format`: the value at a path (`datapath.py`), such as `config.model.width` or
  `bench.rows[tokens=8192].text_tokens_per_second|int`.
- `t:name`: the template `name` of `templates.toml`. A template is a string with `{path as format}` fields, or a table
  `{if = "path", then = "...", else = "..."}` that picks its text by whether a path holds a value.
- `table.name as argument`: a table whose rows come from the data (`TABLES`).

A missing document, path, row, or format fails; nothing falls back to a default.
"""

from __future__ import annotations

import re
import tomllib

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .datapath import MissingValue, resolve, spell, split_key


FIELD = re.compile(r"\{([^{}]+)\}")
NOT_APPLICABLE = "n/a"
# The results name Litmus's own held-out split as a benchmark of its own (`benchmarks[].id`).
OWN_SPLITS = ("test", "validation")


def holds(documents: Mapping[str, Any], path: str) -> bool:
    """Whether a path leads to a value that is not empty."""
    try:
        value = resolve(documents, path)
    except MissingValue:
        return False
    return value is not None and value != "" and value != []


def machine_name(machine: str) -> str:
    """A cost row's machine without its operating system: `Arm Neoverse-V2 / Linux` is `Arm Neoverse-V2`."""
    return machine.rsplit(" / ", 1)[0]


def comparison_counts(results: Mapping[str, Any]) -> dict[str, int | float | str]:
    """How many public two-class benchmarks the comparison scored and how many detectors beside Litmus read them, and the
    standings a sentence would otherwise state by hand, so a new release cannot leave one stale: how many of those
    detectors were measured faster than Litmus's engine on the comparison's core and the fastest of their speeds, and how
    many have a higher mean PR AUC over the benchmarks none of them trained on, and how many such benchmarks there are.
    The processors the speeds were measured on are named too, and `speed_machines_differ` holds a value only when
    Litmus's was not the others' (the baselines are timed once, on the machine of their time)."""
    own = results["model"]["name"]
    rows = [row for row in results["rows"] if row["task"] == "ai_vs_human"]
    benchmarks = [entry["id"] for entry in results["benchmarks"] if entry["kind"] == "two_class" and entry["id"] not in OWN_SPLITS
                  and any(row["benchmark"] == entry["id"] for row in rows)]
    detectors = {row["detector"] for row in rows if row["benchmark"] in benchmarks and row["detector"] != own}
    speeds = {cost["detector"]: cost["tokens_per_second"] for cost in results["costs"] if cost["measured"]}
    own_speed = speeds[f"{own} engine"]
    other_speeds = [speeds[name] for name in detectors if name in speeds]
    assert other_speeds, "no compared detector has a measured speed"
    machines = {cost["detector"]: machine_name(cost["machine"]) for cost in results["costs"] if cost["measured"]}
    own_machine = machines[f"{own} engine"]
    other_machines = sorted({machines[name] for name in detectors if name in machines})
    unseen = {entry["detector"]: entry["pr_auc"] for entry in results["comparison"]["mean_pr_auc_unseen"]}
    return {"benchmarks": len(benchmarks), "detectors": len(detectors),
            "faster_than_litmus": sum(speed > own_speed for speed in other_speeds), "fastest_other_tokens_per_second": max(other_speeds),
            "more_accurate_unseen": sum(value > unseen[own] for name, value in unseen.items() if name != own),
            "unseen_benchmarks": len(results["comparison"]["unseen_benchmarks"]),
            "speed_machine": own_machine, "other_speed_machines": " and ".join(other_machines),
            "speed_machines_differ": "" if other_machines == [own_machine] else "yes"}


BANDS = ("all", "1-128", "129-512", "513-2048", "2049-8192")


def coverage_bands(documents: Mapping[str, Any], argument: str) -> str:
    """Test-split coverage by token band (passages), and for merged chunks of long documents: coverage and mean set size at the
    two levels and the share of human texts called not human, from the student evaluation."""
    rows = documents["evaluation_test"]["rows"]

    def value(evaluation: str, band: str, metric: str, level: str) -> Any:
        found = [row for row in rows if row["evaluation"] == evaluation and str(row["stratum"]) == band and row["metric"] == metric and str(row["level"]) == level]
        return found[0]["value"] if len(found) == 1 else None

    lines = ["| Where | Coverage at 0.05 | Coverage at 0.10 | Mean set size at 0.10 | Human called not human at 0.05 |", "| --- | --- | --- | --- | --- |"]
    for label, evaluation, band in [(f"Test, {name} tokens" if name != "all" else "Test, all", "passages", name) for name in BANDS] + [("Long documents, merged chunks", "long_documents", "all")]:
        cells = [value(evaluation, band, "coverage", "0.05"), value(evaluation, band, "coverage", "0.1"), value(evaluation, band, "mean_set_size", "0.1"),
                 value(evaluation, band, "human_called_not_human", "0.05")]
        if cells[0] is None:
            continue
        lines.append(f"| {label} | " + " | ".join(spell(cell, "f2" if position == 2 else "pct1") if cell is not None else NOT_APPLICABLE for position, cell in enumerate(cells)) + " |")
    if len(lines) == 2:
        raise MissingValue("table.coverage_bands: the test evaluation holds no coverage rows")
    return "\n".join(lines)


def benchmark_matrix(documents: Mapping[str, Any], metric: str) -> str:
    """One row per two-class public benchmark of the results, one column per detector, each cell the metric for AI
    against human (mixed left out); a dagger marks a detector that trained on the benchmark's family."""
    results = documents["results"]
    own = results["model"]["name"]
    rows = [row for row in results["rows"] if row["task"] == "ai_vs_human"]
    scored = {row["benchmark"] for row in rows}
    benchmarks = [entry for entry in results["benchmarks"] if entry["kind"] == "two_class" and entry["id"] in scored and entry["id"] not in OWN_SPLITS]
    detectors = list(dict.fromkeys(row["detector"] for row in rows if row["benchmark"] in {entry["id"] for entry in benchmarks}))
    detectors.sort(key=lambda name: name != own)
    lines = ["| Benchmark | " + " | ".join(detectors) + " |", "| --- | " + " | ".join("---" for _ in detectors) + " |"]
    for entry in benchmarks:
        cells = []
        for detector in detectors:
            found = [row for row in rows if row["benchmark"] == entry["id"] and row["detector"] == detector]
            if len(found) > 1:
                raise ValueError(f"{len(found)} rows for {detector} on {entry['id']}")
            cells.append(NOT_APPLICABLE if not found else spell(found[0][metric], "f3") + (" †" if found[0].get("in_distribution") else ""))
        lines.append(f"| {entry['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLES: dict[str, Callable[[Mapping[str, Any], str], str]] = {"benchmark_matrix": benchmark_matrix, "coverage_bands": coverage_bands}


class Values:
    """Renders slot keys against the data documents and the templates."""

    def __init__(self, documents: Mapping[str, Any], templates: Mapping[str, Any]) -> None:
        self.documents = documents
        self.templates = templates

    @classmethod
    def with_templates(cls, documents: Mapping[str, Any], path: Path) -> Values:
        templates = tomllib.loads(path.read_text(encoding="utf-8"))
        for name, entry in templates.items():
            for text in (entry,) if isinstance(entry, str) else (entry["then"], entry["else"]):
                stripped = FIELD.sub("", text)
                if "{" in stripped or "}" in stripped:
                    raise ValueError(f"{path}: template {name!r} has a brace that is not a {{path as format}} field")
        return cls(documents, templates)

    def interpolate(self, text: str) -> str:
        """The text with each `{path as format}` written out. A line that starts `@if path ` stays only when the path holds a value."""
        lines = []
        for line in text.split("\n"):
            if line.startswith("@if "):
                condition, _, line = line[len("@if "):].partition(" ")
                if not holds(self.documents, condition):
                    continue
            lines.append(FIELD.sub(lambda match: self.field(match.group(1)), line))
        return "\n".join(lines)

    def field(self, key: str) -> str:
        path, spec = split_key(key)
        return spell(resolve(self.documents, path), spec)

    def template(self, name: str) -> str:
        if name not in self.templates:
            raise MissingValue(f"t:{name}: templates.toml has no template named {name!r}")
        entry = self.templates[name]
        if isinstance(entry, str):
            return self.interpolate(entry)
        return self.interpolate(entry["then"] if holds(self.documents, entry["if"]) else entry["else"])

    def render(self, key: str, block: bool) -> str:
        """The text for a slot; an empty text is an error, since a slot that says nothing is a value nobody filled in."""
        name, spec = split_key(key)
        if name.startswith("t:"):
            text = self.template(name[2:]).strip("\n")
        elif name.startswith("table."):
            table = name[len("table."):]
            if table not in TABLES:
                raise MissingValue(f"{key}: no table named {table!r}; the tables are {', '.join(sorted(TABLES))}")
            text = TABLES[table](self.documents, spec)
        else:
            text = spell(resolve(self.documents, name), spec).strip("\n")
        if not text.strip():
            raise MissingValue(f"{key}: the value is empty")
        return text
