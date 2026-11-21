"""Read a value from nested data by a path, and write it as text.

A path is names joined by dots, with list steps in brackets: `config.bank.counts[0]` takes an index, and
`results.rows[benchmark=test,detector=Litmus,task=ai_vs_human].pr_auc` takes the one row of a list whose fields equal
the selector's. A selector that matches no row or several rows is an error, never a guess. A name may hold any
character except `.`, `[` and `]`, so `val/pr_auc` is one name. After the path, ` as spec` names a format.
"""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from typing import Any


class MissingValue(LookupError):
    """A path that data does not hold, with what is there instead."""


def parse_path(path: str) -> list[str | int | dict[str, str]]:
    """Split a path into names, indexes, and selectors."""
    steps: list[str | int | dict[str, str]] = []
    position = 0
    while position < len(path):
        if path[position] == ".":
            position += 1
        elif path[position] == "[":
            end = path.find("]", position)
            if end < 0:
                raise ValueError(f"{path!r}: a bracket is not closed")
            inner = path[position + 1:end]
            if inner.isdigit():
                steps.append(int(inner))
            else:
                pairs = [pair.split("=", 1) for pair in inner.split(",")]
                if any(len(pair) != 2 for pair in pairs):
                    raise ValueError(f"{path!r}: a selector reads [field=value,...], not [{inner}]")
                steps.append({field.strip(): value.strip() for field, value in pairs})
            position = end + 1
        else:
            stops = [index for index in (path.find(".", position), path.find("[", position)) if index >= 0]
            end = min(stops) if stops else len(path)
            steps.append(path[position:end])
            position = end
    if not steps:
        raise ValueError("an empty path")
    return steps


def split_key(key: str) -> tuple[str, str]:
    """A slot key as (path, format): the format follows the last ` as `. A bar is no separator because a Markdown table
    reads it as a cell edge, and the key sits in the cell."""
    head, separator, tail = key.rpartition(" as ")
    if separator and not any(character in tail for character in "[]=,"):
        return head.strip(), tail.strip()
    return key.strip(), ""


def same(cell: object, wanted: str) -> bool:
    """Whether a row's field equals a selector's text; numbers compare by their usual spelling."""
    return str(cell) == wanted


def resolve(documents: Mapping[str, Any], path: str) -> Any:
    """The value at `path`; the first name is a document."""
    steps = parse_path(path)
    first = steps[0]
    if not isinstance(first, str) or first not in documents:
        known = ", ".join(sorted(documents)) or "none"
        raise MissingValue(f"{path}: no data document named {first!r}; the documents are {known}")
    value: Any = documents[first]
    walked = str(first)
    for step in steps[1:]:
        if isinstance(step, str):
            if not isinstance(value, Mapping) or step not in value:
                held = ", ".join(sorted(map(str, value)))[:200] if isinstance(value, Mapping) else type(value).__name__
                raise MissingValue(f"{path}: {walked} has no {step!r} (it holds {held})")
            value = value[step]
            walked += f".{step}"
        elif isinstance(step, int):
            if not isinstance(value, Sequence) or isinstance(value, str) or step >= len(value):
                raise MissingValue(f"{path}: {walked} has no item {step}")
            value = value[step]
            walked += f"[{step}]"
        else:
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise MissingValue(f"{path}: {walked} is not a list, so it cannot be selected from")
            rows = [row for row in value if isinstance(row, Mapping) and all(field in row and same(row[field], wanted) for field, wanted in step.items())]
            if len(rows) != 1:
                raise MissingValue(f"{path}: {len(rows)} rows of {walked} match {step}, not one")
            value = rows[0]
            walked += f"[{','.join(f'{field}={wanted}' for field, wanted in step.items())}]"
    return value


def oxford(items: Sequence[object]) -> str:
    """A list as prose with an Oxford comma."""
    words = [str(item) for item in items]
    if len(words) <= 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + f", and {words[-1]}"


def render_table(table: Mapping[str, Any]) -> str:
    """A Markdown table from `{"columns": [...], "rows": [[...], ...]}`."""
    columns = [str(column) for column in table["columns"]]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table["rows"]:
        if len(row) != len(columns):
            raise ValueError(f"a table row has {len(row)} cells for {len(columns)} columns: {row}")
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def spell(value: Any, spec: str = "") -> str:
    """A value as text in the format `spec`.

    `int` groups thousands; `sig2` rounds to two significant figures and groups thousands, for an approximate count
    (28,267 as 28,000); `f1` to `f4` fix the decimals; `pct0` to `pct2` write a fraction as a percentage with the
    percent sign; `mb1` writes bytes as megabytes; `ms0` writes seconds as milliseconds; `m2` writes a count in millions; `hours1` writes seconds as hours;
    `odds1` writes a distance in log odds as the number of times it multiplies the odds (1.4319 as 4.2); `list` joins with an Oxford comma; `text` takes a string as it is. A float with no format, a missing value, and a
    format that does not fit the value are errors.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"cannot write {value!r} as text")
    if spec == "text" or (not spec and isinstance(value, str)):
        return str(value)
    if spec == "list" or (not spec and isinstance(value, Sequence) and not isinstance(value, str)):
        return oxford(value)
    if isinstance(value, Mapping) and not spec and {"columns", "rows"} <= value.keys():
        return render_table(value)
    if isinstance(value, bool):
        raise ValueError(f"a boolean has no default text; name a format ({value})")
    if isinstance(value, str):
        value = float(value)
    if isinstance(value, (Mapping, Sequence)):
        raise ValueError(f"{value!r} needs a format")
    if not spec:
        if isinstance(value, int):
            return f"{value:,}"
        raise ValueError(f"the float {value} needs a format such as f3 or pct1")
    if spec == "int":
        return f"{round(value):,}"
    if spec == "sig2":
        return f"{round(float(f'{value:.2g}')):,}"
    if spec.startswith("f") and spec[1:].isdigit():
        return f"{value:.{int(spec[1:])}f}"
    if spec.startswith("pct") and spec[3:].isdigit():
        return f"{100 * value:.{int(spec[3:])}f}%"
    if spec == "ms0":
        return f"{value * 1000:,.0f}"
    if spec == "mb1":
        return f"{value / 1e6:.1f} MB"
    if spec == "m2":
        return f"{value / 1e6:.2f}M"
    if spec == "hours1":
        return f"{value / 3600:.1f}"
    if spec == "odds1":
        return f"{math.exp(value):.1f}"
    raise ValueError(f"no format named {spec!r}")
