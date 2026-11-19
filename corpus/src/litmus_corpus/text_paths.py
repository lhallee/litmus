"""The catalog's column expressions: where in a row the text, the generator or the label sits.

`sources.csv` names columns with a small path language:

    text                                    a column
    human_answers[]                         each element of a list column
    generated_message.content               a field of a struct column
    messages[].content (role=assistant)     a field of each element whose `role` is `assistant`
    conversations[].info.answer_content     a nested field of each element
    conversation_a[].content[].text (role=assistant)
    messages (JSON)                         a column holding JSON text, parsed before the path is followed
    response_a | response_b                 several expressions, each its own text (` | ` between them)
    title + selftext                        one text, its parts' texts joined by a blank line
    llama-3.1-70b_edits                     a dot before a digit is part of a name, not a field

A generator expression is paired with the text expressions by position when it has as many alternatives or
more (`model_a | model_b`); one that starts with the same list as its text expression (`completions[].model`
beside `completions[].response`) is read from the same element. `column name` takes the text column's own name.

An expression this grammar cannot read (`see card`, `<model>.content`, `row_json`) raises `PathError`, and the
source is skipped with that reason.
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass


ALTERNATIVE_SEPARATOR = re.compile(r"\s+\|\s+")
CONCATENATION = re.compile(r"\s+\+\s+")
JOINER = "\n\n"  # between the parts of an `a + b` text, as between a title and its body
FILTER_PATTERN = re.compile(r"\s*\((\w+)=([^)]+)\)\s*$")
JSON_PATTERN = re.compile(r"\s*\(JSON(?: string)?\)\s*$", re.IGNORECASE)
STEP_PATTERN = re.compile(r"^[A-Za-z_][\w\-. ]*(?:\[\])?$")
# A dot before a letter steps into a field; a dot before a digit is part of a name (`llama-3.1-70b_edits`).
FIELD_SEPARATOR = re.compile(r"\.(?=[A-Za-z_])")
COLUMN_NAME = "column name"
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking", "think_content", "reasoning_text")


class PathError(ValueError):
    """A column expression the reader cannot follow."""


@dataclass(frozen=True)
class Step:
    field: str
    each: bool  # the field holds a list, and every element is followed


@dataclass(frozen=True)
class TextPath:
    expression: str
    steps: tuple[Step, ...]
    keep: tuple[str, str] | None  # (key, value) an element of the first list must hold
    is_json: bool
    joined: tuple[TextPath, ...] = ()  # for `a + b`: the paths whose texts are joined into one text

    @property
    def column(self) -> str:
        return self.steps[0].field

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column the path reads."""
        return tuple(dict.fromkeys(part.column for part in self.joined)) if self.joined else (self.column,)

    @property
    def is_plain(self) -> bool:
        """A column of text, readable without leaving Arrow."""
        return len(self.steps) == 1 and not self.steps[0].each and not self.is_json and self.keep is None and not self.joined

    def list_prefix(self) -> tuple[Step, ...]:
        """The steps up to and including the first list, which a paired generator path may share."""
        if self.joined:
            return ()
        for index, step in enumerate(self.steps):
            if step.each:
                return self.steps[: index + 1]

        return ()


def parse_path(expression: str) -> TextPath:
    text = expression.strip()
    if CONCATENATION.search(text):
        parts = tuple(parse_path(part) for part in CONCATENATION.split(text))
        return TextPath(expression, parts[0].steps, None, False, parts)
    if not text or "+" in text or "<" in text or text.lower().startswith(("see ", "row_json")):
        raise PathError(f"cannot read the column expression {expression!r}")
    is_json = bool(JSON_PATTERN.search(text))
    text = JSON_PATTERN.sub("", text)
    keep_match = FILTER_PATTERN.search(text)
    keep = (keep_match.group(1), keep_match.group(2).strip()) if keep_match else None
    text = FILTER_PATTERN.sub("", text).strip()
    parts = FIELD_SEPARATOR.split(text)
    if not all(STEP_PATTERN.match(part) for part in parts):
        raise PathError(f"cannot read the column expression {expression!r}")
    steps = tuple(Step(part.removesuffix("[]"), part.endswith("[]")) for part in parts)

    return TextPath(expression, steps, keep, is_json)


def parse_alternatives(expression: str) -> list[TextPath]:
    return [parse_path(part) for part in ALTERNATIVE_SEPARATOR.split(expression.strip()) if part.strip()]


@dataclass(frozen=True)
class Found:
    """A text found in a row: the text, the element it came from (for a paired generator), and its prompt."""

    text: str
    element: object  # the element of the first list, or the row's column value
    prompt: str | None


def with_reasoning(text: str, holder: object) -> str:
    """The text with its reasoning block in front, when the dict that holds it keeps reasoning beside it."""
    if not isinstance(holder, dict):
        return text
    for key in REASONING_KEYS:
        reasoning = holder.get(key)
        if isinstance(reasoning, str) and reasoning.strip() and reasoning.strip() not in text:
            return f"<think>\n{reasoning.strip()}\n</think>\n\n{text}"

    return text


def follow(value: object, steps: tuple[Step, ...], holder: object) -> list[tuple[str, object]]:
    """Every string at the end of `steps` from `value`, each with the dict that held it."""
    if not steps:
        return [(value, holder)] if isinstance(value, str) else []
    step, rest = steps[0], steps[1:]
    if not isinstance(value, dict):
        return []
    inner = value.get(step.field)
    if step.each:
        if not isinstance(inner, list):
            return []
        return [found for element in inner for found in follow_element(element, rest, value)]

    return follow(inner, rest, value)


def follow_element(element: object, rest: tuple[Step, ...], holder: object) -> list[tuple[str, object]]:
    if not rest:
        return [(element, holder)] if isinstance(element, str) else []

    return follow(element, rest, element)


def parsed_column(value: object, path: TextPath) -> object:
    if path.is_json and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    return value


def texts_at(row: dict[str, object], path: TextPath) -> list[Found]:
    """The texts a path finds in one row, in order; an `a + b` path finds one text, its parts' texts joined."""
    if path.joined:
        pieces = [found.text for part in path.joined for found in texts_at(row, part) if found.text.strip()]
        return [Found(JOINER.join(pieces), row, None)] if pieces else []
    value = parsed_column(row.get(path.column), path)
    first, rest = path.steps[0], path.steps[1:]
    if not first.each:
        return [Found(with_reasoning(text, holder), value, None) for text, holder in follow({first.field: value}, path.steps, row)]
    if not isinstance(value, list):
        return []
    found: list[Found] = []
    prompt: str | None = None
    for element in value:
        texts = follow_element(element, rest, element)
        if path.keep is not None and isinstance(element, dict) and str(element.get(path.keep[0])) != path.keep[1]:
            prompt = texts[0][0] if texts else prompt
            continue
        found.extend(Found(with_reasoning(text, holder), element, prompt) for text, holder in texts)

    return found


def generator_at(row: dict[str, object], found: Found, text_path: TextPath, generator_path: TextPath | None) -> str | None:
    """The generator of one found text: from the same element when the paths share their list, else from the row."""
    if generator_path is None:
        return None
    shared = text_path.list_prefix()
    if shared and generator_path.steps[: len(shared)] == shared:
        values = follow_element(found.element, generator_path.steps[len(shared) :], found.element)
    else:
        values = follow({generator_path.column: parsed_column(row.get(generator_path.column), generator_path)}, generator_path.steps, row)
        if not values and not generator_path.steps[-1].each:
            raw = row.get(generator_path.column)
            values = [(str(raw), row)] if raw is not None and not isinstance(raw, (dict, list)) else []

    return values[0][0].strip() if values and values[0][0].strip() else None
