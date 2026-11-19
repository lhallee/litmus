"""Slots: the values and tables of a published document that come from data, never from a hand.

A slot is an HTML comment pair around its current text, which Markdown does not show:

    the model is <!--litmus:model.config_name-->s192_l4<!--/litmus-->, a ternary encoder

An inline slot sits inside a line. A block slot has both comments at the start of a line, and holds a table or a
paragraph:

    <!--litmus:table.benchmark_matrix|roc_auc-->
    | Benchmark | Litmus |
    <!--/litmus-->

The key names what fills it (`values.py`). Filling replaces the text between the comments and keeps the comments, so a
later release fills the same slot again; text outside the comments is the author's and is never touched. A slot that
has never been filled holds `[PENDING]`.
"""

from __future__ import annotations

import re

from collections.abc import Callable
from dataclasses import dataclass


PENDING = "[PENDING]"
SLOT = re.compile(r"<!--litmus:(?P<key>[^>]*?)-->(?P<body>.*?)<!--/litmus-->", re.DOTALL)
# Any other `[PENDING` is a slot nobody converted; it is as unfinished as an empty one.
STRAY = re.compile(r"\[PENDING\b[^\]]*\]")
OPEN_WITHOUT_CLOSE = re.compile(r"<!--litmus:[^>]*-->")


@dataclass(frozen=True)
class Slot:
    """One slot in a text: its key, where its body starts and ends, and whether it is a block."""

    key: str
    start: int
    end: int
    block: bool


def find(text: str) -> list[Slot]:
    """Every slot of `text` in order; a slot that opens and never closes, or opens inside another, is an error."""
    slots = []
    for match in SLOT.finditer(text):
        if "<!--litmus:" in match.group("body"):
            raise ValueError(f"a slot inside the slot {match.group('key')!r}")
        opens = match.start() == 0 or text[match.start() - 1] == "\n"
        closes = match.end("body") > match.start("body") and text[match.end("body") - 1] == "\n"
        slots.append(Slot(key=match.group("key").strip(), start=match.start("body"), end=match.end("body"), block=opens and closes))
    if len(OPEN_WITHOUT_CLOSE.findall(text)) != len(slots):
        raise ValueError("a slot opens with <!--litmus:...--> and never closes with <!--/litmus-->")
    return slots


def fill(text: str, render: Callable[[str, bool], str]) -> str:
    """`text` with every slot's body replaced by `render(key, block)`; an inline body may not span lines."""
    pieces: list[str] = []
    last = 0
    for slot in find(text):
        body = render(slot.key, slot.block)
        if slot.block:
            body = "\n" + body.strip("\n") + "\n"
        elif "\n" in body:
            raise ValueError(f"the inline slot {slot.key!r} would write several lines; put its comments at the start of lines")
        pieces += [text[last:slot.start], body]
        last = slot.end
    pieces.append(text[last:])
    return "".join(pieces)


def unfinished(text: str) -> list[str]:
    """What is left to do in `text`: slots still holding `[PENDING]`, and `[PENDING...]` written outside a slot."""
    found = [f"slot {slot.key!r} is not filled" for slot in find(text) if text[slot.start:slot.end].strip() == PENDING]
    inside = [(slot.start, slot.end) for slot in find(text)]
    for match in STRAY.finditer(text):
        if not any(start <= match.start() < end for start, end in inside):
            found.append(f"{match.group(0)[:80]!r} is not a slot")
    return found


def stale(text: str, render: Callable[[str, bool], str]) -> list[str]:
    """The keys of slots whose text is not what `render` gives now."""
    drifted = []
    for slot in find(text):
        body = render(slot.key, slot.block)
        current = text[slot.start:slot.end]
        wanted = "\n" + body.strip("\n") + "\n" if slot.block else body
        if current != wanted:
            drifted.append(slot.key)
    return drifted


def keys(text: str) -> list[str]:
    return [slot.key for slot in find(text)]
