from __future__ import annotations

import pytest

from litmus_release import slots


TEXT = """before <!--litmus:a.b as int-->[PENDING]<!--/litmus--> after

<!--litmus:table.x-->
[PENDING]
<!--/litmus-->

| row | <!--litmus:c.d-->[PENDING]<!--/litmus--> |
"""


def render(key: str, block: bool) -> str:
    return {"a.b as int": "4,096", "table.x": "| a |\n| - |\n| 1 |", "c.d": "yes"}[key]


def test_inline_and_block_slots_are_found_with_their_kind() -> None:
    found = slots.find(TEXT)
    assert [(slot.key, slot.block) for slot in found] == [("a.b as int", False), ("table.x", True), ("c.d", False)]


def test_fill_changes_only_the_slots_and_keeps_the_comments() -> None:
    filled = slots.fill(TEXT, render)
    assert filled.startswith("before <!--litmus:a.b as int-->4,096<!--/litmus--> after")
    assert "<!--litmus:table.x-->\n| a |\n| - |\n| 1 |\n<!--/litmus-->" in filled
    assert "| row | <!--litmus:c.d-->yes<!--/litmus--> |" in filled
    assert slots.fill(filled, render) == filled
    assert slots.stale(filled, render) == []
    assert slots.unfinished(filled) == []


def test_unfilled_and_stray_pending_are_unfinished() -> None:
    left = slots.unfinished(TEXT + "and [PENDING: a number nobody converted]")
    assert len(left) == 4
    assert any("not a slot" in item for item in left)


def test_a_slot_whose_text_drifted_is_stale() -> None:
    filled = slots.fill(TEXT, render)
    assert slots.stale(filled.replace("4,096", "4,097"), render) == ["a.b as int"]


def test_an_inline_slot_may_not_write_several_lines() -> None:
    with pytest.raises(ValueError, match="several lines"):
        slots.fill("a <!--litmus:k-->[PENDING]<!--/litmus--> b", lambda key, block: "one\ntwo")


@pytest.mark.parametrize("text", ["<!--litmus:k-->x", "<!--litmus:a--><!--litmus:b-->x<!--/litmus--><!--/litmus-->"])
def test_unclosed_and_nested_slots_are_errors(text: str) -> None:
    with pytest.raises(ValueError):
        slots.find(text)
