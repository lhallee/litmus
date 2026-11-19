"""The daily scoring budget shared by containers through one table, each writing only its own key."""

from __future__ import annotations

from litmus_contribute.shards import ContainerName
from litmus_contribute.usage import MemoryTable, UsageLedger

from conftest import Clock


def test_budget_is_shared_and_resets_each_day() -> None:
    table, clock = MemoryTable(), Clock()
    first = UsageLedger(table, ContainerName("one"), daily_seconds=100.0, clock=clock)
    second = UsageLedger(table, ContainerName("two"), daily_seconds=100.0, clock=clock)
    first.add(60.0)
    second.add(30.0)
    assert not first.exhausted()
    assert first.flush() == 60.0  # the other container has not published yet
    assert second.flush() == 90.0
    second.add(15.0)
    assert second.exhausted()
    assert first.flush() == 90.0 and not first.exhausted()
    assert len(table.values) == 2  # one key per container per day

    clock.advance(86_400)
    assert not second.exhausted()
    assert second.flush() == 0.0
