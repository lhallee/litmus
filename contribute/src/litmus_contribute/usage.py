"""A daily budget of scoring seconds shared by every container, so a flood from many addresses cannot run up the bill.

Each container counts its own scoring seconds and, every flush, writes them under its own key (`<UTC day>:<writer>`)
of a shared table (a Modal Dict on Modal), then sums the day's keys. No key is written by two containers, so no
update is lost; the sum lags by at most one flush interval per container.
"""

from __future__ import annotations

import threading
import time

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Protocol

from .shards import ContainerName


class SharedTable(Protocol):
    def put(self, key: str, value: float) -> object: ...

    def items(self) -> Iterable[tuple[str, float]]: ...


class MemoryTable:
    """A process-local table: the tests' stand-in for a Modal Dict."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def put(self, key: str, value: float) -> bool:
        self.values[key] = value
        return True

    def items(self) -> list[tuple[str, float]]:
        return list(self.values.items())


def utc_day(now: float) -> str:
    return datetime.fromtimestamp(now, tz=UTC).strftime("%Y%m%d")


class UsageLedger:
    def __init__(self, table: SharedTable, name: ContainerName, daily_seconds: float,
                 clock: Callable[[], float] = time.time) -> None:
        self.table = table
        self.name = name
        self.daily_seconds = daily_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._day = utc_day(clock())
        self._own = 0.0  # this container's scoring seconds today
        self._others = 0.0  # every other container's, as of the last flush

    def _roll(self) -> None:
        day = utc_day(self.clock())
        if day != self._day:
            self._day, self._own, self._others = day, 0.0, 0.0

    def add(self, seconds: float) -> None:
        with self._lock:
            self._roll()
            self._own += seconds

    def exhausted(self) -> bool:
        with self._lock:
            self._roll()
            return self._own + self._others >= self.daily_seconds

    def flush(self) -> float:
        """Publish this container's seconds and read everyone's; returns the day's total."""
        with self._lock:
            self._roll()
            day, own = self._day, self._own
        own_key = f"{day}:{self.name.value}"
        self.table.put(own_key, own)
        others = sum(value for key, value in self.table.items() if key.startswith(f"{day}:") and key != own_key)
        with self._lock:
            if self._day == day:
                self._others = others
            return self._own + self._others
