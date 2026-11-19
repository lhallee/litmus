"""In-memory rate limits keyed by a salted hash of the address. The salt and the table live only in the container's
memory: nothing here is written, logged or returned, and both die with the container.

On Modal the salt is drawn after a memory snapshot is restored (never inside it), so restored containers do not share
one salt.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    burst: float
    per_second: float


@dataclass
class Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token buckets per (rule, address hash), the least recently used dropped past `max_keys`."""

    def __init__(self, rules: dict[str, Rule], max_keys: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rules = rules
        self.max_keys = max_keys
        self.clock = clock
        self._salt: bytes | None = None  # drawn at the first request, after any snapshot restore
        self._buckets: OrderedDict[tuple[str, bytes], Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def key(self, address: str) -> bytes:
        if self._salt is None:
            self._salt = secrets.token_bytes(32)
        return hmac.new(self._salt, address.encode("utf-8"), hashlib.sha256).digest()[:16]

    def allow(self, rule_name: str, address: str, cost: float = 1.0) -> bool:
        """Spend `cost` tokens of `rule_name` for `address` if it has them."""
        rule = self.rules[rule_name]
        now = self.clock()
        with self._lock:
            slot = (rule_name, self.key(address))
            bucket = self._buckets.pop(slot, None) or Bucket(tokens=rule.burst, updated=now)
            bucket.tokens = min(rule.burst, bucket.tokens + (now - bucket.updated) * rule.per_second)
            bucket.updated = now
            allowed = bucket.tokens >= cost
            if allowed:
                bucket.tokens -= cost
            self._buckets[slot] = bucket
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        return allowed

    def __len__(self) -> int:
        return len(self._buckets)
