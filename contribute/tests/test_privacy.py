"""Nothing identifying reaches storage or the logs, and the rate limiter keeps only salted hashes in memory."""

from __future__ import annotations

import logging
import pytest

from pathlib import Path
from typing import Any
from litmus_contribute.ratelimit import RateLimiter, Rule
from litmus_contribute.shards import list_shards


SENTINELS = {
    "user-agent": "UA-SENTINEL-4242",
    "cookie": "session=COOKIE-SENTINEL-4242",
    "x-forwarded-for": "203.0.113.77",
    "x-real-ip": "198.51.100.23",
    "referer": "https://referrer-sentinel.example/page",
}


def stored_bytes(root: Path) -> str:
    return "".join(path.read_text(encoding="utf-8") for path in root.rglob("*") if path.is_file())


def test_no_identifying_field_in_storage_or_logs(make_site: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    site = make_site()
    version = site.legal.version
    contribution_id = site.client.post("/api/score", json={"text": "Private-free text.", "terms_version": version,
                                                           "consent": True}, headers=SENTINELS).json()["id"]
    site.client.post(f"/api/contributions/{contribution_id}/annotation", json={"annotation": "human"}, headers=SENTINELS)
    site.client.post("/api/removals", json={"text": "Some other text."}, headers=SENTINELS)
    # error paths: a stale terms version, and no consent
    site.client.post("/api/score", json={"text": "x", "terms_version": "stale", "consent": True}, headers=SENTINELS)
    site.client.post("/api/score", json={"text": "x", "terms_version": version, "consent": False}, headers=SENTINELS)
    site.client.__exit__(None, None, None)  # shutdown: the final commit and its log line
    site.client.__enter__()

    assert list_shards(site.root)
    stored = stored_bytes(site.root)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for value in [*SENTINELS.values(), "testclient"]:
        assert value not in stored, value
        assert value not in logged, value
    assert "set-cookie" not in {name.lower() for name in site.client.get("/").headers}


def test_rate_limiter_holds_only_salted_hashes() -> None:
    now = [0.0]
    limiter = RateLimiter({"scores": Rule(burst=2, per_second=1.0)}, max_keys=3, clock=lambda: now[0])
    address = "203.0.113.77"
    assert [limiter.allow("scores", address) for _ in range(3)] == [True, True, False]
    now[0] += 1.0
    assert limiter.allow("scores", address)
    for _, key in limiter._buckets:
        assert address.encode() not in key and len(key) == 16
    other = RateLimiter({"scores": Rule(burst=2, per_second=1.0)}, max_keys=3)
    assert other.key(address) != limiter.key(address)  # a fresh salt per container
    for index in range(5):
        limiter.allow("scores", f"198.51.100.{index}")
    assert len(limiter) == 3
