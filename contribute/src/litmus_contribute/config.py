"""Every cap, window and rate of the contribute site, in one frozen record."""

from __future__ import annotations

from dataclasses import dataclass


MIB = 1_048_576


@dataclass(frozen=True)
class Settings:
    # One text. The byte cap is checked before tokenizing, the token cap after: 131,072 tokens is the longest text the
    # site scores, about 512 KiB of English at about 4 bytes a token, so the byte cap only stops oversized bodies.
    text_max_bytes: int = MIB
    text_max_tokens: int = 131_072
    body_max_bytes: int = 4 * MIB  # room for JSON escaping of the text
    small_body_max_bytes: int = 1024  # an annotation
    # The engine's reading of the longest text, serialized: checked as an invariant before it is stored.
    reading_max_bytes: int = 2 * MIB
    # One scoring at a time per one-core container; the rest wait for it or go to another container.
    scorings_per_container: int = 1

    # How long after a text is stored its annotation may be added or changed.
    annotation_window_seconds: int = 1800

    # Shards: one append-only file per container and window, rolled past a size.
    shard_window_seconds: int = 900
    shard_max_bytes: int = 64 * MIB
    commit_interval_seconds: int = 30
    # Compaction leaves shards younger than this alone, so every container has committed them.
    compaction_grace_seconds: int = 900

    # Rate limits per salted address hash; tokens refill continuously up to the burst.
    scores_burst: float = 10.0
    scores_per_second: float = 10.0 / 60.0
    text_bytes_burst: float = 4.0 * MIB
    text_bytes_per_second: float = 4.0 * MIB / 600.0
    annotations_burst: float = 30.0
    annotations_per_second: float = 30.0 / 60.0
    removals_burst: float = 10.0
    removals_per_second: float = 10.0 / 600.0
    # Per container, whatever the address: the most a flood from many addresses can write to the volume.
    container_bytes_burst: float = 64.0 * MIB
    container_bytes_per_second: float = 1024.0 * MIB / 86_400.0
    rate_limit_max_keys: int = 100_000

    # Scoring seconds per UTC day across every container (the usage ledger): past it the site stops scoring until the
    # next day. 12 core-hours is about $0.57 of CPU at Modal's list price.
    daily_scoring_seconds: float = 12 * 3600.0
    usage_flush_seconds: int = 30

    # Origins a browser may post from besides the site's own (a custom domain, if one is ever added).
    extra_origins: tuple[str, ...] = ()
