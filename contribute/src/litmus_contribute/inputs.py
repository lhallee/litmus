"""Strict parsing of what a visitor sends. An error names the field and the rule, never the value sent."""

from __future__ import annotations

import json
import re

from dataclasses import dataclass
from typing import Any

from .config import Settings


ANNOTATIONS = ("ai", "mixed", "human")
CONTRIBUTION_ID = re.compile(r"[0-9a-f]{36}")  # 12 hex digits of milliseconds since the epoch, 24 of randomness


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ScoreRequest:
    text: str
    terms_version: str
    annotation: str | None


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise RequestError(400, "A JSON object repeats a key.")
    return dict(pairs)


def reject_constant(name: str) -> None:
    raise RequestError(400, f"{name} is not JSON.")


def parse_object(body: bytes, allowed: set[str], required: set[str]) -> dict[str, Any]:
    """A JSON object with only `allowed` keys and every `required` one."""
    try:
        decoded = json.loads(body.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError(400, "The body is not UTF-8 JSON.") from error
    if not isinstance(decoded, dict):
        raise RequestError(400, "The body must be a JSON object.")
    unknown, missing = set(decoded) - allowed, required - set(decoded)
    if unknown or missing:
        raise RequestError(400, f"Unknown fields {sorted(unknown)}, missing fields {sorted(missing)}.")
    return decoded


def parse_text(value: Any, settings: Settings) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(400, "text must be a string that is not blank.")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:  # a lone surrogate escaped in the JSON
        raise RequestError(400, "text is not valid Unicode.") from error
    if size > settings.text_max_bytes:
        raise RequestError(413, f"text is over {settings.text_max_bytes} bytes.")
    return value


def parse_annotation_value(value: Any) -> str | None:
    if value is not None and value not in ANNOTATIONS:
        raise RequestError(400, f"annotation must be one of {list(ANNOTATIONS)} or null.")
    return value


def parse_score_request(body: bytes, settings: Settings, terms_version: str) -> ScoreRequest:
    """`{"text", "terms_version", "consent": true, "annotation"?}`: the terms version must be the one served now, and
    `consent` the JSON value `true`, sent only when the visitor has ticked the box agreeing to that version."""
    fields = parse_object(body, allowed={"text", "terms_version", "consent", "annotation"},
                          required={"text", "terms_version", "consent"})
    if fields["terms_version"] != terms_version:
        raise RequestError(409, "The terms have changed since the page loaded: reload it to see the current ones.")
    if fields["consent"] is not True:
        raise RequestError(400, "consent must be true: tick the box agreeing to the Terms of Use and the Privacy Notice.")
    return ScoreRequest(text=parse_text(fields["text"], settings), terms_version=terms_version,
                        annotation=parse_annotation_value(fields.get("annotation")))


def parse_annotation(body: bytes) -> str | None:
    """`{"annotation": "ai" | "mixed" | "human" | null}`; null clears it."""
    return parse_annotation_value(parse_object(body, allowed={"annotation"}, required={"annotation"})["annotation"])


def parse_removal(body: bytes, settings: Settings) -> str:
    """`{"text"}`: the exact text whose stored copies are to be deleted."""
    return parse_text(parse_object(body, allowed={"text"}, required={"text"})["text"], settings)


def contribution_time_ms(contribution_id: str) -> int:
    """When a contribution was stored, read from its id."""
    if not CONTRIBUTION_ID.fullmatch(contribution_id):
        raise RequestError(404, "No such contribution.")
    return int(contribution_id[:12], 16)
