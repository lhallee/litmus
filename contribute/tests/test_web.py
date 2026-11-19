"""The ASGI app through a test client, against a local directory standing in for the volume."""

from __future__ import annotations

import gzip
import json

from pathlib import Path
from typing import Any
from litmus_contribute.config import Settings
from litmus_contribute.legal import as_served, read_document
from litmus_contribute.shards import list_shards

from conftest import LEGAL, PRIVATE_PAGE


def records(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for shard in list_shards(root)
            for line in (root / shard.path).read_text(encoding="utf-8").splitlines()]


def score(site: Any, text: str = "A short text to check.", **extra: Any) -> Any:
    return site.client.post("/api/score", json={"text": text, "terms_version": site.legal.version, "consent": True, **extra})


def test_page_and_assets(make_site: Any) -> None:
    site = make_site()
    page = site.client.get("/", headers={"accept-encoding": "gzip"})
    assert page.status_code == 200
    assert page.headers["content-type"] == "text/html; charset=utf-8"
    assert page.headers["content-encoding"] == "gzip"
    assert page.headers["cache-control"] == "no-cache"
    assert "connect-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
    assert "Placeholder page" in page.text
    assert "access-control-allow-origin" not in page.headers

    wasm = site.client.get("/assets/litmus-app-0123456789abcdef_bg.wasm", headers={"accept-encoding": "identity"})
    assert wasm.headers["content-type"] == "application/wasm"
    assert wasm.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert wasm.content.startswith(b"\x00asm")
    assert "content-encoding" not in wasm.headers

    script = site.client.get("/assets/litmus-app-0123456789abcdef.js", headers={"accept-encoding": "gzip;q=0"})
    assert script.headers["content-type"] == "text/javascript; charset=utf-8"
    assert "content-encoding" not in script.headers

    again = site.client.get("/", headers={"if-none-match": page.headers["etag"]})
    assert again.status_code == 304

    head = site.client.head("/style.css")
    assert head.status_code == 200 and head.content == b"" and int(head.headers["content-length"]) > 0

    for missing in ("/nope.html", "/../pyproject.toml", "/assets/../../conftest.py"):
        assert site.client.get(missing).status_code == 404


def test_gzip_body_is_the_file(make_site: Any, placeholder_site: Path) -> None:
    site = make_site()
    served = site.client.get("/style.css", headers={"accept-encoding": "gzip"})
    assert served.headers["content-encoding"] == "gzip"
    assert served.content == (placeholder_site / "style.css").read_bytes()  # httpx decoded it
    assert len(gzip.compress(served.content)) < len(served.content)


def test_score_stores_one_row(make_site: Any) -> None:
    site = make_site()
    response = score(site, "Some words to score here.", annotation="ai")
    assert response.status_code == 200
    body = response.json()
    assert body["reading"]["n_tokens"] == 5
    assert body["model_version"] == "fake-model-1" and body["terms_version"] == site.legal.version
    assert body["annotation"] == "ai" and body["annotation_window_seconds"] == 1800
    assert response.headers["cache-control"] == "no-store"

    [stored] = records(site.root)
    assert set(stored) == {"kind", "id", "text", "content_hash", "reading", "model_version", "annotation", "terms_version",
                           "received_at"}
    assert stored["id"] == body["id"] and stored["text"] == "Some words to score here."
    assert stored["terms_version"] == site.legal.version and stored["received_at"] == int(site.clock() * 1000)


def test_terms_version_is_required_and_current(make_site: Any) -> None:
    site = make_site()
    missing = site.client.post("/api/score", json={"text": "No terms version.", "consent": True})
    stale = site.client.post("/api/score", json={"text": "Old terms.", "terms_version": "2020-01-01-v0", "consent": True})
    assert missing.status_code == 400 and stale.status_code == 409
    assert records(site.root) == [] and site.scorer.calls == []


def test_explicit_consent_is_required(make_site: Any) -> None:
    site = make_site()
    version = site.legal.version
    refused = [site.client.post("/api/score", json={"text": "Not agreed.", "terms_version": version, **consent})
               for consent in ({}, {"consent": False}, {"consent": None}, {"consent": "true"}, {"consent": 1},
                               {"consent": "yes"})]
    assert [response.status_code for response in refused] == [400] * 6
    assert "consent" in refused[1].json()["error"]
    assert records(site.root) == [] and site.scorer.calls == []
    assert score(site, "Agreed.").status_code == 200


def test_validation_and_caps(make_site: Any) -> None:
    site = make_site(Settings(text_max_bytes=64, text_max_tokens=5, body_max_bytes=512, scores_burst=100))
    version = site.legal.version
    post = site.client.post
    assert score(site, "x" * 65).status_code == 413
    assert score(site, "one two three four five six").status_code == 413
    assert score(site, "   ").status_code == 400
    assert post("/api/score", json={"text": "a", "terms_version": version, "consent": True, "extra": 1}).status_code == 400
    assert post("/api/score", json={"text": "a", "terms_version": version, "consent": True,
                                    "annotation": "robot"}).status_code == 400
    assert post("/api/score", content=b'{"text": "a", "text": "b", "terms_version": "%s", "consent": true}' % version.encode(),
                headers={"content-type": "application/json"}).status_code == 400
    assert post("/api/score", content=b'{"text": NaN}', headers={"content-type": "application/json"}).status_code == 400
    assert post("/api/score", content=b'{"text": "\\ud800", "terms_version": "%s", "consent": true}' % version.encode(),
                headers={"content-type": "application/json"}).status_code == 400
    assert post("/api/score", content=b"not json", headers={"content-type": "application/json"}).status_code == 400
    assert post("/api/score", content=b"[]", headers={"content-type": "application/json"}).status_code == 400
    assert post("/api/score", content=b"x" * 600, headers={"content-type": "application/json"}).status_code == 413
    assert post("/api/score", data={"text": "a"}).status_code == 415
    assert records(site.root) == []


def test_same_origin_only(make_site: Any) -> None:
    site = make_site()
    body = {"text": "a b", "terms_version": site.legal.version, "consent": True}
    foreign = site.client.post("/api/score", json=body, headers={"origin": "https://evil.example"})
    cross_site = site.client.post("/api/score", json=body, headers={"sec-fetch-site": "cross-site"})
    own = site.client.post("/api/score", json=body, headers={"origin": "http://testserver", "sec-fetch-site": "same-origin"})
    assert foreign.status_code == 403 and cross_site.status_code == 403 and own.status_code == 200
    assert "access-control-allow-origin" not in own.headers


def test_annotation_within_window(make_site: Any) -> None:
    site = make_site()
    contribution_id = score(site).json()["id"]
    site.clock.advance(60)
    labeled = site.client.post(f"/api/contributions/{contribution_id}/annotation", json={"annotation": "mixed"})
    cleared = site.client.post(f"/api/contributions/{contribution_id}/annotation", json={"annotation": None})
    assert labeled.status_code == 202 and labeled.json() == {"id": contribution_id, "annotation": "mixed"}
    assert cleared.status_code == 202
    events = [record for record in records(site.root) if record["kind"] == "annotation"]
    assert [event["annotation"] for event in events] == ["mixed", None]
    assert events[0]["at"] == int(site.clock() * 1000)

    site.clock.advance(1800)
    late = site.client.post(f"/api/contributions/{contribution_id}/annotation", json={"annotation": "ai"})
    assert late.status_code == 410
    assert site.client.post("/api/contributions/not-an-id/annotation", json={"annotation": "ai"}).status_code == 404
    future = f"{int((site.clock() + 3600) * 1000):012x}" + "0" * 24
    assert site.client.post(f"/api/contributions/{future}/annotation", json={"annotation": "ai"}).status_code == 404
    fresh = score(site, "Another text.").json()["id"]
    assert site.client.post(f"/api/contributions/{fresh}/annotation", json={"annotation": "ai!"}).status_code == 400


def test_removal_stores_only_the_hash(make_site: Any) -> None:
    site = make_site()
    text = "Please remove this exact text."
    first = site.client.post("/api/removals", json={"text": text})
    second = site.client.post("/api/removals", json={"text": text})
    assert first.status_code == second.status_code == 202
    assert first.json()["content_hash"] == second.json()["content_hash"]
    shard_text = "".join((site.root / shard.path).read_text(encoding="utf-8") for shard in list_shards(site.root))
    assert text not in shard_text
    assert {record["kind"] for record in records(site.root)} == {"removal"}


def test_rate_limit(make_site: Any) -> None:
    site = make_site(Settings(scores_burst=3, scores_per_second=0.0001))
    statuses = [score(site, f"text number {index}").status_code for index in range(5)]
    assert statuses == [200, 200, 200, 429, 429]


def test_daily_scoring_budget(make_site: Any) -> None:
    site = make_site(Settings(daily_scoring_seconds=0.0))
    assert score(site).status_code == 503
    assert site.scorer.calls == []


def test_container_byte_budget(make_site: Any) -> None:
    site = make_site(Settings(container_bytes_burst=300, container_bytes_per_second=0.0001))
    assert score(site, "short").status_code == 200
    assert score(site, "x " * 200).status_code == 503


def test_terms_served_match_the_files(make_site: Any) -> None:
    site = make_site()
    terms = site.client.get("/api/terms").json()
    notice = read_document(LEGAL / "NOTICE.md")
    assert terms["version"] == notice.version and terms["status"] == notice.status
    assert terms["notice"] == as_served(notice.body, PRIVATE_PAGE)
    assert terms["consent_label"] == as_served(notice.fields["consent_label"], PRIVATE_PAGE)
    assert "(/legal/terms-of-use)" in terms["consent_label"] and "(TERMS_OF_USE.md)" in notice.fields["consent_label"]
    assert terms["links"] == {"terms_of_use": "/legal/terms-of-use", "privacy_notice": "/legal/privacy-notice",
                              "private_page": PRIVATE_PAGE}
    assert terms["limits"]["text_max_tokens"] == 131_072
    assert "Please do not paste sensitive or personal information" in terms["notice"]
    for path, name in (("/legal/terms-of-use", "TERMS_OF_USE.md"), ("/legal/privacy-notice", "PRIVACY_NOTICE.md")):
        served = site.client.get(path)
        assert served.headers["content-type"].startswith("text/plain")
        assert served.text == as_served(read_document(LEGAL / name).body, PRIVATE_PAGE)


def test_healthz(make_site: Any) -> None:
    site = make_site()
    assert site.client.get("/healthz").json() == {"ok": True, "model_version": "fake-model-1",
                                                  "terms_version": site.legal.version}
