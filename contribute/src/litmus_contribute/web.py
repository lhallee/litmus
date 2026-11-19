"""The contribute site as one ASGI app: the static page, scoring, annotation, removal, the terms and a health check.

Nothing identifying is kept: the client address is used only as the input of a salted hash for rate limits, which
never leaves memory, and no request header, address, cookie or body is logged. The site answers same-origin requests
only (no CORS headers; a JSON content type forces a preflight that fails for any other origin).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
import anyio

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings
from .inputs import RequestError, contribution_time_ms, parse_annotation, parse_removal, parse_score_request
from .legal import Legal
from .ratelimit import RateLimiter, Rule
from .scoring import Scorer, serialize_reading
from .shards import AnnotationRecord, ContainerName, ContributionRecord, RemovalRecord, ShardStore, ShardWriter, content_hash
from .site import Asset, accepts_gzip, asset_key
from .usage import UsageLedger


LOG = logging.getLogger("litmus_contribute")
NO_STORE = {"Cache-Control": "no-store"}
SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), interest-cohort=()"),
    # The page talks to its own origin only; inline scripts and wasm compilation are allowed for the app's build.
    (b"content-security-policy", b"default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; "
                                 b"style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
                                 b"frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
]


@dataclass(frozen=True)
class Limits:
    text_max_bytes: int
    text_max_tokens: int
    annotation_window_seconds: int


class SecurityHeaders:
    """Adds the same security headers to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), *SECURITY_HEADERS]
            await send(message)

        await self.app(scope, receive, send_with_headers)


def client_address(request: Request) -> str:
    """The address Modal's proxy puts in the ASGI scope; only ever hashed."""
    return request.client.host if request.client is not None else "unknown"


def check_same_origin(request: Request, extra_origins: tuple[str, ...]) -> None:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/json":
        raise RequestError(415, "Send application/json.")
    host = request.headers.get("host", "")
    allowed = {f"https://{host}", f"{request.url.scheme}://{host}", *extra_origins}
    origin = request.headers.get("origin")
    if origin is not None and origin not in allowed:
        raise RequestError(403, "Requests from other sites are refused.")
    if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        raise RequestError(403, "Requests from other sites are refused.")


async def read_body(request: Request, limit: int) -> bytes:
    """The body, refused as soon as it passes `limit` bytes."""
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > limit):
        raise RequestError(413, f"The body is over {limit} bytes.")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise RequestError(413, f"The body is over {limit} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def new_contribution_id(now_ms: int) -> str:
    """12 hex digits of the time, so the annotation window is checked from the id alone, and 96 random bits."""
    return f"{now_ms:012x}{secrets.token_hex(12)}"


def create_app(settings: Settings, store: ShardStore, scorer: Scorer, assets: dict[str, Asset], legal: Legal,
               usage: UsageLedger, name: ContainerName, private_page: str,
               clock: Callable[[], float] = time.time) -> Starlette:
    limiter = RateLimiter({
        "scores": Rule(settings.scores_burst, settings.scores_per_second),
        "text_bytes": Rule(settings.text_bytes_burst, settings.text_bytes_per_second),
        "annotations": Rule(settings.annotations_burst, settings.annotations_per_second),
        "removals": Rule(settings.removals_burst, settings.removals_per_second),
        "container_bytes": Rule(settings.container_bytes_burst, settings.container_bytes_per_second),
    }, max_keys=settings.rate_limit_max_keys)
    writer = ShardWriter(store, name, settings.shard_window_seconds, settings.shard_max_bytes, clock=clock)
    scoring_slots = asyncio.Semaphore(settings.scorings_per_container)
    background: list[asyncio.Task[None]] = []
    limits = Limits(settings.text_max_bytes, settings.text_max_tokens, settings.annotation_window_seconds)

    def require(rule: str, cost: float, address: str, message: str, status: int = 429) -> None:
        if not limiter.allow(rule, address, cost):
            raise RequestError(status, message)

    async def store_record(record: ContributionRecord | AnnotationRecord | RemovalRecord, size: int) -> None:
        for task in background:  # a failed commit loop fails every later write, loudly
            if task.done():
                task.result()
        require("container_bytes", size, "", "The site is taking no more contributions for now; try again later.", 503)
        await anyio.to_thread.run_sync(writer.append, record)

    async def score(request: Request) -> Response:
        check_same_origin(request, settings.extra_origins)
        address = client_address(request)
        require("scores", 1.0, address, "Too many checks from here; wait a minute.")
        body = await read_body(request, settings.body_max_bytes)
        score_request = parse_score_request(body, settings, legal.version)
        text_bytes = len(score_request.text.encode("utf-8"))
        require("text_bytes", text_bytes, address, "Too much text from here; wait a few minutes.")
        if usage.exhausted():
            raise RequestError(503, "The site has used its scoring budget for today; try again tomorrow (UTC).")
        if await anyio.to_thread.run_sync(scorer.count_tokens, score_request.text) > settings.text_max_tokens:
            raise RequestError(413, f"The text is over {settings.text_max_tokens} tokens.")
        async with scoring_slots:
            started = time.perf_counter()
            reading = await anyio.to_thread.run_sync(scorer.score, score_request.text)
            usage.add(time.perf_counter() - started)
        serialized = serialize_reading(reading, settings.reading_max_bytes)
        now_ms = int(clock() * 1000)
        record = ContributionRecord(kind="contribution", id=new_contribution_id(now_ms), text=score_request.text,
                                    content_hash=content_hash(score_request.text), reading=reading,
                                    model_version=scorer.version, annotation=score_request.annotation,
                                    terms_version=score_request.terms_version, received_at=now_ms)
        await store_record(record, text_bytes + len(serialized))
        return JSONResponse({"id": record["id"], "reading": reading, "model_version": record["model_version"],
                             "terms_version": record["terms_version"], "annotation": record["annotation"],
                             "annotation_window_seconds": settings.annotation_window_seconds}, headers=NO_STORE)

    async def annotate(request: Request) -> Response:
        check_same_origin(request, settings.extra_origins)
        require("annotations", 1.0, client_address(request), "Too many changes from here; wait a minute.")
        contribution_id = request.path_params["contribution_id"]
        received_ms, now_ms = contribution_time_ms(contribution_id), int(clock() * 1000)
        if received_ms > now_ms + 60_000:
            raise RequestError(404, "No such contribution.")
        if now_ms - received_ms > settings.annotation_window_seconds * 1000:
            raise RequestError(410, "The time to label this text has passed.")
        annotation = parse_annotation(await read_body(request, settings.small_body_max_bytes))
        await store_record(AnnotationRecord(kind="annotation", id=contribution_id, annotation=annotation, at=now_ms), 128)
        return JSONResponse({"id": contribution_id, "annotation": annotation}, status_code=202, headers=NO_STORE)

    async def remove(request: Request) -> Response:
        check_same_origin(request, settings.extra_origins)
        require("removals", 1.0, client_address(request), "Too many removals from here; wait a few minutes.")
        text = parse_removal(await read_body(request, settings.body_max_bytes), settings)
        digest = content_hash(text)
        await store_record(RemovalRecord(kind="removal", content_hash=digest, at=int(clock() * 1000)), 128)
        return JSONResponse({"content_hash": digest, "removal": "accepted"}, status_code=202, headers=NO_STORE)

    async def terms(request: Request) -> Response:
        return JSONResponse({"version": legal.version, "status": legal.status, "notice": legal.notice,
                             "consent_label": legal.consent_label,
                             "links": {"terms_of_use": "/legal/terms-of-use", "privacy_notice": "/legal/privacy-notice",
                                       "private_page": private_page},
                             "limits": asdict(limits)}, headers={"Cache-Control": "no-cache"})

    def document(text: str) -> Callable[[Request], Awaitable[Response]]:
        async def serve(request: Request) -> Response:
            return PlainTextResponse(text, headers={"Cache-Control": "no-cache"})
        return serve

    async def healthz(request: Request) -> Response:
        return JSONResponse({"ok": True, "model_version": scorer.version, "terms_version": legal.version}, headers=NO_STORE)

    async def static(request: Request) -> Response:
        asset = assets.get(asset_key(request.url.path))
        if asset is None:
            return PlainTextResponse("Not found.", status_code=404)
        headers = {"ETag": asset.etag, "Cache-Control": asset.cache_control, "Vary": "Accept-Encoding"}
        if asset.etag in [tag.strip() for tag in request.headers.get("if-none-match", "").split(",")]:
            return Response(status_code=304, headers=headers)
        body = asset.body
        if asset.gzipped is not None and accepts_gzip(request.headers.get("accept-encoding", "")):
            body, headers["Content-Encoding"] = asset.gzipped, "gzip"
        if request.method == "HEAD":
            headers["Content-Length"] = str(len(body))
            return Response(status_code=200, headers=headers, media_type=asset.media_type)
        return Response(body, headers=headers, media_type=asset.media_type)

    async def request_error(request: Request, error: Exception) -> Response:
        assert isinstance(error, RequestError)
        return JSONResponse({"error": error.message}, status_code=error.status, headers=NO_STORE)

    async def commit_loop() -> None:
        while True:
            await asyncio.sleep(settings.commit_interval_seconds)
            await anyio.to_thread.run_sync(writer.commit_if_dirty)
            await anyio.to_thread.run_sync(usage.flush)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        LOG.info("litmus-contribute: model %s, terms %s", scorer.version, legal.version)
        background.append(asyncio.create_task(commit_loop()))
        try:
            yield
        finally:
            for task in background:
                task.cancel()
            await anyio.to_thread.run_sync(writer.commit_if_dirty)
            await anyio.to_thread.run_sync(usage.flush)
            LOG.info("litmus-contribute: %d records written, committed at shutdown", writer.records_written)

    routes = [
        Route("/api/score", score, methods=["POST"]),
        Route("/api/contributions/{contribution_id}/annotation", annotate, methods=["POST"]),
        Route("/api/removals", remove, methods=["POST"]),
        Route("/api/terms", terms, methods=["GET"]),
        Route("/legal/terms-of-use", document(legal.terms_of_use), methods=["GET"]),
        Route("/legal/privacy-notice", document(legal.privacy_notice), methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        Route("/{path:path}", static, methods=["GET", "HEAD"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan, exception_handlers={RequestError: request_error},
                    middleware=[Middleware(SecurityHeaders)])
    app.state.writer = writer
    return app
