"""The contribute build of the app, loaded into memory once: each file with its type, a gzip copy, an ETag and a cache
rule. Requests are answered from this table alone, so no request path reaches the filesystem.
"""

from __future__ import annotations

import gzip
import hashlib
import re

from dataclasses import dataclass
from pathlib import Path


MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
    ".wasm": "application/wasm",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".map": "application/json",
}
COMPRESSIBLE = {".html", ".js", ".mjs", ".css", ".json", ".webmanifest", ".wasm", ".svg", ".txt", ".map"}
# A content hash in the file name (`app-1a2b3c4d5e6f7a8b.js`, `app-1a2b3c4d5e6f7a8b_bg.wasm` as trunk names them,
# `style.1a2b3c4d.css`) marks a file that never changes.
HASHED = re.compile(r"[.-][0-9a-f]{8,}(_bg)?\.[A-Za-z0-9]+\Z")
IMMUTABLE = "public, max-age=31536000, immutable"
REVALIDATE = "no-cache"


@dataclass(frozen=True)
class Asset:
    body: bytes
    gzipped: bytes | None  # None when gzip does not save a tenth
    media_type: str
    etag: str
    cache_control: str


def load_asset(path: Path) -> Asset:
    suffix = path.suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise ValueError(f"{path}: no media type for {suffix!r}; add it to MEDIA_TYPES")
    body = path.read_bytes()
    gzipped = gzip.compress(body, compresslevel=9, mtime=0) if suffix in COMPRESSIBLE else None
    if gzipped is not None and len(gzipped) > 0.9 * len(body):
        gzipped = None
    return Asset(body=body, gzipped=gzipped, media_type=MEDIA_TYPES[suffix],
                 etag=f'"{hashlib.sha256(body).hexdigest()[:32]}"',
                 cache_control=IMMUTABLE if HASHED.search(path.name) else REVALIDATE)


def load_site(directory: Path) -> dict[str, Asset]:
    """Every file under `directory` by its URL path (`index.html`, `assets/app-<hash>.js`); it must have an index."""
    assets = {path.relative_to(directory).as_posix(): load_asset(path) for path in sorted(directory.rglob("*")) if path.is_file()}
    if "index.html" not in assets:
        raise ValueError(f"{directory} has no index.html")
    return assets


def asset_key(url_path: str) -> str:
    """`/` and `/dir/` to their index.html; other paths as given, without the leading slash."""
    key = url_path.lstrip("/")
    return f"{key}index.html" if key == "" or key.endswith("/") else key


def accepts_gzip(accept_encoding: str) -> bool:
    for part in accept_encoding.split(","):
        name, _, parameters = part.strip().partition(";")
        if name.strip().lower() == "gzip":
            return parameters.replace(" ", "").lower() not in ("q=0", "q=0.0", "q=0.00", "q=0.000")
    return False
