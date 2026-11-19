"""The contribute site on Modal: one-core CPU containers restored from a memory snapshot with the engine loaded,
scaled to zero when idle and out to at most 32, writing contributions to the `litmus-contributions` volume.

From the repository root:

    PYTHONPATH=contribute/src \
    LITMUS_CONTRIBUTE_SITE=<built contribute page> LITMUS_RELEASE=<release folder> \
    LITMUS_ENGINE_WHEEL=<litmus_engine wheel, linux x86_64, cp312> LITMUS_PRIVATE_PAGE=<the private page's address> \
    modal deploy contribute/src/litmus_contribute/deploy_web.py

The deployment refuses legal documents that are drafts or hold a blank, unless LITMUS_LEGAL_DRAFT=allow (a private
trial only). No secret reaches these containers.
"""

from __future__ import annotations

import os
import modal

from pathlib import Path
from typing import Any


APP_NAME = "litmus-contribute"
VOLUME_NAME = "litmus-contributions"
USAGE_TABLE = "litmus-contribute-usage"
DATA, SITE, LEGAL, RELEASE, CODE = "/data", "/srv/site", "/srv/legal", "/srv/release", "/srv/code"
WEB_PACKAGES = ("starlette", "anyio", "idna", "sniffio", "typing-extensions")
MAX_CONTAINERS = 32  # the ceiling of the scale-out; each container scores one text at a time

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
usage_table = modal.Dict.from_name(USAGE_TABLE, create_if_missing=True)

if modal.is_local():
    from foundry.modal_image import ignore_with_credentials, pin_name, read_pins, workspace_root
    from litmus_contribute.legal import load_legal

    PACKAGE = Path(__file__).resolve().parent
    CONTRIBUTE = PACKAGE.parents[1]
    ROOT = workspace_root(PACKAGE)
    missing = [name for name in ("LITMUS_CONTRIBUTE_SITE", "LITMUS_RELEASE", "LITMUS_ENGINE_WHEEL", "LITMUS_PRIVATE_PAGE")
               if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} (see this file's docstring).")
    private_page = os.environ["LITMUS_PRIVATE_PAGE"]
    legal = load_legal(CONTRIBUTE / "legal", private_page)
    if (legal.status != "approved" or legal.blanks()) and os.environ.get("LITMUS_LEGAL_DRAFT") != "allow":
        raise SystemExit(f"The legal documents are {legal.status} with {len(legal.blanks())} blanks: a maintainer approves them "
                         "(status: approved, no {{BLANK}}) before the site is public.")
    wheel = Path(os.environ["LITMUS_ENGINE_WHEEL"])
    pins = {pin_name(pin): pin for pin in read_pins(ROOT / "environments/research/lock.txt")}
    image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install(*(pins[name] for name in WEB_PACKAGES))
             .add_local_file(wheel, f"/wheels/{wheel.name}", copy=True)
             .run_commands(f"python -m pip install --no-deps /wheels/{wheel.name}")
             .env({"PYTHONPATH": CODE, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                   "LITMUS_PRIVATE_PAGE": private_page})
             .add_local_dir(Path(os.environ["LITMUS_CONTRIBUTE_SITE"]), SITE, ignore=ignore_with_credentials())
             .add_local_dir(Path(os.environ["LITMUS_RELEASE"]), RELEASE, ignore=ignore_with_credentials())
             .add_local_dir(CONTRIBUTE / "legal", LEGAL, ignore=ignore_with_credentials())
             .add_local_dir(PACKAGE, f"{CODE}/litmus_contribute", ignore=ignore_with_credentials(["**/__pycache__"])))
else:
    image = modal.Image.debian_slim()


# cpu=(0.125, 1.0): an eighth of a core requested (Modal's default request), one core at most. Modal bills the higher
# of the request and the use, so a scoring pays for the core it uses and a waiting container for an eighth.
# The price is that a busy host guarantees a scoring
# only the eighth. scaledown_window=5 (2 s to 20 minutes allowed): a container leaves 5 s after its last request,
# since a start from the snapshot takes about 2 s and waiting would be most of the bill at 60 s; the commit at shutdown
# still runs. memory: 512 MiB requested (the engine, its release of about
# 30 MB, one 1 MiB text and its reading), capped at 1 GiB. Each container scores one text at a time
# (Settings.scorings_per_container) and takes up to 8 requests, the autoscaler aiming at 1 so a second scoring starts
# another container. timeout bounds one request; Modal's web proxy answers 303 past 150 s and the browser follows it.
@app.cls(image=image, cpu=(0.125, 1.0), memory=(512, 1024), timeout=300, scaledown_window=5, min_containers=0,
         max_containers=MAX_CONTAINERS, enable_memory_snapshot=True, volumes={DATA: volume})
@modal.concurrent(max_inputs=8, target_inputs=1)
class Contribute:
    @modal.enter(snap=True)
    def load(self) -> None:
        """Load and warm the engine, the page and the documents into memory; captured in the snapshot. Nothing here
        draws randomness, names the container or touches the volume."""
        from litmus_contribute.legal import load_legal
        from litmus_contribute.scoring import EngineScorer
        from litmus_contribute.site import load_site

        self.scorer = EngineScorer(Path(RELEASE))
        self.scorer.score("A short text read once so the engine's pages are resident before the snapshot.")
        self.assets = load_site(Path(SITE))
        self.legal = load_legal(Path(LEGAL), os.environ["LITMUS_PRIVATE_PAGE"])

    @modal.asgi_app()
    def web(self) -> Any:
        from litmus_contribute.config import Settings
        from litmus_contribute.shards import ContainerName, VolumeStore
        from litmus_contribute.usage import UsageLedger
        from litmus_contribute.web import create_app

        settings, name = Settings(), ContainerName()
        return create_app(settings, VolumeStore(Path(DATA), volume), self.scorer, self.assets, self.legal,
                          UsageLedger(usage_table, name, settings.daily_scoring_seconds), name,
                          os.environ["LITMUS_PRIVATE_PAGE"])
