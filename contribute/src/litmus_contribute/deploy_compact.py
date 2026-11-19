"""Compaction on Modal, run by hand (no schedule): closed shards of the `litmus-contributions` volume to the private
Hub dataset, removals applied, the pushed shards deleted only after the push is verified.

From the repository root, a dry run first (it reads the Hub and writes nothing):

    PYTHONPATH=contribute/src modal run contribute/src/litmus_contribute/deploy_compact.py \
        --repo-id <the private dataset's repository>
    ... --no-dry-run
    ... --no-dry-run --squash      # only to finish a run that stopped at the history squash

A run that applies a removal to a file already on the Hub squashes the dataset's history (see `compact.py`). Only
`HF_TOKEN` reaches the container, through an ephemeral secret loaded by `foundry.secrets_env`.
"""

from __future__ import annotations

import os
import modal

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


APP_NAME = "litmus-contribute-compact"
VOLUME_NAME = "litmus-contributions"
DATA, CODE, CARD = "/data", "/srv/code", "/srv/DATASET_CARD.md"
PACKAGES = ("pyarrow", "huggingface-hub")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

if modal.is_local():
    from foundry.modal_image import ignore_with_credentials, pin_name, read_pins, workspace_root
    from foundry.secrets_env import load_secrets_env

    PACKAGE = Path(__file__).resolve().parent
    CONTRIBUTE = PACKAGE.parents[1]
    ROOT = workspace_root(PACKAGE)
    pins = {pin_name(pin): pin for pin in read_pins(ROOT / "environments/research/lock.txt")}
    image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install(*(pins[name] for name in PACKAGES))
             .env({"PYTHONPATH": CODE, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
             .add_local_file(CONTRIBUTE / "DATASET_CARD.md", CARD)
             .add_local_dir(PACKAGE, f"{CODE}/litmus_contribute", ignore=ignore_with_credentials(["**/__pycache__"])))
    load_secrets_env(names=["HF_TOKEN"])
    hub_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
else:
    image = modal.Image.debian_slim()
    hub_secret = modal.Secret.from_dict({})


@app.function(image=image, cpu=(1.0, 2.0), memory=(2048, 4096), timeout=1800, max_containers=1, secrets=[hub_secret],
              volumes={DATA: volume})
def compact_contributions(repo_id: str, dry_run: bool, squash: bool) -> dict[str, Any]:
    from litmus_contribute.compact import HubPublisher, compact
    from litmus_contribute.config import Settings
    from litmus_contribute.shards import VolumeStore

    receipt = compact(VolumeStore(Path(DATA), volume), HubPublisher(repo_id), Settings(), datetime.now(UTC),
                      card=Path(CARD).read_text(encoding="utf-8"), dry_run=dry_run, squash=squash)
    return asdict(receipt)


@app.local_entrypoint()
def main(repo_id: str, dry_run: bool = True, squash: bool = False) -> None:
    receipt = compact_contributions.remote(repo_id, dry_run, squash)
    print({key: value for key, value in receipt.items() if key not in ("consumed", "annotation_only")})
    print(f"{len(receipt['consumed'])} shards compacted, {len(receipt['annotation_only'])} read for annotations only")
