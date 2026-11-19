"""Put a written corpus on the Hub, in a private dataset repository only.

The corpus is kept processed and private. The repository is created private when it does
not exist; an existing one must already be private, or nothing is uploaded. Nothing here makes a repository
public. The Hub token is read from the environment by `huggingface_hub`.
"""

from __future__ import annotations

from pathlib import Path
from huggingface_hub import HfApi


DATASET = "dataset"


class PublicRepositoryError(RuntimeError):
    """The target repository is public, so the corpus may not go there."""


def ensure_private(api: HfApi, repo_id: str) -> None:
    """Create `repo_id` as a private dataset repository, or check that it already is one."""
    api.create_repo(repo_id, repo_type=DATASET, private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type=DATASET).private:
        raise PublicRepositoryError(f"{repo_id} is public; the Litmus corpus goes only to a private repository")


def publish(corpus: Path, repo_id: str, api: HfApi | None = None) -> str:
    """Upload a written corpus folder to the private dataset repository `repo_id`; returns the commit's revision."""
    assert (corpus / "report.json").exists(), f"{corpus} holds no written corpus"
    api = api or HfApi()
    ensure_private(api, repo_id)
    commit = api.upload_folder(folder_path=str(corpus), repo_id=repo_id, repo_type=DATASET, commit_message="Litmus corpus build")

    return commit.oid
