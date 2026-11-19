"""Save a folder of the run to a private Hub repository, resuming where a stopped upload left off.

    python launch/save_run.py --folder /work/embed --repo lhallee/litmus-embeddings --repo-type dataset
    python launch/save_run.py --folder /work --repo lhallee/litmus-run --allow "teacher/**" "student/**" \\
        --ignore "**/wandb/**" "**/last.pt"

The repository is created private when it does not exist; an existing one must already be private, or nothing is
uploaded. Files keep their paths under `--folder`. The Hub token is read from the environment by `huggingface_hub`.
Prints the repository's revision after the upload.
"""

from __future__ import annotations

import argparse

from pathlib import Path
from huggingface_hub import HfApi

from foundry.downloads.hub.rate_limit import upload_resuming


def save(folder: Path, repo_id: str, repo_type: str, allow: list[str], ignore: list[str]) -> str:
    """Upload `folder` to the private repository `repo_id`; returns the revision it ends at."""
    assert folder.is_dir(), f"{folder} is not a folder"
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type=repo_type).private:
        raise RuntimeError(f"{repo_id} is public; a run is saved only to a private repository")
    upload_resuming(api, repo_id=repo_id, folder_path=folder, repo_type=repo_type, allow_patterns=allow or None,
                    ignore_patterns=ignore or None)

    return str(api.repo_info(repo_id, repo_type=repo_type).sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset"))
    parser.add_argument("--allow", nargs="*", default=[])
    parser.add_argument("--ignore", nargs="*", default=[])
    arguments = parser.parse_args()
    print(save(arguments.folder, arguments.repo, arguments.repo_type, arguments.allow, arguments.ignore))


if __name__ == "__main__":
    main()
