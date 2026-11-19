"""Cut a Litmus release: the staged, resumable driver from a trained model on the Hub to a published release.

    python release/cut.py run --model OWNER/NAME@<commit> --version 1.0.0 --work <folder outside the repository>
    python release/cut.py status --work <folder>
    python release/cut.py publish --work <folder> --dry-run

`release/RELEASING.md` is the runbook; `release/SKILL.md` is what an agent loads to follow it.
"""

import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from litmus_release.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
