"""Put the checkers in `tools/security/` and the verifiers in `tools/` on the import path; neither is a package."""

import sys

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
for folder in ("tools/security", "tools"):
    sys.path.insert(0, str(REPOSITORY / folder))
