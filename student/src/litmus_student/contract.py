"""The NumPy reference (`student/reference.py`), loaded as a module: one copy of the release format and its arithmetic.

`reference.py` sits beside `SPEC.md`, outside the package, because the engine's authors read it as the contract; the
package imports it from there instead of copying it.
"""

from __future__ import annotations

import importlib.util
import sys

from pathlib import Path
from types import ModuleType


REFERENCE_PATH = Path(__file__).resolve().parents[2] / "reference.py"
MODULE_NAME = "litmus_student_reference"


def load_reference() -> ModuleType:
    """Import `reference.py` once, under a name no other module uses."""
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    specification = importlib.util.spec_from_file_location(MODULE_NAME, REFERENCE_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load the reference at {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[MODULE_NAME] = module  # dataclasses resolve their module through sys.modules
    specification.loader.exec_module(module)

    return module


reference = load_reference()
