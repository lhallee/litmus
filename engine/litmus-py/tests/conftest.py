"""Where the binding's tests find a release, the fixtures and the `litmus` executable.

Each location is a pytest option or, when the option is absent, an environment variable (engine/README.md, Tests):

    --litmus-release DIR     LITMUS_RELEASE            a trained release folder holding tokenizer.json
    --litmus-cli FILE        LITMUS_CLI                the `litmus` executable built from the same checkout
    --litmus-fixtures DIR    LITMUS_STUDENT_FIXTURES   the folder `reference.py fixtures --out <dir>` wrote
    --litmus-tokenizer DIR   LITMUS_TOKENIZER_DIR      a folder holding ModernBERT's tokenizer.json

A test that needs a location fails, never skips, when it is missing.
"""

from __future__ import annotations

import os
import litmus_engine
import pytest

from pathlib import Path


LOCATIONS = {
    "release": "LITMUS_RELEASE",
    "cli": "LITMUS_CLI",
    "fixtures": "LITMUS_STUDENT_FIXTURES",
    "tokenizer": "LITMUS_TOKENIZER_DIR",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    for name, variable in LOCATIONS.items():
        parser.addoption(f"--litmus-{name}", default=None, help=f"default: ${variable}")


def location(config: pytest.Config, name: str) -> Path:
    value = config.getoption(f"--litmus-{name}") or os.environ.get(LOCATIONS[name])
    assert value, f"pass --litmus-{name} or set {LOCATIONS[name]} (engine/README.md, Tests)"
    path = Path(value)
    assert path.exists(), f"{path} does not exist"
    return path


@pytest.fixture(scope="session")
def release(pytestconfig: pytest.Config) -> Path:
    return location(pytestconfig, "release")


@pytest.fixture(scope="session")
def cli(pytestconfig: pytest.Config) -> Path:
    return location(pytestconfig, "cli")


@pytest.fixture(scope="session")
def fixtures_folder(pytestconfig: pytest.Config) -> Path:
    return location(pytestconfig, "fixtures")


@pytest.fixture(scope="session")
def tokenizer_file(pytestconfig: pytest.Config) -> Path:
    return location(pytestconfig, "tokenizer") / "tokenizer.json"


@pytest.fixture(scope="session")
def model(release: Path) -> litmus_engine.Model:
    """The trained release, loaded once and shared by every test, as the server shares it across requests."""
    return litmus_engine.Model.from_path(str(release))
