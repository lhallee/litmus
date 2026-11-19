from __future__ import annotations

import pytest

from pathlib import Path
from litmus_release import versions


def project(root: Path) -> Path:
    """The files that state a version, with another `version =` line in each that must stay."""
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text('{\n  "name": "litmus",\n  "version": "0.1.0"\n}\n', encoding="utf-8")
    for manifest in versions.MANIFESTS:
        path = root / manifest.path
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"[{'.'.join(manifest.table)}]"
        text = f'[tool.other]\nversion = "9.9.9"\n\n{header}\nname = "x"\nversion = "0.1.0"\n\n[dependencies]\nserde = {{ version = "1" }}\n'
        path.write_bytes(text.replace("\n", "\r\n").encode() if manifest.path.startswith("student") else text.encode())
    for lock in versions.LOCKS:
        path = root / lock.path
        path.parent.mkdir(parents=True, exist_ok=True)
        packages = "".join(f'[[package]]\nname = "{crate}"\nversion = "0.1.0"\n\n' for crate in lock.crates)
        packages += '[[package]]\nname = "serde"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n'
        path.write_text(packages, encoding="utf-8", newline="\n")
    return root


def test_every_place_is_checked_and_set_from_one_version(tmp_path: Path) -> None:
    root = project(tmp_path)
    assert len(versions.problems(root, "1.4.0")) == len(versions.MANIFESTS) + 1 + sum(len(lock.crates) for lock in versions.LOCKS)
    changed = versions.apply(root, "1.4.0")
    assert len(changed) == len(versions.MANIFESTS) + 1 + len(versions.LOCKS)
    assert versions.problems(root, "1.4.0") == []
    assert versions.apply(root, "1.4.0") == []


def test_only_the_release_versions_change_and_line_endings_stay(tmp_path: Path) -> None:
    root = project(tmp_path)
    versions.apply(root, "1.4.0")
    student = (root / "student/pyproject.toml").read_bytes()
    assert b'version = "1.4.0"\r\n' in student and b'version = "9.9.9"\r\n' in student and student.count(b"\r\n") == student.count(b"\n")
    engine = (root / "engine/Cargo.toml").read_text(encoding="utf-8")
    assert 'version = "9.9.9"' in engine and 'serde = { version = "1" }' in engine
    lock = (root / "engine/Cargo.lock").read_text(encoding="utf-8")
    assert 'name = "serde"\nversion = "1.0.0"' in lock and 'name = "litmus-cli"\nversion = "1.4.0"' in lock
    assert '"version": "1.4.0"' in (root / ".claude-plugin/plugin.json").read_text(encoding="utf-8")


def test_a_missing_table_or_package_is_an_error_naming_it(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "app/Cargo.toml").write_text('[other]\nversion = "0.1.0"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match=r"\[package\]"):
        versions.apply(root, "1.4.0")
