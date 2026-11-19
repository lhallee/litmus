"""The model verifier catches a changed or unstated file; the build verifier's flags and checksum parsing."""

import hashlib
import json
import verify_build
import verify_model

from pathlib import Path


def release(folder: Path, state_weights: bool = True) -> None:
    """A release folder of stand-in files whose config.json states their hashes."""
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"weights" * 100)
    (folder / "tokenizer.json").write_text('{"model": {}}', encoding="utf-8")
    config = {"tokenizer": {"sha256": hashlib.sha256((folder / "tokenizer.json").read_bytes()).hexdigest()}}
    if state_weights:
        config["weights"] = {"sha256": hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_a_matching_release_passes(tmp_path: Path) -> None:
    release(tmp_path / "release")
    assert verify_model.verify(tmp_path / "release", []) == []


def test_changed_weights_fail(tmp_path: Path) -> None:
    release(tmp_path / "release")
    with (tmp_path / "release" / "model.safetensors").open("ab") as stream:
        stream.write(b"\x00")
    problems = verify_model.verify(tmp_path / "release", [])
    assert len(problems) == 1 and "model.safetensors" in problems[0]


def test_unstated_weights_fail_until_a_sums_file_states_them(tmp_path: Path) -> None:
    release(tmp_path / "release", state_weights=False)
    assert verify_model.verify(tmp_path / "release", []) != []
    digest = hashlib.sha256((tmp_path / "release" / "model.safetensors").read_bytes()).hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"# a comment\n{digest}  model.safetensors\n", encoding="utf-8")
    assert verify_model.verify(tmp_path / "release", [sums]) == []


def test_build_flags_remove_the_source_and_registry_paths(tmp_path: Path) -> None:
    for native in (True, False):
        flags = verify_build.release_rustflags(tmp_path, native)
        assert f"--remap-path-prefix={tmp_path}={verify_build.SOURCE_ROOT_NAME}" in flags
        assert any(flag.startswith(f"--remap-path-prefix={verify_build.cargo_home()}=") for flag in flags)
    # A WebAssembly module is linked by rust-lld, which takes no MSVC linker flag.
    assert not any("/Brepro" in flag for flag in verify_build.release_rustflags(tmp_path, native=False))


def test_the_repository_checksums_file_parses() -> None:
    hashes = verify_build.published_hashes()
    assert all(len(digest) == 64 for digest in hashes.values())
