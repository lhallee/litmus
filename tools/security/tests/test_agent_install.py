"""The install scripts of the agent skill install a program only when it matches the release's published file list.

Nothing here touches the network: each script installs from a folder (`--from-dir`, `-FromDir`) holding a stand-in
program and a SHA256SUMS file, the same two files a release attaches. What no test can reach is the download itself.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import pytest

from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[3] / ".agents" / "skills" / "litmus" / "scripts"
SH = shutil.which("sh") or next((str(p) for p in (Path("C:/Program Files/Git/usr/bin/sh.exe"), Path("C:/Program Files/Git/bin/sh.exe")) if p.is_file()), None)
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
WORKING = b"#!/bin/sh\necho stand-in litmus\n"
BROKEN = b"#!/bin/sh\nexit 1\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release(folder: Path, asset: str, program: bytes, sums: str | None = None) -> Path:
    """A folder like a release's: the program, and SHA256SUMS naming it (or the text `sums` says)."""
    folder.mkdir(parents=True)
    (folder / asset).write_bytes(program)
    (folder / "SHA256SUMS").write_text(f"{digest(program)}  {asset}\n" if sums is None else sums, encoding="utf-8")
    return folder


def install_sh(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    assert SH is not None, "needs a POSIX shell (Git for Windows provides one) to test install.sh"
    (tmp_path / "scratch").mkdir(exist_ok=True)
    environment = {**os.environ, "TMPDIR": (tmp_path / "scratch").as_posix()}
    return subprocess.run([SH, (SCRIPTS / "install.sh").as_posix(), *arguments], capture_output=True, text=True, env=environment, timeout=120)


def leftovers(tmp_path: Path, bin_folder: Path) -> list[str]:
    """Anything an install left in its temporary folder, or a staged file beside the program."""
    found = [p.name for p in (tmp_path / "scratch").iterdir()]
    found += [p.name for p in bin_folder.iterdir() if p.name.startswith(".litmus.new")] if bin_folder.is_dir() else []
    return found


def test_a_matching_program_is_installed_and_described(tmp_path: Path) -> None:
    source = release(tmp_path / "release", "litmus-linux-x86_64", WORKING)
    done = install_sh(tmp_path, "--from-dir", source.as_posix(), "--dir", (tmp_path / "bin").as_posix(), "--target", "linux-x86_64")
    assert done.returncode == 0, done.stderr
    installed = tmp_path / "bin" / "litmus"
    assert installed.read_bytes() == WORKING
    assert os.access(installed, os.X_OK)
    assert "Installed " in done.stdout and "built in" in done.stdout and "claude mcp add litmus" in done.stdout
    assert "sha" not in done.stdout.lower() and "hash" not in done.stdout.lower(), "what it prints does not advertise the check's method"
    assert leftovers(tmp_path, tmp_path / "bin") == []


def test_the_published_list_may_star_the_name_and_use_capitals(tmp_path: Path) -> None:
    sums = f"# a comment\n{digest(b'other')}  other-file\n{digest(WORKING).upper()} *litmus-macos-arm64\n"
    source = release(tmp_path / "release", "litmus-macos-arm64", WORKING, sums)
    assert install_sh(tmp_path, "--from-dir", source.as_posix(), "--dir", (tmp_path / "bin").as_posix(), "--target", "macos-arm64").returncode == 0


def test_a_program_that_differs_from_the_list_is_not_installed_and_the_old_one_stays(tmp_path: Path) -> None:
    source = release(tmp_path / "release", "litmus-linux-x86_64", WORKING)
    (source / "litmus-linux-x86_64").write_bytes(WORKING + b"echo tampered\n")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "litmus").write_bytes(b"the earlier copy")
    done = install_sh(tmp_path, "--from-dir", source.as_posix(), "--dir", (tmp_path / "bin").as_posix(), "--target", "linux-x86_64")
    assert done.returncode == 1 and "nothing was installed" in done.stderr
    assert (tmp_path / "bin" / "litmus").read_bytes() == b"the earlier copy"
    assert leftovers(tmp_path, tmp_path / "bin") == []


def test_a_list_that_does_not_name_the_program_installs_nothing(tmp_path: Path) -> None:
    source = release(tmp_path / "release", "litmus-linux-x86_64", WORKING, f"{digest(WORKING)}  litmus-macos-arm64\n")
    done = install_sh(tmp_path, "--from-dir", source.as_posix(), "--dir", (tmp_path / "bin").as_posix(), "--target", "linux-x86_64")
    assert done.returncode == 1 and "does not name litmus-linux-x86_64" in done.stderr
    assert not (tmp_path / "bin" / "litmus").exists()


def test_a_later_install_replaces_the_earlier_program(tmp_path: Path) -> None:
    source = release(tmp_path / "release", "litmus-linux-x86_64", WORKING)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "litmus").write_bytes(b"#!/bin/sh\necho old\n")
    done = install_sh(tmp_path, "--from-dir", source.as_posix(), "--dir", (tmp_path / "bin").as_posix(), "--target", "linux-x86_64")
    assert done.returncode == 0 and "replacing the earlier copy" in done.stdout
    assert (tmp_path / "bin" / "litmus").read_bytes() == WORKING


def test_a_program_that_does_not_run_is_reported(tmp_path: Path) -> None:
    source = release(tmp_path / "release", "litmus-linux-x86_64", BROKEN)
    done = install_sh(tmp_path, "--from-dir", source.as_posix(), "--dir", (tmp_path / "bin").as_posix(), "--target", "linux-x86_64")
    assert done.returncode == 3 and "does not run on this computer" in done.stderr


@pytest.mark.parametrize(
    "arguments",
    [("--version", "v1;rm -rf /"), ("--version", ""), ("--target", "plan9-arm"), ("--bogus",), ("--dir",), ("--from-dir",)],
)
def test_bad_arguments_are_refused_before_anything_happens(tmp_path: Path, arguments: tuple[str, ...]) -> None:
    done = install_sh(tmp_path, *arguments)
    assert done.returncode == 2, done.stdout + done.stderr
    assert leftovers(tmp_path, tmp_path / "bin") == []


def test_the_script_names_only_this_projects_releases() -> None:
    text = (SCRIPTS / "install.sh").read_text(encoding="utf-8") + (SCRIPTS / "install.ps1").read_text(encoding="utf-8")
    hosts = {line.split("//", 1)[1].split("/", 1)[0] for line in text.replace('"', "\n").splitlines() if line.startswith("https://")}
    assert hosts == {"github.com"}
    assert text.count("lhallee/litmus") >= 2 and "--insecure" not in text and "-k " not in text


needs_windows = pytest.mark.skipif(sys.platform != "win32", reason="install.ps1 installs the Windows program and refuses other systems")


def install_ps1(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None, "needs Windows PowerShell to test install.ps1"
    command = [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPTS / "install.ps1"), *arguments]
    environment = {**os.environ, "TEMP": str(tmp_path / "scratch"), "TMP": str(tmp_path / "scratch")}
    (tmp_path / "scratch").mkdir(exist_ok=True)
    return subprocess.run(command, capture_output=True, text=True, env=environment, timeout=300)


def windows_program() -> bytes:
    """A real Windows program that exits 0 for --help under another name, standing in for litmus.exe.

    curl.exe, not tar.exe: on Windows 11 build 26300 a copy of tar.exe named litmus.exe exits -536870873.
    """
    return (Path(os.environ["SystemRoot"]) / "System32" / "curl.exe").read_bytes()


@needs_windows
def test_windows_installs_a_matching_program_and_refuses_a_changed_one(tmp_path: Path) -> None:
    program = windows_program()
    source = release(tmp_path / "release", "litmus.exe", program)
    done = install_ps1(tmp_path, "-FromDir", str(source), "-Dir", str(tmp_path / "bin"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "bin" / "litmus.exe").read_bytes() == program
    assert "Installed " in done.stdout and "no PATH or setting was changed" in done.stdout
    assert "sha" not in done.stdout.lower() and "hash" not in done.stdout.lower()
    assert [p.name for p in (tmp_path / "scratch").iterdir()] == [], "the temporary folder is removed, also on success"
    assert not any(p.name.startswith(".litmus.new") for p in (tmp_path / "bin").iterdir())

    changed = release(tmp_path / "changed", "litmus.exe", program)
    (changed / "litmus.exe").write_bytes(program + b"\x00")
    refused = install_ps1(tmp_path, "-FromDir", str(changed), "-Dir", str(tmp_path / "bin2"))
    assert refused.returncode == 1 and "nothing was installed" in refused.stderr
    assert not (tmp_path / "bin2" / "litmus.exe").exists()
    assert [p.name for p in (tmp_path / "scratch").iterdir()] == [], "the temporary folder is removed, also on failure"


@needs_windows
def test_windows_refuses_a_list_that_does_not_name_the_program_and_a_bad_version(tmp_path: Path) -> None:
    source = release(tmp_path / "release", "litmus.exe", b"x", f"{digest(b'x')}  other.exe\n")
    done = install_ps1(tmp_path, "-FromDir", str(source), "-Dir", str(tmp_path / "bin"))
    assert done.returncode == 1 and "does not name litmus.exe" in done.stderr
    bad = install_ps1(tmp_path, "-Version", "v1;calc")
    assert bad.returncode == 2
