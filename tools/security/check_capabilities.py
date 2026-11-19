"""Check that every sensitive capability in the source is declared in `tools/security/capabilities.json`.

A capability is something a reader should know a file can do: reach the network, start another program, read or
write the clipboard or the Windows registry, take global input, read other applications' text, write files, store
data in the browser, run code it loads, or use `unsafe` Rust. The scan matches source lines against the patterns
below, ignoring lines that are only comments, and fails when it finds a capability in a file the manifest does not
declare it for, when the manifest declares one the scan no longer finds, or when a Rust file's count of `unsafe`
differs from the manifest's.

The scan sees direct uses only: a file that calls a function in a declared file inherits that capability without
being listed. It reads text, so a capability hidden by code generation, string concatenation, or a dependency is out
of its reach; `tools/security/check_dependencies.py` covers the Rust dependencies.

    python tools/security/check_capabilities.py             # check; exit status 1 on any mismatch
    python tools/security/check_capabilities.py --report    # every finding as JSON, declared or not
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY / "tools" / "security" / "capabilities.json"
# Build outputs, dependencies, and caches: none of them is source in this repository.
SKIPPED_DIRECTORIES = frozenset({".git", "target", "node_modules", "dist", "__pycache__", ".pytest_cache", "placeholder"})

RUST = (".rs",)
PYTHON = (".py",)
SCRIPT = (".js", ".mjs", ".ts", ".html", ".css")
SHELL = (".sh", ".ps1", "Dockerfile")

# capability -> language -> one regular expression, matched per line. Written so that no pattern matches its own text.
PATTERNS: dict[str, dict[tuple[str, ...], str]] = {
    "network": {
        RUST: r"\bstd::net\b|\bTcp(Stream|Listener)\b|\bUdpSocket\b|\b(reqwest|ureq|hyper)::|\btokio::net\b"
        r"|\bfetch_with_(request|str)\b|\bweb_sys::(Request|WebSocket|XmlHttpRequest)\b",
        PYTHON: r"^\s*(import|from)\s+(requests|httpx|urllib3?|aiohttp|socket|http\.client|huggingface_hub|datasets"
        r"|wandb|modal|fastapi|uvicorn|starlette)\b|\burllib\.request\b|\b(hf_hub_downloa[d]|snapshot_downloa[d])\b"
        r"|\.from_pretrained\(",
        SCRIPT: r"\bfetch\(|\bXMLHttpRequest\b|\bWebSocket\b|\bsendBeacon\b|\bEventSource\b|\bimportScripts\("
        r"|<(script|img|iframe)[^>]+src=[\"']https?:|<link[^>]+href=[\"']https?:|url\(\s*[\"']?https?:",
        SHELL: r"\b(curl|wget|huggingface-cli|Invoke-WebRequest)\b|\bpip3? install\b|\bgit clone\b|\bnpm (install|ci)\b"
        r"|\bcargo install\b|\bapt-get\b|\bhf (download|upload)\b|\bdocker (pull|push)\b",
    },
    "process": {
        RUST: r"\bCommand::new\b|\bprocess::Command\b|\bShellExecute\w*|\bwebbrowser::|\bhyperlink(_to)?\(|\bopen_url\(",
        PYTHON: r"^\s*(import|from)\s+(subprocess|multiprocessing)\b|\bsubprocess\.|\bos\.(system|popen|exec\w*|spawn\w*)\("
        r"|\bProcessPoolExecutor\b",
        SCRIPT: r"\bchild_process\b|\b(spawn|execFile|exec)Sync\(|\bspawn\(|\bexecFile\(",
    },
    "clipboard": {
        RUST: r"\b(Open|Close|Empty|Set|Get)Clipboard\w*|\bEnumClipboardFormats\b|\bcopy_text\(|\bCopyText\b|\barboard\b",
        SCRIPT: r"\bnavigator\.clipboard\b",
    },
    "registry": {
        RUST: r"\bReg(Create|Open|Set|Delete|Query|Close|Get|Enum)\w*\(|\bHKEY_[A-Z_]+|\bSystem::Registry\b|\bwinreg\b",
        PYTHON: r"^\s*(import|from)\s+winreg\b",
    },
    "input": {
        RUST: r"\bglobal_hotkey\b|\bGlobalHotKey\w*|\bRegisterHotKey\b|\bSetWindowsHookEx\w*|\bSendInput\b|\bkeybd_event\b"
        r"|\bGetAsyncKeyState\b|\btray_icon\b",
    },
    "other_apps": {
        RUST: r"\bC?IUIAutomation\w*|\bCUIAutomation\b|\bAXUIElement\w*",
    },
    "filesystem_write": {
        RUST: r"\bfs::(write|rename|remove_file|remove_dir\w*|create_dir\w*|copy|hard_link)\(|\bFile::create\("
        r"|\bOpenOptions\b",
        PYTHON: r"\bopen\([^)]*[\"'][wax]b?\+?[\"']|\.write_(text|bytes)\(|\bshutil\.(copy\w*|move|rmtree)\("
        r"|\bos\.(remove|unlink|rename|replace|makedirs|mkdir)\(|\.unlink\(|\.mkdir\(|\.to_parquet\(|\bsave_file\("
        r"|\btorch\.save\(|\bpq\.write_table\(|\bnp\.save\w*\(",
        SCRIPT: r"\b(writeFile|mkdir|rm|unlink|rename|copyFile|appendFile)(Sync)?\(|\bcreateWriteStream\(",
    },
    "browser_storage": {
        RUST: r"\bweb_sys::Storage\b|\blocal_storage\(",
        SCRIPT: r"\b(localStorage|sessionStorage|indexedDB)\b",
    },
    "dynamic_code": {
        PYTHON: r"(?<![.\w])(eval|exec)\(|\bpickle\.loads?\(|\btorch\.load\(|\bjoblib\.load\(|allow_pickle\s*=\s*True",
        SCRIPT: r"(?<![.\w])eval\(|\bnew Function\(",
    },
}
UNSAFE = re.compile(r"\bunsafe\b")
COMMENT_LINE = {RUST: re.compile(r"^\s*//"), PYTHON: re.compile(r"^\s*#"), SCRIPT: re.compile(r"^\s*(//|\*|/\*)"),
                SHELL: re.compile(r"^\s*#")}


@dataclass(frozen=True)
class Finding:
    """One capability found in one file, with the first line that shows it."""

    path: str
    capability: str
    line: int
    text: str


def language_of(path: Path) -> tuple[str, ...] | None:
    """The language group a file is scanned as, or None for a file the scan does not read."""
    for group in (RUST, PYTHON, SCRIPT, SHELL):
        if path.suffix in group or path.name in group:
            return group
    return None


def ignored_files(root: Path) -> set[str]:
    """The single files `.gitignore` names by exact path: files that exist where the code is written but are not
    part of this repository."""
    ignore = root / ".gitignore"
    if not ignore.is_file():
        return set()
    entries = (line.strip() for line in ignore.read_text(encoding="utf-8").splitlines())
    return {entry for entry in entries if entry and not entry.startswith("#") and not entry.endswith("/") and not any(c in entry for c in "*?[!")}


def source_files(root: Path) -> list[Path]:
    """Every scanned file under `root`, in path order, build outputs, caches, and ignored files left out."""
    ignored = ignored_files(root)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and not SKIPPED_DIRECTORIES.intersection(relative.parts)
            and relative.as_posix() not in ignored
            and language_of(path) is not None
        ):
            files.append(path)
    return files


def scan_file(path: Path, root: Path) -> tuple[list[Finding], int]:
    """The capabilities one file shows, one finding per capability, and its count of `unsafe` (Rust only)."""
    language = language_of(path)
    assert language is not None, path
    relative = path.relative_to(root).as_posix()
    compiled = {capability: re.compile(by_language[language]) for capability, by_language in PATTERNS.items() if language in by_language}
    found: dict[str, Finding] = {}
    unsafe_count = 0
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if COMMENT_LINE[language].match(line):
            continue
        if language == RUST:
            unsafe_count += len(UNSAFE.findall(line.split("//", 1)[0]))
        for capability, pattern in compiled.items():
            if capability not in found and pattern.search(line):
                found[capability] = Finding(relative, capability, number, line.strip()[:160])
    return list(found.values()), unsafe_count


def scan(root: Path) -> tuple[list[Finding], dict[str, int]]:
    """Every finding in the repository, and the count of `unsafe` in each Rust file that has any."""
    findings: list[Finding] = []
    unsafe_counts: dict[str, int] = {}
    for path in source_files(root):
        file_findings, unsafe_count = scan_file(path, root)
        findings += file_findings
        if unsafe_count:
            unsafe_counts[path.relative_to(root).as_posix()] = unsafe_count
    return findings, unsafe_counts


def absent_vendored(manifest: dict[str, Any], root: Path) -> list[str]:
    """The vendored directories the manifest names that this checkout lacks.

    The published repository always holds them. A source tree that keeps the same library elsewhere cannot check the
    declarations under these directories, and the run says so.
    """
    return [directory for directory in manifest["vendored_directories"] if not (root / directory).is_dir()]


def mismatches(manifest: dict[str, Any], findings: list[Finding], unsafe_counts: dict[str, int], absent: list[str]) -> list[str]:
    """Each way the scan and the manifest disagree, as one line of text; empty when they agree."""
    def checked(path: str) -> bool:
        return not any(path.startswith(f"{directory}/") for directory in absent)

    declared = {
        (path, capability) for path, entry in manifest["files"].items() if checked(path) for capability in entry["capabilities"]
    }
    seen = {(finding.path, finding.capability): finding for finding in findings}
    problems = [
        f"undeclared {capability} in {path}:{seen[(path, capability)].line}: {seen[(path, capability)].text}"
        for path, capability in sorted(seen.keys() - declared)
    ]
    problems += [f"declared {capability} in {path}, which the scan no longer finds" for path, capability in sorted(declared - seen.keys())]
    expected_unsafe = {path: entry["count"] for path, entry in manifest["unsafe"].items() if checked(path)}
    for path in sorted(expected_unsafe.keys() | unsafe_counts.keys()):
        if expected_unsafe.get(path, 0) != unsafe_counts.get(path, 0):
            problems.append(f"{path}: `unsafe` appears {unsafe_counts.get(path, 0)} times, the manifest says {expected_unsafe.get(path, 0)}")
    unknown = sorted({capability for entry in manifest["files"].values() for capability in entry["capabilities"]} - PATTERNS.keys())
    problems += [f"the manifest names an unknown capability {capability!r}" for capability in unknown]
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print every finding as JSON and exit 0")
    arguments = parser.parse_args()

    findings, unsafe_counts = scan(REPOSITORY)
    if arguments.report:
        report = {
            "findings": [finding.__dict__ for finding in findings],
            "unsafe": unsafe_counts,
        }
        print(json.dumps(report, indent=2))
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    absent = absent_vendored(manifest, REPOSITORY)
    problems = mismatches(manifest, findings, unsafe_counts, absent)
    for problem in problems:
        print(problem)
    for directory in absent:
        print(f"{directory}/ is not in this checkout, so its declarations were not checked")
    scanned = len(source_files(REPOSITORY))
    print(f"{scanned} files scanned, {len(findings)} capabilities found, {len(problems)} mismatches with {MANIFEST.relative_to(REPOSITORY).as_posix()}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
