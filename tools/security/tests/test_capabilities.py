"""The capability manifest matches the source, and the scan catches what it is meant to catch."""

import json
import check_capabilities

from pathlib import Path


def test_manifest_matches_the_source() -> None:
    manifest = json.loads(check_capabilities.MANIFEST.read_text(encoding="utf-8"))
    findings, unsafe_counts = check_capabilities.scan(check_capabilities.REPOSITORY)
    absent = check_capabilities.absent_vendored(manifest, check_capabilities.REPOSITORY)
    assert check_capabilities.mismatches(manifest, findings, unsafe_counts, absent) == []


def test_every_declaration_says_why() -> None:
    manifest = json.loads(check_capabilities.MANIFEST.read_text(encoding="utf-8"))
    for path, entry in manifest["files"].items():
        assert entry["component"] in manifest["components"], path
        for capability, reason in entry["capabilities"].items():
            assert len(reason.split()) >= 4, f"{path}: the reason for {capability} is too short to inform"
    for path, entry in manifest["unsafe"].items():
        assert entry["count"] > 0 and len(entry["reason"].split()) >= 4, path


PLANTED = {
    "engine/src/net.rs": ("let stream = std::net::TcpStream::connect(\"example.org:80\");\n", "network"),
    "engine/src/spawn.rs": ("let child = std::process::Command::new(\"sh\").spawn();\n", "process"),
    "app/src/keys.rs": ("unsafe { SendInput(&inputs, size) };\n", "input"),
    "training/upload.py": ("import requests\n", "network"),
    "training/run.py": ("import subprocess\n", "process"),
    "training/load.py": ("state = torch.load(path)\n", "dynamic_code"),
    "site/page.js": ("fetch('https://example.org/collect', { method: 'POST' });\n", "network"),
    "site/page.html": ("<script src=\"https://cdn.example.org/tracker.js\"></script>\n", "network"),
}


def plant(root: Path, files: dict[str, tuple[str, str]]) -> None:
    for relative, (text, _) in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_planted_capabilities_are_found_and_undeclared(tmp_path: Path) -> None:
    plant(tmp_path, PLANTED)
    findings, unsafe_counts = check_capabilities.scan(tmp_path)
    found = {(finding.path, finding.capability) for finding in findings}
    assert {(path, capability) for path, (_, capability) in PLANTED.items()} <= found
    assert unsafe_counts == {"app/src/keys.rs": 1}

    empty = {"vendored_directories": [], "components": {}, "files": {}, "unsafe": {}}
    problems = check_capabilities.mismatches(empty, findings, unsafe_counts, [])
    assert len(problems) == len(found) + 1


def test_comment_lines_are_not_capabilities(tmp_path: Path) -> None:
    plant(tmp_path, {
        "engine/src/doc.rs": ("// std::net::TcpStream is never used here\n/// unsafe is not used either\n", ""),
        "training/doc.py": ("# import requests\n", ""),
    })
    findings, unsafe_counts = check_capabilities.scan(tmp_path)
    assert findings == []
    assert unsafe_counts == {}


def test_a_stale_declaration_fails(tmp_path: Path) -> None:
    plant(tmp_path, {"engine/src/quiet.rs": ("fn add(a: u32, b: u32) -> u32 { a + b }\n", "")})
    findings, unsafe_counts = check_capabilities.scan(tmp_path)
    manifest = {
        "vendored_directories": [],
        "components": {"engine": {}},
        "files": {"engine/src/quiet.rs": {"component": "engine", "capabilities": {"network": "once fetched a page"}}},
        "unsafe": {"engine/src/quiet.rs": {"count": 2, "reason": "a count that no longer holds"}},
    }
    problems = check_capabilities.mismatches(manifest, findings, unsafe_counts, [])
    assert len(problems) == 2


def test_ignored_files_are_left_out(tmp_path: Path) -> None:
    plant(tmp_path, {"student/draft.py": ("import requests\n", "network")})
    (tmp_path / ".gitignore").write_text("target/\nstudent/draft.py\n", encoding="utf-8")
    assert check_capabilities.scan(tmp_path)[0] == []
