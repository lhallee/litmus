"""The agent skill, its plugin files, the install scripts, `litmus mcp`, and the pages that describe them agree."""

import json
import re

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SKILL_DIRECTORY = ROOT / ".agents" / "skills" / "litmus"
SKILL = SKILL_DIRECTORY / "SKILL.md"
DOCS = ROOT / "docs" / "USE_WITH_AGENTS.md"
CLIENTS = SKILL_DIRECTORY / "references" / "clients.md"
INTRODUCTION = "Litmus is a free detector of AI written text that says how sure it is"
AGENT_FILES = [SKILL, DOCS, CLIENTS, SKILL_DIRECTORY / "references" / "reading-results.md"]
ASSETS = ["litmus.exe", "litmus-linux-x86_64", "litmus-macos-arm64", "litmus-macos-x86_64"]


def frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """The top-level keys of a SKILL.md's frontmatter (flat `key: value` lines) and its body."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "the opening --- is the file's first line"
    header, body = text[4:].split("\n---\n", 1)
    fields = {}
    for line in header.splitlines():
        match = re.match(r"^([a-z-]+):\s*(.*)$", line)
        assert match, f"a frontmatter line that is not `key: value`: {line!r}"
        fields[match.group(1)] = match.group(2)
    return fields, body


def test_the_skill_follows_the_agent_skills_format() -> None:
    fields, body = frontmatter(SKILL)
    assert set(fields) <= {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}, "only fields every client accepts"
    assert fields["name"] == SKILL_DIRECTORY.name and re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"]) and len(fields["name"]) <= 64
    assert 0 < len(fields["description"]) <= 1024 and "<" not in fields["description"]
    assert 0 < len(fields["compatibility"]) <= 500
    assert len(SKILL.read_text(encoding="utf-8").splitlines()) < 500
    first = next(line for line in body.splitlines() if line.strip() and not line.startswith("#"))
    assert first.startswith(INTRODUCTION + "."), "the first line Litmus introduces itself with, verbatim"


def test_the_plugin_serves_the_one_copy_of_the_skill() -> None:
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "litmus" and [entry["name"] for entry in market["plugins"]] == ["litmus"]
    assert market["plugins"][0]["source"] == "./" and market["owner"]["name"] and market["name"] == "litmus"
    for directory in plugin["skills"]:
        assert directory.startswith("./") and (ROOT / directory / SKILL_DIRECTORY.name / "SKILL.md").is_file()
    workspace = (ROOT / "engine" / "Cargo.toml").read_text(encoding="utf-8")
    assert plugin["version"] == re.search(r'^version = "([^"]+)"', workspace, re.MULTILINE).group(1), "the plugin pins the program's version"
    assert plugin["description"].startswith(INTRODUCTION) and market["description"].startswith(INTRODUCTION)
    servers = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert servers == {"litmus": {"command": "litmus", "args": ["mcp"]}}


def test_every_relative_link_in_the_agent_pages_resolves() -> None:
    for page in AGENT_FILES + [ROOT / "AGENTS.md", ROOT / "PUBLIC_README.md"]:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", page.read_text(encoding="utf-8")):
            if "://" not in target:
                assert (page.parent / target).exists(), f"{page.relative_to(ROOT)} links to {target}, which is missing"


def test_the_registration_commands_agree_across_the_pages() -> None:
    docs, clients, skill = (path.read_text(encoding="utf-8") for path in (DOCS, CLIENTS, SKILL))
    shared = [
        "claude mcp add --scope user litmus -- litmus mcp",
        "claude plugin marketplace add lhallee/litmus && claude plugin install litmus@litmus",
        "codex mcp add litmus -- litmus mcp",
        '[mcp_servers.litmus]\ncommand = "litmus"\nargs = ["mcp"]',
        '{ "mcpServers": { "litmus": { "command": "litmus", "args": ["mcp"] } } }',
        "~/.gemini/config/skills/litmus",
        "~/.agents/skills/litmus",
    ]
    for text in shared:
        assert text in docs and text in clients, text
    assert "references/clients.md" in skill and "scripts/install.sh" in skill and "scripts/install.ps1" in skill


def test_the_pages_the_tools_and_the_scripts_name_the_same_things() -> None:
    server = (ROOT / "engine" / "litmus-cli" / "src" / "mcp.rs").read_text(encoding="utf-8")
    tool_names = set(re.findall(r'"name": "(litmus_check_[a-z]+)"', server))
    assert tool_names == {"litmus_check_text", "litmus_check_file"}
    cap = re.search(r"pub const MAX_TOKENS: usize = ([\d_]+);", server).group(1).replace("_", "")
    formatted = f"{int(cap):,}"
    for page in (SKILL, DOCS):
        text = page.read_text(encoding="utf-8")
        assert all(name in text for name in tool_names) and formatted in text, page.name
    scripts = (SKILL_DIRECTORY / "scripts" / "install.sh").read_text(encoding="utf-8") + (SKILL_DIRECTORY / "scripts" / "install.ps1").read_text(encoding="utf-8")
    releasing = (ROOT / "release" / "RELEASING.md").read_text(encoding="utf-8")
    for asset in ASSETS:
        assert asset in releasing and asset in DOCS.read_text(encoding="utf-8"), asset
        target = asset.removeprefix("litmus-")
        assert asset == "litmus.exe" and asset in scripts or target in scripts, f"the script builds {asset} from its target"


def test_what_the_pages_must_say_about_a_reading_is_said_in_each() -> None:
    required = ["never proof", "98.8%", "English", "accusation"]  # 98.8%: the release's English share of training rows (record.corpus.english_share)
    tool_text = (ROOT / "engine" / "litmus-cli" / "src" / "mcp.rs").read_text(encoding="utf-8")
    assert all(phrase in tool_text for phrase in required), "the server's instructions, guidance, and descriptions"
    # The pages state the English share and the accuracy through slots the release driver fills from the record (release/RELEASING.md),
    # so a typed number here would go stale; the server's own text above is Rust source and still carries its figure.
    for page in (SKILL, DOCS):
        text = page.read_text(encoding="utf-8")
        assert all(phrase in text for phrase in required if phrase != "98.8%"), page.name
        assert "<!--litmus:record.corpus.english_share" in text and "<!--litmus:t:accuracy_summary" in text, "the English share and the accuracy are slots, never typed numbers"
    skill = SKILL.read_text(encoding="utf-8")
    for phrase in ("Short texts are less reliable", "Code is weaker than prose"):
        assert phrase in skill


def test_the_new_prose_has_no_em_dash_emoji_or_spelled_out_percent() -> None:
    files = AGENT_FILES + [ROOT / "engine" / "litmus-cli" / "src" / "mcp.rs", ROOT / ".claude-plugin" / "plugin.json"]
    files += sorted((SKILL_DIRECTORY / "scripts").iterdir())
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "—" not in text, f"an em dash in {path.name}"
        assert not re.search("[\U0001f300-\U0001faff☀-➿]", text), f"an emoji in {path.name}"
        assert not re.search(r"\bpercent\b", text, re.IGNORECASE), f"the word percent in {path.name}"


def test_the_public_pages_do_not_advertise_the_check_by_its_method() -> None:
    readme = (ROOT / "PUBLIC_README.md").read_text(encoding="utf-8")
    section = readme.split("## Use it from an AI agent", 1)[1].split("\n## ", 1)[0]
    assert "USE_WITH_AGENTS.md" in section and not re.search(r"sha|hash|checksum", section, re.IGNORECASE)
    for path in (SKILL, DOCS, CLIENTS):
        text = path.read_text(encoding="utf-8")
        if path != DOCS:
            assert not re.search(r"sha-?256|hash", text, re.IGNORECASE) or path == SKILL and "published file list" in text
