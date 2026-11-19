# Offering the Litmus tools to an agent

`litmus mcp` serves two tools, `litmus_check_text` and `litmus_check_file`, over the Model Context Protocol on standard
input and output. Register it once per client after `litmus` is installed. Use the full path the install script printed
when `litmus` is not on `PATH`. The page [docs/USE_WITH_AGENTS.md](../../../../docs/USE_WITH_AGENTS.md) has the same
steps with their sources and what was tested.

## Claude Code

```sh
claude mcp add --scope user litmus -- litmus mcp          # the tools, for every project
claude plugin marketplace add lhallee/litmus && claude plugin install litmus@litmus   # the skill and the tools together
```

Without the plugin, copy this skill's folder to `~/.claude/skills/litmus` (Claude Code reads `.claude/skills`, not
`.agents/skills`). In a project, `claude mcp add --scope project` writes `.mcp.json`.

## OpenAI Codex CLI

```sh
codex mcp add litmus -- litmus mcp
```

or in `~/.codex/config.toml`:

```toml
[mcp_servers.litmus]
command = "litmus"
args = ["mcp"]
```

Codex reads skills from `.agents/skills` in a repository and from `~/.agents/skills` for the user: copy this skill's
folder to `~/.agents/skills/litmus`. The tools are marked read-only, so a `writes` approval mode runs them without a prompt.

## Google Antigravity

Add to `.agents/mcp_config.json` in the workspace, or `~/.gemini/config/mcp_config.json` for every workspace:

```json
{ "mcpServers": { "litmus": { "command": "litmus", "args": ["mcp"] } } }
```

Skills go in `.agents/skills/litmus` in the workspace, or `~/.gemini/config/skills/litmus` for every workspace.
Antigravity's documentation has listed different folders for its app, IDE, and CLI, so check the client's own skills and
MCP panels after a restart.

## Any other client

Run `litmus mcp` as a stdio server: no arguments beyond the optional `--model DIR`, newline-delimited JSON-RPC on standard
input and output, logs on standard error. It speaks protocol 2026-07-28 (no handshake, `server/discover`), and
2025-11-25 and 2025-06-18 for clients that open with `initialize`. A generic entry:

```json
{ "mcpServers": { "litmus": { "command": "litmus", "args": ["mcp"] } } }
```

A client with no MCP support can call `litmus score --json --no-cache <file | ->` and read the JSON.
