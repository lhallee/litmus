# Use Litmus from an AI agent

Litmus is a free detector of AI written text that says how sure it is. This page is for a person setting up a coding
agent (Claude Code, OpenAI Codex CLI, Google Antigravity, or another client of the Model Context Protocol) to check text
or code with it, and for an agent given only this repository's address that has to install it and use it. The
agent-facing instructions are the skill at [`.agents/skills/litmus/SKILL.md`](../.agents/skills/litmus/SKILL.md); this page
gives the setup for each client, what each piece does with the machine, and what was tested.

A reading is evidence, never proof. It is probabilistic and comes with the origins it cannot rule out. Short texts are
less reliable, and so is code. This release is an English detector: it is trained almost entirely on English (<!--litmus:record.corpus.english_share as pct1-->98.8%<!--/litmus--> of its training text) and tested mainly on English text, so other
languages are not supported yet. We recommend not presenting a reading as an accusation against a person. The accuracy
of the release model is as follows: <!--litmus:t:accuracy_summary-->on held-out test texts it wrongly called 7.4% of human texts not human at the 5% level, and it caught 67.2% of AI texts when 5% of human texts are wrongly flagged (53.0% at 1%)<!--/litmus-->.

## What there is

| Piece | Where | What it is |
| --- | --- | --- |
| `litmus` | A release's program, or `engine/` built from source | The command line, with the model built in. `litmus score --json` reads files and standard input; `litmus mcp` serves the same readings to an agent |
| The skill | [`.agents/skills/litmus/`](../.agents/skills/litmus/SKILL.md) | What Litmus is, how to install and call it, how to read a result, and when not to trust it. One copy for every client |
| The install scripts | [`.agents/skills/litmus/scripts/`](../.agents/skills/litmus/scripts/install.sh) | `install.sh` (Linux, macOS) and `install.ps1` (Windows) |
| The Claude Code plugin | [`.claude-plugin/`](../.claude-plugin/plugin.json), [`.mcp.json`](../.mcp.json) | The skill and the MCP registration as one installable plugin from this repository |

## 1. Install the program

An agent runs one of these from a clone of the repository, or after downloading the script and reading it:

```sh
sh .agents/skills/litmus/scripts/install.sh                                                  # Linux, macOS
powershell -NoProfile -ExecutionPolicy Bypass -File .agents\skills\litmus\scripts\install.ps1   # Windows
```

Each script downloads the program for the computer and the release's published file list from this repository's GitHub
releases, installs nothing unless the program matches the list, installs for the current user without elevation (to
`~/.local/bin/litmus` or `%LOCALAPPDATA%\Litmus\bin\litmus.exe`), changes no shell startup file, and prints what it did,
including the full path to call. Options: `--dir` and `--version` (`-Dir`, `-Version`), `--from-dir` (`-FromDir`) to
install from a folder holding the program and `SHA256SUMS` with no network, and `-AddToPath` on Windows to add the folder
to this user's `PATH`. The program has the model built in, so it is the whole install.

A release attaches `litmus.exe` for Windows x86-64, and `litmus-linux-x86_64`, `litmus-macos-arm64`, and
`litmus-macos-x86_64` for the others, with one `SHA256SUMS` naming each ([`release/RELEASING.md`](../release/RELEASING.md),
step 4). Until a release publishes the program for a system, the script says so and the build from source applies. The
macOS program is built and run by the release workflow's macOS runners and untested otherwise.

**From source**, with Rust (rustup installs the toolchain `engine/rust-toolchain.toml` pins) and a release folder holding
`model.safetensors`, `config.json`, and `tokenizer.json` (<!--litmus:t:hub_model_link-->https://huggingface.co/lhallee/litmus<!--/litmus-->):

```sh
cd engine && cargo build --release --locked          # engine/target/release/litmus
engine/target/release/litmus mcp --model <release folder>
```

## 2. Give the agent the tools and the skill

`litmus mcp` serves two tools over the Model Context Protocol on standard input and output, both marked read-only:

| Tool | Arguments | Returns |
| --- | --- | --- |
| `litmus_check_text` | `text` (up to 131,072 tokens), `detail` (`"summary"` or `"full"`) | One sentence, and in `structuredContent` the `label`, `n_tokens`, `words`, `characters`, `caveats`, a standing `guidance` sentence, and the `reading` |
| `litmus_check_file` | `path` (one file), `detail` | The same, plus `file`. Reads documents (`.txt`, `.md`, `.html`, `.docx`, `.pdf`) as `litmus score` does and any other file as plain text, so source code works |

`reading` is the `reading` of `litmus score --json`, whole at `"full"` and its `verdict` with the token counts at
`"summary"` (the default, since a long text's heat map is large). `words` and `characters` count the text that was read
(`characters` are Unicode scalar values, and a word is a run of characters between spaces or line breaks, so a language
written without spaces counts as one word per run); `litmus score --json` gives the same numbers, and the characters
without spaces, in its `counts` object. A bad argument, an empty text, a text over the cap, or a
file that cannot be read comes back as a tool error (`isError: true`) the agent can read; an unknown tool or a malformed
request is a protocol error. The server loads the model when it starts; if that fails it still answers, and each check
says why it cannot read. Register it once per client; use the full path the install script printed where `litmus` is not
on `PATH`.

### Claude Code

```sh
claude mcp add --scope user litmus -- litmus mcp                                       # the tools
claude plugin marketplace add lhallee/litmus && claude plugin install litmus@litmus     # the skill and the tools
```

The plugin is this repository: `.claude-plugin/marketplace.json` lists one plugin whose source is the repository root,
`plugin.json` points its skills at `.agents/skills/` (so the one copy serves every client), and `.mcp.json` registers
`litmus mcp`. Claude Code names the tools `mcp__plugin_litmus_litmus__litmus_check_text` and
`mcp__plugin_litmus_litmus__litmus_check_file` when installed as the plugin, and `mcp__litmus__...` when registered with
`claude mcp add`. Do both and you get the tools twice. Without the plugin, copy `.agents/skills/litmus` to
`~/.claude/skills/litmus`, since Claude Code reads skills from `.claude/skills`. Inside a clone of this repository,
`.mcp.json` also offers the server to Claude Code, which asks before using a project server.

### OpenAI Codex CLI

```sh
codex mcp add litmus -- litmus mcp
mkdir -p ~/.agents/skills && cp -r .agents/skills/litmus ~/.agents/skills/litmus        # the skill, for every project
```

or, in `~/.codex/config.toml` (or `.codex/config.toml` in a trusted project):

```toml
[mcp_servers.litmus]
command = "litmus"
args = ["mcp"]
```

Codex finds skills in `.agents/skills` of the repository it runs in, so inside a clone no copy is needed. It prompts for
tools not marked read-only in the `writes` approval mode (`default_tools_approval_mode`); these tools are marked
read-only. The default `startup_timeout_sec` of 10 seconds is ample, since the engine loads quickly
([engine README](../engine/README.md#command-line)). A text near the 131,072-token cap takes seconds on one core, and on a
slow computer it can pass Codex's default `tool_timeout_sec` of 60, which you can raise if your agent checks texts that
long.

### Google Antigravity

Antigravity reads MCP servers from `.agents/mcp_config.json` in the workspace and `~/.gemini/config/mcp_config.json`
for every workspace, in the same `mcpServers` form:

```json
{ "mcpServers": { "litmus": { "command": "litmus", "args": ["mcp"] } } }
```

It reads skills from `.agents/skills/` in the workspace, so inside a clone no copy is needed; for every workspace copy
`.agents/skills/litmus` to `~/.gemini/config/skills/litmus`, the one global folder that the app, the IDE, and the CLI all
read in one user's tests. Antigravity's own pages list a different global folder for each of the three, and a user's test
found them out of date, so confirm in the client's skills and MCP panels after a restart.

### Any other client

Run `litmus mcp` as a stdio server with no arguments (or `--model DIR`) and the generic entry above. It reads
newline-delimited JSON-RPC from standard input, writes only messages to standard output, logs to standard error, and exits
when standard input closes. A client without MCP can call `litmus score --json --no-cache <file | ->`.

## What the server does with the machine

It makes no network call, writes no file, and keeps no cache (`litmus score` keeps one unless given `--no-cache`). It reads
the text it is sent, the one file named in `litmus_check_file`, and the model. A PDF is read in a child process, the same
as `litmus score`. It refuses a text over 131,072 tokens (the longest the engine is measured at), a file over 32 MiB, and
a message over 48 MiB, none of which `litmus score` limits. The tool takes any readable path, as the agent's own file
tools do, so grant it as you would them. `tools/security/capabilities.json` declares each of these;
[SECURITY.md](../SECURITY.md) and [AUDIT.md](AUDIT.md) say what to check.

## What was tested, and what was not

- `litmus mcp` is driven by a scripted client through both protocol revisions, a good call, bad calls, malformed input, an
  oversized line, and a release that does not load, and each tool's reading equals `litmus score --json`'s on the same text
  (`engine/litmus-cli/src/mcp.rs` tests, always run; `engine/litmus-cli/tests/mcp.rs` on a release folder, run with
  `LITMUS_MODEL=<release folder> cargo test --release --features release-tests -p litmus-cli --test mcp`).
- The install scripts are driven with `--from-dir` and `-FromDir` against stand-in programs: a match installs, a changed
  program, a list that omits it, and a bad version or target install nothing, and a replaced program is replaced
  (`tools/security/tests/test_agent_install.py`, in `python tools/run_tests.py security`).
- Not tested: an install that downloads from a published release, macOS and Linux runs of
  `install.sh` beyond `sh` on Windows, `install.ps1 -AddToPath`, the plugin and marketplace files with `claude plugin
  validate`, and any of Claude Code, Codex, and Antigravity as a live client of `litmus mcp`. The server answers the
  protocol the documents below specify, and it serves the older `initialize` handshake because most clients still open
  with it.

## Sources, read 2026-11-18

- MCP specification 2026-07-28: <https://modelcontextprotocol.io/specification/2026-07-28/basic.md> (messages,
  `_meta`, statelessness, error codes), [versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning.md)
  (dual-era servers, `UnsupportedProtocolVersion`), [stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio.md),
  [server/discover](https://modelcontextprotocol.io/specification/2026-07-28/server/discover.md),
  [tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools.md), and its
  [changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog.md): this revision removes `initialize`
  and `ping`, and adds `server/discover` and a `resultType` on every result.
- The `initialize` handshake of 2025-11-25: <https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle.md>.
- Agent Skills: <https://agentskills.io/specification> (`SKILL.md`, `name` and `description` limits), Claude Code's
  <https://code.claude.com/docs/en/skills> (`.claude/skills`, plugin `skills/`), and Codex's
  <https://learn.chatgpt.com/docs/build-skills> (`.agents/skills`, `~/.agents/skills`).
- Claude Code plugins: <https://code.claude.com/docs/en/plugins-reference> and
  <https://code.claude.com/docs/en/plugin-marketplaces>; MCP: <https://code.claude.com/docs/en/mcp>.
- Codex MCP: <https://learn.chatgpt.com/docs/extend/mcp?surface=cli>.
- Antigravity: Google's <https://antigravity.google/docs/migration/workflows-to-skills/> (workspace `.agents/skills`,
  global `~/.gemini/config/skills`, `SKILL.md` with `name` and `description`), and one user's tests of the skill and MCP
  paths, <https://atamel.dev/posts/2026/07-01_where_agy_agent_skills/> and
  <https://atamel.dev/posts/2026/07-10_where_agy_mcp_servers/>, which are not Google's pages. Google's page for MCP
  configuration was not found, so the `mcp_config.json` locations rest on that user's account.
