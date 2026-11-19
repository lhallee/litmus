# Litmus

Litmus is a free detector of AI written text that says how sure it is.

- **Using Litmus as a tool:** to check whether a text or code is AI-written from your own workflow, read the skill at
  [.agents/skills/litmus/SKILL.md](.agents/skills/litmus/SKILL.md) (what Litmus is, how to install the `litmus` program,
  `litmus mcp` and `litmus score --json`, how to read a result, and when not to trust it);
  [USE_WITH_AGENTS.md](docs/USE_WITH_AGENTS.md) has the setup for Claude Code, Codex, and Antigravity. A reading is
  evidence, never proof, and never an accusation against a person.
- **Auditing this repository:** [AUDIT.md](docs/AUDIT.md) lists what the project claims, where each claim is
  implemented, and the commands that check it; [SECURITY.md](SECURITY.md) says what each part does with the machine
  and the text.
- **Releasing it:** [release/SKILL.md](release/SKILL.md) is the procedure and `python release/cut.py status` says
  where a release stands; the publishing stages run only on the maintainer's word.
- **Changing it:** a new network call, process, file write, clipboard, registry, global-input, or browser-storage
  use, or a new `unsafe` in Rust, is declared with its reason in `tools/security/capabilities.json`, and a changed
  `Cargo.lock` is recorded with `python tools/security/check_dependencies.py --write` and its diff reviewed.
  `python tools/run_tests.py security` fails until both match the source.
