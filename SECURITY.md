# Security

This page says what each part of Litmus does and does not do with your machine, your files, and your text, in
claims you can check. Each claim names the code that implements it and the check that would catch it if it stopped
being true. [AUDIT.md](docs/AUDIT.md) gives the commands in order, for a person or an AI agent reviewing the repository.

Two checks back most of the claims below. `tools/security/check_capabilities.py` scans every source file for network
access, starting processes, the clipboard, the Windows registry, global input, reading other applications' text,
writing files, browser storage, and loading code or pickled data, and fails when any of them appears in a file that
[`tools/security/capabilities.json`](tools/security/capabilities.json) does not declare, with its reason. The same file records
how many times each Rust file uses `unsafe`. `tools/security/check_dependencies.py` fails when a Rust dependency changes
without a matching change to [`tools/security/dependencies.json`](tools/security/dependencies.json), when one is a network crate,
or when its license is not one the project accepts. Both run with `python tools/run_tests.py security`.

## What each part does

### The engine, the command line, and the Python module (`engine/`)

- **No network code.** No file under `engine/` is declared for network access, and `engine/Cargo.lock` holds no HTTP,
  TLS, socket, or async-runtime crate. Check: the capability and dependency checks above.
- **Files it reads:** the files and folders named on the command line (or standard input), the release folder
  (`--model` or `LITMUS_MODEL`), and `tokenizer.json` from the release folder, `LITMUS_TOKENIZER`, or the local
  Hugging Face cache folder. It never downloads the tokenizer; it checks its SHA-256 against the release's
  `config.json` (`engine/litmus-files/src/release.rs`).
- **Files it writes:** a cache of readings, one JSON file per reading, in `%LOCALAPPDATA%\litmus\cache` on Windows,
  `~/Library/Caches/litmus` on macOS, or `$XDG_CACHE_HOME/litmus` on Linux, or `LITMUS_CACHE_DIR`; `--no-cache` turns
  it off (`engine/litmus-files/src/cache.rs`). Output goes to standard output.
- **Processes it starts:** one, itself, with `__pdf-text <file>`, to read a PDF's text in a child process so that a
  malformed PDF cannot crash the run or print into its output (`engine/litmus-files/src/extract.rs`).
- **`litmus mcp`** serves the Model Context Protocol to an AI agent on standard input and output
  (`engine/litmus-cli/src/mcp.rs`). It reads the texts the agent sends, the one file path a `litmus_check_file` call
  names (any readable path, as the agent's own file tools can read), and the model. It writes nothing (no cache, no file),
  starts a process only for a PDF, as above, makes no network call, and writes nothing but protocol messages to standard
  output. It refuses a text over 131,072 tokens, a file over 32 MiB, and a message over 48 MiB. Check: the capability
  check finds no network, file-writing, or process code in the file, and its tests drive it through the protocol.
- **`unsafe` Rust:** SIMD intrinsics in the kernels, used only after a runtime CPU check, with a scalar path that the
  tests require to give the same bits; a counting allocator for `litmus bench`; three Windows console calls that turn
  on color. Every file and count is in `tools/security/capabilities.json`.

### The agent skill and its install scripts (`.agents/skills/litmus/`, `.claude-plugin/`, `.mcp.json`)

- **The skill and the plugin files are text.** `SKILL.md`, its references, the plugin and marketplace manifests, and
  `.mcp.json` (which registers `litmus mcp` for Claude Code) run nothing themselves; the agent that reads them decides what
  to run.
- **The install scripts** (`scripts/install.sh`, `scripts/install.ps1`) download two files over HTTPS from this
  repository's GitHub releases (`github.com/lhallee/litmus`, the latest release or the tag given, nothing else), the
  program for the computer and the release's `SHA256SUMS`. Each computes the program's SHA-256, compares it with the line
  of `SHA256SUMS` that names it, and installs nothing on a difference. They write only the program, to the user's folder
  (`~/.local/bin` or `%LOCALAPPDATA%\Litmus\bin`, or the one named), through a staged file in the same folder that is moved
  into place; they need no elevation and change no shell startup file or setting. The one exception is `install.ps1
  -AddToPath`, which adds the folder to this user's `PATH` (`HKCU\Environment`). Both run the installed program once with
  `--help`. `SHA256SUMS` comes from the same release as the program, so the check catches a damaged or swapped file and
  not a release that was itself altered; `tools/verify_build.py` is the check for that. Check: `tools/security/capabilities.json`
  declares each script's network use, and `tools/security/tests/test_agent_install.py` installs from a local folder and
  requires the refusals.

### The desktop app (`app/`)

- **No network code.** The only network call in the app, `app/src/contribute.rs`, compiles only into the contribute
  site's browser build (`#[cfg(all(target_arch = "wasm32", feature = "contribute"))]`); the desktop build has none,
  and `app/Cargo.lock` holds no network crate. Check: the capability and dependency checks.
- **Files it reads:** the files you open, drop, or check as a batch; the release folder; on Windows, the Segoe UI and
  Cascadia Mono or Consolas fonts from the system's font folder.
- **Files it writes:** its data folder, `%APPDATA%\litmus` on Windows, `~/Library/Application Support/litmus` on
  macOS, or `$XDG_DATA_HOME/litmus` on Linux, or `LITMUS_DATA_DIR`: `settings.json`, the history of readings (off
  in Settings to stop keeping it), a lock file that keeps one app open at a time, and a folder through which a second
  launch hands its file to the first. The reading cache is shared with the command line. Exports go only to the file
  you choose in the save dialog.
- **Quick check is off by default** (`quick_check: false` in `app/src/settings.rs`, asserted by its test). When you
  switch it on in Settings, and only then, the app:
  - registers one global hotkey (Ctrl+Alt+L unless you change it; it must include Ctrl, Alt, or the Windows key) and
    a tray icon (`app/src/quick/mod.rs`);
  - on the hotkey, reads the selected text of the foreground app through Windows UI Automation, and when that app
    offers none, saves every format on the clipboard, simulates Ctrl+C, reads the text, and puts the saved clipboard
    back, marking its own write to stay out of clipboard history (`app/src/quick/windows.rs`);
  - adds a "Check with Litmus" entry to Explorer's right-click menu only if you ask for it in Settings, under
    `HKEY_CURRENT_USER\Software\Classes\SystemFileAssociations\.<ext>\shell\LitmusCheck`, removed again from Settings.
    No other registry key is written.
  - on Linux and macOS, takes the place of the Windows route above with the platform's own command-line tools
    (`app/src/quick/unix.rs`). On Linux, `wl-paste` or `xclip` reads the primary selection, which is the selected text
    itself, so nothing is copied and the clipboard is left alone. On macOS, `osascript` presses Copy through System
    Events, which needs the Accessibility permission, and `pbpaste` and `pbcopy` save the clipboard's text and put it
    back, while any other kind of clipboard content is not saved.
  The selected text is scored on the machine like any other text.
- **`unsafe` Rust:** the Win32 calls of quick check, one call that lets the running app take the focus, and two that
  place the window on an existing screen.

### The browser page (`site/app/` and the app's WebAssembly build)

- **Your text never leaves the page.** The page loads its own WebAssembly module (`site/app/index.html`), and its
  worker makes three same-origin GET requests for the model's files, `model.safetensors`, `config.json`, and
  `tokenizer.json` (`site/app/worker.js`, function `getAll`). Scoring runs in the worker. The page is built without
  the `contribute` feature (`app/scripts/build_web.sh` without `--contribute`), so it compiles no code that sends a
  request. Check: the capability check, and your browser's network panel, which should show only those requests.
- **Browser storage:** the page keeps its settings (theme and detail level) in its own `localStorage`
  (`app/src/settings.rs`); history stays in the page's memory.
- **Not in this repository:** the JavaScript glue `litmus_app.js` that wasm-bindgen 0.2.126 generates at build time.
  Rebuild the page with `app/scripts/build_web.sh` to read it.

### The contribute site (`contribute/`)

The contribute site is the one part of Litmus to which a text is sent. Its page sends a text only after the visitor
ticks a box agreeing to the Terms of Use and the Privacy Notice (the server refuses a text without
`"consent": true`), and its removal form sends `POST /api/removals`. The
page posts the text to the server, which scores it and stores, per row, the text, its SHA-256, the reading, the
model version, the visitor's optional label and its time, the receipt time, and the terms version. It does not store
or log the visitor's address, user agent, cookies (it sets none), referrer, or any account; rate limits key on a
salted HMAC of the address held in memory only. Stored rows are merged into a private Hugging Face dataset, and a
removal request deletes every stored copy of a text at the next merge. [`contribute/README.md`](contribute/README.md)
has the details, and `contribute/tests/test_privacy.py` checks that no address, user agent, cookie, or referrer
reaches storage or logs. Its terms of use and privacy notice are in `contribute/legal/`.

### Training and evaluation (`corpus/`, `teacher/`, `student/`, `benchmarks/`, `launch/`, `training/foundry/`)

This code runs on the trainer's own machine, in Docker. It downloads datasets and models from the Hugging Face Hub
at pinned revisions, logs runs to Weights and Biases, and uploads its outputs to private Hugging Face repositories;
`tools/security/capabilities.json` lists each file that does so. It reads credentials (`HF_TOKEN`, `WANDB_API_KEY`) from
the environment, or from a local secrets file that `training/foundry/secrets_env` loads into the environment without printing
any value, and the repository holds none. It loads PyTorch checkpoints, which are pickle files and can run code, so
we recommend giving it only checkpoints a run of your own wrote; the files that load them are declared.
`training/foundry/` is a vendored library of training helpers, and it is scanned like the rest.

## What these checks do not cover

- **The scan reads text.** It finds direct uses; code generated at build time, a capability reached through a call
  into a declared file, or one hidden by string building would pass it. Read the declared files.
- **Dependencies are checked by name, version, and license, not by content.** A crate could misbehave inside its
  license. `Cargo.lock` pins each crate's checksum, and `cargo build --locked` refuses a changed one.
- **The model file is an opaque blob.** `tools/verify_model.py` checks that a release's files match the hashes the
  release states; it cannot say what the weights compute. The engine reads `model.safetensors` as plain tensors,
  never as code.
- **Prebuilt binaries** are trustworthy only once you have rebuilt them and compared hashes:
  `python tools/verify_build.py --model <model folder> --sums <SHA256SUMS>` rebuilds `litmus-app.exe` and
  `litmus.exe` from the release's commit and says MATCH when they are the published files. See
  [Rebuild and compare hashes](docs/AUDIT.md#rebuild-and-compare-hashes) and its limits.

## Reporting a vulnerability

Report it privately through GitHub's private vulnerability reporting on this repository (the Security tab, then
"Report a vulnerability"), not in a public issue. Include what you found, how to reproduce it, and which version or
commit it affects.
