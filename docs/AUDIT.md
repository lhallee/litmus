# Auditing Litmus

This file is for anyone, person or AI agent, checking whether Litmus does what it says with a user's machine and
text. It lists the claims, where each is implemented, and the commands that test them. It states no conclusions:
run the checks, read the code they point to, and judge for yourself. With Python 3.11 or later, pytest, and Rust
installed, the first three checks take under ten minutes; the rebuild takes longer.

## 1. The claims

1. The engine, the `litmus` command line, and the `litmus_engine` Python module contain no network code.
2. The desktop app contains no network code.
3. The browser page sends no text anywhere. Its only requests are for its own files: the WebAssembly module and the
   model's three files, from the page's own address. The site page that holds it makes one more request when the
   release names a counter, a count of the page's opening that carries no text and is never made on a check.
4. The desktop app touches the clipboard, the Windows registry, global hotkeys, and other applications' text only
   for quick check, which is off until the user switches it on, plus a clipboard copy the user asks for in the
   browser build.
5. The files each program writes are the ones [SECURITY.md](../SECURITY.md) lists.
6. Every `unsafe` block in the Rust code is counted and explained.
7. The Rust dependencies are pinned, none is a network crate, and each one's license is one the project accepts.
8. Two builds of the same source with the pinned toolchain on the same platform are byte-identical: the command line
   wherever the source sits on disk, and the app and its WebAssembly module when built at the same path.
9. The contribute site, the one place a text is sent, stores the text, its reading, and the label, and never the
   sender's address, user agent, cookies, or referrer.
10. `litmus mcp`, which serves AI agents, makes no network call and writes no file, and gives the reading `litmus score
    --json` gives. The agent skill's install scripts download only from this repository's GitHub releases, check the
    program against the release's published file list before installing it, and write only the program.

## 2. Where each claim lives

| Claim | Code | Check |
| --- | --- | --- |
| 1, 2, 4, 5, 6 | Every file declared in [`tools/security/capabilities.json`](../tools/security/capabilities.json), with its reason | `tools/security/check_capabilities.py` |
| 3 | [`site/app/worker.js`](../site/app/worker.js) (`getAll`), [`site/app/index.html`](../site/app/index.html), [`app/src/contribute.rs`](../app/src/contribute.rs) (compiled only with feature `contribute`), [`app/scripts/build_web.sh`](../app/scripts/build_web.sh); the count is [`site/site.js`](../site/site.js) (`countOpening`) and `page_counter` in [`site/assets/links.json`](../site/assets/links.json) | the capability check; a browser's network panel on the built page |
| 4 | [`app/src/settings.rs`](../app/src/settings.rs) (`quick_check: false` and its test), [`app/src/quick/`](../app/src/quick/mod.rs) | the capability check; `app` tests |
| 7 | [`engine/Cargo.lock`](../engine/Cargo.lock), [`app/Cargo.lock`](../app/Cargo.lock), [`tools/security/dependencies.json`](../tools/security/dependencies.json), [`engine/THIRD_PARTY.md`](../engine/THIRD_PARTY.md) | `tools/security/check_dependencies.py` |
| 8 | [`engine/rust-toolchain.toml`](../engine/rust-toolchain.toml), [`app/rust-toolchain.toml`](../app/rust-toolchain.toml), [`tools/verify_build.py`](../tools/verify_build.py) (`release_rustflags`) | `tools/verify_build.py` |
| 9 | [`contribute/src/litmus_contribute/web.py`](../contribute/src/litmus_contribute/web.py), [`contribute/README.md`](../contribute/README.md) | `contribute/tests/test_privacy.py` |
| 10 | [`engine/litmus-cli/src/mcp.rs`](../engine/litmus-cli/src/mcp.rs), [`.agents/skills/litmus/scripts/install.sh`](../.agents/skills/litmus/scripts/install.sh), [`install.ps1`](../.agents/skills/litmus/scripts/install.ps1) | the capability check; `cargo test -p litmus-cli`; `tools/security/tests/test_agent_install.py`; `LITMUS_MODEL=<release folder> cargo test --release --features release-tests -p litmus-cli --test mcp` |

## 3. The checks

Run these from the repository's root.

### Capabilities and dependencies (about 10 seconds, Python standard library only)

```sh
python tools/security/check_capabilities.py             # fails on any undeclared or stale capability, or a changed unsafe count
python tools/security/check_capabilities.py --report     # every finding, declared or not, as JSON
python tools/security/check_dependencies.py              # fails on an unlisted crate, a network crate, or an unaccepted license
```

Read the patterns at the top of `tools/security/check_capabilities.py` before trusting its result, and read each declared
file's matching lines. The checker's own tests plant a network call, a process spawn, synthetic input, and a
pickle load in a temporary tree and require the scan to find them: `python -m pytest tools/security/tests`.

### Search for network calls yourself (about 1 minute)

Do not rely on the checker alone. With ripgrep:

```sh
rg -n 'std::net|TcpStream|UdpSocket|reqwest|ureq|hyper::|fetch_with|web_sys::Request' engine app
rg -n 'fetch\(|XMLHttpRequest|WebSocket|sendBeacon|EventSource|importScripts' site app
rg -n '^\s*(import|from)\s+(requests|httpx|urllib|socket|aiohttp|http\.client)' --type py
rg -n 'name = "(reqwest|hyper|ureq|h2|tokio|rustls|native-tls|openssl|socket2)"' engine/Cargo.lock app/Cargo.lock
rg -n 'https?://' .agents/skills/litmus/scripts                  # the install scripts name github.com/lhallee/litmus and nothing else
rg -n 'fs::|File::|OpenOptions|Command::|std::net' engine/litmus-cli/src/mcp.rs   # only reads of the one named file, and none of the others
```

### The test suites (2 to 10 minutes for the Rust and security suites)

```sh
python tools/run_tests.py --list          # what each suite proves
python tools/run_tests.py security engine app
python tools/run_tests.py                 # every suite but the rebuild; the Python suites need each package's dependencies
```

### Rebuild and compare hashes

```sh
python tools/verify_build.py              # the command line and the WebAssembly module, each built twice
python tools/verify_build.py --artifacts app
python tools/verify_build.py --model <release model folder> --sums <the release's SHA256SUMS>   # rebuild litmus-app.exe and litmus.exe and compare with the release
```

With `--model` and `--sums` the script rebuilds the release's two programs with its model built in, at the fixed
path, through the same functions `release/build_release.py` uses, and ends with `MATCH: ...` when both are the
published files byte for byte; without them it builds each artifact twice and compares the two.

The script builds each artifact twice with the toolchain that `rust-toolchain.toml` pins and the flags in
`release_rustflags`, and compares SHA-256 hashes. The command line is built from copies of `engine/` and `app/` at
two different paths. The app and the WebAssembly module are built twice from fresh copies at one fixed path
(`C:\litmus-build` on Windows, `/tmp/litmus-build` elsewhere, or `--fixed-root`), because Cargo hashes the absolute
path of the engine's crates, which sit outside the app's Cargo workspace, into symbol names, and that changes the
code's layout; the script's docstring explains this. When
`release/SHA256SUMS` lists a release's artifacts, it compares against those too; when it lists none, the script says
it compared two local builds. To check a model release's files against its stated hashes:

```sh
python tools/verify_model.py path/to/release --sums release/SHA256SUMS
```

## 4. Out of scope

- The Python training, corpus, and benchmark code runs on the trainer's machine with network access by design; the
  capability check lists what it downloads and uploads, and nothing here constrains it further.
- Quick check outside Windows (`app/src/quick/unix.rs`): the Linux path was run on X11 under a virtual display,
  and the macOS path is compiled and tested by the release workflow and has never been run on a Mac.
- Reproducibility is verified on Windows x86-64 only (see [DEVELOPING.md](DEVELOPING.md#verify-the-build) for what is
  measured). Other platforms are untested.
- What the model computes: a hash check proves which file you have, not what it does.

## 5. What to distrust until you have checked it

- **Any prebuilt binary**, including one attached to a release, until
  `tools/verify_build.py --model <model folder> --sums <SHA256SUMS>` prints MATCH for it from this source at the
  release's commit.
- **The model file**, `model.safetensors`, which is an opaque blob of tensors. Check its hash with
  `tools/verify_model.py`; the engine parses it as safetensors and never executes it.
- **Generated code not in this repository**: the wasm-bindgen glue `litmus_app.js` of the browser page.
- **The checkers themselves**: they are code in this repository like any other. Read them.
- **This file and SECURITY.md**: they are claims. The checks above are what test them.
