# Cutting a release

A release is one trained model, named by its Hugging Face repository and commit, turned into everything Litmus
publishes: the Windows, Linux, and macOS programs, the browser page on GitHub Pages, the model's card on the Hub, the
numbers in every document, and the ad. One driver, `release/cut.py`, runs it in stages and remembers what is done, so a
person or an agent with no other context follows this page to the end.

```sh
python release/cut.py run --work <work folder> --model <owner/name>@<40-character commit> --version <X.Y.Z>
python release/cut.py status --work <work folder>
```

The first call writes the plan (the model, the version, the date) into the work folder, which must be outside the
repository, and runs every stage it can. Every later call reads the plan; a different `--model` or `--version` for the
same work folder is refused. A stage that has run is skipped until something it read changes, and `status` says which
stages are done, stale (and why), or waiting. A stage that cannot run on this machine stops the run with exit code 3 and
says what to do; any other failure exits with 1 and names the file or number at fault.

Nothing in this driver sends anything anywhere except the three gated stages (publish, macos, pages), and they run only
with `--confirm`, which is the person's word and never an agent's choice. `--dry-run` prints what each would do and
writes no receipt.

## What a release starts from

1. **A model on the Hub, at a commit.** The model repository holds the release folder (`model.safetensors`,
   `config.json`, `tokenizer.json`, as the student's export writes them) and a record folder, which is how every
   number about training and evaluation reaches the documents:

   | In the record folder | What |
   | --- | --- |
   | `record.toml` | `[identity] weights_sha256`, the corpus, split, and training facts the documents quote (`python release/cut.py record-template` lists every key), and `[benchmarks] litmus_trained_on` (the benchmarks whose generators or rows the model trained on, which the results tables mark) |
   | `evaluation_test.json`, `evaluation_test.csv` | The test evaluation from the student's `evaluate` |
   | `evaluation_validation.csv` | The validation evaluation, when the documents quote it |
   | `teacher_metrics.csv`, `teacher_uncertainty.csv` | The teacher's calibration, for a model that has a teacher; a labels-only build has none, and the documents' teacher rows drop out |
   | other `.json`, `.toml`, `.csv` | Any further file a document's slot names, by its stem (`corpus_report.json` is `corpus_report`) |

   `python release/cut.py bundle --release <release folder> --out <new folder> --record <record.toml> --evaluation-test
   <folder> [--evaluation-validation <folder>] [--teacher <folder>] [--file name=path ...]` gathers these into
   `release/` and `record/` folders to upload with the model, and ties `record.toml` to the weights by their SHA-256. The
   `fetch` stage refuses a record that states other weights.
2. **The stored baselines.** The other detectors are scored once, on a GPU machine the person names, and reused by every
   release. Their score files and cost rows live in a private Hub repository at a pinned commit that
   `release/baselines.toml` names; `fetch` refuses an empty pin and says what to upload. To add a benchmark or a
   detector, score it, upload its files, and pin the new commit there.
3. **A version.** `MAJOR.MINOR.PATCH`, chosen by the person. See [Versions and the Hub](#versions-and-the-hub).
4. **Machines.** The driver runs on the machine it is started on and never connects to another. The stages that need
   Windows, Linux, or macOS say so in `status`; the person starts them on a machine of that kind (any machine that is
   theirs and named for the purpose; an agent asks which) and brings the result back with `adopt`. Scoring and
   measuring are CPU only. Nothing here needs a GPU, and none of it needs the workspace's `ws` command: the public
   repository runs this page as written.

## The stages

| Stage | Where | What it does | Waits on |
| --- | --- | --- | --- |
| `fetch` | any | Downloads the release folder, the record folder, and the pinned baselines; checks every hash | |
| `record` | any | Checks the record against every `record.*` key the documents read, and names each missing one | fetch |
| `version` | any | Sets the release version in every manifest of the checkout, or checks it ([below](#versions-and-the-hub)) | |
| `score` | any | Scores Litmus with the shipped engine on the rows the baselines read, and checks the rows are the same | fetch |
| `cost` | any | Measures Litmus's cost per 1,000 words on one core, for the cost table | fetch |
| `results` | any | Writes `results.json`, the comparison tables, and the figures | record, score, cost |
| `bench` | any | Runs `litmus bench` for speed and memory on one core | fetch |
| `browser` | by hand | The browser page's speed, a person's measurement with a real browser, adopted ([below](#the-browser-page)) | bench |
| `docs` | any | Fills every slot of every document and the site's data from the data files | record, results, bench, browser |
| `film` | any | Copies the film's README preview into `docs/images`, once the film's record names this release's weights | fetch |
| `build-windows` | Windows | `release/build_release.py`: the app, the command line, the site, the browser page, and the contribute page | docs |
| `build-linux` | Linux x86-64 with Docker | `release/build_linux.sh`: the app and the command line in the pinned image | docs |
| `verify` | any | Checks hashes, manifests, commit, and the release block of each build against this release | both builds |
| `reproduce` | Windows | Rebuilds both Windows programs twice and compares them with the build | build-windows |
| `assemble` | any | Lays out what GitHub, the Pages branch, and the Hub receive, and sends nothing | verify, reproduce |
| `publish` | any, gated | The Hub upload and tag, and the GitHub release as a draft | assemble |
| `macos` | any, gated | The macOS programs from the release-build workflow, attached to the release once it is public | publish |
| `pages` | any, gated | The site and browser page as one ordinary commit on `gh-pages`, once the release is public and holds the macOS programs the site links | publish |
| `ad` | Windows, optional | Re-renders the ad from this release | verify |

`run` goes as far as `assemble` by default (`--until`, `--from`, `--only`, and `--force` choose others). The gated
stages have their own commands (`publish`, `macos`, `pages`, each with `--dry-run` or `--confirm`).

**A release of the same weights.** A release that changes only what no benchmark score reads (the calibration's
`between_sources`, which widens the range of the chance of AI) reads every text to the same score as the release before
it, so it takes that release's scores and cost instead of reading sixteen benchmarks again:

```sh
python release/cut.py adopt score --work <work folder> --from <the earlier release's work folder>
python release/cut.py adopt cost --work <work folder> --from <the earlier release's work folder>
```

`adopt` refuses an earlier release whose weights, tokenizer, or config differ in anything else, a stage that did not
finish there, files changed since it ran, and scores of other rows than the baselines this release fetched. The
receipt names the folder the result came from and when and where it was made.

## Every number comes from the model

A number in a document is a slot, a comment pair around its text (shown here with spaces, so that this page is not itself
filled; the real comments have none):

```
<!-- litmus:t:released_size -->`s192_l4` (width 192, 4 layers, 1.79M ternary weights)<!-- /litmus -->
```

The key names where the value comes from: a path into a data document and a format (`record.corpus.english_share as
pct1`), a sentence or table written once in `release/templates.toml` (`t:accuracy_summary`), or a table the driver makes
(`table.benchmark_matrix`). A slot not yet filled holds the word PENDING in square brackets. The prose around a slot is the author's; the number
is the data's, so nobody types one. The data documents are the plan, the release facts, `release/targets.toml`, the
model's `config.json`, facts derived from the model, the record folder's files by stem, the results, the speed and
browser measurements, and the builds' manifests. `release/litmus_release/datapath.py` says how a path and a format read:
names, `[n]` indexes, and `[field=value,...]` row selectors that must match exactly one row, and the formats `int`, `f0`
to `f4`, `pct0` to `pct2`, `mb1`, `m2`, `ms0`, `hours1`, `list`, and `text`. A value that is missing, a selector that
matches no row or two, a float with no format, and an empty result are errors that name the slot.

- `python release/cut.py slots list --work <work folder>` lists every slot and its state, `slots check` exits non-zero
  while any is unfilled, stale, or a bracketed PENDING outside a slot.
- The `docs` stage fills the slots in `docs/`, `release/model_card.md`, `release/dataset_card.md`,
  `release/release_notes.md`, `PUBLIC_README.md`, `docs/USE_WITH_AGENTS.md`, and the agent skill's `SKILL.md` and
  `references/*.md`, rewrites `site/assets/results.json` and `links.json` and the site pages' typed numbers, and writes
  `release/SHA256SUMS` for the model's files. Filling keeps the comment pairs, so the next release refills them.
- The build and publish stages refuse while any slot is unfinished. To add a number to a document, write a slot, never
  a figure; if its source is missing from the record, add the key to `record.toml` and say so in the training run's
  record.
- A figure the Rust source states (the MCP server's instruction text names the English share) is not a slot yet; check
  it by hand against the record when you cut a release.

## The path of a release

1. **Fetch and check.** `run` with `--model` and `--version`. It stops at the first stage that cannot proceed and says
   why: a missing record key, an unpinned baselines repository, the browser measurement. Fix what it names and run the
   same command again; finished stages are skipped.
2. **The browser page's speed.** Serve the built page (`build-windows`'s `site/`, or a trial build of the same source)
   and read the document `measure/bench.json` names, of that many tokens, in a real browser with one worker (`litmus
   bench --print-text --tokens <that many> --model <work folder>/model > document.txt` writes the same text the
   program was timed on). Write
   `<work folder>/measure/browser.json` with `weights_sha256`, `document_tokens`, `tokens_per_second`, `seconds`,
   `browser`, `threads`, and `machine`, then `python release/cut.py adopt browser --work <work folder> --from <file>`.
   The driver refuses a file for other weights or another document length.
3. **Documents.** `run` fills them in the `docs` stage, then stops. Read the diff of the documents and the site, because
   the prose around a slot may no longer be true of this model (a section about a teacher, a sentence that names a
   size); that is a person's or an agent's edit, and a slot's text is never edited by hand.
4. **Commit.** The release commit holds the version in every manifest, the filled documents, the site's data, and
   `release/SHA256SUMS`. Committing and pushing is the person's word. A build records the commit it was made at, and
   `verify` refuses a build whose commit is not the checkout's; `--allow-dirty` builds with uncommitted source for a
   trial that is never published. The workspace's authoring tree reaches the public repository through `ws publish`;
   build and publish from a clone of the public repository at the release commit (`git clone
   https://github.com/lhallee/litmus.git <folder>`, then `--repository <folder>` on every later call), so the manifest
   names a commit anyone can check out. A work folder moves between checkouts that hold the same files.
5. **Build.** `build-windows` on the Windows machine, `build-linux` on a Linux machine. The home page shows the ad's
   film, which is not in the repository: copy the wide cut for the web, its poster, its animated preview, and its
   record (`litmus_wide_site.mp4`, `litmus_wide_poster.png`, `litmus_wide_preview.webp`, `litmus_wide_film.json`, from
   `node render.mjs wide` in `demo/`) into `<work folder>/media` first; the `film` stage, just before the builds,
   copies the preview into `docs/images` for the README, and refuses a film that was measured with other weights than
   this release's (`demo/README.md`, A new release's film). When the driver is started on
   the wrong one, it stops with the exact steps: copy the work folder there (without `build/` and `publish/`; its
   receipts are what the stage checks), run `python release/cut.py run --work <copy> --only build-linux` from a checkout
   of the release commit, and bring the result back with `python release/cut.py adopt build-linux --work <work folder>
   --from <copy>`. `adopt` refuses a build that embeds other weights, another version, or files that do not match its
   `SHA256SUMS`.
6. **Verify and reproduce.** `run --from verify`. `reproduce` rebuilds the Windows programs from the checkout twice and
   compares them with the build, as a user does with `tools/verify_build.py --model`.
7. **Assemble.** Lays out `<work folder>/publish/` with `github/`, `pages/`, and `hub/`, and sends nothing.
8. **Publish, with the person's word.** `python release/cut.py publish --work <work folder> --dry-run` prints every
   command and call. Then, once the person says to:

   ```sh
   python release/cut.py publish --work <work folder> --confirm
   ```

   It checks the gate first: every earlier stage done, no unfinished slot, a clean tree, `HEAD` equal to the commit both
   builds were made at and pushed to `origin`, and the tag absent. It uploads `publish/hub/` to the Hub model repository
   as one commit and tags it `vX.Y.Z`, then creates the GitHub release as a draft. Nothing already published is
   overwritten: an existing tag stops it. Review the draft on GitHub, and publish it
   (`gh release edit vX.Y.Z --repo lhallee/litmus --draft=false`).
9. **macOS.** `macos --confirm`, once the release is public, starts the release-build workflow against the tag (the
   workflow downloads the model from the release); wait for it, download its artifacts into `<work folder>/macos/`,
   and the stage attaches the programs and replaces the release's `SHA256SUMS` with one that also names them.
   Replacing that file needs the person's word.
10. **Pages.** `pages --confirm`, once the release is public: the site and browser page replace the `gh-pages`
    branch's files in one ordinary commit, never a force-push. The model's files go into the branch's history (about
    13 MB for the smallest size). The site links the macOS programs (`targets.toml` lists `"macos"`), so the stage
    refuses while the release does not hold all four. If the macOS build cannot be had, `pages --confirm
    --ahead-of-macos` publishes the site anyway, and its macOS links resolve once the programs are attached.
11. **The contribute site.** It is a deployment, not a stage: follow `contribute/README.md` with
    `LITMUS_CONTRIBUTE_SITE=<work folder>/build/windows/contribute`, `LITMUS_RELEASE=<work folder>/model`, the Linux wheel
    of `litmus_engine`, and `LITMUS_PRIVATE_PAGE=<pages_url>app/`. Set `[contribute] site_url` in
    `release/targets.toml` once it is live.
12. **The ad.** `run --only ad` re-renders `demo/` with this release's numbers and the footage the driver is given
    (`--ad-clips`, for the Linux app, which needs a machine that has a Linux desktop).

## The assets

The GitHub release attaches, by these names, which the site and the install scripts for AI agents
([`docs/USE_WITH_AGENTS.md`](../docs/USE_WITH_AGENTS.md)) read:

| Asset | From |
| --- | --- |
| `litmus-app.exe`, `litmus.exe` | the Windows build |
| `litmus-app-linux-x86_64`, `litmus-linux-x86_64` | the Linux build's `litmus-app` and `litmus` |
| `litmus-app-macos-arm64`, `litmus-macos-arm64`, `litmus-app-macos-x86_64`, `litmus-macos-x86_64` | the `macos` stage, after publishing |
| `model.safetensors`, `config.json`, `tokenizer.json` | the model |
| `manifest.json`, `manifest-linux.json`, `results.json` | the builds and the results |
| `SHA256SUMS` | one line per attached program and file; the install scripts read the line that names their program |
| `NOTES.md` | the release notes, which are `release/release_notes.md` |

The names are in `release/targets.toml` (`[programs.assets]`); the build scripts keep each platform's plain names
(`litmus`, `litmus-app`), and `assemble` renames them, so the scripts and the workflow need no change. A platform whose
program is not attached gets the install script's message to build from source. `release/SHA256SUMS` in the repository
lists the model's three files only, because the release-build workflow checks the model it downloads against it; a
program's hash cannot be in the commit it is built from, so the programs' hashes live in the release's `SHA256SUMS`.

A download loses the executable bit on Linux and macOS: `chmod +x litmus litmus-app`. macOS also marks a downloaded
program as quarantined and, since the programs are not code-signed, refuses to open it until the mark is removed
(`xattr -d com.apple.quarantine <file>`) or the person allows it under Privacy & Security in System Settings. A user checks a download with
`sha256sum -c --ignore-missing SHA256SUMS`, or in PowerShell `Get-FileHash` against its line, and can rebuild both
Windows programs from the tagged source with `tools/verify_build.py --model <folder>`. An unsigned binary makes Windows
SmartScreen show "unknown publisher".

## Versions and the Hub

The release version is stated in these places, and `python release/cut.py version check <X.Y.Z>` lists the ones that
differ while `version set <X.Y.Z>` writes them all, changing only the version lines and keeping each file's line endings
(the `version` stage does the same for the plan's version):

- `engine/Cargo.toml` `[workspace.package]`, which the engine, files, CLI, and Python crates inherit;
- `app/Cargo.toml` `[package]`;
- the `pyproject.toml` of `corpus`, `teacher`, `student`, `benchmarks`, and `contribute`;
- `.claude-plugin/plugin.json`;
- the `litmus-*` packages of `engine/Cargo.lock` and `app/Cargo.lock`;
- the built `manifest.json` and the site's `release.version`, written by `build_release.py --release-version` and
  checked by `verify`.

The demo's `package.json` is the ad's own and carries no release version.

**Proposed Hub layout, not created.** One public model repository, `lhallee/litmus` (`targets.toml` `[hub] model_repo`).
Its `main` branch is the latest release and each version is a tag, `v1.0.0`, so a user pins
`revision="v1.0.0"` and the model's page stays one address. A new version is a new commit and a new tag on the same
repository, never an overwrite of a tagged one. A Hub collection, "Litmus", holds the model, any earlier model
repository (for example the first build), and the corpus card `lhallee/litmus-corpus`. The trained models before a
release stay in their own private repositories at their own commits; `--model` names one of them, and the release copies
its release folder into the public repository.

## What each build stage does

### Windows

```sh
python release/build_release.py --release <release folder> --out <empty folder outside the repository> --release-version <X.Y.Z> --media <film folder>
```

The driver runs this with the fetched release folder. The script refuses the random-weights placeholder (a trial uses
`--allow-placeholder`), a tokenizer whose hash differs from the one `config.json` states, and an output folder that is
not empty (`--replace` deletes an earlier output of the script). A clean build takes 7.3 to 8.4 minutes on an AMD
Ryzen 7 3800X (the command line 1.6 to 1.7, the app 2.8 to 3.5, the two browser modules 2.8 to 3.1), most of it
link-time optimization; `--incremental` keeps the build folder and takes about 3 minutes after a source change. It
writes:

| Path | What | Goes to |
| --- | --- | --- |
| `litmus-app.exe` | The desktop app, the release built in: one file, no install, runs offline | The GitHub release |
| `litmus.exe` | The command line, the release built in | The GitHub release |
| `site/` | The eight pages, and `app/` with the browser page, its worker, and the model's three files | The `gh-pages` branch |
| `contribute/` | The page the contribute server serves (`LITMUS_CONTRIBUTE_SITE`) | The contribute deployment |
| `manifest.json` | Model version, release version, commit and source digest, toolchain, build settings, sizes | The GitHub release |
| `SHA256SUMS` | SHA-256 of every file above | The build's own check |

The programs are built by `tools/verify_build.py`'s own functions, the ones its `--model` check runs: from a fresh copy
of the source at `C:\litmus-build`, with the release's model files at `C:\litmus-build\model`, the pinned toolchain,
`--locked`, the same path-remapping flags, and the C runtime linked in. The build checks every page and script for a
path that starts at the server's root, which would break under `/litmus/`, and runs `litmus.exe` once to confirm its
built-in release reads a text exactly as the release folder does.

### Linux

On a Linux x86_64 machine with Docker (`DOCKER="sudo docker"` where docker needs sudo):

```sh
bash release/build_linux.sh <release folder> <empty folder outside the repository>
```

It runs `build_release.py --native-only` in maturin's manylinux2014 image, pinned by digest (CentOS 7, glibc 2.17,
GCC 10, the image `engine/litmus-py/build_wheel.sh` builds the Linux wheel in), and writes `litmus` and `litmus-app`
for `x86_64-unknown-linux-gnu` with the release built in, `manifest.json`, and `SHA256SUMS`. It fails if either
program needs a glibc symbol newer than 2.17, the oldest glibc Rust's Linux target supports, so the programs run on
RHEL and CentOS 7, Debian 8, Ubuntu 14.04, and every later x86_64 distribution. The build takes 4.5 minutes on 8
cores; two clean builds of one source and release gave the same bytes.

Both programs link only glibc (`libc`, `libm`) and `libgcc_s`. The app loads the rest when it starts, so the build
needs none of it:

| For | Libraries (Debian and Ubuntu packages) |
| --- | --- |
| A window on X11 | `libX11.so.6`, `libX11-xcb.so.1`, `libXcursor.so.1`, `libXi.so.6`, `libxkbcommon.so.0`, `libxkbcommon-x11.so.0` (`libx11-6`, `libx11-xcb1`, `libxcursor1`, `libxi6`, `libxkbcommon0`, `libxkbcommon-x11-0`) |
| A window on Wayland (from the dependencies' source; not run) | `libwayland-client.so.0`, `libwayland-egl.so.1`, `libEGL.so.1`, `libxkbcommon.so.0` |
| Drawing | OpenGL 2 or later through `libGL.so.1` (GLX) or `libEGL.so.1`: the system's driver, or Mesa's (`libgl1`, `libgl1-mesa-dri`) |
| Open file | The desktop's file-chooser portal (`xdg-desktop-portal`) over D-Bus, else `zenity` |
| Quick check | `xclip` on X11, `wl-paste` (`wl-clipboard`) on Wayland; the hotkey works on X11 only, and there is no tray |

Every desktop distribution has the X11 and OpenGL libraries; a server or a minimal container needs them installed.

### The checks by hand

`verify` and `reproduce` do the hashes and the rebuild. Use each output as a person would too, with the network not
needed:

- Copy `litmus-app.exe` alone into an empty folder and run it: it opens in the dark theme at the Simple level with no
  console window; check a text and see a reading. `litmus.exe score <file>` reads the same text the same way.
- Serve `site/` under a `/litmus/` prefix (as GitHub Pages does), open `/litmus/app/`, and check a text; the
  browser's network panel shows only the page's own files and the model's three.
- Serve `contribute/` with the contribute server (`contribute/README.md`) and confirm, at a desktop width and about
  400 pixels wide in a window 800 pixels high, that the notice, the consent box, Check, and Remove a text are all on
  screen; that Check stays off until the box is ticked; and that a check, a label, and a removal reach the server.

`manifest.json` records `modules_before_wasm_bindgen_sha256`; `python tools/verify_build.py --artifacts wasm` rebuilds
the browser module at the same fixed path and prints its hash, which is the `app_module` one.

## Tests

`python tools/run_tests.py release` runs the driver's tests (stage receipts and staleness, version setting, every slot
and its data, the record bundle, the assets, the gates and dry runs). They use synthetic data and make no network call;
`python tools/run_tests.py security` checks the capability declarations.
