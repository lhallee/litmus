# Litmus app

What a person sees and touches of Litmus: one egui (eframe 0.37) program that runs natively on Windows, macOS and
Linux and compiles to WebAssembly for the github.io page, in three builds of one UI. The engine
([`../engine`](../engine/README.md)) does the reading; this crate shows it. Licensed under PolyForm Noncommercial
1.0.0 like the rest of the project; the crates it compiles in are in [`../engine/THIRD_PARTY.md`](../engine/THIRD_PARTY.md).

`release/build_release.py` builds the three shipped forms with the release built in.

## Layout

| Path | Holds |
| --- | --- |
| `Cargo.toml` | Its own workspace (below), the features `embedded`, `contribute` and `release-tests`, and the `web` profile |
| `build.rs`, `src/icon_resource.rs` | The Windows icon resource: the mark drawn at 16 to 256 pixels into a `.res` file the MSVC linker takes, so `litmus-app.exe` shows the mark in Explorer |
| `src/main.rs` | The desktop entry: one app at a time, the window's saved place; `--check <file>`, `--check-text <text>`, `--write-icon <svg>`; `__pdf-text <file>`, the PDF child process |
| `src/app.rs` | State, events from the scorer, the keyboard shortcuts (`SHORTCUTS`, the one list), dropped files, requests from later launches |
| `src/views/` | The screens: `check` (the editor, then the reading over the tinted text), `reading` (the verdict hero and the detail panel), `heat`, `explain`, `batch`, `history`, `settings`; `controls` (buttons, tabs, the segmented switch, cards) and `table` (the sortable tables) give them one style |
| `src/instance.rs`, `src/window_place.rs`, `src/sorting.rs` | One app at a time and the handoff of a later launch's request; the window's saved place; sorting the tables |
| `src/scoring.rs` | Scoring off the UI thread: a worker thread (desktop), a Web Worker (browser), the site's server (contribute) |
| `src/wording.rs`, `strip.rs`, `heatmap.rs`, `history.rs`, `model.rs`, `contribute.rs`, `brand.rs`, `theme.rs` | The logic the screens use, with unit tests |
| `src/web.rs` | The browser's `start` and the `WorkerEngine` the page's worker drives |
| `src/demo.rs`, `src/demo/quick.rs`, `src/png.rs` | A scripted run for screenshots and end-to-end checks (desktop), and quick check's scripts |
| `assets/litmus_mark.svg`, `assets/litmus_icon.svg` | The mark, and the icon written from `brand.rs` (a test keeps them equal) |
| `scripts/build_web.sh` | Builds the browser page into a dist folder |
| `tests/release.rs`, `tests/reading_small.json` | Tests against a real release; a reading of a 14-token text by the placeholder, written by `litmus score --json` |
| `../site/app/index.html`, `../site/app/worker.js` | The browser page's host (full-window canvas and loader) and its Web Worker |

Files and the cache come from the crate `../engine/litmus-files`, which the command line
shares: text of txt, md, html, docx and pdf files, the content-hash cache, release folders, folder walks, and
the CSV and JSON exports. The app and `litmus` therefore write the same CSV columns and share one cache folder.

**Why its own workspace.** The app depends on `litmus-engine` and `litmus-files` by path but is not a member of
`engine/Cargo.toml`: the GUI brings about 200 crates (winit, glutin, accesskit, the Windows bindings), which would
otherwise enter the engine's lock file, every `cargo test` of the engine and the Python wheel's build. Its release
profile matches the engine's (fat LTO, one codegen unit); the `web` profile optimizes the UI for size and keeps the
engine at `opt-level = 3`.

## The three builds

All commands from the repository's root.

**Desktop.**

```sh
cd app
LITMUS_MODEL=<release folder> CARGO_TARGET_DIR=<target> cargo run --release
```

The model is the first of: the folder chosen in Settings, `LITMUS_MODEL`, a folder named `release` beside the
executable, and the release built into the binary (`--features embedded` with `LITMUS_EMBED_DIR=<release folder>`, as
the command line's).

**The shipped `litmus-app.exe` carries its model**: `release/build_release.py` builds it with
`--features embedded`, so the download is one file that runs offline with nothing beside it. The release adds its
13 MB to the program for 1.0.0's d192 L4, and more for 1.1's d512 L6. Developers still load another release from a folder: Settings, `LITMUS_MODEL`,
or a `release` folder beside the program win over the built-in one. The C runtime is linked in (`crt-static`), so
the program needs no Visual C++ Redistributable. [`../release/RELEASING.md`](../release/RELEASING.md) has the steps. Settings and the history live in the data folder (`%APPDATA%\litmus`, `~/Library/Application
Support/litmus`, `$XDG_DATA_HOME/litmus`; `LITMUS_DATA_DIR` overrides); the cache is the command line's
(`%LOCALAPPDATA%\litmus\cache` and the like; `LITMUS_CACHE_DIR` overrides).

**Browser (the github.io page).**

```sh
WASM_BINDGEN=<wasm-bindgen 0.2.126> CARGO_TARGET_DIR=<target> \
  app/scripts/build_web.sh <dist> <release folder>
python -m http.server 8000 --directory <dist>        # then open http://localhost:8000/
```

The script builds the cdylib for `wasm32-unknown-unknown` with SIMD128, runs `wasm-bindgen --target web`, and copies
`site/app/index.html`, `site/app/worker.js`, the icon as `favicon.svg` and the release's three files beside it.
`wasm-bindgen-cli` must be the version the crate pins (`=0.2.126`); install it with
`cargo install wasm-bindgen-cli --version 0.2.126 --locked --root <folder>`. For the published page, the dist
folder's contents go to `site/app/` of the github.io branch (the three model files included, about 16 MB for 1.0.0's d192 L4).

**Contribute.**

```sh
CARGO_TARGET_DIR=<target> app/scripts/build_web.sh --contribute <dist>
```

The same page with the feature `contribute`, built to the contract in [`../contribute/README.md`](../contribute/README.md)
("What the contribute page must send and show"): no worker and no model files. On load it reads `GET /api/terms`
and shows, under the text box and directly above the consent box and the Check button, the notice as served
(`contribute::paragraphs`: its paragraphs, its bold sentence, and every link underlined in the link ink, in body
type). The consent box is a real checkbox labeled with the served `consent_label` (the Terms of Use and the Privacy
Notice are links); it starts unticked on every load and is unticked again whenever the served version changes, and
it lives in memory only. Check stays off until it is ticked. A check sends `POST /api/score` with
`{"text", "terms_version", "consent": true}`; a 409 (new terms) reads the terms again, unticks the box, and says
beside it that nothing was sent. The words "removal form" in the notice, and the Remove a text button beside Check,
open the removal form over the page, which sends `POST /api/removals` with `{"text"}`, checks that the server's
SHA-256 is the page's own of the text sent, and says that every stored copy of that exact text is deleted within 30
days and that models already trained on it are not retrained. After a reading (its strip and band shown as on every
build) the page offers the labels AI, Mixed and Not AI (`POST /api/contributions/{id}/annotation`) within the site's
window; the contribution's id stays in memory. The notice, the box, Check, and Remove a text fit an 800-pixel-high
window at desktop width and at 400 pixels wide (the box shrinks to fit, the first screen's question is left out on a
phone, and a screen too short scrolls). The dist folder is what the contribute server's `LITMUS_CONTRIBUTE_SITE`
serves.

Sizes (Windows x86-64, `release/build_release.py`, a trained d192 L4 release built in):
`litmus-app.exe` 24.0 MB and `litmus.exe` 16.5 MB, the C runtime linked in. Browser: the page (host page, worker,
glue, and `litmus_app_bg.wasm`) 5.0 MB, 1.8 MB gzipped, plus the release's three files, 13.5 MB (10.9 MB gzipped).
No `wasm-opt` pass is run.

## What a reading shows

**Three levels of detail**, simple by default. The level is the user's choice, remembered (the settings file on the desktop, the
page's local storage in the browser), switched by "More detail" under the strip, the Detail switch beside a
reading, Settings, or Ctrl+M. Exports always carry every field, and the JSON exports the text's `counts` object as
`litmus score --json` does (the CSV columns are unchanged); the level is about the screen. No level hides
uncertainty: an uncertain reading says "Uncertain" and shows its band, "Unlike reference" stays its own state, and
the placeholder warning stays on every screen.

- **Simple** (the default): before a check, one sentence, the text box and one primary Check button; after it, the
  verdict word, one plain sentence (`wording::plain_sentence`, in the site's words), one muted line with the text's size
  (`1,234 words, 7,890 characters, 1,650 tokens`, from `litmus-files`' `counts`, counted once as the reading opens),
  the strip with its band, and the user's own text tinted as the heat map, with Edit to go back to editing. Nothing
  else.
- **Detailed** adds a panel beside the reading (under it on a narrow window): P(not human) with its interval, what
  the 10% set keeps, tokens, characters without spaces, and time; what a reading is and our recommendation for using one
  (`wording::RECOMMENDATION`: a reading is an estimate, not proof, and we recommend not using a reading alone to
  accuse anyone); the minimap of a long text; and the reading of the part of the text clicked. With a release that
  carries its masked language head (1.1 on), it also adds the text's **surprise** (`student/SPEC.md`, Surprise): minus
  the mean log probability the model gives each token seeing the rest of the text, the perplexity, and the share of
  each origin's calibration texts the model found at least as surprising, with what it is not
  (`wording::SURPRISE_CAUTION`). It is read in a pass of its own only when a reading's details are first shown, on the
  scoring thread (in the browser, the worker), and stops when a new reading starts; the contribute site does not show it.
- **Full** adds every other number, how the reading was made, the model, and the exports.

The pieces:

- **Verdict**: the conformal set at 0.10. One origin reads "Human", "AI" or "Mixed"; several read "Uncertain: human
  or mixed" (named, in strip order, with the Oxford comma for three); none reads "Unlike reference". When the set is
  not empty but the text is out of reference, the set's word stays and a "? unlike reference" caution sits beside
  it, as on the site. A marker beside the headline differs in shape by call (disc, ring, question mark).
- **The strip**: blue `#2F6FDB` to purple `#8E5BC9` to red `#D9483B`, a triangle marker at `strip.position`, the pale
  band `#C9B8E6` from `band_lo` to `band_hi` bracketed at both ends, the words Human, Mixed and AI under it; washed out
  with a question mark when the strip is unlike the reference.
- **Numbers** (Detailed and Full): P(not human) with its interval, the three-way probabilities, the conformal p-values, the sets at 10%
  and 5%, whether the text is called not human at 10% (left out when the set is empty), tokens and sections, the
  time taken or "from the cache", the closeness p-value, the model and when it was read; then the flags (out of
  reference; for long texts the mixed-document signal or its absence).
- **Heat map**: the user's text itself, after a check, with each 128-token cell tinted by its place on the scale, lighter toward human and deeper
  toward AI so lightness alone orders the cells; cells unlike the reference underlined; a minimap of the whole
  document; a click on a cell (in the text or the minimap) shows its numbers and the reading of the window centered
  nearest it, with the other windows over it. The text is laid out in pages of 12 cells and only the pages in view
  are painted, so a 131,072-token text stays smooth.
- **How this was read**: how sections combine (merged p-values, the widened interval, the mixed-document test);
  per section, the bank evidence (mean, share, spread and its percentile per origin), each conformal p-value in
  words with the calibration group's token lengths ("Of n human calibration texts of lengths 0-15 tokens, k read at
  most this human-like: p = ..."), the sets, the Venn-Abers probability, the closeness count, and the nearest bank texts
  with origin, cosine, P(same origin), tokens and their recorded fields.
- **Placeholder honesty**: a release whose `config.json` says `"provenance": {"kind": "random"}` (what
  `reference.py init` writes; a trained export records its checkpoint instead) shows "Placeholder model, readings are
  meaningless" on every screen. No sidecar file is needed.

Every color-coded state also has a word or a shape. Light and dark themes use the site's tokens
(`site/style.css`), the origin inks included for verdict words; dark is the default on first launch in all three
builds, and a saved choice wins after that. One corner
radius (6 points), spacing on a 4-point grid, a filled primary button (Check) and quiet secondary ones, and a focus
ring in the mixed purple on every control; `theme::tests` checks every ink against every ground at 4.5:1.

**Type.** On Windows the desktop app uses the system's Segoe UI (regular and semibold) and Cascadia Mono or
Consolas, the faces the site's `system-ui` stack shows there, read from the system's font folder and never shipped;
the browser and other systems use egui's bundled Ubuntu Light and Hack. Hierarchy is size and weight: the verdict at
34 points semibold, body text at 15, numbers in the mono face.

**Keyboard.** Every action has a control reachable by Tab (egui with AccessKit, so screen readers see each control
by role and name), and the shortcuts are listed once, in `app::SHORTCUTS`, which Settings shows: Ctrl+Enter checks,
Esc cancels or closes Settings, Ctrl+E edits the text again, Ctrl+O opens a file, Ctrl+M changes the detail level,
Ctrl+1, 2 and 3 switch screens, Ctrl+, opens Settings, Ctrl+Shift+L switches the theme. The browser leaves Ctrl+O,
Ctrl+E and Ctrl+1 to 3 to the browser.

**States.** First launch: one sentence, Sample and Open file beside Check, and the drop hint. Scoring: the text stays
on screen with the progress in tokens and sections and a Cancel button (the reading runs off the UI thread). A file
opened, dropped, or handed over by Explorer is read on the scoring thread as well ("Opening notes.pdf"), so a large
PDF never freezes the window, and its text fills the editor when its reading arrives. Errors say what happened and
what to do (a file that does not read, an empty file or a text with nothing to read, no model with a button to
Settings). Under 32 tokens a reading adds that a few words carry little evidence. A text file saved as UTF-16 with its
byte-order mark (Notepad's "Unicode") is read as such (`litmus_files::extract::decode`).

**Desktop.** One app at a time: the first launch holds a lock in the data folder (`instance.lock`, the system's own
file lock, so a crash leaves nothing stale); a later launch, or Explorer's "Check with Litmus", leaves its request
in `inbox/` and exits, and the running app comes to the front and checks the file or text. The window's size,
place and maximized state are saved a second after they settle and when it closes, and restored unless the saved
place is on no screen. `litmus-app.exe` carries the mark as its icon resource.

## Quick check (desktop, opt-in)

Select text in any application, press a hotkey, and a small window by the pointer shows the reading. Off by default; nothing about it runs, registers or writes until the user
ticks it in Settings, beside one sentence on what it does (`quick::SENTENCE`).

- **When on**: the app stays in the background when its window is closed, with a tray icon (Check copied text, Open
  Litmus, Pause quick check, Quit), and a global hotkey (default Ctrl+Alt+L, changeable in Settings; it must hold
  Ctrl, Alt or the Windows key). The choice is saved, so the hotkey is registered again at the next launch only
  because the user switched it on.
- **On the hotkey**: the selected text is read on a thread of its own from the app the hotkey was pressed in, and
  only then, since a new window takes the focus, a small borderless, always-on-top window opens by the pointer with
  the reading, scored locally: the verdict, the strip with its band, and "Open full reading" (the main window, with heat map and explanation); it follows the detail level: Simple is
  the verdict word and the strip; at every level a line under the strip gives the selected text's words, characters,
  and tokens; Detailed adds P(not human) with its interval, and under 32 tokens that a few words carry little evidence; Full adds how the text was read. The placeholder warning shows at every
  level. Checked at 100% and 150% display scaling (`LITMUS_DEMO_ZOOM=1.5`). It opens below and to the right of the
  pointer, above or to the left of it where the screen's work area would cut it off, and always inside that work
  area. It closes on Esc, when it loses the focus after holding it (a click elsewhere), or after 25 seconds unless
  the pointer rests on it. Windows opens it without the focus when the app has no right to take it (it was in the
  background, and the simulated copy was the last input): then the popup reads the state of Esc and the mouse
  buttons, about 30 times a second while it is up, so Esc or a click outside it still closes it; neither is taken
  from the app in front. The reading enters the history only if "Keep a history" is on.
- **Its own scoring thread**: quick check's readings run on a second thread that shares the loaded model (the engine
  is `Send` and `Sync`), so a popup never waits behind a batch or a long reading in the main window; a popup that
  closes, or is replaced by the next press, stops its reading. A press while a capture is under way waits for it (two
  copies at once could each save the other's selection as the clipboard to restore), and a press while the popup
  holds the focus keeps the popup (the selection would be read from the popup itself).
- **Reading the selection on Windows**: UI Automation's text pattern on the focused element first (rich edit
  controls, browsers, Office and most modern apps offer it); if there is none (classic Edit controls such as
  Notepad's offer none), simulated Ctrl+C after the hotkey's keys are released, sent only while the window the hotkey
  was pressed over is still in front, with every memory-backed clipboard format saved first and put back after,
  marked to stay out of clipboard history. The tray's item reads the clipboard as it is, because clicking the tray moves
  the focus away from the selection. The source app's own copy may enter Windows' clipboard history; that is the
  source app's write, not one the app can mark.
- **Right-click**: there is no general way to add an item to every application's context menu on Windows. Settings
  offers "Check with Litmus" on files in Explorer (txt, md, markdown, html, htm, docx, pdf): per-user keys
  `HKEY_CURRENT_USER\Software\Classes\SystemFileAssociations\.<ext>\shell\LitmusCheck`, written and removed only by
  that switch; the entry runs `litmus-app --check "<file>"`, which opens the app and checks the file.
- **macOS and Linux** (`src/quick/unix.rs`): Linux is built and run on X11 under a virtual display, where the hotkey
  read a selection set with `xclip` and the popup showed its reading; Wayland and macOS are not run.
  Linux reads the PRIMARY selection (`wl-paste --primary` or `xclip`), so nothing is copied; the hotkey works on X11
  only and there is no tray, so quick check works while the window is open and closing the window quits. macOS copies through System Events
  (`osascript`, needs the Accessibility permission) and saves and restores the clipboard's text only. The macOS
  right-click route, a Services entry, is not built: it needs a service provider in the app bundle.

Code: `src/quick/mod.rs` (hotkey, tray, capture thread, popup state), `src/quick/windows.rs` (UI Automation,
clipboard, Explorer entry), `src/views/quick.rs` (Settings section, popup, tray actions, hiding to the tray).

## The browser build sends nothing

The private page's only network requests are the page's own files (the loader in `site/app/index.html` reads
`litmus_app_bg.wasm` as a stream to show its progress in MB) and the three GETs in `site/app/worker.js`
(`model.safetensors`, `config.json`, `tokenizer.json`, from the page's own origin, streamed so the page shows
"Downloading the model: 6.2 of 16.4 MB"). The detail level and the theme are kept in the page's local storage
(`litmus.settings`), which stays in the browser. By code inspection: without the
`contribute` feature the crate enables no `web-sys` request API (`Request`, `RequestInit`, `Response` and `Headers`
are enabled only by that feature) and calls no `fetch`; `contribute::fetch` compiles only with it. Checked in a
browser as well (below). The text goes to the worker by `postMessage`, which stays inside the page.

At phone width (below 600 points) the page stacks, the placeholder warning keeps to its title, and every control is
at least 40 points tall; the first screen says in one line that the text is read in the page and never leaves it.

## Tests

| Command | Checks |
| --- | --- |
| `cargo test` | Verdict wording, set sentences and the simple screen's plain sentences, number formats, strip geometry, cell tints ordered by lightness, cell-to-text mapping (pages, sections, byte lookup, characters), history (reopening, repeats, removal, a damaged index is an error), settings defaults and the detail level, the contribute API's request and response shapes, PNG encoding, dates, the placeholder test, the icon asset equal to the code, the icon resource's layout, every ink at 4.5:1 on every ground, sorting (direction, stability, gaps last), one app at a time (a second claim finds the lock, requests handed over oldest first and once, a damaged one reported), the window's saved place (sane sizes, jitter is not a move), history entries saved before bands, the quick-check popup's place (below and right of the pointer, flipped above or left at the work area's edges, inside it on any monitor), and a popup without the focus closing on Esc or a click outside it but not on a key or button already held when it opened, nor on a click on itself |
| `cargo test --features desktop-tests --test quick_windows -- --test-threads 1` (Windows) | Quick check against this machine, leaving it as found: the Explorer entry written, found and removed under a temporary root `HKEY_CURRENT_USER\Software\LitmusTest_<pid>` (never `Software\Classes`), which is then deleted; UI Automation reads the selection of a WordPad the test starts and ends; the clipboard is saved, overwritten with text marked to stay out of clipboard history, and put back byte for byte, then put back as the copy path does, the same formats and the history marker |
| `LITMUS_MODEL=<placeholder> cargo test --release --features release-tests` | Against the random placeholder release: cells tile the text and map back from bytes, every cell has the engine's count of windows over it, a misplaced reading is refused; the scorer thread loads, scores, serves an unchanged text from the cache, cancels a long text, scores a folder as a batch with an error row for an empty file, and reads and scores a file opened in the Check screen (its text with the reading, "Opening the file" first, an empty and a missing file as errors); quick check's thread answers while a long main-window reading runs, and its cancel stops its own reading only |
| `LITMUS_MODEL=<trained release> cargo test --release --features desktop-tests,release-tests --test quick_scenes -- --test-threads 1` (Windows) | Every quick-check script of the demo (`src/demo/quick.rs`) on the built app, with the main window hidden in the tray and kept up: reading, loading, batch, replace, escape, expire, full and tray, each passing and ending through the tray's Quit within 90 seconds. Its windows show on the screen for a second or two per script; nothing is typed, clicked, or copied on the desktop |
| `cargo test -p litmus-files` (in `engine/`) | Folder walks, CSV and JSON rows, HTML and entities, the word, character and token counts (empty text, runs of whitespace, CRLF, a non-breaking space, CJK, an emoji of several scalars), text files decoded by their byte-order mark (UTF-16LE and BE, UTF-8 with a mark, an HTML page in UTF-16) |

Not tested by code: the screens themselves (they are run and photographed, below), the hotkey pressed by a person
on a physical keyboard (and so whether Windows gives that popup the focus; the demo's popup had it on some runs and
not on others, and Esc and a click elsewhere close it either way), the tray icon clicked with a mouse (its actions
are driven in-process by the `tray` script), placement across monitors of different scaling, the macOS build, Linux
on Wayland, and the browser build in browsers other than Edge. Quick check on a release build of `litmus-app.exe` is driven end to end on Windows 10, with a
fresh data folder and the machine left as found: the Settings switch ticked through UI Automation; the main window
closed to the tray; Ctrl+Alt+L synthesized only while a Notepad holding a selected text is in front, which takes
the copy path (Notepad's Edit control has no text pattern) and opens the popup with the same reading, field for
field, as `litmus.exe score --json`, the clipboard's formats and text unchanged after; the tray menu's "Check copied
text" with a known text put on the clipboard, marked to stay out of clipboard history, and the clipboard put back;
and the Explorer entry's command, written under a scratch registry root and removed, which hands its file to the
running app. The contribute build runs against the contribute server with the real binding: a check, a label, a
removal, a refusal without consent, and a change of terms.

## Screenshots and end-to-end runs (desktop)

`LITMUS_DEMO=<scene>` with `LITMUS_SCREENSHOT=<file.png>` runs a scene and writes a PNG of the window, then
closes. Scenes: `empty`, `check`, `short`, `long`, `explain`, `batch`, `history`, `settings`, `quick`, `open`, and the
states `scoring` (photographed mid-reading), `error` (a file that does not read), `edit` (the editor over a reading)
and `nomodel` (`LITMUS_MODEL` at an empty folder). A scene runs in the app's logic, which also runs while the window
is hidden in the tray. The demo runs apart from the one-app lock and never saves the window's
place. `LITMUS_THEME=light|dark`, `LITMUS_DETAIL=simple|detailed|full`, `LITMUS_DEMO_ZOOM=1.5` (display scaling),
`LITMUS_DEMO_SIZE=1280x820`, `LITMUS_DEMO_TEXT` (a text file the `check`, `explain` and `edit` scenes check in
place of the sample; the `open` scene opens it as Open file does, any kind the app reads, and prints what it showed,
how long it took, and its slowest frame meanwhile),
`LITMUS_DEMO_FOLDER` (the batch scene's folder) and `LITMUS_DEMO_EXPORT` (where it writes the batch's CSV and JSON
through the Export buttons' functions). `quick` turns quick check on for the run only (not saved, on a hotkey of its
own, Ctrl+Alt+Shift+F10, so it never takes a running Litmus's), hides the window to the tray (`LITMUS_DEMO_WINDOW=shown`
keeps it up), feeds in a capture as the hotkey would, and runs one script, `LITMUS_DEMO_QUICK` (`src/demo/quick.rs`):
`reading` (the default: the popup draws its reading), `loading` (captured while the model reloads: the popup says it
is loading, then shows the reading), `batch` (the reading shows while a batch of `LITMUS_DEMO_FOLDER` still runs),
`replace` (a second capture replaces a long reading that shows its progress; the new reading shows at once, so the
first was cancelled), `escape`, `expire` (after a two-second life), `full` (Open full reading shows it in the main
window), and `tray` (the close button hides to the tray; Pause, Resume, Open). Scripts check the windows themselves
on the screen (that a closed popup's window is gone), and every one ends through the tray's Quit. In the browser,
`#sample` checks the sample once the model loads and `#dark` or `#light` picks the theme.

## Running a trained release

`python release/build_release.py --release <folder> --out <folder>` builds the shipped app, command line, page, and
contribute page around a trained release folder ([`../release/RELEASING.md`](../release/RELEASING.md)). For development,
point the app at the folder (Settings, `LITMUS_MODEL`, or a `release` folder beside the executable). The placeholder
notice disappears by itself, since a trained release's provenance is not `random`.

Checked with an interim trained release (d192 L4, step 4,999): the shipped `litmus-app.exe`, alone in an
empty folder with a fresh data folder, read a human-written and a model-written news story exactly as
`litmus.exe score --json` did, field for field, through its check screen and its batch export; the page under a
`/litmus/` prefix in headless Edge read them, a pasted text, and its own sample with every word, set, count, and
place equal to the command line's and every number within 1e-7 (WebAssembly's floating-point functions differ from
Windows' in the last bits); the contribute page, served by the contribute server with the `litmus_engine` binding,
sent nothing until its consent box was ticked and returned the command line's reading byte for byte.

## Known limits

- The browser's UI font is egui's Ubuntu Light (Ubuntu Font Licence), which has no semibold, so the browser's
  hierarchy is size alone. macOS and Linux use the bundled faces too; the app does not read their system fonts.
- The contribute build cannot tell whether the site's release is the placeholder: the server's API returns
  `model_version` only.
- The icon resource is linked for MSVC targets only; a GNU-toolchain Windows build gets the run-time window icon but
  no icon in Explorer.
