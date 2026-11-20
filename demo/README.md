# Litmus demo: the ad

A 41.5-second vertical ad (1080 by 1920, 30 fps, with music), an 11.5-second teaser, and the same ad as a 1920 by 1080 wide
cut for a README and a web page, drawn as code from real footage of the
app and the page, the engine's own readings, and a measured speed run. Every number on screen is read from a data file
in `data/ad/` with its basis, so a new release is one command away from a new ad. The beats and the timing
are in [STORYBOARD.md](STORYBOARD.md).

## Make it

Node 22, the FFmpeg build from `ffmpeg-static`, and Playwright's Chromium (`npx playwright install chromium` once).
Everything heavy (dependencies, the build, frames, footage, the documents, and the
videos) goes in a work folder outside the repository.

```text
node make.mjs --work <dir> --litmus <litmus.exe> --site <built site folder> --manifest <release manifest.json> --clips <dir>
```

That one command, given a release's programs, re-reads the sample texts, re-runs the pinned speed and memory measurement,
recaptures the browser checker, prepares the footage, and renders the ad, the teaser, and the wide cut into `<dir>/out/`
(`litmus_ad.mp4`, `litmus_teaser.mp4`, `litmus_wide.mp4`, and the wide cut's web encodes, see Embedding) with a still at each
beat of review (`<dir>/out/stills/ad`, `teaser`, `wide`). The desktop footage and the app's memory (`--clips`, holding `app.mp4`,
`app.json`, `quick.mp4`, `quick.json`, `memory.json`) come from the Linux build of the app on a virtual display, recorded
with Docker on an x86_64 machine, over SSH or on this one:

```text
capture/run_linux.sh <ssh key> <user@host> <litmus-app, the Linux build with the release built in> <a long text file> <clips folder>
capture/run_linux.sh local <stage folder> <litmus-app> <a long text file> <clips folder>
```

(over SSH it uses the machine's `~/litmus_run/demo/` only; either way one container at a time, 6 cores, 12 GB, no GPU,
from `capture/Dockerfile`, Ubuntu 24.04 pinned by digest). The Windows build is
never driven: nothing in this folder sends keystrokes or mouse input to a real desktop or records one. The speed
measurement needs Windows (it pins with `start /affinity` and reads Windows' process counters) and PowerShell.

Each step is a script that runs alone:

| Script | Does | Writes |
| --- | --- | --- |
| `tools/read_samples.mjs` | Reads the texts the footage shows with `litmus score --json` and stops if a text no longer reads as its caption says | `data/ad/readings.json` |
| `tools/fetch_sets.mjs` | Draws and fetches the two sets of long documents, written by people and written by AI (see below) | `<work>/sets` |
| `tools/measure_run.mjs` | Reads them with `litmus` pinned to one core, two ways, while `tools/sample_process.ps1` records the process's memory and processor time and the machine's utilization is traced | `data/ad/hype_run.json` |
| `tools/merge_gpu_use.mjs` | Merges the GPU measurements of one or more machines (each row keeps its machine) into the GPU panel's data, and writes their basis | `data/ad/gpu_use.json` |
| `tools/read_app_memory.mjs` | Summarizes the desktop app's memory from the capture's own samples | `data/ad/app_memory.json` |
| `capture/site_capture.mjs` | Loads the built site and the checker at a phone's size in headless Chromium, pastes, presses Check, saves frames | `<work>/checker` |
| `capture/linux_capture.py` | Inside the container: drives the Linux app and an editor on a virtual display and records it; the `memory` scene records nothing and samples the app's memory from `/proc` | `app.mp4`, `quick.mp4`, `memory.json` and their logs |
| `tools/prepare_footage.mjs` | Numbers the frames and finds when things happened in the footage's own time (the capture's white flash gives the offset; a settled reading is found from the footage) | `<work>/footage` |
| `tools/music.mjs` | Synthesizes the music and the sound effects on the beat grid | a WAV |
| `render.mjs` | Bundles `src/`, draws each frame in Chromium, encodes H.264, mixes and loudness-normalizes the sound, and for the wide cut makes the web encodes | the MP4s, a report each, and `litmus_wide_site.mp4`, `litmus_wide_readme.mp4`, `litmus_wide_poster.png` |

## How it is built

| Path | Holds |
| --- | --- |
| `data/ad/timeline.json` | The clock: 104 beats a minute (`authored_bpm` 120 is the tempo the scenes' own durations were written at), 4 to a bar, the shots in bars, the named cues (a sound and a beat each), and the alias that makes `wide` the ad's shots in another frame. The music and the picture both read it |
| `data/ad/readings.json`, `data/ad/hype_run.json`, `data/ad/app_memory.json` | Measured: the engine's readings of the footage's texts, the speed and memory run, and the desktop app's memory, each with its basis |
| `data/ad/gpu_use.json` | Measured on GPU machines by `python -m litmus_benchmarks gpu-use` (`../benchmarks`) and merged by `tools/merge_gpu_use.mjs`: each detector's GPU busy time and GPU memory, the GPU it was measured on, and the same for Litmus's engine, with the basis (what each read, the polls, what was tried). With no file the speed shot is drawn without the GPU panel |
| `data/ad/texts/` | The two texts the footage shows (`ai_essay.txt` pasted into the app; `notes.txt`, the editor's document) |
| `src/ad/` | One file per shot (`hook`, `app`, `quick`, `web`, `hype`, `end`), each drawing for the frame it is given (tall or wide), the clock (`clock.ts`), the footage compositor (`footage.ts`: frames, camera, pointer, card) and the shared pieces (`kit.ts`: ground, captions, the dip through dark at a cut) |
| `src/draw/`, `src/motion.ts`, `src/palette.ts` | The strip, the mark (read from `../app/assets/litmus_mark.svg` at build time), typography, easing, and the site's colors |
| `src/cuts.ts`, `src/main.ts`, `render.mjs` | The three cuts (`ad`, `teaser`, `wide`) and their frames; the page that draws a frame as a pure function of time; bundle, draw, encode |

Every frame is a pure function of time, drawn in headless Chromium on the CPU and encoded to H.264 (a 41.5-second ad takes
about 100 seconds to draw on a 16-thread machine). The footage is real frames, shown inside a rounded card seen through a
camera (a zoom and pan across the capture) with the pointer redrawn from the capture's log, so it can be larger and
smoother than the recording's own cursor. The footage's time is remapped so its moments land on the beats; it is never
sped past what a viewer can follow except across dead time (a pause between a paste and the press of Check), and never
shown slower than it was recorded: the film's tempo stretches the scenes' own motion, not the footage (`inScene` and
`live` in `src/ad/clock.ts`).

**Pace.** The tempo is one number, `bpm` in `data/ad/timeline.json`. The first cut ran at 120 and was too fast; 104 makes
every shot, cut, word, fade, and camera move 15.4% longer (36 s to 41.5 s, 8 s to 9.2 s; the teaser then got one more bar of end card for the site address, 11.5 s) with the cuts still on the bar
line, which more bars per shot could not do (a bar is a quarter of a three-bar shot). The scenes draw in "scene seconds"
at the authored tempo (`authored_bpm`, 120) and `src/main.ts` plays them `authored_bpm / bpm` times slower. Footage
anchors, the "shown at ×" factor, and the check cues are computed in real seconds, so a measured duration is never
stretched on screen. Where a shot gave a caption too little time at 120 (the app's third caption showed for 0.25 s),
its cues were moved, not the tempo.

Live preview: `node render.mjs ad --work <dir> --preview` prints an address that plays the cut in a browser (space pauses,
the arrow keys step a frame, a click on the bar seeks).

## What the numbers are

Every number is measured and stored with its basis. The values below are the current release's, from the files named.

| Shown | Value | Basis |
| --- | --- | --- |
| The reading of each text in the footage | `data/ad/readings.json` | `litmus score --json` on the text, with the release named in the file |
| Documents read | 120 (60 written by people, 60 written by AI), 595,890 tokens | `data/ad/hype_run.json`, `totals` |
| One core | Every `litmus` process is launched with `start /affinity 40` (core 6 only), so it never runs on another core | the operating system's affinity mask, set at launch |
| Tokens a second | 23,549, one process reading all 120 documents in 25.3 s; 13,786 when each document is read by a new process (0.133 s to start one, median of 7 runs on a one-word text); the engine's own benchmark says 24,229 | `data/ad/hype_run.json`, `totals`; the command line prints its readings only when it has read them all, so a per-document time needs a process per document, and the speed with the model loaded once needs the second pass |
| Memory | 66 MB shown: the largest working set (physical memory held) of any `litmus` process in the run, 65.1 MB; the private part peaked at 49.1 MB; the engine reports 57.2 MB | Windows' process counters read every 50 ms or so by `tools/sample_process.ps1`; the shown figure is rounded up |
| Processing share | `litmus` used 4.7% of the whole machine on average over the per-document run (0.75 of one core's worth, because starting a process is not processing); 5% to 6% at an instant while reading | the processor seconds of the `litmus` processes over the wall seconds and the machine's 16 logical cores; the machine's utilization (17% on average, so the "Other tools" row, the rest of the machine during the run, is about 13%) is `os.cpus()` every 100 ms, on an AMD Ryzen 7 3800X under Windows 10 that was doing other work; the ad names no other process |
| Desktop app memory | 156 MB resident with the model loaded and idle, 172 MB at its peak while checking a 39,763-character text, 80 MB of it private; not shown in the ad | `data/ad/app_memory.json`: the Linux build on a virtual display with software rendering, from `/proc`; the figure includes the graphics stack the app loads, so a machine with a graphics card differs |
| Playback | 43.2 s of recorded time shown in 8.1 s, "shown at ×5"; at the end the figures are the run's average processing and peak memory, not the last instant's | the factor is the recorded duration over the seconds the ad spends on it, computed from the data |
| Tallies | written by people: 57 of 60 read as human, 2 mixed, 1 unlike reference (95%, 3%, 2%), 0 read as AI; written by AI: 60 of 60 read as AI (all five models, with and without formatting) | the verdict is the conformal set at the 10% level: one origin names itself, several read "uncertain"; counts in `data/ad/hype_run.json`, `tally`, `tally_by_model` and `tally_by_formatting` |

The model behind every number here is the interim release named in the data files; they are provisional until the
final release replaces them (see the last section).

## The GPU panel

The speed shot adds one element when `data/ad/gpu_use.json` exists: a panel "GPU busy and GPU memory, by detector" with one row for Litmus
and one for each of the five open detectors the benchmark results cover, each on the same 120 documents: Fast-DetectGPT, Desklib v1.01,
Binoculars, ModernBERT (Drayson), and e5-small-lora (named as the benchmark figures name them). A row is the detector's name over the GPU
it was measured on ("A10" or "GH200"; Litmus's reads "CPU, one core"), a bar and a percentage for the share of time the GPU was busy while it
read (`utilization.gpu` of `nvidia-smi`, averaged), and under the percentage its own process's peak GPU memory. Memory follows the ad's own
convention (the process list's "MB" is MiB, so 1,024 to the GB): 16.0 GB is 16,362 MiB. Litmus's row reads "0%" and "none" only when the
measurement says so (no process of its own on the GPU and no busy time added over the idle baseline); otherwise it shows what was measured.
The bars fill in over a second when the panel appears and are the measured means, not a live reading. The panel never uses a group label:
the "Other tools" row of the process list is the rest of the machine during the Litmus run (the machine minus `litmus`, from
`data/ad/hype_run.json`), and the GPU panel names each detector, so the two cannot be read as the same group.

What the figures are, plainly (the file's `basis` says the same): the other detectors truncate each document to their input limit and read
61,440 tokens in all (Desklib, at 768, 92,160), while Litmus's engine read every token, 595,890. They are figures for what each detector's GPU did
while it read its own share, not for the same text, and nothing on screen says they read the whole documents. The fast detectors finished in a
second or two (ModernBERT 1.4 s and 8 polls, e5-small-lora 0.9 s and 7 polls), so their busy percentages rest on few polls; Fast-DetectGPT and
Binoculars read for 15 s and 5 s. Binoculars does not fit a 24 GB GPU (its worker ran out of memory loading on the A10 at 22.00 of 22.06 GiB), so its
row was measured on a GH200, which is why every row names its GPU. RADAR, OpenAI RoBERTa-base, MAGE Longformer and RoBERTa (HC3) are not shown (not
in the benchmark results; MAGE Longformer does not load under the image's transformers).

Layout. Tall: the panel sits under the two sets, which pushes the facts and the process list up and shortens the process list's two traces;
wide: it sits under the facts at the left and the counts move under the sets.

The scene refuses to invent it: with no `gpu_use.json` the page has no `window.__gpuUse`, the panel is not drawn, and the shot is the one it was
(only the row label differs); a file that is a pilot, lacks Litmus's row, or lacks a detector's figures or its machine stops the render.
`render.mjs --gpu-use <json>` names another file (a layout fixture lives outside `data/ad/`; a file with `"fixture": true` stamps the panel
"FIXTURE, NOT MEASURED"). The figures come from `../benchmarks/src/litmus_benchmarks/gpu_use.py`, run on GPU machines in the `litmus` image (see
`../benchmarks/README.md`); `tools/merge_gpu_use.mjs` merges the machines' JSON into `data/ad/gpu_use.json` (rows carry their machine; the basis and the notes
on what was tried are written in), and `make.mjs` runs neither.

## The documents

The speed segment reads two sets of 60 whole documents each, labeled on screen "Written by people" and "Written by AI".
The selection rule and its seed are in `tools/fetch_sets.mjs` and, with the list's hash, in `data/ad/hype_run.json`;
both sets are drawn at random with the same fixed seed and kept to the same length range, 12,000 to 40,000 characters
(about 3,000 to 9,000 tokens), and no document is cut.

| Set | Source and provenance | License |
| --- | --- | --- |
| Written by people | Rules and proposed rules from the Federal Register published 2017 to 2019 (federalregister.gov API and the plain text of each), drawn from the same ten months of each year; their authorship is certain because they were published before any language model could write them, and the Federal Register is not among Litmus's training sources. Median 16,260 characters | Works of the U.S. government, public domain |
| Written by AI | The test split of the StoryScope dataset (Hugging Face `jjrussell10/storyscope`, revision `befcb672`, published April 2026): stories each written in one generation by GPT-5.4, Claude Sonnet 4.6, DeepSeek V3.2, Kimi K2.5 or Gemini 3 Flash, 12 of each, one per prompt. It is a benchmark split, held out from Litmus's training (`corpus/sources.csv`: `storyscope_test`, `in_first_corpus` no). The text was fetched from the datasets server and checked identical to the pinned revision. Median 22,144 characters | MIT |

How the sets were chosen: the human source is public-domain text with certain dates that is not a training source; the
AI source is a public set of long, model-written documents with named generators and a held-out split, preferred over
writing documents for the demo. Neither was picked by how its documents read, and no source was dropped. Two limits go with the data: the
people-written documents are regulations and the AI-written ones are fiction, so a reading cannot be separated from
genre; and 37 of the 60 AI stories keep the markdown headings or bold lines their generator wrote, which the
regulations lack (the readings are the same, 100% AI, with and without them). The stories are also from the same
generators Litmus's training split came from, so the AI reading is in-distribution. The final measurement should add
a people-written set of the same genre (long fiction from before 2020) and an AI set from generators and a genre the
training did not see.

The ad shows only anonymous tiles and aggregates: no document is named and no named third party's document appears with
a verdict. A reading is evidence about where a text came from, never proof of who wrote it, and the ad's words are about
readings ("read as human", "read as AI", "uncertain or mixed"), not about authorship.

Scientific papers from arXiv were tried in an earlier draw, and set aside: the interim model read 18 of 22 of them as
"mixed", a limit of the interim model on equation-heavy PDF text.

## Sound

The music is made here, not borrowed: `tools/music.mjs` synthesizes it from oscillators, filtered noise, and envelopes
(a kick, a clap, hats, a bass, an arpeggio with a delay, a pad, risers, and four effects: hit, tick, ding, pop), on the
timeline's grid at 104 beats a minute in A minor (Am, F, C, G), with a small reverb. It is deterministic (a seeded noise
generator), so the same timeline always gives the same samples, and its energy follows the shots' sections. The sync is
defined by the grid: every shot is a whole number of bars, every cue a beat in `timeline.json`, and a scene anchors its
footage to its cues, so changing the tempo or a shot's length moves picture and sound together.

The mix is normalized in two passes to -14 LUFS integrated with a true-peak ceiling of -2 dBTP before encoding (AAC at
192 kbit/s, 48 kHz), and measured again on the finished file: see `<out>/litmus_ad.report.json` (a limiter at -2.5 dBFS after the normalizer keeps the encoded file under 0 dBFS; the AAC encode still measures about 1 dB above the ceiling on the ad and the wide cut, at -0.9 dBTP, because the codec overshoots a peak the normalizer placed at -2, and without the limiter the teaser measured +1 dBFS once the tempo changed). Third-party platforms
re-normalize loudness, third-party guides name -14 LUFS as the common target, and none of the platforms was found to
publish a figure of its own.

| Asset | Source and license |
| --- | --- |
| Music and sound effects | Synthesized by `tools/music.mjs` (this repository, PolyForm Noncommercial 1.0.0); no sample, loop, or recording is used |
| Source Serif 4 (headings), Inter (words), JetBrains Mono (numbers) | `@fontsource-variable/source-serif-4` 5.2.9, `@fontsource-variable/inter` 5.2.8, `@fontsource/jetbrains-mono` 5.2.8; SIL Open Font License 1.1 |
| The mark | `../app/assets/litmus_mark.svg` (this repository) |
| Footage of the Litmus app | The Linux build of Litmus, with the Ubuntu and Hack fonts egui bundles (Ubuntu Font Licence 1.0 and MIT) |
| Footage of a text editor | Mousepad (GPL-2.0-or-later), unmodified, with the Noto Sans font (SIL OFL 1.1) and the Adwaita dark theme (LGPL-2.1), shown on a virtual display |
| Text pasted and selected | `data/ad/texts/ai_essay.txt` and two paragraphs of `notes.txt` were written by a language model for this demo and are released with this repository; the sentence is written for the demo; one paragraph is from a nineteenth-century novel (public domain, from a Project Gutenberg mirror) |
| Documents | Federal Register (public domain) and the StoryScope test split (MIT); never shown, only counted |

No third-party logo, stock footage, or music is used. The end card says "Free", "Private", and "Fast", and its last two lines are "Open source" and "Free for noncommercial use"
(Logan, 2026-11-18). The license (PolyForm Noncommercial 1.0.0) is source available, not open source in the OSI sense; the
wording is his decision and worth a second look before the final render.

## Platform settings

All three platforms take vertical 9:16 at 1080 by 1920, H.264 in MP4, with AAC audio, and 30 fps is documented for Reels;
this ad is 30 fps at constant frame rate. The video is encoded at CRF 16 capped at 14 Mbit/s (high profile, BT.709, yuv420p,
faststart); the finished ad averages 3.6 Mbit/s because most of it is still. YouTube recommends 8 Mbit/s for 1080p at 30
fps; the CRF encode is above the visual quality that rate gives and under the caps platforms re-encode to. X accepts the
same file.

The first frame carries the hook ("Is this AI?" fully drawn at frame 0, with its first beat), the hook line is on screen
for the first two seconds, and every spoken-style line is a burned-in caption, so the ad reads with the sound off. Text stays
inside x 70 to 1010 and, for anything that must be read, y 270 to 1480; Meta's ad percentages put the top margin at about 14%
(269 px), the bottom at up to 35%, and the sides at 6%, and Reels crops its profile grid to 3:4, so the centered column is
what survives. The phone and the cards extend lower, but only with parts that need no reading (a status bar, the foot
of the text).

## Embedding

The wide cut (`litmus_wide.mp4`, 1920 by 1080, 41.5 s, burned-in captions, so it reads muted) is made for the GitHub
README and the github.io site. `node render.mjs wide` (and `make.mjs`) also writes, beside it in `<out>/`:

| File | What | Size |
| --- | --- | --- |
| `litmus_wide_site.mp4` | 1920 by 1080, H.264 high profile, CRF 21 capped at 6 Mbit/s, AAC 128 kbit/s, index at the front | 12.0 MB |
| `litmus_wide_readme.mp4` | 1280 by 720, CRF 19 capped at 3 Mbit/s, AAC 96 kbit/s, the best quality that stays under 9.5 MB (the encoder steps CRF up until it does) | 8.1 MB |
| `litmus_wide_poster.png` | 1920 by 1080, the last frame, the whole end card with the address `lhallee.github.io/litmus`: the poster of the site's video | 0.46 MB |
| `litmus_wide_preview.webp` | The whole cut as a silent animated WebP, 800 pixels wide at 10 frames a second, quality 70 (stepped down by 10 until it fits 6 MB): the README's moving image, linked to the site | 4.7 MB |

`litmus_wide.mp4` itself is 18.2 MB (CRF 16, the master); `--web-only` remakes just these four from it. The release's
`film` stage copies the preview from `<work>/media/` to `docs/images/`, so a new film reaches the README with the release.

**What the research found (read 2026-11-18).**

| Question | Finding | Source |
| --- | --- | --- |
| What GitHub renders in a README | A video dragged into the web editor is uploaded and its address (`https://github.com/user-attachments/assets/<id>`, an observed form) is put in the text; on its own line it renders as an inline player. MP4, MOV, and WebM are supported, and GitHub recommends H.264 for the widest browser support. A `<video>` tag, or the address wrapped in other markup, did not render a player in April 2024 tests, and no way to autoplay or loop an uploaded video was found. An image link is the fallback that always works; animated GIF, WebP, and AVIF play on their own as images, silently | [Attaching files](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files); [a 2024 test of each method](https://github.com/george-hawkins/video-in-github-markdown) (third party, April 2024) |
| Size limits | Upload: 10 MB for a video in a repository owned by a user or organization on a free plan, 100 MB on a paid plan; 10 MB for an image or GIF; 25 MB for other files. A file committed through the browser: 25 MiB; git warns at 50 MiB and GitHub blocks at 100 MiB | [Attaching files](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files); [About large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) |
| The site | Pages sites may be up to 1 GB, about 1 GB is advised for the source repository, about 100 GB a month of soft bandwidth, a deployment is cut off at 10 minutes | [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits) |
| Autoplay in a page | A video autoplays when it has no audible track or its audio is muted; `playsinline` is required for autoplay in Safari; a failed `play()` call (`NotAllowedError`) is the page's cue to leave the poster and controls | [MDN, Autoplay guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay) |
| Release assets | A release asset's address redirects to a short-lived signed address that sends the file as `application/octet-stream` with `Content-Disposition: attachment` and no CORS header, with byte ranges allowed (read from the headers of an unrelated public asset) | checked with `curl -I` on 2026-11-18 |

**Recommendation.**

- **README: the animated preview (`litmus_wide_preview.webp`, 4.7 MB), linked to the site.** Decided 2026-11-19 after
  the README's linked poster was found not to render (it was never copied to `docs/images/`) and Logan asked for a GIF,
  something shorter, or a picture. An animated image plays on its own, loops, and needs no manual upload, so a release
  refreshes it with no step by hand. A GIF of the whole cut was 19 to 23 MB even at 640 to 800 pixels, over the 10 MB image
  limit and slow to load, while animated WebP at 800 pixels is 4.7 MB with the captions sharp; current Chrome, Edge,
  Firefox, and Safari play animated WebP. The sound plays on the site, which the image links to. The uploaded video
  (`litmus_wide_readme.mp4`) stays available for a player with sound, but it needs the manual upload described below.
- **Site: a `<video>` element on the page, from the site's own files (`litmus_wide_site.mp4` and the poster).** The site
  makes no request to any other host (`site/README.md`), a release asset cannot serve a `<video>` reliably (the
  redirect, the forced download type, no CORS), and Pages allows far more than 12 MB. The files live in the `gh-pages`
  branch under `assets/media/`, which `release/RELEASING.md` step 5 already replaces by hand and which already carries the
  66 MB model, and not in `main`. The release build copies only `site/assets` from git (`assemble_site` in
  `release/build_release.py`) and its link check wants every `src` to be a file in the built folder, so the video must be
  copied into the built folder as one more step; this task did not touch either file.

**Markup for the README** (`PUBLIC_README.md`, under the one-line description and the site's links; applied):

```html
<a href="https://lhallee.github.io/litmus/"><img alt="Litmus in a short film, playing silently: ..." src="docs/images/litmus_wide_preview.webp" width="100%"></a>
```

For a player with sound instead, upload `litmus_wide_readme.mp4` through the GitHub editor and put the address it
inserts on a line of its own (`https://github.com/user-attachments/assets/<id>`).

**Markup for the site** (a page of `site/`, for example under the home page's one-line description; not applied here).
It starts muted and in a loop where the browser allows it, shows controls so a visitor can unmute, and leaves the poster
and controls where autoplay is refused (iOS Low Power Mode is a known case; this was not tested) or the visitor asks for
reduced motion:

```html
<video class="film" controls muted loop playsinline preload="metadata" width="1920" height="1080" poster="assets/media/litmus_wide_poster.png">
  <source src="assets/media/litmus_wide_site.mp4" type="video/mp4">
  <a href="assets/media/litmus_wide_site.mp4">Watch the film (MP4, 12 MB)</a>
</video>
```

```css
.film { display: block; width: 100%; height: auto; aspect-ratio: 16 / 9; border-radius: 12px; background: #16141A; }
```

```js
// in site.js: play on load unless the visitor asked for less motion; a refusal leaves the poster and the controls
const film = document.querySelector('video.film');
if (film && !matchMedia('(prefers-reduced-motion: reduce)').matches) film.play().catch(() => {});
```

Looping goes from the held end card straight to the hook (a hard cut, which the hook's first frame, fully drawn, takes
well). The video has an audio track; `muted` keeps it silent until the visitor unmutes.

**Where the files live.** In the work folder's `out/`, never in this tree: they are build outputs, regenerated by every
release's `make.mjs`. `ws validate` budgets only experiment folders (1 MB), `results/` files (5 MB, 25 MB a folder), and
figure sets (a warning at 5 MB); it names no limit for a file under `projects/`, so a 12 MB MP4 there would pass, and the
workspace rule against large generated files in authored folders is the one that forbids it. The video is final only
once the final model's numbers are in (the speed shot and the tallies are the interim model's), so nothing is published
before then.

**Not verified.** No upload to GitHub was made (it needs the browser and Logan's account), so the inline player, its
mobile behavior in the GitHub app or a phone's browser, and whether the repository's plan is free or paid are untested; the
`<video>` markup was not run on a real Pages deployment or on iOS (it was not opened in a browser at all); the 2024 test of
the `<video>` tag in a README is third-party and 19 months old.

## What it touches

For the project's capability declarations (`tools/security/capabilities.json`): the demo starts processes (`ffmpeg`
through `ffmpeg-static`, headless Chromium through Playwright, `litmus` through `cmd.exe /c start /affinity` for
`score` and `bench`, `powershell.exe -File tools/sample_process.ps1` to read the memory and processor time of processes
named `litmus` (it reads no other process), `node`, and for capture `ssh`, `scp`, and `docker` on a machine you name);
reaches the network (the Federal Register's public API and document text, one request every 1.2 seconds; the Hugging
Face datasets server's rows of one public dataset, one request a second; and `npm ci`), serves the built site from a
local HTTP server on 127.0.0.1 for the capture, reads the per-core CPU times and the memory in use of the machine, and
writes files only in the work folder and in `data/ad/`. It never reads a credential and never touches the Windows desktop.

## Research behind the choices

How programmatic and agent-written product videos are made in 2026, what each tool is good at, and the license, read
from each project's own pages and package metadata (versions and license fields from the npm registry); several
comparison articles are vendor blogs and are marked as such.

| Tool | What it is and what it is good at | License | Link |
| --- | --- | --- | --- |
| Remotion 4.0.545 | React components rendered frame by frame, a Studio, cloud rendering on AWS Lambda, and official Agent Skills (rule files that teach a coding agent its API, installed with `npx skills add remotion-dev/skills` or a Claude Code plugin). Best for templated, data-driven video and for teams that write React | The Remotion License: source available, not open source. Free for an individual, for a for-profit organization with up to 3 employees, and for non-profits; a paid company license otherwise (third-party pricing pages list $25 a seat a month, or $0.01 a render with a $100 monthly minimum; check remotion.dev) | [license](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md), [Claude Code guide](https://www.remotion.dev/docs/ai/claude-code) |
| HyperFrames 0.9.67 (HeyGen) | Compositions are plain HTML and CSS with GSAP timelines registered as seekable; headless Chrome seeks each frame and FFmpeg encodes (the architecture this folder uses); audio is `<audio>` elements with timing attributes and a mixer; a CLI (`init`, `preview`, `render`) and skills for Claude Code and other agents. Young: 0.9.x, published the same day it was read, with some skills reserved for a paid tier | Apache-2.0 | [repository](https://github.com/heygen-com/hyperframes), [Claude Code skills roundup](https://hyperframes.heygen.com/blog/best-claude-code-skills-for-video) (vendor) |
| Motion Canvas 3.17.2 | TypeScript generator scenes drawn on a canvas, with a real-time editor; best for narrated technical explainers synced to a voiceover. Last stable release in early 2025 | MIT | [repository](https://github.com/motion-canvas/motion-canvas) |
| Revideo 0.11.0 | A fork of Motion Canvas rebuilt as a library for headless, batch rendering (ads at scale, automated shorts); small ecosystem | MIT | [repository](https://github.com/redotvideo/revideo) |
| GSAP 3.15.1 | The standard timeline and easing library; since 2025 free for commercial use under its own "no charge" license, formerly paid plugins (SplitText, MorphSVG) included; forbids building a competing no-code animation tool | GSAP Standard License (source available, not OSI) | [license](https://gsap.com/community/standard-license/) |
| Theatre.js (@theatre/core 0.7.2) | A visual animation sequencer with an editor, for hand-keyframed motion | Apache-2.0 | package metadata only |
| Rive (@rive-app/canvas 2.51.0), Lottie (lottie-web 5.13.0) | Pre-authored vector animation played in a page: Rive from its editor, Lottie from After Effects. Good for icons and UI motion; both need their authoring tool, which this ad does not | MIT (runtimes) | package metadata only |
| Three.js 0.187.0 | WebGL scenes and shaders for particle or gradient backgrounds; in a headless browser it renders on the CPU, so slowly | MIT | package metadata only |

What the practice sources agree on, for short vertical video ([Hootsuite 2026](https://blog.hootsuite.com/short-form-video/),
[Jetfuel](https://jetfuel.agency/how-to-optimize-short-form-video-content-for-success/), [Clipspeed](https://www.clipspeed.ai/blog/hook-first-second-video-retention.html),
[Biteable](https://biteable.com/blog/best-practices-short-social-video/), [Revid](https://www.revid.ai/blog/content-creation-best-practices)):
land the hook in the first one to two seconds, with the key information in the picture and not only in speech; burn captions
into the video; and check the video muted. The often-quoted share of viewers who watch without sound (92%) traces to a
2019 survey of 5,616 U.S. consumers and is dated, and one platform's own research finds its users value sound, so this
ad is built to work either way and carries music. For specs ([Sprout Social](https://sproutsocial.com/insights/social-media-video-specs-guide/),
[Hookspring](https://hookspring.com/blog/short-form-video-specs), [Postplanify safe zones](https://postplanify.com/blog/social-media-safe-zones-2026-complete-guide)),
the guides disagree about safe zones and bitrates and most of their figures are not platform-published, so the ad keeps
to the conservative union (see Platform settings). On agent-written video, the practitioners' consistent warning
([Demo Editor](https://demoeditor.com/articles/remotion-claude-code-launch-video), [Digital Applied](https://www.digitalapplied.com/blog/video-as-code-remotion-agentic-generation-2026))
is that code-made video looks generic unless real captures, real timing, and real taste go into it, which is why this ad
is built from real footage, on a beat grid, with the pointer and the camera authored.

**The stack, and why.** The ad keeps this folder's own renderer: TypeScript scenes that are pure functions of time,
bundled by esbuild, drawn frame by frame in headless Chromium by Playwright, and encoded by FFmpeg. It is the same
architecture HyperFrames packages, and it already does what the ad needs: deterministic frames, real frame sequences
composited with a camera, data-driven scenes, and audio mixed in a separate step. Adopting HyperFrames would add a
dependency whose API was moving daily and whose licensing has a paid tier, to replace code that works; GSAP would add a
license with a no-compete clause for easing curves that `src/motion.ts` already has; Motion Canvas and Revideo solve
voiceover-synced explainers and headless batch rendering, neither of which this ad is. Remotion is the tool most people
reach for with Claude Code, and the choice not to use it is a licensing one stated on the merits: Remotion's free tier
covers an individual and this is a noncommercial project, so it would cost its author nothing, but this repository is
published under PolyForm Noncommercial and anyone at a company of four or more who re-rendered the ad from it would need
a Remotion company license, which the repository should not impose. Plain HTML and SVG drawn by Playwright and encoded by
FFmpeg do the same job under permissive terms.

## Dependencies

The film's source is part of Litmus, under the PolyForm Noncommercial License 1.0.0 (`../LICENSE`). Nothing here is
bundled into Litmus itself; these tools render the ad.

| Dependency | Version | License | Role |
| --- | --- | --- | --- |
| esbuild | 0.28.2 | MIT | Bundles the TypeScript and fonts |
| playwright-core | 1.58.2 | Apache-2.0 | Drives headless Chromium to draw frames and to capture the page |
| Chromium (Playwright's build) | 1208 | BSD-3-Clause and others (Chromium's own notices) | Draws the frames |
| ffmpeg-static | 5.3.0 | GPL-3.0-or-later (the package and its FFmpeg 6.1.1 build from gyan.dev) | Encodes H.264 and AAC and measures loudness; run as a separate program, so its license does not reach the ad or this source |
| typescript | 5.9.3 | Apache-2.0 | Type checking only (`tsc -p tsconfig.json`) |
| @fontsource-variable/source-serif-4, @fontsource-variable/inter, @fontsource/jetbrains-mono | 5.2.9, 5.2.8, 5.2.8 | SIL Open Font License 1.1 | The ad's three faces |
| Docker image `capture/Dockerfile` | Ubuntu 22.04 with FFmpeg, Mousepad, openbox, and the packages named in it | each package's own license | The virtual display the Linux footage is recorded on |

## A new release's film

The film released with 1.0.0 was never rendered from it. `data/ad/readings.json`, `data/ad/hype_run.json` and
`data/ad/app_memory.json` are the interim model's (step 4,999, measured 2026-11-18), so the film shows 23,549 tokens a
second and 66 MB where 1.0.0 measured 28,267 tokens a second in `litmus bench`. Logan asked on 2026-11-19 for the film to
be rendered again for 1.1. In order:

1. Run the release through `verify` as usual. The builds take the film already in `<work>/media/`, the old one.
2. On the Windows desktop with Docker Desktop running (Logan allowed it for short Docker jobs on 2026-11-19; it runs the
   x86_64 Linux build natively, which the aarch64 GH200s cannot), capture from the release's Linux build:
   `bash capture/run_linux.sh local <work>/ad/capture_stage <work>/build/linux/litmus-app <work>/measure/document.txt <work>/ad/clips`.
   The image build and the three scenes take minutes. Over SSH, the same script runs on any x86_64 Linux machine with Docker.
3. On the Windows desktop, with the machine otherwise quiet (the speed run is pinned to one core and counts the
   machine's utilization): `npm ci` and `npx playwright install chromium` in `demo/` once, then
   `py release/cut.py run --work <work> --only ad --ad-clips <clips folder>`. It reads the footage's texts again and
   stops if the new model reads one differently from its caption (then replace the text and capture again), measures
   speed and memory, recaptures the browser checker, and renders every cut into `<work>/ad/out/`.
4. Review the stills in `<work>/ad/out/stills/wide/`, then copy `litmus_wide_site.mp4`, `litmus_wide_poster.png`, and
   `litmus_wide_preview.webp` from `<work>/ad/out/` to `<work>/media/`.
5. Run `film`, `build-windows`, and `verify` again, so the README's preview and the site's film are the new ones, before
   `assemble`.

- TODO(film_render_1_1): render the film from the 1.1 release by the steps above; it waits on the 1.1 release.

What else to know before rendering:

- The GPU panel (`data/ad/gpu_use.json`) keeps its 2026-11-18 measurement unless `python -m litmus_benchmarks gpu-use`
  is run again on a GPU machine; it measures the other detectors, which a new Litmus does not change.
- The two document sets differ in genre and the AI set is in-distribution (see "The documents"); the final measurement
  should use a genre-matched, people-written set and an AI set the training did not see.
- The memory figure is the command line's. The desktop app holds more (156 to 172 MB on the Linux build with software
  rendering), and a release with a different model or engine changes both.
- The footage's texts were chosen so the interim model reads them clearly (the model-written essay as AI, a short sentence
  as uncertain, a novel's narration as human); `tools/read_samples.mjs` stops the pipeline if the final model reads one
  differently, and the text is then replaced and `capture/run_linux.sh` run again.
- The scenes anchor the footage's press of Check to the reading's arrival; a release that reads at another speed moves the
  `app_check` and `web_check` sound cues by the difference (`data/ad/timeline.json`).
- The Linux footage is recorded from the Linux build; the final Windows and Linux programs are rebuilt from the release
  folder, so record again from the final Linux build.
- The tallies on screen are whatever the final model reads; nothing in the scene assumes 100% or 0%.
