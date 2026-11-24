# Litmus site

The public site at `https://lhallee.github.io/litmus/`: the product's one home, for people who use Litmus rather than
develop it. Plain static files, one stylesheet, one small script, no framework, no build step, and no request to any
other host. Every link is relative, so the folder serves under `/litmus/` or at any other prefix.

| File | Holds |
| --- | --- |
| `index.html` | Home: the one-line description, the two primary actions, a screenshot, the short film, four points, what a reading tells you, the three ways and where the text goes, and a results teaser |
| `use.html` | The in-browser app (`app/`) in a frame under the site's navigation |
| `download.html` | The Windows download and its size, the first run, how to check a download (collapsed), why download (speed, quick check), other platforms, the command line |
| `how.html` | How a reading is made, plain first, each step with a "More detail" layer, and what it cannot tell. Each diagram is drawn twice: a wide `svg.wide` and a stacked `svg.narrow` shown under 48rem; the shared arrow marker and strip gradient sit in one `svg.svg-defs` at the top |
| `reading.html` | Reading a result: verdicts, the strip, the chance of AI and its range, the heat map, our recommendations |
| `benchmarks.html` | Results, drawn from `assets/results.json` and `assets/figures/` |
| `privacy.html`, `license.html` | Where the text goes in each of the three ways, how to check it, the contribute site's terms in short; the license |
| `style.css` | Tokens for both looks from `app/src/theme.rs`. Dark is the look of every visit, with or without JavaScript and whatever the system asks for; the light look applies only under `data-theme="light"`, which the toggle sets |
| `site.js` | The theme toggle (one `localStorage` key, `litmus.site.theme`, the only way to the light look; `use.html` opens the app at `app/#embed-dark` or `app/#embed-light`, which hides the app's own toggle, and changes that fragment in place when the toggle is pressed, so the app follows without reloading), and filling numbers, addresses, figures, and tables from `assets/` |
| `litmus_mark.svg` | The mark, a byte-identical copy of `../app/assets/litmus_mark.svg` |
| `assets/screens/` | Screenshots of the program, copied from `docs/images/` (`reading_*` from `app_simple_*`, `explain_*` from `app_full_explain_*`) |
| `assets/results.json`, `assets/links.json` | The numbers and the addresses that are not known when a page is written (below) |
| `assets/figures/` | The results figures (below) |
| `assets/media/` | The home page's film: `litmus_wide_site.mp4` and its poster `litmus_wide_poster.png`, the wide cut's web encodes from `demo/` (`node render.mjs wide`). Not kept in git: a release copies them into the built site, whose link check requires them |
| `app/` | The host page and worker of the in-browser app; the release build adds the module and the model's files |

Every page carries the same header and footer; when you change one, change all eight pages.

## Data the pages read

No number or address that changes with a release is typed into a page. `site.js` reads two files from `assets/` and
fills the elements marked with `data-value`, `data-show`, `data-hide`, and `data-link`. A value that is missing,
`null`, or `""` leaves the page's own wording ("being finalized", "coming soon", "not available yet"), so an empty
file is a valid state and never shows an invented value. With JavaScript off, elements marked `needs-js` (the
tables, figures, and filled-in facts) are hidden, links keep their written `href`, and the Results page says the
numbers are in `assets/results.json`. Never name a data attribute `data-href` or `data-src`: the release build's link
check reads any `href=` or `src=` as a file the folder must hold.

### `assets/results.json`

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | integer | 1 |
| `model.name`, `model.version`, `model.date` | string | The detector named in `rows` as ours (`Litmus`), its release version, and the date of the results (YYYY-MM-DD) |
| `primary_task` | string | The task shown by default: `ai_vs_human`, `any_ai_vs_human`, or `mixed_vs_human` |
| `texts_per_benchmark` | integer | The seeded sample size per benchmark set |
| `benchmarks[]` | list | In display order: `id` (the benchmark pipeline's `benchmark` name, or `test` for Litmus's own test split), `name`, `about`, and `kind`, `two_class` or `flag_rate` (a set with one origin, such as APT-Eval) |
| `rows[]` | list | One row per benchmark, detector, and task (below) |
| `costs[]` | list | `detector`, `tokens_per_second` (one processor core), `measured` (true) or estimated (false), `machine` (string) |
| `release.*` | see below | The facts of the current release |

A row of `rows[]`:

| Field | Type | Meaning |
| --- | --- | --- |
| `benchmark` | string | A `benchmarks[].id` |
| `detector` | string | As named in the comparison; `model.name` marks ours |
| `task` | string | `ai_vs_human`, `any_ai_vs_human`, or `mixed_vs_human` |
| `pr_auc`, `roc_auc`, `balanced_mcc` | number in [0, 1] or [-1, 1] | Two-class metrics |
| `tpr_at_fpr_1`, `tpr_at_fpr_5` | number in [0, 1] | True positive rate at 1% and 5% false positive rate |
| `flagged_at_fpr_1`, `flagged_at_fpr_5` | number in [0, 1] | For `flag_rate` sets: share of the set flagged at the detector's 1% and 5% FPR thresholds |
| `<metric>_low`, `<metric>_high` | number | The 95% bootstrap interval of each metric above |
| `positives`, `negatives` | integer | Texts in each class; for a `flag_rate` set, `texts` instead |
| `in_distribution` | boolean or `null` | True when this detector trained on this benchmark (its official training split, or for another detector possibly its test texts), false when it did not, `null` when no source says. The tables mark true rows with a "T" and a legend. Written by `python -m litmus_benchmarks results` from `benchmarks/catalog.csv` for Litmus and `TRAINED_ON` for the others |

`release.*`: `version`, `date`, `app_exe_bytes` and `app_exe_sha256` (`litmus-app.exe`), `cli_exe_bytes` and
`cli_exe_sha256` (`litmus.exe`), `browser_model_bytes` (the model's three files, as `manifest.json`'s
`sizes.browser_model` counts them), `browser_page_bytes` (`sizes.browser_page`), `app_tokens_per_second` and
`browser_tokens_per_second` (one core of `speed_machine`, the processor's name), `speed_document_tokens`,
`app_document_seconds`, and `browser_document_seconds` (the document those speeds were measured on). When the speed
measurement also read that document on every logical processor, as the app does, `app_processors`,
`app_tokens_per_second_all_processors`, and `app_document_seconds_all_processors` hold that reading, and the pages say
nothing of it while they are absent. `chance_range` is present only for a release whose `config.json` carries
`calibration.between_sources` (1.2.0 on). It holds `level_percent`, `odds` (how many times the odds move either way),
`log_odds`, `human_sources` and `not_human_sources` (the sources the fit read), and the range the programs print at
a chance of 5%, 20%, 50%, 80%, 95%, and 99% as `at_5_low`, `at_5_high`, and so on. `reading.html` and `how.html` read
them from `results.release.chance_range.*`, so a page never types a range.

### Example strips

An example strip is `<div class="strip" data-chance="0.93">` with its `strip-band` and `strip-mark`. The mark sits at
the chance (`--x`), and the band covers the chance's range (`--lo` and `--hi`), which the docs stage computes: it
widens the chance by the release's `between_sources` half width in log odds, as the engine does. A strip for a long
document whose sections disagree also carries `data-interval="low,high"`, the interval to widen instead of the
chance alone. Never type `--lo` or `--hi`: the stage rewrites them, and it refuses a strip whose mark is not at its
chance or whose chance lies outside its interval. For a release without `calibration.between_sources` the band is the
interval as given.

### `assets/links.json`

| Field | Meaning |
| --- | --- |
| `schema` | 1 |
| `training_guide` | "How Litmus was trained, step by step" (`docs/TRAINING.md` on GitHub); the pages also carry this address for readers without JavaScript |
| `contribute_site` | The contribute site's address; empty shows "coming soon" |
| `page_counter` | The address the browser page counts its openings at (`[counter] count_url` in `release/targets.toml`); empty means nothing is counted, and `privacy.html` words itself by it |
| `model_on_hugging_face` | The model's Hugging Face page; empty shows "coming soon" |
| `download_macos_app`, `download_macos_intel_app`, `download_linux_app` | A program's download (macOS on Apple silicon, macOS on Intel, Linux x86-64); empty shows "not available yet" |

The Windows downloads need no entry: the pages link to
`https://github.com/lhallee/litmus/releases/latest/download/litmus-app.exe` (and `litmus.exe`, `SHA256SUMS`), which
GitHub serves from the latest release.

### `assets/figures/`

`<name>_dark.png` and `<name>_light.png` for each of `headline`, `raid`, `pareto`, `strata`, and `origins`, drawn for
the site's dark and light grounds, with no text beyond axis labels and panel letters: the captions are in
`benchmarks.html`. A missing figure leaves a "being finalized" box.

## Preview

Serve the folder under a prefix, as GitHub Pages does, for example with Python's `http.server` from the folder that
holds a `litmus` copy of it, and open `/litmus/`. `use.html` needs the release build's module and model files in
`app/`. Links to the repository point at `lhallee/litmus`.
