# Developing Litmus

This page is for people who build Litmus from source, change it, run its tests, check its security claims, or read
its full results. The [README](../README.md) is for people who use it. [SECURITY.md](../SECURITY.md) says what each part
does with the machine and the text, and [AUDIT.md](AUDIT.md) lists the checks a reviewer, or a reviewer's AI agent,
can run.

## Run it from source

Every way in reads a release folder holding `model.safetensors`, `config.json`, and `tokenizer.json`. Releases are
at <!--litmus:t:hub_model_link-->https://huggingface.co/lhallee/litmus<!--/litmus-->. A placeholder with random weights shows how everything
works without one; its readings mean nothing, and the app says "Placeholder model, readings are
meaningless" on every screen.

```sh
pip install numpy huggingface_hub
python student/reference.py init --config s192_l4 --out placeholder
hf download answerdotai/ModernBERT-large tokenizer.json --revision 45bb4654a4d5aaff24dd11d4781fa46d39bf8c13 --local-dir placeholder
```

The tokenizer is ModernBERT's, at the revision `student/reference.py` pins.

**The browser page.** To build it you need Rust with the `wasm32-unknown-unknown` target and `wasm-bindgen-cli`
0.2.126, the version the app pins:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126 --locked
app/scripts/build_web.sh dist placeholder
python -m http.server 8000 --directory dist        # then open http://localhost:8000/
```

The page's only requests are for its own files and the model's three, from its own address; the text you check goes
to the page's worker and nowhere else. Only a build with the `contribute` feature, which is the separate contribute
site's, compiles any code that sends a request.

**The desktop app** (built, tested, and used on Windows; the release workflow builds and tests it on Linux and on
macOS, starts it there, and compares its reading with the command line's):

```sh
cd app
LITMUS_MODEL=../placeholder cargo run --release
```

It opens at the Simple detail level in the dark theme; "More detail", the Detail switch, or Ctrl+M moves to Full. It keeps a history and a cache, checks a folder as a batch, and exports CSV and JSON. An opt-in quick check,
off by default and switched on in Settings, reads the text selected in any app on a global hotkey and shows the verdict
in a small window by the pointer. The [app's README](../app/README.md) has the rest.

**The command line:**

```sh
cd engine
cargo build --release
./target/release/litmus score --model ../placeholder essay.txt
./target/release/litmus score --model ../placeholder --map --explain report.pdf
./target/release/litmus info --model ../placeholder
```

`litmus score` reads files (`.txt`, `.md`, `.html`, `.docx`, `.pdf`), folders recursively, or standard input (`-`),
and prints the verdict, the chance of AI with its range, the origins not ruled out, and the strip in characters as
well as color. `--json` prints every field, `--csv` one row per file, and `LITMUS_MODEL` can stand in for `--model`.
A text's 256-token blocks are spread over the machine's processors (`--threads N` sets how many), and the reading is
the same bit for bit on any number of them.
`litmus mcp` serves the same readings to an AI agent over the Model Context Protocol; [Use Litmus from an AI
agent](USE_WITH_AGENTS.md) has the setup, and its tests are in the [engine's README](../engine/README.md#tests).

**The Python module** (`litmus_engine`, a wheel for Python 3.10 and later built from the same engine):

```sh
pip install maturin==1.16.0
maturin build --release --locked -m engine/litmus-py/Cargo.toml --out dist
pip install --no-index --find-links dist litmus-engine
```

```python
import litmus_engine

model = litmus_engine.Model.from_path("placeholder")
reading = model.score(text)
reading["label"]                       # "human", "AI", "mixed", "AI or mixed", "uncertain" or "unlike reference"
reading["chance"]["value"], reading["chance"]["interval"]    # the chance of AI and its range
reading["verdict"]["sets"]["0.1"]      # the origins not ruled out at the 10% level
reading["cells"]                       # the heat map, one cell per 128 tokens
```

`score` releases the GIL, so threads can share one model. The [engine's README](../engine/README.md) lists every call.

## Tests

Each Cargo workspace pins its compiler in `rust-toolchain.toml` (Rust 1.100.0, which rustup installs on first use),
so every build of the same source uses the same compiler. The engine supports Rust 1.91 or newer and the app 1.95 or
newer; the app is its own Cargo workspace, so its GUI dependencies stay out of the engine's lock file and the wheel.

One command runs every suite and reports each one's result and time:

```sh
python tools/run_tests.py                 # every suite except the rebuild
python tools/run_tests.py security engine app
python tools/run_tests.py --all           # also rebuild and compare hashes
python tools/run_tests.py --list          # what each suite proves
```

| Suite | What passing it shows | Time |
| --- | --- | --- |
| `security` | Every network, process, clipboard, registry, input, file-writing, storage, and dynamic-code use in the source is declared in `tools/security/capabilities.json` and every `unsafe` is counted; the Rust dependencies match their lock files, include no network crate, and carry accepted licenses; the model verifier catches a changed file | 2 s |
| `engine` | The engine's kernels agree across SIMD backends and with the scalar path; the release format, chunks, windows, cells, and the strip; file reading, folder walks, and the cache; the command line | 99 s from a clean build |
| `app` | The app's wording, strip geometry, heat-map cells, history, settings, sorting, one-instance lock, and icon | 174 s from a clean build |
| `corpus`, `teacher`, `student`, `benchmarks`, `contribute` | Each Python package on small fixtures, on a CPU, offline | not measured |
| `build` (with `--all`) | Two builds of the command line, from different paths, and of the WebAssembly module, at one path, give the same bytes | about 7 minutes |

Times are from one run on an AMD Ryzen 7 3800X under Windows 10, with other work on the machine.

The engine's parity tests against `student/reference.py`, which read the fixtures `reference.py fixtures` writes, and
the wheel's tests are listed in the [engine's README](../engine/README.md). Each Python package declares its pinned
dependencies in its `pyproject.toml`; install them before running its suite. Training runs in Docker on one GPU, and
[`launch/`](../launch/README.md) gives the order.

## Check the security claims

[SECURITY.md](../SECURITY.md) states what each part does with the machine, the files, and the text, each claim tied to
its code and to a check that fails if it stops being true.

- **A capability manifest.** [`tools/security/capabilities.json`](../tools/security/capabilities.json) lists every file that can
  reach the network, start a process, use the clipboard, the registry, or global input, read another app's text,
  write files, use browser storage, or load pickled data, with the reason, and counts every `unsafe` in the Rust
  code. `tools/security/check_capabilities.py`, which needs nothing beyond Python's standard library, scans the source and
  fails on any use the manifest does not declare.
- **Pinned dependencies.** `tools/security/check_dependencies.py` fails when a Rust dependency changes without a reviewed
  record in [`tools/security/dependencies.json`](../tools/security/dependencies.json), when one is a network crate, or when its
  license is not one the project accepts.
- **A guide for auditors and their AI agents.** [AUDIT.md](AUDIT.md) lists the claims, where each is implemented,
  the commands that test them, and what to distrust until checked. It gives checks, not conclusions.

A change that adds a network call, a process, a file write, the clipboard, the registry, global input, browser
storage, or `unsafe` Rust declares it with its reason in `tools/security/capabilities.json`; a changed `Cargo.lock` is
recorded with `python tools/security/check_dependencies.py --write` and its diff reviewed.

## Verify the build

```sh
python tools/verify_build.py                          # the command line and the WebAssembly module
python tools/verify_build.py --artifacts cli wasm app # and the desktop app
python tools/verify_build.py --model <release folder> --sums <SHA256SUMS> --twice   # a release's two programs
python tools/verify_model.py path/to/release --sums release/SHA256SUMS
```

`tools/verify_build.py` snapshots `engine/` and `app/`, builds each artifact twice with the pinned toolchain,
`--locked` dependencies, and flags that keep the source's path, the Cargo registry's path, and the link time out of
the output, and compares SHA-256 hashes. With `--model` and `--sums` it rebuilds a release's two programs with its
model built in, at the fixed path, through the same functions `release/build_release.py` uses, and ends with
`MATCH: ...` when both are the published files byte for byte; without them it builds each artifact twice and
compares the two.
`tools/verify_model.py` checks a release folder's files against the hashes its `config.json` and any `SHA256SUMS`
state.

Measured on one machine, Windows 10 x86-64, Rust 1.100.0, MSVC linker 14.44.35207:

| Artifact | Built | Result |
| --- | --- | --- |
| `litmus.exe`, the command line | from two copies at different paths, one with a space in it | byte-identical |
| `litmus_app.wasm`, the browser page's module before wasm-bindgen | twice, from fresh copies at one path | byte-identical |
| `litmus-app.exe`, the desktop app | twice, from fresh copies at one path | byte-identical |
| `litmus.exe` and `litmus-app.exe` with a release's model built in, as shipped | twice from a clean state at the fixed path, compared with `release/build_release.py`'s output | byte-identical, MATCH |
| `litmus_app.wasm` and `litmus-app.exe` | from two different paths | different |

The app's builds depend on the path because the engine's crates are path dependencies outside the app's Cargo
workspace, and Cargo hashes their absolute path into every symbol name, which changes the code's layout as well as
the names; stripping symbols does not remove the difference. So `verify_build.py` builds the app and its module at
one fixed path, `C:\litmus-build` on Windows or `/tmp/litmus-build` elsewhere (`--fixed-root` changes it), and a
release built the same way can be reproduced only at that path. The command line has no such dependency.

What this does not show: builds on another Windows machine, whose linker or Windows SDK may differ; macOS and Linux,
which are untested; and the browser page's final module and JavaScript glue, which wasm-bindgen 0.2.126 writes after
the build and which this measurement does not cover.

## How it works, in full

The model is a BitNet b1.58 encoder whose weights are all -1, 0, or +1. It turns a text into the mean and variance
of its token vectors, a pair head on top gives the probability that two texts share an origin, and a reading
compares a text with every vector in a bank of known human, AI, and mixed reference texts. It is trained from
scratch on the true labels of pairs and runs with integer additions on a CPU, reading a long document in chunks
of at most 8,192 tokens and the heat map from windows of 512 tokens every 128. A frozen encoder, ModernBERT-base,
embeds the corpus for the clustered split and serves as a reference detector in training. Calibration
on texts held out for it turns the comparison into split conformal p-values for each origin, a Venn-Abers probability
with an interval, and a flag for text unlike the reference.

The corpus is public datasets only, with no text generated for it: human text dated before 2022, AI text that names
its generator, and mixed text that names how it was mixed where the source says. The texts are kept as written,
tagged by format, language, domain, and length, balanced inside those groups, and split into train, validation, test,
bank, and calibration sets of whole clusters found by e-means clustering, so near-copies never cross sets. The corpus
holds <!--litmus:record.corpus.rows.all as int-->14,828,377<!--/litmus--> rows (<!--litmus:record.corpus.rows.human as int-->7,138,302<!--/litmus--> human, <!--litmus:record.corpus.rows.ai as int-->7,094,051<!--/litmus--> AI, <!--litmus:record.corpus.rows.mixed as int-->596,024<!--/litmus--> mixed); the compiled corpus is private, and its
recipe, [`corpus/sources.csv`](../corpus/sources.csv) and the pipeline, is public.

[How Litmus is made](TRAINING.md) is the full training method, with each formula mapped to the function that
computes it, and the site's [How it works](../site/how.html) page walks through each step. The contracts are
[`student/SPEC.md`](../student/SPEC.md) with its NumPy reference [`student/reference.py`](../student/reference.py), which
the engine reproduces, and [`teacher/CALIBRATION.md`](../teacher/CALIBRATION.md).

## Results in full

### Validation and test

The checkpoint is chosen on validation, and test is read once. The metrics are for AI against human with mixed text
left out: PR AUC, ROC AUC, balanced MCC (the Matthews correlation coefficient on class-balanced samples at its best
threshold), and the true positive rate (TPR) at a 1% and a 5% false positive rate (FPR). The AI-or-mixed and mixed
tasks, 95% intervals, and every breakdown are in <!--litmus:t:data_files-->the release's [`results.json`](https://github.com/lhallee/litmus/releases/tag/v1.2.0)<!--/litmus-->.

<!--litmus:t:overview_table-->
| Split | Reader | Task | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | TPR at 5% FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test | Litmus, as released | AI against human | 0.937 | 0.934 | 0.697 | 0.530 | 0.671 |
| Test | Litmus, as released | AI or mixed against human | 0.945 | 0.900 | 0.621 | 0.385 | 0.557 |
| Test | Litmus, as released | Mixed against human | 0.861 | 0.866 | 0.559 | 0.240 | 0.442 |
<!--/litmus-->

Released size: <!--litmus:t:released_size-->`s512_l6` (width 512, 6 layers, 18.94M ternary weights)<!--/litmus-->.

<!--litmus:t:uncertainty_table-->
| Uncertainty on test | Value |
| --- | --- |
| Human texts called not human at the 5% level | 7.4% |
| Texts whose 10% set holds their true origin | 90.1% |
| Mean size of the 10% set | 1.87 |
| Heat-map boundaries placed within one cell of the truth, on spliced documents | 58.9% |
<!--/litmus-->

### Public benchmarks

Every detector reads the same seeded sample of each benchmark's labeled test split (<!--litmus:results.texts_per_benchmark as int-->3,000<!--/litmus-->), and the open detectors are run again on those texts rather than quoted. Each cell is ROC AUC for AI
against human, with mixed text left out; PR AUC, balanced MCC, TPR at 1% and 5% FPR, and 95% bootstrap intervals
drawn the same way for every detector are in <!--litmus:t:data_files-->the release's [`results.json`](https://github.com/lhallee/litmus/releases/tag/v1.2.0)<!--/litmus-->. The corpus builder holds each of
these splits out of every set Litmus trains, chooses, or calibrates on.

<!--litmus:table.benchmark_matrix as roc_auc-->
| Benchmark | Litmus | Binoculars | Desklib v1.01 | Fast-DetectGPT | ModernBERT RAID+MAGE (Drayson) | ai-vs-human-checker | e5-small-lora |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAID, extra | 0.886 | 0.761 | 0.883 † | 0.600 | 0.885 † | 0.700 † | 0.773 † |
| MAGE, test | 0.829 † | 0.624 | 0.929 | 0.569 | 0.986 † | 0.853 † | 0.761 |
| MAGE, unseen domain and model | 0.967 | 0.969 | 0.988 | 0.638 | 0.929 † | 0.973 † | 0.951 |
| MAGE, paraphrased | 0.898 | 0.662 | 0.873 | 0.420 | 0.719 † | 0.891 † | 0.833 |
| SemEval-2024 Task 8 A, monolingual | 0.994 † | 0.896 | 0.999 | 0.336 | 0.950 | 0.996 † | 0.918 |
| SemEval-2024 Task 8 A, multilingual | 0.981 † | 0.887 | 0.964 | 0.489 | 0.842 | 0.926 † | 0.758 |
| Beemo | 0.829 | 0.788 | 0.797 | 0.549 | 0.678 | 0.780 | 0.690 |
| MixSet | 0.285 | 0.499 | 0.660 | 0.420 | 0.636 | 0.463 | 0.607 |
| CoDET-M4 | 0.722 | 0.886 | 0.772 | 0.535 | 0.429 | 0.598 | 0.614 |
| DroidCollection | 0.818 † | 0.831 | 0.665 | 0.609 | 0.446 | 0.604 | 0.519 |
| autextification_en | 0.817 | 0.909 | 0.951 | 0.861 | 0.875 | 0.815 | 0.757 |
| autextification_es | 0.554 | 0.892 | 0.887 | 0.831 | 0.538 | 0.519 | 0.620 |
| editlens | 0.996 | 0.952 | 0.988 | 0.685 | 0.946 | 0.993 | 0.846 |
| hc3 | 0.993 | 0.993 | 0.999 | 0.767 | 0.988 | 0.999 | 0.934 |
<!--/litmus-->

`ai-vs-human-checker` is Litmus's predecessor ([lhallee/ai-vs-human-checker](https://huggingface.co/lhallee/ai-vs-human-checker)).
The ModernBERT RAID+MAGE classifier was trained on MAGE and RAID, and e5-small-lora on RAID's training split, so
their numbers on those benchmarks are in distribution. APT-Eval (human text polished by a model) and its human
originals hold one origin each, so neither has a ROC AUC on its own; the share of each that every detector flags is in the flag-rate table that follows. RAID's hidden test set is scored only by its public
leaderboard. Commercial detectors are not run here, and the site's [Results](../site/benchmarks.html) page says what a
comparison across protocols cannot claim.

<!--litmus:results_md.flag_rates-->
**Flag rates on APT-Eval** (share flagged; thresholds at 1% and 5% FPR are set on the human texts of the benchmarks every detector scored)

| Detector | Set | Texts | At 1% FPR | At 5% FPR | At its own threshold |
| --- | --- | --- | --- | --- | --- |
| Binoculars | APT-Eval polished | 3,000 | 0.018 (0.013 to 0.024) | 0.072 (0.062 to 0.083) | 0.066 (0.058 to 0.076) |
| Desklib v1.01 | APT-Eval polished | 3,000 | 0.120 (0.107 to 0.133) | 0.219 (0.205 to 0.235) | 0.503 (0.486 to 0.520) |
| e5-small-lora | APT-Eval polished | 3,000 | 0.000 (0.000 to 0.000) | 0.019 (0.013 to 0.024) | 0.823 (0.809 to 0.836) |
| Fast-DetectGPT | APT-Eval polished | 3,000 | 0.010 (0.006 to 0.014) | 0.062 (0.053 to 0.073) | none documented |
| ModernBERT RAID+MAGE (Drayson) | APT-Eval polished | 3,000 | 0.002 (0.000 to 0.004) | 0.006 (0.004 to 0.009) | 0.388 (0.371 to 0.406) |
| ai-vs-human-checker (predecessor) | APT-Eval polished | 3,000 | 0.089 (0.076 to 0.104) | 0.192 (0.177 to 0.208) | 0.298 (0.283 to 0.314) |
| **Litmus** | APT-Eval polished | 3,000 | 0.099 (0.079 to 0.111) | 0.160 (0.147 to 0.173) | 0.568 (0.550 to 0.585) |
| Binoculars | APT-Eval originals | 300 | 0.000 (0.000 to 0.000) | 0.007 (0.000 to 0.017) | 0.007 (0.000 to 0.017) |
| Desklib v1.01 | APT-Eval originals | 300 | 0.007 (0.000 to 0.017) | 0.020 (0.007 to 0.037) | 0.140 (0.097 to 0.183) |
| e5-small-lora | APT-Eval originals | 300 | 0.000 (0.000 to 0.000) | 0.003 (0.000 to 0.010) | 0.597 (0.540 to 0.650) |
| Fast-DetectGPT | APT-Eval originals | 300 | 0.013 (0.003 to 0.027) | 0.090 (0.053 to 0.123) | none documented |
| ModernBERT RAID+MAGE (Drayson) | APT-Eval originals | 300 | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.157 (0.117 to 0.200) |
| ai-vs-human-checker (predecessor) | APT-Eval originals | 300 | 0.000 (0.000 to 0.000) | 0.003 (0.000 to 0.010) | 0.040 (0.020 to 0.063) |
| **Litmus** | APT-Eval originals | 300 | 0.003 (0.000 to 0.010) | 0.003 (0.000 to 0.010) | 0.107 (0.073 to 0.140) |
<!--/litmus-->

### Ternary weights and integer attention

Two comparisons on validation, neither on the released model. In a one-hour proof-of-concept run
(`litmus_student.poc`) on the predecessor's two-origin corpus, the smallest ternary student (s192_l4) read PR AUC 0.9673 against 0.9676 for a
full-precision twin of the same size trained on the same batches, so ternary weights cost nothing measurable there.
The predecessor's teacher read 0.9901 on the same bank, and the students' curves were still rising when the hour
ended. Training the same student twice on the same batches, with and without integer attention (8-bit queries, keys,
and values, softmax in float), gave PR AUC 0.9386 and 0.9371 after 306 steps; the two stayed within 0.008 of each
other at every evaluation with no consistent sign, and training with integer attention was 15% slower.

### Speed and size

The engine's speed depends on the model's size. It was measured on releases of nine sizes filled with random weights, on one core of an AMD Ryzen 7 3800X under Windows 10 with the AVX2 kernels. A full scoring call ran at
21,926 tokens per second on an 8,192-token text for the smallest size (width 192, 4 layers) and 4,214 for the largest
(width 384, 8 layers), flat from 1,024 to 131,072 tokens. A 350-token passage took 17 to 83 ms, and the whole process
peaked at 56 to 70 MB. In WebAssembly, measured in Node 22 on one thread rather than in a browser, the smallest size
scored 3,400 to 4,000 tokens per second. Every size's numbers, and where the time goes, are in the
[engine's README](../engine/README.md).

| Built artifact | Size |
| --- | --- |
| `litmus-app.exe`, Windows x86-64, no model built in | 10.3 MB |
| Browser page, `litmus_app_bg.wasm` | 4.9 MB (1.7 MB gzipped) |
| Release files of the smallest size with random weights (`model.safetensors`, `config.json`, `tokenizer.json`) | 16.4 MB |

| Released model | Value |
| --- | --- |
| Tokens per second on one core, 8,192-token text | <!--litmus:bench.rows[tokens=8192].text_tokens_per_second as int-->4,920<!--/litmus--> |
| Peak memory of the process | <!--litmus:bench.rows[tokens=8192].process_peak_bytes as mb1-->63.1 MB<!--/litmus--> |
| Release files | <!--litmus:model.files_bytes as mb1-->18.5 MB<!--/litmus--> |
| Single-core cost, ms per 1,000 words (the other detectors' are in the benchmark tables) | <!--litmus:results.costs[detector=Litmus engine].seconds_per_1k_words as ms0-->349<!--/litmus--> |

## Repository map

| Folder | Holds | Language |
| --- | --- | --- |
| [`corpus/`](../corpus/README.md) | The source catalog and the pipeline that reads, standardizes, stratifies, checks, balances, and writes the corpus | Python |
| [`teacher/`](../teacher/README.md) | Embedding with the frozen encoder, the e-means split, the pair head's search, the bank, calibration, and the export | Python |
| [`student/`](../student/README.md) | The BitNet encoder, distillation, export, evaluation, the release format (`SPEC.md`), and its NumPy reference | Python |
| [`engine/`](../engine/README.md) | The Rust engine, the `litmus` command line, and the `litmus_engine` Python module | Rust |
| [`app/`](../app/README.md) | The desktop and browser app | Rust |
| [`site/`](../site/README.md) | The static site: what it is, how it works, reading a result, benchmarks, license, privacy | HTML, CSS |
| [`benchmarks/`](../benchmarks/README.md) | The benchmark catalog, published numbers of other detectors, scoring, metrics, and figures | Python |
| [`launch/`](../launch/README.md) | The Docker image and the scripts that run the training on one GPU | Shell, Python |
| [`release/`](../release/README.md) | The model card and the dataset card, the release build script and its steps, and the release checksums | Markdown, Python |
| [`contribute/`](../contribute/README.md) | The contribute site's server | Python |
| [`demo/`](../demo/README.md) | The launch film, as code | TypeScript |
| `tools/` | `run_tests.py` (every suite), `verify_build.py` (a release's programs rebuilt and compared, or two builds compared), `verify_model.py` (a release's hashes), and `security/`: the capability manifest and its checker, the dependency record and its checker, and their tests | Python, JSON |
| `training/foundry/` | Vendored library code that the training code imports | Python |
| `docs/` | This guide, the audit guide, the guide to using Litmus from an AI agent, the training method (`TRAINING.md` and `training/`), and the images of the README | Markdown, SVG, PNG |
| `.agents/skills/litmus/` | The agent skill (one copy for Claude Code, Codex, and Antigravity), its reference pages, and the install scripts for Linux, macOS, and Windows | Markdown, shell, PowerShell |
| `.claude-plugin/`, `.mcp.json` | The Claude Code plugin and marketplace manifests and the MCP registration that point at the skill | JSON |
