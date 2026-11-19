# Litmus engine

The inference engine of Litmus, a detector of model-written text: a ternary (BitNet b1.58) encoder that runs on one
CPU core by integer addition, scored against a reference bank with calibrated uncertainty, as a Rust library, a
command line (`litmus`), a WebAssembly build and a Python module (`litmus_engine`). The contract it implements is
[`../student/SPEC.md`](../student/SPEC.md) and [`../student/reference.py`](../student/reference.py) (the reference
wins where the two disagree).

It reads format 1 releases (chunks and sliding windows), format 2 releases with SPEC.md's optional integer attention
(`model.attention: "integer"`) and format 3 releases with int8 output (`model.output: "int8"`: int8 query and key codes
and an int8 bank, compared by integer dot products), in parity with `reference.py` on its fixtures. The ternary products,
attention, and the float64 products run on AVX2, SIMD128, and NEON, and the feed-forward's `down` product skips zero
inputs per token (the column path, same bits). The measurements below come from random-weight releases of the
production sizes, except the column path's, which come from a trained release.

| Crate | What |
| --- | --- |
| `litmus-engine` | The library: kernels, tokenizer, streaming encoder, head, bank, calibration, the document reading. No file system, threads or clock |
| `litmus-cli` | The `litmus` binary: score files, folders or standard input; benchmark; serve AI agents over the Model Context Protocol |
| `litmus-py` | The Python module `litmus_engine` (PyO3, built with maturin) |

Licensed under PolyForm Noncommercial 1.0.0 (every `Cargo.toml`; [`../LICENSING.md`](../LICENSING.md)); the crates
compiled in are listed with their licenses in [THIRD_PARTY.md](THIRD_PARTY.md), all permissive.

## What a reading is

`score(text)` returns a `DocumentReading` (SPEC.md, Fields):

- **`verdict`**: the document's call. Conformal p-values per origin (human, ai, mixed) and the sets at levels 0.05
  and 0.10 (the origins not ruled out; one origin is a call, several are uncertain, none is unlike the reference),
  the Venn-Abers probability of not human with its interval, three-way probabilities, the closeness p-value and the
  out-of-reference flag, the **litmus strip** (position 0 human, 0.5 mixed, 1 AI, and a band over what the 0.10 set
  cannot rule out), and the mixed-document signal. `document.label()` is the verdict in one word.
- **`chunks`**: a text of more than `chunk_tokens` (8,192) tokens is read as `ceil(n / 8192)` equal chunks, each a
  text of its own with a fresh encoder state; each chunk carries the full reading and its explanation (`explain`:
  the calibration counts behind each p-value and the five nearest bank texts with their metadata). The verdict merges
  the chunks: p-values as `min(1, 2 * mean)`, valid under any dependence; the probability as the mean, its interval
  widened by the chunks' disagreement; the mixed-document signal at 0.05 / C per chunk. One chunk is its own verdict.
- **`windows`**: inside each chunk, windows of 512 tokens every 128 (and one ending at the chunk's end), each read as
  a text of its own length from differences of the chunk's running float64 sums, without a second encoder pass.
- **`cells`**: the heat map, one cell per 128 tokens: the overlap-weighted mean strip position of the windows over it,
  their spread, and the widest band.

Every chunk, window and cell carries its place: token positions `start`, `end`, and `start_byte`, `end_byte`,
`start_char`, `end_char` into the original text (characters are Unicode scalar values; the cells and the chunks tile
the text).

## Library

```rust
use litmus_engine::{Engine, Flow};

let engine = Engine::load(&model_bytes, &config_bytes, &tokenizer_bytes)?; // model.safetensors, config.json, tokenizer.json
let document = engine.score_with(text, &mut |progress| {
    eprintln!("{} of {} chunks", progress.chunks_done, progress.chunks_total);
    if cancelled() { Flow::Cancel } else { Flow::Continue }
})?;
let verdict = &document.verdict;
println!("{} P(not human) {:.2} [{:.2}, {:.2}], strip at {:.2}", document.label().word(),
         verdict.probability, verdict.interval[0], verdict.interval[1], verdict.strip.position);
for cell in &document.cells {
    println!("characters {}..{}: {:.2}", cell.place.start_char, cell.place.end_char, cell.position);
}
```

| Item | Does |
| --- | --- |
| `Engine::load(model, config, tokenizer)` | Load a release from bytes, with the fastest backend this processor has |
| `score(text)`, `score_with(text, progress)` | The `DocumentReading`; `progress` is called every ~8 KB of text and may return `Flow::Cancel` (`Error::Cancelled`) |
| `count_tokens(text)`, `tokenize(text)` | Tokens of the standard reading; `Token { id, start, end }` with byte offsets into the original text |
| `score_ids(ids, piece)` | Score token ids directly, fed in pieces (the contract's level) |
| `info()`, `backend()`, `set_backend(b)` | Sizes, levels, license, provenance; the kernel backend (`Backend::Scalar` defines the integers, every backend gives the same) |
| `Error` | `Load(message)`, `Empty` (no tokens), `Cancelled`, `Backend(name)` |

The text is read twice: once to count its tokens (which fixes the chunks), once through the encoder, a chunk at a
time. `Progress` reports `tokens_total` and `chunks_total` as 0 during the counting pass. All readings serialize with
serde in the field names of `reference.py`. `Engine` is `Send` and `Sync`; calls on several threads share it.

## Python interface

The module `litmus_engine` (crate `litmus-py`) is a thin wrapper: every computation is in `litmus-engine`. The
contribute service ([`../contribute/`](../contribute/README.md)) codes against this surface.

```python
import litmus_engine

model = litmus_engine.Model.from_path("path/to/release")       # Model is Engine; tokenizer.json in the folder
model = litmus_engine.Engine(model_bytes, config_bytes, tokenizer_bytes, backend=None)
reading = model.score(text, timeout_seconds=None, max_tokens=None)  # dict, JSON-ready
reading["label"]                       # "human", "AI", "mixed", "uncertain" or "unlike reference"
reading["verdict"]["probability"], reading["verdict"]["interval"], reading["verdict"]["strip"]
reading["chunks"], reading["windows"], reading["cells"]        # as in "What a reading is"
model.count_tokens(text)               # int, the tokens score reads
model.version                          # str: provenance's "version", else "sha256:" + 16 hex of model then config
model.tokenize(text)                   # [(id, start_byte, end_byte), ...]
model.info()                           # dict: sizes, chunk and window sizes, bank, levels, license, backend, version
litmus_engine.Engine.backends()        # backends this processor supports
litmus_engine.Engine.standard_reading(text)
```

| Call | Detail |
| --- | --- |
| `Engine.from_path(path, tokenizer=None, backend=None)` | `path` is the release folder or its `model.safetensors`; reads `model.safetensors`, `config.json` and `tokenizer.json` (or the `tokenizer` path), and refuses a tokenizer whose SHA-256 differs from the one `config.json` pins, as the command line does. No `LITMUS_TOKENIZER` or Hugging Face cache fallback: the folder is the whole release |
| `score(text, *, timeout_seconds=None, max_tokens=None)` | The reading `litmus score --json` prints under `reading`, bit for bit, plus `label`. Releases the GIL while it scores and serializes; raises `ValueError` for a text with no tokens or more than `max_tokens` (stopped during the counting pass), `TimeoutError` past the timeout, `KeyboardInterrupt` on Ctrl-C (each checked every ~8 KB) |
| Errors | `FileNotFoundError` for a missing folder or file, naming it; `ValueError` for a release or tokenizer that does not load or an unknown backend. A Rust panic surfaces as `pyo3_runtime.PanicException`, never an abort |

**Threads.** `Engine` is immutable once loaded, so one instance is shared by every thread. Concurrent `score` calls
run in parallel, each with its own buffers, and give the same readings as serial calls; they contend only on the
tokenizer's word-cache lock. On one core they share the core, so each takes about as long as all of them in turn.

**Files.** The module reads the release's three files when it loads and writes nothing: it calls the engine
directly and does not use `litmus-files`, so neither the reading cache nor the PDF reader is in the wheel.

**Wheels.** abi3 for CPython 3.10 and later (`abi3-py310`): one wheel serves the contribute server's 3.12 and any
later interpreter without a rebuild, and the limited API costs nothing measurable since the work is in Rust with the
GIL released. The floor is 3.10 because `cargo build` of this Cargo workspace compiles `litmus-py` against the
`python` on PATH, and a higher floor fails that build where `python` is older. PyO3 is pinned at 0.25.1, the version
`Cargo.lock` holds, and maturin at 1.16.0. On Windows: `maturin build --release --locked -m
engine/litmus-py/Cargo.toml --out <dir>`. For Linux x86_64 (manylinux2014), in maturin's pinned Docker image with the
pinned Rust, on any machine with Docker:

```sh
bash engine/litmus-py/build_wheel.sh <project folder holding engine/> <output folder>
python -m venv <env> && <env>/bin/pip install <output folder>/litmus_engine-*.whl
cd /tmp && <env>/bin/python <project folder>/engine/litmus-py/smoke_test.py <release folder>
```

`smoke_test.py` loads the release, scores two texts, prints the readings as JSON, and exits non-zero on any failure.

Measured on one laptop core (AVX2, one thread) with an interim trained release, best of three: 8,192 tokens in
0.348 s through the module against 0.349 s for `litmus score` less its 0.100 s of start and load; 131,072 tokens in
5.80 s against 5.60 s (22,600 against 23,400 tokens a second; single runs varied by more than that).
The reading of a 131,072-token text is 1.15 MB of JSON.

## Command line

```text
litmus score [--json | --csv] [--map] [--explain] [--no-cache] [--no-color] <file | folder | ->...
litmus bench [--tokens 128,1024,8192,131072] [--repeat 3] [--json] [--text FILE] [--column-percent N] [--print-text]
litmus info
litmus mcp
  --model DIR (or $LITMUS_MODEL)   --tokenizer FILE (or DIR/tokenizer.json, $LITMUS_TOKENIZER, the Hugging Face cache)
  --backend scalar|avx2|avx2-sign|neon|neon-dotprod|simd128
```

`--backend` picks the instruction set of the ternary products and of attention (`scalar` is the portable code
throughout, as on a processor without SIMD); every backend gives the same bits. The head's and the bank's float64
products take AVX2, SIMD128 or NEON whenever the processor or the build has it, and an int8 bank's dot products take
AVX2 where the processor has it and the portable loops elsewhere.

The default table prints the verdict word, P(not human) with its interval, the set at 0.10, the strip and the token
count; the strip is drawn in color (blue `#2F6FDB` human, purple `#8E5BC9` mixed, red `#D9483B` AI, pale band `#C9B8E6`)
where the terminal supports it and always with characters too (`=` band, `|` position, `?` unlike the reference),
never by color alone. Under each row a line gives the text's counts, such as `1,234 words, 7,890 characters, 1,650
tokens`. `--map` adds each chunk of a long text and the heat map, one row per cell with its characters;
`--explain` says how the chunks combine and, per chunk, the bank evidence, the conformal counts and the nearest bank
texts. `--json` prints every field, and beside `reading` a `counts` object of the text read (`words`, `characters`,
`characters_no_spaces`, `tokens`): characters are Unicode scalar values, a word is a run of non-whitespace (a language
written without spaces counts as one word per run), and `tokens` is the reading's `n_tokens`. The counts are of the text
as extracted from the file, before the engine normalizes it, and come from `litmus-files` (`counts.rs`), not from the
engine, so `reading` and the Python module's readings are unchanged. Folders are read recursively (`.txt`, `.md`, `.html`, `.docx`, `.pdf`); readings
are cached by the SHA-256 of the file's bytes and of the release (`LITMUS_CACHE_DIR`, else the platform cache
folder).

**PDF quality.** PDFs go through `pdf-extract` (pure Rust) in a child process, because it prints diagnostics to
standard output and panics on some files. It reads the text layer only (no OCR), maps ligatures, and follows the
page's drawing order, so multi-column pages can interleave columns and headers and footers stay in. Of three report
PDFs tried, two read well and one failed on a font map (`adobe-cmap-parser` panic), which the table
reports as that file's error.

**`litmus mcp`** serves the Model Context Protocol on standard input and output for an AI agent
([`../docs/USE_WITH_AGENTS.md`](../docs/USE_WITH_AGENTS.md); `litmus-cli/src/mcp.rs`). It takes only `--model`,
`--tokenizer` and `--backend`, loads the release as `score` does, and offers two read-only tools, `litmus_check_text`
(`text`, `detail`) and `litmus_check_file` (`path`, `detail`). A result is one sentence and, in `structuredContent`, the
`label`, `n_tokens`, `words`, `characters` (as `counts` above), `caveats`, a standing `guidance` sentence and the `reading`: the `reading` of `litmus score --json`
whole at `detail: "full"`, and its `verdict` with the token counts at `"summary"`, the default. It follows the protocol
revision 2026-07-28 (no handshake: `server/discover`, and `_meta` on every request) and answers `initialize` for
2025-11-25 and 2025-06-18 clients, `ping` and `tools/list` in both, and ignores notifications. A bad argument, an empty
text, a text over 131,072 tokens (8 MiB of text, or a 32 MiB file) is a tool error the agent can read; an unknown tool or a
malformed message is a JSON-RPC error; a line that is not JSON gets a parse error and the server goes on. A release that
does not load does not stop the server; each check then says why. It uses no network, writes no file (no cache) and logs
to standard error only. `litmus score` has none of these limits.

## Kernel design

Weights are BitNet b1.58 ternary in the release's 2-bit layout (code `t + 1`, four per byte, low bits first, the
`I2_S` layout of bitnet.cpp), activations int8 with one absmax scale per token. The product is integer:
`sum_c code[r, c] * q[t, c] - sum_c q[t, c]`, then one float multiply per output.

- **Scalar** defines the answer. The SIMD backends give the same integers on every shape and remainder
  (`tests/kernels.rs`). It multiplies by the weight instead of branching on it, since a branch on ternary weights
  is close to a coin flip, and the branch-free sum is an integer reduction the compiler vectorizes for the target
  (SSE2 on any x86-64): 4.8 to 5.4 G additions a second on the machine below. It is the path of a processor
  without AVX2, NEON or SIMD128.
- **AVX2** (x86-64): a tile of 16 weight rows is unpacked once into L1 through a 256-entry table, then
  `vpmaddubsw` multiplies the unsigned codes (0 to 2) by signed int8 activations, 4 rows by 2 tokens per register
  block, with 16-bit sums flushed by `vpmaddwd` every 64 steps of 32 columns before they can overflow, and the
  activation sum subtracted once. A variant on `vpsignb` (`avx2-sign`) runs equal at 192 by 192 and 11 to 30% slower at larger shapes.
- **NEON** (aarch64): `sdot` where the processor has dot product (inline assembly; the intrinsic is unstable),
  `smlal` otherwise. **WASM SIMD128**: `i16x8.extmul` with a flush every 127 steps.
- **Why not bitnet.cpp's lookup tables (TL1, TL2).** They win for one token at a time (a language model decoding),
  where each weight is used once and memory dominates. An encoder reads blocks of up to 256 tokens through the same
  weights, so the product is compute-bound and the multiply-add form reuses every unpacked weight across tokens;
  bitnet.cpp itself ships its embedding models with `I2_S`.

Measured on an AMD Ryzen 7 3800X (one core, Criterion, 256 tokens): the AVX2 form runs 65 to 117 G ternary
additions per second (192 by 192: 65; 384 by 1,536: 116; 1,536 by 1,536: 115), the scalar form 4.8 to 5.4 G.

The float parts (norms, rotary, attention, SiLU, head, bank) use multiply then add, never a fused multiply-add, and
round-half-even by a magic number, so every path agrees bit for bit. **Attention follows the backend**: tiles of 4 rows by 16 columns in AVX2 (0.9 ms for a 256-token
block at width 192), SIMD128 (`dense_simd128.rs`) and NEON (`dense_neon.rs`) with the same per-lane order of
sums, each with its own vector `exp`, and the portable loops for `scalar`. In V8 the SIMD128 tiles read a block at
width 192 in 2.3 ms, against 4.1 ms for the portable loops. The head and the bank
run once per chunk for the chunk and all its windows together, as float64 products of two rows by four
(`dense::dots_f64`), so the bank is read once per chunk; the pair probability, the per-origin mean and spread and the
top cosines are gathered in that one pass. A chunk that is one window of its own length (a text of at most 512
tokens) has one pooled vector, so the bank is read once for it, and the window's reading is the chunk's without its
explanation.

**Int8 output** (`codes.rs`; SPEC.md, Int8 output) is the head and bank of a format 3 release. The head's query and key
are quantized to int8 codes, one scale per vector (`quantize_output`), and the bank is stored as int8 codes, each half
of a row padded with zeros to a multiple of 32 bytes, with the inverse norms of its halves computed at load. A bank
text costs four int8 dot products (the text's query and key against the text's query and key, two batched products of
a tile of 32 bank rows with every text vector of the chunk and its windows): `dots_i8` on AVX2 multiplies with
`vpmaddubsw` on the absolute value of one side against the other side with its sign (exact while no code is -128, which
neither the quantizer nor the load allows), adds pairs into 32 bits with `vpmaddwd` and reduces four bank rows at a
time with horizontal adds; every other processor takes the portable loops, which the compiler vectorizes, and
`tests/kernels.rs` checks both against the definition. The sums are exact int32, so every path returns the same
integers; the cosines, the pair probability and everything after are float64 from the sums and the inverse norms, in
the order SPEC.md gives, so the readings agree with `reference.py` to float64 rounding wherever the codes do.

**Integer attention** (`integer_attention.rs`; SPEC.md, Integer attention) is the attention of a format 2 release
that names `model.attention: "integer"`; a format 1 release reads with float attention. Queries and keys become int8 per
token and head (kept as 16-bit pairs), values int8 per block and head, and both products are integer: AVX2 forms the
scores with `vpmaddwd` (two columns of eight keys a register) and the weighted values with `vpmaddubsw` (unsigned
weights in 127ths times signed values, two tokens a lane), the latter summed in 16-bit lanes over spans of 128
tokens, which SPEC.md's bound keeps exact; SIMD128 uses `i32x4.dot_i16x8_s` for both, NEON `smull` and `addp`. The
float work left is each row's softmax: the vector `exp`, a float64 sum in a fixed lane order, and one rounding to
127ths. Every backend gives the portable path's bits, and the portable path follows SPEC.md's steps exactly (a test
writes them out loop by loop). A 256-token block at width 192 takes 0.62 ms in AVX2 against 0.88 ms for float
attention (1.4 times as fast; width 384: 1.24 against 1.85 ms), and 2.0 against 2.3 ms in V8's SIMD128. The gain is
smaller than the product count suggests (a `vpmaddwd` does 16 multiply-adds, a float multiply and add 8): the
softmax's exponentials are the same work in both modes.

## Streaming and memory

Each layer holds only its open block (256 rows, offset by half a block on odd layers), so a chunk of any length
streams through the encoder; the chunk's running float64 sums are recorded at every window boundary (at most 66
marks of 2d values per chunk) and dropped when the chunk is read. Beyond the input text, scoring holds the weights,
the block buffers, one chunk's marks and the readings it returns (one per chunk, window and cell).

## Measurements

`litmus bench` on random releases of the nine production sizes (`reference.py init`, 4,096 bank texts per origin,
so a bank of 12,288), AMD Ryzen 7 3800X, Windows 10, one core: the process pinned with
`start /wait /b /affinity 4 litmus.exe bench ...` (logical processor 2), AVX2 backend. Tokens per second from the
text (standard reading, tokenizing twice, encoder, head, bank, calibration, windows and cells), the fastest of 5 runs
(2 at 131,072) and the better of two rounds, with other work keeping the machine 40 to 45% busy; single runs varied
by up to a factor of two. The releases use float attention. The read pass is the head and the bank for one chunk and its 61 windows, timed alone; the
scoring heap is the allocation peak while scoring; the process peak is the resident peak of the whole `litmus`
process, engine (25 to 29 MB on the heap, the bank 12.6 MB of it) included.

| Size | 128 tokens | 1,024 | 8,192 | 131,072 (16 chunks) | Read pass at 8,192 | Scoring heap at 131,072 | Process peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d192 L4 | 16,133/s (8 ms) | 21,731/s (47 ms) | 21,926/s (374 ms) | 21,995/s (6.0 s) | 34 ms (9%) | 5.6 MB | 56 MB |
| d192 L6 | 12,571/s (10 ms) | 14,779/s (69 ms) | 15,231/s (538 ms) | 15,225/s (8.6 s) | 33 ms (6%) | 6.0 MB | 57 MB |
| d192 L8 | 10,019/s (13 ms) | 11,609/s (88 ms) | 11,624/s (705 ms) | 11,377/s (11.5 s) | 33 ms (5%) | 6.4 MB | 58 MB |
| d256 L4 | 11,913/s (11 ms) | 14,963/s (68 ms) | 14,735/s (556 ms) | 14,822/s (8.8 s) | 35 ms (6%) | 6.6 MB | 59 MB |
| d256 L6 | 8,938/s (14 ms) | 9,925/s (103 ms) | 10,034/s (816 ms) | 10,258/s (12.8 s) | 34 ms (4%) | 7.2 MB | 60 MB |
| d256 L8 | 7,199/s (18 ms) | 7,731/s (132 ms) | 7,913/s (1.0 s) | 7,743/s (16.9 s) | 35 ms (3%) | 7.7 MB | 62 MB |
| d384 L4 | 7,426/s (17 ms) | 8,488/s (121 ms) | 8,335/s (983 ms) | 8,422/s (15.6 s) | 38 ms (4%) | 8.7 MB | 65 MB |
| d384 L6 | 5,287/s (24 ms) | 5,734/s (179 ms) | 5,607/s (1.5 s) | 5,592/s (23.4 s) | 38 ms (3%) | 9.5 MB | 67 MB |
| d384 L8 | 4,148/s (31 ms) | 4,278/s (239 ms) | 4,214/s (1.9 s) | 4,226/s (31.0 s) | 38 ms (2%) | 10.3 MB | 70 MB |

At 8,192 tokens the whole pipeline sustains 39 to 62 G ternary additions per second (from `d192 L4` to `d384 L8`). Throughput is
flat from 1,024 to 131,072 tokens: a long text is 16 chunks of the same work, in the same memory. For
comparison, the predecessor checker took 1.11 s and peaked at 1.56 GB for a 266-word passage; a 350-token passage
takes 17 ms on `d192 L4` and 83 ms on `d384 L8`, in a process of 56 to 70 MB. In WebAssembly (Node 22, V8, one
thread) the SIMD128 kernel runs 4 to 14 G additions per second, and `d192 L4` scores 3,400 to 4,000 tokens per second
with its float parts (attention, head, bank) on the portable code and 5,400 to 5,650 with them on SIMD128 (below).

### Where the time goes

`litmus bench` built with `--features profile` prints each encoder stage's share of a scoring call (two clock reads
around a stage, `src/profile.rs`; no release build sets the feature). Same machine and pinning, 8,192 tokens,
float attention:

| Stage | d192 L4 | d256 L6 | d384 L8 |
| --- | --- | --- | --- |
| Ternary products | 24.0 us, 50% | 56.1 us, 59% | 152.2 us, 66% |
| Attention (scores, softmax, values; float) | 15.0 us, 31% | 27.0 us, 28% | 57.9 us, 25% |
| RMS norms and quantizing | 3.8 us, 8% | 7.0 us, 7% | 13.7 us, 6% |
| Head and bank | 4.1 us, 9% | 4.2 us, 4% | 4.5 us, 2% |
| Rotary, activation, embedding, pooling, the rest | 0.9 us, 2% | 1.3 us, 2% | 2.9 us, 1% |
| A token | 47.8 us | 95.6 us | 231.2 us |

The ternary products run at 62 to 75 G additions a second inside the pipeline. Attention is the largest part that
is not additions; integer attention (below) reduces it. There is no VNNI kernel (AVX-512 VNNI or AVX-VNNI).

### Integer attention

The same machine, release sizes and pinning, under heavier load from other work (one core read `d192 L4` at 47.5 us
a token and 93.3 us an hour later with the same binary), so the two kinds of attention were measured interleaved:
each size and pass ran the float release and the same tensors released with integer attention (format 2, `config.json`
changed, the tensors identical) one after the other. Microseconds a token from the text: the best of two rounds of 5
runs at 8,192 tokens (the slower round's best in brackets), the best of 2 at 131,072; stage times from the faster of
two profiled rounds (one at the `scalar` rows, at 1,024 tokens); the scoring heap is the allocation peak while scoring
131,072 tokens and the process peak the resident peak of the whole `litmus` process. Absolute numbers carry the load;
compare within a size.

AVX2:

| Size | Attention | 8,192 tokens, us a token | 131,072 | Ternary | Attention | Norms, quantizing | Head, bank | Scoring heap | Process peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d192 L4 | float | 88.4 (93.4) | 92.9 | 47.6 us, 51% | 32.4 us, 35% | 3.8 us, 4% | 8.0 us, 9% | 5.61 MB | 56.2 MB |
| d192 L4 | integer | 80.4 (84.5) | 81.5 | 54.8 us, 68% | 9.6 us, 12% | 7.7 us, 10% | 8.0 us, 10% | 5.56 MB | 56.2 MB |
| d256 L6 | float | 195.5 (197.8) | 98.3, load lifted | 127.8 us, 57% | 64.2 us, 29% | 15.7 us, 7% | 8.2 us, 4% | 7.15 MB | 60.0 MB |
| d256 L6 | integer | 173.9 (183.4) | 89.2, load lifted | 104.3 us, 58% | 38.4 us, 21% | 18.9 us, 11% | 8.2 us, 5% | 7.11 MB | 59.9 MB |
| d384 L8 | float | 227.0 (238.8) | 232.6 | 171.4 us, 67% | 61.6 us, 24% | 15.2 us, 6% | 4.6 us, 2% | 10.28 MB | 69.7 MB |
| d384 L8 | integer | 214.6 (225.7) | 211.3 | 163.0 us, 72% | 39.4 us, 17% | 14.5 us, 6% | 4.6 us, 2% | 10.24 MB | 69.6 MB |

`scalar` (1,024 tokens; attention runs the portable code too):

| Size | Attention | us a token | Ternary | Attention | Process peak |
| --- | --- | --- | --- | --- | --- |
| d192 L4 | float | 818 | 707 us, 84% | 120 us, 14% | 55.1 MB |
| d192 L4 | integer | 954 | 643 us, 74% | 219 us, 25% | 55.1 MB |
| d256 L6 | float | 976 | 860 us, 88% | 97 us, 10% | 57.1 MB |
| d256 L6 | integer | 1,065 | 865 us, 81% | 187 us, 18% | 57.1 MB |
| d384 L8 | float | 2,809 | 2,554 us, 92% | 194 us, 7% | 62.6 MB |
| d384 L8 | integer | 2,950 | 2,555 us, 87% | 373 us, 13% | 62.7 MB |

WebAssembly (Node 22, V8, one thread, unpinned, `d192 L4`, 9,950 tokens, the best of 3; `scripts/wasm_probe.mjs`):

| Build | Float attention | Integer attention |
| --- | --- | --- |
| SIMD128 | 5,416 and 5,648 tokens/s (two runs) | 4,630 and 5,580 tokens/s (attention alone, d192: 1.8 to 2.0 ms a block against 1.9 to 2.3 float) |
| Without SIMD128 (older browsers) | 861 tokens/s | 837 tokens/s |

Read pass in V8 SIMD128 (head and bank, a chunk of 8,192 and its windows): 83 ms.

What it says:

- **Integer attention is faster with SIMD, modestly.** Read in the same minutes, a format 2 release takes 5 to 11%
  less time a token than the float release at 8,192 tokens (d192: 80.4 against 88.4 us; d256: 173.9 against 195.5;
  d384: 214.6 against 227.0), and attention falls from 24 to 35% of a token to 12 to 21%. Alone, a 256-token block
  takes 0.62 ms against 0.88 (width 192) and 1.24 against 1.85 ms (width 384), Criterion (`benches/blocks.rs`). The
  per-stage times of single profiled runs swing with the load (norms at d192 read 3.8 and 7.7 us for the same code).
  It is slower where there is no SIMD: the portable integer loops multiply in 32 bits, which SSE2 lacks, so on the
  `scalar` path and in WebAssembly without SIMD128 integer attention costs 4 to 17% more than float attention. A
  format 1 release reads with float attention on every path.
- **Memory:** integer attention's buffers are 16-bit and 8-bit, 45 KB less of scoring heap (5.56 against 5.61 MB at
  d192). The bank, 12.6 MB of float32 widened from the file's float16, is the largest piece of the engine's heap (a
  format 3 bank is held as int8 codes, [below](#int8-output)).

### Int8 output

`d192 L4` random releases (`reference.py init`) with 2,048 bank texts per origin, written as format 1 (float
attention, float16 bank), format 2 (integer attention) and format 3 (int8 output, with float or integer attention): the
same tensors, the bank the same random unit vectors as float16 or as int8 codes. The same machine and pinning as above,
interleaved: the best of three rounds of 9 runs at 128 and at 8,192 tokens, from the text, by `litmus bench --json`
(`seconds`, `read_pass_seconds`). Formats 1 and 2 were run by the build before int8 output and by the build after
it. The read pass is the head and the bank for one chunk and its windows (one window at 128 tokens, 61 at 8,192), timed
alone. The first table has the head of the trained interim release (query and key width 64, hidden 256), the second
`configs.json`'s (128, 512).

Width 64:

| Release | Model file | 128 tokens | read pass | 8,192 tokens | read pass | Engine heap |
| --- | --- | --- | --- | --- | --- | --- |
| Format 1, before | 9.07 MB | 5.56 ms | 0.50 ms | 339.3 ms | 13.44 ms | 15.5 MB |
| Format 1, after | 9.07 MB | 5.57 ms | 0.42 ms | 341.4 ms | 13.51 ms | 15.5 MB |
| Format 3, float attention | 8.28 MB | 5.35 ms | 0.26 ms | 338.1 ms | 11.46 ms | 13.2 MB |
| Format 2, before | 9.07 MB | 5.43 ms | 0.49 ms | 311.5 ms | 13.44 ms | 15.5 MB |
| Format 2, after | 9.07 MB | 5.30 ms | 0.43 ms | 311.7 ms | 13.51 ms | 15.5 MB |
| Format 3, integer attention | 8.28 MB | 5.12 ms | 0.26 ms | 310.3 ms | 11.46 ms | 13.2 MB |

Width 128:

| Release | Model file | 128 tokens | read pass | 8,192 tokens | read pass | Engine heap |
| --- | --- | --- | --- | --- | --- | --- |
| Format 1, before | 11.04 MB | 6.14 ms | 0.69 ms | 346.9 ms | 18.67 ms | 19.5 MB |
| Format 1, after | 11.04 MB | 5.84 ms | 0.64 ms | 346.5 ms | 18.76 ms | 19.5 MB |
| Format 3, float attention | 9.46 MB | 5.41 ms | 0.30 ms | 338.6 ms | 13.02 ms | 14.8 MB |
| Format 2, before | 11.04 MB | 5.90 ms | 0.69 ms | 320.9 ms | 18.64 ms | 19.5 MB |
| Format 2, after | 11.04 MB | 5.59 ms | 0.64 ms | 318.8 ms | 18.88 ms | 19.5 MB |
| Format 3, integer attention | 9.46 MB | 5.20 ms | 0.30 ms | 313.5 ms | 13.06 ms | 14.8 MB |

What it says:

- **The compare is faster, the text barely:** the read pass of a chunk and its 61 windows takes 15% less (width 64) and
  30% less (width 128) in format 3 than in format 1 or 2, and a short text's read pass takes 47% and 57% less. A whole
  text takes 4 to 12% less at 128 tokens and 0.3 to 2.4% less at 8,192: the encoder is the rest. The sigmoid, the
  running sums and the float64 cosines are the same work in every format, so the gain is the products', which grows
  with the width.
- **A short text's bank is read once:** formats 1 and 2 read a 128-token text's bank in 0.05 to 0.07 ms less in the
  build after, with identical readings (`litmus score --json` is byte for byte the same on the interim release and on
  the four fixture releases).
- **Size:** the bank is `2w + 17` bytes a text instead of `4w + 17`: the 6,144 texts shrink the file by 0.79 MB at
  width 64 and 1.57 MB at width 128. In memory the engine holds a format 3 bank as int8 codes (each half padded to 32
  bytes) with the inverse norms of its halves, 1.0 MB with the token counts and metadata at width 64 against 3.3 MB
  of float32 vectors and norms, so the engine's heap is 2.3 MB smaller at width 64 and 4.7 MB at width 128.

### The column path for `down`

**Why.** In a trained release (width 192, 4 layers, an interim checkpoint), reading real documents, the int8 inputs
of the feed-forward's `down` product (squared-ReLU outputs after the norm and the per-token quantization, SPEC.md
section 3 step 6) are zero 68.2%, 72.1%, 85.4%, and 91.9% of the time in layers 0 to 3 (the `profile` build counts
them). Every other product reads inputs that are 1% to 2% zero, and random-weight releases give 57% to 59% in every
layer, so only a trained release says what a release will see. Zero weights cannot help a dense SIMD kernel; zero
activations can, because the product can skip them per token.

**The kernel** (`kernels::matmul_f32_columns`, `ColumnPairs`). `down` is stored a second time by column pairs: for
pair `k`, each row's codes `t + 1` of columns `2k` and `2k + 1` as two adjacent bytes. For a token, a 32-pair mask
(one vector compare, then a walk over its set bits) finds the pairs with a nonzero input, and only those are added:
`vpmaddubsw` adds `c0 * q_2k + c1 * q_2k+1` into 16 rows a register, 12 registers of rows a pass, widened every 64
pairs (AVX2); `i32x4.dot_i16x8_s` on the codes widened to 16 bits (SIMD128); `smlal` into 16-bit lanes widened by
`sadalp` every 127 pairs (NEON, compiled, not run); 16-bit multiplies on each row's two codes read as one 16-bit lane
(scalar). The code form's correction `sum q` is the same as the dense path's, so the integers are exact and the output
is the dense path's bits: each token takes the column path when its nonzero pairs are at most
`kernels::column_limit`, else the dense path (the dense tokens are gathered and run through the tiles). Pairs, not
single columns: a pair costs one multiply-add for two inputs, and skipping only pairs whose two inputs are both zero
still does less work than single columns at every zero rate (`(1 + z) / 2` multiply-adds per nonzero input).

**The limit**, from the kernel alone on synthetic activations (independent zeros; `benches/kernels.rs`,
`-- down_columns`; Criterion, 64 tokens, pinned), microseconds a token for one `down` product:

| Shape (d by f) | Path | 0% zeros | 50% | 70% | 80% | 85% | 90% | 95% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 by 768 | AVX2 dense | 1.48 | 1.48 | 1.48 | 1.48 | 1.48 | 1.48 | 1.48 |
| 192 by 768 | AVX2 columns | 1.85 | 1.59 | 1.16 | 0.86 | 0.69 | 0.50 | 0.28 |
| 256 by 1,024 | AVX2 dense | 2.58 | 2.58 | 2.58 | 2.58 | 2.58 | 2.58 | 2.58 |
| 256 by 1,024 | AVX2 columns | 4.47 | 3.51 | 2.47 | 1.78 | 1.43 | 0.99 | 0.50 |
| 384 by 1,536 | AVX2 dense | 5.63 | 5.63 | 5.63 | 5.63 | 5.63 | 5.63 | 5.63 |
| 384 by 1,536 | AVX2 columns | 10.5 | 7.81 | 5.33 | 3.83 | 3.05 | 2.11 | 1.15 |
| 512 by 2,048 | AVX2 dense | 9.95 | 9.95 | 9.95 | 9.95 | 9.95 | 9.95 | 9.95 |
| 512 by 2,048 | AVX2 columns (the better of two runs) | 15.5 | 12.4 | 8.83 | 7.32 | 4.97 | 3.42 | 1.82 |
| 192 by 768 | scalar dense | 27.0 | 27.0 | 27.0 | 27.0 | 27.0 | 27.0 | 27.0 |
| 192 by 768 | scalar columns | 46.4 | 39.1 | 23.5 | 16.4 | 12.8 | 8.9 | 4.6 |
| 384 by 1,536 | scalar dense | 106 | 106 | 106 | 106 | 106 | 106 | 106 |
| 384 by 1,536 | scalar columns | 185 | 138 | 93.8 | 68.5 | 52.8 | 35.7 | 17.9 |
| 192 by 768 | SIMD128 dense (V8) | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 192 by 768 | SIMD128 columns (V8) | 7.30 | 6.00 | 4.24 | 3.26 | | 2.14 | 1.51 |
| 384 by 1,536 | SIMD128 dense (V8) | 38.3 | 38.3 | 38.3 | 38.3 | 38.3 | 38.3 | 38.3 |
| 384 by 1,536 | SIMD128 columns (V8) | 26.6 | 22.0 | 16.4 | 12.5 | | 7.64 | 5.12 |
| 192 by 768 | without SIMD128, dense (V8) | 71.9 | 71.9 | 71.9 | 71.9 | 71.9 | 71.9 | 71.9 |
| 192 by 768 | without SIMD128, columns (V8) | 61.4 | 43.8 | 30.6 | 22.2 | | 12.2 | 6.66 |

The dense path does not depend on the zeros (its row is the median of its runs). The V8 rows are the best of three
rounds of `scripts/wasm_probe.mjs` (two without SIMD128), 20 repeats each, unpinned; 85% was not run there. On AVX2 the
column path costs 1.25 times the dense path per multiply-add while its codes stay in L2 (192 by 768: 147 KB) and 1.7
to 1.9 times past it (memory-bound: a pair's column is a cache line or more, read once per token, where the dense path
reuses each unpacked weight across 64 tokens), so the limit is 75% of the pairs nonzero when the codes fit in 192 KB,
else 50%. The scalar path is compute-bound at every size (1.7 to 1.9 times): 55%. In V8 the column path beats the
dense kernel even with no zeros, with SIMD128 (0.7 times) and without it (0.85 times), so every token takes it in
WebAssembly (100%): its 32-bit `i32x4.dot_i16x8_s` sums are cheaper there than the dense SIMD128 kernel's `extmul`
and 16-bit flushes. NEON takes the native scalar limit, unmeasured. On the trained release, whole tokens agreed (one
profiled round each): on AVX2 a limit of 70%, 90%, or 100% read 21.0 us of ternary products a token against 22.1 at
50% and 23.0 with the dense path alone; on the scalar path 50% and 70% read 289 to 293 us against 290 to 308 at 100%
and 331 to 337 dense.

**On the trained release**, real text (this repository's README and `student/SPEC.md` with a design document,
concatenated: 26,323 tokens, read by `litmus bench --text`), AMD Ryzen 7 3800X, one core pinned, with other work
loading the machine. Five rounds, each running the build with the column path and with it off
(`--column-percent 0`) one after the other; microseconds a token from the text, the best round and the range:

| Backend | Tokens | Column path | Dense path alone | Tokens on the column path, layers 0 to 3 |
| --- | --- | --- | --- | --- |
| AVX2 | 8,192 | 41.7 (41.7 to 43.0) | 43.0 (43.0 to 44.4) | 100%, 100%, 100%, 100% |
| scalar | 2,048 | 344.4 (344.4 to 348.4) | 385.4 (385.4 to 396.6) | 79%, 95%, 100%, 100% |
| SIMD128 (V8) | 26,323 | 145.9 (145.9 to 150.3) | 172.5 (172.5 to 179.3) | 100% (three rounds, unpinned, the whole text) |
| Without SIMD128 (V8) | 26,323 | 901 (901 to 912) | 1,113 (1,113 to 1,117) | 100% (two rounds; at the native scalar limit of 55%, 925 and 938) |

The column path saves 3% of a token's time on AVX2 (inside the rounds' range), 11% on the scalar path, and 15% and
19% in V8 with and without SIMD128. The whole JSON reading of the trained release on this text is byte-identical with
and without the column path on `avx2`, `avx2-sign` and `scalar` (231,236 bytes each). Random-weight releases of other
sizes, the same text (three AVX2 rounds at 8,192 tokens, two scalar rounds at 2,048; best round, microseconds a
token):

| Release | Zeros in `down`'s inputs | AVX2 column path, dense alone | Scalar column path, dense alone | Engine heap without and with the column codes |
| --- | --- | --- | --- | --- |
| random d192 L4 | 57 to 58% | 45.9, 44.8 (every token on the column path) | 393.4, 386.8 (none) | 25.3, 25.9 MB |
| random d384 L8 | 57 to 59% | 231.7, 231.6 (none) | 2,809, 2,842 (none) | 29.3, 34.0 MB |
| random d512 L12 | 57 to 60% | 554.1, 534.9 (none) | 7,386, 7,244 (none) | 23.1, 35.7 MB |

At random weights' 58% the column path is break-even at d192 and off at d384 and d512, and the times move inside
the noise (2 to 4% either way between rounds). These rows say nothing about a trained release of those sizes; only
its own zero rates will.

**What a higher zero rate would buy.** At d192 L4 on AVX2, `down` with the dense path takes 4 x 1.48 = 5.9 us of a
41.7 us token; with the trained rates it takes about 3.4 us (the kernel table at 68%, 72%, 85%, and 92%). At a uniform
zero rate it would take 4.6 us at 70%, 3.4 at 80%, 2.0 at 90%, and 1.1 at 95%: from the measured 41.7 us, 90% zeros
in every layer would save about 1.4 us a token (3%), 95% about 2.3 us (5.5%), and a free `down` 3.4 us (8%). On the
scalar path (`down` 108 us of 385 dense, about 67 with the column path) 90% would save 31 us of 344 (9%) and 95% 49 us
(14%). In V8 with SIMD128 (`down` 40 us of 172 dense, about 13 with the column path, matching the measured 27 us
saved) 90% would save about 4.5 us of 146 (3%) and 95% about 7 us (5%). At d384, where ternary products are 67 to 74%
of a token, 90% zeros in every layer would save about 3.5 us a layer, 28 us of a 232 us token at L8 (12%), 95% about
36 us (15%). So sparser feed-forward activations pay most at the larger widths and on the scalar and WebAssembly
paths; at d192 on AVX2 the remaining gain is small.

**`up` cannot skip work** the same way: which of its outputs the squared ReLU will zero is known only after the
product, and its inputs (the normed residual stream) are 1% to 2% zero.

**Memory.** The column codes are `f * d` bytes a layer, four times the packed `down`: the engine's heap grows by
0.59 MB at d192 L4 (19.3 to 19.9 MB), 4.7 MB at d384 L8 and 12.6 MB at d512 L12 (23.1 to 35.7 MB, a 54% larger
engine heap). Scoring buffers grow by a few kilobytes (the pair lists and one row of sums). A release whose zero rates
stay near random weights' carries the codes for nothing on AVX2 at d256 and above.

## Tests

| Test | Checks |
| --- | --- |
| `cargo test -p litmus-engine` | Kernels: every SIMD backend equals scalar on all shapes and remainders, 16-bit lane bounds, packing; the column path's integers equal the dense path's and the naive sums, and the per-token choice gives the dense path's float bits at every limit, on random inputs at 0 to 99% zeros and on all zero, no zero, one nonzero, every input -128 or 127, and every weight +1 or -1 across the 16-bit widening bounds; float attention on every backend equals the portable path; integer attention on every backend equals the portable path, which equals SPEC.md's steps written out loop by loop, including rows whose weights sum past 255 and work buffers reused across blocks; float64 products the same on every path; the int8 output's dot products on every path equal the definition at every shape and at the extremes, and its quantizer follows SPEC.md; format 1, 2 and 3 configs and what each refuses; chunks, windows, cells; the strip; numpy's `interp` |
| `--features parity --test tokenizer_parity` | 429 texts tokenize exactly as Python `tokenizers` (`scripts/tokenizer_fixture.py` writes the fixture) |
| `--features parity --test fixture_parity` | `reference.py`'s fixtures (`tiny`, `s192_l4`, the integer-attention `tiny_integer`, `s192_l4_integer`, and the int8-output `tiny_int8`, `s192_l4_integer_int8`, whose codes may differ from the reference's by one and whose readings are then held to twice the table, and whose head and bank, fed the reference's own pooled vectors, return its codes and readings to float64 rounding; an int8 bank is checked at load): standard reading and ids exactly; each chunk's pooled vector, query, key and window vectors; every chunk, window and cell reading and the verdict within SPEC.md's tolerances, a short text that a flipped activation moved at the tolerances scaled by its length (SPEC.md, Parity, Short texts); the verdict on the `combinations` cases; `score(text)` equals `score_ids`; streaming in pieces of 1, 7, W - 1, W, W + 1 equals reading at once; places tile the text; the dense path alone, the column path alone and the measured choice give the same reading, pooled vectors and token states on every backend |
| `cargo test -p litmus-cli` | `litmus mcp`'s protocol with a stand-in scorer: `server/discover`, `tools/list` and `tools/call` with the per-request metadata, the `initialize` handshake of 2025-11-25 and 2025-06-18, `ping`, notifications, bad arguments, unknown tools and methods, an unsupported version, batches, parse errors, a line over the limit, the token and byte caps, and a release that does not load |
| `LITMUS_MODEL=<release folder> cargo test --release --features release-tests -p litmus-cli --test mcp` | The real `litmus mcp` on a release: both protocol revisions through a scripted client, five texts (short, a paragraph, code, German and Japanese, a three-chunk one) by string and by file, and the `reading` of each equal to `litmus score --json --no-cache`'s, the summary's verdict equal to it, bad calls, malformed input, a release that does not load, and a clean exit when input closes |
| `python -m pytest engine/litmus-py/tests` | The wheel: readings match the fixtures; on a trained release, `score` equals `litmus score --json` bit for bit on empty, whitespace, one-word, short, non-ASCII, one-chunk, exactly 8,192-token, 8,193-token and three-chunk texts, `count_tokens` equals the engine's count, the 131,072-token reading is finite JSON under the server's 2 MiB, the token cap, timeout and load errors (missing files, a refused release, a tokenizer off its pin), threads give the serial readings and run in parallel, another thread runs Python while one scores (the GIL is released), and the contribute server's `scoring.py` runs unchanged on the module |

Set `LITMUS_STUDENT_FIXTURES` to the folder `reference.py fixtures --out <dir>` wrote, `LITMUS_TOKENIZER_DIR` to a
folder holding ModernBERT's `tokenizer.json`, `LITMUS_RELEASE` to a trained release folder, and `LITMUS_CLI` to the
`litmus` executable built from the same checkout (or pass `--litmus-fixtures`, `--litmus-tokenizer`,
`--litmus-release`, `--litmus-cli` to pytest).

## Builds

| Target | Command | State |
| --- | --- | --- |
| Windows x86-64 | `cargo build --release` | Built, tested, measured |
| WebAssembly | `RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown -p litmus-engine --example wasm_probe`, then `node scripts/wasm_probe.mjs <wasm> <release> <tokenizer.json>` | Built with and without SIMD128; in Node 22 (V8) the SIMD128 ternary product equals scalar, and float attention, integer attention and the float64 products equal the portable path, on every shape |
| Windows aarch64 | `cargo check --target aarch64-pc-windows-msvc -p litmus-engine --tests --benches --examples` | Compiles, the NEON attention, integer attention, float64 products and column path included; NEON not run. The pinned toolchain lists no aarch64 target; `cargo +stable` (the same 1.100.0) has it |
| Windows x86-64, wheel | `maturin build --release --locked -m litmus-py/Cargo.toml` | Built and tested with CPython 3.14 (abi3, 3.10 and later) |
| Linux x86-64, wheel | `bash litmus-py/build_wheel.sh <project folder> <output folder>` (Docker) | Built in the pinned manylinux2014 image and smoke-tested with `litmus-py/smoke_test.py` |
| Linux x86-64 | `cargo build --release --locked`; the released programs with `release/build_linux.sh` (Docker, glibc 2.17) | Built and tested: every suite above; `litmus` reads the same words, sets, labels and counts as the Windows program, with floats up to 3.5e-8 apart (the platforms' math libraries differ in `exp`, `tanh` and `ln`) |

Not tested: macOS, a browser page (only Node's V8), NEON at run time (the ternary products, attention, the
float64 products, and the column path), parity against `reference.py` on a trained release (a trained release is
measured for speed, and its readings are byte-identical with and without the column path), integer attention and int8
output on a trained release, and the int8 dot products on WebAssembly and NEON, which take the portable loops.
