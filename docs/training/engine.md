# 8. The engine

[Front page](../TRAINING.md) · [1. Data](data.md) · [2. Splits](splits.md) · [3. Model](model.md) ·
[4. Training](training.md) · [5. Calibration](calibration.md) · [6. Experiments](experiments.md) ·
[7. Evaluation](evaluation.md) · **8. Engine** · [9. Reproduce](reproduce.md)

The trained model is read by one engine, written in Rust, in the Windows, macOS, and Linux programs, the `litmus`
command line, the Python module `litmus_engine`, and, compiled to WebAssembly, the browser page. It reads a release
(`model.safetensors` and `config.json`) on the processor alone, with no graphics card, and gives the same reading as
the NumPy reference within stated tolerances. The app and `litmus score` spread one text over every logical processor
and give the same reading to the last digit as one core does; the speeds below are measured on one core. This part is a summary; [`engine/README.md`](../../engine/README.md) has every measurement,
and [`docs/AUDIT.md`](../AUDIT.md) the checks anyone can run.

- [The release file](#the-release-file)
- [Packed ternary kernels](#packed-ternary-kernels)
- [Comparing against the bank in integers](#comparing-against-the-bank-in-integers)
- [Skipping zero inputs](#skipping-zero-inputs)
- [Streaming in fixed memory](#streaming-in-fixed-memory)
- [The browser build](#the-browser-build)
- [Parity with the reference](#parity-with-the-reference)
- [Reproducible builds](#reproducible-builds)

## The release file

A release is two files ([`student/SPEC.md`](../../student/SPEC.md), section 6). `model.safetensors` holds the int8
embedding rows and their scales; every ternary matrix as 2-bit codes $`t + 1`$ (0 for −1, 1 for 0, 2 for +1), four to a
byte, low bits first, with its one scale $`\gamma`$; the RMS norms' gains; the head's float16 weights and the pair
logit's scale and bias; the bank's queries and keys (float16, or int8 codes in a release with int8 output), origins,
token counts, and metadata codes; and the sorted calibration scores that let a reading explain its counts.
`config.json` holds the sizes, the tokenizer's pinned revision and SHA-256, the reading's chunk and window sizes, the
calibration table, the bank's metadata names, and the provenance. Code 3 is invalid and fails the load.

A release with int8 output (format 3, `model.output: "int8"`) stores the head's query and key as int8 codes with one
scale per vector, the same quantizer the encoder uses for its activations, and the bank the same way, so a bank text
takes $`2w + 17`$ bytes instead of $`4w + 17`$ (width $`w`$ of the query and key). The compare is then four exact
integer dot products and a few float64 operations, and the engine holds the bank in memory as int8 codes.

## Packed ternary kernels

The product is $`\sum_c \mathrm{code}_{rc}\, \hat{x}_c - \sum_c \hat{x}_c`$, an exact integer, then one float multiply
per output: subtracting $`\sum_c \hat{x}_c`$ turns codes 0, 1, 2 back into weights −1, 0, +1.

- **Portable (scalar)** code defines the answer. It multiplies by the code instead of branching on it (a branch on a
  ternary weight is close to a coin flip), and the compiler vectorizes the integer sum for any x86-64.
- **AVX2** unpacks a tile of 16 weight rows once into the L1 cache through a 256-entry table, then multiplies unsigned
  codes by signed int8 activations with `vpmaddubsw`, 4 rows by 2 tokens per register block, widening the 16-bit
  sums every 64 steps before they can overflow. 65 to 117 billion ternary additions a second on one core of an AMD
  Ryzen 7 3800X.
- **NEON** (Arm) uses `sdot` where the processor has it; **WebAssembly SIMD128** uses `i16x8.extmul`.
- **Why not lookup tables.** bitnet.cpp's lookup kernels ([Wang et al., 2024](https://arxiv.org/abs/2410.16144)) win
  when each weight is used once, as when a language model writes one token at a time. An encoder reads blocks of up to
  256 tokens through the same weights, so the multiply-add form, which reuses each unpacked weight across tokens, is
  faster here.

Every backend gives the scalar path's integers on every shape, and the float parts (norms, rotary, attention, head,
bank) use multiply then add in a fixed order, never a fused multiply-add, so every path agrees bit for bit.

## Comparing against the bank in integers

With int8 output, the text's query and key and every bank row are int8 codes, so the products behind a reading's
cosines and pair probabilities are integer dot products, exact in 32 bits. AVX2 forms them with `vpmaddubsw` on the
absolute value of one side against the other side carrying its sign (valid because no code is −128), then `vpmaddwd`,
four bank rows at a time; other processors take portable loops that the compiler vectorizes. Each cosine is the integer
dot product times the two vectors' inverse norms in float64, in a fixed order, so every backend returns the same
readings. On the same random-weight `s192_l4` release as float16, a chunk's compare (one chunk and its 61 windows) takes 15%
to 30% less time, a 128-token text's 47% to 57% less, and the bank is about half the size on disk and a third of the size in memory; a whole
text takes 0.3% to 12% less, because the encoder is most of the work. A text read through one window reads the bank once.

## Skipping zero inputs

The squared ReLU makes most inputs of the feed-forward's `down` product exactly zero in a trained model (68% to 92% by
layer in an interim `s192_l4` release; [E10](experiments.md#e10-skipping-zero-inputs-of-the-down-product)). A dense
kernel cannot use zero *weights* (they are scattered, about a third of each matrix), but it can skip zero *inputs*,
which differ per token. The engine stores `down` a second time by column pairs; for each token it finds the pairs with
a nonzero input and adds only those, taking this column path when a token's nonzero pairs are under a limit measured
per backend, and the dense path otherwise. The integers are the dense path's, so the output bits are identical. On the
interim release it saved 3% of a token's time with AVX2, 11% on the portable path, and 15% and 19% in WebAssembly with
and without SIMD128. The column codes cost $`f \cdot d`$ bytes a layer of memory.

## Streaming in fixed memory

Each layer holds only its open block of 256 rows (offset by half a block on odd layers), so a chunk of any length
streams through the encoder; the chunk's running float64 sums are recorded at every window boundary and dropped when
the chunk is read. Memory does not grow with the text: a 131,072-token text is 16 chunks of the same work, and
throughput is flat from 1,024 to 131,072 tokens. With random weights, `s192_l4` reads 21,926 tokens a second at 8,192
tokens and 21,995 at 131,072, in a process of 56 MB ([`engine/README.md`](../../engine/README.md), Measurements).
The release: <!--litmus:t:release_speed-->4,920 tokens a second on one core and a process peak of 63.1 MB (`litmus bench` on AMD Ryzen 7 3800X 8-Core Processor, Windows 10)<!--/litmus-->.

## The browser build

The browser page runs the same engine compiled to WebAssembly, with SIMD128 kernels for the ternary products,
attention, and the column path, and portable code where a browser lacks SIMD128. The page fetches its own files only
(the WebAssembly module and the release's three files) and sends no text anywhere. In Node 22's V8, with random
weights, `s192_l4` scores about 5,400 to 5,650 tokens a second with SIMD128 and about 860 without.

## Parity with the reference

[`student/reference.py`](../../student/reference.py) is the answer; the engine reproduces it within the tolerances of
[`student/SPEC.md`](../../student/SPEC.md), Parity. Token ids are exact. Float sums in another order can round one
quantized activation the other way, which moves a token state by up to about 0.05 but a pooled vector, averaged over
the text, by about 0.001, so values agree within tolerances, not bit for bit:

| Value | Tolerance |
| --- | --- |
| Token ids | exact |
| Pooled vector | $`\lvert \Delta \rvert \le 3 \times 10^{-3} (1 + \lvert v \rvert)`$ |
| Query, key; mean, spread, share, score, closeness | $`\lvert \Delta \rvert \le 10^{-3}`$ |
| Three-way probabilities, the place's position and span (`strip`) | $`\lvert \Delta \rvert \le 2 \times 10^{-3}`$ |
| Grid p-values, sets, calls | equal, except within $`10^{-3}`$ of a table threshold |

In a release with int8 output the head's query and key are integer codes, so the engine's codes are compared exactly
and the readings from the same codes agree to float64 rounding; the tolerances above apply to the rest. A pooled
difference of about $`10^{-3}`$ can still flip a code by one in many components of a long text, which moves a reading
by up to about $`2 \times 10^{-3}`$, so the engine's tests allow twice the table for a reading whose codes moved.

The engine's tests check this on fixed random releases (`reference.py fixtures`) of the smallest production size and a
tiny one, in both attention modes and with float16 and int8 output, every chunk, window, cell, and verdict, and streaming in pieces of any size against
reading at once. The export checks each trained release against the reference on fixture texts before it is written
(`litmus_student.export`). For the release: <!--litmus:record.export.parity-->on 64 validation texts the worst pooled difference from the model is 0.0026 (bound 0.01) and the median 2.6e-4, the worst int8 query and key code differs by one step (bound 2), a streamed text scores exactly as a whole one, and the surprise's mean log probability differs by 0.0094 at worst (bound 0.02) and 2.3e-4 at the median (bound 2e-3)<!--/litmus-->.

## Reproducible builds

Two builds of the same source with the pinned Rust toolchain give byte-identical binaries: `tools/verify_build.py`
builds the command line and the WebAssembly module twice and compares their SHA-256 hashes, and
`tools/verify_model.py` checks a model release against its stated hashes. Verified on Windows x86-64; other platforms
are untested. See [`docs/AUDIT.md`](../AUDIT.md) for every claim the project makes about what it does with a machine
and a text, where each is implemented, and the commands that check it, and [`docs/DEVELOPING.md`](../DEVELOPING.md)
for building from source.

---

Previous: [7. Evaluation](evaluation.md) · Next: [9. Reproduce it](reproduce.md)
