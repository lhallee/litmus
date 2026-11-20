# Litmus student: release file and inference arithmetic

This file and [reference.py](reference.py) are the contract between the student (PyTorch, `src/litmus_student/`) and
the Rust engine (`../engine/`). The engine reproduces `reference.py` within the tolerances of
[Parity](#parity); where this text and `reference.py` disagree, `reference.py` is the answer and this text is the bug.

The code and the model it describes (weights, bank, calibration tables) are licensed under PolyForm Noncommercial
1.0.0 ([LICENSE](../LICENSE), [LICENSING.md](../LICENSING.md)); this document is CC BY-NC 4.0. Every copy passes
along the license and its notice: `Required Notice: Copyright Logan Hallee (https://github.com/lhallee/litmus)`.
A release's `config.json` names its license in `license`.

## Formats

A release's `config.json` names its format in `format_version`.

- **Format 1** reads attention in float ([3. The encoder](#3-the-encoder)) and does not carry `model.attention`.
- **Format 2** adds one optional mode, integer attention. A release whose `config.json` names
  `model.attention: "integer"` is format 2 and reads attention as [Integer attention](#integer-attention): int8
  queries, keys and values, integer products for the scores and for the weighted values, and float work only in each
  row's softmax. Nothing else in the file differs from format 1, and the tensors are the same. An engine that reads
  format 2 reads `model.attention` (`"float"` or `"integer"`) and runs [Integer attention](#integer-attention) in step
  4 of section 3 when it is `"integer"`; an engine that does not know format 2 refuses such a release instead of
  reading it with float attention. `export.py --attention integer` writes one from a checkpoint trained with either
  attention.
- **Format 3** adds one more mode, int8 output. A release whose `config.json` names `model.output: "int8"` is format 3
  and names `model.attention` as format 2 does. Its head's query and key are int8 codes, its bank is stored as int8
  codes, and the pair logit is computed from integer dot products ([Int8 output](#int8-output)). The encoder and the
  head's two layers are as in the other formats. An engine that does not know format 3 refuses such a release. Formats
  1 and 2 are read and written exactly as before. `export.py` writes one from a checkpoint trained with
  `train.py --int8-output` (and, with an explicit option, from another checkpoint).

## Fixtures for the engine

`python reference.py fixtures --out DIR` writes, deterministically (the same bytes on every run):

| Folder | Holds |
| --- | --- |
| `tiny/` | A random release of a small config (width 64, depth 2, block 16, factorized embedding; chunks of 96 tokens, windows of 32 every 16) whose texts cross many blocks, windows and, for the long text, four chunks |
| `s192_l4/` | A random release of the smallest production config (chunks of 8,192, windows of 512 every 128) |
| `tiny_integer/`, `s192_l4_integer/` | The same weights released with [Integer attention](#integer-attention) (format 2), with their own calibration |
| `tiny_int8/`, `s192_l4_integer_int8/` | The same weights released with [Int8 output](#int8-output) (format 3), the first with float attention and the second with integer attention, each with its own bank (the same random unit vectors as int8 codes) and calibration |

Each folder holds `model.safetensors`, `config.json` (with a calibration table that the teacher's `fit_calibration`
fitted to synthetic readings) and `fixtures.json`: for each of eight texts (short, one token, prose, control
characters and blank lines, literal special tokens, non-English with emoji, code, and a long text of many blocks and
windows), the text, its standard reading, its token ids, the first chunk's residual stream (first four rows after
the embedding and after each layer) and first four token states, each chunk's bounds, pooled vector, query, key and
window vectors, and the whole reading of [The reading](#the-reading): verdict, chunks with their explanations,
windows and cells (the random bank carries synthetic metadata); and `combinations`, chunk readings with the verdict
each gives, one of them a mixed document. The fixtures hold calls of not human, sets of none to three origins, and
chunks and windows out of reference. In the format 3 sets a chunk's `query` and `key` are the int8 codes (integers),
and each chunk also holds `window_query` and `window_key`, the codes of its windows. A random model is meaningless as a
detector; it tests arithmetic only. `python reference.py init --config <json> --out <dir> [--attention integer]
[--output int8]` writes a random release for any config, and `python reference.py score --release <dir> --text <file>
[--feed N]` prints one reading.

## 1. Text to token ids

1. **Standard reading**, the corpus's (`../corpus/src/litmus_corpus/standardize.py`, `standard_reading`, which owns
   it; `tests/test_reference.py` checks the two agree). The text is UTF-8. In order, each a replace-all of a regular
   expression:
   1. Unicode NFC.
   2. `\r\n?` to `\n`.
   3. Delete the characters U+0000-U+0008, U+000B, U+000C, U+000E-U+001F, U+007F-U+009F, U+00AD, U+180E, U+200B,
      U+200E, U+200F, U+202A-U+202E, U+2060-U+2064, U+2066-U+206F, U+FEFF and U+FFF9-U+FFFB. Tab, newline, and the
      zero-width joiner and non-joiner (U+200C, U+200D) stay.
   4. `[ \t\p{Zs}]+(\n|$)` to the captured group: spaces, tabs and space separators (U+0020, U+00A0, U+1680,
      U+2000-U+200A, U+202F, U+205F, U+3000) before each newline and at the end are deleted.
   5. `[\t\n\f\r ]+$` to nothing: trailing whitespace and newlines at the end of the text.
   6. `^(?:[ \t\p{Zs}]*\n)+` to nothing: blank lines at the start.
   7. `\n{4,}` to `\n\n\n`: at most two blank lines in a row.

   `$` and `^` are the end and start of the whole text.
2. **Tokenize** with the `tokenizer.json` of `answerdotai/ModernBERT-large` at revision
   `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` (sha256 `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1`),
   as Hugging Face `tokenizers` does with `encode(text, add_special_tokens=False)`: NFC normalizer, byte-level
   pre-tokenizer with the GPT-2 regex and no prefix space, BPE, and the 116 added tokens matched in the raw text
   (a literal `[CLS]` in the text is id 50281). No `[CLS]` or `[SEP]` is added. Ids lie in `[0, 50368)`. The
   fixtures were written with `tokenizers` 0.22.2 (the Python package; the training image pins 0.23.2); the Rust
   crate `tokenizers` loads the same file, and the fixtures' ids check it.
3. A text of zero tokens has no reading; the engine reports it as empty.

Streaming (section 5) is over token ids. The ids of the whole text are the contract; an engine that tokenizes
incrementally must produce exactly those ids.

## 2. Quantization

Following BitNet b1.58 (Ma et al. 2024, arXiv 2402.17764, section 2, equations 1 to 3; the official code in
"The Era of 1-bit LLMs: Training Tips, Code and FAQ", section 3, figure 3; the subln norms and squared ReLU of
BitNet b1.58 2B4T, arXiv 2504.12285, section 2).

**Round** is round half to even (`numpy.rint`, `torch.round`, Rust `f32::round_ties_even`), everywhere in this file.

**Ternary weights** (absmean, per tensor). Training holds a float matrix `W` of shape `(d_out, d_in)`;
`gamma = max(mean(|W|), 1e-5)` and `t = clamp(round(W * (1 / gamma)), -1, 1)`. The file stores `t` and `gamma`; the
weight the arithmetic uses is `t * gamma`.

**8-bit activations** (absmax, per token, symmetric, no zero point). For one token's vector `x` of width `d_in`
(float32): `m = max(max_i |x_i|, 1e-5)`, `s = 127 / m`, `q_i = clamp(round(x_i * s), -128, 127)` (int8).

**Ternary product.** `acc_j = sum_i t_ji * q_i` is an exact integer (int32; `|acc| < 2^24` for every width here),
computed by additions and subtractions only. The output is `y_j = float32(acc_j) * (gamma / s)`, with `gamma / s`
one float32 division per token and tensor. This is the "scaling factors applied after the linear operation" of the
FAQ, section 3.

**BitLinear** is the sequence: RMS norm with a learned gain, activation quantization, ternary product. Several
ternary products may share one norm and one quantized input (Q, K and V do).

**RMS norm.** `r = 1 / sqrt(mean_i(x_i^2) + 1e-6)`; `y_i = (x_i * r) * g_i`, float32.

## 3. The encoder

Hyperparameters come from `config.json` (`model`): vocabulary `V = 50368`, embedding width `e`, width `d`, depth
`L`, heads `h` of width `d_h = d / h`, feed-forward width `f`, block size `W` (even), rotary base `rope_base`.
The encoder reads one **chunk** ([Chunks](#chunks)) at a time, as a text of its own: positions `p` count tokens
from 0 at the start of the chunk, and `n` below is the chunk's length.

**Embedding.** `x0_p = float32(embed.weight[id_p]) * embed.scale[id_p]` (width `e`; int8 rows in `[-127, 127]`,
`round(row / scale)` with `scale = max(max |row|, 1e-5) / 127`, one float32 scale per row). When `e != d`, `x0_p = BitLinear(embed.proj)(x0_p)`: RMS norm with `embed.proj_norm`, quantization,
ternary product to width `d`. The residual stream `x` is float32.

**Layer `l`** (`l = 0 .. L-1`), shift `sh_l = 0` for even `l` and `W/2` for odd `l`:

1. `a = RMSNorm(x; attn_norm)`, quantized once; `Q, K, V = ternary products with q, k, v` (each `(n, d)`), split into
   heads `(n, h, d_h)`.
2. **Blocks.** Token `p` belongs to block `floor((p + sh_l) / W)`. A block holds at most `W` tokens; on odd layers
   the first block holds tokens `0 .. W/2 - 1`; the last block of a chunk holds whatever tokens remain. Attention
   is bidirectional inside a block and nothing crosses a block. There is no padding and no global token.
3. **Rotary positions** inside the block: `r_p = (p + sh_l) mod W`. For `j = 0 .. d_h/2 - 1`,
   `theta_j = r_p * rope_base^(-2j / d_h)`, `c = float32(cos(theta_j))`, `sn = float32(sin(theta_j))`, with the angle
   and both functions in float64 (a table of `W * d_h / 2` pairs). The halves rotate (the rotate-half convention):
   `u'_j = u_j * c - u_{j + d_h/2} * sn` and `u'_{j + d_h/2} = u_{j + d_h/2} * c + u_j * sn`, applied to Q and K.
4. **Attention** per head and block, float32: `S_ij = (Q_i . K_j) * float32(1 / sqrt(d_h))` for `i, j` in the
   block; `P_ij = exp(S_ij - max_k S_ik) / sum_k exp(S_ik - max_k S_ik)`; `O_i = sum_j P_ij V_j`. These are the only
   float multiplications of activations by activations in the encoder. A format 2 release whose `model.attention` is
   `"integer"` replaces this step by [Integer attention](#integer-attention).
5. `x = x + BitLinear(o)(O)`: heads concatenated to `(n, d)`, RMS norm with `attn_out_norm`, quantization, ternary
   product.
6. `u = RMSNorm(x; ffn_norm)`, quantized, ternary product with `up` to width `f`; `g = max(u, 0)^2` (squared ReLU);
   `x = x + BitLinear(down)(g)` with `ffn_out_norm` (width `f`) as its norm, back to width `d`.

**Token states.** `h_p = RMSNorm(x_p; final_norm)` after the last layer.

No biases anywhere in the encoder. A token's state depends only on tokens at most `L * W` positions away, so the cost
of a document is linear in its length.

### Integer attention

An optional mode, off unless a format 2 release's `config.json` names `model.attention: "integer"`. It replaces step
4 above, per head and block of `n` tokens, with `Q`, `K` (after rotary positions) and `V` float32 `(n, d_h)` and
`eps = act_eps`; **round** is round half to even, as everywhere:

1. **Queries and keys**, one scale per token: `s_i = 127 / max(max_c |u_ic|, eps)` (float32), `u'_ic = clamp(round(u_ic
   * s_i), -127, 127)`. **Values**, one scale for the block and head: `s_v = 127 / max(max_{j,c} |V_jc|, eps)`,
   `V'_jc = clamp(round(V_jc * s_v), -127, 127)`. The block is the text's own block of step 2.
2. **Scores**: `A_ij = sum_c Q'_ic K'_jc`, an exact integer (`|A| <= d_h * 127^2 < 2^24`, so heads of at most 1,040
   columns); `S_ij = (float32(A_ij) * x_i) * y_j` with `x_i = float32(1 / sqrt(d_h)) / s_q_i` and `y_j = 1 / s_k_j`,
   each a float32 operation.
3. **Weights**, the only float work left: `E_ij = exp*(S_ij - max_k S_ik)`, `z_i = sum_j E_ij` summed in float64 (in any
   order: the float64 sum of at most a block of float32 values moves `r_i` by a float32 rounding about once in 10^7
   rows), `r_i = float32(127 / z_i)`, and `p_ij = round(E_ij * r_i)`, an integer in `[0, 127]`: the weight in 127ths.
4. **Outputs**: `B_ic = sum_j p_ij V'_jc`, an exact integer; `O_ic = float32(B_ic) * t` with `t = 1 / (127 * s_v)`.

`exp*` is `exp` for `x <= 0` as one sequence of float32 operations, each rounded, none fused (Cephes `expf`'s
reduction and polynomial; the engine's `dense::exp_nonpositive`, `reference.exp_nonpositive`), so every
implementation gets the same bits from the same scores: `x = max(x, -87)`; `a = x * L + M`, `n = a - M`, `e =
bits(a) - bits(M)` (integer); `r = (x - n * C1) - n * C2`; `p = P0`, then `p = p * r + P_k` for `k = 1 .. 5`; `p = p *
(r * r) + r + 1`; the result is `p * 2^e` (the float32 whose bits are `(e + 127) << 23`, times `p`). The constants as
float32 bit patterns: `M` = 0x4B400000 (`1.5 * 2^23`), `L` = 0x3FB8AA3B (`log2 e`), `C1` = 0x3F318000, `C2` = 0xB95E8083,
`P0 .. P5` = 0x39506967, 0x3AB743CE, 0x3C088908, 0x3D2AA9C1, 0x3E2AAAAA, 0x3F000000. Its error is at most 4e-7 of
`exp(x)` above -87.

**Why these definitions.** A weight is a rounding to 127ths, so the float steps that feed it are pinned down: the same
exponential, a sum whose order cannot matter, and one rounding of `127 / z`. With them the reference, the PyTorch
model in eval mode and the engine round the same weights from the same scores, and the integer products are exact
everywhere. What remains between implementations is the float rounding upstream of attention (the RMS norms' sums),
as in float attention ([Parity](#parity)).

**Bounds the engine relies on.** A row's weights sum to at most `127 + n / 2` (each rounding adds at most a half), so
any 128 consecutive weights of a row sum to at most 191 and their products with `|V'| <= 127` to at most 24,257: the
weighted values can be summed in 16-bit lanes over spans of 128 tokens. Two weights sum to at most 128, so a pair of
weight-by-value products fits 16 bits (`vpmaddubsw`, which multiplies unsigned bytes by signed bytes).

**Training and export.** `model.py`'s `integer_block_attention` is this mode: its training forward rounds with
straight-through gradients and the float32 `softmax`; in eval mode (`integer_block_attention_eval`) it computes the
steps above exactly, with the value scale over each text's own block of a packed stream (the rows a row attends to),
not the stream's block. `python -m litmus_student.export --attention integer` releases a checkpoint in this mode,
whichever attention it was trained with: the model reads with integer attention from the start of the export, so the
bank, the calibration and the export's own parity check are all computed in the mode the release names, and the
provenance records `attention: {release, trained}`. Without `--attention` a checkpoint is released with the attention
it was trained with; a float release is format 1.

## 4. Pooling and the head

### Chunks

`config.json`'s `reading` sets `chunk_tokens` (`C_max`; 8,192 in a release, the teacher's own context, so every
chunk is a text the student was distilled on with a direct teacher target), `window_width` (512) and
`window_stride` (128, dividing `window_width`). A text of `n` tokens is cut into `C = ceil(n / C_max)` chunks; chunk
`c` (`c = 0 .. C-1`) holds the text's positions `[floor(c * n / C), floor((c + 1) * n / C))` in integer
arithmetic, so the chunks differ in length by at most one token and none exceeds `C_max`. A text of at most `C_max`
tokens is one chunk. Each chunk is read as a text of its own (section 3, positions from 0): the model state starts
fresh at every chunk boundary, so a chunk's vector is exactly what that token span gives as a standalone input.
The engine knows `n` before it reads, since the ids are the contract (section 1).

### Running totals

Inside a chunk, accumulate in float64 and in token order `T_p = sum_{i < p} h_i`, `Q_p = sum_{i < p} h_i^2`
(elementwise), recording `(T_p, Q_p)` at every position `p` where a window (below) starts or ends; `T_0 = Q_0 = 0`.
The pooled vector of positions `[a, b)` is, with `t = b - a`, `mean = (T_b - T_a) / t`,
`var = max((Q_b - Q_a) / t - mean^2, 0)`, both float64, then `float32(concat(mean, var))`, width `2d`. The chunk's
own vector is that of `[0, m)`, `m` the chunk's length.

**Precision.** A window's vector is a difference of two totals, not a sum over its own states, so it carries the
totals' float64 rounding. With unit final-norm gains each component of `h_i` is at most `sqrt(d)` in size, so over a
chunk of 8,192 tokens of width 384, each component of `T` stays under 1.6e5 and of `Q` under 3.2e6, where one
float64 rounding is at most 3e-11 and 5e-10. Even if all 8,192 additions rounded the same way, the error would be
2.4e-7 and 3.8e-6; divided by a window's 512 tokens, 5e-10 on the mean and 8e-9 on the mean square, below the
pooled vector's own float32 resolution (6e-8 relative) and far below the [Parity](#parity) tolerance of 3e-3.
`tests/test_reference.py` checks it on 8,192 random states of width 384 against pooling each window's states
directly (worst gap 1.2e-7 relative, one float32 step) and on the `tiny` config.

### Windows and the heat map

In a chunk of `m` tokens, with `w = window_width` and `s = window_stride`: when `m <= w` there is one window,
`[0, m)`; otherwise the windows are `[a, a + w)` for `a = 0, s, 2s, ...` while `a + w <= m`, and, when the last of
those does not end at `m`, also the chunk's final window `[m - w, m)`. A chunk of 8,192 tokens has 61 windows. Each
window's vector comes from the running totals, so windows cost no second pass through the encoder, only the head
and the bank (an engine may read a window as soon as its end total exists and drop totals it no longer needs).
Each window gets the [calibrated reading](#calibrated-reading) of a text of its own length (`t` tokens, so the
table's length group of `t`), without `explain`.

**The caveat.** A window's token states were computed with neighbors outside the window in view (attention reaches
up to `L * W` positions), unlike a standalone text of those tokens, which is what training and calibration read.
Its reading is therefore close to, not equal to, the reading of the same tokens alone. `reference.window_context_gap`
measures the gap. On the random `s192_l4` fixture release and the fixture's long text (2,691 tokens, 19 windows
compared with the same tokens alone), the pooled vectors differ by up to 0.31 (`abs / (1 + abs)`, mean 0.14), the
queries and keys by up to 0.033, the strip position by 0.011 on average and 0.031 at most; the 0.10 sets agree in
74% of windows and the side of 0.5 in 95%. Those are random weights reading random words: they show the gap is
real and moves sets near their thresholds, not how large it is for a trained student. `evaluate.py` measures that
on real text (`window_context`, interior windows apart from those touching a text's ends) and is the number to
quote.

**Cells.** The heat map of a chunk has one cell per stride: `[j s, min((j + 1) s, m))` for
`j = 0 .. ceil(m / s) - 1`, the last possibly shorter. With `o_k` the number of the cell's tokens that window `k`
holds (only windows with `o_k > 0` cover the cell; every cell has at least one) and `x_k` that window's strip
position, the cell's **position** is `sum o_k x_k / sum o_k`, its **spread** `sqrt(sum o_k (x_k - position)^2 /
sum o_k)`, its band `[min_k band_lo_k, max_k band_hi_k]` over the covering windows' [strips](#the-litmus-strip),
`windows` their count and `unlike_reference` true when any covering window is unlike the reference. Cells and
windows carry document positions (the chunk's start added). The app, the command line and the figures draw these
values and nothing else.

**The map describes; the verdict decides.** The document's verdict comes from its chunks, never from the most
extreme window or cell: with 61 overlapping windows per chunk, the most extreme of them is a maximum over many
looks at one text, and a window's set holds its level only for that window. A run of cells on the other side of the
strip is where to look, not a call.

**Head** (float32 arithmetic; weights stored float16 and widened exactly):
`z = clamp((pooled - head.input_mean) / head.input_scale, -c, c)` with `c` the float32 of `config.json`'s
`head.clamp` (10 by default, as the teacher's head has it; the distillation search may choose another);
`a = silu(head.fc1.weight @ z + head.fc1.bias)` with
`silu(v) = v / (1 + exp(-v))`; `o = head.fc2.weight @ a + head.fc2.bias`, width `2w`; the query is `o[:w]`, the key
`o[w:]`, each divided by `max(||.||_2, 1e-8)`. In a format 3 release the division is replaced by the codes of
[Int8 output](#int8-output).

### Int8 output

An optional mode, off unless a format 3 release's `config.json` names `model.output: "int8"`. It changes the head's last
step, the bank's storage and the compare; the encoder, the head's two layers and everything after the bank reading are
as in the other formats. **Round** is round half to even, as everywhere.

**Codes.** Let `o` be the head's output above, the `2w` float32 values before any division by a norm, and `x` the query's
`o[:w]` or the key's `o[w:]`. Each is quantized on its own, as [Integer attention](#integer-attention) quantizes a query
or a key: `s = 127 / max(max_c |x_c|, eps)` (float32, with `eps = act_eps` of `config.json` as a float32) and
`code_c = clamp(round(x_c * s), -127, 127)` (int8, `x_c * s` a float32 product). The text's query is its code vector `Q`
and its key `K`, with no division by a norm. The largest component of a vector with `max |x_c| >= eps` has code +127 or
-127, no code is -128, and a zero vector has zero codes.

**Norms.** For a vector of codes `c`, `N(c) = sum_i c_i^2` is an exact integer (at most `127^2 w`, under 2^31 for any `w`
up to 133,000, which an engine checks at load), and `u(c) = 1 / sqrt(N(c))` in float64, the square root and the division
each correctly rounded, or `u(c) = 0` when `N(c) = 0`. The unit vector of the codes is `c * u(c)`. The norms of a bank
text's two halves are computed when the release loads and never stored.

**Bank.** `bank.vectors` is I8 `(N, 2w)`: for each bank text the codes of its query then those of its key, from the same
head and the same quantizer. A code of -128 fails the load, as does any other dtype in a format 3 release (and I8 in
formats 1 and 2).

**Compare.** For the text's codes `(Q, K)` and bank text `j`'s `(Q_j, K_j)`, four integer dot products, each an exact
sum of `w` products of codes, so the same int32 in any order: `qq_j = Q . Q_j`, `qk_j = Q . K_j`, `kq_j = K . Q_j`,
`kk_j = K . K_j`. The cosine of two code vectors `a` and `b` whose dot product is `D` is
`cos(a, b) = (float64(D) * u(a)) * u(b)`: two float64 multiplications in that order, the text's vector first, none
fused. Then, in float64:

- the pair logit `s_j = scale * (cos(Q, K_j) + cos(K, Q_j)) / 2 + bias`, evaluated as `((scale * (a + b)) / 2) + bias` for
  the two cosines `a` and `b`, and `p_j = (1 + tanh(s_j / 2)) / 2`, as in the [bank reading](#bank-reading): the pair
  logit of the unit vectors of the codes;
- the cosine of the text and the bank text, which closeness and the nearest entries read, `(cos(Q, Q_j) + cos(K, K_j)) /
  2`: the cosine of the two concatenations `[Q, K]` and `[Q_j, K_j]` of unit vectors, as `v . b_j / (||v|| ||b_j||)` is
  for float vectors.

Means, spreads, closeness and everything after are as in the bank reading. Integers are exact; the float steps are
IEEE operations in a fixed order (a square root, a division, multiplications, additions), so every implementation gets
the same bits from the same codes, but for the last bit of `tanh`, which math libraries give differently, as in the
other formats.

**Training and export.** `model.py`'s `StudentHead(config, int8_output=True)` (`train.py --int8-output`) applies the
quantizer above to the query and to the key, straight-through in training (the gradient passes as if the rounding were
the identity), and normalizes the dequantized vectors, so the loss sees the unit vector of the codes and the model
learns the vectors a release stores. In eval mode its unit vectors are `c * u(c)` in float32, and quantizing a unit
vector of codes returns the codes. A checkpoint records the switch as `int8_output`. `python -m litmus_student.export`
releases a checkpoint trained with it as format 3 (`--output int8`, and the default for such a checkpoint); a
checkpoint trained without it is refused `--output int8` unless `--quantize-after-training` says to quantize its head's
output after the fact, and the provenance then records `output: {release, trained}`. The bank, the calibration and the
export's own check are computed with the codes the release stores. The export's check compares codes: the median text
within one code of the PyTorch model's and the worst within two.

## The reading

The default reading carries its own uncertainty. It is computed in float64 from the float32 query and key, for each chunk and each window,
in two steps: the **bank reading**, defined here, and the **calibrated reading**, which is the teacher's
`apply_calibration` (`../teacher/src/litmus_teacher/calibrate.py`) applied to that bank reading with the table in
`config.json`. The document's **verdict** combines its chunks' calibrated readings by that file's `combine_chunks`.
That file owns the calibration table and its methods; this section restates its arithmetic so the engine can be
written from here, and `tests/test_reference.py` checks that `reference.py` equals `apply_calibration` and
`chunk_uncertainty` on the fixtures. Origins are indexed human 0, ai 1, mixed 2.

### Bank reading

The bank holds unit query and key pairs of reference texts of each origin (`bank.vectors`, float16, widened
exactly to float64; in a format 3 release int8 codes, whose compare is that of [Int8 output](#int8-output), the same
reading of their unit vectors). For bank vector `j`, with `[scale, bias] = head.pair`:

- `s_j = scale * (q . k_j + q_j . k) / 2 + bias`, the same-origin logit of the text and bank text `j`;
- `p_j = (1 + tanh(s_j / 2)) / 2`, its sigmoid.

Per origin `o`, over the bank texts of that origin: **mean** `m_o = mean_j p_j` and **spread**
`sd_o = sqrt(mean_j (p_j - m_o)^2)` (population). **Closeness**: with `v = [q, k]` and each bank row `b_j`, the cosine
`v . b_j / (||v|| ||b_j||)`; closeness is the mean of the `top_k` (16) largest over the whole bank.

### Calibrated reading

The table is fitted per **length group**: `n` is the span's token count, its bin is
`#{e in length_edges[1:] : e <= n}` (bins of `[0, 16)`, `[16, 32)`, ..., `[8192, 32768)` and `[32768, inf)`), and
each kind of entry in the table is a list of groups, each with `bins` (the length bins it holds; every bin is in
exactly one group of each list). "The group" below is the entry whose `bins` holds the text's bin.

- **Share and score.** `r_o = m_o / (m_0 + m_1 + m_2)`, the normalized reading; the score is `1 - r_0`.
- **Conformal p-values** (Mondrian split conformal, by true origin and length group). For origin `o`, the group of
  `conformal[origin name]` holds `thresholds`, one per level `u` of `grid` (ascending, `null` allowed). The p-value
  is the smallest `u` with `thresholds[u]` not null and `r_o < thresholds[u]`, else 1. (This is the grid upper bound
  of `(1 + #{calibration texts of origin o in the group with r_o <= r}) / (n + 1)`, exact at 0.05 and 0.10.)
- **Sets and calls.** For each of `levels` (0.05, 0.10): the set is `{o : p_o > level}`; the text is **called not
  human** at a level when `p_human <= level`, which bounds the share of human text like the calibration set that is
  so called. A set of more than one origin reads as uncertain, an empty set as unlike the reference.
- **Probability of not human, with an interval** (inductive Venn-Abers). In the group of `probability`:
  `block = #{e in edges : e <= score}`, `p0 = p0[block]`, `p1 = p1[block]`; the probability is
  `p1 / (1 - p0 + p1)` and the interval `[p0, p1]`.
- **Three-way probabilities.** In the group of `three_way`: `z_o = slope * ln(max(r_o, 1e-12)) + bias[o]`, then a
  softmax over the three (subtract the largest `z` first).
- **Spread percentiles.** In the group of `spread`: for each origin, `numpy.interp(sd_o, group[origin name],
  spread_levels)` (the quantiles of the calibration texts' spreads, ascending; below the first, `spread_levels[0]`;
  above the last, `spread_levels[-1]`).
- **Out of reference.** In the group of `closeness`, the grid p-value of the closeness exactly as for conformal
  p-values; the text is out of reference when it is at or under `out_of_reference_level` (0.01).

### The litmus strip

One drawing for the engine, the app and the figures: a strip from 0 (human, blue) through 0.5 (mixed, purple) to 1
(AI, red). With the three-way probabilities `P_h, P_a, P_m` (they sum to one), the **position** is
`x = P_a + P_m / 2`. Each origin has an anchor: human 0, mixed 0.5, AI 1. The **uncertain band** is what the
conformal set at level 0.10 cannot rule out: with `G` that set, `[min(x, min_{o in G} anchor_o),
max(x, max_{o in G} anchor_o)]`, drawn as a soft band over the strip with a mark at `x`. A set of one origin gives
the band from `x` to that origin's anchor, a set of all three the whole strip. An empty set gives the band `[x, x]`,
and the strip is drawn as unlike the reference, as is any text out of reference. `strip` in the reading holds
`position`, `band_lo`, `band_hi`, `level` (0.10) and `unlike_reference` (the set is empty or the text is out of
reference).

### Chunks and the verdict

Every chunk carries the full reading above at its own length, with `explain`. The document's **verdict** is
`combine_chunks` of the teacher's `calibrate.py` over its `C` chunks' readings, at level `a = 0.05` for the
mixed-document signal (`../teacher/CALIBRATION.md`, Long documents). With `p_{c,o}` chunk `c`'s grid p-value of
origin `o`, `P_c` its probability of not human, `[p0_c, p1_c]` its interval, `T_c` its three-way probabilities and
`k_c` its closeness p-value:

- **p-values**: for `C = 1`, the chunk's own; for `C >= 2`, `p_o = min(1, 2 * mean_c p_{c,o})`. Twice the arithmetic
  mean of p-values is a p-value under any dependence between them, and no smaller factor suffices in general
  (Vovk and Wang, "Combining p-values via averaging", Biometrika 2020, arXiv 1212.4966; after Ruschendorf 1982 and
  Meng 1993). Chunks of one document resemble each other, so a rule that assumes independence would overstate the
  evidence. The merged p-values are not on the grid. **Sets and calls** follow from them exactly as for one text:
  the set at each level is `{o : p_o > level}`, called not human when `p_human <= level`.
- **Probability of not human**: `P = mean_c P_c`, a linear pool, which assumes nothing about dependence (a product
  would count the document's shared evidence `C` times). With `s` the population standard deviation of the `P_c`
  (`disagreement`), the **interval** is `[max(0, min(mean_c p0_c, P - s)), min(1, max(mean_c p1_c, P + s))]`: the
  mean Venn-Abers interval when the chunks agree, wider when they do not. It describes and carries no coverage
  guarantee. `lowest` and `highest` are the smallest and largest `P_c`.
- **Three-way probabilities**: `mean_c T_c`. The verdict's [strip](#the-litmus-strip) is drawn from them and the
  merged 0.10 set.
- **Out of reference**: the merged closeness p-value `min(1, 2 * mean_c k_c)` (`k_1` for one chunk) at or under
  `out_of_reference_level`.
- **Mixed document** (`C >= 2`): with `chunk_level = a / C`, chunk `c` is a **model chunk** when
  `p_{c,human} <= chunk_level` and a **human chunk** when `p_{c,ai} <= chunk_level` and `p_{c,mixed} <= chunk_level`.
  `mixed_document` is true when one chunk is a model chunk and a different chunk is a human chunk (one chunk that is
  both is not enough). By the union bound over the chunks, valid under any dependence, a wholly human document whose
  chunks are each like the human calibration texts fires it with probability at most `a`, and so does a wholly ai
  one. It is the located form of mixed: the chunks say where.

The guarantees hold for chunks exchangeable with the calibration texts of their length. Calibration texts are whole
texts and passages of at most a chunk, not chunks of longer documents; that difference is a stated limit, measured
on the corpus's whole long documents by `evaluate.py` (coverage of the merged sets, the share of human documents
called not human, and how often the mixed-document signal fires, by number of chunks).

### Explanation

Every reading explains itself from numbers a person can check, so that how each prediction and its uncertainty are
made can be read off the reading. `explain` holds:

- **`conformal`**, per origin `o`: from the release's sorted calibration scores of the text's group of
  `conformal[o]` (`explain.conformal.<o>.scores` between consecutive `offsets`, groups in the table's order), `n`
  (the calibration texts of origin `o` in the group), `count = #{scores <= r_o}` (binary search, the right side),
  `p = (1 + count) / (n + 1)` (the exact split conformal p-value, of which the grid p-value is the ceiling on
  `grid`), and `bins`. Said in words: "of `n` human calibration texts of this length, `count` read at most this
  human" (for human, at least as model-like). `tests/test_reference.py` checks that `p` rounds up on the grid to
  the reading's p-value.
- **`closeness`**: the same from `explain.closeness.scores` (every calibration text of the group of `closeness`):
  `n`, `count = #{scores <= closeness}`, `p`, `bins`. In words: "`count` of `n` calibration texts of this length sat
  as far from the reference as this one".
- **`nearest`**: the `NEAREST` (5) bank entries of highest cosine (ties to the lower index), each with `index`,
  `origin`, `cosine`, `probability` (`p_j`), `tokens`, and every metadata field of `bank.fields` (the corpus's
  `source`, `generator`, `domain`, `era`, `format`, `language`), decoded from `bank.meta.<field>` codes. The bank
  carries no text, since the sources' terms differ; a reader looks an entry up by its metadata.

### Fields

A document's reading in `reference.py` (JSON) has:

- `n_tokens` and `chunk_tokens`;
- `verdict`: `chunks`, `level` (0.05), `p_values`, `sets` and `not_human` (keyed `"0.05"`, `"0.1"`), `probability`,
  `interval`, `disagreement`, `lowest`, `highest`, `three_way`, `closeness_p`, `out_of_reference`, `strip`,
  `chunk_level`, `model_chunks`, `human_chunks` and `mixed_document`;
- `chunks`: each a reading (`n_tokens`, `length_bin`, `mean`, `spread`, `share`, `score`, `p_values`, `sets`,
  `not_human`, `probability`, `interval`, `three_way`, `spread_percentile`, `closeness`, `closeness_p`,
  `out_of_reference`, `strip`, `explain`) with `start` and `end`;
- `windows`: each the same reading without `explain`, with `start`, `end` and `chunk`;
- `cells`: `start`, `end`, `position`, `spread`, `band_lo`, `band_hi`, `windows`, `unlike_reference`.

Positions are the document's. For a text of one chunk, `verdict` repeats the chunk's values.

### Surprise

A release trained with the masked language objective (`train.py --mlm-share`) carries its masked language head, and
reads a text's **surprise** on request, in a pass of its own beside the reading: each token's log probability under
the head with nothing masked, every chunk read as a text of its own, averaged over the text. This mean is the
unmasked pseudo log likelihood (PPLL), in nats; higher is less surprising.

For a chunk's final token states `H` (`(c, d)`, after `final_norm`) and its ids:

1. `G = rms_norm(BitLinear(H; mlm.proj_norm, mlm.proj), mlm.norm)`, `(c, e)`: the BitLinear of section 3 from
   width `d` to the embedding width `e`.
2. The output rows are the embedding's own int8 rows, `R_v = embed.weight[v] * embed.scale[v]` in float32, and the
   logits `l_v = mlm.scale * (G R_v) + mlm.bias[v]` in float32, `(c, V)`.
3. Each position's log probability is `l_{id} - log sum_v exp(l_v)`, the log-sum-exp in float64.

The reading is `n_tokens`, `mean_log_probability` (the mean of step 3 over every token of every chunk), `perplexity`
(`exp(-mean_log_probability)`), and `percentile`: for each origin (human, AI, mixed), the share of that origin's
calibration texts whose mean is at or below the text's, interpolated linearly between the 101 quantiles of
`config.json`'s `surprise` table and 0 or 1 beyond its ends. A model's surprise at a text is evidence about the text
and the model together, not a verdict: it rises on any text unlike the training data, as well as on human writing.

## 5. Streaming

A document of any length is read in bounded memory, with the same answer as reading each chunk at once. The chunk
bounds follow from `n` (section 1 gives the ids first). Inside a chunk every operation is per token or per block,
so each layer is a block processor fed in position order:

- **Per layer `l`**, the float32 inputs `x` of its open block: tokens from the block's first position to the last
  received, at most `W` rows of width `d`. When the next position `p` received satisfies `(p + sh_l) mod W = 0`, or
  the chunk ends, the open block is complete: the layer runs on it and passes its outputs, in order, to layer
  `l + 1` (the last layer to the final norm and the running totals).
- **The position counter** `p` of the next token in the chunk (block membership and rotary positions depend on it).
- **Running totals:** the chunk's float64 `T` and `Q` and the totals recorded at window edges still needed.

At a chunk's end every open block is completed, the chunk and its windows are read, and all state is dropped:
the next chunk starts from nothing. Nothing crosses a block boundary but these, and nothing crosses a chunk
boundary but the chunks' and windows' readings (a few hundred bytes each). Memory is at most `L * W * d` floats of
open blocks, plus the head and the bank, plus `2d` float64 totals per recorded edge (at most `w / s + 2` of them
when windows are read as soon as they end), plus the readings, whatever the length. The whole-chunk path of
`reference.py` runs each layer on all its blocks at once; the streaming path feeds tokens in pieces of any size,
across chunk boundaries; the two agree within [Parity](#parity), and the tests check piece sizes 1, 7, `W - 1`,
`W`, `W + 1`, `C_max - 1` and the whole text, on texts of one to three chunks.

## 6. The release file

A release is a directory of two files.

**`model.safetensors`**, the [safetensors](https://github.com/huggingface/safetensors) format: an 8-byte
little-endian header length, a JSON header, then the tensors back to back in the header's order, little-endian.
Names, with `i` the layer:

| Name | Dtype | Shape | Holds |
| --- | --- | --- | --- |
| `embed.weight` | I8 | `(V, e)` | Token rows |
| `embed.scale` | F32 | `(V,)` | One scale per row |
| `embed.proj_norm` | F32 | `(e,)` | Only when `e != d` |
| `embed.proj.weight`, `embed.proj.scale` | U8, F32 | `(d, e/4)`, `(1,)` | Only when `e != d` |
| `layers.i.attn_norm` | F32 | `(d,)` | Shared by q, k, v |
| `layers.i.attn.{q,k,v,o}.weight` | U8 | `(d, d/4)` | Packed ternary |
| `layers.i.attn.{q,k,v,o}.scale` | F32 | `(1,)` | `gamma` |
| `layers.i.attn_out_norm` | F32 | `(d,)` | Before `o` |
| `layers.i.ffn_norm` | F32 | `(d,)` | Before `up` |
| `layers.i.ffn.up.weight`, `.scale` | U8, F32 | `(f, d/4)`, `(1,)` | |
| `layers.i.ffn_out_norm` | F32 | `(f,)` | Before `down` |
| `layers.i.ffn.down.weight`, `.scale` | U8, F32 | `(d, f/4)`, `(1,)` | |
| `final_norm` | F32 | `(d,)` | |
| `head.input_mean`, `head.input_scale` | F32 | `(2d,)` | Standardization |
| `head.fc1.weight`, `head.fc1.bias` | F16 | `(hh, 2d)`, `(hh,)` | |
| `head.fc2.weight`, `head.fc2.bias` | F16 | `(2w, hh)`, `(2w,)` | |
| `head.pair` | F32 | `(2,)` | `scale`, `bias` of the pair logit |
| `bank.vectors` | F16 (I8 in format 3) | `(N, 2w)` | Unit query then unit key (in format 3 the int8 codes of the query then of the key, in [-127, 127]), rows grouped by origin: human, ai, mixed |
| `bank.origin` | U8 | `(N,)` | 0, 1, 2, ascending |
| `bank.tokens` | I32 | `(N,)` | Each bank text's token count |
| `bank.meta.<field>` | U16 | `(N,)` | Each bank text's code into `config.json`'s `bank.fields.<field>` |
| `explain.conformal.<origin>.scores`, `.offsets` | F32, I64 | `(n_o,)`, `(g_o + 1,)` | Calibration shares of that origin, ascending within each group of `conformal.<origin>` |
| `explain.closeness.scores`, `.offsets` | F32, I64 | `(n,)`, `(g + 1,)` | Calibration closeness, ascending within each group of `closeness` |
| `mlm.proj_norm` | F32 | `(d,)` | Optional, with every `mlm.*` tensor: the [Surprise](#surprise) head |
| `mlm.proj.weight`, `mlm.proj.scale` | U8, F32 | `(e, d/4)`, `(1,)` | Packed ternary, `d` to `e` |
| `mlm.norm` | F32 | `(e,)` | |
| `mlm.scale` | F32 | `(1,)` | The logits' scale |
| `mlm.bias` | F32 | `(V,)` | One bias per token |

**Packed ternary.** Row-major `(d_out, d_in)`; the code of a weight is `t + 1` (0 for -1, 1 for 0, 2 for +1; the
I2_S code of bitnet.cpp, Wang et al. 2024, arXiv 2410.16144, section 2, table 1), and weight `(r, c)` sits in byte
`(r, c / 4)` at bits `2 * (c mod 4)` and `2 * (c mod 4) + 1`. Code 3 is invalid and fails the load. Every `d_in`
is a multiple of 4. An engine relays the weights into its own kernel layout (bitnet.cpp's TL1 and TL2 lookup tables,
or plain add and subtract) at load time; the file fixes only the values.

**`config.json`**: `format` (`"litmus-student"`), `format_version` (1; 2 for a release that names its attention; 3 for
one that names its output), `license` (`"PolyForm-Noncommercial-1.0.0"`), `tokenizer` (repository, revision, sha256,
`add_special_tokens: false`), `model` (`vocab_size`, `embed_width`, `width`, `depth`, `heads`, `ffn_width`,
`block_size`, `rope_base`, `norm_eps`, `act_eps`, `weight_eps`, in formats 2 and 3 `attention`, `"float"` or
`"integer"`, and in format 3 only `output`, `"int8"`; format 1 has no `attention` and is float attention, formats 1 and
2 have no `output` and are float output), `head` (`hidden`, `out_width`, `clamp`), `reading` (`chunk_tokens`, `window_width`,
`window_stride`; 8,192, 512 and 128 in a release), `text_reading` (only in a release trained on a reading of its
texts other than the standard one, which it names; no release has one yet), `bank` (`origins`, `counts`,
`split`, and `fields`: each metadata field's names, which `bank.meta.<field>` codes index), `calibration` (below),
`surprise` (with the `mlm.*` tensors only: `split`, `texts` per origin, and `quantiles`, each origin's 101 calibration
quantiles of the mean token log probability, 0% to 100%; [Surprise](#surprise)),
`provenance` (checkpoint, step, training run, validation numbers, data, seed) and `weights` (`file`,
`"model.safetensors"`, and the `sha256` of that file as written; an engine ignores it and
`tools/verify_model.py` checks it). JSON numbers are float64 written
in shortest round-trip form.

### Calibration table

`config.json`'s `calibration` is the teacher's table as `fit_calibration` writes it (`format`
`"litmus-calibration"`, `version` 1; `../teacher/CALIBRATION.md` documents it): `length_edges`, `levels`, `grid`,
`spread_levels`, `top_k`, `out_of_reference_level`, and the group lists `conformal` (keyed by origin name),
`probability`, `three_way`, `spread` and `closeness`, each entry with `bins` and `n`. Its floats are rounded to seven
significant digits by the fit, so the table the engine reads is the table Python applies. `export.py` fits it on
the student's own bank readings of the `calibration` split; each release carries its own, about 27 KB of JSON, and
beside it in the safetensors file the sorted calibration scores of [Explanation](#explanation) (4 bytes per
calibration text and table, about 0.24 MB for 30,000 texts).

## Sizes

For the size sweep (`configs.json`; `python -m litmus_student.model` prints this table), with `e = 128`, `W = 256`,
`f = 4d`, heads of 64, head hidden 512, query and key width 128 each, and a bank of 4,096 texts per origin. **Ternary
weights** include the embedding's projection. **Adds** are the ternary additions and subtractions per token (an
upper bound on the arithmetic: a zero of `t` adds nothing, though the engine's dense kernels walk the zeros at the
same cost as any other weight); **multiplies** are the float32 attention multiply-adds per token at full
blocks. **Model** is the safetensors file without the bank; the bank and its metadata add 6.5 MB (below). The random `s192_l4` fixture, with
a bank of 64 per origin, is a 7.9 MB file.

| Config | d | L | Ternary weights | Adds per token | Multiplies per token | Model MB | Tokens/s, one core | 120,000 tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `s192_l4` | 192 | 4 | 1.79M | 1.79M | 0.39M | 7.8 | 9,144 | 13 s |
| `s192_l6` | 192 | 6 | 2.68M | 2.68M | 0.59M | 8.0 | 6,119 | 20 s |
| `s192_l8` | 192 | 8 | 3.56M | 3.56M | 0.79M | 8.2 | 4,598 | 26 s |
| `s256_l4` | 256 | 4 | 3.18M | 3.18M | 0.52M | 8.3 | 5,401 | 22 s |
| `s256_l6` | 256 | 6 | 4.75M | 4.75M | 0.79M | 8.7 | 3,612 | 33 s |
| `s256_l8` | 256 | 8 | 6.32M | 6.32M | 1.05M | 9.1 | 2,713 | 44 s |
| `s384_l4` | 384 | 4 | 7.13M | 7.13M | 0.79M | 9.5 | 2,527 | 47 s |
| `s384_l6` | 384 | 6 | 10.67M | 10.67M | 1.18M | 10.4 | 1,688 | 71 s |
| `s384_l8` | 384 | 8 | 14.20M | 14.20M | 1.57M | 11.3 | 1,268 | 95 s |

Throughput assumes 2e10 ternary adds and 2e10 float32 multiply-adds per second on one core. The adds rate is from
bitnet.cpp (arXiv 2410.16144, table 5): a 700M BitNet b1.58 at 80 tokens per second on two threads of an
i7-13700H is about 2.8e10 weight operations per second per thread, in decoding, where the weights cannot be reused
across tokens; an encoder reuses each weight across a block, so the rate is a floor. The engine measures it. The
embedding (6.6 MB at `e = 128`) is most of every model file; `e = 64` halves it.

**The bank** is sized by the student head's query and key width `w`, not by the teacher's: each entry is
`4w + 17` bytes in formats 1 and 2 (the float16 query and key, the origin, the token count and six 2-byte metadata
codes) and `2w + 17` bytes in format 3 (the same with the query and key as int8 codes), so a bank of `N` texts per
origin is `3N(4w + 17)` or `3N(2w + 17)` bytes. The compare's norms are computed at load and take memory then, not
file. Against the browser budget of about 30 MB for model and bank together, float16 / int8:

| `w` (query and key each) | Bytes per entry | 2,048 per origin | 4,096 per origin | 8,192 per origin |
| --- | --- | --- | --- | --- |
| 64 | 273 / 145 | 1.7 / 0.9 MB | 3.4 / 1.8 MB | 6.7 / 3.6 MB |
| 128 (every config here) | 529 / 273 | 3.3 / 1.7 MB | 6.5 / 3.4 MB | 13.0 / 6.7 MB |
| 192 | 785 / 401 | 4.8 / 2.5 MB | 9.6 / 4.9 MB | 19.3 / 9.9 MB |
| 256 | 1,041 / 529 | 6.4 / 3.3 MB | 12.8 / 6.5 MB | 25.6 / 13.0 MB |
| 1,024 (the first checker's head) | 4,113 / 2,065 | 25.3 / 12.7 MB | 50.5 / 25.4 MB | 101.1 / 50.8 MB |

With `w = 128` and 4,096 per origin, the largest config (`s384_l8`) ships at about 11.3 + 6.5 + 0.3 = 18 MB (15 MB with
an int8 bank) and the smallest (`s192_l4`) at about 14.6 MB (11.5 MB). Validation chooses the bank size (the design's
open question of a held-out bank against a stratified one is chosen the same way); the export takes `--bank-per-origin`
and, for the draw of the bank's texts, `--bank-recipe`.

## Parity

Ids are exact. The engine's float32 sums run in another order than NumPy's, and a rounding that lands on the other
side of a half moves one quantized activation by one step; inside a block that reaches every token of the block, so
token states differ by up to about 0.05 while pooled vectors, averaged over the document, differ by about 1e-3.
Values therefore agree within a tolerance, not bit for bit. Measured between `reference.py` and the independent
float32 PyTorch implementation of `src/litmus_student/model.py` in eval mode, on random models of 1,500-token random
texts (`tests/test_parity.py`): token states up to 0.064 apart, pooled vectors `abs / (1 + abs(value))` up to
1.5e-3 (`s384_l8`) and 0.9e-3 (`s192_l4`), queries and keys up to 2e-4. The engine meets:

| Value | Tolerance |
| --- | --- |
| Token ids | exact |
| Layer and token states in the fixture | a debugging aid: about 1e-5 until a quantized activation flips, then up to 0.1 |
| Pooled vector | `abs <= 3e-3 * (1 + abs(value))` |
| Query, key | `abs <= 1e-3` (format 3: the codes, integers, each at most 1 from the reference's; see below) |
| Mean, spread, share, score, closeness | `abs <= 1e-3` |
| Three-way probabilities, spread percentiles, strip position and band | `abs <= 2e-3` |
| Probability of not human and its interval | equal while the Venn-Abers block is the same (they are table values) |
| Grid p-values, `closeness_p`, sets, calls, Venn-Abers block | equal, except where the engine's share, score or closeness lies within 1e-3 of a table threshold or block edge, where it may fall on the other side |
| Window vectors (`window_pooled`) and window readings | as the pooled vector and the chunk reading above |
| Cells | position and spread `abs <= 2e-3`; band and `unlike_reference` as the windows' strips |
| Verdict | merged p-values within 1e-12, and sets and calls equal, while every chunk's grid p-values are (they are twice means of grid values); probability, interval, three-way and disagreement `abs <= 2e-3`; `mixed_document`, `model_chunks` and `human_chunks` follow the chunks' grid p-values |
| Nearest entries (`explain.nearest`) | cosine `abs <= 1e-3`, probability `abs <= 2e-3` (one pair's probability, the steepest function of a cosine in the reading); entries whose cosines are within 1e-3 may swap places |
| Surprise: mean log probability | `abs <= 2e-3` (a text under 512 tokens: times `sqrt(512 / n)`); percentiles `abs <= 1e-2` |
| Streaming against whole chunks, in one implementation | `abs <= 1e-6` on every value above (the reference: equal to 1e-9) |

**Short texts.** A flip moves a short text further than the table allows, because few tokens average it out. The
engine against the reference on 480 random texts of 2 to 543 tokens, half in each attention mode, on the random
`s192_l4`: 23 of the 336 texts under 128 tokens (7%) left the table and none of the 144 longer ones, with pooled
vectors up to 2.5e-2 apart at 2 to 4 tokens, 1.4e-2 at 5 to 11, 1.1e-2 at 17 to 19, 6.6e-3 at 33 to 36 and 5.8e-3
at 65 to 70, and queries and keys up to 4.4e-3. Which texts flip depends on the machine that wrote the fixture,
since NumPy's sums differ between processors. A text of `n < 512` tokens that leaves the table therefore meets the
table's continuous tolerances times `sqrt(512 / n)`, with its discrete values not compared, and the engine's test
fails a fixture set in which more than half of the short texts do.

**Integer attention** meets the same table. Given the same queries, keys and values, the reference and the PyTorch
model in eval mode give the same bits (`tests/test_integer_attention.py`), and the engine's every backend gives the
portable path's bits (`tests/kernels.rs`); what separates implementations is the float rounding upstream, as in float
attention, but a weight that rounds the other way moves an output by up to `max |V| / 127`, more than one flipped
int8 activation does. Measured between the reference and the PyTorch model (random `s192_l4`): at
1,500 tokens, pooled vectors up to 1.9e-3 apart (float attention: 0.9e-3) and queries and keys up to 3.2e-4 (1.4e-4),
inside the table on every text. Shorter texts of random models exceed the table now and then in both modes: of 40
random texts each, pooled vectors beyond 3e-3 in 8% (9 tokens), 25% (64) and 28% (300) of texts with integer attention,
against 12%, 8% and 2% with float attention. On a briefly trained `tiny` model, 2 of 150 validation texts (1 to 80
tokens) exceed the query and key tolerance with integer attention and none with float attention. The engine on the
`tiny_integer` and `s192_l4_integer` fixtures: pooled vectors up to 2.2e-4 and 6.9e-4 apart, window vectors 8.7e-4 and
2.1e-3, queries and keys 6.0e-5 and 1.1e-4, strip positions 2.5e-5 and 4.3e-4. Sets, calls and the verdict agree.

**Int8 output** (format 3) has two parts, an exact one and one that can flip. What is exact: given the same pooled
vector, the engine's head gives the reference's codes (the head's float32 output is computed in float64 sums and
rounded, NumPy's in float32, so the two differ in the last bit of `o`, which moves a code only for a component within
that of half a step), and from the same codes the compare is the same bits up to `tanh`: on every chunk and window of
the `tiny_int8` and `s192_l4_integer_int8` fixtures, the engine fed the reference's pooled vectors returns the
reference's codes and its mean, spread, share and closeness within 3e-16. What can still flip: a text's own code at a
rounding boundary. A pooled vector that differs by 1e-3 (the encoder's rounding, [above](#parity)) moves the head's
output by about as much, which is 0.13 codes at the largest component, so a component near half a step rounds the other
way, one code apart, in many of a long text's 128 components; it moves the unit vector by about `1 / ||c||` (2e-3 for
`w = 128`), more than the float table's 1e-3 on a query or key, and a reading by more than the table's tolerances. On
`s192_l4_integer_int8` (engine against reference, integer attention; pooled vectors up to 1.2e-3 apart) every code is at
most one from the reference's, 18 of the 20 readings of the 2,691-token text have a code that is not, and those readings
are within 2.6e-4 on mean, 1.6e-4 on spread, 1.2e-4 on closeness, 1.9e-3 on three-way probabilities, 2.5e-3 on spread
percentiles, 1.5e-3 on strip positions, 8.0e-4 on cell positions and 1.0e-3 on the verdict's three-way probabilities;
sets, calls and Venn-Abers blocks agree except near their thresholds, as before. On `tiny_int8` (float attention,
pooled vectors 1.7e-7 apart) every code is equal and every value is within 3e-15. A reading with a code that is not the
reference's therefore meets the table's continuous tolerances times 2, its discrete values compared as before; one
whose codes are all the reference's meets the table. Between the reference and the PyTorch model in eval mode, on a
briefly trained `tiny` model with int8 output (150 validation texts of 1 to 80 tokens), the median text's codes are
equal and the worst text's are one apart.

**On a trained model** the table's query and key line is too tight for short texts; the engine is tested against
the table on its fixtures, and the export checks a trained release against the bounds at the end of this paragraph.
On a trained checkpoint (`s192_l4`, a 64-wide head with clamp 30, float attention, step 999), the reference against the PyTorch model on 113 slices of four documents: queries and
keys, which are unit vectors with components up to 0.44, differ by 1e-4 at the median at every length from 128 to
4,096 tokens and by under 2e-7 at the median below that, and at worst by 5.2e-4 (16 tokens), 6.8e-4 (32), 8.6e-4
(64), 2.1e-3 (128), 6.0e-4 (256), 4.0e-4 (512) and 1e-4 (4,096); pooled vectors at worst 1.8e-3 (32 tokens). A short
text is one or two flipped activations away from another short text and has few tokens to average them over. The
export's own check (`export.verify_release`) therefore holds the median text to 5e-4 on queries and keys and 1e-3 on
the pooled vector, where a wrong weight or norm would show on every text, and the worst text to 1e-2 on both.
