//! The ternary product: a matrix of weights in {-1, 0, +1} times a batch of 8-bit activation rows, which is every
//! linear layer of the encoder (SPEC.md, section 2).
//!
//! The sum `acc = sum_i t_i * q_i` is an exact integer, so every backend returns the same bits. The scalar path
//! defines the answer; the SIMD paths are tested against it on random inputs of every shape and remainder.
//!
//! Layout. Weights stay in the release file's I2_S packing (2 bits a weight, code `t + 1`, four to a byte), which is
//! the smallest exact form, and are unpacked [`TILE_ROWS`] rows at a time into a tile that stays in L1 while every
//! token of the batch streams past it. An encoder runs a whole block of tokens through each matrix, so the unpack
//! costs one table lookup per four weights per batch, under 1% of the products at 64 tokens or more.
//!
//! Backends (chosen at run time, [`Backend::detect`]):
//! - `Avx2`: the tile holds codes `t + 1` in {0, 1, 2} as unsigned bytes; `vpmaddubsw` multiplies them with the
//!   signed activations and adds pairs into 16-bit lanes, and `sum_i q_i` is subtracted once per output. A 4 rows by 2
//!   tokens register block reuses each weight load twice and each activation load four times.
//! - `Avx2Sign`: the tile holds `t` as signed bytes; `vpsignb` negates or zeroes each activation (a pure
//!   add-and-subtract form) and `vpmaddubsw` against ones adds pairs. Kept to measure against `Avx2`.
//! - `NeonDot` (`sdot`, Armv8.2 dot product) and `Neon` (`smlal` into 16-bit lanes) on aarch64.
//! - `Simd128` (`i16x8.extmul` into 16-bit lanes) on wasm32 built with `+simd128`.
//!
//! The column path ([`ColumnPairs`], [`matmul_f32_columns`]). The feed-forward's `down` matrix reads squared-ReLU
//! activations, which a trained model leaves mostly zero after quantizing (69 to 92% in a trained release).
//! For such a token the product walks only the nonzero inputs: `acc += q_i * column_i`. Columns are taken in pairs
//! `(2k, 2k + 1)` stored interleaved as codes `t + 1`, so one `vpmaddubsw` (or `i32x4.dot_i16x8_s`, or `smlal`) adds
//! `c0 * q_2k + c1 * q_2k+1` into 16 rows at once, and a pair is skipped when both its activations are zero. The
//! integers are exact either way, so the result equals the dense path bit for bit; each token takes the column path
//! when its nonzero pairs are under [`column_limit`], else the dense path.

use serde::{Deserialize, Serialize};

pub mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;

/// Activation rows are padded with zeros to a multiple of this many columns.
pub const LANES: usize = 32;

/// Rows of weights unpacked into one tile.
pub const TILE_ROWS: usize = 16;

/// The instruction set that runs the ternary product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Backend {
    Scalar,
    Avx2,
    Avx2Sign,
    Neon,
    NeonDot,
    Simd128,
}

/// How a tile stores a weight: the code `t + 1` as an unsigned byte, or `t` as a signed byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TileForm {
    Codes,
    Signed,
}

impl Backend {
    /// The fastest backend this processor supports.
    pub fn detect() -> Self {
        let available = Self::available();
        for preferred in [Backend::Avx2, Backend::NeonDot, Backend::Neon, Backend::Simd128] {
            if available.contains(&preferred) {
                return preferred;
            }
        }
        Backend::Scalar
    }

    /// Every backend this processor supports, the scalar path first.
    pub fn available() -> Vec<Self> {
        [Backend::Scalar, Backend::Avx2, Backend::Avx2Sign, Backend::Neon, Backend::NeonDot, Backend::Simd128]
            .into_iter()
            .filter(|backend| backend.is_available())
            .collect()
    }

    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 | Backend::Avx2Sign => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(target_arch = "aarch64")]
            Backend::NeonDot => {
                std::arch::is_aarch64_feature_detected!("neon") && std::arch::is_aarch64_feature_detected!("dotprod")
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Avx2 => "avx2",
            Backend::Avx2Sign => "avx2-sign",
            Backend::Neon => "neon",
            Backend::NeonDot => "neon-dotprod",
            Backend::Simd128 => "simd128",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        [Backend::Scalar, Backend::Avx2, Backend::Avx2Sign, Backend::Neon, Backend::NeonDot, Backend::Simd128]
            .into_iter()
            .find(|backend| backend.name() == name)
    }

    fn form(self) -> TileForm {
        match self {
            Backend::Avx2 => TileForm::Codes,
            _ => TileForm::Signed,
        }
    }
}

/// One ternary matrix `(rows, cols)` in the release file's packing, with its scale `gamma`.
#[derive(Clone, Debug)]
pub struct PackedTernary {
    rows: usize,
    cols: usize,
    packed: Vec<u8>, // (rows, cols / 4)
    gamma: f32,
}

impl PackedTernary {
    /// Checks the packing: `cols` a multiple of 4, `rows * cols / 4` bytes, and no code 3.
    pub fn new(rows: usize, cols: usize, packed: Vec<u8>, gamma: f32) -> Result<Self, String> {
        if cols % 4 != 0 {
            return Err(format!("a packed ternary matrix needs a multiple of 4 columns, not {cols}"));
        }
        if packed.len() != rows * cols / 4 {
            return Err(format!("{} bytes for a ({rows}, {cols}) packed ternary matrix", packed.len()));
        }
        if let Some(position) = packed.iter().position(|&byte| has_code_three(byte)) {
            return Err(format!("packed ternary byte {position} holds the invalid code 3"));
        }
        if !gamma.is_finite() {
            return Err(format!("ternary scale {gamma} is not finite"));
        }
        Ok(Self { rows, cols, packed, gamma })
    }

    /// Packs values in {-1, 0, 1}, row-major `(rows, cols)`.
    pub fn from_values(rows: usize, cols: usize, values: &[i8], gamma: f32) -> Self {
        assert_eq!(values.len(), rows * cols, "one value per weight");
        assert!(cols % 4 == 0, "columns must be a multiple of 4");
        let packed = values
            .chunks_exact(4)
            .map(|four| {
                four.iter().enumerate().fold(0u8, |byte, (slot, &value)| {
                    assert!((-1..=1).contains(&value), "ternary values only");
                    byte | (((value + 1) as u8) << (2 * slot))
                })
            })
            .collect();
        Self { rows, cols, packed, gamma }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// The weight at `(row, col)`, in {-1, 0, 1}.
    pub fn value(&self, row: usize, col: usize) -> i8 {
        let byte = self.packed[row * self.cols / 4 + col / 4];
        ((byte >> (2 * (col % 4))) & 3) as i8 - 1
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.packed.len()
    }
}

fn has_code_three(byte: u8) -> bool {
    (0..4).any(|slot| (byte >> (2 * slot)) & 3 == 3)
}

/// A batch of activation rows quantized to 8 bits per token (SPEC.md, section 2): values `q`, their sum per row (for
/// the code form's correction) and the scale `s` per row.
#[derive(Clone, Debug, Default)]
pub struct QuantRows {
    n: usize,
    cols: usize,
    stride: usize,
    q: Vec<i8>,   // (n, stride), zero past `cols`
    sum: Vec<i32>, // (n,)
    s: Vec<f32>,   // (n,)
}

impl QuantRows {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn scales(&self) -> &[f32] {
        &self.s[..self.n]
    }

    pub fn values(&self, row: usize) -> &[i8] {
        &self.q[row * self.stride..row * self.stride + self.cols]
    }

    fn reset(&mut self, n: usize, cols: usize) {
        self.n = n;
        self.cols = cols;
        self.stride = cols.div_ceil(LANES) * LANES;
        let len = n * self.stride;
        if self.q.len() < len {
            self.q.resize(len, 0);
        }
        if self.sum.len() < n {
            self.sum.resize(n, 0);
            self.s.resize(n, 0.0);
        }
    }

    /// Quantize `n` rows of `cols` floats: `s = 127 / max(max |x|, eps)`, `q = clamp(round(x * s), -128, 127)`,
    /// rounding half to even.
    pub fn quantize(&mut self, x: &[f32], n: usize, cols: usize, eps: f32) {
        assert_eq!(x.len(), n * cols, "quantize takes n rows of cols values");
        self.reset(n, cols);
        for row in 0..n {
            let source = &x[row * cols..(row + 1) * cols];
            let target = &mut self.q[row * self.stride..(row + 1) * self.stride];
            let (s, sum) = crate::dense::quantize_row(source, &mut target[..cols], eps);
            target[cols..].fill(0);
            self.s[row] = s;
            self.sum[row] = sum;
        }
    }

    /// Rows `tokens` of these rows, in that order, into `into`.
    fn copy_rows(&self, tokens: &[usize], into: &mut QuantRows) {
        into.reset(tokens.len(), self.cols);
        assert_eq!(into.stride, self.stride);
        for (row, &t) in tokens.iter().enumerate() {
            into.q[row * self.stride..(row + 1) * self.stride].copy_from_slice(&self.q[t * self.stride..(t + 1) * self.stride]);
            into.sum[row] = self.sum[t];
            into.s[row] = self.s[t];
        }
    }

    /// Rows of given integer values with given scales (tests and benchmarks).
    pub fn from_values(values: &[i8], n: usize, cols: usize, scales: &[f32]) -> Self {
        assert_eq!(values.len(), n * cols, "n rows of cols values");
        assert_eq!(scales.len(), n, "one scale per row");
        let mut rows = Self::new();
        rows.reset(n, cols);
        for row in 0..n {
            let source = &values[row * cols..(row + 1) * cols];
            let target = &mut rows.q[row * rows.stride..(row + 1) * rows.stride];
            target[..cols].copy_from_slice(source);
            target[cols..].fill(0);
            rows.sum[row] = source.iter().map(|&value| value as i32).sum();
            rows.s[row] = scales[row];
        }
        rows
    }
}

/// Reusable buffers of the ternary product: one unpacked tile and its integer outputs, and the column path's lists.
#[derive(Debug, Default)]
pub struct KernelScratch {
    tile: Vec<u8>, // (TILE_ROWS, stride)
    acc: Vec<i32>, // (n, TILE_ROWS)
    dense_rows: QuantRows,
    dense_tokens: Vec<usize>,
    offsets: Vec<u32>,     // (cols / 2,) byte offset of each nonzero pair's codes
    pairs: Vec<u16>,       // (cols / 2,) the pair's two activations, low byte first
    column_acc: Vec<i32>,  // (height,)
}

impl KernelScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bytes(&self) -> usize {
        self.tile.capacity()
            + 4 * self.acc.capacity()
            + self.dense_rows.q.capacity()
            + 8 * self.dense_rows.sum.capacity()
            + 8 * self.dense_tokens.capacity()
            + 4 * self.offsets.capacity()
            + 2 * self.pairs.capacity()
            + 4 * self.column_acc.capacity()
    }
}

/// A ternary matrix `(rows, cols)` stored a second time by column pairs for the column path: pair `k` holds, for
/// every row `r` of `height` (rows padded to a multiple of 16), the codes `t + 1` of columns `2k` and `2k + 1` as two
/// adjacent bytes; padding rows hold code 1 (weight 0). `cols * height` bytes, four times the packed matrix.
#[derive(Clone, Debug)]
pub struct ColumnPairs {
    rows: usize,
    cols: usize,
    height: usize,
    codes: Vec<u8>, // (cols / 2, height, 2)
}

impl ColumnPairs {
    pub fn new(weights: &PackedTernary) -> Self {
        let (rows, cols) = (weights.rows, weights.cols);
        let height = rows.div_ceil(16) * 16;
        let mut codes = vec![1u8; cols * height];
        for r in 0..rows {
            for c in 0..cols {
                codes[(c / 2) * 2 * height + 2 * r + c % 2] = (weights.value(r, c) + 1) as u8;
            }
        }
        Self { rows, cols, height, codes }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.codes.len()
    }
}

/// The largest count of nonzero activation pairs for which a token takes the column path over `columns` on
/// `backend`, where the column path's time on synthetic activations crossed the dense path's (engine README,
/// the column path; a Zen 2 core with 512 KB of L2). AVX2's column path keeps pace with the dense path
/// only while the codes stay in L2 (d = 192, 147 KB: 1.2 times the dense time with no zeros), and is memory-bound
/// past it (d = 256 to 512: 1.5 to 1.9 times); the scalar path is compute-bound at every size (1.7 to 1.9 times).
/// In V8 the column path beats the dense path even with no zeros, with SIMD128 (0.67 times its time) and without it
/// (the scalar path, 0.8 to 0.9 times), so every token takes it in WebAssembly. NEON is not measured and
/// takes the native scalar path's limit.
pub fn column_limit(backend: Backend, columns: &ColumnPairs) -> usize {
    let percent = match backend {
        Backend::Avx2 | Backend::Avx2Sign if columns.bytes() <= 192 * 1024 => 75,
        Backend::Avx2 | Backend::Avx2Sign => 50,
        Backend::Scalar if cfg!(target_arch = "wasm32") => 100,
        Backend::Scalar | Backend::Neon | Backend::NeonDot => 55,
        Backend::Simd128 => 100,
    };
    columns.cols / 2 * percent / 100
}

/// Collect a token's nonzero activation pairs, in increasing order, into `offsets` (each pair's byte offset in the
/// codes) and `pairs` (each `cols / 2` long); returns their count. A mask of 32 pairs at a time (which the compiler
/// forms with a vector compare), then one step per nonzero pair.
fn gather_pairs(q: &[i8], height: usize, offsets: &mut [u32], pairs: &mut [u16]) -> usize {
    let half = q.len() / 2;
    assert!(q.len() % 2 == 0 && offsets.len() >= half && pairs.len() >= half);
    let pair_at = |k: usize| u16::from_le_bytes([q[2 * k] as u8, q[2 * k + 1] as u8]);
    let mut count = 0;
    let mut k0 = 0;
    while k0 < half {
        let span = (half - k0).min(32);
        let mut mask = 0u32;
        if span == 32 {
            let block: &[i8; 64] = q[2 * k0..2 * k0 + 64].try_into().unwrap();
            for i in 0..32 {
                mask |= (((block[2 * i] | block[2 * i + 1]) != 0) as u32) << i;
            }
        } else {
            for i in 0..span {
                mask |= ((pair_at(k0 + i) != 0) as u32) << i;
            }
        }
        while mask != 0 {
            let k = k0 + mask.trailing_zeros() as usize;
            offsets[count] = (k * 2 * height) as u32;
            pairs[count] = pair_at(k);
            count += 1;
            mask &= mask - 1;
        }
        k0 += span;
    }
    count
}

/// The code sums of one token over its nonzero pairs: `acc[r] = sum_k c(r, 2k) q_2k + c(r, 2k + 1) q_2k+1`.
fn column_sums(backend: Backend, columns: &ColumnPairs, offsets: &[u32], pairs: &[u16], acc: &mut [i32]) {
    let height = columns.height;
    assert!(acc.len() == height && offsets.len() == pairs.len());
    // `gather_pairs` writes the offsets in increasing order, so the last bounds them all.
    debug_assert!(offsets.windows(2).all(|two| two[0] < two[1]));
    assert!(offsets.last().is_none_or(|&last| last as usize + 2 * height <= columns.codes.len()), "pair offsets inside the codes");
    acc.fill(0);
    match backend {
        Backend::Scalar => scalar::columns(&columns.codes, height, offsets, pairs, acc),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the caller asserted that the processor has AVX2; offsets were checked above.
        Backend::Avx2 | Backend::Avx2Sign => unsafe { avx2::columns(&columns.codes, height, offsets, pairs, acc) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the caller asserted that the processor has NEON; offsets were checked above.
        Backend::Neon | Backend::NeonDot => unsafe { neon::columns(&columns.codes, height, offsets, pairs, acc) },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        // SAFETY: offsets were checked above.
        Backend::Simd128 => unsafe { simd128::columns(&columns.codes, height, offsets, pairs, acc) },
        #[allow(unreachable_patterns)]
        _ => unreachable!("backend {} is not compiled for this target", backend.name()),
    }
}

/// The exact integer product by the column path alone, every token: `out[t * rows + r] = sum_c t_rc * q_tc`.
pub fn column_i32(backend: Backend, columns: &ColumnPairs, x: &QuantRows, out: &mut [i32], scratch: &mut KernelScratch) {
    assert_eq!(columns.cols, x.cols, "the matrix's columns must match the activations' width");
    assert!(backend.is_available(), "backend {} is not available on this processor", backend.name());
    let rows = columns.rows;
    assert_eq!(out.len(), x.n * rows, "out holds n rows of the matrix's outputs");
    let half = columns.cols / 2;
    scratch.offsets.resize(half, 0);
    scratch.pairs.resize(half, 0);
    scratch.column_acc.resize(columns.height, 0);
    for t in 0..x.n {
        let count = gather_pairs(x.values(t), columns.height, &mut scratch.offsets, &mut scratch.pairs);
        column_sums(backend, columns, &scratch.offsets[..count], &scratch.pairs[..count], &mut scratch.column_acc);
        for (target, &sum) in out[t * rows..(t + 1) * rows].iter_mut().zip(&scratch.column_acc) {
            *target = sum - x.sum[t];
        }
    }
}

const fn unpack_table(signed: bool) -> [[u8; 4]; 256] {
    let mut table = [[0u8; 4]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut slot = 0;
        while slot < 4 {
            let code = ((byte >> (2 * slot)) & 3) as u8;
            // Code 3 never reaches a tile: PackedTernary::new refuses it.
            table[byte][slot] = if signed { (code as i8 - 1) as u8 } else { code };
            slot += 1;
        }
        byte += 1;
    }
    table
}

static UNPACK_CODES: [[u8; 4]; 256] = unpack_table(false);
static UNPACK_SIGNED: [[u8; 4]; 256] = unpack_table(true);

/// Unpack rows `[r0, r0 + rows)` into a tile of `TILE_ROWS` rows of `stride` bytes; padding holds the zero weight.
fn unpack_tile(weights: &PackedTernary, r0: usize, rows: usize, stride: usize, form: TileForm, tile: &mut [u8]) {
    let (table, zero) = match form {
        TileForm::Codes => (&UNPACK_CODES, 1u8),
        TileForm::Signed => (&UNPACK_SIGNED, 0u8),
    };
    let packed_cols = weights.cols / 4;
    for row in 0..TILE_ROWS {
        let target = &mut tile[row * stride..(row + 1) * stride];
        if row < rows {
            let source = &weights.packed[(r0 + row) * packed_cols..(r0 + row + 1) * packed_cols];
            for (four, &byte) in target.chunks_exact_mut(4).zip(source) {
                four.copy_from_slice(&table[byte as usize]);
            }
            target[weights.cols..].fill(zero);
        } else {
            target.fill(zero);
        }
    }
}

/// Run the product tile by tile; `emit(r0, rows, acc)` receives the exact sums `acc[t * TILE_ROWS + r]` of rows
/// `[r0, r0 + rows)` for every token `t`.
fn run(
    backend: Backend,
    weights: &PackedTernary,
    x: &QuantRows,
    scratch: &mut KernelScratch,
    mut emit: impl FnMut(usize, usize, &[i32]),
) {
    assert_eq!(weights.cols, x.cols, "the matrix's columns must match the activations' width");
    assert!(backend.is_available(), "backend {} is not available on this processor", backend.name());
    let stride = x.stride;
    let n = x.n;
    if scratch.tile.len() < TILE_ROWS * stride {
        scratch.tile.resize(TILE_ROWS * stride, 0);
    }
    if scratch.acc.len() < n * TILE_ROWS {
        scratch.acc.resize(n * TILE_ROWS, 0);
    }
    let form = backend.form();
    let tile = &mut scratch.tile[..TILE_ROWS * stride];
    let acc = &mut scratch.acc[..n * TILE_ROWS];
    let q = &x.q[..n * stride];
    let sums = &x.sum[..n];
    for r0 in (0..weights.rows).step_by(TILE_ROWS) {
        let rows = TILE_ROWS.min(weights.rows - r0);
        unpack_tile(weights, r0, rows, stride, form, tile);
        match backend {
            Backend::Scalar => scalar::tile(tile, stride, q, n, acc),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `run` asserted that the processor has AVX2.
            Backend::Avx2 => unsafe { avx2::tile_codes(tile, stride, q, sums, n, acc) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above.
            Backend::Avx2Sign => unsafe { avx2::tile_sign(tile, stride, q, n, acc) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: `run` asserted that the processor has NEON.
            Backend::Neon => unsafe { neon::tile_mlal(tile, stride, q, n, acc) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: `run` asserted that the processor has the dot product extension.
            Backend::NeonDot => unsafe { neon::tile_dot(tile, stride, q, n, acc) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => simd128::tile(tile, stride, q, n, acc),
            #[allow(unreachable_patterns)]
            _ => unreachable!("backend {} is not compiled for this target", backend.name()),
        }
        let _ = sums;
        emit(r0, rows, acc);
    }
}

/// The exact integer product: `out[t * rows + r] = sum_c t_rc * q_tc`, `(n, rows)`.
pub fn matmul_i32(backend: Backend, weights: &PackedTernary, x: &QuantRows, out: &mut [i32], scratch: &mut KernelScratch) {
    let rows_total = weights.rows;
    assert_eq!(out.len(), x.n * rows_total, "out holds n rows of the matrix's outputs");
    run(backend, weights, x, scratch, |r0, rows, acc| {
        for t in 0..x.n {
            out[t * rows_total + r0..t * rows_total + r0 + rows].copy_from_slice(&acc[t * TILE_ROWS..t * TILE_ROWS + rows]);
        }
    });
}

/// The ternary product scaled to floats (SPEC.md, section 2): `y = float32(acc) * (gamma / s_t)`, written into
/// `out[t * out_stride + r]`, or added to it when `accumulate` (the residual stream).
pub fn matmul_f32(
    backend: Backend,
    weights: &PackedTernary,
    x: &QuantRows,
    out: &mut [f32],
    out_stride: usize,
    accumulate: bool,
    scratch: &mut KernelScratch,
) {
    let n = x.n;
    assert!(out_stride >= weights.rows, "output rows must hold the matrix's outputs");
    assert!(out.len() >= (n.max(1) - 1) * out_stride + weights.rows || n == 0, "output too short");
    let gamma = weights.gamma;
    let scales = &x.s[..n];
    run(backend, weights, x, scratch, |r0, rows, acc| {
        for t in 0..n {
            let factor = gamma / scales[t];
            let target = &mut out[t * out_stride + r0..t * out_stride + r0 + rows];
            let source = &acc[t * TILE_ROWS..t * TILE_ROWS + rows];
            epilogue(target, source, 0, factor, accumulate);
        }
    });
}

/// `y = float32(a - correction) * factor`, written or added: the one float step of every path.
#[inline(always)]
fn epilogue(target: &mut [f32], source: &[i32], correction: i32, factor: f32, accumulate: bool) {
    if accumulate {
        for (y, &a) in target.iter_mut().zip(source) {
            *y += (a - correction) as f32 * factor;
        }
    } else {
        for (y, &a) in target.iter_mut().zip(source) {
            *y = (a - correction) as f32 * factor;
        }
    }
}

/// [`matmul_f32`] with a per-token choice: a token with fewer than `column_below` nonzero activation pairs (of
/// `cols / 2`) takes the column path over `columns`, the others the dense path over `weights`, both the same matrix.
/// The integers are exact on both paths, so the output is the same bits for any `column_below`; 0 is the dense path
/// alone, `cols / 2 + 1` the column path alone, and [`column_limit`] the measured choice. Returns the tokens that took
/// the column path.
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32_columns(
    backend: Backend,
    weights: &PackedTernary,
    columns: &ColumnPairs,
    x: &QuantRows,
    out: &mut [f32],
    out_stride: usize,
    accumulate: bool,
    column_below: usize,
    scratch: &mut KernelScratch,
) -> usize {
    let n = x.n;
    assert!(columns.rows == weights.rows && columns.cols == weights.cols, "the column pairs are this matrix's");
    assert_eq!(weights.cols, x.cols, "the matrix's columns must match the activations' width");
    assert!(backend.is_available(), "backend {} is not available on this processor", backend.name());
    assert!(out_stride >= weights.rows, "output rows must hold the matrix's outputs");
    assert!(out.len() >= (n.max(1) - 1) * out_stride + weights.rows || n == 0, "output too short");
    if column_below == 0 {
        matmul_f32(backend, weights, x, out, out_stride, accumulate, scratch);
        return 0;
    }
    let (rows, half, gamma) = (weights.rows, weights.cols / 2, weights.gamma);
    scratch.offsets.resize(half, 0);
    scratch.pairs.resize(half, 0);
    scratch.column_acc.resize(columns.height, 0);
    let mut dense_tokens = std::mem::take(&mut scratch.dense_tokens);
    dense_tokens.clear();
    for t in 0..n {
        let count = gather_pairs(x.values(t), columns.height, &mut scratch.offsets, &mut scratch.pairs);
        if count >= column_below {
            dense_tokens.push(t);
            continue;
        }
        column_sums(backend, columns, &scratch.offsets[..count], &scratch.pairs[..count], &mut scratch.column_acc);
        let factor = gamma / x.s[t];
        epilogue(&mut out[t * out_stride..t * out_stride + rows], &scratch.column_acc[..rows], x.sum[t], factor, accumulate);
    }
    let column_tokens = n - dense_tokens.len();
    if dense_tokens.len() == n {
        matmul_f32(backend, weights, x, out, out_stride, accumulate, scratch);
    } else if !dense_tokens.is_empty() {
        let mut subset = std::mem::take(&mut scratch.dense_rows);
        x.copy_rows(&dense_tokens, &mut subset);
        let scales = &subset.s[..dense_tokens.len()];
        run(backend, weights, &subset, scratch, |r0, rows, acc| {
            for (j, &t) in dense_tokens.iter().enumerate() {
                let factor = gamma / scales[j];
                let target = &mut out[t * out_stride + r0..t * out_stride + r0 + rows];
                epilogue(target, &acc[j * TILE_ROWS..j * TILE_ROWS + rows], 0, factor, accumulate);
            }
        });
        scratch.dense_rows = subset;
    }
    scratch.dense_tokens = dense_tokens;
    column_tokens
}
