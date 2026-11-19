//! Integer attention (SPEC.md, Integer attention), the optional mode a format 2 release names in `model.attention`:
//! queries and keys are int8 with one scale per token and head, values int8 with one scale per block and head, and
//! both products of attention are integer products. Per head and block:
//!
//! - scores `S_ij = (float32(sum_c q_ic k_jc) * x_i) * y_j`, with `x_i = float32(1 / sqrt(d_h)) / s_q_i` and
//!   `y_j = 1 / s_k_j`;
//! - weights `p_ij = round(E_ij * r_i)` in whole 127ths, `E_ij = exp(S_ij - max_k S_ik)` by `dense::exp_nonpositive`,
//!   `r_i = float32(127 / z_i)` and `z_i` the float64 sum of the row's `E`;
//! - outputs `O_ic = float32(sum_j p_ij v_jc) * t`, `t = 1 / (127 * s_v)`.
//!
//! The integers are exact on every path and the float steps run in one order, so every backend gives the same bits
//! (`tests/kernels.rs`; WebAssembly in `scripts/wasm_probe.mjs`). SIMD128 (`i32x4.dot_i16x8_s`) and NEON
//! (`smull` and `addp`) work on 16-bit pairs throughout (`dense_simd128.rs`, `dense_neon.rs`). AVX2 forms the scores with `vpmaddwd` on 16-bit queries and keys (two columns a lane) and the
//! weighted values with `vpmaddubsw` on unsigned weights and signed values (two tokens a lane), summed in 16-bit
//! lanes over spans of `SPAN` (128) tokens: a span's weights sum to at most `127 + SPAN / 2`, so its sums stay within
//! `127 * 191 < 2^15`.

use crate::dense::{exp_nonpositive, max_abs, round_even_small, row_max};
use crate::kernels::Backend;

/// Attention weights are whole 127ths.
pub const LEVELS: f32 = 127.0;
/// Keys of a score tile.
pub(crate) const KEYS: usize = 16;
/// Columns of an output tile.
pub(crate) const COLS: usize = 32;
/// Tokens whose weighted values are summed in 16-bit lanes before they are widened.
#[cfg(target_arch = "x86_64")]
const SPAN: usize = 128;

/// Buffers of one head's integer attention over one block, reused for every block.
#[derive(Debug, Default)]
pub struct IntegerScratch {
    q: Vec<i16>,      // (n, d_h): this head's query integers
    x: Vec<f32>,      // (n,): the queries' factors
    kt: Vec<i16>,     // (d_h / 2, n_pad, 2): key integers, two columns of each key side by side, zero past n
    y: Vec<f32>,      // (n_pad,): the keys' factors, zero past n
    vt: Vec<i8>,      // (n_pad / 2, dh_pad, 2): value integers, two tokens of each column side by side, zero past n and d_h
    scores: Vec<f32>, // (4, n_pad): four rows of scores, then of exponentials
    weights: Vec<u8>, // (4, n_pad): four rows of weights, zero past n
}

impl IntegerScratch {
    pub fn bytes(&self) -> usize {
        2 * (self.q.capacity() + self.kt.capacity())
            + 4 * (self.x.capacity() + self.y.capacity() + self.scores.capacity())
            + self.vt.capacity()
            + self.weights.capacity()
    }
}

/// The widest head integer attention reads (SPEC.md: an integer score must be exact in float32).
pub const MAX_HEAD: usize = 1040;

/// `clamp(round(value * s), -127, 127)`.
#[inline(always)]
fn level(value: f32, s: f32) -> i32 {
    let rounded = round_even_small(value * s).clamp(-127.0, 127.0);
    // SAFETY: `rounded` is an integer in [-127, 127].
    unsafe { rounded.to_int_unchecked::<i32>() }
}

#[inline(always)]
fn levels_i16(row: &[f32], s: f32, out: &mut [i16]) {
    for (target, &value) in out.iter_mut().zip(row) {
        *target = level(value, s) as i16;
    }
}

#[inline(always)]
fn levels_i8(row: &[f32], s: f32, out: &mut [i8]) {
    for (target, &value) in out.iter_mut().zip(row) {
        *target = level(value, s) as i8;
    }
}

/// Integer attention inside one block for every head. `q`, `k`, `v` and `o` are `(n, d)` with heads side by side;
/// `eps` is the release's `act_eps`.
#[allow(clippy::too_many_arguments)]
pub fn block_attention(
    backend: Backend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    n: usize,
    d: usize,
    heads: usize,
    eps: f32,
    scratch: &mut IntegerScratch,
) {
    assert!(backend.is_available(), "backend {} is not available on this processor", backend.name());
    match backend {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the backend is available, so the processor has AVX2.
        Backend::Avx2 | Backend::Avx2Sign => unsafe { body_avx2(q, k, v, o, n, d, heads, eps, scratch) },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Backend::Simd128 => body(q, k, v, o, n, d, heads, eps, scratch, Path::Simd128),
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the backend is available, so the processor has NEON.
        Backend::Neon | Backend::NeonDot => unsafe { body_neon(q, k, v, o, n, d, heads, eps, scratch) },
        _ => body(q, k, v, o, n, d, heads, eps, scratch, Path::Portable),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn body_neon(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], n: usize, d: usize, heads: usize, eps: f32, scratch: &mut IntegerScratch) {
    body(q, k, v, o, n, d, heads, eps, scratch, Path::Neon);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn body_avx2(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], n: usize, d: usize, heads: usize, eps: f32, scratch: &mut IntegerScratch) {
    body(q, k, v, o, n, d, heads, eps, scratch, Path::Avx2);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn body(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], n: usize, d: usize, heads: usize, eps: f32, scratch: &mut IntegerScratch, path: Path) {
    assert!(heads > 0 && d % heads == 0 && (d / heads) % 2 == 0, "heads of even width");
    let d_h = d / heads;
    assert!(d_h * 127 * 127 < 1 << 24, "an integer score is exact in float32");
    assert!(q.len() >= n * d && k.len() >= n * d && v.len() >= n * d && o.len() >= n * d, "(n, d) rows");
    let half = d_h / 2;
    let n_pad = n.div_ceil(KEYS) * KEYS;
    let dh_pad = d_h.div_ceil(COLS) * COLS;
    let inv_sqrt = (1.0 / (d_h as f64).sqrt()) as f32;
    let IntegerScratch { q: qi, x, kt, y, vt, scores, weights } = scratch;
    qi.clear();
    qi.resize(n * d_h, 0);
    x.clear();
    x.resize(n, 0.0);
    kt.clear();
    kt.resize(half * n_pad * 2, 0);
    y.clear();
    y.resize(n_pad, 0.0);
    vt.clear();
    vt.resize(n_pad * dh_pad, 0);
    scores.clear();
    scores.resize(4 * n_pad, 0.0);
    weights.clear();
    weights.resize(4 * n_pad, 0);
    for head in 0..heads {
        let column = head * d_h;
        // Queries and keys: one scale per token; values: one scale for the block.
        let mut top_v = 0f32;
        for i in 0..n {
            let row = &q[i * d + column..i * d + column + d_h];
            let s = 127.0f32 / max_abs(row).max(eps);
            x[i] = inv_sqrt / s;
            levels_i16(row, s, &mut qi[i * d_h..(i + 1) * d_h]);
            let row = &k[i * d + column..i * d + column + d_h];
            let s = 127.0f32 / max_abs(row).max(eps);
            y[i] = 1.0 / s;
            let mut key = [0i16; MAX_HEAD];
            levels_i16(row, s, &mut key[..d_h]);
            for (p, pair) in key[..d_h].chunks_exact(2).enumerate() {
                kt[(p * n_pad + i) * 2..(p * n_pad + i) * 2 + 2].copy_from_slice(pair);
            }
            let m = max_abs(&v[i * d + column..i * d + column + d_h]);
            top_v = if m > top_v { m } else { top_v };
        }
        let s_v = 127.0f32 / top_v.max(eps);
        let t = 1.0f32 / (LEVELS * s_v);
        for j0 in (0..n).step_by(2) {
            let (mut first, mut second) = ([0i8; MAX_HEAD], [0i8; MAX_HEAD]);
            levels_i8(&v[j0 * d + column..j0 * d + column + d_h], s_v, &mut first[..d_h]);
            if j0 + 1 < n {
                levels_i8(&v[(j0 + 1) * d + column..(j0 + 1) * d + column + d_h], s_v, &mut second[..d_h]);
            }
            let target = &mut vt[(j0 / 2) * dh_pad * 2..(j0 / 2) * dh_pad * 2 + 2 * d_h];
            for ((pair, &a), &b) in target.chunks_exact_mut(2).zip(&first[..d_h]).zip(&second[..d_h]) {
                pair[0] = a;
                pair[1] = b;
            }
        }
        let mut i0 = 0;
        while i0 < n {
            let rows = (n - i0).min(4);
            // Rows past the block repeat its last row; their results are not stored.
            let row_of = |r: usize| i0 + r.min(rows - 1);
            let index = [row_of(0), row_of(1), row_of(2), row_of(3)];
            let factors = index.map(|i| x[i]);
            for j0 in (0..n_pad).step_by(KEYS) {
                match path {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 is present; the buffers are sized above.
                    Path::Avx2 => unsafe { avx2::scores(qi, d_h, index, kt, n_pad, j0, factors, y, scores) },
                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    Path::Simd128 => crate::dense_simd128::integer_scores(qi, d_h, index, kt, n_pad, j0, factors, y, scores),
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is present; the buffers are sized above.
                    Path::Neon => unsafe { crate::dense_neon::integer_scores(qi, d_h, index, kt, n_pad, j0, factors, y, scores) },
                    Path::Portable => portable::scores(qi, d_h, index, kt, n_pad, j0, factors, y, scores),
                }
            }
            for r in 0..rows {
                let (row, levels) = (&mut scores[r * n_pad..r * n_pad + n], &mut weights[r * n_pad..(r + 1) * n_pad]);
                match path {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 is present.
                    Path::Avx2 => unsafe { avx2::softmax_levels(row, levels) },
                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    Path::Simd128 => crate::dense_simd128::softmax_levels(row, levels),
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is present.
                    Path::Neon => unsafe { crate::dense_neon::softmax_levels(row, levels) },
                    Path::Portable => portable::softmax_levels(row, levels),
                }
            }
            for c0 in (0..dh_pad).step_by(COLS) {
                let mut acc = [[0i32; COLS]; 4];
                match path {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 is present; the buffers are sized above.
                    Path::Avx2 => unsafe { avx2::weighted(weights, n_pad, vt, dh_pad, c0, &mut acc) },
                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    Path::Simd128 => crate::dense_simd128::integer_weighted(weights, n_pad, vt, dh_pad, c0, &mut acc),
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is present; the buffers are sized above.
                    Path::Neon => unsafe { crate::dense_neon::integer_weighted(weights, n_pad, vt, dh_pad, c0, &mut acc) },
                    Path::Portable => portable::weighted(weights, n_pad, vt, dh_pad, c0, &mut acc),
                }
                let width = COLS.min(d_h - c0);
                for (r, sums) in acc.iter().enumerate().take(rows) {
                    let at = (i0 + r) * d + column + c0;
                    for (out, &sum) in o[at..at + width].iter_mut().zip(sums) {
                        *out = sum as f32 * t;
                    }
                }
            }
            i0 += rows;
        }
    }
}

/// The portable tiles; the SIMD tiles compute the same integers and the same float steps in the same order.
mod portable {
    use super::{COLS, KEYS, LEVELS, exp_nonpositive, round_even_small, row_max};

    /// Scores of 4 query rows (`index`) against keys `[j0, j0 + KEYS)`: lanes over keys, so it vectorizes.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn scores(qi: &[i16], d_h: usize, index: [usize; 4], kt: &[i16], n_pad: usize, j0: usize, x: [f32; 4], y: &[f32], out: &mut [f32]) {
        let mut acc = [[0i32; KEYS]; 4];
        let queries = index.map(|i| &qi[i * d_h..(i + 1) * d_h]);
        for p in 0..d_h / 2 {
            let keys = &kt[(p * n_pad + j0) * 2..(p * n_pad + j0 + KEYS) * 2];
            for (sums, query) in acc.iter_mut().zip(&queries) {
                let (a, b) = (query[2 * p] as i32, query[2 * p + 1] as i32);
                for (sum, pair) in sums.iter_mut().zip(keys.chunks_exact(2)) {
                    *sum += a * pair[0] as i32 + b * pair[1] as i32;
                }
            }
        }
        for (r, sums) in acc.iter().enumerate() {
            for (l, &sum) in sums.iter().enumerate() {
                out[r * n_pad + j0 + l] = (sum as f32 * x[r]) * y[j0 + l];
            }
        }
    }

    /// `row` to its exponentials and `levels` to the weights `round(E * r)`, zero past the row.
    #[inline(always)]
    pub fn softmax_levels(row: &mut [f32], levels: &mut [u8]) {
        let n = row.len();
        let top = row_max(row);
        let full = n / 8 * 8;
        let mut lanes = [0f64; 4];
        for (j, value) in row[..full].iter_mut().enumerate() {
            *value = exp_nonpositive(*value - top);
            lanes[j % 4] += *value as f64;
        }
        let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for value in &mut row[full..] {
            *value = exp_nonpositive(*value - top);
            total += *value as f64;
        }
        let r = (LEVELS as f64 / total) as f32;
        for (level, &value) in levels.iter_mut().zip(row.iter()) {
            // SAFETY: `value * r` lies in [0, 127] (to rounding), so its rounding is an integer in [0, 127].
            *level = unsafe { round_even_small(value * r).to_int_unchecked::<i32>() } as u8;
        }
        levels[n..].fill(0);
    }

    /// `acc[r][l] = sum_j weights[r][j] * v[j][c0 + l]`: lanes over columns, so it vectorizes.
    #[inline(always)]
    pub fn weighted(weights: &[u8], n_pad: usize, vt: &[i8], dh_pad: usize, c0: usize, acc: &mut [[i32; COLS]; 4]) {
        for p in 0..n_pad / 2 {
            let values = &vt[(p * dh_pad + c0) * 2..(p * dh_pad + c0 + COLS) * 2];
            for (r, sums) in acc.iter_mut().enumerate() {
                let (a, b) = (weights[r * n_pad + 2 * p] as i32, weights[r * n_pad + 2 * p + 1] as i32);
                for (sum, pair) in sums.iter_mut().zip(values.chunks_exact(2)) {
                    *sum += a * pair[0] as i32 + b * pair[1] as i32;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{COLS, KEYS, LEVELS, SPAN, exp_nonpositive, round_even_small};
    use crate::dense::{exp8, scale_and_max};

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scores(qi: &[i16], d_h: usize, index: [usize; 4], kt: &[i16], n_pad: usize, j0: usize, x: [f32; 4], y: &[f32], out: &mut [f32]) {
        let half = d_h / 2;
        assert!(index.iter().all(|&i| (i + 1) * d_h <= qi.len()));
        assert!(half == 0 || ((half - 1) * n_pad + j0 + KEYS) * 2 <= kt.len());
        assert!(j0 + KEYS <= y.len() && 3 * n_pad + j0 + KEYS <= out.len());
        let zero = _mm256_setzero_si256();
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (zero, zero, zero, zero, zero, zero, zero, zero);
        let (queries, keys) = (qi.as_ptr(), kt.as_ptr());
        let pair = |i: usize, p: usize| -> __m256i {
            // SAFETY: row `i` of `qi` holds `d_h` values and `2 p + 1 < d_h` (asserted above).
            unsafe { _mm256_set1_epi32((queries.add(i * d_h + 2 * p) as *const i32).read_unaligned()) }
        };
        for p in 0..half {
            // SAFETY: bounds asserted above.
            unsafe {
                let base = keys.add((p * n_pad + j0) * 2);
                let (k0, k1) = (_mm256_loadu_si256(base as *const __m256i), _mm256_loadu_si256(base.add(16) as *const __m256i));
                let (b0, b1, b2, b3) = (pair(index[0], p), pair(index[1], p), pair(index[2], p), pair(index[3], p));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(b0, k0));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(b0, k1));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(b1, k0));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(b1, k1));
                a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(b2, k0));
                a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(b2, k1));
                a6 = _mm256_add_epi32(a6, _mm256_madd_epi16(b3, k0));
                a7 = _mm256_add_epi32(a7, _mm256_madd_epi16(b3, k1));
            }
        }
        // SAFETY: bounds asserted above.
        unsafe {
            let (y0, y1) = (_mm256_loadu_ps(y.as_ptr().add(j0)), _mm256_loadu_ps(y.as_ptr().add(j0 + 8)));
            let target = out.as_mut_ptr();
            for (r, (low, high)) in [(a0, a1), (a2, a3), (a4, a5), (a6, a7)].into_iter().enumerate() {
                let factor = _mm256_set1_ps(x[r]);
                _mm256_storeu_ps(target.add(r * n_pad + j0), _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(low), factor), y0));
                _mm256_storeu_ps(target.add(r * n_pad + j0 + 8), _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(high), factor), y1));
            }
        }
    }

    /// `portable::softmax_levels`, eight lanes at a time: the same exponentials, sums and roundings.
    #[target_feature(enable = "avx2")]
    pub unsafe fn softmax_levels(row: &mut [f32], levels: &mut [u8]) {
        let n = row.len();
        assert!(levels.len() >= n);
        // SAFETY: AVX2 is present (this function requires it).
        let top = unsafe { scale_and_max(row, None) };
        let full = n / 8 * 8;
        let pointer = row.as_mut_ptr();
        let shift = _mm256_set1_ps(top);
        let mut lanes = _mm256_setzero_pd();
        for at in (0..full).step_by(8) {
            // SAFETY: `at + 8 <= n`.
            unsafe {
                let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(pointer.add(at)), shift));
                _mm256_storeu_ps(pointer.add(at), e);
                lanes = _mm256_add_pd(lanes, _mm256_cvtps_pd(_mm256_castps256_ps128(e)));
                lanes = _mm256_add_pd(lanes, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(e)));
            }
        }
        let mut l = [0f64; 4];
        // SAFETY: `l` holds four doubles.
        unsafe { _mm256_storeu_pd(l.as_mut_ptr(), lanes) };
        let mut total = (l[0] + l[1]) + (l[2] + l[3]);
        for value in &mut row[full..] {
            *value = exp_nonpositive(*value - top);
            total += *value as f64;
        }
        let r = (LEVELS as f64 / total) as f32;
        let (factor, magic) = (_mm256_set1_ps(r), _mm256_set1_ps(12_582_912.0));
        let target = levels.as_mut_ptr();
        for at in (0..full).step_by(8) {
            // SAFETY: `at + 8 <= n <= levels.len()`.
            unsafe {
                let scaled = _mm256_mul_ps(_mm256_loadu_ps(pointer.add(at)), factor);
                let integers = _mm256_cvttps_epi32(_mm256_sub_ps(_mm256_add_ps(scaled, magic), magic));
                let shorts = _mm256_packs_epi32(integers, integers);
                let bytes = _mm256_packus_epi16(shorts, shorts);
                (target.add(at) as *mut i32).write_unaligned(_mm256_cvtsi256_si32(bytes));
                (target.add(at + 4) as *mut i32).write_unaligned(_mm256_extract_epi32::<4>(bytes));
            }
        }
        for j in full..n {
            // SAFETY: as in `portable::softmax_levels`.
            levels[j] = unsafe { round_even_small(row[j] * r).to_int_unchecked::<i32>() } as u8;
        }
        levels[n..].fill(0);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn weighted(weights: &[u8], n_pad: usize, vt: &[i8], dh_pad: usize, c0: usize, acc: &mut [[i32; COLS]; 4]) {
        let pairs = n_pad / 2;
        assert!(weights.len() >= 4 * n_pad && (pairs == 0 || ((pairs - 1) * dh_pad + c0 + COLS) * 2 <= vt.len()));
        let (values, levels) = (vt.as_ptr(), weights.as_ptr());
        let pair = |r: usize, p: usize| -> __m256i {
            // SAFETY: row `r` of `weights` holds `n_pad` bytes and `2 p + 1 < n_pad`.
            unsafe { _mm256_set1_epi16((levels.add(r * n_pad + 2 * p) as *const i16).read_unaligned()) }
        };
        let mut p0 = 0;
        while p0 < pairs {
            let p1 = (p0 + SPAN / 2).min(pairs);
            let zero = _mm256_setzero_si256();
            let (mut c0r, mut c1r, mut c2r, mut c3r, mut c4r, mut c5r, mut c6r, mut c7r) = (zero, zero, zero, zero, zero, zero, zero, zero);
            for p in p0..p1 {
                // SAFETY: bounds asserted above.
                unsafe {
                    let base = values.add((p * dh_pad + c0) * 2);
                    let (v0, v1) = (_mm256_loadu_si256(base as *const __m256i), _mm256_loadu_si256(base.add(32) as *const __m256i));
                    let (b0, b1, b2, b3) = (pair(0, p), pair(1, p), pair(2, p), pair(3, p));
                    c0r = _mm256_add_epi16(c0r, _mm256_maddubs_epi16(b0, v0));
                    c1r = _mm256_add_epi16(c1r, _mm256_maddubs_epi16(b0, v1));
                    c2r = _mm256_add_epi16(c2r, _mm256_maddubs_epi16(b1, v0));
                    c3r = _mm256_add_epi16(c3r, _mm256_maddubs_epi16(b1, v1));
                    c4r = _mm256_add_epi16(c4r, _mm256_maddubs_epi16(b2, v0));
                    c5r = _mm256_add_epi16(c5r, _mm256_maddubs_epi16(b2, v1));
                    c6r = _mm256_add_epi16(c6r, _mm256_maddubs_epi16(b3, v0));
                    c7r = _mm256_add_epi16(c7r, _mm256_maddubs_epi16(b3, v1));
                }
            }
            for (r, (low, high)) in [(c0r, c1r), (c2r, c3r), (c4r, c5r), (c6r, c7r)].into_iter().enumerate() {
                for (half, sums) in [low, high].into_iter().enumerate() {
                    // SAFETY: each row of `acc` holds COLS = 32 sums; `half * 16 + 16 <= 32`.
                    unsafe {
                        let at = acc[r].as_mut_ptr().add(half * 16) as *mut __m256i;
                        let first = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sums));
                        let second = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(sums));
                        _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), first));
                        _mm256_storeu_si256(at.add(1), _mm256_add_epi32(_mm256_loadu_si256(at.add(1)), second));
                    }
                }
            }
            p0 = p1;
        }
    }
}

/// `block_attention` on the portable path whatever the processor (tests compare every backend with it bit for bit).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn block_attention_portable(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], n: usize, d: usize, heads: usize, eps: f32, scratch: &mut IntegerScratch) {
    body(q, k, v, o, n, d, heads, eps, scratch, Path::Portable);
}
