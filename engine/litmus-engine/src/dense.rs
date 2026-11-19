//! The small float pieces of the encoder (SPEC.md, sections 2 and 3): RMS norm, 8-bit quantization, rotary
//! positions, block attention, the exponential and half-precision widening.
//!
//! Loops run over fixed lanes of 8 so the compiler vectorizes them on every target with the same order of
//! operations; no multiply-add is fused, so scalar and vector builds give the same bits.

use crate::kernels::Backend;

/// `round(x)` half to even for `|x| < 2^22`: adding and removing `1.5 * 2^23` leaves the nearest integer, ties to
/// even, in the default rounding mode. Plain additions, so it vectorizes on every target.
#[inline(always)]
pub fn round_even_small(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    (x + MAGIC) - MAGIC
}

/// The largest `|x|`.
#[inline]
pub fn max_abs(x: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let mut chunks = x.chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            let magnitude = value.abs();
            *lane = if magnitude > *lane { magnitude } else { *lane };
        }
    }
    let mut m = lanes.iter().fold(0f32, |m, &value| if value > m { value } else { m });
    for &value in chunks.remainder() {
        m = if value.abs() > m { value.abs() } else { m };
    }
    m
}

/// Per-token absmax 8-bit quantization of one row (SPEC.md, section 2): returns `s` and `sum q`.
#[inline]
pub fn quantize_row(x: &[f32], q: &mut [i8], eps: f32) -> (f32, i32) {
    let m = max_abs(x).max(eps);
    let s = 127.0f32 / m;
    let mut sum = 0i32;
    for (out, &value) in q.iter_mut().zip(x) {
        // |value * s| <= 127 (to rounding), inside the magic-number range.
        let rounded = round_even_small(value * s).clamp(-128.0, 127.0);
        // SAFETY: `rounded` is an integer in [-128, 127].
        let integer = unsafe { rounded.to_int_unchecked::<i32>() };
        *out = integer as i8;
        sum += integer;
    }
    (s, sum)
}

/// RMS norm with a gain (SPEC.md, section 2): `r = 1 / sqrt(mean(x^2) + eps)`, `y = (x * r) * g`. The squares are
/// float32 as in the reference and are summed in float64, then the mean is rounded to float32.
#[inline]
pub fn rms_norm(x: &[f32], gain: &[f32], eps: f32, y: &mut [f32]) {
    debug_assert!(x.len() == gain.len() && x.len() == y.len());
    let mut lanes = [0f64; 4];
    let mut chunks = x.chunks_exact(4);
    for chunk in &mut chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane += (value * value) as f64;
        }
    }
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for &value in chunks.remainder() {
        total += (value * value) as f64;
    }
    let mean = (total / x.len() as f64) as f32;
    let r = 1.0f32 / (mean + eps).sqrt();
    for ((out, &value), &g) in y.iter_mut().zip(x).zip(gain) {
        *out = (value * r) * g;
    }
}

/// Cosine and sine tables of the rotary angles `r * base^(-2j / d_h)`, computed in float64, stored float32:
/// `(W, d_h / 2)` each.
pub fn rope_tables(block_size: usize, head_width: usize, base: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_width / 2;
    let frequencies: Vec<f64> = (0..half).map(|j| base.powf(-2.0 * j as f64 / head_width as f64)).collect();
    let mut cosine = Vec::with_capacity(block_size * half);
    let mut sine = Vec::with_capacity(block_size * half);
    for r in 0..block_size {
        for &frequency in &frequencies {
            let angle = r as f64 * frequency;
            cosine.push(angle.cos() as f32);
            sine.push(angle.sin() as f32);
        }
    }
    (cosine, sine)
}

/// Rotate the halves of one head's vector `u` (width `d_h`) by one row of the tables.
#[inline]
pub fn rope(u: &mut [f32], cosine: &[f32], sine: &[f32]) {
    let half = u.len() / 2;
    let (first, second) = u.split_at_mut(half);
    for j in 0..half {
        let (a, b, c, s) = (first[j], second[j], cosine[j], sine[j]);
        first[j] = a * c - b * s;
        second[j] = b * c + a * s;
    }
}

/// `exp(x)` in float32 for `x <= 0` (softmax), about one unit in the last place: Cephes `expf`'s range reduction
/// and polynomial, written without branches so it vectorizes.
#[inline(always)]
pub fn exp_nonpositive(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    let x = x.max(-87.0);
    let shifted = x * LOG2E + MAGIC;
    let n = shifted - MAGIC; // round(x / ln 2)
    let exponent = (shifted.to_bits() as i32).wrapping_sub(MAGIC.to_bits() as i32);
    let r = (x - n * C1) - n * C2;
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5.000_000_1e-1;
    p = p * (r * r) + r + 1.0;
    let scale = f32::from_bits(((exponent + 127) << 23) as u32);
    p * scale
}

/// Columns of a score tile and of an output tile.
pub(crate) const TILE: usize = 16;

/// Buffers of one head's attention over one block.
#[derive(Debug, Default)]
pub struct AttentionScratch {
    kt: Vec<f32>,     // (d_h, n_pad): this head's keys, transposed, zero past n
    vh: Vec<f32>,     // (n, dh_pad): this head's values, zero past d_h
    scores: Vec<f32>, // (4, n_pad): four rows of scores, then probabilities
}

impl AttentionScratch {
    pub fn bytes(&self) -> usize {
        4 * (self.kt.capacity() + self.vh.capacity() + self.scores.capacity())
    }
}

/// Bidirectional softmax attention inside one block for every head (SPEC.md, section 3, step 4), float32:
/// `S = (Q_i . K_j) * scale`, `P = softmax_j(S)`, `O_i = sum_j P_ij V_j`. `q`, `k`, `v` and `o` are `(n, d)` with
/// heads side by side. Tiles of 4 rows by 16 columns stay in registers; the backend picks the tiles (AVX2 on
/// x86-64, SIMD128 on WebAssembly, NEON on aarch64, the portable loops for `Scalar`), and every one sums each output
/// lane in the same order, so the results are the same bits.
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
    scratch: &mut AttentionScratch,
) {
    assert!(backend.is_available(), "backend {} is not available on this processor", backend.name());
    match backend {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the backend is available, so the processor has AVX2.
        Backend::Avx2 | Backend::Avx2Sign => unsafe { block_attention_avx2(q, k, v, o, n, d, heads, scratch) },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Backend::Simd128 => block_attention_body(q, k, v, o, n, d, heads, scratch, Tiles::Simd128),
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the backend is available, so the processor has NEON.
        Backend::Neon | Backend::NeonDot => unsafe { block_attention_neon(q, k, v, o, n, d, heads, scratch) },
        _ => block_attention_body(q, k, v, o, n, d, heads, scratch, Tiles::Portable),
    }
}

/// Which tiles run attention's products and softmax.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tiles {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// `block_attention` on the portable tiles whatever the processor (tests compare every backend with it bit for bit).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn block_attention_portable(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], n: usize, d: usize, heads: usize, scratch: &mut AttentionScratch) {
    block_attention_body(q, k, v, o, n, d, heads, scratch, Tiles::Portable);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn block_attention_neon(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], n: usize, d: usize, heads: usize, scratch: &mut AttentionScratch) {
    block_attention_body(q, k, v, o, n, d, heads, scratch, Tiles::Neon);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn block_attention_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    n: usize,
    d: usize,
    heads: usize,
    scratch: &mut AttentionScratch,
) {
    block_attention_body(q, k, v, o, n, d, heads, scratch, Tiles::Avx2);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn block_attention_body(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    n: usize,
    d: usize,
    heads: usize,
    scratch: &mut AttentionScratch,
    tiles: Tiles,
) {
    let d_h = d / heads;
    let scale = (1.0 / (d_h as f64).sqrt()) as f32;
    let n_pad = n.div_ceil(TILE) * TILE;
    let dh_pad = d_h.div_ceil(TILE) * TILE;
    scratch.kt.clear();
    scratch.kt.resize(d_h * n_pad, 0.0);
    scratch.vh.clear();
    scratch.vh.resize(n * dh_pad, 0.0);
    scratch.scores.resize(4 * n_pad, 0.0);
    let AttentionScratch { kt, vh, scores } = scratch;
    for head in 0..heads {
        let column = head * d_h;
        for j in 0..n {
            for c in 0..d_h {
                kt[c * n_pad + j] = k[j * d + column + c];
                vh[j * dh_pad + c] = v[j * d + column + c];
            }
        }
        let mut i0 = 0;
        while i0 < n {
            let rows = (n - i0).min(4);
            // Rows past the block repeat its last row; their results are not stored.
            let row_of = |r: usize| {
                let i = i0 + r.min(rows - 1);
                &q[i * d + column..i * d + column + d_h]
            };
            let query = [row_of(0), row_of(1), row_of(2), row_of(3)];
            for j0 in (0..n_pad).step_by(TILE) {
                match tiles {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 is present; the tile lies inside `kt` and `scores` (sized above).
                    Tiles::Avx2 => unsafe { tiles_avx2::scores(&query, kt, n_pad, j0, d_h, scores) },
                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    Tiles::Simd128 => crate::dense_simd128::scores(&query, kt, n_pad, j0, d_h, scores),
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is present; the tile lies inside `kt` and `scores` (sized above).
                    Tiles::Neon => unsafe { crate::dense_neon::scores(&query, kt, n_pad, j0, d_h, scores) },
                    Tiles::Portable => tiles::scores(&query, kt, n_pad, j0, d_h, scores),
                }
            }
            for r in 0..rows {
                let row = &mut scores[r * n_pad..r * n_pad + n];
                match tiles {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 is present.
                    Tiles::Avx2 => unsafe { tiles_avx2::softmax(row, scale) },
                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    Tiles::Simd128 => crate::dense_simd128::softmax(row, scale),
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is present.
                    Tiles::Neon => unsafe { crate::dense_neon::softmax(row, scale) },
                    Tiles::Portable => softmax_scaled(row, scale),
                }
            }
            for c0 in (0..dh_pad).step_by(TILE) {
                let mut acc = [[0f32; TILE]; 4];
                match tiles {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 is present; rows j < n of `vh` and of the four score rows are in range.
                    Tiles::Avx2 => unsafe { tiles_avx2::weighted(scores, n_pad, vh, dh_pad, c0, n, &mut acc) },
                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    Tiles::Simd128 => crate::dense_simd128::weighted(scores, n_pad, vh, dh_pad, c0, n, &mut acc),
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is present; as for AVX2.
                    Tiles::Neon => unsafe { crate::dense_neon::weighted(scores, n_pad, vh, dh_pad, c0, n, &mut acc) },
                    Tiles::Portable => tiles::weighted(scores, n_pad, vh, dh_pad, c0, n, &mut acc),
                }
                let width = TILE.min(d_h - c0);
                for r in 0..rows {
                    let at = (i0 + r) * d + column + c0;
                    o[at..at + width].copy_from_slice(&acc[r][..width]);
                }
            }
            i0 += rows;
        }
    }
}

/// Portable tiles: each output lane sums its products in order, as the SIMD tiles do.
mod tiles {
    use super::TILE;

    /// Scores of 4 query rows against keys `[j0, j0 + TILE)`: `scores[r][j] = sum_c q_r[c] * kt[c][j]`.
    #[inline(always)]
    pub fn scores(query: &[&[f32]; 4], kt: &[f32], n_pad: usize, j0: usize, d_h: usize, scores: &mut [f32]) {
        let mut acc = [[0f32; TILE]; 4];
        for c in 0..d_h {
            let keys = &kt[c * n_pad + j0..c * n_pad + j0 + TILE];
            for r in 0..4 {
                let weight = query[r][c];
                for l in 0..TILE {
                    acc[r][l] += weight * keys[l];
                }
            }
        }
        for r in 0..4 {
            scores[r * n_pad + j0..r * n_pad + j0 + TILE].copy_from_slice(&acc[r]);
        }
    }

    /// `acc[r][l] = sum_j p_r[j] * vh[j][c0 + l]`.
    #[inline(always)]
    pub fn weighted(scores: &[f32], n_pad: usize, vh: &[f32], dh_pad: usize, c0: usize, n: usize, acc: &mut [[f32; TILE]; 4]) {
        for j in 0..n {
            let values = &vh[j * dh_pad + c0..j * dh_pad + c0 + TILE];
            for r in 0..4 {
                let weight = scores[r * n_pad + j];
                for l in 0..TILE {
                    acc[r][l] += weight * values[l];
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use tiles_avx2::{exp8, scale_and_max};

/// The largest value of a row: eight lanes of maxima (so it vectorizes), the same value as any order.
#[inline]
pub fn row_max(x: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; 8];
    let mut chunks = x.chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane = if value > *lane { value } else { *lane };
        }
    }
    let mut m = lanes.iter().fold(f32::NEG_INFINITY, |m, &value| if value > m { value } else { m });
    for &value in chunks.remainder() {
        m = if value > m { value } else { m };
    }
    m
}

/// AVX2 tiles: the portable tiles' arithmetic, eight lanes at a time, multiply then add (never fused).
#[cfg(target_arch = "x86_64")]
mod tiles_avx2 {
    use std::arch::x86_64::*;

    use super::TILE;

    #[target_feature(enable = "avx2")]
    pub unsafe fn scores(query: &[&[f32]; 4], kt: &[f32], n_pad: usize, j0: usize, d_h: usize, scores: &mut [f32]) {
        assert!(kt.len() >= (d_h - 1) * n_pad + j0 + TILE && scores.len() >= 3 * n_pad + j0 + TILE);
        assert!(query.iter().all(|row| row.len() >= d_h));
        let zero = _mm256_setzero_ps();
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (zero, zero, zero, zero, zero, zero, zero, zero);
        let keys = kt.as_ptr();
        for c in 0..d_h {
            // SAFETY: bounds asserted above.
            unsafe {
                let base = keys.add(c * n_pad + j0);
                let (k0, k1) = (_mm256_loadu_ps(base), _mm256_loadu_ps(base.add(8)));
                let w0 = _mm256_set1_ps(*query[0].get_unchecked(c));
                let w1 = _mm256_set1_ps(*query[1].get_unchecked(c));
                let w2 = _mm256_set1_ps(*query[2].get_unchecked(c));
                let w3 = _mm256_set1_ps(*query[3].get_unchecked(c));
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(w0, k0));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(w0, k1));
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(w1, k0));
                a3 = _mm256_add_ps(a3, _mm256_mul_ps(w1, k1));
                a4 = _mm256_add_ps(a4, _mm256_mul_ps(w2, k0));
                a5 = _mm256_add_ps(a5, _mm256_mul_ps(w2, k1));
                a6 = _mm256_add_ps(a6, _mm256_mul_ps(w3, k0));
                a7 = _mm256_add_ps(a7, _mm256_mul_ps(w3, k1));
            }
        }
        let out = scores.as_mut_ptr();
        // SAFETY: bounds asserted above.
        unsafe {
            for (r, (low, high)) in [(a0, a1), (a2, a3), (a4, a5), (a6, a7)].into_iter().enumerate() {
                _mm256_storeu_ps(out.add(r * n_pad + j0), low);
                _mm256_storeu_ps(out.add(r * n_pad + j0 + 8), high);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn weighted(scores: &[f32], n_pad: usize, vh: &[f32], dh_pad: usize, c0: usize, n: usize, acc: &mut [[f32; TILE]; 4]) {
        assert!(n == 0 || (vh.len() >= (n - 1) * dh_pad + c0 + TILE && scores.len() >= 3 * n_pad + n));
        let zero = _mm256_setzero_ps();
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (zero, zero, zero, zero, zero, zero, zero, zero);
        let (values, weights) = (vh.as_ptr(), scores.as_ptr());
        for j in 0..n {
            // SAFETY: bounds asserted above.
            unsafe {
                let base = values.add(j * dh_pad + c0);
                let (v0, v1) = (_mm256_loadu_ps(base), _mm256_loadu_ps(base.add(8)));
                let w0 = _mm256_set1_ps(*weights.add(j));
                let w1 = _mm256_set1_ps(*weights.add(n_pad + j));
                let w2 = _mm256_set1_ps(*weights.add(2 * n_pad + j));
                let w3 = _mm256_set1_ps(*weights.add(3 * n_pad + j));
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(w0, v0));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(w0, v1));
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(w1, v0));
                a3 = _mm256_add_ps(a3, _mm256_mul_ps(w1, v1));
                a4 = _mm256_add_ps(a4, _mm256_mul_ps(w2, v0));
                a5 = _mm256_add_ps(a5, _mm256_mul_ps(w2, v1));
                a6 = _mm256_add_ps(a6, _mm256_mul_ps(w3, v0));
                a7 = _mm256_add_ps(a7, _mm256_mul_ps(w3, v1));
            }
        }
        for (r, (low, high)) in [(a0, a1), (a2, a3), (a4, a5), (a6, a7)].into_iter().enumerate() {
            // SAFETY: each row of `acc` holds 16 floats.
            unsafe {
                _mm256_storeu_ps(acc[r].as_mut_ptr(), low);
                _mm256_storeu_ps(acc[r].as_mut_ptr().add(8), high);
            }
        }
    }

    /// `exp_nonpositive` on eight lanes, the same operations in the same order.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn exp8(x: __m256) -> __m256 {
        let magic = _mm256_set1_ps(12_582_912.0);
        let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
        let shifted = _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)), magic);
        let n = _mm256_sub_ps(shifted, magic);
        let exponent = _mm256_sub_epi32(_mm256_castps_si256(shifted), _mm256_castps_si256(magic));
        let r = _mm256_sub_ps(
            _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.693_359_4))),
            _mm256_mul_ps(n, _mm256_set1_ps(-2.121_944_4e-4)),
        );
        let mut p = _mm256_set1_ps(1.987_569_1e-4);
        for coefficient in [1.398_199_9e-3f32, 8.333_452e-3, 4.166_579_6e-2, 1.666_666_5e-1, 5.000_000_1e-1] {
            p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(coefficient));
        }
        p = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(p, _mm256_mul_ps(r, r)), r), _mm256_set1_ps(1.0));
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23));
        _mm256_mul_ps(p, scale)
    }

    /// Multiply `s` by `scale` in place, when given, and return its largest value: eight lanes of maxima, which give
    /// the scalar loop's value since a maximum is exact in any order.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn scale_and_max(s: &mut [f32], scale: Option<f32>) -> f32 {
        let full = s.len() / 8 * 8;
        let pointer = s.as_mut_ptr();
        let mut top = _mm256_set1_ps(f32::NEG_INFINITY);
        for at in (0..full).step_by(8) {
            // SAFETY: `at + 8 <= s.len()`.
            unsafe {
                let mut value = _mm256_loadu_ps(pointer.add(at));
                if let Some(scale) = scale {
                    value = _mm256_mul_ps(value, _mm256_set1_ps(scale));
                    _mm256_storeu_ps(pointer.add(at), value);
                }
                top = _mm256_max_ps(top, value);
            }
        }
        let mut lanes = [0f32; 8];
        // SAFETY: `lanes` holds eight floats.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), top) };
        let mut maximum = lanes.iter().fold(f32::NEG_INFINITY, |m, &value| if value > m { value } else { m });
        for value in &mut s[full..] {
            if let Some(scale) = scale {
                *value *= scale;
            }
            maximum = if *value > maximum { *value } else { maximum };
        }
        maximum
    }

    /// `softmax_scaled` with eight lanes; the same sums in the same order.
    #[target_feature(enable = "avx2")]
    pub unsafe fn softmax(s: &mut [f32], scale: f32) {
        // SAFETY: AVX2 is present (this function requires it).
        let maximum = unsafe { scale_and_max(s, Some(scale)) };
        let top = _mm256_set1_ps(maximum);
        let mut lanes = _mm256_setzero_ps();
        let full = s.len() / 8 * 8;
        let pointer = s.as_mut_ptr();
        for at in (0..full).step_by(8) {
            // SAFETY: `at + 8 <= s.len()`.
            unsafe {
                let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(pointer.add(at)), top));
                _mm256_storeu_ps(pointer.add(at), e);
                lanes = _mm256_add_ps(lanes, e);
            }
        }
        let mut l = [0f32; 8];
        // SAFETY: `l` holds eight floats.
        unsafe { _mm256_storeu_ps(l.as_mut_ptr(), lanes) };
        let mut total = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
        for value in &mut s[full..] {
            *value = super::exp_nonpositive(*value - maximum);
            total += *value;
        }
        let divisor = _mm256_set1_ps(total);
        for at in (0..full).step_by(8) {
            // SAFETY: `at + 8 <= s.len()`.
            unsafe { _mm256_storeu_ps(pointer.add(at), _mm256_div_ps(_mm256_loadu_ps(pointer.add(at)), divisor)) };
        }
        for value in &mut s[full..] {
            *value /= total;
        }
    }
}

/// `s = softmax(s * scale)` in place, with the reference's order: scale, subtract the maximum, exponentiate, divide
/// by the sum.
#[inline]
fn softmax_scaled(s: &mut [f32], scale: f32) {
    for value in s.iter_mut() {
        *value *= scale;
    }
    let maximum = row_max(s);
    let mut lanes = [0f32; 8];
    let mut chunks = s.chunks_exact_mut(8);
    for chunk in &mut chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk.iter_mut()) {
            *value = exp_nonpositive(*value - maximum);
            *lane += *value;
        }
    }
    let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for value in chunks.into_remainder() {
        *value = exp_nonpositive(*value - maximum);
        total += *value;
    }
    for value in s.iter_mut() {
        *value /= total;
    }
}

/// `out[i * rows_b + j] = sum_c a[i, c] * b[j, c]` in float64, for `a` `(rows_a, k)` float32 and `b` `(rows_b, k)`
/// float64: the head's two layers and the bank product, batched over a chunk's windows so each weight row is read
/// once. Every path sums in the same order (four lanes over `c mod 4`, then `(l0 + l1) + (l2 + l3)`, then the tail),
/// with no fused multiply-add, so the answer is the same on every processor.
pub fn dots_f64(a: &[f32], b: &[f64], k: usize, out: &mut [f64]) {
    assert!(k > 0 && a.len() % k == 0 && b.len() % k == 0, "rows of k values");
    let (rows_a, rows_b) = (a.len() / k, b.len() / k);
    assert_eq!(out.len(), rows_a * rows_b, "one output per pair of rows");
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2; shapes are checked above.
        unsafe { dots_f64_avx2(a, b, k, out) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if rows_a > 0 {
        crate::dense_simd128::dots_f64(a, b, k, out);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: the processor has NEON; shapes are checked above.
        unsafe { crate::dense_neon::dots_f64(a, b, k, out) };
        return;
    }
    dots_f64_portable(a, b, k, out);
}

/// `dots_f64` on the portable loops whatever the processor (tests compare every path with it bit for bit).
#[doc(hidden)]
pub fn dots_f64_portable(a: &[f32], b: &[f64], k: usize, out: &mut [f64]) {
    let rows_b = b.len() / k;
    for i in 0..a.len() / k {
        let row = &a[i * k..(i + 1) * k];
        for j in 0..rows_b {
            out[i * rows_b + j] = dot_lanes(row, &b[j * k..(j + 1) * k]);
        }
    }
}

#[inline]
fn dot_lanes(a: &[f32], b: &[f64]) -> f64 {
    let mut lanes = [0f64; 4];
    let full = a.len() / 4 * 4;
    for c in (0..full).step_by(4) {
        for lane in 0..4 {
            lanes[lane] += a[c + lane] as f64 * b[c + lane];
        }
    }
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for c in full..a.len() {
        total += a[c] as f64 * b[c];
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dots_f64_avx2(a: &[f32], b: &[f64], k: usize, out: &mut [f64]) {
    use std::arch::x86_64::*;
    let (rows_a, rows_b) = (a.len() / k, b.len() / k);
    let full = k / 4 * 4;
    let reduce = |v: __m256d| -> f64 {
        let mut lanes = [0f64; 4];
        // SAFETY: `lanes` holds four doubles.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), v) };
        (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    };
    let finish = |s: __m256d, i: usize, j: usize, out: &mut [f64]| {
        // The scalar order: the lanes' sum, then each tail product in turn.
        let mut total = reduce(s);
        for c in full..k {
            total += a[i * k + c] as f64 * b[j * k + c];
        }
        out[i * rows_b + j] = total;
    };
    let mut i = 0;
    while i < rows_a {
        // Two rows of a against four of b: eight independent sums, six loads per 32 products.
        let mut first_j = 0;
        if i + 1 < rows_a {
            let (pa0, pa1) = (a[i * k..].as_ptr(), a[(i + 1) * k..].as_ptr());
            while first_j + 4 <= rows_b {
                let pb = b[first_j * k..].as_ptr();
                let mut s = [_mm256_setzero_pd(); 8];
                for c in (0..full).step_by(4) {
                    // SAFETY: rows i, i + 1 of a and first_j..first_j + 4 of b hold k values each and c + 4 <= k.
                    unsafe {
                        let x0 = _mm256_cvtps_pd(_mm_loadu_ps(pa0.add(c)));
                        let x1 = _mm256_cvtps_pd(_mm_loadu_ps(pa1.add(c)));
                        for (r, y) in [pb.add(c), pb.add(k + c), pb.add(2 * k + c), pb.add(3 * k + c)].into_iter().enumerate() {
                            let y = _mm256_loadu_pd(y);
                            s[r] = _mm256_add_pd(s[r], _mm256_mul_pd(x0, y));
                            s[4 + r] = _mm256_add_pd(s[4 + r], _mm256_mul_pd(x1, y));
                        }
                    }
                }
                for r in 0..4 {
                    finish(s[r], i, first_j + r, out);
                    finish(s[4 + r], i + 1, first_j + r, out);
                }
                first_j += 4;
            }
        }
        let rows = if i + 1 < rows_a { 2 } else { 1 };
        for row in i..i + rows {
            let pa = a[row * k..].as_ptr();
            let mut j = if rows == 2 { first_j } else { 0 };
            while j + 4 <= rows_b {
                let pb = b[j * k..].as_ptr();
                let mut s = [_mm256_setzero_pd(); 4];
                for c in (0..full).step_by(4) {
                    // SAFETY: this row of a and rows j..j + 4 of b hold k values each and c + 4 <= k.
                    unsafe {
                        let x = _mm256_cvtps_pd(_mm_loadu_ps(pa.add(c)));
                        for (r, y) in [pb.add(c), pb.add(k + c), pb.add(2 * k + c), pb.add(3 * k + c)].into_iter().enumerate() {
                            s[r] = _mm256_add_pd(s[r], _mm256_mul_pd(x, _mm256_loadu_pd(y)));
                        }
                    }
                }
                for (r, sum) in s.into_iter().enumerate() {
                    finish(sum, row, j + r, out);
                }
                j += 4;
            }
            while j < rows_b {
                let pb = b[j * k..].as_ptr();
                let mut sum = _mm256_setzero_pd();
                for c in (0..full).step_by(4) {
                    // SAFETY: as above, for one row of b.
                    unsafe { sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(pa.add(c))), _mm256_loadu_pd(pb.add(c)))) };
                }
                finish(sum, row, j, out);
                j += 1;
            }
        }
        i += rows;
    }
}

/// IEEE half to single precision, exactly.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exponent = ((h >> 10) & 0x1f) as u32;
    let mantissa = (h as u32) & 0x3ff;
    match exponent {
        0 => {
            let magnitude = mantissa as f32 * (1.0 / 16_777_216.0); // mantissa * 2^-24, exact
            f32::from_bits(sign | magnitude.to_bits())
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_even_matches_std() {
        let mut x = -300.0f32;
        while x <= 300.0 {
            assert_eq!(round_even_small(x), x.round_ties_even(), "at {x}");
            x += 0.125;
        }
        for x in [0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -127.5, 0.49999997, 1e-30] {
            assert_eq!(round_even_small(x), x.round_ties_even(), "at {x}");
        }
    }

    #[test]
    fn exp_is_close() {
        let mut x = 0.0f32;
        while x > -87.0 {
            let (got, want) = (exp_nonpositive(x), (x as f64).exp());
            assert!(((got as f64) - want).abs() <= 4e-7 * want, "exp({x}) = {got}, want {want}");
            x -= 0.0137;
        }
        assert_eq!(exp_nonpositive(0.0), 1.0);
    }

    #[test]
    fn f16_widens_exactly() {
        for h in 0..=u16::MAX {
            let exponent = (h >> 10) & 0x1f;
            let got = f16_to_f32(h);
            if exponent == 31 && (h & 0x3ff) != 0 {
                assert!(got.is_nan());
                continue;
            }
            // Compare with a float64 decoding of the half.
            let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let mantissa = (h & 0x3ff) as f64;
            let want = match exponent {
                0 => sign * mantissa * 2f64.powi(-24),
                31 => sign * f64::INFINITY,
                e => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(e as i32 - 15),
            };
            assert_eq!(got as f64, want, "half {h:#06x}");
        }
    }

    #[test]
    fn dots_are_the_same_on_every_path() {
        for (rows_a, rows_b, k) in [(1, 1, 4), (3, 9, 256), (5, 7, 13), (2, 4, 3), (16, 122, 256)] {
            let a: Vec<f32> = (0..rows_a * k).map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) / 3.0).collect();
            let b: Vec<f64> = (0..rows_b * k).map(|i| ((i * 53 % 97) as f64 / 97.0 - 0.5) / 7.0).collect();
            let mut fast = vec![0.0; rows_a * rows_b];
            dots_f64(&a, &b, k, &mut fast);
            let portable: Vec<f64> =
                (0..rows_a).flat_map(|i| (0..rows_b).map(move |j| (i, j))).map(|(i, j)| dot_lanes(&a[i * k..(i + 1) * k], &b[j * k..(j + 1) * k])).collect();
            assert_eq!(fast, portable, "shape {rows_a} x {rows_b} x {k}");
            let exact: f64 = (0..k).map(|c| a[c] as f64 * b[c]).sum();
            assert!((fast[0] - exact).abs() < 1e-12);
        }
    }

    #[test]
    fn quantize_matches_formula() {
        let x: Vec<f32> = (0..300).map(|i| ((i as f32) * 0.731).sin() * (1.0 + i as f32 * 0.01)).collect();
        let mut q = vec![0i8; x.len()];
        let (s, sum) = quantize_row(&x, &mut q, 1e-5);
        let m = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(s, 127.0 / m);
        for (&value, &quantized) in x.iter().zip(&q) {
            assert_eq!(quantized as f32, (value * s).round_ties_even().clamp(-128.0, 127.0));
        }
        assert_eq!(sum, q.iter().map(|&v| v as i32).sum::<i32>());
    }
}
