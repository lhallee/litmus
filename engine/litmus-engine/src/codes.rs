//! Int8 output (SPEC.md, Int8 output): the codes of the head's query and key, their inverse norms, and the integer
//! dot products of the compare against an int8 bank.
//!
//! Every integer here is exact (a dot product is a sum of at most `w` products of codes of at most 127 in magnitude,
//! under 2^31 for the widths [`crate::release::ReleaseConfig::parse`] allows), so every path returns the same bits and
//! [`dots_i8_portable`] defines the answer. The AVX2 path multiplies with `vpmaddubsw` on the absolute value of one
//! side and the other side's sign-adjusted value, which is exact while no code is -128; the quantizer never writes
//! -128 and the bank's load refuses it. Other processors take the portable loops, which the compiler vectorizes.

use crate::dense::{max_abs, round_even_small};

/// Codes are stored in rows padded with zeros to a multiple of this many bytes, one vector register of the AVX2 path.
pub const LANE: usize = 32;

/// The width of a padded row of `width` codes.
pub fn padded(width: usize) -> usize {
    width.div_ceil(LANE) * LANE
}

/// The int8 codes of one query or key (SPEC.md, Int8 output): `s = 127 / max(max |x|, eps)` in float32 and
/// `code = clamp(round(x * s), -127, 127)`, rounding half to even.
#[inline]
pub fn quantize_output(x: &[f32], eps: f32, codes: &mut [i8]) {
    assert_eq!(x.len(), codes.len(), "one code per component");
    let s = 127.0f32 / max_abs(x).max(eps);
    for (code, &value) in codes.iter_mut().zip(x) {
        // |value * s| <= 127 to rounding, inside the magic-number range; `as` saturates and sends NaN to 0.
        *code = round_even_small(value * s).clamp(-127.0, 127.0) as i32 as i8;
    }
}

/// `sum c^2` of one vector of codes: an exact integer.
pub fn squared_norm(codes: &[i8]) -> i32 {
    codes.iter().map(|&c| c as i32 * c as i32).sum()
}

/// `1 / sqrt(sum c^2)` in float64, the square root and the division each correctly rounded, and 0 for a zero vector.
pub fn inverse_norm(codes: &[i8]) -> f64 {
    match squared_norm(codes) {
        0 => 0.0,
        squares => 1.0 / (squares as f64).sqrt(),
    }
}

/// `out[j * rows_a + i] = sum_c a[i, c] * b[j, c]` for `a` `(rows_a, stride)` and `b` `(rows_b, stride)` rows of int8
/// codes in `[-127, 127]` (padding zero), exact int32 sums.
pub fn dots_i8(a: &[i8], b: &[i8], stride: usize, out: &mut [i32]) {
    assert!(stride > 0 && a.len() % stride == 0 && b.len() % stride == 0, "rows of `stride` codes");
    assert_eq!(out.len(), a.len() / stride * (b.len() / stride), "one sum per pair of rows");
    #[cfg(target_arch = "x86_64")]
    if stride % LANE == 0 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2; the shapes are checked above and `stride` is a multiple of 32.
        unsafe { dots_i8_avx2(a, b, stride, out) };
        return;
    }
    dots_i8_portable(a, b, stride, out);
}

/// `dots_i8` on the portable loops whatever the processor (tests compare every path with it).
#[doc(hidden)]
pub fn dots_i8_portable(a: &[i8], b: &[i8], stride: usize, out: &mut [i32]) {
    let rows_a = a.len() / stride;
    for (j, y) in b.chunks_exact(stride).enumerate() {
        for (i, x) in a.chunks_exact(stride).enumerate() {
            out[j * rows_a + i] = x.iter().zip(y).map(|(&p, &q)| p as i32 * q as i32).sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dots_i8_avx2(a: &[i8], b: &[i8], stride: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let (rows_a, rows_b, chunks) = (a.len() / stride, b.len() / stride, stride / LANE);
    let ones = _mm256_set1_epi16(1);
    // |x| times y with the sign of x is x times y, and `vpmaddubsw` adds adjacent pairs of such products into 16 bits
    // (at most 2 * 127 * 127), which `vpmaddwd` against ones widens to 32.
    let step = |acc: __m256i, x: __m256i, y: __m256i| -> __m256i {
        _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(y, x)), ones))
    };
    let total = |v: __m256i| -> i32 {
        let mut lanes = [0i32; 8];
        // SAFETY: `lanes` holds eight 32-bit integers.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v) };
        lanes.iter().sum()
    };
    for j in 0..rows_b {
        let pb = b[j * stride..].as_ptr();
        let mut i = 0;
        // Four rows of `a` against one of `b`: four accumulators, reduced together by horizontal adds.
        while i + 4 <= rows_a {
            let pa = a[i * stride..].as_ptr();
            let mut acc = [_mm256_setzero_si256(); 4];
            for c in 0..chunks {
                // SAFETY: rows i..i + 4 of `a` and row j of `b` hold `stride` codes each and 32 * (c + 1) <= stride.
                unsafe {
                    let y = _mm256_loadu_si256(pb.add(LANE * c) as *const __m256i);
                    for (r, sum) in acc.iter_mut().enumerate() {
                        *sum = step(*sum, _mm256_loadu_si256(pa.add(r * stride + LANE * c) as *const __m256i), y);
                    }
                }
            }
            let pairs = _mm256_hadd_epi32(_mm256_hadd_epi32(acc[0], acc[1]), _mm256_hadd_epi32(acc[2], acc[3]));
            let sums = _mm_add_epi32(_mm256_castsi256_si128(pairs), _mm256_extracti128_si256(pairs, 1));
            // SAFETY: out holds rows_a * rows_b sums and j * rows_a + i + 4 <= rows_a * rows_b.
            unsafe { _mm_storeu_si128(out.as_mut_ptr().add(j * rows_a + i) as *mut __m128i, sums) };
            i += 4;
        }
        while i < rows_a {
            let pa = a[i * stride..].as_ptr();
            let mut acc = _mm256_setzero_si256();
            for c in 0..chunks {
                // SAFETY: row i of `a` and row j of `b` hold `stride` codes each and 32 * (c + 1) <= stride.
                unsafe { acc = step(acc, _mm256_loadu_si256(pa.add(LANE * c) as *const __m256i), _mm256_loadu_si256(pb.add(LANE * c) as *const __m256i)) };
            }
            out[j * rows_a + i] = total(acc);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_matches_the_formula() {
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.731).sin() * (1.0 + i as f32 * 0.05)).collect();
        let mut codes = vec![0i8; x.len()];
        quantize_output(&x, 1e-5, &mut codes);
        let m = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = 127.0f32 / m;
        for (&value, &code) in x.iter().zip(&codes) {
            assert_eq!(code as f32, (value * s).round_ties_even().clamp(-127.0, 127.0));
        }
        assert_eq!(codes.iter().map(|&c| c.unsigned_abs()).max(), Some(127));
        // Ties go to the even code at a scale of exactly one; the zero vector has zero codes and a zero inverse norm.
        let mut ties = [0i8; 6];
        quantize_output(&[127.0, 0.5, 1.5, 2.5, -0.5, -1.5], 1e-5, &mut ties);
        assert_eq!(ties, [127, 0, 2, 2, 0, -2]);
        quantize_output(&[0.0; 8], 1e-5, &mut codes[..8]);
        assert!(codes[..8].iter().all(|&c| c == 0));
        assert_eq!(inverse_norm(&codes[..8]), 0.0);
        assert_eq!(inverse_norm(&[3, 4, 0]), 1.0 / 5.0);
    }

    #[test]
    fn dots_agree_with_the_definition_on_every_shape() {
        for (rows_a, rows_b, stride) in [(1, 1, 32), (4, 2, 64), (7, 5, 64), (13, 3, 96), (9, 2, 192), (3, 4, 40)] {
            let a: Vec<i8> = (0..rows_a * stride).map(|i| ((i * 37 + 11) % 255) as i32 as i8).map(|v| v.max(-127)).collect();
            let b: Vec<i8> = (0..rows_b * stride).map(|i| ((i * 53 + 29) % 255) as i32 as i8).map(|v| v.max(-127)).collect();
            let (mut fast, mut portable) = (vec![0i32; rows_a * rows_b], vec![0i32; rows_a * rows_b]);
            dots_i8(&a, &b, stride, &mut fast);
            dots_i8_portable(&a, &b, stride, &mut portable);
            assert_eq!(fast, portable, "shape {rows_a} x {rows_b} x {stride}");
            let by_hand: i32 = (0..stride).map(|c| a[c] as i32 * b[c] as i32).sum();
            assert_eq!(fast[0], by_hand);
        }
    }
}
