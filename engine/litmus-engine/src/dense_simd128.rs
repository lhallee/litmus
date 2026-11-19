//! WebAssembly SIMD128 tiles of attention, float (`dense::block_attention`) and integer
//! (`integer_attention::block_attention`): the portable tiles' arithmetic four lanes at a time, each output lane
//! summing its products in the portable order, multiply then add (never fused), so the bits are the portable path's.
//! Tiles of 4 rows by 8 columns keep a pass's accumulators within the 16 registers V8 has on x86-64.

use std::arch::wasm32::*;

use crate::dense::{TILE, exp_nonpositive, round_even_small};
use crate::integer_attention::{COLS, KEYS, LEVELS};

#[inline(always)]
fn splat(value: f32) -> v128 {
    f32x4_splat(value)
}

/// `dense::exp_nonpositive` on four lanes, the same operations in the same order.
#[inline(always)]
pub fn exp4(x: v128) -> v128 {
    let magic = splat(12_582_912.0);
    let x = f32x4_max(x, splat(-87.0));
    let shifted = f32x4_add(f32x4_mul(x, splat(std::f32::consts::LOG2_E)), magic);
    let n = f32x4_sub(shifted, magic);
    let exponent = i32x4_sub(shifted, magic);
    let r = f32x4_sub(f32x4_sub(x, f32x4_mul(n, splat(0.693_359_4))), f32x4_mul(n, splat(-2.121_944_4e-4)));
    let mut p = splat(1.987_569_1e-4);
    for coefficient in [1.398_199_9e-3f32, 8.333_452e-3, 4.166_579_6e-2, 1.666_666_5e-1, 5.000_000_1e-1] {
        p = f32x4_add(f32x4_mul(p, r), splat(coefficient));
    }
    p = f32x4_add(f32x4_add(f32x4_mul(p, f32x4_mul(r, r)), r), splat(1.0));
    f32x4_mul(p, i32x4_shl(i32x4_add(exponent, i32x4_splat(127)), 23))
}

/// Multiply `s` by `scale` in place, when given, and return its largest value (exact in any order).
#[inline(always)]
fn scale_and_max(s: &mut [f32], scale: Option<f32>) -> f32 {
    let full = s.len() / 4 * 4;
    let pointer = s.as_mut_ptr();
    let mut top = splat(f32::NEG_INFINITY);
    for at in (0..full).step_by(4) {
        // SAFETY: `at + 4 <= s.len()`; v128 loads and stores are unaligned.
        unsafe {
            let mut value = v128_load(pointer.add(at) as *const v128);
            if let Some(scale) = scale {
                value = f32x4_mul(value, splat(scale));
                v128_store(pointer.add(at) as *mut v128, value);
            }
            top = f32x4_pmax(top, value);
        }
    }
    let lanes = [f32x4_extract_lane::<0>(top), f32x4_extract_lane::<1>(top), f32x4_extract_lane::<2>(top), f32x4_extract_lane::<3>(top)];
    let mut maximum = lanes.iter().fold(f32::NEG_INFINITY, |m, &value| if value > m { value } else { m });
    for value in &mut s[full..] {
        if let Some(scale) = scale {
            *value *= scale;
        }
        maximum = if *value > maximum { *value } else { maximum };
    }
    maximum
}

// ---------------------------------------------------------------- float attention

/// `dense::tiles::scores`: `scores[r][j] = sum_c q_r[c] * kt[c][j]` for keys `[j0, j0 + TILE)`.
pub fn scores(query: &[&[f32]; 4], kt: &[f32], n_pad: usize, j0: usize, d_h: usize, scores: &mut [f32]) {
    assert!(kt.len() >= (d_h - 1) * n_pad + j0 + TILE && scores.len() >= 3 * n_pad + j0 + TILE);
    assert!(query.iter().all(|row| row.len() >= d_h));
    let (keys, out) = (kt.as_ptr(), scores.as_mut_ptr());
    for half in [0, 8] {
        let mut acc = [[splat(0.0); 2]; 4];
        for c in 0..d_h {
            // SAFETY: bounds asserted above.
            let (k0, k1) = unsafe {
                let base = keys.add(c * n_pad + j0 + half);
                (v128_load(base as *const v128), v128_load(base.add(4) as *const v128))
            };
            for (sums, row) in acc.iter_mut().zip(query) {
                // SAFETY: each query row holds at least d_h values.
                let w = splat(unsafe { *row.get_unchecked(c) });
                sums[0] = f32x4_add(sums[0], f32x4_mul(w, k0));
                sums[1] = f32x4_add(sums[1], f32x4_mul(w, k1));
            }
        }
        for (r, sums) in acc.iter().enumerate() {
            // SAFETY: bounds asserted above.
            unsafe {
                v128_store(out.add(r * n_pad + j0 + half) as *mut v128, sums[0]);
                v128_store(out.add(r * n_pad + j0 + half + 4) as *mut v128, sums[1]);
            }
        }
    }
}

/// `dense::tiles::weighted`: `acc[r][l] = sum_j p_r[j] * vh[j][c0 + l]`.
pub fn weighted(scores: &[f32], n_pad: usize, vh: &[f32], dh_pad: usize, c0: usize, n: usize, acc: &mut [[f32; TILE]; 4]) {
    assert!(n == 0 || (vh.len() >= (n - 1) * dh_pad + c0 + TILE && scores.len() >= 3 * n_pad + n));
    let (values, weights) = (vh.as_ptr(), scores.as_ptr());
    for half in [0, 8] {
        let mut sums = [[splat(0.0); 2]; 4];
        for j in 0..n {
            // SAFETY: bounds asserted above.
            let (v0, v1) = unsafe {
                let base = values.add(j * dh_pad + c0 + half);
                (v128_load(base as *const v128), v128_load(base.add(4) as *const v128))
            };
            for (r, row) in sums.iter_mut().enumerate() {
                // SAFETY: bounds asserted above.
                let w = splat(unsafe { *weights.add(r * n_pad + j) });
                row[0] = f32x4_add(row[0], f32x4_mul(w, v0));
                row[1] = f32x4_add(row[1], f32x4_mul(w, v1));
            }
        }
        for (r, row) in sums.iter().enumerate() {
            // SAFETY: each row of `acc` holds TILE = 16 floats and `half + 8 <= 16`.
            unsafe {
                v128_store(acc[r].as_mut_ptr().add(half) as *mut v128, row[0]);
                v128_store(acc[r].as_mut_ptr().add(half + 4) as *mut v128, row[1]);
            }
        }
    }
}

/// `dense::softmax_scaled` four lanes at a time: two vectors of four give the AVX2 path's eight lane sums.
pub fn softmax(s: &mut [f32], scale: f32) {
    let maximum = scale_and_max(s, Some(scale));
    let top = splat(maximum);
    let (mut low, mut high) = (splat(0.0), splat(0.0));
    let full = s.len() / 8 * 8;
    let pointer = s.as_mut_ptr();
    for at in (0..full).step_by(8) {
        // SAFETY: `at + 8 <= s.len()`.
        unsafe {
            let e0 = exp4(f32x4_sub(v128_load(pointer.add(at) as *const v128), top));
            let e1 = exp4(f32x4_sub(v128_load(pointer.add(at + 4) as *const v128), top));
            v128_store(pointer.add(at) as *mut v128, e0);
            v128_store(pointer.add(at + 4) as *mut v128, e1);
            low = f32x4_add(low, e0);
            high = f32x4_add(high, e1);
        }
    }
    let l = [
        f32x4_extract_lane::<0>(low),
        f32x4_extract_lane::<1>(low),
        f32x4_extract_lane::<2>(low),
        f32x4_extract_lane::<3>(low),
        f32x4_extract_lane::<0>(high),
        f32x4_extract_lane::<1>(high),
        f32x4_extract_lane::<2>(high),
        f32x4_extract_lane::<3>(high),
    ];
    let mut total = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
    for value in &mut s[full..] {
        *value = exp_nonpositive(*value - maximum);
        total += *value;
    }
    let divisor = splat(total);
    let quarters = s.len() / 4 * 4;
    for at in (0..quarters).step_by(4) {
        // SAFETY: `at + 4 <= s.len()`.
        unsafe { v128_store(pointer.add(at) as *mut v128, f32x4_div(v128_load(pointer.add(at) as *const v128), divisor)) };
    }
    for value in &mut s[quarters..] {
        *value /= total;
    }
}

// ---------------------------------------------------------------- float64 products

/// `dense::dots_f64` with two `f64x2` sums per pair of rows, holding the AVX2 path's four lanes (`c mod 4`): the same
/// sums in the same order, then `(l0 + l1) + (l2 + l3)` and the tail. One row of `a` against four of `b` at a time.
pub fn dots_f64(a: &[f32], b: &[f64], k: usize, out: &mut [f64]) {
    let (rows_a, rows_b) = (a.len() / k, b.len() / k);
    let full = k / 4 * 4;
    let finish = |low: v128, high: v128, i: usize, j: usize| -> f64 {
        let mut total = (f64x2_extract_lane::<0>(low) + f64x2_extract_lane::<1>(low)) + (f64x2_extract_lane::<0>(high) + f64x2_extract_lane::<1>(high));
        for c in full..k {
            total += a[i * k + c] as f64 * b[j * k + c];
        }
        total
    };
    for i in 0..rows_a {
        let pa = a[i * k..(i + 1) * k].as_ptr();
        let mut j = 0;
        while j < rows_b {
            let count = (rows_b - j).min(4);
            let mut sums = [[f64x2_splat(0.0); 2]; 4];
            for c in (0..full).step_by(4) {
                // SAFETY: row i of a holds k values and c + 4 <= k.
                let x = unsafe { v128_load(pa.add(c) as *const v128) };
                let (x0, x1) = (f64x2_promote_low_f32x4(x), f64x2_promote_low_f32x4(i64x2_shuffle::<1, 1>(x, x)));
                for (r, pair) in sums.iter_mut().enumerate().take(count) {
                    // SAFETY: row j + r of b holds k values and c + 4 <= k.
                    let (y0, y1) = unsafe {
                        let pb = b.as_ptr().add((j + r) * k + c);
                        (v128_load(pb as *const v128), v128_load(pb.add(2) as *const v128))
                    };
                    pair[0] = f64x2_add(pair[0], f64x2_mul(x0, y0));
                    pair[1] = f64x2_add(pair[1], f64x2_mul(x1, y1));
                }
            }
            for (r, pair) in sums.iter().enumerate().take(count) {
                out[i * rows_b + j + r] = finish(pair[0], pair[1], i, j + r);
            }
            j += count;
        }
    }
}

// ---------------------------------------------------------------- integer attention

/// `integer_attention::portable::scores` with `i32x4.dot_i16x8_s` (two columns of four keys a vector).
#[allow(clippy::too_many_arguments)]
pub fn integer_scores(qi: &[i16], d_h: usize, index: [usize; 4], kt: &[i16], n_pad: usize, j0: usize, x: [f32; 4], y: &[f32], out: &mut [f32]) {
    let half = d_h / 2;
    assert!(index.iter().all(|&i| (i + 1) * d_h <= qi.len()));
    assert!(half == 0 || ((half - 1) * n_pad + j0 + KEYS) * 2 <= kt.len());
    assert!(j0 + KEYS <= y.len() && 3 * n_pad + j0 + KEYS <= out.len());
    let (queries, keys) = (qi.as_ptr(), kt.as_ptr());
    for part in [0, 8] {
        let mut acc = [[i32x4_splat(0); 2]; 4];
        for p in 0..half {
            // SAFETY: bounds asserted above.
            let (k0, k1) = unsafe {
                let base = keys.add((p * n_pad + j0 + part) * 2);
                (v128_load(base as *const v128), v128_load(base.add(8) as *const v128))
            };
            for (sums, &i) in acc.iter_mut().zip(&index) {
                // SAFETY: row `i` of `qi` holds `d_h` values and `2 p + 1 < d_h`.
                let b = i32x4_splat(unsafe { (queries.add(i * d_h + 2 * p) as *const i32).read_unaligned() });
                sums[0] = i32x4_add(sums[0], i32x4_dot_i16x8(b, k0));
                sums[1] = i32x4_add(sums[1], i32x4_dot_i16x8(b, k1));
            }
        }
        // SAFETY: bounds asserted above.
        unsafe {
            let (y0, y1) = (v128_load(y.as_ptr().add(j0 + part) as *const v128), v128_load(y.as_ptr().add(j0 + part + 4) as *const v128));
            for (r, sums) in acc.iter().enumerate() {
                let factor = splat(x[r]);
                let target = out.as_mut_ptr().add(r * n_pad + j0 + part);
                v128_store(target as *mut v128, f32x4_mul(f32x4_mul(f32x4_convert_i32x4(sums[0]), factor), y0));
                v128_store(target.add(4) as *mut v128, f32x4_mul(f32x4_mul(f32x4_convert_i32x4(sums[1]), factor), y1));
            }
        }
    }
}

/// `integer_attention::portable::softmax_levels` four lanes at a time, its float64 lane sums included.
pub fn softmax_levels(row: &mut [f32], levels: &mut [u8]) {
    let n = row.len();
    assert!(levels.len() >= n);
    let top = scale_and_max(row, None);
    let shift = splat(top);
    let full = n / 8 * 8;
    let pointer = row.as_mut_ptr();
    let (mut first, mut second) = (f64x2_splat(0.0), f64x2_splat(0.0)); // lanes 0, 1 and lanes 2, 3 of j mod 4
    for at in (0..full).step_by(8) {
        // SAFETY: `at + 8 <= n`.
        unsafe {
            let e0 = exp4(f32x4_sub(v128_load(pointer.add(at) as *const v128), shift));
            let e1 = exp4(f32x4_sub(v128_load(pointer.add(at + 4) as *const v128), shift));
            v128_store(pointer.add(at) as *mut v128, e0);
            v128_store(pointer.add(at + 4) as *mut v128, e1);
            first = f64x2_add(first, f64x2_promote_low_f32x4(e0));
            second = f64x2_add(second, f64x2_promote_low_f32x4(i64x2_shuffle::<1, 1>(e0, e0)));
            first = f64x2_add(first, f64x2_promote_low_f32x4(e1));
            second = f64x2_add(second, f64x2_promote_low_f32x4(i64x2_shuffle::<1, 1>(e1, e1)));
        }
    }
    let l = [f64x2_extract_lane::<0>(first), f64x2_extract_lane::<1>(first), f64x2_extract_lane::<0>(second), f64x2_extract_lane::<1>(second)];
    let mut total = (l[0] + l[1]) + (l[2] + l[3]);
    for value in &mut row[full..] {
        *value = exp_nonpositive(*value - top);
        total += *value as f64;
    }
    let r = (LEVELS as f64 / total) as f32;
    let (factor, magic) = (splat(r), splat(12_582_912.0));
    let target = levels.as_mut_ptr();
    for at in (0..full).step_by(8) {
        // SAFETY: `at + 8 <= n <= levels.len()`.
        unsafe {
            let round = |at: usize| {
                let scaled = f32x4_mul(v128_load(pointer.add(at) as *const v128), factor);
                i32x4_trunc_sat_f32x4(f32x4_sub(f32x4_add(scaled, magic), magic))
            };
            let shorts = i16x8_narrow_i32x4(round(at), round(at + 4));
            let bytes = u8x16_narrow_i16x8(shorts, shorts);
            (target.add(at) as *mut i64).write_unaligned(i64x2_extract_lane::<0>(bytes));
        }
    }
    for j in full..n {
        // SAFETY: as in `integer_attention::portable::softmax_levels`.
        levels[j] = unsafe { round_even_small(row[j] * r).to_int_unchecked::<i32>() } as u8;
    }
    levels[n..].fill(0);
}

/// `integer_attention::portable::weighted`: weights and values widened to 16 bits, `i32x4.dot_i16x8_s` over two
/// tokens of four columns a vector.
pub fn integer_weighted(weights: &[u8], n_pad: usize, vt: &[i8], dh_pad: usize, c0: usize, acc: &mut [[i32; COLS]; 4]) {
    let pairs = n_pad / 2;
    assert!(weights.len() >= 4 * n_pad && (pairs == 0 || ((pairs - 1) * dh_pad + c0 + COLS) * 2 <= vt.len()));
    let (values, levels) = (vt.as_ptr(), weights.as_ptr());
    for part in (0..COLS).step_by(8) {
        let mut sums = [[i32x4_splat(0); 2]; 4];
        for p in 0..pairs {
            // SAFETY: bounds asserted above; 16 bytes hold 8 columns of two tokens.
            let raw = unsafe { v128_load(values.add((p * dh_pad + c0 + part) * 2) as *const v128) };
            let (v0, v1) = (i16x8_extend_low_i8x16(raw), i16x8_extend_high_i8x16(raw));
            for (r, row) in sums.iter_mut().enumerate() {
                // SAFETY: row `r` of `weights` holds `n_pad` bytes and `2 p + 1 < n_pad`.
                let pair = unsafe { (levels.add(r * n_pad + 2 * p) as *const u16).read_unaligned() } as i32;
                let b = i32x4_splat((pair & 0xff) | ((pair >> 8) << 16));
                row[0] = i32x4_add(row[0], i32x4_dot_i16x8(b, v0));
                row[1] = i32x4_add(row[1], i32x4_dot_i16x8(b, v1));
            }
        }
        for (r, row) in sums.iter().enumerate() {
            // SAFETY: each row of `acc` holds COLS = 32 sums and `part + 8 <= 32`.
            unsafe {
                v128_store(acc[r].as_mut_ptr().add(part) as *mut v128, row[0]);
                v128_store(acc[r].as_mut_ptr().add(part + 4) as *mut v128, row[1]);
            }
        }
    }
}
