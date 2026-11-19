//! aarch64 NEON tiles of attention, float (`dense::block_attention`) and integer
//! (`integer_attention::block_attention`): the portable tiles' arithmetic four lanes at a time, each output lane
//! summing its products in the portable order, multiply then add (`fmul` and `fadd`, never the fused `fmla`), so the
//! bits are the portable path's. Compiled for `aarch64-pc-windows-msvc`; not run on an Arm processor.

use std::arch::aarch64::*;

use crate::dense::{TILE, exp_nonpositive, round_even_small};
use crate::integer_attention::{COLS, KEYS, LEVELS};

/// `dense::exp_nonpositive` on four lanes, the same operations in the same order.
#[target_feature(enable = "neon")]
#[inline]
pub fn exp4(x: float32x4_t) -> float32x4_t {
    let magic = vdupq_n_f32(12_582_912.0);
    let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
    let shifted = vaddq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)), magic);
    let n = vsubq_f32(shifted, magic);
    let exponent = vsubq_s32(vreinterpretq_s32_f32(shifted), vreinterpretq_s32_f32(magic));
    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n, vdupq_n_f32(0.693_359_4))), vmulq_f32(n, vdupq_n_f32(-2.121_944_4e-4)));
    let mut p = vdupq_n_f32(1.987_569_1e-4);
    for coefficient in [1.398_199_9e-3f32, 8.333_452e-3, 4.166_579_6e-2, 1.666_666_5e-1, 5.000_000_1e-1] {
        p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(coefficient));
    }
    p = vaddq_f32(vaddq_f32(vmulq_f32(p, vmulq_f32(r, r)), r), vdupq_n_f32(1.0));
    vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(exponent, vdupq_n_s32(127)))))
}

/// Multiply `s` by `scale` in place, when given, and return its largest value (exact in any order).
#[target_feature(enable = "neon")]
fn scale_and_max(s: &mut [f32], scale: Option<f32>) -> f32 {
    let full = s.len() / 4 * 4;
    let pointer = s.as_mut_ptr();
    let mut top = vdupq_n_f32(f32::NEG_INFINITY);
    for at in (0..full).step_by(4) {
        // SAFETY: `at + 4 <= s.len()`.
        unsafe {
            let mut value = vld1q_f32(pointer.add(at));
            if let Some(scale) = scale {
                value = vmulq_f32(value, vdupq_n_f32(scale));
                vst1q_f32(pointer.add(at), value);
            }
            top = vmaxq_f32(top, value);
        }
    }
    let mut maximum = vmaxvq_f32(top);
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
#[target_feature(enable = "neon")]
pub unsafe fn scores(query: &[&[f32]; 4], kt: &[f32], n_pad: usize, j0: usize, d_h: usize, scores: &mut [f32]) {
    assert!(kt.len() >= (d_h - 1) * n_pad + j0 + TILE && scores.len() >= 3 * n_pad + j0 + TILE);
    assert!(query.iter().all(|row| row.len() >= d_h));
    let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
    let keys = kt.as_ptr();
    for c in 0..d_h {
        // SAFETY: bounds asserted above.
        let k = unsafe {
            let base = keys.add(c * n_pad + j0);
            [vld1q_f32(base), vld1q_f32(base.add(4)), vld1q_f32(base.add(8)), vld1q_f32(base.add(12))]
        };
        for (sums, row) in acc.iter_mut().zip(query) {
            // SAFETY: each query row holds at least d_h values.
            let w = vdupq_n_f32(unsafe { *row.get_unchecked(c) });
            for (sum, &key) in sums.iter_mut().zip(&k) {
                *sum = vaddq_f32(*sum, vmulq_f32(w, key));
            }
        }
    }
    for (r, sums) in acc.iter().enumerate() {
        for (l, &sum) in sums.iter().enumerate() {
            // SAFETY: bounds asserted above.
            unsafe { vst1q_f32(scores.as_mut_ptr().add(r * n_pad + j0 + 4 * l), sum) };
        }
    }
}

/// `dense::tiles::weighted`: `acc[r][l] = sum_j p_r[j] * vh[j][c0 + l]`.
#[target_feature(enable = "neon")]
pub unsafe fn weighted(scores: &[f32], n_pad: usize, vh: &[f32], dh_pad: usize, c0: usize, n: usize, acc: &mut [[f32; TILE]; 4]) {
    assert!(n == 0 || (vh.len() >= (n - 1) * dh_pad + c0 + TILE && scores.len() >= 3 * n_pad + n));
    let mut sums = [[vdupq_n_f32(0.0); 4]; 4];
    let (values, weights) = (vh.as_ptr(), scores.as_ptr());
    for j in 0..n {
        // SAFETY: bounds asserted above.
        let v = unsafe {
            let base = values.add(j * dh_pad + c0);
            [vld1q_f32(base), vld1q_f32(base.add(4)), vld1q_f32(base.add(8)), vld1q_f32(base.add(12))]
        };
        for (r, row) in sums.iter_mut().enumerate() {
            // SAFETY: bounds asserted above.
            let w = vdupq_n_f32(unsafe { *weights.add(r * n_pad + j) });
            for (sum, &value) in row.iter_mut().zip(&v) {
                *sum = vaddq_f32(*sum, vmulq_f32(w, value));
            }
        }
    }
    for (r, row) in sums.iter().enumerate() {
        for (l, &sum) in row.iter().enumerate() {
            // SAFETY: each row of `acc` holds TILE = 16 floats.
            unsafe { vst1q_f32(acc[r].as_mut_ptr().add(4 * l), sum) };
        }
    }
}

/// `dense::softmax_scaled` four lanes at a time: two vectors of four give the AVX2 path's eight lane sums.
#[target_feature(enable = "neon")]
pub unsafe fn softmax(s: &mut [f32], scale: f32) {
    let maximum = scale_and_max(s, Some(scale));
    let top = vdupq_n_f32(maximum);
    let (mut low, mut high) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let full = s.len() / 8 * 8;
    let pointer = s.as_mut_ptr();
    for at in (0..full).step_by(8) {
        // SAFETY: `at + 8 <= s.len()`.
        unsafe {
            let e0 = exp4(vsubq_f32(vld1q_f32(pointer.add(at)), top));
            let e1 = exp4(vsubq_f32(vld1q_f32(pointer.add(at + 4)), top));
            vst1q_f32(pointer.add(at), e0);
            vst1q_f32(pointer.add(at + 4), e1);
            low = vaddq_f32(low, e0);
            high = vaddq_f32(high, e1);
        }
    }
    let (mut l, mut h) = ([0f32; 4], [0f32; 4]);
    // SAFETY: four floats each.
    unsafe {
        vst1q_f32(l.as_mut_ptr(), low);
        vst1q_f32(h.as_mut_ptr(), high);
    }
    let mut total = ((l[0] + l[1]) + (l[2] + l[3])) + ((h[0] + h[1]) + (h[2] + h[3]));
    for value in &mut s[full..] {
        *value = exp_nonpositive(*value - maximum);
        total += *value;
    }
    let divisor = vdupq_n_f32(total);
    let quarters = s.len() / 4 * 4;
    for at in (0..quarters).step_by(4) {
        // SAFETY: `at + 4 <= s.len()`.
        unsafe { vst1q_f32(pointer.add(at), vdivq_f32(vld1q_f32(pointer.add(at)), divisor)) };
    }
    for value in &mut s[quarters..] {
        *value /= total;
    }
}

// ---------------------------------------------------------------- float64 products

/// `dense::dots_f64` with two `float64x2_t` sums per pair of rows, holding the AVX2 path's four lanes (`c mod 4`):
/// the same sums in the same order (`fmul` then `fadd`), then `(l0 + l1) + (l2 + l3)` and the tail.
#[target_feature(enable = "neon")]
pub unsafe fn dots_f64(a: &[f32], b: &[f64], k: usize, out: &mut [f64]) {
    let (rows_a, rows_b) = (a.len() / k, b.len() / k);
    let full = k / 4 * 4;
    for i in 0..rows_a {
        let pa = a[i * k..(i + 1) * k].as_ptr();
        let mut j = 0;
        while j < rows_b {
            let count = (rows_b - j).min(4);
            let mut sums = [[vdupq_n_f64(0.0); 2]; 4];
            for c in (0..full).step_by(4) {
                // SAFETY: row i of a holds k values and c + 4 <= k.
                let x = unsafe { vld1q_f32(pa.add(c)) };
                let (x0, x1) = (vcvt_f64_f32(vget_low_f32(x)), vcvt_high_f64_f32(x));
                for (r, pair) in sums.iter_mut().enumerate().take(count) {
                    // SAFETY: row j + r of b holds k values and c + 4 <= k.
                    let (y0, y1) = unsafe {
                        let pb = b.as_ptr().add((j + r) * k + c);
                        (vld1q_f64(pb), vld1q_f64(pb.add(2)))
                    };
                    pair[0] = vaddq_f64(pair[0], vmulq_f64(x0, y0));
                    pair[1] = vaddq_f64(pair[1], vmulq_f64(x1, y1));
                }
            }
            for (r, pair) in sums.iter().enumerate().take(count) {
                let mut total = (vgetq_lane_f64::<0>(pair[0]) + vgetq_lane_f64::<1>(pair[0])) + (vgetq_lane_f64::<0>(pair[1]) + vgetq_lane_f64::<1>(pair[1]));
                for c in full..k {
                    total += a[i * k + c] as f64 * b[(j + r) * k + c];
                }
                out[i * rows_b + j + r] = total;
            }
            j += count;
        }
    }
}

// ---------------------------------------------------------------- integer attention

/// `pmaddwd` on NEON: `[a0 b0 + a1 b1, a2 b2 + a3 b3, ...]` from 16-bit lanes.
#[target_feature(enable = "neon")]
#[inline]
fn pairs_dot(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    let high = vmull_high_s16(a, b);
    vpaddq_s32(low, high)
}

/// `integer_attention::portable::scores`, two columns of four keys a vector.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn integer_scores(qi: &[i16], d_h: usize, index: [usize; 4], kt: &[i16], n_pad: usize, j0: usize, x: [f32; 4], y: &[f32], out: &mut [f32]) {
    let half = d_h / 2;
    assert!(index.iter().all(|&i| (i + 1) * d_h <= qi.len()));
    assert!(half == 0 || ((half - 1) * n_pad + j0 + KEYS) * 2 <= kt.len());
    assert!(j0 + KEYS <= y.len() && 3 * n_pad + j0 + KEYS <= out.len());
    let (queries, keys) = (qi.as_ptr(), kt.as_ptr());
    let mut acc = [[vdupq_n_s32(0); 4]; 4];
    for p in 0..half {
        // SAFETY: bounds asserted above.
        let k = unsafe {
            let base = keys.add((p * n_pad + j0) * 2);
            [vld1q_s16(base), vld1q_s16(base.add(8)), vld1q_s16(base.add(16)), vld1q_s16(base.add(24))]
        };
        for (sums, &i) in acc.iter_mut().zip(&index) {
            // SAFETY: row `i` of `qi` holds `d_h` values and `2 p + 1 < d_h`.
            let b = vreinterpretq_s16_s32(vdupq_n_s32(unsafe { (queries.add(i * d_h + 2 * p) as *const i32).read_unaligned() }));
            for (sum, &key) in sums.iter_mut().zip(&k) {
                *sum = vaddq_s32(*sum, pairs_dot(b, key));
            }
        }
    }
    for (r, sums) in acc.iter().enumerate() {
        let factor = vdupq_n_f32(x[r]);
        for (l, &sum) in sums.iter().enumerate() {
            // SAFETY: bounds asserted above.
            unsafe {
                let keys_factor = vld1q_f32(y.as_ptr().add(j0 + 4 * l));
                vst1q_f32(out.as_mut_ptr().add(r * n_pad + j0 + 4 * l), vmulq_f32(vmulq_f32(vcvtq_f32_s32(sum), factor), keys_factor));
            }
        }
    }
}

/// `integer_attention::portable::softmax_levels` four lanes at a time, its float64 lane sums included.
#[target_feature(enable = "neon")]
pub unsafe fn softmax_levels(row: &mut [f32], levels: &mut [u8]) {
    let n = row.len();
    assert!(levels.len() >= n);
    let top = scale_and_max(row, None);
    let shift = vdupq_n_f32(top);
    let full = n / 8 * 8;
    let pointer = row.as_mut_ptr();
    let (mut first, mut second) = (vdupq_n_f64(0.0), vdupq_n_f64(0.0)); // lanes 0, 1 and lanes 2, 3 of j mod 4
    for at in (0..full).step_by(8) {
        // SAFETY: `at + 8 <= n`.
        unsafe {
            let e0 = exp4(vsubq_f32(vld1q_f32(pointer.add(at)), shift));
            let e1 = exp4(vsubq_f32(vld1q_f32(pointer.add(at + 4)), shift));
            vst1q_f32(pointer.add(at), e0);
            vst1q_f32(pointer.add(at + 4), e1);
            first = vaddq_f64(first, vcvt_f64_f32(vget_low_f32(e0)));
            second = vaddq_f64(second, vcvt_high_f64_f32(e0));
            first = vaddq_f64(first, vcvt_f64_f32(vget_low_f32(e1)));
            second = vaddq_f64(second, vcvt_high_f64_f32(e1));
        }
    }
    let total_lanes = [vgetq_lane_f64::<0>(first), vgetq_lane_f64::<1>(first), vgetq_lane_f64::<0>(second), vgetq_lane_f64::<1>(second)];
    let mut total = (total_lanes[0] + total_lanes[1]) + (total_lanes[2] + total_lanes[3]);
    for value in &mut row[full..] {
        *value = exp_nonpositive(*value - top);
        total += *value as f64;
    }
    let r = (LEVELS as f64 / total) as f32;
    let (factor, magic) = (vdupq_n_f32(r), vdupq_n_f32(12_582_912.0));
    for at in (0..full).step_by(8) {
        // SAFETY: `at + 8 <= n <= levels.len()`.
        unsafe {
            let round = |at: usize| {
                let scaled = vmulq_f32(vld1q_f32(pointer.add(at)), factor);
                vcvtq_s32_f32(vsubq_f32(vaddq_f32(scaled, magic), magic))
            };
            let shorts = vcombine_s16(vqmovn_s32(round(at)), vqmovn_s32(round(at + 4)));
            vst1_u8(levels.as_mut_ptr().add(at), vqmovun_s16(shorts));
        }
    }
    for j in full..n {
        // SAFETY: as in `integer_attention::portable::softmax_levels`.
        levels[j] = unsafe { round_even_small(row[j] * r).to_int_unchecked::<i32>() } as u8;
    }
    levels[n..].fill(0);
}

/// `integer_attention::portable::weighted`: values widened to 16 bits, two tokens of four columns a vector.
#[target_feature(enable = "neon")]
pub unsafe fn integer_weighted(weights: &[u8], n_pad: usize, vt: &[i8], dh_pad: usize, c0: usize, acc: &mut [[i32; COLS]; 4]) {
    let pairs = n_pad / 2;
    assert!(weights.len() >= 4 * n_pad && (pairs == 0 || ((pairs - 1) * dh_pad + c0 + COLS) * 2 <= vt.len()));
    let (values, levels) = (vt.as_ptr(), weights.as_ptr());
    for part in (0..COLS).step_by(16) {
        let mut sums = [[vdupq_n_s32(0); 4]; 4];
        for p in 0..pairs {
            // SAFETY: bounds asserted above; 32 bytes hold 16 columns of two tokens.
            let v = unsafe {
                let base = values.add((p * dh_pad + c0 + part) * 2);
                let (a, b) = (vld1q_s8(base), vld1q_s8(base.add(16)));
                [vmovl_s8(vget_low_s8(a)), vmovl_high_s8(a), vmovl_s8(vget_low_s8(b)), vmovl_high_s8(b)]
            };
            for (r, row) in sums.iter_mut().enumerate() {
                // SAFETY: row `r` of `weights` holds `n_pad` bytes and `2 p + 1 < n_pad`.
                let pair = unsafe { (levels.add(r * n_pad + 2 * p) as *const u16).read_unaligned() } as i32;
                let b = vreinterpretq_s16_s32(vdupq_n_s32((pair & 0xff) | ((pair >> 8) << 16)));
                for (sum, &value) in row.iter_mut().zip(&v) {
                    *sum = vaddq_s32(*sum, pairs_dot(b, value));
                }
            }
        }
        for (r, row) in sums.iter().enumerate() {
            for (l, &sum) in row.iter().enumerate() {
                // SAFETY: each row of `acc` holds COLS = 32 sums and `part + 16 <= 32`.
                unsafe { vst1q_s32(acc[r].as_mut_ptr().add(part + 4 * l), sum) };
            }
        }
    }
}
