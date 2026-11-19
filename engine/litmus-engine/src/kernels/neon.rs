//! aarch64 ternary products over one unpacked tile of signed weights (module documentation in `mod.rs`).
//!
//! `tile_dot` uses the Armv8.2 dot product (`sdot`, four products into each 32-bit lane), through inline assembly
//! because its intrinsic is not stable. `tile_mlal` uses `smlal`, products into 16-bit lanes, two a step of 16
//! columns, so a lane gains at most 256 a step and is widened every [`CHUNK`] steps.

use std::arch::aarch64::*;

use super::TILE_ROWS;

/// Steps of 16 columns accumulated in 16 bits before widening: `127 * 256 = 32512`.
const CHUNK: usize = 127;

#[target_feature(enable = "neon,dotprod")]
#[inline]
fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: `sdot` reads and writes vector registers only; the function requires the dot product extension.
    unsafe {
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

/// Sums of 4 tile rows times 2 activation rows over `steps` steps of 16 columns: index `token * 4 + row`.
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn block_dot(w: *const i8, a: *const i8, stride: usize, steps: usize, tokens: usize) -> [i32; 8] {
    let mut c = [vdupq_n_s32(0); 8];
    for k in 0..steps {
        let offset = k * 16;
        // SAFETY: the caller guarantees 4 tile rows and `tokens` activation rows of `stride >= steps * 16` bytes.
        unsafe {
            let w0 = vld1q_s8(w.add(offset));
            let w1 = vld1q_s8(w.add(stride + offset));
            let w2 = vld1q_s8(w.add(2 * stride + offset));
            let w3 = vld1q_s8(w.add(3 * stride + offset));
            for token in 0..tokens {
                let a0 = vld1q_s8(a.add(token * stride + offset));
                c[token * 4] = sdot(c[token * 4], w0, a0);
                c[token * 4 + 1] = sdot(c[token * 4 + 1], w1, a0);
                c[token * 4 + 2] = sdot(c[token * 4 + 2], w2, a0);
                c[token * 4 + 3] = sdot(c[token * 4 + 3], w3, a0);
            }
        }
    }
    let mut out = [0i32; 8];
    for (value, lane) in out.iter_mut().zip(c) {
        *value = vaddvq_s32(lane);
    }
    out
}

/// As `block_dot`, with `smlal` into 16-bit lanes.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn block_mlal(w: *const i8, a: *const i8, stride: usize, steps: usize, tokens: usize) -> [i32; 8] {
    let mut wide = [vdupq_n_s32(0); 8];
    let mut k0 = 0;
    while k0 < steps {
        let k1 = (k0 + CHUNK).min(steps);
        let mut c = [vdupq_n_s16(0); 8];
        for k in k0..k1 {
            let offset = k * 16;
            // SAFETY: as in block_dot.
            unsafe {
                let w0 = vld1q_s8(w.add(offset));
                let w1 = vld1q_s8(w.add(stride + offset));
                let w2 = vld1q_s8(w.add(2 * stride + offset));
                let w3 = vld1q_s8(w.add(3 * stride + offset));
                for token in 0..tokens {
                    let a0 = vld1q_s8(a.add(token * stride + offset));
                    for (row, weights) in [w0, w1, w2, w3].into_iter().enumerate() {
                        let lane = &mut c[token * 4 + row];
                        *lane = vmlal_s8(*lane, vget_low_s8(weights), vget_low_s8(a0));
                        *lane = vmlal_high_s8(*lane, weights, a0);
                    }
                }
            }
        }
        for (total, lane) in wide.iter_mut().zip(c) {
            *total = vpadalq_s16(*total, lane);
        }
        k0 = k1;
    }
    let mut out = [0i32; 8];
    for (value, lane) in out.iter_mut().zip(wide) {
        *value = vaddvq_s32(lane);
    }
    out
}

/// The column path over rows `[r0, r0 + 8 * R)`: `smlal` of each pair's codes against `(q0, q1)` into 16-bit lanes
/// (a lane gains at most `2 * 128 = 256` a pair), widened with `sadalp` every [`CHUNK`] pairs, which also adds each
/// row's two products.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn column_block<const R: usize>(codes: *const u8, r0: usize, offsets: &[u32], pairs: &[u16], acc: *mut i32) {
    let mut i0 = 0;
    while i0 < offsets.len() {
        let i1 = (i0 + CHUNK).min(offsets.len());
        let mut c = [[vdupq_n_s16(0); 2]; R];
        for i in i0..i1 {
            let q = vreinterpret_s8_u16(vdup_n_u16(pairs[i]));
            let qq = vcombine_s8(q, q);
            // SAFETY: the caller checked that every offset leaves `2 * height` bytes, and `r0 + 8 * R <= height`.
            let base = unsafe { codes.add(offsets[i] as usize + 2 * r0) } as *const i8;
            for (j, lanes) in c.iter_mut().enumerate() {
                // SAFETY: as above, 16 bytes at row r0 + 8 j.
                let w = unsafe { vld1q_s8(base.add(16 * j)) };
                lanes[0] = vmlal_s8(lanes[0], vget_low_s8(w), q);
                lanes[1] = vmlal_high_s8(lanes[1], w, qq);
            }
        }
        for (j, lanes) in c.iter().enumerate() {
            // SAFETY: `acc` holds `height >= r0 + 8 * R` sums.
            unsafe {
                let p = acc.add(r0 + 8 * j);
                vst1q_s32(p, vpadalq_s16(vld1q_s32(p), lanes[0]));
                vst1q_s32(p.add(4), vpadalq_s16(vld1q_s32(p.add(4)), lanes[1]));
            }
        }
        i0 = i1;
    }
}

/// The column path: `acc[r] += sum over pairs of codes(r) . (q0, q1)`, rows in blocks of at most 64.
///
/// # Safety
/// The processor must support NEON; every offset must leave `2 * height` bytes of `codes`.
#[target_feature(enable = "neon")]
pub unsafe fn columns(codes: &[u8], height: usize, offsets: &[u32], pairs: &[u16], acc: &mut [i32]) {
    assert!(height % 16 == 0 && acc.len() >= height && offsets.len() == pairs.len());
    let (w, a) = (codes.as_ptr(), acc.as_mut_ptr());
    let mut r0 = 0;
    while r0 < height {
        let loads = ((height - r0) / 8).min(8);
        // SAFETY: rows r0..r0 + 8 * loads lie inside `height`; the caller vouches for the offsets.
        unsafe {
            match loads {
                2 => column_block::<2>(w, r0, offsets, pairs, a),
                4 => column_block::<4>(w, r0, offsets, pairs, a),
                6 => column_block::<6>(w, r0, offsets, pairs, a),
                _ => column_block::<8>(w, r0, offsets, pairs, a),
            }
        }
        r0 += 8 * loads;
    }
}

fn check(tile: &[u8], stride: usize, q: &[i8], n: usize, acc: &[i32]) {
    assert!(stride % 16 == 0 && tile.len() >= TILE_ROWS * stride && q.len() >= n * stride && acc.len() >= n * TILE_ROWS);
}

/// `sum t q` with `sdot`.
///
/// # Safety
/// The processor must support NEON and the dot product extension.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn tile_dot(tile: &[u8], stride: usize, q: &[i8], n: usize, acc: &mut [i32]) {
    check(tile, stride, q, n, acc);
    let w = tile.as_ptr() as *const i8;
    let mut t = 0;
    while t < n {
        let tokens = (n - t).min(2);
        for r in (0..TILE_ROWS).step_by(4) {
            // SAFETY: rows r..r+4 of the tile and rows t..t+tokens of q lie inside the slices (checked above).
            let s = unsafe { block_dot(w.add(r * stride), q.as_ptr().add(t * stride), stride, stride / 16, tokens) };
            for token in 0..tokens {
                acc[(t + token) * TILE_ROWS + r..(t + token) * TILE_ROWS + r + 4].copy_from_slice(&s[token * 4..token * 4 + 4]);
            }
        }
        t += tokens;
    }
}

/// `sum t q` with `smlal`.
///
/// # Safety
/// The processor must support NEON.
#[target_feature(enable = "neon")]
pub unsafe fn tile_mlal(tile: &[u8], stride: usize, q: &[i8], n: usize, acc: &mut [i32]) {
    check(tile, stride, q, n, acc);
    let w = tile.as_ptr() as *const i8;
    let mut t = 0;
    while t < n {
        let tokens = (n - t).min(2);
        for r in (0..TILE_ROWS).step_by(4) {
            // SAFETY: as in tile_dot.
            let s = unsafe { block_mlal(w.add(r * stride), q.as_ptr().add(t * stride), stride, stride / 16, tokens) };
            for token in 0..tokens {
                acc[(t + token) * TILE_ROWS + r..(t + token) * TILE_ROWS + r + 4].copy_from_slice(&s[token * 4..token * 4 + 4]);
            }
        }
        t += tokens;
    }
}
