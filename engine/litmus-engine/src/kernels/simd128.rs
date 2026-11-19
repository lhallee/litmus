//! WebAssembly SIMD128 ternary products over one unpacked tile of signed weights (module documentation in
//! `mod.rs`). `i16x8.extmul` gives products in 16-bit lanes, two a lane a step of 16 columns, widened every
//! [`CHUNK`] steps (`127 * 256 = 32512`).

use std::arch::wasm32::*;

use super::TILE_ROWS;

const CHUNK: usize = 127;

/// Sums of 4 tile rows times `tokens` (1 or 2) activation rows: index `token * 4 + row`.
#[inline(always)]
fn block(tile: &[u8], r: usize, q: &[i8], t: usize, stride: usize, tokens: usize) -> [i32; 8] {
    let steps = stride / 16;
    let mut wide = [i32x4_splat(0); 8];
    let mut k0 = 0;
    while k0 < steps {
        let k1 = (k0 + CHUNK).min(steps);
        let mut c = [i16x8_splat(0); 8];
        for k in k0..k1 {
            let offset = k * 16;
            let weights: [v128; 4] = std::array::from_fn(|row| {
                let start = (r + row) * stride + offset;
                // SAFETY: the slice bound check reads 16 bytes in range; v128_load is unaligned.
                unsafe { v128_load(tile[start..start + 16].as_ptr() as *const v128) }
            });
            for token in 0..tokens {
                let start = (t + token) * stride + offset;
                // SAFETY: the slice bound check above reads 16 bytes in range; v128_load is unaligned.
                let a0 = unsafe { v128_load(q[start..start + 16].as_ptr() as *const v128) };
                for (row, &w) in weights.iter().enumerate() {
                    let lane = &mut c[token * 4 + row];
                    *lane = i16x8_add(*lane, i16x8_extmul_low_i8x16(w, a0));
                    *lane = i16x8_add(*lane, i16x8_extmul_high_i8x16(w, a0));
                }
            }
        }
        for (total, lane) in wide.iter_mut().zip(c) {
            *total = i32x4_add(*total, i32x4_extadd_pairwise_i16x8(lane));
        }
        k0 = k1;
    }
    wide.map(|lane| {
        i32x4_extract_lane::<0>(lane) + i32x4_extract_lane::<1>(lane) + i32x4_extract_lane::<2>(lane) + i32x4_extract_lane::<3>(lane)
    })
}

/// The column path over rows `[r0, r0 + 8 * R)`: each pair's 16 bytes of codes (8 rows) widened to 16 bits, and
/// `i32x4.dot_i16x8_s` against `(q0, q1)` gives each row's `c0 q0 + c1 q1` in a 32-bit lane.
#[inline(always)]
fn column_block<const R: usize>(codes: &[u8], r0: usize, offsets: &[u32], pairs: &[u16], acc: &mut [i32]) {
    let mut c = [[i32x4_splat(0); 2]; R];
    let w = codes.as_ptr();
    for (&offset, &pair) in offsets.iter().zip(pairs) {
        let q0 = pair as u8 as i8 as i16 as u16 as u32;
        let q1 = (pair >> 8) as u8 as i8 as i16 as u16 as u32;
        let q = u32x4_splat(q0 | (q1 << 16));
        let base = offset as usize + 2 * r0;
        for (j, lanes) in c.iter_mut().enumerate() {
            // SAFETY: the caller checked that every offset leaves `2 * height` bytes, and `r0 + 8 * R <= height`;
            // v128_load is unaligned.
            let codes = unsafe { v128_load(w.add(base + 16 * j) as *const v128) };
            lanes[0] = i32x4_add(lanes[0], i32x4_dot_i16x8(u16x8_extend_low_u8x16(codes), q));
            lanes[1] = i32x4_add(lanes[1], i32x4_dot_i16x8(u16x8_extend_high_u8x16(codes), q));
        }
    }
    for (j, lanes) in c.iter().enumerate() {
        for (half, lane) in lanes.iter().enumerate() {
            let at = r0 + 8 * j + 4 * half;
            let sums = [
                i32x4_extract_lane::<0>(*lane),
                i32x4_extract_lane::<1>(*lane),
                i32x4_extract_lane::<2>(*lane),
                i32x4_extract_lane::<3>(*lane),
            ];
            for (total, sum) in acc[at..at + 4].iter_mut().zip(sums) {
                *total += sum;
            }
        }
    }
}

/// The column path: `acc[r] += sum over pairs of codes(r) . (q0, q1)`, rows in blocks of at most 48.
///
/// # Safety
/// Every offset must leave `2 * height` bytes of `codes`.
pub unsafe fn columns(codes: &[u8], height: usize, offsets: &[u32], pairs: &[u16], acc: &mut [i32]) {
    assert!(height % 16 == 0 && acc.len() >= height && offsets.len() == pairs.len());
    let mut r0 = 0;
    while r0 < height {
        let loads = ((height - r0) / 8).min(6);
        match loads {
            2 => column_block::<2>(codes, r0, offsets, pairs, acc),
            4 => column_block::<4>(codes, r0, offsets, pairs, acc),
            _ => column_block::<6>(codes, r0, offsets, pairs, acc),
        }
        r0 += 8 * loads;
    }
}

/// `sum t q` over a tile of signed weights.
pub fn tile(tile: &[u8], stride: usize, q: &[i8], n: usize, acc: &mut [i32]) {
    assert!(stride % 16 == 0 && tile.len() >= TILE_ROWS * stride && q.len() >= n * stride && acc.len() >= n * TILE_ROWS);
    let mut t = 0;
    while t < n {
        let tokens = (n - t).min(2);
        for r in (0..TILE_ROWS).step_by(4) {
            let s = block(tile, r, q, t, stride, tokens);
            for token in 0..tokens {
                acc[(t + token) * TILE_ROWS + r..(t + token) * TILE_ROWS + r + 4].copy_from_slice(&s[token * 4..token * 4 + 4]);
            }
        }
        t += tokens;
    }
}
