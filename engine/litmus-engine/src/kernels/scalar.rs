//! The scalar ternary product, which defines the answer every SIMD path must equal bit for bit.
//!
//! A weight of +1 adds the activation, -1 subtracts it, 0 skips it, written as a multiply by the weight with no
//! branch: random ternary weights make a branch per weight a coin flip the processor mispredicts half the time, and the
//! branch-free sum is a plain integer reduction the compiler vectorizes for whatever the target has (SSE2 on any
//! x86-64, NEON on aarch64, SIMD128 in a WebAssembly build that enables it). The sums are exact integers either way.

use super::TILE_ROWS;

/// Pairs added into 16-bit sums before the column path widens them: a row gains at most `2 * 128 + 2 * 128 = 512` a
/// pair.
const COLUMN_CHUNK: usize = 64;

/// The column path over `ROWS` rows from `r0`: each row's two codes read as one little-endian 16-bit lane (low byte
/// column `2k`), so the products are 16-bit multiplies the compiler vectorizes without shuffles.
#[inline(always)]
fn column_block<const ROWS: usize>(codes: &[u8], height: usize, r0: usize, offsets: &[u32], pairs: &[u16], acc: &mut [i32]) {
    debug_assert!(r0 + ROWS <= height);
    for (offsets, pairs) in offsets.chunks(COLUMN_CHUNK).zip(pairs.chunks(COLUMN_CHUNK)) {
        let mut sums = [0i16; ROWS];
        for (&offset, &pair) in offsets.iter().zip(pairs) {
            let q0 = pair as u8 as i8 as i16;
            let q1 = (pair >> 8) as u8 as i8 as i16;
            let start = offset as usize + 2 * r0;
            let column = &codes[start..start + 2 * ROWS];
            for (sum, two) in sums.iter_mut().zip(column.chunks_exact(2)) {
                let both = u16::from_le_bytes([two[0], two[1]]);
                *sum += (both & 0xff) as i16 * q0 + (both >> 8) as i16 * q1;
            }
        }
        for (total, &sum) in acc[r0..r0 + ROWS].iter_mut().zip(&sums) {
            *total += sum as i32;
        }
    }
}

/// The column path (`mod.rs`, [`super::ColumnPairs`]): for each nonzero pair at byte `offset` of `codes` with
/// activations `pair` (low byte first), `acc[r] += codes[offset + 2r] * q0 + codes[offset + 2r + 1] * q1` over
/// `height` rows (a multiple of 16), in blocks of 64 rows.
pub fn columns(codes: &[u8], height: usize, offsets: &[u32], pairs: &[u16], acc: &mut [i32]) {
    assert!(height % 16 == 0 && acc.len() >= height);
    let mut r0 = 0;
    while r0 + 64 <= height {
        column_block::<64>(codes, height, r0, offsets, pairs, acc);
        r0 += 64;
    }
    while r0 < height {
        column_block::<16>(codes, height, r0, offsets, pairs, acc);
        r0 += 16;
    }
}

/// `acc[t * TILE_ROWS + r] = sum_c tile[r][c] * q[t][c]` over a tile of signed weights `(TILE_ROWS, stride)` and
/// activations `(n, stride)`.
pub fn tile(tile: &[u8], stride: usize, q: &[i8], n: usize, acc: &mut [i32]) {
    for t in 0..n {
        let activations = &q[t * stride..(t + 1) * stride];
        for r in 0..TILE_ROWS {
            let weights = &tile[r * stride..(r + 1) * stride];
            acc[t * TILE_ROWS + r] = weights.iter().zip(activations).fold(0i32, |sum, (&weight, &activation)| sum + weight as i8 as i32 * activation as i32);
        }
    }
}
