//! AVX2 ternary products over one unpacked tile (module documentation in `mod.rs`).
//!
//! Both forms add 16-bit pair sums into 16-bit lanes for at most [`CHUNK`] steps of 32 columns, then widen to 32
//! bits. In the code form a lane gains at most `2 * 2 * 128 = 512` in magnitude a step, so 64 steps stay within
//! 16 bits (`64 * 512 = 32768`, and only `-32768` is reachable at that bound); in the sign form at most 256.

use std::arch::x86_64::*;

use super::TILE_ROWS;

/// Steps of 32 columns accumulated in 16 bits before widening.
const CHUNK: usize = 64;

/// One step's products of a weight vector and an activation vector, as 16 pair sums. `SIGN`: weights are `t` as
/// signed bytes and `vpsignb` negates or zeroes each activation; otherwise weights are the codes `t + 1` as unsigned
/// bytes multiplied by `vpmaddubsw`.
#[target_feature(enable = "avx2")]
#[inline]
fn step<const SIGN: bool>(weights: __m256i, activations: __m256i) -> __m256i {
    if SIGN {
        _mm256_maddubs_epi16(_mm256_set1_epi8(1), _mm256_sign_epi8(activations, weights))
    } else {
        _mm256_maddubs_epi16(weights, activations)
    }
}

/// The horizontal sums of eight vectors of eight 32-bit lanes, in order.
#[target_feature(enable = "avx2")]
#[inline]
fn sum8(v: [__m256i; 8]) -> __m256i {
    let h01 = _mm256_hadd_epi32(v[0], v[1]);
    let h23 = _mm256_hadd_epi32(v[2], v[3]);
    let h45 = _mm256_hadd_epi32(v[4], v[5]);
    let h67 = _mm256_hadd_epi32(v[6], v[7]);
    let h0123 = _mm256_hadd_epi32(h01, h23);
    let h4567 = _mm256_hadd_epi32(h45, h67);
    let low = _mm256_permute2x128_si256(h0123, h4567, 0x20);
    let high = _mm256_permute2x128_si256(h0123, h4567, 0x31);
    _mm256_add_epi32(low, high)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load(pointer: *const u8) -> __m256i {
    unsafe { _mm256_loadu_si256(pointer as *const __m256i) }
}

/// Sums of 4 tile rows times 2 activation rows over `steps` steps of 32 columns: lane `token * 4 + row`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn block_4x2<const SIGN: bool>(w: *const u8, a: *const u8, stride: usize, steps: usize) -> [i32; 8] {
    let ones = _mm256_set1_epi16(1);
    let mut total = _mm256_setzero_si256();
    let mut k0 = 0;
    while k0 < steps {
        let k1 = (k0 + CHUNK).min(steps);
        let mut c0 = _mm256_setzero_si256();
        let mut c1 = c0;
        let mut c2 = c0;
        let mut c3 = c0;
        let mut c4 = c0;
        let mut c5 = c0;
        let mut c6 = c0;
        let mut c7 = c0;
        for k in k0..k1 {
            let offset = k * 32;
            // SAFETY: the caller guarantees 4 tile rows and 2 activation rows of `stride >= steps * 32` bytes.
            let (a0, a1, w0, w1, w2, w3) = unsafe {
                (
                    load(a.add(offset)),
                    load(a.add(stride + offset)),
                    load(w.add(offset)),
                    load(w.add(stride + offset)),
                    load(w.add(2 * stride + offset)),
                    load(w.add(3 * stride + offset)),
                )
            };
            c0 = _mm256_add_epi16(c0, step::<SIGN>(w0, a0));
            c1 = _mm256_add_epi16(c1, step::<SIGN>(w1, a0));
            c2 = _mm256_add_epi16(c2, step::<SIGN>(w2, a0));
            c3 = _mm256_add_epi16(c3, step::<SIGN>(w3, a0));
            c4 = _mm256_add_epi16(c4, step::<SIGN>(w0, a1));
            c5 = _mm256_add_epi16(c5, step::<SIGN>(w1, a1));
            c6 = _mm256_add_epi16(c6, step::<SIGN>(w2, a1));
            c7 = _mm256_add_epi16(c7, step::<SIGN>(w3, a1));
        }
        let wide = [
            _mm256_madd_epi16(c0, ones),
            _mm256_madd_epi16(c1, ones),
            _mm256_madd_epi16(c2, ones),
            _mm256_madd_epi16(c3, ones),
            _mm256_madd_epi16(c4, ones),
            _mm256_madd_epi16(c5, ones),
            _mm256_madd_epi16(c6, ones),
            _mm256_madd_epi16(c7, ones),
        ];
        total = _mm256_add_epi32(total, sum8(wide));
        k0 = k1;
    }
    let mut out = [0i32; 8];
    // SAFETY: `out` holds 32 bytes.
    unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, total) };
    out
}

/// Sums of 4 tile rows times 1 activation row: lanes 0 to 3.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn block_4x1<const SIGN: bool>(w: *const u8, a: *const u8, stride: usize, steps: usize) -> [i32; 8] {
    let ones = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();
    let mut total = zero;
    let mut k0 = 0;
    while k0 < steps {
        let k1 = (k0 + CHUNK).min(steps);
        let mut c0 = zero;
        let mut c1 = zero;
        let mut c2 = zero;
        let mut c3 = zero;
        for k in k0..k1 {
            let offset = k * 32;
            // SAFETY: as in block_4x2, with one activation row.
            let (a0, w0, w1, w2, w3) = unsafe {
                (
                    load(a.add(offset)),
                    load(w.add(offset)),
                    load(w.add(stride + offset)),
                    load(w.add(2 * stride + offset)),
                    load(w.add(3 * stride + offset)),
                )
            };
            c0 = _mm256_add_epi16(c0, step::<SIGN>(w0, a0));
            c1 = _mm256_add_epi16(c1, step::<SIGN>(w1, a0));
            c2 = _mm256_add_epi16(c2, step::<SIGN>(w2, a0));
            c3 = _mm256_add_epi16(c3, step::<SIGN>(w3, a0));
        }
        let wide = [
            _mm256_madd_epi16(c0, ones),
            _mm256_madd_epi16(c1, ones),
            _mm256_madd_epi16(c2, ones),
            _mm256_madd_epi16(c3, ones),
            zero,
            zero,
            zero,
            zero,
        ];
        total = _mm256_add_epi32(total, sum8(wide));
        k0 = k1;
    }
    let mut out = [0i32; 8];
    // SAFETY: `out` holds 32 bytes.
    unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, total) };
    out
}

#[target_feature(enable = "avx2")]
#[inline]
fn tile_generic<const SIGN: bool>(tile: &[u8], stride: usize, q: &[i8], sums: &[i32], n: usize, acc: &mut [i32]) {
    assert!(stride % 32 == 0 && tile.len() >= TILE_ROWS * stride && q.len() >= n * stride && acc.len() >= n * TILE_ROWS);
    assert!(SIGN || sums.len() >= n, "the code form needs every row's sum");
    let steps = stride / 32;
    let w = tile.as_ptr();
    let a = q.as_ptr() as *const u8;
    let mut t = 0;
    while t + 2 <= n {
        let correction = if SIGN { [0, 0] } else { [sums[t], sums[t + 1]] };
        for r in (0..TILE_ROWS).step_by(4) {
            // SAFETY: rows r..r+4 of the tile and rows t, t+1 of q lie inside the slices (asserted above).
            let s = unsafe { block_4x2::<SIGN>(w.add(r * stride), a.add(t * stride), stride, steps) };
            for token in 0..2 {
                for row in 0..4 {
                    acc[(t + token) * TILE_ROWS + r + row] = s[token * 4 + row] - correction[token];
                }
            }
        }
        t += 2;
    }
    if t < n {
        let correction = if SIGN { 0 } else { sums[t] };
        for r in (0..TILE_ROWS).step_by(4) {
            // SAFETY: as above, with row t of q.
            let s = unsafe { block_4x1::<SIGN>(w.add(r * stride), a.add(t * stride), stride, steps) };
            for row in 0..4 {
                acc[t * TILE_ROWS + r + row] = s[row] - correction;
            }
        }
    }
}

/// Pairs added into 16-bit lanes before the column path widens them: a lane gains at most `2 * 128 + 2 * 128 = 512`
/// a pair, the bound of the dense code form.
const COLUMN_CHUNK: usize = 64;

/// The column path over rows `[r0, r0 + 16 * R)`: `R` registers of 16 rows, each pair one `vpmaddubsw` a register.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn column_block<const R: usize>(codes: *const u8, r0: usize, offsets: &[u32], pairs: &[u16], acc: *mut i32) {
    let mut i0 = 0;
    while i0 < offsets.len() {
        let i1 = (i0 + COLUMN_CHUNK).min(offsets.len());
        let mut c = [_mm256_setzero_si256(); R];
        for i in i0..i1 {
            let q = _mm256_set1_epi16(pairs[i] as i16);
            // SAFETY: the caller checked that every offset leaves `2 * height` bytes, and `r0 + 16 * R <= height`.
            let base = unsafe { codes.add(offsets[i] as usize + 2 * r0) };
            for (j, lane) in c.iter_mut().enumerate() {
                // SAFETY: as above, 32 bytes at row r0 + 16 j.
                let w = unsafe { load(base.add(32 * j)) };
                *lane = _mm256_add_epi16(*lane, _mm256_maddubs_epi16(w, q));
            }
        }
        for (j, lane) in c.iter().enumerate() {
            let low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(*lane));
            let high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(*lane));
            // SAFETY: `acc` holds `height >= r0 + 16 * R` sums.
            unsafe {
                let p = acc.add(r0 + 16 * j) as *mut __m256i;
                _mm256_storeu_si256(p, _mm256_add_epi32(_mm256_loadu_si256(p), low));
                _mm256_storeu_si256(p.add(1), _mm256_add_epi32(_mm256_loadu_si256(p.add(1)), high));
            }
        }
        i0 = i1;
    }
}

/// The column path: `acc[r] += sum over pairs of codes(r) . (q0, q1)`, rows in passes of at most 12 registers,
/// split evenly (256 rows as 8 and 8, not 12 and 4).
///
/// # Safety
/// The processor must support AVX2; every offset must leave `2 * height` bytes of `codes`.
#[target_feature(enable = "avx2")]
pub unsafe fn columns(codes: &[u8], height: usize, offsets: &[u32], pairs: &[u16], acc: &mut [i32]) {
    assert!(height % 16 == 0 && acc.len() >= height && offsets.len() == pairs.len());
    let (w, a) = (codes.as_ptr(), acc.as_mut_ptr());
    let mut passes_left = (height / 16).div_ceil(12);
    let mut r0 = 0;
    while r0 < height {
        // At most 12: the registers left never exceed 12 per pass left.
        let registers = ((height - r0) / 16).div_ceil(passes_left);
        passes_left -= 1;
        // SAFETY: rows r0..r0 + 16 * registers lie inside `height`; the caller vouches for the offsets.
        unsafe {
            match registers {
                1 => column_block::<1>(w, r0, offsets, pairs, a),
                2 => column_block::<2>(w, r0, offsets, pairs, a),
                3 => column_block::<3>(w, r0, offsets, pairs, a),
                4 => column_block::<4>(w, r0, offsets, pairs, a),
                5 => column_block::<5>(w, r0, offsets, pairs, a),
                6 => column_block::<6>(w, r0, offsets, pairs, a),
                7 => column_block::<7>(w, r0, offsets, pairs, a),
                8 => column_block::<8>(w, r0, offsets, pairs, a),
                9 => column_block::<9>(w, r0, offsets, pairs, a),
                10 => column_block::<10>(w, r0, offsets, pairs, a),
                11 => column_block::<11>(w, r0, offsets, pairs, a),
                _ => column_block::<12>(w, r0, offsets, pairs, a),
            }
        }
        r0 += 16 * registers;
    }
}

/// Code form: `sum (t + 1) q - sum q`.
///
/// # Safety
/// The processor must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn tile_codes(tile: &[u8], stride: usize, q: &[i8], sums: &[i32], n: usize, acc: &mut [i32]) {
    tile_generic::<false>(tile, stride, q, sums, n, acc)
}

/// Sign form: `sum sign(t) q`.
///
/// # Safety
/// The processor must support AVX2. Activations must lie in [-127, 127]: `vpsignb` cannot negate -128.
#[target_feature(enable = "avx2")]
pub unsafe fn tile_sign(tile: &[u8], stride: usize, q: &[i8], n: usize, acc: &mut [i32]) {
    tile_generic::<true>(tile, stride, q, &[], n, acc)
}
