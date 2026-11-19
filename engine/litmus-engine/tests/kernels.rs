//! Every SIMD backend equals the scalar path bit for bit, on random inputs of every shape and remainder: rows across
//! tile boundaries, columns across vector and 16-bit chunk boundaries, odd and even token counts, and the extreme
//! values that bound the 16-bit lanes.

use litmus_engine::kernels::{self, Backend, ColumnPairs, KernelScratch, PackedTernary, QuantRows};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

fn naive(weights: &PackedTernary, values: &[i8], n: usize) -> Vec<i32> {
    let (rows, cols) = (weights.rows(), weights.cols());
    let mut out = vec![0i32; n * rows];
    for t in 0..n {
        for r in 0..rows {
            out[t * rows + r] = (0..cols).map(|c| weights.value(r, c) as i32 * values[t * cols + c] as i32).sum();
        }
    }
    out
}

/// Activation range a backend accepts: `Avx2Sign` cannot negate -128.
fn lowest(backend: Backend) -> i32 {
    if backend == Backend::Avx2Sign { -127 } else { -128 }
}

fn check(rows: usize, cols: usize, n: usize, random: &mut Lcg, extreme: Option<(i8, i8)>) {
    let values: Vec<i8> = match extreme {
        Some((weight, _)) => vec![weight; rows * cols],
        None => (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect(),
    };
    let weights = PackedTernary::from_values(rows, cols, &values, 0.75);
    let scales: Vec<f32> = (0..n).map(|t| 0.5 + t as f32 * 0.25).collect();
    for backend in Backend::available() {
        let floor = lowest(backend);
        let activations: Vec<i8> = match extreme {
            Some((_, value)) => vec![value.max(floor as i8); n * cols],
            None => (0..n * cols).map(|_| (floor + (random.next() % (128 - floor) as u32) as i32) as i8).collect(),
        };
        let x = QuantRows::from_values(&activations, n, cols, &scales);
        let want = naive(&weights, &activations, n);
        let mut scratch = KernelScratch::new();
        let mut got = vec![0i32; n * rows];
        kernels::matmul_i32(backend, &weights, &x, &mut got, &mut scratch);
        assert_eq!(got, want, "{} rows {rows} cols {cols} tokens {n} extreme {extreme:?}", backend.name());

        // The float epilogue is the same arithmetic on every backend.
        let mut floats = vec![1.5f32; n * rows];
        kernels::matmul_f32(backend, &weights, &x, &mut floats, rows, true, &mut scratch);
        for t in 0..n {
            let factor = 0.75f32 / scales[t];
            for r in 0..rows {
                assert_eq!(floats[t * rows + r], 1.5 + want[t * rows + r] as f32 * factor, "{} float output", backend.name());
            }
        }
    }
}

#[test]
fn backends_equal_scalar_on_every_remainder() {
    let mut random = Lcg(1);
    for rows in [1, 3, 4, 5, 15, 16, 17, 33] {
        for cols in [4, 8, 28, 32, 36, 64, 92, 128, 192, 260] {
            for n in [1, 2, 3, 5, 8] {
                check(rows, cols, n, &mut random, None);
            }
        }
    }
}

#[test]
fn backends_equal_scalar_on_model_shapes() {
    let mut random = Lcg(2);
    for (rows, cols) in [(192, 192), (768, 192), (192, 768), (384, 1536), (64, 2048), (40, 2052), (24, 4100)] {
        for n in [1, 7, 64] {
            check(rows, cols, n, &mut random, None);
        }
    }
}

#[test]
fn sixteen_bit_lanes_never_overflow() {
    let mut random = Lcg(3);
    // Every weight +1 or -1 against every activation at an extreme: the largest sums the lanes must hold, across the
    // 64-step (2,048-column) widening boundary of the code form and the 127-step one of the 16-bit forms.
    for cols in [2044, 2048, 2052, 4064, 4096, 4100, 8192] {
        for weight in [-1i8, 1] {
            for activation in [127i8, -128, -127] {
                check(5, cols, 3, &mut random, Some((weight, activation)));
            }
        }
    }
}

/// Activations for the column path's checks, `n` rows of `cols`.
#[derive(Clone, Copy, Debug)]
enum Pattern {
    /// Each value zero with this probability in 1,000, else uniform in [-128, 127] without 0.
    Zeros(u32),
    AllZero,
    /// Every value nonzero.
    NoZero,
    /// One nonzero value per row, at column `(row * 7919 + 3) % cols`, of the given value.
    One(i8),
    /// Every value this one.
    Every(i8),
    /// Every value -128 or 127, alternating by column, a run of 300 zeros from column 5.
    Extremes,
}

fn activations(pattern: Pattern, n: usize, cols: usize, random: &mut Lcg) -> Vec<i8> {
    fn nonzero(random: &mut Lcg) -> i8 {
        loop {
            let value = (random.next() % 256) as i32 - 128;
            if value != 0 {
                return value as i8;
            }
        }
    }
    (0..n * cols)
        .map(|i| {
            let (row, col) = (i / cols, i % cols);
            match pattern {
                Pattern::Zeros(per_mille) => {
                    if random.next() % 1000 < per_mille { 0 } else { nonzero(random) }
                }
                Pattern::AllZero => 0,
                Pattern::NoZero => nonzero(random),
                Pattern::One(value) => {
                    if col == (row * 7919 + 3) % cols { value } else { 0 }
                }
                Pattern::Every(value) => value,
                Pattern::Extremes => {
                    if (5..305).contains(&col) { 0 } else if col % 2 == 0 { -128 } else { 127 }
                }
            }
        })
        .collect()
}

/// The column path's integers equal the naive sums and the dense path's, and the per-token choice gives the dense
/// path's float bits at every limit, written and accumulated, on every backend.
fn check_columns(rows: usize, cols: usize, n: usize, weights_of: Option<i8>, pattern: Pattern, random: &mut Lcg) {
    let values: Vec<i8> = match weights_of {
        Some(weight) => vec![weight; rows * cols],
        None => (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect(),
    };
    let weights = PackedTernary::from_values(rows, cols, &values, 0.375);
    let columns = ColumnPairs::new(&weights);
    let q = activations(pattern, n, cols, random);
    let scales: Vec<f32> = (0..n).map(|t| 0.3 + t as f32 * 0.17).collect();
    let x = QuantRows::from_values(&q, n, cols, &scales);
    let want = naive(&weights, &q, n);
    let half = cols / 2;
    let label = format!("rows {rows} cols {cols} tokens {n} weights {weights_of:?} {pattern:?}");
    for backend in Backend::available() {
        let mut scratch = KernelScratch::new();
        let mut got = vec![0i32; n * rows];
        kernels::column_i32(backend, &columns, &x, &mut got, &mut scratch);
        assert_eq!(got, want, "{} column path, {label}", backend.name());
        if backend == Backend::Avx2Sign && q.contains(&-128) {
            continue; // the dense sign form cannot negate -128 (as in `lowest`)
        }
        let out_stride = rows + 3;
        for accumulate in [false, true] {
            let start: Vec<f32> = (0..n * out_stride).map(|i| (i % 13) as f32 * 0.25 - 1.0).collect();
            let mut dense = start.clone();
            kernels::matmul_f32(backend, &weights, &x, &mut dense, out_stride, accumulate, &mut scratch);
            for below in [0, 1, 2, half / 4, kernels::column_limit(backend, &columns) + 1, half, half + 1] {
                let mut mixed = start.clone();
                let taken = kernels::matmul_f32_columns(backend, &weights, &columns, &x, &mut mixed, out_stride, accumulate, below, &mut scratch);
                assert!(taken <= n);
                let same = mixed.iter().zip(&dense).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{} below {below} accumulate {accumulate}, {label}", backend.name());
            }
        }
    }
}

#[test]
fn column_path_equals_dense_on_every_shape() {
    let mut random = Lcg(11);
    for rows in [1, 5, 16, 17, 33, 192] {
        for cols in [4, 8, 36, 260, 768] {
            for n in [1, 2, 7] {
                for pattern in [Pattern::Zeros(0), Pattern::Zeros(500), Pattern::Zeros(900), Pattern::Zeros(990)] {
                    check_columns(rows, cols, n, None, pattern, &mut random);
                }
            }
        }
    }
    for (rows, cols) in [(192, 768), (256, 1024), (384, 1536), (512, 2048), (40, 2052)] {
        for pattern in [Pattern::Zeros(700), Pattern::Zeros(850), Pattern::Zeros(950)] {
            check_columns(rows, cols, 64, None, pattern, &mut random);
        }
    }
}

#[test]
fn column_path_equals_dense_on_adversarial_inputs() {
    let mut random = Lcg(12);
    let patterns = [
        Pattern::AllZero,
        Pattern::NoZero,
        Pattern::One(1),
        Pattern::One(-128),
        Pattern::One(127),
        Pattern::Every(-128),
        Pattern::Every(127),
        Pattern::Extremes,
    ];
    for (rows, cols) in [(1, 4), (17, 36), (192, 768), (512, 2048)] {
        for pattern in patterns {
            check_columns(rows, cols, 5, None, pattern, &mut random);
        }
    }
    // Every weight +1 or -1 against every activation at an extreme: the largest sums the 16-bit lanes must hold,
    // across the column path's widening every 64 pairs (AVX2) or 127 (NEON), with 2,050 pairs.
    for cols in [256, 260, 512, 4100] {
        for weight in [-1i8, 1] {
            for pattern in [Pattern::Every(-128), Pattern::Every(127), Pattern::Extremes] {
                check_columns(33, cols, 3, Some(weight), pattern, &mut random);
            }
        }
    }
}

#[test]
fn column_pairs_hold_the_matrix() {
    let mut random = Lcg(13);
    let values: Vec<i8> = (0..17 * 12).map(|_| (random.next() % 3) as i8 - 1).collect();
    let weights = PackedTernary::from_values(17, 12, &values, 1.0);
    let columns = ColumnPairs::new(&weights);
    assert_eq!((columns.rows(), columns.cols()), (17, 12));
    // 17 rows pad to 32, two bytes a row, six pairs.
    assert_eq!(columns.bytes(), 12 * 32);
}

#[test]
fn packing_round_trips_and_refuses_code_three() {
    let values: Vec<i8> = (0..64).map(|i| (i % 3) as i8 - 1).collect();
    let weights = PackedTernary::from_values(4, 16, &values, 1.0);
    for r in 0..4 {
        for c in 0..16 {
            assert_eq!(weights.value(r, c), values[r * 16 + c]);
        }
    }
    assert!(PackedTernary::new(1, 4, vec![0b1100_0000], 1.0).is_err());
    assert!(PackedTernary::new(1, 4, vec![0b1001_0110], 1.0).is_ok());
    assert!(PackedTernary::new(1, 6, vec![0, 0], 1.0).is_err());
}

#[test]
fn detection_picks_an_available_backend() {
    let backend = Backend::detect();
    assert!(backend.is_available());
    assert!(Backend::available().contains(&Backend::Scalar));
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        assert_eq!(backend, Backend::Avx2);
    }
}

#[test]
fn attention_is_the_same_on_every_path() {
    use litmus_engine::dense::{self, AttentionScratch};
    let mut random = Lcg(4);
    for (n, d, heads) in [(1, 64, 2), (5, 64, 2), (16, 64, 2), (37, 192, 3), (256, 192, 3), (130, 384, 6), (9, 12, 3)] {
        let mut draw = || (0..n * d).map(|_| (random.next() % 20_001) as f32 / 5_000.0 - 2.0).collect::<Vec<f32>>();
        let (q, k, v) = (draw(), draw(), draw());
        let (mut fast, mut portable) = (vec![0f32; n * d], vec![0f32; n * d]);
        dense::block_attention_portable(&q, &k, &v, &mut portable, n, d, heads, &mut AttentionScratch::default());
        for backend in Backend::available() {
            dense::block_attention(backend, &q, &k, &v, &mut fast, n, d, heads, &mut AttentionScratch::default());
            assert_eq!(fast, portable, "attention on {} n {n} d {d} heads {heads}", backend.name());
        }
        dense::block_attention(Backend::detect(), &q, &k, &v, &mut fast, n, d, heads, &mut AttentionScratch::default());
        // Against a float64 attention, to the float32 rounding of a 256-term sum.
        let d_h = d / heads;
        for head in 0..heads {
            for i in 0..n {
                let scores: Vec<f64> = (0..n)
                    .map(|j| (0..d_h).map(|c| q[i * d + head * d_h + c] as f64 * k[j * d + head * d_h + c] as f64).sum::<f64>() / (d_h as f64).sqrt())
                    .collect();
                let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f64 = e.iter().sum();
                for c in 0..d_h {
                    let want: f64 = (0..n).map(|j| e[j] / total * v[j * d + head * d_h + c] as f64).sum();
                    let got = fast[i * d + head * d_h + c] as f64;
                    assert!((got - want).abs() < 1e-4, "attention value {got} against {want}");
                }
            }
        }
    }
}

/// SPEC.md, Integer attention, written out loop by loop: the integers, then each float step in the stated order.
fn integer_attention_by_the_book(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize, heads: usize, eps: f32) -> Vec<f32> {
    use litmus_engine::dense::{exp_nonpositive, max_abs};
    let d_h = d / heads;
    let level = |x: f32, s: f32| (x * s).round_ties_even().clamp(-127.0, 127.0) as i32;
    let mut out = vec![0f32; n * d];
    for head in 0..heads {
        let at = |rows: &[f32], i: usize| rows[i * d + head * d_h..i * d + (head + 1) * d_h].to_vec();
        let scale = |row: &[f32]| 127.0f32 / max_abs(row).max(eps);
        let (s_q, s_k): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (scale(&at(q, i)), scale(&at(k, i)))).unzip();
        let s_v = 127.0f32 / (0..n).map(|j| max_abs(&at(v, j))).fold(0f32, f32::max).max(eps);
        for i in 0..n {
            let x = (1.0 / (d_h as f64).sqrt()) as f32 / s_q[i];
            let scores: Vec<f32> = (0..n)
                .map(|j| {
                    let integer: i32 = (0..d_h).map(|c| level(at(q, i)[c], s_q[i]) * level(at(k, j)[c], s_k[j])).sum();
                    (integer as f32 * x) * (1.0 / s_k[j])
                })
                .collect();
            let top = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|&s| exp_nonpositive(s - top)).collect();
            let r = (127.0 / e.iter().map(|&value| value as f64).sum::<f64>()) as f32;
            let weights: Vec<i32> = e.iter().map(|&value| (value * r).round_ties_even() as i32).collect();
            assert!(weights.iter().all(|w| (0..=127).contains(w)));
            for c in 0..d_h {
                let sum: i32 = (0..n).map(|j| weights[j] * level(at(v, j)[c], s_v)).sum();
                out[i * d + head * d_h + c] = sum as f32 * (1.0 / (127.0 * s_v));
            }
        }
    }
    out
}

#[test]
fn integer_attention_is_the_same_on_every_path_and_follows_the_spec() {
    use litmus_engine::integer_attention::{self, IntegerScratch};
    let mut random = Lcg(5);
    let shapes = [(1, 64, 2), (2, 64, 2), (5, 64, 2), (16, 64, 2), (17, 64, 1), (37, 192, 3), (256, 192, 3), (130, 384, 6), (9, 12, 3), (300, 128, 2)];
    for (n, d, heads) in shapes {
        for spread in [0.5f32, 4.0, 40.0] {
            let mut draw = || (0..n * d).map(|_| ((random.next() % 20_001) as f32 / 10_000.0 - 1.0) * spread).collect::<Vec<f32>>();
            let (q, k, v) = (draw(), draw(), draw());
            let mut portable = vec![0f32; n * d];
            integer_attention::block_attention_portable(&q, &k, &v, &mut portable, n, d, heads, 1e-5, &mut IntegerScratch::default());
            for backend in Backend::available() {
                let mut fast = vec![0f32; n * d];
                integer_attention::block_attention(backend, &q, &k, &v, &mut fast, n, d, heads, 1e-5, &mut IntegerScratch::default());
                assert_eq!(fast, portable, "integer attention on {} n {n} d {d} heads {heads} spread {spread}", backend.name());
            }
            if n <= 130 {
                assert_eq!(portable, integer_attention_by_the_book(&q, &k, &v, n, d, heads, 1e-5), "by the book, n {n} d {d}");
            }
        }
    }
    // One scratch reused across blocks of different lengths gives what fresh scratch gives.
    let mut scratch = IntegerScratch::default();
    for n in [256, 3, 128, 1, 200] {
        let (d, heads) = (192, 3);
        let draw = |seed: usize| (0..n * d).map(|i| ((i * 37 + seed * 101) % 997) as f32 / 499.0 - 1.0).collect::<Vec<f32>>();
        let (q, k, v) = (draw(1), draw(2), draw(3));
        let (mut reused, mut fresh) = (vec![0f32; n * d], vec![0f32; n * d]);
        integer_attention::block_attention(Backend::detect(), &q, &k, &v, &mut reused, n, d, heads, 1e-5, &mut scratch);
        integer_attention::block_attention(Backend::detect(), &q, &k, &v, &mut fresh, n, d, heads, 1e-5, &mut IntegerScratch::default());
        assert_eq!(reused, fresh, "reused scratch at n {n}");
    }
}

#[test]
fn integer_attention_sums_stay_inside_sixteen_bits() {
    // Large weighted sums: every value at +127 or -127, and rows of equal scores, whose 127 / n weights all round up
    // (n = 255, 256 and 263, past the 256 that W allows in a release), or one dominant key.
    use litmus_engine::integer_attention::{self, IntegerScratch};
    for (n, peaked) in [(256, false), (256, true), (255, false), (263, false)] {
        let (d, heads) = (64, 1);
        let q: Vec<f32> = (0..n * d).map(|i| if peaked && i / d == 0 { 1.0 } else { 0.25 }).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..n * d).map(|i| if (i / d + i % d) % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut portable = vec![0f32; n * d];
        integer_attention::block_attention_portable(&q, &k, &v, &mut portable, n, d, heads, 1e-5, &mut IntegerScratch::default());
        for backend in Backend::available() {
            let mut fast = vec![0f32; n * d];
            integer_attention::block_attention(backend, &q, &k, &v, &mut fast, n, d, heads, 1e-5, &mut IntegerScratch::default());
            assert_eq!(fast, portable, "{} at n {n}", backend.name());
        }
    }
}

/// The integer dot products of the int8 compare (SPEC.md, Int8 output), `codes::dots_i8`, against the definition: a sum of
/// products of codes in [-127, 127], exact in int32, on random codes of every shape and remainder (rows across the
/// four-row blocks of the AVX2 path, strides of one to six vectors) and on the extremes that bound a lane.
#[test]
fn int8_dots_equal_the_definition_on_every_path() {
    use litmus_engine::codes::{self, dots_i8, dots_i8_portable};
    let mut random = Lcg(0x1234_5678);
    let code = |random: &mut Lcg| (random.next() % 255) as i32 - 127;
    for stride in [32, 64, 96, 128, 192] {
        for (rows_a, rows_b) in [(1, 1), (3, 2), (4, 1), (5, 7), (32, 62), (33, 5), (17, 124)] {
            let a: Vec<i8> = (0..rows_a * stride).map(|_| code(&mut random) as i8).collect();
            let b: Vec<i8> = (0..rows_b * stride).map(|_| code(&mut random) as i8).collect();
            let mut want = vec![0i32; rows_a * rows_b];
            for j in 0..rows_b {
                for i in 0..rows_a {
                    want[j * rows_a + i] = (0..stride).map(|c| a[i * stride + c] as i32 * b[j * stride + c] as i32).sum();
                }
            }
            let (mut fast, mut portable) = (vec![0i32; want.len()], vec![0i32; want.len()]);
            dots_i8(&a, &b, stride, &mut fast);
            dots_i8_portable(&a, &b, stride, &mut portable);
            assert_eq!(portable, want, "portable, {rows_a} x {rows_b} x {stride}");
            assert_eq!(fast, want, "{rows_a} x {rows_b} x {stride}");
        }
        // Every code at the extreme, same and opposite signs, and a zero row.
        for (x, y) in [(127i8, 127i8), (127, -127), (-127, -127), (-127, 127), (0, 127)] {
            let (a, b) = (vec![x; 5 * stride], vec![y; 3 * stride]);
            let mut got = vec![0i32; 15];
            dots_i8(&a, &b, stride, &mut got);
            assert!(got.iter().all(|&sum| sum == x as i32 * y as i32 * stride as i32), "extreme {x} {y} at stride {stride}");
        }
    }
    // A half padded with zeros past the width reads as the width alone.
    let (w, stride) = (40, codes::padded(40));
    let q: Vec<i8> = (0..w).map(|_| code(&mut random) as i8).collect();
    let (mut padded_a, mut padded_b) = (vec![0i8; stride], vec![0i8; stride]);
    padded_a[..w].copy_from_slice(&q);
    padded_b[..w].copy_from_slice(&q);
    let mut got = [0i32];
    dots_i8(&padded_a, &padded_b, stride, &mut got);
    assert_eq!(got[0], codes::squared_norm(&q));
}

/// The output quantizer (SPEC.md, Int8 output) against its formula on vectors of every scale, ties included.
#[test]
fn output_codes_follow_the_specification() {
    use litmus_engine::codes::quantize_output;
    let mut random = Lcg(99);
    for scale in [1e-6f32, 0.003, 1.0, 40.0, 3e4] {
        for width in [1, 7, 64, 128] {
            let x: Vec<f32> = (0..width).map(|_| ((random.next() % 20_001) as f32 / 10_000.0 - 1.0) * scale).collect();
            let mut got = vec![0i8; width];
            quantize_output(&x, 1e-5, &mut got);
            let m = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-5);
            let s = 127.0f32 / m;
            let want: Vec<i8> = x.iter().map(|&v| (v * s).round_ties_even().clamp(-127.0, 127.0) as i8).collect();
            assert_eq!(got, want, "width {width} scale {scale}");
            assert!(got.iter().all(|&c| c != i8::MIN));
        }
    }
}
