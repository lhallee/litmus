//! A WebAssembly module that checks the SIMD128 ternary product against the scalar path and times the engine in a
//! JavaScript engine (`scripts/wasm_probe.mjs` drives it). Build:
//! `RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown --example wasm_probe`.

use std::cell::RefCell;

use litmus_engine::dense::{self, AttentionScratch};
use litmus_engine::integer_attention::{self, IntegerScratch};
use litmus_engine::kernels::{self, Backend, ColumnPairs, KernelScratch, PackedTernary, QuantRows};
use litmus_engine::Engine;

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static BUFFERS: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

/// A buffer of `length` bytes the host fills; returns its address. Buffers live until `probe_reset`.
#[unsafe(no_mangle)]
pub extern "C" fn probe_alloc(length: usize) -> *mut u8 {
    BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();
        buffers.push(vec![0u8; length]);
        buffers.last_mut().unwrap().as_mut_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn probe_reset() {
    BUFFERS.with(|buffers| buffers.borrow_mut().clear());
}

/// The SIMD128 backend against the scalar one on random shapes: the number of shapes that differ (0 is a pass), or
/// `u32::MAX` when the module was built without SIMD128.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_check() -> u32 {
    if !Backend::Simd128.is_available() {
        return u32::MAX;
    }
    let mut random = Lcg(9);
    let mut mismatches = 0;
    for (rows, cols, n) in [(1, 4, 1), (5, 36, 3), (17, 92, 2), (33, 260, 5), (192, 192, 7), (40, 2052, 3), (24, 4100, 2)] {
        let values: Vec<i8> = (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect();
        let activations: Vec<i8> = (0..n * cols).map(|_| ((random.next() % 256) as i32 - 128) as i8).collect();
        let weights = PackedTernary::from_values(rows, cols, &values, 1.0);
        let x = QuantRows::from_values(&activations, n, cols, &vec![1.0; n]);
        let (mut fast, mut slow) = (vec![0i32; n * rows], vec![0i32; n * rows]);
        let mut scratch = KernelScratch::new();
        kernels::matmul_i32(Backend::Simd128, &weights, &x, &mut fast, &mut scratch);
        kernels::matmul_i32(Backend::Scalar, &weights, &x, &mut slow, &mut scratch);
        mismatches += (fast != slow) as u32;
    }
    // The column path against the dense path, on both backends: random zero rates, all zero, one nonzero, extremes.
    for (rows, cols, zeros) in [(1, 4, 0u32), (17, 36, 500), (192, 768, 900), (192, 768, 1000), (33, 4100, 0), (512, 2048, 990)] {
        for extreme in [None, Some(-128i8), Some(127)] {
            let n = 5;
            let values: Vec<i8> = (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect();
            let activations: Vec<i8> = (0..n * cols)
                .map(|i| match extreme {
                    Some(value) => value,
                    None if i % cols == 3 => 1,
                    None if random.next() % 1000 < zeros => 0,
                    None => ((random.next() % 256) as i32 - 128) as i8,
                })
                .collect();
            let weights = PackedTernary::from_values(rows, cols, &values, 1.0);
            let columns = ColumnPairs::new(&weights);
            let x = QuantRows::from_values(&activations, n, cols, &vec![1.0; n]);
            let (mut column, mut dense) = (vec![0i32; n * rows], vec![0i32; n * rows]);
            let mut scratch = KernelScratch::new();
            kernels::matmul_i32(Backend::Scalar, &weights, &x, &mut dense, &mut scratch);
            for backend in [Backend::Simd128, Backend::Scalar] {
                kernels::column_i32(backend, &columns, &x, &mut column, &mut scratch);
                mismatches += (column != dense) as u32;
            }
        }
    }
    mismatches
}

/// The `down` product `(rows, cols)` on `tokens` rows whose activations are zero at `zeros` percent, `repeats` times,
/// by the dense path alone (`columns` 0) or the column path alone (1), on the fastest backend; the host times it.
#[unsafe(no_mangle)]
pub extern "C" fn column_run(rows: usize, cols: usize, tokens: usize, zeros: u32, repeats: usize, columns: u32) -> f32 {
    let mut random = Lcg(8);
    let values: Vec<i8> = (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect();
    let activations: Vec<i8> = (0..tokens * cols)
        .map(|_| if random.next() % 100 < zeros { 0 } else { (1 + random.next() % 127) as i8 * if random.next() % 2 == 0 { 1 } else { -1 } })
        .collect();
    let weights = PackedTernary::from_values(rows, cols, &values, 1.0);
    let pairs = ColumnPairs::new(&weights);
    let x = QuantRows::from_values(&activations, tokens, cols, &vec![1.0; tokens]);
    let below = if columns == 1 { cols / 2 + 1 } else { 0 };
    let mut out = vec![0f32; tokens * rows];
    let mut scratch = KernelScratch::new();
    for _ in 0..repeats {
        kernels::matmul_f32_columns(Backend::detect(), &weights, &pairs, &x, &mut out, rows, true, below, &mut scratch);
    }
    out[0]
}

/// The `down` product's path in the loaded engine: 0 the measured choice, 1 the dense path alone.
#[unsafe(no_mangle)]
pub extern "C" fn engine_dense_only(dense: u32) {
    use litmus_engine::model::ColumnChoice;
    ENGINE.with(|slot| {
        if let Some(engine) = slot.borrow_mut().as_mut() {
            engine.set_column_choice(if dense == 1 { ColumnChoice::Dense } else { ColumnChoice::Measured });
        }
    });
}

/// Run the ternary product `repeats` times on a `(rows, cols)` matrix and `tokens` rows; the host times it.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_run(rows: usize, cols: usize, tokens: usize, repeats: usize) -> f32 {
    let mut random = Lcg(3);
    let values: Vec<i8> = (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect();
    let activations: Vec<i8> = (0..tokens * cols).map(|_| ((random.next() % 255) as i32 - 127) as i8).collect();
    let weights = PackedTernary::from_values(rows, cols, &values, 1.0);
    let x = QuantRows::from_values(&activations, tokens, cols, &vec![1.0; tokens]);
    let mut out = vec![0f32; tokens * rows];
    let mut scratch = KernelScratch::new();
    for _ in 0..repeats {
        kernels::matmul_f32(Backend::detect(), &weights, &x, &mut out, rows, false, &mut scratch);
    }
    out[0]
}

fn floats(n: usize, seed: u64) -> Vec<f32> {
    let mut random = Lcg(seed);
    (0..n).map(|_| (random.next() % 20_001) as f32 / 5_000.0 - 2.0).collect()
}

/// Float (`integer` 0) or integer (1) attention over one block of `n` tokens of width `d`, `repeats` times, on the
/// fastest backend (`portable` 1: the portable path); the host times it.
#[unsafe(no_mangle)]
pub extern "C" fn attention_run(n: usize, d: usize, heads: usize, repeats: usize, integer: u32, portable: u32) -> f32 {
    let (q, k, v) = (floats(n * d, 1), floats(n * d, 2), floats(n * d, 3));
    let mut o = vec![0f32; n * d];
    let backend = if portable == 1 { Backend::Scalar } else { Backend::detect() };
    let (mut float_scratch, mut integer_scratch) = (AttentionScratch::default(), IntegerScratch::default());
    for _ in 0..repeats {
        if integer == 1 {
            integer_attention::block_attention(backend, &q, &k, &v, &mut o, n, d, heads, 1e-5, &mut integer_scratch);
        } else {
            dense::block_attention(backend, &q, &k, &v, &mut o, n, d, heads, &mut float_scratch);
        }
    }
    o[0]
}

/// RMS norm and quantization of `n` rows of width `d`, `repeats` times; the host times it.
#[unsafe(no_mangle)]
pub extern "C" fn norm_run(n: usize, d: usize, repeats: usize) -> f32 {
    let x = floats(n * d, 4);
    let gain = vec![1.0f32; d];
    let mut normed = vec![0f32; n * d];
    let mut quant = QuantRows::new();
    for _ in 0..repeats {
        for row in 0..n {
            dense::rms_norm(&x[row * d..(row + 1) * d], &gain, 1e-6, &mut normed[row * d..(row + 1) * d]);
        }
        quant.quantize(&normed, n, d, 1e-5);
    }
    normed[0]
}

/// Every float and integer attention path of this build against the portable path, bit for bit: the number of
/// shapes that differ (0 is a pass).
#[unsafe(no_mangle)]
pub extern "C" fn attention_check() -> u32 {
    let mut mismatches = 0;
    for (n, d, heads) in [(1, 64, 2), (5, 64, 2), (16, 64, 2), (37, 192, 3), (256, 192, 3), (130, 384, 6), (9, 12, 3)] {
        let (q, k, v) = (floats(n * d, 5), floats(n * d, 6), floats(n * d, 7));
        let (mut fast, mut portable) = (vec![0f32; n * d], vec![0f32; n * d]);
        dense::block_attention(Backend::detect(), &q, &k, &v, &mut fast, n, d, heads, &mut AttentionScratch::default());
        dense::block_attention_portable(&q, &k, &v, &mut portable, n, d, heads, &mut AttentionScratch::default());
        mismatches += (fast != portable) as u32;
        integer_attention::block_attention(Backend::detect(), &q, &k, &v, &mut fast, n, d, heads, 1e-5, &mut IntegerScratch::default());
        integer_attention::block_attention_portable(&q, &k, &v, &mut portable, n, d, heads, 1e-5, &mut IntegerScratch::default());
        mismatches += (fast != portable) as u32;
    }
    for (rows_a, rows_b, k) in [(1, 1, 4), (3, 9, 256), (5, 7, 13), (2, 4, 3), (16, 122, 256), (7, 5, 512)] {
        let a = floats(rows_a * k, 8);
        let b: Vec<f64> = floats(rows_b * k, 9).iter().map(|&x| x as f64 / 3.0).collect();
        let (mut fast, mut portable) = (vec![0f64; rows_a * rows_b], vec![0f64; rows_a * rows_b]);
        dense::dots_f64(&a, &b, k, &mut fast);
        dense::dots_f64_portable(&a, &b, k, &mut portable);
        mismatches += (fast != portable) as u32;
    }
    mismatches
}

fn bytes(pointer: *const u8, length: usize) -> &'static [u8] {
    // SAFETY: the host passes a buffer from `probe_alloc` that it filled, and keeps it until `probe_reset`.
    unsafe { std::slice::from_raw_parts(pointer, length) }
}

/// Load a release; 0 on success.
#[unsafe(no_mangle)]
pub extern "C" fn engine_load(model: *const u8, model_length: usize, config: *const u8, config_length: usize, tokenizer: *const u8, tokenizer_length: usize) -> i32 {
    match Engine::load(bytes(model, model_length), bytes(config, config_length), bytes(tokenizer, tokenizer_length)) {
        Ok(engine) => {
            ENGINE.with(|slot| *slot.borrow_mut() = Some(engine));
            0
        }
        Err(_) => 1,
    }
}

/// The head and the bank for a chunk of `length` tokens and its windows, on synthetic states (`Engine::read_pass`);
/// the host times it.
#[unsafe(no_mangle)]
pub extern "C" fn engine_read_pass(length: usize) -> u32 {
    ENGINE.with(|slot| slot.borrow().as_ref().map_or(0, |engine| engine.read_pass(length) as u32))
}

/// Score a UTF-8 text; returns its token count (negative on error) and leaves the reading's score in `out`.
#[unsafe(no_mangle)]
pub extern "C" fn engine_score(text: *const u8, length: usize) -> f64 {
    let Ok(text) = std::str::from_utf8(bytes(text, length)) else { return -1.0 };
    ENGINE.with(|slot| match slot.borrow().as_ref().map(|engine| engine.score(text)) {
        Some(Ok(document)) => document.n_tokens as f64,
        _ => -2.0,
    })
}


