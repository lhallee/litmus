//! The ternary product on the encoder's shapes, every available backend: throughput in ternary additions (weights
//! times tokens) per second. `cargo bench -p litmus-engine --bench kernels`.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litmus_engine::kernels::{self, Backend, KernelScratch, PackedTernary, QuantRows};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

fn inputs(rows: usize, cols: usize, n: usize) -> (PackedTernary, QuantRows) {
    let mut random = Lcg(7);
    let values: Vec<i8> = (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect();
    let activations: Vec<i8> = (0..n * cols).map(|_| ((random.next() % 255) as i32 - 127) as i8).collect();
    (PackedTernary::from_values(rows, cols, &values, 0.5), QuantRows::from_values(&activations, n, cols, &vec![1.0; n]))
}

fn products(c: &mut Criterion) {
    // (rows, cols): q/k/v/o at widths 192, 384 and 1,536; up and down at d = 192 and 384 (f = 4d).
    let shapes = [(192, 192), (384, 384), (768, 192), (192, 768), (1536, 384), (384, 1536), (1536, 1536)];
    for tokens in [256usize, 64] {
        let mut group = c.benchmark_group(format!("ternary_{tokens}_tokens"));
        group.sample_size(20).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2));
        for &(rows, cols) in &shapes {
            let (weights, x) = inputs(rows, cols, tokens);
            let mut out = vec![0f32; tokens * rows];
            let mut scratch = KernelScratch::new();
            group.throughput(Throughput::Elements((rows * cols * tokens) as u64));
            for backend in Backend::available() {
                if backend == Backend::Scalar && rows * cols > 384 * 384 {
                    continue;
                }
                group.bench_with_input(BenchmarkId::new(backend.name(), format!("{rows}x{cols}")), &(), |b, _| {
                    b.iter(|| kernels::matmul_f32(backend, &weights, &x, &mut out, rows, false, &mut scratch))
                });
            }
        }
        group.finish();
    }
}

/// The feed-forward's `down` product on 64 tokens whose activations are zero at a given rate (independently), by the
/// dense path alone and by the column path alone: where the column path starts to pay. `-- down_columns`.
fn down_columns(c: &mut Criterion) {
    use kernels::ColumnPairs;
    for (rows, cols) in [(192, 768), (256, 1024), (384, 1536), (512, 2048)] {
        let mut group = c.benchmark_group(format!("down_columns_{rows}x{cols}"));
        group.sample_size(20).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));
        let tokens = 64;
        let mut random = Lcg(8);
        let values: Vec<i8> = (0..rows * cols).map(|_| (random.next() % 3) as i8 - 1).collect();
        let weights = PackedTernary::from_values(rows, cols, &values, 0.5);
        let columns = ColumnPairs::new(&weights);
        group.throughput(Throughput::Elements(tokens as u64));
        for zeros in [0u32, 50, 70, 80, 85, 90, 95] {
            let activations: Vec<i8> = (0..tokens * cols)
                .map(|_| if random.next() % 100 < zeros { 0 } else { (1 + random.next() % 127) as i8 * if random.next() % 2 == 0 { 1 } else { -1 } })
                .collect();
            let x = QuantRows::from_values(&activations, tokens, cols, &vec![1.0; tokens]);
            let mut out = vec![0f32; tokens * rows];
            let mut scratch = KernelScratch::new();
            for backend in Backend::available() {
                if backend == Backend::Avx2Sign {
                    continue;
                }
                for (path, below) in [("dense", 0), ("columns", cols / 2 + 1)] {
                    group.bench_with_input(BenchmarkId::new(format!("{}_{path}", backend.name()), format!("{zeros}%")), &(), |b, _| {
                        b.iter(|| kernels::matmul_f32_columns(backend, &weights, &columns, &x, &mut out, rows, true, below, &mut scratch))
                    });
                }
            }
        }
        group.finish();
    }
}

criterion_group!(benches, products, down_columns);
criterion_main!(benches);
