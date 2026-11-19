//! The float pieces of one block on the encoder's shapes: attention, RMS norm with quantization, rotary positions.
//! `cargo bench -p litmus-engine --bench blocks`.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litmus_engine::dense::{self, AttentionScratch};
use litmus_engine::integer_attention::{self, IntegerScratch};
use litmus_engine::kernels::{Backend, QuantRows};

fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n).map(|i| (((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8) as f32 / 16_777_216.0 - 0.5) * 4.0).collect()
}

fn pieces(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_pieces");
    group.sample_size(20).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2));
    let n = 256;
    for (d, heads) in [(192usize, 3usize), (384, 6)] {
        let (q, k, v) = (values(n * d, 1), values(n * d, 2), values(n * d, 3));
        let mut o = vec![0f32; n * d];
        let mut scratch = AttentionScratch::default();
        group.throughput(Throughput::Elements(n as u64));
        let mut integer = IntegerScratch::default();
        for backend in Backend::available() {
            group.bench_with_input(BenchmarkId::new(format!("attention_{}", backend.name()), d), &(), |b, _| {
                b.iter(|| dense::block_attention(backend, &q, &k, &v, &mut o, n, d, heads, &mut scratch))
            });
            group.bench_with_input(BenchmarkId::new(format!("integer_attention_{}", backend.name()), d), &(), |b, _| {
                b.iter(|| integer_attention::block_attention(backend, &q, &k, &v, &mut o, n, d, heads, 1e-5, &mut integer))
            });
        }
        let gain = vec![1.0f32; d];
        let mut normed = vec![0f32; n * d];
        let mut quant = QuantRows::new();
        group.bench_with_input(BenchmarkId::new("norm_and_quantize", d), &(), |b, _| {
            b.iter(|| {
                for row in 0..n {
                    dense::rms_norm(&q[row * d..(row + 1) * d], &gain, 1e-6, &mut normed[row * d..(row + 1) * d]);
                }
                quant.quantize(&normed, n, d, 1e-5);
            })
        });
        let (cosine, sine) = dense::rope_tables(256, d / heads, 10_000.0);
        let mut rotated = q.clone();
        let half = d / heads / 2;
        group.bench_with_input(BenchmarkId::new("rope", d), &(), |b, _| {
            b.iter(|| {
                for row in 0..n {
                    for head in 0..heads {
                        let at = row * d + head * 2 * half;
                        dense::rope(&mut rotated[at..at + 2 * half], &cosine[row * half..(row + 1) * half], &sine[row * half..(row + 1) * half]);
                    }
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, pieces);
criterion_main!(benches);
