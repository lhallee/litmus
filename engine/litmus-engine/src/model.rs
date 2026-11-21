//! The ternary encoder (SPEC.md, section 3) and its streaming runner (section 5): each layer holds only its open
//! block, so a document of any length reads in fixed memory, one chunk at a time, and each chunk's running sums for
//! its pooled vector and its windows come out of the same pass (section 4).

use crate::Error;
use crate::chunks::ChunkPool;
use crate::dense::{self, AttentionScratch};
use crate::integer_attention::{self, IntegerScratch};
use crate::kernels::{self, Backend, ColumnPairs, KernelScratch, PackedTernary, QuantRows};
use crate::release::{Attention, ModelConfig, Safetensors};
use crate::surprise::{LanguageHead, LanguageScratch};

/// Tokens embedded at a time.
const EMBED_BATCH: usize = 256;
/// Tokens per pass through the feed-forward layers, which keeps the `(n, f)` hidden rows in L2.
const FFN_BATCH: usize = 64;

struct Layer {
    attn_norm: Vec<f32>,
    q: PackedTernary,
    k: PackedTernary,
    v: PackedTernary,
    attn_out_norm: Vec<f32>,
    o: PackedTernary,
    ffn_norm: Vec<f32>,
    up: PackedTernary,
    ffn_out_norm: Vec<f32>,
    down: PackedTernary,
    /// `down` again by column pairs: its inputs are squared-ReLU outputs, mostly zero in a trained model.
    down_columns: ColumnPairs,
}

/// Which path the feed-forward's `down` product takes per token (`kernels::matmul_f32_columns`). Every choice gives
/// the same bits; only the time differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColumnChoice {
    /// The column path below the measured limit (`kernels::column_limit`), else the dense path.
    #[default]
    Measured,
    /// The dense path for every token.
    Dense,
    /// The column path for every token.
    Columns,
    /// The column path when at most this percent of the input pairs are nonzero (for measurement).
    Percent(usize),
}

/// The encoder's weights.
pub struct Encoder {
    pub vocab: usize,
    pub embed_width: usize,
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub ffn_width: usize,
    pub block_size: usize,
    /// Float attention, or the integer attention of a format 2 release.
    pub attention: Attention,
    /// The `down` product's path per token.
    pub column_choice: ColumnChoice,
    norm_eps: f32,
    act_eps: f32,
    embed: Vec<i8>,        // (V, e)
    embed_scale: Vec<f32>, // (V,)
    proj: Option<(Vec<f32>, PackedTernary)>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>, // (d,)
    cosine: Vec<f32>,     // (W, d_h / 2)
    sine: Vec<f32>,       // (W, d_h / 2)
    /// The masked language head of the surprise reading, when the release has one.
    pub(crate) language: Option<LanguageHead>,
}

fn ternary(tensors: &Safetensors, name: &str, rows: usize, cols: usize) -> Result<PackedTernary, Error> {
    let packed = tensors.u8s(&format!("{name}.weight"), &[rows, cols / 4])?;
    let scale = tensors.f32s(&format!("{name}.scale"), &[1])?;
    PackedTernary::new(rows, cols, packed, scale[0]).map_err(|error| Error::Load(format!("{name}: {error}")))
}

impl Encoder {
    pub fn load(config: &ModelConfig, tensors: &Safetensors) -> Result<Self, Error> {
        let (v, e, d, f) = (config.vocab_size, config.embed_width, config.width, config.ffn_width);
        let proj = if e != d {
            Some((tensors.f32s("embed.proj_norm", &[e])?, ternary(tensors, "embed.proj", d, e)?))
        } else {
            None
        };
        let layers = (0..config.depth)
            .map(|i| {
                let p = format!("layers.{i}.");
                let down = ternary(tensors, &format!("{p}ffn.down"), d, f)?;
                Ok(Layer {
                    attn_norm: tensors.f32s(&format!("{p}attn_norm"), &[d])?,
                    q: ternary(tensors, &format!("{p}attn.q"), d, d)?,
                    k: ternary(tensors, &format!("{p}attn.k"), d, d)?,
                    v: ternary(tensors, &format!("{p}attn.v"), d, d)?,
                    attn_out_norm: tensors.f32s(&format!("{p}attn_out_norm"), &[d])?,
                    o: ternary(tensors, &format!("{p}attn.o"), d, d)?,
                    ffn_norm: tensors.f32s(&format!("{p}ffn_norm"), &[d])?,
                    up: ternary(tensors, &format!("{p}ffn.up"), f, d)?,
                    ffn_out_norm: tensors.f32s(&format!("{p}ffn_out_norm"), &[f])?,
                    down_columns: ColumnPairs::new(&down),
                    down,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let (cosine, sine) = dense::rope_tables(config.block_size, d / config.heads, config.rope_base);
        Ok(Self {
            vocab: v,
            embed_width: e,
            width: d,
            depth: config.depth,
            heads: config.heads,
            ffn_width: f,
            block_size: config.block_size,
            attention: config.attention(),
            column_choice: ColumnChoice::Measured,
            norm_eps: config.norm_eps as f32,
            act_eps: config.act_eps as f32,
            embed: tensors.i8s("embed.weight", &[v, e])?,
            embed_scale: tensors.f32s("embed.scale", &[v])?,
            proj,
            layers,
            final_norm: tensors.f32s("final_norm", &[d])?,
            cosine,
            sine,
            language: LanguageHead::load(tensors, v, e, d)?,
        })
    }

    /// Whether the release has the masked language head of the surprise reading.
    pub fn has_language(&self) -> bool {
        self.language.is_some()
    }

    /// Bytes of weights held.
    pub fn bytes(&self) -> usize {
        let ternary_bytes: usize = self
            .layers
            .iter()
            .map(|l| l.q.bytes() + l.k.bytes() + l.v.bytes() + l.o.bytes() + l.up.bytes() + l.down.bytes() + l.down_columns.bytes())
            .sum::<usize>()
            + self.proj.as_ref().map_or(0, |(_, p)| p.bytes());
        let floats: usize = self
            .layers
            .iter()
            .map(|l| l.attn_norm.len() + l.attn_out_norm.len() + l.ffn_norm.len() + l.ffn_out_norm.len())
            .sum::<usize>()
            + self.final_norm.len()
            + self.embed_scale.len()
            + self.cosine.len()
            + self.sine.len();
        self.embed.len() + ternary_bytes + 4 * floats
    }

    /// Ternary additions per token (an upper bound: zero weights are skipped).
    pub fn adds_per_token(&self) -> usize {
        let (d, f) = (self.width, self.ffn_width);
        self.depth * (4 * d * d + 2 * d * f) + self.proj.as_ref().map_or(0, |_| self.embed_width * d)
    }

    fn shift(&self, layer: usize) -> usize {
        if layer % 2 == 0 { 0 } else { self.block_size / 2 }
    }
}

/// Buffers of one block's arithmetic, reused for every block.
#[derive(Default)]
pub struct BlockScratch {
    normed: Vec<f32>,
    quant: QuantRows,
    kernel: KernelScratch,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    hidden: Vec<f32>,
    raw_embed: Vec<f32>,
    attention: AttentionScratch,
    integer: IntegerScratch,
}

impl BlockScratch {
    fn ensure(&mut self, encoder: &Encoder) {
        let rows = encoder.block_size.max(EMBED_BATCH);
        let wide = rows * encoder.width.max(encoder.embed_width);
        for buffer in [&mut self.q, &mut self.k, &mut self.v, &mut self.o] {
            buffer.resize(encoder.block_size * encoder.width, 0.0);
        }
        self.normed.resize(wide.max(FFN_BATCH * encoder.ffn_width), 0.0);
        self.hidden.resize(FFN_BATCH * encoder.ffn_width, 0.0);
        if encoder.proj.is_some() {
            self.raw_embed.resize(EMBED_BATCH * encoder.embed_width, 0.0);
        }
    }

    pub fn bytes(&self) -> usize {
        4 * (self.normed.capacity()
            + self.q.capacity()
            + self.k.capacity()
            + self.v.capacity()
            + self.o.capacity()
            + self.hidden.capacity()
            + self.raw_embed.capacity())
            + self.kernel.bytes()
            + self.attention.bytes()
            + self.integer.bytes()
            + self.quant.stride() * self.quant.n()
    }
}

/// `y = BitLinear(x)`: RMS norm rows with `gain`, quantize, ternary product; written or added into `out`.
#[allow(clippy::too_many_arguments)]
fn bitlinear(
    backend: Backend,
    x: &[f32],
    n: usize,
    gain: &[f32],
    weights: &PackedTernary,
    out: &mut [f32],
    accumulate: bool,
    norm_eps: f32,
    act_eps: f32,
    normed: &mut [f32],
    quant: &mut QuantRows,
    kernel: &mut KernelScratch,
) {
    let cols = weights.cols();
    timed!(Norm, {
        for row in 0..n {
            dense::rms_norm(&x[row * cols..(row + 1) * cols], gain, norm_eps, &mut normed[row * cols..(row + 1) * cols]);
        }
    });
    timed!(Quantize, quant.quantize(&normed[..n * cols], n, cols, act_eps));
    timed!(Ternary, kernels::matmul_f32(backend, weights, quant, out, weights.rows(), accumulate, kernel));
}

/// Embed token ids: `(n,)` to `(n, d)` rows of `out`.
fn embed(encoder: &Encoder, backend: Backend, ids: &[u32], out: &mut [f32], scratch: &mut BlockScratch) {
    let e = encoder.embed_width;
    let n = ids.len();
    let target: &mut [f32] = if encoder.proj.is_some() { &mut scratch.raw_embed } else { out };
    assert!(target.len() >= n * e, "embedding buffers hold a batch");
    timed!(Embed, {
        for (row, &id) in ids.iter().enumerate() {
            let id = id as usize;
            assert!(id < encoder.vocab, "token id {id} outside the vocabulary");
            let scale = encoder.embed_scale[id];
            for (value, &weight) in target[row * e..(row + 1) * e].iter_mut().zip(&encoder.embed[id * e..(id + 1) * e]) {
                *value = weight as f32 * scale;
            }
        }
    });
    if let Some((gain, proj)) = &encoder.proj {
        let BlockScratch { normed, quant, kernel, raw_embed, .. } = scratch;
        bitlinear(backend, &raw_embed[..n * e], n, gain, proj, out, false, encoder.norm_eps, encoder.act_eps, normed, quant, kernel);
    }
}

/// Layer `index` on one of its blocks, in place: `x` is `(n, d)`, its first row at position `first`.
fn layer_block(encoder: &Encoder, backend: Backend, index: usize, x: &mut [f32], first: usize, scratch: &mut BlockScratch) {
    let layer = &encoder.layers[index];
    let (d, f, h) = (encoder.width, encoder.ffn_width, encoder.heads);
    let d_h = d / h;
    let half = d_h / 2;
    let n = x.len() / d;
    let (w, shift) = (encoder.block_size, encoder.shift(index));
    debug_assert_eq!((first + shift) / w, (first + n - 1 + shift) / w, "a block never crosses a block boundary");
    let (norm_eps, act_eps) = (encoder.norm_eps, encoder.act_eps);
    let BlockScratch { normed, quant, kernel, q, k, v, o, hidden, attention, integer, .. } = scratch;

    // Attention.
    timed!(Norm, {
        for row in 0..n {
            dense::rms_norm(&x[row * d..(row + 1) * d], &layer.attn_norm, norm_eps, &mut normed[row * d..(row + 1) * d]);
        }
    });
    timed!(Quantize, quant.quantize(&normed[..n * d], n, d, act_eps));
    timed!(Ternary, {
        kernels::matmul_f32(backend, &layer.q, quant, &mut q[..n * d], d, false, kernel);
        kernels::matmul_f32(backend, &layer.k, quant, &mut k[..n * d], d, false, kernel);
        kernels::matmul_f32(backend, &layer.v, quant, &mut v[..n * d], d, false, kernel);
    });
    timed!(Rotary, {
        for row in 0..n {
            let r = (first + row + shift) % w;
            let (cosine, sine) = (&encoder.cosine[r * half..(r + 1) * half], &encoder.sine[r * half..(r + 1) * half]);
            for head in 0..h {
                let at = row * d + head * d_h;
                dense::rope(&mut q[at..at + d_h], cosine, sine);
                dense::rope(&mut k[at..at + d_h], cosine, sine);
            }
        }
    });
    timed!(Attention, match encoder.attention {
        Attention::Float => dense::block_attention(backend, &q[..n * d], &k[..n * d], &v[..n * d], &mut o[..n * d], n, d, h, attention),
        Attention::Integer => {
            integer_attention::block_attention(backend, &q[..n * d], &k[..n * d], &v[..n * d], &mut o[..n * d], n, d, h, act_eps, integer)
        }
    });
    bitlinear(backend, &o[..n * d], n, &layer.attn_out_norm, &layer.o, x, true, norm_eps, act_eps, normed, quant, kernel);

    // Feed-forward, a few tokens at a time.
    let mut start = 0;
    while start < n {
        let rows = (n - start).min(FFN_BATCH);
        let xs = &mut x[start * d..(start + rows) * d];
        bitlinear(backend, xs, rows, &layer.ffn_norm, &layer.up, &mut hidden[..rows * f], false, norm_eps, act_eps, normed, quant, kernel);
        timed!(Activation, {
            for value in hidden[..rows * f].iter_mut() {
                let r = if *value > 0.0 { *value } else { 0.0 };
                *value = r * r;
            }
        });
        // `down` with the per-token choice of the column path: the same bits as `bitlinear`.
        timed!(Norm, {
            for row in 0..rows {
                dense::rms_norm(&hidden[row * f..(row + 1) * f], &layer.ffn_out_norm, norm_eps, &mut normed[row * f..(row + 1) * f]);
            }
        });
        timed!(Quantize, quant.quantize(&normed[..rows * f], rows, f, act_eps));
        let below = match encoder.column_choice {
            ColumnChoice::Measured => kernels::column_limit(backend, &layer.down_columns) + 1,
            ColumnChoice::Dense => 0,
            ColumnChoice::Columns => f / 2 + 1,
            ColumnChoice::Percent(percent) => f / 2 * percent / 100 + 1,
        };
        let _column_tokens = timed!(Ternary, kernels::matmul_f32_columns(backend, &layer.down, &layer.down_columns, quant, xs, d, true, below, kernel));
        #[cfg(feature = "profile")]
        {
            let zeros: usize = (0..rows).map(|t| quant.values(t).iter().filter(|&&value| value == 0).count()).sum();
            crate::profile::count_down(index, zeros as u64, (rows * f) as u64, rows as u64, _column_tokens as u64);
        }
        start += rows;
    }
}

/// The blocks of layer `index` over a chunk of `n` tokens, as position spans `[start, end)` in order: a block ends
/// where `Stream::push` closes one, at every position whose shifted place is a multiple of the block size.
fn layer_blocks(encoder: &Encoder, index: usize, n: usize) -> Vec<(usize, usize)> {
    let (w, shift) = (encoder.block_size, encoder.shift(index));
    let mut spans = Vec::with_capacity(n / w + 2);
    let mut start = 0;
    while start < n {
        let end = (start + w - (start + shift) % w).min(n);
        spans.push((start, end));
        start = end;
    }
    spans
}

/// Rows `(n, d)` cut into the given consecutive spans of positions: each span's first position and its rows.
fn split_rows<'a>(mut x: &'a mut [f32], spans: &[(usize, usize)], d: usize) -> Vec<(usize, &'a mut [f32])> {
    let mut pieces = Vec::with_capacity(spans.len());
    for &(start, end) in spans {
        let (head, tail) = x.split_at_mut((end - start) * d);
        pieces.push((start, head));
        x = tail;
    }
    pieces
}

/// Run `work` on every item, on as many threads as there are scratches (the caller's thread is one of them), each
/// thread with a scratch of its own. Items are taken in order, one at a time, by whichever thread is free.
#[cfg(not(target_arch = "wasm32"))]
fn fan_out<T: Send>(items: Vec<T>, scratches: &mut [BlockScratch], work: impl Fn(T, &mut BlockScratch) + Sync) {
    let threads = scratches.len().min(items.len());
    let (mine, others) = scratches.split_first_mut().expect("a scratch for the caller's thread");
    let queue = std::sync::Mutex::new(items.into_iter());
    let take = || queue.lock().expect("no thread panics holding the queue").next();
    std::thread::scope(|scope| {
        for scratch in others.iter_mut().take(threads.saturating_sub(1)) {
            let (take, work) = (&take, &work);
            scope.spawn(move || {
                while let Some(item) = take() {
                    work(item, scratch);
                }
            });
        }
        while let Some(item) = take() {
            work(item, mine);
        }
    });
}

/// A whole chunk through the encoder with its blocks spread over threads, one `BlockScratch` a thread: the token
/// states `(n, d)` are embedded into `x`, every layer runs over its blocks in place, and the final norm gives the
/// states the pool sums in position order. Each block is the arithmetic of `Stream::close` on the same rows, and a
/// block does not read another's, so the pool is the stream's bit for bit on any number of threads. The states take
/// `4 n d` bytes, where the stream holds one block a layer. `after_stage` is called after the embedding (0) and
/// after each layer (1 to the depth) and stops the run, returning `None`, when it returns false.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn run_chunk(
    encoder: &Encoder,
    backend: Backend,
    ids: &[u32],
    window_width: usize,
    window_stride: usize,
    x: &mut Vec<f32>,
    scratches: &mut [BlockScratch],
    after_stage: &mut dyn FnMut(usize) -> bool,
) -> Option<ChunkPool> {
    let (n, d) = (ids.len(), encoder.width);
    assert!(n > 0, "a chunk holds at least one token");
    let batches: Vec<(usize, usize)> = (0..n).step_by(EMBED_BATCH).map(|start| (start, (start + EMBED_BATCH).min(n))).collect();
    let layers: Vec<Vec<(usize, usize)>> = (0..encoder.depth).map(|index| layer_blocks(encoder, index, n)).collect();
    // No stage has more pieces than this, so no more threads than this find work.
    let widest = layers.iter().map(Vec::len).max().unwrap_or(1).max(batches.len());
    let threads = scratches.len().min(widest).max(1);
    let scratches = &mut scratches[..threads];
    scratches.iter_mut().for_each(|scratch| scratch.ensure(encoder));
    x.clear();
    x.resize(n * d, 0.0);
    fan_out(split_rows(x, &batches, d), scratches, |(first, rows), scratch| {
        let count = rows.len() / d;
        embed(encoder, backend, &ids[first..first + count], rows, scratch);
    });
    if !after_stage(0) {
        return None;
    }
    for (index, blocks) in layers.iter().enumerate() {
        fan_out(split_rows(x, blocks, d), scratches, |(first, rows), scratch| layer_block(encoder, backend, index, rows, first, scratch));
        if !after_stage(index + 1) {
            return None;
        }
    }
    let mut pool = ChunkPool::new(d, n, window_width, window_stride);
    let mut state = vec![0f32; d];
    for row in x.chunks_exact(d) {
        timed!(Norm, dense::rms_norm(row, &encoder.final_norm, encoder.norm_eps, &mut state));
        timed!(Pool, pool.add(&state));
    }
    Some(pool)
}

/// The first rows of every stage of a document's first chunk, for parity tests.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub keep: usize,
    /// After the embedding, then after each layer: `depth + 1` lists of `keep` rows of width `d`.
    pub layer_states: Vec<Vec<Vec<f32>>>,
    /// The final token states: `keep` rows.
    pub token_states: Vec<Vec<f32>>,
}

/// One chunk at a time streaming through the encoder: each layer holds only its open block, and every chunk starts
/// from a fresh state, as a text of its own.
pub struct Stream<'e> {
    encoder: &'e Encoder,
    backend: Backend,
    window_width: usize,
    window_stride: usize,
    open: Vec<Vec<f32>>, // per layer: (W, d)
    open_rows: Vec<usize>,
    open_start: Vec<usize>,
    received: Vec<usize>,
    pending: Vec<u32>,
    embedded: Vec<f32>, // (EMBED_BATCH, d)
    states: Vec<f32>,   // (W, d) final token states of one block
    scratch: BlockScratch,
    pool: ChunkPool,
    trace: Option<Trace>,
    kept: Option<Trace>,
    fed: usize,
    /// The surprise pass's buffers and running sum, when the stream reads it.
    language: Option<LanguageScratch>,
}

impl<'e> Stream<'e> {
    /// A stream whose first chunk has `length` tokens; `keep` rows of the first chunk's stages are traced.
    pub fn new(encoder: &'e Encoder, backend: Backend, length: usize, window_width: usize, window_stride: usize, keep: usize) -> Self {
        let (d, w) = (encoder.width, encoder.block_size);
        let mut scratch = BlockScratch::default();
        scratch.ensure(encoder);
        let trace = (keep > 0).then(|| Trace { keep, layer_states: vec![Vec::new(); encoder.depth + 1], token_states: Vec::new() });
        Self {
            encoder,
            backend,
            window_width,
            window_stride,
            open: (0..encoder.depth).map(|_| vec![0.0; w * d]).collect(),
            open_rows: vec![0; encoder.depth],
            open_start: vec![0; encoder.depth],
            received: vec![0; encoder.depth],
            pending: Vec::with_capacity(EMBED_BATCH),
            embedded: vec![0.0; EMBED_BATCH * d],
            states: vec![0.0; w * d],
            scratch,
            pool: ChunkPool::new(d, length, window_width, window_stride),
            trace,
            kept: None,
            fed: 0,
            language: None,
        }
    }

    /// Also read every token's log probability under the masked language head (SPEC.md, Surprise); chunks hold at
    /// most `chunk_tokens` tokens.
    pub fn with_surprise(mut self, chunk_tokens: usize) -> Self {
        assert!(self.encoder.has_language(), "the surprise reading needs the release's masked language head");
        let encoder = self.encoder;
        self.language = Some(LanguageScratch::new(encoder.block_size, encoder.embed_width, encoder.vocab, chunk_tokens));
        self
    }

    /// The sum of the log probabilities read so far, and their number (`None` without `with_surprise`).
    pub fn surprise_sum(&self) -> Option<(f64, usize)> {
        self.language.as_ref().map(|language| (language.sum, language.count))
    }

    /// Bytes held by the stream's buffers and its chunk's sums (not the weights).
    pub fn bytes(&self) -> usize {
        4 * (self.open.iter().map(Vec::capacity).sum::<usize>() + self.embedded.capacity() + self.states.capacity())
            + self.scratch.bytes()
            + self.pool.bytes()
            + self.language.as_ref().map_or(0, LanguageScratch::bytes)
    }

    /// Tokens of the current chunk fed so far.
    pub fn tokens(&self) -> usize {
        self.fed
    }

    pub fn feed(&mut self, id: u32) {
        assert!(self.fed < self.pool.length, "fed more tokens than the chunk holds");
        self.fed += 1;
        self.pending.push(id);
        if let Some(language) = &mut self.language {
            language.ids.push(id);
        }
        if self.pending.len() == EMBED_BATCH {
            self.flush_pending();
        }
    }

    pub fn feed_all(&mut self, ids: &[u32]) {
        for &id in ids {
            self.feed(id);
        }
    }

    /// Close every open block and return the finished chunk's sums; the stream then reads a fresh chunk of
    /// `next_length` tokens (0 when none follows).
    pub fn finish_chunk(&mut self, next_length: usize) -> Result<ChunkPool, Error> {
        self.flush_pending();
        for index in 0..self.encoder.depth {
            if self.open_rows[index] > 0 {
                self.close(index);
            }
        }
        if self.pool.count == 0 {
            return Err(Error::Empty);
        }
        assert_eq!(self.pool.count, self.pool.length, "every token of the chunk reached the pool");
        if self.kept.is_none() {
            self.kept = self.trace.take();
        }
        self.received.fill(0);
        self.open_start.fill(0);
        self.fed = 0;
        if let Some(language) = &mut self.language {
            assert_eq!(language.ids.len(), self.pool.length, "every token of the chunk reached the language head");
            language.ids.clear();
        }
        let next = ChunkPool::new(self.encoder.width, next_length, self.window_width, self.window_stride);
        Ok(std::mem::replace(&mut self.pool, next))
    }

    /// The first chunk's traced rows, once it is finished.
    pub fn take_trace(&mut self) -> Option<Trace> {
        self.kept.take()
    }

    fn flush_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let n = self.pending.len();
        let d = self.encoder.width;
        let mut embedded = std::mem::take(&mut self.embedded);
        embed(self.encoder, self.backend, &self.pending, &mut embedded[..n * d], &mut self.scratch);
        self.pending.clear();
        self.record(0, self.received[0], &embedded[..n * d]);
        self.push(0, &embedded[..n * d]);
        self.embedded = embedded;
    }

    fn record(&mut self, stage: usize, first: usize, rows: &[f32]) {
        let d = self.encoder.width;
        if let Some(trace) = &mut self.trace {
            for (offset, row) in rows.chunks_exact(d).enumerate() {
                if first + offset < trace.keep {
                    if stage <= self.encoder.depth {
                        trace.layer_states[stage].push(row.to_vec());
                    } else {
                        trace.token_states.push(row.to_vec());
                    }
                }
            }
        }
    }

    /// The next rows `(c, d)` of layer `index`'s input, in position order.
    fn push(&mut self, index: usize, x: &[f32]) {
        let (d, w) = (self.encoder.width, self.encoder.block_size);
        let shift = self.encoder.shift(index);
        let c = x.len() / d;
        let mut first = 0;
        while first < c {
            let position = self.received[index];
            if self.open_rows[index] > 0 && (position + shift) % w == 0 {
                self.close(index);
            }
            let room = w - (position + shift) % w;
            let take = room.min(c - first);
            if self.open_rows[index] == 0 {
                self.open_start[index] = position;
            }
            let at = self.open_rows[index] * d;
            self.open[index][at..at + take * d].copy_from_slice(&x[first * d..(first + take) * d]);
            self.open_rows[index] += take;
            self.received[index] += take;
            first += take;
        }
    }

    fn close(&mut self, index: usize) {
        let d = self.encoder.width;
        let rows = self.open_rows[index];
        let start = self.open_start[index];
        let mut block = std::mem::take(&mut self.open[index]);
        self.open_rows[index] = 0;
        layer_block(self.encoder, self.backend, index, &mut block[..rows * d], start, &mut self.scratch);
        self.record(index + 1, start, &block[..rows * d]);
        if index + 1 < self.encoder.depth {
            self.push(index + 1, &block[..rows * d]);
        } else {
            let mut states = std::mem::take(&mut self.states);
            timed!(Norm, {
                for row in 0..rows {
                    dense::rms_norm(&block[row * d..(row + 1) * d], &self.encoder.final_norm, self.encoder.norm_eps, &mut states[row * d..(row + 1) * d]);
                }
            });
            self.record(self.encoder.depth + 1, start, &states[..rows * d]);
            assert_eq!(start, self.pool.count, "token states reach the pool in position order");
            timed!(Pool, {
                for row in states[..rows * d].chunks_exact(d) {
                    self.pool.add(row);
                }
            });
            if let Some(language) = &mut self.language {
                let encoder = self.encoder;
                let head = encoder.language.as_ref().expect("a surprise stream has the language head");
                let e = encoder.embed_width;
                let BlockScratch { normed, quant, kernel, .. } = &mut self.scratch;
                let (norm_eps, act_eps) = (encoder.norm_eps, encoder.act_eps);
                bitlinear(self.backend, &states[..rows * d], rows, &head.proj_norm, &head.proj, &mut language.projected[..rows * e], false, norm_eps, act_eps, normed, quant, kernel);
                for row in 0..rows {
                    dense::rms_norm(&language.projected[row * e..(row + 1) * e], &head.norm, norm_eps, &mut language.normed[row * e..(row + 1) * e]);
                }
                let ids = std::mem::take(&mut language.ids);
                language.add(head, &encoder.embed, &encoder.embed_scale, &ids[start..start + rows]);
                language.ids = ids;
            }
            self.states = states;
        }
        self.open[index] = block;
    }
}
