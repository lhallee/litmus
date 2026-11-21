//! Litmus: a detector of model-written text that runs a ternary (BitNet b1.58) encoder on a CPU by addition.
//!
//! The library reads a release (`model.safetensors`, `config.json`) and ModernBERT's `tokenizer.json` from bytes,
//! streams a document of any length through the encoder in fixed memory, one chunk of at most `chunk_tokens` tokens
//! at a time, and returns the document's verdict, each chunk's reading, and the heat map's windows and cells,
//! uncertainty included (`../student/SPEC.md` is the contract). It uses no file system and no clock, and `score`
//! reads on the calling thread alone, so it runs unchanged in the browser. `Engine::score_threads` gives the same
//! reading, bit for bit, with a chunk's blocks spread over scoped threads of the standard library.
//!
//! ```no_run
//! # fn main() -> Result<(), litmus_engine::Error> {
//! # let (model, config, tokenizer) = (vec![], vec![], vec![]);
//! let engine = litmus_engine::Engine::load(&model, &config, &tokenizer)?;
//! let document = engine.score("Some text to read.")?;
//! println!("{}, chance of AI {}", document.label().word(), engine.chance(&document).words());
//! # Ok(()) }
//! ```

/// `timed!(Stage, expression)`: the expression, its time added to the stage under the `profile` feature.
macro_rules! timed {
    ($stage:ident, $body:expr) => {{
        #[cfg(feature = "profile")]
        let started = std::time::Instant::now();
        let value = $body;
        #[cfg(feature = "profile")]
        $crate::profile::add($crate::profile::Stage::$stage, started.elapsed().as_nanos() as u64);
        value
    }};
}

pub mod chunks;
pub mod codes;
pub mod dense;
#[cfg(target_arch = "aarch64")]
mod dense_neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod dense_simd128;
pub mod integer_attention;
pub mod kernels;
pub mod model;
#[cfg(feature = "profile")]
pub mod profile;
pub mod reading;
pub mod release;
pub mod surprise;
pub mod text;
pub mod tokenizer;
mod unicode14;

use std::fmt;

pub use kernels::Backend;
pub use reading::{Cell, Chance, ChanceRule, ChunkReading, ClosenessCount, ConformalCount, DocumentReading, Drawn, Explain, Label, NearestEntry, Origin, Place, Reading, Strip, Verdict, WindowReading};
pub use release::{MIXED_LEVEL, STRIP_LEVEL};
pub use surprise::Surprise;
pub use tokenizer::Token;

use chunks::{ChunkPool, cell_spans, chunk_bounds, window_spans};
use model::{Encoder, Stream, Trace};
use reading::{Bank, ChunkCall, Explainer, Head, HeadVectors, Reader};
use release::{CalibrationTable, ReleaseConfig, Safetensors};
use tokenizer::Tokenizer;

/// What can go wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The release or the tokenizer could not be read; the message says why.
    Load(String),
    /// The text has no tokens, so no reading.
    Empty,
    /// The progress callback asked to stop.
    Cancelled,
    /// The backend is not available on this processor.
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Load(message) => write!(f, "cannot load: {message}"),
            Error::Empty => write!(f, "the text has no tokens, so no reading"),
            Error::Cancelled => write!(f, "cancelled"),
            Error::Backend(name) => write!(f, "backend {name} is not available on this processor"),
        }
    }
}

impl std::error::Error for Error {}

/// Progress of one scoring call. The text is read twice: first to count its tokens (`tokens_total` and
/// `chunks_total` are 0 until that is done), then through the encoder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    /// Bytes of the original text read so far in this pass, and in all.
    pub bytes_done: usize,
    pub bytes_total: usize,
    /// Tokens read so far in this pass, and in all.
    pub tokens_done: usize,
    pub tokens_total: usize,
    /// Chunks read, and in all.
    pub chunks_done: usize,
    pub chunks_total: usize,
}

/// What a progress callback asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Cancel,
}

/// A summary of the loaded release.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelInfo {
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub ffn_width: usize,
    pub embed_width: usize,
    pub block_size: usize,
    /// `float`, or `integer` for a format 2 release in SPEC.md's integer attention.
    pub attention: release::Attention,
    /// `float`, or `int8` for a format 3 release in SPEC.md's int8 output: the head's int8 codes and an int8 bank.
    pub output: release::Output,
    /// The chunk limit `C_max`, and the heat map's window width and stride.
    pub chunk_tokens: usize,
    pub window_width: usize,
    pub window_stride: usize,
    pub bank: [usize; 3],
    pub levels: Vec<f64>,
    pub strip_level: f64,
    pub mixed_level: f64,
    /// What a reading's `chance` takes from the calibration table: per length bin, the share of not-human texts among
    /// the calibration texts behind the probability, and the half width between sources (SPEC.md, Chance of AI).
    pub chance_rule: ChanceRule,
    /// Ternary additions per token (upper bound).
    pub adds_per_token: usize,
    /// Bytes held by the weights, the head and the bank.
    pub bytes: usize,
    /// Whether the release has the masked language head, so `surprise` reads (SPEC.md, Surprise).
    pub surprise: bool,
    pub provenance: serde_json::Value,
    /// The release's license (SPEC.md): PolyForm-Noncommercial-1.0.0.
    pub license: Option<String>,
}

/// A loaded release and tokenizer. `Send` and `Sync`: one engine can score on several threads at once (each call
/// keeps its own buffers; the tokenizer's word cache is shared behind a lock).
pub struct Engine {
    encoder: Encoder,
    head: Head,
    bank: Bank,
    calibration: CalibrationTable,
    explainer: Explainer,
    tokenizer: Tokenizer,
    backend: Backend,
    info: ModelInfo,
    surprise_table: Option<surprise::SurpriseTable>,
}

impl Engine {
    /// Load a release from the bytes of `model.safetensors`, `config.json` and `tokenizer.json`, with the fastest
    /// backend this processor has.
    pub fn load(model: &[u8], config: &[u8], tokenizer: &[u8]) -> Result<Self, Error> {
        let config = ReleaseConfig::parse(config)?;
        let calibration = config.calibration.clone().ok_or_else(|| Error::Load("config.json has no calibration table".into()))?;
        let tensors = Safetensors::parse(model)?;
        let encoder = Encoder::load(&config.model, &tensors)?;
        let output = config.model.output();
        let head = Head::load(&tensors, config.model.width, config.head.hidden, config.head.out_width, config.head.clamp, output, config.model.act_eps as f32)?;
        let bank = Bank::load(&tensors, config.head.out_width, &config.bank.fields, output)?;
        let explainer = Explainer::load(&tensors, &calibration)?;
        let tokenizer = Tokenizer::from_json(tokenizer)?;
        if tokenizer.vocab_size() > config.model.vocab_size {
            return Err(Error::Load(format!(
                "the tokenizer has {} entries, the model's vocabulary {}",
                tokenizer.vocab_size(),
                config.model.vocab_size
            )));
        }
        let info = ModelInfo {
            width: encoder.width,
            depth: encoder.depth,
            heads: encoder.heads,
            ffn_width: encoder.ffn_width,
            embed_width: encoder.embed_width,
            block_size: encoder.block_size,
            attention: encoder.attention,
            output,
            chunk_tokens: config.reading.chunk_tokens,
            window_width: config.reading.window_width,
            window_stride: config.reading.window_stride,
            bank: bank.counts(),
            levels: calibration.levels.clone(),
            strip_level: release::STRIP_LEVEL,
            mixed_level: release::MIXED_LEVEL,
            chance_rule: calibration.chance_rule(),
            adds_per_token: encoder.adds_per_token(),
            bytes: encoder.bytes() + head.bytes() + bank.bytes() + explainer.bytes() + encoder.language.as_ref().map_or(0, surprise::LanguageHead::bytes),
            surprise: encoder.has_language(),
            license: config.license.clone(),
            provenance: config.provenance.clone(),
        };
        let surprise_table = config.surprise.clone();
        Ok(Self { encoder, head, bank, calibration, explainer, tokenizer, backend: Backend::detect(), info, surprise_table })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Use another backend (every backend gives the same integers; `Scalar` defines them).
    pub fn set_backend(&mut self, backend: Backend) -> Result<(), Error> {
        if !backend.is_available() {
            return Err(Error::Backend(backend.name().into()));
        }
        self.backend = backend;
        Ok(())
    }

    /// The path of the feed-forward's `down` product per token (the measured choice by default). Every choice gives
    /// the same bits; for tests and measurement.
    #[doc(hidden)]
    pub fn set_column_choice(&mut self, choice: model::ColumnChoice) {
        self.encoder.column_choice = choice;
    }

    /// The tokens of a text's standard reading, with byte offsets into the original text.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        self.tokenizer.stream_standard(text, &mut |token| tokens.push(token), &mut |_| true);
        tokens
    }

    /// The number of tokens `score` reads in a text.
    pub fn count_tokens(&self, text: &str) -> usize {
        let mut count = 0;
        self.tokenizer.stream_standard(text, &mut |_| count += 1, &mut |_| true);
        count
    }

    /// Token ids of a text as given, without the standard reading (as `tokenizers` encodes it).
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// The chance of AI of a document's verdict: its probability and interval at even odds (SPEC.md, Chance of AI).
    pub fn chance(&self, document: &DocumentReading) -> Chance {
        document.chance(&self.info.chance_rule)
    }

    /// Score a text.
    pub fn score(&self, text: &str) -> Result<DocumentReading, Error> {
        self.score_with(text, &mut |_| Flow::Continue)
    }

    /// Score a text, calling `progress` after every piece of about 8 KB of text; it may cancel. The text is read
    /// twice: once to count its tokens (which fixes the chunks), once through the encoder, one chunk at a time.
    pub fn score_with(&self, text: &str, progress: &mut dyn FnMut(Progress) -> Flow) -> Result<DocumentReading, Error> {
        let bytes_total = text.len();
        let counted = std::cell::Cell::new(0usize);
        let mut counting = |done: usize| {
            progress(Progress { bytes_done: done, bytes_total, tokens_done: counted.get(), ..Progress::default() }) == Flow::Continue
        };
        if !self.tokenizer.stream_standard(text, &mut |_| counted.set(counted.get() + 1), &mut counting) {
            return Err(Error::Cancelled);
        }
        let n = counted.get();
        if n == 0 {
            return Err(Error::Empty);
        }
        let plan = Plan::new(n, &self.info);
        let reader = self.reader();
        let mut stream = Stream::new(&self.encoder, self.backend, plan.bounds[0].1, self.info.window_width, self.info.window_stride, 0);
        let mut parts = Parts::default();
        let mut boundary_bytes = Vec::with_capacity(plan.boundaries.len());
        let (position, chunks_done) = (std::cell::Cell::new(0usize), std::cell::Cell::new(0usize));
        let mut failure = None;
        let completed = {
            let mut sink = |token: Token| {
                let p = position.get();
                if plan.boundaries.get(boundary_bytes.len()) == Some(&p) {
                    boundary_bytes.push(if p == 0 { 0 } else { token.start });
                }
                stream.feed(token.id);
                position.set(p + 1);
                let chunk = chunks_done.get();
                if p + 1 == plan.bounds[chunk].1 {
                    let next = plan.bounds.get(chunk + 1).map_or(0, |&(a, b)| b - a);
                    match stream.finish_chunk(next) {
                        Ok(pool) => self.read_chunk(&reader, &pool, plan.bounds[chunk].0, chunk, &mut parts, None),
                        Err(error) => failure = Some(error),
                    }
                    chunks_done.set(chunk + 1);
                }
            };
            let mut on_piece = |done: usize| {
                let report = Progress {
                    bytes_done: done,
                    bytes_total,
                    tokens_done: position.get(),
                    tokens_total: n,
                    chunks_done: chunks_done.get(),
                    chunks_total: plan.bounds.len(),
                };
                progress(report) == Flow::Continue
            };
            self.tokenizer.stream_standard(text, &mut sink, &mut on_piece)
        };
        if !completed {
            return Err(Error::Cancelled);
        }
        if let Some(error) = failure {
            return Err(error);
        }
        assert_eq!(position.get(), n, "the second reading of the text gives the same tokens");
        boundary_bytes.push(bytes_total); // the boundary at position n
        let mut document = self.document(n, parts);
        place(&mut document, &plan.boundaries, &boundary_bytes, text);
        Ok(document)
    }

    /// As `score_with`, on up to `threads` threads: each chunk's token states are held whole (`4 * width` bytes a
    /// token of the chunk, where `score_with` holds one block a layer) and every layer's blocks are spread over the
    /// threads (`model::run_chunk`). A block's arithmetic does not depend on the thread that runs it, so the reading
    /// is `score_with`'s bit for bit. One thread is `score_with` itself. `progress` is called while the text is
    /// tokenized (`tokens_total` 0), then after the embedding and after every layer of each chunk; it may cancel.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn score_threads(&self, text: &str, threads: usize, progress: &mut dyn FnMut(Progress) -> Flow) -> Result<DocumentReading, Error> {
        if threads <= 1 {
            return self.score_with(text, progress);
        }
        let bytes_total = text.len();
        let (mut ids, mut starts) = (Vec::new(), Vec::new());
        let read = std::cell::Cell::new(0usize);
        let completed = {
            let mut sink = |token: Token| {
                ids.push(token.id);
                starts.push(token.start);
                read.set(read.get() + 1);
            };
            let mut on_piece = |done: usize| progress(Progress { bytes_done: done, bytes_total, tokens_done: read.get(), ..Progress::default() }) == Flow::Continue;
            self.tokenizer.stream_standard(text, &mut sink, &mut on_piece)
        };
        if !completed {
            return Err(Error::Cancelled);
        }
        let n = ids.len();
        if n == 0 {
            return Err(Error::Empty);
        }
        let plan = Plan::new(n, &self.info);
        let chunks_total = plan.bounds.len();
        let mut document = self.score_chunks(&ids, &plan, threads, &mut |tokens_done, chunks_done| {
            let bytes_done = (bytes_total as u128 * tokens_done as u128 / n as u128) as usize;
            progress(Progress { bytes_done, bytes_total, tokens_done, tokens_total: n, chunks_done, chunks_total }) == Flow::Continue
        })?;
        // The byte offset of each boundary position, as `score_with` records them while it feeds the tokens.
        let boundary_bytes: Vec<usize> = plan.boundaries.iter().map(|&p| if p == 0 { 0 } else if p == n { bytes_total } else { starts[p] }).collect();
        place(&mut document, &plan.boundaries, &boundary_bytes, text);
        Ok(document)
    }

    /// `score_ids` on up to `threads` threads (parity tests): the same reading, bit for bit.
    #[cfg(not(target_arch = "wasm32"))]
    #[doc(hidden)]
    pub fn score_ids_threads(&self, ids: &[u32], threads: usize) -> Result<DocumentReading, Error> {
        if ids.is_empty() {
            return Err(Error::Empty);
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.encoder.vocab) {
            return Err(Error::Load(format!("token id {id} lies outside the vocabulary")));
        }
        self.score_chunks(ids, &Plan::new(ids.len(), &self.info), threads, &mut |_, _| true)
    }

    /// A document's reading from its token ids, every chunk through `model::run_chunk` on up to `threads` threads.
    /// `progress(tokens_done, chunks_done)` is called after each stage of each chunk and stops the scoring when it
    /// returns false. Places carry token positions only.
    #[cfg(not(target_arch = "wasm32"))]
    fn score_chunks(&self, ids: &[u32], plan: &Plan, threads: usize, progress: &mut dyn FnMut(usize, usize) -> bool) -> Result<DocumentReading, Error> {
        // The bank's pass over a chunk and its windows is spread over the same threads.
        let reader = Reader { threads: threads.max(1), ..self.reader() };
        let mut scratches: Vec<model::BlockScratch> = (0..threads.max(1)).map(|_| model::BlockScratch::default()).collect();
        let mut states = Vec::new();
        let mut parts = Parts::default();
        // The embedding and each layer are a stage: a chunk's tokens count as done in equal shares.
        let stages = self.encoder.depth + 1;
        for (chunk, &(start, end)) in plan.bounds.iter().enumerate() {
            let length = end - start;
            let mut after_stage = |stage: usize| progress(start + length * (stage + 1) / stages, chunk);
            let pool = model::run_chunk(&self.encoder, self.backend, &ids[start..end], self.info.window_width, self.info.window_stride, &mut states, &mut scratches, &mut after_stage)
                .ok_or(Error::Cancelled)?;
            self.read_chunk(&reader, &pool, start, chunk, &mut parts, None);
        }
        Ok(self.document(ids.len(), parts))
    }

    /// A text's surprise reading (SPEC.md, Surprise): its mean token log probability under the masked language head
    /// with nothing masked, every chunk read as a text of its own, in a pass of its own. `progress` is called as in
    /// `score_with` and may cancel; a release without the head gives `Error::Load`.
    pub fn surprise_with(&self, text: &str, progress: &mut dyn FnMut(Progress) -> Flow) -> Result<Surprise, Error> {
        if !self.encoder.has_language() {
            return Err(Error::Load("this release has no masked language head, so no surprise reading".into()));
        }
        let bytes_total = text.len();
        let counted = std::cell::Cell::new(0usize);
        let mut counting = |done: usize| {
            progress(Progress { bytes_done: done, bytes_total, tokens_done: counted.get(), ..Progress::default() }) == Flow::Continue
        };
        if !self.tokenizer.stream_standard(text, &mut |_| counted.set(counted.get() + 1), &mut counting) {
            return Err(Error::Cancelled);
        }
        let n = counted.get();
        if n == 0 {
            return Err(Error::Empty);
        }
        let bounds = chunk_bounds(n, self.info.chunk_tokens);
        let mut stream = self.surprise_stream(bounds[0].1);
        let (position, chunks_done) = (std::cell::Cell::new(0usize), std::cell::Cell::new(0usize));
        let mut failure = None;
        let completed = {
            let mut sink = |token: Token| {
                stream.feed(token.id);
                let p = position.get() + 1;
                position.set(p);
                let chunk = chunks_done.get();
                if p == bounds[chunk].1 {
                    let next = bounds.get(chunk + 1).map_or(0, |&(a, b)| b - a);
                    if let Err(error) = stream.finish_chunk(next) {
                        failure = Some(error);
                    }
                    chunks_done.set(chunk + 1);
                }
            };
            let mut on_piece = |done: usize| {
                let report = Progress {
                    bytes_done: done,
                    bytes_total,
                    tokens_done: position.get(),
                    tokens_total: n,
                    chunks_done: chunks_done.get(),
                    chunks_total: bounds.len(),
                };
                progress(report) == Flow::Continue
            };
            self.tokenizer.stream_standard(text, &mut sink, &mut on_piece)
        };
        if !completed {
            return Err(Error::Cancelled);
        }
        if let Some(error) = failure {
            return Err(error);
        }
        let (sum, count) = stream.surprise_sum().expect("a surprise stream keeps its sum");
        assert_eq!(count, n, "every token of the text has its log probability");
        Ok(Surprise::new(sum, n, self.surprise_table.as_ref()))
    }

    /// A text's surprise reading.
    pub fn surprise(&self, text: &str) -> Result<Surprise, Error> {
        self.surprise_with(text, &mut |_| Flow::Continue)
    }

    /// The surprise reading of token ids directly (parity tests).
    #[doc(hidden)]
    pub fn surprise_ids(&self, ids: &[u32]) -> Result<Surprise, Error> {
        if !self.encoder.has_language() {
            return Err(Error::Load("this release has no masked language head, so no surprise reading".into()));
        }
        if ids.is_empty() {
            return Err(Error::Empty);
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.encoder.vocab) {
            return Err(Error::Load(format!("token id {id} lies outside the vocabulary")));
        }
        let bounds = chunk_bounds(ids.len(), self.info.chunk_tokens);
        let mut stream = self.surprise_stream(bounds[0].1);
        for (chunk, &(start, end)) in bounds.iter().enumerate() {
            stream.feed_all(&ids[start..end]);
            let next = bounds.get(chunk + 1).map_or(0, |&(a, b)| b - a);
            stream.finish_chunk(next)?;
        }
        let (sum, count) = stream.surprise_sum().expect("a surprise stream keeps its sum");
        assert_eq!(count, ids.len(), "every token has its log probability");
        Ok(Surprise::new(sum, ids.len(), self.surprise_table.as_ref()))
    }

    fn surprise_stream(&self, first_length: usize) -> Stream<'_> {
        Stream::new(&self.encoder, self.backend, first_length, self.info.window_width, self.info.window_stride, 0).with_surprise(self.info.chunk_tokens)
    }

    /// Score token ids directly, fed in pieces of `piece` tokens (the contract's level; parity tests). Places carry
    /// token positions only; their byte and character offsets are zero.
    pub fn score_ids(&self, ids: &[u32], piece: usize) -> Result<DocumentReading, Error> {
        Ok(self.trace_ids(ids, piece, 0)?.0)
    }

    /// As `score_ids`, also returning each chunk's pooled vector, query, key and window vectors, and the first
    /// `keep` rows of every stage of the first chunk.
    #[doc(hidden)]
    pub fn trace_ids(&self, ids: &[u32], piece: usize, keep: usize) -> Result<(DocumentReading, Vectors), Error> {
        assert!(piece > 0, "pieces of at least one token");
        if ids.is_empty() {
            return Err(Error::Empty);
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.encoder.vocab) {
            return Err(Error::Load(format!("token id {id} lies outside the vocabulary")));
        }
        let plan = Plan::new(ids.len(), &self.info);
        let reader = self.reader();
        let mut stream = Stream::new(&self.encoder, self.backend, plan.bounds[0].1, self.info.window_width, self.info.window_stride, keep);
        let mut parts = Parts::default();
        let mut vectors = Vec::new();
        let (mut chunk, mut position) = (0, 0);
        for piece in ids.chunks(piece) {
            for &id in piece {
                stream.feed(id);
                position += 1;
                if position == plan.bounds[chunk].1 {
                    let next = plan.bounds.get(chunk + 1).map_or(0, |&(a, b)| b - a);
                    let pool = stream.finish_chunk(next)?;
                    self.read_chunk(&reader, &pool, plan.bounds[chunk].0, chunk, &mut parts, Some(&mut vectors));
                    chunk += 1;
                }
            }
        }
        let trace = stream.take_trace().unwrap_or_default();
        Ok((self.document(ids.len(), parts), Vectors { chunks: vectors, trace }))
    }

    /// Bytes held by the stream's buffers while scoring a full chunk (weights excluded).
    pub fn stream_bytes(&self) -> usize {
        Stream::new(&self.encoder, self.backend, self.info.chunk_tokens, self.info.window_width, self.info.window_stride, 0).bytes()
    }

    /// The reading pass alone (head and bank for a chunk and its windows, and its cells) on a chunk of `length`
    /// synthetic token states; returns the number of windows. For `litmus bench`, to time this pass apart from the
    /// encoder.
    #[doc(hidden)]
    pub fn read_pass(&self, length: usize) -> usize {
        let d = self.info.width;
        let mut pool = ChunkPool::new(d, length, self.info.window_width, self.info.window_stride);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut row = vec![0f32; d];
        for _ in 0..length {
            for value in row.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *value = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            pool.add(&row);
        }
        let mut parts = Parts::default();
        self.read_chunk(&self.reader(), &pool, 0, 0, &mut parts, None);
        parts.windows.len()
    }

    /// The head and the bank alone, on given pooled vectors `(v, 2d)` of spans of `n_tokens[i]` tokens: their readings
    /// (the first explained) and, in int8 output, the codes of their queries and keys `(v, w)` each (empty otherwise).
    /// For parity tests, which feed the reference's pooled vectors to isolate the integer compare from the encoder.
    #[doc(hidden)]
    pub fn read_pooled(&self, pooled: &[f32], n_tokens: &[usize]) -> (Vec<Reading>, Vec<i8>, Vec<i8>) {
        let (readings, vectors) = self.reader().read(pooled, n_tokens, true);
        match vectors {
            HeadVectors::Float { .. } => (readings, Vec::new(), Vec::new()),
            HeadVectors::Int8 { queries, keys } => (readings, queries, keys),
        }
    }

    fn reader(&self) -> Reader<'_> {
        Reader { head: &self.head, bank: &self.bank, calibration: &self.calibration, explainer: &self.explainer, threads: 1 }
    }

    /// A finished chunk's reading, its windows' readings and its heat-map cells; the chunk starts at document
    /// position `start`. The chunk and its windows go through the head and the bank as one batch.
    fn read_chunk(&self, reader: &Reader, pool: &ChunkPool, start: usize, index: usize, parts: &mut Parts, vectors: Option<&mut Vec<ChunkVectors>>) {
        let length = pool.length;
        let two_d = 2 * self.info.width;
        let spans = window_spans(length, self.info.window_width, self.info.window_stride);
        // A chunk that is one window of its own length (a short text) has the pooled vector of the window already: the
        // bank is read once, and the window's reading is the chunk's without its explanation.
        let single = spans.as_slice() == [(0, length)];
        let windows_read = if single { 0 } else { spans.len() };
        let mut pooled = vec![0f32; (1 + windows_read) * two_d];
        pool.pooled(0, length, &mut pooled[..two_d]);
        let mut n_tokens = vec![length];
        if !single {
            for (i, &(a, b)) in spans.iter().enumerate() {
                pool.pooled(a, b, &mut pooled[(i + 1) * two_d..(i + 2) * two_d]);
                n_tokens.push(b - a);
            }
        }
        let (readings, head_vectors) = reader.read(&pooled, &n_tokens, true);
        let mut readings = readings.into_iter();
        let chunk_reading = readings.next().expect("the chunk's reading");
        let window_readings: Vec<Reading> = if single { vec![Reading { explain: None, ..chunk_reading.clone() }] } else { readings.collect() };
        let place = |a: usize, b: usize| Place { start: start + a, end: start + b, ..Place::default() };
        parts.chunks.push(ChunkReading { place: place(0, length), reading: chunk_reading });
        let windows: Vec<WindowReading> =
            spans.iter().zip(window_readings).map(|(&(a, b), reading)| WindowReading { place: place(a, b), chunk: index, reading }).collect();
        let cells: Vec<(usize, usize)> = cell_spans(length, self.info.window_stride).into_iter().map(|(a, b)| (start + a, start + b)).collect();
        parts.cells.extend(reading::heat_map(&windows, &cells));
        parts.windows.extend(windows);
        if let Some(vectors) = vectors {
            let w = self.head.out_width();
            let unit = |vector: &[i8]| -> Vec<f32> {
                let inverse = codes::inverse_norm(vector);
                vector.iter().map(|&c| (c as f64 * inverse) as f32).collect()
            };
            let (query, key, query_codes, key_codes) = match &head_vectors {
                HeadVectors::Float { queries, keys } => (queries[..w].to_vec(), keys[..w].to_vec(), Vec::new(), Vec::new()),
                HeadVectors::Int8 { queries, keys } => (unit(&queries[..w]), unit(&keys[..w]), queries[..w].to_vec(), keys[..w].to_vec()),
            };
            // The windows' codes: the rows after the chunk's, or, for a single window, the chunk's own.
            let windows_of = |rows: &[i8]| -> Vec<Vec<i8>> {
                if single { vec![rows[..w].to_vec()] } else { rows[w..].chunks_exact(w).map(<[i8]>::to_vec).collect() }
            };
            let (window_query_codes, window_key_codes) = match &head_vectors {
                HeadVectors::Float { .. } => (Vec::new(), Vec::new()),
                HeadVectors::Int8 { queries, keys } => (windows_of(queries), windows_of(keys)),
            };
            let window_pooled = if single { vec![pooled[..two_d].to_vec()] } else { pooled[two_d..].chunks_exact(two_d).map(<[f32]>::to_vec).collect() };
            vectors.push(ChunkVectors { pooled: pooled[..two_d].to_vec(), query, key, query_codes, key_codes, window_query_codes, window_key_codes, window_pooled });
        }
    }

    fn document(&self, n_tokens: usize, parts: Parts) -> DocumentReading {
        DocumentReading {
            n_tokens,
            chunk_tokens: self.info.chunk_tokens,
            verdict: reading::combine(&self.calibration, &parts.chunks.iter().map(|chunk| ChunkCall::from(&chunk.reading)).collect::<Vec<_>>()),
            chunks: parts.chunks,
            windows: parts.windows,
            cells: parts.cells,
        }
    }
}

/// The chunks of a document of `n` tokens and every position where a chunk, window or cell starts or ends.
struct Plan {
    bounds: Vec<(usize, usize)>,
    boundaries: Vec<usize>, // ascending document positions, 0 first, n last
}

impl Plan {
    fn new(n: usize, info: &ModelInfo) -> Self {
        let bounds = chunk_bounds(n, info.chunk_tokens);
        let mut boundaries = vec![0, n];
        for &(start, end) in &bounds {
            let length = end - start;
            for (a, b) in window_spans(length, info.window_width, info.window_stride).into_iter().chain(cell_spans(length, info.window_stride)) {
                boundaries.extend([start + a, start + b]);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        Self { bounds, boundaries }
    }
}

/// The readings of a document's chunks so far.
#[derive(Default)]
struct Parts {
    chunks: Vec<ChunkReading>,
    windows: Vec<WindowReading>,
    cells: Vec<Cell>,
}

/// One chunk's vectors (parity tests).
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ChunkVectors {
    pub pooled: Vec<f32>,
    /// The unit query and key (in int8 output, the unit vectors of the codes).
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    /// The int8 codes of the query and key in int8 output; empty otherwise.
    pub query_codes: Vec<i8>,
    pub key_codes: Vec<i8>,
    /// Each window's codes, `(v, w)` each in window order; empty in float output.
    pub window_query_codes: Vec<Vec<i8>>,
    pub window_key_codes: Vec<Vec<i8>>,
    pub window_pooled: Vec<Vec<f32>>,
}

/// Each chunk's vectors and the first rows of every stage of the first chunk (parity tests).
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct Vectors {
    pub chunks: Vec<ChunkVectors>,
    pub trace: Trace,
}

/// Byte and character offsets of every chunk, window and cell, from the byte offset of each boundary position.
fn place(document: &mut DocumentReading, positions: &[usize], bytes: &[usize], text: &str) {
    assert_eq!(positions.len(), bytes.len(), "a byte offset for every boundary");
    // Characters before each boundary, in one pass (offsets ascend).
    let mut chars = Vec::with_capacity(bytes.len());
    let (mut byte, mut count) = (0usize, 0usize);
    for &boundary in bytes {
        count += text[byte..boundary].chars().count();
        byte = boundary;
        chars.push(count);
    }
    let fill = |place: &mut Place| {
        let index = |p: usize| positions.binary_search(&p).expect("every place starts and ends at a boundary");
        let (a, b) = (index(place.start), index(place.end));
        (place.start_byte, place.end_byte, place.start_char, place.end_char) = (bytes[a], bytes[b], chars[a], chars[b]);
    };
    document.chunks.iter_mut().for_each(|chunk| fill(&mut chunk.place));
    document.windows.iter_mut().for_each(|window| fill(&mut window.place));
    document.cells.iter_mut().for_each(|cell| fill(&mut cell.place));
}

#[cfg(test)]
mod tests {
    #[test]
    fn engine_is_send_and_sync() {
        fn shareable<T: Send + Sync>() {}
        shareable::<super::Engine>();
    }
}
