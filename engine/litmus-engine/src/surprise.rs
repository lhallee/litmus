//! The surprise reading (SPEC.md, Surprise): each token's log probability under the release's masked language head
//! with nothing masked, averaged over the text (its unmasked pseudo log likelihood), and where that mean falls among
//! each origin's calibration texts. It is read only when asked for, in a pass of its own.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::dense;
use crate::kernels::PackedTernary;
use crate::release::Safetensors;

/// Positions whose logits over the vocabulary are held at once: `(TILE, V)` floats.
const TILE: usize = 16;
/// Quantiles per origin in a release's surprise table: 0%, 1%, ..., 100%.
pub const QUANTILES: usize = 101;

/// The masked language head: a BitLinear from width `d` to the embedding width `e`, an RMS norm, and the token
/// embedding's own rows as the output, with one scale and a bias per token.
pub struct LanguageHead {
    pub(crate) proj_norm: Vec<f32>, // (d,)
    pub(crate) proj: PackedTernary, // (e, d)
    pub(crate) norm: Vec<f32>,      // (e,)
    scale: f32,
    bias: Vec<f32>, // (V,)
}

impl LanguageHead {
    /// The head, or `None` when the release has none (`mlm.proj.weight` absent).
    pub fn load(tensors: &Safetensors, vocab: usize, embed_width: usize, width: usize) -> Result<Option<Self>, Error> {
        if tensors.shape("mlm.proj.weight").is_none() {
            return Ok(None);
        }
        let packed = tensors.u8s("mlm.proj.weight", &[embed_width, width / 4])?;
        let gamma = tensors.f32s("mlm.proj.scale", &[1])?[0];
        let proj = PackedTernary::new(embed_width, width, packed, gamma).map_err(|error| Error::Load(format!("mlm.proj: {error}")))?;
        Ok(Some(Self {
            proj_norm: tensors.f32s("mlm.proj_norm", &[width])?,
            proj,
            norm: tensors.f32s("mlm.norm", &[embed_width])?,
            scale: tensors.f32s("mlm.scale", &[1])?[0],
            bias: tensors.f32s("mlm.bias", &[vocab])?,
        }))
    }

    pub fn bytes(&self) -> usize {
        self.proj.bytes() + 4 * (self.proj_norm.len() + self.norm.len() + self.bias.len() + 1)
    }
}

/// Each origin's calibration texts' mean token log probabilities at the quantiles 0%, 1%, ..., 100%.
#[derive(Clone, Debug, Deserialize)]
pub struct SurpriseQuantiles {
    pub human: Vec<f64>,
    pub ai: Vec<f64>,
    pub mixed: Vec<f64>,
}

/// `config.json`'s `surprise`: the table a text's surprise is placed against.
#[derive(Clone, Debug, Deserialize)]
pub struct SurpriseTable {
    pub split: String,
    /// Calibration texts per origin: human, AI, mixed.
    pub texts: [usize; 3],
    pub quantiles: SurpriseQuantiles,
}

impl SurpriseTable {
    pub fn check(&self) -> Result<(), Error> {
        for (name, values) in [("human", &self.quantiles.human), ("ai", &self.quantiles.ai), ("mixed", &self.quantiles.mixed)] {
            if values.len() != QUANTILES || values.windows(2).any(|pair| pair[1] < pair[0]) || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Load(format!("config.json: surprise quantiles of {name} are not {QUANTILES} rising numbers")));
            }
        }
        Ok(())
    }

    /// Per origin, the share of its calibration texts at or below `value`.
    pub fn percentile(&self, value: f64) -> [f64; 3] {
        [&self.quantiles.human, &self.quantiles.ai, &self.quantiles.mixed].map(|quantiles| percentile(quantiles, value))
    }
}

/// The share of an origin's calibration texts at or below `value`, from its quantiles: linear between the
/// quantiles, 0 below the first, 1 above the last.
pub fn percentile(quantiles: &[f64], value: f64) -> f64 {
    let last = quantiles.len() - 1;
    if value <= quantiles[0] {
        return 0.0;
    }
    if value >= quantiles[last] {
        return 1.0;
    }
    let upper = quantiles.partition_point(|&q| q <= value); // q[upper - 1] <= value < q[upper]
    let (low, high) = (quantiles[upper - 1], quantiles[upper]);
    (upper as f64 - 1.0 + (value - low) / (high - low)) / last as f64
}

/// A text's surprise reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Surprise {
    pub n_tokens: usize,
    /// Mean over tokens of `log p(token | the text)`, in nats; higher is less surprising.
    pub mean_log_probability: f64,
    /// `exp(-mean_log_probability)`.
    pub perplexity: f64,
    /// Human, AI, mixed: the share of that origin's calibration texts at or below this text's mean (`None` when the
    /// release has no table).
    pub percentile: Option<[f64; 3]>,
}

impl Surprise {
    pub fn new(sum: f64, n_tokens: usize, table: Option<&SurpriseTable>) -> Self {
        let mean = sum / n_tokens as f64;
        Self { n_tokens, mean_log_probability: mean, perplexity: (-mean).exp(), percentile: table.map(|table| table.percentile(mean)) }
    }
}

/// Buffers of the surprise pass and its running sum, held by a stream that reads it.
#[derive(Default)]
pub struct LanguageScratch {
    /// The current chunk's token ids, in position order.
    pub(crate) ids: Vec<u32>,
    pub(crate) projected: Vec<f32>, // (W, e)
    pub(crate) normed: Vec<f32>,    // (W, e)
    logits: Vec<f32>,               // (TILE, V)
    row: Vec<f32>,                  // (e,)
    /// Sum of the log probabilities read so far, and their number.
    pub sum: f64,
    pub count: usize,
}

impl LanguageScratch {
    pub fn new(block_size: usize, embed_width: usize, vocab: usize, chunk_tokens: usize) -> Self {
        Self {
            ids: Vec::with_capacity(chunk_tokens),
            projected: vec![0.0; block_size * embed_width],
            normed: vec![0.0; block_size * embed_width],
            logits: vec![0.0; TILE * vocab],
            row: vec![0.0; embed_width],
            sum: 0.0,
            count: 0,
        }
    }

    pub fn bytes(&self) -> usize {
        4 * (self.ids.capacity() + self.projected.capacity() + self.normed.capacity() + self.logits.capacity() + self.row.capacity())
    }

    /// Add the log probability of each of `ids` given its row of `normed[..ids.len() * e]`: logits
    /// `scale * (g . row_v) + bias_v` in float32 over the int8 embedding rows `embed * embed_scale`, and a float64
    /// log-sum-exp.
    pub(crate) fn add(&mut self, head: &LanguageHead, embed: &[i8], embed_scale: &[f32], ids: &[u32]) {
        let e = self.row.len();
        let vocab = head.bias.len();
        for (tile, tile_ids) in ids.chunks(TILE).enumerate() {
            let n = tile_ids.len();
            let g = &self.normed[tile * TILE * e..(tile * TILE + n) * e];
            for v in 0..vocab {
                let scale = head.scale * embed_scale[v];
                for (value, &weight) in self.row.iter_mut().zip(&embed[v * e..(v + 1) * e]) {
                    *value = weight as f32 * scale;
                }
                for t in 0..n {
                    self.logits[t * vocab + v] = dot(&g[t * e..(t + 1) * e], &self.row) + head.bias[v];
                }
            }
            for (t, &id) in tile_ids.iter().enumerate() {
                let logits = &self.logits[t * vocab..(t + 1) * vocab];
                let top = dense::row_max(logits);
                let total: f64 = logits.iter().map(|&l| dense::exp_nonpositive(l - top) as f64).sum();
                self.sum += (logits[id as usize] - top) as f64 - total.ln();
                self.count += 1;
            }
        }
    }
}

/// `a . b` with eight running sums, which the compiler vectorizes.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let (a_chunks, b_chunks) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = a_chunks.remainder().iter().zip(b_chunks.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in a_chunks.zip(b_chunks) {
        for lane in 0..8 {
            sums[lane] += x[lane] * y[lane];
        }
    }
    sums.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates_between_quantiles() {
        let quantiles: Vec<f64> = (0..QUANTILES).map(|i| i as f64).collect();
        assert_eq!(percentile(&quantiles, -1.0), 0.0);
        assert_eq!(percentile(&quantiles, 100.0), 1.0);
        assert!((percentile(&quantiles, 25.5) - 0.255).abs() < 1e-12);
        assert!((percentile(&quantiles, 50.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn dot_matches_a_plain_sum() {
        let a: Vec<f32> = (0..131).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..131).map(|i| (i as f32 * 0.11).cos()).collect();
        let plain: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - plain).abs() < 1e-4);
    }
}
