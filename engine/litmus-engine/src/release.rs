//! The release file (SPEC.md, section 6): `model.safetensors` and `config.json`.

use std::collections::HashMap;

use serde::Deserialize;

use crate::Error;

pub const FORMAT: &str = "litmus-student";
pub const FORMAT_VERSION: u32 = 1;
/// Format 1 plus `model.attention`, the optional integer attention of SPEC.md; a float release stays format 1.
pub const INTEGER_ATTENTION_FORMAT_VERSION: u32 = 2;
/// Format 2 plus `model.output: "int8"` (SPEC.md, Int8 output): int8 query and key and an int8 bank.
pub const INT8_OUTPUT_FORMAT_VERSION: u32 = 3;

/// What the head outputs and the bank stores (SPEC.md, section 4, Head, and Int8 output).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Output {
    /// Float32 unit queries and keys against a float16 bank: every release of format 1 and 2.
    #[default]
    Float,
    /// Int8 codes of the query and the key against an int8 bank, compared by integer dot products: a format 3 release.
    Int8,
}

impl Output {
    pub fn name(self) -> &'static str {
        match self {
            Output::Float => "float",
            Output::Int8 => "int8",
        }
    }
}

/// How the encoder attends (SPEC.md, section 3, step 4, and Integer attention).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Attention {
    /// Float32 scores, softmax and weighted values: every format 1 release.
    #[default]
    Float,
    /// Integer products of int8 queries, keys and values, weights in 127ths: a format 2 release that names it.
    Integer,
}

impl Attention {
    pub fn name(self) -> &'static str {
        match self {
            Attention::Float => "float",
            Attention::Integer => "integer",
        }
    }
}

/// `config.json` of a release.
#[derive(Clone, Debug, Deserialize)]
pub struct ReleaseConfig {
    pub format: String,
    pub format_version: u32,
    pub tokenizer: TokenizerConfig,
    pub model: ModelConfig,
    pub head: HeadConfig,
    pub reading: ReadingConfig,
    pub bank: BankConfig,
    pub calibration: Option<CalibrationTable>,
    #[serde(default)]
    pub provenance: serde_json::Value,
    #[serde(default)]
    pub license: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TokenizerConfig {
    pub repository: String,
    pub revision: String,
    pub sha256: String,
    pub add_special_tokens: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub embed_width: usize,
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub ffn_width: usize,
    pub block_size: usize,
    pub rope_base: f64,
    pub norm_eps: f64,
    pub act_eps: f64,
    #[serde(default)]
    pub weight_eps: f64,
    /// Format 2 only; `None` in format 1, which is float attention.
    #[serde(default)]
    pub attention: Option<Attention>,
    /// Format 3 only; `None` in formats 1 and 2, which are float output.
    #[serde(default)]
    pub output: Option<Output>,
}

impl ModelConfig {
    /// The attention the release reads with (checked against the format by `ReleaseConfig::parse`).
    pub fn attention(&self) -> Attention {
        self.attention.unwrap_or_default()
    }

    /// What the head outputs and the bank stores (checked against the format by `ReleaseConfig::parse`).
    pub fn output(&self) -> Output {
        self.output.unwrap_or_default()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct HeadConfig {
    pub hidden: usize,
    pub out_width: usize,
    /// Standardized head inputs are clamped to +-clamp (10 in every fixture; the distillation may choose another).
    pub clamp: f32,
}

/// How a document is cut (SPEC.md, Chunks; Windows and the heat map).
#[derive(Clone, Debug, Deserialize)]
pub struct ReadingConfig {
    /// `C_max`: the most tokens read as one text; a longer document is read in `ceil(n / C_max)` equal chunks.
    pub chunk_tokens: usize,
    /// The heat map's windows and their stride, which divides the width.
    pub window_width: usize,
    pub window_stride: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BankConfig {
    pub origins: Vec<String>,
    pub counts: Vec<usize>,
    pub split: String,
    /// Each metadata field's names, which `bank.meta.<field>` codes index.
    #[serde(default)]
    pub fields: std::collections::BTreeMap<String, Vec<String>>,
}

/// The calibration format this engine reads (`../teacher/CALIBRATION.md`).
pub const CALIBRATION_FORMAT: &str = "litmus-calibration";
pub const CALIBRATION_VERSION: u32 = 1;
/// The level of the strip's band (SPEC.md, The litmus strip).
pub const STRIP_LEVEL: f64 = 0.10;
/// The family level of the mixed-document signal over a document's chunks (SPEC.md, Chunks).
pub const MIXED_LEVEL: f64 = 0.05;

/// Thresholds aligned with the grid; `null` (minus infinity) as `None`.
#[derive(Clone, Debug, Deserialize)]
pub struct GridGroup {
    pub bins: Vec<usize>,
    pub n: usize,
    pub thresholds: Vec<Option<f64>>,
}

/// Inductive Venn-Abers blocks on the score.
#[derive(Clone, Debug, Deserialize)]
pub struct VennAbersGroup {
    pub bins: Vec<usize>,
    pub edges: Vec<f64>,
    pub p0: Vec<f64>,
    pub p1: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ThreeWayGroup {
    pub bins: Vec<usize>,
    pub slope: f64,
    pub bias: [f64; 3],
}

/// Quantiles of the calibration texts' spreads at `spread_levels`, per origin.
#[derive(Clone, Debug, Deserialize)]
pub struct SpreadGroup {
    pub bins: Vec<usize>,
    pub human: Vec<f64>,
    pub ai: Vec<f64>,
    pub mixed: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ConformalTables {
    pub human: Vec<GridGroup>,
    pub ai: Vec<GridGroup>,
    pub mixed: Vec<GridGroup>,
}

/// The teacher's calibration table (`litmus-calibration` version 1), embedded in `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct CalibrationTable {
    pub format: String,
    pub version: u32,
    pub origins: Vec<String>,
    pub length_edges: Vec<usize>,
    pub levels: Vec<f64>,
    pub grid: Vec<f64>,
    pub spread_levels: Vec<f64>,
    pub top_k: usize,
    pub out_of_reference_level: f64,
    pub conformal: ConformalTables,
    pub probability: Vec<VennAbersGroup>,
    pub three_way: Vec<ThreeWayGroup>,
    pub spread: Vec<SpreadGroup>,
    pub closeness: Vec<GridGroup>,
}

impl ReleaseConfig {
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let config: ReleaseConfig =
            serde_json::from_slice(bytes).map_err(|error| Error::Load(format!("config.json: {error}")))?;
        if config.format != FORMAT || ![FORMAT_VERSION, INTEGER_ATTENTION_FORMAT_VERSION, INT8_OUTPUT_FORMAT_VERSION].contains(&config.format_version) {
            return Err(Error::Load(format!(
                "config.json is {} version {}, not {FORMAT} version {FORMAT_VERSION}, {INTEGER_ATTENTION_FORMAT_VERSION} or {INT8_OUTPUT_FORMAT_VERSION}",
                config.format, config.format_version
            )));
        }
        let model = &config.model;
        let check = |ok: bool, what: &str| if ok { Ok(()) } else { Err(Error::Load(format!("config.json: {what}"))) };
        if config.format_version == FORMAT_VERSION {
            check(model.attention.is_none(), "model.attention needs format version 2 or later")?;
        } else {
            check(model.attention.is_some(), "format versions 2 and 3 name model.attention (float or integer)")?;
        }
        if config.format_version == INT8_OUTPUT_FORMAT_VERSION {
            check(model.output == Some(Output::Int8), "format version 3 names model.output, \"int8\"")?;
        } else {
            check(model.output.is_none(), "model.output needs format version 3")?;
        }
        if model.output() == Output::Int8 {
            check(config.head.out_width > 0 && config.head.out_width * 127 * 127 < 1 << 31, "int8 output needs a query and key width whose integer dot products fit 32 bits")?;
        }
        if model.attention() == Attention::Integer {
            let d_h = model.width / model.heads.max(1);
            check(d_h * 127 * 127 < 1 << 24, "integer attention needs heads of at most 1,040 columns")?;
        }
        check(model.heads > 0 && model.width % model.heads == 0, "width must split into heads")?;
        check((model.width / model.heads) % 2 == 0, "head width must be even")?;
        check(model.block_size >= 2 && model.block_size % 2 == 0, "block size must be even")?;
        check(config.head.clamp > 0.0, "head.clamp must be positive")?;
        let reading = &config.reading;
        check(
            reading.window_stride > 0 && reading.window_width % reading.window_stride == 0 && reading.window_width <= reading.chunk_tokens,
            "window_stride must divide window_width, which is at most chunk_tokens",
        )?;
        check(
            [model.embed_width, model.width, model.ffn_width].iter().all(|w| w % 4 == 0 && *w > 0),
            "ternary input widths must be positive multiples of 4",
        )?;
        check(!config.tokenizer.add_special_tokens, "the student reads ids without added special tokens")?;
        if let Some(table) = &config.calibration {
            table.validate()?;
        }
        Ok(config)
    }
}

/// The group of a table holding a length bin.
pub fn group<'a, T>(groups: &'a [T], bins: impl Fn(&T) -> &[usize], bin: usize) -> &'a T {
    groups.iter().find(|group| bins(group).contains(&bin)).expect("validated: every bin is in one group")
}

impl CalibrationTable {
    /// Length bins: bin `i` holds `[length_edges[i], length_edges[i + 1])`, the last open above.
    pub fn bins(&self) -> usize {
        self.length_edges.len()
    }

    fn validate(&self) -> Result<(), Error> {
        let fail = |what: String| Err(Error::Load(format!("calibration: {what}")));
        if self.format != CALIBRATION_FORMAT || self.version != CALIBRATION_VERSION {
            return fail(format!("{} version {} is not {CALIBRATION_FORMAT} version {CALIBRATION_VERSION}", self.format, self.version));
        }
        if self.origins != ["human", "ai", "mixed"] {
            return fail(format!("origins {:?}, expected human, ai, mixed", self.origins));
        }
        for level in [STRIP_LEVEL, MIXED_LEVEL] {
            if !self.levels.contains(&level) {
                return fail(format!("levels {:?} must hold {level}", self.levels));
            }
        }
        let bins = self.bins();
        let covers = |members: Vec<&[usize]>| {
            let mut all: Vec<usize> = members.into_iter().flatten().copied().collect();
            all.sort_unstable();
            all == (0..bins).collect::<Vec<_>>()
        };
        let grid_ok = |groups: &[GridGroup]| groups.iter().all(|g| g.thresholds.len() == self.grid.len());
        for (name, groups) in [("human", &self.conformal.human), ("ai", &self.conformal.ai), ("mixed", &self.conformal.mixed), ("closeness", &self.closeness)] {
            if !covers(groups.iter().map(|g| g.bins.as_slice()).collect()) || !grid_ok(groups) {
                return fail(format!("{name}: groups must cover every bin once, thresholds aligned with the grid"));
            }
        }
        if !covers(self.probability.iter().map(|g| g.bins.as_slice()).collect())
            || !self.probability.iter().all(|g| g.p0.len() == g.edges.len() + 1 && g.p1.len() == g.edges.len() + 1)
        {
            return fail("probability: groups must cover every bin once, with one more p0 and p1 than edges".into());
        }
        if !covers(self.three_way.iter().map(|g| g.bins.as_slice()).collect()) {
            return fail("three_way: groups must cover every bin once".into());
        }
        let quantiles = self.spread_levels.len();
        if !covers(self.spread.iter().map(|g| g.bins.as_slice()).collect())
            || !self.spread.iter().all(|g| [&g.human, &g.ai, &g.mixed].iter().all(|q| q.len() == quantiles))
        {
            return fail("spread: groups must cover every bin once, one quantile per spread level".into());
        }
        if self.top_k == 0 || self.grid.is_empty() || !self.grid.windows(2).all(|pair| pair[0] < pair[1]) {
            return fail("top_k must be positive and the grid ascending".into());
        }
        Ok(())
    }

    /// The number of edges after the first that are at or under `n_tokens`.
    pub fn length_bin(&self, n_tokens: usize) -> usize {
        self.length_edges[1..].iter().filter(|&&edge| edge <= n_tokens).count()
    }

    /// The smallest grid level whose threshold exceeds `value`, else 1.
    pub fn grid_p_value(&self, value: f64, thresholds: &[Option<f64>]) -> f64 {
        self.grid
            .iter()
            .zip(thresholds)
            .find(|(_, threshold)| threshold.is_some_and(|t| value < t))
            .map_or(1.0, |(&level, _)| level)
    }
}

/// One tensor of a safetensors file, borrowed.
pub struct Tensor<'a> {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

/// The tensors of a safetensors file by name, borrowed from its bytes.
pub struct Safetensors<'a> {
    tensors: HashMap<String, Tensor<'a>>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn dtype_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "I8" | "U8" | "BOOL" => 1,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F32" | "I32" | "U32" => 4,
        "F64" | "I64" | "U64" => 8,
        _ => return None,
    })
}

impl<'a> Safetensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let fail = |what: String| Error::Load(format!("model.safetensors: {what}"));
        if bytes.len() < 8 {
            return Err(fail("shorter than its header length".into()));
        }
        let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let body_start = 8usize.checked_add(header_length).filter(|&end| end <= bytes.len()).ok_or_else(|| fail("header past the end".into()))?;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..body_start]).map_err(|error| fail(format!("header: {error}")))?;
        let body = &bytes[body_start..];
        let mut tensors = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let entry: Entry = serde_json::from_value(value).map_err(|error| fail(format!("{name}: {error}")))?;
            let size = dtype_size(&entry.dtype).ok_or_else(|| fail(format!("{name}: dtype {}", entry.dtype)))?;
            let [begin, end] = entry.data_offsets;
            let expected = entry.shape.iter().product::<usize>() * size;
            if begin > end || end > body.len() || end - begin != expected {
                return Err(fail(format!("{name}: offsets {begin}..{end} do not hold shape {:?}", entry.shape)));
            }
            tensors.insert(name, Tensor { dtype: entry.dtype, shape: entry.shape, data: &body[begin..end] });
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&Tensor<'a>, Error> {
        let tensor = self.tensors.get(name).ok_or_else(|| Error::Load(format!("model.safetensors has no tensor {name}")))?;
        if tensor.dtype != dtype || tensor.shape != shape {
            return Err(Error::Load(format!(
                "tensor {name} is {} {:?}, expected {dtype} {shape:?}",
                tensor.dtype, tensor.shape
            )));
        }
        Ok(tensor)
    }

    pub fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let tensor = self.get(name, "F32", shape)?;
        Ok(tensor.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }

    /// F16 tensor widened to f32.
    pub fn f16s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let tensor = self.get(name, "F16", shape)?;
        Ok(tensor.data.chunks_exact(2).map(|b| crate::dense::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect())
    }

    pub fn i8s(&self, name: &str, shape: &[usize]) -> Result<Vec<i8>, Error> {
        Ok(self.get(name, "I8", shape)?.data.iter().map(|&b| b as i8).collect())
    }

    pub fn u8s(&self, name: &str, shape: &[usize]) -> Result<Vec<u8>, Error> {
        Ok(self.get(name, "U8", shape)?.data.to_vec())
    }

    pub fn u16s(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, Error> {
        Ok(self.get(name, "U16", shape)?.data.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }

    pub fn i32s(&self, name: &str, shape: &[usize]) -> Result<Vec<i32>, Error> {
        Ok(self.get(name, "I32", shape)?.data.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect())
    }

    pub fn i64s(&self, name: &str, shape: &[usize]) -> Result<Vec<i64>, Error> {
        Ok(self.get(name, "I64", shape)?.data.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
    }

    /// A one-dimensional tensor's length.
    pub fn length(&self, name: &str) -> Result<usize, Error> {
        match self.shape(name) {
            Some([n]) => Ok(*n),
            Some(shape) => Err(Error::Load(format!("tensor {name} has shape {shape:?}, expected one dimension"))),
            None => Err(Error::Load(format!("model.safetensors has no tensor {name}"))),
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|tensor| tensor.shape.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(version: u32, attention: Option<&str>) -> Vec<u8> {
        config_with_output(version, attention, None)
    }

    fn config_with_output(version: u32, attention: Option<&str>, output: Option<&str>) -> Vec<u8> {
        let mut model = serde_json::json!({"vocab_size": 50368, "embed_width": 64, "width": 64, "depth": 2, "heads": 2, "ffn_width": 256,
            "block_size": 16, "rope_base": 10000.0, "norm_eps": 1e-6, "act_eps": 1e-5, "weight_eps": 1e-5});
        if let Some(attention) = attention {
            model["attention"] = attention.into();
        }
        if let Some(output) = output {
            model["output"] = output.into();
        }
        serde_json::to_vec(&serde_json::json!({"format": FORMAT, "format_version": version, "model": model,
            "tokenizer": {"repository": "r", "revision": "v", "sha256": "s", "add_special_tokens": false},
            "head": {"hidden": 8, "out_width": 4, "clamp": 10.0}, "reading": {"chunk_tokens": 96, "window_width": 32, "window_stride": 16},
            "bank": {"origins": ["human", "ai", "mixed"], "counts": [1, 1, 1], "split": "bank"}, "calibration": null}))
        .unwrap()
    }

    #[test]
    fn format_one_is_float_and_format_two_names_its_attention() {
        assert_eq!(ReleaseConfig::parse(&config(1, None)).unwrap().model.attention(), Attention::Float);
        assert_eq!(ReleaseConfig::parse(&config(2, Some("integer"))).unwrap().model.attention(), Attention::Integer);
        assert_eq!(ReleaseConfig::parse(&config(2, Some("float"))).unwrap().model.attention(), Attention::Float);
        for (version, attention) in [(1, Some("integer")), (2, None), (2, Some("half")), (3, Some("integer")), (4, Some("integer")), (0, None)] {
            assert!(ReleaseConfig::parse(&config(version, attention)).is_err(), "format {version} with {attention:?}");
        }
    }

    #[test]
    fn format_three_names_its_output_and_the_others_cannot() {
        for attention in ["float", "integer"] {
            let model = ReleaseConfig::parse(&config_with_output(3, Some(attention), Some("int8"))).unwrap().model;
            assert_eq!((model.output(), model.attention().name()), (Output::Int8, attention));
        }
        assert_eq!(ReleaseConfig::parse(&config(1, None)).unwrap().model.output(), Output::Float);
        assert_eq!(ReleaseConfig::parse(&config(2, Some("integer"))).unwrap().model.output(), Output::Float);
        for (version, attention, output) in [
            (3, Some("integer"), None),
            (3, Some("integer"), Some("float")),
            (3, Some("integer"), Some("int4")),
            (3, None, Some("int8")),
            (2, Some("integer"), Some("int8")),
            (1, None, Some("int8")),
            (4, Some("integer"), Some("int8")),
        ] {
            assert!(ReleaseConfig::parse(&config_with_output(version, attention, output)).is_err(), "format {version} with {attention:?} and output {output:?}");
        }
    }
}
