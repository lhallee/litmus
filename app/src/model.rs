//! What the app knows of the loaded release beside the engine: its version, its size, the calibration's length bins
//! for the explanation, and whether it is the random-weights placeholder. A release made by `reference.py init`
//! records `"provenance": {"kind": "random", ...}` in `config.json`; a trained release records its checkpoint
//! instead. That field is the placeholder test; no sidecar file is needed.

use litmus_engine::release::CalibrationTable;
use litmus_engine::{Chance, ChanceRule, DocumentReading, Reading};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelMeta {
    /// The provenance's version, else `sha256:` and 16 hex digits of the model and config (as the Python binding).
    pub version: String,
    /// The release has random weights: every reading is meaningless.
    pub placeholder: bool,
    /// "d192 L4": width and depth.
    pub size: String,
    /// Labels of the calibration's length bins ("128-255"), by bin index.
    pub length_bins: Vec<String>,
    /// Where each length bin starts, in tokens, and per bin the share of not-human texts among the calibration texts
    /// behind the probability with the half width between sources (`litmus_engine::ChanceRule`). Both are empty for
    /// the contribute site's model, whose table the page does not hold.
    pub length_edges: Vec<usize>,
    pub chance_rule: ChanceRule,
    pub chunk_tokens: usize,
    pub window_width: usize,
    pub window_stride: usize,
    pub bank: [usize; 3],
    pub out_of_reference_level: f64,
    pub license: String,
    /// Where it came from: a folder, "built in", or "this site".
    pub source: String,
    /// The release carries its masked language head and surprise table, so a reading's details show its surprise.
    pub surprise: bool,
}

/// A release is the placeholder when its provenance says its weights are random.
pub fn is_placeholder(config: &Value) -> bool {
    config["provenance"]["kind"].as_str() == Some("random")
}

impl ModelMeta {
    /// From the release's `config.json` and its version.
    pub fn from_config(config: &Value, version: String, source: String) -> Result<Self, String> {
        let model = &config["model"];
        let reading = &config["reading"];
        let calibration = &config["calibration"];
        let number = |value: &Value, name: &str| value[name].as_u64().map(|n| n as usize).ok_or_else(|| format!("config.json has no {name}"));
        let bank = config["bank"]["counts"].as_array().ok_or("config.json has no bank counts")?;
        if bank.len() != 3 {
            return Err("config.json's bank counts are not three".into());
        }
        let count = |i: usize| bank[i].as_u64().map(|n| n as usize).ok_or("a bank count is not a number");
        let table = CalibrationTable::parse(calibration.clone()).map_err(|e| e.to_string())?;
        Ok(Self {
            version,
            placeholder: is_placeholder(config),
            size: format!("d{} L{}", number(model, "width")?, number(model, "depth")?),
            length_bins: calibration["length_bins"].as_array().map(|bins| bins.iter().filter_map(|b| b.as_str().map(str::to_owned)).collect()).unwrap_or_default(),
            chance_rule: table.chance_rule(),
            length_edges: table.length_edges,
            chunk_tokens: number(reading, "chunk_tokens")?,
            window_width: number(reading, "window_width")?,
            window_stride: number(reading, "window_stride")?,
            bank: [count(0)?, count(1)?, count(2)?],
            out_of_reference_level: calibration["out_of_reference_level"].as_f64().unwrap_or(f64::NAN),
            license: config["license"].as_str().unwrap_or("").to_owned(),
            source,
            surprise: config["surprise"].is_object(),
        })
    }

    /// The contribute site's model, before the first reading names its version. The server's API does not say
    /// whether its release is the placeholder.
    pub fn server() -> Self {
        Self {
            version: "the site's model".into(),
            placeholder: false,
            size: String::new(),
            length_bins: Vec::new(),
            length_edges: Vec::new(),
            chance_rule: ChanceRule::default(),
            chunk_tokens: 0,
            window_width: 0,
            window_stride: 0,
            bank: [0; 3],
            out_of_reference_level: f64::NAN,
            license: String::new(),
            source: "the contribute site".into(),
            surprise: false,
        }
    }

    /// The chance of AI of a document's verdict by this release's table; none without the table.
    pub fn chance(&self, document: &DocumentReading) -> Option<Chance> {
        (!self.chance_rule.not_human_shares.is_empty()).then(|| document.chance(&self.chance_rule))
    }

    /// The chance of AI of one chunk's or window's reading.
    pub fn chance_of(&self, reading: &Reading) -> Option<Chance> {
        (!self.chance_rule.not_human_shares.is_empty()).then(|| reading.chance(&self.chance_rule))
    }

    /// The chance of AI from a verdict's probability and interval alone, for a history entry kept before the chance
    /// was: the share is that of the first chunk a text of `tokens` tokens is read in.
    pub fn chance_from(&self, probability: f64, interval: [f64; 2], tokens: usize) -> Option<Chance> {
        if tokens == 0 || self.chunk_tokens == 0 {
            return None;
        }
        let (start, end) = litmus_engine::chunks::chunk_bounds(tokens, self.chunk_tokens)[0];
        let bin = self.length_edges.iter().skip(1).filter(|&&edge| edge <= end - start).count();
        (bin < self.chance_rule.not_human_shares.len()).then(|| self.chance_rule.chance(probability, interval, bin))
    }

    /// The token lengths a calibration group covers, in words: "lengths 128-255 and 256-511".
    pub fn bins_words(&self, bins: &[usize]) -> String {
        let names: Vec<String> = bins.iter().map(|&b| self.length_bins.get(b).cloned().unwrap_or_else(|| format!("bin {b}"))).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        format!("lengths {} tokens", crate::wording::join_and(&names))
    }
}

/// The bytes of a release and where they came from (the desktop).
#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use std::path::{Path, PathBuf};

    use litmus_files::release::{self, ReleaseFiles};

    /// Where the desktop app looks for a release, in order: the folder chosen in Settings, `LITMUS_MODEL`, a
    /// `release` folder beside the executable, and the release built into the binary (the `embedded` feature).
    pub fn candidates(chosen: Option<&Path>) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Some(folder) = chosen {
            out.push(folder.to_owned());
        }
        if let Some(folder) = std::env::var_os("LITMUS_MODEL") {
            out.push(PathBuf::from(folder));
        }
        if let Some(beside) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|dir| dir.join("release"))) {
            out.push(beside);
        }
        out
    }

    #[cfg(feature = "embedded")]
    pub fn embedded() -> Option<ReleaseFiles> {
        Some(ReleaseFiles {
            model: include_bytes!(concat!(env!("LITMUS_EMBED_DIR"), "/model.safetensors")).to_vec(),
            config: include_bytes!(concat!(env!("LITMUS_EMBED_DIR"), "/config.json")).to_vec(),
            tokenizer: include_bytes!(concat!(env!("LITMUS_EMBED_DIR"), "/tokenizer.json")).to_vec(),
            source: "built in".into(),
        })
    }

    #[cfg(not(feature = "embedded"))]
    pub fn embedded() -> Option<ReleaseFiles> {
        None
    }

    /// The first release found. A folder that exists but does not read is an error, never skipped.
    pub fn find(chosen: Option<&Path>) -> Result<ReleaseFiles, String> {
        for folder in candidates(chosen) {
            if folder.join("config.json").exists() || chosen == Some(folder.as_path()) {
                return release::read_folder(&folder, None);
            }
        }
        embedded().ok_or_else(|| "No model found. Choose a release folder in Settings (it holds model.safetensors, config.json, and tokenizer.json), set LITMUS_MODEL, or put a folder named release beside the app.".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A release's config with a calibration table of two length bins, texts under 16 tokens and the rest: two
    /// thirds of the short calibration texts are not human and half of the long.
    fn config() -> Value {
        let grid = |bins: &[usize], n: usize| serde_json::json!({"bins": bins, "n": n, "thresholds": [0.1, 0.2]});
        let all = [grid(&[0, 1], 600)];
        serde_json::json!({
            "license": "PolyForm-Noncommercial-1.0.0",
            "model": {"width": 192, "depth": 4},
            "reading": {"chunk_tokens": 8192, "window_width": 512, "window_stride": 128},
            "bank": {"counts": [4096, 4096, 4096]},
            "calibration": {
                "format": "litmus-calibration", "version": 1, "origins": ["human", "ai", "mixed"],
                "length_edges": [0, 16], "length_bins": ["0-15", "16-31"], "levels": [0.05, 0.1], "grid": [0.05, 0.1],
                "spread_levels": [0.5], "top_k": 4, "out_of_reference_level": 0.01,
                "conformal": {"human": [grid(&[0], 100), grid(&[1], 150)], "ai": all, "mixed": all},
                "probability": [
                    {"bins": [0], "n": 300, "edges": [0.5], "p0": [0.2, 0.7], "p1": [0.3, 0.8]},
                    {"bins": [1], "n": 300, "edges": [0.5], "p0": [0.2, 0.7], "p1": [0.3, 0.8]}
                ],
                "three_way": [{"bins": [0, 1], "slope": 1.0, "bias": [0.0, 0.0, 0.0]}],
                "spread": [{"bins": [0, 1], "human": [0.1], "ai": [0.1], "mixed": [0.1]}],
                "closeness": all
            },
            "provenance": {"kind": "random", "seed": 0}
        })
    }

    #[test]
    fn placeholder_comes_from_provenance() {
        let mut config = config();
        let meta = ModelMeta::from_config(&config, "sha256:00".into(), "here".into()).unwrap();
        assert!(meta.placeholder);
        assert_eq!(meta.size, "d192 L4");
        assert_eq!(meta.bins_words(&[0, 1, 7]), "lengths 0-15, 16-31, and bin 7 tokens");
        config["provenance"] = serde_json::json!({"checkpoint": "student.pt", "step": 4000});
        assert!(!ModelMeta::from_config(&config, "v1".into(), "here".into()).unwrap().placeholder);
    }

    #[test]
    fn the_chance_of_ai_comes_from_the_releases_table() {
        let meta = ModelMeta::from_config(&config(), "v1".into(), "here".into()).unwrap();
        assert_eq!(meta.length_edges, [0, 16]);
        let shares = &meta.chance_rule.not_human_shares;
        assert!((shares[0] - 2.0 / 3.0).abs() < 1e-12 && shares[1] == 0.5, "{shares:?}");
        // A history entry kept before the chance was: a text of 10 tokens is read against the short texts, of which
        // two thirds are not human, so a probability of two thirds is one half. A long text keeps its probability,
        // and so does a text of many chunks, each of them long.
        let chance = |tokens: usize| meta.chance_from(2.0 / 3.0, [0.6, 0.7], tokens).map(|c| c.value);
        assert!((chance(10).unwrap() - 0.5).abs() < 1e-12);
        assert!((chance(16).unwrap() - 2.0 / 3.0).abs() < 1e-12 && (chance(20_000).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(chance(0), None, "no tokens, no chunk to read it against");
        // The contribute site's model has no table in the page.
        assert_eq!(ModelMeta::server().chance_from(0.7, [0.6, 0.8], 100), None);
        let mut broken = config();
        broken["calibration"]["probability"][0]["n"] = 100.into();
        assert!(ModelMeta::from_config(&broken, "v1".into(), "here".into()).unwrap_err().contains("length bin 0"));
    }
}
