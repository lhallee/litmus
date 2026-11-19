//! A release folder on disk: `model.safetensors`, `config.json` and ModernBERT's `tokenizer.json` (beside them, named
//! explicitly, in `LITMUS_TOKENIZER`, or in the Hugging Face cache at the release's pinned revision).

use std::path::{Path, PathBuf};

use crate::cache;

/// The bytes of a release and its tokenizer, and where they came from.
pub struct ReleaseFiles {
    pub model: Vec<u8>,
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub source: String,
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The tokenizer file: `explicit`, the release folder, `LITMUS_TOKENIZER`, or the Hugging Face cache at the release's
/// pinned revision.
pub fn tokenizer_path(explicit: Option<&Path>, folder: &Path, config: &serde_json::Value) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return Ok(path.to_owned());
    }
    let beside = folder.join("tokenizer.json");
    if beside.exists() {
        return Ok(beside);
    }
    if let Some(path) = std::env::var_os("LITMUS_TOKENIZER") {
        return Ok(PathBuf::from(path));
    }
    let repository = config["tokenizer"]["repository"].as_str().unwrap_or("answerdotai/ModernBERT-large");
    let revision = config["tokenizer"]["revision"].as_str().unwrap_or("");
    let hub = std::env::var_os("HF_HOME")
        .map(|home| PathBuf::from(home).join("hub"))
        .or_else(|| std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(|home| PathBuf::from(home).join(".cache").join("huggingface").join("hub")));
    if let Some(hub) = hub {
        let path = hub.join(format!("models--{}", repository.replace('/', "--"))).join("snapshots").join(revision).join("tokenizer.json");
        if path.exists() {
            return Ok(path);
        }
    }
    Err(format!("no tokenizer.json: pass --tokenizer, put one in {}, or set LITMUS_TOKENIZER", folder.display()))
}

/// Read a release folder, checking the tokenizer against the hash the release pins when it pins one.
pub fn read_folder(folder: &Path, tokenizer: Option<&Path>) -> Result<ReleaseFiles, String> {
    let config = read(&folder.join("config.json"))?;
    let parsed: serde_json::Value = serde_json::from_slice(&config).map_err(|e| format!("config.json: {e}"))?;
    let tokenizer_file = tokenizer_path(tokenizer, folder, &parsed)?;
    let tokenizer = read(&tokenizer_file)?;
    if let Some(expected) = parsed["tokenizer"]["sha256"].as_str() {
        let actual = cache::sha256(&tokenizer);
        if actual != expected {
            return Err(format!("{} has sha256 {actual}, the release expects {expected}", tokenizer_file.display()));
        }
    }
    Ok(ReleaseFiles { model: read(&folder.join("model.safetensors"))?, config, tokenizer, source: folder.display().to_string() })
}
