//! A content-hash cache of readings: an unchanged file (same bytes, same model, same engine version) is not scored
//! again. One JSON file per reading under the user's cache folder, or `LITMUS_CACHE_DIR`. The command line and the
//! desktop app share the folder, so a file either has read is not read again by the other.

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use litmus_engine::DocumentReading;
use sha2::{Digest, Sha256};

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The key of a model in the cache: 16 hex digits of the hash of the engine's version and the release's files.
pub fn model_key(model_files: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    for bytes in model_files {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    hex(&hasher.finalize())[..16].to_owned()
}

/// A release's version as the Python binding reports it (and the contribute site stores it): the provenance's
/// `version` if it names one, else `sha256:` and the first 16 hex digits of the hash of `model.safetensors` then
/// `config.json`.
pub fn release_version(provenance: &serde_json::Value, model: &[u8], config: &[u8]) -> String {
    if let Some(version) = provenance.get("version").and_then(|v| v.as_str()) {
        return version.to_owned();
    }
    let digest = Sha256::new().chain_update(model).chain_update(config).finalize();
    format!("sha256:{}", hex(&digest[..8]))
}

/// The cache folder: `LITMUS_CACHE_DIR`, else the platform's per-user cache folder.
#[cfg(not(target_arch = "wasm32"))]
pub fn folder() -> Option<PathBuf> {
    if let Some(folder) = std::env::var_os("LITMUS_CACHE_DIR") {
        return Some(PathBuf::from(folder));
    }
    let home = |var: &str| std::env::var_os(var).map(PathBuf::from);
    if cfg!(windows) {
        home("LOCALAPPDATA").map(|base| base.join("litmus").join("cache"))
    } else if cfg!(target_os = "macos") {
        home("HOME").map(|base| base.join("Library").join("Caches").join("litmus"))
    } else {
        home("XDG_CACHE_HOME").or_else(|| home("HOME").map(|base| base.join(".cache"))).map(|base| base.join("litmus"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Cache {
    folder: PathBuf,
    model: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl Cache {
    /// A cache for one model, named by the hash of its files and the engine's version.
    pub fn open(folder: &Path, model_files: &[&[u8]]) -> Self {
        Self { folder: folder.to_owned(), model: model_key(model_files) }
    }

    fn path(&self, content_hash: &str) -> PathBuf {
        self.folder.join(&self.model).join(format!("{content_hash}.json"))
    }

    /// The folder holding this model's readings.
    pub fn model_folder(&self) -> PathBuf {
        self.folder.join(&self.model)
    }

    pub fn get(&self, content_hash: &str) -> Option<DocumentReading> {
        let bytes = std::fs::read(self.path(content_hash)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Store a reading, atomically; a failure to write only loses the cache entry.
    pub fn put(&self, content_hash: &str, reading: &DocumentReading) {
        let path = self.path(content_hash);
        let Some(parent) = path.parent() else { return };
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
        let partial = path.with_extension("json.partial");
        if let Ok(bytes) = serde_json::to_vec(reading)
            && std::fs::write(&partial, bytes).is_ok()
        {
            let _ = std::fs::rename(&partial, &path);
        }
    }

    /// The number of readings cached for this model and their bytes.
    pub fn size(&self) -> (usize, u64) {
        let Ok(entries) = std::fs::read_dir(self.model_folder()) else { return (0, 0) };
        entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|e| e == "json"))
            .fold((0, 0), |(count, bytes), entry| (count + 1, bytes + entry.metadata().map_or(0, |m| m.len())))
    }

    /// Delete this model's cached readings (other models' stay).
    pub fn clear(&self) -> std::io::Result<()> {
        match std::fs::remove_dir_all(self.model_folder()) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            other => other,
        }
    }
}
