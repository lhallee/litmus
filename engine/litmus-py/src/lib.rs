//! `litmus_engine`, the Python binding of the Litmus engine: a thin wrapper, every computation in `litmus-engine`.
//!
//! The reading is `Engine::score_with`'s `DocumentReading`, serialized by serde exactly as `litmus score --json`
//! prints it under `reading`. Scoring releases the GIL, so threads score in parallel on one shared model, and stops
//! early on Ctrl-C, a timeout, or a text past a token cap.
//!
//! This crate reads the three release files itself instead of depending on `litmus-files`, so the wheel holds no
//! PDF reader, child process, or reading cache: the binding never writes a file. The two checks it shares with
//! `litmus_files::release::read_folder` (the tokenizer beside the release, its pinned SHA-256) and the version
//! rule it shares with `litmus_files::cache::release_version` are tested against the command line in `tests/`.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use litmus_engine::{Backend, Engine as Core, Error, Flow};
use pyo3::exceptions::{PyFileNotFoundError, PyKeyboardInterrupt, PyOSError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

/// A loaded Litmus release. The module also names it `Model`. Immutable once loaded, so one instance is shared by
/// every thread; each `score` call keeps its own buffers.
#[pyclass(name = "Engine", module = "litmus_engine", frozen)]
struct Engine {
    core: Core,
    version: String,
}

/// Why a scoring stopped before the engine finished, when the binding asked it to.
enum Stop {
    Timeout,
    OverCap(usize),
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The release's version: its provenance's `version` if it names one, else `sha256:` and the first 16 hex digits of
/// the hash of `model.safetensors` then `config.json` (the rule of `litmus_files::cache::release_version`).
fn release_version(core: &Core, model: &[u8], config: &[u8]) -> String {
    if let Some(version) = core.info().provenance.get("version").and_then(|v| v.as_str()) {
        return version.to_owned();
    }
    let digest = Sha256::new().chain_update(model).chain_update(config).finalize();
    format!("sha256:{}", hex(&digest[..8]))
}

/// Refuse a tokenizer whose SHA-256 differs from the one `config.json` pins, as the command line does.
fn check_tokenizer_pin(config: &[u8], tokenizer: &[u8]) -> PyResult<()> {
    let parsed: serde_json::Value =
        serde_json::from_slice(config).map_err(|e| PyValueError::new_err(format!("cannot load: config.json: {e}")))?;
    if let Some(expected) = parsed["tokenizer"]["sha256"].as_str() {
        let actual = hex(&Sha256::digest(tokenizer));
        if actual != expected {
            return Err(PyValueError::new_err(format!("cannot load: the tokenizer has sha256 {actual}, the release expects {expected}")));
        }
    }
    Ok(())
}

fn read_file(path: &Path) -> PyResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| match e.kind() {
        ErrorKind::NotFound => PyFileNotFoundError::new_err(format!("{}: no such file", path.display())),
        _ => PyOSError::new_err(format!("{}: {e}", path.display())),
    })
}

/// The release folder of a path: the folder itself, or the folder holding a `model.safetensors` named directly.
fn release_folder(path: &Path) -> PyResult<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_owned());
    }
    if !path.exists() {
        return Err(PyFileNotFoundError::new_err(format!("{}: no such release folder", path.display())));
    }
    if path.file_name().is_some_and(|name| name == "model.safetensors") {
        return Ok(path.parent().map_or_else(PathBuf::new, Path::to_owned));
    }
    Err(PyValueError::new_err(format!(
        "{} is not a release: pass its folder (model.safetensors, config.json, tokenizer.json) or its model.safetensors",
        path.display()
    )))
}

fn error(e: Error) -> PyErr {
    match e {
        Error::Empty => PyValueError::new_err("the text has no tokens, so no reading"),
        Error::Load(message) => PyValueError::new_err(format!("cannot load: {message}")),
        Error::Backend(name) => PyValueError::new_err(format!("backend {name} is not available on this processor")),
        Error::Cancelled => PyKeyboardInterrupt::new_err("scoring was cancelled"),
    }
}

/// A value of the library as JSON text, as `litmus score --json` writes it.
fn json_text(value: &impl serde::Serialize) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(format!("cannot serialize: {e}")))
}

/// JSON text as a Python object, through the `json` module (dicts, lists, floats, ints, bools, strings).
fn from_json<'py>(py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?.call_method1("loads", (text,))
}

#[pymethods]
impl Engine {
    /// Load a release from the bytes of `model.safetensors`, `config.json` and ModernBERT's `tokenizer.json`.
    #[new]
    #[pyo3(signature = (model, config, tokenizer, backend = None))]
    fn new(py: Python<'_>, model: &[u8], config: &[u8], tokenizer: &[u8], backend: Option<&str>) -> PyResult<Self> {
        check_tokenizer_pin(config, tokenizer)?;
        let mut core = py.allow_threads(|| Core::load(model, config, tokenizer)).map_err(error)?;
        if let Some(name) = backend {
            let backend = Backend::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown backend {name}")))?;
            core.set_backend(backend).map_err(error)?;
        }
        let version = release_version(&core, model, config);
        Ok(Self { core, version })
    }

    /// Load a release from its folder (`model.safetensors`, `config.json`, and `tokenizer.json` unless `tokenizer`
    /// names it) or from the path of its `model.safetensors`.
    #[staticmethod]
    #[pyo3(signature = (path, tokenizer = None, backend = None))]
    fn from_path(py: Python<'_>, path: PathBuf, tokenizer: Option<PathBuf>, backend: Option<&str>) -> PyResult<Self> {
        let folder = release_folder(&path)?;
        let model = read_file(&folder.join("model.safetensors"))?;
        let config = read_file(&folder.join("config.json"))?;
        let tokenizer = read_file(&tokenizer.unwrap_or_else(|| folder.join("tokenizer.json")))?;
        Self::new(py, &model, &config, &tokenizer, backend)
    }

    /// Score a text: the document's reading as a dict (`n_tokens`, `chunk_tokens`, `verdict`, `chunks`, `windows`,
    /// `cells`, plus `label`, the verdict in one word). Releases the GIL while it runs, serialization included.
    /// Raises `ValueError` for a text with no tokens or with more than `max_tokens` tokens, `TimeoutError` past
    /// `timeout_seconds`, and `KeyboardInterrupt` on Ctrl-C; each is checked about every 8 KB of text.
    #[pyo3(signature = (text, *, timeout_seconds = None, max_tokens = None))]
    fn score<'py>(&self, py: Python<'py>, text: &str, timeout_seconds: Option<f64>, max_tokens: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let deadline = timeout_seconds.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds.max(0.0)));
        let mut stop = None;
        let outcome = py.allow_threads(|| {
            let document = self.core.score_with(text, &mut |progress| {
                // The counting pass reports tokens_done with tokens_total 0; the encoding pass reports the total.
                let tokens = progress.tokens_done.max(progress.tokens_total);
                if let Some(cap) = max_tokens
                    && tokens > cap
                {
                    stop = Some(Stop::OverCap(cap));
                    return Flow::Cancel;
                }
                if deadline.is_some_and(|deadline| Instant::now() > deadline) {
                    stop = Some(Stop::Timeout);
                    return Flow::Cancel;
                }
                // Checking for Ctrl-C takes the GIL for a moment; it is a no-op off the main thread.
                if Python::with_gil(|py| py.check_signals()).is_err() { Flow::Cancel } else { Flow::Continue }
            })?;
            Ok::<_, Error>((json_text(&document), document.label().word()))
        });
        let (json, label) = match (outcome, stop) {
            (Err(Error::Cancelled), Some(Stop::Timeout)) => {
                return Err(PyTimeoutError::new_err("scoring took longer than timeout_seconds"));
            }
            (Err(Error::Cancelled), Some(Stop::OverCap(cap))) => {
                return Err(PyValueError::new_err(format!("the text has more than {cap} tokens (max_tokens)")));
            }
            (outcome, _) => outcome.map_err(error)?,
        };
        let value = from_json(py, json?)?;
        value.set_item("label", label)?;
        Ok(value)
    }

    /// The number of tokens `score` reads in a text.
    fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        py.allow_threads(|| self.core.count_tokens(text))
    }

    /// The release's version, stored with every reading the contribute service keeps.
    #[getter]
    fn version(&self) -> &str {
        &self.version
    }

    /// The tokens of a text's standard reading: `(id, start byte, end byte)` into the UTF-8 text.
    fn tokenize(&self, py: Python<'_>, text: &str) -> Vec<(u32, usize, usize)> {
        py.allow_threads(|| self.core.tokenize(text).into_iter().map(|t| (t.id, t.start, t.end)).collect())
    }

    /// The loaded model: widths, depth, bank sizes, levels, license, provenance and the backend in use.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = from_json(py, json_text(self.core.info())?)?;
        value.set_item("backend", self.core.backend().name())?;
        value.set_item("version", &self.version)?;
        Ok(value)
    }

    /// The backends this processor supports.
    #[staticmethod]
    fn backends() -> Vec<&'static str> {
        Backend::available().into_iter().map(Backend::name).collect()
    }

    /// The standard reading of a text (SPEC.md, section 1.1).
    #[staticmethod]
    fn standard_reading(text: &str) -> String {
        litmus_engine::text::standard_reading(text)
    }

    fn __repr__(&self) -> String {
        let info = self.core.info();
        format!("Engine(version={}, width={}, depth={}, backend={})", self.version, info.width, info.depth, self.core.backend().name())
    }
}

#[pymodule]
#[pyo3(name = "litmus_engine")]
fn litmus_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Engine>()?;
    module.add("Model", module.getattr("Engine")?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
