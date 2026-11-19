//! The browser's entry points. `start` runs the app on the page's canvas. `WorkerEngine` is what `worker.js` (the
//! page's Web Worker) drives: it loads the release from the bytes the worker fetched and scores, posting progress.
//! This module makes no network request of its own; the worker's three GETs are the private page's only ones.

use eframe::wasm_bindgen::{self, prelude::*};
use wasm_bindgen::JsCast;

use crate::settings::ThemeChoice;

/// Run the app on the canvas with this id.
#[wasm_bindgen]
pub async fn start(canvas_id: String) -> Result<(), JsValue> {
    let document = web_sys::window().and_then(|w| w.document()).ok_or("no document")?;
    let canvas = document.get_element_by_id(&canvas_id).ok_or("no canvas with that id")?.dyn_into::<web_sys::HtmlCanvasElement>()?;
    eframe::WebRunner::new()
        .start(canvas, eframe::WebOptions::default(), Box::new(|cc| Ok(Box::new(crate::app::LitmusApp::new(cc)))))
        .await
}

fn hash() -> String {
    web_sys::window().and_then(|w| w.location().hash().ok()).unwrap_or_default()
}

/// `#dark` or `#light` in the page's address picks the theme for this visit.
pub fn theme_from_location() -> Option<ThemeChoice> {
    let hash = hash();
    if hash.contains("dark") {
        Some(ThemeChoice::Dark)
    } else if hash.contains("light") {
        Some(ThemeChoice::Light)
    } else {
        None
    }
}

/// `#sample` in the page's address checks the sample passage once the model loads (for demonstrations).
pub fn sample_from_location() -> bool {
    hash().contains("sample")
}

/// Dropped files' names and texts (or why they could not be read), waiting for the app's next frame.
pub type DroppedQueue = std::rc::Rc<std::cell::RefCell<Vec<(String, Result<String, String>)>>>;

/// Read a dropped file (asynchronously, as browsers do) into the queue the app drains each frame.
pub fn read_dropped(file: eframe::egui::DroppedFileHandle, queue: DroppedQueue, ctx: eframe::egui::Context) {
    wasm_bindgen_futures::spawn_local(async move {
        let name = file.path().display().to_string();
        let result = match file.bytes_async().await {
            Ok(bytes) => litmus_files::extract::text_of(file.path(), &bytes),
            Err(error) => Err(error),
        };
        queue.borrow_mut().push((name, result));
        ctx.request_repaint();
    });
}

/// The engine inside the page's worker.
#[cfg(not(feature = "contribute"))]
#[wasm_bindgen]
pub struct WorkerEngine {
    engine: litmus_engine::Engine,
    version: String,
}

#[cfg(not(feature = "contribute"))]
#[wasm_bindgen]
impl WorkerEngine {
    /// Load a release from the bytes of `model.safetensors`, `config.json` and `tokenizer.json`.
    #[wasm_bindgen(constructor)]
    pub fn new(model: &[u8], config: &[u8], tokenizer: &[u8]) -> Result<WorkerEngine, JsError> {
        let engine = litmus_engine::Engine::load(model, config, tokenizer).map_err(|e| JsError::new(&e.to_string()))?;
        let provenance = &engine.info().provenance;
        let version = litmus_files::cache::release_version(provenance, model, config);
        Ok(Self { engine, version })
    }

    /// The release's version (as the Python binding and the contribute site report it).
    pub fn version(&self) -> String {
        self.version.clone()
    }

    /// The reading of a text as JSON. `progress` is called with (tokens done, tokens total, chunks done, chunks
    /// total, bytes done, bytes total) about every 8 KB of text.
    pub fn score(&self, text: &str, progress: &js_sys::Function) -> Result<String, JsError> {
        let reading = self
            .engine
            .score_with(text, &mut |p| {
                let values = js_sys::Array::new();
                for value in [p.tokens_done, p.tokens_total, p.chunks_done, p.chunks_total, p.bytes_done, p.bytes_total] {
                    values.push(&JsValue::from_f64(value as f64));
                }
                let _ = progress.apply(&JsValue::NULL, &values);
                litmus_engine::Flow::Continue
            })
            .map_err(|e| JsError::new(&match e {
                litmus_engine::Error::Empty => crate::scoring::EMPTY.to_owned(),
                other => other.to_string(),
            }))?;
        serde_json::to_string(&reading).map_err(|e| JsError::new(&e.to_string()))
    }
}
