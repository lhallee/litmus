//! Scoring off the UI thread. Three implementations behind one surface (`Scorer`): on the desktop a worker thread
//! owns the engine and the disk cache; in the browser a Web Worker runs a second instance of this module and the
//! engine, so the page never freezes and a cancel is a terminate; in the contribute build the site's server scores.
//! The UI sends jobs and polls `Event`s; every implementation wakes the UI when an event arrives.

use std::sync::Arc;

use litmus_engine::DocumentReading;

use crate::model::ModelMeta;

/// A finished reading and what it was made from.
#[derive(Clone, Debug)]
pub struct Scored {
    pub reading: Arc<DocumentReading>,
    pub text: Arc<String>,
    /// SHA-256 of the input's bytes (a file's, or the text's).
    pub hash: String,
    /// Seconds the scoring took; `None` when the reading came from the cache.
    pub seconds: Option<f64>,
    /// The contribute site's record of the text, when the site scored it.
    pub contribution: Option<Contribution>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Contribution {
    pub id: String,
    pub model_version: String,
    pub terms_version: String,
    pub annotation_window_seconds: u64,
    pub received_ms: i64,
}

/// How far a job has come.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Progress {
    pub fraction: f32,
    pub words: String,
}

impl Progress {
    /// From the engine's progress: the counting pass is the first 5%, the encoder the rest.
    pub fn of(p: &litmus_engine::Progress) -> Self {
        if p.tokens_total == 0 {
            let fraction = 0.05 * p.bytes_done as f32 / p.bytes_total.max(1) as f32;
            return Self { fraction, words: "Counting tokens".into() };
        }
        let fraction = 0.05 + 0.95 * p.tokens_done as f32 / p.tokens_total as f32;
        let words = format!(
            "{} of {} tokens{}",
            litmus_files::export::thousands(p.tokens_done),
            litmus_files::export::thousands(p.tokens_total),
            if p.chunks_total > 1 { format!(", section {} of {}", (p.chunks_done + 1).min(p.chunks_total), p.chunks_total) } else { String::new() }
        );
        Self { fraction: fraction.min(1.0), words }
    }
}

pub enum Event {
    Loaded(Result<ModelMeta, String>),
    Progress { job: u64, progress: Progress },
    Scored { job: u64, result: Result<Scored, String> },
    /// One file of a batch, by its index in the batch.
    BatchFile { job: u64, index: usize, result: Result<Scored, String> },
    BatchDone { job: u64, cancelled: bool },
    /// Bytes of the model fetched so far and in all (the browser's worker).
    Downloading { done: u64, total: u64 },
    /// The surprise reading of the text whose input hash is `hash` (SPEC.md, Surprise), read only on request.
    #[cfg(not(feature = "contribute"))]
    Surprised { hash: String, result: Result<litmus_engine::Surprise, String> },
    #[cfg(feature = "contribute")]
    Terms(Result<crate::contribute::Terms, String>),
    #[cfg(feature = "contribute")]
    Annotated { id: String, result: Result<Option<crate::contribute::Annotation>, String> },
    /// The site's answer to a removal request for `text`: the SHA-256 it will remove, or why not.
    #[cfg(feature = "contribute")]
    Removed { text: String, result: Result<String, String> },
}

pub const CANCELLED: &str = "Cancelled.";
/// The error of a text with no tokens to read.
pub const EMPTY: &str = "The text has no tokens to read.";

#[cfg(not(target_arch = "wasm32"))]
pub use native::Scorer;
#[cfg(all(target_arch = "wasm32", not(feature = "contribute")))]
pub use worker::Scorer;
#[cfg(all(target_arch = "wasm32", feature = "contribute"))]
pub use server::Scorer;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use eframe::egui;
    use litmus_engine::{Engine, Error, Flow};
    use litmus_files::cache::{self, Cache};
    use litmus_files::extract;

    use super::{CANCELLED, Event, Progress, Scored};
    use crate::model::{self, ModelMeta};

    enum Job {
        Load(Option<PathBuf>),
        Score { job: u64, text: Arc<String>, cancel: Arc<AtomicBool> },
        File { job: u64, path: PathBuf, cancel: Arc<AtomicBool> },
        Batch { job: u64, files: Vec<PathBuf>, cancel: Arc<AtomicBool> },
        Surprise { hash: String, text: Arc<String>, cancel: Arc<AtomicBool> },
        ClearCache,
    }

    /// What both scoring threads read: the loaded engine (`Send` and `Sync`, so one copy serves both) and its cache.
    /// The main thread's loads write it; each reading takes what is there when it starts.
    struct Loaded {
        engine: Option<Arc<Engine>>,
        cache: Option<Arc<Cache>>,
        use_cache: bool,
    }

    type Shared = Arc<Mutex<Loaded>>;

    /// Scoring off the UI thread, on two threads: the main window's readings and batches on one, quick check's on
    /// another, so that a quick check never waits behind a batch or a long reading. Both share the one engine.
    pub struct Scorer {
        jobs: mpsc::Sender<Job>,
        quick: mpsc::Sender<Job>,
        events: mpsc::Receiver<Event>,
        loaded: Shared,
        cancel: Arc<AtomicBool>,
        /// Quick check's newest job and its cancel.
        quick_cancel: (u64, Arc<AtomicBool>),
        /// The running surprise reading's cancel: a new reading or another surprise stops it, so it never delays them.
        surprise_cancel: Arc<AtomicBool>,
        next: u64,
    }

    impl Scorer {
        pub fn new(ctx: &egui::Context) -> Self {
            let (jobs, job_rx) = mpsc::channel();
            let (quick, quick_rx) = mpsc::channel();
            let (event_tx, events) = mpsc::channel();
            let loaded: Shared = Arc::new(Mutex::new(Loaded { engine: None, cache: None, use_cache: true }));
            let main = Worker { events: event_tx.clone(), ctx: ctx.clone(), loaded: loaded.clone() };
            std::thread::Builder::new().name("litmus-scorer".into()).spawn(move || main.run(job_rx)).expect("the scoring thread starts");
            let quick_worker = Worker { events: event_tx, ctx: ctx.clone(), loaded: loaded.clone() };
            std::thread::Builder::new().name("litmus-quick".into()).spawn(move || quick_worker.run(quick_rx)).expect("quick check's scoring thread starts");
            Self {
                jobs,
                quick,
                events,
                loaded,
                cancel: Arc::new(AtomicBool::new(false)),
                quick_cancel: (0, Arc::new(AtomicBool::new(false))),
                surprise_cancel: Arc::new(AtomicBool::new(false)),
                next: 1,
            }
        }

        fn send(&self, job: Job) {
            self.jobs.send(job).expect("the scoring thread is alive");
        }

        /// Load the release from the chosen folder, else the usual places (`model::native::find`).
        pub fn load(&mut self, chosen: Option<String>) {
            self.send(Job::Load(chosen.map(PathBuf::from)));
        }

        /// A new cancel for the main window's next job; a surprise reading still running stops first.
        fn fresh_cancel(&mut self) -> Arc<AtomicBool> {
            self.surprise_cancel.store(true, Ordering::Relaxed);
            self.cancel = Arc::new(AtomicBool::new(false));
            self.cancel.clone()
        }

        /// Read a text's surprise (SPEC.md, Surprise) on the main scoring thread; it ends with `Event::Surprised`
        /// for `hash`, `CANCELLED` when a new reading or another surprise stopped it.
        pub fn surprise(&mut self, text: Arc<String>, hash: String) {
            self.surprise_cancel.store(true, Ordering::Relaxed);
            self.surprise_cancel = Arc::new(AtomicBool::new(false));
            self.send(Job::Surprise { hash, text, cancel: self.surprise_cancel.clone() });
        }

        pub fn score(&mut self, text: Arc<String>) -> u64 {
            let job = self.next;
            self.next += 1;
            let cancel = self.fresh_cancel();
            self.send(Job::Score { job, text, cancel });
            job
        }

        /// Score quick check's text on its own thread, at once even while the main window reads; the main window's
        /// cancel does not reach it, `cancel_quiet` does.
        pub fn score_quiet(&mut self, text: Arc<String>) -> u64 {
            let job = self.next;
            self.next += 1;
            self.quick_cancel = (job, Arc::new(AtomicBool::new(false)));
            self.quick.send(Job::Score { job, text, cancel: self.quick_cancel.1.clone() }).expect("quick check's scoring thread is alive");
            job
        }

        /// Stop quick check's reading `job` if it is the newest (its popup closed or a new one replaced it); it ends
        /// with `CANCELLED`.
        pub fn cancel_quiet(&mut self, job: u64) {
            if self.quick_cancel.0 == job {
                self.quick_cancel.1.store(true, Ordering::Relaxed);
            }
        }

        /// Read a file's text and score it, both on the scoring thread: a large PDF takes seconds to read, and the
        /// window must not freeze meanwhile. The reading's `text` is the file's.
        pub fn score_file(&mut self, path: PathBuf) -> u64 {
            let job = self.next;
            self.next += 1;
            let cancel = self.fresh_cancel();
            self.send(Job::File { job, path, cancel });
            job
        }

        pub fn batch(&mut self, files: Vec<PathBuf>) -> u64 {
            let job = self.next;
            self.next += 1;
            let cancel = self.fresh_cancel();
            self.send(Job::Batch { job, files, cancel });
            job
        }

        pub fn cancel(&mut self) {
            self.cancel.store(true, Ordering::Relaxed);
        }

        pub fn set_cache(&mut self, on: bool) {
            self.loaded.lock().expect("the loaded model").use_cache = on;
        }

        pub fn clear_cache(&mut self) {
            self.send(Job::ClearCache);
        }

        pub fn poll(&mut self) -> Vec<Event> {
            self.events.try_iter().collect()
        }
    }

    struct Worker {
        events: mpsc::Sender<Event>,
        ctx: egui::Context,
        loaded: Shared,
    }

    impl Worker {
        /// The main window's logic drains the events, so it is the one woken; a plain `request_repaint` from this
        /// thread reaches whichever window is mid-pass, quick check's popup included.
        fn emit(&self, event: Event) {
            let _ = self.events.send(event);
            self.ctx.request_repaint_of(egui::ViewportId::ROOT);
        }

        fn run(self, jobs: mpsc::Receiver<Job>) {
            for job in jobs {
                match job {
                    Job::Load(folder) => {
                        let loaded = self.load(folder.as_deref());
                        self.emit(Event::Loaded(loaded));
                    }
                    Job::Score { job, text, cancel } => {
                        let hash = cache::sha256(text.as_bytes());
                        let result = self.read(job, text, hash, &cancel, "");
                        self.emit(Event::Scored { job, result });
                    }
                    Job::File { job, path, cancel } => {
                        self.emit(Event::Progress { job, progress: Progress { fraction: 0.0, words: "Opening the file".into() } });
                        let result = self.read_file(job, &path, &cancel, "");
                        self.emit(Event::Scored { job, result });
                    }
                    Job::Batch { job, files, cancel } => {
                        let total = files.len();
                        for (index, file) in files.into_iter().enumerate() {
                            if cancel.load(Ordering::Relaxed) {
                                break;
                            }
                            let label = format!("File {} of {total}: ", index + 1);
                            let result = self.read_file(job, &file, &cancel, &label);
                            self.emit(Event::BatchFile { job, index, result });
                        }
                        self.emit(Event::BatchDone { job, cancelled: cancel.load(Ordering::Relaxed) });
                    }
                    Job::Surprise { hash, text, cancel } => {
                        let result = self.surprise(&text, &cancel);
                        self.emit(Event::Surprised { hash, result });
                    }
                    Job::ClearCache => {
                        let cache = self.loaded.lock().expect("the loaded model").cache.clone();
                        if let Some(cache) = cache {
                            let _ = cache.clear();
                        }
                    }
                }
            }
        }

        fn load(&self, chosen: Option<&Path>) -> Result<ModelMeta, String> {
            {
                let mut loaded = self.loaded.lock().expect("the loaded model");
                (loaded.engine, loaded.cache) = (None, None);
            }
            let files = model::native::find(chosen)?;
            let engine = Engine::load(&files.model, &files.config, &files.tokenizer).map_err(|e| e.to_string())?;
            let config: serde_json::Value = serde_json::from_slice(&files.config).map_err(|e| format!("config.json: {e}"))?;
            let version = cache::release_version(&config["provenance"], &files.model, &files.config);
            let meta = ModelMeta::from_config(&config, version, files.source.clone())?;
            let cache = cache::folder().map(|folder| Arc::new(Cache::open(&folder, &[&files.model, &files.config, &files.tokenizer])));
            let mut loaded = self.loaded.lock().expect("the loaded model");
            (loaded.engine, loaded.cache) = (Some(Arc::new(engine)), cache);
            Ok(meta)
        }

        /// A text's surprise reading from the loaded engine (never cached: it is read only on request).
        fn surprise(&self, text: &str, cancel: &AtomicBool) -> Result<litmus_engine::Surprise, String> {
            let engine = self.loaded.lock().expect("the loaded model").engine.clone().ok_or("No model is loaded.")?;
            let outcome = engine.surprise_with(text, &mut |_| if cancel.load(Ordering::Relaxed) { Flow::Cancel } else { Flow::Continue });
            match outcome {
                Ok(surprise) => Ok(surprise),
                Err(Error::Cancelled) => Err(CANCELLED.into()),
                Err(Error::Empty) => Err(super::EMPTY.into()),
                Err(error) => Err(error.to_string()),
            }
        }

        fn read_file(&self, job: u64, path: &Path, cancel: &AtomicBool, label: &str) -> Result<Scored, String> {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let hash = cache::sha256(&bytes);
            let text = extract::text_of(path, &bytes)?;
            self.read(job, Arc::new(text), hash, cancel, label)
        }

        /// A reading from the cache, else from the engine with progress and cancel.
        fn read(&self, job: u64, text: Arc<String>, hash: String, cancel: &AtomicBool, label: &str) -> Result<Scored, String> {
            let (engine, cache) = {
                let loaded = self.loaded.lock().expect("the loaded model");
                (loaded.engine.clone(), loaded.cache.clone().filter(|_| loaded.use_cache))
            };
            let engine = engine.ok_or("No model is loaded.")?;
            if let Some(reading) = cache.as_ref().and_then(|cache| cache.get(&hash)) {
                return Ok(Scored { reading: Arc::new(reading), text, hash, seconds: None, contribution: None });
            }
            let started = Instant::now();
            let mut last = Instant::now() - Duration::from_secs(1);
            let outcome = engine.score_with(&text, &mut |p| {
                if cancel.load(Ordering::Relaxed) {
                    return Flow::Cancel;
                }
                if last.elapsed() >= Duration::from_millis(50) {
                    last = Instant::now();
                    let mut progress = Progress::of(&p);
                    progress.words.insert_str(0, label);
                    self.emit(Event::Progress { job, progress });
                }
                Flow::Continue
            });
            let reading = match outcome {
                Ok(reading) => reading,
                Err(Error::Cancelled) => return Err(CANCELLED.into()),
                Err(Error::Empty) => return Err(super::EMPTY.into()),
                Err(error) => return Err(error.to_string()),
            };
            let seconds = started.elapsed().as_secs_f64();
            if let Some(cache) = cache {
                cache.put(&hash, &reading);
            }
            Ok(Scored { reading: Arc::new(reading), text, hash, seconds: Some(seconds), contribution: None })
        }
    }
}

/// The browser's scorer: a module Web Worker (`worker.js`, beside the page) loads this same WebAssembly module,
/// fetches the release's three files from the page's own origin, and scores. Nothing else crosses the network.
#[cfg(all(target_arch = "wasm32", not(feature = "contribute")))]
mod worker {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use std::sync::Arc;

    use eframe::egui;
    use litmus_engine::DocumentReading;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    use super::{Event, Progress, Scored, CANCELLED};
    use crate::model::ModelMeta;

    /// What the worker posts back.
    enum Message {
        Loaded { config: String, version: String },
        Download { done: u64, total: u64 },
        Failed(String),
        Progress { job: u64, progress: Progress },
        Reading { job: u64, json: String, ms: f64 },
        ScoreFailed { job: u64, message: String },
        Surprise { job: u64, json: String },
    }

    fn field(object: &JsValue, name: &str) -> JsValue {
        js_sys::Reflect::get(object, &JsValue::from_str(name)).unwrap_or(JsValue::UNDEFINED)
    }

    fn number(object: &JsValue, name: &str) -> f64 {
        field(object, name).as_f64().unwrap_or(0.0)
    }

    fn string(object: &JsValue, name: &str) -> String {
        field(object, name).as_string().unwrap_or_default()
    }

    fn message(data: &JsValue) -> Option<Message> {
        let job = number(data, "job") as u64;
        Some(match string(data, "type").as_str() {
            "loaded" => Message::Loaded { config: string(data, "config"), version: string(data, "version") },
            "download" => Message::Download { done: number(data, "done") as u64, total: number(data, "total") as u64 },
            "error" if field(data, "job").is_undefined() => Message::Failed(string(data, "message")),
            "error" => Message::ScoreFailed { job, message: string(data, "message") },
            "progress" => {
                let p = litmus_engine::Progress {
                    bytes_done: number(data, "bytes_done") as usize,
                    bytes_total: number(data, "bytes_total") as usize,
                    tokens_done: number(data, "tokens_done") as usize,
                    tokens_total: number(data, "tokens_total") as usize,
                    chunks_done: number(data, "chunks_done") as usize,
                    chunks_total: number(data, "chunks_total") as usize,
                };
                Message::Progress { job, progress: Progress::of(&p) }
            }
            "reading" => Message::Reading { job, json: string(data, "json"), ms: number(data, "ms") },
            "surprise" => Message::Surprise { job, json: string(data, "json") },
            _ => return None,
        })
    }

    struct Pending {
        job: u64,
        text: Arc<String>,
        hash: String,
    }

    pub struct Scorer {
        ctx: egui::Context,
        worker: Option<web_sys::Worker>,
        inbox: Rc<RefCell<Vec<Message>>>,
        _on_message: Option<Closure<dyn FnMut(web_sys::MessageEvent)>>,
        _on_error: Option<Closure<dyn FnMut(web_sys::ErrorEvent)>>,
        pending: Option<Pending>,
        /// The surprise reading the worker is reading: its job and the input hash it answers for.
        surprise: Option<(u64, String)>,
        /// Readings of this session by the hash of their text: an unchanged text is not scored twice.
        cache: HashMap<String, Arc<DocumentReading>>,
        ready: bool,
        out: Vec<Event>,
        next: u64,
    }

    impl Scorer {
        pub fn new(ctx: &egui::Context) -> Self {
            Self {
                ctx: ctx.clone(),
                worker: None,
                inbox: Rc::new(RefCell::new(Vec::new())),
                _on_message: None,
                _on_error: None,
                pending: None,
                surprise: None,
                cache: HashMap::new(),
                ready: false,
                out: Vec::new(),
                next: 1,
            }
        }

        /// Start the worker and have it fetch and load the release.
        pub fn load(&mut self, _chosen: Option<String>) {
            if let Some(worker) = self.worker.take() {
                // Detach the handlers first: their closures are dropped below, and a late message must not reach them.
                worker.set_onmessage(None);
                worker.set_onerror(None);
                worker.terminate();
            }
            self.ready = false;
            let options = web_sys::WorkerOptions::new();
            options.set_type(web_sys::WorkerType::Module);
            let worker = match web_sys::Worker::new_with_options("./worker.js", &options) {
                Ok(worker) => worker,
                Err(error) => {
                    self.out.push(Event::Loaded(Err(format!("The page could not start its worker: {error:?}"))));
                    return;
                }
            };
            let (inbox, ctx) = (self.inbox.clone(), self.ctx.clone());
            let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
                if let Some(message) = message(&event.data()) {
                    inbox.borrow_mut().push(message);
                    ctx.request_repaint();
                }
            });
            let (inbox, ctx) = (self.inbox.clone(), self.ctx.clone());
            let on_error = Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(move |event: web_sys::ErrorEvent| {
                inbox.borrow_mut().push(Message::Failed(format!("The worker stopped: {}", event.message())));
                ctx.request_repaint();
            });
            worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
            worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
            let load = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&load, &"type".into(), &"load".into());
            if let Err(error) = worker.post_message(&load) {
                self.out.push(Event::Loaded(Err(format!("The page could not reach its worker: {error:?}"))));
            }
            self.worker = Some(worker);
            self._on_message = Some(on_message);
            self._on_error = Some(on_error);
        }

        pub fn score(&mut self, text: Arc<String>) -> u64 {
            let job = self.next;
            self.next += 1;
            let hash = litmus_files::cache::sha256(text.as_bytes());
            if let Some(reading) = self.cache.get(&hash) {
                self.out.push(Event::Scored { job, result: Ok(Scored { reading: reading.clone(), text, hash, seconds: None, contribution: None }) });
                self.ctx.request_repaint();
                return job;
            }
            let Some(worker) = &self.worker else {
                self.out.push(Event::Scored { job, result: Err("The model is not loaded.".into()) });
                return job;
            };
            let request = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&request, &"type".into(), &"score".into());
            let _ = js_sys::Reflect::set(&request, &"job".into(), &JsValue::from_f64(job as f64));
            let _ = js_sys::Reflect::set(&request, &"text".into(), &JsValue::from_str(&text));
            if let Err(error) = worker.post_message(&request) {
                self.out.push(Event::Scored { job, result: Err(format!("The page could not reach its worker: {error:?}")) });
                return job;
            }
            self.pending = Some(Pending { job, text, hash });
            job
        }

        /// Read a text's surprise (SPEC.md, Surprise) in the worker, after any reading already sent to it; it ends
        /// with `Event::Surprised` for `hash`.
        pub fn surprise(&mut self, text: Arc<String>, hash: String) {
            let job = self.next;
            self.next += 1;
            let Some(worker) = &self.worker else {
                self.out.push(Event::Surprised { hash, result: Err("The model is not loaded.".into()) });
                return;
            };
            let request = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&request, &"type".into(), &"surprise".into());
            let _ = js_sys::Reflect::set(&request, &"job".into(), &JsValue::from_f64(job as f64));
            let _ = js_sys::Reflect::set(&request, &"text".into(), &JsValue::from_str(&text));
            if let Err(error) = worker.post_message(&request) {
                self.out.push(Event::Surprised { hash, result: Err(format!("The page could not reach its worker: {error:?}")) });
                return;
            }
            self.surprise = Some((job, hash));
        }

        /// Stop the running reading: the worker is terminated and a fresh one loads the release again, so a surprise
        /// reading it held ends too.
        pub fn cancel(&mut self) {
            if let Some(pending) = self.pending.take() {
                self.out.push(Event::Scored { job: pending.job, result: Err(CANCELLED.into()) });
                if let Some((_, hash)) = self.surprise.take() {
                    self.out.push(Event::Surprised { hash, result: Err(CANCELLED.into()) });
                }
                self.load(None);
            }
        }

        pub fn poll(&mut self) -> Vec<Event> {
            let messages: Vec<Message> = self.inbox.borrow_mut().drain(..).collect();
            for message in messages {
                match message {
                    Message::Loaded { config, version } => {
                        self.ready = true;
                        let meta = serde_json::from_str::<serde_json::Value>(&config)
                            .map_err(|e| format!("config.json: {e}"))
                            .and_then(|config| ModelMeta::from_config(&config, version, "this site".into()));
                        self.out.push(Event::Loaded(meta));
                    }
                    Message::Failed(error) => {
                        if let Some(pending) = self.pending.take() {
                            self.out.push(Event::Scored { job: pending.job, result: Err(error.clone()) });
                        }
                        if let Some((_, hash)) = self.surprise.take() {
                            self.out.push(Event::Surprised { hash, result: Err(error.clone()) });
                        }
                        if !self.ready {
                            self.out.push(Event::Loaded(Err(error)));
                        }
                    }
                    Message::Progress { job, progress } => self.out.push(Event::Progress { job, progress }),
                    Message::Download { done, total } => self.out.push(Event::Downloading { done, total }),
                    Message::ScoreFailed { job, message } => {
                        if self.pending.as_ref().is_some_and(|p| p.job == job) {
                            self.pending = None;
                            self.out.push(Event::Scored { job, result: Err(message) });
                        } else if let Some((_, hash)) = self.surprise.take_if(|(surprise, _)| *surprise == job) {
                            self.out.push(Event::Surprised { hash, result: Err(message) });
                        }
                    }
                    Message::Surprise { job, json } => {
                        let Some((_, hash)) = self.surprise.take_if(|(surprise, _)| *surprise == job) else { continue };
                        let result = serde_json::from_str::<litmus_engine::Surprise>(&json).map_err(|e| format!("the surprise reading could not be read: {e}"));
                        self.out.push(Event::Surprised { hash, result });
                    }
                    Message::Reading { job, json, ms } => {
                        let Some(pending) = self.pending.take_if(|p| p.job == job) else { continue };
                        let result = serde_json::from_str::<DocumentReading>(&json).map_err(|e| format!("the reading could not be read: {e}")).map(|reading| {
                            let reading = Arc::new(reading);
                            self.cache.insert(pending.hash.clone(), reading.clone());
                            Scored { reading, text: pending.text, hash: pending.hash, seconds: Some(ms / 1000.0), contribution: None }
                        });
                        self.out.push(Event::Scored { job, result });
                    }
                }
            }
            std::mem::take(&mut self.out)
        }
    }
}

/// The contribute build's scorer: the site's server reads the text (`contribute::fetch`).
#[cfg(all(target_arch = "wasm32", feature = "contribute"))]
mod server {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use eframe::egui;

    use super::{Contribution, Event, Scored, CANCELLED};
    use crate::contribute::{self, fetch, Annotation};

    pub struct Scorer {
        ctx: egui::Context,
        inbox: Rc<RefCell<Vec<Event>>>,
        terms_version: Option<String>,
        /// The visitor's consent box, as the app last set it; no text is sent without it.
        consent: bool,
        running: Option<u64>,
        next: u64,
    }

    /// The site's terms, read again.
    async fn read_terms() -> Event {
        let terms = match fetch::call("GET", contribute::TERMS_PATH, None).await {
            Ok((status, body)) => contribute::parse_terms(status, &body),
            Err(error) => Err(error),
        };
        Event::Terms(terms)
    }

    impl Scorer {
        pub fn new(ctx: &egui::Context) -> Self {
            Self { ctx: ctx.clone(), inbox: Rc::new(RefCell::new(Vec::new())), terms_version: None, consent: false, running: None, next: 1 }
        }

        fn spawn(&self, work: impl std::future::Future<Output = Event> + 'static) {
            let (inbox, ctx) = (self.inbox.clone(), self.ctx.clone());
            wasm_bindgen_futures::spawn_local(async move {
                let event = work.await;
                inbox.borrow_mut().push(event);
                ctx.request_repaint();
            });
        }

        /// Read the site's terms; the model is the site's.
        pub fn load(&mut self, _chosen: Option<String>) {
            self.spawn(read_terms());
        }

        pub fn set_terms_version(&mut self, version: String) {
            self.terms_version = Some(version);
        }

        pub fn set_consent(&mut self, consent: bool) {
            self.consent = consent;
        }

        /// Send the text to the site, only with the visitor's consent. A 409 (the terms changed) reads the terms
        /// again, which unticks the box (`ContributeState::set_terms`).
        pub fn score(&mut self, text: Arc<String>) -> u64 {
            let job = self.next;
            self.next += 1;
            let terms_version = match (&self.terms_version, self.consent) {
                (Some(version), true) => version.clone(),
                (None, _) => {
                    self.inbox.borrow_mut().push(Event::Scored { job, result: Err("The terms have not loaded yet.".into()) });
                    return job;
                }
                (Some(_), false) => {
                    self.inbox.borrow_mut().push(Event::Scored { job, result: Err("Tick the box above the Check button to agree first.".into()) });
                    return job;
                }
            };
            self.running = Some(job);
            let inbox = self.inbox.clone();
            self.spawn(async move {
                let body = contribute::score_body(&text, &terms_version, None);
                let sent_ms = crate::clock::now_ms();
                let answer = fetch::call("POST", contribute::SCORE_PATH, Some(&body)).await;
                if matches!(answer, Ok((409, _))) {
                    let terms = read_terms().await;
                    inbox.borrow_mut().push(terms);
                    return Event::Scored { job, result: Err(contribute::TERMS_CHANGED.into()) };
                }
                let result = match answer {
                    Ok((status, body)) => contribute::parse_score(status, &body).map(|response| Scored {
                        reading: Arc::new(response.reading),
                        hash: litmus_files::cache::sha256(text.as_bytes()),
                        text,
                        seconds: Some((crate::clock::now_ms() - sent_ms) as f64 / 1000.0),
                        contribution: Some(Contribution {
                            id: response.id,
                            model_version: response.model_version,
                            terms_version: response.terms_version,
                            annotation_window_seconds: response.annotation_window_seconds,
                            received_ms: crate::clock::now_ms(),
                        }),
                    }),
                    Err(error) => Err(error),
                };
                Event::Scored { job, result }
            });
            job
        }

        pub fn annotate(&mut self, id: String, annotation: Option<Annotation>) {
            self.spawn(async move {
                let result = match contribute::annotation_path(&id) {
                    Ok(path) => match fetch::call("POST", &path, Some(&contribute::annotation_body(annotation))).await {
                        Ok((status, body)) => contribute::parse_annotation(status, &body, &id),
                        Err(error) => Err(error),
                    },
                    Err(error) => Err(error),
                };
                Event::Annotated { id, result }
            });
        }

        /// Ask the site to delete every stored copy of `text` (open to anyone, no consent needed).
        pub fn remove(&mut self, text: String) {
            self.spawn(async move {
                let result = match fetch::call("POST", contribute::REMOVALS_PATH, Some(&contribute::removal_body(&text))).await {
                    Ok((status, body)) => contribute::parse_removal(status, &body, &text),
                    Err(error) => Err(error),
                };
                Event::Removed { text, result }
            });
        }

        /// The request cannot be recalled; its answer is dropped and the text stays stored by the site.
        pub fn cancel(&mut self) {
            if let Some(job) = self.running.take() {
                self.inbox.borrow_mut().push(Event::Scored { job, result: Err(CANCELLED.into()) });
                self.ctx.request_repaint();
            }
        }

        pub fn poll(&mut self) -> Vec<Event> {
            let events: Vec<Event> = self.inbox.borrow_mut().drain(..).collect();
            events
                .into_iter()
                .filter(|event| match event {
                    // A cancelled job's late answer is dropped.
                    Event::Scored { job, result: Ok(_) } => self.running.take_if(|running| running == job).is_some(),
                    Event::Scored { job, .. } => {
                        self.running.take_if(|running| running == job);
                        true
                    }
                    _ => true,
                })
                .collect()
        }
    }
}
