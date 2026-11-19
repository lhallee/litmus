//! The app's state and its frame: events from the scorer, keyboard shortcuts, dropped files, and the three screens
//! (Check, Batch on the desktop, History).

use std::sync::Arc;

use eframe::egui::{self, Key, KeyboardShortcut, Modifiers};
use litmus_files::counts::TextCounts;

use crate::brand;
use crate::clock;
use crate::history::{self, History};
use crate::model::ModelMeta;
use crate::scoring::{self, Event, Progress, Scored, Scorer};
use crate::settings::Settings;
use crate::theme;
use crate::views;
use crate::wording;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Check,
    #[cfg(not(target_arch = "wasm32"))]
    Batch,
    History,
}

pub enum ModelState {
    Loading,
    Ready(ModelMeta),
    Failed(String),
}

/// A reading on screen.
pub struct Shown {
    pub title: String,
    /// "pasted", a file path, or "contribution <id>".
    pub source: String,
    pub scored: Scored,
    /// The words, characters and tokens of the text, counted once when the reading is opened.
    pub counts: TextCounts,
    pub at_ms: i64,
    pub model: String,
}

impl Shown {
    pub fn new(title: String, source: String, scored: Scored, at_ms: i64, model: String) -> Self {
        let counts = TextCounts::of(&scored.text, scored.reading.n_tokens);
        Self { title, source, scored, counts, at_ms, model }
    }
}

pub struct Running {
    pub job: u64,
    pub progress: Progress,
}

/// A message over the screen: an error says what happened and what to do; an info line confirms something done.
pub enum Notice {
    Error { title: String, detail: String },
    Info(String),
}

impl Notice {
    pub fn error(title: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Error { title: title.into(), detail: detail.into() }
    }
}

/// Every keyboard shortcut, listed once (Settings shows this table): keys, what they do, and whether only the
/// desktop has them (a browser keeps Ctrl+O, Ctrl+E and Ctrl+1 to 3 for itself).
pub const SHORTCUTS: &[(&str, &str, bool)] = &[
    ("Ctrl+Enter", "Check the text", false),
    ("Esc", "Cancel a reading, or close Settings", false),
    ("Ctrl+E", "Edit the text again after a reading", true),
    ("Ctrl+O", "Open a file", true),
    ("Ctrl+M", "Show more or less detail (Simple, Detailed, Full)", false),
    ("Ctrl+1, 2, 3", "Check, Batch, History", true),
    ("Ctrl+,", "Settings", false),
    ("Ctrl+Shift+L", "Switch between the light and dark theme", false),
    ("Tab, Shift+Tab", "Move between controls; Enter or Space presses the one in focus", false),
];

/// One file of a batch.
#[cfg(not(target_arch = "wasm32"))]
pub struct BatchRow {
    pub path: std::path::PathBuf,
    pub state: BatchState,
}

#[cfg(not(target_arch = "wasm32"))]
pub enum BatchState {
    Waiting,
    Done(Scored),
    Failed(String),
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub struct Batch {
    pub rows: Vec<BatchRow>,
    /// The running job and the index of the first row it scores.
    pub job: Option<(u64, usize)>,
    pub progress: Progress,
    pub finished: usize,
}

/// The contribute build's state: the site's terms, the visitor's consent, the label of the shown text, and the
/// removal form. All of it lives in memory only: nothing here reaches cookies or the browser's storage, so every
/// page load starts with the box unticked and no contribution's id is kept.
#[cfg(feature = "contribute")]
#[derive(Default)]
pub struct ContributeState {
    pub terms: Option<Result<crate::contribute::Terms, String>>,
    /// The consent box: unticked on every load and whenever the served terms change.
    pub consent: bool,
    /// The site refused a text because its terms changed; said beside the box until it is ticked again.
    pub terms_changed: bool,
    pub annotation: Option<crate::contribute::Annotation>,
    pub annotation_note: Option<String>,
    pub removal: RemovalForm,
}

/// The removal form: open or not, its text, a request on its way, and the site's answer.
#[cfg(feature = "contribute")]
#[derive(Default)]
pub struct RemovalForm {
    pub open: bool,
    pub text: String,
    pub sending: bool,
    pub answer: Option<Result<String, String>>,
}

#[cfg(feature = "contribute")]
impl ContributeState {
    /// New terms from the site (or a failure to read them): the box is unticked unless the same version is served
    /// again.
    pub fn set_terms(&mut self, terms: Result<crate::contribute::Terms, String>) {
        let old = self.terms.as_ref().and_then(|t| t.as_ref().ok());
        if !crate::contribute::consent_survives(old, terms.as_ref().ok()) {
            self.consent = false;
        }
        self.terms = Some(terms);
    }
}

pub struct LitmusApp {
    pub scorer: Scorer,
    pub model: ModelState,
    pub tab: Tab,
    pub input: String,
    /// Where the editor's text came from: "pasted" or a file path.
    pub input_source: String,
    pub running: Option<Running>,
    pub shown: Option<Shown>,
    /// The editor is open over a reading (after Edit); a new reading closes it.
    pub editing: bool,
    /// Scroll the details to "How this was read" on the next frame (the demo's explain scene).
    pub reveal_explain: bool,
    /// Bytes of the model downloaded so far and in all (the browser), while it loads.
    pub download: Option<(u64, u64)>,
    pub selected_cell: Option<usize>,
    pub explain_chunk: usize,
    pub heat: views::heat::Layout,
    pub history: History,
    pub history_filter: String,
    pub history_sort: Option<crate::sorting::Sort<&'static str>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub batch_sort: Option<crate::sorting::Sort<&'static str>>,
    pub confirm_clear: bool,
    pub settings: Settings,
    pub settings_open: bool,
    pub model_folder_edit: String,
    pub notice: Option<Notice>,
    pub mark: Option<egui::TextureHandle>,
    #[cfg(not(target_arch = "wasm32"))]
    pub batch: Batch,
    #[cfg(not(target_arch = "wasm32"))]
    pub demo: Option<crate::demo::Demo>,
    #[cfg(not(target_arch = "wasm32"))]
    pub quick: crate::views::quick::State,
    /// What the command line or a later launch asked for (`--check <file>`, Explorer's entry), done once the model
    /// loads.
    #[cfg(not(target_arch = "wasm32"))]
    pub pending: Vec<crate::instance::Request>,
    /// The hold on the data folder that makes this the one running app.
    #[cfg(not(target_arch = "wasm32"))]
    pub instance: Option<crate::instance::Instance>,
    /// The window's place as last saved, and a newer one waiting to settle before it is saved.
    #[cfg(not(target_arch = "wasm32"))]
    pub window_saved: Option<crate::window_place::WindowPlace>,
    #[cfg(not(target_arch = "wasm32"))]
    pub window_moved: Option<(crate::window_place::WindowPlace, f64)>,
    /// The touch sizes are on (a narrow browser window).
    pub touch: bool,
    #[cfg(target_arch = "wasm32")]
    pub dropped: crate::web::DroppedQueue,
    #[cfg(target_arch = "wasm32")]
    pub score_on_load: bool,
    #[cfg(feature = "contribute")]
    pub contribute: ContributeState,
}

impl LitmusApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = &cc.egui_ctx;
        #[cfg(not(target_arch = "wasm32"))]
        let (settings, settings_error) = match crate::settings::disk::load() {
            Ok(settings) => (settings, None),
            Err(error) => (Settings::default(), Some(format!("Settings could not be read, so the defaults are used: {error}"))),
        };
        #[cfg(target_arch = "wasm32")]
        let (settings, settings_error): (Settings, Option<String>) = (crate::settings::browser::load(), None);
        #[cfg(not(target_arch = "wasm32"))]
        let demo = crate::demo::Demo::from_env();
        #[cfg(not(target_arch = "wasm32"))]
        let theme_choice = demo.as_ref().and_then(|d| d.theme).unwrap_or(settings.theme);
        #[cfg(target_arch = "wasm32")]
        let theme_choice = crate::web::theme_from_location().unwrap_or(settings.theme);
        theme::install(ctx, theme_choice.preference());
        #[cfg(not(target_arch = "wasm32"))]
        let (history, history_error) = match crate::settings::disk::data_folder().map(|folder| History::open(&folder.join("history"))) {
            Some(Ok(history)) => (history, None),
            Some(Err(error)) => (History::in_memory(), Some(format!("The history could not be opened, so this session's is kept in memory only: {error}"))),
            None => (History::in_memory(), None),
        };
        #[cfg(target_arch = "wasm32")]
        let (history, history_error) = (History::in_memory(), None);
        let mut scorer = Scorer::new(ctx);
        #[cfg(not(target_arch = "wasm32"))]
        scorer.set_cache(settings.use_cache);
        scorer.load(settings.model_folder.clone());
        let size = 96;
        let mark = ctx.load_texture("mark", egui::ColorImage::from_rgba_unmultiplied([size, size], &brand::icon_rgba(size)), egui::TextureOptions::LINEAR);
        let notice = settings_error.or(history_error).map(|error| Notice::error("Starting with defaults", error));
        let app = Self {
            scorer,
            model: ModelState::Loading,
            tab: Tab::Check,
            input: String::new(),
            input_source: "pasted".into(),
            running: None,
            shown: None,
            editing: false,
            reveal_explain: false,
            download: None,
            selected_cell: None,
            explain_chunk: 0,
            heat: Default::default(),
            history,
            history_filter: String::new(),
            history_sort: None,
            #[cfg(not(target_arch = "wasm32"))]
            batch_sort: None,
            confirm_clear: false,
            model_folder_edit: settings.model_folder.clone().unwrap_or_default(),
            settings,
            settings_open: false,
            notice,
            mark: Some(mark),
            #[cfg(not(target_arch = "wasm32"))]
            batch: Batch::default(),
            #[cfg(not(target_arch = "wasm32"))]
            demo,
            #[cfg(not(target_arch = "wasm32"))]
            quick: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            pending: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            instance: None,
            #[cfg(not(target_arch = "wasm32"))]
            window_saved: None,
            #[cfg(not(target_arch = "wasm32"))]
            window_moved: None,
            touch: false,
            #[cfg(target_arch = "wasm32")]
            dropped: Default::default(),
            #[cfg(target_arch = "wasm32")]
            score_on_load: crate::web::sample_from_location(),
            #[cfg(feature = "contribute")]
            contribute: ContributeState::default(),
        };
        #[cfg(not(target_arch = "wasm32"))]
        let app = {
            let mut app = app;
            crate::views::quick::start_if_on(&mut app, ctx);
            app
        };
        app
    }

    /// Record a reading in the history, when the user keeps one.
    pub fn record(&mut self, scored: &Scored, title: &str, source: &str) {
        if !self.settings.keep_history {
            return;
        }
        let model = scored.contribution.as_ref().map(|c| c.model_version.clone()).or_else(|| self.meta().map(|m| m.version.clone())).unwrap_or_default();
        let verdict = &scored.reading.verdict;
        let entry = history::Entry {
            id: 0,
            at_ms: clock::now_ms(),
            title: title.to_owned(),
            source: source.to_owned(),
            hash: scored.hash.clone(),
            headline: wording::verdict_headline(verdict),
            probability: verdict.probability,
            interval: verdict.interval,
            position: verdict.strip.position,
            band: Some([verdict.strip.band_lo, verdict.strip.band_hi]),
            unlike_reference: verdict.strip.unlike_reference,
            tokens: scored.reading.n_tokens,
            seconds: scored.seconds,
            model,
        };
        let body = history::Body { text: scored.text.as_str().to_owned(), reading: scored.reading.as_ref().clone() };
        if let Err(error) = self.history.add(entry, body) {
            self.notice = Some(Notice::error("The reading was not saved to the history", format!("{error}. The reading is still on screen; check that the data folder in Settings can be written.")));
        }
    }

    pub fn meta(&self) -> Option<&ModelMeta> {
        match &self.model {
            ModelState::Ready(meta) => Some(meta),
            _ => None,
        }
    }

    pub fn busy(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        if self.batch.job.is_some() {
            return true;
        }
        self.running.is_some()
    }

    pub fn can_check(&self) -> bool {
        #[cfg(feature = "contribute")]
        if !matches!(self.contribute.terms, Some(Ok(_))) || !self.contribute.consent {
            return false;
        }
        !self.busy() && matches!(self.model, ModelState::Ready(_)) && !self.input.trim().is_empty()
    }

    /// Score the editor's text.
    pub fn check(&mut self) {
        if !self.can_check() {
            return;
        }
        let text = Arc::new(self.input.clone());
        #[cfg(feature = "contribute")]
        self.scorer.set_consent(self.contribute.consent);
        let job = self.scorer.score(text);
        self.running = Some(Running { job, progress: Progress { fraction: 0.0, words: "Starting".into() } });
        self.notice = None;
    }

    pub fn cancel(&mut self) {
        self.scorer.cancel();
    }

    pub fn set_settings(&mut self, settings: Settings) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(error) = crate::settings::disk::save(&settings) {
            self.notice = Some(Notice::error("Settings were not saved", format!("{error}. They apply until Litmus closes.")));
        }
        #[cfg(target_arch = "wasm32")]
        crate::settings::browser::save(&settings);
        self.settings = settings;
    }

    /// Put a reading on screen and record it in the history.
    pub fn show(&mut self, scored: Scored, title: String, source: String) {
        if let Err(error) = crate::heatmap::check_places(&scored.reading, &scored.text) {
            self.notice = Some(Notice::error("This reading does not fit its text", format!("{error}. It is not shown; check the text again, and report this if it repeats.")));
            return;
        }
        self.record(&scored, &title, &source);
        let model = scored.contribution.as_ref().map(|c| c.model_version.clone()).or_else(|| self.meta().map(|m| m.version.clone())).unwrap_or_default();
        self.open(Shown::new(title, source, scored, clock::now_ms(), model));
    }

    /// Show a reading without recording it (from the history or a batch).
    pub fn open(&mut self, shown: Shown) {
        self.selected_cell = None;
        self.editing = false;
        self.explain_chunk = 0;
        self.heat = Default::default();
        #[cfg(feature = "contribute")]
        {
            self.contribute.annotation = None;
            self.contribute.annotation_note = None;
        }
        self.shown = Some(shown);
        self.tab = Tab::Check;
    }

    pub fn open_history(&mut self, id: u64) {
        let Some(entry) = self.history.entries().iter().find(|e| e.id == id).cloned() else { return };
        match self.history.body(id) {
            Ok(body) => {
                self.input = body.text.clone();
                self.input_source = entry.source.clone();
                let scored = Scored { reading: Arc::new(body.reading), text: Arc::new(body.text), hash: entry.hash.clone(), seconds: entry.seconds, contribution: None };
                self.open(Shown::new(entry.title, entry.source, scored, entry.at_ms, entry.model));
            }
            Err(error) => self.notice = Some(Notice::error("That past reading could not be opened", format!("{error}. Its file may have been moved or damaged; delete the entry from the history."))),
        }
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Loaded(Ok(meta)) => {
                self.model = ModelState::Ready(meta);
                self.download = None;
                #[cfg(not(target_arch = "wasm32"))]
                for request in std::mem::take(&mut self.pending) {
                    self.apply(request);
                }
                #[cfg(target_arch = "wasm32")]
                if std::mem::take(&mut self.score_on_load) {
                    self.input = crate::sample::TEXT.to_owned();
                    self.check();
                }
            }
            Event::Loaded(Err(error)) => {
                self.model = ModelState::Failed(error);
                self.download = None;
            }
            Event::Downloading { done, total } => self.download = Some((done, total)),
            Event::Progress { job, progress } => {
                if let Some(running) = self.running.as_mut().filter(|r| r.job == job) {
                    running.progress = progress.clone();
                }
                #[cfg(not(target_arch = "wasm32"))]
                if self.batch.job.is_some_and(|(j, _)| j == job) {
                    self.batch.progress = progress;
                }
            }
            Event::Scored { job, result } => {
                #[cfg(not(target_arch = "wasm32"))]
                let Some(result) = crate::views::quick::route_scored(self, job, result) else { return };
                if self.running.as_ref().is_none_or(|r| r.job != job) {
                    return;
                }
                self.running = None;
                match result {
                    Ok(scored) => {
                        let title = if self.input_source == "pasted" { wording::title_of(&scored.text, 60) } else { file_name(&self.input_source) };
                        let source = match &scored.contribution {
                            Some(contribution) => format!("contribution {}", contribution.id),
                            None => self.input_source.clone(),
                        };
                        self.show(scored, title, source);
                    }
                    Err(error) if error == scoring::CANCELLED => self.notice = Some(Notice::Info("Cancelled; the text is unchanged.".into())),
                    #[cfg(feature = "contribute")]
                    Err(error) if error == crate::contribute::TERMS_CHANGED => self.contribute.terms_changed = true,
                    Err(error) if error == scoring::EMPTY => {
                        self.notice = Some(Notice::error("There is nothing to read", "The text has no words the model can read. Paste or type some sentences, then press Check."));
                    }
                    Err(error) => self.notice = Some(Notice::error("The text could not be checked", format!("{error}. Try again; if it repeats, restart Litmus."))),
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Event::BatchFile { job, index, result } => {
                let Some((_, first)) = self.batch.job.filter(|(j, _)| *j == job) else { return };
                if let Some(row) = self.batch.rows.get_mut(first + index) {
                    row.state = match result {
                        Ok(scored) => BatchState::Done(scored),
                        Err(error) => BatchState::Failed(error),
                    };
                    self.batch.finished += 1;
                    if let BatchState::Done(scored) = &row.state {
                        let (scored, path) = (scored.clone(), row.path.display().to_string());
                        self.record(&scored, &file_name(&path), &path);
                    }
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Event::BatchDone { job, cancelled } => {
                if self.batch.job.is_some_and(|(j, _)| j == job) {
                    self.batch.job = None;
                    self.batch.progress = Progress::default();
                    if cancelled {
                        self.notice = Some(Notice::Info("Batch cancelled; the files not yet read are still waiting.".into()));
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            Event::BatchFile { .. } | Event::BatchDone { .. } => {}
            #[cfg(feature = "contribute")]
            Event::Terms(terms) => {
                if let Ok(terms) = &terms {
                    self.scorer.set_terms_version(terms.version.clone());
                    self.model = ModelState::Ready(ModelMeta::server());
                }
                self.contribute.set_terms(terms);
            }
            #[cfg(feature = "contribute")]
            Event::Removed { text, result } => {
                let form = &mut self.contribute.removal;
                form.sending = false;
                if form.text == text {
                    form.answer = Some(result);
                }
            }
            #[cfg(feature = "contribute")]
            Event::Annotated { id, result } => {
                let current = self.shown.as_ref().and_then(|s| s.scored.contribution.as_ref()).is_some_and(|c| c.id == id);
                if current {
                    match result {
                        Ok(annotation) => {
                            self.contribute.annotation = annotation;
                            self.contribute.annotation_note = Some(match annotation {
                                Some(a) => format!("Thank you: your label \"{}\" is stored with the text.", a.button()),
                                None => "Your label is cleared.".to_owned(),
                            });
                        }
                        Err(error) => self.contribute.annotation_note = Some(error),
                    }
                }
            }
        }
    }

    /// Load a file's text into the editor and check it (the desktop).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_file(&mut self, path: &std::path::Path) {
        let read = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display())).and_then(|bytes| litmus_files::extract::text_of(path, &bytes));
        match read {
            Ok(text) => {
                self.input = text;
                self.input_source = path.display().to_string();
                self.tab = Tab::Check;
                self.editing = true;
                self.check();
            }
            Err(error) => {
                self.notice = Some(Notice::error(
                    format!("{} could not be read", file_name(&path.display().to_string())),
                    format!("{error}. Litmus reads text, Markdown, HTML, Word (.docx) and PDF files with a text layer; try another file, or paste the text."),
                ))
            }
        }
    }

    /// Do what the command line or another launch asked.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply(&mut self, request: crate::instance::Request) {
        use crate::instance::Request;
        if self.meta().is_none() && request != Request::Show {
            self.pending.push(request);
            return;
        }
        match request {
            Request::Show => {}
            Request::CheckFile(path) => {
                if self.busy() {
                    self.pending.push(Request::CheckFile(path));
                    return;
                }
                self.open_file(&path);
            }
            Request::CheckText(text) => {
                if self.busy() {
                    self.pending.push(Request::CheckText(text));
                    return;
                }
                self.input = text;
                self.input_source = "pasted".into();
                self.tab = Tab::Check;
                self.editing = true;
                self.check();
            }
        }
    }

    /// Take over as the one running app: requests from later launches arrive through `instance`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn attach(&mut self, ctx: &egui::Context, instance: Option<crate::instance::Instance>, request: crate::instance::Request) {
        if let Some(instance) = &instance {
            instance.watch(ctx);
        }
        self.instance = instance;
        self.window_saved = self.settings.window;
        if request != crate::instance::Request::Show {
            self.pending.push(request);
        }
    }

    /// Requests from later launches: bring the window forward and do what they ask.
    #[cfg(not(target_arch = "wasm32"))]
    fn requests(&mut self, ctx: &egui::Context) {
        let Some(instance) = &self.instance else { return };
        let taken = instance.take();
        if taken.is_empty() {
            return;
        }
        crate::views::quick::show_main(ctx);
        self.quick.hidden = false;
        for request in taken {
            match request {
                Ok(request) => self.apply(request),
                Err(error) => self.notice = Some(Notice::error("Another launch's request was not understood", error)),
            }
        }
    }

    /// Save the window's place once it has settled for a second after a move or a resize, and when it closes.
    #[cfg(not(target_arch = "wasm32"))]
    fn keep_window_place(&mut self, ctx: &egui::Context) {
        use crate::window_place::WindowPlace;
        if self.demo.is_some() {
            return;
        }
        let (place, now, closing) = ctx.input(|i| (WindowPlace::of(i.viewport()), i.time, i.viewport().close_requested()));
        if let Some(place) = place
            && self.window_saved.is_none_or(|saved| saved.differs(&place))
            && self.window_moved.is_none_or(|(moved, _)| moved.differs(&place))
        {
            self.window_moved = Some((place, now));
        }
        if let Some((place, since)) = self.window_moved {
            if closing || now - since > 1.0 {
                self.window_moved = None;
                self.window_saved = Some(place);
                let mut settings = self.settings.clone();
                settings.window = Some(place);
                self.set_settings(settings);
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(1100));
            }
        }
    }

    /// Show more or less of a reading, remembered.
    pub fn set_detail(&mut self, detail: crate::settings::Detail) {
        if self.settings.detail != detail {
            let mut settings = self.settings.clone();
            settings.detail = detail;
            self.set_settings(settings);
        }
    }

    /// Switch between the light and dark theme, remembered.
    pub fn toggle_theme(&mut self, ctx: &egui::Context) {
        let choice = if ctx.theme() == egui::Theme::Dark { crate::settings::ThemeChoice::Light } else { crate::settings::ThemeChoice::Dark };
        ctx.set_theme(choice.preference());
        let mut settings = self.settings.clone();
        settings.theme = choice;
        self.set_settings(settings);
    }

    /// Add files (folders walked) to the batch.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_to_batch(&mut self, paths: &[std::path::PathBuf]) {
        let mut added = 0;
        for path in paths {
            match litmus_files::walk::files_in(path) {
                Ok(files) => {
                    added += files.len();
                    self.batch.rows.extend(files.into_iter().map(|path| BatchRow { path, state: BatchState::Waiting }));
                }
                Err(error) => self.notice = Some(Notice::error("Those files could not be listed", format!("{error}. Check that the folder exists and can be read."))),
            }
        }
        if added == 0 && self.notice.is_none() {
            self.notice = Some(Notice::Info("No readable files there: Litmus reads .txt, .md, .html, .docx and .pdf.".into()));
        }
        self.tab = Tab::Batch;
    }

    /// Score every waiting file of the batch.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_batch(&mut self) {
        if self.busy() || self.meta().is_none() {
            return;
        }
        // Waiting rows move to the end in order, so the job's indices run contiguously from `first`.
        let (mut rows, waiting): (Vec<BatchRow>, Vec<BatchRow>) = self.batch.rows.drain(..).partition(|row| !matches!(row.state, BatchState::Waiting));
        let first = rows.len();
        let files: Vec<std::path::PathBuf> = waiting.iter().map(|row| row.path.clone()).collect();
        rows.extend(waiting);
        self.batch.rows = rows;
        if files.is_empty() {
            return;
        }
        let job = self.scorer.batch(files);
        self.batch.job = Some((job, first));
        self.batch.finished = 0;
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let check = KeyboardShortcut::new(Modifiers::COMMAND, Key::Enter);
        let (detail, settings, theme, edit) = ctx.input_mut(|i| {
            (
                i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::M)),
                i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Comma)),
                i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::L)),
                cfg!(not(target_arch = "wasm32")) && i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::E)),
            )
        });
        if detail {
            use crate::settings::Detail;
            let next = match self.settings.detail {
                Detail::Simple => Detail::Detailed,
                Detail::Detailed => Detail::Full,
                Detail::Full => Detail::Simple,
            };
            self.set_detail(next);
        }
        if settings {
            self.settings_open = !self.settings_open;
        }
        if theme {
            self.toggle_theme(ctx);
        }
        if edit && self.shown.is_some() && !self.busy() {
            self.editing = true;
            self.tab = Tab::Check;
        }
        let (do_check, do_cancel, tab) = ctx.input_mut(|i| {
            let tab = if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Num1)) {
                Some(Tab::Check)
            } else if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Num3)) {
                Some(Tab::History)
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Num2)) {
                    Some(Tab::Batch)
                } else {
                    None
                }
                #[cfg(target_arch = "wasm32")]
                None
            };
            (i.consume_shortcut(&check), i.key_pressed(Key::Escape), tab)
        });
        if do_check {
            self.check();
        }
        if do_cancel && self.busy() {
            self.cancel();
        } else if do_cancel && self.settings_open {
            self.settings_open = false;
        }
        if let Some(tab) = tab {
            self.tab = tab;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::O)))
            && !self.busy()
            && let Some(path) = views::pick_file()
        {
            self.open_file(&path);
        }
    }

    fn dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let paths: Vec<std::path::PathBuf> = dropped.iter().map(|file| file.path().to_owned()).collect();
            if self.busy() {
                self.notice = Some(Notice::Info("A reading is running: wait for it, or cancel it with Esc, then drop the files again.".into()));
            } else if paths.len() == 1 && paths[0].is_file() && self.tab == Tab::Check {
                self.open_file(&paths[0]);
            } else {
                self.add_to_batch(&paths);
            }
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(file) = dropped.into_iter().next() {
            crate::web::read_dropped(file, self.dropped.clone(), ctx.clone());
        }
    }

    /// Events, before any drawing; this also runs while the window is hidden in the tray.
    fn events(&mut self, ctx: &egui::Context) {
        for event in self.scorer.poll() {
            self.handle(event);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::views::quick::logic(self, ctx);
            self.requests(ctx);
            if !self.busy() && self.meta().is_some() && !self.pending.is_empty() {
                let next = self.pending.remove(0);
                self.apply(next);
            }
            self.keep_window_place(ctx);
        }
        // A phone-sized browser window gets touch-sized controls.
        let touch = cfg!(target_arch = "wasm32") && ctx.content_rect().width() < 600.0;
        if touch != self.touch {
            self.touch = touch;
            ctx.all_styles_mut(|style| theme::style_sizes(style, touch));
        }
    }

    fn frame(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        #[cfg(target_arch = "wasm32")]
        {
            let opened: Vec<(String, Result<String, String>)> = self.dropped.borrow_mut().drain(..).collect();
            for (name, result) in opened {
                match result {
                    Ok(text) => {
                        self.input = text;
                        self.input_source = name;
                        self.tab = Tab::Check;
                        self.editing = true;
                        self.check();
                    }
                    Err(error) => {
                        self.notice = Some(Notice::error(
                            format!("{name} could not be read"),
                            format!("{error}. The page reads text, Markdown, HTML, Word (.docx) and PDF files with a text layer; try another file, or paste the text."),
                        ))
                    }
                }
            }
        }
        self.shortcuts(&ctx);
        self.dropped_files(&ctx);
        views::top_bar(self, ui);
        views::notices(self, ui);
        views::footer(self, ui);
        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Check => views::check::show(self, ui),
            #[cfg(not(target_arch = "wasm32"))]
            Tab::Batch => views::batch::show(self, ui),
            Tab::History => views::history::show(self, ui),
        });
        views::settings::window(self, &ctx);
        #[cfg(feature = "contribute")]
        views::contribute::removal(self, &ctx);
        #[cfg(not(target_arch = "wasm32"))]
        views::quick::popup(self, &ctx);
        if self.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::demo::step(self, &ctx);
    }
}

/// The last part of a path, for titles.
pub fn file_name(path: &str) -> String {
    std::path::Path::new(path).file_name().map_or_else(|| path.to_owned(), |name| name.to_string_lossy().into_owned())
}

impl eframe::App for LitmusApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.events(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.frame(ui);
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }
}
