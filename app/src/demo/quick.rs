//! The quick scene's scripts (`LITMUS_DEMO_QUICK`): each drives one path through quick check with the app's real
//! windows and checks what the windows show, not only the app's state. A capture is fed in as the hotkey's would be;
//! nothing is typed, clicked, or copied on the desktop.
//!
//! - `reading` (the default): a short text's reading shows in the popup.
//! - `loading`: the text is captured while the model loads (a reload, as Settings starts one); the popup says so and
//!   shows the reading once the model is ready, with no second press.
//! - `batch`: a batch of `LITMUS_DEMO_FOLDER` runs in the main window; a quick check's reading must show while it is
//!   still running (quick check reads on its own thread).
//! - `replace`: a second capture replaces a popup whose long reading is still running; its reading must show within
//!   seconds, so the first was cancelled rather than finished.
//! - `escape`: Esc closes the popup, and its window leaves the screen while the main window stays hidden.
//! - `expire`: a popup closes by itself after its life (two seconds here; kept against clicks and keys meanwhile, so
//!   that a person working at the desk does not close it first).
//! - `full`: "Open full reading" closes the popup and shows the reading in the main window.
//! - `tray`: the window's close button keeps the app in the tray; the tray's Pause, Resume and Open; Quit ends it.
//!
//! Every script ends through the tray's Quit, so each also checks that Quit ends the app.

use std::time::{Duration, Instant};

use eframe::egui;

use crate::app::{LitmusApp, ModelState};
use crate::quick::{Keep, TrayAction};
use crate::views::quick::{self as view, DemoReport};

const SHORT: &str = "The bakery on the corner opens at six.";

/// The text the `reading` script captures: the file `LITMUS_DEMO_TEXT` names, so the popup can be photographed over a
/// reading of any kind, else the short sentence.
fn reading_text() -> Result<String, String> {
    match std::env::var_os("LITMUS_DEMO_TEXT") {
        Some(path) => std::fs::read_to_string(&path).map_err(|e| format!("LITMUS_DEMO_TEXT {}: {e}", std::path::Path::new(&path).display())),
        None => Ok(SHORT.to_owned()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Script {
    Reading,
    Loading,
    Batch,
    Replace,
    Escape,
    Expire,
    Full,
    Tray,
}

impl Script {
    pub fn from_env() -> Self {
        match std::env::var("LITMUS_DEMO_QUICK").as_deref() {
            Err(_) | Ok("reading") => Self::Reading,
            Ok("loading") => Self::Loading,
            Ok("batch") => Self::Batch,
            Ok("replace") => Self::Replace,
            Ok("escape") => Self::Escape,
            Ok("expire") => Self::Expire,
            Ok("full") => Self::Full,
            Ok("tray") => Self::Tray,
            Ok(other) => panic!("LITMUS_DEMO_QUICK={other}: use reading, loading, batch, replace, escape, expire, full or tray"),
        }
    }
}

/// What the demo does after a step of a script.
pub enum Next {
    /// Wait for the app to act by itself: the demo asks for no frames, so the app must wake itself as it does in use.
    Wait,
    /// Look again next frame (the app has no reason to wake, as when a window has just closed).
    Poll,
    /// Passed: photograph this window.
    Shoot(egui::ViewportId),
    Fail(String),
}

pub struct Run {
    script: Script,
    step: u8,
    /// When the current step began.
    since: Instant,
    /// The main window stays up (`LITMUS_DEMO_WINDOW=shown`).
    shown: bool,
    folder: Option<std::path::PathBuf>,
    /// When the model was ready with the main window kept up.
    window_since: Option<Instant>,
}

impl Run {
    pub fn new(shown: bool, folder: Option<std::path::PathBuf>) -> Self {
        Self { script: Script::from_env(), step: 0, since: Instant::now(), shown, folder, window_since: None }
    }

    fn next(&mut self) {
        self.step += 1;
        self.since = Instant::now();
    }

    fn elapsed_ms(&self) -> u128 {
        self.since.elapsed().as_millis()
    }
}

fn popup_window() -> bool {
    crate::quick::shows_window(&view::popup_title())
}

fn main_window() -> bool {
    crate::quick::shows_window(crate::brand::NAME)
}

/// Start quick check, hidden to the tray unless the scene keeps the window up, and capture `text`.
fn begin(run: &Run, app: &mut LitmusApp, ctx: &egui::Context, text: &str, keep: Keep) -> Result<(), String> {
    view::demo_start(app, ctx, !run.shown).map_err(|e| format!("quick check did not start: {e}"))?;
    view::demo_capture(app, ctx, text);
    view::demo_hold(app, keep, None);
    Ok(())
}

/// One step of the script, on each pass of the app's logic.
pub fn step(run: &mut Run, app: &mut LitmusApp, ctx: &egui::Context) -> Next {
    let ready = matches!(app.model, ModelState::Ready(_));
    if let ModelState::Failed(error) = &app.model
        && run.script != Script::Loading
    {
        return Next::Fail(format!("the model did not load: {error}"));
    }
    if run.step == 0 {
        if !ready {
            return Next::Wait;
        }
        // A main window kept up first settles in front: shown after its first frame, it takes the focus then, which
        // would take it from a popup opened before.
        if run.shown && run.window_since.get_or_insert_with(Instant::now).elapsed() < Duration::from_millis(700) {
            return Next::Poll;
        }
        let started = match run.script {
            Script::Reading => reading_text().and_then(|text| begin(run, app, ctx, &text, Keep::Always)),
            Script::Escape | Script::Full => begin(run, app, ctx, SHORT, Keep::Always),
            Script::Expire => begin(run, app, ctx, SHORT, Keep::AgainstInput).map(|()| view::demo_hold(app, Keep::AgainstInput, Some(Duration::from_secs(2)))),
            Script::Replace => begin(run, app, ctx, &super::scoring_text(), Keep::Always),
            Script::Loading => {
                // A reload, as Settings' "Load this model" starts one: the model is loading when the text arrives.
                app.model = ModelState::Loading;
                app.scorer.load(app.settings.model_folder.clone());
                begin(run, app, ctx, SHORT, Keep::Always).and_then(|()| {
                    if view::demo_waiting(app) { Ok(()) } else { Err("a text captured while the model loads did not wait for it".into()) }
                })
            }
            Script::Batch => match run.folder.clone() {
                Some(folder) => {
                    app.add_to_batch(&[folder]);
                    app.run_batch();
                    view::demo_start(app, ctx, !run.shown).map_err(|e| format!("quick check did not start: {e}"))
                }
                None => Err("the batch script needs LITMUS_DEMO_FOLDER".into()),
            },
            Script::Tray => view::demo_start(app, ctx, false).map_err(|e| format!("quick check did not start: {e}")),
        };
        if let Err(error) = started {
            return Next::Fail(error);
        }
        run.next();
        // The loading script looks at once, while the model still loads.
        return if run.script == Script::Loading { Next::Poll } else { Next::Wait };
    }
    let drawn = view::demo_drawn(app);
    match (run.script, run.step) {
        (Script::Reading, 1) => {
            if drawn == "shown" {
                eprintln!("litmus demo: the popup shows its reading {} ms after its capture", run.elapsed_ms());
                run.next();
            }
            Next::Wait
        }
        (Script::Reading, _) => {
            // A moment for the window manager to settle the focus, which the popup reports.
            if run.elapsed_ms() < 600 {
                return Next::Poll;
            }
            eprintln!("litmus demo: the popup's window has the focus: {:?}", view::demo_focused(app));
            // The window holds its reading whole: nothing past its right edge, and as tall as what it drew (1.1.0 cut
            // a long headline, the strip's end and the link off at the right).
            let (drawn_height, overflowed) = view::demo_fit(app);
            let window = ctx.input_for(view::popup_id(), |i| i.viewport().inner_rect);
            eprintln!("litmus demo: the popup drew {drawn_height:?} points of height in a window of {:?}", window.map(|r| r.size()));
            if overflowed {
                return Next::Fail("the popup drew its reading past its right edge".into());
            }
            if let (Some(drawn_height), Some(window)) = (drawn_height, window)
                && (window.height() - drawn_height).abs() > 2.0
            {
                return if run.elapsed_ms() > 3000 { Next::Fail(format!("the popup's window is {} points tall and its reading {drawn_height}", window.height())) } else { Next::Poll };
            }
            Next::Shoot(view::popup_id())
        }
        (Script::Loading, 1) => {
            // The model was still loading when the text arrived; the popup says so unless the load was done first.
            match drawn.as_str() {
                "loading" => eprintln!("litmus demo: the popup drew \"loading\" while the model loaded"),
                other if matches!(app.model, ModelState::Loading) => return Next::Fail(format!("the popup drew \"{other}\" while the model loaded")),
                other => eprintln!("litmus demo: the model loaded within a frame; the popup drew \"{other}\""),
            }
            run.next();
            Next::Wait
        }
        (Script::Loading, _) => match drawn.as_str() {
            "shown" => Next::Shoot(view::popup_id()),
            "loading" => Next::Wait,
            scoring if scoring.starts_with("scoring") => Next::Wait,
            other => Next::Fail(format!("after the model loaded the popup drew \"{other}\"")),
        },
        (Script::Batch, 1) => {
            // The batch is under way: now the hotkey's capture.
            if app.batch.progress.words.is_empty() {
                return Next::Wait;
            }
            view::demo_capture(app, ctx, SHORT);
            view::demo_hold(app, Keep::Always, None);
            run.next();
            Next::Wait
        }
        (Script::Batch, _) => {
            if drawn != "shown" {
                return Next::Wait;
            }
            if app.batch.job.is_none() {
                return Next::Fail("the batch finished before the quick reading showed: quick check waited behind it".into());
            }
            eprintln!("litmus demo: quick reading shown {} ms after its capture, the batch at \"{}\"", run.elapsed_ms(), app.batch.progress.words);
            Next::Shoot(view::popup_id())
        }
        (Script::Replace, 1) => {
            // A long selection: the popup says how far its reading has come, then a second capture replaces it.
            if !drawn.contains(" tokens") || run.elapsed_ms() < 500 {
                return if run.elapsed_ms() > 10_000 { Next::Fail(format!("10 seconds into a long reading the popup drew \"{drawn}\", no progress")) } else { Next::Poll };
            }
            eprintln!("litmus demo: the long reading's popup drew \"{drawn}\"");
            view::demo_capture(app, ctx, SHORT);
            view::demo_hold(app, Keep::Always, None);
            run.next();
            Next::Wait
        }
        (Script::Replace, _) => {
            if drawn != "shown" {
                if run.elapsed_ms() > 10_000 {
                    return Next::Fail(format!("the second reading did not show within 10 seconds (the popup drew \"{drawn}\"): the long reading was not cancelled"));
                }
                return Next::Wait;
            }
            eprintln!("litmus demo: the replacing reading showed {} ms after its capture", run.elapsed_ms());
            Next::Shoot(view::popup_id())
        }
        (Script::Escape | Script::Full, 1) => {
            if drawn != "shown" {
                return Next::Wait;
            }
            if !popup_window() {
                return Next::Fail("the popup drew its reading but no popup window is on the screen".into());
            }
            view::demo_report(app, if run.script == Script::Escape { DemoReport::Escape } else { DemoReport::OpenFull });
            run.next();
            Next::Wait
        }
        (Script::Escape, _) => {
            if view::demo_open(app) || popup_window() {
                return if run.elapsed_ms() > 3000 { Next::Fail(format!("3 seconds after Esc the popup is still open (state {}, window {})", view::demo_open(app), popup_window())) } else { Next::Poll };
            }
            if !run.shown && main_window() {
                return Next::Fail("closing the popup showed the main window, which was hidden in the tray".into());
            }
            Next::Shoot(egui::ViewportId::ROOT)
        }
        (Script::Full, _) => {
            let opened = app.shown.as_ref().is_some_and(|s| s.source == "quick check");
            if view::demo_open(app) || popup_window() || !opened || !main_window() {
                return if run.elapsed_ms() > 3000 {
                    Next::Fail(format!("3 seconds after Open full reading: popup open {}, its window {}, reading in the main window {opened}, main window shown {}", view::demo_open(app), popup_window(), main_window()))
                } else {
                    Next::Poll
                };
            }
            Next::Shoot(egui::ViewportId::ROOT)
        }
        (Script::Expire, 1) => {
            if drawn == "shown" {
                run.next();
            }
            Next::Wait
        }
        (Script::Expire, _) => {
            if view::demo_open(app) || popup_window() {
                return if run.elapsed_ms() > 6000 { Next::Fail("the popup did not close 6 seconds after its reading showed, with a life of 2".into()) } else { Next::Poll };
            }
            let after = run.elapsed_ms();
            if after < 1900 {
                return Next::Fail(format!("the popup closed {after} ms after its reading showed, before its life of 2 seconds"));
            }
            eprintln!("litmus demo: the popup closed by itself {after} ms after its reading showed");
            Next::Shoot(egui::ViewportId::ROOT)
        }
        (Script::Tray, 1) => {
            // The window's close button, with quick check on, keeps the app in the tray.
            if !main_window() || run.elapsed_ms() < 500 {
                return Next::Poll;
            }
            ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Close);
            run.next();
            Next::Poll
        }
        (Script::Tray, 2) => {
            if main_window() || !app.quick.hidden {
                return if run.elapsed_ms() > 3000 { Next::Fail("3 seconds after the window was closed with quick check on it is still shown".into()) } else { Next::Poll };
            }
            view::demo_tray(app, ctx, TrayAction::Pause);
            if !view::demo_paused(app) {
                return Next::Fail("Pause did not pause quick check".into());
            }
            view::demo_tray(app, ctx, TrayAction::Pause);
            if view::demo_paused(app) {
                return Next::Fail("the second Pause (Resume) did not resume quick check".into());
            }
            if app.quick.error.is_some() {
                return Next::Fail(format!("Pause and Resume: {}", app.quick.error.clone().unwrap_or_default()));
            }
            view::demo_tray(app, ctx, TrayAction::Open);
            run.next();
            Next::Poll
        }
        (Script::Tray, _) => {
            if !main_window() || app.quick.hidden {
                return if run.elapsed_ms() > 3000 { Next::Fail("3 seconds after the tray's Open the main window is not shown".into()) } else { Next::Poll };
            }
            Next::Shoot(egui::ViewportId::ROOT)
        }
    }
}
