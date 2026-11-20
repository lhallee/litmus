//! Quick check in the app (desktop): its switch and hotkey in Settings, the tray's actions, the hotkey's capture,
//! the popup by the pointer, and staying in the background when the window is closed while it is on.

use std::sync::Arc;

use eframe::egui::{self, RichText, ViewportBuilder, ViewportCommand, ViewportId};
use litmus_engine::STRIP_LEVEL;
use litmus_files::counts::TextCounts;

use crate::app::{LitmusApp, ModelState, Shown};
use crate::quick::{self, Captured, How, Popup, PopupState, Quick, TrayAction};
use crate::scoring::Scored;
use crate::settings::Detail;
use crate::{strip, theme, views, wording};

#[derive(Default)]
pub struct State {
    pub running: Option<Quick>,
    pub error: Option<String>,
    pub hotkey_edit: String,
    /// Quit from the tray: the next close is not turned into hiding.
    pub quitting: bool,
    /// The main window is hidden; the app lives in the tray.
    pub hidden: bool,
    /// What the popup's window draws and reports.
    pub view: Arc<std::sync::Mutex<PopupView>>,
}

/// Start quick check if the user's switch in Settings is on (never otherwise).
pub fn start_if_on(app: &mut LitmusApp, ctx: &egui::Context) {
    app.quick.hotkey_edit = app.settings.hotkey.clone();
    if app.settings.quick_check {
        switch(app, ctx, true);
    }
}

/// Turn quick check on or off, saving the choice.
pub fn switch(app: &mut LitmusApp, ctx: &egui::Context, on: bool) {
    stop_popup_job(app);
    app.quick.running = None; // dropping unregisters the hotkey and removes the tray icon
    app.quick.error = None;
    if on {
        match Quick::start(ctx, &app.settings.hotkey) {
            Ok(running) => app.quick.running = Some(running),
            Err(error) => app.quick.error = Some(error),
        }
    }
    if app.settings.quick_check != on {
        let mut settings = app.settings.clone();
        settings.quick_check = on;
        app.set_settings(settings);
    }
}

pub fn show_main(ctx: &egui::Context) {
    ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::Visible(true));
    ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::Minimized(false));
    ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::Focus);
}

/// Stop the popup's reading if it is still running (the popup is closing or being replaced), so that the quick thread
/// is free for the next one.
fn stop_popup_job(app: &mut LitmusApp) {
    if let Some(job) = app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).and_then(Popup::scoring_job) {
        app.scorer.cancel_quiet(job);
    }
}

fn open_popup(app: &mut LitmusApp, ctx: &egui::Context, state: PopupState) {
    // Points of the main window, its zoom included: egui places a new window at its position times the zoom.
    let ppp = ctx.pixels_per_point();
    let [width, height] = popup_size(app.settings.detail, app.meta().is_some_and(|m| m.placeholder));
    let (at, screen) = match quick::pointer() {
        Some(pointer) => {
            let pixels = |points: f32| (points * ppp).ceil() as i32;
            let size = (pixels(width), pixels(height));
            let (x, y) = quick::popup_origin(pointer, size, (pixels(14.0), pixels(18.0)), quick::work_area_at(pointer));
            (egui::pos2(x as f32 / ppp, y as f32 / ppp), Some([x, y, x + size.0, y + size.1]))
        }
        None => (egui::pos2(200.0, 200.0), None),
    };
    if let Some(running) = app.quick.running.as_mut() {
        let mut popup = Popup::new(state, at);
        popup.screen = screen;
        running.popup = Some(popup);
    }
}

/// A capture's result: a popup with its reading started (or waiting for the model), replacing any popup before it.
fn captured(app: &mut LitmusApp, ctx: &egui::Context, result: Result<Captured, String>) {
    stop_popup_job(app);
    let state = match (result, &app.model) {
        (Err(error), _) => PopupState::Failed(error),
        (Ok(Captured { text, how }), ModelState::Ready(_)) => PopupState::Scoring { job: app.scorer.score_quiet(Arc::new(text)), how, progress: String::new() },
        (Ok(Captured { text, how }), ModelState::Loading) => PopupState::Waiting { text, how },
        (Ok(_), ModelState::Failed(error)) => PopupState::Failed(format!("No model is loaded: {error}")),
    };
    open_popup(app, ctx, state);
}

/// What a tray item does; `Some(true)` asks for a capture of the clipboard.
fn tray_action(app: &mut LitmusApp, ctx: &egui::Context, action: TrayAction) -> Option<bool> {
    let running = app.quick.running.as_mut()?;
    match action {
        TrayAction::CheckCopied => return Some(true),
        TrayAction::Open => {
            app.quick.hidden = false;
            show_main(ctx);
        }
        TrayAction::Pause => {
            let paused = !running.paused;
            if let Err(error) = running.set_paused(paused) {
                app.quick.error = Some(error);
            }
        }
        TrayAction::Quit => {
            app.quick.quitting = true;
            ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::Close);
        }
    }
    None
}

/// Each pass, before any drawing: hotkey and tray, capture results, closing to the tray.
pub fn logic(app: &mut LitmusApp, ctx: &egui::Context) {
    let Some(running) = app.quick.running.as_mut() else { return };
    let mut start_capture = None;
    // A press while the popup holds the focus keeps the popup, since the selection would be read from the popup itself.
    let popup_focused = running.popup.is_some() && app.quick.view.lock().expect("the popup's view").focused == Some(true);
    if running.take_hotkey() && !popup_focused {
        start_capture = Some(false);
    }
    for action in running.take_tray() {
        if let Some(copied_only) = tray_action(app, ctx, action) {
            start_capture = Some(copied_only);
        }
    }
    // The popup opens only once the selection is read: a new window takes the focus, and the selection must be read
    // (or copied) from the app the hotkey was pressed in, never from the popup. A press while a capture is under way
    // waits for it (`Quick::capturing`).
    if let Some(copied_only) = start_capture
        && let Some(running) = app.quick.running.as_mut()
        && !running.capturing()
    {
        running.begin_capture(ctx, copied_only);
    }
    let Some(running) = app.quick.running.as_mut() else { return };
    if let Some(result) = running.poll_capture() {
        captured(app, ctx, result);
    }
    // Closing the window while quick check is on keeps the app in the tray. Linux has no tray to reopen or quit a
    // hidden app from, so there closing the window quits, and the hotkey works while the window is open.
    if crate::demo::QUITTING.load(std::sync::atomic::Ordering::Relaxed) || cfg!(target_os = "linux") {
        app.quick.quitting = true;
    }
    if ctx.input(|i| i.viewport().close_requested()) && !app.quick.quitting {
        ctx.send_viewport_cmd(ViewportCommand::CancelClose);
        ctx.send_viewport_cmd(ViewportCommand::Visible(false));
        app.quick.hidden = true;
    }
    tend_popup(app, ctx);
    // The popup's window is kept alive by the main window's ui pass, which a hidden window skips unless asked. Without
    // the focus the popup watches Esc and the mouse, often enough to see a click (`Popup::dismissed_unfocused`).
    if app.quick.running.as_ref().is_some_and(|r| r.popup.is_some()) {
        let focused = app.quick.view.lock().expect("the popup's view").focused == Some(true);
        ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::RequestPaintWhileHidden);
        ctx.request_repaint_after(std::time::Duration::from_millis(if focused { 250 } else { 30 }));
    }
}

/// The popup's bookkeeping, on every pass of the app's logic: start a waiting text's reading once the model is ready,
/// copy what the popup shows into the view its window draws, and act on what that window reported. All of it runs
/// here, never in the main window's ui pass: on Windows a topmost popup that repaints keeps paint messages from the
/// main window, so that pass can stall while the popup is open, and a view filled there stayed on "Checking" after
/// the reading had finished (1.0.0, Logan's first quick check, 2026-11-19).
fn tend_popup(app: &mut LitmusApp, ctx: &egui::Context) {
    if let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut())
        && let PopupState::Waiting { text, how } = &popup.state
    {
        let next = match &app.model {
            ModelState::Loading => None,
            ModelState::Ready(_) => Some(PopupState::Scoring { job: app.scorer.score_quiet(Arc::new(text.clone())), how: *how, progress: String::new() }),
            ModelState::Failed(error) => Some(PopupState::Failed(format!("No model is loaded: {error}"))),
        };
        if let Some(next) = next {
            popup.set(next);
        }
    }
    let placeholder = app.meta().is_some_and(|m| m.placeholder);
    let model = app.meta().map(|m| m.version.clone()).unwrap_or_default();
    let detail = app.settings.detail;
    let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut()) else { return };
    let reported = {
        let mut view = app.quick.view.lock().expect("the popup's view");
        if view.revision != Some(popup.revision) || view.detail != detail || view.placeholder != placeholder {
            view.content = content(&popup.state);
            (view.revision, view.detail, view.placeholder) = (Some(popup.revision), detail, placeholder);
            ctx.request_repaint_of(popup_id());
        }
        let reported = view.clone();
        (view.escape, view.os_close, view.open_full) = (false, false, false);
        reported
    };
    let ready = matches!(popup.state, PopupState::Shown { .. } | PopupState::Failed(_));
    let watched = reported.focused != Some(true) && {
        let (escape, button) = quick::dismiss_keys();
        popup.dismissed_unfocused(escape, button, quick::pointer())
    };
    let mut close = reported.escape || reported.os_close || watched || popup.lost_focus(reported.focused) || (ready && popup.expired(reported.hovered));
    let full = match &popup.state {
        PopupState::Shown { scored, .. } if reported.open_full => Some(scored.clone()),
        _ => None,
    };
    if let Some(scored) = full {
        let title = wording::title_of(&scored.text, 60);
        app.input = scored.text.as_str().to_owned();
        app.input_source = "pasted".into();
        app.open(Shown::new(title, "quick check".into(), scored, crate::clock::now_ms(), model));
        app.quick.hidden = false;
        show_main(ctx);
        close = true;
    }
    if close {
        stop_popup_job(app);
    }
    if close && let Some(running) = app.quick.running.as_mut() {
        running.popup = None;
        *app.quick.view.lock().expect("the popup's view") = PopupView::default();
        // The window goes in the main window's next ui pass, which no longer shows it.
        ctx.request_repaint_of(ViewportId::ROOT);
    }
}

/// Progress of the popup's reading, for its window (a long selection takes seconds).
pub fn route_progress(app: &mut LitmusApp, job: u64, progress: &crate::scoring::Progress) {
    if let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut())
        && let PopupState::Scoring { job: j, progress: words, .. } = &mut popup.state
        && *j == job
    {
        words.clone_from(&progress.words);
        popup.touch();
    }
}

/// A finished reading for the popup's job, if it is one; otherwise the result goes back to the caller.
pub fn route_scored(app: &mut LitmusApp, job: u64, result: Result<Scored, String>) -> Option<Result<Scored, String>> {
    let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut()) else { return Some(result) };
    let how = match popup.state {
        PopupState::Scoring { job: j, how, .. } if j == job => how,
        _ => return Some(result),
    };
    popup.set(match result {
        Ok(scored) => PopupState::Shown { scored, how },
        Err(error) => PopupState::Failed(error),
    });
    if let PopupState::Shown { scored, .. } = &popup.state {
        let scored = scored.clone();
        let title = wording::title_of(&scored.text, 60);
        app.record(&scored, &title, "quick check");
    }
    None
}

/// What the popup's own window draws, copied from the app by `tend_popup` whenever it changes (the window draws on its
/// own pass, so it holds no reference into the app), and what it reports back.
#[derive(Clone, Default)]
pub struct PopupView {
    content: Content,
    placeholder: bool,
    detail: Detail,
    /// The popup's `revision` that `content` shows.
    revision: Option<u64>,
    /// Reported by the popup's window.
    escape: bool,
    os_close: bool,
    open_full: bool,
    focused: Option<bool>,
    hovered: bool,
}

#[derive(Clone, Default)]
enum Content {
    /// Closed: nothing to draw until the main window's ui pass removes the window.
    #[default]
    Closed,
    Loading,
    Scoring(String),
    Failed(String),
    Shown { headline: String, call: wording::Call, caution: bool, strip: litmus_engine::Strip, probability: String, counts: String, few: bool, how: How },
}

fn content(state: &PopupState) -> Content {
    match state {
        PopupState::Waiting { .. } => Content::Loading,
        PopupState::Scoring { progress, .. } => Content::Scoring(progress.clone()),
        PopupState::Failed(error) => Content::Failed(error.clone()),
        PopupState::Shown { scored, how } => {
            let reading = &scored.reading;
            let v = &reading.verdict;
            Content::Shown {
                headline: wording::verdict_headline(v),
                call: wording::call(v.set(STRIP_LEVEL)),
                caution: v.out_of_reference && !v.set(STRIP_LEVEL).is_none_or(|s| s.is_empty()),
                strip: v.strip,
                probability: wording::probability(v.probability, v.interval),
                counts: TextCounts::of(&scored.text, reading.n_tokens).line(),
                few: reading.n_tokens < wording::FEW_TOKENS,
                how: *how,
            }
        }
    }
}

/// The popup's size in points at a detail level (the window is sized to its content, never scrolled).
fn popup_size(detail: Detail, placeholder: bool) -> [f32; 2] {
    let height = match detail {
        Detail::Simple => 176.0,
        Detail::Detailed => 252.0,
        Detail::Full => 276.0,
    } + if placeholder { 28.0 } else { 0.0 };
    [380.0, height]
}

/// The popup window's drawing, on its own pass.
fn draw(ui: &mut egui::Ui, shared: &std::sync::Mutex<PopupView>) {
    let (escape, focused, hovered, os_close) = ui.input(|i| (i.key_pressed(egui::Key::Escape), i.viewport().focused, i.pointer.has_pointer(), i.viewport().close_requested()));
    let mut view = shared.lock().expect("the popup's view");
    view.escape |= escape;
    view.os_close |= os_close;
    view.focused = focused;
    view.hovered = hovered;
    let mut open_full = false;
    let p = theme::palette(ui.visuals().dark_mode);
    let frame = egui::Frame::new().fill(p.raised).stroke(egui::Stroke::new(1.0, p.rule)).inner_margin(egui::Margin::symmetric(16, 12));
    // No spinner: one repaints the popup every frame, which on Windows starves the main window of paint messages.
    egui::CentralPanel::default().frame(frame).show(ui, |ui| match &view.content {
        Content::Closed => {}
        Content::Loading => {
            ui.add_space(8.0);
            ui.label(RichText::new("Loading the model").size(16.0));
        }
        Content::Scoring(progress) => {
            ui.add_space(8.0);
            ui.label(RichText::new("Checking").size(16.0));
            ui.label(RichText::new(progress).small().color(p.muted));
        }
        Content::Failed(error) => {
            ui.label(RichText::new("No reading").font(theme::strong(18.0)));
            ui.label(RichText::new(error).color(p.muted));
        }
        Content::Shown { headline, call, caution, strip: reading_strip, probability, counts, few, how } => {
            let color = match call {
                wording::Call::One(origin) => p.origin_ink(*origin),
                _ => p.ink,
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new(headline).font(theme::strong(22.0)).color(color));
                if *caution {
                    ui.label(RichText::new("? unlike reference").color(p.muted));
                }
            });
            strip::show(ui, reading_strip, 14.0);
            // At every level: the size of the selected text, which the hotkey's user asked to see at a glance.
            ui.label(RichText::new(counts).small().color(p.muted));
            if view.detail.at_least(Detail::Detailed) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("P(not human)").color(p.muted));
                    ui.label(RichText::new(probability).monospace());
                });
                if *few {
                    ui.label(RichText::new(wording::FEW_WORDS).small().color(p.muted));
                }
            }
            if view.detail.at_least(Detail::Full) {
                let source = match how {
                    How::Accessibility => "Read from the app's selection.",
                    How::Copy => "Copied from the selection; your clipboard was put back.",
                    How::Clipboard => "Read from the clipboard.",
                };
                ui.label(RichText::new(source).small().color(p.muted));
            }
            if view.placeholder {
                ui.label(RichText::new("Placeholder model: readings are meaningless.").small().color(p.caution_ink).background_color(p.caution));
            }
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Max), |ui| {
                open_full = views::controls::link(ui, "Open full reading").clicked();
            });
        }
    });
    view.open_full |= open_full;
    if view.open_full || view.escape || view.os_close {
        ui.ctx().request_repaint_of(ViewportId::ROOT);
    }
}

/// The popup by the pointer: verdict, strip, the text's words, characters and tokens, P(not human), and a link to the
/// full reading. A deferred viewport, kept alive by calling this every frame of the main window's ui pass; what it
/// shows and what it reports are tended in the app's logic (`tend_popup`).
pub fn popup(app: &mut LitmusApp, ctx: &egui::Context) {
    let Some(at) = app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).map(|p| p.at) else { return };
    let placeholder = app.meta().is_some_and(|m| m.placeholder);
    let builder = ViewportBuilder::default()
        .with_title(popup_title())
        .with_decorations(false)
        .with_always_on_top()
        .with_taskbar(false)
        .with_resizable(false)
        .with_position(at)
        .with_inner_size(popup_size(app.settings.detail, placeholder));
    let drawn = app.quick.view.clone();
    ctx.show_viewport_deferred(popup_id(), builder, move |ui, _class| draw(ui, &drawn));
}

/// The Settings section: the switch with its sentence, the hotkey, and the Explorer entry.
pub fn settings_section(app: &mut LitmusApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    views::controls::section(ui, "Quick check");
    ui.label(RichText::new(quick::SENTENCE).weak());
    let mut on = app.settings.quick_check;
    if ui.checkbox(&mut on, "Check selected text with a hotkey, from anywhere").changed() {
        switch(app, ctx, on);
    }
    ui.horizontal(|ui| {
        ui.label("Hotkey");
        ui.add(egui::TextEdit::singleline(&mut app.quick.hotkey_edit).desired_width(140.0).hint_text(quick::DEFAULT_HOTKEY));
        if ui.button("Use this hotkey").clicked() {
            match quick::parse_hotkey(&app.quick.hotkey_edit) {
                Ok(_) => {
                    let mut settings = app.settings.clone();
                    settings.hotkey = app.quick.hotkey_edit.trim().to_owned();
                    app.set_settings(settings);
                    if app.settings.quick_check {
                        switch(app, ctx, true);
                    }
                }
                Err(error) => app.quick.error = Some(error),
            }
        }
    });
    if let Some(running) = &app.quick.running {
        ui.label(RichText::new(if running.paused { "Paused (resume from the tray)." } else { "On: select text anywhere and press the hotkey." }).small());
    }
    if let Some(error) = &app.quick.error {
        views::caution(ui, "Quick check:", error);
    }
    #[cfg(windows)]
    {
        let present = crate::quick::windows::explorer_entry_present(crate::quick::windows::CLASSES);
        let mut entry = present;
        if ui
            .checkbox(&mut entry, "Add \"Check with Litmus\" to Explorer's right-click menu for text, Markdown, HTML, Word and PDF files")
            .on_hover_text("Per-user registry keys under HKEY_CURRENT_USER\\Software\\Classes\\SystemFileAssociations; unticking removes them.")
            .changed()
            && let Err(error) = quick::set_explorer_entry(entry)
        {
            app.quick.error = Some(error);
        }
    }
    ui.label(RichText::new("Other apps' own right-click menus cannot be extended on Windows; the hotkey works in all of them.").small().weak());
}

/// The popup's viewport.
pub fn popup_id() -> ViewportId {
    ViewportId::from_hash_of("litmus_quick_check")
}

/// The popup window's title (no title bar shows it; screen readers and the window list do).
pub fn popup_title() -> String {
    format!("{} quick check", crate::brand::NAME)
}

/// The scripted demo's hotkey: registered while a quick scene runs, never pressed, and unlike any a person uses, so a
/// scene neither takes the user's own Litmus hotkey nor fails because a running Litmus holds it.
const DEMO_HOTKEY: &str = "Ctrl+Alt+Shift+F10";

/// For the scripted demo (`LITMUS_DEMO=quick`): quick check on for this run only (not saved), the main window hidden
/// to the tray if `hide`.
pub fn demo_start(app: &mut LitmusApp, ctx: &egui::Context, hide: bool) -> Result<(), String> {
    app.quick.running = Some(Quick::start(ctx, DEMO_HOTKEY)?);
    if hide {
        ctx.send_viewport_cmd(ViewportCommand::Visible(false));
        app.quick.hidden = true;
    }
    Ok(())
}

/// The demo's capture: `text` read as if it were selected, through the path a hotkey's capture takes.
pub fn demo_capture(app: &mut LitmusApp, ctx: &egui::Context, text: &str) {
    captured(app, ctx, Ok(Captured { text: text.to_owned(), how: How::Accessibility }));
}

/// What keeps the demo's popup up, and a shorter life for it.
pub fn demo_hold(app: &mut LitmusApp, keep: quick::Keep, lifetime: Option<std::time::Duration>) {
    if let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut()) {
        popup.keep = keep;
        if let Some(lifetime) = lifetime {
            popup.lifetime = lifetime;
        }
    }
}

/// What the popup's window draws (its view, not only the app's state: in 1.0.0 the state had the reading while the
/// window still drew "Checking"): `closed`, `loading`, `scoring: <progress>`, `failed: <why>`, or `shown`.
pub fn demo_drawn(app: &LitmusApp) -> String {
    match &app.quick.view.lock().expect("the popup's view").content {
        Content::Closed => "closed".into(),
        Content::Loading => "loading".into(),
        Content::Scoring(progress) => format!("scoring: {progress}"),
        Content::Failed(why) => format!("failed: {why}"),
        Content::Shown { .. } => "shown".into(),
    }
}

/// The popup waits for the model with its captured text.
pub fn demo_waiting(app: &LitmusApp) -> bool {
    app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).is_some_and(|p| matches!(p.state, PopupState::Waiting { .. }))
}

/// Whether the popup's window holds the keyboard focus, as it last reported.
pub fn demo_focused(app: &LitmusApp) -> Option<bool> {
    app.quick.view.lock().expect("the popup's view").focused
}

/// The popup is open, in the app's state.
pub fn demo_open(app: &LitmusApp) -> bool {
    app.quick.running.as_ref().is_some_and(|r| r.popup.is_some())
}

/// The popup's reading while it runs.
pub fn demo_job(app: &LitmusApp) -> Option<u64> {
    app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).and_then(Popup::scoring_job)
}

/// What the popup's window reports, as if a person pressed Esc on it or clicked "Open full reading".
#[derive(Clone, Copy, Debug)]
pub enum DemoReport {
    Escape,
    OpenFull,
}

pub fn demo_report(app: &mut LitmusApp, report: DemoReport) {
    let mut view = app.quick.view.lock().expect("the popup's view");
    match report {
        DemoReport::Escape => view.escape = true,
        DemoReport::OpenFull => view.open_full = true,
    }
}

/// A tray item, as if clicked.
pub fn demo_tray(app: &mut LitmusApp, ctx: &egui::Context, action: TrayAction) {
    assert_ne!(action, TrayAction::CheckCopied, "the demo never reads the clipboard");
    tray_action(app, ctx, action);
}

/// Quick check is paused, as the tray's Pause item leaves it.
pub fn demo_paused(app: &LitmusApp) -> bool {
    app.quick.running.as_ref().is_some_and(|r| r.paused)
}
