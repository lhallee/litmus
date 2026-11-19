//! Quick check in the app (desktop): its switch and hotkey in Settings, the tray's actions, the hotkey's capture,
//! the popup by the pointer, and staying in the background when the window is closed while it is on.

use std::sync::Arc;
use std::time::Instant;

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

fn open_popup(app: &mut LitmusApp, ctx: &egui::Context, state: PopupState) {
    let ppp = ctx.input(|i| i.viewport().native_pixels_per_point).unwrap_or(1.0);
    let at = quick::pointer().map_or(egui::pos2(200.0, 200.0), |(x, y)| egui::pos2(x as f32 / ppp + 14.0, y as f32 / ppp + 18.0));
    if let Some(running) = app.quick.running.as_mut() {
        running.popup = Some(Popup { state, at, opened: Instant::now(), focused_since: None, pinned: false });
    }
}

/// Each pass, before any drawing: hotkey and tray, capture results, closing to the tray.
pub fn logic(app: &mut LitmusApp, ctx: &egui::Context) {
    let Some(running) = app.quick.running.as_mut() else { return };
    let mut start_capture = None;
    if running.take_hotkey() {
        start_capture = Some(false);
    }
    for action in running.take_tray() {
        match action {
            TrayAction::CheckCopied => start_capture = Some(true),
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
    }
    let Some(running) = app.quick.running.as_mut() else { return };
    // The popup opens only once the selection is read: a new window takes the focus, and the selection must be read
    // (or copied) from the app the hotkey was pressed in, never from the popup.
    if let Some(copied_only) = start_capture {
        running.begin_capture(ctx, copied_only);
    }
    let Some(running) = app.quick.running.as_mut() else { return };
    if let Some(result) = running.poll_capture() {
        let ready = matches!(app.model, ModelState::Ready(_));
        let state = match result {
            Ok(_) if !ready => PopupState::Failed("The model is still loading; try again in a moment.".into()),
            Ok(Captured { text, how }) => PopupState::Scoring { job: app.scorer.score_quiet(Arc::new(text)), how },
            Err(error) => PopupState::Failed(error),
        };
        open_popup(app, ctx, state);
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
    // The popup is drawn from the main window's pass, which a hidden window skips unless asked.
    if app.quick.running.as_ref().is_some_and(|r| r.popup.is_some()) {
        ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::RequestPaintWhileHidden);
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
    }
}

/// A finished reading for the popup's job, if it is one; otherwise the result goes back to the caller.
pub fn route_scored(app: &mut LitmusApp, job: u64, result: Result<Scored, String>) -> Option<Result<Scored, String>> {
    let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut()) else { return Some(result) };
    let how = match popup.state {
        PopupState::Scoring { job: j, how } if j == job => how,
        _ => return Some(result),
    };
    popup.state = match result {
        Ok(scored) => PopupState::Shown { scored, how },
        Err(error) => PopupState::Failed(error),
    };
    popup.opened = Instant::now();
    if let PopupState::Shown { scored, .. } = &popup.state {
        let scored = scored.clone();
        let title = wording::title_of(&scored.text, 60);
        app.record(&scored, &title, "quick check");
    }
    None
}

/// What the popup's own window draws, copied from the app each frame (the window draws on its own pass, so it holds
/// no reference into the app), and what it reports back.
#[derive(Clone, Default)]
pub struct PopupView {
    content: Content,
    placeholder: bool,
    detail: Detail,
    /// Reported by the popup's window.
    escape: bool,
    os_close: bool,
    open_full: bool,
    focused: Option<bool>,
    hovered: bool,
}

#[derive(Clone, Default)]
enum Content {
    #[default]
    Reading,
    Scoring,
    Failed(String),
    Shown { headline: String, call: wording::Call, caution: bool, strip: litmus_engine::Strip, probability: String, counts: String, few: bool, how: How },
}

fn content(state: &PopupState) -> Content {
    match state {
        PopupState::Scoring { .. } => Content::Scoring,
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
    egui::CentralPanel::default().frame(frame).show(ui, |ui| match &view.content {
        Content::Reading | Content::Scoring => {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(if matches!(view.content, Content::Reading) { "Reading the selected text" } else { "Checking" }).size(16.0));
            });
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
    if view.open_full || view.escape {
        ui.ctx().request_repaint_of(ViewportId::ROOT);
    }
}

/// The popup by the pointer: verdict, strip, the text's words, characters and tokens, P(not human), and a link to the full reading. A deferred
/// viewport, kept alive by calling this every frame of the main window's pass.
pub fn popup(app: &mut LitmusApp, ctx: &egui::Context) {
    let Some(at) = app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).map(|p| p.at) else { return };
    let placeholder = app.meta().is_some_and(|m| m.placeholder);
    let model = app.meta().map(|m| m.version.clone()).unwrap_or_default();
    let shared = app.quick.view.clone();
    let reported = {
        let mut view = shared.lock().expect("the popup's view");
        let popup = app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).expect("checked above");
        view.content = content(&popup.state);
        view.placeholder = placeholder;
        view.detail = app.settings.detail;
        let reported = view.clone();
        (view.escape, view.os_close, view.open_full) = (false, false, false);
        reported
    };
    let builder = ViewportBuilder::default()
        .with_title(format!("{} quick check", crate::brand::NAME))
        .with_decorations(false)
        .with_always_on_top()
        .with_taskbar(false)
        .with_resizable(false)
        .with_position(at)
        .with_inner_size(popup_size(app.settings.detail, placeholder));
    let drawn = shared.clone();
    ctx.show_viewport_deferred(popup_id(), builder, move |ui, _class| draw(ui, &drawn));
    let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut()) else { return };
    let ready = matches!(popup.state, PopupState::Shown { .. } | PopupState::Failed(_));
    let mut close = reported.escape || reported.os_close || popup.lost_focus(reported.focused) || (ready && popup.expired(reported.hovered));
    if reported.open_full
        && let PopupState::Shown { scored, .. } = &popup.state
    {
        let scored = scored.clone();
        let title = wording::title_of(&scored.text, 60);
        app.input = scored.text.as_str().to_owned();
        app.input_source = "pasted".into();
        app.open(Shown::new(title, "quick check".into(), scored, crate::clock::now_ms(), model));
        app.quick.hidden = false;
        show_main(ctx);
        close = true;
    }
    if close && let Some(running) = app.quick.running.as_mut() {
        running.popup = None;
        *app.quick.view.lock().expect("the popup's view") = PopupView::default();
    }
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

/// For the scripted demo (`LITMUS_DEMO=quick`): quick check on for this run only (not saved), the main window
/// hidden to the tray, and `text` read as if it were selected.
pub fn demo_start(app: &mut LitmusApp, ctx: &egui::Context, text: &str) -> Result<(), String> {
    app.quick.running = Some(Quick::start(ctx, &app.settings.hotkey)?);
    ctx.send_viewport_cmd(ViewportCommand::Visible(false));
    app.quick.hidden = true;
    let job = app.scorer.score_quiet(Arc::new(text.to_owned()));
    open_popup(app, ctx, PopupState::Scoring { job, how: How::Accessibility });
    if let Some(popup) = app.quick.running.as_mut().and_then(|r| r.popup.as_mut()) {
        popup.pinned = true;
    }
    Ok(())
}

/// The demo's popup has its reading.
pub fn demo_shown(app: &LitmusApp) -> bool {
    app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).is_some_and(|p| matches!(p.state, PopupState::Shown { .. }))
}
