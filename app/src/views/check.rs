//! The Check screen. Before a check: one inviting sentence, the text box, and one primary Check button. After a
//! check the text box becomes the reading: the verdict over the user's own text, tinted cell by cell, with an Edit
//! action back to plain editing. At the Detailed and Full levels the numbers sit in a panel beside it (under it on
//! a narrow window).

use eframe::egui::{self, CornerRadius, Margin, RichText, Stroke};

use crate::app::{LitmusApp, ModelState};
use crate::settings::Detail;
use crate::theme;
use crate::views::{self, controls, dark};

/// Below this width the details stack under the reading.
const NARROW: f32 = 900.0;

pub fn show(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let reading = app.shown.is_some() && !app.editing;
    let wide = ui.available_width() >= NARROW;
    let side = reading && wide && app.settings.detail != Detail::Simple;
    let p = theme::palette(dark(ui));
    if side {
        let width = if app.settings.detail == Detail::Full { 460.0 } else { 360.0 };
        let max = (ui.available_width() * 0.55).max(width);
        egui::Panel::right("details")
            .resizable(true)
            .default_size(width)
            .min_size(300.0)
            .max_size(max)
            .frame(egui::Frame::new().fill(p.raised).inner_margin(Margin::symmetric(20, 20)))
            .show(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("details_scroll").auto_shrink([false, false]).show(ui, |ui| views::reading::details(app, ui));
            });
    }
    egui::CentralPanel::default().frame(egui::Frame::new().fill(p.ground).inner_margin(Margin::symmetric(0, if cfg!(feature = "contribute") && ui.available_width() < 600.0 { 8 } else { 20 }))).show(ui, |ui| {
        if reading {
            reading_screen(app, ui, side);
        } else {
            editor_screen(app, ui);
        }
    });
}

/// The verdict over the tinted text; on a narrow window everything scrolls together.
fn reading_screen(app: &mut LitmusApp, ui: &mut egui::Ui, side: bool) {
    let stacked = !side && app.settings.detail != Detail::Simple;
    let narrow = ui.available_width() < NARROW;
    if narrow || stacked {
        egui::ScrollArea::vertical().id_salt("reading_scroll").auto_shrink([false, false]).show(ui, |ui| {
            views::column(ui, views::MEASURE, |ui| {
                views::reading::hero(app, ui);
                if stacked {
                    ui.add_space(16.0);
                    controls::card(ui, 16, |ui| {
                        ui.set_width(ui.available_width());
                        views::reading::details(app, ui);
                    });
                }
                ui.add_space(16.0);
                let height = (ui.ctx().content_rect().height() * 0.7).clamp(240.0, 900.0);
                text_card(app, ui, height);
            });
        });
        return;
    }
    views::column(ui, views::MEASURE, |ui| {
        views::reading::hero(app, ui);
        ui.add_space(20.0);
        let height = (ui.available_height() - 104.0).max(160.0);
        text_card(app, ui, height);
    });
}

/// The user's text as the heat map, with Edit, the minimap of a long document, and the scale of the tints.
fn text_card(app: &mut LitmusApp, ui: &mut egui::Ui, height: f32) {
    let p = theme::palette(dark(ui));
    let detailed = app.settings.detail.at_least(Detail::Detailed);
    controls::card(ui, 16, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Your text").font(theme::strong(15.0)));
            if let Some(shown) = &app.shown {
                // A file's name; nothing for a pasted text or the contribute site's record of one. The text's size is
                // the counts line in the hero above.
                if shown.source != "pasted" && shown.scored.contribution.is_none() {
                    ui.label(RichText::new(format!("from {}", crate::app::file_name(&shown.source))).small().color(p.muted));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let busy = app.busy();
                let hint = if cfg!(target_arch = "wasm32") { None } else { Some("Ctrl+E") };
                if controls::quiet(ui, "Edit", hint, !busy).on_hover_text("Back to editing this text").clicked() {
                    app.editing = true;
                }
            });
        });
        if detailed && views::heat::wants_minimap(app) {
            ui.add_space(4.0);
            views::heat::minimap(app, ui);
            ui.add_space(4.0);
        }
        ui.add_space(4.0);
        views::heat::text(app, ui, height, detailed);
    });
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        views::heat::legend(ui);
        if detailed {
            ui.label(RichText::new("  Click a part of the text for its own reading.").small().color(p.muted));
        }
    });
}

/// The text box, its one primary action, and the states around it: loading, no model, scoring. On the contribute
/// page the site's notice, the consent box and the buttons sit under the box, measured on the frame before, so the
/// box takes the height left and everything needed to consent, check, or remove stays on screen.
fn editor_screen(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let narrow = ui.available_width() < 600.0;
    if cfg!(feature = "contribute") {
        // A screen too short for all of it scrolls rather than hide the Check button.
        let total = ui.available_height();
        egui::ScrollArea::vertical().id_salt("contribute_editor_scroll").auto_shrink([false, false]).show(ui, |ui| editor_column(app, ui, narrow, total));
    } else {
        let total = ui.available_height();
        editor_column(app, ui, narrow, total);
    }
}

fn editor_column(app: &mut LitmusApp, ui: &mut egui::Ui, narrow: bool, total: f32) {
    let p = theme::palette(dark(ui));
    let top = ui.min_rect().top();
    views::column(ui, views::MEASURE, |ui| {
        if app.shown.is_none() {
            // The contribute page's notice under the box says what happens to the text, so the page opens with the
            // question alone, and on a phone with the box itself.
            if cfg!(feature = "contribute") {
                if !narrow {
                    ui.label(RichText::new("Is this text written by a person, a model, or both?").font(theme::strong(26.0)));
                    ui.add_space(12.0);
                }
            } else {
                ui.label(RichText::new("Is this text written by a person, a model, or both?").font(theme::strong(if narrow { 22.0 } else { 26.0 })));
                let hint = if cfg!(target_arch = "wasm32") {
                    "Paste your text and press Check. It is read right here in this page and never leaves it."
                } else {
                    "Paste text, open a file, or drop one on the window, then press Check."
                };
                ui.label(RichText::new(hint).size(16.0).color(p.muted));
                ui.add_space(16.0);
            }
        } else {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Edit the text, then check it again.").size(16.0).color(p.muted));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if controls::quiet(ui, "Back to the reading", None, !app.busy()).clicked() {
                        app.editing = false;
                    }
                });
            });
            ui.add_space(8.0);
        }
        model_state(app, ui);
        let used = ui.min_rect().bottom() - top;
        let below_id = egui::Id::unique("contribute_below_the_box");
        let below = if cfg!(feature = "contribute") {
            // Measured on the frame before, plus the box's own frame (12 points of margin and up to 2 of stroke on
            // each side), which `height` leaves out.
            ui.ctx().data(|d| d.get_temp::<f32>(below_id)).unwrap_or(440.0) + 28.0
        } else if app.running.is_some() {
            72.0
        } else {
            64.0
        };
        let floor = if cfg!(feature = "contribute") { if narrow { 40.0 } else { 96.0 } } else if narrow { 180.0 } else { 220.0 };
        let height = (total - used - below).max(floor);
        editor(app, ui, height);
        let after_box = ui.min_rect().bottom();
        #[cfg(feature = "contribute")]
        {
            ui.add_space(if narrow { 8.0 } else { 12.0 });
            views::contribute::notice(app, ui);
            ui.add_space(if narrow { 6.0 } else { 10.0 });
            views::contribute::consent(app, ui);
        }
        ui.add_space(if cfg!(feature = "contribute") && narrow { 8.0 } else { 12.0 });
        if app.running.is_some() {
            progress(app, ui);
        } else {
            buttons(app, ui);
        }
        if cfg!(feature = "contribute") {
            // What sits under the box, plus the panel's margin, for the next frame's layout.
            let measured = ui.min_rect().bottom() - after_box + if narrow { 12.0 } else { 24.0 };
            if (measured + 28.0 - below).abs() > 1.0 {
                ui.ctx().data_mut(|d| d.insert_temp(below_id, measured));
                ui.ctx().request_repaint();
            }
        }
    });
}

/// The model loading (with its download in the browser) or missing, said with what to do.
fn model_state(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let p = theme::palette(dark(ui));
    match &app.model {
        ModelState::Ready(_) => {}
        ModelState::Loading => {
            controls::card(ui, 12, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.spinner();
                    match app.download {
                        Some((done, total)) if total > 0 => {
                            ui.label(format!("Downloading the model: {} of {}", megabytes(done), megabytes(total)));
                        }
                        _ => {
                            ui.label("Loading the model");
                        }
                    }
                });
                if let Some((done, total)) = app.download.filter(|(_, total)| *total > 0) {
                    bar(ui, done as f32 / total as f32);
                    ui.label(RichText::new("Once, into this page: the model runs here, and your text is never sent anywhere.").small().color(p.muted));
                }
            });
            ui.add_space(12.0);
        }
        ModelState::Failed(error) => {
            let error = error.clone();
            controls::card(ui, 16, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("No model is loaded, so nothing can be checked yet").font(theme::strong(16.0)).color(p.error));
                ui.label(RichText::new(error).color(p.muted));
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button("Choose a model folder in Settings").clicked() {
                    app.settings_open = true;
                }
                #[cfg(target_arch = "wasm32")]
                ui.label("Reload the page to try again.");
            });
            ui.add_space(12.0);
        }
    }
}

/// "6.2 MB".
pub fn megabytes(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1_000_000.0)
}

fn editor(app: &mut LitmusApp, ui: &mut egui::Ui, height: f32) {
    let p = theme::palette(dark(ui));
    let running = app.running.is_some();
    let id = egui::Id::unique("litmus_editor");
    let focused = ui.memory(|m| m.has_focus(id));
    let stroke = if focused { Stroke::new(2.0, p.focus) } else { Stroke::new(1.0, p.rule) };
    egui::Frame::new().fill(p.field).stroke(stroke).corner_radius(CornerRadius::same(theme::RADIUS)).inner_margin(Margin::symmetric(16, 12)).show(ui, |ui| {
        egui::ScrollArea::vertical().id_salt("editor_scroll").max_height(height).min_scrolled_height(height).auto_shrink([false, false]).show(ui, |ui| {
            let rows = ((height - 32.0) / 24.0).floor().max(4.0) as usize;
            let edit = egui::TextEdit::multiline(&mut app.input)
                .id(id)
                .font(egui::FontId::proportional(16.0))
                .desired_width(f32::INFINITY)
                .desired_rows(rows)
                .frame(egui::Frame::NONE)
                .margin(Margin::ZERO)
                .interactive(!running)
                .hint_text(RichText::new("Paste or type text here").size(16.0).color(p.muted))
                .lock_focus(false);
            let response = ui.add(edit);
            if response.changed() && app.input.trim().is_empty() {
                app.input_source = "pasted".into();
            }
        });
    });
}

fn bar(ui: &mut egui::Ui, fraction: f32) {
    let p = theme::palette(dark(ui));
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(3), p.rule);
    let filled = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * fraction.clamp(0.0, 1.0), rect.height()));
    ui.painter().rect_filled(filled, CornerRadius::same(3), p.primary);
}

/// While a text is read: how far, and Cancel. The window never freezes: the reading runs off the UI thread.
fn progress(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let p = theme::palette(dark(ui));
    let Some(running) = &app.running else { return };
    let (fraction, words) = (running.progress.fraction, running.progress.words.clone());
    let mut cancel = false;
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width((ui.available_width() - 120.0).max(120.0));
            ui.label(RichText::new(if words.is_empty() { "Reading".to_owned() } else { format!("Reading: {words}") }).color(p.muted));
            bar(ui, fraction);
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            cancel = ui.button("Cancel").on_hover_text("Esc").clicked();
        });
    });
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    if cancel {
        app.cancel();
    }
}

fn buttons(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let p = theme::palette(dark(ui));
    ui.horizontal_wrapped(|ui| {
        #[cfg(feature = "contribute")]
        let unticked = !app.contribute.consent;
        #[cfg(not(feature = "contribute"))]
        let unticked = false;
        let reason = if unticked {
            Some("Tick the box above to agree to the terms first.")
        } else if matches!(app.model, ModelState::Loading) {
            Some("The model is still loading.")
        } else if matches!(app.model, ModelState::Failed(_)) {
            Some("No model is loaded.")
        } else if app.busy() {
            Some("A reading is running.")
        } else if app.input.trim().is_empty() {
            Some("Paste or type some text first.")
        } else {
            None
        };
        let hint = if app.touch { None } else { Some("Ctrl+Enter") };
        let response = controls::primary(ui, "Check", hint, app.can_check());
        let response = match reason {
            Some(reason) => response.on_hover_text(reason),
            None => response,
        };
        if response.clicked() {
            app.check();
        }
        ui.add_space(4.0);
        // On a phone the contribute page keeps one row of buttons, Remove a text on it.
        let phone_contribute = cfg!(feature = "contribute") && app.touch;
        if !phone_contribute && controls::quiet(ui, "Sample", None, !app.busy()).on_hover_text("A short passage to try Litmus with").clicked() {
            app.input = crate::sample::TEXT.to_owned();
            app.input_source = "pasted".into();
        }
        #[cfg(not(target_arch = "wasm32"))]
        if controls::quiet(ui, "Open file", None, !app.busy()).on_hover_text("Ctrl+O: text, Markdown, HTML, Word or PDF").clicked()
            && let Some(path) = views::pick_file()
        {
            app.open_file(&path);
        }
        if !app.input.is_empty() && controls::quiet(ui, "Clear", None, !app.busy()).clicked() {
            app.input.clear();
            app.input_source = "pasted".into();
        }
        #[cfg(feature = "contribute")]
        if controls::quiet(ui, "Remove a text", None, true).on_hover_text("Ask the site to delete a text it stored").clicked() {
            app.contribute.removal.open = true;
        }
        if !app.input.is_empty() && !phone_contribute {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("{} characters", litmus_files::export::thousands(app.input.chars().count()))).small().color(p.muted));
            });
        }
    });
}
