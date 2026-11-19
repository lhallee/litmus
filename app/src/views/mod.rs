//! The screens and the pieces they share: the top bar, notices, the footer, origin markers and exports.

pub mod check;
#[cfg(feature = "contribute")]
pub mod contribute;
pub mod controls;
pub mod explain;
pub mod heat;
pub mod history;
pub mod reading;
pub mod settings;
pub mod table;
#[cfg(not(target_arch = "wasm32"))]
pub mod batch;
#[cfg(not(target_arch = "wasm32"))]
pub mod quick;

use eframe::egui::{self, Color32, CornerRadius, Margin, RichText, Sense, Stroke, Vec2};
use litmus_engine::Origin;
use litmus_files::export;

use crate::app::{LitmusApp, ModelState, Notice, Shown, Tab};
use crate::brand;
use crate::theme;

/// The widest the reading and the editor grow: a comfortable measure for text.
pub const MEASURE: f32 = 880.0;

pub fn dark(ui: &egui::Ui) -> bool {
    ui.visuals().dark_mode
}

/// A filled marker whose shape names the origin (circle human, diamond mixed, square AI) in its color, so the
/// origin reads without color.
pub fn origin_marker(ui: &mut egui::Ui, origin: Origin) -> egui::Response {
    let size = 11.0;
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let painter = ui.painter();
    let fill = theme::rgb(brand::litmus(origin.strip_anchor()));
    let ink = Stroke::new(1.0, ui.visuals().text_color());
    let c = rect.center();
    let r = size * 0.45;
    match origin {
        Origin::Human => {
            painter.circle(c, r, fill, ink);
        }
        Origin::Mixed => {
            let points = vec![c + Vec2::new(0.0, -r * 1.1), c + Vec2::new(r * 1.1, 0.0), c + Vec2::new(0.0, r * 1.1), c + Vec2::new(-r * 1.1, 0.0)];
            painter.add(egui::Shape::convex_polygon(points, fill, ink));
        }
        Origin::Ai => {
            painter.rect(egui::Rect::from_center_size(c, Vec2::splat(r * 1.8)), 0.0, fill, ink, egui::StrokeKind::Inside);
        }
    }
    response
}

/// A marker and the origin's word.
pub fn origin_label(ui: &mut egui::Ui, origin: Origin) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        origin_marker(ui, origin);
        ui.label(crate::wording::origin_word(origin));
    });
}

/// A caution box: a triangle, a semibold title and a sentence, on the caution color.
pub fn caution(ui: &mut egui::Ui, title: &str, body: &str) {
    let p = theme::palette(dark(ui));
    egui::Frame::new().fill(p.caution).corner_radius(CornerRadius::same(theme::RADIUS)).inner_margin(Margin::symmetric(12, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            controls::caution_icon(ui, p.caution_ink);
            ui.label(RichText::new(title).font(theme::strong(14.0)).color(p.caution_ink));
            if !body.is_empty() {
                ui.label(RichText::new(body).color(p.caution_ink));
            }
        });
    });
}

/// The top bar: the mark and name, the screens, the theme switch and Settings. Nothing else: the model and its
/// hash are in Settings.
pub fn top_bar(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let p = theme::palette(dark(ui));
    egui::Panel::top("top").frame(egui::Frame::new().fill(p.ground).inner_margin(Margin::symmetric(16, 4))).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if let Some(mark) = &app.mark {
                ui.add(egui::Image::new(mark).fit_to_exact_size(Vec2::new(20.0, 28.0)));
            }
            ui.add_space(2.0);
            ui.label(RichText::new(brand::NAME).font(theme::strong(19.0)));
            ui.add_space(20.0);
            let tabs: &[(Tab, &str, &str)] = &[
                (Tab::Check, "Check", "Ctrl+1"),
                #[cfg(not(target_arch = "wasm32"))]
                (Tab::Batch, "Batch", "Ctrl+2"),
                (Tab::History, "History", "Ctrl+3"),
            ];
            for (tab, name, keys) in tabs {
                if controls::nav_tab(ui, app.tab == *tab, name, keys).clicked() {
                    app.tab = *tab;
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if controls::quiet(ui, "Settings", None, true).on_hover_text("Ctrl+,").clicked() {
                    app.settings_open = !app.settings_open;
                }
                if controls::theme_toggle(ui).clicked() {
                    app.toggle_theme(&ui.ctx().clone());
                }
                if let ModelState::Loading = app.model {
                    ui.add_space(8.0);
                    ui.label(RichText::new("Loading the model").color(p.muted).small());
                    ui.spinner();
                }
            });
        });
    });
}

/// The placeholder warning (always, at every level, in one line), and errors and confirmations with what to do.
pub fn notices(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let placeholder = app.meta().is_some_and(|m| m.placeholder);
    if !placeholder && app.notice.is_none() {
        return;
    }
    let p = theme::palette(dark(ui));
    if placeholder {
        egui::Panel::top("placeholder").frame(egui::Frame::new().fill(p.caution).inner_margin(Margin::symmetric(16, 6))).show(ui, |ui| {
            let narrow = ui.available_width() < 600.0;
            ui.horizontal_wrapped(|ui| {
                controls::caution_icon(ui, p.caution_ink);
                ui.label(RichText::new("Placeholder model: readings are meaningless.").font(theme::strong(14.0)).color(p.caution_ink));
                if !narrow {
                    ui.label(RichText::new("It has random weights and is here to show the app; the trained model replaces it.").color(p.caution_ink).small())
                        .on_hover_text("Its config.json says provenance kind \"random\".");
                }
            });
        });
    }
    let mut close = false;
    if let Some(notice) = &app.notice {
        egui::Panel::top("notice").frame(egui::Frame::new().fill(p.ground).inner_margin(Margin { left: 16, right: 16, top: 12, bottom: 0 })).show(ui, |ui| {
            let (edge, title, detail) = match notice {
                Notice::Error { title, detail } => (p.error, Some(title.as_str()), detail.as_str()),
                Notice::Info(text) => (p.rule, None, text.as_str()),
            };
            let shown = egui::Frame::new().fill(p.raised).stroke(Stroke::new(1.0, p.rule)).corner_radius(CornerRadius::same(theme::RADIUS)).inner_margin(Margin { left: 16, right: 8, top: 6, bottom: 6 }).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        close = controls::quiet(ui, "Dismiss", None, true).clicked();
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center).with_main_wrap(true), |ui| {
                            if let Some(title) = title {
                                ui.label(RichText::new(title).font(theme::strong(15.0)));
                            }
                            ui.label(RichText::new(detail).color(p.muted));
                        });
                    });
                });
            });
            let r = shown.response.rect;
            ui.painter().rect_filled(egui::Rect::from_min_max(r.left_top(), egui::pos2(r.left() + 4.0, r.bottom())), CornerRadius { nw: theme::RADIUS, sw: theme::RADIUS, ne: 0, se: 0 }, edge);
        });
    }
    if close {
        app.notice = None;
    }
}

/// One quiet line: where the text is read.
pub fn footer(app: &mut LitmusApp, ui: &mut egui::Ui) {
    // The contribute page's notice, above its Check button, says this already, and the height goes to the box.
    if cfg!(feature = "contribute") {
        return;
    }
    let p = theme::palette(dark(ui));
    egui::Panel::bottom("footer").frame(egui::Frame::new().fill(p.ground).inner_margin(Margin::symmetric(16, 6))).show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            let words = if cfg!(feature = "contribute") {
                "Contribute site: texts you check are sent to this site's server, scored there, and stored with their readings."
            } else if cfg!(target_arch = "wasm32") {
                "Private: your text never leaves this page."
            } else {
                "Private: everything is read on this computer."
            };
            privacy_dot(ui, p.human_ink);
            ui.label(RichText::new(words).small().color(p.muted));
            #[cfg(not(target_arch = "wasm32"))]
            if app.settings.detail.at_least(crate::settings::Detail::Full) {
                ui.label(RichText::new(format!("History: {}{}.", plural(app.history.entries().len(), "reading"), if app.history.is_persistent() { "" } else { " (this session only)" })).small().color(p.muted));
            }
            #[cfg(target_arch = "wasm32")]
            let _ = &app;
        });
    });
}

fn privacy_dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 12.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.0, color);
}

/// "1 reading", "3 readings".
pub fn plural(n: usize, noun: &str) -> String {
    format!("{} {noun}{}", export::thousands(n), if n == 1 { "" } else { "s" })
}

/// The rows of exports for one shown reading.
pub fn reading_row(shown: &Shown) -> export::Row<'_> {
    export::Row { name: &shown.title, hash: &shown.scored.hash, cached: shown.scored.seconds.is_none(), outcome: Ok(&shown.scored.reading), counts: Some(&shown.counts) }
}

/// Offer text as a file: a save dialog on the desktop, the clipboard in the browser.
pub fn deliver(app: &mut LitmusApp, ctx: &egui::Context, suggested: &str, filter: (&str, &str), contents: String) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = ctx;
        let Some(path) = rfd::FileDialog::new().set_file_name(suggested).add_filter(filter.0, &[filter.1]).save_file() else { return };
        app.notice = Some(match std::fs::write(&path, contents) {
            Ok(()) => Notice::Info(format!("Saved {}.", path.display())),
            Err(error) => Notice::error(format!("{} was not saved", path.display()), format!("{error}. Choose a folder you can write to.")),
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (suggested, filter);
        ctx.copy_text(contents);
        app.notice = Some(Notice::Info("Copied to the clipboard.".into()));
    }
}

/// A file chosen in the system's dialog.
#[cfg(not(target_arch = "wasm32"))]
pub fn pick_file() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new().add_filter("Text, Markdown, HTML, Word, PDF", &["txt", "text", "md", "markdown", "html", "htm", "docx", "pdf"]).pick_file()
}

/// A centered column at most `MEASURE` wide with 24-point gutters, for the editor and the reading.
pub fn column<R>(ui: &mut egui::Ui, width: f32, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let available = ui.available_width();
    let gutter = if available < 600.0 { 12.0 } else { 24.0 };
    let inner = (available - 2.0 * gutter).min(width).max(120.0);
    let left = ((available - inner) / 2.0).max(0.0);
    ui.horizontal_top(|ui| {
        ui.add_space(left);
        ui.vertical(|ui| {
            ui.set_width(inner);
            add(ui)
        })
        .inner
    })
    .inner
}
