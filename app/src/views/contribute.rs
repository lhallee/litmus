//! The contribute page's own parts (feature `contribute`; `../contribute/README.md`, "What the contribute page must
//! send and show"): the site's notice as served, directly above the consent box and the Check button, in body type
//! with its bold sentence and its links underlined in the link ink; the consent box with the served label, unticked
//! on every load; the removal form, opened from the notice's words "removal form" or the Remove a text button; and
//! the visitor's label of a reading.

use eframe::egui::{self, Color32, RichText, Role, Sense, Vec2, WidgetInfo};

use crate::app::LitmusApp;
use crate::contribute::{self, Annotation, Span};
use crate::theme;
use crate::views::{self, controls, dark};

/// The notice's and the consent label's type: the page's body text.
const BODY: f32 = 15.0;

/// The notice's line height: tighter on a phone, so the notice, the box and Check fit one screen.
fn line(ui: &egui::Ui) -> f32 {
    if ui.ctx().content_rect().width() < 600.0 { 18.0 } else { 21.0 }
}

/// What a click in served Markdown asked for.
#[derive(Default)]
struct Clicked {
    removal_form: bool,
    words: bool,
}

/// Words in the body type. The browser's bundled face has no bold, so bold words are drawn twice, half a point
/// apart, which reads as bold at this size.
fn words(ui: &mut egui::Ui, text: &str, bold: bool, ink: Color32, sense: Sense) -> egui::Response {
    let (at, galley, response) = egui::Label::new(RichText::new(text).size(BODY).line_height(Some(line(ui))).color(ink)).sense(sense).selectable(false).layout_in_ui(ui);
    response.widget_info(|| WidgetInfo::labeled(Role::Label, true, text));
    if ui.is_rect_visible(response.rect) {
        ui.painter().galley(at, galley.clone(), ink);
        if bold {
            ui.painter().galley(at + Vec2::new(0.6, 0.0), galley, ink);
        }
    }
    response
}

/// A link in the body type: underlined, in the link ink, always, with the focus ring the app's controls share.
fn link(ui: &mut egui::Ui, text: &str, ink: Color32) -> egui::Response {
    let rich = RichText::new(text).size(BODY).line_height(Some(line(ui))).color(ink);
    let (at, galley, response) = egui::Label::new(rich).sense(Sense::click()).selectable(false).layout_in_ui(ui);
    response.widget_info(|| WidgetInfo::labeled(Role::Link, true, text));
    if ui.is_rect_visible(response.rect) {
        // The underline just under the letters of each row (the text format's underline sits at the foot of the
        // taller line, against the next row's capitals).
        for row in &galley.rows {
            let (Some(first), Some(last)) = (row.row.glyphs.first(), row.row.glyphs.last()) else { continue };
            let origin = at + row.pos.to_vec2();
            let y = origin.y + BODY + 3.0;
            let (left, right) = (origin.x + first.pos.x, origin.x + last.pos.x + last.advance_width);
            ui.painter().line_segment([egui::pos2(left, y), egui::pos2(right, y)], egui::Stroke::new(1.0, ink));
        }
        ui.painter().galley(at, galley, ink);
    }
    controls::focus_ring(ui, &response);
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// One paragraph of served Markdown, wrapped to the width. A link opens in a new tab, so the page (the text and the
/// box) stays as it is; the words "removal form" open the form. `sense` lets the words themselves be clicked (the
/// consent label's words tick its box).
fn paragraph(ui: &mut egui::Ui, spans: &[Span], sense: Sense) -> Clicked {
    // Every row the height of a line: the wrapped layout takes its row height from the spacing it starts with, and
    // a run that starts mid-row would otherwise get a taller first row.
    let row = line(ui);
    ui.scope(|ui| {
        ui.spacing_mut().interact_size.y = row;
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        ui.horizontal_wrapped(|ui| runs(ui, spans, sense)).inner
    })
    .inner
}

/// The spans of one paragraph, in a wrapped row.
fn runs(ui: &mut egui::Ui, spans: &[Span], sense: Sense) -> Clicked {
    let p = theme::palette(dark(ui));
    let mut clicked = Clicked::default();
    for span in spans {
        match span {
            Span::Link { text, url } => {
                if link(ui, text, p.human_ink).on_hover_text(url).clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(url));
                }
            }
            Span::Text { text, bold } => {
                let mut rest = text.as_str();
                while let Some(at) = rest.find(contribute::REMOVAL_FORM) {
                    clicked.words |= words(ui, &rest[..at], *bold, p.ink, sense).clicked();
                    clicked.removal_form |= link(ui, contribute::REMOVAL_FORM, p.human_ink).on_hover_text("Opens the removal form on this page").clicked();
                    rest = &rest[at + contribute::REMOVAL_FORM.len()..];
                }
                if !rest.is_empty() {
                    clicked.words |= words(ui, rest, *bold, p.ink, sense).clicked();
                }
            }
        }
    }
    clicked
}

/// The site's notice, as served, above the consent box: or why it is not there yet.
pub fn notice(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let terms = match &app.contribute.terms {
        None => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Loading the site's terms; checking starts once they are shown.");
            });
            return;
        }
        Some(Err(error)) => {
            let error = error.clone();
            views::caution(ui, "The site's terms did not load, so checking is off.", &error);
            return;
        }
        Some(Ok(terms)) => terms.clone(),
    };
    let margin = if app.touch { 6 } else { 14 };
    let open = controls::card(ui, margin, |ui| {
        ui.set_width(ui.available_width());
        let mut open = false;
        for (index, spans) in contribute::paragraphs(&terms.notice).iter().enumerate() {
            if index > 0 {
                ui.add_space(if app.touch { 3.0 } else { 6.0 });
            }
            open |= paragraph(ui, spans, Sense::hover()).removal_form;
        }
        open
    });
    if open {
        app.contribute.removal.open = true;
    }
}

/// The consent box and its served label (the documents' names are links), and the terms' version. The box is a real
/// checkbox whose accessible name is the label in plain words; it is never ticked by the page.
pub fn consent(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let Some(Ok(terms)) = &app.contribute.terms else { return };
    let (label, version) = (terms.consent_label.clone(), format!("Terms version {}{}", terms.version, if terms.status == "draft" { " (draft)" } else { "" }));
    let plain = contribute::notice_plain(&label);
    let touch = app.touch;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().icon_width = if touch { 24.0 } else { 20.0 };
        ui.spacing_mut().icon_width_inner = if touch { 14.0 } else { 11.0 };
        let ticked = app.contribute.consent;
        let response = ui.add(egui::Checkbox::without_text(&mut app.contribute.consent));
        response.widget_info(|| WidgetInfo::selected(Role::CheckBox, true, ticked, &plain));
        ui.add_space(6.0);
        ui.vertical(|ui| {
            let mut clicked = Clicked::default();
            for spans in contribute::paragraphs(&label) {
                let more = paragraph(ui, &spans, Sense::click());
                clicked.words |= more.words;
                clicked.removal_form |= more.removal_form;
            }
            if clicked.words {
                app.contribute.consent = !app.contribute.consent;
            }
            if clicked.removal_form {
                app.contribute.removal.open = true;
            }
            if app.contribute.consent {
                app.contribute.terms_changed = false;
            }
            if app.contribute.terms_changed {
                // In place of the version line, so the page keeps its height: the site refused the text for new terms.
                let p = theme::palette(dark(ui));
                ui.label(RichText::new("New terms: nothing was sent; tick the box again.").small().color(p.caution_ink).background_color(p.caution))
                    .on_hover_text(version);
            } else {
                ui.label(RichText::new(version).small().weak());
            }
        });
    });
}

/// The removal form, over the page: a text box, Remove, and what happens next. Open to anyone, no box to tick.
pub fn removal(app: &mut LitmusApp, ctx: &egui::Context) {
    if !app.contribute.removal.open {
        return;
    }
    let width = (ctx.content_rect().width() - 40.0).clamp(240.0, 560.0);
    let rows = if ctx.content_rect().height() < 700.0 || app.touch { 5 } else { 8 };
    let editor_text = app.input.clone();
    let modal = egui::Modal::new(egui::Id::unique("litmus_removal_form")).show(ctx, |ui| {
        let p = theme::palette(dark(ui));
        ui.set_width(width);
        let form = &mut app.contribute.removal;
        ui.label(RichText::new("Remove a text").font(theme::strong(20.0)));
        ui.add_space(4.0);
        ui.label(RichText::new(
            "Paste the exact text that was checked here. Every stored copy of that exact text is deleted from our storage \
             and our private dataset within 30 days. Models already trained on it are not retrained. Anyone can ask this \
             for any text.",
        ).size(BODY));
        ui.add_space(8.0);
        // One height whether the box is empty or full, so the buttons under it stay where they are.
        let height = rows as f32 * 22.0 + 16.0;
        let edit = egui::TextEdit::multiline(&mut form.text)
            .desired_width(f32::INFINITY)
            .min_size(Vec2::new(0.0, height))
            .font(egui::FontId::proportional(BODY))
            .hint_text("The exact text to remove");
        let changed = egui::ScrollArea::vertical().id_salt("removal_scroll").max_height(height).show(ui, |ui| ui.add(edit)).inner.changed();
        if changed {
            form.answer = None;
        }
        ui.add_space(8.0);
        let mut close = false;
        ui.horizontal_wrapped(|ui| {
            let ready = !form.text.is_empty() && !form.sending;
            if controls::primary(ui, if form.sending { "Sending" } else { "Remove" }, None, ready).clicked() && ready {
                form.sending = true;
                form.answer = None;
                app.scorer.remove(form.text.clone());
            }
            if !editor_text.is_empty() && controls::quiet(ui, "Use the text from the box", None, !form.sending).clicked() {
                form.text = editor_text.clone();
                form.answer = None;
            }
            close = controls::quiet(ui, "Close", None, true).clicked();
        });
        match &form.answer {
            Some(Ok(_)) => {
                ui.add_space(8.0);
                ui.label(RichText::new(
                    "Received. Every stored copy of this exact text will be deleted within 30 days. Only the same text, \
                     character for character, matches it.",
                ).size(BODY).color(p.human_ink));
            }
            Some(Err(error)) => {
                ui.add_space(8.0);
                views::caution(ui, "The request was not taken:", error);
            }
            None => {}
        }
        close
    });
    if modal.inner || modal.should_close() {
        app.contribute.removal.open = false;
    }
}

/// The visitor's label of the shown text, within the site's window for labels.
pub fn labels(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let Some(contribution) = app.shown.as_ref().and_then(|s| s.scored.contribution.clone()) else { return };
    let open = crate::clock::now_ms() - contribution.received_ms < contribution.annotation_window_seconds as i64 * 1000;
    ui.add_space(8.0);
    controls::card(ui, 12, |ui| {
        ui.label(RichText::new("Do you know where this text came from? Your label is stored with it (optional).").font(theme::strong(15.0)));
        ui.horizontal(|ui| {
            for annotation in Annotation::ALL {
                let selected = app.contribute.annotation == Some(annotation);
                if ui.add_enabled(open, egui::Button::selectable(selected, annotation.button())).clicked() {
                    let next = if selected { None } else { Some(annotation) };
                    app.scorer.annotate(contribution.id.clone(), next);
                }
            }
            if !open {
                ui.label(RichText::new("The time to label this text has passed.").weak());
            }
        });
        if let Some(note) = &app.contribute.annotation_note {
            ui.label(RichText::new(note).weak());
        }
    });
}
