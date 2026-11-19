//! A reading, at the level of detail the user chose: Simple by default, more detail on request.
//!
//! - The hero, at every level: the verdict word with its marker, one plain sentence, the strip with its band, and
//!   the cautions that keep it honest (unlike the reference, a few words, a mixed document).
//! - Detailed adds the headline numbers (P(not human) with its interval, the set at 10%, tokens, time), what a
//!   reading is with our recommendation for using one, and the reading of a clicked part of the text.
//! - Full adds every other number, how the reading was made, the model, and the exports.

use eframe::egui::{self, RichText};
use litmus_engine::{Origin, STRIP_LEVEL};
use litmus_files::export;

use crate::app::LitmusApp;
use crate::settings::Detail;
use crate::strip;
use crate::views::{self, controls, dark};
use crate::wording::{self, Call};
use crate::{brand, theme};

/// The verdict, its sentence, the strip and the cautions.
pub fn hero(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let Some(shown) = &app.shown else { return };
    let reading = shown.scored.reading.clone();
    let counts_line = shown.counts.line();
    let verdict = &reading.verdict;
    let set = verdict.set(STRIP_LEVEL);
    let call = wording::call(set);
    let p = theme::palette(dark(ui));
    let word_color = match call {
        Call::One(origin) => p.origin_ink(origin),
        Call::Uncertain | Call::UnlikeReference => p.ink,
    };
    let narrow = ui.available_width() < 520.0;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 12.0;
        verdict_marker(ui, call, verdict.strip.position, if narrow { 24.0 } else { 30.0 });
        ui.label(RichText::new(wording::verdict_headline(verdict)).font(theme::strong(if narrow { 26.0 } else { 34.0 })).color(word_color));
        if verdict.out_of_reference && call != Call::UnlikeReference {
            egui::Frame::new().fill(p.caution).corner_radius(egui::CornerRadius::same(theme::RADIUS)).inner_margin(egui::Margin::symmetric(8, 3)).show(ui, |ui| {
                ui.label(RichText::new("? unlike reference").color(p.caution_ink).font(theme::strong(13.0)))
                    .on_hover_text("The text sits farther from the reference bank than the calibration texts of its length; we recommend reading this verdict with caution.");
            });
        }
    });
    ui.add_space(2.0);
    ui.label(RichText::new(wording::plain_sentence(set)).size(17.0).color(p.muted));
    if verdict.out_of_reference && call != Call::UnlikeReference {
        ui.label(RichText::new(wording::OUT_OF_REFERENCE).size(15.0).color(p.muted));
    }
    if reading.n_tokens < wording::FEW_TOKENS {
        ui.label(RichText::new(wording::FEW_WORDS).size(15.0).color(p.muted));
    }
    ui.label(RichText::new(counts_line).size(14.0).color(p.muted));
    ui.add_space(12.0);
    strip::show(ui, &verdict.strip, if narrow { 20.0 } else { 26.0 });
    for (title, body) in wording::flags(&reading) {
        // At the simple level only the finding that changes the reading is shown; its absence is a detail.
        if title == "Mixed document" {
            ui.add_space(4.0);
            views::caution(ui, title, &body);
        }
    }
    if app.settings.detail == Detail::Simple {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            if controls::link(ui, "More detail").on_hover_text("Ctrl+M").clicked() {
                app.set_detail(Detail::Detailed);
            }
        });
    }
    #[cfg(feature = "contribute")]
    views::contribute::labels(app, ui);
}

/// The detail level's switch, the headline numbers, and at Full every number, the explanation and the exports.
pub fn details(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let options: Vec<(Detail, &str)> = Detail::ALL.iter().map(|d| (*d, d.name())).collect();
    ui.horizontal(|ui| {
        controls::section(ui, "Detail");
        ui.add_space(4.0);
        if let Some(detail) = controls::segmented(ui, app.settings.detail, &options, "detail_level") {
            app.set_detail(detail);
        }
    });
    ui.add_space(12.0);
    let Some(shown) = &app.shown else { return };
    let reading = shown.scored.reading.clone();
    let v = &reading.verdict;
    controls::section(ui, "The numbers");
    egui::Grid::new("headline_numbers").num_columns(2).spacing([16.0, 2.0]).show(ui, |ui| {
        controls::fact(ui, "P(not human)", wording::probability(v.probability, v.interval), Some("The calibrated probability that the text is not purely human-written, with its interval. We recommend reading the interval before the number."));
        let set = v.set(STRIP_LEVEL);
        let words: Vec<&str> = [Origin::Human, Origin::Mixed, Origin::Ai].into_iter().filter(|o| set.is_some_and(|s| s.contains(o))).map(wording::origin_word).collect();
        let kept = if words.is_empty() { "none".to_owned() } else { wording::join_and(&words) };
        controls::fact(ui, "Not ruled out", format!("{kept} (at 10%)"), Some("The origins the reading cannot rule out at the 10% level: the verdict."));
        controls::fact(
            ui,
            "Tokens",
            if v.chunks > 1 { format!("{} in {} sections", export::thousands(reading.n_tokens), v.chunks) } else { export::thousands(reading.n_tokens) },
            None,
        );
        controls::fact(
            ui,
            "Characters without spaces",
            export::thousands(shown.counts.characters_no_spaces),
            Some("Characters that are not spaces, tabs, or line breaks. A word is a run of characters between spaces, so text written without spaces counts as one word per run."),
        );
        let time = match shown.scored.seconds {
            Some(seconds) if shown.scored.contribution.is_some() => format!("{} (on the site's server)", wording::duration(seconds)),
            Some(seconds) => wording::duration(seconds),
            None => "from the cache".to_owned(),
        };
        controls::fact(ui, "Time", time, None);
    });
    if let Some((_, body)) = wording::flags(&reading).into_iter().find(|(title, _)| *title == "No mixed-document signal") {
        ui.add_space(4.0);
        ui.label(RichText::new(body).small().weak());
    }
    ui.add_space(8.0);
    ui.label(RichText::new(wording::RECOMMENDATION).small().color(theme::palette(dark(ui)).muted));
    ui.add_space(16.0);
    views::heat::cell_detail(app, ui);
    if !app.settings.detail.at_least(Detail::Full) {
        return;
    }
    ui.add_space(16.0);
    full_numbers(app, ui);
    ui.add_space(16.0);
    let heading = ui.label(egui::RichText::new("How this was read").font(theme::strong(13.0)).color(theme::palette(dark(ui)).muted));
    if std::mem::take(&mut app.reveal_explain) {
        heading.scroll_to_me(Some(egui::Align::Min));
    }
    ui.add_space(4.0);
    views::explain::show(app, ui);
    ui.add_space(16.0);
    controls::section(ui, "Export every field");
    ui.horizontal_wrapped(|ui| exports(app, ui));
}

/// A swatch beside the headline: the strip color at the position, shaped by the call (a disc for one origin, a
/// ring for uncertain, a question mark for unlike reference).
fn verdict_marker(ui: &mut egui::Ui, call: Call, position: f64, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(size), egui::Sense::hover());
    let ink = ui.visuals().text_color();
    let fill = theme::rgb(brand::litmus(position));
    let painter = ui.painter();
    match call {
        Call::One(_) => {
            painter.circle(rect.center(), size * 0.42, fill, egui::Stroke::new(1.5, ink));
        }
        Call::Uncertain => {
            painter.circle_stroke(rect.center(), size * 0.34, egui::Stroke::new(size * 0.18, fill));
            painter.circle_stroke(rect.center(), size * 0.46, egui::Stroke::new(1.0, ink));
        }
        Call::UnlikeReference => {
            painter.circle_stroke(rect.center(), size * 0.42, egui::Stroke::new(1.5, ink));
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "?", theme::strong(size * 0.62), ink);
        }
    }
}

/// Every number of the verdict that the headline leaves out.
fn full_numbers(app: &LitmusApp, ui: &mut egui::Ui) {
    let Some(shown) = &app.shown else { return };
    let reading = &shown.scored.reading;
    let v = &reading.verdict;
    controls::section(ui, "Every number");
    egui::Grid::new("full_numbers").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
        let p = theme::palette(dark(ui));
        ui.label(RichText::new("Three-way").color(p.muted)).on_hover_text("Calibrated probabilities of human, mixed and AI; they sum to one.");
        ui.horizontal(|ui| {
            for origin in [Origin::Human, Origin::Mixed, Origin::Ai] {
                views::origin_marker(ui, origin);
                ui.label(RichText::new(format!("{} {:.2}", wording::origin_word(origin), v.three_way[origin.index()])).monospace());
            }
        });
        ui.end_row();
        controls::fact(
            ui,
            "Conformal p",
            format!("human {:.3}, mixed {:.3}, AI {:.3}", v.p_values[0], v.p_values[2], v.p_values[1]),
            Some("Per origin: how typical the text is of that origin's calibration texts. An origin with p at or under 0.10 is ruled out at the 10% level."),
        );
        controls::fact(ui, "Not ruled out", format!("at 10% {}, at 5% {}", export::set_words(v.set(0.1)), export::set_words(v.set(0.05))), None);
        if !v.set(STRIP_LEVEL).is_none_or(|set| set.is_empty()) {
            controls::fact(
                ui,
                "Not human at 10%",
                if v.not_human.get("0.1").copied().unwrap_or(false) { "called".to_owned() } else { "not called".to_owned() },
                Some("For human text like the calibration texts of its length, at most 10% is called not human."),
            );
        }
        controls::fact(
            ui,
            "Closeness p",
            format!("{:.3}{}", v.closeness_p, if v.out_of_reference { ", out of reference" } else { "" }),
            Some("How close the text sits to the reference bank, against calibration texts of its length; very low means out of reference."),
        );
        controls::fact(ui, "Model", shown.model.clone(), None);
        controls::fact(ui, "Read at", crate::clock::format_utc(shown.at_ms), None);
    });
}

fn exports(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let Some(shown) = &app.shown else { return };
    let (json, csv) = {
        let row = views::reading_row(shown);
        (serde_json::to_string_pretty(&export::json_value(&row)).unwrap_or_default(), export::csv(&[row]))
    };
    let verb = if cfg!(target_arch = "wasm32") { "Copy" } else { "Export" };
    let ctx = ui.ctx().clone();
    if ui.button(format!("{verb} CSV")).clicked() {
        views::deliver(app, &ctx, "litmus_reading.csv", ("CSV", "csv"), csv);
    }
    if ui.button(format!("{verb} JSON")).on_hover_text("Every field of the reading").clicked() {
        views::deliver(app, &ctx, "litmus_reading.json", ("JSON", "json"), json);
    }
}
