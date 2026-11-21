//! How a reading was made, from the engine's own numbers: how the sections combine, the evidence against the
//! reference bank, the conformal counts in plain words with the calibration group they come from, the probability,
//! the closeness to the reference, and the nearest reference texts with their origin and cosine.

use eframe::egui::{self, RichText};
use litmus_engine::{Chance, DocumentReading, Origin, Reading};
use litmus_files::export::thousands;

use crate::app::LitmusApp;
use crate::model::ModelMeta;
use crate::views;
use crate::wording::{self, origin_word};

pub fn show(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let Some(shown) = &app.shown else { return };
    let reading = shown.scored.reading.clone();
    let meta = app.meta().cloned();
    let chance = app.chance(&shown.scored);
    ui.scope(|ui| {
        combine(ui, &reading, chance.as_ref());
        if reading.chunks.len() > 1 {
            ui.horizontal(|ui| {
                ui.label("Section");
                egui::ComboBox::from_id_salt("explain_chunk")
                    .selected_text(format!("{} of {}", app.explain_chunk + 1, reading.chunks.len()))
                    .show_ui(ui, |ui| {
                        for (index, chunk) in reading.chunks.iter().enumerate() {
                            let label = format!(
                                "{} (characters {} to {}), {}",
                                index + 1,
                                thousands(chunk.place.start_char),
                                thousands(chunk.place.end_char),
                                wording::reading_headline(&chunk.reading).line()
                            );
                            ui.selectable_value(&mut app.explain_chunk, index, label);
                        }
                    });
            });
        }
        let index = app.explain_chunk.min(reading.chunks.len().saturating_sub(1));
        if let Some(chunk) = reading.chunks.get(index) {
            chunk_account(ui, &chunk.reading, meta.as_ref());
        }
    });
}

/// How a chance of AI came from a probability of not human: the share of not-human calibration texts it divides out,
/// and what its range holds when the release's table carries the shift between sources.
fn at_even_odds(chance: Option<&Chance>) -> String {
    chance.map_or_else(String::new, |chance| {
        let range = if chance.between_sources > 0.0 {
            format!(
                " The range is the calibration's own interval widened by how far the chance moves from one source of text to another, {:.1} times the odds either way.",
                chance.between_sources.exp()
            )
        } else {
            String::new()
        };
        format!(
            " Of the calibration texts of this length, {} are not human to begin with, so for a text as likely human as not the chance of AI is {}.{range} The strip marks that chance and its pale band covers that range.",
            wording::percent_fine(chance.reference_share),
            chance.sentence()
        )
    })
}

/// How several sections make one verdict (one section is its own verdict).
fn combine(ui: &mut egui::Ui, document: &DocumentReading, chance: Option<&Chance>) {
    let v = &document.verdict;
    if v.chunks <= 1 {
        ui.label(format!(
            "The text is {} tokens, read in one pass (a section holds at most {} tokens), so its verdict is that section's reading.",
            thousands(document.n_tokens),
            thousands(document.chunk_tokens)
        ));
        ui.add_space(6.0);
        return;
    }
    ui.label(RichText::new("How the sections combine").font(crate::theme::strong(14.0)));
    ui.label(format!(
        "{} tokens were read as {} sections of about {} each (at most {}), each as a text of its own.",
        thousands(document.n_tokens),
        v.chunks,
        thousands(document.n_tokens / v.chunks),
        thousands(document.chunk_tokens)
    ));
    ui.label(format!(
        "Merged p-values are twice the sections' mean, valid however the sections depend on each other. They are human {:.3}, mixed {:.3}, and AI {:.3}.",
        v.p_values[0], v.p_values[2], v.p_values[1]
    ));
    ui.label(format!(
        "The probability of not human, {}, is the sections' mean. Its range, {} to {}, is their mean range widened by their disagreement ({:.3}). The sections run from {} to {}.{}",
        wording::percent_fine(v.probability),
        wording::percent_fine(v.interval[0]),
        wording::percent_fine(v.interval[1]),
        v.disagreement,
        wording::percent_fine(v.lowest),
        wording::percent_fine(v.highest),
        at_even_odds(chance)
    ));
    ui.label(format!(
        "The mixed-document test at {} per section ({} over {} sections) finds {} section{} read as model-written and {} as human, so {}.",
        wording::format_percent(v.chunk_level),
        wording::format_percent(v.level),
        v.chunks,
        v.model_chunks,
        if v.model_chunks == 1 { "" } else { "s" },
        v.human_chunks,
        if v.mixed_document { "the document reads as mixed" } else { "there is no sign of a mixed document" }
    ));
    ui.add_space(8.0);
}

fn chunk_account(ui: &mut egui::Ui, r: &Reading, meta: Option<&ModelMeta>) {
    let mono = |text: String| RichText::new(text).monospace();
    let bins = |bins: &[usize]| meta.map_or_else(|| format!("length bins {bins:?}"), |m| m.bins_words(bins));
    ui.label(RichText::new("Against the reference bank").font(crate::theme::strong(14.0)));
    ui.label(format!(
        "The head scores the text against every reference text. Each origin's mean is the mean probability that the text shares an origin with a reference text of that origin. The text is {} tokens (length bin {}).",
        thousands(r.n_tokens),
        r.length_bin
    ));
    egui::Grid::new("bank_evidence").striped(true).num_columns(5).spacing([16.0, 3.0]).show(ui, |ui| {
        for heading in ["Origin", "Mean", "Share", "Spread", "Spread percentile"] {
            ui.label(RichText::new(heading).font(crate::theme::strong(14.0)));
        }
        ui.end_row();
        for origin in [Origin::Human, Origin::Mixed, Origin::Ai] {
            let o = origin.index();
            views::origin_label(ui, origin);
            ui.label(mono(format!("{:.3}", r.mean[o])));
            ui.label(mono(format!("{:.0}%", 100.0 * r.share[o])));
            ui.label(mono(format!("{:.3}", r.spread[o])));
            ui.label(mono(format!("{:.0}", 100.0 * r.spread_percentile[o])));
            ui.end_row();
        }
    });
    ui.add_space(8.0);
    ui.label(RichText::new("How typical the text is of each origin (conformal p-values)").font(crate::theme::strong(14.0)));
    let likeness = |origin: Origin| match origin {
        Origin::Human => "human-like",
        Origin::Ai => "AI-like",
        Origin::Mixed => "mixed-like",
    };
    match &r.explain {
        Some(explain) => {
            for count in &explain.conformal {
                ui.horizontal_wrapped(|ui| {
                    views::origin_marker(ui, count.origin);
                    ui.label(format!(
                        "Of {} {} calibration texts of {}, {} read at most this {}, so p = {:.3} (the reading uses {:.3}, its grid value).",
                        thousands(count.n),
                        origin_word(count.origin),
                        bins(&count.bins),
                        thousands(count.count),
                        likeness(count.origin),
                        count.p,
                        r.p_values[count.origin.index()]
                    ));
                });
            }
        }
        None => {
            ui.label(format!("human {:.3}, mixed {:.3}, AI {:.3}", r.p_values[0], r.p_values[2], r.p_values[1]));
        }
    }
    for (key, set) in &r.sets {
        let level: f64 = key.parse().unwrap_or(f64::NAN);
        let empty = set.is_empty();
        let called = r.not_human.get(key).copied().unwrap_or(false);
        ui.label(format!(
            "{}{}",
            wording::set_sentence(Some(set), level),
            if empty {
                String::new()
            } else if called {
                " The text is called not human at this level.".to_owned()
            } else {
                " The text is not called not human at this level.".to_owned()
            }
        ));
    }
    ui.add_space(8.0);
    ui.label(RichText::new("Probability").font(crate::theme::strong(14.0)));
    ui.label(format!(
        "Of the calibration texts of this length that score like this one, {} are not human (range {} to {}). That is the Venn-Abers calibration of the score {:.3}, one minus the human share.{}",
        wording::percent_fine(r.probability),
        wording::percent_fine(r.interval[0]),
        wording::percent_fine(r.interval[1]),
        r.score,
        at_even_odds(meta.and_then(|m| m.chance_of(r)).as_ref())
    ));
    ui.label(format!(
        "The three-way probabilities are human {:.2}, mixed {:.2}, and AI {:.2}. They place the text at P(AI) + P(mixed) / 2 = {:.2} between human (0) and AI (1), which colors the verdict's marker and the tint of the text.",
        r.three_way[0], r.three_way[2], r.three_way[1], r.strip.position
    ));
    ui.add_space(8.0);
    ui.label(RichText::new("Closeness to the reference").font(crate::theme::strong(14.0)));
    match &r.explain {
        Some(explain) => {
            let c = &explain.closeness;
            ui.label(format!(
                "The mean of the text's highest cosines to the bank is {:.3}. Of {} calibration texts of {}, {} sat as far from the reference, so p = {:.3}{}.",
                r.closeness,
                thousands(c.n),
                bins(&c.bins),
                thousands(c.count),
                c.p,
                if r.out_of_reference { ", which puts the text out of reference" } else { "" }
            ));
        }
        None => {
            ui.label(format!("Closeness {:.3}, p = {:.3}.", r.closeness, r.closeness_p));
        }
    }
    let Some(explain) = &r.explain else { return };
    ui.add_space(8.0);
    ui.label(RichText::new("Nearest reference texts").font(crate::theme::strong(14.0)));
    ui.label(RichText::new("The bank texts most like this one by cosine, with what the release records of them.").weak());
    egui::ScrollArea::horizontal().id_salt("nearest_scroll").show(ui, |ui| nearest(ui, explain));
}

fn nearest(ui: &mut egui::Ui, explain: &litmus_engine::Explain) {
    let mono = |text: String| RichText::new(text).monospace();
    egui::Grid::new("nearest").striped(true).num_columns(6).spacing([14.0, 3.0]).show(ui, |ui| {
        for heading in ["Bank text", "Origin", "Cosine", "P(same origin)", "Tokens", "Recorded"] {
            ui.label(RichText::new(heading).font(crate::theme::strong(14.0)));
        }
        ui.end_row();
        for entry in &explain.nearest {
            ui.label(mono(format!("#{}", entry.index)));
            views::origin_label(ui, entry.origin);
            ui.label(mono(format!("{:.3}", entry.cosine)));
            ui.label(mono(format!("{:.2}", entry.probability)));
            ui.label(mono(thousands(entry.tokens.max(0) as usize)));
            let recorded: Vec<String> = entry.metadata.iter().filter(|(_, value)| !value.is_empty()).map(|(field, value)| format!("{field} {value}")).collect();
            ui.label(recorded.join(", "));
            ui.end_row();
        }
    });
}
