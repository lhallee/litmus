//! The History screen: past readings, newest first, searchable and sortable; a row opens its reading, and each can
//! be deleted. The simple level shows the first words, the verdict and a small strip; the Full level adds when, the
//! numbers and the model.

use eframe::egui::{self, RichText};
use litmus_files::export;

use crate::app::LitmusApp;
use crate::history::Entry;
use crate::settings::Detail;
use crate::sorting::{self, Value};
use crate::views::table::{self, Column};
use crate::views::controls;
use crate::{clock, strip, theme, wording};

const TEXT: &str = "Text";
const VERDICT: &str = "Verdict";
const STRIP: &str = "Strip";
const WHEN: &str = "When (UTC)";
const PROBABILITY: &str = wording::CHANCE;
const TOKENS: &str = "Tokens";
const MODEL: &str = "Model";
const DELETE: &str = "";

fn columns(detail: Detail) -> Vec<Column> {
    let mut columns = vec![Column::text(TEXT, 0.0), Column::text(VERDICT, 250.0), Column::number(STRIP, 150.0)];
    if detail.full() {
        columns.extend([Column::number(WHEN, 164.0), Column::number(PROBABILITY, 150.0), Column::number(TOKENS, 90.0), Column::text(MODEL, 130.0)]);
    }
    columns.push(Column::plain(DELETE, 84.0));
    columns
}

fn value(app: &LitmusApp, entry: &Entry, key: &str) -> Value {
    match key {
        TEXT => Value::Text(entry.title.clone()),
        VERDICT => Value::Number(entry.position),
        WHEN => Value::Number(entry.at_ms as f64),
        // The strip marks the chance of AI, so both columns sort by it.
        STRIP | PROBABILITY => app.entry_chance(entry).map_or(Value::Missing, |chance| Value::Number(chance.value)),
        TOKENS => Value::Number(entry.tokens as f64),
        MODEL => Value::Text(entry.model.clone()),
        _ => Value::Missing,
    }
}

pub fn show(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let p = controls::palette(ui);
    egui::Frame::new().inner_margin(egui::Margin::symmetric(24, 20)).show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(crate::views::plural(app.history.entries().len(), "reading")).font(theme::strong(17.0)));
            ui.label(RichText::new(if app.history.is_persistent() { "kept on this computer" } else { "kept until this page closes" }).color(p.muted));
            ui.add_space(12.0);
            ui.add(egui::TextEdit::singleline(&mut app.history_filter).hint_text("Search").desired_width(220.0).margin(egui::Margin::symmetric(10, 7)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if app.confirm_clear {
                    if ui.button(RichText::new(format!("Delete all {}", app.history.entries().len())).color(p.error)).clicked() {
                        if let Err(error) = app.history.clear() {
                            app.notice = Some(crate::app::Notice::error("The history was not cleared", error));
                        }
                        app.confirm_clear = false;
                    }
                    if controls::quiet(ui, "Keep them", None, true).clicked() {
                        app.confirm_clear = false;
                    }
                    ui.label("Delete every past reading? This cannot be undone.");
                } else if controls::quiet(ui, "Clear history", None, !app.history.entries().is_empty()).clicked() {
                    app.confirm_clear = true;
                }
            });
        });
        ui.add_space(12.0);
        let filter = app.history_filter.to_lowercase();
        let entries: Vec<&Entry> = app
            .history
            .entries()
            .iter()
            .rev()
            .filter(|e| filter.is_empty() || [&e.title, &e.source, &e.headline].iter().any(|field| field.to_lowercase().contains(&filter)))
            .collect();
        if entries.is_empty() {
            ui.add_space(48.0);
            ui.vertical_centered(|ui| {
                if app.history.entries().is_empty() {
                    ui.label(RichText::new("No readings yet").font(theme::strong(22.0)));
                    ui.label(RichText::new("Every text you check is listed here, so you can open its reading again.").size(16.0).color(p.muted));
                } else {
                    ui.label(RichText::new("Nothing matches the search.").size(16.0).color(p.muted));
                }
            });
            return;
        }
        let columns = columns(app.settings.detail);
        let widths = table::widths(&columns, ui.available_width());
        let mut sort = app.history_sort.filter(|s| columns.iter().any(|c| c.title == s.key));
        table::header(ui, &columns, &widths, &mut sort);
        app.history_sort = sort;
        let order: Vec<usize> = match sort {
            Some(sort) => sorting::order(&entries.iter().map(|e| value(app, e, sort.key)).collect::<Vec<_>>(), sort.descending),
            None => (0..entries.len()).collect(),
        };
        let ids: Vec<u64> = order.iter().map(|&i| entries[i].id).collect();
        let mut open = None;
        let mut delete = None;
        egui::ScrollArea::vertical().id_salt("history_rows").auto_shrink([false, false]).show_rows(ui, table::ROW, ids.len(), |ui, range| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for (position, &id) in ids[range.clone()].iter().enumerate() {
                let Some(entry) = app.history.entries().iter().find(|e| e.id == id) else { continue };
                let response = table::row(ui, ui.scope_id().with(("history_row", id)), &widths, (range.start + position) % 2 == 1, &format!("Open the reading of {}", entry.title), |ui, column| match columns[column].title {
                    TEXT => table::text(ui, entry.title.clone()),
                    VERDICT => table::verdict(ui, &wording::Headline::from_line(&entry.headline), entry.position),
                    STRIP => {
                        strip::mini(ui, &strip::Drawn::of(app.entry_chance(entry).as_ref(), entry.unlike_reference), 126.0, 20.0);
                    }
                    WHEN => table::number(ui, clock::format_utc(entry.at_ms).trim_end_matches(" UTC").to_owned()),
                    PROBABILITY => table::number(ui, app.entry_chance(entry).map_or_else(String::new, |chance| chance.words())),
                    TOKENS => table::number(ui, export::thousands(entry.tokens)),
                    MODEL => table::text(ui, RichText::new(&entry.model).monospace().color(p.muted)),
                    _ => {
                        if controls::quiet(ui, "Delete", None, true).on_hover_text("Delete this reading").clicked() {
                            delete = Some(id);
                        }
                    }
                });
                if response.on_hover_text(&entry.source).clicked() {
                    open = Some(id);
                }
            }
        });
        if let Some(id) = delete {
            if let Err(error) = app.history.remove(id) {
                app.notice = Some(crate::app::Notice::error("That reading was not deleted", error));
            }
        } else if let Some(id) = open {
            app.open_history(id);
        }
    });
}
