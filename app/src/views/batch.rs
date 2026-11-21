//! The Batch screen (desktop): files and folders in a sortable table, scored one after another with progress and
//! cancel, exported as CSV or JSON. A row opens its reading. The simple level shows the file, the verdict and a
//! small strip, so a list can be scanned by color and position; the Full level adds the numbers.

use eframe::egui::{self, RichText};
use litmus_files::counts::TextCounts;
use litmus_files::export;

use crate::app::{BatchState, LitmusApp, Shown};
use crate::settings::Detail;
use crate::sorting::{self, Value};
use crate::views::table::{self, Column};
use crate::views::{self, controls};
use crate::{strip, theme, wording};

const FILE: &str = "File";
const VERDICT: &str = "Verdict";
const STRIP: &str = "Strip";
const PROBABILITY: &str = wording::CHANCE;
const TOKENS: &str = "Tokens";
const TIME: &str = "Time";
const HASH: &str = "SHA-256";

fn columns(detail: Detail) -> Vec<Column> {
    let mut columns = vec![Column::text(FILE, 0.0), Column::text(VERDICT, 250.0), Column::number(STRIP, 150.0)];
    if detail.full() {
        columns.extend([Column::number(PROBABILITY, 150.0), Column::number(TOKENS, 90.0), Column::number(TIME, 90.0), Column::text(HASH, 130.0)]);
    }
    columns
}

/// A row's value in a column, for sorting.
fn value(app: &LitmusApp, state: &BatchState, path: &std::path::Path, key: &str) -> Value {
    if key == FILE {
        return Value::Text(crate::app::file_name(&path.display().to_string()));
    }
    let BatchState::Done(scored) = state else { return Value::Missing };
    let v = &scored.reading.verdict;
    match key {
        VERDICT => Value::Number(v.strip.position),
        // The strip marks the chance of AI, so both columns sort by it.
        STRIP | PROBABILITY => app.chance(scored).map_or(Value::Missing, |chance| Value::Number(chance.value)),
        TOKENS => Value::Number(scored.reading.n_tokens as f64),
        TIME => scored.seconds.map_or(Value::Number(0.0), Value::Number),
        HASH => Value::Text(scored.hash.clone()),
        _ => Value::Missing,
    }
}

pub fn show(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let p = controls::palette(ui);
    egui::Frame::new().inner_margin(egui::Margin::symmetric(24, 20)).show(ui, |ui| {
        toolbar(app, ui);
        if let Some((_, first)) = app.batch.job {
            let total = app.batch.rows.len() - first;
            let done = app.batch.finished;
            ui.add_space(8.0);
            ui.label(RichText::new(format!("{done} of {total} files read{}", if app.batch.progress.words.is_empty() { String::new() } else { format!(". {}", app.batch.progress.words) })).color(p.muted));
            ui.add(egui::ProgressBar::new(done as f32 / total.max(1) as f32).desired_height(6.0));
        }
        if app.batch.rows.is_empty() {
            ui.add_space(48.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Check many files at once").font(theme::strong(22.0)));
                ui.label(RichText::new("Add files or a folder, or drop them on the window. Text, Markdown, HTML, Word, and PDF files are read, and folders are walked.").size(16.0).color(p.muted));
                ui.label(RichText::new("An unchanged file is not read twice, because readings are kept by each file's contents.").color(p.muted));
            });
            return;
        }
        ui.add_space(12.0);
        rows(app, ui);
    });
}

fn toolbar(app: &mut LitmusApp, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        let idle = !app.busy();
        let waiting = app.batch.rows.iter().filter(|r| matches!(r.state, BatchState::Waiting)).count();
        let label = if waiting > 0 { format!("Check {waiting} file{}", if waiting == 1 { "" } else { "s" }) } else { "Check".to_owned() };
        if controls::primary(ui, &label, None, idle && waiting > 0 && app.meta().is_some()).clicked() {
            app.run_batch();
        }
        if app.batch.job.is_some() && ui.button("Cancel").on_hover_text("Esc").clicked() {
            app.cancel();
        }
        if controls::quiet(ui, "Add files", None, idle).clicked()
            && let Some(paths) = rfd::FileDialog::new().add_filter(views::FILE_FILTER, &["txt", "text", "md", "markdown", "html", "htm", "docx", "pdf"]).pick_files()
        {
            app.add_to_batch(&paths);
        }
        if controls::quiet(ui, "Add folder", None, idle).on_hover_text("Every readable file inside, recursively").clicked()
            && let Some(folder) = rfd::FileDialog::new().pick_folder()
        {
            app.add_to_batch(&[folder]);
        }
        if !app.batch.rows.is_empty() && controls::quiet(ui, "Clear", None, idle).clicked() {
            app.batch = Default::default();
        }
        let any = app.batch.rows.iter().any(|r| !matches!(r.state, BatchState::Waiting));
        if any {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let ctx = ui.ctx().clone();
                if controls::quiet(ui, "Export JSON", None, idle).on_hover_text("Every field of every reading, as one JSON array").clicked() {
                    let contents = batch_json(app);
                    views::deliver(app, &ctx, "litmus_batch.json", ("JSON", "json"), contents);
                }
                if controls::quiet(ui, "Export CSV", None, idle).on_hover_text("Every field, the command line's columns").clicked() {
                    let contents = batch_csv(app);
                    views::deliver(app, &ctx, "litmus_batch.csv", ("CSV", "csv"), contents);
                }
            });
        }
    });
}

fn rows(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let p = controls::palette(ui);
    let columns = columns(app.settings.detail);
    let widths = table::widths(&columns, ui.available_width());
    let mut sort = app.batch_sort.filter(|s| columns.iter().any(|c| c.title == s.key));
    table::header(ui, &columns, &widths, &mut sort);
    app.batch_sort = sort;
    let order: Vec<usize> = match sort {
        Some(sort) => {
            let values: Vec<Value> = app.batch.rows.iter().map(|row| value(app, &row.state, &row.path, sort.key)).collect();
            sorting::order(&values, sort.descending)
        }
        None => (0..app.batch.rows.len()).collect(),
    };
    let mut open = None;
    egui::ScrollArea::vertical().id_salt("batch_rows").auto_shrink([false, false]).show_rows(ui, table::ROW, order.len(), |ui, range| {
        ui.spacing_mut().item_spacing.y = 0.0;
        for (position, &index) in order[range.clone()].iter().enumerate() {
            let row = &app.batch.rows[index];
            let name = crate::app::file_name(&row.path.display().to_string());
            let response = table::row(ui, ui.scope_id().with(("batch_row", index)), &widths, (range.start + position) % 2 == 1, &format!("Open the reading of {name}"), |ui, column| {
                match (columns[column].title, &row.state) {
                    (FILE, _) => {
                        table::text(ui, name.clone());
                    }
                    (VERDICT, BatchState::Waiting) => {
                        ui.label(RichText::new("waiting").color(p.muted));
                    }
                    (VERDICT, BatchState::Failed(error)) => {
                        let words = if error == crate::scoring::EMPTY { "No words to read".to_owned() } else { format!("Not read ({})", error.trim_end_matches('.')) };
                        ui.add(egui::Label::new(RichText::new(words).color(p.error)).truncate()).on_hover_text(error);
                    }
                    (VERDICT, BatchState::Done(scored)) => table::verdict(ui, &wording::verdict_headline(&scored.reading.verdict), scored.reading.verdict.strip.position),
                    (STRIP, BatchState::Done(scored)) => {
                        strip::mini(ui, &strip::Drawn::of(app.chance(scored).as_ref(), scored.reading.verdict.strip.unlike_reference), 126.0, 20.0);
                    }
                    (PROBABILITY, BatchState::Done(scored)) => table::number(ui, app.chance(scored).map_or_else(String::new, |chance| chance.words())),
                    (TOKENS, BatchState::Done(scored)) => table::number(ui, export::thousands(scored.reading.n_tokens)),
                    (TIME, BatchState::Done(scored)) => table::number(ui, scored.seconds.map_or("cached".to_owned(), wording::duration)),
                    (HASH, BatchState::Done(scored)) => {
                        ui.label(RichText::new(scored.hash.chars().take(12).collect::<String>()).monospace().color(p.muted)).on_hover_text(&scored.hash);
                    }
                    _ => {}
                }
            });
            let hover = row.path.display().to_string();
            if response.on_hover_text(hover).clicked() && matches!(row.state, BatchState::Done(_)) {
                open = Some(index);
            }
        }
    });
    if let Some(index) = open
        && let BatchState::Done(scored) = &app.batch.rows[index].state
    {
        let path = app.batch.rows[index].path.display().to_string();
        let scored = scored.clone();
        app.input = scored.text.as_str().to_owned();
        app.input_source = path.clone();
        let model = app.meta().map(|m| m.version.clone()).unwrap_or_default();
        app.open(Shown::new(crate::app::file_name(&path), path, scored, crate::clock::now_ms(), model));
    }
}

/// A finished row for export: its name, hash, whether it came from the cache, its reading or error, its counts, and
/// its chance of AI.
type OutRow<'a> = (String, String, bool, Result<&'a litmus_engine::DocumentReading, &'a str>, Option<TextCounts>, Option<litmus_engine::Chance>);

fn rows_out(app: &LitmusApp) -> Vec<OutRow<'_>> {
    app.batch
        .rows
        .iter()
        .filter_map(|row| match &row.state {
            BatchState::Waiting => None,
            BatchState::Done(scored) => {
                let counts = TextCounts::of(&scored.text, scored.reading.n_tokens);
                Some((row.path.display().to_string(), scored.hash.clone(), scored.seconds.is_none(), Ok(scored.reading.as_ref()), Some(counts), app.chance(scored)))
            }
            BatchState::Failed(error) => Some((row.path.display().to_string(), String::new(), false, Err(error.as_str()), None, None)),
        })
        .collect()
}

fn export_row<'a>((name, hash, cached, outcome, counts, chance): &'a OutRow<'a>) -> export::Row<'a> {
    export::Row { name, hash, cached: *cached, outcome: *outcome, counts: counts.as_ref(), chance: *chance }
}

/// The batch's finished rows in the command line's CSV columns.
pub fn batch_csv(app: &LitmusApp) -> String {
    let rows = rows_out(app);
    export::csv(&rows.iter().map(export_row).collect::<Vec<_>>())
}

/// The batch's finished rows as one JSON array of the command line's objects.
pub fn batch_json(app: &LitmusApp) -> String {
    let rows = rows_out(app);
    let values: Vec<serde_json::Value> = rows.iter().map(|row| export::json_value(&export_row(row))).collect();
    serde_json::to_string_pretty(&values).expect("readings serialize")
}
