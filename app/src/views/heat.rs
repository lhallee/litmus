//! The heat map: the user's own text with each 128-token cell tinted by its place on the litmus scale, a slim
//! minimap of a long document over it, and the reading of a clicked part of the text. Lightness carries the scale
//! as well as hue (lighter is more human-like); cells whose windows are unlike the reference are underlined.

use std::sync::Arc;

use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use eframe::epaint::text::{LayoutJob, TextFormat};
use litmus_engine::{Cell, DocumentReading, STRIP_LEVEL};
use litmus_files::export;

use crate::app::LitmusApp;
use crate::heatmap::{self, Page};
use crate::views::{controls, dark};
use crate::{strip, theme, wording};

/// Below this many cells a document is short enough to need no minimap.
pub const MINIMAP_CELLS: usize = 8;

/// Page heights at a wrap width, kept until the width, the reading or the selection changes.
#[derive(Default)]
pub struct Layout {
    key: Option<(u32, usize, Option<usize>, bool)>,
    pages: Vec<Page>,
    heights: Vec<f32>,
    /// A cell to bring into view on the next frame.
    pub scroll_to_cell: Option<usize>,
}

/// The colors a page is drawn in.
#[derive(Clone, Copy)]
struct Look {
    dark: bool,
    ink: Color32,
    muted: Color32,
}

fn job(text: &str, cells: &[Cell], page: &Page, width: f32, selected: Option<usize>, look: Look) -> LayoutJob {
    let Look { dark, ink, muted } = look;
    let mut job = LayoutJob { text: text[page.bytes.clone()].to_owned(), ..Default::default() };
    job.wrap.max_width = width;
    let font = FontId::proportional(16.0);
    for (range, index) in heatmap::sections(cells, page) {
        let cell = &cells[index];
        let underline = if selected == Some(index) {
            Stroke::new(2.0, ink)
        } else if cell.unlike_reference {
            Stroke::new(1.0, muted)
        } else {
            Stroke::NONE
        };
        let format = TextFormat { font_id: font.clone(), color: ink, background: theme::cell_tint(cell.position, dark), underline, line_height: Some(26.0), ..Default::default() };
        job.sections.push(eframe::epaint::text::LayoutSection { leading_space: 0.0, byte_range: eframe::epaint::text::ByteIndex(range.start)..eframe::epaint::text::ByteIndex(range.end), format });
    }
    job
}

/// The tinted text in a scroll area `height` tall; a click on a cell selects it when `clickable`.
pub fn text(app: &mut LitmusApp, ui: &mut egui::Ui, height: f32, clickable: bool) {
    let Some(shown) = &app.shown else { return };
    let reading: Arc<DocumentReading> = shown.scored.reading.clone();
    let text = shown.scored.text.clone();
    let cells = &reading.cells;
    if cells.is_empty() {
        ui.label("This reading has no heat map.");
        return;
    }
    let dark = dark(ui);
    let look = Look { dark, ink: ui.visuals().text_color(), muted: ui.visuals().weak_text_color() };
    let width = (ui.available_width() - 12.0).max(120.0);
    let selected = app.selected_cell.filter(|_| clickable);
    let key = (width.round() as u32, Arc::as_ptr(&reading) as usize, selected, dark);
    if app.heat.key != Some(key) {
        let pages = heatmap::pages(cells, heatmap::CELLS_PER_PAGE);
        let heights = pages.iter().map(|page| ui.fonts_mut(|fonts| fonts.layout_job(job(&text, cells, page, width, selected, look)).size().y)).collect();
        app.heat.key = Some(key);
        app.heat.pages = pages;
        app.heat.heights = heights;
    }
    let total: f32 = app.heat.heights.iter().sum();
    let mut scroll = egui::ScrollArea::vertical().id_salt("heat_text").auto_shrink([false, true]).max_height(height);
    if let Some(cell) = app.heat.scroll_to_cell.take() {
        let page = cell / heatmap::CELLS_PER_PAGE;
        let y: f32 = app.heat.heights.iter().take(page).sum();
        scroll = scroll.vertical_scroll_offset((y - 8.0).max(0.0));
    }
    let mut clicked = None;
    scroll.show_viewport(ui, |ui, viewport| {
        ui.set_min_height(total);
        ui.set_min_width(width);
        let origin = ui.min_rect().min;
        let mut y = 0.0;
        for (index, page) in app.heat.pages.iter().enumerate() {
            let height = app.heat.heights[index];
            if y + height >= viewport.min.y && y <= viewport.max.y {
                let galley = ui.fonts_mut(|fonts| fonts.layout_job(job(&text, cells, page, width, selected, look)));
                let rect = Rect::from_min_size(origin + Vec2::new(0.0, y), galley.size());
                if clickable {
                    let response = ui.interact(rect, ui.scope_id().with(("heat_page", index)), Sense::click());
                    if response.clicked()
                        && let Some(pointer) = response.interact_pointer_pos()
                    {
                        let cursor = galley.cursor_from_pos(pointer - rect.min);
                        let byte = page.bytes.start + heatmap::byte_of_char(&text[page.bytes.clone()], cursor.index.0);
                        clicked = heatmap::cell_at_byte(cells, byte);
                    }
                    if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                }
                ui.painter().galley(rect.min, galley, look.ink);
            }
            y += height;
        }
    });
    if let Some(index) = clicked {
        app.selected_cell = if app.selected_cell == Some(index) { None } else { Some(index) };
    }
}

/// Whether a reading is long enough for the minimap.
pub fn wants_minimap(app: &LitmusApp) -> bool {
    app.shown.as_ref().is_some_and(|s| s.scored.reading.cells.len() >= MINIMAP_CELLS)
}

/// The whole document in one slim bar, a block per cell in proportion to its tokens; a click selects the cell and
/// scrolls the text to it.
pub fn minimap(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let Some(shown) = &app.shown else { return };
    let reading = shown.scored.reading.clone();
    let cells = &reading.cells;
    if cells.is_empty() {
        return;
    }
    let dark = dark(ui);
    let p = theme::palette(dark);
    let height = 12.0;
    let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::click());
    let painter = ui.painter_at(rect.expand(3.0));
    let total = cells.last().map_or(1, |c| c.place.end).max(1) as f32;
    let x_of = |token: usize| rect.left() + rect.width() * token as f32 / total;
    for (index, cell) in cells.iter().enumerate() {
        let block = Rect::from_x_y_ranges(x_of(cell.place.start)..=x_of(cell.place.end), rect.y_range());
        painter.rect_filled(block, 0.0, theme::rgb(crate::brand::litmus(cell.position)).gamma_multiply(if dark { 0.85 } else { 1.0 }));
        if cell.unlike_reference {
            painter.line_segment([Pos2::new(block.left(), block.bottom() + 2.0), Pos2::new(block.right(), block.bottom() + 2.0)], Stroke::new(1.5, p.muted));
        }
        if app.selected_cell == Some(index) {
            painter.rect_stroke(block.expand(2.0), CornerRadius::same(2), Stroke::new(2.0, p.ink), egui::StrokeKind::Outside);
        }
    }
    if let Some(pointer) = response.hover_pos() {
        let token = ((pointer.x - rect.left()) / rect.width() * total).max(0.0) as usize;
        let index = cells.partition_point(|c| c.place.end <= token).min(cells.len() - 1);
        let cell = &cells[index];
        response.clone().on_hover_text(format!("Part {} of {}: characters {} to {}", index + 1, cells.len(), export::thousands(cell.place.start_char), export::thousands(cell.place.end_char)));
        if response.clicked() {
            app.selected_cell = Some(index);
            app.heat.scroll_to_cell = Some(index);
        }
    }
}

/// The scale of the tints, in words: lighter leans human, deeper leans AI.
pub fn legend(ui: &mut egui::Ui) {
    let dark = dark(ui);
    let p = theme::palette(dark);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(RichText::new("Leans human").small().color(p.human_ink));
        let (rect, _) = ui.allocate_exact_size(Vec2::new(96.0, 12.0), Sense::hover());
        let steps = 8;
        for i in 0..steps {
            let t = i as f64 / (steps - 1) as f64;
            let w = rect.width() / steps as f32;
            let block = Rect::from_min_size(Pos2::new(rect.left() + w * i as f32, rect.top()), Vec2::new(w, rect.height()));
            ui.painter().rect_filled(block, 0.0, theme::cell_tint(t, dark));
        }
        ui.painter().rect_stroke(rect, CornerRadius::same(2), Stroke::new(1.0, p.rule), egui::StrokeKind::Outside);
        ui.label(RichText::new("leans AI").small().color(p.ai_ink));
    });
}

/// The reading of the clicked part of the text, or how to get one.
pub fn cell_detail(app: &mut LitmusApp, ui: &mut egui::Ui) {
    let Some(shown) = &app.shown else { return };
    let reading = shown.scored.reading.clone();
    let p = theme::palette(dark(ui));
    controls::section(ui, "This part of the text");
    let Some(index) = app.selected_cell.filter(|&i| i < reading.cells.len()) else {
        ui.label(RichText::new("Click any part of the text to see how it reads on its own.").color(p.muted));
        return;
    };
    let cell = &reading.cells[index];
    let covering = heatmap::covering_windows(&reading.windows, cell);
    let Some(&nearest) = covering.first() else { return };
    let window = &reading.windows[nearest];
    let r = &window.reading;
    controls::card(ui, 12, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new(wording::reading_headline(r)).font(theme::strong(17.0)));
        strip::mini(ui, &r.strip, ui.available_width().min(260.0), 18.0);
        egui::Grid::new("cell_numbers").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
            controls::fact(ui, "Part", format!("{} of {}", index + 1, reading.cells.len()), None);
            controls::fact(ui, "Characters", format!("{} to {}", export::thousands(cell.place.start_char), export::thousands(cell.place.end_char)), None);
            controls::fact(ui, "P(not human)", wording::probability(r.probability, r.interval), None);
        });
        ui.label(
            RichText::new(format!(
                "Read as the {}-token window centered nearest this part{}. A window describes where to look; the verdict above decides.",
                export::thousands(r.n_tokens),
                if cell.unlike_reference { ", which sits unlike the reference" } else { "" }
            ))
            .small()
            .color(p.muted),
        );
        if app.settings.detail.at_least(crate::settings::Detail::Full) {
            ui.label(
                RichText::new(format!(
                    "Cell tokens {} to {}: position {:.2} (spread {:.2}) from {} window{}; widest band {:.2} to {:.2}. Window tokens {} to {}: p-values human {:.3}, mixed {:.3}, AI {:.3}; not ruled out at 10% {}.",
                    export::thousands(cell.place.start),
                    export::thousands(cell.place.end),
                    cell.position,
                    cell.spread,
                    cell.windows,
                    if cell.windows == 1 { "" } else { "s" },
                    cell.band_lo,
                    cell.band_hi,
                    export::thousands(window.place.start),
                    export::thousands(window.place.end),
                    r.p_values[0],
                    r.p_values[2],
                    r.p_values[1],
                    export::set_words(r.set(STRIP_LEVEL))
                ))
                .monospace()
                .small(),
            );
            if covering.len() > 1 {
                ui.collapsing(format!("The other {} windows over this part", covering.len() - 1), |ui| {
                    for &other in &covering[1..] {
                        let w = &reading.windows[other];
                        ui.horizontal(|ui| {
                            strip::mini(ui, &w.reading.strip, 90.0, 14.0);
                            ui.label(
                                RichText::new(format!(
                                    "tokens {} to {}: {}, {}",
                                    export::thousands(w.place.start),
                                    export::thousands(w.place.end),
                                    wording::reading_headline(&w.reading),
                                    wording::probability(w.reading.probability, w.reading.interval)
                                ))
                                .small(),
                            );
                        });
                    }
                });
            }
        }
    });
}
