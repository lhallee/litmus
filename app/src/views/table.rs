//! The tables of Batch and History, in one style: a header whose titles sort the rows (a second click reverses),
//! rows 36 points tall with a quiet stripe and a hover fill, numbers right-aligned in the mono face, and a click (or
//! Enter on a row in focus) that opens the row's reading.

use eframe::egui::{self, Align, CornerRadius, Layout, Pos2, Rect, Response, Sense, Stroke, UiBuilder, Vec2, Role, WidgetInfo};

use crate::sorting::Sort;
use crate::theme;
use crate::views::controls;

pub const ROW: f32 = 36.0;
const PAD: f32 = 12.0;

/// A column: its title (also its sort key), its width (0 takes the room the others leave), and whether it holds
/// numbers (right-aligned, sorted high to low first).
#[derive(Clone, Copy)]
pub struct Column {
    pub title: &'static str,
    pub width: f32,
    pub numeric: bool,
    pub sortable: bool,
}

impl Column {
    pub const fn text(title: &'static str, width: f32) -> Self {
        Self { title, width, numeric: false, sortable: true }
    }

    pub const fn number(title: &'static str, width: f32) -> Self {
        Self { title, width, numeric: true, sortable: true }
    }

    pub const fn plain(title: &'static str, width: f32) -> Self {
        Self { title, width, numeric: false, sortable: false }
    }
}

/// The columns' widths across `available`, the flexible one at least 140.
pub fn widths(columns: &[Column], available: f32) -> Vec<f32> {
    let fixed: f32 = columns.iter().map(|c| c.width).sum();
    let flexible = columns.iter().filter(|c| c.width == 0.0).count().max(1) as f32;
    let rest = ((available - fixed) / flexible).max(140.0);
    columns.iter().map(|c| if c.width == 0.0 { rest } else { c.width }).collect()
}

fn cells(rect: Rect, widths: &[f32]) -> Vec<Rect> {
    let mut x = rect.left();
    widths
        .iter()
        .map(|w| {
            let cell = Rect::from_x_y_ranges((x + PAD)..=(x + w - PAD).max(x + PAD), rect.y_range());
            x += w;
            cell
        })
        .collect()
}

/// The header; a click on a sortable title changes `sort`.
pub fn header(ui: &mut egui::Ui, columns: &[Column], widths: &[f32], sort: &mut Option<Sort<&'static str>>) {
    let p = controls::palette(ui);
    let total: f32 = widths.iter().sum();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(total, 32.0), Sense::hover());
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, p.rule));
    for (column, cell) in columns.iter().zip(cells(rect, widths)) {
        let current = sort.filter(|s| s.key == column.title);
        let ink = if current.is_some() { p.ink } else { p.muted };
        let galley = ui.painter().layout_no_wrap(column.title.to_owned(), theme::strong(13.0), ink);
        let arrow = 10.0;
        let width = galley.size().x + if current.is_some() { arrow + 4.0 } else { 0.0 };
        let left = if column.numeric { cell.right() - width } else { cell.left() };
        let label = Rect::from_min_size(Pos2::new(left, cell.center().y - galley.size().y / 2.0), Vec2::new(width, galley.size().y));
        if column.sortable {
            let response = ui.interact(label.expand2(Vec2::new(6.0, 6.0)), ui.scope_id().with(("sort", column.title)), Sense::click());
            response.widget_info(|| WidgetInfo::labeled(Role::Button, true, format!("Sort by {}", column.title)));
            if response.hovered() {
                ui.painter().rect_filled(response.rect, CornerRadius::same(theme::RADIUS), p.hover);
            }
            controls::focus_ring(ui, &response);
            if response.clicked() {
                *sort = Some(Sort::clicked(*sort, column.title, column.numeric));
            }
            response.on_hover_text(format!("Sort by {}", column.title.to_lowercase()));
        }
        ui.painter().galley(label.min, galley.clone(), ink);
        if let Some(current) = current {
            let x = label.min.x + galley.size().x + 4.0 + arrow / 2.0;
            let y = label.center().y;
            let h = arrow * 0.35;
            let points = if current.descending {
                vec![Pos2::new(x - h, y - h * 0.6), Pos2::new(x + h, y - h * 0.6), Pos2::new(x, y + h * 0.8)]
            } else {
                vec![Pos2::new(x - h, y + h * 0.6), Pos2::new(x + h, y + h * 0.6), Pos2::new(x, y - h * 0.8)]
            };
            ui.painter().add(egui::Shape::convex_polygon(points, ink, Stroke::NONE));
        }
    }
}

/// One row: `cell` draws column `i` into its own area. The row's response clicks when the row is clicked outside
/// a control of its own (such as a Delete button).
pub fn row(ui: &mut egui::Ui, id: egui::Id, widths: &[f32], stripe: bool, label: &str, mut cell: impl FnMut(&mut egui::Ui, usize)) -> Response {
    let p = controls::palette(ui);
    let total: f32 = widths.iter().sum();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(total, ROW), Sense::hover());
    let response = ui.interact(rect, id, Sense::click());
    response.widget_info(|| WidgetInfo::labeled(Role::Button, true, label));
    let fill = if response.hovered() {
        Some(p.hover)
    } else if stripe {
        Some(ui.visuals().faint_bg_color)
    } else {
        None
    };
    if let Some(fill) = fill {
        ui.painter().rect_filled(rect, CornerRadius::same(theme::RADIUS), fill);
    }
    if response.has_focus() {
        ui.painter().rect_stroke(rect.shrink(1.0), CornerRadius::same(theme::RADIUS), Stroke::new(2.0, p.focus), egui::StrokeKind::Inside);
    }
    for (index, area) in cells(rect, widths).into_iter().enumerate() {
        let mut child = ui.new_child(UiBuilder::new().max_rect(area).layout(Layout::left_to_right(Align::Center)));
        child.set_clip_rect(area.intersect(ui.clip_rect()));
        cell(&mut child, index);
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A verdict in a row: a small marker shaped by the call (a disc for a call, one origin or "AI or mixed", a ring for
/// uncertain, a question mark for unlike reference) and the headline, a call in its ink and an uncertain one's
/// origins muted after the word.
pub fn verdict(ui: &mut egui::Ui, headline: &crate::wording::Headline, position: f64) {
    use crate::wording::Call;
    let p = controls::palette(ui);
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(14.0), Sense::hover());
    let c = rect.center();
    let fill = theme::rgb(crate::brand::litmus(position));
    let ink = p.ink;
    match headline.call() {
        Call::UnlikeReference => {
            ui.painter().circle_stroke(c, 6.0, Stroke::new(1.2, ink));
            ui.painter().text(c, egui::Align2::CENTER_CENTER, "?", theme::strong(10.0), ink);
        }
        Call::Uncertain => {
            ui.painter().circle_stroke(c, 4.5, Stroke::new(2.5, fill));
            ui.painter().circle_stroke(c, 6.5, Stroke::new(1.0, ink));
        }
        Call::One(_) | Call::AiOrMixed => {
            ui.painter().circle(c, 5.5, fill, Stroke::new(1.0, ink));
        }
    }
    let mut job = egui::text::LayoutJob::default();
    let font = egui::FontId::proportional(15.0);
    job.append(&headline.word, 0.0, egui::TextFormat { font_id: font.clone(), color: crate::views::headline_ink(headline, &p), ..Default::default() });
    if !headline.among.is_empty() {
        job.append(&headline.among, 6.0, egui::TextFormat { font_id: font, color: p.muted, ..Default::default() });
    }
    ui.add(egui::Label::new(job).truncate());
}

/// A number in the mono face, right-aligned in its cell.
pub fn number(ui: &mut egui::Ui, text: String) {
    let ink = ui.visuals().text_color();
    let galley = ui.painter().layout_no_wrap(text, egui::FontId::monospace(14.0), ink);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), galley.size().y), Sense::hover());
    ui.painter().galley(Pos2::new(rect.right() - galley.size().x, rect.top()), galley, ink);
}

/// Something narrower than its cell (a small strip), right-aligned like the numbers.
pub fn right(ui: &mut egui::Ui, width: f32) {
    ui.add_space((ui.available_width() - width).max(0.0));
}

/// A cell's text, cut to its width.
pub fn text(ui: &mut egui::Ui, text: impl Into<egui::WidgetText>) {
    ui.add(egui::Label::new(text).truncate());
}
