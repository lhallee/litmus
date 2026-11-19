//! The app's controls, in one style: a primary button that reads as primary (Check), quiet buttons for secondary
//! actions, a segmented switch (the detail level), tabs for the three screens, an icon button for the theme, cards,
//! and a visible focus ring on every one of them. Each reports itself to AccessKit by role and label.

use eframe::egui::{self, Align2, Color32, CornerRadius, Margin, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Vec2, Role, WidgetInfo};

use crate::theme::{self, Palette};
use crate::views::dark;

pub fn palette(ui: &egui::Ui) -> Palette {
    theme::palette(dark(ui))
}

fn radius() -> CornerRadius {
    CornerRadius::same(theme::RADIUS)
}

/// The focus ring: two points outside the control, in the mixed purple.
pub fn focus_ring(ui: &egui::Ui, response: &Response) {
    if response.has_focus() {
        ui.painter().rect_stroke(response.rect.expand(2.0), CornerRadius::same(theme::RADIUS + 2), Stroke::new(2.0, palette(ui).focus), StrokeKind::Outside);
    }
}

/// A button laid out from its label (and a quieter hint after it, such as a shortcut), painted by `paint`.
fn button_shell(ui: &mut egui::Ui, label: &str, hint: Option<&str>, enabled: bool, font: egui::FontId, min_width: f32) -> (Response, std::sync::Arc<egui::Galley>, Option<std::sync::Arc<egui::Galley>>) {
    let padding = ui.spacing().button_padding;
    let galley = ui.painter().layout_no_wrap(label.to_owned(), font, Color32::PLACEHOLDER);
    let hint = hint.map(|hint| ui.painter().layout_no_wrap(hint.to_owned(), egui::FontId::proportional(12.5), Color32::PLACEHOLDER));
    let hint_width = hint.as_ref().map_or(0.0, |h| h.size().x + 12.0);
    let height = ui.spacing().interact_size.y.max(galley.size().y + 2.0 * padding.y);
    let size = Vec2::new((galley.size().x + hint_width + 2.0 * padding.x).max(min_width), height);
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (_, response) = ui.allocate_exact_size(size, sense);
    response.widget_info(|| WidgetInfo::labeled(Role::Button, enabled, label));
    (response, galley, hint)
}

fn paint_label(ui: &egui::Ui, rect: Rect, galley: std::sync::Arc<egui::Galley>, hint: Option<std::sync::Arc<egui::Galley>>, ink: Color32, hint_ink: Color32) {
    let total = galley.size().x + hint.as_ref().map_or(0.0, |h| h.size().x + 12.0);
    let left = rect.center().x - total / 2.0;
    let y = rect.center().y - galley.size().y / 2.0;
    let after = left + galley.size().x + 12.0;
    ui.painter().galley(Pos2::new(left, y), galley, ink);
    if let Some(hint) = hint {
        let hy = rect.center().y - hint.size().y / 2.0;
        ui.painter().galley(Pos2::new(after, hy), hint, hint_ink);
    }
}

/// The primary action (Check): filled in the human ink, semibold, with its shortcut beside it.
pub fn primary(ui: &mut egui::Ui, label: &str, hint: Option<&str>, enabled: bool) -> Response {
    let (response, galley, hint) = button_shell(ui, label, hint, enabled, theme::strong(15.0), 112.0);
    let p = palette(ui);
    let fill = if !enabled {
        p.rule
    } else if response.is_pointer_button_down_on() {
        p.primary.gamma_multiply(0.82)
    } else if response.hovered() {
        lighten(p.primary, dark(ui))
    } else {
        p.primary
    };
    let ink = if enabled { p.on_primary } else { p.muted };
    ui.painter().rect_filled(response.rect, radius(), fill);
    paint_label(ui, response.rect, galley, hint, ink, ink.gamma_multiply(0.75));
    focus_ring(ui, &response);
    if enabled {
        response.clone().on_hover_cursor(egui::CursorIcon::PointingHand);
    }
    response
}

fn lighten(color: Color32, dark: bool) -> Color32 {
    let toward = if dark { Color32::WHITE } else { Color32::BLACK };
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * 0.12).round() as u8;
    Color32::from_rgb(mix(color.r(), toward.r()), mix(color.g(), toward.g()), mix(color.b(), toward.b()))
}

/// A secondary action: no frame until the pointer is on it, muted ink that darkens on hover.
pub fn quiet(ui: &mut egui::Ui, label: &str, hint: Option<&str>, enabled: bool) -> Response {
    let (response, galley, hint) = button_shell(ui, label, hint, enabled, egui::FontId::proportional(15.0), 0.0);
    let p = palette(ui);
    if enabled && (response.hovered() || response.is_pointer_button_down_on()) {
        ui.painter().rect_filled(response.rect, radius(), if response.is_pointer_button_down_on() { p.press } else { p.hover });
    }
    let ink = if !enabled {
        p.muted.gamma_multiply(0.6)
    } else if response.hovered() {
        p.ink
    } else {
        p.muted
    };
    paint_label(ui, response.rect, galley, hint, ink, p.muted.gamma_multiply(0.8));
    focus_ring(ui, &response);
    response
}

/// A link-like action in the human ink ("More detail").
pub fn link(ui: &mut egui::Ui, label: &str) -> Response {
    let p = palette(ui);
    let font = egui::FontId::proportional(14.0);
    let galley = ui.painter().layout_no_wrap(label.to_owned(), font, p.human_ink);
    let (rect, response) = ui.allocate_exact_size(galley.size() + Vec2::new(4.0, 8.0), Sense::click());
    response.widget_info(|| WidgetInfo::labeled(Role::Link, true, label));
    let at = rect.left_center() + Vec2::new(2.0, -galley.size().y / 2.0);
    if response.hovered() {
        let y = at.y + galley.size().y + 1.0;
        ui.painter().line_segment([Pos2::new(at.x, y), Pos2::new(at.x + galley.size().x, y)], Stroke::new(1.0, p.human_ink));
    }
    ui.painter().galley(at, galley, p.human_ink);
    focus_ring(ui, &response);
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A segmented switch over `options`; returns the option clicked, if any.
pub fn segmented<T: Copy + PartialEq>(ui: &mut egui::Ui, current: T, options: &[(T, &str)], id: &str) -> Option<T> {
    let p = palette(ui);
    let font = egui::FontId::proportional(14.0);
    let height = ui.spacing().interact_size.y.clamp(30.0, 36.0);
    let galleys: Vec<_> = options.iter().map(|(_, name)| ui.painter().layout_no_wrap((*name).to_owned(), font.clone(), Color32::PLACEHOLDER)).collect();
    let widths: Vec<f32> = galleys.iter().map(|g| g.size().x + 24.0).collect();
    let (outer, _) = ui.allocate_exact_size(Vec2::new(widths.iter().sum::<f32>() + 4.0, height), Sense::hover());
    ui.painter().rect(outer, radius(), p.field, Stroke::new(1.0, p.rule), StrokeKind::Inside);
    let mut x = outer.left() + 2.0;
    let mut chosen = None;
    for (index, ((value, name), galley)) in options.iter().zip(galleys).enumerate() {
        let rect = Rect::from_min_size(Pos2::new(x, outer.top() + 2.0), Vec2::new(widths[index], height - 4.0));
        x += widths[index];
        let response = ui.interact(rect, ui.scope_id().with((id, index)), Sense::click());
        let selected = *value == current;
        response.widget_info(|| WidgetInfo::selected(Role::RadioButton, true, selected, *name));
        if selected {
            ui.painter().rect_filled(rect, CornerRadius::same(theme::RADIUS - 2), p.press);
        } else if response.hovered() {
            ui.painter().rect_filled(rect, CornerRadius::same(theme::RADIUS - 2), p.hover);
        }
        let ink = if selected || response.hovered() { p.ink } else { p.muted };
        let galley = if selected { ui.painter().layout_no_wrap((*name).to_owned(), theme::strong(14.0), ink) } else { galley };
        ui.painter().galley(rect.center() - galley.size() / 2.0, galley, ink);
        if response.has_focus() {
            ui.painter().rect_stroke(rect, CornerRadius::same(theme::RADIUS - 2), Stroke::new(2.0, p.focus), StrokeKind::Inside);
        }
        if response.clicked() && !selected {
            chosen = Some(*value);
        }
    }
    chosen
}

/// One of the screens' tabs in the top bar: the current one in ink with the mixed purple under it.
pub fn nav_tab(ui: &mut egui::Ui, selected: bool, label: &str, hint: &str) -> Response {
    let p = palette(ui);
    let font = if selected { theme::strong(15.0) } else { egui::FontId::proportional(15.0) };
    let galley = ui.painter().layout_no_wrap(label.to_owned(), font, Color32::PLACEHOLDER);
    let (rect, response) = ui.allocate_exact_size(Vec2::new(galley.size().x + 20.0, 40.0), Sense::click());
    response.widget_info(|| WidgetInfo::selected(Role::Tab, true, selected, label));
    let ink = if selected || response.hovered() { p.ink } else { p.muted };
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, ink);
    let line = if selected {
        Some(p.focus)
    } else if response.hovered() {
        Some(p.rule)
    } else {
        None
    };
    if let Some(color) = line {
        let y = rect.bottom() - 2.0;
        ui.painter().line_segment([Pos2::new(rect.left() + 10.0, y), Pos2::new(rect.right() - 10.0, y)], Stroke::new(2.0, color));
    }
    focus_ring(ui, &response);
    response.on_hover_text(hint)
}

/// A square icon button: a moon to switch to dark, a sun to switch to light.
pub fn theme_toggle(ui: &mut egui::Ui) -> Response {
    let p = palette(ui);
    let is_dark = dark(ui);
    let side = 32.0;
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), Sense::click());
    let label = if is_dark { "Switch to the light theme" } else { "Switch to the dark theme" };
    response.widget_info(|| WidgetInfo::labeled(Role::Button, true, label));
    if response.hovered() {
        ui.painter().rect_filled(rect, radius(), p.hover);
    }
    let ink = if response.hovered() { p.ink } else { p.muted };
    let c = rect.center();
    if is_dark {
        // A sun: a disc and eight rays.
        ui.painter().circle_filled(c, 4.5, ink);
        for i in 0..8 {
            let a = i as f32 * std::f32::consts::FRAC_PI_4;
            let d = Vec2::new(a.cos(), a.sin());
            ui.painter().line_segment([c + d * 7.0, c + d * 9.5], Stroke::new(1.6, ink));
        }
    } else {
        // A moon: a disc with a second disc of the ground cut from it.
        let fill = if response.hovered() { p.hover } else { p.ground };
        ui.painter().circle_filled(c, 8.0, ink);
        ui.painter().circle_filled(c + Vec2::new(4.0, -3.5), 7.0, fill);
    }
    focus_ring(ui, &response);
    response.on_hover_text(label)
}

/// A card: the raised surface with a rule around it.
pub fn card<R>(ui: &mut egui::Ui, margin: i8, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let p = palette(ui);
    egui::Frame::new().fill(p.raised).stroke(Stroke::new(1.0, p.rule)).corner_radius(radius()).inner_margin(Margin::same(margin)).show(ui, add).inner
}

/// A small heading over a group of controls or numbers.
pub fn section(ui: &mut egui::Ui, title: &str) {
    let p = palette(ui);
    ui.label(egui::RichText::new(title).font(theme::strong(13.0)).color(p.muted));
}

/// A label and a value in a two-column grid row (the value in the mono face).
pub fn fact(ui: &mut egui::Ui, label: &str, value: String, hover: Option<&str>) {
    let p = palette(ui);
    let response = ui.label(egui::RichText::new(label).color(p.muted));
    if let Some(hover) = hover {
        response.on_hover_text(hover);
    }
    ui.label(egui::RichText::new(value).monospace());
    ui.end_row();
}

/// The caution triangle, drawn (the bundled fonts have no warning sign).
pub fn caution_icon(ui: &mut egui::Ui, ink: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), Sense::hover());
    let r = rect.shrink(1.0);
    let points = vec![r.center_top(), r.right_bottom(), r.left_bottom()];
    ui.painter().add(egui::Shape::convex_polygon(points, Color32::TRANSPARENT, Stroke::new(1.6, ink)));
    ui.painter().text(r.center() + Vec2::new(0.0, 2.5), Align2::CENTER_CENTER, "!", theme::strong(10.0), ink);
}
