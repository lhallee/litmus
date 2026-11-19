//! The litmus strip, drawn as the site draws it (`site/style.css`, `.strip`): a rounded bar from blue (human, 0)
//! through purple (mixed, 0.5) to red (AI, 1), the pale dashed band over what the conformal set cannot rule out, and
//! the mark, a bar of ink at the reading's position. A text unlike the reference gets a washed-out bar in a dashed
//! outline and a question mark for its mark, and no band when every origin is ruled out. The scale is labeled in
//! words at both ends and the middle, so the strip reads without telling red from blue.

use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Vec2};
use litmus_engine::Strip;

use crate::brand;
use crate::theme;

/// Where a position and a band fall across a strip drawn from `left` over `width`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub left: f32,
    pub width: f32,
}

impl Geometry {
    /// The x of a strip position, clamped to the strip; a position that is not a number sits in the middle.
    pub fn x(&self, position: f64) -> f32 {
        let p = if position.is_finite() { position.clamp(0.0, 1.0) } else { 0.5 };
        self.left + self.width * p as f32
    }

    /// The band's ends, in order, at least `min_width` apart (a band of no width is still drawn as a sliver).
    pub fn band(&self, strip: &Strip, min_width: f32) -> (f32, f32) {
        let (a, b) = (self.x(strip.band_lo), self.x(strip.band_hi));
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if hi - lo >= min_width {
            return (lo, hi);
        }
        let middle = (lo + hi) / 2.0;
        let half = min_width / 2.0;
        let lo = (middle - half).max(self.left);
        (lo, (lo + min_width).min(self.left + self.width))
    }
}

/// Whether a strip has a band to draw: a text unlike the reference whose set is empty has none (the site's "there
/// is no band"); the engine then gives a band of no width.
pub fn has_band(strip: &Strip) -> bool {
    !(strip.unlike_reference && (strip.band_hi - strip.band_lo).abs() < 1e-9)
}

fn scale_color(t: f64, washed: bool) -> Color32 {
    let color = brand::litmus(t);
    theme::rgb(if washed { brand::mix(color, (0x9a, 0x96, 0x90), 0.62) } else { color })
}

/// Paint the gradient of the scale into `rect`, its ends rounded.
pub fn paint_scale(painter: &egui::Painter, rect: Rect, washed: bool) {
    const STOPS: usize = 48;
    let r = rect.height() / 2.0;
    let t_of = |x: f32| ((x - rect.left()) / rect.width()) as f64;
    painter.circle_filled(Pos2::new(rect.left() + r, rect.center().y), r, scale_color(t_of(rect.left() + r * 0.5), washed));
    painter.circle_filled(Pos2::new(rect.right() - r, rect.center().y), r, scale_color(t_of(rect.right() - r * 0.5), washed));
    let (left, right) = (rect.left() + r, rect.right() - r);
    let mut mesh = egui::Mesh::default();
    for stop in 0..=STOPS {
        let x = left + (right - left) * stop as f32 / STOPS as f32;
        let color = scale_color(t_of(x), washed);
        mesh.colored_vertex(Pos2::new(x, rect.top()), color);
        mesh.colored_vertex(Pos2::new(x, rect.bottom()), color);
        if stop > 0 {
            let i = (stop as u32) * 2;
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i, i + 1);
        }
    }
    painter.add(Shape::mesh(mesh));
}

fn dashed_rect(painter: &egui::Painter, rect: Rect, stroke: Stroke) {
    let path = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom(), rect.left_top()];
    painter.extend(Shape::dashed_line(&path, stroke, 5.0, 3.0));
}

/// The full strip with its words under it, the bar `height` tall. Hovering shows the numbers.
pub fn show(ui: &mut egui::Ui, strip: &Strip, height: f32) -> egui::Response {
    let p = theme::palette(ui.visuals().dark_mode);
    let over = (height * 0.4).max(6.0);
    let label_font = FontId::proportional(if height >= 20.0 { 13.0 } else { 12.0 });
    let total = Vec2::new(ui.available_width().max(160.0), over * 2.0 + height + 18.0);
    let (rect, response) = ui.allocate_exact_size(total, Sense::hover());
    let painter = ui.painter_at(rect.expand(4.0));
    let inset = height * 0.5;
    let bar = Rect::from_min_size(Pos2::new(rect.left() + inset, rect.top() + over), Vec2::new(rect.width() - 2.0 * inset, height));
    let geometry = Geometry { left: bar.left(), width: bar.width() };
    let washed = strip.unlike_reference;
    paint_scale(&painter, bar, washed);
    if washed {
        dashed_rect(&painter, bar.expand(3.0), Stroke::new(1.5, p.band_edge));
    }
    if has_band(strip) {
        let (lo, hi) = geometry.band(strip, 6.0);
        let band = Rect::from_x_y_ranges(lo..=hi, (bar.top() - over * 0.5)..=(bar.bottom() + over * 0.5));
        painter.rect_filled(band, CornerRadius::same(4), p.band_wash);
        dashed_rect(&painter, band, Stroke::new(1.5, p.band_edge));
    }
    let x = geometry.x(strip.position);
    if washed {
        let r = height * 0.5 + 2.0;
        painter.circle(Pos2::new(x, bar.center().y), r, p.raised, Stroke::new(2.0, p.ink));
        painter.text(Pos2::new(x, bar.center().y), Align2::CENTER_CENTER, "?", theme::strong(height * 0.62), p.ink);
    } else {
        let w = (height * 0.24).clamp(3.0, 6.0);
        let mark = Rect::from_x_y_ranges((x - w / 2.0)..=(x + w / 2.0), (bar.top() - over)..=(bar.bottom() + over));
        painter.rect(mark, CornerRadius::same(3), p.ink, Stroke::new(2.0, p.ground), StrokeKind::Outside);
    }
    // Words under the scale, so the strip reads without color.
    let y = bar.bottom() + over + 2.0;
    painter.text(Pos2::new(bar.left() - inset, y), Align2::LEFT_TOP, "Human", label_font.clone(), p.human_ink);
    painter.text(Pos2::new(bar.center().x, y), Align2::CENTER_TOP, "Mixed", label_font.clone(), p.mixed_ink);
    painter.text(Pos2::new(bar.right() + inset, y), Align2::RIGHT_TOP, "AI", label_font, p.ai_ink);
    response.on_hover_text(hover_words(strip))
}

fn hover_words(strip: &Strip) -> String {
    format!(
        "Mark at {:.2} (0 human, 0.5 mixed, 1 AI).{} {}",
        strip.position,
        if has_band(strip) {
            format!(" Pale band from {:.2} to {:.2}: what the set at {:.0}% cannot rule out.", strip.band_lo.min(strip.band_hi), strip.band_hi.max(strip.band_lo), strip.level * 100.0)
        } else {
            String::new()
        },
        if strip.unlike_reference { "Washed out with a question mark: the text is unlike the reference." } else { "" }
    )
}

/// A small strip for tables: the bar, the band and the mark in `width` by `height`.
pub fn mini(ui: &mut egui::Ui, strip: &Strip, width: f32, height: f32) -> egui::Response {
    let p = theme::palette(ui.visuals().dark_mode);
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    let painter = ui.painter_at(rect.expand(1.0));
    let bar_height = (height * 0.42).round().max(4.0);
    let bar = Rect::from_center_size(rect.center(), Vec2::new(width - 6.0, bar_height));
    paint_scale(&painter, bar, strip.unlike_reference);
    let geometry = Geometry { left: bar.left(), width: bar.width() };
    if has_band(strip) {
        let (lo, hi) = geometry.band(strip, 3.0);
        let band = Rect::from_x_y_ranges(lo..=hi, (bar.top() - 2.0)..=(bar.bottom() + 2.0));
        painter.rect_filled(band, CornerRadius::same(2), p.band_wash);
        painter.rect_stroke(band, CornerRadius::same(2), Stroke::new(1.0, p.band_edge), StrokeKind::Inside);
    }
    let x = geometry.x(strip.position);
    if strip.unlike_reference {
        painter.circle(Pos2::new(x, rect.center().y), height * 0.42, p.raised, Stroke::new(1.5, p.ink));
        painter.text(Pos2::new(x, rect.center().y), Align2::CENTER_CENTER, "?", theme::strong(height * 0.6), p.ink);
    } else {
        let mark = Rect::from_x_y_ranges((x - 1.5)..=(x + 1.5), rect.top() + 1.0..=rect.bottom() - 1.0);
        painter.rect(mark, CornerRadius::same(1), p.ink, Stroke::new(1.0, p.ground), StrokeKind::Outside);
    }
    response.on_hover_text(hover_words(strip))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(position: f64, band_lo: f64, band_hi: f64) -> Strip {
        Strip { position, band_lo, band_hi, level: 0.1, unlike_reference: false }
    }

    #[test]
    fn positions_map_across_the_strip() {
        let g = Geometry { left: 10.0, width: 200.0 };
        assert_eq!(g.x(0.0), 10.0);
        assert_eq!(g.x(0.5), 110.0);
        assert_eq!(g.x(1.0), 210.0);
        assert_eq!(g.x(1.7), 210.0);
        assert_eq!(g.x(-0.2), 10.0);
        assert_eq!(g.x(f64::NAN), 110.0);
    }

    #[test]
    fn bands_are_ordered_and_never_vanish() {
        let g = Geometry { left: 0.0, width: 100.0 };
        assert_eq!(g.band(&strip(0.7, 0.5, 1.0), 3.0), (50.0, 100.0));
        assert_eq!(g.band(&strip(0.7, 1.0, 0.5), 3.0), (50.0, 100.0));
        let (lo, hi) = g.band(&strip(0.3, 0.3, 0.3), 3.0);
        assert!((hi - lo - 3.0).abs() < 1e-4 && lo <= 30.0 && hi >= 30.0);
        let (lo, hi) = g.band(&strip(0.0, 0.0, 0.0), 3.0);
        assert_eq!((lo, hi), (0.0, 3.0), "a sliver at the edge stays inside the strip");
    }

    #[test]
    fn only_an_empty_set_loses_its_band() {
        assert!(has_band(&strip(0.3, 0.3, 0.3)), "a confident reading keeps its sliver");
        assert!(!has_band(&Strip { unlike_reference: true, ..strip(0.5, 0.5, 0.5) }));
        assert!(has_band(&Strip { unlike_reference: true, ..strip(0.5, 0.2, 0.7) }), "out of reference with a set keeps its band");
    }
}
