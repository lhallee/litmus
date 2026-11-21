//! The litmus strip, drawn as the site draws it (`site/style.css`, `.strip`): a rounded bar from blue (human, 0)
//! through purple to red (AI, 1), the mark, a bar of ink at the reading's chance of AI, and the pale dashed band over
//! the chance's range, which is the interval the chance is given with (SPEC.md, Chance of AI). A text unlike the
//! reference gets a washed-out bar in a dashed outline, a question mark for its mark, and no band. The scale is
//! labeled in words at both ends, so the strip reads without telling red from blue.

use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Vec2};
pub use litmus_engine::Drawn;
use litmus_engine::reading::percent;

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

    /// The band's ends, in order, at least `min_width` apart and inside the strip: a range too narrow to see is
    /// widened evenly about its middle, so it is still drawn, hugging the mark.
    pub fn band(&self, band: [f64; 2], min_width: f32) -> (f32, f32) {
        let (a, b) = (self.x(band[0]), self.x(band[1]));
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if hi - lo >= min_width {
            return (lo, hi);
        }
        let right = self.left + self.width;
        let lo = ((lo + hi) / 2.0 - min_width / 2.0).clamp(self.left, (right - min_width).max(self.left));
        (lo, (lo + min_width).min(right))
    }
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

/// How far the mark and the band reach over a bar `bar` tall, above it and below it.
fn overhang(bar: f32) -> f32 {
    (bar * 0.4).max(6.0)
}

/// The height `show` takes for a bar `bar` tall: the bar, the mark's reach over both edges, and the words under it.
pub fn height(bar: f32) -> f32 {
    overhang(bar) * 2.0 + bar + 18.0
}

/// The width of the full strip's mark for a bar `bar` tall, without its outline.
fn mark_width(bar: f32) -> f32 {
    (bar * 0.24).clamp(3.0, 6.0)
}

/// The full strip with its words under it, the bar `height` tall. Hovering shows the numbers.
pub fn show(ui: &mut egui::Ui, drawn: &Drawn, height: f32) -> egui::Response {
    let p = theme::palette(ui.visuals().dark_mode);
    let over = overhang(height);
    let label_font = FontId::proportional(if height >= 20.0 { 13.0 } else { 12.0 });
    let total = Vec2::new(ui.available_width().max(160.0), self::height(height));
    let (rect, response) = ui.allocate_exact_size(total, Sense::hover());
    let painter = ui.painter_at(rect.expand(4.0));
    let inset = height * 0.5;
    let bar = Rect::from_min_size(Pos2::new(rect.left() + inset, rect.top() + over), Vec2::new(rect.width() - 2.0 * inset, height));
    let geometry = Geometry { left: bar.left(), width: bar.width() };
    let washed = drawn.unlike_reference;
    paint_scale(&painter, bar, washed);
    if washed {
        dashed_rect(&painter, bar.expand(3.0), Stroke::new(1.5, p.band_edge));
    }
    let w = mark_width(height);
    if let Some(range) = drawn.band {
        // A range under a point or two would hide behind the mark and its outline: it keeps a rim on both sides.
        let (lo, hi) = geometry.band(range, w + 12.0);
        let band = Rect::from_x_y_ranges(lo..=hi, (bar.top() - over * 0.5)..=(bar.bottom() + over * 0.5));
        painter.rect_filled(band, CornerRadius::same(4), p.band_wash);
        dashed_rect(&painter, band, Stroke::new(1.5, p.band_edge));
    }
    if let Some(mark) = drawn.mark {
        let x = geometry.x(mark);
        if washed {
            let r = height * 0.5 + 2.0;
            painter.circle(Pos2::new(x, bar.center().y), r, p.raised, Stroke::new(2.0, p.ink));
            painter.text(Pos2::new(x, bar.center().y), Align2::CENTER_CENTER, "?", theme::strong(height * 0.62), p.ink);
        } else {
            let mark = Rect::from_x_y_ranges((x - w / 2.0)..=(x + w / 2.0), (bar.top() - over)..=(bar.bottom() + over));
            painter.rect(mark, CornerRadius::same(3), p.ink, Stroke::new(2.0, p.ground), StrokeKind::Outside);
        }
    }
    // Words under the scale, so the strip reads without color.
    let y = bar.bottom() + over + 2.0;
    painter.text(Pos2::new(bar.left() - inset, y), Align2::LEFT_TOP, "Human", label_font.clone(), p.human_ink);
    painter.text(Pos2::new(bar.right() + inset, y), Align2::RIGHT_TOP, "AI", label_font, p.ai_ink);
    response.on_hover_text(hover_words(drawn))
}

/// What the strip says in words: the chance at the mark, the range under the band, and why a strip is washed out.
pub fn hover_words(drawn: &Drawn) -> String {
    let Some(mark) = drawn.mark else {
        return "The strip runs from human at the left to AI at the right. This reading has no chance of AI to mark, because the model's calibration is not loaded.".to_owned();
    };
    let mut words = format!("The mark is the chance of AI, {}.", percent(mark));
    // The range in the words every other place writes it with: whole percentages that never reach 0% or 100%.
    match (drawn.band, drawn.range()) {
        (Some(_), Some((lo, hi))) => words.push_str(&format!(" The pale band is its range, {lo:.0}% to {hi:.0}%.")),
        (Some(_), None) => words.push_str(" Its range is about a point wide or less, so the pale band hugs the mark."),
        (None, _) => {}
    }
    if drawn.unlike_reference {
        words.push_str(" Washed out with a question mark because the text is unlike the reference, so the chance is no evidence either way.");
    }
    words
}

/// A small strip for tables: the bar, the band and the mark in `width` by `height`. The band is drawn at its true
/// width, so a range narrower than the mark stays behind it.
pub fn mini(ui: &mut egui::Ui, drawn: &Drawn, width: f32, height: f32) -> egui::Response {
    let p = theme::palette(ui.visuals().dark_mode);
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    let painter = ui.painter_at(rect.expand(1.0));
    let bar_height = (height * 0.42).round().max(4.0);
    let bar = Rect::from_center_size(rect.center(), Vec2::new(width - 6.0, bar_height));
    paint_scale(&painter, bar, drawn.unlike_reference);
    let geometry = Geometry { left: bar.left(), width: bar.width() };
    if let Some(range) = drawn.band {
        let (lo, hi) = geometry.band(range, 0.0);
        if hi - lo > 5.0 {
            let band = Rect::from_x_y_ranges(lo..=hi, (bar.top() - 2.0)..=(bar.bottom() + 2.0));
            painter.rect_filled(band, CornerRadius::same(2), p.band_wash);
            painter.rect_stroke(band, CornerRadius::same(2), Stroke::new(1.0, p.band_edge), StrokeKind::Inside);
        }
    }
    if let Some(mark) = drawn.mark {
        let x = geometry.x(mark);
        if drawn.unlike_reference {
            painter.circle(Pos2::new(x, rect.center().y), height * 0.42, p.raised, Stroke::new(1.5, p.ink));
            painter.text(Pos2::new(x, rect.center().y), Align2::CENTER_CENTER, "?", theme::strong(height * 0.6), p.ink);
        } else {
            let mark = Rect::from_x_y_ranges((x - 1.5)..=(x + 1.5), rect.top() + 1.0..=rect.bottom() - 1.0);
            painter.rect(mark, CornerRadius::same(1), p.ink, Stroke::new(1.0, p.ground), StrokeKind::Outside);
        }
    }
    response.on_hover_text(hover_words(drawn))
}

#[cfg(test)]
mod tests {
    use super::*;
    use litmus_engine::Chance;

    fn chance(value: f64, interval: [f64; 2]) -> Chance {
        Chance { value, interval, reference_share: 0.5, between_sources: 0.0 }
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
        assert_eq!(g.band([0.5, 1.0], 3.0), (50.0, 100.0));
        assert_eq!(g.band([1.0, 0.5], 3.0), (50.0, 100.0));
        let (lo, hi) = g.band([0.3, 0.3], 3.0);
        assert!((hi - lo - 3.0).abs() < 1e-4 && lo <= 30.0 && hi >= 30.0);
        assert_eq!(g.band([0.0, 0.0], 3.0), (0.0, 3.0), "a sliver at the left edge stays inside the strip");
        assert_eq!(g.band([1.0, 1.0], 3.0), (97.0, 100.0), "and one at the right edge");
        assert_eq!(g.band([0.5, 0.5], 0.0), (50.0, 50.0), "no least width leaves the range as it is");
        assert_eq!(g.band([0.25, 0.375], 0.0), (25.0, 37.5));
    }

    #[test]
    fn the_hover_words_give_the_chance_and_its_range() {
        assert_eq!(hover_words(&Drawn::of(Some(&chance(0.62, [0.4, 0.8])), false)), "The mark is the chance of AI, 62%. The pale band is its range, 40% to 80%.");
        assert_eq!(
            hover_words(&Drawn::of(Some(&chance(0.5643, [0.5635, 0.5663])), false)),
            "The mark is the chance of AI, 56%. Its range is about a point wide or less, so the pale band hugs the mark."
        );
        // An end is written as the tables write it, never as 100%.
        assert_eq!(hover_words(&Drawn::of(Some(&chance(0.99, [0.959, 0.998])), false)), "The mark is the chance of AI, 99%. The pale band is its range, 96% to 99%.");
        assert!(hover_words(&Drawn::of(Some(&chance(0.62, [0.4, 0.8])), true)).ends_with("so the chance is no evidence either way."));
        assert!(hover_words(&Drawn::of(None, false)).contains("no chance of AI to mark"));
    }
}
