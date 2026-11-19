//! The name, the litmus colors and the mark. The colors are the site's (`site/style.css`): blue `#2F6FDB` (human),
//! purple `#8E5BC9` (mixed), red `#D9483B` (AI), the pale band `#C9B8E6` (what cannot be ruled out) and paper
//! `#F3EEE4`.
//!
//! The mark is a cursive capital L drawn as one strip that runs blue through the top loop and the stem, purple
//! through the lower loop and red along the tail, the color graded along the stroke. It is the mark
//! `assets/litmus_mark.svg` with the three strokes joined into one: `icon_svg` writes `assets/litmus_icon.svg` (a
//! test keeps the two equal) and `icon_rgba` draws the same strip into pixels for the window icon and the in-app mark.

/// The product's display name, in one place: the public name is not final.
pub const NAME: &str = "Litmus";

pub type Rgb = (u8, u8, u8);

pub const BLUE: Rgb = (0x2f, 0x6f, 0xdb);
pub const PURPLE: Rgb = (0x8e, 0x5b, 0xc9);
pub const RED: Rgb = (0xd9, 0x48, 0x3b);
pub const BAND: Rgb = (0xc9, 0xb8, 0xe6);
pub const PAPER: Rgb = (0xf3, 0xee, 0xe4);

/// Linear mix of two colors, `t` in [0, 1].
pub fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    let channel = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
    (channel(a.0, b.0), channel(a.1, b.1), channel(a.2, b.2))
}

/// The litmus color at a strip position: 0 human (blue), 0.5 mixed (purple), 1 AI (red).
pub fn litmus(position: f64) -> Rgb {
    let x = if position.is_finite() { position.clamp(0.0, 1.0) } else { 0.5 };
    if x <= 0.5 { mix(BLUE, PURPLE, x / 0.5) } else { mix(PURPLE, RED, (x - 0.5) / 0.5) }
}

/// Relative luminance (WCAG) of a color, for choosing ink and checking lightness steps.
pub fn luminance(color: Rgb) -> f64 {
    let linear = |c: u8| {
        let c = c as f64 / 255.0;
        if c <= 0.039_28 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    0.2126 * linear(color.0) + 0.7152 * linear(color.1) + 0.0722 * linear(color.2)
}

type Point = (f64, f64);

/// The strip of the mark as cubic Bézier segments in the mark's coordinates (view box 34 10 142 198), in drawing
/// order: three blue, three purple, one red.
const MARK: [[Point; 4]; 7] = [
    [(70.0, 95.0), (100.0, 95.0), (140.0, 70.0), (138.0, 45.0)],
    [(138.0, 45.0), (137.0, 28.0), (118.0, 26.0), (112.0, 45.0)],
    [(112.0, 45.0), (106.0, 70.0), (104.0, 110.0), (96.0, 140.0)],
    [(96.0, 140.0), (92.0, 160.0), (86.0, 178.0), (70.0, 182.0)],
    [(70.0, 182.0), (56.0, 185.0), (50.0, 172.0), (62.0, 166.0)],
    [(62.0, 166.0), (76.0, 160.0), (92.0, 172.0), (104.0, 180.0)],
    [(104.0, 180.0), (120.0, 190.0), (140.0, 192.0), (160.0, 178.0)],
];
/// The segments the color reaches purple and red at (the mark's color changes), and how far the grade runs on
/// each side of a change, in segments.
const PURPLE_FROM: f64 = 3.0;
const RED_FROM: f64 = 6.0;
const GRADE: f64 = 0.45;
/// The view box (x, y, width, height) and the stroke width of the drawing.
const VIEW: (f64, f64, f64, f64) = (34.0, 10.0, 142.0, 198.0);
const STROKE: f64 = 16.0;
/// At or under this many pixels the stroke is drawn heavier, as the mark asks.
const SMALL: usize = 32;

fn cubic(c: &[Point; 4], t: f64) -> Point {
    let u = 1.0 - t;
    let (a, b, cc, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    (a * c[0].0 + b * c[1].0 + cc * c[2].0 + d * c[3].0, a * c[0].1 + b * c[1].1 + cc * c[2].1 + d * c[3].1)
}

/// The color of the strip at `s`, its place along the mark in segments (0 to 7).
pub fn mark_color(s: f64) -> Rgb {
    let ramp = |from: f64| ((s - (from - GRADE)) / (2.0 * GRADE)).clamp(0.0, 1.0);
    let toward_purple = mix(BLUE, PURPLE, ramp(PURPLE_FROM));
    mix(toward_purple, RED, ramp(RED_FROM))
}

/// The mark as an SVG document: the strip cut into short pieces, each in its color along the grade.
pub fn icon_svg() -> String {
    const PIECES: usize = 6;
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\" role=\"img\">\n  <title>{NAME}</title>\n  \
         <desc>A cursive capital L drawn as one strip, its color graded along the stroke: blue through the top loop and \
         stem (human), purple through the lower loop (mixed), red along the tail (AI). Written by the app's \
         brand::icon_svg from the mark, litmus_mark.svg.</desc>\n  <g fill=\"none\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n",
        VIEW.0, VIEW.1, VIEW.2, VIEW.3, STROKE
    );
    for (index, segment) in MARK.iter().enumerate() {
        for piece in 0..PIECES {
            let (t0, t1) = (piece as f64 / PIECES as f64, (piece + 1) as f64 / PIECES as f64);
            let part = split(segment, t0, t1);
            let color = mark_color(index as f64 + (t0 + t1) / 2.0);
            out.push_str(&format!(
                "    <path d=\"M{} {} C{} {} {} {} {} {}\" stroke=\"#{:02X}{:02X}{:02X}\"/>\n",
                round2(part[0].0),
                round2(part[0].1),
                round2(part[1].0),
                round2(part[1].1),
                round2(part[2].0),
                round2(part[2].1),
                round2(part[3].0),
                round2(part[3].1),
                color.0,
                color.1,
                color.2
            ));
        }
    }
    out.push_str("  </g>\n</svg>\n");
    out
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// The part of a cubic between `t0` and `t1` (de Casteljau, twice).
fn split(c: &[Point; 4], t0: f64, t1: f64) -> [Point; 4] {
    fn lerp(a: Point, b: Point, t: f64) -> Point {
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
    }
    fn tail(c: &[Point; 4], t: f64) -> [Point; 4] {
        let (ab, bc, cd) = (lerp(c[0], c[1], t), lerp(c[1], c[2], t), lerp(c[2], c[3], t));
        let (abc, bcd) = (lerp(ab, bc, t), lerp(bc, cd, t));
        [lerp(abc, bcd, t), bcd, cd, c[3]]
    }
    fn head(c: &[Point; 4], t: f64) -> [Point; 4] {
        let (ab, bc, cd) = (lerp(c[0], c[1], t), lerp(c[1], c[2], t), lerp(c[2], c[3], t));
        let (abc, bcd) = (lerp(ab, bc, t), lerp(bc, cd, t));
        [c[0], ab, abc, lerp(abc, bcd, t)]
    }
    let rest = tail(c, t0);
    if t1 >= 1.0 { rest } else { head(&rest, (t1 - t0) / (1.0 - t0)) }
}

/// The mark drawn into a square of `size` pixels, transparent around it, as unpremultiplied RGBA rows.
pub fn icon_rgba(size: usize) -> Vec<u8> {
    assert!(size >= 8, "an icon of at least 8 pixels");
    const STEPS: usize = 24;
    let scale = size as f64 / VIEW.3; // the view box is taller than wide: fit the height
    let offset_x = (size as f64 - VIEW.2 * scale) / 2.0;
    let half = if size <= SMALL { STROKE * 1.35 } else { STROKE } * scale / 2.0;
    // Each segment as a polyline in pixels, with its place along the strip.
    let polylines: Vec<Vec<(Point, f64)>> = MARK
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            (0..=STEPS)
                .map(|step| {
                    let t = step as f64 / STEPS as f64;
                    let (x, y) = cubic(segment, t);
                    (((x - VIEW.0) * scale + offset_x, (y - VIEW.1) * scale), index as f64 + t)
                })
                .collect()
        })
        .collect();
    let mut pixels = vec![[0f64; 4]; size * size]; // premultiplied
    for polyline in &polylines {
        for y in 0..size {
            for x in 0..size {
                let p = (x as f64 + 0.5, y as f64 + 0.5);
                let mut best = (f64::INFINITY, 0.0);
                for pair in polyline.windows(2) {
                    let ((a, sa), (b, sb)) = (pair[0], pair[1]);
                    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                    let length = dx * dx + dy * dy;
                    let t = if length > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / length).clamp(0.0, 1.0) } else { 0.0 };
                    let (qx, qy) = (a.0 + t * dx - p.0, a.1 + t * dy - p.1);
                    let distance = (qx * qx + qy * qy).sqrt();
                    if distance < best.0 {
                        best = (distance, sa + (sb - sa) * t);
                    }
                }
                let coverage = (half + 0.5 - best.0).clamp(0.0, 1.0);
                if coverage > 0.0 {
                    let color = mark_color(best.1);
                    let pixel = &mut pixels[y * size + x];
                    let keep = 1.0 - coverage;
                    pixel[0] = color.0 as f64 / 255.0 * coverage + pixel[0] * keep;
                    pixel[1] = color.1 as f64 / 255.0 * coverage + pixel[1] * keep;
                    pixel[2] = color.2 as f64 / 255.0 * coverage + pixel[2] * keep;
                    pixel[3] = coverage + pixel[3] * keep;
                }
            }
        }
    }
    let mut out = Vec::with_capacity(size * size * 4);
    for pixel in pixels {
        let alpha = pixel[3];
        let channel = |c: f64| if alpha > 0.0 { (c / alpha * 255.0).round().clamp(0.0, 255.0) as u8 } else { 0 };
        out.extend([channel(pixel[0]), channel(pixel[1]), channel(pixel[2]), (alpha * 255.0).round() as u8]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn litmus_scale_runs_blue_purple_red() {
        assert_eq!(litmus(0.0), BLUE);
        assert_eq!(litmus(0.5), PURPLE);
        assert_eq!(litmus(1.0), RED);
        assert_eq!(litmus(-3.0), BLUE);
        assert_eq!(litmus(f64::NAN), PURPLE);
    }

    #[test]
    fn mark_grades_along_the_strip() {
        assert_eq!(mark_color(0.0), BLUE);
        assert_eq!(mark_color(4.5), PURPLE);
        assert_eq!(mark_color(7.0), RED);
    }

    #[test]
    fn the_strip_is_continuous() {
        for pair in MARK.windows(2) {
            assert_eq!(pair[0][3], pair[1][0], "each segment starts where the last ends");
        }
    }

    #[test]
    fn icon_asset_matches_the_code() {
        let asset = include_str!("../assets/litmus_icon.svg").replace("\r\n", "\n");
        assert_eq!(asset, icon_svg(), "regenerate assets/litmus_icon.svg with `cargo run -- --write-icon`");
    }

    #[test]
    fn icon_has_ink_and_a_clear_ground() {
        let size = 64;
        let rgba = icon_rgba(size);
        assert_eq!(rgba.len(), size * size * 4);
        assert_eq!(rgba[3], 0, "the corner is transparent");
        let opaque = rgba.as_chunks::<4>().0.iter().filter(|p| p[3] == 255).count();
        assert!(opaque > size * size / 20 && opaque < size * size / 2, "{opaque} opaque pixels");
    }
}
