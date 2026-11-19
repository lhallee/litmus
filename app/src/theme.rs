//! Light and dark looks, in the tokens of the project's site (`site/style.css`): paper `#F3EEE4` and surface
//! `#FBF8F2` with ink `#1E1B22` in the light, `#16141A` and `#201D25` with ink `#EEE8DC` in the dark, and each
//! origin's ink (`--human-ink` and the like) for words in an origin's color. One corner radius, spacing on a 4-point
//! grid, a focus ring in the mixed purple (the site's `:focus-visible` outline), and the fonts.
//!
//! Fonts: on Windows the app uses the system's Segoe UI (regular and semibold) and Cascadia Mono or Consolas, the
//! faces the site's `system-ui` and monospace stacks resolve to there, read from the system's font folder at start
//! and never shipped; elsewhere, and in the browser, egui's bundled Ubuntu Light and Hack (`engine/THIRD_PARTY.md`).

use eframe::egui::{self, Color32, CornerRadius, FontDefinitions, FontFamily, FontId, Stroke, TextStyle, Theme, Visuals};
use litmus_engine::Origin;

use crate::brand::{self, Rgb};

/// The one corner radius of controls, cards and windows.
pub const RADIUS: u8 = 6;
/// The family of semibold text (headlines, labels that lead); the regular face where no semibold exists.
pub const STRONG: &str = "strong";

pub fn rgb(color: Rgb) -> Color32 {
    Color32::from_rgb(color.0, color.1, color.2)
}

/// Grounds and inks of a look.
#[derive(Clone, Copy)]
pub struct Palette {
    pub ground: Color32,
    pub raised: Color32,
    pub field: Color32,
    pub ink: Color32,
    pub muted: Color32,
    pub rule: Color32,
    /// A control's fill under the pointer, and pressed.
    pub hover: Color32,
    pub press: Color32,
    /// The caution color of notices (the placeholder model, flags), with the ink that reads on it.
    pub caution: Color32,
    pub caution_ink: Color32,
    /// Text in an origin's color, at least 4.5:1 on the ground (the site's `--human-ink`, `--mixed-ink`, `--ai-ink`).
    pub human_ink: Color32,
    pub mixed_ink: Color32,
    pub ai_ink: Color32,
    /// The primary button's fill and the text on it.
    pub primary: Color32,
    pub on_primary: Color32,
    /// The band's wash and edge, and the ring around the strip's mark.
    pub band_wash: Color32,
    pub band_edge: Color32,
    pub focus: Color32,
    pub error: Color32,
}

pub fn palette(dark: bool) -> Palette {
    if dark {
        Palette {
            ground: Color32::from_rgb(0x16, 0x14, 0x1a),
            raised: Color32::from_rgb(0x20, 0x1d, 0x25),
            field: Color32::from_rgb(0x1b, 0x19, 0x20),
            ink: Color32::from_rgb(0xee, 0xe8, 0xdc),
            muted: Color32::from_rgb(0xb9, 0xb2, 0xa8),
            rule: Color32::from_rgb(0x3a, 0x35, 0x40),
            hover: Color32::from_rgb(0x2b, 0x27, 0x31),
            press: Color32::from_rgb(0x35, 0x30, 0x3c),
            caution: Color32::from_rgb(0x4a, 0x3a, 0x12),
            caution_ink: Color32::from_rgb(0xff, 0xe3, 0xa0),
            human_ink: Color32::from_rgb(0x8d, 0xb2, 0xf5),
            mixed_ink: Color32::from_rgb(0xc3, 0xa3, 0xec),
            ai_ink: Color32::from_rgb(0xf2, 0x90, 0x7f),
            primary: Color32::from_rgb(0x8d, 0xb2, 0xf5),
            on_primary: Color32::from_rgb(0x16, 0x14, 0x1a),
            band_wash: Color32::from_rgba_unmultiplied(0xc9, 0xb8, 0xe6, 158),
            band_edge: Color32::from_rgb(0xe6, 0xdd, 0xf3),
            focus: Color32::from_rgb(0xb0, 0x88, 0xe0),
            error: Color32::from_rgb(0xf2, 0x90, 0x7f),
        }
    } else {
        Palette {
            ground: rgb(brand::PAPER),
            raised: Color32::from_rgb(0xfb, 0xf8, 0xf2),
            field: Color32::from_rgb(0xff, 0xfd, 0xf9),
            ink: Color32::from_rgb(0x1e, 0x1b, 0x22),
            muted: Color32::from_rgb(0x4f, 0x4a, 0x55),
            rule: Color32::from_rgb(0xd6, 0xcd, 0xbe),
            hover: Color32::from_rgb(0xec, 0xe5, 0xd8),
            press: Color32::from_rgb(0xe2, 0xd9, 0xc8),
            caution: Color32::from_rgb(0xf4, 0xdc, 0x96),
            caution_ink: Color32::from_rgb(0x3b, 0x2c, 0x05),
            human_ink: Color32::from_rgb(0x1f, 0x55, 0xb5),
            mixed_ink: Color32::from_rgb(0x6b, 0x3f, 0xa6),
            ai_ink: Color32::from_rgb(0xa8, 0x32, 0x1f),
            primary: Color32::from_rgb(0x1f, 0x55, 0xb5),
            on_primary: Color32::from_rgb(0xfb, 0xf8, 0xf2),
            band_wash: Color32::from_rgba_unmultiplied(0xc9, 0xb8, 0xe6, 199),
            band_edge: Color32::from_rgb(0x6e, 0x61, 0x78),
            focus: rgb(brand::PURPLE),
            error: Color32::from_rgb(0xa8, 0x32, 0x1f),
        }
    }
}

impl Palette {
    /// An origin's ink, for its word.
    pub fn origin_ink(&self, origin: Origin) -> Color32 {
        match origin {
            Origin::Human => self.human_ink,
            Origin::Mixed => self.mixed_ink,
            Origin::Ai => self.ai_ink,
        }
    }
}

fn visuals(dark: bool) -> Visuals {
    let p = palette(dark);
    let mut v = if dark { Visuals::dark() } else { Visuals::light() };
    let radius = CornerRadius::same(RADIUS);
    v.panel_fill = p.ground;
    v.window_fill = p.raised;
    v.window_stroke = Stroke::new(1.0, p.rule);
    v.window_corner_radius = CornerRadius::same(RADIUS + 4);
    v.menu_corner_radius = radius;
    v.extreme_bg_color = p.field;
    v.text_edit_bg_color = Some(p.field);
    v.faint_bg_color = if dark { Color32::from_rgb(0x1d, 0x1a, 0x22) } else { Color32::from_rgb(0xef, 0xe9, 0xdd) };
    v.code_bg_color = p.hover;
    v.override_text_color = Some(p.ink);
    v.weak_text_color = Some(p.muted);
    v.hyperlink_color = p.human_ink;
    v.selection.bg_fill = rgb(brand::mix(brand::PURPLE, if dark { (0x16, 0x14, 0x1a) } else { (255, 255, 255) }, 0.55));
    v.selection.stroke = Stroke::new(2.0, p.focus);
    v.text_cursor.stroke = Stroke::new(2.0, p.ink);
    v.window_shadow = egui::epaint::Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(if dark { 110 } else { 40 }) };
    v.popup_shadow = egui::epaint::Shadow { offset: [0, 4], blur: 12, spread: 0, color: Color32::from_black_alpha(if dark { 90 } else { 30 }) };
    v.striped = false;
    let w = &mut v.widgets;
    w.noninteractive.bg_stroke = Stroke::new(1.0, p.rule);
    w.noninteractive.fg_stroke = Stroke::new(1.0, p.ink);
    w.noninteractive.corner_radius = radius;
    w.noninteractive.bg_fill = p.raised;
    w.noninteractive.weak_bg_fill = p.raised;
    for (state, fill, stroke) in [(&mut w.inactive, p.raised, Stroke::new(1.0, p.rule)), (&mut w.hovered, p.hover, Stroke::new(1.0, p.muted)), (&mut w.active, p.press, Stroke::new(2.0, p.focus)), (&mut w.open, p.hover, Stroke::new(1.0, p.muted))] {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = stroke;
        state.fg_stroke = Stroke::new(1.5, p.ink);
        state.corner_radius = radius;
        state.expansion = 0.0;
    }
    v
}

/// The text styles and spacing, on a 4-point grid. `touch` raises every control to at least 40 points tall (the
/// browser on a phone).
pub fn style_sizes(style: &mut egui::Style, touch: bool) {
    style.text_styles = [
        (TextStyle::Heading, FontId::new(22.0, FontFamily::Name(STRONG.into()))),
        (TextStyle::Body, FontId::new(15.0, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(14.0, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(15.0, FontFamily::Proportional)),
        (TextStyle::Small, FontId::new(12.5, FontFamily::Proportional)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = if touch { egui::vec2(16.0, 10.0) } else { egui::vec2(12.0, 6.0) };
    style.spacing.interact_size = egui::vec2(40.0, if touch { 40.0 } else { 32.0 });
    style.spacing.icon_width = 16.0;
    style.spacing.icon_width_inner = 10.0;
    style.spacing.icon_spacing = 8.0;
    style.spacing.window_margin = egui::Margin::same(20);
    style.spacing.menu_margin = egui::Margin::same(8);
    style.spacing.indent = 20.0;
    style.spacing.scroll.bar_width = 8.0;
    style.spacing.scroll.floating = true;
    style.interaction.selectable_labels = false;
}

/// The fonts: the system's UI faces on Windows (never shipped), egui's bundled faces otherwise.
fn fonts() -> FontDefinitions {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut fonts = FontDefinitions::default();
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut strong = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    #[cfg(windows)]
    {
        use eframe::egui::FontData;
        let folder = std::env::var_os("WINDIR").map_or_else(|| std::path::PathBuf::from("C:\\Windows"), std::path::PathBuf::from).join("Fonts");
        let read = |name: &str| std::fs::read(folder.join(name)).ok();
        let mut add = |key: &str, bytes: Option<Vec<u8>>, family: FontFamily, list: Option<&mut Vec<String>>| {
            let Some(bytes) = bytes else { return false };
            fonts.font_data.insert(key.to_owned(), std::sync::Arc::new(FontData::from_owned(bytes)));
            match list {
                Some(list) => list.insert(0, key.to_owned()),
                None => fonts.families.entry(family).or_default().insert(0, key.to_owned()),
            }
            true
        };
        let regular = read("segoeui.ttf");
        let has_regular = add("segoe_ui", regular, FontFamily::Proportional, None);
        let semibold = read("seguisb.ttf").or_else(|| read("segoeuib.ttf"));
        if !add("segoe_ui_semibold", semibold, FontFamily::Proportional, Some(&mut strong)) && has_regular {
            strong.insert(0, "segoe_ui".to_owned());
        }
        let mono = read("CascadiaMono.ttf").or_else(|| read("consola.ttf"));
        add("system_mono", mono, FontFamily::Monospace, None);
    }
    fonts.families.insert(FontFamily::Name(STRONG.into()), strong);
    fonts
}

/// Install both looks, the fonts, the sizes, and the user's choice of theme.
pub fn install(ctx: &egui::Context, preference: egui::ThemePreference) {
    ctx.set_fonts(fonts());
    ctx.set_visuals_of(Theme::Light, visuals(false));
    ctx.set_visuals_of(Theme::Dark, visuals(true));
    ctx.all_styles_mut(|style| style_sizes(style, false));
    ctx.set_theme(preference);
}

/// A font in the strong (semibold) family.
pub fn strong(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(STRONG.into()))
}

/// The tint of a heat-map cell: the cell's litmus color laid over the ground, stronger toward AI, so lightness alone
/// orders the cells (lighter is more human-like, darker or deeper more AI-like) for readers who cannot tell the hues.
pub fn cell_tint(position: f64, dark: bool) -> Color32 {
    let p = if position.is_finite() { position.clamp(0.0, 1.0) } else { 0.5 };
    let strength = 0.16 + 0.5 * p;
    let ground = if dark { (0x16, 0x14, 0x1a) } else { brand::PAPER };
    rgb(brand::mix(ground, brand::litmus(p), strength))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lum(c: Color32) -> f64 {
        brand::luminance((c.r(), c.g(), c.b()))
    }

    fn contrast(a: Color32, b: Color32) -> f64 {
        let (x, y) = (lum(a), lum(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }

    #[test]
    fn cell_tints_order_by_lightness() {
        for dark in [false, true] {
            let steps: Vec<f64> = (0..=10).map(|i| lum(cell_tint(i as f64 / 10.0, dark))).collect();
            if dark {
                // On the dark ground the AI end is the most saturated and brightest step.
                assert!(steps.windows(2).all(|w| w[1] >= w[0] - 1e-3), "{steps:?}");
            } else {
                assert!(steps.windows(2).all(|w| w[1] < w[0]), "{steps:?}");
            }
            assert!((steps[0] - steps[10]).abs() > 0.05, "the ends differ clearly in lightness: {steps:?}");
        }
    }

    #[test]
    fn inks_read_on_every_ground() {
        for dark in [false, true] {
            let p = palette(dark);
            for ink in [p.ink, p.muted, p.human_ink, p.mixed_ink, p.ai_ink] {
                for ground in [p.ground, p.raised, p.field] {
                    assert!(contrast(ink, ground) >= 4.5, "dark {dark}: {ink:?} on {ground:?} is {:.2}:1", contrast(ink, ground));
                }
            }
            assert!(contrast(p.on_primary, p.primary) >= 4.5, "the primary button's text reads");
            assert!(contrast(p.caution_ink, p.caution) >= 4.5, "the caution banner reads");
        }
    }
}
