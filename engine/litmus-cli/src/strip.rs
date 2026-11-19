//! The litmus strip in a terminal: blue (human) through purple (mixed) to red (AI), a pale band over what the
//! conformal set cannot rule out, and a mark at the reading's position. Colors are the site's (`site/style.css`): blue
//! `#2F6FDB`, purple `#8E5BC9`, red `#D9483B`, band `#C9B8E6`. Without color the same strip is drawn in characters;
//! the words beside it always carry the verdict.

use litmus_engine::Strip;

const BLUE: (u8, u8, u8) = (0x2f, 0x6f, 0xdb);
const PURPLE: (u8, u8, u8) = (0x8e, 0x5b, 0xc9);
const RED: (u8, u8, u8) = (0xd9, 0x48, 0x3b);
const BAND: (u8, u8, u8) = (0xc9, 0xb8, 0xe6);
const DIM: (u8, u8, u8) = (0x40, 0x40, 0x46);

/// Cells of the strip.
pub const WIDTH: usize = 20;

fn mix(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let channel = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
    (channel(a.0, b.0), channel(a.1, b.1), channel(a.2, b.2))
}

/// The litmus color at a position in [0, 1].
pub fn color(position: f64) -> (u8, u8, u8) {
    let x = position.clamp(0.0, 1.0);
    if x <= 0.5 { mix(BLUE, PURPLE, x / 0.5) } else { mix(PURPLE, RED, (x - 0.5) / 0.5) }
}

fn cell_of(x: f64) -> usize {
    ((x.clamp(0.0, 1.0) * WIDTH as f64) as usize).min(WIDTH - 1)
}

/// The strip as terminal text, `WIDTH` cells wide.
pub fn render(strip: &Strip, colored: bool) -> String {
    let mark = cell_of(strip.position);
    let (band_lo, band_hi) = (cell_of(strip.band_lo), cell_of(strip.band_hi));
    let mut out = String::new();
    for cell in 0..WIDTH {
        let center = (cell as f64 + 0.5) / WIDTH as f64;
        let in_band = cell >= band_lo && cell <= band_hi;
        if colored {
            let hue = color(center);
            let background = if in_band { mix(hue, BAND, 0.55) } else { mix(hue, DIM, 0.8) };
            if cell == mark {
                let ink = color(strip.position);
                out.push_str(&format!("\x1b[48;2;{};{};{}m\x1b[38;2;{};{};{}m\u{2588}", background.0, background.1, background.2, ink.0, ink.1, ink.2));
            } else {
                out.push_str(&format!("\x1b[48;2;{};{};{}m ", background.0, background.1, background.2));
            }
        } else {
            out.push(if cell == mark { '|' } else if in_band { '=' } else { '-' });
        }
    }
    if colored {
        out.push_str("\x1b[0m");
    }
    if strip.unlike_reference {
        out.push_str(if colored { " \x1b[2m?\x1b[0m" } else { " ?" });
    } else {
        out.push_str("  ");
    }
    out
}

/// Whether to color: standard output is a terminal, `NO_COLOR` is unset, and (on Windows) the console accepts
/// escape sequences.
pub fn color_supported() -> bool {
    use std::io::IsTerminal;
    if std::env::var_os("NO_COLOR").is_some() || !std::io::stdout().is_terminal() {
        return false;
    }
    enable_escapes()
}

#[cfg(windows)]
fn enable_escapes() -> bool {
    // SAFETY: plain Win32 calls on this process's own standard output handle.
    unsafe extern "system" {
        fn GetStdHandle(handle: u32) -> *mut core::ffi::c_void;
        fn GetConsoleMode(handle: *mut core::ffi::c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: *mut core::ffi::c_void, mode: u32) -> i32;
    }
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode = 0u32;
        if GetConsoleMode(handle, &mut mode) == 0 {
            return false;
        }
        mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0 || SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
    }
}

#[cfg(not(windows))]
fn enable_escapes() -> bool {
    std::env::var("TERM").map_or(true, |term| term != "dumb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_strip_marks_the_band_and_position() {
        let strip = Strip { position: 0.8, band_lo: 0.5, band_hi: 1.0, level: 0.1, unlike_reference: false };
        assert_eq!(render(&strip, false), "----------======|===  ");
        assert_eq!(color(0.0), BLUE);
        assert_eq!(color(0.5), PURPLE);
        assert_eq!(color(1.0), RED);
    }
}
