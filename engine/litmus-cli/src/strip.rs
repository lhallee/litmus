//! The litmus strip in a terminal: blue (human) through purple to red (AI), a mark at the reading's chance of AI, and
//! a pale band over the chance's range (`litmus_engine::Drawn`). Colors are the site's (`site/style.css`): blue
//! `#2F6FDB`, purple `#8E5BC9`, red `#D9483B`, band `#C9B8E6`. Without color the same strip is drawn in characters;
//! the words beside it always carry the verdict. A cell is 5 points of chance wide, so a range narrower than that
//! shows as the mark's cell alone.

use litmus_engine::Drawn;

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
pub fn render(strip: &Drawn, colored: bool) -> String {
    let mark = strip.mark.map(|position| (cell_of(position), position));
    let band = strip.band.map(|[lo, hi]| (cell_of(lo), cell_of(hi)));
    let mut out = String::new();
    for cell in 0..WIDTH {
        let center = (cell as f64 + 0.5) / WIDTH as f64;
        let in_band = band.is_some_and(|(lo, hi)| cell >= lo && cell <= hi);
        let at_mark = mark.filter(|&(at, _)| at == cell);
        if colored {
            let hue = color(center);
            let background = if in_band { mix(hue, BAND, 0.55) } else { mix(hue, DIM, 0.8) };
            if let Some((_, position)) = at_mark {
                let ink = color(position);
                out.push_str(&format!("\x1b[48;2;{};{};{}m\x1b[38;2;{};{};{}m\u{2588}", background.0, background.1, background.2, ink.0, ink.1, ink.2));
            } else {
                out.push_str(&format!("\x1b[48;2;{};{};{}m ", background.0, background.1, background.2));
            }
        } else {
            out.push(if at_mark.is_some() { '|' } else if in_band { '=' } else { '-' });
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
    fn plain_strip_marks_the_chance_and_its_range() {
        let strip = Drawn { mark: Some(0.8), band: Some([0.5, 1.0]), unlike_reference: false };
        assert_eq!(render(&strip, false), "----------======|===  ");
        // A range narrower than a cell is the mark's cell alone, and a text unlike the reference has no band.
        assert_eq!(render(&Drawn { mark: Some(0.56), band: Some([0.558, 0.562]), unlike_reference: false }, false), "-----------|--------  ");
        assert_eq!(render(&Drawn { mark: Some(0.56), band: None, unlike_reference: true }, false), "-----------|-------- ?");
        assert_eq!(render(&Drawn { mark: None, band: None, unlike_reference: false }, false), "--------------------  ");
        assert_eq!(color(0.0), BLUE);
        assert_eq!(color(0.5), PURPLE);
        assert_eq!(color(1.0), RED);
    }
}
