//! Where the main window was (desktop): its outer position and inner size in points, and whether it was maximized,
//! saved in the settings when it settles after a move or a resize and when the app closes, and restored at the next
//! launch. A saved place that is not a sane size, or (on Windows) whose title bar would land on no monitor, is not
//! used, so a disconnected screen never hides the window.

use serde::{Deserialize, Serialize};

/// The smallest window the app lays out for.
pub const MIN_SIZE: [f32; 2] = [420.0, 360.0];
/// The first launch's size.
pub const DEFAULT_SIZE: [f32; 2] = [1200.0, 800.0];
/// Larger than any screen: a saved size past this is damaged.
const MAX_SIDE: f32 = 16_384.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowPlace {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub maximized: bool,
}

impl WindowPlace {
    /// The place to restore, with the size raised to the minimum; `None` when the numbers are not a window's.
    pub fn sane(self) -> Option<Self> {
        let finite = [self.x, self.y, self.width, self.height].iter().all(|v| v.is_finite());
        if !finite || self.width > MAX_SIDE || self.height > MAX_SIDE || self.x.abs() > MAX_SIDE * 4.0 || self.y.abs() > MAX_SIDE * 4.0 {
            return None;
        }
        Some(Self { width: self.width.max(MIN_SIZE[0]), height: self.height.max(MIN_SIZE[1]), ..self })
    }

    /// Whether two places differ by more than half a point anywhere (a move or a resize, not rounding).
    pub fn differs(&self, other: &Self) -> bool {
        self.maximized != other.maximized
            || [(self.x, other.x), (self.y, other.y), (self.width, other.width), (self.height, other.height)].iter().any(|(a, b)| (a - b).abs() > 0.5)
    }

    /// The window's place from egui's view of it; `None` while it is minimized, hidden, or not yet placed.
    pub fn of(viewport: &eframe::egui::ViewportInfo) -> Option<Self> {
        if viewport.minimized == Some(true) || viewport.visible() == Some(false) {
            return None;
        }
        let maximized = viewport.maximized == Some(true);
        let (outer, inner) = (viewport.outer_rect?, viewport.inner_rect?);
        Some(Self { x: outer.min.x, y: outer.min.y, width: inner.width(), height: inner.height(), maximized })
    }

    /// Whether the window's title bar would be on a screen (Windows asks the system; elsewhere it is assumed).
    pub fn on_a_screen(&self) -> bool {
        #[cfg(windows)]
        {
            use windows::Win32::Foundation::POINT;
            use windows::Win32::Graphics::Gdi::{MONITOR_DEFAULTTONULL, MonitorFromPoint};
            // SAFETY: plain Win32 queries with values, no pointers kept.
            let dpi = unsafe { windows::Win32::UI::HiDpi::GetDpiForSystem() }.max(96) as f32 / 96.0;
            let point = POINT { x: ((self.x + self.width / 2.0) * dpi) as i32, y: ((self.y + 12.0) * dpi) as i32 };
            // SAFETY: as above.
            !unsafe { MonitorFromPoint(point, MONITOR_DEFAULTTONULL) }.is_invalid()
        }
        #[cfg(not(windows))]
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn place(x: f32, y: f32, width: f32, height: f32) -> WindowPlace {
        WindowPlace { x, y, width, height, maximized: false }
    }

    #[test]
    fn saved_places_are_checked() {
        assert_eq!(place(10.0, 20.0, 900.0, 700.0).sane(), Some(place(10.0, 20.0, 900.0, 700.0)));
        assert_eq!(place(10.0, 20.0, 100.0, 50.0).sane(), Some(place(10.0, 20.0, MIN_SIZE[0], MIN_SIZE[1])), "a tiny window is raised to the minimum");
        assert_eq!(place(f32::NAN, 0.0, 900.0, 700.0).sane(), None);
        assert_eq!(place(0.0, 0.0, 90_000.0, 700.0).sane(), None);
        assert_eq!(place(-1.0e7, 0.0, 900.0, 700.0).sane(), None);
    }

    #[test]
    fn small_jitter_is_not_a_move() {
        let a = place(10.0, 20.0, 900.0, 700.0);
        assert!(!a.differs(&place(10.3, 19.8, 900.4, 700.0)));
        assert!(a.differs(&place(14.0, 20.0, 900.0, 700.0)));
        assert!(a.differs(&WindowPlace { maximized: true, ..a }));
    }
}
