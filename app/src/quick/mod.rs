//! Quick check (desktop, opt-in, off by default): select text in any app, press a hotkey, and a small window by the
//! pointer shows the reading. Switched on only from Settings; while on, the app stays in the background with a tray
//! icon (Check copied text, Open Litmus, Pause, Quit) and a global hotkey the user can change.
//!
//! The selected text is read locally, through the platform's accessibility interface where the foreground app
//! offers it (UI Automation's text pattern on Windows), else by simulating Copy with the clipboard saved first and
//! restored after. Nothing leaves the machine. Per platform: `windows.rs` (built and tested on Windows), `unix.rs`
//! (macOS and Linux: Linux on X11 built and run, macOS never compiled).
//!
//! On Windows, Settings can also add a "Check with Litmus" entry to Explorer's right-click menu for the file kinds
//! the app reads (per-user registry keys, removed again from Settings). There is no general way to add an item to
//! every application's own right-click menu; the hotkey is that way.

#[cfg(windows)]
pub mod windows;
#[cfg(all(unix, not(target_arch = "wasm32")))]
pub mod unix;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use eframe::egui::ViewportId;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

/// The default hotkey.
pub const DEFAULT_HOTKEY: &str = "Ctrl+Alt+L";
/// The popup closes by itself after this long unless the pointer rests on it.
pub const POPUP_SECONDS: u64 = 25;

/// The one sentence Settings shows beside the switch.
#[cfg(target_os = "linux")]
pub const SENTENCE: &str = "Quick check lets a hotkey read the text selected in any app while Litmus is open and show its reading \
in a small window; the selection is read as it is, so nothing is copied, and it never leaves this computer. The hotkey works \
on X11, not on Wayland.";
/// The one sentence Settings shows beside the switch.
#[cfg(not(target_os = "linux"))]
pub const SENTENCE: &str = "Quick check keeps Litmus running in the background (tray icon) so that a hotkey reads the text \
selected in any app and shows its reading in a small window by the pointer; the text is read from the app's accessibility \
interface, or else by copying it with your clipboard saved first and put back after, and it never leaves this computer.";

/// A hotkey from its words ("Ctrl+Alt+L"). At least one of Ctrl, Alt or the Windows / Command key is required, so
/// the hotkey never takes over plain typing.
pub fn parse_hotkey(words: &str) -> Result<HotKey, String> {
    let hotkey = HotKey::try_from(words.trim()).map_err(|e| format!("\"{words}\" is not a hotkey: {e}"))?;
    let modifiers = hotkey.mods;
    use global_hotkey::hotkey::Modifiers;
    if !modifiers.intersects(Modifiers::CONTROL | Modifiers::ALT | Modifiers::META) {
        return Err(format!("\"{words}\" needs Ctrl, Alt or the Windows key, so that it does not take over plain typing"));
    }
    Ok(hotkey)
}

/// Text the hotkey captured, and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Captured {
    pub text: String,
    pub how: How,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    Accessibility,
    Copy,
    Clipboard,
}

/// What the popup shows.
pub enum PopupState {
    /// Captured while the model was still loading: read as soon as it is ready, with no second press.
    Waiting { text: String, how: How },
    /// The reading's job, and its progress in words once the engine reports some.
    Scoring { job: u64, how: How, progress: String },
    Shown { scored: crate::scoring::Scored, how: How },
    Failed(String),
}

/// What keeps a popup up besides the pointer resting on it: nothing for a person's, more for the scripted demo's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    Nothing,
    /// The clock still closes it; the focus, Esc and clicks do not (a person at the desk may click while a script
    /// waits for the clock).
    AgainstInput,
    /// Up whatever the focus, the input and the clock (the demo's screenshot).
    Always,
}

pub struct Popup {
    pub state: PopupState,
    /// Where to open, in the main window's points on the screen.
    pub at: eframe::egui::Pos2,
    pub opened: Instant,
    /// Since when the popup has held the focus. A window that briefly gets the focus and loses it at once (Windows
    /// keeps the app in front when another process opens a window) is not closed by that.
    pub focused_since: Option<Instant>,
    pub keep: Keep,
    /// Unique to each change of `state`, so the popup's window is redrawn when it changes and only then.
    pub revision: u64,
    /// How long a reading stays up unless the pointer rests on it (`POPUP_SECONDS`; the scripted demo's is shorter).
    pub lifetime: Duration,
    /// Where it is on the screen, in physical pixels: left, top, right, bottom.
    pub screen: Option<[i32; 4]>,
    /// Esc and every mouse button have been seen up since it opened (`dismissed_unfocused`).
    armed: bool,
}

static REVISIONS: AtomicU64 = AtomicU64::new(1);

impl Popup {
    pub fn new(state: PopupState, at: eframe::egui::Pos2) -> Self {
        Self {
            state,
            at,
            opened: Instant::now(),
            focused_since: None,
            keep: Keep::Nothing,
            revision: REVISIONS.fetch_add(1, Ordering::Relaxed),
            lifetime: Duration::from_secs(POPUP_SECONDS),
            screen: None,
            armed: false,
        }
    }

    /// While the popup does not hold the focus (Windows opens a window without it when the app was in the background
    /// and had no right to take it), Esc and a click elsewhere never reach it; it watches their state instead. True
    /// once Esc, or a mouse button with the pointer outside the popup, goes down after all were seen up since it
    /// opened, so a key or button already held then does not close it. Neither is taken from the app in front.
    pub fn dismissed_unfocused(&mut self, escape: bool, button: bool, pointer: Option<(i32, i32)>) -> bool {
        if !escape && !button {
            self.armed = true;
            return false;
        }
        let over = pointer.zip(self.screen).is_some_and(|((x, y), [left, top, right, bottom])| (left..right).contains(&x) && (top..bottom).contains(&y));
        self.armed && self.keep == Keep::Nothing && (escape || (button && !over))
    }

    /// Show another state, restarting the clock that closes the popup.
    pub fn set(&mut self, state: PopupState) {
        self.state = state;
        self.opened = Instant::now();
        self.touch();
    }

    /// Mark the state as changed in place (progress), so the popup's window is redrawn.
    pub fn touch(&mut self) {
        self.revision = REVISIONS.fetch_add(1, Ordering::Relaxed);
    }

    /// The reading's job while it is still being read (to cancel when the popup goes).
    pub fn scoring_job(&self) -> Option<u64> {
        match self.state {
            PopupState::Scoring { job, .. } => Some(job),
            _ => None,
        }
    }

    pub fn expired(&self, hovered: bool) -> bool {
        self.keep != Keep::Always && !hovered && self.opened.elapsed() > self.lifetime
    }

    /// Track the focus; true when the popup held it for a moment and then lost it (a click elsewhere).
    pub fn lost_focus(&mut self, focused: Option<bool>) -> bool {
        match focused {
            Some(true) => {
                self.focused_since.get_or_insert_with(Instant::now);
                false
            }
            Some(false) => {
                let held = self.focused_since.take().is_some_and(|since| since.elapsed() > Duration::from_millis(600));
                held && self.keep == Keep::Nothing
            }
            None => false,
        }
    }
}

/// What the tray menu asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrayAction {
    CheckCopied,
    Open,
    Pause,
    Quit,
}

/// The hotkey and the tray while quick check is on.
pub struct Quick {
    manager: GlobalHotKeyManager,
    hotkey: HotKey,
    pub paused: bool,
    #[cfg(not(target_os = "linux"))]
    tray: Option<Tray>,
    capture: Option<mpsc::Receiver<Result<Captured, String>>>,
    pub popup: Option<Popup>,
}

// Linux has no tray: tray-icon needs a GTK loop the app does not run, so it is not a dependency there (Cargo.toml).
#[cfg(not(target_os = "linux"))]
struct Tray {
    _icon: tray_icon::TrayIcon,
    pause: tray_icon::menu::MenuItem,
    ids: Vec<(tray_icon::menu::MenuId, TrayAction)>,
}

#[cfg(not(target_os = "linux"))]
fn tray(ctx: &eframe::egui::Context) -> Result<Tray, String> {
    use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
    let menu = Menu::new();
    let check = MenuItem::new("Check copied text", true, None);
    let open = MenuItem::new(format!("Open {}", crate::brand::NAME), true, None);
    let pause = MenuItem::new("Pause quick check", true, None);
    let quit = MenuItem::new("Quit", true, None);
    let separator = PredefinedMenuItem::separator();
    for item in [&check as &dyn tray_icon::menu::IsMenuItem, &open, &pause, &separator, &quit] {
        menu.append(item).map_err(|e| e.to_string())?;
    }
    let size = 32;
    let icon = tray_icon::Icon::from_rgba(crate::brand::icon_rgba(size), size as u32, size as u32).map_err(|e| e.to_string())?;
    let built = tray_icon::TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip(format!("{}: quick check is on", crate::brand::NAME))
        .with_icon(icon)
        .build()
        .map_err(|e| e.to_string())?;
    let waker = ctx.clone();
    MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
        MENU_EVENTS.lock().expect("the tray's queue").push(event.id);
        waker.request_repaint_of(ViewportId::ROOT);
    }));
    let ids = vec![(check.id().clone(), TrayAction::CheckCopied), (open.id().clone(), TrayAction::Open), (pause.id().clone(), TrayAction::Pause), (quit.id().clone(), TrayAction::Quit)];
    Ok(Tray { _icon: built, pause, ids })
}

#[cfg(not(target_os = "linux"))]
static MENU_EVENTS: std::sync::Mutex<Vec<tray_icon::menu::MenuId>> = std::sync::Mutex::new(Vec::new());
static HOTKEY_EVENTS: std::sync::Mutex<Vec<GlobalHotKeyEvent>> = std::sync::Mutex::new(Vec::new());

impl Quick {
    /// Register the hotkey and show the tray icon. Called only when the user's switch in Settings is on.
    pub fn start(ctx: &eframe::egui::Context, hotkey_words: &str) -> Result<Self, String> {
        let hotkey = parse_hotkey(hotkey_words)?;
        let manager = GlobalHotKeyManager::new().map_err(|e| format!("the hotkey service did not start: {e}"))?;
        manager.register(hotkey).map_err(|e| format!("{hotkey_words} could not be registered (another app may hold it): {e}"))?;
        // Every waker names the main window: its logic drains these queues, and a plain `request_repaint` from another
        // thread reaches whichever window is mid-pass, the popup's included.
        let waker = ctx.clone();
        GlobalHotKeyEvent::set_event_handler(Some(move |event: GlobalHotKeyEvent| {
            HOTKEY_EVENTS.lock().expect("the hotkey queue").push(event);
            waker.request_repaint_of(ViewportId::ROOT);
        }));
        #[cfg(not(target_os = "linux"))]
        let tray = Some(tray(ctx)?);
        Ok(Self {
            manager,
            hotkey,
            paused: false,
            #[cfg(not(target_os = "linux"))]
            tray,
            capture: None,
            popup: None,
        })
    }

    pub fn set_paused(&mut self, paused: bool) -> Result<(), String> {
        if paused == self.paused {
            return Ok(());
        }
        if paused {
            self.manager.unregister(self.hotkey).map_err(|e| e.to_string())?;
        } else {
            self.manager.register(self.hotkey).map_err(|e| e.to_string())?;
        }
        self.paused = paused;
        #[cfg(not(target_os = "linux"))]
        if let Some(tray) = &self.tray {
            tray.pause.set_text(if paused { "Resume quick check" } else { "Pause quick check" });
        }
        Ok(())
    }

    /// Whether the hotkey was pressed since the last call.
    pub fn take_hotkey(&self) -> bool {
        let events: Vec<GlobalHotKeyEvent> = HOTKEY_EVENTS.lock().expect("the hotkey queue").drain(..).collect();
        events.iter().any(|e| e.id() == self.hotkey.id() && e.state() == HotKeyState::Pressed) && !self.paused
    }

    #[cfg(target_os = "linux")]
    pub fn take_tray(&self) -> Vec<TrayAction> {
        Vec::new()
    }

    #[cfg(not(target_os = "linux"))]
    pub fn take_tray(&self) -> Vec<TrayAction> {
        let Some(tray) = &self.tray else { return Vec::new() };
        let ids: Vec<tray_icon::menu::MenuId> = MENU_EVENTS.lock().expect("the tray's queue").drain(..).collect();
        ids.iter().filter_map(|id| tray.ids.iter().find(|(known, _)| known == id).map(|(_, action)| *action)).collect()
    }

    /// Start reading the selection (or the clipboard) on a thread of its own; `poll_capture` gives the result.
    pub fn begin_capture(&mut self, ctx: &eframe::egui::Context, copied_only: bool) {
        let (sender, receiver) = mpsc::channel();
        let waker = ctx.clone();
        std::thread::spawn(move || {
            let result = if copied_only { platform::clipboard_text().map(|text| Captured { text, how: How::Clipboard }) } else { platform::selected_text() };
            let _ = sender.send(result);
            waker.request_repaint_of(ViewportId::ROOT);
        });
        self.capture = Some(receiver);
    }

    /// A capture is under way. Another press waits for it: two copies at once would each save the clipboard, and the
    /// second would save the first's copied selection and put that back instead of the user's own contents.
    pub fn capturing(&self) -> bool {
        self.capture.is_some()
    }

    pub fn poll_capture(&mut self) -> Option<Result<Captured, String>> {
        let result = self.capture.as_ref()?.try_recv().ok()?;
        self.capture = None;
        Some(result)
    }
}

impl Drop for Quick {
    fn drop(&mut self) {
        let _ = self.manager.unregister(self.hotkey);
        GlobalHotKeyEvent::set_event_handler(None::<fn(GlobalHotKeyEvent)>);
        #[cfg(not(target_os = "linux"))]
        tray_icon::menu::MenuEvent::set_event_handler(None::<fn(tray_icon::menu::MenuEvent)>);
    }
}

/// The pointer's place on the screen in physical pixels, where the platform says.
pub fn pointer() -> Option<(i32, i32)> {
    platform::pointer()
}

/// Whether Esc, and whether a mouse button, is down now (Windows; elsewhere never).
pub fn dismiss_keys() -> (bool, bool) {
    platform::dismiss_keys()
}

/// Whether this process shows a window with this title (Windows; elsewhere always false).
pub fn shows_window(title: &str) -> bool {
    platform::shows_window(title)
}

/// The work area (the screen less the taskbar) of the monitor nearest a point, in physical pixels: left, top, right,
/// bottom. `None` where the platform does not say.
pub fn work_area_at(point: (i32, i32)) -> Option<[i32; 4]> {
    platform::work_area_at(point)
}

/// Where the popup's top-left corner goes, in physical pixels: `gap` below and to the right of the pointer, flipped
/// above or to the left of it where the work area would cut it off, and always inside the work area, so a selection
/// near a screen's edge never opens a popup half off the screen.
pub fn popup_origin(pointer: (i32, i32), size: (i32, i32), gap: (i32, i32), work: Option<[i32; 4]>) -> (i32, i32) {
    let (mut x, mut y) = (pointer.0 + gap.0, pointer.1 + gap.1);
    let Some([left, top, right, bottom]) = work else { return (x, y) };
    if x + size.0 > right {
        x = pointer.0 - gap.0 - size.0;
    }
    if y + size.1 > bottom {
        y = pointer.1 - gap.1 - size.1;
    }
    (x.clamp(left, (right - size.0).max(left)), y.clamp(top, (bottom - size.1).max(top)))
}

#[cfg(windows)]
use self::windows as platform;
#[cfg(all(unix, not(target_arch = "wasm32")))]
use self::unix as platform;

/// The Explorer entry (Windows only; elsewhere an error that says so).
pub fn set_explorer_entry(on: bool) -> Result<(), String> {
    #[cfg(windows)]
    {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        if on { windows::add_explorer_entry(windows::CLASSES, &exe) } else { windows::remove_explorer_entry(windows::CLASSES) }
    }
    #[cfg(not(windows))]
    {
        let _ = on;
        Err("the file-manager entry exists on Windows only; on macOS a Services entry is the native route (not built yet)".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotkeys_need_a_modifier() {
        assert!(parse_hotkey("Ctrl+Alt+L").is_ok());
        assert!(parse_hotkey(" ctrl+shift+K ").is_ok());
        assert!(parse_hotkey("Super+Digit5").is_ok());
        assert!(parse_hotkey("L").unwrap_err().contains("needs Ctrl"));
        assert!(parse_hotkey("Shift+L").unwrap_err().contains("needs Ctrl"));
        assert!(parse_hotkey("Ctrl+Nope").is_err());
        assert_eq!(parse_hotkey("Ctrl+Alt+L").unwrap().id(), parse_hotkey("alt+ctrl+l").unwrap().id());
    }

    #[test]
    fn a_popup_without_the_focus_closes_on_esc_or_a_click_elsewhere() {
        let popup = || {
            let mut popup = Popup::new(PopupState::Failed("test".into()), eframe::egui::pos2(0.0, 0.0));
            popup.screen = Some([100, 100, 480, 276]);
            popup
        };
        let (inside, outside) = (Some((200, 150)), Some((50, 50)));
        let mut held = popup();
        assert!(!held.dismissed_unfocused(true, false, outside), "Esc already down when the popup opened");
        assert!(!held.dismissed_unfocused(false, true, outside), "a button already down when it opened");
        assert!(!held.dismissed_unfocused(false, false, outside), "all up");
        assert!(held.dismissed_unfocused(true, false, inside), "then Esc closes it, wherever the pointer is");
        let mut click = popup();
        click.dismissed_unfocused(false, false, outside);
        assert!(!click.dismissed_unfocused(false, true, inside), "a click on the popup gives it the focus instead");
        assert!(click.dismissed_unfocused(false, true, outside), "a click elsewhere closes it");
        for keep in [Keep::AgainstInput, Keep::Always] {
            let mut kept = popup();
            kept.keep = keep;
            kept.dismissed_unfocused(false, false, outside);
            assert!(!kept.dismissed_unfocused(true, true, outside), "the demo's popup kept {keep:?} stays");
        }
    }

    #[test]
    fn the_popup_stays_on_the_screen() {
        let work = Some([0, 0, 1920, 1040]); // a 1080-pixel screen less a 40-pixel taskbar
        let (size, gap) = ((380, 176), (14, 18));
        assert_eq!(popup_origin((500, 400), size, gap, work), (514, 418), "below and right of the pointer");
        assert_eq!(popup_origin((1800, 400), size, gap, work), (1800 - 14 - 380, 418), "left of a pointer near the right edge");
        assert_eq!(popup_origin((500, 1000), size, gap, work), (514, 1000 - 18 - 176), "above a pointer near the taskbar");
        assert_eq!(popup_origin((1900, 1030), size, gap, work), (1900 - 14 - 380, 1030 - 18 - 176), "both, in the corner");
        let second = Some([1920, -200, 3840, 880]); // a second monitor to the right, set higher
        assert_eq!(popup_origin((1925, -195), size, gap, second), (1939, -177), "on the monitor it is on, negative coordinates included");
        assert_eq!(popup_origin((10, 10), size, gap, Some([0, 0, 300, 100])), (0, 0), "a work area smaller than the popup keeps its top left");
        assert_eq!(popup_origin((500, 400), size, gap, None), (514, 418), "no work area known: by the pointer");
    }
}
