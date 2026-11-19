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

use std::sync::mpsc;
use std::time::{Duration, Instant};

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
    Scoring { job: u64, how: How },
    Shown { scored: crate::scoring::Scored, how: How },
    Failed(String),
}

pub struct Popup {
    pub state: PopupState,
    /// Where to open, in the main window's points on the screen.
    pub at: eframe::egui::Pos2,
    pub opened: Instant,
    /// Since when the popup has held the focus. A window that briefly gets the focus and loses it at once (Windows
    /// keeps the app in front when another process opens a window) is not closed by that.
    pub focused_since: Option<Instant>,
    /// Kept open whatever the focus and the clock (the scripted demo's screenshot).
    pub pinned: bool,
}

impl Popup {
    pub fn expired(&self, hovered: bool) -> bool {
        !self.pinned && !hovered && self.opened.elapsed() > Duration::from_secs(POPUP_SECONDS)
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
                held && !self.pinned
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
        waker.request_repaint();
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
        let waker = ctx.clone();
        GlobalHotKeyEvent::set_event_handler(Some(move |event: GlobalHotKeyEvent| {
            HOTKEY_EVENTS.lock().expect("the hotkey queue").push(event);
            waker.request_repaint();
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
            waker.request_repaint();
        });
        self.capture = Some(receiver);
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
}
