//! Quick check on Windows: the selected text through UI Automation, else by simulated Copy with the clipboard saved
//! and restored; the pointer's place; and the Explorer entry under `HKEY_CURRENT_USER\Software\Classes`.

use std::path::Path;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, HGLOBAL, HWND, POINT};
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromPoint};
use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData, GetClipboardSequenceNumber, IsClipboardFormatAvailable, OpenClipboard,
    RegisterClipboardFormatW, SetClipboardData,
};
use windows::Win32::System::Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock};
use windows::Win32::System::Registry::{HKEY, HKEY_CURRENT_USER, KEY_READ, KEY_WRITE, REG_OPTION_NON_VOLATILE, REG_SZ, RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegOpenKeyExW, RegSetValueExW};
use windows::Win32::UI::Accessibility::{CUIAutomation, IUIAutomation, IUIAutomationElement, IUIAutomationTextPattern, UIA_TextPatternId};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT, KEYEVENTF_KEYUP, SendInput, VIRTUAL_KEY, VK_C, VK_CONTROL, VK_LWIN, VK_MENU,
    VK_RWIN, VK_SHIFT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, FindWindowExW, GetCursorPos, GetForegroundWindow, GetWindowThreadProcessId, HWND_MESSAGE, IsWindowVisible, WINDOW_EX_STYLE,
    WINDOW_STYLE,
};
use windows::core::{HSTRING, PCWSTR, w};

use super::{Captured, How};

const CF_UNICODETEXT: u32 = 13;
/// Clipboard formats that hold GDI handles, not memory: they cannot be copied byte for byte (an image also sits on
/// the clipboard as `CF_DIB`, which can).
const HANDLE_FORMATS: [u32; 6] = [2, 3, 9, 14, 0x0080, 0x0082];

/// The file kinds Explorer offers "Check with Litmus" for.
pub const EXTENSIONS: [&str; 7] = ["txt", "md", "markdown", "html", "htm", "docx", "pdf"];
/// The per-user classes key the Explorer entry lives under.
pub const CLASSES: &str = "Software\\Classes";
const VERB: &str = "LitmusCheck";

/// The text selected in the foreground app: UI Automation first, then simulated Copy.
pub fn selected_text() -> Result<Captured, String> {
    if let Some(text) = accessibility_selection().filter(|t| !t.trim().is_empty()) {
        return Ok(Captured { text, how: How::Accessibility });
    }
    let text = copy_selection()?;
    if text.trim().is_empty() {
        return Err("No selected text was found. Select some text, then press the hotkey.".into());
    }
    Ok(Captured { text, how: How::Copy })
}

/// The clipboard's text as it is (the tray's "Check copied text").
pub fn clipboard_text() -> Result<String, String> {
    let _open = Opened::new(None)?;
    unicode_text().filter(|t| !t.trim().is_empty()).ok_or_else(|| "The clipboard holds no text.".to_owned())
}

pub fn pointer() -> Option<(i32, i32)> {
    let mut point = POINT::default();
    // SAFETY: GetCursorPos writes the POINT it is given.
    unsafe { GetCursorPos(&mut point).ok()? };
    Some((point.x, point.y))
}

/// Whether Esc, and whether a mouse button, is down now: read from the keys' state, never taken from the app in front
/// (a popup without the focus watches them so that Esc and a click elsewhere still close it).
pub fn dismiss_keys() -> (bool, bool) {
    use windows::Win32::UI::Input::KeyboardAndMouse::{VK_ESCAPE, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON};
    (held(VK_ESCAPE), [VK_LBUTTON, VK_RBUTTON, VK_MBUTTON].into_iter().any(held))
}

/// Whether this process shows a top-level window with this title (the scripted demo checks that a closed popup's
/// window is gone from the screen, not only from the app's state).
pub fn shows_window(title: &str) -> bool {
    let title = HSTRING::from(title);
    let mut after = None;
    // SAFETY: window lookups by title, and reads of each found window's process and visibility.
    unsafe {
        while let Ok(window) = FindWindowExW(None, after, None, &title) {
            let mut process = 0u32;
            GetWindowThreadProcessId(window, Some(&mut process));
            if process == std::process::id() && IsWindowVisible(window).as_bool() {
                return true;
            }
            after = Some(window);
        }
    }
    false
}

/// The work area of the monitor nearest a point, in physical pixels.
pub fn work_area_at((x, y): (i32, i32)) -> Option<[i32; 4]> {
    let mut info = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
    // SAFETY: MonitorFromPoint takes values; GetMonitorInfoW writes the MONITORINFO it is given, its size set above.
    unsafe { GetMonitorInfoW(MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONEAREST), &mut info).ok().ok()? };
    let work = info.rcWork;
    Some([work.left, work.top, work.right, work.bottom])
}

fn automation() -> Option<IUIAutomation> {
    // SAFETY: COM is initialized for this thread before the instance is made; a thread already initialized is fine.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).ok()
    }
}

/// The selection of the focused element, through its text pattern, if it has one.
fn accessibility_selection() -> Option<String> {
    let automation = automation()?;
    // SAFETY: plain COM calls on interfaces this thread created.
    let element = unsafe { automation.GetFocusedElement().ok()? };
    selection_of(&element).ok()
}

/// The selected text of an element that offers UI Automation's text pattern, or which step said no.
pub fn selection_of(element: &IUIAutomationElement) -> Result<String, String> {
    // SAFETY: plain COM calls on live interfaces.
    unsafe {
        let pattern: IUIAutomationTextPattern = element.GetCurrentPatternAs(UIA_TextPatternId).map_err(|e| format!("no text pattern: {e}"))?;
        let ranges = pattern.GetSelection().map_err(|e| format!("no selection: {e}"))?;
        let mut text = String::new();
        for index in 0..ranges.Length().map_err(|e| e.to_string())? {
            let range = ranges.GetElement(index).map_err(|e| e.to_string())?;
            text.push_str(&range.GetText(-1).map_err(|e| e.to_string())?.to_string());
        }
        Ok(text)
    }
}

/// The selected text of a window (tests drive a window of their own this way).
pub fn selection_of_window(hwnd: HWND) -> Result<String, String> {
    let automation = automation().ok_or("UI Automation is not available")?;
    // SAFETY: plain COM call; the window belongs to another thread of this process.
    let element = unsafe { automation.ElementFromHandle(hwnd) }.map_err(|e| format!("no element: {e}"))?;
    selection_of(&element)
}

/// A window that owns the clipboard while the app writes it (an owner is required for `SetClipboardData`).
struct Owner(HWND);

impl Owner {
    fn new() -> Result<Self, String> {
        // SAFETY: a message-only STATIC window, destroyed on drop.
        let hwnd = unsafe { CreateWindowExW(WINDOW_EX_STYLE(0), w!("STATIC"), w!("litmus-clipboard"), WINDOW_STYLE(0), 0, 0, 0, 0, Some(HWND_MESSAGE), None, None, None) }
            .map_err(|e| format!("no clipboard window: {e}"))?;
        Ok(Self(hwnd))
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        // SAFETY: the window was made by `new` on this thread.
        unsafe {
            let _ = DestroyWindow(self.0);
        }
    }
}

/// The clipboard, open until dropped. Another app may hold it for a moment, so opening retries.
struct Opened;

impl Opened {
    fn new(owner: Option<HWND>) -> Result<Self, String> {
        let started = Instant::now();
        loop {
            // SAFETY: OpenClipboard is paired with CloseClipboard on drop.
            if unsafe { OpenClipboard(owner) }.is_ok() {
                return Ok(Self);
            }
            if started.elapsed() > Duration::from_millis(500) {
                return Err("Another app holds the clipboard.".into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Opened {
    fn drop(&mut self) {
        // SAFETY: opened by `Opened::new`.
        unsafe {
            let _ = CloseClipboard();
        }
    }
}

/// The open clipboard's Unicode text, if any.
fn unicode_text() -> Option<String> {
    // SAFETY: the clipboard is open; the handle's memory is read under GlobalLock and not kept.
    unsafe {
        if IsClipboardFormatAvailable(CF_UNICODETEXT).is_err() {
            return None;
        }
        let handle = GetClipboardData(CF_UNICODETEXT).ok()?;
        let memory = HGLOBAL(handle.0);
        let pointer = GlobalLock(memory) as *const u16;
        if pointer.is_null() {
            return None;
        }
        let units = GlobalSize(memory) / 2;
        let slice = std::slice::from_raw_parts(pointer, units);
        let end = slice.iter().position(|&u| u == 0).unwrap_or(units);
        let text = String::from_utf16_lossy(&slice[..end]);
        let _ = GlobalUnlock(memory);
        Some(text)
    }
}

/// Every format on the clipboard that is plain memory, copied out.
#[derive(Debug, PartialEq, Eq)]
pub struct Saved(Vec<(u32, Vec<u8>)>);

impl Saved {
    pub fn formats(&self) -> Vec<u32> {
        self.0.iter().map(|(format, _)| *format).collect()
    }
}

/// Copy out the clipboard's contents.
pub fn save_clipboard() -> Result<Saved, String> {
    let _open = Opened::new(None)?;
    let mut saved = Vec::new();
    // SAFETY: the clipboard is open; each handle's memory is copied under GlobalLock.
    unsafe {
        let mut format = EnumClipboardFormats(0);
        while format != 0 {
            if !HANDLE_FORMATS.contains(&format)
                && let Ok(handle) = GetClipboardData(format)
                && !handle.is_invalid()
            {
                let memory = HGLOBAL(handle.0);
                let size = GlobalSize(memory);
                let pointer = GlobalLock(memory) as *const u8;
                if !pointer.is_null() {
                    saved.push((format, std::slice::from_raw_parts(pointer, size).to_vec()));
                    let _ = GlobalUnlock(memory);
                }
            }
            format = EnumClipboardFormats(format);
        }
    }
    Ok(Saved(saved))
}

/// Put saved contents back on the clipboard.
pub fn restore_clipboard(saved: &Saved) -> Result<(), String> {
    let owner = Owner::new()?;
    let _open = Opened::new(Some(owner.0))?;
    // SAFETY: the clipboard is open and owned; each block is handed to the system by SetClipboardData.
    unsafe {
        EmptyClipboard().map_err(|e| format!("the clipboard could not be emptied: {e}"))?;
        for (format, bytes) in &saved.0 {
            let memory = GlobalAlloc(GMEM_MOVEABLE, bytes.len().max(1)).map_err(|e| e.to_string())?;
            let pointer = GlobalLock(memory) as *mut u8;
            if pointer.is_null() {
                return Err("no memory for the clipboard".into());
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
            let _ = GlobalUnlock(memory);
            SetClipboardData(*format, Some(HANDLE(memory.0))).map_err(|e| format!("clipboard format {format}: {e}"))?;
        }
    }
    Ok(())
}

/// The clipboard format that tells clipboard history and the cloud clipboard to skip the contents beside it.
pub fn exclusion_format() -> u32 {
    // SAFETY: registering a format name has no other effect.
    unsafe { RegisterClipboardFormatW(w!("ExcludeClipboardContentFromMonitorProcessing")) }
}

/// Contents with the format that tells clipboard history and the cloud clipboard to skip them.
fn private(mut formats: Vec<(u32, Vec<u8>)>) -> Saved {
    let exclude = exclusion_format();
    formats.retain(|(format, _)| *format != exclude);
    formats.push((exclude, vec![0]));
    Saved(formats)
}

/// Put text on the clipboard, marked so that clipboard history and cloud clipboard skip it (for tests).
pub fn set_private_text(text: &str) -> Result<(), String> {
    let mut units: Vec<u16> = text.encode_utf16().collect();
    units.push(0);
    let bytes: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
    restore_clipboard(&private(vec![(CF_UNICODETEXT, bytes)]))
}

/// Put saved contents back as `restore_clipboard` does, marked so that clipboard history does not record them a second
/// time (the copy path's own write; the source app's copy before it is the source app's).
pub fn restore_clipboard_privately(saved: &Saved) -> Result<(), String> {
    restore_clipboard(&private(saved.0.clone()))
}

fn key(vk: VIRTUAL_KEY, up: bool) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 { ki: KEYBDINPUT { wVk: vk, wScan: 0, dwFlags: if up { KEYEVENTF_KEYUP } else { KEYBD_EVENT_FLAGS(0) }, time: 0, dwExtraInfo: 0 } },
    }
}

fn held(vk: VIRTUAL_KEY) -> bool {
    // SAFETY: a read of the key's state.
    (unsafe { GetAsyncKeyState(vk.0 as i32) } as u16 & 0x8000) != 0
}

/// Simulate Copy in the foreground app and read what it put on the clipboard, with the clipboard's contents saved
/// before and put back after.
fn copy_selection() -> Result<String, String> {
    // The window the hotkey was pressed over: Copy goes to it or to nothing, never to a window that came in front
    // since.
    // SAFETY: a read of the foreground window's handle.
    let target = unsafe { GetForegroundWindow() };
    // The hotkey's own keys are probably still down; Copy is sent once they are up, or Alt+C would go instead.
    let started = Instant::now();
    while [VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN].iter().any(|&vk| held(vk)) {
        if started.elapsed() > Duration::from_millis(1500) {
            return Err("Release the hotkey's keys, then the selection can be copied.".into());
        }
        std::thread::sleep(Duration::from_millis(15));
    }
    // SAFETY: a read of the foreground window's handle.
    if target.is_invalid() || unsafe { GetForegroundWindow() } != target {
        return Err("Another window came in front before the selection could be copied. Press the hotkey again.".into());
    }
    let saved = save_clipboard()?;
    // SAFETY: reading a counter.
    let before = unsafe { GetClipboardSequenceNumber() };
    let inputs = [key(VK_CONTROL, false), key(VK_C, false), key(VK_C, true), key(VK_CONTROL, true)];
    // SAFETY: SendInput reads the array it is given.
    let sent = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
        return Err("Windows refused the simulated Copy (an app running as administrator is in front?).".into());
    }
    let started = Instant::now();
    // SAFETY: reading a counter.
    while unsafe { GetClipboardSequenceNumber() } == before {
        if started.elapsed() > Duration::from_millis(700) {
            return Err("Nothing was copied. No text seems to be selected in the app in front.".into());
        }
        std::thread::sleep(Duration::from_millis(15));
    }
    std::thread::sleep(Duration::from_millis(30)); // let the app finish writing every format
    let text = {
        let _open = Opened::new(None)?;
        unicode_text().unwrap_or_default()
    };
    restore_clipboard_privately(&saved).map_err(|e| format!("The text was read, but the clipboard could not be put back ({e})."))?;
    Ok(text)
}

fn wide(text: &str) -> HSTRING {
    HSTRING::from(text)
}

fn set_value(key: HKEY, name: Option<&str>, value: &str) -> Result<(), String> {
    let units: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
    let bytes: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
    let name = name.map(wide);
    let name_ptr = name.as_ref().map_or(PCWSTR::null(), |n| PCWSTR(n.as_ptr()));
    // SAFETY: the key is open for writing; the data is a NUL-terminated UTF-16 string.
    let status = unsafe { RegSetValueExW(key, name_ptr, None, REG_SZ, Some(&bytes)) };
    if status.is_err() { Err(format!("registry value: error {}", status.0)) } else { Ok(()) }
}

fn create(path: &str) -> Result<HKEY, String> {
    let mut key = HKEY::default();
    // SAFETY: creates or opens a per-user key; closed by the caller.
    let status = unsafe { RegCreateKeyExW(HKEY_CURRENT_USER, &wide(path), None, PCWSTR::null(), REG_OPTION_NON_VOLATILE, KEY_WRITE, None, &mut key, None) };
    if status.is_err() { Err(format!("HKEY_CURRENT_USER\\{path}: error {}", status.0)) } else { Ok(key) }
}

/// Add "Check with Litmus" to Explorer's right-click menu for the readable file kinds, under
/// `HKEY_CURRENT_USER\<root>\SystemFileAssociations\.<ext>\shell\LitmusCheck`.
pub fn add_explorer_entry(root: &str, exe: &Path) -> Result<(), String> {
    let exe = exe.display().to_string();
    for extension in EXTENSIONS {
        let verb = format!("{root}\\SystemFileAssociations\\.{extension}\\shell\\{VERB}");
        let key = create(&verb)?;
        let result = set_value(key, None, &format!("Check with {}", crate::brand::NAME)).and_then(|()| set_value(key, Some("Icon"), &format!("\"{exe}\",0")));
        // SAFETY: closing the key opened above.
        unsafe {
            let _ = RegCloseKey(key);
        }
        result?;
        let command = create(&format!("{verb}\\command"))?;
        let result = set_value(command, None, &format!("\"{exe}\" --check \"%1\""));
        // SAFETY: closing the key opened above.
        unsafe {
            let _ = RegCloseKey(command);
        }
        result?;
    }
    Ok(())
}

/// Remove every key `add_explorer_entry` made (the file-kind keys above them may be shared, so they stay).
pub fn remove_explorer_entry(root: &str) -> Result<(), String> {
    for extension in EXTENSIONS {
        let verb = format!("{root}\\SystemFileAssociations\\.{extension}\\shell\\{VERB}");
        // SAFETY: deletes a per-user key and its subkeys; a missing key is not an error.
        let status = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, &wide(&verb)) };
        if status.is_err() && status.0 != 2 {
            return Err(format!("HKEY_CURRENT_USER\\{verb}: error {}", status.0));
        }
    }
    Ok(())
}

/// Whether the entry is there (for the switch's state in Settings).
pub fn explorer_entry_present(root: &str) -> bool {
    let verb = format!("{root}\\SystemFileAssociations\\.txt\\shell\\{VERB}\\command");
    let mut key = HKEY::default();
    // SAFETY: opens for reading and closes.
    unsafe {
        let status = RegOpenKeyExW(HKEY_CURRENT_USER, &wide(&verb), None, KEY_READ, &mut key);
        if status.is_ok() {
            let _ = RegCloseKey(key);
            return true;
        }
    }
    false
}

/// Delete a key tree under `HKEY_CURRENT_USER` (tests remove their scratch root with it).
pub fn delete_tree(path: &str) -> Result<(), String> {
    // SAFETY: deletes a per-user key; a missing key is not an error.
    let status = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, &wide(path)) };
    if status.is_err() && status.0 != 2 { Err(format!("error {}", status.0)) } else { Ok(()) }
}
