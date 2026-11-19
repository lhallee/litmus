//! Quick check on macOS and Linux. Linux is built and run (x86_64, X11 under Xvfb: the hotkey read the PRIMARY
//! selection through `xclip`); the Wayland route and macOS are written and never run. It uses only the standard
//! library and the platforms' own command-line tools:
//!
//! - Linux: the PRIMARY selection is the selected text itself, so nothing is copied and the clipboard is untouched:
//!   `wl-paste --primary --no-newline` (Wayland) or `xclip -o -selection primary` (X11). The global hotkey works on
//!   X11 only (global-hotkey's limit); Wayland offers no global hotkeys to apps. No tray (README).
//! - macOS: simulated Copy through System Events (`osascript`), with the clipboard's text saved (`pbpaste`) and put
//!   back (`pbcopy`); clipboard contents other than text are not saved yet, and System Events needs the
//!   Accessibility permission. The native right-click route, a Services entry, is not built.

#[cfg(target_os = "macos")]
use std::io::Write;
use std::process::{Command, Stdio};

use super::{Captured, How};

fn run(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program).args(arguments).stdin(Stdio::null()).output().map_err(|e| format!("{program}: {e}"))?;
    if !output.status.success() {
        return Err(format!("{program} failed"));
    }
    String::from_utf8(output.stdout).map_err(|_| format!("{program} returned text that is not UTF-8"))
}

#[cfg(target_os = "macos")]
fn write_clipboard(text: &str) -> Result<(), String> {
    let mut child = Command::new("pbcopy").stdin(Stdio::piped()).spawn().map_err(|e| format!("pbcopy: {e}"))?;
    child.stdin.take().ok_or("pbcopy: no input")?.write_all(text.as_bytes()).map_err(|e| format!("pbcopy: {e}"))?;
    child.wait().map_err(|e| format!("pbcopy: {e}"))?;
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn selected_text() -> Result<Captured, String> {
    let saved = run("pbpaste", &[]).unwrap_or_default();
    run("osascript", &["-e", "tell application \"System Events\" to keystroke \"c\" using command down"])?;
    std::thread::sleep(std::time::Duration::from_millis(250));
    let text = run("pbpaste", &[])?;
    write_clipboard(&saved)?;
    if text.trim().is_empty() || text == saved {
        return Err("No selected text was found. Select some text, then press the hotkey.".into());
    }
    Ok(Captured { text, how: How::Copy })
}

#[cfg(target_os = "macos")]
pub fn clipboard_text() -> Result<String, String> {
    run("pbpaste", &[])
}

#[cfg(not(target_os = "macos"))]
pub fn selected_text() -> Result<Captured, String> {
    let text = run("wl-paste", &["--primary", "--no-newline"]).or_else(|_| run("xclip", &["-o", "-selection", "primary"]))?;
    if text.trim().is_empty() {
        return Err("No selected text was found. Select some text, then press the hotkey.".into());
    }
    Ok(Captured { text, how: How::Accessibility })
}

#[cfg(not(target_os = "macos"))]
pub fn clipboard_text() -> Result<String, String> {
    run("wl-paste", &["--no-newline"]).or_else(|_| run("xclip", &["-o", "-selection", "clipboard"]))
}

/// The pointer's place is not read on these platforms; the popup opens by the main window's last place.
pub fn pointer() -> Option<(i32, i32)> {
    None
}
