//! Quick check's Windows parts against this machine, leaving it as found: the Explorer entry under a scratch
//! registry root (never `Software\Classes`), UI Automation on a WordPad the test starts and ends, and the clipboard
//! saved, overwritten with text marked to stay out of clipboard history, and put back byte for byte.
//! Run with `cargo test --features desktop-tests --test quick_windows -- --test-threads 1`.
#![cfg(all(windows, feature = "desktop-tests"))]

use litmus_app::quick::windows as quick;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::System::Registry::{HKEY, HKEY_CURRENT_USER, KEY_READ, RegCloseKey, RegOpenKeyExW};
use windows::Win32::UI::WindowsAndMessaging::{FindWindowExW, FindWindowW, SendMessageW};
use windows::core::{HSTRING, w};

fn key_exists(path: &str) -> bool {
    let mut key = HKEY::default();
    // SAFETY: opens for reading and closes.
    unsafe {
        if RegOpenKeyExW(HKEY_CURRENT_USER, &HSTRING::from(path), None, KEY_READ, &mut key).is_ok() {
            let _ = RegCloseKey(key);
            return true;
        }
    }
    false
}

#[test]
fn explorer_entry_adds_and_removes_under_a_scratch_root() {
    let real_before = quick::explorer_entry_present(quick::CLASSES);
    let scratch = format!("Software\\LitmusTest_{}", std::process::id());
    let root = format!("{scratch}\\Classes");
    assert!(!key_exists(&scratch), "the scratch root is new");
    let exe = std::env::current_exe().unwrap();
    quick::add_explorer_entry(&root, &exe).unwrap();
    assert!(quick::explorer_entry_present(&root));
    for extension in quick::EXTENSIONS {
        assert!(key_exists(&format!("{root}\\SystemFileAssociations\\.{extension}\\shell\\LitmusCheck\\command")), "{extension}");
    }
    quick::remove_explorer_entry(&root).unwrap();
    assert!(!quick::explorer_entry_present(&root));
    quick::remove_explorer_entry(&root).unwrap(); // removing twice is fine
    quick::delete_tree(&scratch).unwrap();
    assert!(!key_exists(&scratch), "nothing is left behind");
    assert_eq!(quick::explorer_entry_present(quick::CLASSES), real_before, "the real classes key is as it was");
}

#[test]
fn ui_automation_reads_a_selection_in_another_process() {
    // WordPad, started and ended by this test, holds a rich edit control, which offers UI Automation's text pattern;
    // the selection is read the way the hotkey reads the app in front. (Classic Edit controls, as in Notepad, offer
    // no text pattern; for them quick check copies instead.)
    let mut wordpad = std::process::Command::new(r"C:\Program Files\Windows NT\Accessories\wordpad.exe").spawn().unwrap();
    let edit = (0..100)
        .find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            // SAFETY: window lookups by class.
            unsafe {
                let top = FindWindowW(w!("WordPadClass"), None).ok()?;
                let mut pid = 0u32;
                windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId(top, Some(&mut pid));
                if pid != wordpad.id() {
                    return None;
                }
                FindWindowExW(Some(top), None, w!("RICHEDIT50W"), None).ok()
            }
        })
        .expect("WordPad's rich edit control appears");
    let text = HSTRING::from("The bakery on the corner opens at six.");
    // SAFETY: WM_SETTEXT and EM_SETSEL on the started WordPad's rich edit control.
    unsafe {
        SendMessageW(edit, 0x000C, None, Some(LPARAM(text.as_ptr() as isize))); // WM_SETTEXT
        SendMessageW(edit, 0x00B1, Some(WPARAM(4)), Some(LPARAM(10))); // EM_SETSEL: "bakery"
    }
    let selected = quick::selection_of_window(edit);
    wordpad.kill().unwrap();
    wordpad.wait().unwrap();
    assert_eq!(selected, Ok("bakery".to_owned()));
}

#[test]
fn the_clipboard_is_put_back_as_it_was() {
    let before = quick::save_clipboard().unwrap();
    quick::set_private_text("litmus clipboard test").unwrap();
    assert_eq!(litmus_app::quick::windows::clipboard_text().unwrap(), "litmus clipboard test");
    quick::restore_clipboard(&before).unwrap();
    let after = quick::save_clipboard().unwrap();
    assert_eq!(after, before, "every saved format comes back byte for byte");
}
