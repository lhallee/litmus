//! Quick check's scripted scenes against the built app (`LITMUS_DEMO=quick`, `src/demo/quick.rs`): every script, with
//! the main window hidden in the tray and kept up, must pass (its picture written, no error note) and the app must end
//! through the tray's Quit within 90 seconds. Each run opens the app's windows on the screen for a second or two and
//! feeds in its captures itself: nothing is typed, clicked, or copied on the desktop. `LITMUS_MODEL` names a trained
//! release, whose long readings take long enough to watch. Run with
//! `LITMUS_MODEL=<release> cargo test --release --features desktop-tests,release-tests --test quick_scenes -- --test-threads 1`.
#![cfg(all(windows, feature = "desktop-tests", feature = "release-tests"))]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const SCRIPTS: [&str; 8] = ["reading", "loading", "batch", "replace", "escape", "expire", "full", "tray"];

fn release() -> PathBuf {
    PathBuf::from(std::env::var_os("LITMUS_MODEL").expect("LITMUS_MODEL names a trained release folder for these tests"))
}

fn scratch(name: &str) -> PathBuf {
    let folder = std::env::temp_dir().join(format!("litmus_scenes_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    std::fs::create_dir_all(&folder).unwrap();
    folder
}

/// One script in one window mode: what it printed, or why it failed.
fn run(script: &str, window: &str, batch: &Path) -> Result<String, String> {
    let folder = scratch(&format!("{script}_{window}"));
    let shot = folder.join("shot.png");
    let mut child = Command::new(env!("CARGO_BIN_EXE_litmus-app"))
        .env("LITMUS_DATA_DIR", folder.join("data"))
        .env("LITMUS_CACHE_DIR", folder.join("cache"))
        .env("LITMUS_MODEL", release())
        .env("LITMUS_DEMO", "quick")
        .env("LITMUS_DEMO_QUICK", script)
        .env("LITMUS_DEMO_WINDOW", window)
        .env("LITMUS_DEMO_FOLDER", batch)
        .env("LITMUS_SCREENSHOT", &shot)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(90) {
            child.kill().unwrap();
            child.wait().unwrap();
            return Err(format!("{script} ({window}): the app did not end within 90 seconds"));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut printed = String::new();
    child.stderr.take().unwrap().read_to_string(&mut printed).unwrap();
    let note = std::fs::read_to_string(shot.with_extension("error.txt")).ok();
    let outcome = match (status.success(), shot.exists(), note) {
        (_, _, Some(note)) => Err(format!("{script} ({window}): {note}")),
        (true, true, None) => Ok(printed.trim().to_owned()),
        (success, written, None) => Err(format!("{script} ({window}): exit {success}, picture written {written}: {printed}")),
    };
    std::fs::remove_dir_all(&folder).unwrap();
    outcome
}

#[test]
fn every_quick_check_script_passes_with_the_window_hidden_and_shown() {
    // The batch script's folder: two texts long enough that the batch is still reading when the quick check shows.
    let batch = scratch("batch_files");
    let text = (1..=4000).map(|n| format!("{n}. {}", litmus_app::sample::TEXT)).collect::<Vec<_>>().join("\n\n");
    for name in ["one.txt", "two.txt"] {
        std::fs::write(batch.join(name), &text).unwrap();
    }
    let mut failures = Vec::new();
    for window in ["hidden", "shown"] {
        for script in SCRIPTS {
            match run(script, window, &batch) {
                Ok(printed) => eprintln!("{script} ({window}): passed. {printed}"),
                Err(error) => failures.push(error),
            }
        }
    }
    std::fs::remove_dir_all(&batch).unwrap();
    assert!(failures.is_empty(), "{failures:#?}");
}
