//! One app at a time (desktop). The first launch holds an exclusive lock on `instance.lock` in the data folder for
//! as long as it runs. A later launch (a second double-click, or Explorer's "Check with Litmus", which runs
//! `litmus-app --check <file>`) finds the lock held, leaves its request in `inbox/` (written whole, then renamed into
//! place) and exits; the running app picks the request up within a moment, comes to the front, and checks the file
//! or text. The lock is the operating system's, so a crashed app leaves no stale lock behind.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// What a launch asks of the app.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    /// Come to the front.
    Show,
    /// Check this file (an absolute path).
    CheckFile(PathBuf),
    /// Check this text.
    CheckText(String),
}

impl Request {
    /// The request of a command line (the arguments after the program's name): none is `Show`,
    /// `--check <file>` checks a file, `--check-text <text>` checks a text. Anything else is an error.
    pub fn from_args(arguments: &[String]) -> Result<Self, String> {
        match arguments {
            [] => Ok(Self::Show),
            [flag, file] if flag == "--check" => {
                let path = std::path::absolute(file).map_err(|e| format!("{file}: {e}"))?;
                Ok(Self::CheckFile(path))
            }
            [flag, text] if flag == "--check-text" => Ok(Self::CheckText(text.clone())),
            [flag] if flag == "--check" || flag == "--check-text" => Err(format!("{flag} takes one argument")),
            other => Err(format!("unknown arguments {other:?}: use --check <file> or --check-text <text>")),
        }
    }
}

/// The running app's hold on the data folder.
pub struct Instance {
    _lock: File,
    inbox: PathBuf,
}

/// Whether this launch is the first.
pub enum Claim {
    First(Instance),
    /// Another app holds the lock: send it the request instead.
    Running,
}

fn inbox(folder: &Path) -> PathBuf {
    folder.join("inbox")
}

/// Take the lock in `folder`, or learn that another app holds it.
pub fn claim(folder: &Path) -> Result<Claim, String> {
    std::fs::create_dir_all(inbox(folder)).map_err(|e| format!("{}: {e}", folder.display()))?;
    let path = folder.join("instance.lock");
    let file = OpenOptions::new().create(true).truncate(false).write(true).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(Claim::First(Instance { _lock: file, inbox: inbox(folder) })),
        Err(std::fs::TryLockError::WouldBlock) => Ok(Claim::Running),
        Err(std::fs::TryLockError::Error(error)) => Err(format!("{}: {error}", path.display())),
    }
}

/// Leave a request for the running app.
pub fn send(folder: &Path, request: &Request) -> Result<(), String> {
    let inbox = inbox(folder);
    std::fs::create_dir_all(&inbox).map_err(|e| format!("{}: {e}", inbox.display()))?;
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let name = format!("{nanos:032}_{}", std::process::id());
    let partial = inbox.join(format!("{name}.partial"));
    let bytes = serde_json::to_vec(request).map_err(|e| e.to_string())?;
    std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
    let path = inbox.join(format!("{name}.json"));
    std::fs::rename(&partial, &path).map_err(|e| format!("{}: {e}", path.display()))
}

fn waiting(inbox: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(inbox) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    paths.sort();
    paths
}

impl Instance {
    /// The requests other launches left, oldest first, each removed once read. A request that does not read is
    /// reported as an error and removed too, so it is not read again.
    pub fn take(&self) -> Vec<Result<Request, String>> {
        waiting(&self.inbox)
            .into_iter()
            .map(|path| {
                let read = std::fs::read(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| serde_json::from_slice::<Request>(&bytes).map_err(|e| e.to_string()))
                    .map_err(|e| format!("a request from another launch ({}) did not read: {e}", path.display()));
                let _ = std::fs::remove_file(&path);
                read
            })
            .collect()
    }

    /// Wake the app when a request arrives: a thread looks in the inbox a few times a second.
    pub fn watch(&self, ctx: &eframe::egui::Context) {
        let (inbox, ctx) = (self.inbox.clone(), ctx.clone());
        let _ = std::thread::Builder::new().name("litmus-inbox".into()).spawn(move || {
            loop {
                if !waiting(&inbox).is_empty() {
                    ctx.request_repaint();
                }
                std::thread::sleep(Duration::from_millis(300));
            }
        });
    }
}

/// Let the running app take the foreground when this launch hands it a request (Windows refuses a background
/// process the focus unless the foreground one allows it).
pub fn allow_foreground() {
    #[cfg(windows)]
    // SAFETY: a plain Win32 call with a constant argument.
    unsafe {
        let _ = windows::Win32::UI::WindowsAndMessaging::AllowSetForegroundWindow(windows::Win32::UI::WindowsAndMessaging::ASFW_ANY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!("litmus_instance_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        folder
    }

    #[test]
    fn arguments_make_requests() {
        let args = |a: &[&str]| a.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(Request::from_args(&[]).unwrap(), Request::Show);
        assert_eq!(Request::from_args(&args(&["--check-text", "some words"])).unwrap(), Request::CheckText("some words".into()));
        match Request::from_args(&args(&["--check", "notes.md"])).unwrap() {
            Request::CheckFile(path) => assert!(path.is_absolute() && path.ends_with("notes.md"), "{path:?}"),
            other => panic!("{other:?}"),
        }
        assert!(Request::from_args(&args(&["--check"])).is_err());
        assert!(Request::from_args(&args(&["--frobnicate"])).is_err());
    }

    #[test]
    fn a_second_launch_hands_over_its_request() {
        let folder = folder("handoff");
        let Claim::First(first) = claim(&folder).unwrap() else { panic!("the first claim holds the lock") };
        assert!(matches!(claim(&folder).unwrap(), Claim::Running), "a second claim finds the lock held");
        send(&folder, &Request::CheckText("one".into())).unwrap();
        send(&folder, &Request::CheckFile(PathBuf::from("C:/two.txt"))).unwrap();
        std::fs::write(inbox(&folder).join("zzz_damaged.json"), b"{not json").unwrap();
        let taken = first.take();
        assert_eq!(taken.len(), 3);
        assert_eq!(taken[0], Ok(Request::CheckText("one".into())), "oldest first");
        assert_eq!(taken[1], Ok(Request::CheckFile(PathBuf::from("C:/two.txt"))));
        assert!(taken[2].as_ref().is_err_and(|e| e.contains("did not read")));
        assert!(first.take().is_empty(), "each request is read once");
        drop(first);
        assert!(matches!(claim(&folder).unwrap(), Claim::First(_)), "the lock is free once the app ends");
        std::fs::remove_dir_all(&folder).unwrap();
    }
}
