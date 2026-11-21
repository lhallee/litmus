//! Past readings. On the desktop they persist in the app's data folder: `history/index.json` lists the entries and
//! `history/<id>.json` holds each one's text and reading, every file written whole and then renamed into place. In
//! the browser the history lives for the session only, in memory.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use litmus_engine::DocumentReading;
use serde::{Deserialize, Serialize};

/// Entries kept; the oldest go first.
pub const LIMIT: usize = 2000;
#[cfg(not(target_arch = "wasm32"))]
const FORMAT: &str = "litmus-history";
#[cfg(not(target_arch = "wasm32"))]
const VERSION: u32 = 1;

/// One past reading, as the history list shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub id: u64,
    pub at_ms: i64,
    pub title: String,
    /// Where the text came from: "pasted", a file's path, or the contribute site's id.
    pub source: String,
    /// SHA-256 of the text's bytes (or the file's).
    pub hash: String,
    pub headline: String,
    /// The verdict's probability of not human and its interval, as the release's table gives them.
    pub probability: f64,
    pub interval: [f64; 2],
    /// The verdict's chance of AI, what the list shows and its small strip draws (entries saved before it was kept
    /// have none, and the list restates their probability by the loaded release's table).
    #[serde(default)]
    pub chance: Option<litmus_engine::Chance>,
    /// The verdict's place between human, mixed and AI (`P_ai + P_mixed / 2`), which colors the verdict's marker.
    pub position: f64,
    #[serde(default)]
    pub unlike_reference: bool,
    pub tokens: usize,
    /// Seconds the scoring took; `None` when the reading came from the cache.
    pub seconds: Option<f64>,
    /// The model that read it (its version, or its hash).
    pub model: String,
}

/// The text and the reading behind an entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Body {
    pub text: String,
    pub reading: DocumentReading,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Serialize, Deserialize)]
struct Index {
    format: String,
    version: u32,
    entries: Vec<Entry>,
}

pub struct History {
    #[cfg(not(target_arch = "wasm32"))]
    folder: Option<PathBuf>,
    entries: Vec<Entry>,
    memory: HashMap<u64, Body>,
    next_id: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let partial = path.with_extension("json.partial");
    std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

impl History {
    /// A history that lives in memory only (the browser).
    pub fn in_memory() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            folder: None,
            entries: Vec::new(),
            memory: HashMap::new(),
            next_id: 1,
        }
    }

    /// Open the history in `folder`, creating it when missing. A damaged index is an error, never silently
    /// replaced.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(folder: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
        let index = folder.join("index.json");
        let entries = match std::fs::read(&index) {
            Ok(bytes) => {
                let parsed: Index = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", index.display()))?;
                if parsed.format != FORMAT || parsed.version != VERSION {
                    return Err(format!("{}: format {} version {}, expected {FORMAT} version {VERSION}", index.display(), parsed.format, parsed.version));
                }
                parsed.entries
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(format!("{}: {error}", index.display())),
        };
        let next_id = entries.iter().map(|e| e.id).max().unwrap_or(0) + 1;
        Ok(Self { folder: Some(folder.to_owned()), entries, memory: HashMap::new(), next_id })
    }

    /// Entries, oldest first.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn is_persistent(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.folder.is_some();
        #[cfg(target_arch = "wasm32")]
        false
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn body_path(&self, id: u64) -> Option<PathBuf> {
        self.folder.as_ref().map(|folder| folder.join(format!("{id}.json")))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_index(&self) -> Result<(), String> {
        let Some(folder) = &self.folder else { return Ok(()) };
        let index = Index { format: FORMAT.to_owned(), version: VERSION, entries: self.entries.clone() };
        let bytes = serde_json::to_vec(&index).map_err(|e| e.to_string())?;
        write_atomic(&folder.join("index.json"), &bytes)
    }

    #[cfg(target_arch = "wasm32")]
    fn save_index(&self) -> Result<(), String> {
        Ok(())
    }

    fn drop_body(&mut self, id: u64) {
        self.memory.remove(&id);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = self.body_path(id) {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Add a reading; a repeat of the newest entry's text and model replaces it. Returns the entry's id.
    pub fn add(&mut self, mut entry: Entry, body: Body) -> Result<u64, String> {
        if let Some(last) = self.entries.last()
            && last.hash == entry.hash
            && last.model == entry.model
            && last.source == entry.source
        {
            let id = last.id;
            self.entries.pop();
            self.drop_body(id);
        }
        entry.id = self.next_id;
        self.next_id += 1;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = self.body_path(entry.id) {
            let bytes = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
            write_atomic(&path, &bytes)?;
        } else {
            self.memory.insert(entry.id, body);
        }
        #[cfg(target_arch = "wasm32")]
        self.memory.insert(entry.id, body);
        let id = entry.id;
        self.entries.push(entry);
        while self.entries.len() > LIMIT {
            let old = self.entries.remove(0).id;
            self.drop_body(old);
        }
        self.save_index()?;
        Ok(id)
    }

    /// The text and reading of an entry.
    pub fn body(&self, id: u64) -> Result<Body, String> {
        if let Some(body) = self.memory.get(&id) {
            return Ok(body.clone());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = self.body_path(id) {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            return serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()));
        }
        Err(format!("no history entry {id}"))
    }

    pub fn remove(&mut self, id: u64) -> Result<(), String> {
        self.entries.retain(|entry| entry.id != id);
        self.drop_body(id);
        self.save_index()
    }

    /// Forget every entry and delete their files.
    pub fn clear(&mut self) -> Result<(), String> {
        let ids: Vec<u64> = self.entries.iter().map(|e| e.id).collect();
        self.entries.clear();
        for id in ids {
            self.drop_body(id);
        }
        self.save_index()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn entry(hash: &str, source: &str) -> Entry {
        Entry {
            id: 0,
            at_ms: 1,
            title: "t".into(),
            source: source.into(),
            hash: hash.into(),
            headline: "Human".into(),
            probability: 0.2,
            interval: [0.1, 0.3],
            chance: Some(litmus_engine::Chance::at_even_odds(0.2, [0.1, 0.3], 2.0 / 3.0)),
            position: 0.1,
            unlike_reference: false,
            tokens: 5,
            seconds: Some(0.01),
            model: "m".into(),
        }
    }

    fn body(text: &str) -> Body {
        let reading: DocumentReading = serde_json::from_str(include_str!("../tests/reading_small.json")).expect("the fixture reading parses");
        Body { text: text.into(), reading }
    }

    fn folder(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!("litmus_history_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        folder
    }

    #[test]
    fn history_survives_reopening() {
        let folder = folder("reopen");
        let mut history = History::open(&folder).unwrap();
        let first = history.add(entry("a", "pasted"), body("one")).unwrap();
        let second = history.add(entry("b", "x.txt"), body("two")).unwrap();
        drop(history);
        let history = History::open(&folder).unwrap();
        assert_eq!(history.entries().iter().map(|e| e.id).collect::<Vec<_>>(), [first, second]);
        assert_eq!(history.body(second).unwrap().text, "two");
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn repeats_replace_the_newest_and_removal_deletes_files() {
        let folder = folder("repeat");
        let mut history = History::open(&folder).unwrap();
        let first = history.add(entry("a", "pasted"), body("one")).unwrap();
        let again = history.add(entry("a", "pasted"), body("one")).unwrap();
        assert_ne!(first, again);
        assert_eq!(history.entries().len(), 1);
        assert!(!folder.join(format!("{first}.json")).exists());
        history.remove(again).unwrap();
        assert!(history.entries().is_empty() && !folder.join(format!("{again}.json")).exists());
        history.add(entry("c", "pasted"), body("three")).unwrap();
        history.clear().unwrap();
        assert!(History::open(&folder).unwrap().entries().is_empty());
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_damaged_index_is_an_error() {
        let folder = folder("damaged");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("index.json"), b"{not json").unwrap();
        assert!(History::open(&folder).is_err());
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn entries_of_earlier_versions_still_read() {
        let mut value = serde_json::to_value(entry("a", "pasted")).unwrap();
        let object = value.as_object_mut().unwrap();
        // The first entries kept neither field, and an entry of 1.1.0 kept its probability alone, which the list
        // restates by the loaded release's table.
        object.remove("unlike_reference");
        object.remove("chance");
        // 1.1.0 also kept the band of the strip it drew, over the origins not ruled out. The strip now bands the
        // chance's range, so that field is read past.
        object.insert("band".into(), serde_json::json!([0.0, 0.5]));
        let old: Entry = serde_json::from_value(value).unwrap();
        assert_eq!((old.chance, old.unlike_reference), (None, false));
        assert_eq!((old.probability, old.interval, old.position), (0.2, [0.1, 0.3], 0.1));
        assert!(serde_json::to_value(&old).unwrap().get("band").is_none(), "and is not written again");
    }

    #[test]
    fn memory_history_keeps_bodies() {
        let mut history = History::in_memory();
        let id = history.add(entry("a", "pasted"), body("one")).unwrap();
        assert_eq!(history.body(id).unwrap().text, "one");
        assert!(!history.is_persistent());
    }
}
