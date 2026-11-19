//! What the Litmus command line (`litmus-cli`) and the app (`../app`) share around the engine: the text of the files
//! they read (plain text, Markdown, HTML, Word and PDF), the content-hash cache of readings, release folders, folder
//! walks, the CSV and JSON export of readings, and the word, character and token counts shown beside a reading. The
//! engine itself stays free of files, threads and clocks.
//!
//! On WebAssembly the file system parts (the disk cache, release folders, folder walks and PDF) are left out; text of
//! HTML and Word bytes, hashes and exports work everywhere.

pub mod cache;
pub mod counts;
pub mod export;
pub mod extract;
#[cfg(not(target_arch = "wasm32"))]
pub mod release;
#[cfg(not(target_arch = "wasm32"))]
pub mod walk;
