//! The Litmus app: one egui (eframe) program for the desktop (Windows, macOS, Linux) and the browser.
//!
//! - Desktop: paste, open or drop files and folders, batch with progress and cancel, history across launches, a
//!   cache of readings, CSV and JSON export, a model folder setting (`main.rs`).
//! - Browser: the same screens with paste, score, heat map, explanation and this session's history; the engine runs
//!   in a Web Worker on the release fetched from the page's own origin (`web.rs`, `site/app/`).
//! - Contribute (feature `contribute`): the browser page, scored by the contribute site's server, with its consent
//!   notice and the visitor's label (`contribute.rs`).
//!
//! The non-UI logic (verdict wording, strip geometry, cell-to-text mapping, history, exports, the contribute API's
//! shapes) is in plain modules with unit tests; the screens are in `views`.

pub mod app;
pub mod brand;
pub mod clock;
pub mod contribute;
pub mod heatmap;
pub mod history;
pub mod model;
pub mod sample;
pub mod scoring;
pub mod settings;
pub mod sorting;
pub mod strip;
pub mod theme;
pub mod views;
pub mod window_place;
pub mod wording;

#[cfg(not(target_arch = "wasm32"))]
pub mod demo;
#[cfg(not(target_arch = "wasm32"))]
pub mod icon_resource;
#[cfg(not(target_arch = "wasm32"))]
pub mod instance;
#[cfg(not(target_arch = "wasm32"))]
pub mod png;
#[cfg(not(target_arch = "wasm32"))]
pub mod quick;
#[cfg(target_arch = "wasm32")]
pub mod web;

pub use app::LitmusApp;
