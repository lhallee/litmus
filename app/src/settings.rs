//! The desktop app's settings and data folder. Settings persist as `settings.json` in the data folder; the history
//! sits beside them in `history/`. `LITMUS_DATA_DIR` moves the folder (for tests and demos).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemeChoice {
    System,
    Light,
    /// The default on first launch.
    #[default]
    Dark,
}

impl ThemeChoice {
    pub fn preference(self) -> eframe::egui::ThemePreference {
        match self {
            ThemeChoice::System => eframe::egui::ThemePreference::System,
            ThemeChoice::Light => eframe::egui::ThemePreference::Light,
            ThemeChoice::Dark => eframe::egui::ThemePreference::Dark,
        }
    }
}

/// How much of a reading the screens show: Simple by default, Full a click away. The exports always carry every
/// field; the level is about the screen only, and neither level hides uncertainty: an uncertain reading says so and
/// shows its band at both. There were three levels through 1.1.0; Logan, 2026-11-21: "simple and full are the only
/// detail levels we need", so a saved `detailed` now reads as Full.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Detail {
    /// The verdict word, one plain sentence, the strip with its band, and the text tinted.
    #[default]
    Simple,
    /// Adds the numbers, the minimap of a long document, the reading of a clicked part of the text, how the reading
    /// was made, the model, and the exports.
    #[serde(alias = "detailed")]
    Full,
}

impl Detail {
    pub const ALL: [Detail; 2] = [Detail::Simple, Detail::Full];

    pub fn name(self) -> &'static str {
        match self {
            Detail::Simple => "Simple",
            Detail::Full => "Full",
        }
    }

    /// The other level, for the shortcut that switches between the two.
    pub fn other(self) -> Detail {
        match self {
            Detail::Simple => Detail::Full,
            Detail::Full => Detail::Simple,
        }
    }

    pub fn full(self) -> bool {
        self == Detail::Full
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// A release folder chosen by the user; `None` looks in the usual places.
    pub model_folder: Option<String>,
    pub theme: ThemeChoice,
    /// Reuse the readings of unchanged texts.
    pub use_cache: bool,
    /// Record readings in the history.
    pub keep_history: bool,
    /// Quick check: off until the user switches it on in Settings.
    pub quick_check: bool,
    pub hotkey: String,
    /// How much of a reading the screens show.
    pub detail: Detail,
    /// Where the main window was when the app last ran (desktop).
    pub window: Option<crate::window_place::WindowPlace>,
}

impl Default for Settings {
    fn default() -> Self {
        Self { model_folder: None, theme: ThemeChoice::Dark, use_cache: true, keep_history: true, quick_check: false, hotkey: "Ctrl+Alt+L".into(), detail: Detail::Simple, window: None }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub mod disk {
    use std::path::PathBuf;

    use super::Settings;

    /// `LITMUS_DATA_DIR`, else the platform's per-user data folder.
    pub fn data_folder() -> Option<PathBuf> {
        if let Some(folder) = std::env::var_os("LITMUS_DATA_DIR") {
            return Some(PathBuf::from(folder));
        }
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        if cfg!(windows) {
            var("APPDATA").map(|base| base.join("litmus"))
        } else if cfg!(target_os = "macos") {
            var("HOME").map(|base| base.join("Library").join("Application Support").join("litmus"))
        } else {
            var("XDG_DATA_HOME").or_else(|| var("HOME").map(|home| home.join(".local").join("share"))).map(|base| base.join("litmus"))
        }
    }

    /// The saved settings; missing is the defaults, damaged is an error.
    pub fn load() -> Result<Settings, String> {
        let Some(path) = data_folder().map(|folder| folder.join("settings.json")) else { return Ok(Settings::default()) };
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
            Err(error) => Err(format!("{}: {error}", path.display())),
        }
    }

    pub fn save(settings: &Settings) -> Result<(), String> {
        let folder = data_folder().ok_or("no data folder on this system")?;
        std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
        let path = folder.join("settings.json");
        let partial = folder.join("settings.json.partial");
        let bytes = serde_json::to_vec_pretty(settings).map_err(|e| e.to_string())?;
        std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
        std::fs::rename(&partial, &path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The browser keeps the settings in the page's local storage (this browser only; nothing is sent anywhere). A
/// browser that refuses storage (a private window) keeps them for the visit.
#[cfg(target_arch = "wasm32")]
pub mod browser {
    use super::Settings;

    const KEY: &str = "litmus.settings";

    fn storage() -> Option<web_sys::Storage> {
        web_sys::window()?.local_storage().ok().flatten()
    }

    /// The saved settings; missing or damaged is the defaults (the page has nothing else to fall back on).
    pub fn load() -> Settings {
        storage().and_then(|s| s.get_item(KEY).ok().flatten()).and_then(|json| serde_json::from_str(&json).ok()).unwrap_or_default()
    }

    pub fn save(settings: &Settings) {
        if let (Some(storage), Ok(json)) = (storage(), serde_json::to_string(settings)) {
            let _ = storage.set_item(KEY, &json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_fill_missing_fields() {
        let settings: Settings = serde_json::from_str(r#"{"theme": "light"}"#).unwrap();
        assert_eq!(settings, Settings { theme: ThemeChoice::Light, ..Settings::default() }, "a saved choice wins");
        let settings: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(settings.theme, ThemeChoice::Dark, "dark is the default on first launch");
        assert!(!settings.quick_check, "quick check is off until the user switches it on");
        assert!(settings.keep_history && settings.use_cache);
        assert_eq!(settings.detail, Detail::Simple, "the simple screen is the default");
        let settings: Settings = serde_json::from_str(r#"{"detail": "full", "window": {"x": 4, "y": 8, "width": 900, "height": 700, "maximized": false}}"#).unwrap();
        assert_eq!(settings.detail, Detail::Full);
        assert_eq!(settings.window.map(|w| w.width), Some(900.0));
    }

    #[test]
    fn there_are_two_detail_levels_and_an_old_choice_still_loads() {
        assert!(Detail::Full.full() && !Detail::Simple.full());
        assert_eq!(Detail::ALL.map(Detail::name), ["Simple", "Full"]);
        assert_eq!((Detail::Simple.other(), Detail::Full.other()), (Detail::Full, Detail::Simple));
        // Settings saved by 1.1.0 and earlier may hold the middle level that no longer exists.
        let settings: Settings = serde_json::from_str(r#"{"detail": "detailed"}"#).unwrap();
        assert_eq!(settings.detail, Detail::Full);
        assert_eq!(serde_json::to_string(&Detail::Full).unwrap(), r#""full""#);
    }
}
