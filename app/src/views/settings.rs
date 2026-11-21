//! The Settings window: how much a reading shows, the theme, the keyboard shortcuts, and on the desktop the model
//! folder, the cache and the history, quick check, and where the app keeps its data. The model's version and hash
//! live here, in About, not in the top bar.

use eframe::egui::{self, RichText};

use crate::app::{LitmusApp, ModelState, SHORTCUTS};
use crate::settings::{Detail, ThemeChoice};
use crate::theme;
use crate::views::controls;

pub fn window(app: &mut LitmusApp, ctx: &egui::Context) {
    if !app.settings_open {
        return;
    }
    let mut open = app.settings_open;
    let width = (ctx.content_rect().width() - 32.0).min(600.0);
    let height = (ctx.content_rect().height() - 160.0).clamp(200.0, 680.0);
    egui::Window::new(RichText::new("Settings").font(theme::strong(17.0)))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(width)
        .min_width(width)
        .max_width(width)
        .anchor(egui::Align2::CENTER_TOP, [0.0, 64.0])
        .min_height(height)
        .max_height(height)
        .vscroll(true)
        .show(ctx, |ui| contents(app, ui, ctx));
    app.settings_open = open;
}

fn contents(app: &mut LitmusApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let p = controls::palette(ui);
    controls::section(ui, "Detail");
    let options: Vec<(Detail, &str)> = Detail::ALL.iter().map(|d| (*d, d.name())).collect();
    if let Some(detail) = controls::segmented(ui, app.settings.detail, &options, "settings_detail") {
        app.set_detail(detail);
    }
    ui.label(
        RichText::new(match app.settings.detail {
            Detail::Simple => "The verdict, one sentence, the strip, and your text tinted. Uncertainty always shows.",
            Detail::Full => "Adds the numbers, the reading of any part you click, how the reading was made, the model, and the exports.",
        })
        .color(p.muted),
    );
    ui.add_space(16.0);
    controls::section(ui, "Theme");
    let themes = [(ThemeChoice::Dark, "Dark"), (ThemeChoice::Light, "Light"), (ThemeChoice::System, "Follow the system")];
    if let Some(choice) = controls::segmented(ui, app.settings.theme, &themes, "settings_theme") {
        ctx.set_theme(choice.preference());
        let mut settings = app.settings.clone();
        settings.theme = choice;
        app.set_settings(settings);
    }
    #[cfg(not(target_arch = "wasm32"))]
    desktop(app, ui, ctx);
    ui.add_space(16.0);
    controls::section(ui, "Keyboard");
    egui::Grid::new("shortcuts").num_columns(2).spacing([20.0, 6.0]).show(ui, |ui| {
        for (keys, what, desktop_only) in SHORTCUTS {
            if *desktop_only && cfg!(target_arch = "wasm32") {
                continue;
            }
            ui.label(RichText::new(*keys).monospace());
            ui.label(*what);
            ui.end_row();
        }
    });
    ui.add_space(16.0);
    controls::section(ui, "About");
    ui.label(format!("{} {}, an open detector of AI-written text. Source available under the PolyForm Noncommercial License 1.0.0.", crate::brand::NAME, env!("CARGO_PKG_VERSION")));
    match &app.model {
        ModelState::Ready(meta) => {
            ui.label(RichText::new(format!("Model {} {}", meta.size, meta.version).trim().to_owned()).monospace().color(p.muted));
            ui.label(RichText::new(format!("From {}{}", meta.source, if meta.license.is_empty() { String::new() } else { format!(", license {}", meta.license) })).small().color(p.muted));
        }
        ModelState::Loading => {
            ui.label(RichText::new("The model is loading.").color(p.muted));
        }
        ModelState::Failed(error) => {
            ui.label(RichText::new(format!("No model. {error}")).color(p.error));
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(folder) = crate::settings::disk::data_folder() {
        ui.label(RichText::new(format!("Settings and history are kept in {}", folder.display())).small().color(p.muted));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn desktop(app: &mut LitmusApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    use crate::app::Notice;
    let p = controls::palette(ui);
    ui.add_space(16.0);
    controls::section(ui, "Model");
    ui.label(RichText::new("A release folder holds model.safetensors, config.json, and tokenizer.json. Left empty, Litmus looks at LITMUS_MODEL, then a folder named release beside the app, then the model built in.").color(p.muted));
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut app.model_folder_edit).hint_text("Release folder").desired_width(ui.available_width() - 100.0).margin(egui::Margin::symmetric(10, 7)));
        if ui.button("Browse").clicked()
            && let Some(folder) = rfd::FileDialog::new().pick_folder()
        {
            app.model_folder_edit = folder.display().to_string();
        }
    });
    ui.horizontal(|ui| {
        let idle = !app.busy();
        if ui.add_enabled(idle, egui::Button::new("Load this model")).clicked() {
            let folder = app.model_folder_edit.trim().to_owned();
            let mut settings = app.settings.clone();
            settings.model_folder = (!folder.is_empty()).then_some(folder);
            app.set_settings(settings.clone());
            app.model = ModelState::Loading;
            app.scorer.load(settings.model_folder);
        }
        if controls::quiet(ui, "Use the default places", None, idle).clicked() {
            app.model_folder_edit.clear();
            let mut settings = app.settings.clone();
            settings.model_folder = None;
            app.set_settings(settings);
            app.model = ModelState::Loading;
            app.scorer.load(None);
        }
    });
    ui.add_space(16.0);
    controls::section(ui, "Readings");
    let mut use_cache = app.settings.use_cache;
    if ui.checkbox(&mut use_cache, "Reuse the reading of a text read before (same contents, same model)").changed() {
        let mut settings = app.settings.clone();
        settings.use_cache = use_cache;
        app.set_settings(settings);
        app.scorer.set_cache(use_cache);
    }
    let mut keep = app.settings.keep_history;
    if ui.checkbox(&mut keep, "Keep a history of readings on this computer").changed() {
        let mut settings = app.settings.clone();
        settings.keep_history = keep;
        app.set_settings(settings);
    }
    ui.horizontal_wrapped(|ui| {
        if controls::quiet(ui, "Clear this model's cache", None, true).clicked() {
            app.scorer.clear_cache();
            app.notice = Some(Notice::Info("The cache for this model is cleared.".into()));
        }
        if let Some(folder) = litmus_files::cache::folder() {
            ui.label(RichText::new(format!("Shared with the litmus command line, in {}", folder.display())).small().color(p.muted));
        }
    });
    ui.add_space(16.0);
    crate::views::quick::settings_section(app, ui, ctx);
}
