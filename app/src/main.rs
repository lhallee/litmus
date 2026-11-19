//! The desktop app. `litmus-app` opens the window (or brings the running one forward); `litmus-app --check <file>`
//! and `--check-text <text>` check a file or a text, in the running app when there is one; `--write-icon <file.svg>`
//! writes the mark; `__pdf-text <file>` is the child process that reads a PDF's text (as the command line's).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    use eframe::egui;
    use litmus_app::instance::{self, Claim, Request};
    use litmus_app::window_place::{DEFAULT_SIZE, MIN_SIZE};
    use litmus_app::{brand, settings};

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.first().map(String::as_str) {
        Some(litmus_files::extract::PDF_CHILD_ARGUMENT) => {
            let outcome = arguments.get(1).ok_or_else(|| "a PDF file".to_owned()).and_then(|path| litmus_files::extract::pdf_child(std::path::Path::new(path)));
            return if outcome.is_ok() { std::process::ExitCode::SUCCESS } else { std::process::ExitCode::from(2) };
        }
        Some("--write-icon") => {
            let Some(path) = arguments.get(1) else {
                eprintln!("--write-icon takes a file");
                return std::process::ExitCode::from(2);
            };
            return match std::fs::write(path, brand::icon_svg()) {
                Ok(()) => std::process::ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("{path}: {error}");
                    std::process::ExitCode::from(2)
                }
            };
        }
        _ => {}
    }
    let request = match Request::from_args(&arguments) {
        Ok(request) => request,
        Err(error) => {
            eprintln!("{error}");
            return std::process::ExitCode::from(2);
        }
    };
    // One app at a time: a later launch hands its request to the running app and ends. The scripted demo runs
    // apart from that, so that its scenes never reach a running app.
    let demo = std::env::var_os("LITMUS_DEMO").is_some();
    let folder = settings::disk::data_folder();
    let held = match (&folder, demo) {
        (Some(folder), false) => match instance::claim(folder) {
            Ok(Claim::First(held)) => Some(held),
            Ok(Claim::Running) => {
                instance::allow_foreground();
                return match instance::send(folder, &request) {
                    Ok(()) => std::process::ExitCode::SUCCESS,
                    Err(error) => {
                        eprintln!("{}: the running app could not be reached: {error}", brand::NAME);
                        std::process::ExitCode::from(1)
                    }
                };
            }
            Err(error) => {
                eprintln!("{}: running without the one-app lock: {error}", brand::NAME);
                None
            }
        },
        _ => None,
    };
    let saved = if demo { None } else { settings::disk::load().ok().and_then(|s| s.window).and_then(|w| w.sane()).filter(|w| w.on_a_screen()) };
    let size = 128;
    let icon = egui::IconData { rgba: brand::icon_rgba(size), width: size as u32, height: size as u32 };
    let inner = std::env::var("LITMUS_DEMO_SIZE")
        .ok()
        .and_then(|s| s.split_once('x').and_then(|(w, h)| Some([w.parse().ok()?, h.parse().ok()?])))
        .or(saved.map(|w| [w.width, w.height]))
        .unwrap_or(DEFAULT_SIZE);
    let mut viewport = egui::ViewportBuilder::default().with_title(brand::NAME).with_inner_size(inner).with_min_inner_size(MIN_SIZE).with_icon(icon).with_drag_and_drop(true);
    if let Some(place) = saved {
        viewport = viewport.with_position([place.x, place.y]).with_maximized(place.maximized);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    let started = eframe::run_native(
        brand::NAME,
        options,
        Box::new(move |cc| {
            let mut app = litmus_app::LitmusApp::new(cc);
            app.attach(&cc.egui_ctx, held, request);
            Ok(Box::new(app))
        }),
    );
    match started {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", brand::NAME);
            std::process::ExitCode::from(1)
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
