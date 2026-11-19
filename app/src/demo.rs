//! A scripted run of the desktop app for screenshots and end-to-end checks, set by environment variables:
//!
//! - `LITMUS_DEMO`: the scene, `check`, `explain`, `long`, `batch`, `history`, `empty`, `settings` or `quick` (quick check on for
//!   the run only, the window hidden to the tray, a short text read as if selected; the PNG is of the popup).
//! - `LITMUS_SCREENSHOT`: where to write the PNG of the window once the scene is ready; the app then closes.
//! - `LITMUS_THEME`: `light` or `dark` for this run.
//! - `LITMUS_DEMO_FOLDER`: the folder the `batch` scene scores; `LITMUS_DEMO_EXPORT`: a folder to write the batch's
//!   `litmus_batch.csv` and `litmus_batch.json` into, through the same functions as the Export buttons.
//! - `LITMUS_DEMO_CELL`: the heat-map cell to select (default 2).
//! - `LITMUS_DETAIL`: `simple`, `detailed` or `full` for this run (not saved).
//! - `LITMUS_DEMO_ZOOM`: a zoom factor, as a display scaled to 150% shows the app (`1.5`).
//! - `LITMUS_DEMO_TEXT`: a UTF-8 text file the `check`, `explain` and `edit` scenes check in place of the sample, so
//!   that a release's reading of a chosen text can be photographed and compared with the command line's.
//!
//! The scenes of states: `short` (a few words), `scoring` (a long text, photographed while it is read), `error` (a
//! file that cannot be read), `nomodel` (`LITMUS_MODEL` pointing at no release), and `edit` (the editor reopened
//! over a reading).
//!
//! A scene that fails (no model, an error, 120 seconds without a result) writes `<screenshot>.error.txt` instead.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use eframe::egui;

use crate::app::{LitmusApp, ModelState, Tab};
use crate::settings::ThemeChoice;

/// Set when a scene fails, so that closing is not turned into hiding to the tray.
pub static QUITTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scene {
    Check,
    Explain,
    Long,
    Batch,
    History,
    Empty,
    Quick,
    Settings,
    Short,
    Scoring,
    Error,
    NoModel,
    Edit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    WaitModel,
    WaitResult,
    Settle(u32),
    Shooting,
    Done,
}

pub struct Demo {
    scene: Scene,
    stage: Stage,
    shot: Option<PathBuf>,
    pub theme: Option<ThemeChoice>,
    folder: Option<PathBuf>,
    export: Option<PathBuf>,
    cell: usize,
    text: Option<PathBuf>,
    started: Instant,
    image: Arc<Mutex<Option<Arc<egui::ColorImage>>>>,
    /// When the scene began to settle: windows fade in, so a shot waits a moment as well as a few frames.
    settling: Option<Instant>,
    pub detail: Option<crate::settings::Detail>,
    pub zoom: Option<f32>,
}

impl Demo {
    pub fn from_env() -> Option<Self> {
        let scene = match std::env::var("LITMUS_DEMO").ok()?.as_str() {
            "check" => Scene::Check,
            "explain" => Scene::Explain,
            "long" => Scene::Long,
            "batch" => Scene::Batch,
            "history" => Scene::History,
            "empty" => Scene::Empty,
            "quick" => Scene::Quick,
            "settings" => Scene::Settings,
            "short" => Scene::Short,
            "scoring" => Scene::Scoring,
            "error" => Scene::Error,
            "nomodel" => Scene::NoModel,
            "edit" => Scene::Edit,
            other => panic!("LITMUS_DEMO={other}: use check, explain, long, batch, history, empty, quick, settings, short, scoring, error, nomodel or edit"),
        };
        let theme = std::env::var("LITMUS_THEME").ok().map(|t| match t.as_str() {
            "dark" => ThemeChoice::Dark,
            "light" => ThemeChoice::Light,
            other => panic!("LITMUS_THEME={other}: use light or dark"),
        });
        Some(Self {
            scene,
            stage: Stage::WaitModel,
            shot: std::env::var_os("LITMUS_SCREENSHOT").map(PathBuf::from),
            theme,
            folder: std::env::var_os("LITMUS_DEMO_FOLDER").map(PathBuf::from),
            export: std::env::var_os("LITMUS_DEMO_EXPORT").map(PathBuf::from),
            cell: std::env::var("LITMUS_DEMO_CELL").ok().and_then(|c| c.parse().ok()).unwrap_or(2),
            text: std::env::var_os("LITMUS_DEMO_TEXT").map(PathBuf::from),
            started: Instant::now(),
            image: Arc::new(Mutex::new(None)),
            settling: None,
            detail: std::env::var("LITMUS_DETAIL").ok().map(|d| match d.as_str() {
                "simple" => crate::settings::Detail::Simple,
                "detailed" => crate::settings::Detail::Detailed,
                "full" => crate::settings::Detail::Full,
                other => panic!("LITMUS_DETAIL={other}: use simple, detailed or full"),
            }),
            zoom: std::env::var("LITMUS_DEMO_ZOOM").ok().map(|z| z.parse().expect("LITMUS_DEMO_ZOOM is a number")),
        })
    }
}

fn fail(demo: &mut Demo, ctx: &egui::Context, error: &str) {
    QUITTING.store(true, std::sync::atomic::Ordering::Relaxed);
    eprintln!("litmus demo: {error}");
    if let Some(shot) = &demo.shot {
        let _ = std::fs::write(shot.with_extension("error.txt"), error);
    }
    demo.stage = Stage::Done;
    ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Close);
}

/// The text the check, explain and edit scenes read: the file `LITMUS_DEMO_TEXT` names, else the sample. A file
/// that does not read fails the scene rather than falling back to the sample.
fn scene_text(file: Option<&std::path::Path>) -> Result<String, String> {
    match file {
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("LITMUS_DEMO_TEXT {}: {e}", path.display())),
        None => Ok(crate::sample::TEXT.to_owned()),
    }
}

/// The long scene's text: the sample, numbered and repeated past one section (8,192 tokens).
fn long_text() -> String {
    (1..=36).map(|n| format!("{n}. {}", crate::sample::TEXT)).collect::<Vec<_>>().join("\n\n")
}

/// Advance the scene by one frame.
pub fn step(app: &mut LitmusApp, ctx: &egui::Context) {
    let Some(mut demo) = app.demo.take() else { return };
    ctx.request_repaint();
    if let Some(detail) = demo.detail.take() {
        app.settings.detail = detail;
    }
    if let Some(zoom) = demo.zoom.take() {
        ctx.set_zoom_factor(zoom);
    }
    // The pointer rests wherever the window opened: no tooltips in the pictures.
    ctx.all_styles_mut(|style| style.interaction.tooltip_delay = 3600.0);
    if demo.started.elapsed().as_secs() > 120 && demo.stage != Stage::Done {
        let popup = app.quick.running.as_ref().and_then(|r| r.popup.as_ref()).map(|p| match &p.state {
            crate::quick::PopupState::Scoring { job, .. } => format!("scoring job {job}"),
            crate::quick::PopupState::Shown { .. } => "shown".to_owned(),
            crate::quick::PopupState::Failed(error) => format!("failed: {error}"),
        });
        let message = format!("the scene did not finish within 120 seconds (stage {:?}, popup {popup:?}, model loaded {})", demo.stage, app.meta().is_some());
        fail(&mut demo, ctx, &message);
    }
    match demo.stage {
        Stage::WaitModel => match &app.model {
            ModelState::Loading => {}
            ModelState::Failed(_) if demo.scene == Scene::NoModel => demo.stage = Stage::Settle(12),
            ModelState::Failed(error) => {
                let error = error.clone();
                fail(&mut demo, ctx, &format!("the model did not load: {error}"));
            }
            ModelState::Ready(_) => {
                match demo.scene {
                    Scene::NoModel => fail(&mut demo, ctx, "the nomodel scene found a model: point LITMUS_MODEL at an empty folder"),
                    Scene::Short => {
                        app.input = "The bakery on the corner opens at six.".to_owned();
                        app.input_source = "pasted".into();
                        app.check();
                    }
                    Scene::Scoring => {
                        app.input = (0..12).map(|_| long_text()).collect::<Vec<_>>().join("

");
                        app.input_source = "pasted".into();
                        app.check();
                    }
                    Scene::Error => match &demo.folder {
                        Some(folder) => app.open_file(&folder.join("missing.docx")),
                        None => fail(&mut demo, ctx, "the error scene needs LITMUS_DEMO_FOLDER"),
                    },
                    Scene::Check | Scene::Explain | Scene::Edit => match scene_text(demo.text.as_deref()) {
                        Ok(text) => {
                            app.input = text;
                            app.input_source = "pasted".into();
                            app.check();
                        }
                        Err(error) => fail(&mut demo, ctx, &error),
                    },
                    Scene::Long => {
                        app.input = long_text();
                        app.input_source = "pasted".into();
                        app.check();
                    }
                    Scene::Batch => match &demo.folder {
                        Some(folder) => {
                            app.add_to_batch(std::slice::from_ref(folder));
                            app.run_batch();
                        }
                        None => fail(&mut demo, ctx, "the batch scene needs LITMUS_DEMO_FOLDER"),
                    },
                    Scene::History => app.tab = Tab::History,
                    Scene::Empty => {}
                    Scene::Settings => app.settings_open = true,
                    Scene::Quick => {
                        if let Err(error) = crate::views::quick::demo_start(app, ctx, "The bakery on the corner opens at six.") {
                            fail(&mut demo, ctx, &format!("quick check did not start: {error}"));
                        }
                    }
                }
                if demo.stage != Stage::Done {
                    demo.stage = Stage::WaitResult;
                }
            }
        },
        Stage::WaitResult if demo.scene == Scene::Quick => {
            if crate::views::quick::demo_shown(app) {
                demo.stage = Stage::Settle(12);
            }
        }
        Stage::WaitResult if demo.scene == Scene::Scoring => {
            if app.running.as_ref().is_some_and(|r| r.progress.fraction > 0.2) {
                demo.stage = Stage::Settle(1);
            } else if !app.busy() {
                fail(&mut demo, ctx, "the scoring scene finished before it could be photographed");
            }
        }
        Stage::WaitResult if demo.scene == Scene::Error => {
            if matches!(app.notice, Some(crate::app::Notice::Error { .. })) {
                demo.stage = Stage::Settle(12);
            } else if !app.busy() {
                fail(&mut demo, ctx, "the error scene showed no error");
            }
        }
        Stage::WaitResult => {
            if !app.busy() {
                if let Some(crate::app::Notice::Error { title, detail }) = &app.notice {
                    let error = format!("{title}: {detail}");
                    fail(&mut demo, ctx, &error);
                } else {
                    match demo.scene {
                        Scene::Explain => {
                            app.settings.detail = crate::settings::Detail::Full;
                            app.reveal_explain = true;
                        }
                        Scene::Edit => {
                            app.editing = true;
                            ctx.memory_mut(|m| m.request_focus(egui::Id::unique("litmus_editor")));
                        }
                        Scene::Check | Scene::Long => {
                            let cells = app.shown.as_ref().map_or(0, |s| s.scored.reading.cells.len());
                            if cells > 0 {
                                app.selected_cell = Some(demo.cell.min(cells - 1));
                            }
                        }
                        Scene::Batch => {
                            if let Some(folder) = &demo.export {
                                let written = std::fs::create_dir_all(folder)
                                    .and_then(|()| std::fs::write(folder.join("litmus_batch.csv"), crate::views::batch::batch_csv(app)))
                                    .and_then(|()| std::fs::write(folder.join("litmus_batch.json"), crate::views::batch::batch_json(app)));
                                if let Err(error) = written {
                                    fail(&mut demo, ctx, &format!("export: {error}"));
                                }
                            }
                        }
                        Scene::History | Scene::Empty | Scene::Quick | Scene::Settings | Scene::Short | Scene::Scoring | Scene::Error | Scene::NoModel => {}
                    }
                    if demo.stage != Stage::Done {
                        // A window fades in: give it time to finish.
                        demo.stage = Stage::Settle(if demo.scene == Scene::Settings { 60 } else { 12 });
                    }
                }
            }
        }
        Stage::Settle(0) if demo.settling.get_or_insert_with(Instant::now).elapsed().as_millis() < 700 && demo.scene != Scene::Scoring => {}
        Stage::Settle(0) => match demo.shot.clone() {
            Some(_) => {
                let slot = demo.image.clone();
                let waker = ctx.clone();
                let callback = egui::ScreenshotCallback::new(move |image| {
                    *slot.lock().expect("the screenshot slot") = Some(image);
                    waker.request_repaint();
                });
                let target = if demo.scene == Scene::Quick { crate::views::quick::popup_id() } else { egui::ViewportId::ROOT };
                ctx.send_viewport_cmd_to(target, egui::ViewportCommand::Screenshot(callback));
                demo.stage = Stage::Shooting;
            }
            None => demo.stage = Stage::Done,
        },
        Stage::Settle(frames) => demo.stage = Stage::Settle(frames - 1),
        Stage::Shooting => {
            let image = demo.image.lock().expect("the screenshot slot").take();
            if let (Some(image), Some(shot)) = (image, demo.shot.clone()) {
                let [width, height] = image.size;
                let rgba: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_srgba_unmultiplied()).collect();
                match std::fs::write(&shot, crate::png::encode(width as u32, height as u32, &rgba)) {
                    Ok(()) => {
                        eprintln!("litmus demo: wrote {}", shot.display());
                        demo.stage = Stage::Done;
                        app.quick.quitting = true;
                        ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Close);
                    }
                    Err(error) => fail(&mut demo, ctx, &format!("{}: {error}", shot.display())),
                }
            }
        }
        Stage::Done => {}
    }
    app.demo = Some(demo);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_text_is_the_named_file_else_the_sample() {
        assert_eq!(scene_text(None).unwrap(), crate::sample::TEXT);
        let file = std::env::temp_dir().join(format!("litmus_demo_text_{}.txt", std::process::id()));
        std::fs::write(&file, "A text chosen for the scene.").unwrap();
        assert_eq!(scene_text(Some(&file)).unwrap(), "A text chosen for the scene.");
        std::fs::remove_file(&file).unwrap();
        let error = scene_text(Some(&file)).unwrap_err();
        assert!(error.starts_with("LITMUS_DEMO_TEXT "), "a missing file is an error, never the sample: {error}");
    }
}
