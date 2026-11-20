//! The app's logic against a real release: `LITMUS_MODEL` names a release folder (the random-weights placeholder
//! does). Run with `cargo test --features release-tests`; the feature without the variable fails, never skips.
#![cfg(feature = "release-tests")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use litmus_app::heatmap;
use litmus_app::scoring::{Event, Scorer};
use litmus_engine::Engine;

fn release() -> PathBuf {
    PathBuf::from(std::env::var_os("LITMUS_MODEL").expect("LITMUS_MODEL names a release folder for these tests"))
}

fn engine() -> Engine {
    let files = litmus_files::release::read_folder(&release(), None).expect("the release reads");
    Engine::load(&files.model, &files.config, &files.tokenizer).expect("the release loads")
}

/// A text of two sections: the sample, numbered and repeated, with some characters outside ASCII.
fn long_text() -> String {
    (1..=36).map(|n| format!("{n}. Caf\u{e9} \u{2014} {}", litmus_app::sample::TEXT)).collect::<Vec<_>>().join("\n\n")
}

#[test]
fn cells_map_onto_the_text() {
    let engine = engine();
    for text in [litmus_app::sample::TEXT.to_owned(), long_text()] {
        let reading = engine.score(&text).unwrap();
        heatmap::check_places(&reading, &text).unwrap();
        let pages = heatmap::pages(&reading.cells, heatmap::CELLS_PER_PAGE);
        assert_eq!(pages.first().unwrap().bytes.start, 0);
        assert_eq!(pages.last().unwrap().bytes.end, text.len());
        for pair in pages.windows(2) {
            assert_eq!(pair[0].bytes.end, pair[1].bytes.start, "pages tile the text");
        }
        for (index, cell) in reading.cells.iter().enumerate() {
            assert_eq!(heatmap::cell_at_byte(&reading.cells, cell.place.start_byte), Some(index));
            let covering = heatmap::covering_windows(&reading.windows, cell);
            assert!(!covering.is_empty(), "cell {index} has a window over it");
            assert_eq!(covering.len(), cell.windows, "the engine's count of windows over cell {index}");
        }
        let page = &pages[0];
        let local = &text[page.bytes.clone()];
        for (range, index) in heatmap::sections(&reading.cells, page) {
            assert!(local.is_char_boundary(range.start) && local.is_char_boundary(range.end), "cell {index} cuts a character");
        }
        assert!(!litmus_app::wording::verdict_headline(&reading.verdict).is_empty());
        if reading.verdict.chunks > 1 {
            assert!(litmus_app::wording::flags(&reading).iter().any(|(title, _)| title.contains("ixed")), "a long text says whether it reads as mixed");
        }
    }
    assert!(engine.count_tokens(&long_text()) > 8192, "the long text spans two sections");
}

#[test]
fn a_misplaced_reading_is_refused() {
    let engine = engine();
    let reading = engine.score(litmus_app::sample::TEXT).unwrap();
    assert!(heatmap::check_places(&reading, &litmus_app::sample::TEXT[..100]).is_err());
}

fn wait(scorer: &mut Scorer, mut done: impl FnMut(&Event) -> bool) -> Vec<Event> {
    let started = Instant::now();
    let mut seen = Vec::new();
    loop {
        for event in scorer.poll() {
            let finished = done(&event);
            seen.push(event);
            if finished {
                return seen;
            }
        }
        assert!(started.elapsed() < Duration::from_secs(120), "the scorer answers within two minutes");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn the_scorer_thread_scores_caches_cancels_and_batches() {
    let cache = std::env::temp_dir().join(format!("litmus_app_cache_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    // SAFETY: set before the scorer thread starts; no other thread reads the environment in this test binary's run of it.
    unsafe { std::env::set_var("LITMUS_CACHE_DIR", &cache) };
    let mut scorer = Scorer::new(&egui::Context::default());
    scorer.load(Some(release().display().to_string()));
    let events = wait(&mut scorer, |e| matches!(e, Event::Loaded(_)));
    let Some(Event::Loaded(Ok(meta))) = events.last() else { panic!("the model loads") };
    assert!(meta.placeholder, "the test release is the random placeholder");

    let text = Arc::new(litmus_app::sample::TEXT.to_owned());
    let job = scorer.score(text.clone());
    let events = wait(&mut scorer, |e| matches!(e, Event::Scored { .. }));
    let Some(Event::Scored { job: answered, result: Ok(first) }) = events.last() else { panic!("the text scores") };
    assert_eq!(*answered, job);
    assert!(first.seconds.is_some(), "the first reading is computed");
    scorer.score(text);
    let events = wait(&mut scorer, |e| matches!(e, Event::Scored { .. }));
    let Some(Event::Scored { result: Ok(second), .. }) = events.last() else { panic!("the text scores again") };
    assert!(second.seconds.is_none(), "an unchanged text comes from the cache");
    assert_eq!(second.reading, first.reading);

    scorer.score(Arc::new(long_text()));
    scorer.cancel();
    let events = wait(&mut scorer, |e| matches!(e, Event::Scored { .. }));
    let Some(Event::Scored { result: Err(error), .. }) = events.last() else { panic!("a cancelled reading is an error") };
    assert_eq!(error, litmus_app::scoring::CANCELLED);

    let folder = std::env::temp_dir().join(format!("litmus_app_batch_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join("a.txt"), litmus_app::sample::TEXT).unwrap();
    std::fs::write(folder.join("b.html"), "<p>A short page about <b>bread</b>.</p>").unwrap();
    std::fs::write(folder.join("c.md"), "   ").unwrap();
    let files = litmus_files::walk::files_in(&folder).unwrap();
    let job = scorer.batch(files);
    let events = wait(&mut scorer, |e| matches!(e, Event::BatchDone { .. }));
    let results: Vec<(usize, bool)> = events
        .iter()
        .filter_map(|e| match e {
            Event::BatchFile { job: j, index, result } if *j == job => Some((*index, result.is_ok())),
            _ => None,
        })
        .collect();
    assert_eq!(results, [(0, true), (1, true), (2, false)], "the blank file is an error row, the others read");

    // A file opened in the Check screen is read and scored on the scoring thread; its reading carries its text.
    let job = scorer.score_file(folder.join("b.html"));
    let events = wait(&mut scorer, |e| matches!(e, Event::Scored { .. }));
    let Some(Event::Scored { job: answered, result: Ok(page) }) = events.last() else { panic!("the page reads") };
    assert_eq!((*answered, page.text.as_str()), (job, "A short page about bread."));
    assert!(events.iter().any(|e| matches!(e, Event::Progress { job: j, progress } if *j == job && progress.words == "Opening the file")));
    for (name, expected) in [("c.md", litmus_app::scoring::EMPTY), ("missing.docx", "missing.docx")] {
        scorer.score_file(folder.join(name));
        let events = wait(&mut scorer, |e| matches!(e, Event::Scored { .. }));
        let Some(Event::Scored { result: Err(error), .. }) = events.last() else { panic!("{name} does not read") };
        assert!(error.contains(expected), "{name}: {error}");
    }
    std::fs::remove_dir_all(&folder).unwrap();
    std::fs::remove_dir_all(&cache).unwrap();
}

/// Quick check reads on a thread of its own: its reading arrives while the main window's long reading still runs, and
/// a popup that closes stops its own reading without touching the main window's.
#[test]
fn quick_check_never_waits_for_the_main_window() {
    let mut scorer = Scorer::new(&egui::Context::default());
    scorer.set_cache(false);
    scorer.load(Some(release().display().to_string()));
    let events = wait(&mut scorer, |e| matches!(e, Event::Loaded(_)));
    assert!(matches!(events.last(), Some(Event::Loaded(Ok(_)))), "the model loads");
    // Long enough that the main reading still runs at the end, even with a small model on a fast machine; it is
    // cancelled there.
    let very_long = Arc::new((0..64).map(|_| long_text()).collect::<Vec<_>>().join("\n\n"));
    let main = scorer.score(very_long.clone());
    let quick = scorer.score_quiet(Arc::new(litmus_app::sample::TEXT.to_owned()));
    let events = wait(&mut scorer, |e| matches!(e, Event::Scored { .. }));
    let Some(Event::Scored { job, result: Ok(_) }) = events.last() else { panic!("the quick text scores") };
    assert_eq!(*job, quick, "the quick reading arrives first, while the long one runs");

    let long_quick = scorer.score_quiet(very_long);
    wait(&mut scorer, |e| matches!(e, Event::Progress { job, .. } if *job == long_quick));
    scorer.cancel_quiet(main); // not quick check's job: nothing happens
    scorer.cancel_quiet(long_quick);
    let events = wait(&mut scorer, |e| matches!(e, Event::Scored { job, .. } if *job == long_quick));
    let Some(Event::Scored { result: Err(error), .. }) = events.last() else { panic!("a cancelled quick reading is an error") };
    assert_eq!(error, litmus_app::scoring::CANCELLED);
    assert!(!events.iter().any(|e| matches!(e, Event::Scored { job, .. } if *job == main)), "the main window's reading still runs");
    scorer.cancel();
    let events = wait(&mut scorer, |e| matches!(e, Event::Scored { job, .. } if *job == main));
    let Some(Event::Scored { result: Err(error), .. }) = events.last() else { panic!("the main window's cancel stops its reading") };
    assert_eq!(error, litmus_app::scoring::CANCELLED);
}
