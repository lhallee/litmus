//! The engine reproduces `../student/reference.py` on its fixtures within SPEC.md's tolerances (Parity): the standard
//! reading and token ids exactly; each chunk's pooled vector, query, key and window vectors; the whole reading of
//! every chunk and window, the heat map's cells and the document's verdict; and streaming in pieces of 1, 7, W - 1, W
//! and W + 1 tokens equals reading at once. A text under 512 tokens that reads beyond the table is held to the table
//! times `sqrt(512 / n)` with its discrete values left out (a flipped activation; SPEC.md, Parity), and the set
//! fails if more than half of its short texts read so.
//!
//! `LITMUS_STUDENT_FIXTURES` names the folder `reference.py fixtures` wrote (`tiny/`, `s192_l4/`, and the same in
//! integer attention, `tiny_integer/`, `s192_l4_integer/`, and with int8 output, `tiny_int8/`, `s192_l4_integer_int8/`);
//! `LITMUS_TOKENIZER_DIR` names a folder with ModernBERT's `tokenizer.json`.
#![cfg(feature = "parity")]

use std::path::PathBuf;

use litmus_engine::reading::{ChunkCall, combine, level_key};
use litmus_engine::release::{Output, ReleaseConfig};
use litmus_engine::model::ColumnChoice;
use litmus_engine::{Backend, DocumentReading, Engine, Origin, Place, Reading, Vectors, text};
use serde_json::Value;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} (engine/README.md, Tests)")))
}

fn load(set: &str) -> (Engine, Value, Value) {
    let folder = env_path("LITMUS_STUDENT_FIXTURES").join(set);
    let read = |name: &str| std::fs::read(folder.join(name)).unwrap_or_else(|e| panic!("{}: {e}", folder.join(name).display()));
    let tokenizer = std::fs::read(env_path("LITMUS_TOKENIZER_DIR").join("tokenizer.json")).unwrap();
    let engine = Engine::load(&read("model.safetensors"), &read("config.json"), &tokenizer).unwrap();
    let config: Value = serde_json::from_slice(&read("config.json")).unwrap();
    let fixtures: Value = serde_json::from_slice(&read("fixtures.json")).unwrap();
    (engine, config, fixtures)
}

/// Largest deviations seen, by value kind, for the report.
#[derive(Default, Debug)]
struct Worst {
    entries: Vec<(String, f64)>,
}

impl Worst {
    fn see(&mut self, kind: &str, deviation: f64) {
        match self.entries.iter_mut().find(|(name, _)| name == kind) {
            Some((_, worst)) => *worst = worst.max(deviation),
            None => self.entries.push((kind.to_owned(), deviation)),
        }
    }

    fn merge(&mut self, other: Worst) {
        for (kind, deviation) in other.entries {
            self.see(&kind, deviation);
        }
    }
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap_or_else(|| panic!("not an array: {value}")).iter().map(|v| v.as_f64().unwrap()).collect()
}

/// Thresholds and edges near which the engine may fall on the other side (SPEC.md, Parity).
fn near_any(value: f64, edges: impl IntoIterator<Item = f64>) -> bool {
    edges.into_iter().any(|edge| (edge - value).abs() <= 1e-3)
}

fn group<'a>(groups: &'a Value, bin: usize) -> &'a Value {
    groups.as_array().unwrap().iter().find(|g| g["bins"].as_array().unwrap().iter().any(|b| b.as_u64() == Some(bin as u64))).unwrap()
}

fn thresholds(group: &Value) -> Vec<f64> {
    group["thresholds"].as_array().unwrap().iter().filter_map(Value::as_f64).collect()
}

fn origin_name(origin: Origin) -> &'static str {
    origin.name()
}

/// How a text is held to SPEC.md's table (Parity): at the table itself, with every discrete value compared, or, for
/// a short text that a flipped activation moved, at the table times `scale` with the discrete values left out.
#[derive(Clone, Copy)]
struct Hold {
    scale: f64,
    discrete: bool,
}

const TABLE: Hold = Hold { scale: 1.0, discrete: true };
/// The most an int8 query or key code may differ from the reference's (SPEC.md, Parity, Int8 output), in codes.
const CODE_GAP: f64 = 1.0;
/// A reading whose query or key has a code the reference's does not is held to this multiple of the table's continuous
/// tolerances, its discrete values compared as usual (SPEC.md, Parity, Int8 output): one code is a step of 1/127 of the
/// vector's largest component, which the pair logit's steep sigmoid and the spread's percentile turn into up to 1.9e-3
/// in the three-way probabilities and 2.5e-3 in a spread percentile (the measured largest, on `s192_l4_integer_int8`).
const MOVED_CODE_SCALE: f64 = 2.0;

/// Which readings of a text have a query or key code that differs from the reference's, in int8 output (else empty):
/// the document's chunks and windows, in order.
#[derive(Default)]
struct Moved {
    chunks: Vec<bool>,
    windows: Vec<bool>,
}

fn held(hold: Hold, moved: Option<&bool>) -> Hold {
    if moved == Some(&true) { Hold { scale: hold.scale * MOVED_CODE_SCALE, ..hold } } else { hold }
}

fn codes(value: &Value) -> Vec<i64> {
    value.as_array().unwrap_or_else(|| panic!("not an array: {value}")).iter().map(|v| v.as_i64().unwrap()).collect()
}
/// Texts shorter than this may be read at the scaled table: `sqrt(SHORT_TEXT / n)` for a text of `n` tokens.
const SHORT_TEXT: usize = 512;

/// Compare one reading; a discrete value may differ only where the continuous value it is read from lies within
/// 1e-3 of a table threshold or edge. Returns whether every discrete value equals the reference's.
fn compare(label: &str, got: &Reading, want: &Value, calibration: &Value, hold: Hold, worst: &mut Worst, failures: &mut Vec<String>) -> bool {
    let Hold { scale, discrete } = hold;
    let mut fail = |what: String| failures.push(format!("{label}: {what}"));
    let close = |kind: &str, a: f64, b: f64, tolerance: f64, worst: &mut Worst, fail: &mut dyn FnMut(String)| {
        let deviation = (a - b).abs();
        worst.see(kind, deviation);
        if deviation > tolerance || deviation.is_nan() {
            fail(format!("{kind}: {a} against {b}"));
        }
    };
    if got.n_tokens as u64 != want["n_tokens"].as_u64().unwrap() {
        fail(format!("n_tokens {} against {}", got.n_tokens, want["n_tokens"]));
        return false;
    }
    let bin = want["length_bin"].as_u64().unwrap() as usize;
    if got.length_bin != bin {
        fail(format!("length bin {} against {bin}", got.length_bin));
    }
    for (kind, a, b) in [("mean", got.mean, floats(&want["mean"])), ("spread", got.spread, floats(&want["spread"])), ("share", got.share, floats(&want["share"]))] {
        for o in 0..3 {
            close(kind, a[o], b[o], 1e-3 * scale, worst, &mut fail);
        }
    }
    let score = want["score"].as_f64().unwrap();
    close("score", got.score, score, 1e-3 * scale, worst, &mut fail);
    let closeness = want["closeness"].as_f64().unwrap();
    close("closeness", got.closeness, closeness, 1e-3 * scale, worst, &mut fail);
    let share = floats(&want["share"]);

    // Grid p-values, sets and calls: equal unless the share is near a threshold of its group.
    let p_values = floats(&want["p_values"]);
    let mut p_free = [true; 3];
    for origin in Origin::ALL {
        let o = origin.index();
        let near = near_any(share[o], thresholds(group(&calibration["conformal"][origin_name(origin)], bin)));
        p_free[o] = !near;
        if got.p_values[o] != p_values[o] {
            if near {
                worst.see("p-value flips near a threshold", 1.0);
            } else if discrete {
                fail(format!("p-value of {} {} against {}", origin.name(), got.p_values[o], p_values[o]));
            }
        }
    }
    for (key, set) in &got.sets {
        let want_set: Vec<&str> = want["sets"][key].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let got_set: Vec<&str> = set.iter().map(|o| o.name()).collect();
        if got_set != want_set && p_free.iter().all(|&free| free) && discrete {
            fail(format!("set at {key} {got_set:?} against {want_set:?}"));
        }
        if got.not_human[key] != want["not_human"][key].as_bool().unwrap() && p_free[0] && discrete {
            fail(format!("not-human call at {key}"));
        }
    }

    // Venn-Abers: equal while the block is the same.
    let venn = group(&calibration["probability"], bin);
    let block_free = !near_any(score, floats(&venn["edges"]));
    let interval = floats(&want["interval"]);
    let probability = want["probability"].as_f64().unwrap();
    if block_free && discrete {
        close("probability (Venn-Abers)", got.probability, probability, 1e-12, worst, &mut fail);
        close("interval", got.interval[0], interval[0], 1e-12, worst, &mut fail);
        close("interval", got.interval[1], interval[1], 1e-12, worst, &mut fail);
    }
    let three_way = floats(&want["three_way"]);
    let percentile = floats(&want["spread_percentile"]);
    for o in 0..3 {
        close("three-way", got.three_way[o], three_way[o], 2e-3 * scale, worst, &mut fail);
        close("spread percentile", got.spread_percentile[o], percentile[o], 2e-3 * scale, worst, &mut fail);
    }
    let closeness_free = !near_any(closeness, thresholds(group(&calibration["closeness"], bin)));
    if closeness_free && discrete {
        if got.closeness_p != want["closeness_p"].as_f64().unwrap() {
            fail(format!("closeness p-value {} against {}", got.closeness_p, want["closeness_p"]));
        }
        if got.out_of_reference != want["out_of_reference"].as_bool().unwrap() {
            fail("out-of-reference flag".into());
        }
    }
    // The explanation: counts equal unless the value sits within 1e-3 of a calibration score; nearest entries the
    // same unless their cosines are within 1e-3 of each other.
    let explain = &want["explain"];
    let free = p_free.iter().all(|&free| free) && block_free && closeness_free && discrete;
    // Whether every discrete value the verdict and the cells read came out the same as the reference's.
    let same = got.p_values.as_slice() == floats(&want["p_values"]).as_slice()
        && got.interval.as_slice() == floats(&want["interval"]).as_slice()
        && got.closeness_p == want["closeness_p"].as_f64().unwrap()
        && got.sets.iter().all(|(key, set)| set.iter().map(|o| o.name()).eq(want["sets"][key].as_array().unwrap().iter().map(|v| v.as_str().unwrap())))
        && got.strip.unlike_reference == want["strip"]["unlike_reference"].as_bool().unwrap();
    let Some(got_explain) = &got.explain else {
        if !explain.is_null() {
            fail("explanation missing".into());
        }
        compare_strip(got, want, free, scale, worst, &mut fail);
        return same;
    };
    if explain.is_null() {
        fail("explanation where the reference has none".into());
    }
    for (index, count) in got_explain.conformal.iter().enumerate() {
        let expected = &explain["conformal"][index];
        if count.n as u64 != expected["n"].as_u64().unwrap() {
            fail(format!("explain: conformal n of {}", count.origin.name()));
        }
        let deviation = (count.count as i64 - expected["count"].as_i64().unwrap()).abs();
        worst.see("explain conformal count", deviation as f64);
        if deviation > 0 && p_free[index] && discrete {
            fail(format!("explain: conformal count of {} {} against {}", count.origin.name(), count.count, expected["count"]));
        }
    }
    if (got_explain.closeness.count as i64 - explain["closeness"]["count"].as_i64().unwrap()).abs() > 2 && discrete {
        fail(format!("explain: closeness count {} against {}", got_explain.closeness.count, explain["closeness"]["count"]));
    }
    for (entry, expected) in got_explain.nearest.iter().zip(explain["nearest"].as_array().unwrap()) {
        close("explain nearest cosine", entry.cosine, expected["cosine"].as_f64().unwrap(), 1e-3 * scale, worst, &mut fail);
        if entry.index as u64 == expected["index"].as_u64().unwrap() {
            // One pair's probability, the steepest function of a cosine in the reading: twice the table's 1e-3.
            close("explain nearest probability", entry.probability, expected["probability"].as_f64().unwrap(), 2e-3 * scale, worst, &mut fail);
            if entry.tokens != expected["tokens"].as_i64().unwrap() || entry.origin.name() != expected["origin"].as_str().unwrap() {
                fail(format!("explain: nearest entry {} tokens or origin", entry.index));
            }
            for (field, value) in &entry.metadata {
                if expected[field].as_str() != Some(value.as_str()) {
                    fail(format!("explain: nearest entry {} field {field}", entry.index));
                }
            }
        } else {
            worst.see("explain nearest order swaps (near ties)", 1.0);
        }
    }
    compare_strip(got, want, free, scale, worst, &mut fail);
    same
}

fn compare_strip(got: &Reading, want: &Value, free: bool, scale: f64, worst: &mut Worst, fail: &mut dyn FnMut(String)) {
    let strip = &want["strip"];
    let mut close = |kind: &str, a: f64, b: f64| {
        worst.see(kind, (a - b).abs());
        if (a - b).abs() > 2e-3 * scale {
            fail(format!("{kind}: {a} against {b}"));
        }
    };
    close("strip position", got.strip.position, strip["position"].as_f64().unwrap());
    if free {
        close("strip band", got.strip.band_lo, strip["band_lo"].as_f64().unwrap());
        close("strip band", got.strip.band_hi, strip["band_hi"].as_f64().unwrap());
        if got.strip.unlike_reference != strip["unlike_reference"].as_bool().unwrap() {
            fail("strip unlike_reference".into());
        }
    }
}

fn compare_document(label: &str, got: &DocumentReading, want: &Value, calibration: &Value, hold: Hold, moved: &Moved, worst: &mut Worst, failures: &mut Vec<String>) {
    let scale = hold.scale;
    let span = |value: &Value| (value["start"].as_u64().unwrap() as usize, value["end"].as_u64().unwrap() as usize);
    let (chunks, windows, cells) = (want["chunks"].as_array().unwrap(), want["windows"].as_array().unwrap(), want["cells"].as_array().unwrap());
    if got.n_tokens as u64 != want["n_tokens"].as_u64().unwrap() || got.chunk_tokens as u64 != want["chunk_tokens"].as_u64().unwrap() {
        failures.push(format!("{label}: n_tokens or chunk_tokens"));
    }
    if got.chunks.len() != chunks.len() || got.windows.len() != windows.len() || got.cells.len() != cells.len() {
        failures.push(format!(
            "{label}: {} chunks, {} windows, {} cells against {}, {}, {}",
            got.chunks.len(),
            got.windows.len(),
            got.cells.len(),
            chunks.len(),
            windows.len(),
            cells.len()
        ));
        return;
    }
    let mut chunks_same = true;
    for (index, (chunk, expected)) in got.chunks.iter().zip(chunks).enumerate() {
        if (chunk.place.start, chunk.place.end) != span(expected) {
            failures.push(format!("{label} chunk {index}: span"));
        }
        chunks_same &= compare(&format!("{label} chunk {index}"), &chunk.reading, expected, calibration, held(hold, moved.chunks.get(index)), worst, failures);
    }
    let mut windows_same = true;
    for (index, (window, expected)) in got.windows.iter().zip(windows).enumerate() {
        if (window.place.start, window.place.end) != span(expected) || window.chunk as u64 != expected["chunk"].as_u64().unwrap() {
            failures.push(format!("{label} window {index}: span or chunk"));
        }
        windows_same &= compare(&format!("{label} window {index}"), &window.reading, expected, calibration, held(hold, moved.windows.get(index)), worst, failures);
    }
    for (index, (cell, expected)) in got.cells.iter().zip(cells).enumerate() {
        if (cell.place.start, cell.place.end) != span(expected) || cell.windows as u64 != expected["windows"].as_u64().unwrap() {
            failures.push(format!("{label} cell {index}: span or window count"));
        }
        for (kind, a, key, apply) in [
            ("cell position", cell.position, "position", true),
            ("cell spread", cell.spread, "spread", true),
            ("cell band", cell.band_lo, "band_lo", windows_same),
            ("cell band", cell.band_hi, "band_hi", windows_same),
        ] {
            let deviation = (a - expected[key].as_f64().unwrap()).abs();
            worst.see(kind, deviation);
            if apply && deviation > 2e-3 * scale {
                failures.push(format!("{label} cell {index}: {kind} {a} against {}", expected[key]));
            }
        }
    }
    // The verdict: its discrete values are exact while every chunk's are; the continuous ones within tolerance.
    let (verdict, expected) = (&got.verdict, &want["verdict"]);
    let mut check = |kind: &str, a: f64, b: f64, tolerance: f64| {
        worst.see(kind, (a - b).abs());
        if (a - b).abs() > tolerance {
            failures.push(format!("{label} verdict: {kind} {a} against {b}"));
        }
    };
    let three_way = floats(&expected["three_way"]);
    for o in 0..3 {
        check("verdict three-way", verdict.three_way[o], three_way[o], 2e-3 * scale);
    }
    check("verdict strip position", verdict.strip.position, expected["strip"]["position"].as_f64().unwrap(), 2e-3 * scale);
    check("verdict chunk level", verdict.chunk_level, expected["chunk_level"].as_f64().unwrap(), 1e-15);
    if !chunks_same {
        worst.see("verdict discrete values skipped (a chunk flipped)", 1.0);
        return;
    }
    let p_values = floats(&expected["p_values"]);
    for o in 0..3 {
        check("verdict merged p-value", verdict.p_values[o], p_values[o], 1e-12);
    }
    check("verdict probability", verdict.probability, expected["probability"].as_f64().unwrap(), 1e-12);
    check("verdict disagreement", verdict.disagreement, expected["disagreement"].as_f64().unwrap(), 1e-12);
    let interval = floats(&expected["interval"]);
    check("verdict interval", verdict.interval[0], interval[0], 1e-12);
    check("verdict interval", verdict.interval[1], interval[1], 1e-12);
    check("verdict closeness p", verdict.closeness_p, expected["closeness_p"].as_f64().unwrap(), 1e-12);
    let counts = (verdict.model_chunks as u64, verdict.human_chunks as u64, verdict.mixed_document, verdict.chunks as u64);
    let want_counts = (
        expected["model_chunks"].as_u64().unwrap(),
        expected["human_chunks"].as_u64().unwrap(),
        expected["mixed_document"].as_bool().unwrap(),
        expected["chunks"].as_u64().unwrap(),
    );
    if counts != want_counts {
        failures.push(format!("{label} verdict: chunk counts {counts:?} against {want_counts:?}"));
    }
    for level in [0.05, 0.1] {
        let key = level_key(level);
        let got_set: Vec<&str> = verdict.sets[&key].iter().map(|o| o.name()).collect();
        let want_set: Vec<&str> = expected["sets"][&key].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        if got_set != want_set {
            failures.push(format!("{label} verdict: set at {key} {got_set:?} against {want_set:?}"));
        }
    }
}

/// Two JSON values equal but for float rounding (1e-12): same keys, same strings and booleans, same counts.
fn same_json(label: &str, got: &Value, want: &Value, worst: &mut Worst, failures: &mut Vec<String>) {
    match (got, want) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            let deviation = (a.as_f64().unwrap() - b.as_f64().unwrap()).abs();
            worst.see("combination value", deviation);
            if deviation > 1e-12 {
                failures.push(format!("{label}: {a} against {b}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            let (mut keys_a, mut keys_b): (Vec<_>, Vec<_>) = (a.keys().collect(), b.keys().collect());
            keys_a.sort();
            keys_b.sort();
            if keys_a != keys_b {
                failures.push(format!("{label}: fields {keys_a:?} against {keys_b:?}"));
                return;
            }
            for key in a.keys() {
                same_json(&format!("{label}.{key}"), &a[key], &b[key], worst, failures);
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (index, (x, y)) in a.iter().zip(b).enumerate() {
                same_json(&format!("{label}[{index}]"), x, y, worst, failures);
            }
        }
        _ if got == want => {}
        _ => failures.push(format!("{label}: {got} against {want}")),
    }
}

/// One text's vectors and document reading against the fixture's, held as `hold` says: the largest deviations and
/// the values outside the tolerances.
fn compare_text(label: &str, record: &Value, reading: &DocumentReading, vectors: &Vectors, calibration: &Value, hold: Hold) -> (Worst, Vec<String>) {
    let (mut worst, mut failures) = (Worst::default(), Vec::new());
    // Layer and token states: a debugging aid (SPEC.md, Parity); report the largest deviation.
    for (stage, rows) in record["layer_states"].as_array().unwrap().iter().enumerate() {
        for (row, values) in rows.as_array().unwrap().iter().enumerate() {
            for (a, b) in vectors.trace.layer_states[stage][row].iter().zip(floats(values)) {
                worst.see(&format!("layer state {stage}"), (*a as f64 - b).abs());
            }
        }
    }
    for (row, values) in record["token_states"].as_array().unwrap().iter().enumerate() {
        for (a, b) in vectors.trace.token_states[row].iter().zip(floats(values)) {
            worst.see("token state", (*a as f64 - b).abs());
        }
    }
    let want_chunks = record["chunks"].as_array().unwrap();
    assert_eq!(vectors.chunks.len(), want_chunks.len(), "{label}: chunks");
    let mut moved = Moved::default();
    for (chunk, (got, want)) in vectors.chunks.iter().zip(want_chunks).enumerate() {
        let window_pooled: Vec<f32> = got.window_pooled.concat();
        let want_windows: Vec<f64> = want["window_pooled"].as_array().unwrap().iter().flat_map(floats).collect();
        for (kind, got, want) in [("pooled (relative)", &got.pooled, floats(&want["pooled"])), ("window pooled (relative)", &window_pooled, want_windows)] {
            assert_eq!(got.len(), want.len(), "{label} chunk {chunk}: {kind} length");
            for (a, b) in got.iter().zip(want) {
                let a = *a as f64;
                worst.see(kind, (a - b).abs() / (1.0 + b.abs()));
                if (a - b).abs() > 3e-3 * hold.scale * (1.0 + b.abs()) {
                    failures.push(format!("{label} chunk {chunk}: {kind} {a} against {b}"));
                }
            }
        }
        if got.query_codes.is_empty() {
            for (kind, got, want) in [("query", &got.query, &want["query"]), ("key", &got.key, &want["key"])] {
                for (a, b) in got.iter().zip(floats(want)) {
                    worst.see(kind, (*a as f64 - b).abs());
                    if (*a as f64 - b).abs() > 1e-3 * hold.scale {
                        failures.push(format!("{label} chunk {chunk}: {kind} {a} against {b}"));
                    }
                }
            }
        } else {
            // Int8 output: the fixture holds the reference's codes (integers), and a code differs only where the head's float
            // output lay within rounding error of half a step (SPEC.md, Parity).
            for (kind, got, want) in [("query code", &got.query_codes, &want["query"]), ("key code", &got.key_codes, &want["key"])] {
                assert_eq!(got.len(), want.as_array().unwrap().len(), "{label} chunk {chunk}: {kind} width");
                for (a, b) in got.iter().zip(floats(want)) {
                    let gap = (*a as f64 - b).abs();
                    worst.see(kind, gap);
                    worst.see(&format!("{kind}s differing"), f64::from(gap != 0.0));
                    if gap > (CODE_GAP * hold.scale).ceil() {
                        failures.push(format!("{label} chunk {chunk}: {kind} {a} against {b}"));
                    }
                }
            }
            let differs = |got: &[i8], want: &Value| got.iter().zip(codes(want)).any(|(&a, b)| a as i64 != b);
            moved.chunks.push(differs(&got.query_codes, &want["query"]) || differs(&got.key_codes, &want["key"]));
            assert_eq!(got.window_query_codes.len(), want["window_query"].as_array().unwrap().len(), "{label} chunk {chunk}: windows");
            for (window, (query, key)) in got.window_query_codes.iter().zip(&got.window_key_codes).enumerate() {
                let (want_query, want_key) = (&want["window_query"][window], &want["window_key"][window]);
                let gap = query.iter().zip(codes(want_query)).chain(key.iter().zip(codes(want_key))).map(|(&a, b)| (a as i64 - b).abs()).max().unwrap();
                worst.see("window code", gap as f64);
                if gap as f64 > (CODE_GAP * hold.scale).ceil() {
                    failures.push(format!("{label} chunk {chunk} window {window}: a code is {gap} steps from the reference's"));
                }
                moved.windows.push(differs(query, want_query) || differs(key, want_key));
            }
        }
    }
    if !moved.chunks.is_empty() {
        worst.see("most readings with a moved code in a text", moved.chunks.iter().chain(&moved.windows).filter(|&&moved| moved).count() as f64);
    }
    compare_document(label, reading, &record["reading"], calibration, hold, &moved, &mut worst, &mut failures);
    (worst, failures)
}

/// Int8 output only: the engine's head and bank on the reference's own pooled vectors, which isolates the integer compare
/// from the encoder's float rounding. Every code equals the reference's but for a head output within rounding error of a
/// half (one step apart), and a reading whose codes all equal the reference's is the reference's to float64 rounding.
fn check_pooled(engine: &Engine, record: &Value, label: &str, worst: &mut Worst, failures: &mut Vec<String>) {
    let reading = &record["reading"];
    let windows = reading["windows"].as_array().unwrap();
    for (chunk, want) in record["chunks"].as_array().unwrap().iter().enumerate() {
        let (start, end) = (want["start"].as_u64().unwrap() as usize, want["end"].as_u64().unwrap() as usize);
        let in_chunk: Vec<&Value> = windows.iter().filter(|window| window["chunk"].as_u64() == Some(chunk as u64)).collect();
        let mut pooled: Vec<f32> = floats(&want["pooled"]).iter().map(|&v| v as f32).collect();
        let mut n_tokens = vec![end - start];
        for (window, vector) in in_chunk.iter().zip(want["window_pooled"].as_array().unwrap()) {
            pooled.extend(floats(vector).iter().map(|&v| v as f32));
            n_tokens.push((window["end"].as_u64().unwrap() - window["start"].as_u64().unwrap()) as usize);
        }
        let (readings, queries, keys) = engine.read_pooled(&pooled, &n_tokens);
        let w = queries.len() / readings.len();
        let wanted: Vec<&Value> = std::iter::once(&reading["chunks"][chunk]).chain(in_chunk.iter().copied()).collect();
        for (index, (got, wanted)) in readings.iter().zip(&wanted).enumerate() {
            let (want_query, want_key) = match index {
                0 => (codes(&want["query"]), codes(&want["key"])),
                _ => (codes(&want["window_query"][index - 1]), codes(&want["window_key"][index - 1])),
            };
            let gap = (queries[index * w..(index + 1) * w].iter().zip(&want_query).chain(keys[index * w..(index + 1) * w].iter().zip(&want_key)))
                .map(|(&a, &b)| (a as i64 - b).abs())
                .max()
                .unwrap();
            worst.see("head code on the reference's pooled vector", gap as f64);
            let at = format!("{label} chunk {chunk} reading {index}");
            if gap > 1 {
                failures.push(format!("{at}: a code is {gap} steps from the reference's given the same pooled vector"));
            }
            if gap == 0 {
                for (kind, a, b) in [("mean", got.mean, "mean"), ("spread", got.spread, "spread"), ("share", got.share, "share")] {
                    for (o, want) in floats(&wanted[b]).into_iter().enumerate() {
                        worst.see(&format!("{kind} on the reference's pooled vector"), (a[o] - want).abs());
                        if (a[o] - want).abs() > 1e-9 {
                            failures.push(format!("{at}: {kind} {} against {want}", a[o]));
                        }
                    }
                }
                let closeness = wanted["closeness"].as_f64().unwrap();
                worst.see("closeness on the reference's pooled vector", (got.closeness - closeness).abs());
                if (got.closeness - closeness).abs() > 1e-9 {
                    failures.push(format!("{at}: closeness {} against {closeness}", got.closeness));
                }
            } else {
                worst.see("readings with a code one step off, given the same pooled vector", 1.0);
            }
        }
    }
}

fn run_set(set: &str) {
    let (mut engine, config, fixtures) = load(set);
    let calibration = &config["calibration"];
    let block = config["model"]["block_size"].as_u64().unwrap() as usize;
    let mut worst = Worst::default();
    let mut failures = Vec::new();
    // Short texts read at the scaled table, their deviations apart from the others', and the count of short texts.
    let (mut worst_flipped, mut flipped, mut short) = (Worst::default(), Vec::new(), 0usize);
    for (index, record) in fixtures["texts"].as_array().unwrap().iter().enumerate() {
        let label = format!("{set} text {index}");
        let text = record["text"].as_str().unwrap();
        assert_eq!(text::standard_reading(text), record["standard_reading"].as_str().unwrap(), "{label}: standard reading");
        let ids: Vec<u32> = engine.tokenize(text).iter().map(|token| token.id).collect();
        let want_ids: Vec<u32> = record["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(ids, want_ids, "{label}: token ids");

        let keep = fixtures["keep_rows"].as_u64().unwrap() as usize;
        let (reading, vectors) = engine.trace_ids(&ids, ids.len(), keep).unwrap();
        let (seen, found) = compare_text(&label, record, &reading, &vectors, calibration, TABLE);
        if found.is_empty() || ids.len() >= SHORT_TEXT {
            worst.merge(seen);
            failures.extend(found);
        } else {
            // Beyond the table in a short text: a flipped activation (SPEC.md, Parity), held to the scaled table.
            let hold = Hold { scale: (SHORT_TEXT as f64 / ids.len() as f64).sqrt(), discrete: false };
            let (seen, found) = compare_text(&label, record, &reading, &vectors, calibration, hold);
            worst_flipped.merge(seen);
            failures.extend(found);
            flipped.push(format!("text {index} ({} tokens)", ids.len()));
        }
        short += usize::from(ids.len() < SHORT_TEXT);
        if engine.info().output == Output::Int8 {
            check_pooled(&engine, record, &label, &mut worst, &mut failures);
        }

        // The text path (streamed tokenization) equals the id path but for the places' offsets into the text, which
        // tile it.
        let mut scored = engine.score(text).unwrap();
        let whole = (0, text.len(), 0, text.chars().count());
        for (kind, places) in [("chunks", scored.chunks.iter().map(|c| c.place).collect::<Vec<_>>()), ("cells", scored.cells.iter().map(|c| c.place).collect())] {
            let (first, last) = (places[0], places[places.len() - 1]);
            assert_eq!((first.start_byte, last.end_byte, first.start_char, last.end_char), whole, "{label}: {kind} tile the text");
            for pair in places.windows(2) {
                assert_eq!((pair[0].end_byte, pair[0].end_char), (pair[1].start_byte, pair[1].start_char), "{label}: {kind} are contiguous");
            }
        }
        for window in &scored.windows {
            let place = window.place;
            assert!(text.is_char_boundary(place.start_byte) && text.is_char_boundary(place.end_byte), "{label}: window at character boundaries");
            assert_eq!(text[..place.start_byte].chars().count(), place.start_char, "{label}: characters before a window");
            assert_eq!(text[..place.end_byte].chars().count(), place.end_char, "{label}: characters to a window's end");
        }
        let positions_only = |place: &mut Place| *place = Place { start: place.start, end: place.end, ..Place::default() };
        scored.chunks.iter_mut().for_each(|c| positions_only(&mut c.place));
        scored.windows.iter_mut().for_each(|w| positions_only(&mut w.place));
        scored.cells.iter_mut().for_each(|c| positions_only(&mut c.place));
        assert_eq!(scored, reading, "{label}: score(text) against score_ids");
        // Streaming in pieces equals reading at once, exactly in this implementation.
        for piece in [1, 7, block - 1, block, block + 1] {
            assert_eq!(engine.score_ids(&ids, piece).unwrap(), reading, "{label}: pieces of {piece}");
        }
        // The surprise reading (SPEC.md, Surprise): the mean log probability within 2e-3 (scaled for a short text),
        // percentiles within 1e-2, and the text path equal to the id path.
        assert_eq!(engine.info().surprise, record.get("surprise").is_some(), "{label}: a surprise reading with the head");
        if let Some(want) = record.get("surprise") {
            let got = engine.surprise_ids(&ids).unwrap();
            assert_eq!(engine.surprise(text).unwrap(), got, "{label}: surprise(text) against surprise_ids");
            assert_eq!(got.n_tokens, ids.len(), "{label}: surprise tokens");
            let scale = if ids.len() < SHORT_TEXT { (SHORT_TEXT as f64 / ids.len() as f64).sqrt() } else { 1.0 };
            let deviation = (got.mean_log_probability - want["mean_log_probability"].as_f64().unwrap()).abs();
            worst.see("surprise mean log probability", deviation / scale);
            if deviation > 2e-3 * scale {
                failures.push(format!("{label}: surprise mean log probability {} against {}", got.mean_log_probability, want["mean_log_probability"]));
            }
            let percentile = got.percentile.expect("a release with the head carries its surprise table");
            for (a, b) in percentile.iter().zip(floats(&want["percentile"])) {
                worst.see("surprise percentile", (a - b).abs());
                if (a - b).abs() > 1e-2 {
                    failures.push(format!("{label}: surprise percentile {a} against {b}"));
                }
            }
        }
        // The `down` product's path changes no bit: the dense path alone, the column path alone and the measured
        // choice, on every backend, give the same reading, pooled vectors and token states.
        for backend in Backend::available() {
            engine.set_backend(backend).unwrap();
            for choice in [ColumnChoice::Dense, ColumnChoice::Columns, ColumnChoice::Measured] {
                engine.set_column_choice(choice);
                let (got, got_vectors) = engine.trace_ids(&ids, ids.len(), keep).unwrap();
                let at = format!("{label}: {} with {choice:?}", backend.name());
                assert_eq!(got, reading, "{at}: reading");
                assert_eq!(got_vectors.trace.token_states, vectors.trace.token_states, "{at}: token states");
                for (a, b) in got_vectors.chunks.iter().zip(&vectors.chunks) {
                    assert_eq!((&a.pooled, &a.window_pooled), (&b.pooled, &b.window_pooled), "{at}: pooled vectors");
                }
            }
        }
        engine.set_backend(Backend::detect()).unwrap();
        engine.set_column_choice(ColumnChoice::Measured);
    }
    // The verdict alone, on chunk readings chosen to exercise it (a mixed document, one just short of it, ...).
    let table = ReleaseConfig::parse(&serde_json::to_vec(&config).unwrap()).unwrap().calibration.unwrap();
    for (index, case) in fixtures["combinations"].as_array().unwrap().iter().enumerate() {
        let chunks: Vec<ChunkCall> = serde_json::from_value(case["chunks"].clone()).unwrap();
        let got = serde_json::to_value(combine(&table, &chunks)).unwrap();
        same_json(&format!("{set} combination {index}"), &got, &case["verdict"], &mut worst, &mut failures);
    }
    eprintln!("{set}: largest deviations from reference.py:");
    for (kind, deviation) in &worst.entries {
        eprintln!("  {kind:<32} {deviation:.3e}");
    }
    if !flipped.is_empty() {
        eprintln!("{set}: short texts moved by a flipped activation, held to the scaled table: {}", flipped.join(", "));
        for (kind, deviation) in &worst_flipped.entries {
            eprintln!("  {kind:<32} {deviation:.3e}");
        }
    }
    // A flip moves about 1 short text in 15 (SPEC.md, Parity); most of them moved is an error in the engine.
    assert!(2 * flipped.len() <= short, "{set}: {} of {short} short texts read beyond the table: {}", flipped.len(), flipped.join(", "));
    assert!(failures.is_empty(), "{set}: {} values outside SPEC.md's tolerances:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn tiny_matches_reference() {
    run_set("tiny");
}

#[test]
fn s192_l4_matches_reference() {
    run_set("s192_l4");
}

/// The same random weights as `tiny`, released in SPEC.md's integer attention (format 2).
#[test]
fn tiny_integer_matches_reference() {
    run_set("tiny_integer");
}

#[test]
fn s192_l4_integer_matches_reference() {
    run_set("s192_l4_integer");
}

/// The same random weights as `tiny`, released with int8 output (format 3, float attention): int8 query and key codes
/// and an int8 bank.
#[test]
fn tiny_int8_matches_reference() {
    run_set("tiny_int8");
}

/// An int8 bank holds codes in [-127, 127] and is stored I8: a -128, a float16 bank under a format 3 config and an int8
/// bank under a format 1 config each fail to load.
#[test]
fn int8_banks_are_checked_at_load() {
    let folder = env_path("LITMUS_STUDENT_FIXTURES");
    let read = |set: &str, name: &str| std::fs::read(folder.join(set).join(name)).unwrap();
    let tokenizer = std::fs::read(env_path("LITMUS_TOKENIZER_DIR").join("tokenizer.json")).unwrap();
    let (model, config) = (read("tiny_int8", "model.safetensors"), read("tiny_int8", "config.json"));
    Engine::load(&model, &config, &tokenizer).unwrap();

    // The bank's first byte, from the safetensors header: 8 bytes of length, the JSON, then the tensors.
    let header_length = u64::from_le_bytes(model[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&model[8..8 + header_length]).unwrap();
    let begin = 8 + header_length + header["bank.vectors"]["data_offsets"][0].as_u64().unwrap() as usize;
    let mut corrupt = model.clone();
    corrupt[begin] = 0x80;
    let error = Engine::load(&corrupt, &config, &tokenizer).err().expect("-128 refused").to_string();
    assert!(error.contains("-128"), "{error}");

    let float_bank = read("tiny", "model.safetensors");
    let error = Engine::load(&float_bank, &config, &tokenizer).err().expect("a float16 bank refused").to_string();
    assert!(error.contains("bank.vectors") && error.contains("I8"), "{error}");
    let error = Engine::load(&model, &read("tiny", "config.json"), &tokenizer).err().expect("an int8 bank under format 1 refused").to_string();
    assert!(error.contains("bank.vectors") && error.contains("F16"), "{error}");
}

/// The same weights as `s192_l4_integer` with int8 output (format 3, integer attention).
#[test]
fn s192_l4_integer_int8_matches_reference() {
    run_set("s192_l4_integer_int8");
}
