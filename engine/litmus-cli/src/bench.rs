//! `litmus bench`: tokens per second and memory for documents of given lengths, on the calling thread only unless
//! `--threads` asks for more. Pin the process to one core from outside (`start /affinity 4` on Windows, `taskset -c 2`
//! on Linux) for single-core numbers.

use std::time::Instant;

use litmus_engine::model::ColumnChoice;
use litmus_engine::{Backend, Engine, Flow};

use crate::memory;

pub struct BenchOptions {
    pub tokens: Vec<usize>,
    pub repeat: usize,
    /// Threads a scoring call spreads a chunk's blocks over (`Engine::score_threads`); 1 is the calling thread alone.
    pub threads: usize,
    pub json: bool,
    /// A UTF-8 file whose first tokens are read instead of the generated text.
    pub text: Option<std::path::PathBuf>,
    /// The column path's limit in percent of nonzero input pairs (0: the dense path alone); else the measured one.
    pub column_percent: Option<usize>,
    /// Print each length's text to standard output and measure nothing, so the same document can be read elsewhere
    /// (the browser page's speed is measured on it).
    pub print_text: bool,
}

const WORDS: &[&str] = &[
    "the", "river", "model", "carried", "a", "quiet", "ledger", "of", "signals", "through", "careful", "measure", "and",
    "several", "readers", "found", "that", "during", "long", "winters", "however", "protein", "windows", "remained",
    "open", "because", "although", "written", "by", "hand", "each", "section", "bank", "origin", "human", "observed",
    "results", "in", "their", "own", "words", "which", "is", "not", "what", "we", "expected", "from", "it",
];

/// A deterministic English-like text of at least `tokens` tokens, cut at a token boundary.
fn text_of_tokens(engine: &Engine, tokens: usize) -> (String, usize) {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut text = String::new();
    let mut sentences = 0;
    // About 1.3 tokens a word; overshoot, then cut.
    while text.len() < tokens * 6 + 64 {
        let length = 6 + (next() % 14) as usize;
        for i in 0..length {
            let word = WORDS[(next() % WORDS.len() as u64) as usize];
            if i == 0 {
                let mut chars = word.chars();
                text.extend(chars.next().map(|c| c.to_ascii_uppercase()));
                text.push_str(chars.as_str());
            } else {
                text.push(' ');
                text.push_str(word);
            }
            if i + 1 < length && next() % 9 == 0 {
                text.push(',');
            }
        }
        text.push('.');
        sentences += 1;
        text.push_str(if sentences % 7 == 0 { "\n\n" } else { " " });
    }
    let all = engine.tokenize(&text);
    assert!(all.len() > tokens, "the generated text is long enough");
    let cut = all[tokens].start;
    let text = text[..cut].trim_end().to_owned();
    let count = engine.tokenize(&text).len();
    (text, count)
}

/// The first `tokens` tokens of `source`, cut at a token boundary; an error when it holds no more than that.
fn text_from_file(engine: &Engine, source: &str, tokens: usize) -> Result<(String, usize), String> {
    let all = engine.tokenize(source);
    if all.len() <= tokens {
        return Err(format!("--text holds {} tokens, not more than the {tokens} asked for", all.len()));
    }
    let text = source[..all[tokens].start].trim_end().to_owned();
    let count = engine.tokenize(&text).len();
    Ok((text, count))
}

/// One more scoring of `ids`, with the share of its time each encoder stage took; the head and the bank are the
/// read pass, and the rest is what no stage holds (rows copied between layers, the readings, the clock itself).
#[cfg(feature = "profile")]
fn stage_shares(engine: &Engine, ids: &[u32], read_pass: f64) -> Result<(), String> {
    use litmus_engine::profile::{STAGES, take};

    take();
    litmus_engine::profile::take_down(engine.info().depth);
    let started = Instant::now();
    std::hint::black_box(engine.score_ids(ids, 4096).map_err(|e| e.to_string())?);
    let total = started.elapsed().as_secs_f64();
    let nanos = take();
    let staged: f64 = nanos.iter().map(|&n| n as f64 * 1e-9).sum();
    let micros = 1e6 / ids.len() as f64;
    let mut line = format!("{:>8} stages, microseconds a token ({:.1} in all):", "", total * micros);
    for ((_, name), &n) in STAGES.iter().zip(&nanos) {
        let seconds = n as f64 * 1e-9;
        line.push_str(&format!(" {name} {:.2} ({:.0}%)", seconds * micros, 100.0 * seconds / total));
    }
    let rest = (total - staged - read_pass).max(0.0);
    line.push_str(&format!(" head+bank {:.2} ({:.0}%) rest {:.2} ({:.0}%)", read_pass * micros, 100.0 * read_pass / total, rest * micros, 100.0 * rest / total));
    println!("{line}");
    let down = litmus_engine::profile::take_down(engine.info().depth);
    let mut line = format!("{:>8} down inputs zero, and tokens on the column path, by layer:", "");
    for (layer, [zeros, values, tokens, column]) in down.iter().enumerate() {
        line.push_str(&format!(" L{layer} {:.1}% {:.1}%", 100.0 * *zeros as f64 / *values as f64, 100.0 * *column as f64 / *tokens as f64));
    }
    println!("{line}");
    Ok(())
}

pub fn run(options: &BenchOptions, model: &[u8], config: &[u8], tokenizer: &[u8], backend: Option<&str>) -> Result<bool, String> {
    let before_load = memory::heap_now();
    let started = Instant::now();
    let mut engine = Engine::load(model, config, tokenizer).map_err(|e| e.to_string())?;
    let load_seconds = started.elapsed().as_secs_f64();
    let engine_heap = memory::heap_now() - before_load;
    if let Some(name) = backend {
        engine.set_backend(Backend::from_name(name).ok_or_else(|| format!("unknown backend {name}"))?).map_err(|e| e.to_string())?;
    }
    match options.column_percent {
        Some(0) => engine.set_column_choice(ColumnChoice::Dense),
        Some(percent) => engine.set_column_choice(ColumnChoice::Percent(percent)),
        None => {}
    }
    let info = engine.info().clone();
    if options.print_text {
        let source = match &options.text {
            Some(path) => Some(std::fs::read_to_string(path).map_err(|error| format!("--text {}: {error}", path.display()))?),
            None => None,
        };
        for &tokens in &options.tokens {
            let (text, _) = match &source {
                Some(source) => text_from_file(&engine, source, tokens)?,
                None => text_of_tokens(&engine, tokens),
            };
            print!("{text}"); // no line break added: it would be part of the document
        }
        return Ok(true);
    }
    let threads = options.threads.max(1);
    if !options.json {
        println!(
            "model d={} L={} f={} W={} bank={:?}; backend {}; {}; load {:.0} ms; engine heap {:.1} MB (weights, head, bank {:.1} MB; tokenizer the rest)",
            info.width,
            info.depth,
            info.ffn_width,
            info.block_size,
            info.bank,
            engine.backend().name(),
            if threads == 1 { "one thread".to_owned() } else { format!("{threads} threads") },
            load_seconds * 1e3,
            engine_heap as f64 / 1e6,
            info.bytes as f64 / 1e6
        );
        println!(
            "{:>8} {:>7} {:>8} {:>12} {:>12} {:>14} {:>16} {:>10} {:>12}",
            "TOKENS", "CHUNKS", "WINDOWS", "TEXT TOK/S", "IDS TOK/S", "ADDS/S (IDS)", "SCORING HEAP MB", "SECONDS", "READ PASS"
        );
    }
    let source = match &options.text {
        Some(path) => Some(std::fs::read_to_string(path).map_err(|error| format!("--text {}: {error}", path.display()))?),
        None => None,
    };
    for &tokens in &options.tokens {
        let (text, count) = match &source {
            Some(source) => text_from_file(&engine, source, tokens)?,
            None => text_of_tokens(&engine, tokens),
        };
        let ids: Vec<u32> = engine.tokenize(&text).iter().map(|t| t.id).collect();
        let mut best_text = f64::INFINITY;
        let mut best_ids = f64::INFINITY;
        let mut heap = 0usize;
        let mut windows = 0usize;
        let mut chunk_lengths: Vec<usize> = Vec::new();
        for _ in 0..options.repeat {
            memory::reset_peak();
            let base = memory::heap_now();
            let started = Instant::now();
            let reading = engine.score_threads(&text, threads, &mut |_| Flow::Continue).map_err(|e| e.to_string())?;
            best_text = best_text.min(started.elapsed().as_secs_f64());
            heap = heap.max(memory::heap_peak() - base);
            assert_eq!(reading.n_tokens, count, "the reading counts the text's tokens");
            windows = reading.windows.len();
            chunk_lengths = reading.chunks.iter().map(|chunk| chunk.reading.n_tokens).collect();
            drop(reading);
            let started = Instant::now();
            let reading = if threads == 1 { engine.score_ids(&ids, 4096) } else { engine.score_ids_threads(&ids, threads) }.map_err(|e| e.to_string())?;
            best_ids = best_ids.min(started.elapsed().as_secs_f64());
            drop(reading);
        }
        // The reading pass (head and bank for each chunk and its windows) timed alone, on synthetic states.
        let mut read_pass = 0.0;
        for &length in &chunk_lengths {
            let mut best = f64::INFINITY;
            for _ in 0..options.repeat {
                let started = Instant::now();
                std::hint::black_box(engine.read_pass(length));
                best = best.min(started.elapsed().as_secs_f64());
            }
            read_pass += best;
        }
        let text_rate = count as f64 / best_text;
        let ids_rate = count as f64 / best_ids;
        let adds = info.adds_per_token as f64 * ids_rate;
        if options.json {
            println!(
                "{}",
                serde_json::json!({"width": info.width, "depth": info.depth, "backend": engine.backend().name(), "threads": threads, "tokens": count,
                    "text_tokens_per_second": text_rate, "ids_tokens_per_second": ids_rate, "ternary_adds_per_second": adds,
                    "scoring_heap_bytes": heap, "engine_heap_bytes": engine_heap, "seconds": best_text,
                    "chunks": chunk_lengths.len(), "windows": windows, "read_pass_seconds": read_pass, "bank": info.bank,
                    "process_peak_bytes": memory::process_peak()})
            );
        } else {
            println!(
                "{count:>8} {:>7} {windows:>8} {text_rate:>12.0} {ids_rate:>12.0} {adds:>14.3e} {:>16.2} {best_text:>10.4} {:>7.1} ms {:>3.0}%",
                chunk_lengths.len(),
                heap as f64 / 1e6,
                read_pass * 1e3,
                100.0 * read_pass / best_text
            );
        }
        #[cfg(feature = "profile")]
        if !options.json {
            stage_shares(&engine, &ids, read_pass)?;
        }
    }
    if !options.json {
        match memory::process_peak() {
            Some(peak) => println!("process peak resident memory: {:.1} MB", peak as f64 / 1e6),
            None => println!("process peak resident memory: not reported on this system"),
        }
    }
    Ok(true)
}
