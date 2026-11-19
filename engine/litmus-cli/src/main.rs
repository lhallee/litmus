//! `litmus`: score files, folders or standard input for model-written text, and measure the engine.
//!
//! ```text
//! litmus score [options] <file | folder | ->...
//! litmus bench [options]
//! litmus info  [options]
//! litmus mcp   [options]
//! ```

mod bench;
mod mcp;
mod memory;
mod strip;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use litmus_engine::{Backend, DocumentReading, Engine, Reading, STRIP_LEVEL};
use litmus_files::counts::TextCounts;
use litmus_files::export::{self, set_words, thousands};
use litmus_files::release::{self, ReleaseFiles};
use litmus_files::{cache, extract, walk};

#[global_allocator]
static ALLOCATOR: memory::Counting = memory::Counting;

const USAGE: &str = "\
Litmus: a detector of model-written text, on one CPU core.

USAGE:
  litmus score [options] <file | folder | ->...   Score files (txt, md, html, docx, pdf), folders (recursively)
                                                  or standard input (-)
  litmus bench [options]                          Tokens per second and peak memory on this machine
  litmus info  [options]                          The loaded model and the processor's backends
  litmus mcp   [options]                          Serve the Model Context Protocol on standard input and output, so an AI
                                                  agent can check a text or a file (tools litmus_check_text and
                                                  litmus_check_file); no network, no files written; takes only the
                                                  model options

SCORE OPTIONS:
  --json             One JSON object per file, every field of the reading and the text's counts
  --csv              One CSV row per file
  --map              Also print the heat map (one row per stride, 128 tokens) and, for long texts, each chunk
  --explain          Explain each reading: how chunks combine, bank evidence, conformal counts, calibration
  --no-cache         Score again even if the file is unchanged
  --no-color         Draw the strip in plain characters

MODEL OPTIONS (all commands):
  --model DIR        A release folder: model.safetensors, config.json and (optionally) tokenizer.json
                     [default: $LITMUS_MODEL, or the model built into this binary]
  --tokenizer FILE   ModernBERT's tokenizer.json [default: DIR/tokenizer.json, $LITMUS_TOKENIZER, or the
                     Hugging Face cache]
  --backend NAME     scalar, avx2, avx2-sign, neon, neon-dotprod or simd128 [default: the fastest available]

BENCH OPTIONS:
  --tokens LIST      Document lengths in tokens [default: 128,1024,8192,131072]
  --repeat N         Runs per length; the fastest counts [default: 3]
  --text FILE        Read the first tokens of this UTF-8 file instead of generated text (it must be long enough)
  --column-percent N The feed-forward's down product takes the column path for a token when at most N% of its input
                     pairs are nonzero (0: the dense path alone) [default: the measured limit]; same bits either way
  --json             One JSON object per measurement
  --print-text       Print the text of each length and measure nothing, to read the same document elsewhere

The verdict is the conformal set at level 0.10: one origin (human, AI, mixed), several (uncertain), or none
(unlike reference). P(not human) is the calibrated probability with its interval. The strip runs from human (blue)
through mixed (purple) to AI (red); the pale band is what the conformal set cannot rule out. A text of more than
8,192 tokens is read in equal chunks, and the verdict combines them.
";

/// Command-line options.
#[derive(Default)]
struct Options {
    command: String,
    inputs: Vec<String>,
    model: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    backend: Option<String>,
    json: bool,
    csv: bool,
    map: bool,
    explain: bool,
    no_cache: bool,
    no_color: bool,
    tokens: Vec<usize>,
    repeat: usize,
    text: Option<PathBuf>,
    column_percent: Option<usize>,
    print_text: bool,
}

fn parse(arguments: Vec<String>) -> Result<Options, String> {
    let mut options = Options { repeat: 3, tokens: vec![128, 1024, 8192, 131_072], ..Options::default() };
    let mut iter = arguments.into_iter();
    options.command = iter.next().ok_or("a command: score, bench or info")?;
    while let Some(argument) = iter.next() {
        let mut value = |name: &str| iter.next().ok_or_else(|| format!("{name} needs a value"));
        match argument.as_str() {
            "--json" => options.json = true,
            "--print-text" => options.print_text = true,
            "--csv" => options.csv = true,
            "--map" => options.map = true,
            "--explain" => options.explain = true,
            "--no-cache" => options.no_cache = true,
            "--no-color" => options.no_color = true,
            "--model" => options.model = Some(PathBuf::from(value("--model")?)),
            "--tokenizer" => options.tokenizer = Some(PathBuf::from(value("--tokenizer")?)),
            "--backend" => options.backend = Some(value("--backend")?),
            "--repeat" => options.repeat = value("--repeat")?.parse().map_err(|_| "--repeat takes a number")?,
            "--text" => options.text = Some(PathBuf::from(value("--text")?)),
            "--column-percent" => {
                options.column_percent = Some(value("--column-percent")?.parse().map_err(|_| "--column-percent takes a number")?)
            }
            "--tokens" => {
                options.tokens = value("--tokens")?
                    .split(',')
                    .map(|n| n.trim().replace('_', "").parse::<usize>().map_err(|_| format!("--tokens: {n} is not a number")))
                    .collect::<Result<_, _>>()?
            }
            "-h" | "--help" => return Err(String::new()),
            "-" => options.inputs.push(argument),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => options.inputs.push(argument),
        }
    }
    if options.json && options.csv {
        return Err("--json and --csv exclude each other".into());
    }
    Ok(options)
}

#[cfg(feature = "embedded")]
fn embedded() -> Option<ReleaseFiles> {
    Some(ReleaseFiles {
        model: include_bytes!(concat!(env!("LITMUS_EMBED_DIR"), "/model.safetensors")).to_vec(),
        config: include_bytes!(concat!(env!("LITMUS_EMBED_DIR"), "/config.json")).to_vec(),
        tokenizer: include_bytes!(concat!(env!("LITMUS_EMBED_DIR"), "/tokenizer.json")).to_vec(),
        source: "built in".into(),
    })
}

#[cfg(not(feature = "embedded"))]
fn embedded() -> Option<ReleaseFiles> {
    None
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn model_files(options: &Options) -> Result<ReleaseFiles, String> {
    let folder = options.model.clone().or_else(|| std::env::var_os("LITMUS_MODEL").map(PathBuf::from));
    let Some(folder) = folder else {
        return embedded().ok_or_else(|| "no model: pass --model DIR or set LITMUS_MODEL (this binary has none built in)".to_owned());
    };
    release::read_folder(&folder, options.tokenizer.as_deref())
}

fn load_engine(options: &Options, files: &ReleaseFiles) -> Result<Engine, String> {
    let mut engine = Engine::load(&files.model, &files.config, &files.tokenizer).map_err(|e| e.to_string())?;
    if let Some(name) = &options.backend {
        let backend = Backend::from_name(name).ok_or_else(|| format!("unknown backend {name}"))?;
        engine.set_backend(backend).map_err(|e| e.to_string())?;
    }
    Ok(engine)
}

struct Row {
    name: String,
    hash: String,
    cached: bool,
    outcome: Result<DocumentReading, String>,
    /// The words, characters and tokens of the text that was read (none for a file that did not read).
    counts: Option<TextCounts>,
}

/// What the table's verdict line adds in words: out of reference, the chunks' spread, a mixed document, and where
/// the heat map runs hottest and coolest.
fn notes(document: &DocumentReading) -> String {
    let verdict = &document.verdict;
    let mut notes = Vec::new();
    if verdict.out_of_reference {
        notes.push("out of reference".to_owned());
    }
    if verdict.chunks > 1 {
        notes.push(format!("{} chunks, P(not human) {:.2} to {:.2}", verdict.chunks, verdict.lowest, verdict.highest));
    }
    if verdict.mixed_document {
        notes.push("mixed document: a chunk reads as model-written and another as human".to_owned());
    }
    if document.cells.len() > 1 {
        let (low, high) = export::map_range(document);
        notes.push(format!("heat map {low:.2} to {high:.2} over {} cells", document.cells.len()));
    }
    notes.join("; ")
}

fn print_table(rows: &[Row], options: &Options) {
    let colored = !options.no_color && strip::color_supported();
    println!("{:<17} {:<20} {:<18} {:<strip_width$} {:>8}  FILE", "VERDICT", "P(NOT HUMAN)", "SET AT 0.10", "STRIP", "TOKENS", strip_width = strip::WIDTH + 2);
    for row in rows {
        match &row.outcome {
            Err(error) => println!("{:<17} {error}  {}", "error", row.name),
            Ok(document) => {
                let verdict = &document.verdict;
                let mut line = format!(
                    "{:<17} {:<20} {:<18} {} {:>8}  {}",
                    document.label().word(),
                    format!("{:.2} [{:.2}, {:.2}]", verdict.probability, verdict.interval[0], verdict.interval[1]),
                    set_words(verdict.set(STRIP_LEVEL)),
                    strip::render(&verdict.strip, colored),
                    thousands(document.n_tokens),
                    row.name,
                );
                let notes = notes(document);
                if !notes.is_empty() {
                    line.push_str(&format!("  ({notes})"));
                }
                println!("{line}");
                if let Some(counts) = &row.counts {
                    println!("  {}", counts.line());
                }
                if options.map {
                    print_map(document, colored);
                }
                if options.explain {
                    print!("{}", explain_document(document));
                }
            }
        }
    }
}

/// `--map`: each chunk's reading when there are several, then the heat map, one row per cell.
fn print_map(document: &DocumentReading, colored: bool) {
    if document.chunks.len() > 1 {
        for (index, chunk) in document.chunks.iter().enumerate() {
            let r = &chunk.reading;
            println!(
                "  {:<15} {:<20} {:<18} {} {:>8}  chunk {} (characters {} to {})",
                r.label(STRIP_LEVEL).word(),
                format!("{:.2} [{:.2}, {:.2}]", r.probability, r.interval[0], r.interval[1]),
                set_words(r.set(STRIP_LEVEL)),
                strip::render(&r.strip, colored),
                thousands(r.n_tokens),
                index + 1,
                thousands(chunk.place.start_char),
                thousands(chunk.place.end_char),
            );
        }
    }
    println!("  Heat map: one cell per {} tokens, each the overlap-weighted mean of the windows over it", document.cells.first().map_or(0, |c| c.place.end - c.place.start));
    for cell in &document.cells {
        let band = litmus_engine::Strip { position: cell.position, band_lo: cell.band_lo, band_hi: cell.band_hi, level: STRIP_LEVEL, unlike_reference: cell.unlike_reference };
        println!(
            "    {} {:.2} (spread {:.2}, {} windows)  characters {} to {}{}",
            strip::render(&band, colored),
            cell.position,
            cell.spread,
            cell.windows,
            thousands(cell.place.start_char),
            thousands(cell.place.end_char),
            if cell.unlike_reference { "  unlike reference" } else { "" },
        );
    }
}

/// `--explain`: how the chunks combine into the verdict, then each chunk's own account.
fn explain_document(document: &DocumentReading) -> String {
    let verdict = &document.verdict;
    let mut out = String::new();
    if verdict.chunks > 1 {
        out.push_str(&format!(
            "  How the verdict was made: {} tokens read as {} chunks of about {} (a chunk is at most {} tokens, each read as a text of its own).\n",
            thousands(document.n_tokens),
            verdict.chunks,
            thousands(document.n_tokens / verdict.chunks),
            thousands(document.chunk_tokens)
        ));
        out.push_str(&format!(
            "    Merged p-values (twice the chunks' mean, valid however the chunks depend on each other): human {:.3}, AI {:.3}, mixed {:.3}\n",
            verdict.p_values[0], verdict.p_values[1], verdict.p_values[2]
        ));
        out.push_str(&format!(
            "    P(not human) {:.3}, the chunks' mean; interval [{:.3}, {:.3}], their mean interval widened by their disagreement {:.3}\n",
            verdict.probability, verdict.interval[0], verdict.interval[1], verdict.disagreement
        ));
        out.push_str(&format!(
            "    Mixed-document test at {:.4} per chunk ({} over {} chunks): {} chunk(s) read as model-written, {} as human; {}\n",
            verdict.chunk_level,
            verdict.level,
            verdict.chunks,
            verdict.model_chunks,
            verdict.human_chunks,
            if verdict.mixed_document { "a mixed document" } else { "no sign of a mixed document" }
        ));
    }
    for (index, chunk) in document.chunks.iter().enumerate() {
        if document.chunks.len() > 1 {
            out.push_str(&format!("  Chunk {} (characters {} to {}):\n", index + 1, thousands(chunk.place.start_char), thousands(chunk.place.end_char)));
        }
        out.push_str(&explain_text(&chunk.reading));
    }
    out
}

/// The plain-words account of one reading, from the reading's own numbers.
fn explain_text(reading: &Reading) -> String {
    let names = ["human", "AI", "mixed"];
    let likeness = ["human", "model-like (AI)", "mixed"];
    let mut out = String::new();
    out.push_str(&format!("  How this reading was made ({} tokens, length bin {}):\n", thousands(reading.n_tokens), reading.length_bin));
    out.push_str("    Against the reference bank, the mean probability that this text shares an origin with a bank text:\n");
    for o in 0..3 {
        out.push_str(&format!(
            "      {:<5} {:.3}, {:.0}% of the total (spread {:.3}, percentile {:.0} among calibration texts)\n",
            names[o],
            reading.mean[o],
            100.0 * reading.share[o],
            reading.spread[o],
            100.0 * reading.spread_percentile[o]
        ));
    }
    let Some(explain) = &reading.explain else { return out };
    out.push_str("    Conformal p-values, from the calibration texts of the same origin and length:\n");
    for count in &explain.conformal {
        let o = count.origin.index();
        out.push_str(&format!(
            "      {:<5} of {} {} calibration texts, {} read at most this {}: p = {:.3} (reported on the grid as {})\n",
            names[o],
            thousands(count.n),
            names[o],
            thousands(count.count),
            likeness[o],
            count.p,
            reading.p_values[o]
        ));
    }
    for (key, set) in &reading.sets {
        let set: Vec<&str> = set.iter().map(|o| names[o.index()]).collect();
        out.push_str(&format!(
            "    At level {key}: not ruled out {}; {}\n",
            if set.is_empty() { "none (unlike the reference)".to_owned() } else { set.join(", ") },
            if reading.not_human[key] { "called not human" } else { "not called not human" }
        ));
    }
    out.push_str(&format!(
        "    P(not human) {:.3}, interval [{:.3}, {:.3}] (Venn-Abers on the score {:.3}); three-way: human {:.2}, AI {:.2}, mixed {:.2}\n",
        reading.probability, reading.interval[0], reading.interval[1], reading.score, reading.three_way[0], reading.three_way[1], reading.three_way[2]
    ));
    let closeness = &explain.closeness;
    out.push_str(&format!(
        "    Closeness to the bank {:.3}: {} of {} calibration texts of this length sat as far from the reference (p = {:.3}){}\n",
        reading.closeness,
        thousands(closeness.count),
        thousands(closeness.n),
        closeness.p,
        if reading.out_of_reference { "; out of reference" } else { "" }
    ));
    out.push_str("    Nearest bank texts:\n");
    for entry in &explain.nearest {
        let metadata: Vec<String> = entry.metadata.iter().filter(|(_, value)| !value.is_empty()).map(|(field, value)| format!("{field} {value}")).collect();
        out.push_str(&format!(
            "      #{:<6} {:<5} cosine {:.3}, P(same origin) {:.2}, {} tokens{}{}\n",
            entry.index,
            names[entry.origin.index()],
            entry.cosine,
            entry.probability,
            thousands(entry.tokens.max(0) as usize),
            if metadata.is_empty() { "" } else { "; " },
            metadata.join(", ")
        ));
    }
    out
}

fn export_rows(rows: &[Row]) -> Vec<export::Row<'_>> {
    rows.iter()
        .map(|row| export::Row { name: &row.name, hash: &row.hash, cached: row.cached, outcome: row.outcome.as_ref().map_err(String::as_str), counts: row.counts.as_ref() })
        .collect()
}

fn score(options: &Options) -> Result<bool, String> {
    if options.inputs.is_empty() {
        return Err("score needs files, folders or - for standard input".into());
    }
    let names = walk::collect(&options.inputs)?;
    let files = model_files(options)?;
    let engine = load_engine(options, &files)?;
    let cache = (!options.no_cache).then(cache::folder).flatten().map(|folder| cache::Cache::open(&folder, &[&files.model, &files.config, &files.tokenizer]));
    let mut rows = Vec::new();
    let mut all_ok = true;
    for name in names {
        let bytes = if name == "-" { extract::stdin_text() } else { read(Path::new(&name)) };
        let bytes = match bytes {
            Ok(bytes) => bytes,
            Err(error) => {
                all_ok = false;
                rows.push(Row { name, hash: String::new(), cached: false, outcome: Err(error), counts: None });
                continue;
            }
        };
        let hash = cache::sha256(&bytes);
        // The text is extracted even when the reading is cached: the counts are of the text, which the cache does not hold.
        let path = if name == "-" { PathBuf::from("stdin.txt") } else { PathBuf::from(&name) };
        let text = match extract::text_of(&path, &bytes) {
            Ok(text) => text,
            Err(error) => {
                all_ok = false;
                rows.push(Row { name, hash, cached: false, outcome: Err(error), counts: None });
                continue;
            }
        };
        if let Some(reading) = cache.as_ref().and_then(|cache| cache.get(&hash)) {
            let counts = Some(TextCounts::of(&text, reading.n_tokens));
            rows.push(Row { name, hash, cached: true, outcome: Ok(reading), counts });
            continue;
        }
        let outcome = engine.score(&text).map_err(|e| e.to_string());
        if let (Some(cache), Ok(reading)) = (&cache, &outcome) {
            cache.put(&hash, reading);
        }
        all_ok &= outcome.is_ok();
        let counts = outcome.as_ref().ok().map(|reading| TextCounts::of(&text, reading.n_tokens));
        rows.push(Row { name, hash, cached: false, outcome, counts });
    }
    if options.json {
        print!("{}", export::json_lines(&export_rows(&rows)));
    } else if options.csv {
        print!("{}", export::csv(&export_rows(&rows)));
    } else {
        print_table(&rows, options);
    }
    Ok(all_ok)
}

fn info(options: &Options) -> Result<bool, String> {
    let available: Vec<&str> = Backend::available().iter().map(|b| b.name()).collect();
    println!("backends on this processor: {} (fastest {})", available.join(", "), Backend::detect().name());
    let files = model_files(options)?;
    let engine = load_engine(options, &files)?;
    println!("model: {}", files.source);
    println!("{}", serde_json::to_string_pretty(engine.info()).unwrap_or_default());
    Ok(true)
}

/// `litmus mcp`: load the release the other commands load and serve on standard input and output until it closes. A
/// release that fails to load does not stop the server: each check then answers with the reason, which an agent can read.
fn serve_mcp(options: &Options) -> Result<bool, String> {
    if !options.inputs.is_empty() {
        return Err("mcp takes no files; the agent sends texts and paths through the protocol".into());
    }
    if options.json || options.csv || options.map || options.explain || options.no_cache || options.no_color {
        return Err("mcp takes only --model, --tokenizer and --backend".into());
    }
    let engine = model_files(options).and_then(|files| load_engine(options, &files).map(|engine| (engine, files.source)));
    let (scorer, source): (Box<dyn mcp::Scorer>, String) = match engine {
        Ok((engine, source)) => (Box::new(mcp::Loaded(engine)), source),
        Err(error) => {
            eprintln!("litmus mcp: {error}");
            (Box::new(mcp::Missing(error)), "none".to_owned())
        }
    };
    eprintln!("litmus mcp: serving, model: {source}");
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    mcp::serve(scorer.as_ref(), &mut stdin.lock(), &mut stdout.lock()).map_err(|e| format!("mcp: {e}"))?;
    Ok(true)
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let options = match parse(arguments) {
        Ok(options) => options,
        Err(error) => {
            if !error.is_empty() {
                eprintln!("litmus: {error}\n");
            }
            eprint!("{USAGE}");
            return if error.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(2) };
        }
    };
    let outcome = match options.command.as_str() {
        "score" => score(&options),
        "bench" => model_files(&options).and_then(|files| bench::run(&options_bench(&options), &files.model, &files.config, &files.tokenizer, options.backend.as_deref())),
        "info" => info(&options),
        "mcp" => serve_mcp(&options),
        extract::PDF_CHILD_ARGUMENT => match options.inputs.first() {
            Some(path) => extract::pdf_child(Path::new(path)),
            None => Err("__pdf-text takes a file".into()),
        },
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(true)
        }
        other => Err(format!("unknown command {other}")),
    };
    match outcome {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("litmus: {error}");
            ExitCode::from(2)
        }
    }
}

fn options_bench(options: &Options) -> bench::BenchOptions {
    bench::BenchOptions { tokens: options.tokens.clone(), repeat: options.repeat.max(1), json: options.json, text: options.text.clone(), column_percent: options.column_percent, print_text: options.print_text }
}
