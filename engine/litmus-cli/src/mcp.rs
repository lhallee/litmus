//! `litmus mcp`: the Model Context Protocol over standard input and output, so an AI agent can read a text or a file
//! with the program it already installed. It makes no network call, writes no file, and keeps no state between
//! requests; the engine is the one `litmus score` uses and the reading is the one `litmus score --json` prints.
//!
//! The protocol follows the 2026-07-28 revision, which has no handshake: every request carries its protocol version
//! and capabilities in `_meta`, and `server/discover` stands in for `initialize`. Clients that still open with
//! `initialize` (2025-11-25 and 2025-06-18) are served too, since most of them do today. Messages are one JSON object
//! a line; nothing but messages goes to standard output; the server leaves when standard input closes.
//! Sources, with the date read, are in `docs/USE_WITH_AGENTS.md`.

use std::io::{BufRead, Read, Write};
use std::path::Path;

use litmus_engine::{DocumentReading, Engine, STRIP_LEVEL};
use litmus_files::counts::TextCounts;
use litmus_files::export::{set_words, thousands};
use litmus_files::extract;
use serde_json::{Value, json};

/// The revision with per-request metadata, and the two earlier ones answered after an `initialize`.
const MODERN: &str = "2026-07-28";
const LEGACY: [&str; 2] = ["2025-11-25", "2025-06-18"];
const VERSION_KEY: &str = "io.modelcontextprotocol/protocolVersion";
const CAPABILITIES_KEY: &str = "io.modelcontextprotocol/clientCapabilities";
const SERVER_INFO_KEY: &str = "io.modelcontextprotocol/serverInfo";

/// The longest text read, in tokens (the largest length the engine is measured at: 16 chunks), and the largest input in
/// bytes tried before tokens are counted. `litmus score` has no cap; an agent can hand a server anything.
pub const MAX_TOKENS: usize = 131_072;
const MAX_TEXT_BYTES: usize = 8 * 1024 * 1024;
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;
/// A message line longer than this is refused unread (a text of `MAX_TEXT_BYTES` escaped as JSON fits).
const MAX_LINE_BYTES: usize = 48 * 1024 * 1024;
/// Under this many tokens the reading carries a caveat; it marks where one sentence or two ends, not a measured limit.
const SHORT_TEXT_TOKENS: usize = 128;

const INSTRUCTIONS: &str = "Litmus is a free detector of AI written text that says how sure it is. Call \
litmus_check_text with a text, or litmus_check_file with a path, to read whether it looks human, AI, or mixed. A reading \
is evidence, never proof: it comes with the origins it cannot rule out and an interval, and an \"uncertain\" or \"unlike \
reference\" reading means no call. Short texts are less reliable, 98.8% of the model's training text was \
English, so other languages are not supported yet, code is weaker than prose, and we recommend never presenting a \
reading alone as an accusation against a person.";

const GUIDANCE: &str = "A reading is evidence about where a text came from, never proof of who wrote it. It is \
probabilistic, with a set of origins it cannot rule out. It is less reliable for short texts and for code, and it is \
trained almost entirely on English, so other languages are not supported yet. We recommend never presenting \
it alone as an accusation against a person.";

const TEXT_DESCRIPTION: &str = "Read whether a text looks human, AI, or mixed, with calibrated uncertainty. Returns the \
verdict (human, AI, mixed, uncertain, or unlike reference), the origins that cannot be ruled out at the 10% level, \
P(not human) with an interval, and the word, character, and token counts. A reading is evidence, never proof, and never an accusation against \
a person: short texts are less reliable, the model is trained almost entirely on English (other languages are not \
supported yet), and code is weaker than prose. An \"uncertain\" or \"unlike reference\" verdict means no call. Runs \
on this computer with no network. detail \"full\" adds every chunk, window, and heat-map cell, exactly as `litmus score \
--json` prints them, which is large for long texts.";

const FILE_DESCRIPTION: &str = "Read whether a file looks human, AI, or mixed, with calibrated uncertainty. Reads one \
file by path (txt, md, html, docx, or pdf as documents; any other file as plain text, so source code works) and returns \
what litmus_check_text returns. Use an absolute path. A reading is evidence, never proof, and never an accusation \
against a person: short texts are less reliable, the model is trained almost entirely on English (other languages are \
not supported yet), and code is weaker than prose. Reads only the one file, writes nothing, uses no network.";

/// What the server needs from the engine, so the protocol can be tested without a model.
pub trait Scorer {
    fn count_tokens(&self, text: &str) -> usize;
    fn score(&self, text: &str) -> Result<DocumentReading, String>;
}

/// The loaded engine.
pub struct Loaded(pub Engine);

impl Scorer for Loaded {
    fn count_tokens(&self, text: &str) -> usize {
        self.0.count_tokens(text)
    }

    fn score(&self, text: &str) -> Result<DocumentReading, String> {
        self.0.score(text).map_err(|e| e.to_string())
    }
}

/// A server whose model failed to load: it still answers, and every check says why it cannot read.
pub struct Missing(pub String);

impl Scorer for Missing {
    fn count_tokens(&self, _text: &str) -> usize {
        0
    }

    fn score(&self, _text: &str) -> Result<DocumentReading, String> {
        Err(format!("no model is loaded: {}", self.0))
    }
}

fn server_info() -> Value {
    json!({"name": "litmus", "title": "Litmus", "version": env!("CARGO_PKG_VERSION")})
}

fn error(id: Option<&Value>, code: i64, message: &str, data: Option<Value>) -> Value {
    let mut error = json!({"code": code, "message": message});
    if let Some(data) = data {
        error["data"] = data;
    }
    match id {
        Some(id) => json!({"jsonrpc": "2.0", "id": id, "error": error}),
        None => json!({"jsonrpc": "2.0", "error": error}),
    }
}

fn tools() -> Value {
    let annotations = json!({"readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false});
    let detail = json!({
        "type": "string",
        "enum": ["summary", "full"],
        "default": "summary",
        "description": "summary: the document verdict only. full: also every chunk, window, and heat-map cell, as `litmus score --json` prints them"
    });
    json!([
        {
            "name": "litmus_check_text",
            "title": "Check a text for AI-written content",
            "description": TEXT_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text or code to read. At most 131,072 tokens."},
                    "detail": detail
                },
                "required": ["text"],
                "additionalProperties": false
            },
            "annotations": annotations
        },
        {
            "name": "litmus_check_file",
            "title": "Check a file for AI-written content",
            "description": FILE_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path of one file. A folder is not read; call once per file."},
                    "detail": detail
                },
                "required": ["path"],
                "additionalProperties": false
            },
            "annotations": annotations
        }
    ])
}

/// The caveats that follow from this reading, in plain words.
fn caveats(document: &DocumentReading) -> Vec<String> {
    let verdict = &document.verdict;
    let mut caveats = Vec::new();
    if document.n_tokens < SHORT_TEXT_TOKENS {
        caveats.push(format!(
            "Short text ({} tokens): a sentence or two carries little evidence, so expect wide sets and intervals.",
            document.n_tokens
        ));
    }
    if verdict.out_of_reference || verdict.set(STRIP_LEVEL).is_none_or(|set| set.is_empty()) {
        caveats.push("The text is unlike the reference texts the model learned from, so the reading is no evidence either way.".to_owned());
    } else if verdict.set(STRIP_LEVEL).is_some_and(|set| set.len() > 1) {
        caveats.push("More than one origin is not ruled out, so the reading makes no call.".to_owned());
    }
    if verdict.mixed_document {
        caveats.push("Different chunks of this document read differently; ask for detail \"full\" to see where.".to_owned());
    }
    caveats
}

/// The one plain sentence that leads a result.
fn sentence(document: &DocumentReading, counts: &TextCounts) -> String {
    let verdict = &document.verdict;
    format!(
        "Litmus reads this text as {} (not ruled out at the 10% level: {}); P(not human) {:.2}, interval {:.2} to {:.2}, {}.",
        document.label().word(),
        set_words(verdict.set(STRIP_LEVEL)),
        verdict.probability,
        verdict.interval[0],
        verdict.interval[1],
        counts.line()
    )
}

fn failure(message: impl Into<String>) -> Value {
    json!({"content": [{"type": "text", "text": message.into()}], "isError": true})
}

/// A successful reading as a tool result: the sentence, then the structured reading. `counts` are the words,
/// characters and tokens of the text that was read.
fn success(document: &DocumentReading, counts: &TextCounts, full: bool, file: Option<&str>) -> Result<Value, String> {
    let reading = if full {
        serde_json::to_value(document).map_err(|e| e.to_string())?
    } else {
        json!({"n_tokens": document.n_tokens, "chunk_tokens": document.chunk_tokens, "verdict": document.verdict})
    };
    let text = sentence(document, counts);
    let mut structured = json!({
        "label": document.label().word(),
        "n_tokens": document.n_tokens,
        "words": counts.words,
        "characters": counts.characters,
        "caveats": caveats(document),
        "guidance": GUIDANCE,
        "detail": if full { "full" } else { "summary" },
        "reading": reading
    });
    if let Some(file) = file {
        structured["file"] = json!(file);
    }
    Ok(json!({"content": [{"type": "text", "text": text}], "structuredContent": structured, "isError": false}))
}

/// Read one text: refuse an empty or oversized one, else score it.
fn check_text(scorer: &dyn Scorer, text: &str, full: bool, file: Option<&str>) -> Value {
    if text.len() > MAX_TEXT_BYTES {
        return failure(format!("The text is {} bytes; at most {MAX_TEXT_BYTES} are read. Split it and check the parts.", text.len()));
    }
    let tokens = scorer.count_tokens(text);
    if tokens > MAX_TOKENS {
        return failure(format!("The text has {} tokens; at most {} are read. Split it and check the parts.", thousands(tokens), thousands(MAX_TOKENS)));
    }
    match scorer.score(text).and_then(|document| success(&document, &TextCounts::of(text, document.n_tokens), full, file)) {
        Ok(result) => result,
        Err(message) => failure(message),
    }
}

/// Read one file the way `litmus score` does: documents by their kind, anything else as plain text.
fn check_file(scorer: &dyn Scorer, path: &str, full: bool) -> Value {
    let location = Path::new(path);
    let metadata = match std::fs::metadata(location) {
        Ok(metadata) => metadata,
        Err(e) => return failure(format!("{path}: {e}")),
    };
    if metadata.is_dir() {
        return failure(format!("{path} is a folder; call litmus_check_file once per file."));
    }
    if metadata.len() > MAX_FILE_BYTES {
        return failure(format!("{path} is {} bytes; at most {MAX_FILE_BYTES} are read.", metadata.len()));
    }
    let bytes = match std::fs::read(location) {
        Ok(bytes) => bytes,
        Err(e) => return failure(format!("{path}: {e}")),
    };
    match extract::text_of(location, &bytes) {
        Ok(text) => check_text(scorer, &text, full, Some(path)),
        Err(message) => failure(format!("{path}: {message}")),
    }
}

/// The state of one process: which era the client opened in.
pub struct Server<'a> {
    scorer: &'a dyn Scorer,
    legacy: bool,
}

impl<'a> Server<'a> {
    pub fn new(scorer: &'a dyn Scorer) -> Self {
        Server { scorer, legacy: false }
    }

    /// The reply to one message, or none for a notification.
    pub fn handle(&mut self, message: &Value) -> Option<Value> {
        let Some(object) = message.as_object() else {
            return Some(error(None, -32600, "a message is one JSON object (batches are not supported)", None));
        };
        let id = object.get("id").filter(|id| id.is_string() || id.is_i64() || id.is_u64());
        let Some(method) = object.get("method").and_then(Value::as_str) else {
            // A response from the client is ignored, since this server never sends a request; anything else is nonsense.
            if object.contains_key("result") || object.contains_key("error") {
                return None;
            }
            return Some(error(id, -32600, "a message needs a method", None));
        };
        let Some(id) = id else {
            // A notification (initialized, cancelled, anything): nothing is sent back. A request with a null or
            // fractional id is not a notification, but it has no usable id to answer under either.
            if object.contains_key("id") {
                return Some(error(None, -32600, "a request id is a string or an integer", None));
            }
            return None;
        };
        if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Some(error(Some(id), -32600, "jsonrpc must be \"2.0\"", None));
        }
        let params = object.get("params").cloned().unwrap_or(Value::Null);
        if !(params.is_null() || params.is_object()) {
            return Some(error(Some(id), -32602, "params must be an object", None));
        }
        Some(match self.route(method, &params) {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err((code, message, data)) => error(Some(id), code, &message, data),
        })
    }

    fn route(&mut self, method: &str, params: &Value) -> Result<Value, (i64, String, Option<Value>)> {
        let invalid = |message: &str| (-32602, message.to_owned(), None);
        if method == "initialize" {
            self.legacy = true;
            let requested = params.get("protocolVersion").and_then(Value::as_str).ok_or_else(|| invalid("initialize needs a protocolVersion"))?;
            let version = if LEGACY.contains(&requested) { requested } else { LEGACY[0] };
            return Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": server_info(),
                "instructions": INSTRUCTIONS
            }));
        }
        let meta = params.get("_meta").and_then(Value::as_object).filter(|meta| meta.contains_key(VERSION_KEY));
        let modern = meta.is_some();
        if !modern && !self.legacy && method != "ping" {
            return Err(invalid(&format!(
                "send _meta[\"{VERSION_KEY}\"] and _meta[\"{CAPABILITIES_KEY}\"] with each request (protocol {MODERN}), or send initialize first (protocol {})",
                LEGACY[0]
            )));
        }
        if let Some(meta) = meta {
            let requested = meta[VERSION_KEY].as_str().ok_or_else(|| invalid(&format!("_meta[\"{VERSION_KEY}\"] must be a string")))?;
            if requested != MODERN {
                let mut supported = vec![MODERN];
                supported.extend(LEGACY);
                return Err((-32022, "Unsupported protocol version".to_owned(), Some(json!({"supported": supported, "requested": requested}))));
            }
            if !meta.get(CAPABILITIES_KEY).is_some_and(Value::is_object) {
                return Err(invalid(&format!("_meta[\"{CAPABILITIES_KEY}\"] is required and must be an object")));
            }
        }
        let mut result = match method {
            // Removed from 2026-07-28 but harmless, and earlier clients send it.
            "ping" => json!({}),
            "server/discover" if modern => {
                let mut supported = vec![MODERN];
                supported.extend(LEGACY);
                json!({"supportedVersions": supported, "capabilities": {"tools": {}}, "instructions": INSTRUCTIONS, "ttlMs": 3_600_000, "cacheScope": "public"})
            }
            "tools/list" => {
                if modern {
                    json!({"tools": tools(), "ttlMs": 3_600_000, "cacheScope": "public"})
                } else {
                    json!({"tools": tools()})
                }
            }
            "tools/call" => self.call(params)?,
            _ => return Err((-32601, format!("Method not found: {method}"), None)),
        };
        if modern {
            result["resultType"] = json!("complete");
            result["_meta"] = json!({ SERVER_INFO_KEY: server_info() });
        }
        Ok(result)
    }

    fn call(&self, params: &Value) -> Result<Value, (i64, String, Option<Value>)> {
        let name = params.get("name").and_then(Value::as_str).ok_or((-32602, "tools/call needs a tool name".to_owned(), None))?;
        let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
        let Some(arguments) = arguments.as_object() else {
            return Ok(failure("arguments must be an object."));
        };
        let full = match arguments.get("detail") {
            None => false,
            Some(Value::String(detail)) if detail == "summary" => false,
            Some(Value::String(detail)) if detail == "full" => true,
            Some(other) => return Ok(failure(format!("detail is \"summary\" or \"full\", not {other}."))),
        };
        let argument = |key: &str| -> Result<&str, Value> {
            arguments.get(key).and_then(Value::as_str).ok_or_else(|| failure(format!("{key} is required and must be a string.")))
        };
        let result = match name {
            "litmus_check_text" => argument("text").map(|text| check_text(self.scorer, text, full, None)),
            "litmus_check_file" => argument("path").map(|path| check_file(self.scorer, path, full)),
            other => return Err((-32602, format!("Unknown tool: {other}"), None)),
        };
        Ok(result.unwrap_or_else(|failure| failure))
    }
}

/// One line of at most `limit` bytes, without its line break. A longer line is skipped to its end and reported.
fn read_line(reader: &mut dyn BufRead, limit: usize) -> std::io::Result<Option<Result<Vec<u8>, usize>>> {
    let mut line = Vec::new();
    let read = Read::take(&mut *reader, limit as u64 + 1).read_until(b'\n', &mut line)?;
    if read == 0 {
        return Ok(None);
    }
    if line.last() == Some(&b'\n') {
        line.pop();
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        return Ok(Some(Ok(line)));
    }
    if line.len() <= limit {
        return Ok(Some(Ok(line))); // the last line, without a break
    }
    let mut skipped = line.len();
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        match buffer.iter().position(|&b| b == b'\n') {
            Some(at) => {
                skipped += at;
                reader.consume(at + 1);
                break;
            }
            None => {
                skipped += buffer.len();
                let all = buffer.len();
                reader.consume(all);
            }
        }
    }
    Ok(Some(Err(skipped)))
}

fn send(output: &mut dyn Write, message: &Value) -> std::io::Result<()> {
    // `to_string` writes one line: every line break inside a string is escaped.
    writeln!(output, "{message}")?;
    output.flush()
}

/// Serve until the input closes.
pub fn serve(scorer: &dyn Scorer, input: &mut dyn BufRead, output: &mut dyn Write) -> std::io::Result<()> {
    serve_limited(scorer, input, output, MAX_LINE_BYTES)
}

fn serve_limited(scorer: &dyn Scorer, input: &mut dyn BufRead, output: &mut dyn Write, line_limit: usize) -> std::io::Result<()> {
    let mut server = Server::new(scorer);
    while let Some(line) = read_line(input, line_limit)? {
        let reply = match line {
            Err(skipped) => Some(error(None, -32600, &format!("a message line of {skipped} bytes is over the limit of {line_limit}"), None)),
            Ok(bytes) if bytes.iter().all(u8::is_ascii_whitespace) => None,
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(message) => server.handle(&message),
                Err(e) => Some(error(None, -32700, &format!("Parse error: {e}"), None)),
            },
        };
        if let Some(reply) = reply {
            send(output, &reply)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reading of a text of `n_tokens` tokens with the 0.10 set `set`, as `litmus score --json` would hold it.
    fn document(n_tokens: usize, set: &[&str], mixed_document: bool) -> DocumentReading {
        let strip = json!({"position": 0.8, "band_lo": 0.4, "band_hi": 1.0, "level": 0.1, "unlike_reference": set.is_empty()});
        let sets = json!({"0.05": set, "0.1": set});
        serde_json::from_value(json!({
            "n_tokens": n_tokens,
            "chunk_tokens": 8192,
            "verdict": {
                "chunks": 1, "level": 0.05, "p_values": [0.02, 0.4, 0.1], "sets": sets, "not_human": {"0.05": true, "0.1": true},
                "probability": 0.81, "interval": [0.62, 0.9], "disagreement": 0.0, "lowest": 0.81, "highest": 0.81,
                "three_way": [0.2, 0.7, 0.1], "closeness_p": 0.5, "out_of_reference": false, "strip": strip,
                "chunk_level": 0.05, "model_chunks": 0, "human_chunks": 0, "mixed_document": mixed_document
            },
            "chunks": [], "windows": [], "cells": []
        }))
        .expect("a reading")
    }

    /// Counts words as tokens and returns a fixed reading, so the protocol is tested without a model.
    struct Stub(DocumentReading);

    impl Scorer for Stub {
        fn count_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }

        fn score(&self, text: &str) -> Result<DocumentReading, String> {
            if text.trim().is_empty() { Err("the text has no tokens, so no reading".into()) } else { Ok(self.0.clone()) }
        }
    }

    fn modern(method: &str, id: u64, params: Value) -> Value {
        let mut params = params;
        params["_meta"] = json!({VERSION_KEY: MODERN, CAPABILITIES_KEY: {}, "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"}});
        json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
    }

    fn run(server: &mut Server, message: Value) -> Value {
        server.handle(&message).expect("a reply")
    }

    #[test]
    fn modern_requests_are_served_without_a_handshake() {
        let stub = Stub(document(200, &["ai"], false));
        let mut server = Server::new(&stub);
        let discover = run(&mut server, modern("server/discover", 1, json!({})));
        assert_eq!(discover["result"]["resultType"], "complete");
        assert_eq!(discover["result"]["supportedVersions"][0], MODERN);
        assert_eq!(discover["result"]["_meta"][SERVER_INFO_KEY]["name"], "litmus");
        let list = run(&mut server, modern("tools/list", 2, json!({})));
        let names: Vec<&str> = list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["litmus_check_text", "litmus_check_file"]);
        assert!(list["result"]["ttlMs"].is_u64() && list["result"]["cacheScope"] == "public");
        let call = run(&mut server, modern("tools/call", 3, json!({"name": "litmus_check_text", "arguments": {"text": "one two three"}})));
        let result = &call["result"];
        assert_eq!(result["resultType"], "complete");
        assert_eq!(result["isError"], false);
        assert_eq!(result["structuredContent"]["label"], "AI");
        assert_eq!(result["structuredContent"]["reading"]["verdict"]["probability"], 0.81);
        assert!(result["structuredContent"]["reading"].get("windows").is_none(), "summary leaves the heat map out");
        assert_eq!(result["structuredContent"]["words"], 3, "the words of \"one two three\"");
        assert_eq!(result["structuredContent"]["characters"], 13);
        assert_eq!(result["structuredContent"]["n_tokens"], 200, "the tokens are the reading's own");
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("Litmus reads this text as AI (not ruled out at the 10% level: {AI}); P(not human) 0.81"), "{text}");
        assert!(text.ends_with(", 3 words, 13 characters, 200 tokens."), "{text}");
        assert_eq!(text.matches('.').count(), 4, "one sentence: {text}");
    }

    #[test]
    fn legacy_clients_open_with_initialize() {
        let stub = Stub(document(200, &["human", "ai"], false));
        let mut server = Server::new(&stub);
        let list_first = server.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})).unwrap();
        assert_eq!(list_first["error"]["code"], -32602, "no _meta and no initialize");
        let init = run(&mut server, json!({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}}}));
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(init["result"]["capabilities"], json!({"tools": {}}));
        assert!(init["result"].get("resultType").is_none());
        let newer = run(&mut server, json!({"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {"protocolVersion": "2030-01-01"}}));
        assert_eq!(newer["result"]["protocolVersion"], LEGACY[0], "an unknown version gets the server's latest legacy one");
        assert_eq!(server.handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})), None);
        assert_eq!(run(&mut server, json!({"jsonrpc": "2.0", "id": 4, "method": "ping"}))["result"], json!({}));
        let list = run(&mut server, json!({"jsonrpc": "2.0", "id": "a", "method": "tools/list"}));
        assert_eq!(list["id"], "a");
        assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 2);
        assert!(list["result"].get("resultType").is_none());
        let call = run(&mut server, json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "litmus_check_text", "arguments": {"text": "a b", "detail": "full"}}}));
        assert_eq!(call["result"]["structuredContent"]["label"], "uncertain");
        assert_eq!((call["result"]["structuredContent"]["words"].as_u64(), call["result"]["structuredContent"]["characters"].as_u64()), (Some(2), Some(3)));
        assert!(call["result"]["structuredContent"]["reading"]["windows"].is_array());
        assert!(call["result"]["structuredContent"]["caveats"][0].as_str().unwrap().contains("More than one origin"));
    }

    #[test]
    fn bad_requests_are_told_apart() {
        let stub = Stub(document(200, &["ai"], false));
        let mut server = Server::new(&stub);
        let old = run(&mut server, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": {VERSION_KEY: "1900-01-01", CAPABILITIES_KEY: {}}}}));
        assert_eq!(old["error"]["code"], -32022);
        assert_eq!(old["error"]["data"]["requested"], "1900-01-01");
        assert_eq!(old["error"]["data"]["supported"][0], MODERN);
        let no_capabilities = run(&mut server, json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": {VERSION_KEY: MODERN}}}));
        assert_eq!(no_capabilities["error"]["code"], -32602);
        assert_eq!(run(&mut server, modern("resources/list", 3, json!({})))["error"]["code"], -32601);
        assert_eq!(run(&mut server, modern("tools/call", 4, json!({"name": "nope", "arguments": {}})))["error"]["code"], -32602);
        assert_eq!(run(&mut server, modern("tools/call", 5, json!({"arguments": {}})))["error"]["code"], -32602);
        for arguments in [json!({}), json!({"text": 5}), json!({"text": "a", "detail": "most"}), json!("text")] {
            let reply = run(&mut server, modern("tools/call", 6, json!({"name": "litmus_check_text", "arguments": arguments})));
            assert_eq!(reply["result"]["isError"], true, "{arguments}");
            assert!(reply.get("error").is_none());
        }
        let empty = run(&mut server, modern("tools/call", 7, json!({"name": "litmus_check_text", "arguments": {"text": "  "}})));
        assert_eq!(empty["result"]["isError"], true);
        assert!(empty["result"]["content"][0]["text"].as_str().unwrap().contains("no tokens"));
        let missing = run(&mut server, modern("tools/call", 8, json!({"name": "litmus_check_file", "arguments": {"path": "no/such/file.txt"}})));
        assert_eq!(missing["result"]["isError"], true);
        assert!(missing["result"]["content"][0]["text"].as_str().unwrap().contains("no/such/file.txt"));
        let wrong_jsonrpc = server.handle(&json!({"jsonrpc": "1.0", "id": 9, "method": "ping"})).unwrap();
        assert_eq!(wrong_jsonrpc["error"]["code"], -32600);
        assert_eq!(server.handle(&json!([{"jsonrpc": "2.0", "id": 1, "method": "ping"}])).unwrap()["error"]["code"], -32600);
        assert_eq!(server.handle(&json!({"jsonrpc": "2.0", "id": null, "method": "ping"})).unwrap()["error"]["code"], -32600);
        assert_eq!(server.handle(&json!({"jsonrpc": "2.0", "id": 1.5, "method": "ping"})).unwrap()["error"]["code"], -32600);
        assert_eq!(server.handle(&json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}})), None);
        assert_eq!(server.handle(&json!({"jsonrpc": "2.0", "id": 1, "result": {}})), None, "a response from the client is ignored");
    }

    #[test]
    fn long_and_short_texts_are_handled() {
        let stub = Stub(document(40, &["ai"], true));
        let mut server = Server::new(&stub);
        let short = run(&mut server, modern("tools/call", 1, json!({"name": "litmus_check_text", "arguments": {"text": "one two"}})));
        let caveats = short["result"]["structuredContent"]["caveats"].as_array().unwrap();
        assert!(caveats.iter().any(|c| c.as_str().unwrap().starts_with("Short text (40 tokens)")));
        assert!(caveats.iter().any(|c| c.as_str().unwrap().contains("Different chunks")));
        let long = "word ".repeat(MAX_TOKENS + 1);
        let refused = run(&mut server, modern("tools/call", 2, json!({"name": "litmus_check_text", "arguments": {"text": long}})));
        assert_eq!(refused["result"]["isError"], true);
        assert!(refused["result"]["content"][0]["text"].as_str().unwrap().contains("131,072"));
        let huge = "x".repeat(MAX_TEXT_BYTES + 1);
        let refused = run(&mut server, modern("tools/call", 3, json!({"name": "litmus_check_text", "arguments": {"text": huge}})));
        assert_eq!(refused["result"]["isError"], true);
    }

    #[test]
    fn a_missing_model_is_reported_by_each_check() {
        let missing = Missing("no model: pass --model DIR".to_owned());
        let mut server = Server::new(&missing);
        assert_eq!(run(&mut server, modern("tools/list", 1, json!({})))["result"]["tools"].as_array().unwrap().len(), 2);
        let call = run(&mut server, modern("tools/call", 2, json!({"name": "litmus_check_text", "arguments": {"text": "hello"}})));
        assert_eq!(call["result"]["isError"], true);
        assert!(call["result"]["content"][0]["text"].as_str().unwrap().contains("no model is loaded"));
    }

    fn lines(input: &[u8], limit: usize) -> Vec<Value> {
        let stub = Stub(document(200, &["ai"], false));
        let mut output = Vec::new();
        serve_limited(&stub, &mut std::io::Cursor::new(input.to_vec()), &mut output, limit).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.is_empty() || text.ends_with('\n'));
        text.lines().map(|line| serde_json::from_str(line).expect("each output line is one JSON message")).collect()
    }

    #[test]
    fn the_loop_keeps_one_message_a_line_and_survives_bad_input() {
        let ping = |id: u64| format!("{}\n", json!({"jsonrpc": "2.0", "id": id, "method": "ping"}));
        let mut input = Vec::new();
        input.extend(b"\n   \n");
        input.extend(b"{not json\n");
        input.extend([0xff, 0xfe, b'\n']); // not UTF-8
        input.extend(ping(1).as_bytes());
        input.extend(format!("{}\r\n", json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).as_bytes());
        input.extend(ping(2).as_bytes());
        input.extend(format!("{}", json!({"jsonrpc": "2.0", "id": 3, "method": "ping"})).as_bytes()); // no final break
        let replies = lines(&input, 1 << 20);
        let codes: Vec<i64> = replies.iter().filter_map(|r| r["error"]["code"].as_i64()).collect();
        assert_eq!(codes, [-32700, -32700]);
        let ids: Vec<u64> = replies.iter().filter_map(|r| r["id"].as_u64()).collect();
        assert_eq!(ids, [1, 2, 3]);
    }

    #[test]
    fn an_oversized_line_is_skipped_and_the_next_is_read() {
        let mut input = vec![b'x'; 300];
        input.push(b'\n');
        input.extend(format!("{}\n", json!({"jsonrpc": "2.0", "id": 7, "method": "ping"})).as_bytes());
        let replies = lines(&input, 100);
        assert_eq!(replies[0]["error"]["code"], -32600);
        assert!(replies[0]["error"]["message"].as_str().unwrap().contains("300"));
        assert_eq!(replies[1]["id"], 7);
        assert_eq!(replies.len(), 2);
    }
}
