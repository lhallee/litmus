//! `litmus mcp` against a scripted client, on a real release: the handshake-free revision (2026-07-28) and the
//! `initialize` one (2025-11-25), a good call, bad calls, malformed input, a PDF-free file check, and the reading each
//! tool returns compared with `litmus score --json` on the same text.
//!
//! `LITMUS_MODEL` names a release folder holding `model.safetensors`, `config.json` and `tokenizer.json`
//! (engine/README.md, Tests). Run it with `cargo test --release --features release-tests --test mcp`.
#![cfg(feature = "release-tests")]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

const MODERN: &str = "2026-07-28";

const PARAGRAPH: &str = "The committee met on Tuesday to review the proposal. After a long discussion about costs and timelines, the members agreed to postpone the vote until the spring, when the new figures would be ready. Several of them said they would rather wait than decide on numbers nobody trusted, and the chair, who had hoped to finish that evening, agreed that a month would change little.";
const SHORT: &str = "The road was closed for three weeks, so she sold yesterday's bread at the far end of the detour.";
const CODE: &str = "def total(items):\n    \"\"\"Add up the prices, skipping empty entries.\"\"\"\n    return sum(i.price for i in items if i is not None)\n";
const NON_ENGLISH: &str = "Der Ausschuss tagte am Dienstag, um den Vorschlag zu prüfen. Nach langer Diskussion über Kosten und Zeitpläne beschlossen die Mitglieder, die Abstimmung auf das Frühjahr zu verschieben.\n\n日本語の文章も読めますが、このモデルは主に英語で学習されています。";

fn release() -> PathBuf {
    PathBuf::from(std::env::var("LITMUS_MODEL").expect("set LITMUS_MODEL to a release folder (engine/README.md, Tests)"))
}

fn litmus() -> Command {
    Command::new(env!("CARGO_BIN_EXE_litmus"))
}

struct Client {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
    next_id: u64,
}

impl Client {
    fn start(model: &Path) -> Client {
        let mut child = litmus()
            .arg("mcp")
            .arg("--model")
            .arg(model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start litmus mcp");
        let input = child.stdin.take();
        let output = BufReader::new(child.stdout.take().expect("stdout"));
        Client { child, input, output, next_id: 1 }
    }

    /// Send one raw line and read the one reply line, which must be one JSON message.
    fn raw(&mut self, line: &[u8]) -> Value {
        let input = self.input.as_mut().expect("input open");
        input.write_all(line).unwrap();
        input.write_all(b"\n").unwrap();
        input.flush().unwrap();
        self.reply()
    }

    fn reply(&mut self) -> Value {
        let mut reply = String::new();
        let read = self.output.read_line(&mut reply).unwrap();
        assert!(read > 0, "the server closed its output");
        assert!(reply.ends_with('\n') && !reply[..reply.len() - 1].contains('\n'));
        serde_json::from_str(&reply).unwrap_or_else(|e| panic!("stdout holds only messages, got {reply:?}: {e}"))
    }

    fn send(&mut self, message: &Value) {
        let input = self.input.as_mut().expect("input open");
        writeln!(input, "{message}").unwrap();
        input.flush().unwrap();
    }

    /// A request with the per-request metadata of 2026-07-28.
    fn modern(&mut self, method: &str, mut params: Value) -> Value {
        params["_meta"] = json!({
            "io.modelcontextprotocol/protocolVersion": MODERN,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "litmus-test", "version": "0"}
        });
        self.legacy(method, params)
    }

    /// A request as it is after `initialize`.
    fn legacy(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let reply = self.reply();
        assert_eq!(reply["id"], id);
        assert_eq!(reply["jsonrpc"], "2.0");
        reply
    }

    fn call(&mut self, tool: &str, arguments: Value) -> Value {
        self.modern("tools/call", json!({"name": tool, "arguments": arguments}))
    }

    /// Close the input and require a clean exit.
    fn finish(mut self) {
        drop(self.input.take());
        assert!(self.child.wait().unwrap().success(), "the server exits cleanly when its input closes");
    }
}

/// `litmus score --json --no-cache` on the text, as the file holds it.
fn cli_reading(model: &Path, file: &Path) -> Value {
    let output = litmus().arg("score").arg("--json").arg("--no-cache").arg("--model").arg(model).arg(file).output().expect("run litmus score");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
    serde_json::from_slice(&output.stdout).expect("one JSON object")
}

#[test]
fn the_server_speaks_both_protocol_eras_and_reads_as_the_command_line_does() {
    let model = release();
    let folder = std::env::temp_dir().join(format!("litmus_mcp_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    std::fs::create_dir_all(&folder).unwrap();
    let long: String = (0..60).map(|i| format!("{PARAGRAPH} (part {i}) ")).collect::<String>().repeat(4);
    let texts = [("short.txt", SHORT.to_owned()), ("paragraph.txt", PARAGRAPH.to_owned()), ("code.py", CODE.to_owned()), ("german_japanese.txt", NON_ENGLISH.to_owned()), ("long.txt", long)];
    for (name, text) in &texts {
        std::fs::write(folder.join(name), text).unwrap();
    }

    let mut client = Client::start(&model);

    // 2026-07-28: no handshake, a discover call, then the tools.
    let discover = client.modern("server/discover", json!({}));
    assert_eq!(discover["result"]["resultType"], "complete");
    assert!(discover["result"]["supportedVersions"].as_array().unwrap().iter().any(|v| v == MODERN));
    assert!(discover["result"]["instructions"].as_str().unwrap().starts_with("Litmus is a free detector of AI written text that says how sure it is"));
    let list = client.modern("tools/list", json!({}));
    let tools = list["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.iter().map(|t| t["name"].as_str().unwrap()).collect::<Vec<_>>(), ["litmus_check_text", "litmus_check_file"]);
    for tool in tools {
        assert_eq!(tool["inputSchema"]["type"], "object");
        assert_eq!(tool["annotations"]["readOnlyHint"], true);
        assert_eq!(tool["annotations"]["openWorldHint"], false);
        let description = tool["description"].as_str().unwrap();
        for phrase in ["never proof", "English", "code is weaker", "short texts are less reliable", "never an accusation"] {
            assert!(description.contains(phrase), "{} says {phrase:?}", tool["name"]);
        }
    }

    // Each text, by the string tool and by the file tool, against `litmus score --json`.
    for (name, text) in &texts {
        let path = folder.join(name);
        let expected = cli_reading(&model, &path);
        let by_text = client.call("litmus_check_text", json!({"text": text, "detail": "full"}));
        let by_file = client.call("litmus_check_file", json!({"path": path.to_str().unwrap(), "detail": "full"}));
        for reply in [&by_text, &by_file] {
            let result = &reply["result"];
            assert_eq!(result["isError"], false, "{name}: {result}");
            assert_eq!(result["resultType"], "complete");
            let structured = &result["structuredContent"];
            assert_eq!(structured["reading"], expected["reading"], "{name}: the reading is the command line's");
            assert_eq!(structured["label"], expected["label"], "{name}");
            assert_eq!(structured["n_tokens"], expected["reading"]["n_tokens"]);
            assert_eq!(structured["words"], expected["counts"]["words"], "{name}: the command line's word count");
            assert_eq!(structured["characters"], expected["counts"]["characters"], "{name}: the command line's character count");
            assert_eq!(expected["counts"]["tokens"], expected["reading"]["n_tokens"], "{name}");
            assert!(structured["guidance"].as_str().unwrap().contains("never proof"));
            let sentence = result["content"][0]["text"].as_str().unwrap();
            assert!(sentence.starts_with("Litmus reads this text as "), "{sentence}");
            assert_eq!(result["content"].as_array().unwrap().len(), 1);
        }
        assert_eq!(by_file["result"]["structuredContent"]["file"], path.to_str().unwrap());
        let summary = client.call("litmus_check_text", json!({"text": text}));
        let structured = &summary["result"]["structuredContent"];
        assert_eq!(structured["detail"], "summary");
        assert_eq!(structured["reading"]["verdict"], expected["reading"]["verdict"], "{name}: the summary keeps the verdict whole");
        assert!(structured["reading"].get("windows").is_none() && structured["reading"].get("cells").is_none());
    }
    // The short text is flagged as short, and the long one is read in several chunks.
    let short = client.call("litmus_check_text", json!({"text": SHORT}));
    assert!(short["result"]["structuredContent"]["caveats"].to_string().contains("Short text"));
    let long = client.call("litmus_check_text", json!({"text": texts[4].1}));
    assert!(long["result"]["structuredContent"]["reading"]["verdict"]["chunks"].as_u64().unwrap() > 1);

    // Bad calls: a protocol error for an unknown tool, a tool error for bad arguments and files, none of them fatal.
    assert_eq!(client.call("no_such_tool", json!({}))["error"]["code"], -32602);
    for arguments in [json!({}), json!({"text": 3}), json!({"text": "x", "detail": "everything"}), json!({"text": "   \n"})] {
        let reply = client.call("litmus_check_text", arguments.clone());
        assert_eq!(reply["result"]["isError"], true, "{arguments}");
    }
    let missing = client.call("litmus_check_file", json!({"path": folder.join("absent.txt").to_str().unwrap()}));
    assert_eq!(missing["result"]["isError"], true);
    let directory = client.call("litmus_check_file", json!({"path": folder.to_str().unwrap()}));
    assert!(directory["result"]["content"][0]["text"].as_str().unwrap().contains("folder"));
    let old = client.legacy("tools/list", json!({"_meta": {"io.modelcontextprotocol/protocolVersion": "1900-01-01", "io.modelcontextprotocol/clientCapabilities": {}}}));
    assert_eq!(old["error"]["code"], -32022);
    assert_eq!(client.modern("resources/list", json!({}))["error"]["code"], -32601);

    // Malformed input: the server answers with a parse error and keeps going.
    assert_eq!(client.raw(b"{this is not json")["error"]["code"], -32700);
    assert_eq!(client.raw(&[0xff, 0xfe, 0xfd])["error"]["code"], -32700);
    assert_eq!(client.raw(b"[1, 2]")["error"]["code"], -32600);
    assert_eq!(client.raw(br#"{"jsonrpc": "2.0", "id": 99, "method": "ping"}"#)["result"], json!({}));
    let again = client.call("litmus_check_text", json!({"text": PARAGRAPH}));
    assert_eq!(again["result"]["isError"], false);
    client.send(&json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}));
    client.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    assert_eq!(client.raw(br#"{"jsonrpc": "2.0", "id": 100, "method": "ping"}"#)["id"], 100, "notifications get no reply");
    client.finish();

    // 2025-11-25: the same server, a client that opens with initialize.
    let mut client = Client::start(&model);
    let init = client.legacy("initialize", json!({"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "legacy", "version": "0"}}));
    assert_eq!(init["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(init["result"]["serverInfo"]["name"], "litmus");
    assert!(init["result"]["capabilities"]["tools"].is_object());
    client.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    assert_eq!(client.legacy("ping", json!({}))["result"], json!({}));
    assert_eq!(client.legacy("tools/list", json!({}))["result"]["tools"].as_array().unwrap().len(), 2);
    let expected = cli_reading(&model, &folder.join("paragraph.txt"));
    let call = client.legacy("tools/call", json!({"name": "litmus_check_text", "arguments": {"text": PARAGRAPH, "detail": "full"}}));
    assert_eq!(call["result"]["structuredContent"]["reading"], expected["reading"]);
    assert!(call["result"].get("resultType").is_none());
    client.finish();

    std::fs::remove_dir_all(&folder).unwrap();
}

#[test]
fn a_release_that_does_not_load_is_reported_by_each_check() {
    let nowhere = std::env::temp_dir().join(format!("litmus_mcp_no_model_{}", std::process::id()));
    let mut client = Client::start(&nowhere);
    assert_eq!(client.modern("tools/list", json!({}))["result"]["tools"].as_array().unwrap().len(), 2);
    let reply = client.call("litmus_check_text", json!({"text": PARAGRAPH}));
    assert_eq!(reply["result"]["isError"], true);
    assert!(reply["result"]["content"][0]["text"].as_str().unwrap().contains("no model is loaded"));
    client.finish();
}

#[test]
fn mcp_refuses_arguments_that_belong_to_other_commands() {
    let model = release();
    for arguments in [vec!["--json"], vec!["essay.txt"], vec!["--no-cache"]] {
        let output = litmus().arg("mcp").arg("--model").arg(&model).args(&arguments).stdin(Stdio::null()).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{arguments:?}");
        assert!(output.stdout.is_empty(), "nothing but protocol messages may reach standard output");
    }
}
