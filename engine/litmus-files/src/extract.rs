//! Text from the files the command line and the app read: plain text and Markdown as written, HTML with its markup
//! removed, Word documents (`.docx`, the paragraphs of `word/document.xml`), and PDF through `pdf-extract`.

#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
use std::path::Path;

/// The argument that makes a Litmus executable (the command line or the app) print a PDF's text and exit: the
/// executable calls `pdf_child` when its first argument is this.
pub const PDF_CHILD_ARGUMENT: &str = "__pdf-text";

/// The kinds of file `litmus score` and the app read, by extension.
pub fn readable(path: &Path) -> bool {
    matches!(extension(path).as_str(), "txt" | "text" | "md" | "markdown" | "html" | "htm" | "docx" | "pdf")
}

fn extension(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

/// The text of a file's bytes, by the path's extension (plain text when unknown).
pub fn text_of(path: &Path, bytes: &[u8]) -> Result<String, String> {
    match extension(path).as_str() {
        "html" | "htm" => Ok(html_text(&String::from_utf8_lossy(bytes))),
        "docx" => docx_text(bytes),
        "pdf" => pdf_text(path),
        _ => Ok(String::from_utf8_lossy(bytes).into_owned()),
    }
}

/// Read standard input to the end.
#[cfg(not(target_arch = "wasm32"))]
pub fn stdin_text() -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).map_err(|e| format!("standard input: {e}"))?;
    Ok(bytes)
}

/// The line after which the PDF child prints the text (`pdf-extract` prints its own diagnostics to standard output).
#[cfg(not(target_arch = "wasm32"))]
const PDF_MARKER: &str = "\u{1e}litmus pdf text\u{1e}\n";

#[cfg(target_arch = "wasm32")]
fn pdf_text(_path: &Path) -> Result<String, String> {
    Err("PDF files are read by the desktop app and the command line, not in the browser".to_owned())
}

/// The text of a PDF, extracted in a child process (this executable with `__pdf-text <file>`): `pdf-extract` prints
/// diagnostics to standard output and panics on some malformed files, and neither may reach this run's output.
#[cfg(not(target_arch = "wasm32"))]
fn pdf_text(path: &Path) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot start the PDF reader: {e}"))?;
    let output = std::process::Command::new(exe)
        .arg(PDF_CHILD_ARGUMENT)
        .arg(path)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("cannot start the PDF reader: {e}"))?;
    let stdout = String::from_utf8(output.stdout).map_err(|_| "the PDF reader returned text that is not UTF-8".to_owned())?;
    match (output.status.success(), stdout.rfind(PDF_MARKER)) {
        (true, Some(at)) => Ok(stdout[at + PDF_MARKER.len()..].to_owned()),
        _ => Err("the PDF reader failed on this file".to_owned()),
    }
}

/// The child's side: print the marker, then the PDF's text; an unreadable file exits non-zero.
#[cfg(not(target_arch = "wasm32"))]
pub fn pdf_child(path: &Path) -> Result<bool, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = pdf_extract::extract_text_from_mem(&bytes).map_err(|e| format!("PDF: {e}"))?;
    print!("{PDF_MARKER}{text}");
    Ok(true)
}

/// Decode the entities HTML and XML text use.
fn decode_entities(text: &str, out: &mut String) {
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let end = rest[1..].find([';', '&', '<', ' ']).map(|i| i + 1).filter(|&i| rest.as_bytes()[i] == b';' && i <= 12);
        let Some(end) = end else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let name = &rest[1..end];
        let decoded = match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some('\u{a0}'),
            "mdash" => Some('\u{2014}'),
            "ndash" => Some('\u{2013}'),
            "hellip" => Some('\u{2026}'),
            "lsquo" => Some('\u{2018}'),
            "rsquo" => Some('\u{2019}'),
            "ldquo" => Some('\u{201c}'),
            "rdquo" => Some('\u{201d}'),
            "copy" => Some('\u{a9}'),
            "reg" => Some('\u{ae}'),
            "trade" => Some('\u{2122}'),
            "deg" => Some('\u{b0}'),
            "middot" => Some('\u{b7}'),
            "bull" => Some('\u{2022}'),
            "times" => Some('\u{d7}'),
            "euro" => Some('\u{20ac}'),
            _ if name.starts_with("#x") || name.starts_with("#X") => u32::from_str_radix(&name[2..], 16).ok().and_then(char::from_u32),
            _ if name.starts_with('#') => name[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match decoded {
            Some(c) => out.push(c),
            None => out.push_str(&rest[..=end]),
        }
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
}

const BLOCK_TAGS: &[&str] = &[
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
    "header", "footer", "blockquote", "pre", "hr", "dt", "dd", "figcaption", "main", "nav", "aside", "td", "th",
];

/// HTML to text: drop comments, scripts, styles and tags; a block-level tag starts a new line; whitespace inside
/// text collapses as a browser shows it, except inside `<pre>`.
pub fn html_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut text = String::new();
    let mut rest = html;
    let mut preformatted = 0usize;
    let flush = |text: &mut String, out: &mut String, preformatted: bool| {
        let mut decoded = String::with_capacity(text.len());
        decode_entities(text, &mut decoded);
        if preformatted {
            out.push_str(&decoded);
        } else {
            let mut last_space = out.ends_with([' ', '\n']) || out.is_empty();
            for c in decoded.chars() {
                if c.is_whitespace() && c != '\u{a0}' {
                    if !last_space {
                        out.push(' ');
                        last_space = true;
                    }
                } else {
                    out.push(c);
                    last_space = false;
                }
            }
        }
        text.clear();
    };
    while let Some(at) = rest.find('<') {
        text.push_str(&rest[..at]);
        rest = &rest[at..];
        if rest.starts_with("<!--") {
            rest = rest.find("-->").map_or("", |end| &rest[end + 3..]);
            continue;
        }
        let Some(close) = rest.find('>') else {
            text.push_str(rest);
            rest = "";
            break;
        };
        let tag = &rest[1..close];
        rest = &rest[close + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        if !closing && (name == "script" || name == "style" || name == "noscript" || name == "template") {
            let end_tag = format!("</{name}");
            let lower = rest.to_ascii_lowercase();
            rest = lower.find(&end_tag).map_or("", |end| &rest[end..]);
            continue;
        }
        if BLOCK_TAGS.contains(&name.as_str()) {
            flush(&mut text, &mut out, preformatted > 0);
            while out.ends_with(' ') {
                out.pop();
            }
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            if name == "pre" {
                preformatted = if closing { preformatted.saturating_sub(1) } else { preformatted + 1 };
            }
            if !closing && matches!(name.as_str(), "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "blockquote" | "table" | "pre") && !out.is_empty() && !out.ends_with("\n\n") {
                out.push('\n');
            }
        }
    }
    text.push_str(rest);
    flush(&mut text, &mut out, preformatted > 0);
    out.trim().to_owned()
}

/// A file inside a ZIP archive (the central directory, stored or deflated entries).
fn zip_entry(bytes: &[u8], wanted: &str) -> Result<Vec<u8>, String> {
    let u16_at = |at: usize| -> Result<usize, String> {
        bytes.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize).ok_or_else(|| "truncated ZIP".to_owned())
    };
    let u32_at = |at: usize| -> Result<usize, String> {
        bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).ok_or_else(|| "truncated ZIP".to_owned())
    };
    let search_from = bytes.len().saturating_sub(22 + 65_535);
    let end = (search_from..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or("not a ZIP archive (no end of central directory)")?;
    let entries = u16_at(end + 10)?;
    let mut at = u32_at(end + 16)?;
    for _ in 0..entries {
        if bytes.get(at..at + 4) != Some(&[0x50, 0x4b, 0x01, 0x02]) {
            return Err("damaged ZIP central directory".into());
        }
        let method = u16_at(at + 10)?;
        let compressed = u32_at(at + 20)?;
        let name_length = u16_at(at + 28)?;
        let extra_length = u16_at(at + 30)?;
        let comment_length = u16_at(at + 32)?;
        let local = u32_at(at + 42)?;
        let name = bytes.get(at + 46..at + 46 + name_length).ok_or("truncated ZIP")?;
        if name == wanted.as_bytes() {
            let data_start = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
            let data = bytes.get(data_start..data_start + compressed).ok_or("truncated ZIP entry")?;
            return match method {
                0 => Ok(data.to_vec()),
                8 => miniz_oxide::inflate::decompress_to_vec(data).map_err(|e| format!("ZIP entry {wanted}: {e:?}")),
                other => Err(format!("ZIP entry {wanted} uses compression method {other}")),
            };
        }
        at += 46 + name_length + extra_length + comment_length;
    }
    Err(format!("no {wanted} in the archive"))
}

/// The text of a Word document: `<w:t>` runs, a tab per `<w:tab/>`, a line per `<w:br/>` and per paragraph.
pub fn docx_text(bytes: &[u8]) -> Result<String, String> {
    let xml = zip_entry(bytes, "word/document.xml")?;
    let xml = String::from_utf8_lossy(&xml);
    let mut out = String::new();
    let mut rest: &str = &xml;
    while let Some(at) = rest.find('<') {
        rest = &rest[at..];
        let Some(close) = rest.find('>') else { break };
        let tag = &rest[1..close];
        rest = &rest[close + 1..];
        let name = tag.split([' ', '/']).next().unwrap_or("");
        match name {
            "w:t" if !tag.ends_with('/') => {
                let end = rest.find("</w:t>").unwrap_or(rest.len());
                decode_entities(&rest[..end], &mut out);
                rest = &rest[end..];
            }
            "w:tab" => out.push('\t'),
            "w:br" | "w:cr" => out.push('\n'),
            "" if tag == "/w:p" => out.push('\n'),
            _ => {}
        }
    }
    Ok(out.trim_end().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_keeps_text_and_lines() {
        let html = "<html><head><style>p{}</style><script>var x = '<p>';</script></head><body><h1>Title</h1>\
                    <p>One &amp; two&nbsp;three\n   words.</p><!-- gone --><ul><li>a</li><li>b &#x263A;</li></ul></body></html>";
        assert_eq!(html_text(html), "Title\n\nOne & two\u{a0}three words.\na\nb \u{263a}");
    }

    #[test]
    fn entities_decode() {
        let mut out = String::new();
        decode_entities("a &lt;b&gt; &#65; &unknown; & c", &mut out);
        assert_eq!(out, "a <b> A &unknown; & c");
    }
}
