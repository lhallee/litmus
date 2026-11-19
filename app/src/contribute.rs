//! The contribute site's API, as the contribute build of the page uses it (`../contribute/README.md`, Endpoints, and
//! "What the contribute page must send and show"): `GET /api/terms` for the notice, the consent box's label and their
//! version; `POST /api/score` with `{"text", "terms_version", "consent": true, "annotation"?}`, sent only while the
//! visitor's box is ticked; `POST /api/contributions/{id}/annotation` with `{"annotation"}`; and `POST /api/removals`
//! with `{"text"}`, open to anyone. The request and response shapes and the served Markdown's reading are plain data
//! and tested natively; the calls themselves (`fetch`) exist only in the browser build with the `contribute` feature.

use litmus_engine::DocumentReading;
use serde::{Deserialize, Serialize};

pub const TERMS_PATH: &str = "/api/terms";
pub const SCORE_PATH: &str = "/api/score";
pub const REMOVALS_PATH: &str = "/api/removals";
/// The words of the notice that name the removal form; the page shows them as a link that opens it.
pub const REMOVAL_FORM: &str = "removal form";
/// The scorer's error when the site refuses a text because its terms changed (409); the page has read the new terms
/// by then, shown them, and unticked the box, and says so beside it.
pub const TERMS_CHANGED: &str = "the site's terms changed";

/// A visitor's label of a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Annotation {
    Ai,
    Mixed,
    Human,
}

impl Annotation {
    pub const ALL: [Annotation; 3] = [Annotation::Ai, Annotation::Mixed, Annotation::Human];

    /// The words on the label's button.
    pub fn button(self) -> &'static str {
        match self {
            Annotation::Ai => "AI",
            Annotation::Mixed => "Mixed",
            Annotation::Human => "Not AI",
        }
    }
}

/// The body of `POST /api/score`: the text, the version of the terms the visitor agreed to, `consent` (always the
/// JSON `true`: the page builds a request only while the consent box is ticked, and the server refuses one without
/// it with a 400), and the visitor's label when one was chosen before checking.
#[derive(Debug, Serialize)]
pub struct ScoreRequest<'a> {
    pub text: &'a str,
    pub terms_version: &'a str,
    pub consent: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotation: Option<Annotation>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ScoreResponse {
    pub id: String,
    /// The engine's document reading (the server adds `label`, which the app recomputes from the set).
    pub reading: DocumentReading,
    pub model_version: String,
    pub terms_version: String,
    pub annotation: Option<Annotation>,
    pub annotation_window_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Links {
    pub terms_of_use: String,
    pub privacy_notice: String,
    pub private_page: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Limits {
    pub text_max_bytes: usize,
    pub text_max_tokens: usize,
    pub annotation_window_seconds: u64,
}

/// The notice shown above the consent box and the Check button, the box's label, their version, and their status
/// (`draft` or `approved`). `notice` and `consent_label` are Markdown, shown as served.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Terms {
    pub version: String,
    pub status: String,
    pub notice: String,
    pub consent_label: String,
    pub links: Links,
    pub limits: Limits,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: String,
}

#[derive(Deserialize)]
struct AnnotationResponse {
    id: String,
    annotation: Option<Annotation>,
}

#[derive(Deserialize)]
struct RemovalResponse {
    content_hash: String,
    removal: String,
}

/// A run of the served Markdown as the page shows it: words (bold or not), or a link.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Span {
    Text { text: String, bold: bool },
    Link { text: String, url: String },
}

/// The served Markdown as paragraphs of spans. A blank line parts paragraphs and a single line break is a space;
/// `**` turns bold on and off; `[words](url)` is a link; a bare `https://` or `http://` address is a link to itself,
/// with any punctuation that ends a sentence after it kept as text. Anything else, brackets included, stays as it is.
pub fn paragraphs(markdown: &str) -> Vec<Vec<Span>> {
    markdown
        .replace("\r\n", "\n")
        .split("\n\n")
        .map(|paragraph| paragraph.split('\n').map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>().join(" "))
        .filter(|paragraph| !paragraph.is_empty())
        .map(|paragraph| spans(&paragraph))
        .collect()
}

fn spans(paragraph: &str) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let mut words = String::new();
    let mut bold = false;
    let mut rest = paragraph;
    let flush = |out: &mut Vec<Span>, words: &mut String, bold: bool| {
        if words.is_empty() {
            return;
        }
        match out.last_mut() {
            Some(Span::Text { text, bold: was }) if *was == bold => text.push_str(words),
            _ => out.push(Span::Text { text: words.clone(), bold }),
        }
        words.clear();
    };
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("**") {
            flush(&mut out, &mut words, bold);
            bold = !bold;
            rest = after;
            continue;
        }
        if let Some(after) = rest.strip_prefix('[') {
            // `[words](url)`, where the words hold no bracket of their own.
            let link = after
                .find("](")
                .filter(|&close| !after[..close].contains('['))
                .and_then(|close| after[close + 2..].find(')').map(|end| (close, close + 2 + end)));
            if let Some((close, end)) = link {
                flush(&mut out, &mut words, bold);
                out.push(Span::Link { text: after[..close].replace("**", ""), url: after[close + 2..end].to_owned() });
                rest = &after[end + 1..];
                continue;
            }
        }
        let at_word_start = words.is_empty() || words.ends_with(|c: char| c.is_whitespace() || c == '(');
        if at_word_start && (rest.starts_with("https://") || rest.starts_with("http://")) {
            let length = rest.find(char::is_whitespace).unwrap_or(rest.len());
            let url = rest[..length].trim_end_matches(['.', ',', ':', ';', '!', '?', ')']);
            flush(&mut out, &mut words, bold);
            out.push(Span::Link { text: url.to_owned(), url: url.to_owned() });
            rest = &rest[url.len()..];
            continue;
        }
        let next = rest.chars().next().expect("rest is not empty");
        words.push(next);
        rest = &rest[next.len_utf8()..];
    }
    flush(&mut out, &mut words, bold);
    out
}

/// The served Markdown in plain words (the consent box's accessible label): emphasis marks dropped, each link its
/// words, paragraphs parted by a blank line.
pub fn notice_plain(markdown: &str) -> String {
    paragraphs(markdown)
        .iter()
        .map(|spans| {
            spans
                .iter()
                .map(|span| match span {
                    Span::Text { text, .. } | Span::Link { text, .. } => text.as_str(),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Whether a ticked consent box stays ticked when the site's terms are read again: only when the same version is
/// served again. New terms, or terms that did not load, untick it.
pub fn consent_survives(old: Option<&Terms>, new: Option<&Terms>) -> bool {
    matches!((old, new), (Some(old), Some(new)) if old.version == new.version)
}

/// The body of a scoring request, made only once the visitor has ticked the consent box.
pub fn score_body(text: &str, terms_version: &str, annotation: Option<Annotation>) -> String {
    serde_json::to_string(&ScoreRequest { text, terms_version, consent: true, annotation }).expect("a request serializes")
}

/// The path of a contribution's annotation; ids are 36 lowercase hex digits (12 of time, 24 random).
pub fn annotation_path(id: &str) -> Result<String, String> {
    if id.len() == 36 && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Ok(format!("/api/contributions/{id}/annotation"))
    } else {
        Err(format!("{id:?} is not a contribution id"))
    }
}

/// The body of an annotation request; `None` clears the label.
pub fn annotation_body(annotation: Option<Annotation>) -> String {
    serde_json::json!({ "annotation": annotation }).to_string()
}

/// The body of a removal request.
pub fn removal_body(text: &str) -> String {
    serde_json::json!({ "text": text }).to_string()
}

/// What the server said went wrong, in words for the visitor.
pub fn error_message(status: u16, body: &str) -> String {
    let said = serde_json::from_str::<ErrorBody>(body).map(|e| e.error).ok();
    let fallback = match status {
        400 => "The site needs your agreement first: tick the box above the Check button.",
        409 => "The terms have changed since the page loaded: read the new ones and tick the box again.",
        410 => "The time to label this text has passed.",
        413 => "The text is too long for the site.",
        429 => "Too many requests from here; wait a minute.",
        503 => "The site has reached today's limit; try again tomorrow, or use the private page.",
        _ => "The site could not answer.",
    };
    format!("{} (HTTP {status})", said.unwrap_or_else(|| fallback.to_owned()))
}

pub fn parse_terms(status: u16, body: &str) -> Result<Terms, String> {
    if status != 200 {
        return Err(error_message(status, body));
    }
    serde_json::from_str(body).map_err(|e| format!("the terms could not be read: {e}"))
}

pub fn parse_score(status: u16, body: &str) -> Result<ScoreResponse, String> {
    if status != 200 {
        return Err(error_message(status, body));
    }
    serde_json::from_str(body).map_err(|e| format!("the reading could not be read: {e}"))
}

/// The label the server stored (202 Accepted).
pub fn parse_annotation(status: u16, body: &str, id: &str) -> Result<Option<Annotation>, String> {
    if status != 202 {
        return Err(error_message(status, body));
    }
    let response: AnnotationResponse = serde_json::from_str(body).map_err(|e| format!("the answer could not be read: {e}"))?;
    if response.id != id {
        return Err(format!("the site answered for contribution {}, not {id}", response.id));
    }
    Ok(response.annotation)
}

/// A removal the server accepted (202), checked against the text sent: the server names the SHA-256 of the text it
/// received, which must be the page's own hash of it. Returns that hash.
pub fn parse_removal(status: u16, body: &str, text: &str) -> Result<String, String> {
    if status != 202 {
        return Err(error_message(status, body));
    }
    let response: RemovalResponse = serde_json::from_str(body).map_err(|e| format!("the answer could not be read: {e}"))?;
    let sent = litmus_files::cache::sha256(text.as_bytes());
    if response.removal != "accepted" || response.content_hash != sent {
        return Err(format!("the site answered {:?} for text {}, but this page sent text {sent}", response.removal, response.content_hash));
    }
    Ok(response.content_hash)
}

/// The calls, in the browser only.
#[cfg(all(target_arch = "wasm32", feature = "contribute"))]
pub mod fetch {
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;

    fn text(error: JsValue) -> String {
        error.as_string().unwrap_or_else(|| format!("{error:?}"))
    }

    /// A same-origin request with a JSON body (or none), answered with its status and body text.
    pub async fn call(method: &str, path: &str, body: Option<&str>) -> Result<(u16, String), String> {
        let init = web_sys::RequestInit::new();
        init.set_method(method);
        init.set_mode(web_sys::RequestMode::SameOrigin);
        if let Some(body) = body {
            init.set_body(&JsValue::from_str(body));
        }
        let request = web_sys::Request::new_with_str_and_init(path, &init).map_err(text)?;
        if body.is_some() {
            request.headers().set("Content-Type", "application/json").map_err(text)?;
        }
        let window = web_sys::window().ok_or("no window")?;
        let response: web_sys::Response = JsFuture::from(window.fetch_with_request(&request)).await.map_err(text)?.dyn_into().map_err(text)?;
        let status = response.status();
        let body = JsFuture::from(response.text().map_err(text)?).await.map_err(text)?.as_string().unwrap_or_default();
        Ok((status, body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const READING: &str = include_str!("../tests/reading_small.json");

    #[test]
    fn score_requests_match_the_server() {
        let body: serde_json::Value = serde_json::from_str(&score_body("Some text.", "2026-11-17", None)).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"text": "Some text.", "terms_version": "2026-11-17", "consent": true}),
            "consent is the JSON true, and there is no annotation key when there is none"
        );
        let body: serde_json::Value = serde_json::from_str(&score_body("x", "v1", Some(Annotation::Human))).unwrap();
        assert_eq!(body["annotation"], "human");
        assert_eq!(body["consent"], serde_json::Value::Bool(true));
        assert_eq!(annotation_body(Some(Annotation::Ai)), r#"{"annotation":"ai"}"#);
        assert_eq!(annotation_body(None), r#"{"annotation":null}"#);
        assert_eq!(removal_body("A \"quoted\" text."), r#"{"text":"A \"quoted\" text."}"#);
    }

    #[test]
    fn score_responses_parse_with_the_servers_label() {
        let mut reading: serde_json::Value = serde_json::from_str(READING).unwrap();
        reading["label"] = "unlike reference".into(); // the binding adds it
        let id = "0192a0b0c0d0e0f0a1b2c3d4e5f6a7b8c9d0";
        let body = serde_json::json!({"id": id, "reading": reading, "model_version": "sha256:0123456789abcdef",
            "terms_version": "v1", "annotation": null, "annotation_window_seconds": 1800}).to_string();
        let response = parse_score(200, &body).unwrap();
        assert_eq!(response.id, id);
        assert_eq!(response.reading.n_tokens, 14);
        assert_eq!(response.annotation, None);
        assert_eq!(response.annotation_window_seconds, 1800);
    }

    #[test]
    fn errors_carry_the_servers_words() {
        assert_eq!(error_message(409, r#"{"error": "The terms have changed."}"#), "The terms have changed. (HTTP 409)");
        assert!(parse_score(400, "").unwrap_err().contains("tick the box"));
        assert!(parse_score(413, "not json").unwrap_err().contains("too long"));
        assert!(parse_score(200, "{}").unwrap_err().contains("could not be read"));
    }

    #[test]
    fn terms_and_annotations_parse() {
        let body = r#"{"version": "2026-11-17.1", "status": "draft", "notice": "The text you check here is sent to our server.",
            "consent_label": "I agree to the [Terms of Use](/legal/terms-of-use) and the [Privacy Notice](/legal/privacy-notice), and I am 18 or older.",
            "links": {"terms_of_use": "/legal/terms-of-use", "privacy_notice": "/legal/privacy-notice", "private_page": "https://example.github.io/litmus/app/"},
            "limits": {"text_max_bytes": 1048576, "text_max_tokens": 131072, "annotation_window_seconds": 1800}}"#;
        let terms = parse_terms(200, body).unwrap();
        assert_eq!(terms.version, "2026-11-17.1");
        assert_eq!(terms.limits.text_max_tokens, 131_072);
        assert_eq!(notice_plain(&terms.consent_label), "I agree to the Terms of Use and the Privacy Notice, and I am 18 or older.");
        assert!(parse_terms(200, &body.replace("\"consent_label\"", "\"other\"")).is_err(), "terms without a consent label are refused");
        let newer = Terms { version: "2026-11-18.1".into(), ..terms.clone() };
        assert!(consent_survives(Some(&terms), Some(&terms.clone())), "the same terms read again keep the box ticked");
        assert!(!consent_survives(Some(&terms), Some(&newer)), "new terms untick it");
        assert!(!consent_survives(Some(&terms), None) && !consent_survives(None, Some(&terms)), "terms that did not load untick it");
        let id = "0192a0b0c0d0e0f0a1b2c3d4e5f6a7b8c9d0";
        assert_eq!(annotation_path(id).unwrap(), format!("/api/contributions/{id}/annotation"));
        assert!(annotation_path("../../etc").is_err());
        let answer = format!(r#"{{"id": "{id}", "annotation": "mixed"}}"#);
        assert_eq!(parse_annotation(202, &answer, id).unwrap(), Some(Annotation::Mixed));
        assert!(parse_annotation(410, r#"{"error": "Too late."}"#, id).unwrap_err().starts_with("Too late."));
        assert!(parse_annotation(202, &answer, "0192a0b0c0d0e0f0a1b2c3d4e5f6a7b8c9d1").is_err());
    }

    #[test]
    fn removals_are_checked_against_the_text_sent() {
        let text = "Some text.";
        let hash = litmus_files::cache::sha256(text.as_bytes());
        let answer = format!(r#"{{"content_hash": "{hash}", "removal": "accepted"}}"#);
        assert_eq!(parse_removal(202, &answer, text).unwrap(), hash);
        assert!(parse_removal(202, &answer, "Some text").unwrap_err().contains("this page sent"), "a different text's hash is refused");
        assert!(parse_removal(429, r#"{"error": "Too many removals from here; wait a few minutes."}"#, text).unwrap_err().starts_with("Too many removals"));
    }

    #[test]
    fn notice_shows_without_markdown() {
        let notice = "By pressing **Check** you agree to the [Terms of Use](/legal/terms-of-use) and the [Privacy Notice](/legal/privacy-notice). Stored.";
        assert_eq!(notice_plain(notice), "By pressing Check you agree to the Terms of Use and the Privacy Notice. Stored.");
        assert_eq!(notice_plain("A [bracket] stays, [and [a link](/x) inside]."), "A [bracket] stays, [and a link inside].");
        assert_eq!(notice_plain("Open [unclosed"), "Open [unclosed");
        assert_eq!(notice_plain("Use https://lhallee.github.io/litmus/: there"), "Use https://lhallee.github.io/litmus/: there");
    }

    #[test]
    fn the_notice_keeps_its_paragraphs_bold_and_links() {
        let notice = "First line\nwraps here.\n\n**Please do not paste sensitive or personal information.** You are responsible.\n\n\
                      We recommend https://lhallee.github.io/litmus/app/: there it stays. See the [Terms](/legal/terms-of-use).";
        let shown = paragraphs(notice);
        assert_eq!(shown.len(), 3);
        assert_eq!(shown[0], vec![Span::Text { text: "First line wraps here.".into(), bold: false }]);
        assert_eq!(
            shown[1],
            vec![
                Span::Text { text: "Please do not paste sensitive or personal information.".into(), bold: true },
                Span::Text { text: " You are responsible.".into(), bold: false },
            ]
        );
        let url = "https://lhallee.github.io/litmus/app/";
        assert_eq!(
            shown[2],
            vec![
                Span::Text { text: "We recommend ".into(), bold: false },
                Span::Link { text: url.into(), url: url.into() },
                Span::Text { text: ": there it stays. See the ".into(), bold: false },
                Span::Link { text: "Terms".into(), url: "/legal/terms-of-use".into() },
                Span::Text { text: ".".into(), bold: false },
            ]
        );
    }

    #[test]
    fn the_served_notice_reads_as_written() {
        // The repository's notice, with its private page filled in as the server does (contribute/legal.py).
        let markdown = include_str!("../../contribute/legal/NOTICE.md");
        let body = markdown.replace("\r\n", "\n").splitn(3, "---\n").nth(2).expect("front matter, then the body").replace("{private_page}", "https://lhallee.github.io/litmus/app/");
        let shown = paragraphs(&body);
        let bold = shown.iter().flatten().find_map(|span| match span {
            Span::Text { text, bold: true } => Some(text.as_str()),
            _ => None,
        });
        assert_eq!(bold, Some("Please do not paste sensitive or personal information, yours or anyone else's."));
        assert!(matches!(&shown[1][0], Span::Text { bold: true, .. }), "the second paragraph opens with the bold sentence");
        assert!(notice_plain(&body).contains(REMOVAL_FORM), "the notice names the removal form, which the page links to its form");
        assert!(shown.iter().flatten().any(|span| matches!(span, Span::Link { url, .. } if url == "https://lhallee.github.io/litmus/app/")));
    }
}
