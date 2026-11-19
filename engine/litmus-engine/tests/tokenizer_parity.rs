//! The tokenizer gives exactly the ids of Hugging Face `tokenizers` 0.22 (`scripts/tokenizer_fixture.py`), and
//! streaming the standard reading in pieces gives exactly the ids of reading it at once.
//!
//! `LITMUS_TOKENIZER_DIR` names a folder with `tokenizer.json` and `tokenizer_fixture.json`.
#![cfg(feature = "parity")]

use std::path::PathBuf;

use litmus_engine::text;
use litmus_engine::tokenizer::Tokenizer;

fn folder() -> PathBuf {
    PathBuf::from(std::env::var("LITMUS_TOKENIZER_DIR").expect("set LITMUS_TOKENIZER_DIR (engine/README.md, Tests)"))
}

fn load() -> (Tokenizer, Vec<(String, Vec<u32>)>) {
    let folder = folder();
    let tokenizer = Tokenizer::from_json(&std::fs::read(folder.join("tokenizer.json")).unwrap()).unwrap();
    let records: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(folder.join("tokenizer_fixture.json")).unwrap()).unwrap();
    let cases = records
        .into_iter()
        .map(|record| {
            let text = record["text"].as_str().unwrap().to_owned();
            let ids = record["ids"].as_array().unwrap().iter().map(|id| id.as_u64().unwrap() as u32).collect();
            (text, ids)
        })
        .collect();
    (tokenizer, cases)
}

#[test]
fn ids_equal_hugging_face_tokenizers() {
    let (tokenizer, cases) = load();
    assert!(cases.len() > 400, "the fixture holds the full set");
    let mut failures = Vec::new();
    for (index, (text, want)) in cases.iter().enumerate() {
        let got = tokenizer.encode(text);
        if &got != want {
            let first = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            failures.push(format!("case {index} ({} chars): first difference at token {first}: got {:?} want {:?}; text {:?}",
                text.chars().count(), &got[first..(first + 6).min(got.len())], &want[first..(first + 6).min(want.len())],
                text.chars().take(80).collect::<String>()));
        }
    }
    assert!(failures.is_empty(), "{} of {} texts differ:\n{}", failures.len(), cases.len(), failures.join("\n"));
}

#[test]
fn streaming_pieces_equal_whole_reading() {
    let (tokenizer, cases) = load();
    // Long texts made of the fixture's texts, so pieces are cut at many places.
    let joined: Vec<String> = (0..4)
        .map(|offset| cases.iter().skip(offset).step_by(4).map(|(text, _)| text.as_str()).collect::<Vec<_>>().join(" \n"))
        .collect();
    for text in joined.iter().chain(cases.iter().map(|(text, _)| text)) {
        let whole = tokenizer.encode(&text::standard_reading(text));
        let mut streamed = Vec::new();
        let mut last_end = 0;
        tokenizer.stream_standard(text, &mut |token| {
            assert!(token.start <= token.end && token.end <= text.len(), "offsets inside the text");
            assert!(token.start >= last_end.min(token.start), "offsets ascend");
            last_end = token.end;
            streamed.push(token.id)
        }, &mut |_| true);
        assert_eq!(streamed, whole, "streamed ids differ for a text of {} bytes", text.len());
    }
}
