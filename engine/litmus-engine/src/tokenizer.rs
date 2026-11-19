//! ModernBERT's byte-level BPE tokenizer (`answerdotai/ModernBERT-large`, `tokenizer.json`), as Hugging Face
//! `tokenizers` 0.22 encodes with `add_special_tokens=False` (SPEC.md, section 1.2):
//!
//! 1. split out the added tokens that are matched on the raw text (`normalized: false`), leftmost-longest;
//! 2. NFC on what remains;
//! 3. split out the added tokens matched on normalized text (`normalized: true`: runs of 2 to 24 spaces, `[unusedN]`,
//!    the `|||..|||` placeholders), leftmost-longest;
//! 4. the GPT-2 pre-tokenizer pattern
//!    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, written out by hand;
//! 5. BPE over each word's bytes, lowest merge rank first, ties to the leftmost pair.
//!
//! Implemented here rather than through the `tokenizers` crate: no regex engine, no thread pool, a few hundred
//! lines, and it streams. The parity test checks it against the Python library.

use std::sync::Mutex;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use serde_json::Value;
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

use crate::Error;
use crate::text::{self, Cleaned};

/// The FxHash function (rustc's): fast for short keys, and no denial-of-service concern for a local tokenizer.
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Words cached at most; past this the cache stops growing (as in `tokenizers`).
const CACHE_CAPACITY: usize = 16_384;
/// Words longer than this many bytes are not cached.
const CACHE_WORD_BYTES: usize = 64;
/// Original bytes read per piece when streaming.
const PIECE_BYTES: usize = 8 * 1024;

/// One token: its id and the original byte range it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    pub start: usize,
    pub end: usize,
}

struct AddedToken {
    content: Vec<u8>,
    id: u32,
    /// The match also takes the white space before it (`lstrip`) or after it (`rstrip`), as in `tokenizers`.
    lstrip: bool,
    rstrip: bool,
}

/// Leftmost-longest matcher over a few patterns, bucketed by first byte.
#[derive(Default)]
struct AddedSet {
    by_first: Vec<Vec<usize>>, // 256 buckets of indices into `tokens`, longest first
    tokens: Vec<AddedToken>,
}

impl AddedSet {
    fn new(tokens: Vec<AddedToken>) -> Self {
        let mut by_first = vec![Vec::new(); 256];
        for (index, token) in tokens.iter().enumerate() {
            by_first[token.content[0] as usize].push(index);
        }
        for bucket in &mut by_first {
            bucket.sort_by_key(|&index| std::cmp::Reverse(tokens[index].content.len()));
        }
        Self { by_first, tokens }
    }

    /// The leftmost match in `text[from..]`, longest at that position, widened over white space by `lstrip` (not
    /// before `from`, the end of the previous match) and `rstrip`: `(start, end, id)`.
    fn find(&self, text: &str, from: usize) -> Option<(usize, usize, u32)> {
        if self.tokens.is_empty() {
            return None;
        }
        let bytes = text.as_bytes();
        for start in from..bytes.len() {
            for &index in &self.by_first[bytes[start] as usize] {
                let token = &self.tokens[index];
                if bytes[start..].starts_with(&token.content) {
                    let mut begin = start;
                    let mut end = start + token.content.len();
                    if token.lstrip {
                        begin = text[from..start]
                            .char_indices()
                            .rev()
                            .find(|&(_, c)| !c.is_whitespace())
                            .map_or(from, |(i, c)| from + i + c.len_utf8());
                    }
                    if token.rstrip {
                        end = text[end..].char_indices().find(|&(_, c)| !c.is_whitespace()).map_or(text.len(), |(i, _)| end + i);
                    }
                    return Some((begin, end, token.id));
                }
            }
        }
        None
    }

    /// Whether a piece boundary at `at` could change a match: an `lstrip` token right after it, or an `rstrip`
    /// token ending just before the white space before it.
    fn touches_strip(&self, text: &str, at: usize) -> bool {
        let before = text[..at].trim_end_matches(char::is_whitespace);
        self.tokens.iter().any(|token| {
            (token.lstrip && text.as_bytes()[at..].starts_with(&token.content))
                || (token.rstrip && before.as_bytes().ends_with(&token.content))
        })
    }
}

/// The tokenizer: vocabulary, merges and added tokens.
pub struct Tokenizer {
    vocab: FxMap<Vec<u8>, u32>,
    lengths: Vec<u16>, // byte length of each vocabulary id
    byte_ids: [u32; 256],
    merges: FxMap<u64, (u32, u32)>, // (left id, right id) -> (rank, merged id)
    raw_added: AddedSet,
    normalized_added: AddedSet,
    cuts_allowed: bool,
    cache: Mutex<FxMap<Box<[u8]>, Box<[u32]>>>, // a Mutex so an engine can be shared between threads
}

/// GPT-2's map from bytes to printable characters.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for byte in 0..256u32 {
        let printable = (0x21..=0x7e).contains(&byte) || (0xa1..=0xac).contains(&byte) || (0xae..=0xff).contains(&byte);
        table[byte as usize] = if printable {
            char::from_u32(byte).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
    }
    table
}

fn json_error(what: impl std::fmt::Display) -> Error {
    Error::Load(format!("tokenizer.json: {what}"))
}

impl Tokenizer {
    /// Load `tokenizer.json`. Refuses anything this implementation does not reproduce exactly.
    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        let root: Value = serde_json::from_slice(bytes).map_err(json_error)?;
        let expect = |path: &str, value: Option<&Value>, want: &Value| -> Result<(), Error> {
            if value == Some(want) { Ok(()) } else { Err(json_error(format!("{path} is {value:?}, expected {want}"))) }
        };
        expect("normalizer.type", root.pointer("/normalizer/type"), &Value::from("NFC"))?;
        expect("pre_tokenizer.type", root.pointer("/pre_tokenizer/type"), &Value::from("ByteLevel"))?;
        expect("pre_tokenizer.add_prefix_space", root.pointer("/pre_tokenizer/add_prefix_space"), &Value::from(false))?;
        expect("pre_tokenizer.use_regex", root.pointer("/pre_tokenizer/use_regex"), &Value::from(true))?;
        expect("model.type", root.pointer("/model/type"), &Value::from("BPE"))?;
        for (key, want) in [("dropout", Value::Null), ("continuing_subword_prefix", Value::Null), ("end_of_word_suffix", Value::Null)] {
            let value = root.pointer(&format!("/model/{key}"));
            if !(value.is_none() || value == Some(&want)) {
                return Err(json_error(format!("model.{key} is {value:?}, expected null")));
            }
        }
        for key in ["byte_fallback", "ignore_merges"] {
            let value = root.pointer(&format!("/model/{key}"));
            if !(value.is_none() || value == Some(&Value::from(false))) {
                return Err(json_error(format!("model.{key} is {value:?}, expected false")));
            }
        }

        let to_byte: HashMap<char, u8> = bytes_to_unicode().iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let decode = |token: &str| -> Result<Vec<u8>, Error> {
            token.chars().map(|c| to_byte.get(&c).copied().ok_or_else(|| json_error(format!("vocab entry {token:?} is not byte-level")))).collect()
        };

        let added_ids: std::collections::HashSet<u64> = root
            .get("added_tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| json_error("no added_tokens"))?
            .iter()
            .filter_map(|token| token.get("id").and_then(Value::as_u64))
            .collect();
        let vocab_json = root.pointer("/model/vocab").and_then(Value::as_object).ok_or_else(|| json_error("no model.vocab"))?;
        let mut vocab: FxMap<Vec<u8>, u32> = FxMap::default();
        vocab.reserve(vocab_json.len());
        for (token, id) in vocab_json {
            let id = id.as_u64().ok_or_else(|| json_error("vocab ids must be integers"))?;
            // Vocabulary entries that are added tokens (the space runs, the placeholders) are matched as added
            // tokens, never reached by merges; some are not byte-level strings at all.
            if added_ids.contains(&id) {
                continue;
            }
            if vocab.insert(decode(token)?, id as u32).is_some() {
                return Err(json_error(format!("two vocab entries decode to the bytes of {token:?}")));
            }
        }
        // Every byte that valid UTF-8 can hold has an entry; 0xC0, 0xC1 and 0xF5 to 0xFF never occur in a `&str`.
        let mut byte_ids = [NONE; 256];
        for byte in 0..256usize {
            match vocab.get(&vec![byte as u8]) {
                Some(&id) => byte_ids[byte] = id,
                None if byte == 0xc0 || byte == 0xc1 || byte >= 0xf5 => {}
                None => return Err(json_error(format!("no vocab entry for byte {byte:#04x}"))),
            }
        }

        let merges_json = root.pointer("/model/merges").and_then(Value::as_array).ok_or_else(|| json_error("no model.merges"))?;
        let mut merges: FxMap<u64, (u32, u32)> = FxMap::default();
        merges.reserve(merges_json.len());
        for (rank, merge) in merges_json.iter().enumerate() {
            let (left, right) = match merge {
                Value::String(pair) => pair.split_once(' ').ok_or_else(|| json_error(format!("merge {pair:?}")))?,
                Value::Array(pair) if pair.len() == 2 => (
                    pair[0].as_str().ok_or_else(|| json_error("merge parts must be strings"))?,
                    pair[1].as_str().ok_or_else(|| json_error("merge parts must be strings"))?,
                ),
                other => return Err(json_error(format!("merge {other}"))),
            };
            let (left, right) = (decode(left)?, decode(right)?);
            let joined = [left.as_slice(), right.as_slice()].concat();
            let id_of = |bytes: &[u8]| vocab.get(bytes).copied().ok_or_else(|| json_error(format!("merge part {bytes:?} not in vocab")));
            let key = ((id_of(&left)? as u64) << 32) | id_of(&right)? as u64;
            merges.entry(key).or_insert((rank as u32, id_of(&joined)?));
        }

        let added_json = root.get("added_tokens").and_then(Value::as_array).ok_or_else(|| json_error("no added_tokens"))?;
        let (mut raw, mut normalized) = (Vec::new(), Vec::new());
        let mut cuts_allowed = true;
        for token in added_json {
            let field = |key: &str| token.get(key).ok_or_else(|| json_error(format!("added token without {key}")));
            if field("single_word")?.as_bool() != Some(false) {
                return Err(json_error(format!("added token {token} sets single_word, which this tokenizer does not reproduce")));
            }
            let flag = |key: &str| field(key)?.as_bool().ok_or_else(|| json_error(format!("added token {key} must be true or false")));
            let (lstrip, rstrip) = (flag("lstrip")?, flag("rstrip")?);
            let content = field("content")?.as_str().ok_or_else(|| json_error("added token content"))?.to_owned();
            let id = field("id")?.as_u64().ok_or_else(|| json_error("added token id"))? as u32;
            if content.is_empty() {
                return Err(json_error("an empty added token"));
            }
            // A piece boundary must never fall inside an added token (text::is_safe_cut).
            let bytes = content.as_bytes();
            let spaces_only = bytes.iter().all(|&b| b == b' ');
            if content.contains('\n') || (!spaces_only && content.contains(' ')) {
                cuts_allowed = false;
            }
            let entry = AddedToken { content: content.into_bytes(), id, lstrip, rstrip };
            if field("normalized")?.as_bool() == Some(true) { normalized.push(entry) } else { raw.push(entry) }
        }

        let mut lengths = vec![0u16; vocab.values().copied().max().unwrap_or(0) as usize + 1];
        for (bytes, &id) in &vocab {
            lengths[id as usize] = bytes.len() as u16;
        }

        Ok(Self {
            vocab,
            lengths,
            byte_ids,
            merges,
            raw_added: AddedSet::new(raw),
            normalized_added: AddedSet::new(normalized),
            cuts_allowed,
            cache: Mutex::new(FxMap::default()),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.raw_added.tokens.len() + self.normalized_added.tokens.len()
    }

    /// Token ids of a text as given (no standard reading), as `tokenizers` encodes it.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        self.encode_piece(text, &mut |id, _, _| ids.push(id));
        ids
    }

    /// Tokenize the standard reading of `text` in pieces, calling `sink` with each token (ids exactly those of the
    /// whole reading; byte offsets into the original `text`), and `progress(bytes read)` after each piece; a false
    /// return from `progress` stops early and returns false.
    pub fn stream_standard(
        &self,
        text: &str,
        sink: &mut dyn FnMut(Token),
        progress: &mut dyn FnMut(usize) -> bool,
    ) -> bool {
        let mut cleaned = Cleaned::default();
        let mut start = 0;
        while start < text.len() {
            let end = if self.cuts_allowed { text::next_cut(text, start + PIECE_BYTES, &|at| !self.raw_added.touches_strip(text, at) && !self.normalized_added.touches_strip(text, at)) } else { text.len() };
            text::clean(text, start, end, start == 0, end == text.len(), &mut cleaned);
            let origin = &cleaned.origin;
            self.encode_piece(&cleaned.text, &mut |id, a, b| {
                sink(Token { id, start: origin[a] as usize, end: origin[b] as usize })
            });
            start = end;
            if !progress(start) {
                return false;
            }
        }
        true
    }

    /// Steps 1 to 5 on one piece; `sink(id, start, end)` with byte offsets into `piece`.
    fn encode_piece(&self, piece: &str, sink: &mut dyn FnMut(u32, usize, usize)) {
        let bytes = piece.as_bytes();
        let mut at = 0;
        while at < bytes.len() {
            let found = self.raw_added.find(piece, at);
            let stop = found.map_or(bytes.len(), |(start, _, _)| start);
            if stop > at {
                self.encode_normalizable(&piece[at..stop], at, sink);
            }
            match found {
                Some((start, end, id)) => {
                    sink(id, start, end);
                    at = end;
                }
                None => break,
            }
        }
    }

    /// NFC, then the normalized added tokens, then words.
    fn encode_normalizable(&self, segment: &str, base: usize, sink: &mut dyn FnMut(u32, usize, usize)) {
        if text::is_nfc14_quick(segment) {
            self.encode_normalized(segment, &mut |id, a, b| sink(id, base + a, base + b));
        } else {
            // Offsets of a changed segment are coarse: the whole segment.
            let normalized = text::nfc14(segment);
            let end = base + segment.len();
            self.encode_normalized(&normalized, &mut |id, _, _| sink(id, base, end));
        }
    }

    fn encode_normalized(&self, segment: &str, sink: &mut dyn FnMut(u32, usize, usize)) {
        let bytes = segment.as_bytes();
        let mut at = 0;
        while at < bytes.len() {
            let found = self.normalized_added.find(segment, at);
            let stop = found.map_or(bytes.len(), |(start, _, _)| start);
            if stop > at {
                self.encode_words(&segment[at..stop], at, sink);
            }
            match found {
                Some((start, end, id)) => {
                    sink(id, start, end);
                    at = end;
                }
                None => break,
            }
        }
    }

    fn encode_words(&self, segment: &str, base: usize, sink: &mut dyn FnMut(u32, usize, usize)) {
        let mut at = 0;
        let mut ids = Vec::new();
        while at < segment.len() {
            let end = next_word(segment, at);
            let word = &segment.as_bytes()[at..end];
            ids.clear();
            self.bpe(word, &mut ids);
            // Byte offsets of each token inside the word, from the lengths of the merged byte strings.
            let mut offset = base + at;
            for &id in &ids {
                let length = self.token_bytes(id);
                sink(id, offset, (offset + length).min(base + end));
                offset += length;
            }
            at = end;
        }
    }

    /// Byte length of a vocabulary token (every BPE output is in the vocabulary).
    fn token_bytes(&self, id: u32) -> usize {
        self.lengths[id as usize] as usize
    }

    /// BPE of one word's bytes.
    fn bpe(&self, word: &[u8], out: &mut Vec<u32>) {
        if word.len() <= CACHE_WORD_BYTES
            && let Some(ids) = self.cache.lock().expect("the word cache is never poisoned").get(word)
        {
            out.extend_from_slice(ids);
            return;
        }
        let start = out.len();
        merge_word(word, &self.byte_ids, &self.merges, out);
        if word.len() <= CACHE_WORD_BYTES {
            let mut cache = self.cache.lock().expect("the word cache is never poisoned");
            if cache.len() < CACHE_CAPACITY {
                cache.insert(word.into(), out[start..].into());
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: u32,
    next: u32,
    alive: bool,
}

const NONE: u32 = u32::MAX;

/// Merge a word's bytes by rank: a min-heap on (rank, position), stale entries skipped, as `tokenizers`'
/// `Word::merge_all`.
fn merge_word(word: &[u8], byte_ids: &[u32; 256], merges: &FxMap<u64, (u32, u32)>, out: &mut Vec<u32>) {
    let n = word.len();
    let mut symbols: Vec<Symbol> = (0..n)
        .map(|i| Symbol {
            id: byte_ids[word[i] as usize],
            prev: if i == 0 { NONE } else { i as u32 - 1 },
            next: if i + 1 == n { NONE } else { i as u32 + 1 },
            alive: true,
        })
        .collect();
    let pair = |a: u32, b: u32| merges.get(&(((a as u64) << 32) | b as u64)).copied();
    // Max-heap of Reverse keys: lowest rank, then lowest position.
    let mut heap: BinaryHeap<std::cmp::Reverse<(u32, u32, u32)>> = BinaryHeap::with_capacity(n);
    for i in 0..n.saturating_sub(1) {
        if let Some((rank, merged)) = pair(symbols[i].id, symbols[i + 1].id) {
            heap.push(std::cmp::Reverse((rank, i as u32, merged)));
        }
    }
    while let Some(std::cmp::Reverse((rank, position, merged))) = heap.pop() {
        let left = symbols[position as usize];
        if !left.alive || left.next == NONE {
            continue;
        }
        let right_position = left.next as usize;
        let right = symbols[right_position];
        if pair(left.id, right.id) != Some((rank, merged)) {
            continue;
        }
        symbols[position as usize].id = merged;
        symbols[position as usize].next = right.next;
        symbols[right_position].alive = false;
        if right.next != NONE {
            symbols[right.next as usize].prev = position;
        }
        if left.prev != NONE
            && let Some((rank, merged)) = pair(symbols[left.prev as usize].id, merged)
        {
            heap.push(std::cmp::Reverse((rank, left.prev, merged)));
        }
        if right.next != NONE
            && let Some((rank, merged)) = pair(merged, symbols[right.next as usize].id)
        {
            heap.push(std::cmp::Reverse((rank, position, merged)));
        }
    }
    let mut i = 0u32;
    while i != NONE {
        out.push(symbols[i as usize].id);
        i = symbols[i as usize].next;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

#[inline]
fn class(c: char) -> Class {
    if c.is_ascii() {
        return match c {
            'a'..='z' | 'A'..='Z' => Class::Letter,
            '0'..='9' => Class::Number,
            ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}' => Class::Space,
            _ => Class::Other,
        };
    }
    if c.is_whitespace() {
        return Class::Space;
    }
    if !text::assigned_in_unicode14(c) {
        return Class::Other; // unassigned in tokenizers' tables
    }
    match c.general_category_group() {
        GeneralCategoryGroup::Letter => Class::Letter,
        GeneralCategoryGroup::Number => Class::Number,
        _ => Class::Other,
    }
}

/// The end of the GPT-2 pattern's match at `at` (alternatives tried in order, as the regex does).
fn next_word(text: &str, at: usize) -> usize {
    let rest = &text[at..];
    let bytes = rest.as_bytes();
    // 's 't 're 've 'm 'll 'd
    if bytes[0] == b'\'' && bytes.len() >= 2 {
        match bytes[1] {
            b's' | b't' | b'm' | b'd' => return at + 2,
            b'r' | b'v' if bytes.len() >= 3 && bytes[2] == b'e' => return at + 3,
            b'l' if bytes.len() >= 3 && bytes[2] == b'l' => return at + 3,
            _ => {}
        }
    }
    let mut chars = rest.char_indices();
    let (_, first) = chars.next().unwrap();
    let first_class = class(first);
    // ` ?X+` for X in letters, numbers, other: an optional leading space, then a run of one class.
    let (run_class, run_start) = if first == ' ' {
        match rest[1..].chars().next().map(class) {
            Some(next) if next != Class::Space => (Some(next), 1),
            _ => (None, 0),
        }
    } else if first_class != Class::Space {
        (Some(first_class), 0)
    } else {
        (None, 0)
    };
    if let Some(run_class) = run_class {
        let end = rest[run_start..].char_indices().find(|&(_, c)| class(c) != run_class).map_or(rest.len(), |(i, _)| run_start + i);
        return at + end;
    }
    // `\s+(?!\S)|\s+`: a run of white space; when a non-space follows, it gives up its last character if it has more
    // than one.
    let mut end = rest.len();
    let mut last_start = 0;
    for (i, c) in rest.char_indices() {
        if class(c) != Class::Space {
            end = i;
            break;
        }
        last_start = i;
    }
    if end == rest.len() || last_start == 0 { at + end } else { at + last_start }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < text.len() {
            let end = next_word(text, at);
            out.push(&text[at..end]);
            at = end;
        }
        out
    }

    #[test]
    fn pre_tokenizer_follows_the_pattern() {
        assert_eq!(words("Hello world"), ["Hello", " world"]);
        assert_eq!(words("it's we've they'll"), ["it", "'s", " we", "'ve", " they", "'ll"]);
        assert_eq!(words("a\n\nb"), ["a", "\n", "\n", "b"]);
        assert_eq!(words("a\t b"), ["a", "\t", " b"]);
        assert_eq!(words("x  "), ["x", "  "]);
        assert_eq!(words("12ab..."), ["12", "ab", "..."]);
        assert_eq!(words(" ..."), [" ..."]);
        assert_eq!(words("a \n b"), ["a", " \n", " b"]);
        assert_eq!(words("'x"), ["'", "x"]);
    }

    #[test]
    fn bytes_to_unicode_is_gpt2s() {
        let table = bytes_to_unicode();
        assert_eq!(table[b' ' as usize], '\u{120}');
        assert_eq!(table[b'\n' as usize], '\u{10a}');
        assert_eq!(table[b'!' as usize], '!');
        assert_eq!(table[0xad], '\u{143}');
    }
}
