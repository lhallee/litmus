//! The standard reading (SPEC.md, section 1.1) with a map back to the original text, and the cut points at which
//! a long text can be read in pieces with exactly the token ids of reading it at once.
//!
//! The reading, in order: Unicode NFC; `\r\n?` to `\n`; delete the control and format characters of SPEC.md (tab,
//! newline and the zero-width joiner and non-joiner stay); delete runs of tabs and space separators (`Zs`) before
//! each `\n` and at the end; delete trailing white space and newlines at the end; delete blank lines at the start;
//! replace runs of four or more `\n` by three.

use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};

/// A piece of text after the standard reading, with the original byte offset of every byte.
#[derive(Debug, Default, Clone)]
pub struct Cleaned {
    pub text: String,
    /// `origin[i]` is the original byte offset of cleaned byte `i`; `origin[text.len()]` is the piece's end.
    pub origin: Vec<u32>,
}

/// Whether a character was assigned in Unicode 14.0, the version of `tokenizers`' tables (and of Python 3.11's
/// `unicodedata`, which wrote the fixtures).
#[inline]
pub fn assigned_in_unicode14(c: char) -> bool {
    let code = c as u32;
    if code < 0x378 {
        return true;
    }
    let ranges = &crate::unicode14::ASSIGNED;
    let index = ranges.partition_point(|&(_, end)| end < code);
    index < ranges.len() && ranges[index].0 <= code
}

/// NFC with Unicode 14.0's character repertoire: a character assigned later is unassigned there, so a starter that
/// neither decomposes nor composes, and it passes through; the text between such characters normalizes on its own
/// (normalization data of assigned characters never changes between versions).
pub fn nfc14(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if !assigned_in_unicode14(c) {
            out.extend(text[start..i].nfc());
            out.push(c);
            start = i + c.len_utf8();
        }
    }
    out.extend(text[start..].nfc());
    out
}

/// Whether NFC (Unicode 14.0) leaves `text` unchanged, quickly.
pub fn is_nfc14_quick(text: &str) -> bool {
    text.is_ascii() || (is_nfc_quick(text.chars()) == IsNormalized::Yes && text.chars().all(assigned_in_unicode14))
}

/// Deleted by the reading: U+0000-0008, 000B, 000C, 000E-001F, 007F-009F, 00AD, 180E, 200B, 200E, 200F,
/// 202A-202E, 2060-2064, 2066-206F, FEFF, FFF9-FFFB.
fn deleted(character: char) -> bool {
    matches!(
        character as u32,
        0x00..=0x08
            | 0x0b
            | 0x0c
            | 0x0e..=0x1f
            | 0x7f..=0x9f
            | 0xad
            | 0x180e
            | 0x200b
            | 0x200e
            | 0x200f
            | 0x202a..=0x202e
            | 0x2060..=0x2064
            | 0x2066..=0x206f
            | 0xfeff
            | 0xfff9..=0xfffb
    )
}

/// Tab or a space separator (category `Zs`): what the reading deletes before a newline and at the end.
fn tab_or_space_separator(character: char) -> bool {
    matches!(character as u32, 0x09 | 0x20 | 0xa0 | 0x1680 | 0x2000..=0x200a | 0x202f | 0x205f | 0x3000)
}

/// NFC by runs that start at an ASCII character: no composition or reordering crosses an ASCII character, so the
/// runs normalize independently. A run already in NFC keeps exact offsets; a changed run maps every output
/// character to the run's start.
fn nfc_with_offsets(piece: &str, base: usize, out: &mut Vec<(char, u32)>) {
    if is_nfc14_quick(piece) {
        out.extend(piece.char_indices().map(|(i, c)| (c, (base + i) as u32)));
        return;
    }
    let bytes = piece.as_bytes();
    let mut start = 0;
    while start < piece.len() {
        // The run: one character, then every non-ASCII character up to the next ASCII byte.
        let first_len = piece[start..].chars().next().map_or(1, char::len_utf8);
        let mut end = start + first_len;
        while end < bytes.len() && bytes[end] >= 0x80 {
            end += 1;
        }
        let run = &piece[start..end];
        if is_nfc14_quick(run) {
            out.extend(run.char_indices().map(|(i, c)| (c, (base + start + i) as u32)));
        } else {
            out.extend(nfc14(run).chars().map(|c| (c, (base + start) as u32)));
        }
        start = end;
    }
}

/// The standard reading of `text[start..end]`: `first` when the piece starts the text (leading blank lines go),
/// `last` when it ends it (trailing white space goes).
pub fn clean(text: &str, start: usize, end: usize, first: bool, last: bool, out: &mut Cleaned) {
    let mut chars: Vec<(char, u32)> = Vec::with_capacity(end - start);
    nfc_with_offsets(&text[start..end], start, &mut chars);

    // `\r\n?` to `\n`, then the deletions, in one pass.
    let mut stage: Vec<(char, u32)> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let (c, offset) = chars[i];
        if c == '\r' {
            stage.push(('\n', offset));
            i += if i + 1 < chars.len() && chars[i + 1].0 == '\n' { 2 } else { 1 };
            continue;
        }
        if !deleted(c) {
            stage.push((c, offset));
        }
        i += 1;
    }

    // Tabs and space separators before each `\n`, and at the end of the text.
    let mut trimmed: Vec<(char, u32)> = Vec::with_capacity(stage.len());
    for &(c, offset) in &stage {
        if c == '\n' {
            while trimmed.last().is_some_and(|&(previous, _)| tab_or_space_separator(previous)) {
                trimmed.pop();
            }
        }
        trimmed.push((c, offset));
    }
    if last {
        while trimmed.last().is_some_and(|&(previous, _)| tab_or_space_separator(previous)) {
            trimmed.pop();
        }
        // `[\t\n\f\r ]+$`
        while trimmed.last().is_some_and(|&(previous, _)| matches!(previous, '\t' | '\n' | '\u{c}' | '\r' | ' ')) {
            trimmed.pop();
        }
    }

    // `^(?:[\t Zs]*\n)+`: whole blank lines at the start of the text.
    let mut skip = 0;
    if first {
        let mut at = 0;
        loop {
            while at < trimmed.len() && tab_or_space_separator(trimmed[at].0) {
                at += 1;
            }
            if at < trimmed.len() && trimmed[at].0 == '\n' {
                at += 1;
                skip = at;
            } else {
                break;
            }
        }
    }

    // Runs of four or more `\n` become three.
    out.text.clear();
    out.origin.clear();
    let mut run = 0;
    for &(c, offset) in &trimmed[skip..] {
        if c == '\n' {
            run += 1;
            if run > 3 {
                continue;
            }
        } else {
            run = 0;
        }
        let mut buffer = [0u8; 4];
        let encoded = c.encode_utf8(&mut buffer);
        out.text.push_str(encoded);
        out.origin.extend(std::iter::repeat_n(offset, encoded.len()));
    }
    out.origin.push(end as u32);
}

/// The standard reading of a whole text.
pub fn standard_reading(text: &str) -> String {
    let mut out = Cleaned::default();
    clean(text, 0, text.len(), true, true, &mut out);
    out.text
}

/// Whether the text may be cut before byte `at`, so that the standard reading and the token ids of the two sides,
/// read apart, equal those of the whole. Two patterns qualify (the caller checks that no added token contains a
/// space between other characters or a line feed):
/// - an ASCII letter or digit, one space, an ASCII letter or digit: the space starts the next word;
/// - a character that is neither white space nor deleted, one `\n`, then another such character: the line feed is a
///   token of its own and touches no run that the reading or the pre-tokenizer would join.
pub fn is_safe_cut(text: &str, at: usize) -> bool {
    let bytes = text.as_bytes();
    if at < 2 || at + 1 > bytes.len() || !text.is_char_boundary(at) {
        return false;
    }
    if bytes[at] == b' ' {
        return bytes[at - 1].is_ascii_alphanumeric() && at + 1 < bytes.len() && bytes[at + 1].is_ascii_alphanumeric();
    }
    if bytes[at - 1] == b'\n' {
        let solid = |c: Option<char>| c.is_some_and(|c| !c.is_whitespace() && !deleted(c) && c != '\r');
        return solid(text[..at - 1].chars().next_back()) && solid(text[at..].chars().next());
    }
    false
}

/// The first safe cut at or after `from` that `also` accepts, or the end of the text.
pub fn next_cut(text: &str, from: usize, also: &dyn Fn(usize) -> bool) -> usize {
    let bytes = text.as_bytes();
    let mut at = from.max(2);
    while at < bytes.len() {
        if (bytes[at] == b' ' || bytes[at - 1] == b'\n') && is_safe_cut(text, at) && also(at) {
            return at;
        }
        at += 1;
    }
    text.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reading_follows_the_spec() {
        let text = "Line one\r\nLine two \u{a0}\r\n\r\n\r\n\r\n\r\nAfter\u{200b} a\u{7} b\t \n\u{0085}x \t\n\n";
        assert_eq!(standard_reading(text), "Line one\nLine two\n\n\nAfter a b\nx");
        assert_eq!(standard_reading("cafe\u{301}"), "caf\u{e9}");
        assert_eq!(standard_reading("a \u{200b}\nb"), "a\nb");
        assert_eq!(standard_reading(" \t\n\u{3000}\n  x\n"), "  x");
        assert_eq!(standard_reading("x\r\ry\u{ad}\u{200d}"), "x\n\ny\u{200d}");
        assert_eq!(standard_reading("\n\u{200b}\n\n\n"), "");
        assert_eq!(standard_reading("a\n\u{200b}\n\n\nb"), "a\n\n\nb");
    }

    #[test]
    fn offsets_point_into_the_original() {
        let text = "ab\r\ncd\u{200b}e  \nf";
        let mut out = Cleaned::default();
        clean(text, 0, text.len(), true, true, &mut out);
        assert_eq!(out.text, "ab\ncde\nf");
        let origins: Vec<u32> = out.text.char_indices().map(|(i, _)| out.origin[i]).collect();
        assert_eq!(origins, vec![0, 1, 2, 4, 5, 9, 12, 13]);
        assert_eq!(*out.origin.last().unwrap(), text.len() as u32);
    }

    #[test]
    fn cuts_are_where_promised() {
        assert!(is_safe_cut("ab cd", 2));
        assert!(!is_safe_cut("ab  cd", 2));
        assert!(!is_safe_cut("a, cd", 2));
        assert!(is_safe_cut("ab\ncd", 3));
        assert!(!is_safe_cut("ab\n\ncd", 4));
        assert!(!is_safe_cut("ab \ncd", 4));
        assert!(!is_safe_cut("ab\n cd", 3));
        assert!(!is_safe_cut("ab\n\u{200b}cd", 3));
        assert!(is_safe_cut("\u{4e2d}\n\u{6587}", 4));
    }
}
