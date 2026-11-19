//! The size of a text in words, characters and tokens, which the command line, `litmus mcp` and the app show beside a
//! reading. A presentation statistic: the engine neither computes nor reads it, and the reading's fields are unchanged.
//!
//! - `characters` counts Unicode scalar values (`str::chars`), so an emoji built from several scalars counts as
//!   several, and a line break of two characters (CRLF) counts as two.
//! - `characters_no_spaces` counts the scalar values that are not whitespace (`char::is_whitespace`).
//! - `words` counts maximal runs of non-whitespace (`str::split_whitespace`). A language written without spaces, such
//!   as Chinese or Japanese, counts as one word per run of characters between spaces or line breaks.
//! - `tokens` is the reading's own token count (`n_tokens`), passed in and never recounted.
//!
//! The text counted is the text as given, after a file's text is extracted and before the engine normalizes it.

use serde_json::{Value, json};

use crate::export::thousands;

/// Words, characters and tokens of one text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextCounts {
    pub words: usize,
    pub characters: usize,
    pub characters_no_spaces: usize,
    pub tokens: usize,
}

impl TextCounts {
    /// The counts of `text`, with `tokens` the reading's token count of that same text.
    pub fn of(text: &str, tokens: usize) -> Self {
        let (mut characters, mut spaces) = (0, 0);
        for c in text.chars() {
            characters += 1;
            spaces += usize::from(c.is_whitespace());
        }
        Self { words: text.split_whitespace().count(), characters, characters_no_spaces: characters - spaces, tokens }
    }

    /// The `counts` object of `litmus score --json`.
    pub fn json(&self) -> Value {
        json!({"words": self.words, "characters": self.characters, "characters_no_spaces": self.characters_no_spaces, "tokens": self.tokens})
    }

    /// The one line every surface shows, such as `1,234 words, 7,890 characters, 1,650 tokens`.
    pub fn line(&self) -> String {
        let count = |n: usize, noun: &str| format!("{} {noun}{}", thousands(n), if n == 1 { "" } else { "s" });
        format!("{}, {}, {}", count(self.words, "word"), count(self.characters, "character"), count(self.tokens, "token"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words, characters and characters without spaces of `text` (tokens fixed at 0).
    fn counted(text: &str) -> (usize, usize, usize) {
        let counts = TextCounts::of(text, 0);
        (counts.words, counts.characters, counts.characters_no_spaces)
    }

    #[test]
    fn empty_text_counts_nothing() {
        assert_eq!(TextCounts::of("", 0), TextCounts::default());
        assert_eq!(counted("   \n\t "), (0, 6, 0), "whitespace alone is characters but no words");
    }

    #[test]
    fn plain_ascii() {
        assert_eq!(counted("Hello, world!"), (2, 13, 12));
        assert_eq!(counted("one"), (1, 3, 3));
    }

    #[test]
    fn runs_of_spaces_tabs_and_newlines_are_one_separator() {
        assert_eq!(counted("a   b"), (2, 5, 2));
        assert_eq!(counted("a\tb\nc\n"), (3, 6, 3));
        assert_eq!(counted("  lead and trail  "), (3, 18, 12));
        assert_eq!(counted("a\n\n\nb"), (2, 5, 2));
    }

    #[test]
    fn crlf_is_two_characters_and_one_separator() {
        assert_eq!(counted("a\r\nb"), (2, 4, 2));
        assert_eq!(counted("one\r\ntwo\r\n"), (2, 10, 6));
    }

    #[test]
    fn a_non_breaking_space_separates_words() {
        assert_eq!(counted("a\u{a0}b"), (2, 3, 2));
        assert_eq!(counted("a\u{3000}b"), (2, 3, 2), "the ideographic space too");
    }

    #[test]
    fn cjk_text_is_one_word_per_run() {
        assert_eq!(counted("日本語のテキスト"), (1, 8, 8));
        assert_eq!(counted("日本語 のテキスト"), (2, 9, 8));
    }

    #[test]
    fn an_emoji_of_several_scalars_counts_each_scalar() {
        // Man, zero-width joiner, woman, zero-width joiner, girl: one glyph, five scalar values, none whitespace.
        assert_eq!(counted("\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}"), (1, 5, 5));
        // A flag is two regional indicators; a thumbs up with a skin tone is two scalars.
        assert_eq!(counted("\u{1F1EB}\u{1F1F7} \u{1F44D}\u{1F3FD}"), (2, 5, 4));
    }

    #[test]
    fn the_token_count_is_passed_through() {
        assert_eq!(TextCounts::of("a b", 7).tokens, 7);
    }

    #[test]
    fn the_line_and_the_json() {
        let counts = TextCounts { words: 1234, characters: 7890, characters_no_spaces: 6600, tokens: 1650 };
        assert_eq!(counts.line(), "1,234 words, 7,890 characters, 1,650 tokens");
        assert_eq!(TextCounts::of("a", 1).line(), "1 word, 1 character, 1 token");
        assert_eq!(counts.json(), json!({"words": 1234, "characters": 7890, "characters_no_spaces": 6600, "tokens": 1650}));
    }
}
