//! What a reading says in words. The verdict is the conformal set at level 0.10, named by the engine's `Label`: one
//! origin is a call ("Human", "AI", "Mixed"); AI and mixed with human ruled out read "AI or mixed"; human with another
//! origin reads "Uncertain" with the origins named beside the word ("human or mixed"); none is "Unlike reference".
//! The app never shows one origin when the set holds more.
//!
//! The app's sentences are written for a reader who has never met a p-value: short, with few colons and semicolons,
//! and with probabilities as percentages. `tests::app_text_keeps_to_plain_punctuation` holds the sentences of this
//! file to that.

use litmus_engine::reading::percent;
use litmus_engine::{Chance, DocumentReading, Label, Origin, Reading, STRIP_LEVEL, Verdict};

/// An origin as the app writes it inside a sentence.
pub fn origin_word(origin: Origin) -> &'static str {
    match origin {
        Origin::Human => "human",
        Origin::Ai => "AI",
        Origin::Mixed => "mixed",
    }
}

/// Origins in strip order (human, mixed, AI), the order the app names them in.
fn strip_order(set: &[Origin]) -> Vec<Origin> {
    let mut origins = set.to_vec();
    origins.sort_by(|a, b| a.strip_anchor().total_cmp(&b.strip_anchor()));
    origins.dedup();
    origins
}

/// Words joined as English lists them: "a", "a or b", "a, b, or c" (the Oxford comma).
pub fn join_or(words: &[&str]) -> String {
    match words {
        [] => String::new(),
        [one] => (*one).to_owned(),
        [a, b] => format!("{a} or {b}"),
        [rest @ .., last] => format!("{}, or {last}", rest.join(", ")),
    }
}

/// The kind of verdict, for the marker shape and the color swatch beside the headline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Call {
    One(Origin),
    /// Human is ruled out and neither AI nor mixed is. It is a call, that a model wrote or edited the text.
    AiOrMixed,
    Uncertain,
    UnlikeReference,
}

/// The call of a conformal set (`None` when the release has no set at the level), by the engine's `Label`.
pub fn call(set: Option<&Vec<Origin>>) -> Call {
    match Label::of(set) {
        Label::Human => Call::One(Origin::Human),
        Label::Ai => Call::One(Origin::Ai),
        Label::Mixed => Call::One(Origin::Mixed),
        Label::AiOrMixed => Call::AiOrMixed,
        Label::Uncertain => Call::Uncertain,
        Label::UnlikeReference => Call::UnlikeReference,
    }
}

const AI_OR_MIXED: &str = "AI or mixed";
const UNCERTAIN: &str = "Uncertain";
const UNLIKE_REFERENCE: &str = "Unlike reference";

/// What an uncertain headline of 1.1.0 and earlier named beside its word where the verdict is now "AI or mixed".
const AMONG_MIXED_AND_AI: &str = "mixed or AI";

/// A verdict as the app writes it: the word, and beside an uncertain one the origins the reading could not tell
/// apart ("human or mixed"), which every screen draws in a quieter ink after the word, as the site's cards do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Headline {
    /// "Human", "AI", "Mixed", "AI or mixed", "Uncertain", or "Unlike reference".
    pub word: String,
    /// The origins an uncertain reading keeps, in strip order: "human or mixed", "human, mixed, or AI". Empty
    /// otherwise.
    pub among: String,
}

impl Headline {
    /// The headline on one line, for lists, accessible labels and the history: "Human", "Uncertain (human or mixed)".
    pub fn line(&self) -> String {
        if self.among.is_empty() { self.word.clone() } else { format!("{} ({})", self.word, self.among) }
    }

    /// A headline from its line, as the history keeps it. Litmus 1.1.0 and earlier kept "Uncertain: human or mixed",
    /// and called "Uncertain: mixed or AI" what now reads "AI or mixed", so a reading kept then lists as it would
    /// read today.
    pub fn from_line(line: &str) -> Self {
        let line = line.trim();
        match line.strip_prefix(UNCERTAIN) {
            Some(rest) => {
                let among = rest.trim_matches(|c: char| c == ':' || c == '(' || c == ')' || c.is_whitespace());
                if among == AMONG_MIXED_AND_AI { Self { word: AI_OR_MIXED.to_owned(), among: String::new() } } else { Self { word: UNCERTAIN.to_owned(), among: among.to_owned() } }
            }
            None => Self { word: line.to_owned(), among: String::new() },
        }
    }

    /// The kind of verdict the word names, for the marker's shape and the word's ink.
    pub fn call(&self) -> Call {
        match self.word.as_str() {
            "Human" => Call::One(Origin::Human),
            "AI" => Call::One(Origin::Ai),
            "Mixed" => Call::One(Origin::Mixed),
            AI_OR_MIXED => Call::AiOrMixed,
            UNLIKE_REFERENCE => Call::UnlikeReference,
            _ => Call::Uncertain,
        }
    }
}

/// The headline of a conformal set.
pub fn headline(set: Option<&Vec<Origin>>) -> Headline {
    let (word, among) = match call(set) {
        Call::UnlikeReference => (UNLIKE_REFERENCE, String::new()),
        Call::One(Origin::Human) => ("Human", String::new()),
        Call::One(Origin::Ai) => ("AI", String::new()),
        Call::One(Origin::Mixed) => ("Mixed", String::new()),
        Call::AiOrMixed => (AI_OR_MIXED, String::new()),
        Call::Uncertain => {
            let origins = strip_order(set.map(Vec::as_slice).unwrap_or(&[]));
            (UNCERTAIN, join_or(&origins.iter().map(|o| origin_word(*o)).collect::<Vec<_>>()))
        }
    };
    Headline { word: word.to_owned(), among }
}

/// The verdict's headline at the strip's level (0.10).
pub fn verdict_headline(verdict: &Verdict) -> Headline {
    headline(verdict.set(STRIP_LEVEL))
}

/// A reading's (a chunk's or a window's) headline at the strip's level.
pub fn reading_headline(reading: &Reading) -> Headline {
    headline(reading.set(STRIP_LEVEL))
}

/// One sentence on what the set at `level` rules out and keeps.
pub fn set_sentence(set: Option<&Vec<Origin>>, level: f64) -> String {
    let percent = format_percent(level);
    let kept = strip_order(set.map(Vec::as_slice).unwrap_or(&[]));
    let ruled_out: Vec<&str> = [Origin::Human, Origin::Mixed, Origin::Ai].into_iter().filter(|o| !kept.contains(o)).map(origin_word).collect();
    let kept_words: Vec<&str> = kept.iter().map(|o| origin_word(*o)).collect();
    match kept.len() {
        0 => format!("At the {percent} level every origin is ruled out. The text is unlike the reference texts the model was calibrated on, so no call is made."),
        1 => format!("At the {percent} level only {} is not ruled out ({} ruled out).", kept_words[0], join_and(&ruled_out)),
        3 => format!("At the {percent} level no origin can be ruled out, so human, mixed, and AI all remain possible."),
        _ => format!("At the {percent} level {} cannot be told apart, and {} is ruled out.", join_and(&kept_words), join_and(&ruled_out)),
    }
}

/// The origins a set keeps, in strip order and in words: "human and mixed", or "none" for an empty set.
pub fn kept(set: Option<&Vec<Origin>>) -> String {
    let words: Vec<&str> = strip_order(set.map(Vec::as_slice).unwrap_or(&[])).iter().map(|o| origin_word(*o)).collect();
    if words.is_empty() { "none".to_owned() } else { join_and(&words) }
}

/// Under this many tokens a reading adds that a few words carry little evidence.
pub const FEW_TOKENS: usize = 32;

/// The sentence a short text adds beside its verdict.
pub const FEW_WORDS: &str = "A few words carry little evidence, so a text this short rarely gets a firm reading.";

/// The same in the quick-check popup, where one short line has to do.
pub const FEW_WORDS_SHORT: &str = "A few words carry little evidence.";

/// The one plain sentence under the verdict on the simple screen, in the words of the site's table of verdicts
/// (`site/reading.html`): what the reading means, never more confident than its set.
pub fn plain_sentence(set: Option<&Vec<Origin>>) -> String {
    let kept = strip_order(set.map(Vec::as_slice).unwrap_or(&[]));
    let ruled_out: Vec<&str> = [Origin::Human, Origin::Mixed, Origin::Ai].into_iter().filter(|o| !kept.contains(o)).map(origin_word).collect();
    match call(set) {
        Call::UnlikeReference => "This text is unlike every text the model was calibrated on, so it makes no call either way.".to_owned(),
        Call::One(Origin::Human) => "It reads like text a person wrote. AI and mixed are both ruled out.".to_owned(),
        Call::One(Origin::Ai) => "It reads like text a language model wrote. Human and mixed are both ruled out.".to_owned(),
        Call::One(Origin::Mixed) => "It reads like text a person and a model both wrote or edited.".to_owned(),
        Call::AiOrMixed => "It reads like text a language model wrote or edited. Human is ruled out, and the reading cannot tell AI from mixed.".to_owned(),
        Call::Uncertain if kept.len() == 3 => "Nothing could be ruled out, so human, mixed, and AI all remain possible.".to_owned(),
        Call::Uncertain => {
            let words: Vec<&str> = kept.iter().map(|o| origin_word(*o)).collect();
            format!("It could be {}, and the reading cannot tell them apart. {} is ruled out.", join_or(&words), capitalize(&join_and(&ruled_out)))
        }
    }
}

/// What a reading is, and our recommendation for using one, where the detail of a reading is shown (the Full level,
/// never the Simple one).
pub const RECOMMENDATION: &str = "A reading is an estimate, not proof that anyone used a language model. We recommend not using a reading alone to accuse anyone.";

/// The caution beside a verdict whose text sits far from the reference while its set still names an origin.
pub const OUT_OF_REFERENCE: &str = "The text sits far from the reference texts, so we recommend reading this with caution.";

fn capitalize(words: &str) -> String {
    let mut chars = words.chars();
    chars.next().map_or_else(String::new, |first| first.to_uppercase().chain(chars).collect())
}

/// Words joined with "and" (the Oxford comma for three or more).
pub fn join_and(words: &[&str]) -> String {
    match words {
        [] => String::new(),
        [one] => (*one).to_owned(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

/// A level as a percentage: 0.1 to "10%", 0.05 to "5%", 0.0125 to "1.25%".
pub fn format_percent(level: f64) -> String {
    let percent = level * 100.0;
    let text = format!("{percent:.2}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    format!("{text}%")
}

/// Where a text's surprise falls among each origin's calibration texts (human, AI, mixed): the share of each that
/// the model found at least as surprising.
pub fn surprise_sentence(percentile: [f64; 3]) -> String {
    let share = |p: f64| format!("{:.0}%", 100.0 * p);
    format!(
        "Less surprising to the model than {} of the human-written calibration texts, {} of the AI-written, and {} of the mixed.",
        share(percentile[0]),
        share(percentile[1]),
        share(percentile[2])
    )
}

/// What the surprise reading is, and what it is not.
pub const SURPRISE_CAUTION: &str = "Surprise is how unexpected each token is to the model, given the rest of the text. It is evidence about the text and the model together, not a verdict, because unusual topics, formats, and languages raise it too.";

/// What the app calls the engine's `Chance`, the calibrated probability that a text is not purely human-written
/// restated at even odds: the name a reader meets on every screen. `Chance::words` writes it ("69% (62-75%)").
pub const CHANCE: &str = "Chance of AI";

/// What the chance of AI is, where the pointer rests on it.
pub const CHANCE_MEANING: &str = "How likely it is that a language model wrote or edited some of this text, taking a text to be as likely human as not before it is read. The range in parentheses is how far that chance moves from one kind of text to another, and we recommend reading it before the number.";

/// A probability to a tenth of a percent, for the account of how a reading was made: "69.3%".
pub fn percent_fine(value: f64) -> String {
    format!("{:.1}%", 100.0 * value)
}

/// Seconds, readably: "47 ms", "1.3 s", "2 min 5 s".
pub fn duration(seconds: f64) -> String {
    if seconds < 1.0 {
        format!("{:.0} ms", seconds * 1000.0)
    } else if seconds < 60.0 {
        format!("{seconds:.1} s")
    } else {
        format!("{} min {:.0} s", (seconds / 60.0).floor(), seconds % 60.0)
    }
}

/// The flags a reading raises, each as a short title and a sentence. `chance` is the verdict's chance of AI, when
/// the release's table is at hand, for the range of the sections' chances.
pub fn flags(document: &DocumentReading, chance: Option<&Chance>) -> Vec<(&'static str, String)> {
    let verdict = &document.verdict;
    let mut out = Vec::new();
    if verdict.out_of_reference {
        out.push((
            "Out of reference",
            format!(
                "The text sits farther from the reference bank than nearly every calibration text of its length (closeness p = {:.3}), and readings of such text are less reliable.",
                verdict.closeness_p
            ),
        ));
    }
    if verdict.chunks > 1 {
        if verdict.mixed_document {
            out.push((
                "Mixed document",
                format!(
                    "Of {} sections, {} read as model-written and {} as human at {} per section, so the document looks partly written by a model.",
                    verdict.chunks,
                    verdict.model_chunks,
                    verdict.human_chunks,
                    format_percent(verdict.chunk_level)
                ),
            ));
        } else {
            // The lowest and the highest section, at even odds as the verdict's chance is.
            let range = chance.map_or_else(String::new, |chance| {
                let ends = Chance::at_even_odds(verdict.probability, [verdict.lowest, verdict.highest], chance.reference_share).interval;
                format!(", with a chance of AI from {} to {}", percent(ends[0]), percent(ends[1]))
            });
            out.push((
                "No mixed-document signal",
                format!(
                    "Read as {} sections{range}. No section reads as model-written while another reads as human at {} per section.",
                    verdict.chunks,
                    format_percent(verdict.chunk_level)
                ),
            ));
        }
    }
    out
}

/// A short title for a text: its first words, at most `limit` characters.
pub fn title_of(text: &str, limit: usize) -> String {
    let words: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if words.chars().count() <= limit {
        return if words.is_empty() { "(empty)".to_owned() } else { words };
    }
    let cut: String = words.chars().take(limit).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head).to_owned();
    format!("{cut}...")
}

#[cfg(test)]
mod tests {
    use super::*;
    use Origin::{Ai, Human, Mixed};

    fn line(set: Option<&Vec<Origin>>) -> String {
        headline(set).line()
    }

    #[test]
    fn headlines_never_hide_uncertainty() {
        assert_eq!(line(Some(&vec![Human])), "Human");
        assert_eq!(line(Some(&vec![Ai])), "AI");
        assert_eq!(line(Some(&vec![Mixed])), "Mixed");
        assert_eq!(line(Some(&vec![Mixed, Human])), "Uncertain (human or mixed)");
        assert_eq!(line(Some(&vec![Human, Ai])), "Uncertain (human or AI)");
        assert_eq!(line(Some(&vec![Ai, Human, Mixed])), "Uncertain (human, mixed, or AI)");
        assert_eq!(line(Some(&vec![])), "Unlike reference");
        assert_eq!(line(None), "Unlike reference");
        assert_eq!(headline(Some(&vec![Mixed, Human])), Headline { word: "Uncertain".into(), among: "human or mixed".into() });
        assert_eq!(call(Some(&vec![Human, Mixed])), Call::Uncertain);
        assert_eq!(call(Some(&vec![Ai])), Call::One(Ai));
        assert_eq!(call(Some(&vec![])), Call::UnlikeReference);
    }

    #[test]
    fn human_ruled_out_is_a_verdict_of_its_own() {
        // Logan, 2026-11-21: "Yes, rename it". The 1.1 model gives no test passage "AI" alone, and 78% of the AI
        // passages the set of AI and mixed, which read "Uncertain" until now. Either order of the set reads the same,
        // and a set that keeps human stays uncertain.
        for set in [vec![Ai, Mixed], vec![Mixed, Ai]] {
            assert_eq!(headline(Some(&set)), Headline { word: "AI or mixed".into(), among: String::new() });
            assert_eq!(line(Some(&set)), "AI or mixed");
            assert_eq!(call(Some(&set)), Call::AiOrMixed);
            assert_eq!(litmus_engine::Label::of(Some(&set)).word(), "AI or mixed", "the command line and the tools say the same");
        }
        assert_eq!(call(Some(&vec![Human, Ai, Mixed])), Call::Uncertain);
        assert_eq!(call(Some(&vec![Human, Ai])), Call::Uncertain);
    }

    #[test]
    fn a_headline_reads_back_from_its_line() {
        for set in [vec![], vec![Human], vec![Ai], vec![Mixed], vec![Human, Mixed], vec![Mixed, Ai], vec![Human, Ai], vec![Human, Mixed, Ai]] {
            let written = headline(Some(&set));
            assert_eq!(Headline::from_line(&written.line()), written, "{set:?}");
            assert_eq!(written.call(), call(Some(&set)), "{set:?}");
        }
        // What the history of 1.1.0 and earlier holds.
        assert_eq!(Headline::from_line("Uncertain: human or mixed"), Headline { word: "Uncertain".into(), among: "human or mixed".into() });
        assert_eq!(Headline::from_line("Uncertain: human, mixed, or AI").among, "human, mixed, or AI");
        assert_eq!(Headline::from_line("Unlike reference").call(), Call::UnlikeReference);
        // A reading kept as uncertain between mixed and AI lists under the verdict it would get today.
        for kept in ["Uncertain: mixed or AI", "Uncertain (mixed or AI)"] {
            assert_eq!(Headline::from_line(kept), headline(Some(&vec![Ai, Mixed])), "{kept}");
            assert_eq!(Headline::from_line(kept).call(), Call::AiOrMixed, "{kept}");
        }
    }

    #[test]
    fn set_sentences_name_what_is_kept_and_ruled_out() {
        assert_eq!(set_sentence(Some(&vec![Ai]), 0.1), "At the 10% level only AI is not ruled out (human and mixed ruled out).");
        assert_eq!(set_sentence(Some(&vec![Human, Mixed]), 0.1), "At the 10% level human and mixed cannot be told apart, and AI is ruled out.");
        assert!(set_sentence(Some(&vec![]), 0.05).starts_with("At the 5% level every origin is ruled out"));
        assert!(set_sentence(Some(&vec![Human, Mixed, Ai]), 0.1).contains("human, mixed, and AI"));
    }

    #[test]
    fn plain_sentences_never_overstate() {
        assert_eq!(plain_sentence(Some(&vec![Human])), "It reads like text a person wrote. AI and mixed are both ruled out.");
        assert_eq!(plain_sentence(Some(&vec![Ai, Mixed])), "It reads like text a language model wrote or edited. Human is ruled out, and the reading cannot tell AI from mixed.");
        assert_eq!(plain_sentence(Some(&vec![Human, Mixed])), "It could be human or mixed, and the reading cannot tell them apart. AI is ruled out.");
        assert_eq!(plain_sentence(Some(&vec![Human, Ai])), "It could be human or AI, and the reading cannot tell them apart. Mixed is ruled out.");
        assert!(plain_sentence(Some(&vec![Ai, Human, Mixed])).contains("human, mixed, and AI all remain possible"));
        assert!(plain_sentence(Some(&vec![])).contains("makes no call"));
        assert!(plain_sentence(None).contains("makes no call"));
    }

    /// Every sentence this file can write, over every set.
    fn sentences() -> Vec<String> {
        let sets = [vec![], vec![Human], vec![Ai], vec![Mixed], vec![Human, Mixed], vec![Mixed, Ai], vec![Human, Ai], vec![Human, Mixed, Ai]];
        let mut all: Vec<String> = [FEW_WORDS, FEW_WORDS_SHORT, RECOMMENDATION, OUT_OF_REFERENCE, SURPRISE_CAUTION, CHANCE, CHANCE_MEANING].map(str::to_owned).to_vec();
        for set in &sets {
            all.extend([plain_sentence(Some(set)), set_sentence(Some(set), 0.1), set_sentence(Some(set), 0.05), headline(Some(set)).line()]);
        }
        let chance = Chance { value: 0.62, interval: [0.4, 0.8], reference_share: 0.5, between_sources: 0.0 };
        all.extend([surprise_sentence([0.2, 0.5, 0.9]), chance.words(), chance.sentence(), Chance { interval: [0.62, 0.62], ..chance }.words()]);
        all
    }

    #[test]
    fn app_text_keeps_to_plain_punctuation() {
        // Logan, 2026-11-21: "Let's use far less semicolons and colons in the app text itself." No sentence here has
        // either, nor an em dash, nor the specification's "P(not human)".
        for sentence in sentences() {
            for mark in [":", ";", "\u{2014}", "\u{2013}", "P(not human)"] {
                assert!(!sentence.contains(mark), "{mark:?} in {sentence:?}");
            }
        }
    }

    #[test]
    fn advice_on_using_a_reading_is_our_recommendation() {
        // What a reader does with a reading is theirs to decide: the app recommends, in our own voice, and never
        // tells them what they must or should do.
        for sentence in [OUT_OF_REFERENCE, RECOMMENDATION] {
            assert!(sentence.contains("We recommend") || sentence.contains("we recommend"), "{sentence}");
            for rule in ["must", "should", "never use", "do not use", "don't"] {
                assert!(!sentence.to_lowercase().contains(rule), "{rule:?} in {sentence}");
            }
        }
        assert!(RECOMMENDATION.contains("We recommend not using a reading alone to accuse anyone"));
    }

    #[test]
    fn a_long_documents_sections_give_their_chances_at_even_odds() {
        // Five sections whose probabilities of not human run from 0.5 to 0.8, read against calibration texts of
        // which two thirds are not human: 0.5 is one third at even odds, and 0.8 is two thirds.
        let mut reading: DocumentReading = serde_json::from_str(include_str!("../tests/reading_small.json")).expect("the fixture reading parses");
        reading.verdict.chunks = 5;
        (reading.verdict.lowest, reading.verdict.highest, reading.verdict.mixed_document) = (0.5, 0.8, false);
        let chance = Chance::at_even_odds(reading.verdict.probability, reading.verdict.interval, 2.0 / 3.0);
        let sentence = |chance: Option<&Chance>| flags(&reading, chance).into_iter().find(|(title, _)| *title == "No mixed-document signal").expect("a document of several sections").1;
        assert!(sentence(Some(&chance)).starts_with("Read as 5 sections, with a chance of AI from 33% to 67%. No section"), "{}", sentence(Some(&chance)));
        // Without the release's table (the contribute site's page) the sentence leaves the chances out.
        assert!(sentence(None).starts_with("Read as 5 sections. No section"), "{}", sentence(None));
        assert_eq!(percent_fine(0.6934), "69.3%");
    }

    #[test]
    fn numbers_read_plainly() {
        assert_eq!(format_percent(0.1), "10%");
        assert_eq!(format_percent(0.05), "5%");
        assert_eq!(format_percent(0.0125), "1.25%");
        assert_eq!(duration(0.047), "47 ms");
        assert_eq!(duration(1.26), "1.3 s");
        assert_eq!(duration(125.0), "2 min 5 s");
        assert_eq!(join_or(&["a", "b", "c"]), "a, b, or c");
        assert_eq!(join_and(&["a", "b"]), "a and b");
    }

    #[test]
    fn titles_cut_at_a_word() {
        assert_eq!(title_of("  One two\nthree  ", 40), "One two three");
        assert_eq!(title_of("alpha beta gamma delta", 12), "alpha beta...");
        assert_eq!(title_of("   ", 10), "(empty)");
    }
}
