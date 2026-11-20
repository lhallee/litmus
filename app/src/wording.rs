//! What a reading says in words. The verdict is the conformal set at level 0.10: one origin is a call ("Human",
//! "AI", "Mixed"), several are named as uncertain ("Uncertain: human or mixed"), none is "Unlike reference". The app
//! never shows a bare confident label when the set holds more than one origin.

use litmus_engine::{DocumentReading, Origin, Reading, STRIP_LEVEL, Verdict};

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
    Uncertain,
    UnlikeReference,
}

/// The call of a conformal set (`None` when the release has no set at the level).
pub fn call(set: Option<&Vec<Origin>>) -> Call {
    let origins = strip_order(set.map(Vec::as_slice).unwrap_or(&[]));
    match origins.as_slice() {
        [] => Call::UnlikeReference,
        [one] => Call::One(*one),
        _ => Call::Uncertain,
    }
}

/// The headline of a conformal set: "Human", "AI", "Mixed", "Uncertain: human or mixed", or "Unlike reference".
pub fn headline(set: Option<&Vec<Origin>>) -> String {
    let origins = strip_order(set.map(Vec::as_slice).unwrap_or(&[]));
    match origins.as_slice() {
        [] => "Unlike reference".to_owned(),
        [Origin::Human] => "Human".to_owned(),
        [Origin::Ai] => "AI".to_owned(),
        [Origin::Mixed] => "Mixed".to_owned(),
        several => format!("Uncertain: {}", join_or(&several.iter().map(|o| origin_word(*o)).collect::<Vec<_>>())),
    }
}

/// The verdict's headline at the strip's level (0.10).
pub fn verdict_headline(verdict: &Verdict) -> String {
    headline(verdict.set(STRIP_LEVEL))
}

/// A reading's (a chunk's or a window's) headline at the strip's level.
pub fn reading_headline(reading: &Reading) -> String {
    headline(reading.set(STRIP_LEVEL))
}

/// One sentence on what the set at `level` rules out and keeps.
pub fn set_sentence(set: Option<&Vec<Origin>>, level: f64) -> String {
    let percent = format_percent(level);
    let kept = strip_order(set.map(Vec::as_slice).unwrap_or(&[]));
    let ruled_out: Vec<&str> = [Origin::Human, Origin::Mixed, Origin::Ai].into_iter().filter(|o| !kept.contains(o)).map(origin_word).collect();
    let kept_words: Vec<&str> = kept.iter().map(|o| origin_word(*o)).collect();
    match kept.len() {
        0 => format!("At the {percent} level every origin is ruled out: the text is unlike the reference texts the model was calibrated on, so no call is made."),
        1 => format!("At the {percent} level only {} is not ruled out ({} ruled out).", kept_words[0], join_and(&ruled_out)),
        3 => format!("At the {percent} level no origin can be ruled out: human, mixed, and AI all remain possible."),
        _ => format!("At the {percent} level {} cannot be told apart; {} is ruled out.", join_and(&kept_words), join_and(&ruled_out)),
    }
}

/// Under this many tokens a reading adds that a few words carry little evidence.
pub const FEW_TOKENS: usize = 32;

/// The sentence a short text adds beside its verdict.
pub const FEW_WORDS: &str = "A few words carry little evidence, so a text this short rarely gets a firm reading.";

/// The one plain sentence under the verdict on the simple screen, in the words of the site's table of verdicts
/// (`site/reading.html`): what the reading means, never more confident than its set.
pub fn plain_sentence(set: Option<&Vec<Origin>>) -> String {
    let kept = strip_order(set.map(Vec::as_slice).unwrap_or(&[]));
    let ruled_out: Vec<&str> = [Origin::Human, Origin::Mixed, Origin::Ai].into_iter().filter(|o| !kept.contains(o)).map(origin_word).collect();
    match kept.as_slice() {
        [] => "This text is unlike every text the model was calibrated on, so it makes no call either way.".to_owned(),
        [Origin::Human] => "It reads like text a person wrote; AI and mixed are both ruled out.".to_owned(),
        [Origin::Ai] => "It reads like text a language model wrote; human and mixed are both ruled out.".to_owned(),
        [Origin::Mixed] => "It reads like text a person and a model both wrote or edited.".to_owned(),
        [_, _, _] => "Nothing could be ruled out: human, mixed, and AI all remain possible.".to_owned(),
        several => {
            let words: Vec<&str> = several.iter().map(|o| origin_word(*o)).collect();
            format!("It could be {}; the reading cannot tell them apart. {} is ruled out.", join_or(&words), capitalize(&join_and(&ruled_out)))
        }
    }
}

/// What a reading is, and our recommendation for using one, where the detail of a reading is shown (Detailed and
/// Full, never the Simple level).
pub const RECOMMENDATION: &str = "A reading is an estimate, not proof that anyone used a language model. We recommend not using a reading alone to accuse anyone.";

/// The caution beside a verdict whose text sits far from the reference while its set still names an origin.
pub const OUT_OF_REFERENCE: &str = "The text sits far from the reference texts; we recommend reading this with caution.";

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
pub const SURPRISE_CAUTION: &str = "Surprise is how unexpected each token is to the model, given the rest of the text. It is evidence about the text and the model together, not a verdict: unusual topics, formats, and languages raise it too.";

/// A probability with its interval: "0.62 [0.41, 0.80]".
pub fn probability(value: f64, interval: [f64; 2]) -> String {
    format!("{value:.2} [{:.2}, {:.2}]", interval[0], interval[1])
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

/// The flags a reading raises, each as a short title and a sentence.
pub fn flags(document: &DocumentReading) -> Vec<(&'static str, String)> {
    let verdict = &document.verdict;
    let mut out = Vec::new();
    if verdict.out_of_reference {
        out.push((
            "Out of reference",
            format!(
                "The text sits farther from the reference bank than nearly every calibration text of its length (closeness p = {:.3}); readings of such text are less reliable.",
                verdict.closeness_p
            ),
        ));
    }
    if verdict.chunks > 1 {
        if verdict.mixed_document {
            out.push((
                "Mixed document",
                format!(
                    "Of {} sections, {} read as model-written and {} as human at {} per section: the document looks partly written by a model.",
                    verdict.chunks,
                    verdict.model_chunks,
                    verdict.human_chunks,
                    format_percent(verdict.chunk_level)
                ),
            ));
        } else {
            out.push((
                "No mixed-document signal",
                format!(
                    "Read as {} sections (P(not human) from {:.2} to {:.2}); no section reads as model-written while another reads as human at {} per section.",
                    verdict.chunks,
                    verdict.lowest,
                    verdict.highest,
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

    #[test]
    fn headlines_never_hide_uncertainty() {
        assert_eq!(headline(Some(&vec![Human])), "Human");
        assert_eq!(headline(Some(&vec![Ai])), "AI");
        assert_eq!(headline(Some(&vec![Mixed])), "Mixed");
        assert_eq!(headline(Some(&vec![Mixed, Human])), "Uncertain: human or mixed");
        assert_eq!(headline(Some(&vec![Ai, Mixed])), "Uncertain: mixed or AI");
        assert_eq!(headline(Some(&vec![Human, Ai])), "Uncertain: human or AI");
        assert_eq!(headline(Some(&vec![Ai, Human, Mixed])), "Uncertain: human, mixed, or AI");
        assert_eq!(headline(Some(&vec![])), "Unlike reference");
        assert_eq!(headline(None), "Unlike reference");
        assert_eq!(call(Some(&vec![Human, Mixed])), Call::Uncertain);
        assert_eq!(call(Some(&vec![Ai])), Call::One(Ai));
        assert_eq!(call(Some(&vec![])), Call::UnlikeReference);
    }

    #[test]
    fn set_sentences_name_what_is_kept_and_ruled_out() {
        assert_eq!(set_sentence(Some(&vec![Ai]), 0.1), "At the 10% level only AI is not ruled out (human and mixed ruled out).");
        assert_eq!(set_sentence(Some(&vec![Human, Mixed]), 0.1), "At the 10% level human and mixed cannot be told apart; AI is ruled out.");
        assert!(set_sentence(Some(&vec![]), 0.05).starts_with("At the 5% level every origin is ruled out"));
        assert!(set_sentence(Some(&vec![Human, Mixed, Ai]), 0.1).contains("human, mixed, and AI"));
    }

    #[test]
    fn plain_sentences_never_overstate() {
        assert_eq!(plain_sentence(Some(&vec![Human])), "It reads like text a person wrote; AI and mixed are both ruled out.");
        assert_eq!(plain_sentence(Some(&vec![Ai, Mixed])), "It could be mixed or AI; the reading cannot tell them apart. Human is ruled out.");
        assert_eq!(plain_sentence(Some(&vec![Human, Ai])), "It could be human or AI; the reading cannot tell them apart. Mixed is ruled out.");
        assert!(plain_sentence(Some(&vec![Ai, Human, Mixed])).contains("human, mixed, and AI all remain possible"));
        assert!(plain_sentence(Some(&vec![])).contains("makes no call"));
        assert!(plain_sentence(None).contains("makes no call"));
        for sentence in [plain_sentence(Some(&vec![Mixed])), FEW_WORDS.to_owned(), OUT_OF_REFERENCE.to_owned(), RECOMMENDATION.to_owned()] {
            assert!(!sentence.contains('\u{2014}'), "no em dashes: {sentence}");
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
    fn numbers_read_plainly() {
        assert_eq!(format_percent(0.1), "10%");
        assert_eq!(format_percent(0.05), "5%");
        assert_eq!(format_percent(0.0125), "1.25%");
        assert_eq!(probability(0.6234, [0.4, 0.801]), "0.62 [0.40, 0.80]");
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
