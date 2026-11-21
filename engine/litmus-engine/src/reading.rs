//! The head and the reading (SPEC.md, sections 4 and "The reading"): a pooled vector to a unit query and key; the
//! bank reading (per-origin mean and spread of the same-origin probabilities, closeness by cosine); the calibrated
//! reading of the teacher's table (`../teacher/CALIBRATION.md`): grid conformal p-values and sets by length group, the
//! Venn-Abers probability of not human with its interval, three-way probabilities, spread percentiles, the
//! out-of-reference test and the litmus strip; a document's verdict from its chunks (the teacher's `combine_chunks`);
//! and the heat map's cells from a chunk's windows. Every reading is computed in float64 from the float32 query and
//! key (or, in int8 output, from the integer dot products of their codes), and a chunk's windows go through the head and the bank together, so each weight row is read once per chunk.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::codes;
use crate::dense::dots_f64;
use crate::release::{self, CalibrationTable, MIXED_LEVEL, Output, STRIP_LEVEL, Safetensors};

const LOG_FLOOR: f64 = 1e-12;
/// Merged p-values of several chunks: twice their mean (valid under any dependence; the teacher's `merge_p_values`).
const MERGE_FACTOR: f64 = 2.0;
/// Bank rows per tile of the batched bank product.
const BANK_TILE: usize = 32;

/// Where a text comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    Human,
    Ai,
    Mixed,
}

impl Origin {
    pub const ALL: [Origin; 3] = [Origin::Human, Origin::Ai, Origin::Mixed];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Origin::Human => "human",
            Origin::Ai => "ai",
            Origin::Mixed => "mixed",
        }
    }

    /// The origin's place on the litmus strip: human 0 (blue), mixed 0.5 (purple), AI 1 (red).
    pub fn strip_anchor(self) -> f64 {
        match self {
            Origin::Human => 0.0,
            Origin::Ai => 1.0,
            Origin::Mixed => 0.5,
        }
    }
}

/// The key of a level in `sets` and `not_human`: `0.05`, `0.1` (Python's `f"{level:g}"` for these levels).
pub fn level_key(level: f64) -> String {
    format!("{level}")
}

/// A reading's place between the three origins (SPEC.md, The reading's place): `position` from 0 (human, blue)
/// through 0.5 (mixed, purple) to 1 (AI, red), and the span `[band_lo, band_hi]` from it to the anchors of the
/// origins the conformal set at `level` (0.10) cannot rule out. The heat map's cells are colored by `position`. The
/// strip itself is drawn from the chance of AI (`Drawn`); releases through 1.1.0 drew this span as its band.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Strip {
    /// `P_ai + P_mixed / 2` of the three-way probabilities.
    pub position: f64,
    pub band_lo: f64,
    pub band_hi: f64,
    pub level: f64,
    /// The conformal set is empty or the text is out of reference: draw the strip as unlike the reference.
    pub unlike_reference: bool,
}

impl Strip {
    pub fn width(&self) -> f64 {
        self.band_hi - self.band_lo
    }
}

/// The chance of AI (SPEC.md, Chance of AI): the probability of not human restated at even odds. The table's
/// probability is right for texts mixed as the calibration texts of the span's length are (two thirds of the short
/// ones are not human), so a reading that says nothing lands on that share, not on one half. Dividing its odds by the
/// calibration texts' odds gives what the same evidence says of a text as likely human as not beforehand: a reading
/// that says nothing lands on one half, where the strip's mark sits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chance {
    pub value: f64,
    /// The chance's range: the probability's interval restated the same way, then widened on each side by
    /// `between_sources` in log odds.
    pub interval: [f64; 2],
    /// The share of not-human texts among the calibration texts behind the probability.
    pub reference_share: f64,
    /// How far the range was widened on each side, in log odds: the half width that holds the table's level of the
    /// pairings of one human source with one not-human source (SPEC.md, Chance of AI). Zero for a table without the
    /// fit, and for a chance kept by Litmus 1.1.0 and earlier.
    #[serde(default)]
    pub between_sources: f64,
}

/// What the chance of AI reads from a release's calibration table (`CalibrationTable::chance_rule`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChanceRule {
    /// Per length bin, the share of not-human texts among the calibration texts behind the probability.
    pub not_human_shares: Vec<f64>,
    /// The half width, in log odds, by which the chance's range is widened on each side for the shift between
    /// sources of text; zero for a table without the fit (Litmus 1.1.0 and earlier).
    pub between_sources: f64,
    /// The share of source pairings the half width holds (0.9), when the table has the fit.
    pub level: Option<f64>,
}

impl ChanceRule {
    /// The chance of a probability and its interval read in length bin `bin`.
    pub fn chance(&self, probability: f64, interval: [f64; 2], bin: usize) -> Chance {
        Chance::at_even_odds(probability, interval, self.not_human_shares[bin]).widened(self.between_sources)
    }
}

impl Chance {
    /// A probability and its interval, calibrated where a share `reference_share` of the texts is not human, at
    /// even odds.
    pub fn at_even_odds(probability: f64, interval: [f64; 2], reference_share: f64) -> Self {
        let restate = |p: f64| {
            let not_human = p * (1.0 - reference_share);
            not_human / (not_human + (1.0 - p) * reference_share)
        };
        Self { value: restate(probability), interval: interval.map(restate), reference_share, between_sources: 0.0 }
    }

    /// The same chance with its range widened by `log_odds` on each side: the low end's odds divided by
    /// `exp(log_odds)` and the high end's multiplied by it, so ends of 0 and 1 stay where they are and the value
    /// stays inside.
    pub fn widened(self, log_odds: f64) -> Self {
        if log_odds == 0.0 {
            return self; // a table without the fit: the range to the bit as the calibration gave it
        }
        let moved = |p: f64, by: f64| {
            let odds = p * by.exp();
            odds / (odds + 1.0 - p)
        };
        let (low, high) = (self.interval[0].min(self.interval[1]), self.interval[0].max(self.interval[1]));
        Self { interval: [moved(low, -log_odds), moved(high, log_odds)], between_sources: self.between_sources + log_odds, ..self }
    }

    /// The ends of the range as whole percentages, in order, or `None` when the range is narrower than a point or its
    /// ends round to one number: "69% (69-69%)" would say nothing, nor "56% (56-57%)" of a range from 56.4% to
    /// 56.6%. An end is never written under 1% or over 99%, as the value never claims certainty, so a range inside
    /// the last point of either end is left out (">99%", not ">99% (99-100%)").
    fn range(&self) -> Option<(f64, f64)> {
        whole_range(self.interval[0], self.interval[1])
    }

    /// The chance with its range, as the app's tables and the command line show it: "69% (62-75%)", or "69%" when the
    /// range rounds away.
    pub fn words(&self) -> String {
        match self.range() {
            Some((lo, hi)) => format!("{} ({lo:.0}-{hi:.0}%)", percent(self.value)),
            None => percent(self.value),
        }
    }

    /// The chance with its range inside a sentence: "69%, likely between 62% and 75%", or "69%".
    pub fn sentence(&self) -> String {
        match self.range() {
            Some((lo, hi)) => format!("{}, likely between {lo:.0}% and {hi:.0}%", percent(self.value)),
            None => percent(self.value),
        }
    }
}

/// The litmus strip as the app, the command line and the figures draw it (SPEC.md, The litmus strip): a scale from
/// 0 (human, blue) through purple to 1 (AI, red), a mark at the chance of AI, and a pale band over the chance's
/// range, which is the interval the chance is given with. The band is that interval and nothing wider: with a table
/// that carries the shift between sources it is short near either end and wide in the middle, and without one a
/// reading of one chunk has a range a fraction of a point wide, and its band hugs the mark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drawn {
    /// The chance of AI, where the mark sits; none where no calibration table is at hand, and the scale is then bare.
    pub mark: Option<f64>,
    /// The chance's range, its ends in order and holding the mark; none for a text unlike the reference, whose range
    /// says nothing.
    pub band: Option<[f64; 2]>,
    /// The text is unlike the reference: the strip is washed out and its mark a question mark.
    pub unlike_reference: bool,
}

impl Drawn {
    /// The drawing of a chance; `unlike_reference` is the reading's `strip.unlike_reference`.
    pub fn of(chance: Option<&Chance>, unlike_reference: bool) -> Self {
        let mark = chance.map(|chance| chance.value).filter(|value| value.is_finite());
        let band = match (chance, mark) {
            (Some(chance), Some(mark)) if !unlike_reference && chance.interval.iter().all(|end| end.is_finite()) => {
                let [a, b] = chance.interval;
                Some([a.min(b).min(mark).clamp(0.0, 1.0), a.max(b).max(mark).clamp(0.0, 1.0)])
            }
            _ => None,
        };
        Self { mark, band, unlike_reference }
    }

    /// The band's width in points of chance, 0 without a band.
    pub fn range_points(&self) -> f64 {
        self.band.map_or(0.0, |[lo, hi]| 100.0 * (hi - lo))
    }

    /// The band's ends as the whole percentages a chance's range is written with, or `None` without a band or for
    /// one too narrow to write (`whole_range`).
    pub fn range(&self) -> Option<(f64, f64)> {
        self.band.and_then(|[lo, hi]| whole_range(lo, hi))
    }
}

/// The ends of a range as whole percentages, in order, or `None` when the range is narrower than a point or its ends
/// round to one number. An end is never written under 1% or over 99%, as a chance never claims certainty.
pub fn whole_range(a: f64, b: f64) -> Option<(f64, f64)> {
    let (low, high) = (a.min(b), a.max(b));
    let end = |v: f64| (100.0 * v).round().clamp(1.0, 99.0);
    (low.is_finite() && high.is_finite() && high - low >= 0.01 - 1e-9 && end(high) > end(low)).then(|| (end(low), end(high)))
}

/// A probability as a whole percentage that never claims certainty: "<1%", "69%", ">99%".
pub fn percent(value: f64) -> String {
    if !value.is_finite() {
        return "unknown".to_owned();
    }
    let rounded = (100.0 * value).round();
    if rounded < 1.0 {
        "<1%".to_owned()
    } else if rounded > 99.0 {
        ">99%".to_owned()
    } else {
        format!("{rounded:.0}%")
    }
}

/// The verdict of a conformal set, in a word or two. Every program that names a verdict takes it from here, so the
/// app, the command line, the MCP tools, the exports and the Python module call the same set the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Human,
    Ai,
    Mixed,
    /// Human is ruled out and neither AI nor mixed is: the reading says a model wrote or edited the text and cannot
    /// say which. Litmus 1.1.0 and earlier called this set uncertain; it is the verdict most model-written text gets
    /// (78% of the AI test passages of 1.1.0, none of which read AI alone), so it has a name of its own.
    AiOrMixed,
    /// Human and at least one other origin are not ruled out.
    Uncertain,
    /// No origin is likely: the text is unlike the reference.
    UnlikeReference,
}

impl Label {
    pub fn word(self) -> &'static str {
        match self {
            Label::Human => "human",
            Label::Ai => "AI",
            Label::Mixed => "mixed",
            Label::AiOrMixed => "AI or mixed",
            Label::Uncertain => "uncertain",
            Label::UnlikeReference => "unlike reference",
        }
    }

    /// The verdict of a set of origins not ruled out, in any order (`None` when the release has no set at the level).
    pub fn of(set: Option<&Vec<Origin>>) -> Label {
        let kept = |origin: Origin| set.is_some_and(|set| set.contains(&origin));
        match (kept(Origin::Human), kept(Origin::Ai), kept(Origin::Mixed)) {
            (false, false, false) => Label::UnlikeReference,
            (true, false, false) => Label::Human,
            (false, true, false) => Label::Ai,
            (false, false, true) => Label::Mixed,
            (false, true, true) => Label::AiOrMixed,
            (true, _, _) => Label::Uncertain,
        }
    }
}

/// The reading of one span read as a text of its own: a chunk or a window (SPEC.md, Fields).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reading {
    pub n_tokens: usize,
    pub length_bin: usize,
    /// Per origin (human, ai, mixed): the mean same-origin probability against the bank's texts of that origin.
    pub mean: [f64; 3],
    /// Per origin: the population standard deviation of those probabilities.
    pub spread: [f64; 3],
    /// `mean` normalized to sum to one.
    pub share: [f64; 3],
    /// `1 - share_human`, the two-class reading.
    pub score: f64,
    /// Grid conformal p-value per origin, by length group (an upper bound, exact at the levels).
    pub p_values: [f64; 3],
    /// Per level of the table (`"0.05"`, `"0.1"`): the origins not ruled out, `p_o > level`. `Label` names what a
    /// set reads as.
    pub sets: BTreeMap<String, Vec<Origin>>,
    /// Per level: `p_human <= level`. For human text like the calibration texts of its length, at most a share
    /// `level` is so called.
    pub not_human: BTreeMap<String, bool>,
    /// Probability of not human (inductive Venn-Abers) and its interval `[p0, p1]`.
    pub probability: f64,
    pub interval: [f64; 2],
    /// Calibrated probabilities of human, ai and mixed (they sum to one).
    pub three_way: [f64; 3],
    /// Percentile of each origin's spread among calibration texts of the length group.
    pub spread_percentile: [f64; 3],
    /// Mean of the `top_k` highest cosines to the bank, and its grid p-value.
    pub closeness: f64,
    pub closeness_p: f64,
    pub out_of_reference: bool,
    pub strip: Strip,
    /// How the reading was made: calibration counts and the nearest bank entries (chunks only; windows go without).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<Explain>,
}

impl Reading {
    /// The origins not ruled out at `level`, if the table has it.
    pub fn set(&self, level: f64) -> Option<&Vec<Origin>> {
        self.sets.get(&level_key(level))
    }

    /// The verdict of the conformal set at `level` (0.10 is the strip's).
    pub fn label(&self, level: f64) -> Label {
        Label::of(self.set(level))
    }

    /// The chance of AI of this span; `rule` is the release's `ModelInfo::chance_rule`.
    pub fn chance(&self, rule: &ChanceRule) -> Chance {
        rule.chance(self.probability, self.interval, self.length_bin)
    }

    /// The strip drawn of this span: its mark at the chance of AI, its band over the chance's range.
    pub fn drawn(&self, rule: &ChanceRule) -> Drawn {
        Drawn::of(Some(&self.chance(rule)), self.strip.unlike_reference)
    }
}

/// Token positions `[start, end)` of a span of the document and its place in the original text. Spans tile the
/// text: a span starting at token 0 starts at byte 0, one ending at the last token ends at the text's end, and
/// otherwise a boundary falls at the first byte of the token after it. Scoring token ids (no text) leaves the byte
/// and character offsets at zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Place {
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub start_byte: usize,
    #[serde(default)]
    pub end_byte: usize,
    /// The same in characters (Unicode scalar values).
    #[serde(default)]
    pub start_char: usize,
    #[serde(default)]
    pub end_char: usize,
}

/// A chunk's reading, with its explanation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChunkReading {
    #[serde(flatten)]
    pub place: Place,
    #[serde(flatten)]
    pub reading: Reading,
}

/// A window's reading: a span of `window_width` tokens of one chunk, pooled from that chunk's token states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowReading {
    #[serde(flatten)]
    pub place: Place,
    pub chunk: usize,
    #[serde(flatten)]
    pub reading: Reading,
}

/// One cell of the heat map: one stride of a chunk, read from the windows that cover it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    #[serde(flatten)]
    pub place: Place,
    /// Overlap-weighted mean strip position of the covering windows, and their weighted standard deviation.
    pub position: f64,
    pub spread: f64,
    /// The lowest `band_lo` and the highest `band_hi` of those windows' strips.
    pub band_lo: f64,
    pub band_hi: f64,
    pub windows: usize,
    /// Some covering window is unlike the reference.
    pub unlike_reference: bool,
}

/// A document's verdict from its chunks (the teacher's `combine_chunks`; one chunk keeps its own values).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub chunks: usize,
    /// The family level of the mixed-document signal (0.05).
    pub level: f64,
    /// Merged conformal p-values: `min(1, 2 * mean)` over chunks, valid under any dependence between them.
    pub p_values: [f64; 3],
    pub sets: BTreeMap<String, Vec<Origin>>,
    pub not_human: BTreeMap<String, bool>,
    /// Mean of the chunks' probabilities of not human; the interval is their mean Venn-Abers interval widened by
    /// the chunks' disagreement (population standard deviation of their probabilities).
    pub probability: f64,
    pub interval: [f64; 2],
    pub disagreement: f64,
    pub lowest: f64,
    pub highest: f64,
    /// Mean of the chunks' three-way probabilities.
    pub three_way: [f64; 3],
    pub closeness_p: f64,
    pub out_of_reference: bool,
    pub strip: Strip,
    /// `level / chunks`, the Bonferroni level of the mixed-document signal.
    pub chunk_level: f64,
    /// Chunks whose human p-value is at or under `chunk_level`, and chunks whose ai and mixed p-values both are.
    pub model_chunks: usize,
    pub human_chunks: usize,
    /// One chunk is a model chunk and a different chunk a human chunk.
    pub mixed_document: bool,
}

impl Verdict {
    pub fn set(&self, level: f64) -> Option<&Vec<Origin>> {
        self.sets.get(&level_key(level))
    }

    /// The verdict of the merged conformal set at `level` (0.10 is the strip's).
    pub fn label(&self, level: f64) -> Label {
        Label::of(self.set(level))
    }
}

/// The reading of a whole document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DocumentReading {
    pub n_tokens: usize,
    /// The release's chunk limit `C_max`; a longer document is read in `ceil(n / C_max)` equal chunks.
    pub chunk_tokens: usize,
    pub verdict: Verdict,
    pub chunks: Vec<ChunkReading>,
    /// The heat map's windows and cells, in position order.
    pub windows: Vec<WindowReading>,
    pub cells: Vec<Cell>,
}

impl DocumentReading {
    /// The document's verdict, at the strip's level (0.10).
    pub fn label(&self) -> Label {
        self.verdict.label(STRIP_LEVEL)
    }

    /// The verdict's chance of AI: its probability and interval at even odds, by the calibration texts of the first
    /// chunk's length (chunks are equal to a token), widened for the shift between sources; `rule` is the release's
    /// `ModelInfo::chance_rule`.
    pub fn chance(&self, rule: &ChanceRule) -> Chance {
        rule.chance(self.verdict.probability, self.verdict.interval, self.chunks[0].reading.length_bin)
    }

    /// The strip drawn of the verdict: its mark at the chance of AI, its band over the chance's range.
    pub fn drawn(&self, rule: &ChanceRule) -> Drawn {
        Drawn::of(Some(&self.chance(rule)), self.verdict.strip.unlike_reference)
    }
}

/// The head's queries and keys of `v` pooled vectors, `(v, w)` each: float32 unit vectors, or, in int8 output, the
/// int8 codes of each query and key (SPEC.md, Int8 output).
pub enum HeadVectors {
    Float { queries: Vec<f32>, keys: Vec<f32> },
    Int8 { queries: Vec<i8>, keys: Vec<i8> },
}

/// The head: standardization clamped at `head.clamp` deviations, a SiLU layer, a linear layer to a query and a key.
pub struct Head {
    input_mean: Vec<f32>,
    input_scale: Vec<f32>,
    fc1: Vec<f32>, // (hh, 2d)
    fc1_bias: Vec<f32>,
    fc2: Vec<f32>, // (2w, hh)
    fc2_bias: Vec<f32>,
    pair_scale: f64,
    pair_bias: f64,
    clamp: f32,
    hidden: usize,
    out: usize, // w
    output: Output,
    act_eps: f32,
}

impl Head {
    pub fn load(tensors: &Safetensors, width: usize, hidden: usize, out: usize, clamp: f32, output: Output, act_eps: f32) -> Result<Self, Error> {
        let pair = tensors.f32s("head.pair", &[2])?;
        Ok(Self {
            input_mean: tensors.f32s("head.input_mean", &[2 * width])?,
            input_scale: tensors.f32s("head.input_scale", &[2 * width])?,
            fc1: tensors.f16s("head.fc1.weight", &[hidden, 2 * width])?,
            fc1_bias: tensors.f16s("head.fc1.bias", &[hidden])?,
            fc2: tensors.f16s("head.fc2.weight", &[2 * out, hidden])?,
            fc2_bias: tensors.f16s("head.fc2.bias", &[2 * out])?,
            pair_scale: pair[0] as f64,
            pair_bias: pair[1] as f64,
            clamp,
            hidden,
            out,
            output,
            act_eps,
        })
    }

    pub fn output(&self) -> Output {
        self.output
    }

    pub fn bytes(&self) -> usize {
        4 * (self.input_mean.len() + self.input_scale.len() + self.fc1.len() + self.fc1_bias.len() + self.fc2.len() + self.fc2_bias.len())
    }

    pub fn out_width(&self) -> usize {
        self.out
    }

    /// Pooled statistics `(v, 2d)` to queries and keys `(v, w)` each, with the matrix sums in float64: float32 unit
    /// vectors, or the int8 codes of the head's output in int8 output.
    pub fn apply(&self, pooled: &[f32]) -> HeadVectors {
        let width = self.input_mean.len();
        assert!(!pooled.is_empty() && pooled.len() % width == 0, "rows of 2d pooled statistics");
        let count = pooled.len() / width;
        let z: Vec<f64> = pooled
            .chunks_exact(width)
            .flat_map(|row| {
                row.iter()
                    .zip(&self.input_mean)
                    .zip(&self.input_scale)
                    .map(|((&x, &m), &s)| ((x - m) / s).clamp(-self.clamp, self.clamp) as f64)
            })
            .collect(); // (v, 2d)
        let mut first = vec![0.0; self.hidden * count]; // (hh, v)
        dots_f64(&self.fc1, &z, width, &mut first);
        let mut a = vec![0.0f64; count * self.hidden]; // (v, hh)
        for i in 0..self.hidden {
            for v in 0..count {
                let value = first[i * count + v] as f32 + self.fc1_bias[i];
                a[v * self.hidden + i] = (value / (1.0 + (-value).exp())) as f64;
            }
        }
        let mut second = vec![0.0; 2 * self.out * count]; // (2w, v)
        dots_f64(&self.fc2, &a, self.hidden, &mut second);
        let mut o = vec![0f32; 2 * self.out];
        let outputs = |v: usize, o: &mut [f32]| {
            for (i, value) in o.iter_mut().enumerate() {
                *value = second[i * count + v] as f32 + self.fc2_bias[i];
            }
        };
        if self.output == Output::Int8 {
            let (mut queries, mut keys) = (vec![0i8; count * self.out], vec![0i8; count * self.out]);
            for v in 0..count {
                outputs(v, &mut o);
                codes::quantize_output(&o[..self.out], self.act_eps, &mut queries[v * self.out..(v + 1) * self.out]);
                codes::quantize_output(&o[self.out..], self.act_eps, &mut keys[v * self.out..(v + 1) * self.out]);
            }
            return HeadVectors::Int8 { queries, keys };
        }
        let (mut queries, mut keys) = (vec![0f32; count * self.out], vec![0f32; count * self.out]);
        for v in 0..count {
            outputs(v, &mut o);
            let unit = |values: &[f32], into: &mut [f32]| {
                let norm = (values.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt() as f32).max(1e-8);
                for (target, &x) in into.iter_mut().zip(values) {
                    *target = x / norm;
                }
            };
            unit(&o[..self.out], &mut queries[v * self.out..(v + 1) * self.out]);
            unit(&o[self.out..], &mut keys[v * self.out..(v + 1) * self.out]);
        }
        HeadVectors::Float { queries, keys }
    }
}

/// The reference bank: query and key pairs grouped by origin (float16 unit vectors widened to float32, each row's norm;
/// or int8 codes with their inverse norms), and what is known of each bank text (its token count and metadata codes;
/// the bank carries no text).
pub struct Bank {
    vectors: BankVectors,
    counts: [usize; 3],
    out: usize,
    tokens: Vec<i32>,                               // (m,)
    metadata: Vec<(String, Vec<String>, Vec<u16>)>, // field, its names, (m,) codes
}

/// The bank's vectors: float16 widened to float32 with each row's norm, or, in int8 output, the int8 codes of every
/// query and key (each half padded with zeros to `stride` bytes) with their inverse norms (SPEC.md, Int8 output).
enum BankVectors {
    Float {
        vectors: Vec<f32>, // (m, 2w)
        norms: Vec<f64>,   // (m,) float64 norm of each row
    },
    Int8 {
        stride: usize,
        queries: Vec<i8>,          // (m, stride)
        keys: Vec<i8>,             // (m, stride)
        inverse_queries: Vec<f64>, // (m,) `1 / sqrt(sum c^2)` of each query's codes, 0 for a zero vector
        inverse_keys: Vec<f64>,    // (m,)
    },
}

impl Bank {
    pub fn load(tensors: &Safetensors, out: usize, fields: &BTreeMap<String, Vec<String>>, output: Output) -> Result<Self, Error> {
        let rows = tensors.length("bank.origin")?;
        let origins = tensors.get("bank.origin", "U8", &[rows])?.data;
        if !origins.windows(2).all(|pair| pair[0] <= pair[1]) || origins.iter().any(|&o| o > 2) {
            return Err(Error::Load("bank.origin must be 0, 1, 2 in ascending order".into()));
        }
        let mut counts = [0usize; 3];
        for &origin in origins {
            counts[origin as usize] += 1;
        }
        if counts.contains(&0) {
            return Err(Error::Load(format!("the bank needs texts of every origin, has {counts:?}")));
        }
        let vectors = match output {
            Output::Float => {
                let vectors = tensors.f16s("bank.vectors", &[rows, 2 * out])?;
                let norms = vectors.chunks_exact(2 * out).map(|row| row.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt()).collect();
                BankVectors::Float { vectors, norms }
            }
            Output::Int8 => {
                let stored = tensors.i8s("bank.vectors", &[rows, 2 * out])?;
                if stored.contains(&i8::MIN) {
                    return Err(Error::Load("bank.vectors holds the code -128, which an int8 output never has".into()));
                }
                let stride = codes::padded(out);
                let (mut queries, mut keys) = (vec![0i8; rows * stride], vec![0i8; rows * stride]);
                for (j, row) in stored.chunks_exact(2 * out).enumerate() {
                    queries[j * stride..j * stride + out].copy_from_slice(&row[..out]);
                    keys[j * stride..j * stride + out].copy_from_slice(&row[out..]);
                }
                let inverse = |rows: &[i8]| -> Vec<f64> { rows.chunks_exact(stride).map(codes::inverse_norm).collect() };
                BankVectors::Int8 { stride, inverse_queries: inverse(&queries), inverse_keys: inverse(&keys), queries, keys }
            }
        };
        let tokens = tensors.i32s("bank.tokens", &[rows])?;
        let metadata = fields
            .iter()
            .map(|(field, names)| {
                let codes = tensors.u16s(&format!("bank.meta.{field}"), &[rows])?;
                if codes.iter().any(|&code| code as usize >= names.len()) {
                    return Err(Error::Load(format!("bank.meta.{field} holds a code past its {} names", names.len())));
                }
                Ok((field.clone(), names.clone(), codes))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Self { vectors, counts, out, tokens, metadata })
    }

    pub fn len(&self) -> usize {
        self.counts.iter().sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }

    pub fn bytes(&self) -> usize {
        let vectors = match &self.vectors {
            BankVectors::Float { vectors, norms } => 4 * vectors.len() + 8 * norms.len(),
            BankVectors::Int8 { queries, keys, inverse_queries, inverse_keys, .. } => queries.len() + keys.len() + 8 * (inverse_queries.len() + inverse_keys.len()),
        };
        vectors + 4 * self.tokens.len() + self.metadata.iter().map(|(_, _, codes)| 2 * codes.len()).sum::<usize>()
    }

    fn origin_of(&self, j: usize) -> Origin {
        let (human, ai) = (self.counts[0], self.counts[1]);
        if j < human {
            Origin::Human
        } else if j < human + ai {
            Origin::Ai
        } else {
            Origin::Mixed
        }
    }
}

/// Bank entries an explanation lists, nearest first.
pub const NEAREST: usize = 5;

/// The counts behind one conformal p-value: of `n` calibration texts of `origin` in the length group, `count` read
/// at most this much like `origin`; `p = (1 + count) / (n + 1)` is the exact p-value the grid value bounds above.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalCount {
    pub origin: Origin,
    pub n: usize,
    pub count: usize,
    pub p: f64,
    pub bins: Vec<usize>,
}

/// The counts behind the closeness p-value: `count` of `n` calibration texts of this length sat as far from the
/// reference as this one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClosenessCount {
    pub n: usize,
    pub count: usize,
    pub p: f64,
    pub bins: Vec<usize>,
}

/// One of the bank texts nearest the text (by cosine), with what the release knows of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NearestEntry {
    pub index: usize,
    pub origin: Origin,
    pub cosine: f64,
    /// The pair probability that the text and this bank text share an origin.
    pub probability: f64,
    pub tokens: i64,
    /// The bank's metadata fields (source, generator, domain, era, ...), by name.
    #[serde(flatten)]
    pub metadata: BTreeMap<String, String>,
}

/// How a reading was made, in numbers a person can check (SPEC.md, Explanation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Explain {
    pub conformal: Vec<ConformalCount>,
    pub closeness: ClosenessCount,
    pub nearest: Vec<NearestEntry>,
}

/// The calibration texts' own sorted scores per group (`explain.*` tensors).
pub struct Explainer {
    conformal: [Vec<Vec<f64>>; 3], // [origin][group] ascending
    closeness: Vec<Vec<f64>>,      // [group] ascending
}

impl Explainer {
    pub fn load(tensors: &Safetensors, table: &CalibrationTable) -> Result<Self, Error> {
        let cells = |prefix: &str, groups: &[release::GridGroup]| -> Result<Vec<Vec<f64>>, Error> {
            let n = tensors.length(&format!("{prefix}.scores"))?;
            let scores: Vec<f64> = tensors.f32s(&format!("{prefix}.scores"), &[n])?.into_iter().map(f64::from).collect();
            let offsets = tensors.i64s(&format!("{prefix}.offsets"), &[groups.len() + 1])?;
            if offsets.last().copied() != Some(n as i64) || offsets[0] != 0 {
                return Err(Error::Load(format!("{prefix}.offsets must run from 0 to {n}")));
            }
            groups
                .iter()
                .enumerate()
                .map(|(index, group)| {
                    let cell = scores[offsets[index] as usize..offsets[index + 1] as usize].to_vec();
                    if cell.len() != group.n || !cell.windows(2).all(|pair| pair[0] <= pair[1]) {
                        return Err(Error::Load(format!("{prefix}: group {index} must hold its {} scores, ascending", group.n)));
                    }
                    Ok(cell)
                })
                .collect()
        };
        Ok(Self {
            conformal: [
                cells("explain.conformal.human", &table.conformal.human)?,
                cells("explain.conformal.ai", &table.conformal.ai)?,
                cells("explain.conformal.mixed", &table.conformal.mixed)?,
            ],
            closeness: cells("explain.closeness", &table.closeness)?,
        })
    }

    pub fn bytes(&self) -> usize {
        8 * (self.conformal.iter().flatten().map(Vec::len).sum::<usize>() + self.closeness.iter().map(Vec::len).sum::<usize>())
    }
}

/// `numpy.interp(value, x, y)` for one value, including its handling of exact knots.
fn interp(value: f64, x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if value < x[0] {
        return y[0];
    }
    if value >= x[n - 1] {
        return y[n - 1];
    }
    let j = x.partition_point(|&knot| knot <= value) - 1;
    if value == x[j] {
        return y[j];
    }
    let slope = (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
    slope * (value - x[j]) + y[j]
}

/// A bank entry among the nearest: cosine, index, pair probability.
type Near = (f64, usize, f64);

/// The bank reading of one query and key, gathered in one pass over the bank.
struct BankReading {
    mean: [f64; 3],
    spread: [f64; 3],
    closeness: f64,
    /// The highest cosines, highest first, ties to the lower index.
    nearest: Vec<Near>,
}

/// Running per-origin sums of the pair probabilities (shifted by each origin's first value, for accuracy) and the
/// highest cosines of one vector.
struct BankSums {
    shift: [f64; 3],
    sum: [f64; 3],
    squares: [f64; 3],
    top: Vec<Near>,
    keep: usize,
}

impl BankSums {
    #[inline]
    fn add(&mut self, origin: usize, first_of_origin: bool, near: Near) {
        let p = near.2;
        if first_of_origin {
            self.shift[origin] = p;
        }
        let deviation = p - self.shift[origin];
        self.sum[origin] += deviation;
        self.squares[origin] += deviation * deviation;
        if self.top.len() < self.keep || near.0 > self.top[self.top.len() - 1].0 {
            let at = self.top.partition_point(|entry| entry.0 >= near.0);
            self.top.insert(at, near);
            self.top.truncate(self.keep);
        }
    }
}

/// Everything a reading needs besides the encoder.
pub struct Reader<'a> {
    pub head: &'a Head,
    pub bank: &'a Bank,
    pub calibration: &'a CalibrationTable,
    pub explainer: &'a Explainer,
    /// Threads the bank's pass is spread over (`bank_readings`); 1 reads on the calling thread alone.
    pub threads: usize,
}

impl Reader<'_> {
    /// The bank readings of `v` queries and keys `(v, w)` each, in one pass over the bank: the four products of a
    /// text with a bank row come from two float64 products, `[k, q] . [q_j, k_j]` (the pair logit's sum) and
    /// `[q, k] . [q_j, k_j]` (the cosine's numerator).
    fn bank_readings_float(&self, bank_vectors: &[f32], bank_norms: &[f64], queries: &[f32], keys: &[f32]) -> Vec<BankReading> {
        let w = self.bank.out;
        let count = queries.len() / w;
        let (m, counts) = (self.bank.len(), self.bank.counts);
        let mut rows = vec![0f64; 2 * count * 2 * w]; // (2v, 2w): [q, k] then [k, q] per vector
        let mut norms = vec![0f64; count];
        for v in 0..count {
            let (q, k) = (&queries[v * w..(v + 1) * w], &keys[v * w..(v + 1) * w]);
            let row = &mut rows[2 * v * 2 * w..(2 * v + 2) * 2 * w];
            for i in 0..w {
                row[i] = q[i] as f64;
                row[w + i] = k[i] as f64;
                row[2 * w + i] = k[i] as f64;
                row[3 * w + i] = q[i] as f64;
            }
            norms[v] = row[..2 * w].iter().map(|x| x * x).sum::<f64>().sqrt();
        }
        let keep = self.calibration.top_k.max(NEAREST).min(m);
        let mut sums: Vec<BankSums> =
            (0..count).map(|_| BankSums { shift: [0.0; 3], sum: [0.0; 3], squares: [0.0; 3], top: Vec::with_capacity(keep + 1), keep }).collect();
        let (scale, bias) = (self.head.pair_scale, self.head.pair_bias);
        let starts = [0, counts[0], counts[0] + counts[1]];
        let mut tile = vec![0f64; BANK_TILE * 2 * count];
        let mut first = 0;
        while first < m {
            let rows_here = BANK_TILE.min(m - first);
            let out = &mut tile[..rows_here * 2 * count];
            dots_f64(&bank_vectors[first * 2 * w..(first + rows_here) * 2 * w], &rows, 2 * w, out);
            for r in 0..rows_here {
                let j = first + r;
                let origin = self.bank.origin_of(j).index();
                let first_of_origin = j == starts[origin];
                let row_norm = bank_norms[j];
                for (v, sum) in sums.iter_mut().enumerate() {
                    let cosine_numerator = out[r * 2 * count + 2 * v];
                    let logit_sum = out[r * 2 * count + 2 * v + 1];
                    let logit = scale * logit_sum / 2.0 + bias;
                    let p = 0.5 * (1.0 + (0.5 * logit).tanh());
                    sum.add(origin, first_of_origin, (cosine_numerator / (norms[v] * row_norm), j, p));
                }
            }
            first += rows_here;
        }
        self.finish(sums)
    }

    /// The bank readings of `v` queries and keys in int8 output (SPEC.md, Int8 output), in one pass over the bank: the
    /// four integer dot products of a text with a bank row (the query and the key of each against the row's query and
    /// key) come from two batched products of the bank's queries and then its keys with every text vector, exact in
    /// int32, and the rest is the specified float64 arithmetic, in its order.
    fn bank_readings_int8(&self, queries: &[i8], keys: &[i8]) -> Vec<BankReading> {
        let BankVectors::Int8 { stride, queries: bank_queries, keys: bank_keys, inverse_queries, inverse_keys } = &self.bank.vectors else {
            unreachable!("an int8 head reads an int8 bank (checked at load)")
        };
        let stride = *stride;
        let w = self.bank.out;
        let count = queries.len() / w;
        let (m, counts) = (self.bank.len(), self.bank.counts);
        let mut text = vec![0i8; 2 * count * stride]; // (2v, stride): the query then the key of each text, padded with zeros
        let mut inverse = vec![[0f64; 2]; count]; // (v, 2): the inverse norms of the query and of the key
        for v in 0..count {
            let (q, k) = (&queries[v * w..(v + 1) * w], &keys[v * w..(v + 1) * w]);
            text[2 * v * stride..2 * v * stride + w].copy_from_slice(q);
            text[(2 * v + 1) * stride..(2 * v + 1) * stride + w].copy_from_slice(k);
            inverse[v] = [codes::inverse_norm(q), codes::inverse_norm(k)];
        }
        let keep = self.calibration.top_k.max(NEAREST).min(m);
        let mut sums: Vec<BankSums> =
            (0..count).map(|_| BankSums { shift: [0.0; 3], sum: [0.0; 3], squares: [0.0; 3], top: Vec::with_capacity(keep + 1), keep }).collect();
        let (scale, bias) = (self.head.pair_scale, self.head.pair_bias);
        let starts = [0, counts[0], counts[0] + counts[1]];
        // tile_q[(2v + t) * rows + r] is the dot product of text vector 2v + t (t = 0: its query, 1: its key) with the
        // query of bank row r of the tile; tile_k the same against the row's key.
        let (mut tile_q, mut tile_k) = (vec![0i32; BANK_TILE * 2 * count], vec![0i32; BANK_TILE * 2 * count]);
        let mut first = 0;
        while first < m {
            let rows_here = BANK_TILE.min(m - first);
            let span = first * stride..(first + rows_here) * stride;
            codes::dots_i8(&bank_queries[span.clone()], &text, stride, &mut tile_q[..rows_here * 2 * count]);
            codes::dots_i8(&bank_keys[span], &text, stride, &mut tile_k[..rows_here * 2 * count]);
            let (tile_q, tile_k) = (&tile_q[..rows_here * 2 * count], &tile_k[..rows_here * 2 * count]);
            for r in 0..rows_here {
                let j = first + r;
                let origin = self.bank.origin_of(j).index();
                let first_of_origin = j == starts[origin];
                let (inverse_query, inverse_key) = (inverse_queries[j], inverse_keys[j]);
                for (v, sum) in sums.iter_mut().enumerate() {
                    let dot = |tile: &[i32], part: usize| tile[(2 * v + part) * rows_here + r] as f64;
                    let (qq, kq, qk, kk) = (dot(tile_q, 0), dot(tile_q, 1), dot(tile_k, 0), dot(tile_k, 1));
                    let [inverse_text_query, inverse_text_key] = inverse[v];
                    let pair = (qk * inverse_text_query) * inverse_key + (kq * inverse_text_key) * inverse_query;
                    let logit = scale * pair / 2.0 + bias;
                    let p = 0.5 * (1.0 + (0.5 * logit).tanh());
                    let cosine = ((qq * inverse_text_query) * inverse_query + (kk * inverse_text_key) * inverse_key) / 2.0;
                    sum.add(origin, first_of_origin, (cosine, j, p));
                }
            }
            first += rows_here;
        }
        self.finish(sums)
    }

    /// The readings of one pass over the bank: each text's per-origin mean and spread of the pair probabilities, the
    /// mean of its `top_k` highest cosines, and its nearest entries.
    fn finish(&self, sums: Vec<BankSums>) -> Vec<BankReading> {
        let (m, counts) = (self.bank.len(), self.bank.counts);
        let top_k = self.calibration.top_k.min(m);
        sums.into_iter()
            .map(|sum| {
                let mut mean = [0.0; 3];
                let mut spread = [0.0; 3];
                for origin in 0..3 {
                    let n = counts[origin] as f64;
                    let average = sum.sum[origin] / n;
                    mean[origin] = sum.shift[origin] + average;
                    spread[origin] = (sum.squares[origin] / n - average * average).max(0.0).sqrt();
                }
                let closeness = sum.top[..top_k].iter().map(|near| near.0).sum::<f64>() / top_k as f64;
                BankReading { mean, spread, closeness, nearest: sum.top }
            })
            .collect()
    }

    /// The bank readings of the head's queries and keys, `(v, w)` each. With `threads` over one, the vectors are cut
    /// into that many runs and each run takes its own pass over the bank on a scoped thread. A vector's sums add up in
    /// the bank's order whichever run holds it, and a dot product does not depend on the vectors beside it
    /// (`dots_f64`, `dots_i8`), so the readings are those of one pass, bit for bit.
    fn bank_readings(&self, vectors: &HeadVectors) -> Vec<BankReading> {
        let count = match vectors {
            HeadVectors::Float { queries, .. } => queries.len() / self.bank.out,
            HeadVectors::Int8 { queries, .. } => queries.len() / self.bank.out,
        };
        #[cfg(not(target_arch = "wasm32"))]
        if self.threads > 1 && count > 1 {
            let size = count.div_ceil(self.threads.min(count));
            let runs: Vec<(usize, usize)> = (0..count).step_by(size).map(|first| (first, (first + size).min(count))).collect();
            return std::thread::scope(|scope| {
                let others: Vec<_> = runs[1..].iter().map(|&run| scope.spawn(move || self.bank_readings_of(vectors, run))).collect();
                let mut readings = self.bank_readings_of(vectors, runs[0]);
                for other in others {
                    readings.extend(other.join().expect("a pass over the bank does not panic"));
                }
                readings
            });
        }
        self.bank_readings_of(vectors, (0, count))
    }

    /// The bank readings of vectors `first..end` of the head's queries and keys.
    fn bank_readings_of(&self, vectors: &HeadVectors, (first, end): (usize, usize)) -> Vec<BankReading> {
        let span = first * self.bank.out..end * self.bank.out;
        match (&self.bank.vectors, vectors) {
            (BankVectors::Float { vectors: bank, norms }, HeadVectors::Float { queries, keys }) => self.bank_readings_float(bank, norms, &queries[span.clone()], &keys[span]),
            (BankVectors::Int8 { .. }, HeadVectors::Int8 { queries, keys }) => self.bank_readings_int8(&queries[span.clone()], &keys[span]),
            _ => unreachable!("a release's head and bank have the same output (checked at load)"),
        }
    }

    /// The counts behind the p-values and the nearest bank entries (SPEC.md, Explanation).
    fn explain(&self, bank: &BankReading, share: &[f64; 3], bin: usize) -> Explain {
        let table = self.calibration;
        let groups = [&table.conformal.human, &table.conformal.ai, &table.conformal.mixed];
        let conformal = Origin::ALL
            .into_iter()
            .map(|origin| {
                let o = origin.index();
                let index = groups[o].iter().position(|g| g.bins.contains(&bin)).expect("validated: every bin has a group");
                let cell = &self.explainer.conformal[o][index];
                let count = cell.partition_point(|&score| score <= share[o]);
                ConformalCount { origin, n: cell.len(), count, p: (1 + count) as f64 / (cell.len() + 1) as f64, bins: groups[o][index].bins.clone() }
            })
            .collect();
        let index = table.closeness.iter().position(|g| g.bins.contains(&bin)).expect("validated: every bin has a group");
        let cell = &self.explainer.closeness[index];
        let count = cell.partition_point(|&score| score <= bank.closeness);
        let closeness = ClosenessCount { n: cell.len(), count, p: (1 + count) as f64 / (cell.len() + 1) as f64, bins: table.closeness[index].bins.clone() };
        let nearest = bank
            .nearest
            .iter()
            .take(NEAREST)
            .map(|&(cosine, j, probability)| NearestEntry {
                index: j,
                origin: self.bank.origin_of(j),
                cosine,
                probability,
                tokens: self.bank.tokens[j] as i64,
                metadata: self.bank.metadata.iter().map(|(field, names, codes)| (field.clone(), names[codes[j] as usize].clone())).collect(),
            })
            .collect();
        Explain { conformal, closeness, nearest }
    }

    /// Every calibrated output of one bank reading of a span of `n_tokens` tokens.
    fn calibrate(&self, bank: &BankReading, n_tokens: usize, explain: bool) -> Reading {
        let table = self.calibration;
        let bin = table.length_bin(n_tokens);
        let total: f64 = bank.mean.iter().sum();
        let share = bank.mean.map(|m| m / total);
        let score = 1.0 - share[0];

        let conformal = [&table.conformal.human, &table.conformal.ai, &table.conformal.mixed];
        let p_values: [f64; 3] = std::array::from_fn(|origin| {
            let group = release::group(conformal[origin], |g| &g.bins, bin);
            table.grid_p_value(share[origin], &group.thresholds)
        });
        let (sets, not_human) = calls(&table.levels, &p_values);

        let venn = release::group(&table.probability, |g| &g.bins, bin);
        let block = venn.edges.partition_point(|&edge| edge <= score);
        let (p0, p1) = (venn.p0[block], venn.p1[block]);
        let probability = p1 / (1.0 - p0 + p1);

        let softmax = release::group(&table.three_way, |g| &g.bins, bin);
        let mut logits: [f64; 3] = std::array::from_fn(|o| softmax.slope * share[o].max(LOG_FLOOR).ln() + softmax.bias[o]);
        let largest = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for logit in &mut logits {
            *logit -= largest;
        }
        let exps = logits.map(f64::exp);
        let sum: f64 = exps.iter().sum();
        let three_way = exps.map(|e| e / sum);

        let spreads = release::group(&table.spread, |g| &g.bins, bin);
        let quantiles = [&spreads.human, &spreads.ai, &spreads.mixed];
        let spread_percentile: [f64; 3] = std::array::from_fn(|o| interp(bank.spread[o], quantiles[o], &table.spread_levels));

        let closeness_group = release::group(&table.closeness, |g| &g.bins, bin);
        let closeness_p = table.grid_p_value(bank.closeness, &closeness_group.thresholds);
        let out_of_reference = closeness_p <= table.out_of_reference_level;
        let strip = strip(&three_way, &sets[&level_key(STRIP_LEVEL)], out_of_reference);
        let explain = explain.then(|| self.explain(bank, &share, bin));

        Reading {
            n_tokens,
            length_bin: bin,
            mean: bank.mean,
            spread: bank.spread,
            share,
            score,
            p_values,
            sets,
            not_human,
            probability,
            interval: [p0, p1],
            three_way,
            spread_percentile,
            closeness: bank.closeness,
            closeness_p,
            out_of_reference,
            strip,
            explain,
        }
    }

    /// The readings of `v` pooled vectors `(v, 2d)` of spans of `n_tokens[i]` tokens, through the head and the bank
    /// together; the first is explained if `explain_first`. Also returns the head's queries and keys `(v, w)` each.
    pub fn read(&self, pooled: &[f32], n_tokens: &[usize], explain_first: bool) -> (Vec<Reading>, HeadVectors) {
        let vectors = self.head.apply(pooled);
        let banks = self.bank_readings(&vectors);
        assert_eq!(banks.len(), n_tokens.len(), "one length per pooled vector");
        let readings = banks.iter().zip(n_tokens).enumerate().map(|(i, (bank, &n))| self.calibrate(bank, n, explain_first && i == 0)).collect();
        (readings, vectors)
    }
}

/// The sets and the not-human calls at each level from three p-values.
fn calls(levels: &[f64], p_values: &[f64; 3]) -> (BTreeMap<String, Vec<Origin>>, BTreeMap<String, bool>) {
    let sets = levels.iter().map(|&level| (level_key(level), Origin::ALL.into_iter().filter(|o| p_values[o.index()] > level).collect())).collect();
    let not_human = levels.iter().map(|&level| (level_key(level), p_values[0] <= level)).collect();
    (sets, not_human)
}

/// The strip from the three-way probabilities and the conformal set at the strip's level.
fn strip(three_way: &[f64; 3], set: &[Origin], out_of_reference: bool) -> Strip {
    let position = three_way[1] + 0.5 * three_way[2];
    let anchors = set.iter().map(|origin| origin.strip_anchor());
    let band_lo = anchors.clone().fold(position, f64::min);
    let band_hi = anchors.fold(position, f64::max);
    Strip { position, band_lo, band_hi, level: STRIP_LEVEL, unlike_reference: set.is_empty() || out_of_reference }
}

/// p-values of several chunks merged into one per origin: one chunk keeps its own, more give `min(1, 2 * mean)`.
fn merge_p_values(values: &[f64]) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    (MERGE_FACTOR * (values.iter().sum::<f64>() / values.len() as f64)).min(1.0)
}

/// What a document's verdict reads of each chunk's reading.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct ChunkCall {
    pub p_values: [f64; 3],
    pub probability: f64,
    pub interval: [f64; 2],
    pub three_way: [f64; 3],
    pub closeness_p: f64,
}

impl From<&Reading> for ChunkCall {
    fn from(reading: &Reading) -> Self {
        Self {
            p_values: reading.p_values,
            probability: reading.probability,
            interval: reading.interval,
            three_way: reading.three_way,
            closeness_p: reading.closeness_p,
        }
    }
}

/// A document's verdict from its chunks' readings (the teacher's `combine_chunks` at level 0.05).
pub fn combine(table: &CalibrationTable, chunks: &[ChunkCall]) -> Verdict {
    assert!(!chunks.is_empty(), "a document has at least one chunk");
    let count = chunks.len();
    let n = count as f64;
    let column = |f: &dyn Fn(&ChunkCall) -> f64| -> Vec<f64> { chunks.iter().map(f).collect() };
    let p_values: [f64; 3] = std::array::from_fn(|o| merge_p_values(&column(&|r| r.p_values[o])));
    let (sets, not_human) = calls(&table.levels, &p_values);
    let probabilities = column(&|r| r.probability);
    let probability = probabilities.iter().sum::<f64>() / n;
    let disagreement = (probabilities.iter().map(|p| (p - probability) * (p - probability)).sum::<f64>() / n).sqrt();
    let low = column(&|r| r.interval[0]).iter().sum::<f64>() / n;
    let high = column(&|r| r.interval[1]).iter().sum::<f64>() / n;
    let interval = [low.min(probability - disagreement).clamp(0.0, 1.0), high.max(probability + disagreement).clamp(0.0, 1.0)];
    let closeness_p = merge_p_values(&column(&|r| r.closeness_p));
    let out_of_reference = closeness_p <= table.out_of_reference_level;
    let three_way: [f64; 3] = std::array::from_fn(|o| column(&|r| r.three_way[o]).iter().sum::<f64>() / n);
    let chunk_level = MIXED_LEVEL / n;
    let model: Vec<usize> = (0..count).filter(|&c| chunks[c].p_values[0] <= chunk_level).collect();
    let human: Vec<usize> = (0..count).filter(|&c| chunks[c].p_values[1] <= chunk_level && chunks[c].p_values[2] <= chunk_level).collect();
    let mixed_document = !model.is_empty() && !human.is_empty() && !(model.len() == 1 && human.len() == 1 && model[0] == human[0]);
    let strip = strip(&three_way, &sets[&level_key(STRIP_LEVEL)], out_of_reference);
    Verdict {
        chunks: count,
        level: MIXED_LEVEL,
        p_values,
        sets,
        not_human,
        probability,
        interval,
        disagreement,
        lowest: probabilities.iter().copied().fold(f64::INFINITY, f64::min),
        highest: probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        three_way,
        closeness_p,
        out_of_reference,
        strip,
        chunk_level,
        model_chunks: model.len(),
        human_chunks: human.len(),
        mixed_document,
    }
}

/// The heat map's cells of one chunk from its windows; `cells` are document token spans, one stride each.
pub fn heat_map(windows: &[WindowReading], cells: &[(usize, usize)]) -> Vec<Cell> {
    cells
        .iter()
        .map(|&(first, last)| {
            let covering: Vec<(f64, &WindowReading)> = windows
                .iter()
                .filter(|window| window.place.start < last && window.place.end > first)
                .map(|window| ((window.place.end.min(last) - window.place.start.max(first)) as f64, window))
                .collect();
            assert!(!covering.is_empty(), "every cell lies under at least one window");
            let weight: f64 = covering.iter().map(|(overlap, _)| overlap).sum();
            let position = covering.iter().map(|(overlap, window)| overlap * window.reading.strip.position).sum::<f64>() / weight;
            let variance = covering.iter().map(|(overlap, window)| overlap * (window.reading.strip.position - position).powi(2)).sum::<f64>() / weight;
            Cell {
                place: Place { start: first, end: last, ..Place::default() },
                position,
                spread: variance.sqrt(),
                band_lo: covering.iter().map(|(_, window)| window.reading.strip.band_lo).fold(f64::INFINITY, f64::min),
                band_hi: covering.iter().map(|(_, window)| window.reading.strip.band_hi).fold(f64::NEG_INFINITY, f64::max),
                windows: covering.len(),
                unlike_reference: covering.iter().any(|(_, window)| window.reading.strip.unlike_reference),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_is_numpys() {
        let x = [0.0, 1.0, 1.0, 3.0];
        let y = [10.0, 20.0, 30.0, 0.0];
        assert_eq!(interp(-1.0, &x, &y), 10.0);
        assert_eq!(interp(0.0, &x, &y), 10.0);
        assert_eq!(interp(0.5, &x, &y), 15.0);
        assert_eq!(interp(1.0, &x, &y), 30.0);
        assert_eq!(interp(2.0, &x, &y), 15.0);
        assert_eq!(interp(3.0, &x, &y), 0.0);
        assert_eq!(interp(9.0, &x, &y), 0.0);
    }

    #[test]
    fn strip_spans_the_set() {
        let s = strip(&[0.2, 0.6, 0.2], &[Origin::Ai, Origin::Mixed], false);
        assert!((s.position - 0.7).abs() < 1e-12);
        assert_eq!((s.band_lo, s.band_hi), (0.5, 1.0));
        assert!(!s.unlike_reference);
        let empty = strip(&[0.2, 0.6, 0.2], &[], false);
        assert!(empty.unlike_reference && empty.width() == 0.0);
    }

    #[test]
    fn a_set_has_one_verdict_in_any_order() {
        use Origin::{Ai, Human, Mixed};
        let label = |set: &[Origin]| Label::of(Some(&set.to_vec()));
        assert_eq!((label(&[]), Label::of(None)), (Label::UnlikeReference, Label::UnlikeReference));
        assert_eq!((label(&[Human]), label(&[Ai]), label(&[Mixed])), (Label::Human, Label::Ai, Label::Mixed));
        // Human ruled out with AI and mixed both kept is a verdict of its own, which 1.1.0 called uncertain.
        for set in [[Ai, Mixed], [Mixed, Ai]] {
            assert_eq!((label(&set), label(&set).word()), (Label::AiOrMixed, "AI or mixed"));
        }
        // A set that keeps human beside another origin makes no call.
        for set in [vec![Human, Ai], vec![Mixed, Human], vec![Human, Ai, Mixed], vec![Mixed, Ai, Human]] {
            assert_eq!((label(&set), label(&set).word()), (Label::Uncertain, "uncertain"), "{set:?}");
        }
    }

    #[test]
    fn the_chance_of_ai_is_the_probability_at_even_odds() {
        // Two thirds of the calibration texts are not human: a probability of two thirds is no evidence, so one half.
        let nothing = Chance::at_even_odds(2.0 / 3.0, [0.6, 0.7], 2.0 / 3.0);
        assert!((nothing.value - 0.5).abs() < 1e-12);
        // Odds of 0.6 : 0.4 over 2 : 1 are 3 : 4, and of 0.7 : 0.3 over 2 : 1 are 7 : 6.
        assert!((nothing.interval[0] - 3.0 / 7.0).abs() < 1e-12 && (nothing.interval[1] - 7.0 / 13.0).abs() < 1e-12);
        // The 1.1 release's reading of a short human text: "Human" beside a probability of 0.518 became 0.349.
        assert!((Chance::at_even_odds(0.517684852203761, [0.516129, 0.5193548], 2.0 / 3.0).value - 0.3492).abs() < 1e-4);
        // Calibration texts already at even odds leave the probability as it is, and certainty stays certain.
        assert_eq!(Chance::at_even_odds(0.31, [0.25, 0.4], 0.5), Chance { value: 0.31, interval: [0.25, 0.4], reference_share: 0.5, between_sources: 0.0 });
        for share in [0.2, 0.5, 2.0 / 3.0] {
            let ends = Chance::at_even_odds(0.0, [0.0, 1.0], share);
            assert_eq!((ends.value, ends.interval), (0.0, [0.0, 1.0]));
        }
        // Order is kept, so the interval still holds the value.
        let chance = Chance::at_even_odds(0.9, [0.85, 0.97], 0.6);
        assert!(chance.interval[0] < chance.value && chance.value < chance.interval[1]);
    }

    #[test]
    fn the_strip_is_drawn_from_the_chance_and_its_range() {
        let chance = |value: f64, interval: [f64; 2]| Chance { value, interval, reference_share: 0.5, between_sources: 0.0 };
        let drawn = Drawn::of(Some(&chance(0.62, [0.4, 0.8])), false);
        assert_eq!(drawn, Drawn { mark: Some(0.62), band: Some([0.4, 0.8]), unlike_reference: false });
        assert!((drawn.range_points() - 40.0).abs() < 1e-9);
        // The ends in either order, and a mark outside its interval stretches the band to itself.
        assert_eq!(Drawn::of(Some(&chance(0.62, [0.8, 0.4])), false).band, Some([0.4, 0.8]));
        assert_eq!(Drawn::of(Some(&chance(0.9, [0.4, 0.8])), false).band, Some([0.4, 0.9]));
        // Without the shift between sources a reading of one chunk has a range a fraction of a point wide: still a
        // band, of that width and no wider, and too narrow to write.
        assert_eq!(Drawn::of(Some(&chance(0.5643, [0.5635, 0.5663])), false).band, Some([0.5635, 0.5663]));
        assert_eq!(Drawn::of(Some(&chance(0.5643, [0.5635, 0.5663])), false).range(), None);
        // A band is written as its chance's range is: whole percentages that never reach 0% or 100%.
        assert_eq!(drawn.range(), Some((40.0, 80.0)));
        assert_eq!(Drawn::of(Some(&chance(0.99, [0.959, 0.998])), false).range(), Some((96.0, 99.0)));
        assert_eq!(Drawn::of(Some(&chance(0.62, [0.4, 0.8])), true).range(), None, "no band, no range");
        // A text unlike the reference keeps its mark and loses its band; no chance, no mark.
        let unlike = Drawn::of(Some(&chance(0.62, [0.4, 0.8])), true);
        assert_eq!((unlike.mark, unlike.band, unlike.unlike_reference), (Some(0.62), None, true));
        assert_eq!(Drawn::of(None, false), Drawn { mark: None, band: None, unlike_reference: false });
        assert_eq!(Drawn::of(Some(&chance(f64::NAN, [0.4, 0.8])), false).mark, None);
    }

    #[test]
    fn a_chance_reads_as_a_percentage_with_a_range() {
        let chance = |value: f64, interval: [f64; 2]| Chance { value, interval, reference_share: 0.5, between_sources: 0.0 };
        assert_eq!(chance(0.6234, [0.4, 0.801]).words(), "62% (40-80%)");
        assert_eq!(chance(0.6234, [0.4, 0.801]).sentence(), "62%, likely between 40% and 80%");
        // An interval narrower than a point is left out, in either order of its ends, also where its ends round apart.
        assert_eq!(chance(0.69, [0.69, 0.69]).words(), "69%");
        assert_eq!(chance(0.69, [0.694, 0.686]).sentence(), "69%");
        assert_eq!(chance(0.5643, [0.5635, 0.5663]).words(), "56%");
        assert_eq!(chance(0.56, [0.55, 0.56]).words(), "56% (55-56%)", "a point wide");
        assert_eq!(chance(0.5, [0.9, 0.2]).words(), "50% (20-90%)");
        // No reading claims certainty.
        assert_eq!((percent(0.0), percent(0.004), percent(0.006), percent(0.994), percent(0.996), percent(1.0)), ("<1%".into(), "<1%".into(), "1%".into(), "99%".into(), ">99%".into(), ">99%".into()));
        // Nor does a range: its ends stay within 1% and 99%, and a range inside the last point of an end is left out.
        assert_eq!(chance(0.9993, [0.97, 1.0]).words(), ">99% (97-99%)");
        assert_eq!(chance(0.9993, [0.991, 0.9999]).words(), ">99%");
        assert_eq!(chance(0.002, [0.0004, 0.009]).words(), "<1%");
        assert_eq!(chance(0.02, [0.004, 0.07]).sentence(), "2%, likely between 1% and 7%");
        assert_eq!(percent(f64::NAN), "unknown");
    }

    #[test]
    fn the_range_widens_by_the_shift_between_sources() {
        // The 1.2 table's half width, 1.432 log odds: the odds 4.19 times either way. A chance of one half with no
        // interval of its own ranges from 19.3% to 80.7%, and one of 95% from 81.9% to 98.8%.
        let half_width = 1.431902;
        let at = |value: f64| Chance::at_even_odds(value, [value, value], 0.5).widened(half_width);
        let near = |a: f64, b: f64| (a - b).abs() < 5e-4;
        assert!(near(at(0.5).interval[0], 0.193) && near(at(0.5).interval[1], 0.807), "{:?}", at(0.5));
        assert!(near(at(0.95).interval[0], 0.819) && near(at(0.95).interval[1], 0.988), "{:?}", at(0.95));
        assert_eq!((at(0.5).value, at(0.5).between_sources, at(0.5).words()), (0.5, half_width, "50% (19-81%)".to_owned()));
        // The calibration's own interval is widened, not replaced, whichever way its ends are given, and certainty
        // at an end stays certainty.
        let wide = Chance::at_even_odds(0.6, [0.7, 0.5], 0.5).widened(half_width);
        assert!(near(wide.interval[0], 0.193) && wide.interval[1] > at(0.7).interval[1] - 1e-9 && wide.interval[1] < 0.91, "{wide:?}");
        assert_eq!(Chance::at_even_odds(0.5, [0.0, 1.0], 0.5).widened(half_width).interval, [0.0, 1.0]);
        // A table without the fit (Litmus 1.1.0 and earlier) leaves the range as the calibration gave it.
        let rule = ChanceRule { not_human_shares: vec![2.0 / 3.0, 0.5], between_sources: 0.0, level: None };
        assert_eq!(rule.chance(0.31, [0.25, 0.4], 1), Chance::at_even_odds(0.31, [0.25, 0.4], 0.5));
        let fitted = ChanceRule { between_sources: half_width, level: Some(0.9), ..rule };
        assert_eq!(fitted.chance(0.31, [0.25, 0.4], 1), Chance::at_even_odds(0.31, [0.25, 0.4], 0.5).widened(half_width));
        // A chance kept before the field was reads back with no widening.
        let kept: Chance = serde_json::from_str(r#"{"value":0.2,"interval":[0.1,0.3],"reference_share":0.5}"#).unwrap();
        assert_eq!(kept.between_sources, 0.0);
    }

    #[test]
    fn level_keys_are_pythons() {
        assert_eq!(level_key(0.05), "0.05");
        assert_eq!(level_key(0.1), "0.1");
        assert_eq!(merge_p_values(&[0.3]), 0.3);
        assert_eq!(merge_p_values(&[0.1, 0.2]), 0.30000000000000004);
        assert_eq!(merge_p_values(&[0.9, 0.8]), 1.0);
    }

    #[test]
    fn nearest_keeps_ties_in_index_order() {
        let mut sums = BankSums { shift: [0.0; 3], sum: [0.0; 3], squares: [0.0; 3], top: Vec::new(), keep: 3 };
        for (j, cosine) in [0.5, 0.9, 0.5, 0.7, 0.9, 0.1].into_iter().enumerate() {
            sums.add(0, j == 0, (cosine, j, 0.5));
        }
        assert_eq!(sums.top.iter().map(|near| near.1).collect::<Vec<_>>(), vec![1, 4, 3]);
    }
}
