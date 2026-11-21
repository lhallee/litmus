# Reading a Litmus result, field by field

`litmus score --json` prints one object a line per input: `{"file", "sha256", "cached", "label", "chance", "reading",
"counts"}`, or `{"file", "sha256", "error"}` when that input could not be read. The MCP tools return the same `label`,
`chance`, and `reading` in `structuredContent` (`reading` holds `n_tokens`, `chunk_tokens`, and `verdict` at the default
`"summary"` detail, and every field below at `"full"`), plus `n_tokens`, `words`, `characters`, `caveats`, `guidance`,
`detail`, and for the file tool `file`. The full definitions are in [`student/SPEC.md`](../../../../student/SPEC.md) and
[`engine/README.md`](../../../../engine/README.md).

`counts` (`words`, `characters`, `characters_no_spaces`, `tokens`) sizes the text that was read, as the user gave it and
before any normalizing: `characters` are Unicode scalar values, a word is a run of non-whitespace (so a language written
without spaces counts as one word per run), and `tokens` is `reading.n_tokens`. It is a statistic of the text, not part of
the reading.

## The chance of AI, `chance`

| Field | Type | Meaning |
| --- | --- | --- |
| `value` | number | The chance of AI: the verdict's probability of not human restated at even odds, for a text as likely human as not beforehand. This is the number the app shows and the one to report |
| `interval` | `[low, high]` | The chance's range: the probability's interval restated the same way, then widened on each side by how far the chance moves between sources of text (`between_sources`). It is short near either end and wide in the middle, and a long text's is wider still when its chunks disagree. The app's strip draws its mark at `value` and its band over this range |
| `reference_share` | number | The share of not-human texts among the calibration texts of this length, which the restating divides out (about two thirds for short texts, one half for long ones) |
| `between_sources` | number | The half width, in log odds, the range was widened by on each side: the low end's odds are divided by `exp(between_sources)` and the high end's multiplied by it. It holds the chance for 90% of the pairings of one human source with one AI or mixed source of calibration text. 0 with the model of Litmus 1.1.0 and earlier, whose range is the calibration's interval alone |

## The verdict, `reading.verdict`

| Field | Type | Meaning |
| --- | --- | --- |
| `sets` | `{"0.05": [origin...], "0.1": [origin...]}` | The origins not ruled out at each level, from `human`, `ai`, `mixed`. One origin is a call, `ai` and `mixed` together are the call `AI or mixed` (human ruled out), human beside another is uncertain, none is unlike the reference. `label` names the 0.1 set |
| `p_values` | `[human, ai, mixed]` | The conformal p-value of each origin: how typical this text is of that origin among calibration texts of its length. An origin is ruled out at a level when its p-value is at or under it |
| `not_human` | `{"0.05": bool, "0.1": bool}` | `p_human` at or under the level. Among human texts like the calibration texts, at most that share are called so |
| `probability`, `interval` | number, `[low, high]` | The probability of not human, with its interval, calibrated on texts mixed as the calibration texts of this length are. `chance` restates both at even odds. For several chunks, the mean, with the interval widened by the chunks' disagreement |
| `three_way` | `[human, ai, mixed]` | Calibrated probabilities of each origin, summing to one |
| `strip` | `{position, band_lo, band_hi, level, unlike_reference}` | The reading's place between the origins: `position` 0 is human, 0.5 mixed, 1 AI, and `band_lo` to `band_hi` spans the origins the 0.1 set cannot rule out. The heat map is colored by `position`. The strip the app draws now marks `chance.value` and bands `chance.interval` instead |
| `out_of_reference`, `closeness_p` | bool, number | Whether the text is unlike every reference text, and the p-value of its closeness to them |
| `chunks`, `lowest`, `highest`, `disagreement` | | How many chunks the text was read in, the lowest and highest chunk probability, and their spread |
| `mixed_document`, `model_chunks`, `human_chunks` | | One chunk reads as model-written and a different one as human, tested at a family level of 0.05 across the chunks |

## The rest of `reading`

| Field | Meaning |
| --- | --- |
| `n_tokens`, `chunk_tokens` | The text's length in tokens, and the chunk size (8,192). Over that, the text is read as `ceil(n / 8192)` equal chunks |
| `chunks` | Each chunk's own reading, with its place and an `explain` block: the calibration counts behind each p-value and the five nearest reference texts with their metadata |
| `windows` | Windows of 512 tokens every 128 inside each chunk, each read as a text of its own length |
| `cells` | The heat map: one cell per 128 tokens, with the overlap-weighted mean `position` of the windows over it (0 human, 0.5 mixed, 1 AI), their `spread`, and the widest span of origins they keep |
| `start_char`, `end_char` (also `start_byte`, `end_byte`, `start`, `end`) | A chunk, window, or cell's place in the original text, in Unicode characters, bytes, and tokens |

## What a reading can and cannot show

- It shows which origins the text resembles, calibrated against held-out texts. It does not show who wrote the text,
  how much of a mixed text a model wrote, or why.
- A cell with a high `position` marks where to look in a long text. It is not a verdict on that passage alone, and the
  document's `label` decides.
- The reference texts are mostly English prose from before 2022 (human) and from named generators (AI). A text
  unlike them gets `unlike reference` or a wide set, and the right reading of either is "no call".
- Short texts are less reliable: a sentence or two carries little evidence, so the set stays wide and the chance of AI
  stays nearer one half.
- This release is an English detector: the model is trained almost entirely on English (<!--litmus:record.corpus.english_share as pct1-->98.8%<!--/litmus--> of its training text) and tested mainly on English text; other languages are not supported yet.
- Accuracy: for the release model, <!--litmus:t:accuracy_summary-->on held-out test texts it wrongly called 7.4% of human texts not human at the 5% level, and it caught 67.2% of AI texts when 5% of human texts are wrongly flagged (53.0% at 1%)<!--/litmus-->.
