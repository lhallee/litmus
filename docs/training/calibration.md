# 5. Calibration and the reading

[Front page](../TRAINING.md) · [1. Data](data.md) · [2. Splits](splits.md) · [3. Model](model.md) ·
[4. Training](training.md) · **5. Calibration** · [6. Experiments](experiments.md) · [7. Evaluation](evaluation.md) ·
[8. Engine](engine.md) · [9. Reproduce](reproduce.md)

The model gives a text three **shares**, $`r_{\mathrm{human}}`$, $`r_{\mathrm{AI}}`$, $`r_{\mathrm{mixed}}`$
([3. The model](model.md#the-reference-bank)). A share of 0.57 is not a probability: what it means depends on how
texts of known origin, and of about the same length, score. **Calibration** answers that by comparison with the
**calibration split**, texts the model never trained on, read through the released model and bank when the release is
exported. It fits a small table (about 27 KB of JSON) that ships inside the release, and every reading applies it.
The table's exact layout is [`teacher/CALIBRATION.md`](../../teacher/CALIBRATION.md); the code is
`litmus_teacher.calibrate` (`fit_calibration` fits, `apply_calibration` applies).

- [Length groups](#length-groups)
- [Conformal p-values and sets](#conformal-p-values-and-sets)
- [What Uncertain means](#what-uncertain-means)
- [The probability of not human, with an interval](#the-probability-of-not-human-with-an-interval)
- [Three-way probabilities and the strip](#three-way-probabilities-and-the-strip)
- [The mechanism's own signals](#the-mechanisms-own-signals)
- [Long documents: combining chunks](#long-documents-combining-chunks)
- [A reading worked end to end](#a-reading-worked-end-to-end)
- [What the guarantee covers](#what-the-guarantee-covers)
- [Where in the code](#where-in-the-code)

## Length groups

A short text carries little evidence, so its shares scatter more than a long text's. Every part of the table is
therefore fitted per **length group**. Token counts fall into bins that double from 16:
$`[0, 16), [16, 32), \dots, [4096, 8192), [8192, 32768)`$, and $`[32768, \infty)`$. Walking from the shortest bin,
bins are merged into a group until every cell the table needs holds at least 200 calibration texts, and a short last
group joins the one before. A text uses the group that holds its own length (a window, the group of 512 tokens).

## Conformal p-values and sets

**Split conformal prediction** asks, for each origin $`o`$ in turn: *if this text were of origin $`o`$, how unusual
would its reading be among calibration texts that really are of origin $`o`$ and of its length?* The **conformity**
of the hypothesis “origin $`o`$” is the text's own share $`r_o`$, which needs no fitting. With $`n`$ calibration texts of
origin $`o`$ in the text's length group, and $`r_o^{(1)}, \dots, r_o^{(n)}`$ their shares for $`o`$:

```math
p_o = \frac{1 + \#\{\, i : r_o^{(i)} \le r_o \,\}}{n + 1}
```

the share of origin-$`o`$ calibration texts that looked at most as much like $`o`$ as this text does, counting the text
itself. A small $`p_o`$ means few real texts of origin $`o`$ look this little like $`o`$, so $`o`$ is **ruled out**:

- the **set** at level $`\alpha`$ is $`\{\, o : p_o \gt  \alpha \,\}`$, reported at $`\alpha = 0.05`$ and $`0.10`$;
- the text is **called not human** at level $`\alpha`$ when $`p_{\mathrm{human}} \le \alpha`$.

This is Mondrian conformal prediction: calibrated separately for each true origin and length group, so the bound below
holds within each.

**The guarantee.** If a human text is exchangeable with the human calibration texts of its length group (drawn the same
way, from the same kinds of sources), the chance that it is called not human at level $`\alpha`$ is at most
$`\alpha`$. This bounds false flags of human writing. It says nothing about how much AI text is caught; that is
measured ([7. Evaluation](evaluation.md)).

**How the table stores it.** The table does not store calibration scores for this formula; it stores, for each level
$`u`$ of a fixed grid (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, ..., 0.10, 0.15, 0.2, 0.25, 0.3, 0.4, ..., 0.9), the
order statistic $`t_u = r_o^{(\lfloor u(n+1) \rfloor)}`$ (the $`\lfloor u(n+1) \rfloor`$-th smallest calibration share).
Then $`p_o \le u`$ exactly when $`r_o \lt  t_u`$, and the reported p-value is the smallest grid level $`u`$ with
$`r_o \lt  t_u`$, or 1. That is the exact p-value rounded up to the grid, so set membership at 0.05 and 0.10 is exact. The
release also ships the sorted calibration shares, so a reading can state its count in words: “of 19 human calibration
texts of this length, 1 read at most this human”.

### Worked example

Made-up numbers, computed with `grid_thresholds` and `grid_p_values`. A length group holds $`n = 19`$ human
calibration texts whose human shares are

0.12, 0.21, 0.28, 0.33, 0.37, 0.40, 0.44, 0.47, 0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.71, 0.75, 0.81.

| Text's $`r_{\mathrm{human}}`$ | Calibration texts at or below it | Exact $`p_{\mathrm{human}}`$ | Grid $`p`$ | Human in the set at 0.05? | at 0.10? | Called not human at 0.10? |
| --- | --- | --- | --- | --- | --- | --- |
| 0.20 | 1 (0.12) | (1 + 1) / 20 = 0.10 | 0.10 | yes | **no** | **yes** |
| 0.30 | 3 | (1 + 3) / 20 = 0.20 | 0.20 | yes | yes | no |
| 0.567 | 11 | (1 + 11) / 20 = 0.60 | 0.60 | yes | yes | no |

The grid check for the first row: $`t_{0.10}`$ is the $`\lfloor 0.10 \times 20 \rfloor = 2`$nd smallest share, 0.21, and
$`0.20 \lt  0.21`$, so $`p \le 0.10`$. With so few texts the smallest possible p-value is $`1/20 = 0.05`$, reached only by a
text below every calibration share; a real group holds at least 200 texts, and its p-values reach 0.005.

## What Uncertain means

The verdict word comes from the set at the 10% level:

```mermaid
flowchart LR
  S{"Origins not ruled out<br>at the 10% level"} -->|"exactly one"| A["Human, AI, or Mixed"]
  S -->|"two or three"| B["Uncertain:<br>the strip's band shows which"]
  S -->|"none"| C["Unlike reference"]
```

*The verdict follows from the set. A text far from the bank is also marked unlike the reference, whatever its set.*

**Uncertain** is an answer, not a failure: it says the evidence cannot separate two or three origins at this level.
It is common for short texts, for mixed text (which by construction resembles both of its parents), and for kinds of
text the calibration set holds few of. An **empty set** means that every origin is ruled out: the text looks less like
each origin than nearly all calibration texts of that origin do, so it resembles none of the reference closely enough
for a call.

## The probability of not human, with an interval

A p-value is not a probability that the text is AI. For a probability, Litmus uses **inductive Venn-Abers**
calibration ([Vovk and Petej, 2014](https://arxiv.org/abs/1211.0025)) on the two-class score
$`1 - r_{\mathrm{human}}`$, per length group:

1. The calibration scores of the group are cut at their quantiles into at most 32 **blocks** of at least 50 texts.
2. For a text that falls in block $`g`$, an increasing (isotonic) fit of “not human” against the blocks is made twice:
   once with the text added as human, giving $`p_0`$, and once with it added as not human, giving $`p_1 \ge p_0`$.
3. The **interval** is $`[p_0, p_1]`$, and the **probability** is

```math
P = \frac{p_1}{1 - p_0 + p_1}
```

Venn predictors are calibrated in a precise sense for any score, with no assumed shape. The interval is narrow where
many calibration texts scored alike and wide where they are scarce, so it shows the evidence behind the number.

*Worked example (made up; `venn_abers_blocks` on 200 calibration scores, 100 human and 100 not).* The scores fall into
four blocks of 50, with edges 0.241, 0.450, and 0.726:

| Block | Score range | Not human of 50 | $`p_0`$ | $`p_1`$ | Probability |
| --- | --- | --- | --- | --- | --- |
| 0 | below 0.241 | 0 | 0.000 | 0.020 | 0.019 |
| 1 | 0.241 to 0.450 | 10 | 0.196 | 0.216 | 0.212 |
| 2 | 0.450 to 0.726 | 41 | 0.804 | 0.824 | 0.808 |
| 3 | 0.726 and above | 49 | 0.961 | 0.980 | 0.962 |

A text scoring 0.433 is in block 1: added as human, the block holds 10 of 51 not human, $`p_0 = 0.196`$; added as not
human, 11 of 51, $`p_1 = 0.216`$. Its probability of not human is $`0.216 / (1 - 0.196 + 0.216) = 0.212`$, with the
interval $`[0.196, 0.216]`$. (Here the block rates already increase, so the isotonic fit changes nothing.)

## Three-way probabilities and the strip

The **three-way probabilities** are a temperature fit with a per-origin offset, per length group:

```math
P_o = \mathrm{softmax}_o\big(\beta \cdot \ln \max(r_o, 10^{-12}) + c_o\big), \qquad c_{\mathrm{human}} = 0
```

with $`\beta`$ and $`c_o`$ fitted by Newton's method on the calibration texts' log loss. They are a reading, not a set.

**The litmus strip** draws them: the **position** is $`x = P_{\mathrm{AI}} + P_{\mathrm{mixed}} / 2`$, from 0 (human, blue)
through 0.5 (mixed, purple) to 1 (AI, red). Each origin has an anchor (human 0, mixed 0.5, AI 1), and the **band**
covers what the 10% set cannot rule out: from $`\min(x, \text{smallest anchor in the set})`$ to
$`\max(x, \text{largest anchor in the set})`$. A set of one origin draws the band from $`x`$ to its anchor; a set of all
three, the whole strip.

## The mechanism's own signals

- **Spread.** Per origin, the spread of the pair probabilities against that origin's bank, reported as a percentile
  among calibration texts of the group. A high percentile says the bank disagrees about this text more than usual.
- **Closeness.** The mean of the 16 highest cosines to any bank text, turned into a p-value against the calibration
  texts of the group (all origins pooled) exactly as above. At 0.01 or below, the text is **out of reference**: it
  sits farther from the bank than 99% of calibration texts of its length, and the strip is drawn as unlike the
  reference.
- **Explanation.** Every chunk's reading carries its evidence: for each origin, the calibration count behind its
  p-value; for closeness, the same; and the five nearest bank texts with their origin, cosine, pair probability,
  token count, source, generator, domain, era, format, and language.

## Long documents: combining chunks

A document longer than 8,192 tokens is read as $`C`$ chunks ([3. The model](model.md#long-texts-chunks-windows-and-cells)),
each calibrated at its own length. The chunks of one document resemble each other, so no rule that assumes they are
independent is used.

**p-values.** Per origin, twice the chunks' mean, capped at 1:

```math
p_o = \min\!\Big(1,\ \frac{2}{C} \sum_{c=1}^{C} p_{c,o}\Big)
```

Twice the arithmetic mean of p-values is a valid p-value under any dependence between them, and no smaller factor
suffices in general ([Vovk and Wang, 2020](https://arxiv.org/abs/1212.4966), published in *Biometrika*). Sets and
calls follow from the merged p-values as for one text.

- **Probabilities.** The mean over chunks (a linear pool, which assumes nothing about dependence). The interval is
  $`[\min(\overline{p_0}, P - s),\ \max(\overline{p_1}, P + s)]`$, clipped to $`[0, 1]`$, with $`s`$ the standard deviation
  of the chunks' probabilities: the mean Venn-Abers interval when the chunks agree, wider when they do not. It
  describes and carries no coverage guarantee.
- **Mixed document.** With $`a = 0.05`$, a chunk is a **model chunk** when its human p-value is at most $`a / C`$, and a
  **human chunk** when its AI and mixed p-values both are. The signal fires when one chunk is a model chunk and a
  different chunk a human chunk. By the union bound over chunks, valid under any dependence, a wholly human document
  fires it with probability at most $`a`$, and so does a wholly AI one.

*Worked example (made up).* Three chunks with human p-values 0.04, 0.30, and 0.09 merge to
$`\min(1, 2 \times 0.1433) = 0.287`$: human is not ruled out for the document, though one chunk alone would rule it out at
5%. For the mixed-document signal the chunk level is $`0.05 / 3 = 0.0167`$; if chunk 1's human p-value were 0.004 and
chunk 3's AI and mixed p-values 0.010 and 0.012, chunk 1 would be a model chunk and chunk 3 a human chunk, and the
signal would fire, locating the change.

## A reading worked end to end

The text of the [bank example](model.md#the-reference-bank), shares $`r = (0.567, 0.081, 0.352)`$ and score 0.433, in a
length group whose human calibration texts are those of the conformal example. Made-up values are marked.

| Step | Value |
| --- | --- |
| Human p-value | 11 of 19 human calibration texts at or below 0.567: $`p = 12/20 = 0.60`$ |
| AI and mixed p-values (made up) | 0.01 and 0.30 |
| Set at 10% | {human, mixed}: AI is ruled out (0.01 ≤ 0.10) |
| Verdict | **Uncertain**: human or mixed |
| Probability of not human | score 0.433 is in Venn-Abers block 1: 0.212, interval [0.196, 0.216] |
| Three-way probabilities (made-up fit with $`\beta = 1`$, $`c = 0`$) | equal to the shares, 0.567, 0.081, 0.352 |
| Strip | position $`x = 0.081 + 0.352/2 = 0.257`$; band from 0 (human's anchor) to 0.5 (mixed's anchor) |

## What the guarantee covers

The conformal bound holds for text exchangeable with the calibration set: the same sources, formats, languages, and
generators in the same mix. Text from a new model, a language other than English, or an unusual format is not, so
coverage and set size are measured on the test split and on every outside benchmark, by length, format, language,
generator, and domain, and where coverage fails off-distribution the results say so ([7. Evaluation](evaluation.md)).
The release's calibration and test sets hold English and undetermined-language text only, so its guarantee speaks
for English text, and other languages are measured only where a benchmark holds them. Calibration
texts are whole texts and passages of at most 8,192 tokens, not chunks of longer documents; that difference is a stated
limit, measured on the corpus's long documents.

The release's table was fitted on <!--litmus:t:calibration_fit-->19,996 human, 19,996 AI, and 4,995 mixed calibration texts, in 12 length groups<!--/litmus-->.

## Where in the code

| Formula | Function (`litmus_teacher.calibrate`) | Engine side (`student/reference.py`) |
| --- | --- | --- |
| Shares and score | `three_way` (shares), `not_human_score` | `ReferenceModel` |
| Length bins and groups | `strata.length_bins`, `merge_bins`, `group_of_bins` | `group_index` |
| Conformal thresholds and grid p-values | `grid_thresholds`, `grid_p_values` | `Calibration` |
| Venn-Abers blocks, $`p_0, p_1`$ | `venn_abers_blocks`, `isotonic` | `Calibration` |
| Three-way fit | `fit_three_way` | `Calibration` |
| Fit and apply the whole table | `fit_calibration`, `apply_calibration` | `Calibration` |
| Strip and band | | `litmus_strip` |
| Merging chunks; the mixed-document signal | `merge_p_values`, `combine_chunks` | `ReferenceModel` |
| Explanation counts | | `Explanation`, `explain_tensors` |
| Fitting the release's table on its own bank readings | `litmus_student.export` | |

---

Previous: [4. Training](training.md) · Next: [6. The experiments](experiments.md)
