# The Litmus calibration table

`calibration.json` turns a scorer's readings of a text into a calibrated verdict: a conformal set of origins with
p-values, a probability of not human with an interval, three-way probabilities, and the mechanism's own signals. It
is fitted on the `calibration` split alone by `litmus_teacher.calibrate.fit_calibration`, from the teacher's or the
student's readings alike, and applied by `apply_calibration`; the student's `SPEC.md` embeds it and the Rust engine
reads it. `apply_calibration` is the reference: an engine matches it on the same readings and table. A table fitted
on synthetic readings of 30,000 texts is 27 KB of JSON.

## What the scorer must provide

For each text, against a bank of vectors of each origin (`human`, `ai`, `mixed`, in that order):

| Reading | Shape | Definition |
| --- | --- | --- |
| `mean` | (3,) | Mean over the origin's bank vectors of `sigmoid(pair_logit(text, bank vector))` |
| `spread` | (3,) | Population standard deviation of those probabilities |
| `closeness` | () | Mean of the `top_k` highest cosines between the text's query-and-key vector and any bank vector, all origins pooled |
| `tokens` | () | ModernBERT tokens of the text without special tokens (the corpus `tokens` column) |

`pair_logit(a, b) = scale * (q_a . k_b + q_b . k_a) / 2 + bias`, with `q` and `k` the unit halves of a vector.

Derived from them: `r = mean / sum(mean)` (the normalized three-way reading) and `score = 1 - r[human]` (the
two-class reading, ai or mixed against human).

## Top-level keys

| Key | Meaning |
| --- | --- |
| `format`, `version` | `"litmus-calibration"`, `1`; refuse anything else |
| `origins` | `["human", "ai", "mixed"]`: the order of every per-origin array |
| `length_edges` | Token edges; bin `i` holds `[edges[i], edges[i + 1])`, the last bin is open above. Bin of `t` = the number of edges after the first that are `<= t` |
| `length_bins` | Readable bin names, for display |
| `levels` | `[0.05, 0.1]`: the levels every reading reports a set at |
| `grid` | The p-value grid, ascending; 1 is implicit |
| `spread_levels` | Quantile levels of the `spread` tables |
| `top_k` | The `k` of `closeness` |
| `out_of_reference_level` | `0.01`: closeness p-value at or under which a text is flagged out of reference |
| `min_rows`, `calibration_rows` | The merge threshold and the rows fitted on, for the record |
| `conformal`, `probability`, `three_way`, `spread`, `closeness` | The five tables below |

Every table is a list of **length groups**. Each group has `bins` (the length bins it covers, consecutive) and `n`
(calibration rows behind it). The groups of one table cover every length bin exactly once, so every text finds its
group. Groups differ between tables, since each merges thin bins for its own cells (rule below). Floats are rounded
to 7 significant digits; `null` stands for minus infinity.

## Conformal sets: `conformal`

`conformal` maps each origin name to its list of groups; each group has `thresholds`, aligned with `grid`.

For a text, for each origin `y`: find the group of `y` that holds the text's length bin; the p-value is the smallest
`grid[j]` with `r[y] < thresholds[j]` (skip `null`), or 1 if none. Origin `y` is in the set at level `a` when its
p-value is greater than `a`. The text is **called not human at level `a` when its human p-value is at or under
`a`**. A set of more than one origin reads as uncertain; an empty set as unlike the reference.

Why this is exact: `thresholds[j]` is the `floor(grid[j] * (n + 1))`-th smallest `r[y]` among calibration texts
of origin `y` in the group, so `r[y] < thresholds[j]` exactly when the split conformal p-value
`(1 + #{calibration r[y] <= r}) / (n + 1)` is at most `grid[j]`. The reported p-value is therefore an upper bound,
equal to the true one's grid ceiling, and set membership at 0.05 and 0.1 is exact. For human text exchangeable with
the calibration set, the share called not human is at most the level, in each length group.

## Probability of not human: `probability`

Inductive Venn-Abers on `score`. Each group has `edges` (ascending block boundaries) and `p0`, `p1` (one more entry
than `edges`). Block `g` = the number of `edges` that are `<= score`. Then `p0 = p0[g]`, `p1 = p1[g]`, the
interval is `[p0, p1]` and the probability is `p1 / (1 - p0 + p1)`.

## Three-way probabilities: `three_way`

Each group has `slope` and `bias` (3 values, human first, always 0). The probabilities are
`softmax(slope * log(max(r, 1e-12)) + bias)`.

## Spread percentile: `spread`

Each group has, per origin name, the quantiles of `spread[origin]` among calibration texts of the group (all origins
pooled) at `spread_levels`. A text's percentile is linear interpolation of its spread on those quantiles, clamped to
the first and last level. A high percentile says the text's pair probabilities disagree across that bank more than
usual for its length.

## Out of reference: `closeness`

Each group has `thresholds` aligned with `grid`, built like the conformal ones from the calibration texts' closeness
(all origins pooled). The closeness p-value is the smallest `grid[j]` with `closeness < thresholds[j]`, else 1, and
the text is out of reference when it is at or under `out_of_reference_level`: it sits farther from the bank than
99% of calibration texts of its length.

## Long documents: chunks

A document of more than 8,192 tokens is read in `C` chunks of equal length, each a text of its own (the student's
`SPEC.md`, Chunks), and the table is applied to each chunk at its own length. `combine_chunks` makes the
document's verdict from the chunks' verdicts; with one chunk it returns that chunk's values unchanged.

- **p-values**: per origin, `min(1, 2 * mean_c p_c)` over the chunks' grid p-values. Twice the arithmetic mean of
  p-values is a p-value under any dependence between them, and no smaller factor suffices in general (Vovk and Wang,
  "Combining p-values via averaging", Biometrika 2020, arXiv 1212.4966, after Ruschendorf 1982 and Meng 1993).
  Chunks of one document resemble each other, so no rule that assumes independence is used. The sets and the
  not-human call at each level follow from the merged p-values as for one text, and keep their bound for a document
  whose every chunk is exchangeable with the calibration texts of its length; that chunks of long documents are
  not drawn like the calibration texts is a stated limit, measured by the student's long-document evaluation.
- **Probabilities**: the probability of not human and the three-way probabilities are the means over chunks (a
  linear pool, which assumes nothing about dependence; a product of chunk probabilities would count shared evidence
  `C` times). The interval is `[min(mean p0, P - s), max(mean p1, P + s)]`, clipped to `[0, 1]`, with `P` the
  pooled probability and `s` the population standard deviation of the chunks' probabilities (`disagreement`), so it
  is the mean Venn-Abers interval when the chunks agree and widens when they do not. It describes; it carries no
  coverage guarantee.
- **Out of reference**: the chunks' closeness p-values merged the same way, at or under `out_of_reference_level`.
- **Mixed document**: with `a` the level (0.05) and `C >= 2`, a chunk is a **model chunk** when its human p-value
  is at or under `a / C`, and a **human chunk** when its ai and mixed p-values both are. The signal fires when one
  chunk is a model chunk and a different chunk a human chunk: the document is part human and part model, and the
  chunks say where. By the union bound over chunks (Bonferroni, valid under any dependence), a wholly human
  document fires it with probability at most `a`, and so does a wholly ai one.

`section_uncertainty` (a summary over sections of one reading) is kept for the teacher's own use; the student's
reading no longer calls it.

## How length groups are merged

Per table, the cells are: `conformal`, the calibration texts of that origin; `probability`, human and not human;
`three_way`, each origin; `spread` and `closeness`, all texts. Walk the length bins from the shortest, adding each
to the current group, and close the group once every cell holds at least `min_rows` (200) rows; a short last group
joins the one before it. A fit whose rows over all lengths fall under `min_rows` in a cell raises an error.

## What the guarantees cover

The conformal bound holds for text exchangeable with the calibration set: the same sources, formats, languages and
generators in the same mix. Coverage and set size are measured on `test` and on every outside benchmark, by length
bin, format, language, generator and domain (`uncertainty.csv`); where coverage fails off-distribution, the card
says so.
