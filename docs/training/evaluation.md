# 7. Evaluation

[Front page](../TRAINING.md) · [1. Data](data.md) · [2. Splits](splits.md) · [3. Model](model.md) ·
[4. Training](training.md) · [5. Calibration](calibration.md) · [6. Experiments](experiments.md) · **7. Evaluation** ·
[8. Engine](engine.md) · [9. Reproduce](reproduce.md)

The released model is measured three ways: on its own **test split**, read once after every choice was made; on
**public benchmarks**, beside other detectors scored the same way; and for **cost**, on one processor core. The
results page of the website and [`benchmarks/README.md`](../../benchmarks/README.md) show the numbers; this part says
how they are made.

- [The metrics](#the-metrics)
- [The held-out test split](#the-held-out-test-split)
- [Long and spliced documents](#long-and-spliced-documents)
- [The public benchmarks](#the-public-benchmarks)
- [The detectors compared](#the-detectors-compared)
- [Intervals](#intervals)
- [Cost on one core](#cost-on-one-core)
- [The release's results](#the-releases-results)
- [Where in the code](#where-in-the-code)

## The metrics

Detection is scored on $`\mathrm{score} = 1 - r_{\mathrm{human}}`$ (or a calibrated probability of not human, which ranks
texts the same way within a length group), with AI text (and, in some tasks, mixed text) as the **positive** class
and human text as the **negative**. Every benchmark and every slice reports all of these, always together:

| Metric | What it measures | Formula |
| --- | --- | --- |
| **PR AUC** (average precision) | How well the positives rank above the negatives, weighted toward the top of the ranking | $`\mathrm{AP} = \sum_k (R_k - R_{k-1})\, P_k`$, over thresholds at each positive, with $`P`$ precision and $`R`$ recall |
| **ROC AUC** | The chance that a random positive scores above a random negative | $`\Pr(\mathrm{score}^{+} \gt  \mathrm{score}^{-})`$ |
| **Balanced MCC** | Correlation between the call and the truth at the best threshold, on class-balanced subsamples so the class ratio does not move it | $`\dfrac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}`$ |
| **TPR at 1% and 5% FPR** | The share of AI text caught when the threshold flags only 1% (or 5%) of human text | threshold at the human scores' 99th (95th) percentile |

*Worked example (made up).* Four human texts score 0.1, 0.3, 0.4, 0.8 and four AI texts 0.35, 0.6, 0.7, 0.9.

| Metric | Computation | Value |
| --- | --- | --- |
| ROC AUC | 12 of the 16 (AI, human) pairs have the AI text higher | 0.750 |
| PR AUC | Ranked: AI, human, AI, AI, human, AI, human, human; precision at each AI text 1/1, 2/3, 3/4, 4/6 | 0.771 |
| MCC at threshold 0.5 | TP 3, FN 1, TN 3, FP 1: $`(9 - 1) / \sqrt{4 \cdot 4 \cdot 4 \cdot 4}`$ | 0.500 |
| Best-threshold MCC | At 0.35: TP 4, FP 2, TN 2, FN 0: $`8 / \sqrt{6 \cdot 4 \cdot 2 \cdot 4}`$ | 0.577 |
| TPR at 25% FPR | Threshold above 0.4, so one human text (0.8) of four is flagged; AI flagged: 0.6, 0.7, 0.9 | 75% |

**Tasks.** `ai_vs_human` leaves mixed text out; `any_ai_vs_human` counts mixed text as AI; `mixed_vs_human` asks
whether mixed text is told from human. **Calibration** is measured on the same texts: the coverage of the conformal
sets (the share whose true origin is in the set) and their mean size at 5% and 10%, and the share of human texts called
not human, by length bin, format, language, generator, and domain.

## The held-out test split

The test split ([2. The splits](splits.md)) is read **once**, after the size, the checkpoint, the bank, and the
calibration were chosen on validation. Passages of each origin, spread evenly over the test clusters
(<!--litmus:record.evaluation.passages_per_origin as int-->2,000<!--/litmus--> an origin), are read as a user's text is, through the release's head, bank, and calibration,
and reported overall and by token band (1 to 128, 129 to 512, 513 to 2,048, 2,049 to 8,192 tokens), with every metric
above and the conformal coverage and set size. The release build chose its held-out sets from English and
undetermined-language rows only ([2. The splits](splits.md#step-3-give-whole-clusters-to-sets-with-moats)), so the
test split measures English text; other languages are measured only by the benchmarks that hold them.

## Long and spliced documents

Two readings test what training never sees directly:

- **Long documents.** The corpus's whole documents of 8,193 to 131,072 tokens (never trained on) are read in chunks
  and judged by their combined verdict, by number of chunks (2, 3 to 4, 5 to 8, 9 to 16): the detection metrics, the
  coverage and size of the merged sets, the share of human documents called not human, and how often the
  mixed-document signal fires for each origin.
- **Spliced documents.** Held-out human and AI passages are joined into documents of one chunk with known boundaries
  (human then AI, AI then human, human-AI-human, AI-human-AI). For each heat-map cell, the map is right when the cell's
  position falls on its true origin's side of 0.5; each boundary is located from the cells and its error reported in
  tokens; and the same documents are mapped by non-overlapping windows too, to measure whether sliding windows help.
- **Window context gap.** Each window's vector came from token states that saw neighbors outside the window. Per
  window, the gap between that reading and the reading of the same tokens alone, and whether their 10% sets agree.

## The public benchmarks

The benchmarks are listed exactly in [`benchmarks/catalog.csv`](../../benchmarks/catalog.csv): repository,
configuration, split, and revision. Each is scored on one seeded sample of at most 3,000 texts
(`--max-rows 3000 --seed 0`, as [`release/baselines.toml`](../../release/baselines.toml) pins), the same sampled rows
for every detector, beside its own official metric.

| Benchmark | What it tests | Official metric | For Litmus |
| --- | --- | --- | --- |
| RAID test | 11 models, 11 genres, 4 decoding strategies, 12 adversarial attacks; labels hidden | TPR at 5% FPR with per-domain thresholds, via the leaderboard | In distribution: RAID's training split is trained on. Scored only by the leaderboard, to which this release has not been submitted |
| RAID extra | Python code, Czech news, German news | as RAID, computed locally | Czech and German: fully unseen. Code: <!--litmus:record.corpus.raid_extra_code-->trained on (its 244,947 code rows are training rows), so code is not scored<!--/litmus--> |
| MAGE test | 27 generators, 10 human sources | AvgRec and AUROC | In distribution: MAGE's training split is trained on |
| MAGE out of distribution | Unseen domain and model (GPT-4); and its paraphrased version | AvgRec and AUROC | Fully unseen |
| SemEval-2024 Task 8, subtask A | Monolingual and multilingual | Accuracy | In distribution: its training split is trained on |
| Beemo | Expert- and model-edited model text | AUROC per pair of versions | Fully unseen |
| MixSet | Mixed text by 12 mixing operations | F1 | Fully unseen |
| APT-Eval and its human originals | AI-polished human text | Share flagged, by polishing degree | Fully unseen |
| DroidCollection | Code in 9 languages, with refined and adversarial machine code | F1 | In distribution: its training and development splits are trained on |
| CoDET-M4 | Code in Python, Java, C++ | F1, accuracy | Fully unseen |
| HC3 | Human and ChatGPT answers to the same questions | None official | Fully unseen |
| EditLens | Human-written, AI-edited, and AI-generated text | Accuracy and macro F1 over the three | Fully unseen |
| AuTexTification, English and Spanish | Human and generated text from anonymized generators | Macro F1 | Fully unseen |

<!--litmus:record.corpus.catalog_check-->
Every catalogued benchmark split was read whole and held apart (2,578,146 rows), and a training row that shares a text, an opening, a link or a source document with a benchmark row is held apart with it: 111,904 rows of the first corpus were newly held apart this way, 92,890 of them COLING 2025 training rows that share source documents with its dev set. The code rows of RAID extra (244,947) train, so its benchmark is the Czech and German news alone.
<!--/litmus-->

Each result is reported overall and by domain, generator, attack, and length bin (1 to 50, 51 to 150, 151 to 400,
401 to 1,000, over 1,000 words).

## The detectors compared

Other detectors are rerun on the same sampled texts where they are open, and only quoted where they are commercial
or published elsewhere. A detector trained on a benchmark's training split is in distribution on that benchmark, and
the tables say so.

| Detector | Kind | In distribution on |
| --- | --- | --- |
| Litmus | This model | RAID, MAGE, SemEval-2024 Task 8, DroidCollection ([1. The data](data.md#benchmark-families-trained-on-and-fully-unseen)) |
| The predecessor, `ai-vs-human-checker` | Frozen ModernBERT-base, pair head, bank | Its corpus read MAGE's test split, Ghostbuster's test split, HC3, and SemEval-2024's training and development data, so its numbers on those are contaminated |
| e5-small-lora | Classifier | RAID (trained on RAID's training split, per its card) |
| ModernBERT RAID+MAGE (Drayson) | Classifier | RAID and MAGE (per its card) |
| Desklib v1.01 | Classifier | not stated in our sources |
| Binoculars (Falcon-7B pair) | Zero-shot: compares two language models' perplexities, no detector training | none |
| Fast-DetectGPT (Falcon-7B-instruct) | Zero-shot: conditional probability curvature, no detector training | none |
| Commercial detectors and leaderboard entries | Quoted from their sources, never rerun | as each source states |

Quoted numbers are drawn only beside the same benchmark, subset, and metric, because different protocols are not one
ranking ([`benchmarks/README.md`](../../benchmarks/README.md), What the comparison cannot claim).

## Intervals

Every metric carries a **stratified bootstrap 95% interval**: the texts are resampled with replacement within each
class, 1,000 times overall and 200 times per slice, and the interval is the 2.5th to 97.5th percentile of the metric
over the resamples. The same resamples are used for every detector, so a difference between two detectors is paired.
A sample of at most 3,000 texts carries the stated intervals and no more.

## Cost on one core

Each detector runs alone in a fresh process on one thread of one machine (`OMP_NUM_THREADS=1` set before Python,
`torch.set_num_threads(1)`), on the same 200 texts after one warm-up, at batch size 1, in float32 for PyTorch detectors
and as shipped for Litmus's engine. The result is milliseconds per 1,000 whitespace words (no tokenizer favored), with
peak resident memory and model size on disk. The two 7B zero-shot methods are measured on 20 texts, and the result says
so. The cost figure plots TPR at 1% FPR against this cost and marks measured and estimated points apart.

## The release's results

| Measure | Value |
| --- | --- |
| Test split, AI against human: PR AUC, ROC AUC, balanced MCC, TPR at 1% and 5% FPR, with 95% intervals | <!--litmus:t:test_ai_vs_human-->PR AUC 0.937 (0.931 to 0.944), ROC AUC 0.934 (0.926 to 0.941), balanced MCC 0.697, TPR at 1% FPR 0.530, at 5% FPR 0.671<!--/litmus--> |
| Test split, AI or mixed against human; mixed against human | <!--litmus:t:test_other_tasks-->AI or mixed against human: PR AUC 0.945, ROC AUC 0.900; mixed against human: PR AUC 0.861, ROC AUC 0.866<!--/litmus--> |
| Conformal coverage and mean set size at 5% and 10% on test, by length band | <!--litmus:t:test_coverage_all-->94.3% at 5% and 90.1% at 10%, mean set size 2.08 and 1.87 (by token band: the table of the model card)<!--/litmus--> |
| Share of human test texts called not human at 5% and 10% | <!--litmus:t:test_human_called-->7.4% at 5% and 12.5% at 10%<!--/litmus--> |
| Long documents, by number of chunks | <!--litmus:t:test_long_documents-->PR AUC 0.998, ROC AUC 0.999, coverage 85.7% at 10% (merged chunks; by number of chunks in `evaluation_test.csv`)<!--/litmus--> |
| Spliced documents: cell accuracy and boundary error, sliding against fixed windows | <!--litmus:t:test_spliced-->sliding windows: cell accuracy 78.4%, median boundary error 105.00 tokens; fixed sections: cell accuracy 77.8%, median boundary error 124.00 tokens<!--/litmus--> |
| Window context gap | <!--litmus:t:test_window_gap-->mean position gap 0.012 (95th percentile 0.059), mean pooled gap 0.112, on interior windows<!--/litmus--> |
| Every benchmark, every detector | <!--litmus:t:benchmarks_pointer-->6 other detectors on 14 benchmarks: the tables of [DEVELOPING.md](../DEVELOPING.md#results-in-full)<!--/litmus--> |
| RAID leaderboard: TPR at 5% and 1% FPR, all attacks and none | <!--litmus:record.evaluation.raid_leaderboard-->Not submitted: RAID's test labels are hidden, and a submission is made only on the maintainer's word<!--/litmus--> |
| Cost: ms per 1,000 words, peak memory, size on disk | <!--litmus:t:cost_row-->349 ms per 1,000 words on one core (Arm Neoverse-V2), a process peak of 65.1 MB, 18.5 MB on disk<!--/litmus--> |

## Where in the code

| What | Function or file |
| --- | --- |
| Test, long, and spliced readings; the window gap | `litmus_student.evaluate`; `reference.window_context_gap` |
| Coverage and set size by stratum | `litmus_teacher.calibrate.uncertainty_rows` |
| Benchmark loading and the held-out catalog | `benchmarks/catalog.csv`; `litmus_benchmarks` (`score`) |
| Metrics with bootstrap intervals | `litmus_benchmarks.evaluate.bootstrap_metrics`; `foundry.eval.bootstrap.stratified_bootstrap_indices`; `foundry.metrics.balanced_mcc` |
| Single-core cost | `litmus_benchmarks` (`cost`) |
| Figures | `litmus_benchmarks` (`figures`) |

---

Previous: [6. The experiments](experiments.md) · Next: [8. The engine](engine.md)
