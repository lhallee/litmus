---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: LICENSE
language:
- en
pipeline_tag: text-classification
tags:
- ai-text-detection
- machine-generated-text-detection
- bitnet
- ternary
- conformal-prediction
- calibration
---

<!--
Model card of a Litmus release; it becomes the model repository's README.md. A value that comes from a measurement or a record is a slot the release process fills from the release's data; nothing here is a measurement of the released model unless it says where it was measured. Sources: engine/README.md, teacher/CALIBRATION.md, student/SPEC.md, benchmarks/README.md,
LICENSING.md and LICENSE.
-->

# Litmus

Litmus is a free detector of AI written text that says how sure it is.

Litmus reads a text and says whether it looks written by a person, by a language model (AI), or by both (mixed), and
every reading carries its own uncertainty: the set of origins it cannot rule out at the 5% and 10% levels, and a
calibrated probability that the text is not human, with an interval. The model is a ternary (BitNet b1.58) encoder
with a pair head, trained on the true origins of its training texts, that scores a text against a bank of reference
texts and runs on one CPU core.

This release is an English detector: almost all of its training text and all of its held-out test text is English,
and other languages are not supported yet. Short texts are less reliable than long ones.

A reading is evidence about a text, never proof of who wrote it. The conformal sets and the interval exist so that no
one reads more into a single number than it can carry.

## Model details

| | |
| --- | --- |
| Developed by | Logan Hallee |
| Model type | BitNet b1.58 encoder: ternary weights (-1, 0, +1), 8-bit activations, block-local attention with blocks shifted by half on alternate layers, rotary positions, no global attention |
| Size | <!--litmus:t:released_size-->`s192_l4` (width 192, 4 layers, 1.79M ternary weights)<!--/litmus--> |
| Release file | `model.safetensors`, `config.json` (with the calibration table), `tokenizer.json` |
| Tokenizer | ModernBERT's (50,368 entries, byte-level, so no text is unknown) |
| Teacher | <!--litmus:record.model.teacher-->None: this release is trained on the true labels alone; labels alone read level with or ahead of distillation at equal steps (E4), and the second build has no teacher export<!--/litmus--> |
| Origins | `human` (0), `ai` (1), `mixed` (2) |
| Languages | English; <!--litmus:record.model.languages_reported-->98.8% of the training split is English and every held-out test text is English or undetermined, so this release is an English detector; other languages are read but not supported yet<!--/litmus--> |
| License | PolyForm Noncommercial 1.0.0 for the weights, the reference bank and the calibration tables; source available, not open source |
| Required notice | `Required Notice: Copyright Logan Hallee (https://github.com/lhallee/litmus)` |
| Code | https://github.com/lhallee/litmus |
| Predecessor | [lhallee/ai-vs-human-checker](https://huggingface.co/lhallee/ai-vs-human-checker) |
| Version | <!--litmus:release.version-->1.0.0<!--/litmus--> |

## Uses

**Intended.** Checking a text, a file or a folder of files for model-written passages, on a personal computer or in
the browser, by anyone using it noncommercially: individuals, researchers, schools, universities, charities, public
research bodies and government institutions. The page at the project's github.io site runs the model in the browser
and sends the text nowhere.

**Not licensed.** Commercial use, which the license does not permit.

**Our recommendations.** What someone does with a reading is their own decision. We recommend:

- not making a decision about a person on a reading alone, and weighing it as one piece of evidence beside drafts,
  version history, and the writer's account;
- not treating the strip position or the probability as a share of the text a model wrote;
- not relying on readings flagged as unlike the reference, or on text far from the calibration mix (languages,
  formats, or generators the reference holds little of), where the guarantees below do not hold.

## How it works

1. **Encoder.** The ternary encoder reads a text of up to 8,192 tokens and pools its last token states into their
   mean and variance. A pair head turns the pooled vector into a unit query and key, each stored as int8, and the
   same-origin logit of two texts is `scale * (q_a . k_b + q_b . k_a) / 2 + bias`.
2. **Training.** The encoder and head are trained on every pair across two batches of texts against the true
   same-origin labels, in the release's own arithmetic (ternary weights, integer attention, the int8 query and key)
   from the first step. Earlier builds also distilled from a frozen encoder teacher; labels alone read level with or
   ahead of distillation at equal steps, so this release has no teacher. Several sizes were compared; the release is
   <!--litmus:record.model.size_rule-->`s192_l4`, the smallest: in the release ablations `s256_l6` read 0.0059 PR AUC above it at equal steps, under the 0.01 a larger size must gain for 1.9 times the training time a step and about 2.7 times the arithmetic a token on one core<!--/litmus-->.
3. **Bank reading.** A text's query and key are compared with every vector of the reference bank. Per origin, the
   reading is the mean and the population spread of the same-origin probabilities, and the normalized means
   `r = mean / sum(mean)` give the three-way reading; the score is `1 - r[human]`. The bank holds vectors and
   metadata (source, generator, domain, era, format, language), never text.
4. **Calibration**, fitted on the `calibration` split alone, by origin and length group:
   split conformal p-values per origin and the sets at 0.05 and 0.10; an inductive Venn-Abers probability of not
   human with its interval; three-way probabilities; spread percentiles; and an out-of-reference flag when the
   closeness p-value is at or under 0.01. The tables ship in `config.json`.
5. **Long documents.** A text over 8,192 tokens is read in `ceil(n / 8192)` equal chunks, each a text of its own.
   The verdict merges the chunks: p-values as `min(1, 2 * mean)`, valid under any dependence (Vovk and Wang,
   Biometrika 2020, arXiv 1212.4966); the probability as the mean, its interval widened by the chunks' disagreement;
   a mixed-document signal at `0.05 / C` per chunk. A heat map of 128-token cells comes from windows of 512 tokens
   every 128, computed from running sums without a second pass. The map describes; the verdict decides.

The verdict's one word is taken from the set at 0.10: one origin names it, several read `uncertain`, none reads
`unlike reference`. The litmus strip places the reading at `P_ai + P_mixed / 2` from 0 (human) to 1 (AI), with a band
over what the 0.10 set cannot rule out.

## Training data

A private corpus compiled from public datasets only; nothing was generated for it. See the dataset card
(<!--litmus:target.hub.dataset_repo-->lhallee/litmus-corpus<!--/litmus-->) for the sources, the standard reading, the strata and the splits.

- Human text is dated before 2022 by a date column or the source's own date; model text names its generator; mixed
  text names how it was mixed where the source says.
- Text is kept raw (source code, non-English text, mixed and edited text, reasoning traces and chat scaffolding
  included), with a format stratum (`plain`, `markdown`, `list`, `table`, `code`, `math`, `chat`), language, domain
  and length bin on every row, and the classes balanced inside each cell as far as public data allows.
- Splits are whole clusters from e-means clustering of a frozen encoder's embeddings (ModernBERT-base), with the
  nearest clusters of every held-out cluster, and training rows sharing a document with a held-out row, removed:
  `train`, `validation` (checkpoint), `test` (read once), `calibration`, and mixed rows reserved for the bank. The
  held-out sets were chosen from English text only.
- The test split of every benchmark in the comparison is held out of every split.

| | Human | AI | Mixed |
| --- | --- | --- | --- |
| Training rows | <!--litmus:record.splits.rows[set=train].human as int-->6,657,360<!--/litmus--> | <!--litmus:record.splits.rows[set=train].ai as int-->6,355,055<!--/litmus--> | <!--litmus:record.splits.rows[set=train].mixed as int-->478,915<!--/litmus--> |
| Bank vectors | <!--litmus:model.bank.human as int-->2,048<!--/litmus--> | <!--litmus:model.bank.ai as int-->2,048<!--/litmus--> | <!--litmus:model.bank.mixed as int-->2,048<!--/litmus--> |
| Calibration rows | <!--litmus:record.splits.rows[set=calibration].human as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=calibration].ai as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=calibration].mixed as int-->4,995<!--/litmus--> |
| Test rows | <!--litmus:record.splits.rows[set=test].human as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=test].ai as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=test].mixed as int-->4,996<!--/litmus--> |

Bank construction, held-out clusters or a stratified draw from every training cluster: <!--litmus:record.model.bank_variant-->from training rows for human and AI text and from 4,096 reserved mixed rows that never trained, 2,048 an origin (the `mixed_out` recipe)<!--/litmus-->.

## Training procedure

<!--litmus:record.training.compute_summary-->Labels only, on one NVIDIA GH200: `s192_l4` trained for 32,500 steps of 1,024 texts (a third from each origin, drawn uniformly with replacement) in 3 h 5 min, with ternary weights, integer attention and the int8 output in the forward pass from the first step and the search's head and loss settings. The checkpoint was chosen on validation by a patience rule: once every origin had a full pass (step 19,466), training continued until ten readings in a row brought no new best validation PR AUC, and the release is the best reading in that region, reading 55 (step 27,499). Export, validation and one test read took 15 minutes; the release needs no GPU.<!--/litmus-->

## Evaluation

Every split and benchmark reports PR AUC, ROC AUC, balanced MCC (MCC on class-balanced subsamples at its best
threshold) and the true positive rate at 1% and 5% false positive rate, each with a stratified bootstrap 95% interval,
for three tasks: AI against human (mixed left out), AI or mixed against human, and mixed against human.

<!--litmus:t:overview_table-->
| Split | Reader | Task | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | TPR at 5% FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test | Litmus, as released | AI against human | 0.914 | 0.917 | 0.667 | 0.351 | 0.621 |
| Test | Litmus, as released | AI or mixed against human | 0.950 | 0.913 | 0.657 | 0.331 | 0.603 |
| Test | Litmus, as released | Mixed against human | 0.905 | 0.909 | 0.644 | 0.310 | 0.586 |
<!--/litmus-->

Outside benchmarks (one seeded sample of each, the same rows for every detector, official metric beside ours): RAID
extra's Czech and German news (its code rows are training rows), MAGE test, unseen-domain and paraphrase sets,
SemEval-2024 Task 8 A monolingual and multilingual, Beemo, APT-Eval and its human originals, MixSet, EditLens,
DroidCollection, CoDET-M4, AuTexTification English and Spanish, and HC3. RAID itself is scored only by its
leaderboard, and this release has not been submitted. The tables below are the benchmarks Litmus was scored on, with
every rerun open detector beside it.

<!--litmus:results_md.benchmarks-->
**autextification_en**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.914 (0.902 to 0.925) | 0.909 (0.898 to 0.919) | 0.683 (0.661 to 0.710) | 0.328 (0.227 to 0.426) | 1,510 / 1,490 |
| Desklib v1.01 | 0.954 (0.947 to 0.961) | 0.951 (0.943 to 0.958) | 0.768 (0.748 to 0.793) | 0.536 (0.499 to 0.605) | 1,510 / 1,490 |
| e5-small-lora | 0.769 (0.751 to 0.786) | 0.757 (0.740 to 0.773) | 0.386 (0.363 to 0.422) | 0.148 (0.120 to 0.183) | 1,510 / 1,490 |
| Fast-DetectGPT | 0.889 (0.878 to 0.899) | 0.861 (0.848 to 0.873) | 0.594 (0.573 to 0.624) | 0.425 (0.367 to 0.477) | 1,510 / 1,490 |
| ModernBERT RAID+MAGE (Drayson) | 0.851 (0.832 to 0.872) | 0.875 (0.862 to 0.887) | 0.620 (0.595 to 0.651) | 0.042 (0.012 to 0.148) | 1,510 / 1,490 |
| ai-vs-human-checker (predecessor) | 0.826 (0.811 to 0.841) | 0.815 (0.800 to 0.829) | 0.477 (0.456 to 0.510) | 0.224 (0.143 to 0.267) | 1,510 / 1,490 |
| **Litmus** | 0.718 (0.697 to 0.740) | 0.740 (0.723 to 0.758) | 0.405 (0.375 to 0.439) | 0.005 (0.000 to 0.026) | 1,510 / 1,490 |

**autextification_es**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.922 (0.914 to 0.930) | 0.892 (0.881 to 0.904) | 0.639 (0.620 to 0.670) | 0.430 (0.354 to 0.490) | 1,660 / 1,340 |
| Desklib v1.01 | 0.908 (0.898 to 0.918) | 0.887 (0.876 to 0.898) | 0.595 (0.573 to 0.625) | 0.353 (0.308 to 0.404) | 1,660 / 1,340 |
| e5-small-lora | 0.680 (0.662 to 0.699) | 0.620 (0.600 to 0.639) | 0.194 (0.171 to 0.232) | 0.062 (0.048 to 0.081) | 1,660 / 1,340 |
| Fast-DetectGPT | 0.869 (0.857 to 0.880) | 0.831 (0.817 to 0.845) | 0.515 (0.490 to 0.549) | 0.278 (0.221 to 0.330) | 1,660 / 1,340 |
| ModernBERT RAID+MAGE (Drayson) | 0.613 (0.595 to 0.634) | 0.538 (0.518 to 0.560) | 0.107 (0.090 to 0.146) | 0.044 (0.020 to 0.058) | 1,660 / 1,340 |
| ai-vs-human-checker (predecessor) | 0.563 (0.547 to 0.583) | 0.519 (0.498 to 0.540) | 0.083 (0.054 to 0.123) | 0.010 (0.003 to 0.024) | 1,660 / 1,340 |
| **Litmus** | 0.609 (0.591 to 0.628) | 0.580 (0.561 to 0.601) | 0.142 (0.115 to 0.180) | 0.000 (0.000 to 0.000) | 1,660 / 1,340 |

**Beemo**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.960 (0.954 to 0.966) | 0.788 (0.763 to 0.814) | 0.459 (0.436 to 0.535) | 0.125 (0.064 to 0.181) | 2,352 / 327 |
| Desklib v1.01 | 0.959 (0.950 to 0.966) | 0.797 (0.771 to 0.822) | 0.500 (0.455 to 0.556) | 0.050 (0.014 to 0.136) | 2,352 / 327 |
| e5-small-lora | 0.926 (0.915 to 0.937) | 0.690 (0.659 to 0.719) | 0.326 (0.277 to 0.378) | 0.015 (0.004 to 0.031) | 2,352 / 327 |
| Fast-DetectGPT | 0.901 (0.890 to 0.912) | 0.549 (0.515 to 0.582) | 0.115 (0.098 to 0.185) | 0.040 (0.025 to 0.060) | 2,352 / 327 |
| ModernBERT RAID+MAGE (Drayson) | 0.917 (0.904 to 0.929) | 0.678 (0.642 to 0.710) | 0.336 (0.292 to 0.400) | 0.010 (0.003 to 0.019) | 2,352 / 327 |
| ai-vs-human-checker (predecessor) | 0.952 (0.943 to 0.961) | 0.780 (0.756 to 0.807) | 0.449 (0.406 to 0.501) | 0.011 (0.005 to 0.133) | 2,352 / 327 |
| **Litmus** | 0.948 (0.938 to 0.959) | 0.795 (0.764 to 0.825) | 0.531 (0.487 to 0.587) | 0.008 (0.002 to 0.027) | 2,352 / 327 |

**CoDET-M4**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.906 (0.896 to 0.917) | 0.886 (0.873 to 0.899) | 0.662 (0.637 to 0.692) | 0.357 (0.294 to 0.446) | 1,522 / 1,477 |
| Desklib v1.01 | 0.755 (0.735 to 0.776) | 0.772 (0.756 to 0.789) | 0.422 (0.396 to 0.454) | 0.068 (0.051 to 0.089) | 1,522 / 1,477 |
| e5-small-lora | 0.580 (0.561 to 0.600) | 0.614 (0.594 to 0.633) | 0.212 (0.186 to 0.249) | 0.009 (0.003 to 0.022) | 1,522 / 1,477 |
| Fast-DetectGPT | 0.499 (0.486 to 0.515) | 0.535 (0.515 to 0.556) | 0.134 (0.110 to 0.171) | 0.001 (0.000 to 0.004) | 1,522 / 1,477 |
| ModernBERT RAID+MAGE (Drayson) | 0.454 (0.440 to 0.469) | 0.429 (0.409 to 0.450) | 0.057 (0.029 to 0.091) | 0.004 (0.001 to 0.009) | 1,522 / 1,477 |
| ai-vs-human-checker (predecessor) | 0.590 (0.570 to 0.611) | 0.598 (0.578 to 0.618) | 0.172 (0.146 to 0.210) | 0.026 (0.010 to 0.044) | 1,522 / 1,477 |
| **Litmus** | 0.643 (0.626 to 0.665) | 0.713 (0.696 to 0.732) | 0.371 (0.356 to 0.408) | 0.005 (0.002 to 0.015) | 1,522 / 1,477 |

**DroidCollection**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.844 (0.829 to 0.860) | 0.831 (0.815 to 0.848) | 0.572 (0.549 to 0.606) | 0.347 (0.204 to 0.421) | 1,205 / 1,430 |
| Desklib v1.01 | 0.607 (0.583 to 0.632) | 0.665 (0.644 to 0.685) | 0.256 (0.228 to 0.299) | 0.037 (0.025 to 0.056) | 1,205 / 1,430 |
| e5-small-lora | 0.493 (0.474 to 0.517) | 0.519 (0.497 to 0.542) | 0.070 (0.055 to 0.116) | 0.019 (0.008 to 0.038) | 1,205 / 1,430 |
| Fast-DetectGPT | 0.571 (0.549 to 0.599) | 0.609 (0.588 to 0.630) | 0.177 (0.161 to 0.223) | 0.022 (0.014 to 0.048) | 1,205 / 1,430 |
| ModernBERT RAID+MAGE (Drayson) | 0.412 (0.398 to 0.428) | 0.446 (0.424 to 0.469) | 0.097 (0.069 to 0.135) | 0.008 (0.003 to 0.014) | 1,205 / 1,430 |
| ai-vs-human-checker (predecessor) | 0.572 (0.550 to 0.597) | 0.604 (0.583 to 0.625) | 0.189 (0.156 to 0.226) | 0.029 (0.018 to 0.053) | 1,205 / 1,430 |
| **Litmus** † | 0.620 (0.600 to 0.646) | 0.719 (0.700 to 0.739) | 0.381 (0.348 to 0.416) | 0.000 (0.000 to 0.008) | 1,205 / 1,430 |

**editlens**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.959 (0.952 to 0.966) | 0.952 (0.943 to 0.961) | 0.774 (0.753 to 0.803) | 0.638 (0.576 to 0.718) | 999 / 996 |
| Desklib v1.01 | 0.989 (0.986 to 0.992) | 0.988 (0.984 to 0.991) | 0.899 (0.882 to 0.920) | 0.866 (0.829 to 0.906) | 999 / 996 |
| e5-small-lora | 0.850 (0.830 to 0.869) | 0.846 (0.829 to 0.862) | 0.559 (0.533 to 0.597) | 0.169 (0.078 to 0.230) | 999 / 996 |
| Fast-DetectGPT | 0.727 (0.706 to 0.749) | 0.685 (0.661 to 0.707) | 0.325 (0.295 to 0.369) | 0.167 (0.134 to 0.202) | 999 / 996 |
| ModernBERT RAID+MAGE (Drayson) | 0.908 (0.886 to 0.931) | 0.946 (0.934 to 0.956) | 0.830 (0.805 to 0.857) | 0.080 (0.035 to 0.177) | 999 / 996 |
| ai-vs-human-checker (predecessor) | 0.994 (0.992 to 0.996) | 0.993 (0.989 to 0.996) | 0.935 (0.923 to 0.952) | 0.931 (0.896 to 0.953) | 999 / 996 |
| **Litmus** | 0.978 (0.964 to 0.989) | 0.987 (0.981 to 0.991) | 0.932 (0.915 to 0.948) | 0.717 (0.304 to 0.924) | 999 / 996 |

**hc3**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.988 (0.981 to 0.993) | 0.993 (0.988 to 0.996) | 0.957 (0.947 to 0.969) | 0.952 (0.902 to 0.969) | 988 / 2,012 |
| Desklib v1.01 | 0.998 (0.996 to 0.999) | 0.999 (0.997 to 0.999) | 0.968 (0.962 to 0.979) | 0.977 (0.962 to 0.987) | 988 / 2,012 |
| e5-small-lora | 0.886 (0.869 to 0.902) | 0.934 (0.924 to 0.943) | 0.744 (0.722 to 0.772) | 0.382 (0.343 to 0.474) | 988 / 2,012 |
| Fast-DetectGPT | 0.662 (0.635 to 0.689) | 0.767 (0.748 to 0.787) | 0.456 (0.423 to 0.488) | 0.141 (0.099 to 0.179) | 988 / 2,012 |
| ModernBERT RAID+MAGE (Drayson) | 0.954 (0.935 to 0.974) | 0.988 (0.984 to 0.991) | 0.916 (0.905 to 0.932) | 0.786 (0.572 to 0.843) | 988 / 2,012 |
| ai-vs-human-checker (predecessor) | 0.998 (0.996 to 0.999) | 0.999 (0.998 to 0.999) | 0.970 (0.960 to 0.979) | 0.977 (0.959 to 0.987) | 988 / 2,012 |
| **Litmus** | 0.991 (0.987 to 0.993) | 0.994 (0.992 to 0.996) | 0.937 (0.921 to 0.949) | 0.894 (0.872 to 0.949) | 988 / 2,012 |

**MAGE**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.682 (0.660 to 0.703) | 0.624 (0.602 to 0.646) | 0.378 (0.350 to 0.407) | 0.095 (0.062 to 0.170) | 1,415 / 1,585 |
| Desklib v1.01 | 0.930 (0.922 to 0.938) | 0.929 (0.920 to 0.937) | 0.690 (0.668 to 0.719) | 0.543 (0.496 to 0.599) | 1,415 / 1,585 |
| e5-small-lora | 0.720 (0.695 to 0.743) | 0.761 (0.743 to 0.778) | 0.381 (0.360 to 0.414) | 0.039 (0.009 to 0.077) | 1,415 / 1,585 |
| Fast-DetectGPT | 0.648 (0.628 to 0.666) | 0.569 (0.548 to 0.589) | 0.318 (0.295 to 0.348) | 0.196 (0.166 to 0.227) | 1,415 / 1,585 |
| ModernBERT RAID+MAGE (Drayson) † | 0.984 (0.977 to 0.989) | 0.986 (0.982 to 0.989) | 0.899 (0.885 to 0.915) | 0.823 (0.764 to 0.859) | 1,415 / 1,585 |
| ai-vs-human-checker (predecessor) † | 0.868 (0.856 to 0.880) | 0.853 (0.839 to 0.867) | 0.568 (0.543 to 0.599) | 0.413 (0.383 to 0.451) | 1,415 / 1,585 |
| **Litmus** † | 0.790 (0.773 to 0.805) | 0.796 (0.779 to 0.810) | 0.435 (0.407 to 0.464) | 0.253 (0.179 to 0.285) | 1,415 / 1,585 |

**MAGE unseen**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.977 (0.971 to 0.982) | 0.969 (0.961 to 0.977) | 0.854 (0.834 to 0.882) | 0.764 (0.724 to 0.819) | 800 / 762 |
| Desklib v1.01 | 0.990 (0.987 to 0.993) | 0.988 (0.983 to 0.991) | 0.884 (0.866 to 0.906) | 0.860 (0.825 to 0.902) | 800 / 762 |
| e5-small-lora | 0.950 (0.940 to 0.959) | 0.951 (0.941 to 0.960) | 0.762 (0.740 to 0.795) | 0.472 (0.401 to 0.549) | 800 / 762 |
| Fast-DetectGPT | 0.721 (0.695 to 0.749) | 0.638 (0.609 to 0.666) | 0.374 (0.333 to 0.423) | 0.124 (0.075 to 0.174) | 800 / 762 |
| ModernBERT RAID+MAGE (Drayson) † | 0.900 (0.876 to 0.927) | 0.929 (0.915 to 0.941) | 0.734 (0.703 to 0.768) | 0.102 (0.043 to 0.215) | 800 / 762 |
| ai-vs-human-checker (predecessor) † | 0.975 (0.966 to 0.981) | 0.973 (0.966 to 0.979) | 0.815 (0.796 to 0.848) | 0.716 (0.624 to 0.784) | 800 / 762 |
| **Litmus** | 0.958 (0.948 to 0.965) | 0.957 (0.947 to 0.964) | 0.805 (0.777 to 0.832) | 0.456 (0.422 to 0.604) | 800 / 762 |

**MAGE paraphrased**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.802 (0.785 to 0.820) | 0.662 (0.640 to 0.685) | 0.259 (0.223 to 0.299) | 0.059 (0.046 to 0.087) | 1,600 / 762 |
| Desklib v1.01 | 0.935 (0.927 to 0.943) | 0.873 (0.858 to 0.887) | 0.573 (0.542 to 0.608) | 0.334 (0.286 to 0.411) | 1,600 / 762 |
| e5-small-lora | 0.913 (0.904 to 0.922) | 0.833 (0.817 to 0.848) | 0.528 (0.486 to 0.556) | 0.275 (0.208 to 0.324) | 1,600 / 762 |
| Fast-DetectGPT | 0.613 (0.599 to 0.630) | 0.420 (0.394 to 0.446) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.001) | 1,600 / 762 |
| ModernBERT RAID+MAGE (Drayson) † | 0.822 (0.806 to 0.841) | 0.719 (0.698 to 0.743) | 0.345 (0.306 to 0.385) | 0.034 (0.015 to 0.072) | 1,600 / 762 |
| ai-vs-human-checker (predecessor) † | 0.941 (0.932 to 0.950) | 0.891 (0.876 to 0.904) | 0.616 (0.583 to 0.652) | 0.341 (0.265 to 0.424) | 1,600 / 762 |
| **Litmus** | 0.952 (0.945 to 0.958) | 0.909 (0.896 to 0.921) | 0.633 (0.610 to 0.664) | 0.389 (0.366 to 0.501) | 1,600 / 762 |

**MixSet**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.859 (0.837 to 0.879) | 0.499 (0.437 to 0.557) | 0.210 (0.181 to 0.285) | 0.100 (0.072 to 0.164) | 500 / 100 |
| Desklib v1.01 | 0.897 (0.871 to 0.919) | 0.660 (0.600 to 0.718) | 0.322 (0.222 to 0.409) | 0.074 (0.028 to 0.108) | 500 / 100 |
| e5-small-lora | 0.865 (0.838 to 0.896) | 0.607 (0.544 to 0.675) | 0.212 (0.152 to 0.342) | 0.010 (0.002 to 0.064) | 500 / 100 |
| Fast-DetectGPT | 0.817 (0.790 to 0.844) | 0.420 (0.361 to 0.481) | 0.058 (0.024 to 0.175) | 0.024 (0.002 to 0.076) | 500 / 100 |
| ModernBERT RAID+MAGE (Drayson) | 0.875 (0.843 to 0.908) | 0.636 (0.574 to 0.694) | 0.287 (0.200 to 0.397) | 0.004 (0.000 to 0.032) | 500 / 100 |
| ai-vs-human-checker (predecessor) | 0.783 (0.756 to 0.813) | 0.463 (0.391 to 0.527) | 0.108 (0.043 to 0.191) | 0.000 (0.000 to 0.000) | 500 / 100 |
| **Litmus** | 0.844 (0.818 to 0.871) | 0.547 (0.475 to 0.611) | 0.212 (0.119 to 0.307) | 0.000 (0.000 to 0.090) | 500 / 100 |

**RAID extra (cs, de)**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.991 (0.990 to 0.993) | 0.761 (0.727 to 0.793) | 0.531 (0.449 to 0.609) | 0.292 (0.278 to 0.402) | 2,911 / 88 |
| Desklib v1.01 † | 0.996 (0.994 to 0.997) | 0.883 (0.846 to 0.919) | 0.687 (0.602 to 0.773) | 0.365 (0.350 to 0.486) | 2,911 / 88 |
| e5-small-lora † | 0.992 (0.990 to 0.993) | 0.773 (0.737 to 0.806) | 0.506 (0.479 to 0.603) | 0.388 (0.372 to 0.464) | 2,911 / 88 |
| Fast-DetectGPT | 0.979 (0.972 to 0.984) | 0.600 (0.547 to 0.647) | 0.324 (0.234 to 0.425) | 0.001 (0.000 to 0.055) | 2,911 / 88 |
| ModernBERT RAID+MAGE (Drayson) † | 0.994 (0.991 to 0.997) | 0.885 (0.837 to 0.929) | 0.739 (0.668 to 0.811) | 0.091 (0.081 to 0.170) | 2,911 / 88 |
| ai-vs-human-checker (predecessor) † | 0.987 (0.985 to 0.990) | 0.700 (0.656 to 0.747) | 0.368 (0.308 to 0.478) | 0.122 (0.110 to 0.288) | 2,911 / 88 |
| **Litmus** | 0.993 (0.992 to 0.994) | 0.807 (0.776 to 0.834) | 0.598 (0.544 to 0.631) | 0.474 (0.456 to 0.541) | 2,911 / 88 |

**SemEval-24 mono**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.940 (0.932 to 0.947) | 0.896 (0.883 to 0.909) | 0.812 (0.795 to 0.831) | 0.805 (0.769 to 0.828) | 1,574 / 1,426 |
| Desklib v1.01 | 1.000 (0.999 to 1.000) | 0.999 (0.999 to 1.000) | 0.989 (0.984 to 0.994) | 0.997 (0.989 to 0.999) | 1,574 / 1,426 |
| e5-small-lora | 0.922 (0.913 to 0.931) | 0.918 (0.908 to 0.927) | 0.681 (0.661 to 0.710) | 0.356 (0.302 to 0.451) | 1,574 / 1,426 |
| Fast-DetectGPT | 0.476 (0.461 to 0.494) | 0.336 (0.318 to 0.356) | 0.073 (0.051 to 0.110) | 0.019 (0.011 to 0.042) | 1,574 / 1,426 |
| ModernBERT RAID+MAGE (Drayson) | 0.897 (0.878 to 0.917) | 0.950 (0.940 to 0.959) | 0.857 (0.839 to 0.875) | 0.001 (0.000 to 0.008) | 1,574 / 1,426 |
| ai-vs-human-checker (predecessor) † | 0.997 (0.996 to 0.998) | 0.996 (0.993 to 0.998) | 0.975 (0.968 to 0.982) | 0.982 (0.975 to 0.989) | 1,574 / 1,426 |
| **Litmus** † | 0.995 (0.993 to 0.997) | 0.994 (0.992 to 0.996) | 0.944 (0.933 to 0.956) | 0.945 (0.901 to 0.958) | 1,574 / 1,426 |

**SemEval-24 multi**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| Binoculars | 0.929 (0.921 to 0.936) | 0.887 (0.875 to 0.900) | 0.707 (0.687 to 0.730) | 0.651 (0.621 to 0.693) | 1,594 / 1,406 |
| Desklib v1.01 | 0.973 (0.969 to 0.977) | 0.964 (0.958 to 0.969) | 0.812 (0.795 to 0.834) | 0.780 (0.738 to 0.812) | 1,594 / 1,406 |
| e5-small-lora | 0.827 (0.815 to 0.840) | 0.758 (0.742 to 0.775) | 0.505 (0.482 to 0.533) | 0.361 (0.291 to 0.392) | 1,594 / 1,406 |
| Fast-DetectGPT | 0.633 (0.616 to 0.651) | 0.489 (0.468 to 0.511) | 0.229 (0.213 to 0.262) | 0.119 (0.100 to 0.142) | 1,594 / 1,406 |
| ModernBERT RAID+MAGE (Drayson) | 0.826 (0.806 to 0.847) | 0.842 (0.827 to 0.857) | 0.619 (0.592 to 0.647) | 0.004 (0.000 to 0.027) | 1,594 / 1,406 |
| ai-vs-human-checker (predecessor) † | 0.947 (0.941 to 0.953) | 0.926 (0.917 to 0.935) | 0.724 (0.704 to 0.750) | 0.662 (0.627 to 0.691) | 1,594 / 1,406 |
| **Litmus** † | 0.899 (0.885 to 0.913) | 0.916 (0.906 to 0.926) | 0.682 (0.661 to 0.709) | 0.122 (0.095 to 0.154) | 1,594 / 1,406 |

**Litmus test**

| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |
| --- | --- | --- | --- | --- | --- |
| **Litmus** | 0.914 (0.905 to 0.923) | 0.917 (0.909 to 0.926) | 0.667 (0.646 to 0.691) | 0.351 (0.325 to 0.378) | 2,000 / 2,000 |

† trained on this benchmark (its training split, or its test texts). ‡ read a different sample of the benchmark than the other detectors, so its intervals are not paired with theirs. Intervals are bootstrap 95%.
<!--/litmus-->

What the comparison cannot claim: protocols differ between sources, so quoted numbers appear only beside the same
benchmark, subset and metric; some other detectors trained on RAID or MAGE; Litmus's held-out claim covers the
cataloged benchmark splits only; one sample per benchmark supports its intervals and no more.

## Uncertainty and coverage

The conformal guarantee: for human text exchangeable with the calibration set, the share called not human at level
`a` is at most `a`, in each length group. It covers text from the same sources, formats, languages and generators in
the same mix, and nothing else.

<!--litmus:table.coverage_bands-->
| Where | Coverage at 0.05 | Coverage at 0.10 | Mean set size at 0.10 | Human called not human at 0.05 |
| --- | --- | --- | --- | --- |
| Test, all | 94.4% | 89.5% | 1.64 | 5.9% |
| Test, 1-128 tokens | 94.2% | 89.2% | 1.82 | 6.2% |
| Test, 129-512 tokens | 94.7% | 90.2% | 1.18 | 4.1% |
| Test, 513-2048 tokens | 95.4% | 91.6% | 1.00 | 6.5% |
| Test, 2049-8192 tokens | 95.9% | 90.4% | 0.92 | 6.5% |
| Long documents, merged chunks | 98.1% | 96.2% | 0.98 | 3.8% |
<!--/litmus-->

<!--litmus:record.findings.coverage_failures-->On test the 10% set covers 0.8952 of passages against the 0.90 it targets (human 0.8970, AI 0.9055, mixed 0.8830), and 10.3% of human passages are called not human at the 10% level against at most 10%, so coverage falls just short for mixed text and human passages here.<!--/litmus--> <!--litmus:t:spliced_sentence-->On spliced documents the sliding windows gave 78.3% of cells their true origin and placed 55.9% of boundaries within one cell (128 tokens) of the truth. The window context gap on real text, over interior windows, averaged 0.014 in position and 0.139 in the pooled vector.<!--/litmus-->

## Speed and memory

Measured on releases of the nine candidate sizes with random weights and a bank of 4,096 texts per origin (engine
README): AMD Ryzen 7 3800X, Windows 10, one core, AVX2, the fastest of 5 runs on
a machine 40 to 45% busy with other work. A whole scoring call ran at 4,214 (d384 L8) to 21,926 (d192 L4) tokens per
second on 8,192 tokens, flat from 1,024 to 131,072 tokens, with a process peak of 56 to 70 MB. In WebAssembly (Node 22,
one thread) d192 L4 scored 3,400 to 4,000 tokens per second.

<!--litmus:t:speed_paragraph-->The released size (`s192_l4`) reads 28,022 tokens per second on an 8,192-token text with a process peak of 37.9 MB, on AMD Ryzen 7 3800X 8-Core Processor under Windows 10 (one core, avx2 kernels). Its release files are 10.8 MB, and its single-core cost is 53 ms per 1,000 words, beside the other detectors in the benchmark tables.<!--/litmus-->

## Bias, risks and limitations

- **False flags happen by design.** At the 5% level, up to 5% of human text like the calibration texts is called
  not human. How likely a flagged text is to be AI depends on how much AI text is among the texts checked.
- **Human text predates 2022.** Every human training and reference text is dated before 2022, so text about later
  events, or in later styles, has no human counterpart in the reference; whether that biases readings of recent human text is <!--litmus:record.findings.recent_human_text-->unmeasured: no held-out set holds human text written after 2021<!--/litmus-->.
- **Long model text is scarce** in public data and nothing was generated to fill it, so readings of long model text
  rest on less evidence. Chunks of long documents are not drawn like the calibration texts, a stated limit of the
  merged sets.
- **Heat-map windows** are read with neighbors in view, so a window's reading is close to, not equal to, the reading
  of those tokens alone.
- **Mixed labels** come from the sources' own descriptions of how a text was mixed.
- **Languages other than English** have less data. <!--litmus:record.findings.language_coverage-->No held-out set of this release holds a non-English text: 98.8% of the training split is English and the test, calibration and validation sets were chosen from English and undetermined-language rows only, so other languages are measured by the benchmarks alone (RAID extra's Czech and German news, AuTexTification Spanish, SemEval-2024's multilingual set).<!--/litmus-->

## How to use

Command line: `litmus score [--json] [--map] [--explain] <file | folder | ->` with `--model` pointing to the release
folder. Python:

```python
import litmus_engine

model = litmus_engine.Model.from_path("path/to/release")
reading = model.score(text)
reading["label"]                                   # "human", "AI", "mixed", "uncertain" or "unlike reference"
reading["verdict"]["probability"], reading["verdict"]["interval"], reading["verdict"]["strip"]
```

Every reading can explain itself: the calibration counts behind each p-value ("of `n` human calibration texts of this
length, `k` read at most this human"), the five nearest bank texts with their metadata, and each chunk's reading.

## License

PolyForm Noncommercial 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0). Pass the license and the
Required Notice line along with every copy. Documentation and figures are CC BY-NC 4.0.

## Citation

<!--litmus:t:citation-->Hallee, L. (2026). Litmus: a free detector of AI written text that says how sure it is [Computer software]. https://github.com/lhallee/litmus<!--/litmus-->.
