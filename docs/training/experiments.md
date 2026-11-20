# 6. The experiments

[Front page](../TRAINING.md) · [1. Data](data.md) · [2. Splits](splits.md) · [3. Model](model.md) ·
[4. Training](training.md) · [5. Calibration](calibration.md) · **6. Experiments** · [7. Evaluation](evaluation.md) ·
[8. Engine](engine.md) · [9. Reproduce](reproduce.md)

Every design choice that was tested has a card here, in the same form: what the experiment asked, in plain terms and
then precisely; its arms, data, and budget; what was measured and how the outcome was decided; the result with its
numbers and the number of seeds; what it changed in the design; and what it does not show.

**E1 to E10** were run on earlier data, before the release build, and their numbers describe those runs, not the
release. **A1 to A5** are the short equal-budget ablations that tested the release's recipe before its run; they ran
on the first build, distilled from its teacher, while the release trains on the second build with labels alone. Every
metric is defined
in [7. Evaluation](evaluation.md#the-metrics); “PR AUC, human against AI” leaves mixed text out.

| Card | Question | Outcome |
| --- | --- | --- |
| [E1](#e1-ternary-weights-against-full-precision) | Do ternary weights cost accuracy? | No measurable cost |
| [E2](#e2-a-per-token-teacher-loss) | Does matching the teacher at every token help? | No gain, three times the cost |
| [E3](#e3-integer-attention) | Does integer attention cost accuracy? | No measurable cost; faster with SIMD |
| [E4](#e4-distillation-from-a-frozen-teacher-against-labels-only) | Does distillation beat labels alone? | No: labels alone level or ahead |
| [E5](#e5-model-size) | Do larger encoders learn more per step? | Larger is ahead on MCC at equal steps, early |
| [E6](#e6-readout-pooled-pairs-token-pairs-or-token-labels) | Which readout reads best? | Pooled pairs with the bank |
| [E7](#e7-searching-the-teachers-pair-head) | What does searching the teacher's head buy? | 0.047 lower validation pair loss |
| [E8](#e8-searching-the-distillation-settings) | Which head and loss settings for the student? | Hidden 256, output 64, clamp 30 |
| [E9](#e9-bank-held-out-clusters-or-a-draw-from-training-clusters) | Which bank reads better? | The draw from training clusters |
| [E10](#e10-skipping-zero-inputs-of-the-down-product) | Can the engine skip the feed-forward's zeros? | Yes: 3% to 19% faster, same bits |
| [A1 to A5](#the-release-ablations) | Weight levels, float-first, pooling, output width, sparsity | <!--litmus:record.ablations.summary-->Nothing beat the recipe by its margin: ternary weights with one scale a tensor, quantized from step 0, mean and variance pooling, a 64-wide output, no sparsity penalty; the int8 output is trained in at no measurable cost<!--/litmus--> |

## The earlier data

- **The predecessor corpus** ([`lhallee/many-generator-detection-corpus`](https://huggingface.co/datasets/lhallee/many-generator-detection-corpus)):
  5.17 million prose passages of two origins, human and AI, with stored ModernBERT-base embeddings, and the
  predecessor detector ([`lhallee/ai-vs-human-checker`](https://huggingface.co/lhallee/ai-vs-human-checker)): a frozen
  ModernBERT-base with a pair head and a bank. Used by E1 to E3, before the new corpus existed.
- **The first build** of this repository's corpus: 12,653,367 rows (5,110,107 human, 7,186,256 AI, 357,004 mixed),
  compiled from the priority-1 sources with a budget of 2e9 new training tokens, embedded with ModernBERT-base, and
  split into 3,934 e-means clusters. It did not yet include benchmark training splits. Used by E4 to E9.

---

## E1. Ternary weights against full precision

**Finding: ternary weights and 8-bit activations cost nothing measurable at the smallest size.**

*In plain terms.* Three small students learned side by side from the same examples: two with every weight −1, 0, or
+1, and one identical twin with ordinary floating-point weights. After an hour, the ternary student of the same size
detected AI text as well as its twin, and a larger ternary student did slightly better.

| | |
| --- | --- |
| **Question** | Does quantizing weights to ternary and activations to 8 bits cost detection accuracy, against the same architecture in full precision on the same batches? |
| **Arms** | Ternary `s192_l4`; ternary `s256_l6`; a full-precision twin of `s192_l4` (same architecture, float weights and activations) |
| **Data and budget** | The predecessor corpus's passages (1.5 million) with its frozen encoder and head as the teacher; every arm distilled from it on the same batches; 1,076 steps of 1,024 passages (about 316,000 tokens a step, 340 million in all, under one pass); 60 minutes on one H100 GPU |
| **Measured** | Validation PR AUC, ROC AUC, and MCC on 10,000 passages against a bank of 8,192 training passages per origin. Rule fixed in advance: *go* if the best ternary student is within 0.02 PR AUC of the teacher and within 0.01 of its twin |
| **Seeds** | One per arm |

| Reader | PR AUC | ROC AUC | MCC | Training tokens a second |
| --- | --- | --- | --- | --- |
| Ternary `s192_l4` | 0.9673 | 0.9609 | 0.7982 | 355,464 |
| Ternary `s256_l6` | **0.9721** | **0.9665** | **0.8219** | 212,570 |
| Full-precision twin, `s192_l4` | 0.9676 | 0.9613 | 0.8027 | 499,308 |
| Teacher, same bank | 0.9901 | 0.9887 | 0.8986 | |

| | |
| --- | --- |
| **Result** | Verdict *go*. The smallest ternary student is 0.0003 PR AUC below its twin; the best ternary student is 0.0045 above the twin and 0.0180 below the teacher. All three curves were still rising over the last quarter of the run |
| **What it changed** | The ternary design was kept; the full-precision twin is a control only, never released |
| **Limits** | Two origins, not three; distilled students, not trained on labels; one seed; a short run with every curve still rising; equal steps are not equal time (the twin trains about 40% faster per token on the GPU) |

---

## E2. A per-token teacher loss

**Finding: matching the teacher's state at every token gave no gain per step and tripled the cost of a step.**

*In plain terms.* Every training signal gave a text one target. This experiment added a target at every token, the
teacher's own vector for that token, to see whether a denser signal teaches faster.

| | |
| --- | --- |
| **Question** | Does a per-token loss on the final states, against the teacher's token states, help the smallest ternary student per step? |
| **Arms** | `s192_l4` without and with the per-token loss at weight 1, on the same batches |
| **Data and budget** | The predecessor corpus and teacher, as in E1; stopped after 15 of a planned 35 minutes |
| **Measured** | Validation PR AUC at equal steps; the cost of a step |
| **Seeds** | One per arm |

| Step | Plain | Per-token loss |
| --- | --- | --- |
| 9 | 0.741 | 0.722 |
| 101 | 0.887 | 0.881 |
| 194 | **0.921** | 0.917 |

| | |
| --- | --- |
| **Result** | The per-token arm was behind at every reading. Its step cost about three plain steps, because the teacher's encoder must run inside the loop |
| **What it changed** | The per-token loss was not adopted (it stays in `litmus_student.dense`, measured, off) |
| **Limits** | One weight; final-layer states only; targets computed in the loop rather than cached (caching would remove the cost); stopped early |

---

## E3. Integer attention

**Finding: integer queries, keys, values, and attention weights cost nothing measurable when trained in, and read 5% to
11% faster with SIMD.**

*In plain terms.* Attention is the one part of the encoder that multiplies two computed quantities, so it stayed in
floating point. This experiment made it integer too and checked whether accuracy suffered.

| | |
| --- | --- |
| **Question** | Does [integer attention](model.md#attention-inside-shifted-blocks) (int8 queries, keys, and values, weights in 127ths, float only in the softmax) cost detection accuracy? |
| **Arms** | `s192_l4` with float attention and with integer attention trained in, on the same batches |
| **Data and budget** | The predecessor corpus and teacher; 306 steps of 1,024 passages, micro-batch 65,536 tokens, on a shared GPU |
| **Measured** | Validation PR AUC, ROC AUC, and MCC at every reading; training speed; the engine's time a token on one core |
| **Seeds** | One per arm |

| Arm, step 305 | PR AUC | ROC AUC | MCC | Training tokens a second |
| --- | --- | --- | --- | --- |
| Float attention | 0.9371 | 0.9197 | **0.7211** | 150,626 |
| Integer attention | **0.9386** | **0.9228** | 0.7181 | 128,534 |

| | |
| --- | --- |
| **Result** | The two stayed within 0.008 PR AUC of each other at every reading, with no consistent sign. Training with the mode is 15% slower. In the engine at 8,192 tokens with AVX2, an integer-attention release takes 5% to 11% less time a token (width 192: 80.4 against 88.4 µs); without SIMD it costs 4% to 17% more ([`engine/README.md`](../../engine/README.md)) |
| **What it changed** | Integer attention became an optional release format (format 2) in the specification, the reference, the engine, and the export, which can release a float-trained model in integer mode. The release uses <!--litmus:model.attention-->integer<!--/litmus--> |
| **Limits** | 306 steps; one size; one seed; the engine's speed is from random-weight releases under load, read interleaved |

---

## E4. Distillation from a frozen teacher against labels only

**Finding: the student trained on true labels alone was level with or ahead of the distilled student on every
detection measure, so the release trains on labels alone.**

*In plain terms.* A large frozen model (the teacher) can show a small model what to compute: its vector for each text
and its judgment of each pair. The alternative is to tell the small model only which texts share an origin. The same
small model was trained both ways on identical examples, and the plain way read as well or better.

| | |
| --- | --- |
| **Question** | Does distilling from a frozen teacher (ModernBERT-base with the searched pair head of E7) beat training on the true-label pair loss alone, for the same student on the same batches? |
| **Arms** | `s192_l4` distilled: all five terms of [the loss](training.md#the-loss) (embedding, query and key, soft pairs, ranked pairs, true labels), with E8's weights; `s192_l4` on labels only: the four teacher terms off and their whole weight (7.47) on the true-label term. Same seed, batches, head, and schedule |
| **Data and budget** | The first build; 1,024 texts a step; a time box of about five hours on one H100 shared by three runs; the distilled arm stopped at step 2,150 |
| **Measured** | Validation, the same rows and bank for both arms, at equal steps: PR AUC, ROC AUC, and balanced MCC of AI or mixed against human; PR AUC of human against AI; the teacher on the same rows: 0.923, 0.899, 0.702, and 0.770 |
| **Seeds** | One per arm |

| Arm | Step | PR AUC | ROC AUC | MCC | PR AUC, human against AI | Pair-label loss |
| --- | --- | --- | --- | --- | --- | --- |
| Distilled | 999 | 0.8972 | 0.8593 | **0.5892** | 0.7567 | **0.5746** |
| Labels only | 999 | **0.9067** | **0.8699** | 0.5814 | **0.8158** | 0.6378 |
| Distilled | 1,999 | 0.9102 | 0.8693 | 0.5930 | 0.7674 | **0.5614** |
| Labels only | 1,999 | **0.9146** | **0.8756** | **0.5935** | **0.8230** | 0.6464 |

| | |
| --- | --- |
| **Result** | Labels only is ahead on PR AUC and ROC AUC at both readings, level on MCC (0.008 behind at step 999, level at 1,999), and well ahead on human against AI (0.8230 against 0.7674), where it also passes the teacher (0.770), which a distilled student is held to. Only the raw pair-label loss favors the distilled arm, a difference in the logits' scale rather than in ranking; calibration sets the scale in a release |
| **What it changed** | The release trains on labels alone, from scratch, with no teacher. The frozen encoder is still used to embed and split the corpus; the runs on the first build logged the teacher's validation numbers beside the student's |
| **Limits** | One seed; about a fifth of the planned run; a teacher that read worse than the student on human against AI, so a stronger teacher could change the answer. Published gains from distillation into ternary models come from converting pretrained models (for example [BitNet Distillation](https://arxiv.org/abs/2510.13998)), a different setting from training from scratch |

The labels-only `s192_l4` continued to step 5,999: PR AUC 0.9262 at step 2,999 and 0.9309 at step 4,999 (its lowest
validation loss: ROC AUC 0.9051, MCC 0.6830, human against AI 0.8238), with its learning rate not yet decayed.

---

## E5. Model size

**Finding: at equal steps a larger encoder was ahead on MCC early in training, with PR AUC close; the comparison was
stopped far from convergence.**

*In plain terms.* Larger models are slower on one core but may learn more. Three sizes were trained on labels alone
and read at the same step counts.

| | |
| --- | --- |
| **Question** | At equal training steps, how much do wider and deeper encoders gain over the smallest, against their cost on one core? |
| **Arms** | `s192_l4` (1.79M ternary weights), `s384_l8` (14.20M), `s512_l12` (37.81M), all on labels only with E8's head settings |
| **Data and budget** | The first build; 1,024 texts a step; the larger two started later in the same time box on a shared GPU, with micro-batches of 65,536 and 32,768 tokens, and were stopped at steps 1,050 and 500 |
| **Measured** | Validation as in E4 at equal steps; the engine's tokens a second on one core |
| **Seeds** | One per arm |

| Size | Step | PR AUC | ROC AUC | MCC | PR AUC, human against AI | One core, 8,192 tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `s192_l4` | 999 | 0.9067 | 0.8699 | 0.5814 | 0.8158 | 21,926 tokens a second |
| `s384_l8` | 499 | 0.8996 | 0.8517 | 0.5471 | 0.7854 | 4,214 tokens a second |
| `s384_l8` | 999 | **0.9104** | **0.8823** | **0.6276** | **0.8183** | |
| `s512_l12` | 249 | 0.8475 | 0.7612 | 0.4192 | 0.6391 | about 1,900 tokens a second |
| `s512_l12` | 499 | 0.8918 | 0.8541 | 0.5887 | 0.7593 | |

| | |
| --- | --- |
| **Result** | At step 999, `s384_l8` is 0.004 PR AUC and 0.046 MCC above `s192_l4`. At step 499, `s512_l12` is 0.008 PR AUC below `s384_l8` and 0.042 MCC above. While distilled, `s384_l8` at step 499 read about what `s192_l4` read at step 999 (PR AUC 0.8919 against 0.8972). On one core `s384_l8` reads about a fifth as many tokens a second as `s192_l4`, and `s512_l12` about half as many again |
| **What it changed** | Sizes up to `s512_l12` stay candidates. The release size is <!--litmus:record.model.size_rule-->`s512_l6`, chosen by the maintainer for 1.1 (Logan, 2026-11-19: "512 hidden size 6 layers, that should be a good size") in place of 1.0.0's `s192_l4`, so that one encoder also serves as the masked language model; no size search was run for this release<!--/litmus--> |
| **Limits** | Very early in training (a few hundred to a thousand steps); one seed; equal steps are not equal compute; the larger arms shared the GPU; speeds are from random-weight releases (`s512_l12` measured once) |

---

## E6. Readout: pooled pairs, token pairs, or token labels

**Finding: pooling a text and comparing it with the bank read better than scoring tokens one by one.**

*In plain terms.* Litmus reduces a text to one vector and compares it with known texts. Two alternatives score every
token instead: one compares each token with the bank, the other scores each token against one learned vector per
origin with no bank at all. The bank-based readout of whole texts won.

| | |
| --- | --- |
| **Question** | Which readout of the same encoder detects best: the pooled pair head with the bank, the pair head on every token with the bank, or a per-token classifier with no bank? |
| **Arms** | `pooled_pairs` (the design: mean and variance pooling, the pair head, the bank); `token_pairs` (the pair head on every token, a text's query and key the mean of its tokens', read against the bank); `token_labels` (each token scored against one learned vector per origin, a text's logits the mean of its tokens', cross-entropy, no bank). All `s192_l4`, from the same weights, on the same batches, labels only |
| **Data and budget** | The first build; 1,001 steps each with the whole schedule, on a shared GPU |
| **Measured** | Validation as in E4; and, on documents spliced from held-out human and AI passages, the share of tokens whose own contribution falls on their origin's side, balanced over the two origins |
| **Seeds** | One per arm |

| Arm | PR AUC | ROC AUC | MCC | PR AUC, human against AI | Spliced tokens on their own side |
| --- | --- | --- | --- | --- | --- |
| `pooled_pairs` | **0.9211** | **0.8843** | **0.6185** | **0.8489** | not measured (no per-token score) |
| `token_pairs` | 0.8811 | 0.8427 | 0.5618 | 0.7472 | 0.601 (human 0.736, AI 0.466) |
| `token_labels` | 0.9057 | 0.8534 | 0.5570 | 0.8201 | 0.660 (human 0.752, AI 0.568) |

| | |
| --- | --- |
| **Result** | The pooled pair head with the bank leads the per-token classifier by 0.015 PR AUC and 0.062 MCC. The hybrid's readings swung during training while its loss fell, so its number says little about the idea; a likely cause, not confirmed, is the gap between the per-step statistics it trains with and the running statistics it reads with |
| **What it changed** | The pooled pair head with the bank stays the readout. The heat map keeps using windows of pooled vectors ([3. The model](model.md#long-texts-chunks-windows-and-cells)) rather than per-token scores |
| **Limits** | One seed; 1,001 steps; the hybrid's instability unresolved; this comparison's code is not part of this repository |

---

## E7. Searching the teacher's pair head

**Finding: a Bayesian search of the teacher's head lowered its validation pair loss by 0.047 over the default, and the
gain held across three seeds.**

| | |
| --- | --- |
| **Question** | How much does searching the pair head's settings on the frozen embeddings improve on the default recipe? |
| **Arms** | A Bayesian search (tree-structured Parzen estimator, a median pruner, the default recipe as trial 0) over learning rate, weight decay, dropout, depth (1 to 3) and widths (512 to 8,192), output width (256 to 2,048), batch, warmup, schedule, input clamp, the pair logit's scale and bias, and label smoothing; heads trained in lockstep on embeddings held on the GPU |
| **Data and budget** | The first build's ModernBERT-base embeddings; a 30-minute box (trials in flight finish), each trial at most 8,000 steps; the best configurations retrained on three seeds |
| **Measured** | Each trial's lowest validation pair loss on 16 fixed pairs of balanced validation batches; no trial reads test, calibration, or a bank |
| **Seeds** | Three for the runs of record |
| **Result** | 21 trials (14 complete, 7 pruned). The best (hidden widths 8,192 then 1,024, output width 2,048, batch 1,536, learning rate 5.4e-4, dropout 0.10, label smoothing 0.1) read 0.4795 against the default's 0.5263; retrained on three seeds, 0.4873, 0.4861, and 0.4873 |
| **What it changed** | This head is the teacher of E4 and the reference logged beside the students trained on the first build; the release has no teacher |
| **Limits** | Whether 8,000-step trials rank settings as full training would is untested; the GPU was shared |

---

## E8. Searching the distillation settings

**Finding: the search settled the student head's widths and the loss weights; the schedule kept its defaults.**

| | |
| --- | --- |
| **Question** | Which student head settings and loss weights train the smallest student best? |
| **Arms** | A Bayesian search (tree-structured Parzen estimator, the defaults as trial 0) over the loss weights (each teacher term as a log ratio to the true-label term, their total weight, and the temperature), the head's hidden width (256 to 1,024), query and key width (64 or 128), and clamp (3 to 30 deviations), and the schedule (peak learning rate, second peak, warmup, weight decay, texts per step). Successive halving read each trial at rungs of 16,384, 49,152, and 147,456 texts and stopped all but the best third at each |
| **Data and budget** | The first build and the teacher of E7; `s192_l4`; three workers on one GPU, 40 minutes |
| **Measured** | The validation pair loss against true labels. Rule fixed before any result: the best trial at the top rung supplies its loss and head settings if the rank agreement (Spearman) between the last two rungs is at least 0.6, and its schedule too only if it is at least 0.8 |
| **Seeds** | One per trial |
| **Result** | 14 trials reached the top rung, with rank agreement 0.6 between the last two rungs. By the rule, the best trial's loss and head settings were taken (validation pair loss 0.6189): head hidden width 256, output width 64, clamp 30, and loss weights totalling 7.47, most of it on the query-and-key and ranked-pair terms; the schedule kept its defaults |
| **What it changed** | E4 and E5 ran with these head settings, and E4's labels-only arm put the whole 7.47 on the true-label term. The release's head is <!--litmus:t:head_widths-->hidden width 256, output width 64<!--/litmus--> |
| **Limits** | One size; short trials; settings chosen for distillation, then reused for labels only |

---

## E9. Bank: held-out clusters or a draw from training clusters

**Finding: a bank drawn evenly from training clusters read better than a bank of held-out clusters.**

| | |
| --- | --- |
| **Question** | Should the reference bank hold texts from clusters no training text shares, or an even draw from every training cluster, which covers every region of the corpus? ([2. The splits](splits.md#step-3-give-whole-clusters-to-sets-with-moats)) |
| **Arms** | Bank (a), held-out clusters; bank (b), a stratified draw from training clusters, from documents no held-out set holds |
| **Data and budget** | The first build; the teacher's head of E7; several bank sizes |
| **Measured** | Validation PR AUC, ROC AUC, and MCC through each bank |
| **Seeds** | One |
| **Result** | Bank (b) at 1,024 texts per origin: 0.898, 0.913, and 0.720. Bank (a): 0.871, 0.893, and 0.711, at every size tried |
| **What it changed** | Bank (b) was chosen for the teacher. The release's bank is drawn <!--litmus:record.model.bank_variant-->the withheld reference set, 8,192 texts an origin from clusters no other set holds and that never trained (the `reference` recipe), embedded again at every validation reading and by the export<!--/litmus--> |
| **Limits** | The teacher's bank, not the student's; one split |

---

## E10. Skipping zero inputs of the down product

**Finding: in a trained model most inputs of the feed-forward's down product are exactly zero, and skipping them makes
the engine 3% to 19% faster with identical readings.**

| | |
| --- | --- |
| **Question** | The squared ReLU sets every negative entry to exactly 0, so many inputs of $`W_{\mathrm{down}}`$ are zero. Can a kernel that skips them per token beat the dense kernel, with the same bits? |
| **Arms** | The dense ternary kernel; a column-pair kernel that adds only the pairs of columns whose inputs are not both zero, chosen per token by a measured limit |
| **Data and budget** | An interim trained `s192_l4` release; real text of 26,323 tokens; one core of an AMD Ryzen 7 3800X, pinned, five rounds interleaved |
| **Measured** | Zero rates of each layer's inputs; microseconds a token from text to reading; whether the whole reading is byte-identical |
| **Result** | Zero inputs of `down`: 68%, 72%, 85%, and 92% in layers 0 to 3 (other products' inputs: 1% to 2%; weights: about a third zero, scattered). Time a token saved: 3% with AVX2, 11% on the portable path, 15% in WebAssembly with SIMD128, and 19% without. The JSON reading is byte-identical with and without the column path |
| **What it changed** | The engine reads `down` by the column path when a token's nonzero pairs are under the limit ([8. The engine](engine.md#skipping-zero-inputs)), and A5 asks whether training can raise the zero rate |
| **Limits** | One trained release of one size; zero rates of other sizes and of the release are not measured here (<!--litmus:record.profile.zero_rates-->at 1,000 steps the release ablations' arms without a penalty read 61% to 81% zeros by layer (A5); the released checkpoint's own rates are not measured<!--/litmus-->) |

---

## The release ablations

Short runs at equal budget, each changing one thing against the current recipe, to settle the open design questions
before the single release run. All five share: <!--litmus:record.ablations.shared-->`s192_l4`, 1,000 steps of 1,024 texts in 65,536-token micro-batches, the release arithmetic (ternary weights, integer attention) from the first step, distilled from the first build's teacher export with the distillation search's settings, seed 0, on the first build's token store; readings every 250 steps on the first build's validation split (PR AUC, ROC AUC and balanced MCC of AI or mixed against human, and PR AUC of human against AI); one arm at a time on an NVIDIA A10, and the int8 output's second seed on a GH200. Two runs of one recipe differ by up to 0.007 PR AUC on one seed, so a difference under that is a tie, and a change to the released format needs 0.01 on two seeds<!--/litmus-->. The
release run differs from that setup in its data (the second build) and its loss (true labels alone, no teacher);
whether the ablations' rankings carry over to it is untested.

### A1. Weight levels

| | |
| --- | --- |
| **Question** | Do weights with more levels than ternary read better at equal training? The engine's AVX2 kernel already multiplies unpacked 2-bit codes by activations, so four levels in the same 2 bits would cost about the same per token ([ParetoQ](https://arxiv.org/abs/2502.02631) reports 2-bit above 1.58-bit for a large language model) |
| **Arms** | <!--litmus:record.ablations.a1..arms-->`baseline`: ternary codes with one absmean scale a tensor; `quaternary`: four levels with a learned scale per output row; `channel_scales`: ternary codes with a learned scale per output row<!--/litmus--> |
| **Data and budget** | <!--litmus:record.ablations.a1..data-->The shared setup above<!--/litmus--> |
| **Measured** | <!--litmus:record.ablations.a1..measured-->Validation at step 999, as above<!--/litmus--> |
| **Result** | <!--litmus:record.ablations.a1..result-->Baseline 0.9189, 0.8838, 0.6206, 0.8412; quaternary 0.9166, 0.8841, 0.6196, 0.8287 (0.0023 behind, a tie, with human against AI 0.0125 lower); channel scales 0.9153, 0.8775, 0.6041, 0.8233 (behind on every measure)<!--/litmus--> |
| **What it changed** | <!--litmus:record.ablations.a1..changed-->Nothing: the weights stay ternary with one scale a tensor<!--/litmus--> |
| **Limits** | <!--litmus:record.ablations.a1..limits-->One seed each; 1,000 steps; no second seed was run, since no arm came within reach of its bar<!--/litmus--> |

### A2. Float-first training

| | |
| --- | --- |
| **Question** | Does training in full precision first and switching to ternary for the last part of the run beat quantized training from the first step? |
| **Arms** | <!--litmus:record.ablations.a2..arms-->`baseline`: the release arithmetic from step 0; `float_first_60`: full precision for the first 60% of the steps, then the release arithmetic with the optimizer state kept<!--/litmus--> |
| **Data and budget** | <!--litmus:record.ablations.a2..data-->The shared setup above<!--/litmus--> |
| **Measured** | <!--litmus:record.ablations.a2..measured-->Validation every 250 steps, as above, and the time a step<!--/litmus--> |
| **Result** | <!--litmus:record.ablations.a2..result-->Ahead while in full precision (step 249: PR AUC 0.8825 against 0.8640), level at step 499 (0.8972 against 0.8987), 0.0010 behind at step 999 (0.9179 against 0.9189); 1.31 s a step in full precision against 2.2 s<!--/litmus--> |
| **What it changed** | <!--litmus:record.ablations.a2..changed-->Nothing: training in the release arithmetic from the first step costs nothing measurable, so the recipe keeps it<!--/litmus--> |
| **Limits** | <!--litmus:record.ablations.a2..limits-->One seed; at 1,000 steps the switch comes at step 600, so a longer run could differ<!--/litmus--> |

### A3. Pooling

| | |
| --- | --- |
| **Question** | Is the mean and variance of the token states the best way to pool a text into one vector? |
| **Arms** | <!--litmus:record.ablations.a3..arms-->`baseline`: mean and variance of the token states; `pool_mean_variance_max`: mean, variance and the column maximum; `pool_max`: the column maximum alone<!--/litmus--> |
| **Data and budget** | <!--litmus:record.ablations.a3..data-->The shared setup above<!--/litmus--> |
| **Measured** | <!--litmus:record.ablations.a3..measured-->Validation at step 999, as above<!--/litmus--> |
| **Result** | <!--litmus:record.ablations.a3..result-->Baseline 0.9189, 0.8838, 0.6206, 0.8412; mean, variance and maximum 0.9228, 0.8862, 0.6180, 0.8494 (+0.0039); maximum alone 0.9247, 0.8876, 0.6188, 0.8541 (+0.0058)<!--/litmus--> |
| **What it changed** | <!--litmus:record.ablations.a3..changed-->Nothing: mean and variance stay, since the windows of the heat map come from running totals, which give no maximum, and a pooling change needs 0.01<!--/litmus--> |
| **Limits** | <!--litmus:record.ablations.a3..limits-->One seed each; inside what one seed shows by chance<!--/litmus--> |

### A4. Output width

| | |
| --- | --- |
| **Question** | How wide must the query and key be? A narrower output shrinks the bank in the release (each bank text costs $`4w + 17`$ bytes) |
| **Arms** | <!--litmus:record.ablations.a4..arms-->`baseline`: 64 wide; `head_32` and `head_256`: 32 and 256 wide; `matryoshka`: 128 wide with the pair loss summed over its prefixes of 16, 32, 64 and 128, read at each prefix<!--/litmus--> |
| **Data and budget** | <!--litmus:record.ablations.a4..data-->The shared setup above<!--/litmus--> |
| **Measured** | <!--litmus:record.ablations.a4..measured-->Validation at step 999, as above<!--/litmus--> |
| **Result** | <!--litmus:record.ablations.a4..result-->Baseline 0.9189; `head_32` 0.9142 (0.0047 behind); `head_256` 0.9111 (0.0078 behind); Matryoshka whole 128 0.9161, first 64 0.9143, first 32 0.9167, first 16 0.9179<!--/litmus--> |
| **What it changed** | <!--litmus:record.ablations.a4..changed-->Nothing: 64 stays, the narrowest width within the 0.003 a narrower output must hold; wider read worse, and the Matryoshka prefixes were not adopted<!--/litmus--> |
| **Limits** | <!--litmus:record.ablations.a4..limits-->One seed each; the bank is 2,048 texts an origin in the release, where a narrower output saves little<!--/litmus--> |

### A5. Activation sparsity

| | |
| --- | --- |
| **Question** | Can training make more of the `down` product's inputs zero, so the column path of E10 skips more work, without costing accuracy? At width 192, 90% zeros in every layer would save about 3% a token on AVX2 and 9% on the portable path; at width 384 about 12% ([`engine/README.md`](../../engine/README.md)) |
| **Arms** | <!--litmus:record.ablations.a5..arms-->`baseline`: no penalty; `ffn_l1_low` and `ffn_l1_high`: an L1 penalty of weight 3 and 30 on the int8 inputs of the feed-forward's `down` product<!--/litmus--> |
| **Data and budget** | <!--litmus:record.ablations.a5..data-->The shared setup above<!--/litmus--> |
| **Measured** | <!--litmus:record.ablations.a5..measured-->Validation at step 999, as above, and the share of zero codes in `down`'s inputs by layer<!--/litmus--> |
| **Result** | <!--litmus:record.ablations.a5..result-->Without a penalty 0.9189 with 61% to 81% zeros by layer (the other arms of the pass); weight 3: 0.9080 (0.0109 behind), 74%, 83%, 87%, 91% zeros; weight 30: 0.9099 (0.0090 behind, MCC 0.0288 lower), 93%, 95%, 96%, 98% zeros<!--/litmus--> |
| **What it changed** | <!--litmus:record.ablations.a5..changed-->No penalty: both strengths cost more than one seed shows by chance, so the release keeps the zeros that come free<!--/litmus--> |
| **Limits** | <!--litmus:record.ablations.a5..limits-->One seed each; zero shares are read at the last reading of a 1,000-step arm<!--/litmus--> |

---

Previous: [5. Calibration and the reading](calibration.md) · Next: [7. Evaluation](evaluation.md)
