# 4. Training

[Front page](../TRAINING.md) · [1. Data](data.md) · [2. Splits](splits.md) · [3. Model](model.md) · **4. Training** ·
[5. Calibration](calibration.md) · [6. Experiments](experiments.md) · [7. Evaluation](evaluation.md) ·
[8. Engine](engine.md) · [9. Reproduce](reproduce.md)

Training adjusts the latent weights of the encoder and the head so that two texts of the same origin get a high pair
logit and two texts of different origins a low one. The model learns from the origins alone: one loss, over every
pair of texts in a training step. It is trained from scratch, with quantization in the forward pass from the first
step. The code is `student/src/litmus_student/train.py`.

- [The loss](#the-loss)
- [A 3 by 3 example](#a-3-by-3-example)
- [What a step draws](#what-a-step-draws)
- [Packing texts into blocks](#packing-texts-into-blocks)
- [The gradient-cached step](#the-gradient-cached-step)
- [Optimizer and schedule](#optimizer-and-schedule)
- [What is measured during training, and the checkpoint](#what-is-measured-during-training-and-the-checkpoint)
- [The release run](#the-release-run)
- [Where in the code](#where-in-the-code)

## The loss

A step holds $`b`$ training texts with origins $`y_1, \dots, y_b \in \{0, 1, 2\}`$ (human, AI, mixed). The model gives
each its unit query and key; stacked, $`Q`$ and $`K`$ are $`b \times w`$. All pair logits at once
([3. The model](model.md#the-pair-logit)):

```math
S = a_s \cdot \frac{Q K^{\top} + K Q^{\top}}{2} + b_s \qquad (b \times b)
```

Entry $`S_{ij}`$ is the logit that texts $`i`$ and $`j`$ share an origin. The **target** of each pair is whether they
do:

```math
Y_{ij} = \mathbf{1}[\,y_i = y_j\,] \qquad (b \times b)
```

The **true-label pair loss** is the binary cross-entropy of every pair against its target, averaged over the
$`b(b-1)`$ pairs of two different texts (a text with itself is left out):

```math
\mathcal{L}_{\mathrm{pair}} = \frac{1}{b(b-1)} \sum_{i \neq j} \Big[ -Y_{ij} \log \sigma(S_{ij}) - (1 - Y_{ij}) \log\big(1 - \sigma(S_{ij})\big) \Big]
```

and the loss the optimizer reduces is $`\mathcal{L} = \lambda\, \mathcal{L}_{\mathrm{pair}}`$, with $`\lambda`$ the weight
the distillation search's settings gave all terms together, put on this one ([E4](experiments.md#e4-distillation-from-a-frozen-teacher-against-labels-only)).
For the release, $`\lambda`$ = <!--litmus:record.training.loss_weight-->7.47<!--/litmus-->. The cross-entropy is computed
from the logits in a form that keeps the gradient of a confidently right pair
([Barron et al., 2026](https://arxiv.org/abs/2610.31642)), and equals the formula above.

Its gradient with respect to one logit is simply the error of that pair's probability:

```math
\frac{\partial \mathcal{L}}{\partial S_{ij}} = \lambda\, \frac{\sigma(S_{ij}) - Y_{ij}}{b(b-1)}
```

A step thus pulls together the queries and keys of every same-origin pair and pushes apart every different-origin
pair, in proportion to how wrong each is. One step with $`b = 1{,}024`$ texts trains on 1,047,552 pairs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/training/pair_matrix_dark.svg">
  <img alt="A 6 by 6 grid of the pairs of six texts, two human, two AI, two mixed: same-origin cells are 1, different-origin cells 0, and the diagonal is hatched and left out; beside it, the pair logit, the target, and the loss." src="../images/training/pair_matrix_light.svg" width="100%">
</picture>

*The targets of a step with two texts of each origin. The diagonal is left out; every other cell is one pair.*

| Symbol | What it is | Kind | Shape | Real size |
| --- | --- | --- | --- | --- |
| $`b`$ | Texts in a step | setting | | <!--litmus:record.training.texts_per_step as int-->1,024<!--/litmus--> |
| $`Q, K`$ | Unit queries and keys of the step's texts | computed | $`b \times w`$ each | |
| $`S`$ | Pair logits | computed | $`b \times b`$ | |
| $`Y`$ | Same-origin targets | computed from labels | $`b \times b`$ | |
| $`\lambda`$ | Loss weight | setting | scalar | <!--litmus:record.training.loss_weight-->7.47<!--/litmus--> |
| $`a_s, b_s`$ | Pair logit scale and bias | fixed after training | scalars | |

<details>
<summary>More detail: the distillation terms, which the release does not use</summary>

The training code can also match a frozen **teacher** (ModernBERT-base and a pair head trained on its embeddings), with
four more terms beside the true-label one. Per text: the student's standardized pooled vector, through a learned
linear map, against the teacher's embedding ($`1 - \cos`$ plus mean squared error); and the student's query and key,
through learned maps, against the teacher's ($`1 - \cos`$). Per pair: the binary cross-entropy of the student's logit
against the teacher's probability $`\sigma(S^{T}_{ij} / \tau)`$; and a cross-entropy between each row's softmax over
the student's and the teacher's logits, which keeps the teacher's order of nearest texts. With default weights 1, 1,
1, 0.5, and 1 (true labels last). Trained against the same student on labels alone, the distilled student was level
or behind on every detection measure, so the release trains on labels alone
([E4](experiments.md#e4-distillation-from-a-frozen-teacher-against-labels-only)). The terms are in
`litmus_student.distill.distillation_losses`, and the switch is `train.py --labels-only`.

</details>

## A 3 by 3 example

Made-up numbers, computed with the formulas above. Three texts, two human and one AI, with toy query and key vectors
of width 2, scale $`a_s = 10`$, and bias $`b_s = -4`$. Text 3 is human but its vectors lean toward the AI text's.

| Text | Origin | Query $`q`$ | Key $`k`$ |
| --- | --- | --- | --- |
| 1 | human | (0.981, 0.196) | (0.992, −0.124) |
| 2 | AI | (−0.287, 0.958) | (−0.217, 0.976) |
| 3 | human | (0.530, 0.848) | (0.406, 0.914) |

The symmetric dot products $`(q_i^{\top} k_j + q_j^{\top} k_i) / 2`$, the logits, and the probabilities:

| Pair | Symmetric dot | Logit $`S_{ij}`$ | $`\sigma(S_{ij})`$ | Target $`Y_{ij}`$ | Loss | What it shows |
| --- | --- | --- | --- | --- | --- | --- |
| 1 and 2 | −0.213 | −6.13 | 0.002 | 0 | 0.002 | A different-origin pair scored correctly: almost no loss |
| 1 and 3 | 0.499 | 0.99 | 0.729 | 1 | 0.316 | A same-origin pair scored right but unsure: some loss |
| 2 and 3 | 0.736 | 3.36 | 0.966 | 0 | **3.391** | A different-origin pair scored as the same: a large loss |

Each pair appears twice in the matrix ($`S`$ is symmetric), so the six off-diagonal cells give

```math
\mathcal{L}_{\mathrm{pair}} = \frac{2 \times (0.002 + 0.316 + 3.391)}{6} = 1.236
```

and the gradients on the logits, $`(\sigma(S_{ij}) - Y_{ij}) / 6`$ per cell (with $`\lambda = 1`$), are $`+0.0004`$ for pair
1 and 2 (leave it), $`-0.045`$ for pair 1 and 3 (raise it), and $`+0.161`$ for pair 2 and 3 (lower it, most of all). The
optimizer therefore moves text 3's query and key away from the AI text's and toward text 1's.

## What a step draws

Each step draws $`b`$ texts from the training split, **a third of each origin** (the remainder to the first), so every
step sees all nine kinds of pair (human-human, human-AI, ..., mixed-mixed). Within an origin, texts are drawn
**uniformly**, so the step inherits the corpus's balance of format, language, length, and domain cell by cell. (An
option draws an origin's e-means clusters uniformly instead and one text within each; it is off, because each origin
would then get its own mix of cells, which a model can read as the origin.)

Eligible texts are training texts of at most 8,192 tokens, never a long document kept for evaluation. Texts are
**never cropped**: a text is read whole, at its own length, from 8 to 8,192 tokens, as inference reads a chunk.

## Packing texts into blocks

A step's texts are packed into **micro-batches** of a fixed number of token positions (65,536 by default), each one
flat stream, so no position is spent on padding inside a block:

- A text longer than $`B/2`$ starts at a multiple of $`B`$, so its blocks are whole blocks of the stream under both
  shifts (0 and $`B/2`$).
- Texts of at most $`B/2`$ tokens share half-blocks, which no block of either shift splits.
- Long texts go first, largest first; short ones fill the free room, first fit, largest first.

Every layer reshapes the stream into blocks and attends under a mask of **same text and same text block**, so a token
never attends to another text or across its own text's block boundary, exactly as when the text is read alone. The
stream's length is padded to a multiple of 8,192 positions so that tensor shapes repeat.

## The gradient-cached step

The loss compares every pair of the step's texts, so it needs all $`b`$ queries and keys at once, but the activations
of a thousand texts of up to 8,192 tokens do not fit in a GPU's memory. **Gradient caching**
([Gao et al., 2021](https://arxiv.org/abs/2101.06983)) computes the exact gradient of the full-step loss with the
memory of one micro-batch:

```mermaid
flowchart TD
  A["Pass 1, no gradients:<br>each micro-batch through the encoder,<br>pooled vectors kept (b × 2d)"] --> B["Head and loss on all b texts:<br>the full b × b pair matrix"]
  B --> C["Gradient of the loss with respect to<br>each pooled vector (b × 2d), kept"]
  C --> D["Pass 2, with gradients:<br>each micro-batch again; back-propagate<br>its slice of the kept gradient"]
  D --> E["One optimizer step"]
```

*The step, exact: pass 2 gives each micro-batch's weights the gradient the full matrix assigns them.*

Because the pooled vectors are the only link between the encoder and the loss, the chain rule splits at them:
$`\partial \mathcal{L} / \partial \theta = \sum_{i} (\partial \mathcal{L} / \partial x_{\mathrm{pool},i})^{\top} \, \partial x_{\mathrm{pool},i} / \partial \theta`$.
Pass 1 gives the first factor for every text at once; pass 2 computes the second one micro-batch at a time. The cost
is one extra forward pass, about a third more compute. A memory bank of vectors from earlier steps would be cheaper,
but its vectors come from older weights and carry no gradient, so the step would no longer follow the gradient of its
own loss.

The encoder runs in bfloat16 autocast on the GPU; the head, the loss, and every quantization run in float32.

## Optimizer and schedule

**AdamW** with betas (0.9, 0.95), weight decay on matrices only (latent ternary weights, the embedding, the head's
weights), and the gradient's norm clipped at 1.0. The schedule follows the BitNet b1.58 training notes: a high peak
learning rate decaying linearly to zero, dropped to a lower peak at half the run, and weight decay removed at half
the run. With $`t`$ the step, $`T`$ the planned steps, $`t_w`$ the warmup steps, and peaks $`\eta_1 \gt  \eta_2`$:

```math
\eta(t) =
\begin{cases}
\eta_1 \, (t + 1) / t_w & t \lt  t_w \\
\eta_1 \, (1 - t / T) & t_w \le t \lt  T/2 \\
\eta_2 \, (1 - t / T) & t \ge T/2
\end{cases}
\qquad
\mathrm{wd}(t) =
\begin{cases}
0.1 & t \lt  T/2 \\
0 & t \ge T/2
\end{cases}
```

**The time box.** A run may be given minutes instead of steps. The schedules then read the step
$`\max\!\big(t,\ \min(T, \lfloor T \cdot \mathrm{elapsed} / \mathrm{box} \rfloor)\big)`$, so the learning rate reaches
zero when the box closes however fast the steps run, and the run ends with its low-rate tail instead of being cut off.

*Worked example* with the code's defaults, $`\eta_1 = 1.5 \times 10^{-3}`$, $`\eta_2 = 10^{-3}`$, $`t_w = 100`$, and
$`T = 10{,}000`$:

| Step $`t`$ | Rule | $`\eta(t)`$ | Weight decay |
| --- | --- | --- | --- |
| 50 | warmup, $`\eta_1 \cdot 51 / 100`$ | 7.65e-4 | 0.1 |
| 2,000 | $`\eta_1 \cdot 0.8`$ | 1.2e-3 | 0.1 |
| 4,999 | $`\eta_1 \cdot 0.5001`$ | 7.5e-4 | 0.1 |
| 5,000 | $`\eta_2 \cdot 0.5`$ | 5.0e-4 | 0 |
| 9,000 | $`\eta_2 \cdot 0.1`$ | 1.0e-4 | 0 |

## What is measured during training, and the checkpoint

Every few hundred steps the run reads a fixed **validation** set the same way each time:

1. **The validation loss**: the training loss on a fixed draw of validation texts (4,096 by default).
2. **Detection through the bank**: a bank and validation texts (3,000 per origin by default, spread evenly over
   clusters) are read in evaluation mode, every text chunked as inference reads it. By default the bank is 4,096 texts
   per origin of the bank split, spread evenly over clusters; with `--bank-recipe`, it is the bank the export draws by
   that recipe for the same seed and size, which is how the release run read its own bank (`mixed_out`). From
   $`\mathrm{score} = 1 - r_{\mathrm{human}}`$: PR AUC, ROC AUC, and balanced MCC of AI or mixed text against human
   text, and the PR AUC of human against AI alone ([7. Evaluation](evaluation.md#the-metrics) defines each). When a
   run is given a teacher, the frozen teacher's numbers on the same texts and bank are logged beside them; the release
   run had no teacher.

By default the run keeps the checkpoint with the **lowest validation loss** (`best.pt`) and the latest (`last.pt`).
Test is never read during training. When a teacher is given, the code's rule for choosing among sizes is the smallest
student within 0.01 PR AUC of the teacher on validation, human against AI. The release was not chosen by either
default: a watcher outside `train.py` kept each reading's checkpoint and stopped the run by a patience rule on
validation PR AUC, and the best reading after every origin's first full pass became the release's `best.pt`.
How the release was chosen: <!--litmus:record.training.selection-->by a patience rule (Logan, 2026-11-18), not the lowest validation loss: after every origin had a full pass (step 19,466), every reading's checkpoint was kept, training stopped once ten readings in a row brought no new best validation PR AUC of AI or mixed against human (reading 65, step 32,499), and the release is the best reading in that region, reading 55 (step 27,499, PR AUC 0.9458), ahead of the lowest-loss candidate (reading 47) on every validation row but one MCC. Only `s192_l4` was trained, labels only, so the teacher-margin rule did not apply<!--/litmus-->.

## The release run

| Setting | Code default | Release |
| --- | --- | --- |
| Configuration | | <!--litmus:model.config_name-->s192_l4<!--/litmus--> |
| Teacher | a teacher export and the four distillation terms, unless `--labels-only` | <!--litmus:record.model.teacher-->None: this release is trained on the true labels alone; labels alone read level with or ahead of distillation at equal steps (E4), and the second build has no teacher export<!--/litmus--> |
| Weights | ternary, absmean | <!--litmus:record.ablations.a1.outcome-->ternary, absmean (A1)<!--/litmus--> |
| Precision schedule | quantized from step 0 | <!--litmus:record.ablations.a2.outcome-->quantized from step 0 (A2)<!--/litmus--> |
| Pooling | mean and variance | <!--litmus:record.ablations.a3.outcome-->mean and variance (A3)<!--/litmus--> |
| Head widths $`d_{\mathrm{hid}}`$, $`w`$, clamp | 512, 128, 10 | <!--litmus:record.ablations.a4.outcome-->256, 64, 30 (A4 and E8)<!--/litmus--> |
| Attention | float | <!--litmus:model.attention-->integer<!--/litmus--> |
| Query and key output | float | <!--litmus:model.output-->int8<!--/litmus--> |
| Texts per step $`b`$ | 2,048 | <!--litmus:record.training.texts_per_step as int-->1,024<!--/litmus--> |
| Micro-batch | 65,536 positions | <!--litmus:record.training.micro_batch-->262,144 positions<!--/litmus--> |
| Peaks $`\eta_1`$, $`\eta_2`$; warmup | 1.5e-3, 1e-3; 100 steps | <!--litmus:record.training.peaks_warmup-->1.5e-3, 1e-3; 100 steps (the search's schedule settings were not taken)<!--/litmus--> |
| Steps or time box | 2,000 steps | <!--litmus:record.training.steps_or_box-->a 240-minute box with a 60,000-step cap; stopped by patience at step 32,499 after 3 h 5 min<!--/litmus--> |
| Evaluation interval | 250 steps | <!--litmus:record.training.eval_interval-->500 steps<!--/litmus--> |
| Seed | 0 | <!--litmus:record.training.seed-->0<!--/litmus--> |
| Steps run, texts and tokens seen | | <!--litmus:record.training.seen-->32,500 steps of 1,024 texts, 33.3 million draws: 1.67 passes over the human training rows, 1.75 over the AI and 23.2 over the mixed; about 6.3e9 tokens (568,700 a second for 3 h 5 min)<!--/litmus--> |
| Chosen checkpoint (step) and its validation numbers | | <!--litmus:record.training.chosen-->Reading 55, step 27,499: training's validation PR AUC 0.9458; its release on validation reads PR AUC 0.9491, ROC AUC 0.9128 and balanced MCC 0.6503 (6,000 passages, AI or mixed against human)<!--/litmus--> |
| GPU and wall time | | <!--litmus:record.training.compute-->One NVIDIA GH200: 3 h 5 min of training at 568,700 training tokens a second<!--/litmus--> |

## Where in the code

| What | Function (`litmus_student`) |
| --- | --- |
| Pair logits $`S`$ | `distill.pair_logits` |
| Targets $`Y`$ | `distill.same_origin` |
| The loss, and the distillation terms | `distill.distillation_losses`; the release's weights: `train.labels_only` |
| Binary cross-entropy from logits | `foundry.layers.cross_entropy.binary_cross_entropy_with_logits` |
| Drawing a step's texts | `data.ClusterSampler.draw` |
| Packing into block-aligned streams | `data.pack`, `data.packed_batch`; the mask `model.block_mask` |
| The gradient-cached step | `train.train_step` |
| Learning rate, weight decay, time box | `train.learning_rate`, `train.weight_decay`, `train.scheduled_step` |
| Weight decay on matrices only | `train.parameter_groups` |
| Validation readings and metrics | `train.Evaluation`, `train.chunked_shares`, `train.detection_metrics` |
| Checkpoints and the size rule | `train.save_checkpoint`, `train.release_margin` |

---

Previous: [3. The model](model.md) · Next: [5. Calibration and the reading](calibration.md)
