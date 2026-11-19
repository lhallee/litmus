# Litmus teacher (`litmus_teacher`)

Embedding with a frozen encoder, e-means splits, the pair head, the reference bank, calibration and the export the
student distills from. Runs in Docker on one H100 GPU ([`../launch/`](../launch/README.md)). The calibration table's layout, which the student and the
engine read, is [CALIBRATION.md](CALIBRATION.md).

The teacher's encoder is ModernBERT-base (`answerdotai/ModernBERT-base`, 768 wide, pinned revision `8949b90`).
ModernBERT-large and mmBERT-base are comparison arms embedded on a subsample and read against base on validation.
ModernBERT-base and -large share one tokenizer: their `tokenizer.json` files have the same vocabulary, merges,
added tokens, normalizer and pre-tokenizer, and the same template, `[CLS]` (50281) text `[SEP]` (50282). So the
corpus `tokens` column is either model's count. mmBERT-base wraps a text as `<bos>` (2) text `<eos>` (1). The code
reads each template rather than assuming it.

## Run order

Each step is `python -m litmus_teacher <step>`, is resumable, and writes `run_summary.json` in its `--out`. The
image needs this folder's `pyproject.toml`, `flash-attn` (for packed embedding; `--attention sdpa` runs without it,
padded and slower), and the vendored `foundry` package (in `training/`) on `PYTHONPATH` (logging, metrics, cross entropy).
`F` is the predecessor corpus's `embeddings/modernbert_base_mean_variance.npy`, downloaded beforehand.

| Step | Command (abridged) | Writes |
| --- | --- | --- |
| Pilot | `embed --corpus C --out E/modernbert_base --reuse F --pilot` (again with `--compile`) | `pilot.json`: tokens per second by piece length, projected hours net of reuse, packing check, bfloat16 drift, the reuse check; `drift.npz`, `reuse_drift.npz` |
| Embed | `embed --corpus C --out E/modernbert_base --reuse F` | `embeddings.npy` (n, 2d) float16 in `id` order, `tokens.npy`, `reused.npy`, `slices/*.done` |
| Arms | `embed --corpus C --encoder modernbert_large --out E/modernbert_large --ids I.npy` (and `mmbert_base`) | the same, for the subsample |
| Split | `split --corpus C --embeddings E/modernbert_base --out S` | `split.parquet` (`id`, `cluster`, `split`), `centroids.npy`, `split_summary.json` with the distance report |
| Arm comparison | `train ... --features modernbert_base modernbert_large mmbert_base --common-rows --output-widths 512 2048 --out T_arms` | validation pair loss of each arm on the same texts, `selection.json` |
| Search pilot | `search --corpus C --split S/split.parquet --embedding-root E --out H --experiment-id YYYY-MM-DD_slug --pilot --max-minutes 3` | `H/pilot/search_summary.json`: head-steps per second and the trials a 30-minute box holds |
| Search | `search --corpus C --split S/split.parquet --embedding-root E --out H --experiment-id YYYY-MM-DD_slug --max-minutes 30` | `study.journal`, `trials.json`, `leaderboard.md`, then the runs of record: `heads/<name>/head/`, `retrain.json`, `selection.json` |
| Baselines | `train --corpus C --split S/split.parquet --embedding-root E --out T --experiment-id YYYY-MM-DD_slug --baselines-only --control --logistic` | the random-vector control head (default recipe), `logistic/modernbert_base/predictions.npz` |
| Bank | `bank --corpus C --split S/split.parquet --embedding-root E --train-dir H --out B --drift E/modernbert_base/pilot/drift.npz E/modernbert_base/pilot/reuse_drift.npz` | `bank.safetensors`, `bank_selection.json` with score drift, `readings_{validation,test,calibration}.npz` |
| Control bank | `bank ... --train-dir T --head random_w2048_s0 --variant <chosen> --size <chosen> --out B_control` | the control's readings |
| Calibrate | `calibrate --corpus C --bank-dir B --control-bank-dir B_control --logistic T/logistic/*/predictions.npz --out K` | `calibration.json`, `metrics.csv`, `uncertainty.csv`, `thresholds.json`; reads test once |
| Export | `export --train-dir H --embedding-root E --split S/split.parquet --bank-dir B --calibration K/calibration.json --out X` | `manifest.json`, `head/`, `query_key.npy` (n, 2w) float16, `present.npy` |

[`../launch/teacher_chain.sh`](../launch/teacher_chain.sh) runs the steps after embedding in one go (split, search,
bank, export, baselines, control bank, calibrate), with the export ahead of the baselines so the student can start;
the arm comparison is a separate step.

Every step takes a small mode for a smoke run: `--pilot` (embed, train, search) or `--limit` (embed, split, export).
`calibrate.uncertainty_rows` and `evaluate.model_rows` score any labeled set, such as RAID, from its readings.

## The head's search

The teacher head comes from a Bayesian search (`search.py`), not from fixed runs. Its docstring
states the space, what is fixed and why, the protocol and the choice rule. In brief:

- **Searched**: learning rate, weight decay, dropout, depth (1 to 3) and each hidden width (512 to 8,192), output
  width (256 to 2,048), batch per side, warmup, schedule, input clamp, the pair logit's scale (learned or fixed, and
  its start) and bias start, and label smoothing. Trial 0 is the default recipe.
- **Objective**: each trial's lowest validation pair loss on 16 fixed pairs of balanced validation batches; the median
  pruner reads it every 500 steps. A trial trains at most 8,000 steps. No trial reads test, calibration or a bank.
- **Lockstep**: features sit on the GPU once, and up to 8 heads train in one process on shared batches. A head with a
  smaller batch takes a balanced prefix of the shared draw, so heads of different batch sizes train together.
- **Time box**: `--max-minutes` stops new trials; trials in flight finish. Rerunning the command resumes the study
  from `study.journal`.
- **Runs of record**: the 5 best configurations and the default, 3 seeds each, logged through `foundry.logging`
  (trials make no W&B run), read through the bank on validation. The teacher is the lowest mean pair loss, unless the
  best mean bank PR AUC beats it by more than the seed spread. `retrain.json` reports the gain over the default with
  each configuration's seed standard deviation.

## Reusing the predecessor corpus's vectors

A ModernBERT-base row may copy its vector from the predecessor corpus when its text is byte-identical (`text_unchanged`)
and `tokens + 2 <= 4,096`, the predecessor corpus's cap. The predecessor corpus pooled the same way: mean then variance over the
last states, special tokens included. The `--pilot` run recomputes 2,000 reusable rows, compares them with the stored vectors
(cosine, mean absolute difference, relative L2), and ends in one of three outcomes:

| Outcome | When | Full run |
| --- | --- | --- |
| `safe` | The stored vectors pass the three rules below as they are | Copies them as stored |
| `aligned` | They fail, but a per-column affine map fitted on half the pairs makes the other half pass | Copies them through that map |
| `refused` | They fail even aligned | Copies nothing; every row is computed |

The three rules guard against differences a head could see. Reused rows hold no mixed text, so such a difference
would be correlated with origin:

1. **Shift.** A linear reader tells stored from recomputed vectors at a held-out ROC AUC above 0.6.
2. **Spread.** The same reader, on squared standardized columns, does the same. An affine alignment zeroes every
   column's mean difference by construction, so rule 1 alone could not see what it leaves behind. Rule 2 sees a
   changed scale or a bent column. In the tests a planted non-affine bend is invisible to rule 1 (AUC 0.52 after
   alignment) and caught by rule 2 (0.78) and by rule 3.
3. **Noise.** The 99th percentile of the paired `1 - cosine` exceeds twice the noise between two of our own bfloat16
   packings.

The alignment's form is chosen by validation inside the fitting half: a shift alone or a shift and a scale, either
for all lengths or per length group (under 256, 256 to 1,023, and 1,024 or more tokens). `per_length_error_ratio`
below 1 says the offset depends on token count. The `--pilot` run writes the outcome, the alignment and its form to
`pilot.json`, and the map to `pilot/reuse_alignment.npz`. The full run records the outcome and the alignment in
`run_summary.json` and `meta.json`.

`bank --drift ... reuse_drift.npz reuse_drift_aligned.npz` reads the raw and aligned pairs through the trained head.
Its `reuse_score_check` fails a pair whose two-class reading moves more than twice as much as two packings, or that
has a direction. The full run stops without a `--pilot` reuse check. `--accept-reuse-risk` copies as stored over a
missing check or a `refused` one, and is recorded.

## Cost on an H100

Projected from FLOP counts; the measured embedding time is under "Measured" below. Rates come from forward FLOPs per token at 40%
(ModernBERT-large) and 25% (the base models, whose smaller matrices use the GPU less well) of the H100 PCIe's 756
dense bfloat16 TFLOPS. Per token, ModernBERT-large costs about 0.7 GFLOP and the base models about 0.22, each plus
attention.

| Step | Projection |
| --- | --- |
| Embed, ModernBERT-base (the teacher) | 450k to 800k processed tokens/s: about 0.45 h per billion tokens computed |
| Embed, mmBERT-base (arm) | the same: about 0.45 h per billion tokens |
| Embed, ModernBERT-large (arm) | 300k to 400k tokens/s: about 0.85 h per billion tokens |
| Split | under 0.1 h: e-means is seconds; reading the mean halves and the host stages (2 s per million rows, measured on CPU) dominate |
| Search | 0.5 h box: 75 to 125 trials. A head-step averages about 0.7 TFLOP over the space (2,880 texts per side, 18 million multiply-adds per text), 3 to 5 ms at 30% of peak. A full 8,000-step trial takes 25 to 40 s, and pruning cuts the mean trial to about 60% of that. `search --pilot` measures the rate |
| Runs of record | 18 runs of at most 8,000 steps, about 0.2 to 0.3 h with the bank readings |
| Train (arms, baselines) | 3 to 6 min per run with features held on the GPU once per encoder |
| Bank | under 0.1 h |
| Calibrate | CPU only, minutes |
| Export | 0.1 to 0.2 h, reading every embedding once |

For 5 to 8 million texts at a mean of 500 to 1,000 tokens (2.5 to 8 billion tokens, plus up to 14% window overlap
above 8,190 tokens), ModernBERT-base over everything costs 1.1 to 3.6 GPU hours before reuse. Reuse removes the
rows the predecessor corpus already embedded, which are the short ones (at most 4,094 tokens).

## Measured

On the corpus in [`../corpus/README.md`](../corpus/README.md), one H100 PCIe GPU, in Docker:

- **Embedding** ran padded under SDPA (`--attention sdpa`): the image has no flash-attention build and
  transformers 5.17.0's ModernBERT has no `cu_seq_lens` path, so the packed path is untested. 12,653,367 rows
  took 2.62 hours on a shared GPU; at 262,144 positions a batch the `--pilot` run measured 350,970 tokens a second for pieces up to 512
  tokens, 256,991 up to 2,048, and 111,577 up to 8,192. bfloat16 against float32: median `1 - cosine` 1.2e-5, worst
  7.5e-3, all of the worst under 512 tokens.
- **Reuse: `safe`.** On 2,000 reusable rows a linear reader told stored from recomputed vectors at ROC AUC 0.524
  (shift) and 0.519 (spread), and 4,699,080 predecessor rows copied their stored vectors.
- **Split**: e-means found 3,934 clusters in 5 epochs, in 3 minutes. Mixed text is 2.8% of the corpus and mostly
  short, so human and AI balance each other per cell, a cell keeps at most as many mixed rows as human ones, and a
  set at most 25% of its per-origin target in mixed rows (`--mixed-share`).
- **Search, bank, export, baselines, calibration**: run end to end at toy size, and at full size on the first
  corpus, where the searched head and the bank draw were chosen
  ([`../docs/training/experiments.md`](../docs/training/experiments.md), E7 and E9). The released student trains on
  labels alone, which read level with or ahead of distillation at equal steps (E4), so the release uses the teacher's
  embeddings for the split and no teacher head.

Untested: the packed flash-attention path, `torch.compile`, and whether 8,000 steps rank configurations as
full training would.

The unit tests (`pytest`, CPU, no network, about 30 seconds) cover:

- window placement and exact moment combination;
- packing and slice planning;
- the reuse copy, its eligibility rules, the shift measure and the gate;
- split invariants: disjoint by cluster and doc, moats, balance and caps;
- three-origin pair labels and the head;
- the search on a tiny synthetic study: a head in lockstep trains as it would alone, balanced prefixes, pruning, the
  time box, resuming from the journal (a stale trial failed, the anchor enqueued once), no read of a held-out split,
  the choice rule, and the runs of record through the bank to `selection.json`;
- bank scoring against the NumPy definition, and the bank choice;
- threshold carrying;
- calibration: coverage on exchangeable data, its failure under a shift, exact grid p-values, Venn-Abers bounds,
  merging, and the table round trip.
