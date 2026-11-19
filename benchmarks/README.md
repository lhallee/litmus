# Litmus benchmarks

How Litmus is compared with commercial and open detectors of model-written text on public benchmarks: which
benchmarks and metrics, which detectors we rerun and which we only quote, what the comparison cannot claim, and the
figures. Every number in `published.csv` was read at its source on 2026-11-17 and carries its URL, table and date.

| File | Holds |
| --- | --- |
| `catalog.csv` | 17 benchmark splits with labels on the Hub, keyed exactly by `repo`, `config`, `split` and a pinned `revision`; the corpus builder holds each out of training |
| `published.csv` | 294 published numbers: detector, kind, benchmark, subset, metric, value, protocol, source and who reported it |
| `detectors.csv` | 12 runnable open detectors, ranked by what the comparison gains, and 8 commercial APIs with price, limits and terms |
| `architectures.csv` | Architecture and size of every measured classifier and every RAID leaderboard model public on the Hub, for the throughput estimates |
| `cost_chain.sh` | Single-core cost of named detectors, one process each, in the launch image |
| `src/litmus_benchmarks/gpu_use.py` | GPU utilization and memory of named detectors reading long documents, sampled from `nvidia-smi`, with Litmus's engine under the same sampling (the ad's GPU figure) |
| `src/litmus_benchmarks/` | Loader, scorer adapters, metrics with bootstrap intervals, single-core cost and its estimates, results and figures |

A `config` written `file:<path>` names one file of the repo, read whole, with `split` `train`; MAGE, HC3 and
AuTexTification load through scripts the datasets library no longer runs, so their test files are read directly.

## Headline

Benchmarks with public labels, scored by us on a seeded sample of 2,000 texts each (`--max-rows 2000 --seed 0`):

- **RAID**, the field's shared reference and the only one most vendors appear on. The test labels are hidden, so
  Litmus is scored by submitting predictions to the leaderboard (official metric: TPR at 5% FPR with per-domain
  thresholds, also at 1% FPR); `raid_leaderboard_metrics` reads the result back. `raid_extra` (code, Czech, German)
  is scored locally.
- **MAGE** test and its two wilder sets (unseen domain and model; paraphrase), official AvgRec and AUROC.
- **SemEval-2024 Task 8** subtask A, monolingual and multilingual (M4GT-Bench's data), official accuracy.
- **Beemo** (expert- and LLM-edited text) and **APT-Eval** with its human originals (AI-polished human text), the
  mixed-origin tests; **MixSet**.
- **DroidCollection** and **CoDET-M4** for code; **AuTexTification** English and Spanish; **HC3**.
- **EditLens** is listed, but its gated columns are not in the catalog, so it is not scored.

The corpus reader cannot open EditLens, HC3 or either AuTexTification set, so their texts are not held out of the
corpus by content, and the comparison leaves them out. APT-Eval and its human originals hold
one origin each, so they give flag rates, not a task with both classes.

Every benchmark and stratum reports PR AUC, ROC AUC, balanced MCC (`foundry.metrics.balanced_mcc` from the vendored
`training/foundry` package: MCC on class-balanced subsamples at its best threshold) and TPR at 1% and 5% FPR, each with a stratified bootstrap
95% interval (1,000 replicates overall, 200 per stratum, the same draws for every detector so differences are paired),
beside the benchmark's official metric. Three tasks: `ai_vs_human` (mixed left out), `any_ai_vs_human` (mixed counted
as AI) and `mixed_vs_human`. Strata: domain, generator, attack and length bin (1-50, 51-150, 151-400, 401-1000,
over 1,000 words); a generator is scored against every human text, the rest within the stratum.

## Detectors

**Rerun on a GPU** (`detectors.csv`, ranked): Litmus's predecessor `lhallee/ai-vs-human-checker`, Binoculars
(Falcon-7B pair), Desklib v1.01, Fast-DetectGPT (Falcon-7B-instruct, analytic), e5-small-lora, OpenAI RoBERTa-base,
RADAR, the ModernBERT RAID+MAGE classifier and the MAGE Longformer. At 16,000 texts of up to 512 tokens the two 7B
methods dominate: Binoculars is about 20 GPU minutes and Fast-DetectGPT about 10 by a FLOP estimate (2 x parameters x
tokens at an assumed 200 TFLOP/s), so the two 7B methods can run on a 1,000-text sample per benchmark, and a result
says when they do.

**Quoted, never rerun**: every commercial detector and every leaderboard or paper number. Quoted points are hollow in
the figures. A head-to-head on a fixed 2,000-text subsample (assuming 400 words a text) would cost, at list price:
Sapling $24, ZeroGPT $29.98, Winston AI $98, Originality.ai $179, Pangram $400 ($320 bulk); GPTZero and Copyleaks
publish no API price and Turnitin no public API. Originality.ai's terms forbid publishing comparative benchmarks
without written consent and ZeroGPT's forbid benchmarking a competing detector without authorization.

## What the comparison cannot claim

- **Different protocols are not one ranking.** RAID leaderboard numbers are TPR at fixed FPR on hidden labels; vendor
  model cards report accuracy on their own samples (Pangram's "RAID, random sample" accuracy is not the leaderboard
  metric); papers choose thresholds differently (APT-Eval maximizes accuracy, Beemo reports AUROC only). Figures draw
  a quoted number only beside the same benchmark, subset and metric.
- **Other detectors' training overlaps the benchmarks.** e5-small-lora and SuperAnnotate trained on RAID train, the
  MAGE Longformer on MAGE, the Drayson ModernBERT on MAGE and RAID (its card), Pangram's DAMAGE on some RAID train, and leaderboard
  entries tuned on RAID (one is named "RAID-tuned"). Their numbers on those benchmarks are in-distribution; ours are
  reported as held out only where the corpus builder confirms it.
- **Litmus's held-out guarantee covers the splits in `catalog.csv`, nothing else.** Benchmarks found but not
  cataloged (below) may share sources with the training corpus; numbers on them carry that caveat.
- **Vendor and independent numbers part.** GPTZero on RAID without attacks reads 0.665 in the RAID paper (2024) and
  0.984 in its own 2025 leaderboard submission (a newer model); Pangram reads 100% on Russell et al.'s 300 articles in
  its model card and TPR 0.980 at 2% FPR in the paper; RADAR on RAID paraphrase reads 0.673 (RAID paper) and 0.624
  (DAMAGE, by Pangram's authors); GPTZero's page reports Pangram 4 on DetectRL from its own run. The claims figure
  shows every such pair from `published.csv`.
- A single sample of 2,000 texts per benchmark carries the stated intervals and no more; costs come from one machine.

## Figures

PNG at 300 dpi, no titles, panels lettered, each in the site's dark and light looks on a transparent ground
(`../site/assets/figures/<name>_<theme>.png`, copied to `../docs/images/benchmarks_<name>_<theme>.png`). Litmus is
the one accent (bluish green); every other detector is a neutral gray with its own marker, the same in every figure.
A number we computed is filled with its bootstrap 95% interval as a line, a leaderboard number is hollow, a
detector's number on a benchmark it trained on is faded, and an estimated throughput is lighter still. Origins use the
litmus scale shared with the app: blue human, purple mixed, red AI, with a hatch per origin.

| File | Panels | Shows |
| --- | --- | --- |
| `headline` | A PR AUC, B ROC AUC, C balanced MCC, D TPR at 1% FPR | Each two-class benchmark a row, ordered by Litmus's PR AUC; every rerun detector with its interval |
| `raid` | A per attack, B per language, C overall | RAID extra (Czech and German news): TPR at 5% FPR per attack (each attack's AI texts against every human text), hardest first, with the leaderboard's per-attack numbers of the same detectors hollow; per language; overall beside the leaderboard's quoted entries |
| `pareto` | A unseen, B all, C RAID leaderboard | Accuracy against tokens a second on one core (log), frontier dashed: A and B mean PR AUC, C the leaderboard's TPR at 5% FPR against measured or estimated throughput |
| `strata` | A by length, B by benchmark | A: PR AUC per length bin over the pooled benchmarks; B: detector by benchmark PR AUC, a dagger where the detector trained on it |
| `origins` | A distributions, B flagged share | A: Litmus's probability of not human by origin on its own held-out split; B: share of mixed and AI text flagged at each benchmark's 1%-FPR threshold |

**Which benchmarks a detector trained on.** Each row of the results and each mark in the figures says whether the
detector trained on the benchmark: e5-small-lora, Desklib v1.01 (its card lists RAID) and the Drayson ModernBERT on
RAID train, so RAID extra is in distribution for them; the Drayson ModernBERT also on MAGE; the predecessor on MAGE's
train, validation and test files, RAID train, and SemEval-2024 Task 8 A train and dev (`../corpus/sources.csv`,
`in_first_corpus`); Binoculars and Fast-DetectGPT train on nothing. Litmus's own come from `catalog.csv`'s
`in_distribution` column, or from `--litmus-trained-on` for a build the catalog does not describe. A benchmark's rows
are filtered by its `row_filter` before any metric, so RAID extra is its Czech and German news.

**A fair accuracy for the frontier.** PR AUC (`ai_vs_human`) averaged over the two-class benchmarks that every
compared detector scored and none trained on (panel A), and over every two-class benchmark they all scored (panel B),
each with a bootstrap interval from the same draws for every detector. Averaging only where no detector trained keeps
supervised detectors from gaining on their own training data; requiring every detector's scores keeps the means over
one set. The results file lists both sets.

**Single-core throughput.** Measured on one machine's processor, one thread (`OMP_NUM_THREADS=1` before Python,
`torch.set_num_threads(1)`), each detector in a fresh process in a CPU-only container, the same 200 texts of MAGE's
test file with 100 to 300 words (short enough that no detector truncates) after one warm-up text, batch size 1,
float32 for the PyTorch detectors and Litmus's engine through its Python wheel (the whole reading, windows included).
Binoculars and Fast-DetectGPT, whose 7B models take a minute or more a text on one core, read the first 20 of the
same texts. Throughput is tokens a second in each detector's own tokenizer (these tokenizers give 1.20 to 1.26 tokens
a word on this set) and seconds per 1,000 whitespace words. For a RAID leaderboard entry we cannot run, throughput is estimated only
when its model is public on the Hub: per-token compute is about twice the non-embedding parameters, so the estimate is
the measured throughput of the same-architecture detector nearest in size, scaled by the ratio of non-embedding
parameters (`estimate.py`, sizes in `architectures.csv`). An entry more than ten times larger or smaller than every
measured model of its architecture, or whose model is not public (every commercial API but GPTTrace), gets no
estimate. `cost_chain.sh` runs the measurements in the launch image.

**GPU use.** `python -m litmus_benchmarks gpu-use --documents FOLDER... --out gpu_use.json` reads folders of `.txt` documents with each
named detector in a process of its own, after an idle baseline, while it polls `nvidia-smi` four times a second. Per detector it
writes the mean and peak GPU utilization (`utilization.gpu`, the share of time a kernel was running), the utilization added over the
idle baseline, the peak GPU memory of its own process (the per-process query, matched by process id; the report says when it had to
use the device's memory over idle instead), tokens as the detector reads them, documents, wall seconds and throughput. Litmus is the
engine as shipped, CPU only on one core, under the same polling: its GPU figures are measured, not assumed. `--pilot` reads 12 documents.
The default detectors are the five the benchmark results cover (Fast-DetectGPT, Desklib v1.01, ModernBERT RAID+MAGE (Drayson),
e5-small-lora, Binoculars) and `Litmus engine`; the other open ones with an adapter (`gpu_use.GPU_DETECTORS`) stay selectable with `--detectors`.
Two limits found on 2026-11-18: MAGE Longformer does not load under the launch image's transformers (its config's `id2label` fails
validation), and Binoculars (two 7B models) needs more than a 24 GB GPU (it ran out of memory loading on an NVIDIA A10, 22.00 GiB in use of
22.06) and was measured on a GH200; run it in a separate call with `--detectors Binoculars --out ...`. Every detector but Litmus truncates each
document to its input limit (512 tokens, Desklib 768), so it reads far fewer tokens than Litmus's engine, which reads them all; the JSON says how many
each read, and a fast detector's window is a few polls (e5-small-lora: 7).

## Run

```
python -m litmus_benchmarks score --benchmarks mage beemo --detectors Binoculars "Desklib v1.01" --out scores.csv
python -m litmus_benchmarks metrics --scores scores.csv --out metrics.csv
python -m litmus_benchmarks cost --benchmark mage --detector e5-small-lora --out costs.csv
python -m litmus_benchmarks architectures --out architectures.csv
```

**One command for every result.** From folders of score files, Litmus's `evaluation.json` and the cost rows:

```
python -m litmus_benchmarks results --scores BASELINES LITMUS --evaluation EVALUATION.json \
    --costs costs_small.csv costs_fast_detectgpt.csv costs_binoculars.csv \
    --model-version VERSION --work WORK --site ../site/assets --docs ../docs/images
```

It writes `../site/assets/results.json` (the fields `site/README.md` defines, keeping its `release` block, plus a
`comparison` block with the row filters, the benchmark sets, the means and every text set split or dropped), every
figure in both looks, and in `WORK` the cached metric table, the flag rates, the cost table and `results.md`, the
Markdown tables for the README and the model card. Detectors that scored different samples of one benchmark are
compared within their own sample, and a text any detector left unscored leaves its sample for every detector. Flag
rates on APT-Eval and its human originals are read at each detector's 1% and 5% FPR thresholds, set on the human texts
of the two-class benchmarks every detector scored, and at its own documented threshold. Pass `--litmus-trained-on`
with no names for a build that held every benchmark out.

The metric table is the slow step (about two CPU hours at 1,000 replicates overall and 200 per stratum). `--workers N`
splits it over processes, and on a cloud machine the launch image computes it alone with `--metrics-only` (it needs no
plotting library), into `WORK/metrics_<digest>.csv`; the full command then reads it with `--metrics` and checks that
it covers the same detectors and samples.

Scoring runs in the launch image on a GPU machine (`.[detectors]`); Litmus enters as one more scorer. Tests:
`python -m pytest` in this folder, seconds on a CPU.

`--detectors` takes the adapter names of `src/litmus_benchmarks/detectors.py`: `Litmus`, `ai-vs-human-checker`,
`Binoculars`, `Fast-DetectGPT`, `Desklib v1.01`, `e5-small-lora`, `OpenAI RoBERTa-base (GPT-2)`, `RADAR`,
`ModernBERT RAID+MAGE (Drayson)`, `MAGE Longformer` and `RoBERTa ChatGPT (HC3)`. `Litmus` reads its checkpoint and
release from `LITMUS_CHECKPOINT` and `LITMUS_RELEASE` and scores `1 - P(human)` of its calibrated reading;
`ai-vs-human-checker` needs its own package (`ai_text_checker`) importable. On a GPU machine
[`../launch/benchmark_chain.sh`](../launch/benchmark_chain.sh) scores one benchmark and detector per process into
`scores_<benchmark>__<detector>.csv`; its `DETECTORS=baselines` is the predecessor, Desklib v1.01, the Drayson
ModernBERT, e5-small-lora, Fast-DetectGPT and Binoculars, the six rerun beside Litmus.

## Found but not cataloged

M4 and M4GT-Bench (GitHub `mbzuai-nlp`), DetectRL (GitHub `NLP2CT/DetectRL`), Ghostbuster (GitHub), MULTITuDE
(Zenodo), PAN 2025 Voight-Kampff (scored through TIRA submissions), CUDRT, LLM-DetectAIve, TuringBench (a script over a
zip), DetectRL-X (its test files are held back for a shared task), RealDet, the COLING 2025 Task 1 test set (only
train and dev are on the Hub), AIGCodeSet (its test split's first rows equal its train split's), and EditLens's
Grammarly split. `pangram/WildAI` is labeled by Pangram itself, so it is not ground truth. Published numbers from
several of these are in `published.csv`.

## Unverified

Not checked against their sources: the DroidCollection paper; the SemEval organizer baseline's model and the MAGE
test split's testbed; GPTZero's and Originality.ai's input limits, Copyleaks's price and terms, and Turnitin's terms;
and the Epoch AI and Graphite benchmarks GPTZero cites. Pangram's terms were read only through a page summary.
