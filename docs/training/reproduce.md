# 9. Reproduce it

[Front page](../TRAINING.md) · [1. Data](data.md) · [2. Splits](splits.md) · [3. Model](model.md) ·
[4. Training](training.md) · [5. Calibration](calibration.md) · [6. Experiments](experiments.md) ·
[7. Evaluation](evaluation.md) · [8. Engine](engine.md) · **9. Reproduce**

Everything runs in one Docker image on one GPU machine, each step resumable, each chain skipping a step whose output
exists ([`launch/README.md`](../../launch/README.md)). The corpus build reads public datasets from the Hugging Face
Hub, so it needs network access and, for gated datasets, a Hub account that has accepted their terms.

- [The image](#the-image)
- [The commands, in order](#the-commands-in-order)
- [The compute and what it cost](#the-compute-and-what-it-cost)
- [Every formula and the code that computes it](#every-formula-and-the-code-that-computes-it)

## The image

```sh
docker build -t litmus -f launch/Dockerfile --build-arg PROJECT=. .
```

CUDA 13.0 and PyTorch 2.14.0 by default (`--build-arg CUDA_IMAGE=... --build-arg TORCH_INDEX=cu126` selects CUDA 12.6
for an older driver), Python 3.12, and the pinned [`launch/requirements.txt`](../../launch/requirements.txt). Run the
steps inside it with each package's `src` folder and `student/` on `PYTHONPATH`; outputs go under `/work`.

## The commands, in order

`CORPUS` is `/work/litmus_corpus/corpus/data`. The release's exact arguments are these:

<!--litmus:record.training.command_lines-->
```sh
# Corpus 2, on CPU (launch/corpus2_chain.sh): the new sources built with their benchmarks held out, then the first corpus extended
python -m litmus_corpus build --out /work/corpus2/build --workers 10 --priority 3 --new-token-budget 1e9 --no-first-corpus \
    --family-train --first-index /work/litmus_corpus/index --earlier-build /work/litmus_corpus
python -m litmus_corpus extend --earlier /work/litmus_corpus/corpus/data --build /work/corpus2/build --out /work/corpus2/run2/corpus \
    --workers 10 --trained-first-source raid --trained-first-source mage_train --trained-first-source mage_validation \
    --trained-first-source ghostbuster_train --trained-first-source coling2025_en --trained-first-source semeval2024_a \
    --group-source raid_train --group-source raid_extra_code
# Teacher embeddings, reusing the first corpus's stored vectors for unchanged texts
python -m litmus_teacher embed --corpus /work/corpus2/run2/corpus/data --out /work/corpus2/embed/modernbert_base --attention sdpa \
    --workers 12 --batch-tokens 262144 --earlier-store /work/embed/modernbert_base
# Token store, then the split and the store's rows
python -m litmus_student.data --corpus /work/corpus2/run2/corpus/data --out /work/corpus2/student_store --tokens-only
python -m litmus_teacher split --corpus /work/corpus2/run2/corpus/data --embeddings /work/corpus2/embed/modernbert_base \
    --out /work/corpus2/teacher/split_v11 --keep-docs-together \
    --targets test=20000,calibration=20000,validation=10000,bank=9300,bank_draw=0 --mixed-share 1.0 \
    --rows-per-cluster 1000 --cap 50 --train-language-ratio 1.5 --held-out-languages en und
python -m litmus_student.data --corpus /work/corpus2/run2/corpus/data --out /work/corpus2/student_store_v11 \
    --split /work/corpus2/teacher/split_v11/split.parquet   # the token store's tokens, linked, with this split's rows
# The text store the presentation reads, and the synonym table (WordNet 3.0 through nltk, in a throwaway folder)
python -m litmus_student.data --corpus /work/corpus2/run2/corpus/data --out /work/corpus2/student_texts --texts
bash experiments/tooling/quill/2026-11-19_canonical_retrain/scripts/synonym_table.sh /work/corpus2/synonyms/wordnet.json
# Training, labels only, through launch/student_chain.sh (experiments/.../2026-11-19_canonical_retrain/scripts/v11_queue.sh)
python -m litmus_student.train --config s512_l6 --data /work/corpus2/student_store_v11 --experiment-id 2026-11-19_canonical_retrain \
    --out /work/corpus2/student_v11/s512_l6_labels_only/train --steps 60000 --minutes 0 --texts-per-step 1024 \
    --micro-tokens 65536 --eval-every 500 --memory-fraction 1.0 --settings /work/student/search/summary.json --labels-only \
    --bank-recipe reference --bank-per-origin 8192 --integer-attention --int8-output --cut-ends 0.5 --cut-starts 0.25 \
    --texts /work/corpus2/student_texts --present-workers 16 --synonyms /work/corpus2/synonyms/wordnet.json --uniform \
    --mlm-share 0.5 --mlm-weight 0.1 --epochs 4 --eval-every-epochs 0.5 --select pr_auc --patience 3 --eval-per-origin 10000
python -m litmus_student.export --checkpoint <run>/train/best.pt --data /work/corpus2/student_store_v11 \
    --corpus /work/corpus2/run2/corpus/data --out <run>/release --bank-recipe reference --bank-per-origin 8192
python -m litmus_student.evaluate --checkpoint <run>/train/best.pt --release <run>/release --data /work/corpus2/student_store_v11 \
    --experiment-id 2026-11-19_canonical_retrain --out <run>/evaluation_validation --split validation --long-documents 1500 \
    --spliced-per-cell 100
python -m litmus_student.evaluate ... --out <run>/evaluation_test --split test (the same arguments, read once)
```
<!--/litmus-->

```sh
# 1. Corpus: read, standardize, strata, passages, checks, hold-out, balance, write (the release's second build then
#    extends the first corpus with `litmus_corpus extend`; its arguments are above)
python -m litmus_corpus build --out /work/litmus_corpus --workers N --priority P --new-token-budget B \
    [--skip-benchmark NAME ...]

# 2. Token store (beside the embedding): every row tokenized as the engine reads it
python -m litmus_student.data --corpus $CORPUS --out /work/student_store --tokens-only

# 3. Embed with the frozen encoder: a pilot first, then the full run
python -m litmus_teacher embed --corpus $CORPUS --out /work/embed/modernbert_base --reuse F --pilot
python -m litmus_teacher embed --corpus $CORPUS --out /work/embed/modernbert_base --reuse F

# 4. Split; then, for a distilled build only, the teacher's head search, bank, export, baselines, calibration.
#    A labels-only build, such as the release, needs only the split (its own arguments are above)
bash launch/teacher_chain.sh

# 5. The token store's split and cluster columns
python -m litmus_student.data --corpus $CORPUS --out /work/student_store --split /work/teacher/split/split.parquet

# 6. Training: a pilot, then the run. The release trains on labels alone, from scratch, with no teacher export;
#    a distilled run adds --teacher-export /work/teacher/export and drops --labels-only. The release reads its
#    validation against the bank the export will draw (`reference`, the held-out bank set); its presentation reads
#    the corpus's texts from a text store (`python -m litmus_student.data --corpus $CORPUS --out /work/texts --texts`)
python -m litmus_student.train --config SIZE --data /work/student_store --experiment-id ID --out /work/student/SIZE \
    --labels-only --minutes M [--settings SEARCH/summary.json] [--bank-recipe reference --bank-per-origin N] --pilot
python -m litmus_student.train --config SIZE --data /work/student_store --experiment-id ID --out /work/student/SIZE \
    --labels-only [--settings SEARCH/summary.json] [--bank-recipe reference --bank-per-origin N] \
    [--uniform --texts /work/texts --synonyms SYNONYMS.json --cut-ends 0.5 --cut-starts 0.25 --mlm-share 0.5 \
     --epochs E --eval-every-epochs 0.5 --select pr_auc --patience 3]

# 7. Export the release: weights, bank, calibration, checked against the reference
python -m litmus_student.export --checkpoint /work/student/SIZE/best.pt --data /work/student_store \
    --corpus $CORPUS --out /work/release [--bank-recipe reference --bank-per-origin N] [--attention integer]

# 8. Evaluate: validation, then test once
python -m litmus_student.evaluate --checkpoint /work/student/SIZE/best.pt --release /work/release \
    --data /work/student_store --experiment-id ID --out /work/eval --split validation
python -m litmus_student.evaluate --checkpoint /work/student/SIZE/best.pt --release /work/release \
    --data /work/student_store --experiment-id ID --out /work/eval_test --split test

# 9. Benchmarks: Litmus and the rerun detectors on the held-out benchmarks
LITMUS_CHECKPOINT=/work/student/SIZE/best.pt LITMUS_RELEASE=/work/release \
    OUT=/work/benchmarks DETECTORS=Litmus bash launch/benchmark_chain.sh
OUT=/work/benchmarks DETECTORS=baselines bash launch/benchmark_chain.sh
python -m litmus_benchmarks metrics --scores scores.csv --out metrics.csv
python -m litmus_benchmarks cost --benchmark mage --detector Litmus --out costs.csv
```

`launch/student_chain.sh` runs steps 6 to 9 for one size (`LABELS_ONLY=1` for training on labels alone, and `TEACHER=`
empty for a corpus with no teacher export, as in the release). A step's `--pilot` runs a few hundred steps, reports tokens a
second, memory, and the projected hours, and makes no experiment-tracking run. Then build the engine and read the
release:

```sh
cd engine && cargo build --release
./target/release/litmus info --model /work/release
./target/release/litmus bench --model /work/release --tokens 128,1024,8192,131072
```

Tests: `python -m pytest` in `corpus/`, `teacher/`, `student/`, and `benchmarks/` (CPU, no network), and
`python tools/run_tests.py` for every suite ([`docs/AUDIT.md`](../AUDIT.md)).

## The compute and what it cost

<!--litmus:record.tables.compute-->
| Stage | Machine | Wall time |
| --- | --- | --- |
| Corpus 2: read, standardize, strata, passages, checks, hold-out, balance, extend | NVIDIA A10 machine, 10 CPU cores | 56 min |
| Token store, 4.52e9 tokens | NVIDIA GH200 machine, 12 CPU cores | 17 min |
| Teacher embeddings of 5,003,766 new rows (ModernBERT-base, 9,835,199 rows reused) | NVIDIA GH200 | 75 min |
| Split (e-means, 14,828,377 rows to 14,828 clusters) and the token store's rows | NVIDIA GH200 | 4 min, then the rows |
| The v1.1 split (e-means, five sets, cap 50) and the store's rows | NVIDIA GH200 | 5 min, then the rows |
| Text store (17.4 million texts, 18.1 GB) and the synonym table | NVIDIA GH200 machine, CPU | 3 min |
| Training, `s512_l6`, four epochs, 43,452 steps | NVIDIA GH200 | 19 h 30 min |
| Export, validation and one test read | NVIDIA GH200 | 12 min |
| Benchmarks, 16 sets of at most 3,000 rows | NVIDIA GH200 | 4 min |

The first build's corpus and its stored embeddings (on an NVIDIA H100 and a GH200, 2026-11-17) are reused, not recomputed, and so are 1.0.0's release ablations (on an NVIDIA A10). Litmus's own release needs no GPU: the engine reads on one CPU core.
<!--/litmus-->

For planning, [`teacher/README.md`](../../teacher/README.md) (Cost on an H100) projects each teacher step from FLOP
counts: embedding with ModernBERT-base at about 0.45 GPU hours per billion tokens on an H100, a split in minutes, and
calibration on a CPU in minutes. A training step's cost is measured by its `--pilot`.

## Every formula and the code that computes it

Each part ends with its own table; this is the index.

| Part | Its table |
| --- | --- |
| Corpus: standard reading, passages, keys, hold-out, balance | [1. The data](data.md#where-in-the-code) |
| Embedding, clustering, the greedy assignment, the invariants | [2. The splits](splits.md#where-in-the-code) |
| Ternary, int8, RMS norm, layer, attention, pooling, head, pair logit, bank | [3. The model](model.md#where-in-the-code) |
| Loss, sampling, packing, gradient caching, schedule, checkpoint | [4. Training](training.md#where-in-the-code) |
| Conformal, Venn-Abers, three-way, strip, chunk combination | [5. Calibration](calibration.md#where-in-the-code) |
| Metrics, intervals, cost | [7. Evaluation](evaluation.md#where-in-the-code) |

---

Previous: [8. The engine](engine.md) · Back to the [front page](../TRAINING.md)
