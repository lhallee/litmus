# Litmus launch

The Docker image and the scripts that run Litmus's training on one GPU machine: the corpus, the teacher, the
student, the benchmark scoring, and saving the run to private Hugging Face repositories. Every step runs in the
container, and every chain skips a step whose output already exists, so a stopped chain is run again as it is.

| File | Does |
| --- | --- |
| `Dockerfile` | The `litmus` image: CUDA 13.0 and torch 2.14.0 by default (build arguments `CUDA_IMAGE` and `TORCH_INDEX` select CUDA 12.6 for a 570 driver), Python 3.12, and `requirements.txt` |
| `requirements.txt` | The union of what the corpus, teacher, and student import, pinned |
| `in_container.sh` | Runs one command in the image with the run folder at `/work`, the Hugging Face cache at `/cache`, the source mounted read-only at `/workspace`, and each package on `PYTHONPATH`; `NAME`, `CPUS`, `MEMORY`, `THREADS`, and `GPU=0` set the container |
| `teacher_chain.sh` | After embedding: split, head search with its runs of record, bank, export, then the baselines, the control's bank, and calibration |
| `student_chain.sh` | One student size after the teacher's export: train in a time box, export the release from the checkpoint chosen on validation, evaluate on validation and once on test, score the benchmarks |
| `benchmark_chain.sh` | Scores benchmarks with named detectors, one file per benchmark and detector, each in its own process |
| `save_chain.sh`, `save_run.py` | Upload the embeddings and the run's outputs to private repositories, resuming a stopped upload; a public repository is refused |

`in_container.sh` runs from a copy of the source shipped to the GPU machine, with this project four levels below the
run directory (`tree/projects/tooling/<name>`); in a clone of this repository, build the image with
`docker build -t litmus -f launch/Dockerfile --build-arg PROJECT=. .` and run the steps below in it with the
packages' `src` folders and `student/` on `PYTHONPATH`.

## Run order

Outputs live under `/work`. `CORPUS` is `/work/litmus_corpus/corpus/data`.

1. **Corpus**: `python -m litmus_corpus build --out /work/litmus_corpus --workers N --priority 1 --new-token-budget 2e9`
   (with `--skip-benchmark` for each benchmark the reader cannot open).
2. **Student tokens**, beside the embedding: `python -m litmus_student.data --corpus CORPUS --out /work/student_store --tokens-only`.
3. **Embed**, after a trial run with `--pilot`: `python -m litmus_teacher embed --corpus CORPUS --out /work/embed/modernbert_base --reuse F --attention sdpa`,
   where `F` is the predecessor's stored ModernBERT-base embeddings; rows whose text is unchanged copy them, as stored
   or aligned, when the trial run's reuse check allows it ([`../teacher/README.md`](../teacher/README.md)).
4. **Teacher**: `bash launch/teacher_chain.sh`.
5. **Student rows**: `python -m litmus_student.data --corpus CORPUS --out /work/student_store --split /work/teacher/split/split.parquet`.
6. **Distillation search** (optional): `python -m litmus_student search --data /work/student_store --teacher-export /work/teacher/export --out /work/student/search`.
7. **Students**, one chain per size: `SIZE=s192_l4 MINUTES=150 bash launch/student_chain.sh`.
8. **Baselines**: `OUT=/work/benchmarks DETECTORS=baselines bash launch/benchmark_chain.sh`.
9. **Save**: `bash launch/save_chain.sh`.

## Settings

| Script | Variables (default) |
| --- | --- |
| `teacher_chain.sh` | `ROOT` (`/work/teacher`), `CORPUS`, `EMBED` (`/work/embed`), `EXPERIMENT` (the W&B group), `SEARCH_MINUTES` (12), `TRIAL_STEPS` (4000), `TOP` (2), `SEEDS` (`0 1 2`) |
| `student_chain.sh` | `SIZE` and `MINUTES` (required), `ROOT` (`/work/student`), `DATA` (`/work/student_store`), `CORPUS`, `TEACHER` (`/work/teacher/export`), `EXPERIMENT`, `STEPS` (60000, a cap the time box closes first), `SETTINGS` (the search's `summary.json`), `TEXTS` (1024 per step), `MICRO` (262144 tokens), `EVAL_EVERY` (500), `FRACTION` (of the GPU's memory), `LONG` (1500 long documents), `SPLICED` (100 per cell) |
| `benchmark_chain.sh` | `OUT` and `DETECTORS` (required: comma-separated names, `baselines` for the predecessor and the five GPU detectors, or `gpu_baselines` for the five alone), `BENCHMARK_ROWS` (2000), `BENCHMARKS` (the 12 with labels the corpus holds out), `LITMUS_CHECKPOINT` and `LITMUS_RELEASE` when Litmus is among them |
| `save_chain.sh` | `WHAT` (`embeddings`, `run`, or `all`), `EMBED_REPO`, `RUN_REPO` |

For the corpus in `../corpus/README.md`, the teacher chain's search ran with `SEARCH_MINUTES=30 TRIAL_STEPS=8000
TOP=3`. The predecessor
(`ai-vs-human-checker`) needs its own package on `PYTHONPATH` and `beautifulsoup4` and `requests`, which the image
lacks; `CHECKER_THREADS` sets its threads.
