# Litmus student (`litmus_student`)

The model Litmus ships: a BitNet b1.58 encoder (ternary weights, 8-bit activations, block-local attention with blocks
shifted by half on alternate layers, rotary positions) distilled from the teacher's frozen encoder and pair head, its
release file, and the NumPy reference the Rust engine reproduces. Training runs in Docker on one GPU
([`../launch/`](../launch/README.md)); the tests run on a CPU.

| Path | Holds |
| --- | --- |
| [`SPEC.md`](SPEC.md) | The contract: text to token ids, quantization, the encoder, chunks, windows and the heat map, the reading and its uncertainty, streaming, the release file, sizes, and the parity tolerances |
| [`reference.py`](reference.py) | The same in NumPy; where it and `SPEC.md` disagree, it is the answer. Writes random releases and the engine's fixtures |
| `configs.json` | The sizes: `s192_l4` to `s384_l8` (width 192, 256, or 384; 4, 6, or 8 layers) and `tiny` for tests |
| `src/litmus_student/data.py` | The pre-tokenized token store (`tokens.npy`, `offsets.npy`, `rows.npz`, `meta.json`), sampling, packing, teacher targets |
| `src/litmus_student/model.py`, `distill.py`, `dense.py` | The encoder in PyTorch; the losses against the teacher's embedding, query and key, and pair matrix; the per-token loss (measured, no gain) |
| `src/litmus_student/train.py`, `search.py` | One size per process in a time box; the Bayesian search over the distillation's settings |
| `src/litmus_student/export.py`, `evaluate.py` | A checkpoint to a release with its bank and calibration, checked against `reference.py`; long, spliced, and passage readings through the release |
| `src/litmus_student/poc.py` | The proof of concept on the predecessor's corpus and head |

## Commands

From this folder, with `src` and this folder on `PYTHONPATH` (the launch image sets both):

| Step | Command |
| --- | --- |
| Token store, tokens | `python -m litmus_student.data --corpus CORPUS --out STORE --tokens-only` |
| Token store, rows | `python -m litmus_student.data --corpus CORPUS --out STORE --split SPLIT.parquet` |
| Distillation search | `python -m litmus_student search --data STORE --teacher-export TEACHER_EXPORT --out DIR [--max-minutes 45]` |
| Train one size | `python -m litmus_student.train --config s192_l4 --data STORE --teacher-export TEACHER_EXPORT --experiment-id YYYY-MM-DD_slug --out RUN [--minutes M] [--settings SEARCH/summary.json] [--int8-output]` |
| Export a release | `python -m litmus_student.export --checkpoint RUN/best.pt --data STORE --corpus CORPUS --out RELEASE [--attention integer] [--output int8] [--bank-recipe mixed_out --bank-per-origin N]` |
| Evaluate | `python -m litmus_student.evaluate --checkpoint RUN/best.pt --release RELEASE --data STORE --experiment-id YYYY-MM-DD_slug --out DIR --split validation` |
| Proof of concept | `python -m litmus_student poc --corpus PREDECESSOR_CORPUS --embeddings EMBEDDINGS.npy --checker CHECKER_DIR --out DIR --experiment-id YYYY-MM-DD_slug` |
| Random release | `python reference.py init --config s192_l4 --out DIR` |
| Engine fixtures | `python reference.py fixtures --out DIR` |
| Read one text | `python reference.py score --release DIR --text FILE` |

`TEACHER_EXPORT` is the teacher's `export` folder (`manifest.json`). `train` and `evaluate` log to W&B unless given
`--pilot`. `evaluate --split test` is the release's one reading of test. `export --attention integer` releases a model in
`SPEC.md`'s optional integer-attention mode (format 2); without it the release keeps the checkpoint's own attention.
`train --int8-output` quantizes the head's query and key to int8 in the training forward, so the loss sees the unit
vectors of the int8 codes a release stores; `export` releases such a checkpoint as `SPEC.md`'s int8 output (format 3):
int8 query and key codes, an int8 bank (`2w + 17` bytes a text instead of `4w + 17`) and a compare by integer dot
products. `export --output int8` on a checkpoint trained without the switch is refused unless
`--quantize-after-training` is also given. `--bank-recipe mixed_out` draws the bank's human and ai texts uniformly from
the training split and its mixed texts from the corpus's `bank_draw` split, `--bank-per-origin` of each, and fails
when the corpus has no such split; the default draws the bank from `--bank-split`, spread over clusters.
`reference.py init` needs only NumPy (`--output int8` writes a format 3 release); reading text also needs `tokenizers`.

## Measured

- **Proof of concept.** On validation of the predecessor's two-origin corpus after one hour, the ternary
  `s192_l4` read PR AUC 0.9673 against 0.9676 for its full-precision twin on the same batches, and `s256_l6` 0.9721;
  the predecessor's teacher read 0.9901 through the same bank.
- **Integer attention.** Trained in, it read validation PR AUC 0.9386 against 0.9371 without it after 306 steps on
  the same batches, at 15% slower training.
- **The released student** is `s192_l4`, trained on the true labels alone (no teacher) on the corpus in
  [`../corpus/README.md`](../corpus/README.md), with ternary weights, integer attention and the int8 output trained
  in from the first step; its checkpoint was chosen on validation. Its validation and test readings are in the model
  card ([`../release/model_card.md`](../release/model_card.md)) and in
  [`../docs/training/evaluation.md`](../docs/training/evaluation.md).

## Tests

`python -m pytest` from this folder: the token store, the losses, the per-token loss, the export round trip, integer
attention, int8 output (the quantizer, the integer compare against a by-hand computation, the format and what it
refuses, training with the switch and exporting its checkpoint, the bank recipe), the search's plumbing, the proof of
concept on a tiny corpus, and the reference's chunks, windows, streaming, calibration, and release files. `tests/test_parity.py` holds the PyTorch model to `reference.py` within
the tolerances of `SPEC.md`, Parity.
