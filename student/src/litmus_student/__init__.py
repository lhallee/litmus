"""Litmus's student: a BitNet b1.58 encoder distilled from the frozen teacher, exported to the release of SPEC.md.

Modules: `model` (the encoder and head, quantization-aware), `data` (pre-tokenized corpus, sampling, packing),
`distill` (the losses), `train` (one size per process), `export` (checkpoint to release, verified against
`reference.py`), and `contract` (loads `reference.py`, the NumPy answer).
"""
