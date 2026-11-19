"""End to end on CPU: a synthetic corpus and teacher, a short pilot run, export, the release read by reference.py, and
the evaluation of long documents, spliced documents and the window context gap."""

from __future__ import annotations

import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pathlib import Path
from litmus_student.contract import reference
from litmus_student.data import write_store
from litmus_student.evaluate import evaluate
from litmus_student.evaluate import parser as evaluation_parser
from litmus_student.export import export_release
from litmus_student.train import parser, train


SPLITS = ["bank", "calibration", "train", "validation"]


def synthetic_corpus(directory: Path) -> tuple[Path, Path, Path]:
    """A corpus whose origins use different tokens, its metadata Parquet, and a teacher export whose vectors cluster by origin."""
    generator = np.random.default_rng(0)
    counts = {"train": 300, "validation": 150, "bank": 150, "calibration": 660, "long": 30}
    split = np.concatenate([np.full(count, SPLITS.index("validation" if name == "long" else name)) for name, count in counts.items()])  # (n,)
    n = len(split)
    long_eval = np.arange(n) >= n - counts["long"]  # (n,) whole documents of two to four tiny chunks
    label = np.arange(n) % 3  # (n,)
    lengths = np.where(generator.random(n) < 0.2, generator.integers(9, 60, n), generator.integers(1, 9, n))  # (n,)
    lengths[(split == SPLITS.index("validation")) & (np.arange(n) % 10 == 0)] = 80  # long enough for several windows
    lengths[long_eval] = generator.integers(97, 380, counts["long"])
    texts = [generator.integers(1000 * origin, 1000 * origin + 1000, length) for origin, length in zip(label, lengths, strict=True)]
    write_store(directory / "store", texts, label, split, generator.integers(0, 10, n), SPLITS, long_eval)
    metadata = pa.table({"id": np.arange(n), "source": [f"source-{index % 4}" for index in range(n)],
                         "generator": ["" if origin == 0 else f"model-{index % 2}" for index, origin in enumerate(label)],
                         "domain": ["news"] * n, "era": [["human", "recent", "recent"][origin] for origin in label],
                         "format": ["plain"] * n, "language": ["en"] * n})
    pq.write_table(metadata, directory / "corpus.parquet")

    export = directory / "teacher"
    export.mkdir()
    centers = generator.standard_normal((3, 12))  # (3, 2D)
    np.save(export / "embeddings.npy", (centers[label] + 0.3 * generator.standard_normal((n, 12))).astype(np.float16))
    query_key = generator.standard_normal((3, 2, 6))[label] + 0.3 * generator.standard_normal((n, 2, 6))  # (n, 2, w_T)
    query_key /= np.linalg.norm(query_key, axis=-1, keepdims=True)
    np.save(export / "query_key.npy", query_key.reshape(n, 12).astype(np.float16))
    present = np.ones(n, dtype=bool)  # (n,)
    present[:5] = False
    np.save(export / "present.npy", present)
    manifest = {"head": {"scale": 10.0, "bias": -1.0}, "embeddings": [{"path": "embeddings.npy"}],
                "query_key": {"path": "query_key.npy", "present": "present.npy"}}
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return directory / "store", directory / "corpus.parquet", export


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_time_boxed_run_ends_on_the_clock_with_an_evaluated_checkpoint(tmp_path: Path) -> None:
    store, _, teacher = synthetic_corpus(tmp_path)
    arguments = parser().parse_args([
        "--config", "tiny", "--data", str(store), "--teacher-export", str(teacher), "--experiment-id", "2026-11-17_student_test",
        "--out", str(tmp_path / "run"), "--device", "cpu", "--pilot", "--pilot-steps", "100000", "--steps", "100000",
        "--minutes", "0.05", "--warmup", "3", "--texts-per-step", "60", "--micro-tokens", "256", "--pack-multiple", "256",
        "--eval-every", "100000", "--val-texts", "60", "--eval-per-origin", "30", "--bank-per-origin", "30", "--max-train-tokens", "96",
    ])
    summary = train(arguments)

    assert 0 < summary["val/step"] < 100000 - 1  # the box closed long before the steps ran out
    assert np.isfinite(summary["val/best_loss"]) and (tmp_path / "run" / "best.pt").exists()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_warm_start_begins_from_the_checkpoints_weights(tmp_path: Path) -> None:
    store, _, teacher = synthetic_corpus(tmp_path)
    base = ["--config", "tiny", "--data", str(store), "--teacher-export", str(teacher), "--experiment-id", "2026-11-17_student_test",
            "--device", "cpu", "--pilot", "--warmup", "3", "--texts-per-step", "60", "--micro-tokens", "256", "--pack-multiple", "256",
            "--val-texts", "60", "--eval-per-origin", "30", "--bank-per-origin", "30", "--max-train-tokens", "96", "--labels-only"]
    first = train(parser().parse_args([*base, "--out", str(tmp_path / "first"), "--pilot-steps", "60", "--steps", "60", "--lr", "4e-3",
                                       "--lr-second", "3e-3", "--eval-every", "60"]))
    # One step at a learning rate of zero: the warm run's only reading is of the checkpoint's own weights.
    still = ["--pilot-steps", "1", "--steps", "1", "--lr", "0", "--lr-second", "0", "--eval-every", "1"]
    warm = train(parser().parse_args([*base, *still, "--out", str(tmp_path / "warm"), "--init", str(tmp_path / "first" / "best.pt")]))
    cold = train(parser().parse_args([*base, *still, "--out", str(tmp_path / "cold")]))

    # The one step still moves the head's running statistics, so the warm reading is near the first run's, not equal.
    warm_gap = abs(warm["val/pair_label"] - first["val/pair_label"])
    cold_gap = abs(cold["val/pair_label"] - first["val/pair_label"])
    assert warm_gap < 1e-3 and cold_gap > 10 * warm_gap


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_pilot_run_exports_a_release_the_reference_reads(tmp_path: Path) -> None:
    store, corpus, teacher = synthetic_corpus(tmp_path)
    arguments = parser().parse_args([
        "--config", "tiny", "--data", str(store), "--teacher-export", str(teacher), "--experiment-id", "2026-11-17_student_test",
        "--out", str(tmp_path / "run"), "--device", "cpu", "--pilot", "--pilot-steps", "150", "--steps", "150", "--lr", "4e-3",
        "--lr-second", "3e-3", "--warmup", "3", "--texts-per-step", "60", "--micro-tokens", "256", "--pack-multiple", "256",
        "--eval-every", "75", "--val-texts", "60", "--eval-per-origin", "30", "--bank-per-origin", "30", "--max-train-tokens", "96",
    ])
    summary = train(arguments)
    assert np.isfinite(summary["val/best_loss"]) and 0.0 <= summary["val/roc_auc"] <= 1.0
    assert summary["release/margin"] == 0.01 and summary["release/within_margin"] in (0.0, 1.0)
    assert summary["release/gap"] == pytest.approx(summary["release/teacher_pr_auc_human_ai"] - summary["release/pr_auc_human_ai"])
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "pilot.json").exists()

    release = tmp_path / "release"
    exported = export_release(tmp_path / "run" / "best.pt", store, corpus, release, bank_per_origin=40, check_texts=8,
                              micro_tokens=256, device=torch.device("cpu"))
    assert exported["bank"] == [40, 40, 40] and exported["calibration_rows"] == 660

    loaded = reference.ReferenceModel.load(release)
    ids = np.random.default_rng(1).integers(1000, 2000, 250)  # (n,) three chunks of the tiny config
    whole, _ = loaded.read_whole(ids)
    streamed, _ = loaded.read_streaming(ids, 5)
    assert whole["verdict"] == streamed["verdict"] and len(whole["chunks"]) == 3
    assert [(cell["start"], cell["end"]) for cell in whole["cells"]][-1][1] == 250
    strip = whole["verdict"]["strip"]
    assert 0.0 <= strip["band_lo"] <= strip["position"] <= strip["band_hi"] <= 1.0
    table = loaded.calibration.table
    for chunk in whole["chunks"]:
        for entry in chunk["explain"]["conformal"]:
            group = next(group for group in table["conformal"][entry["origin"]] if chunk["length_bin"] in group["bins"])
            assert entry["n"] == group["n"] and 0 <= entry["count"] <= entry["n"]
    nearest = whole["chunks"][0]["explain"]["nearest"][0]
    assert nearest["source"].startswith("source-") and nearest["era"] in ("human", "recent")

    out = tmp_path / "evaluation"
    report = evaluate(evaluation_parser().parse_args([
        "--checkpoint", str(tmp_path / "run" / "best.pt"), "--release", str(release), "--data", str(store),
        "--experiment-id", "2026-11-17_student_test", "--out", str(out), "--device", "cpu", "--pilot",
        "--spliced-per-cell", "3", "--context-texts", "4", "--passages-per-origin", "12",
    ]))
    assert len(report["passages"]) == 36 and {document["chunks"] for document in report["passages"]} == {1}
    assert len(report["long_documents"]) == 30 and {document["chunks"] for document in report["long_documents"]} <= {2, 3, 4}
    assert report["spliced"] and report["window_context"]
    metrics = {(row["evaluation"], row["metric"]) for row in report["rows"]}
    assert {("passages", "pr_auc"), ("passages", "mcc"), ("passages", "coverage"), ("long_documents", "coverage"), ("long_documents", "pr_auc"), ("spliced_sliding", "cell_accuracy"),
            ("spliced_sections", "boundary_error_median"), ("window_context", "pooled_gap_mean")} <= metrics
    assert (out / "evaluation.csv").exists() and (out / "evaluation.json").exists()
