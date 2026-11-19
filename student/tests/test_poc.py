"""The proof of concept end to end on CPU, on stand-ins: a two-origin corpus, stored embeddings, a random "released"
pair head with its own code and bank; and the verdict's rules."""

from __future__ import annotations

import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pathlib import Path
from litmus_student.poc import parser, poc, rising, verdict
from safetensors.torch import save_file


HEAD_CODE = '''"""A stand-in for the released `fphunt.pairhead`: the same interface, random weights."""

import json
import torch
import torch.nn.functional as F

from pathlib import Path
from torch import nn


class PairHead(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(input_width, output_width)
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))
        self.config = {"input_width": input_width, "output_width": output_width}

    @property
    def scale(self):
        return self.log_scale.exp()

    def forward(self, pooled):
        query, key = self.linear(pooled.float()).chunk(2, dim=-1)
        return torch.cat([F.normalize(query, dim=-1), F.normalize(key, dim=-1)], dim=-1)

    @classmethod
    def load(cls, directory: Path):
        config = json.loads((directory / "pairhead.json").read_text(encoding="utf-8"))
        return cls(config["input_width"], config["output_width"])
'''


def stand_ins(directory: Path) -> tuple[Path, Path, Path]:
    """A corpus of two origins whose words differ, embeddings that cluster by origin, and a checker folder."""
    generator = np.random.default_rng(0)
    words = [["river", "quiet", "ledger", "careful", "window"], ["delve", "tapestry", "nuanced", "robust", "leverage"]]
    corpus = directory / "corpus"
    (corpus / "data").mkdir(parents=True)
    ids = generator.permutation(420)  # (n,)
    label = ids % 2  # (n,)
    lengths = generator.integers(8, 40, len(ids))  # (n,) words
    lengths[0] = 200  # longer than a tiny chunk: dropped and counted
    texts = [" ".join(generator.choice(words[origin], length)) + "." for origin, length in zip(label, lengths, strict=True)]
    for name, rows in (("train", slice(0, 360)), ("validation", slice(360, 420))):
        table = pa.table({"id": ids[rows], "text": texts[rows], "label": pa.array(label[rows], pa.int8()),
                          "cluster": pa.array(generator.integers(0, 6, len(ids[rows])), pa.int16()), "words": pa.array(lengths[rows], pa.int16())})
        pq.write_table(table, corpus / "data" / f"{name}-00000-of-00001.parquet", row_group_size=50)
    order = np.argsort(ids)  # (n,) rows in id order
    centers = generator.standard_normal((2, 12))  # (2, 2D)
    embeddings = (centers[label[order]] + 0.5 * generator.standard_normal((len(ids), 12))).astype(np.float16)  # (n, 2D) in id order
    np.save(directory / "embeddings.npy", embeddings)
    checker = directory / "checker"
    (checker / "fphunt").mkdir(parents=True)
    (checker / "fphunt" / "__init__.py").write_text("", encoding="utf-8")
    (checker / "fphunt" / "pairhead.py").write_text(HEAD_CODE, encoding="utf-8")
    (checker / "pairhead.json").write_text(json.dumps({"input_width": 12, "output_width": 16}), encoding="utf-8")
    bank = torch.nn.functional.normalize(torch.randn(2, 30, 2, 8), dim=-1).reshape(2, 30, 16)  # (origin, k, 2w_T)
    save_file({"human": bank[0].contiguous(), "ai": bank[1].contiguous()}, str(checker / "bank.safetensors"))

    return corpus, directory / "embeddings.npy", checker


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_proof_of_concept_runs_end_to_end(tmp_path: Path) -> None:
    corpus, embeddings, checker = stand_ins(tmp_path)
    report = poc(parser().parse_args([
        "--corpus", str(corpus), "--embeddings", str(embeddings), "--checker", str(checker), "--out", str(tmp_path / "poc"),
        "--experiment-id", "2026-11-17_student_poc_test", "--rows", "300", "--workers", "2", "--small", "tiny", "--middle", "tiny",
        "--texts-per-step", "24", "--micro-tokens", "256", "--pack-multiple", "256", "--warmup", "2", "--bank-per-origin", "20",
        "--steps", "6", "--eval-steps", "3", "--device", "cpu", "--pilot",
    ]))
    tokenized = report["tokenization"]
    assert tokenized["passages"] == 300 + 60 - 1 and tokenized["dropped_over_chunk"] == 1 and tokenized["tokens_per_second"] > 0
    assert report["steps"] == 6 and set(report["arms"]) == {"ternary_small", "ternary_middle", "twin_small"}
    assert not report["arms"]["twin_small"]["quantized"] and report["arms"]["ternary_small"]["quantized"]
    for arm in report["arms"].values():
        assert [point["step"] for point in arm["curve"]] == [2, 5] and 0.0 <= arm["curve"][-1]["pr_auc"] <= 1.0
        assert arm["tokens_per_second"] > 0
    assert report["teacher"]["validation_passages"] == 60 and report["teacher"]["released_bank"] is not None
    assert report["verdict"]["call"] in ("go", "ternary is the problem", "the student is the problem", "inconclusive")
    assert (tmp_path / "poc" / "poc_report.json").exists()


def test_verdict_rules() -> None:
    def call(small: float, middle: float, twin: float, teacher: float) -> str:
        return verdict({"ternary_small": small, "ternary_middle": middle, "twin_small": twin}, teacher)["call"]

    assert call(0.975, 0.978, 0.982, 0.990) == "go"  # best ternary 0.012 short of the teacher, 0.004 of the twin
    assert call(0.960, 0.962, 0.985, 0.990) == "ternary is the problem"
    assert call(0.940, 0.950, 0.950, 0.990) == "the student is the problem"
    assert call(0.968, 0.966, 0.975, 0.990) == "inconclusive"  # ternary within 0.01 of the twin, 0.022 short of the teacher


def test_rising_curve() -> None:
    flat = [{"minutes": minutes, "pr_auc": 0.95} for minutes in (5, 20, 40, 60)]
    climbing = [{"minutes": minutes, "pr_auc": 0.90 + 0.001 * minutes} for minutes in (5, 20, 40, 60)]
    assert not rising(flat) and rising(climbing)
