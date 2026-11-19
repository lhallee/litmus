"""Bank scoring on the GPU path matches the NumPy definition, and the bank choice favors the small."""

from __future__ import annotations

import numpy as np
import torch

from litmus_teacher.bank import bank_readings, choose_bank
from litmus_teacher.calibrate import readings_from_pair_logits, vector_cosines
from litmus_teacher.head import pair_logits


def unit_halves(values: torch.Tensor) -> torch.Tensor:
    # values: (n, 2w)
    query, key = values.chunk(2, dim=-1)
    return torch.cat([torch.nn.functional.normalize(query, dim=-1), torch.nn.functional.normalize(key, dim=-1)], dim=-1)  # (n, 2w)


def test_bank_readings_match_numpy_definition() -> None:
    generator = torch.Generator().manual_seed(0)
    texts = unit_halves(torch.randn(9, 8, generator=generator))  # (n, 2w)
    banks = [unit_halves(torch.randn(size, 8, generator=generator)) for size in (5, 7, 6)]  # three (k_o, 2w)
    tokens = np.arange(9) * 100
    readings = bank_readings(texts, banks, scale=7.0, bias=-0.5, tokens=tokens, top_k=4)
    logits = [pair_logits(texts, bank, 7.0, -0.5).double().numpy() for bank in banks]
    expected = readings_from_pair_logits(logits, vector_cosines(texts.double().numpy(), torch.cat(banks).double().numpy()),
                                         tokens, top_k=4)
    np.testing.assert_allclose(readings.mean, expected.mean, atol=1e-6)
    np.testing.assert_allclose(readings.spread, expected.spread, atol=1e-6)
    np.testing.assert_allclose(readings.closeness, expected.closeness, atol=1e-6)


def test_choose_bank_takes_smallest_within_tolerance() -> None:
    trials = [{"variant": "a", "size": 1024, "pr_auc": 0.950}, {"variant": "a", "size": 4096, "pr_auc": 0.961},
              {"variant": "b", "size": 2048, "pr_auc": 0.9605}, {"variant": "b", "size": 16384, "pr_auc": 0.962}]
    assert choose_bank(trials, 0.002) == {"variant": "b", "size": 2048, "pr_auc": 0.9605}
    assert choose_bank(trials, 0.0)["size"] == 16384
