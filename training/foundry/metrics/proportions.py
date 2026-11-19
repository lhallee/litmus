"""Proportions: the Wilson score interval of a rate and the pooled two-proportion z-test.

Promoted from synteract4.hsp90 (the first version's HSP90-beta partner recovery), whose reference-set rates the atlas
v3 benchmark suite reads the same way.
"""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of successes / n at the normal quantile `z` (95 percent by default), within [0, 1]."""
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / (1 + z**2 / n)
    return max(0.0, center - spread), min(1.0, center + spread)


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """The pooled two-proportion z statistic of x1 / n1 against x2 / n2 and its two-sided p-value; NaN for both when
    the pooled rate is 0 or 1."""
    pooled = (x1 + x2) / (n1 + n2)
    if pooled in (0.0, 1.0):
        return float("nan"), float("nan")
    z = (x1 / n1 - x2 / n2) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return z, math.erfc(abs(z) / math.sqrt(2))
