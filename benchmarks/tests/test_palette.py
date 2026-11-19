from __future__ import annotations

import pandas as pd
import pytest

from pathlib import Path
from litmus_benchmarks.palette import (DETECTOR_COLORS, ORIGIN_COLORS, ORIGIN_HATCHES, ORIGIN_MARKERS, ORIGIN_ORDER,
                                       detector_role, lightness)
from matplotlib.colors import rgb_to_hsv, to_rgb


BENCHMARKS = Path(__file__).resolve().parents[1]


def test_origin_scale_falls_in_lightness_and_differs_in_shape() -> None:
    values = [lightness(ORIGIN_COLORS[origin]) for origin in ORIGIN_ORDER]
    assert all(earlier - later >= 8 for earlier, later in zip(values, values[1:], strict=False)), values
    assert len(set(ORIGIN_MARKERS.values())) == len(set(ORIGIN_HATCHES.values())) == 3


def test_detector_palette_is_neutral_but_for_the_litmus_accent() -> None:
    for role, color in DETECTOR_COLORS.items():
        saturation = rgb_to_hsv(to_rgb(color))[1]
        assert (saturation > 0.5) if role == "litmus" else (saturation < 0.05), role
    accent_hue = rgb_to_hsv(to_rgb(DETECTOR_COLORS["litmus"]))[0]
    for origin, color in ORIGIN_COLORS.items():
        assert abs(accent_hue - rgb_to_hsv(to_rgb(color))[0]) > 0.08, origin


def test_roles() -> None:
    assert detector_role("Litmus 4L", "open") == "litmus"
    assert detector_role("ai-vs-human-checker", "open") == "predecessor"
    assert detector_role("GPTZero", "commercial") == "commercial"
    with pytest.raises(ValueError):
        detector_role("Someone", "research")


def test_published_rows_carry_their_sources() -> None:
    published = pd.read_csv(BENCHMARKS / "published.csv", dtype=str, keep_default_na=False)
    required = ["detector", "kind", "benchmark", "metric", "value", "protocol", "source_url", "source_locator", "date", "reported_by"]
    assert (published[required] != "").all().all()
    assert set(published["kind"]) <= {"commercial", "open"}
    assert published["source_url"].str.startswith("https://").all()
    assert pd.to_numeric(published["value"]).between(0, 1).all()
