"""The one color system of Litmus's benchmark figures, defined once.

Two scales that never share a meaning. The litmus origin scale, which the app uses too, colors anything that shows an
origin or a score from human to AI: blue for human, purple for mixed, red for AI, with lightness falling from human to
AI and a marker and hatch per origin, so the three stay apart in grayscale and for color-blind readers. The detector
palette colors comparisons between detectors: Litmus is the one accent (a bluish green, outside the origin hues), its
predecessor a dark gray, and every other detector a neutral gray, with commercial and open told apart by marker shape.
A number quoted from a paper or leaderboard rather than rerun is drawn hollow.
"""

from __future__ import annotations

import numpy as np

from matplotlib.colors import LinearSegmentedColormap, to_rgb


ORIGIN_ORDER = ("human", "mixed", "ai")
ORIGIN_COLORS = {"human": "#5B9BD5", "mixed": "#8E5BB5", "ai": "#B2182B"}
ORIGIN_MARKERS = {"human": "o", "mixed": "D", "ai": "s"}
ORIGIN_HATCHES = {"human": "", "mixed": "//", "ai": "xx"}
ORIGIN_LABELS = {"human": "Human", "mixed": "Mixed", "ai": "AI"}
LITMUS_SCALE = LinearSegmentedColormap.from_list("litmus", [ORIGIN_COLORS[origin] for origin in ORIGIN_ORDER])

LITMUS = "Litmus"
PREDECESSOR = "ai-vs-human-checker"
DETECTOR_COLORS = {"litmus": "#009E73", "predecessor": "#3A3A3A", "commercial": "#7A7A7A", "open": "#A6A6A6"}
DETECTOR_MARKERS = {"litmus": "*", "predecessor": "P", "commercial": "^", "open": "o"}
METRIC_SCALE = "cividis"
LINE_WIDTH = 1.2
FONT_SIZE = 8

# The site's two looks (site/style.css): ink and rules from its tokens, Litmus's accent and the neutral detector grays
# chosen per ground so each keeps contrast, and the origin hues of the site's litmus scale. Figures are drawn on a
# transparent ground, so the page's own background shows through.
THEMES = {
    "light": {"ink": "#1E1B22", "soft": "#4F4A55", "rule": "#D6CDBE", "litmus": "#00845F", "predecessor": "#3F3A45",
              "commercial": "#6E675E", "open": "#8C857B", "estimated": "#B5AEA3",
              "human": "#2F6FDB", "mixed": "#8E5BC9", "ai": "#D9483B"},
    "dark": {"ink": "#EEE8DC", "soft": "#B9B2A8", "rule": "#3A3540", "litmus": "#3DD6A3", "predecessor": "#E2DBCF",
             "commercial": "#B0A99E", "open": "#948D83", "estimated": "#6C6672",
             "human": "#5B8FE6", "mixed": "#A27AD6", "ai": "#E2675A"},
}
# One marker per rerun detector, so a detector reads the same in every panel; quoted leaderboard entries use the
# kind's marker.
RERUN_MARKERS = {"Litmus": "*", "ai-vs-human-checker": "P", "Desklib v1.01": "s", "ModernBERT RAID+MAGE (Drayson)": "D",
                 "e5-small-lora": "^", "Fast-DetectGPT": "v", "Binoculars": "o", "OpenAI RoBERTa-base (GPT-2)": "h",
                 "RoBERTa ChatGPT (HC3)": "<", "RADAR": ">", "MAGE Longformer": "X"}
SHORT_NAMES = {"ModernBERT RAID+MAGE (Drayson)": "ModernBERT (Drayson)", "ai-vs-human-checker": "Predecessor",
               "OpenAI RoBERTa-base (GPT-2)": "RoBERTa-base (OpenAI)", "OpenAI RoBERTa-large (GPT-2)": "RoBERTa-large (OpenAI)",
               "TMR (Target Mining RoBERTa)": "TMR", "RoBERTa ChatGPT (HC3)": "RoBERTa (HC3)",
               "Tabularis ModernBERT R1.1": "Tabularis ModernBERT", "Litmus engine": "Litmus"}


def short_name(detector: str) -> str:
    return SHORT_NAMES.get(detector, detector)


def detector_role(detector: str, kind: str) -> str:
    """The palette role of a detector: Litmus, its predecessor, or a commercial or open competitor."""
    if detector.startswith(LITMUS):
        return "litmus"
    if detector.startswith(PREDECESSOR):
        return "predecessor"
    if kind not in ("commercial", "open"):
        raise ValueError(f"detector {detector!r} has kind {kind!r}, expected commercial or open")
    return kind


def lightness(color: str) -> float:
    """CIE L* (0 black to 100 white) of an sRGB color, to check that a scale stays ordered without hue."""
    linear = np.array([channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
                       for channel in to_rgb(color)])  # (3,)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    return 116.0 * luminance ** (1.0 / 3.0) - 16.0 if luminance > (6 / 29) ** 3 else luminance * (29 / 3) ** 3
