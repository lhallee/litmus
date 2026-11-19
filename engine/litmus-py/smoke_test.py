"""Smoke test of an installed `litmus_engine` wheel: load a release, score two texts, print the readings as JSON.

    python smoke_test.py <release folder>

Exits non-zero on any failure: the module or the release does not load, a reading is not a finite JSON object, or
its token count disagrees with `count_tokens`. Run it in an environment where only the wheel provides the module,
from a folder that does not hold the crate's source.
"""

from __future__ import annotations

import argparse
import json
import sys
import litmus_engine

from pathlib import Path


TEXTS = (
    "Thanks for the notes. I moved the meeting to Thursday at 3, and Sam will bring the printed drafts.",
    "The valley's orchards have moved uphill over two decades as rainfall fell by about 12%. Growers who stayed "
    "near the river switched to drought-tolerant rootstock, while others leased terraces above the old road. "
    "The cooperative now runs two pressing sheds instead of one, and its members argue every spring about "
    "whose fruit goes first.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", type=Path, help="a release folder: model.safetensors, config.json, tokenizer.json")
    arguments = parser.parse_args()

    print(f"litmus_engine {litmus_engine.__version__} from {litmus_engine.__file__}", file=sys.stderr)
    model = litmus_engine.Model.from_path(str(arguments.release))
    readings = []
    for text in TEXTS:
        tokens = model.count_tokens(text)
        reading = model.score(text)
        assert isinstance(reading, dict), f"the reading is a {type(reading).__name__}"
        assert reading["n_tokens"] == tokens, f"score read {reading['n_tokens']} tokens, count_tokens {tokens}"
        json.dumps(reading, allow_nan=False)  # raises on a value JSON cannot hold
        readings.append({"tokens": tokens, "label": reading["label"], "reading": reading})
    print(json.dumps({"version": model.version, "readings": readings}, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
