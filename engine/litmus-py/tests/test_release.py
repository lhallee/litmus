"""The binding on a trained release: the surface the contribute server codes against, the command line's reading,
the server's limits, errors, and threads.

The reading must equal what `litmus score --json` prints for the same text and release, field for field and bit for
bit: both serialize the engine's `DocumentReading` with serde_json, and the binding adds only `label` and `chance`
(the chance of AI), which the command line prints beside the reading.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import litmus_engine
import pytest

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


CONTRIBUTE_SOURCE = Path(__file__).resolve().parents[3] / "contribute" / "src"
SERVER_TOKEN_CAP = 131_072  # contribute/src/litmus_contribute/config.py, text_max_tokens
SERVER_READING_BYTES = 2 * 1_048_576  # the same file, reading_max_bytes
SENTENCES = (
    "The committee met on a wet Tuesday and spent most of the morning arguing about the budget for the bridge.",
    "My grandmother kept her recipes in a tin box, each card stained with butter and written in pencil.",
    "Nobody expected the ferry to run that late, so the last bus waited at the pier with its engine idling.",
    "In practice, the method works well when the samples are independent and the noise is roughly Gaussian.",
    "We walked along the canal until the light went orange, then turned back past the shuttered boatyard.",
    "The report recommends three changes: clearer signage, longer opening hours, and a second ticket desk.",
    "He fixed the radio with a paper clip, which says more about the radio than about his skill.",
    "Rainfall in the valley has dropped by 12% over two decades, and the orchards have moved uphill.",
    "She read the letter twice, folded it into quarters, and put it in the drawer with the unpaid bills.",
    "Overall, these results suggest that the effect is real but smaller than the first study claimed.",
)


def prose(model: litmus_engine.Model, tokens: int, seed: int) -> str:
    """Varied English prose of exactly `tokens` tokens, the same for the same seed."""
    order = [(seed * 7 + index * 3) % len(SENTENCES) for index in range(len(SENTENCES))]
    paragraphs: list[str] = []
    count = 0
    while count < tokens:
        paragraph = " ".join(SENTENCES[order[(len(paragraphs) + offset) % len(order)]] for offset in range(4))
        paragraphs.append(f"{len(paragraphs) + 1}. {paragraph}")
        count += model.count_tokens(paragraphs[-1]) + 1
    encoded = "\n\n".join(paragraphs).encode("utf-8")
    ends = [end for _, _, end in model.tokenize(encoded.decode("utf-8"))]
    # A cut inside a word can tokenize differently, so step back to the longest cut with exactly `tokens` tokens.
    for end in reversed(ends[max(0, tokens - 32) : tokens]):
        text = encoded[:end].decode("utf-8")
        if model.count_tokens(text) == tokens:
            return text
    raise AssertionError(f"no cut of the prose has exactly {tokens} tokens")


def cli_rows(cli: Path, release: Path, texts: list[str], folder: Path) -> list[dict]:
    """`litmus score --json --no-cache` on each text, written byte for byte to a `.txt` file."""
    paths = []
    for index, text in enumerate(texts):
        path = folder / f"text_{index}.txt"
        path.write_bytes(text.encode("utf-8"))
        paths.append(path)
    completed = subprocess.run(
        [str(cli), "score", "--json", "--no-cache", "--model", str(release), *map(str, paths)],
        capture_output=True,
        check=False,
    )
    assert completed.returncode in (0, 1), completed.stderr.decode("utf-8", errors="replace")
    rows = [json.loads(line) for line in completed.stdout.decode("utf-8").splitlines()]
    assert [row["file"] for row in rows] == [str(path) for path in paths]
    return rows


def finite_numbers(value: object) -> int:
    """The count of numbers in a JSON value, asserting each is finite and that nothing is null."""
    assert value is not None
    if isinstance(value, (bool, str)):
        return 0
    if isinstance(value, (int, float)):
        assert math.isfinite(value)
        return 1
    if isinstance(value, dict):
        return sum(finite_numbers(item) for item in value.values())
    assert isinstance(value, list)
    return sum(finite_numbers(item) for item in value)


@pytest.fixture(scope="module")
def texts(model: litmus_engine.Model) -> dict[str, str]:
    return {
        "empty": "",
        "whitespace": " \n\t ",
        "one word": "Hello",
        "very short": "Thanks, see you at 5.",
        "unicode": "Café crème, naïve résumé, 日本語の文, and an emoji-free line.\r\nSecond line.",
        "one chunk": prose(model, 3_000, seed=1),
        "exactly one chunk": prose(model, 8_192, seed=2),
        "two chunks": prose(model, 8_193, seed=3),
        "three chunks": prose(model, 20_000, seed=4),
    }


def test_contribute_surface(model: litmus_engine.Model, release: Path) -> None:
    assert litmus_engine.Model is litmus_engine.Engine
    assert isinstance(model.version, str)
    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    if "version" in config.get("provenance", {}):
        assert model.version == config["provenance"]["version"]
    else:
        digest = hashlib.sha256((release / "model.safetensors").read_bytes() + (release / "config.json").read_bytes())
        assert model.version == f"sha256:{digest.hexdigest()[:16]}"
    assert isinstance(model.count_tokens("Hello there."), int)
    reading = model.score("Hello there.")
    assert isinstance(reading, dict) and set(reading) >= {"n_tokens", "chunk_tokens", "verdict", "chunks", "windows", "cells", "label", "chance"}
    chance = reading["chance"]
    assert set(chance) == {"value", "interval", "reference_share"} and 0.0 < chance["reference_share"] < 1.0
    assert chance["interval"][0] <= chance["value"] <= chance["interval"][1]


def test_score_equals_command_line(model: litmus_engine.Model, release: Path, cli: Path, texts: dict[str, str], tmp_path: Path) -> None:
    rows = cli_rows(cli, release, list(texts.values()), tmp_path)
    chunk_counts = []
    for (name, text), row in zip(texts.items(), rows, strict=True):
        if "error" in row:
            with pytest.raises(ValueError) as refused:
                model.score(text)
            assert str(refused.value) == row["error"], name
            continue
        reading = model.score(text)
        assert reading.pop("label") == row["label"], name
        assert reading.pop("chance") == row["chance"], name
        assert reading == row["reading"], name
        assert reading["n_tokens"] == model.count_tokens(text), name
        chunk_counts.append(len(reading["chunks"]))
    assert [row.get("error") for row in rows[:2]] == ["the text has no tokens, so no reading"] * 2
    assert chunk_counts == [1, 1, 1, 1, 1, 2, 3]


def test_count_tokens_equals_engine(model: litmus_engine.Model, texts: dict[str, str]) -> None:
    for text in texts.values():
        count = model.count_tokens(text)
        assert count == len(model.tokenize(text))
        if count:
            assert model.score(text)["n_tokens"] == count


def test_longest_text_reading_fits_the_server(model: litmus_engine.Model) -> None:
    text = prose(model, SERVER_TOKEN_CAP, seed=5)
    assert model.count_tokens(text) == SERVER_TOKEN_CAP
    reading = model.score(text, max_tokens=SERVER_TOKEN_CAP)
    assert len(reading["chunks"]) == 16
    assert finite_numbers(reading) > 10_000
    serialized = json.dumps(reading, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")
    assert len(serialized) <= SERVER_READING_BYTES
    assert json.loads(serialized) == reading
    print(f"\nreading of {SERVER_TOKEN_CAP} tokens: {len(serialized):,} bytes of JSON", file=sys.stderr)


def test_token_cap(model: litmus_engine.Model, texts: dict[str, str]) -> None:
    text = texts["one chunk"]
    count = model.count_tokens(text)
    assert model.score(text, max_tokens=count) == model.score(text)
    with pytest.raises(ValueError, match=f"more than {count - 1} tokens"):
        model.score(text, max_tokens=count - 1)
    with pytest.raises(ValueError, match="more than 100 tokens"):
        model.score(texts["three chunks"], max_tokens=100)


def test_scoring_releases_the_gil(model: litmus_engine.Model, texts: dict[str, str]) -> None:
    """While another thread scores, this thread keeps running Python; with the GIL held it could not."""
    text = texts["three chunks"]
    done = threading.Event()
    started = threading.Event()
    spins, baseline_start = 0, time.perf_counter()
    while time.perf_counter() - baseline_start < 0.2:
        spins += 1
        done.is_set()
    free_rate = spins / (time.perf_counter() - baseline_start)

    def score() -> float:
        started.set()
        start = time.perf_counter()
        model.score(text)
        elapsed = time.perf_counter() - start
        done.set()
        return elapsed

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(score)
        started.wait()
        spins = 0
        while not done.is_set():
            spins += 1
            time.perf_counter()
        elapsed = future.result()
    assert elapsed > 0.2
    # Holding the GIL, the scorer would let this loop run only for the first switch interval (5 ms).
    assert spins > 0.25 * free_rate * elapsed, f"{spins} loop turns in {elapsed:.2f} s, {free_rate:.0f} a second when idle"


def test_concurrent_scores_equal_serial_and_run_in_parallel(model: litmus_engine.Model) -> None:
    workers = 4
    inputs = [prose(model, 6_000, seed=10 + index) for index in range(workers)]
    serial = [model.score(text) for text in inputs]
    barrier = threading.Barrier(workers)

    def score(text: str) -> dict:
        barrier.wait()
        return model.score(text)

    cpu_start, wall_start = time.process_time(), time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        concurrent = list(pool.map(score, inputs))
    cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    assert concurrent == serial
    if (os.cpu_count() or 1) >= workers:
        # CPU time over wall time is the number of threads busy at once; 1 if the calls ran one after another.
        assert cpu / wall > 1.5, f"{cpu:.2f} s of CPU in {wall:.2f} s"


def test_load_errors(release: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such release folder"):
        litmus_engine.Model.from_path(str(tmp_path / "missing"))
    by_file = litmus_engine.Model.from_path(str(release / "model.safetensors"))
    assert by_file.version == litmus_engine.Model.from_path(str(release)).version
    with pytest.raises(ValueError, match="is not a release"):
        litmus_engine.Model.from_path(str(release / "config.json"))

    partial = tmp_path / "partial"
    partial.mkdir()
    shutil.copy(release / "model.safetensors", partial)
    shutil.copy(release / "config.json", partial)
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        litmus_engine.Model.from_path(str(partial))
    (partial / "tokenizer.json").write_bytes((release / "tokenizer.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="the release expects"):
        litmus_engine.Model.from_path(str(partial))

    refused = tmp_path / "refused"
    refused.mkdir()
    shutil.copy(release / "tokenizer.json", refused)
    (refused / "config.json").write_text("{}", encoding="utf-8")
    (refused / "model.safetensors").write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="cannot load"):
        litmus_engine.Model.from_path(str(refused))
    with pytest.raises(ValueError, match="unknown backend"):
        litmus_engine.Model.from_path(str(release), backend="abacus")


def test_contribute_scoring_module_with_the_real_binding(release: Path, texts: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """`scoring.py` of the contribute server, unchanged, on this binding instead of its tests' stand-in."""
    monkeypatch.syspath_prepend(str(CONTRIBUTE_SOURCE))
    from litmus_contribute.config import Settings
    from litmus_contribute.scoring import EngineScorer, serialize_reading

    settings = Settings()
    assert (settings.text_max_tokens, settings.reading_max_bytes) == (SERVER_TOKEN_CAP, SERVER_READING_BYTES)
    scorer = EngineScorer(release)
    text = texts["two chunks"]
    assert scorer.count_tokens(text) == 8_193
    serialized = serialize_reading(scorer.score(text), settings.reading_max_bytes)
    assert json.loads(serialized)["n_tokens"] == 8_193
    assert scorer.version == litmus_engine.Model.from_path(str(release)).version
