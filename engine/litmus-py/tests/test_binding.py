"""The Python binding returns the library's reading unchanged and matches `reference.py`'s fixtures.

The random-weight fixture release `s192_l4` (engine/README.md, Tests) checks the arithmetic against the reference;
`test_release.py` checks the trained release against the command line and the contribute server's needs.
"""

from __future__ import annotations

import json
import threading
import litmus_engine
import pytest

from pathlib import Path


@pytest.fixture(scope="module")
def fixture_release(fixtures_folder: Path) -> Path:
    return fixtures_folder / "s192_l4"


@pytest.fixture(scope="module")
def engine(fixture_release: Path, tokenizer_file: Path) -> litmus_engine.Engine:
    return litmus_engine.Engine.from_path(fixture_release, tokenizer=tokenizer_file)


@pytest.fixture(scope="module")
def fixtures(fixture_release: Path) -> dict:
    return json.loads((fixture_release / "fixtures.json").read_text(encoding="utf-8"))


def test_reading_matches_reference(engine: litmus_engine.Engine, fixtures: dict) -> None:
    for record in fixtures["texts"]:
        document, want = engine.score(record["text"]), record["reading"]
        assert document["n_tokens"] == want["n_tokens"] == len(record["ids"])
        assert len(document["chunks"]) == len(want["chunks"]) and len(document["windows"]) == len(want["windows"])
        assert len(document["cells"]) == len(want["cells"])
        verdict, want_verdict = document["verdict"], want["verdict"]
        assert all(abs(a - b) <= 2e-3 for a, b in zip(verdict["three_way"], want_verdict["three_way"], strict=True))
        assert abs(verdict["strip"]["position"] - want_verdict["strip"]["position"]) <= 2e-3
        for chunk, want_chunk in zip(document["chunks"], want["chunks"], strict=True):
            assert (chunk["start"], chunk["end"]) == (want_chunk["start"], want_chunk["end"])
            assert abs(chunk["score"] - want_chunk["score"]) <= 1e-3
            assert [entry["origin"] for entry in chunk["explain"]["conformal"]] == ["human", "ai", "mixed"]
        for cell, want_cell in zip(document["cells"], want["cells"], strict=True):
            assert abs(cell["position"] - want_cell["position"]) <= 2e-3
        text = record["text"]
        assert document["cells"][0]["start_char"] == 0 and document["cells"][-1]["end_char"] == len(text)
        assert document["label"] in {"human", "AI", "mixed", "uncertain", "unlike reference"}
        # The surprise reading (SPEC.md, Surprise), within SPEC.md's tolerance scaled for a short text.
        surprise, want_surprise = engine.surprise(text), record["surprise"]
        scale = max(1.0, (512 / len(record["ids"])) ** 0.5)
        assert surprise["n_tokens"] == len(record["ids"])
        assert abs(surprise["mean_log_probability"] - want_surprise["mean_log_probability"]) <= 2e-3 * scale
        assert all(abs(a - b) <= 1e-2 for a, b in zip(surprise["percentile"], want_surprise["percentile"], strict=True))


def test_int8_output_release_reports_itself_and_matches_reference(fixtures_folder: Path, tokenizer_file: Path) -> None:
    """The same random weights as `s192_l4_integer`, released with int8 output (format 3): `info` names the output and the
    attention, and every reading is the reference's within the tolerances above."""
    release = fixtures_folder / "s192_l4_integer_int8"
    engine = litmus_engine.Engine.from_path(release, tokenizer=tokenizer_file)
    summary = engine.info()
    assert summary["output"] == "int8" and summary["attention"] == "integer"
    float_summary = litmus_engine.Engine.from_path(fixtures_folder / "s192_l4", tokenizer=tokenizer_file).info()
    assert float_summary["output"] == "float" and float_summary["attention"] == "float"
    assert summary["bytes"] < float_summary["bytes"]  # a bank of int8 codes is smaller than one of float16 vectors
    for record in json.loads((release / "fixtures.json").read_text(encoding="utf-8"))["texts"]:
        document, want = engine.score(record["text"]), record["reading"]
        assert document["n_tokens"] == want["n_tokens"] == len(record["ids"])
        assert len(document["chunks"]) == len(want["chunks"]) and len(document["windows"]) == len(want["windows"])
        assert all(abs(a - b) <= 2e-3 for a, b in zip(document["verdict"]["three_way"], want["verdict"]["three_way"], strict=True))
        for chunk, want_chunk in zip(document["chunks"], want["chunks"], strict=True):
            assert abs(chunk["score"] - want_chunk["score"]) <= 1e-3
            assert abs(chunk["closeness"] - want_chunk["closeness"]) <= 1e-3
        for cell, want_cell in zip(document["cells"], want["cells"], strict=True):
            assert abs(cell["position"] - want_cell["position"]) <= 2e-3


def test_tokens_and_standard_reading(engine: litmus_engine.Engine, fixtures: dict) -> None:
    for record in fixtures["texts"]:
        assert litmus_engine.Engine.standard_reading(record["text"]) == record["standard_reading"]
        assert [token[0] for token in engine.tokenize(record["text"])] == record["ids"]
        assert engine.count_tokens(record["text"]) == len(record["ids"])


def test_errors_and_info(engine: litmus_engine.Engine) -> None:
    with pytest.raises(ValueError, match="no tokens"):
        engine.score("")
    summary = engine.info()
    assert summary["width"] == 192 and summary["depth"] == 4 and summary["backend"] in litmus_engine.Engine.backends()
    assert summary["chunk_tokens"] == 8192 and summary["window_width"] == 512 and summary["window_stride"] == 128
    assert summary["license"] == "PolyForm-Noncommercial-1.0.0"
    with pytest.raises(TimeoutError):
        engine.score("word " * 50_000, timeout_seconds=0.0)


def test_bytes_constructor_equals_from_path(engine: litmus_engine.Engine, fixture_release: Path, tokenizer_file: Path, fixtures: dict) -> None:
    from_bytes = litmus_engine.Engine(
        (fixture_release / "model.safetensors").read_bytes(),
        (fixture_release / "config.json").read_bytes(),
        tokenizer_file.read_bytes(),
        backend="scalar",
    )
    assert from_bytes.version == engine.version and from_bytes.info()["backend"] == "scalar"
    text = fixtures["texts"][2]["text"]
    assert from_bytes.score(text) == engine.score(text)  # every backend gives the same bits


def test_threads_share_one_engine(engine: litmus_engine.Engine, fixtures: dict) -> None:
    text = fixtures["texts"][-1]["text"]
    alone = engine.score(text)
    readings: list[dict] = []
    threads = [threading.Thread(target=lambda: readings.append(engine.score(text))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(readings) == 4 and all(reading == alone for reading in readings)
