"""Training presentations (presentation.presented): every style is drawn, the draw depends on the seed alone, code is
left as written, no text is emptied or read slowly, synonyms are swapped only where drawn, and the text store and the
presenter feed training steps a new presentation at every draw (data.TextStore, data.Presenter), drawn in epochs
(data.EpochSampler)."""

from __future__ import annotations

import json
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pathlib import Path
from typing import Any
from litmus_student.contract import reference
from litmus_student.data import EpochSampler, Presenter, TextStore, TokenStore, write_store, write_texts
from litmus_student.presentation import DASH_STYLES, P_SYNONYMS, choices, presented, swapped


TEXT = ("## Results\n\nThe model — trained on “clean” data… works well. It wraps\nat seventy columns -- as old emails do.  "
        "Then it stops.\n\n- first point\n- second point\n\n```python\n# load the data\ndf = pd.read_csv(path)  # “keep”\n```")
CODE = "# load the data\ndf = pd.read_csv(path)  # “keep”"
SEEDS = range(400)
TABLE = {"model": ["simulation"], "clean": ["unclouded"], "works": ["plant"], "data": ["information"], "stops": ["halts"],
         "point": ["detail"], "have": ["hold"]}


def generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def outputs(text: str = TEXT, table: dict[str, list[str]] | None = None) -> list[str]:
    return [presented(text, generator(seed), table) for seed in SEEDS]


def test_a_seed_gives_one_presentation() -> None:
    assert presented(TEXT, generator(17), TABLE) == presented(TEXT, generator(17), TABLE)
    assert len(set(outputs())) > 50  # and seeds give many


def test_every_dash_style_is_drawn() -> None:
    seen = {style: False for style in DASH_STYLES}
    for out in outputs("The model — trained well, then — it stopped."):
        for style in DASH_STYLES:
            seen[style] |= f"model{style}trained" in out
    assert all(seen.values()), seen


def test_quotes_ellipses_and_gaps_both_ways() -> None:
    texts = outputs()
    assert any("“clean”" in out for out in texts) and any('"clean"' in out for out in texts)
    assert any("…" in out for out in texts) and any("data... works" in out for out in texts)
    assert any("do.  Then" in out for out in texts) and any("do. Then" in out for out in texts)


def test_prose_joined_or_wrapped_never_as_it_came() -> None:
    for seed in SEEDS:
        choice = choices(generator(seed))
        out = presented(TEXT, generator(seed))
        prose = [line for line in out.split("\n") if "columns" in line or "wraps" in line]
        if choice["width"] == 0:
            assert len(prose) == 1  # the paragraph on one line
        else:  # wrapped before a phrase was perhaps made bold, so the marks are not counted
            assert all(len(line.replace("**", "")) <= choice["width"] for line in prose)


def test_markdown_rendered_or_kept() -> None:
    texts = outputs()
    assert any("- first point" in out for out in texts) and any("- first point" not in out for out in texts)
    assert any("```python" in out for out in texts) and any("```" not in out for out in texts)


def test_code_left_as_written() -> None:
    assert all(CODE in out for out in outputs(table=TABLE))


def test_never_empty() -> None:
    assert all(out == "---" for out in outputs("---"))


def test_synonyms_swapped_only_in_texts_that_draw_them() -> None:
    swaps = 0
    for seed in range(2000):
        out = presented(TEXT, generator(seed), TABLE)
        changed = any(word in out for word in ("simulation", "unclouded", "information", "halts", "detail", "Simulation"))
        assert not changed or choices(generator(seed))["synonyms"]
        swaps += changed
    assert 0 < swaps < 2000 * P_SYNONYMS  # some drawn texts swap a word; none that did not draw
    always = np.random.default_rng(0)
    out = " ".join(swapped("The Model has data, DATA, `data` and have.", TABLE, always) for _ in range(200))
    assert "Simulation" in out and "information" in out  # its capital kept
    assert "DATA" in out and "hold" not in out  # capitals and function words never swapped


@pytest.mark.parametrize("unit", ["`", "[a](", "**a ", "# ", " ", " " * 999 + "=", "— ", "a -- ", '"a\' '])
def test_linear_on_adversarial_lines(unit: str) -> None:
    line = unit * (200_000 // len(unit))
    started = time.perf_counter()
    presented(line + " end of the line", generator(0), TABLE)
    assert time.perf_counter() - started < 10.0


def test_presenter_presents_anew_at_every_draw(tmp_path: Path, tokenizer: Any) -> None:
    texts = ["The model — trained “well”… works.", "A second, longer text about rivers\nand ledgers.", "日本語", "code = 1\n    x"]
    counts = [len(reference.token_ids(tokenizer, text)) for text in texts]
    ids = [2, 0, 3, 1]
    table = pa.table({"id": ids, "text": texts, "label": pa.array([0, 1, 2, 0], pa.int8()), "tokens": counts,
                      "long_eval": [False, False, True, False]})
    pq.write_table(table, tmp_path / "corpus.parquet")
    write_texts(tmp_path / "corpus.parquet", tmp_path / "texts", batch_rows=3)
    store = TextStore(tmp_path / "texts")
    assert len(store) == 4 and [store.text(corpus_id) for corpus_id in ids] == texts
    (tmp_path / "synonyms.json").write_text(json.dumps({"table": TABLE}), encoding="utf-8")
    presenter = Presenter(tmp_path / "texts", tmp_path / "synonyms.json", max_tokens=6, workers=2, generator=np.random.default_rng(0))
    try:
        rows = np.asarray([2, 0, 0, 1])
        drawn = [presenter.ids(rows) for _ in range(30)]
    finally:
        presenter.close()
    for step in drawn:
        assert all(1 <= len(ids_) <= 6 for ids_ in step)  # at most max_tokens, never empty
    first_text = [tokenizer.decode(step[0].tolist()) for step in drawn]
    assert len(set(first_text)) > 1  # the same text drawn again is presented anew
    assert all("model" in text or "simulation" in text for text in first_text)


def test_epoch_sampler_draws_every_text_once_an_epoch(tmp_path: Path) -> None:
    texts = [np.arange(1, 2 + index % 5) for index in range(30)]
    label = np.asarray([index % 3 for index in range(30)])
    split = np.asarray([0] * 25 + [1] * 5)
    write_store(tmp_path / "store", texts, label, split, np.arange(30) // 4, ["train", "validation"])
    sampler = EpochSampler(TokenStore(tmp_path / "store"), "train", 8192, np.ones(30, dtype=bool))
    rng = np.random.default_rng(0)
    first = np.concatenate([sampler.draw(10, rng) for _ in range(5)])  # two epochs of 25 texts
    assert sorted(first[:25].tolist()) == list(range(25)) and sorted(first[25:].tolist()) == list(range(25))
    assert first[:25].tolist() != first[25:].tolist()  # each epoch its own order
    assert sampler.epoch == 1
