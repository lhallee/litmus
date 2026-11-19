"""SPEC.md's Int8 output (format 3): the quantizer, the integer compare of the reference, the release it writes and reads, the
training switch carried through a checkpoint to an export, and the bank recipe the export can draw."""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any
from litmus_student.contract import reference
from litmus_student.data import BANK_RECIPES, TeacherFiles, TokenStore, recipe_bank_rows, uniform_order, write_store
from litmus_student.export import INT8_TOLERANCES, export_release, model_output, with_output
from litmus_student.model import StudentModel
from litmus_student.train import Evaluation, load_checkpoint


@pytest.fixture(scope="module")
def int8_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tiny_int8_release")
    reference.init_release(reference.named_config("tiny"), directory, seed=3, bank_per_origin=8, calibration_tokens=200, attention="integer", output="int8")
    return directory


@pytest.fixture(scope="module")
def int8_model(int8_release: Path) -> Any:
    return reference.ReferenceModel.load(int8_release)


# ---------------------------------------------------------------- the quantizer


def test_quantize_output_is_the_specified_quantizer(generator: np.random.Generator) -> None:
    X = (generator.standard_normal((6, 40)) * generator.uniform(0.01, 30.0, (6, 1))).astype(np.float32)  # (6, 40)
    codes = reference.quantize_output(X, 1e-5)
    assert codes.dtype == np.int8 and codes.shape == X.shape
    for row, vector in zip(codes, X, strict=True):
        s = np.float32(127.0) / max(np.abs(vector).max(), np.float32(1e-5))
        assert row.tolist() == np.clip(np.rint(vector * s), -127, 127).astype(np.int64).tolist()
        assert np.abs(row.astype(np.int64)).max() == 127  # the largest component is the scale's own
    # A rounding tie goes to the even code: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2 at a scale of exactly 1.
    assert reference.quantize_output(np.asarray([[127.0, 0.5, 1.5, 2.5, -0.5, -1.5]], dtype=np.float32), 1e-5).tolist() == [[127, 0, 2, 2, 0, -2]]
    assert not reference.quantize_output(np.zeros((1, 8), dtype=np.float32), 1e-5).any()  # the zero vector has zero codes
    # An unit vector of codes carries them: quantizing it again returns them.
    unit = codes.astype(np.float32) / np.linalg.norm(codes.astype(np.float32), axis=1, keepdims=True)  # (6, 40)
    assert np.array_equal(reference.quantize_output(unit, 1e-5), codes)


def test_inverse_norms_are_exact_roots_and_zero_for_zero() -> None:
    codes = np.asarray([[3, 4, 0], [0, 0, 0], [-127, 127, 127]], dtype=np.int8)
    inverse = reference.inverse_norms(codes)
    assert inverse[0] == 1.0 / 5.0 and inverse[1] == 0.0 and inverse[2] == 1.0 / math.sqrt(3 * 127 * 127)
    assert reference.inverse_norms(np.zeros(4, dtype=np.int8)) == 0.0


# ---------------------------------------------------------------- the release


def test_int8_release_is_format_3_and_names_what_it_is(int8_release: Path, int8_model: Any) -> None:
    config = json.loads((int8_release / "config.json").read_text(encoding="utf-8"))
    assert config["format_version"] == reference.INT8_OUTPUT_FORMAT_VERSION == 3
    assert config["model"]["output"] == "int8" and config["model"]["attention"] == "integer"
    assert reference.StudentConfig.from_release(config) == reference.named_config("tiny")
    tensors = reference.read_safetensors(int8_release / "model.safetensors")
    w = int8_model.config.head_out
    assert tensors["bank.vectors"].dtype == np.int8 and tensors["bank.vectors"].shape == (24, 2 * w)
    assert tensors["bank.vectors"].min() >= -127 and np.abs(tensors["bank.vectors"].astype(np.int64)).max() == 127
    assert int8_model.output == "int8" and int8_model.attention == "integer"
    q, k = int8_model.head(np.zeros(2 * int8_model.config.width, dtype=np.float32))
    assert q.dtype == np.int8 and k.dtype == np.int8 and q.shape == (w,)


def test_int8_release_with_float_attention_names_it_too(tmp_path: Path) -> None:
    reference.init_release(reference.named_config("tiny"), tmp_path, seed=3, bank_per_origin=4, calibration_tokens=100, output="int8")
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["format_version"] == 3 and config["model"]["attention"] == "float" and config["model"]["output"] == "int8"
    assert reference.ReferenceModel.load(tmp_path).attention == "float"


def test_formats_one_and_two_are_float_output_and_refuse_a_named_one(tiny_release: Path, int8_release: Path) -> None:
    float_config = json.loads((tiny_release / "config.json").read_text(encoding="utf-8"))
    int8_config = json.loads((int8_release / "config.json").read_text(encoding="utf-8"))
    assert reference.release_output(float_config) == "float" and reference.release_output(int8_config) == "int8"
    assert reference.release_attention(int8_config) == "integer"
    with pytest.raises(ValueError, match="model.output needs format 3"):
        reference.release_output({**float_config, "model": {**float_config["model"], "output": "int8"}})
    with pytest.raises(ValueError, match="model.output needs format 3"):
        reference.release_output({**int8_config, "format_version": 2})
    with pytest.raises(ValueError, match="names model.output"):
        reference.release_output({**int8_config, "model": {key: value for key, value in int8_config["model"].items() if key != "output"}})
    with pytest.raises(ValueError, match="names model.output"):
        reference.release_output({**int8_config, "model": {**int8_config["model"], "output": "float"}})
    with pytest.raises(ValueError, match="names model.attention"):
        reference.release_attention({**int8_config, "model": {key: value for key, value in int8_config["model"].items() if key != "attention"}})
    with pytest.raises(ValueError, match="not a litmus-student release"):
        reference.release_output({**int8_config, "format_version": 4})
    assert [reference.release_format_version(a, o) for a, o in (("float", "float"), ("integer", "float"), ("float", "int8"), ("integer", "int8"))] == [1, 2, 3, 3]


def test_a_bank_with_the_code_minus_128_or_float16_in_int8_output_fails_to_load(int8_release: Path, tmp_path: Path) -> None:
    tensors = reference.read_safetensors(int8_release / "model.safetensors")
    config = json.loads((int8_release / "config.json").read_text(encoding="utf-8"))
    bad = {**tensors, "bank.vectors": tensors["bank.vectors"].copy()}
    bad["bank.vectors"][0, 0] = -128
    with pytest.raises(ValueError, match="-128"):
        reference.ReferenceModel(reference.StudentConfig.from_release(config), bad, config["calibration"], config["bank"]["fields"], "integer", "int8")
    half = {**tensors, "bank.vectors": tensors["bank.vectors"].astype(np.float16)}
    with pytest.raises(AssertionError, match="float16"):
        reference.ReferenceModel(reference.StudentConfig.from_release(config), half, config["calibration"], config["bank"]["fields"], "integer", "int8")


def test_int8_bank_is_a_little_over_half_the_size_of_float16(tmp_path: Path) -> None:
    sizes = {}
    for output in ("float", "int8"):
        reference.init_release(reference.named_config("tiny"), tmp_path / output, seed=3, bank_per_origin=200, calibration_tokens=100, output=output)
        sizes[output] = {name: array.nbytes for name, array in reference.read_safetensors(tmp_path / output / "model.safetensors").items()}["bank.vectors"]
    w = reference.named_config("tiny").head_out
    assert sizes == {"float": 600 * 4 * w, "int8": 600 * 2 * w}


# ---------------------------------------------------------------- the compare


def naive_bank_arrays(model: Any, codes: np.ndarray, q: np.ndarray, k: np.ndarray) -> tuple[list[float], list[float]]:
    """SPEC.md's compare with Python integers and `math`, one bank row at a time."""
    # codes: (m, 2w) int8, the release's bank.vectors
    w = model.config.head_out
    bank = codes.astype(np.int64)  # (m, 2w)
    q, k = [int(c) for c in q], [int(c) for c in k]

    def inverse(codes: list[int]) -> float:
        square = sum(c * c for c in codes)
        return 1.0 / math.sqrt(square) if square else 0.0

    def cosine(a: list[int], b: list[int], ia: float, ib: float) -> float:
        return (float(sum(x * y for x, y in zip(a, b, strict=True))) * ia) * ib

    iq, ik = inverse(q), inverse(k)
    p, cosines = [], []
    for row in bank:
        bq, bk = [int(c) for c in row[:w]], [int(c) for c in row[w:]]
        ibq, ibk = inverse(bq), inverse(bk)
        pair = cosine(q, bk, iq, ibk) + cosine(k, bq, ik, ibq)
        logit = model.pair_scale * pair / 2.0 + model.pair_bias
        p.append(0.5 * (1.0 + math.tanh(0.5 * logit)))
        cosines.append((cosine(q, bq, iq, ibq) + cosine(k, bk, ik, ibk)) / 2.0)

    return p, cosines


def test_the_compare_is_the_specified_integer_arithmetic(int8_model: Any, int8_release: Path, generator: np.random.Generator) -> None:
    w = int8_model.config.head_out
    codes = reference.read_safetensors(int8_release / "model.safetensors")["bank.vectors"]  # (m, 2w)
    cases = [(generator.integers(-127, 128, w).astype(np.int8), generator.integers(-127, 128, w).astype(np.int8)) for _ in range(4)]
    cases.append((np.zeros(w, dtype=np.int8), generator.integers(-127, 128, w).astype(np.int8)))  # a zero query reads cosine 0, not NaN
    cases.append((np.full(w, 127, dtype=np.int8), np.zeros(w, dtype=np.int8)))
    for q, k in cases:
        p, cosines = int8_model.bank_arrays(q, k)
        want_p, want_cosines = naive_bank_arrays(int8_model, codes, q, k)
        assert np.isfinite(p).all() and np.isfinite(cosines).all()
        assert cosines.tolist() == want_cosines  # integer sums, two float64 products in a fixed order: the same bits
        assert np.abs(p - np.asarray(want_p)).max() <= 1e-15  # `tanh` of two math libraries
        # And the same numbers as the float arithmetic on the unit vectors of the codes.
        unit_q, unit_k = int8_model.unit_vectors(q, k)
        bank = int8_model.bank_query, int8_model.bank_key
        unit_bank = [b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-300) for b in bank]
        float_cosines = (unit_bank[0] @ unit_q + unit_bank[1] @ unit_k) / 2.0
        assert np.abs(float_cosines - cosines).max() <= 1e-12


def test_a_zero_query_and_key_reads_the_pair_logit_of_the_bias(int8_model: Any) -> None:
    w = int8_model.config.head_out
    zero = np.zeros(w, dtype=np.int8)
    p, cosines = int8_model.bank_arrays(zero, zero)
    assert (cosines == 0.0).all()
    assert np.allclose(p, 0.5 * (1.0 + np.tanh(0.5 * int8_model.pair_bias)), rtol=0, atol=1e-15)


def test_int8_streaming_equals_whole_and_codes_are_what_the_vectors_hold(int8_model: Any, generator: np.random.Generator) -> None:
    config = int8_model.config
    W, C = config.block_size, config.chunk_tokens
    for n in (1, W + 1, C + 1, 211):
        ids = generator.integers(0, config.vocab_size, n)  # (n,)
        whole, vectors = int8_model.read_whole(ids)
        for piece in (1, 7, W - 1, n):
            streamed, streamed_vectors = int8_model.read_streaming(ids, piece)
            assert streamed["verdict"]["p_values"] == whole["verdict"]["p_values"]
            for mine, theirs in zip(streamed_vectors["chunks"], vectors["chunks"], strict=True):
                assert mine["query"].dtype == np.int8 and np.abs(mine["query"].astype(np.int64) - theirs["query"]).max() <= 1
    for chunk in vectors["chunks"]:
        assert chunk["query"].dtype == np.int8 and chunk["key"].dtype == np.int8 and np.abs(chunk["query"].astype(np.int64)).max() <= 127


def test_int8_reads_near_the_float_release_of_the_same_weights(int8_model: Any, tmp_path: Path, generator: np.random.Generator) -> None:
    """The same weights and the same random bank vectors, released in float and in int8 (integer attention in both): the
    int8 codes and the integer compare move a reading only by what quantization to 127ths costs."""
    reference.init_release(reference.named_config("tiny"), tmp_path, seed=3, bank_per_origin=8, calibration_tokens=200, attention="integer")
    floating = reference.ReferenceModel.load(tmp_path)
    for n in (20, 80, 300):
        ids = generator.integers(0, 50000, n)  # (n,)
        a, _ = int8_model.read_whole(ids)
        b, _ = floating.read_whole(ids)
        assert np.abs(np.asarray(a["chunks"][0]["mean"]) - np.asarray(b["chunks"][0]["mean"])).max() < 0.05


# ---------------------------------------------------------------- checkpoints and export


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The synthetic corpus of `test_export` and two checkpoints trained on it for 120 steps of the tiny config, the second
    with `--int8-output`: the store, the corpus and the two best checkpoints."""
    from litmus_student.train import parser, train
    from test_export import synthetic_corpus

    directory = tmp_path_factory.mktemp("trained")
    store, corpus, teacher = synthetic_corpus(directory)
    checkpoints = {}
    for name, flags in (("float", []), ("int8", ["--int8-output"])):
        train(parser().parse_args([
            "--config", "tiny", "--data", str(store), "--teacher-export", str(teacher), "--experiment-id", "2026-11-17_student_test",
            "--out", str(directory / name), "--device", "cpu", "--pilot", "--pilot-steps", "120", "--steps", "120", "--lr", "4e-3",
            "--lr-second", "3e-3", "--warmup", "3", "--texts-per-step", "60", "--micro-tokens", "256", "--pack-multiple", "256",
            "--eval-every", "60", "--val-texts", "60", "--eval-per-origin", "30", "--bank-per-origin", "30", "--max-train-tokens", "96", *flags,
        ]))
        checkpoints[name] = directory / name / "best.pt"

    return {"store": store, "corpus": corpus, **checkpoints}


def test_with_output_switches_the_head_in_place() -> None:
    model = StudentModel(reference.named_config("tiny"))
    assert model_output(model) == "float"
    assert with_output(model, "int8") is model and model_output(model) == "int8"
    with pytest.raises(ValueError, match="not one of"):
        with_output(model, "int4")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_checkpoint_trained_without_the_switch_is_refused_int8_unless_told_to_quantize(trained: dict[str, Path], tmp_path: Path) -> None:
    store, corpus, checkpoint = trained["store"], trained["corpus"], trained["float"]
    assert load_checkpoint(checkpoint, torch.device("cpu"))[1]["int8_output"] is False
    options: dict[str, Any] = {"bank_per_origin": 20, "check_texts": 6, "micro_tokens": 256, "device": torch.device("cpu")}
    with pytest.raises(ValueError, match="trained without --int8-output"):
        export_release(checkpoint, store, corpus, tmp_path / "refused", output="int8", **options)
    assert not (tmp_path / "refused").exists()

    exported = export_release(checkpoint, store, corpus, tmp_path / "after", output="int8", quantize_after_training=True, **options)
    config = json.loads((tmp_path / "after" / "config.json").read_text(encoding="utf-8"))
    assert exported["output"] == "int8" and config["format_version"] == 3 and config["model"]["output"] == "int8"
    assert config["provenance"]["output"] == {"release": "int8", "trained": "float"}
    assert reference.ReferenceModel.load(tmp_path / "after").output == "int8"

    default = export_release(checkpoint, store, corpus, tmp_path / "default", **options)  # no --output: the checkpoint's own, float
    default_config = json.loads((tmp_path / "default" / "config.json").read_text(encoding="utf-8"))
    assert default["output"] == "float" and default_config["format_version"] == 1 and "output" not in default_config["provenance"]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_switch_trains_exports_and_reads_end_to_end(trained: dict[str, Path], tmp_path: Path) -> None:
    """`train.py --int8-output` on the tiny config, the checkpoint recording it, the default export a format 3 release with an
    int8 bank, and the export's own check of the reference against the model on codes."""
    store, corpus = trained["store"], trained["corpus"]
    _, checkpoint = load_checkpoint(trained["int8"], torch.device("cpu"))
    assert checkpoint["int8_output"] is True and checkpoint["arguments"]["int8_output"] is True
    assert checkpoint["model"]["head.fc2.weight"].shape[0] == 2 * reference.named_config("tiny").head_out

    exported = export_release(trained["int8"], store, corpus, tmp_path / "release", bank_per_origin=40, check_texts=16,
                              micro_tokens=256, device=torch.device("cpu"))
    config = json.loads((tmp_path / "release" / "config.json").read_text(encoding="utf-8"))
    assert exported["output"] == "int8" and config["format_version"] == 3 and config["model"]["output"] == "int8"
    assert config["provenance"]["output"] == {"release": "int8", "trained": "int8"} and "attention" not in config["provenance"]
    tensors = reference.read_safetensors(tmp_path / "release" / "model.safetensors")
    assert tensors["bank.vectors"].dtype == np.int8 and tensors["bank.vectors"].shape == (120, 2 * reference.named_config("tiny").head_out)
    assert exported["parity"]["query_key"] <= INT8_TOLERANCES["query_key"] and exported["parity"]["pooled"] <= INT8_TOLERANCES["pooled"]

    loaded = reference.ReferenceModel.load(tmp_path / "release")
    ids = np.random.default_rng(1).integers(1000, 2000, 250)  # (n,) three chunks of the tiny config
    whole, _ = loaded.read_whole(ids)
    streamed, _ = loaded.read_streaming(ids, 5)
    assert whole["verdict"] == streamed["verdict"] and len(whole["chunks"]) == 3
    nearest = whole["chunks"][0]["explain"]["nearest"][0]
    assert -1.0 <= nearest["cosine"] <= 1.0 and nearest["source"].startswith("source-")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_reading_many_texts_at_once_gives_each_text_its_own_reading(trained: dict[str, Path], tmp_path: Path) -> None:
    """`Reader.read_many` (packed forwards, then every chunk against the whole bank in one product) against `Reader.read`
    of each text: the same verdicts bit for bit from the same pooled vectors, and pooled vectors that agree to float32
    rounding when the texts share forwards."""
    from litmus_student.evaluate import Reader, verdicts

    export_release(trained["int8"], trained["store"], trained["corpus"], tmp_path / "release", bank_per_origin=40, check_texts=16,
                   micro_tokens=256, device=torch.device("cpu"))
    reader = Reader(trained["int8"], tmp_path / "release", torch.device("cpu"))
    assert reader.release.output == "int8"
    generator = np.random.default_rng(5)
    # The tiny config: half-blocks of 8, blocks of 16, chunks of 96, so one to four chunks a text.
    texts = [generator.integers(1000, 2000, n) for n in (1, 3, 8, 9, 16, 40, 96, 97, 250, 300, 5, 7, 8, 2)]  # (n_i,) each

    expected, pooled, tokens, owner = [], [], [], []
    for index, ids in enumerate(texts):
        states = [reader.states(ids[start:end]) for start, end in reference.chunk_bounds(len(ids), reader.config.chunk_tokens)]
        reading, vectors = reader.release.read_states(states, windowed=False)
        expected.append(reading["verdict"])
        pooled.extend(vector["pooled"] for vector in vectors)
        tokens.extend(len(each) for each in states)
        owner.extend([index] * len(states))
    assert [verdict["chunks"] for verdict in expected] == [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1]
    assert verdicts(reader.release, np.stack(pooled), tokens, np.asarray(owner), len(texts)) == expected

    chunks = [ids[start:end] for ids in texts for start, end in reference.chunk_bounds(len(ids), reader.config.chunk_tokens)]
    together = reader.pooled(chunks, positions=64)  # (c, 2d) several forwards, short texts sharing half-blocks
    np.testing.assert_allclose(together, np.stack(pooled), rtol=1e-4, atol=1e-5)
    many = reader.read_many(texts)
    assert [verdict["chunks"] for verdict in many] == [verdict["chunks"] for verdict in expected]
    for verdict, one in zip(many, expected, strict=True):
        np.testing.assert_allclose(verdict["three_way"], one["three_way"], atol=1e-3)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_heat_map_reads_its_windows_together(trained: dict[str, Path], tmp_path: Path) -> None:
    """`window_map` reads every window against the whole bank in one product: the same cells, bit for bit, as reading
    the windows one at a time with the release's own `bank_reading`, at the release's stride and at sections."""
    from litmus_student.evaluate import Reader, window_map

    export_release(trained["int8"], trained["store"], trained["corpus"], tmp_path / "release", bank_per_origin=40, check_texts=16,
                   micro_tokens=256, device=torch.device("cpu"))
    reader = Reader(trained["int8"], tmp_path / "release", torch.device("cpu"))
    release, config = reader.release, reader.config
    H = reader.states(np.random.default_rng(6).integers(1000, 2000, config.chunk_tokens))  # (n, d)
    for stride in (config.window_stride, config.window_width):
        pool = reference.ChunkPool(config.width, len(H), config.window_width, stride)
        pool.add(H)
        windows = []
        for start, end in reference.window_spans(len(H), config.window_width, stride):
            q, k = release.head(pool.pooled(start, end))  # (w,) each
            reading = release.calibration.read(release.bank_reading(q, k, release.calibration.top_k), end - start)
            windows.append({**reading, "start": start, "end": end, "chunk": 0})
        one_at_a_time = reference.heat_map(windows, 0, len(H), config.window_stride)
        assert len(one_at_a_time) > 1
        assert window_map(reader, H, config.window_width, stride) == one_at_a_time


# ---------------------------------------------------------------- the bank recipe


def recipe_store(directory: Path, splits: list[str]) -> TokenStore:
    generator = np.random.default_rng(0)
    counts = {name: 90 for name in splits}
    split = np.concatenate([np.full(count, splits.index(name)) for name, count in counts.items()])  # (n,)
    n = len(split)
    label = np.arange(n) % 3  # (n,)
    long_eval = (np.arange(n) % 17) == 0  # (n,) some whole long documents, which no bank holds
    texts = [generator.integers(1000 * origin, 1000 * origin + 1000, generator.integers(1, 40)) for origin in label]
    write_store(directory, texts, label, split, generator.integers(0, 10, n), splits, long_eval)

    return TokenStore(directory)


def test_the_mixed_out_recipe_takes_human_and_ai_from_training_and_mixed_from_the_draw(tmp_path: Path) -> None:
    store = recipe_store(tmp_path / "store", ["bank", "bank_draw", "train", "validation"])
    rows = recipe_bank_rows(store, "mixed_out", 12, seed=5)  # (36,)
    assert len(rows) == 36 and len(set(rows.tolist())) == 36
    assert (store.label[rows] == np.repeat([0, 1, 2], 12)).all()  # grouped by origin, as the release's bank is
    assert not store.long_eval[rows].any()
    sources = [store.splits[store.split[rows[origin * 12]]] for origin in range(3)]
    assert sources == ["train", "train", "bank_draw"] and tuple(sources) == BANK_RECIPES["mixed_out"]
    for origin, split in enumerate(sources):
        assert (store.split[rows[origin * 12 : (origin + 1) * 12]] == store.splits.index(split)).all()
    # Uniform: the first 12 of the `uniform` composition's order of the origin's rows, for the seed.
    members = np.flatnonzero(store.split_mask("train") & ~store.long_eval & (store.label == 0))
    assert rows[:12].tolist() == members[uniform_order(len(members), 5, 0)[:12]].tolist()
    assert recipe_bank_rows(store, "mixed_out", 12, seed=5).tolist() == rows.tolist()  # deterministic
    assert recipe_bank_rows(store, "mixed_out", 12, seed=6).tolist() != rows.tolist()


def test_training_reads_validation_against_the_bank_export_will_draw(tmp_path: Path) -> None:
    # A split with no `bank` set: the checkpoint is chosen with the recipe's bank, the rows export draws for the same seed.
    store = recipe_store(tmp_path / "store", ["bank_draw", "train", "validation"])
    arguments = argparse.Namespace(micro_tokens=256, config="tiny", val_texts=20, eval_per_origin=10, bank_split="bank",
                                   bank_per_origin=12, bank_recipe="mixed_out", seed=5)
    evaluation = Evaluation(store, TeacherFiles.absent(len(store)), arguments, np.random.default_rng(5), torch.device("cpu"))
    assert evaluation.bank_rows.tolist() == sorted(recipe_bank_rows(store, "mixed_out", 12, seed=5).tolist())
    assert set(store.split[evaluation.bank_rows].tolist()) == {store.splits.index("train"), store.splits.index("bank_draw")}


def test_a_recipe_the_corpus_cannot_serve_fails_with_a_message_that_names_it(tmp_path: Path) -> None:
    store = recipe_store(tmp_path / "store", ["bank", "train", "validation"])
    with pytest.raises(ValueError, match=r"'mixed_out' takes mixed texts from the 'bank_draw' split, which this corpus does not have"):
        recipe_bank_rows(store, "mixed_out", 12, seed=0)
    full = recipe_store(tmp_path / "full", ["bank", "bank_draw", "train"])
    with pytest.raises(ValueError, match="needs 500 human texts from the 'train' split, which holds"):
        recipe_bank_rows(full, "mixed_out", 500, seed=0)
    with pytest.raises(ValueError, match="no bank recipe"):
        recipe_bank_rows(full, "nonsense", 4, seed=0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_draws_the_recipe_bank_or_fails_loudly(trained: dict[str, Path], tmp_path: Path) -> None:
    store, corpus, checkpoint = trained["store"], trained["corpus"], trained["int8"]  # its splits: bank, calibration, train, validation
    with pytest.raises(ValueError, match="'bank_draw' split, which this corpus does not have"):
        export_release(checkpoint, store, corpus, tmp_path / "failed", bank_per_origin=10, check_texts=4, micro_tokens=256,
                       device=torch.device("cpu"), bank_recipe="mixed_out")
    with pytest.raises(ValueError, match="no bank recipe"):
        export_release(checkpoint, store, corpus, tmp_path / "unknown", bank_per_origin=10, check_texts=4, micro_tokens=256,
                       device=torch.device("cpu"), bank_recipe="nonsense")
