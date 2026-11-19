"""A trained checkpoint to the release of SPEC.md, with its bank and calibration, verified against `reference.py`.

    python -m litmus_student.export --checkpoint RUN/best.pt --data PRETOKENIZED --corpus CORPUS --out RELEASE

1. The weights: ternary values and scales taken from the model's own ternarization, packed by `reference.py`.
2. The bank: the student's unit queries and keys of `--bank-per-origin` texts of each origin of the bank split,
   drawn uniformly over clusters, stored float16, with each one's token count and corpus metadata (source,
   generator, domain, era, format, language; never its text) so a reading can name its nearest reference texts.
3. The calibration: the student's own bank readings of the `calibration` split (mean and spread of the pair
   probabilities per origin, closeness), computed by `reference.py` from the release's bank and fitted into the
   table by the teacher's `litmus_teacher.calibrate.fit_calibration`, embedded in `config.json`, with the sorted
   calibration scores behind each group, so a reading can say how many calibration texts scored as it did.
4. The check: `reference.py`, whole and streamed, against the model in eval mode on the fixture texts.

Attention. A release reads its checkpoint with the attention it was trained with, unless `--attention` names the
other: `--attention integer` releases a model trained with float attention in the optional integer mode of SPEC.md
(Integer attention; a format 2 release), and the bank, the calibration and the check are then all computed with
integer attention, so the release's tables describe the arithmetic the engine runs. A float release is format 1.

Output. A checkpoint trained with `--int8-output` (the head's query and key quantized to int8 in the training
forward) is released with `--output int8` or by default: SPEC.md's Int8 output (format 3), the bank stored as int8
codes and the pair logit computed from integer dot products. A checkpoint trained without it is refused `--output int8`
unless `--quantize-after-training` says to quantize its head's output after the fact, which the provenance records.

Bank. By default the bank is `--bank-per-origin` texts of each origin of `--bank-split`, spread over clusters.
`--bank-recipe mixed_out` draws the human and ai texts uniformly from the training split and the mixed texts from the
`bank_draw` split (`data.recipe_bank_rows`), `--bank-per-origin` of each; it fails if the corpus lacks
`bank_draw`.
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import pyarrow.parquet as pq
import torch

from pathlib import Path
from typing import Any
from torch import Tensor

from .contract import reference
from .data import BANK_RECIPES, TokenStore, cluster_stratified_rows, recipe_bank_rows
from .model import StudentModel, single_text_batch


def ternary_tensors(name: str, linear: torch.nn.Module) -> dict[str, np.ndarray]:
    T, gamma = linear.ternary()  # (o, i), ()
    return {f"{name}.weight": reference.pack_ternary(T.cpu().numpy().astype(np.int8)),  # (o, i/4)
            f"{name}.scale": np.asarray([gamma.item()], dtype=np.float32)}


def model_attention(model: StudentModel) -> str:
    """The attention the model reads with: "integer" (`integer_block_attention`) or "float"."""
    return "integer" if model.integer_attention else "float"


def with_attention(model: StudentModel, attention: str) -> StudentModel:
    """Read the model's weights with `attention` ("float" or "integer") from here on, in place; the weights are the
    same in both modes, so this is the post-training switch of a release (SPEC.md, Integer attention)."""
    if attention not in reference.ATTENTION_MODES:
        raise ValueError(f"attention {attention!r} is not one of {reference.ATTENTION_MODES}")
    integer = attention == "integer"
    model.integer_attention = integer
    for layer in model.encoder.layers:
        layer.integer_attention = integer

    return model


def model_output(model: StudentModel) -> str:
    """The head's output the model was built with: "int8" (`StudentHead(int8_output=True)`) or "float"."""
    return "int8" if model.head.int8_output else "float"


def with_output(model: StudentModel, output: str) -> StudentModel:
    """Read the head's output as `output` ("float" or "int8") from here on, in place; the head's weights are the same
    in both modes, so this is the post-training switch of a release (SPEC.md, Int8 output)."""
    if output not in reference.OUTPUT_MODES:
        raise ValueError(f"output {output!r} is not one of {reference.OUTPUT_MODES}")
    model.head.int8_output = output == "int8"

    return model


def gain(norm: torch.nn.Module) -> np.ndarray:
    return norm.gain.detach().float().cpu().numpy()  # (k,)


def model_tensors(model: StudentModel) -> dict[str, np.ndarray]:
    """The release tensors of the encoder and the head, without the bank (SPEC.md, section 6)."""
    if not model.quantized:
        raise ValueError("a full-precision twin is a control; no release is made from it")
    encoder, head = model.encoder, model.head
    integers, scale = encoder.embed.quantize(encoder.embed.weight.detach().float())  # (V, e), (V, 1)
    tensors = {"embed.weight": integers.cpu().numpy().astype(np.int8), "embed.scale": scale[:, 0].cpu().numpy()}
    if encoder.proj is not None:
        tensors["embed.proj_norm"] = gain(encoder.proj.norm)
        tensors.update(ternary_tensors("embed.proj", encoder.proj.linear))
    for index, layer in enumerate(encoder.layers):
        prefix = f"layers.{index}."
        tensors[prefix + "attn_norm"] = gain(layer.attn_norm)
        for name in ("q", "k", "v"):
            tensors.update(ternary_tensors(prefix + "attn." + name, getattr(layer, name)))
        tensors[prefix + "attn_out_norm"] = gain(layer.o.norm)
        tensors.update(ternary_tensors(prefix + "attn.o", layer.o.linear))
        tensors[prefix + "ffn_norm"] = gain(layer.up.norm)
        tensors.update(ternary_tensors(prefix + "ffn.up", layer.up.linear))
        tensors[prefix + "ffn_out_norm"] = gain(layer.down.norm)
        tensors.update(ternary_tensors(prefix + "ffn.down", layer.down.linear))
    tensors["final_norm"] = gain(encoder.final_norm)
    mean, input_scale = head.input_statistics()  # (2d,) each
    tensors["head.input_mean"] = mean.detach().float().cpu().numpy()
    tensors["head.input_scale"] = input_scale.detach().float().cpu().numpy()
    for name in ("fc1", "fc2"):
        linear = getattr(head, name)
        tensors[f"head.{name}.weight"] = linear.weight.detach().half().cpu().numpy()
        tensors[f"head.{name}.bias"] = linear.bias.detach().half().cpu().numpy()
    tensors["head.pair"] = np.asarray([head.scale.item(), head.bias.item()], dtype=np.float32)

    return tensors  # name to array (SPEC.md, section 6)


BANK_FIELDS = ("source", "generator", "domain", "era", "format", "language")  # corpus columns a reading names its neighbors by


def bank_metadata(corpus: Path, rows: np.ndarray, fields: tuple[str, ...]) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    """The named corpus columns of the bank rows, dictionary-encoded: field to sorted names, and field to codes (m,)."""
    # rows: (m,) corpus ids
    from .data import parquet_files

    wanted = set(rows.tolist())
    values: dict[str, dict[int, str]] = {field: {} for field in fields}
    for path in parquet_files(corpus):
        table = pq.read_table(path, columns=["id", *fields])
        ids = table["id"].to_numpy()  # (r,)
        keep = np.flatnonzero(np.isin(ids, rows))  # (k,)
        for field in fields:
            column = table[field].take(keep).to_pylist()
            values[field].update({int(ids[index]): "" if value is None else str(value) for index, value in zip(keep, column, strict=True)})
    missing = wanted - set(values[fields[0]])
    if missing:
        raise KeyError(f"{len(missing)} bank rows are not in {corpus}")
    names = {field: sorted(set(values[field].values())) for field in fields}
    if any(len(field_names) > 65535 for field_names in names.values()):
        raise ValueError("a bank metadata field has more names than uint16 codes")
    lookup = {field: {name: code for code, name in enumerate(names[field])} for field in fields}
    codes = {field: np.asarray([lookup[field][values[field][int(row)]] for row in rows], dtype=np.uint16) for field in fields}  # each (m,)

    return names, codes  # field to names, and (m,) each by field


def bank_tensors(vectors: Tensor, origins: np.ndarray, tokens: np.ndarray, codes: dict[str, np.ndarray], output: str = "float",
                 act_eps: float = 1e-5) -> dict[str, np.ndarray]:
    """Bank vectors and their metadata, grouped by origin: (m, 2w) float16 (int8 codes in Int8 output, recovered from
    the model's unit vectors, which carry them exactly), (m,) uint8, (m,) int32, (m,) uint16 each."""
    # vectors: (m, 2w); origins, tokens: (m,); codes: (m,) each, by field
    order = np.argsort(origins, kind="stable")  # (m,)
    assert set(np.unique(origins)) == {0, 1, 2}, "the bank holds every origin"
    ordered = vectors.detach().cpu().numpy()[order]  # (m, 2w) float32
    if output == "int8":
        w = ordered.shape[1] // 2
        stored = np.concatenate([reference.quantize_output(ordered[:, :w], act_eps), reference.quantize_output(ordered[:, w:], act_eps)], axis=1)  # (m, 2w) int8
    else:
        stored = ordered.astype(np.float16)  # (m, 2w)
    tensors = {"bank.vectors": stored, "bank.origin": origins[order].astype(np.uint8), "bank.tokens": tokens[order].astype(np.int32)}
    tensors.update({f"bank.meta.{field}": field_codes[order] for field, field_codes in codes.items()})

    return tensors  # (...) each, by name


@torch.no_grad()
def text_vectors(model: StudentModel, ids: np.ndarray) -> dict[str, np.ndarray]:
    """One text through the model in eval mode: pooled (2d,), query (w,) and key (w,) as float32 NumPy."""
    # ids: (n,)
    device = next(model.parameters()).device
    pooled = model.pooled(single_text_batch(ids, model.config.block_size).to(device))  # (1, 2d)
    query_key = model.head(pooled)[0].float().cpu().numpy()  # (2w,)
    w = model.config.head_out

    return {"pooled": pooled[0].float().cpu().numpy(), "query": query_key[:w], "key": query_key[w:]}  # (2d,), (w,), (w,)


# The model against `reference.py` on the release (SPEC.md, Parity). The two round in different orders, and one
# quantized activation that lands on the other side of a half moves a short text's vectors by a step that pooling
# does not average out: on a trained checkpoint (s192_l4, a 64-wide head) queries and keys
# differed by 1e-4 at the median and by up to 2.1e-3 on texts of 16 to 128 tokens, pooled vectors by up to 1.8e-3.
# So the typical text is held tight, where a wrong weight or a wrong norm would show on every text, and the worst
# text is held to a few flips. Streaming against whole is one implementation and stays exact.
TOLERANCES = {"pooled": 1e-2, "query_key": 1e-2, "stream_score": 1e-6}  # the worst text
TYPICAL_TOLERANCES = {"pooled": 1e-3, "query_key": 5e-4}  # the median text
# In Int8 output a query or key is its codes, and a component of the head's output within rounding error of half a
# step rounds the other way in the other implementation, one code apart; the pooled vector's differences move many
# components that far at once. The check is then on the codes, in code units: the median text within one code and the
# worst within two. A wrong weight or norm moves codes by far more on every text.
INT8_TOLERANCES = {"pooled": 1e-2, "query_key": 2.0, "stream_score": 1e-6}
INT8_TYPICAL_TOLERANCES = {"pooled": 1e-3, "query_key": 1.0}


def verify_release(model: StudentModel, release: Path, texts: list[np.ndarray]) -> dict[str, float]:
    """`reference.py` on the release against the model in eval mode, and streamed against whole, on token id lists.
    Returns the worst text's errors and, as `median_*`, the median text's. With int8 output the query and key errors
    are in codes (the model's own codes are those of its unit vectors)."""
    # texts: (l_i,) each of k
    loaded = reference.ReferenceModel.load(release)
    int8 = loaded.output == "int8"
    tolerances, typical_tolerances = (INT8_TOLERANCES, INT8_TYPICAL_TOLERANCES) if int8 else (TOLERANCES, TYPICAL_TOLERANCES)
    errors: dict[str, list[float]] = {"pooled": [], "query_key": [], "stream_score": []}
    for ids in texts:
        assert len(ids) <= loaded.config.chunk_tokens, "the check reads texts of one chunk"
        mine = text_vectors(model, ids)
        whole, vectors = loaded.read_whole(ids)
        streamed, _ = loaded.read_streaming(ids, max(1, loaded.config.block_size - 1))
        chunk = vectors["chunks"][0]
        pooled_error = np.abs(chunk["pooled"] - mine["pooled"]) / (1.0 + np.abs(chunk["pooled"]))  # (2d,)
        errors["pooled"].append(float(pooled_error.max()))
        mine_query, mine_key = loaded.head_output(mine["query"], mine["key"])  # (w,) each, codes in int8 output
        errors["query_key"].append(max(float(np.abs(chunk["query"].astype(np.float64) - mine_query).max()),
                                       float(np.abs(chunk["key"].astype(np.float64) - mine_key).max())))
        errors["stream_score"].append(abs(whole["chunks"][0]["score"] - streamed["chunks"][0]["score"]))
    worst = {name: max(values) for name, values in errors.items()}
    typical = {name: float(np.median(errors[name])) for name in typical_tolerances}
    if any(worst[name] > tolerances[name] for name in worst) or any(typical[name] > typical_tolerances[name] for name in typical):
        raise AssertionError(f"the release disagrees with the model beyond {tolerances} at worst or {typical_tolerances} at the "
                             f"median: worst {worst}, median {typical}")

    return {**worst, **{f"median_{name}": value for name, value in typical.items()}}


def export_release(checkpoint_path: Path, store_directory: Path, corpus: Path, out: Path, *, bank_split: str = "bank",
                   bank_per_origin: int = 4096, calibration_split: str = "calibration", fields: tuple[str, ...] = BANK_FIELDS,
                   check_texts: int = 64, micro_tokens: int = 65536, seed: int = 0, device: torch.device | None = None,
                   attention: str | None = None, output: str | None = None, quantize_after_training: bool = False,
                   bank_recipe: str = "split") -> dict[str, Any]:
    """Write the release of a checkpoint and verify it; returns a summary. Every vector is computed in float32, the
    arithmetic the release defines, never under autocast. `attention` None releases the attention the checkpoint was
    trained with; "integer" or "float" releases that one, the bank and the calibration computed with it. `output` None
    releases the head's output the checkpoint was trained with; "int8" on a checkpoint trained without
    `--int8-output` is refused unless `quantize_after_training`. `bank_recipe` "split" draws the bank from
    `bank_split`; a `data.BANK_RECIPES` name draws it with `data.recipe_bank_rows`."""
    from .train import load_checkpoint, model_vectors

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if not model.quantized:
        raise ValueError("a full-precision twin is a control; no release is made from it")
    trained = model_attention(model)
    attention = trained if attention is None else attention
    with_attention(model, attention)
    trained_output = model_output(model)
    output = trained_output if output is None else output
    if output == "int8" and trained_output != "int8" and not quantize_after_training:
        raise ValueError("this checkpoint was trained without --int8-output, so its head never learned int8 vectors; train with the "
                         "switch, or pass quantize_after_training (--quantize-after-training) to quantize its output after the fact")
    with_output(model, output)
    model.eval()
    store = TokenStore(store_directory)
    generator = np.random.default_rng(seed)
    if bank_recipe == "split":
        bank_rows = cluster_stratified_rows(store, bank_split, bank_per_origin, generator)  # (m,)
        bank_name = bank_split
    elif bank_recipe in BANK_RECIPES:
        bank_rows = recipe_bank_rows(store, bank_recipe, bank_per_origin, seed)  # (m,)
        bank_name = f"{bank_recipe} ({', '.join(BANK_RECIPES[bank_recipe])})"
        too_long = int((store.length[bank_rows] > model.config.chunk_tokens).sum())
        if too_long:
            raise ValueError(f"{too_long} bank texts are longer than a chunk of {model.config.chunk_tokens} tokens; the bank holds texts read as one chunk")
    else:
        raise ValueError(f"no bank recipe {bank_recipe!r}; known: split, {', '.join(BANK_RECIPES)}")
    bank_vectors = model_vectors(model, store, bank_rows, micro_tokens, device, precise=True)  # (m, 2w)
    names, codes = bank_metadata(corpus, bank_rows, fields)
    tensors = {**model_tensors(model), **bank_tensors(bank_vectors, store.label[bank_rows], store.length[bank_rows], codes, output, model.config.act_eps)}
    counts = [int((store.label[bank_rows] == origin).sum()) for origin in range(3)]
    provenance = {"checkpoint": str(checkpoint_path), "step": checkpoint["step"], "run": checkpoint.get("run"),
                  "validation": checkpoint.get("validation"), "data": str(store_directory), "corpus": str(corpus), "seed": seed}
    if "integer" in (attention, trained):
        provenance["attention"] = {"release": attention, "trained": trained}
    if "int8" in (output, trained_output):
        provenance["output"] = {"release": output, "trained": trained_output}
    config_json = reference.release_config(model.config, counts, bank_name, names, None, provenance, attention, output)
    reference.write_release(out, tensors, config_json)

    # Calibration texts are read as texts of their own, so each is at most a chunk long; whole long documents are not.
    calibration_rows = np.flatnonzero(store.split_mask(calibration_split) & ~store.long_eval)  # (n_c,)
    calibration_vectors = model_vectors(model, store, calibration_rows, micro_tokens, device, precise=True).cpu().numpy()  # (n_c, 2w)
    loaded = reference.ReferenceModel.load(out)
    w = model.config.head_out
    readings = [loaded.bank_reading(*loaded.head_output(row[:w], row[w:])) for row in calibration_vectors.astype(np.float32)]
    table, explained = reference.fitted_calibration(np.asarray([reading["mean"] for reading in readings]),  # (n_c, 3)
                                                    np.asarray([reading["spread"] for reading in readings]),  # (n_c, 3)
                                                    np.asarray([reading["closeness"] for reading in readings]),  # (n_c,)
                                                    store.length[calibration_rows], store.label[calibration_rows].astype(np.int64))
    config_json["calibration"] = table
    reference.write_release(out, {**tensors, **explained}, config_json)

    validation = np.flatnonzero(store.split_mask("validation") & ~store.long_eval)  # (n_v,)
    chosen = generator.choice(validation, min(check_texts, len(validation)), replace=False)  # (k,)
    worst = verify_release(model, out, [store.ids(row) for row in chosen])

    return {"release": str(out), "bank": counts, "calibration_rows": len(calibration_rows), "parity": worst, "attention": attention,
            "output": output, "bank_recipe": bank_recipe,
            "bytes": (out / "model.safetensors").stat().st_size + (out / "config.json").stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="the pre-tokenized corpus of `litmus_student.data`")
    parser.add_argument("--corpus", type=Path, required=True, help="the corpus Parquet, for the bank's metadata")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank-split", default="bank")
    parser.add_argument("--bank-per-origin", type=int, default=4096)
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--fields", nargs="+", default=list(BANK_FIELDS), help="corpus columns of the bank's metadata")
    parser.add_argument("--check-texts", type=int, default=64, help="validation texts the reference checks")
    parser.add_argument("--micro-tokens", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", choices=reference.ATTENTION_MODES, default=None,
                        help="the release's attention; default: the checkpoint's own. `integer` on a float-trained checkpoint "
                             "releases it in SPEC.md's integer mode (format 2), bank and calibration computed that way")
    parser.add_argument("--output", choices=reference.OUTPUT_MODES, default=None,
                        help="the head's output; default: the checkpoint's own (trained with train.py --int8-output or not). `int8` is "
                             "SPEC.md's Int8 output (format 3): int8 query and key, an int8 bank, integer dot products")
    parser.add_argument("--quantize-after-training", action="store_true",
                        help="allow `--output int8` for a checkpoint trained without --int8-output (its head's output is quantized after "
                             "the fact; the provenance records it); without this such a checkpoint is refused")
    parser.add_argument("--bank-recipe", choices=["split", *BANK_RECIPES], default="split",
                        help="split: --bank-per-origin texts of each origin of --bank-split over clusters (default); mixed_out: human and "
                             "ai texts uniformly from the training split, mixed texts from the bank_draw split, --bank-per-origin each")
    arguments = parser.parse_args()
    summary = export_release(arguments.checkpoint, arguments.data, arguments.corpus, arguments.out, bank_split=arguments.bank_split,
                             bank_per_origin=arguments.bank_per_origin, calibration_split=arguments.calibration_split,
                             fields=tuple(arguments.fields), check_texts=arguments.check_texts, micro_tokens=arguments.micro_tokens,
                             seed=arguments.seed, attention=arguments.attention, output=arguments.output,
                             quantize_after_training=arguments.quantize_after_training, bank_recipe=arguments.bank_recipe)
    print(json.dumps(summary, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
