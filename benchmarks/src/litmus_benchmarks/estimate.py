"""Single-core throughput of detectors we cannot run, estimated from a measured detector of the same architecture.

A RAID leaderboard entry is estimated only when its model is public on the Hub, so its architecture (`model_type`)
and size are read from its own files, never assumed. The rule: per-token compute of an encoder or decoder is about
twice its non-embedding parameters (embedding lookups cost nothing), so an entry's throughput is the measured
throughput of the same-architecture detector closest to it in non-embedding parameters, times the ratio of their
non-embedding parameters. An entry more than `MAX_SCALE` times larger or smaller than every measured model of its
architecture, or of an architecture none of the measured detectors shares, is left out rather than guessed.

    python -m litmus_benchmarks architectures --out architectures.csv
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass

from .detectors import CLASSIFIERS, DesklibScorer


EMBEDDING_MARKERS = ("embeddings.", "embed_tokens.", "tok_embeddings.", "wte.", "wpe.")
MAX_SCALE = 10.0
MEASURED_REPOS = {name: (spec.model_id, spec.revision) for name, spec in CLASSIFIERS.items()} | {
    "Desklib v1.01": DesklibScorer.MODEL}
# RAID leaderboard entries quoted in published.csv whose model the leaderboard links on the Hub, as (repo, folder of
# its config and weights). Entries without a public model (every commercial API but GPTTrace, Desklib v1, GLTR and
# LLMDet as submitted) have no estimate.
LEADERBOARD_REPOS = {
    "OpenAI RoBERTa-large (GPT-2)": ("openai-community/roberta-large-openai-detector", ""),
    "SuperAnnotate": ("SuperAnnotate/ai-detector", ""),
    "BERT-tiny-4M": ("ShantanuT01/BERT-tiny-RAID", ""),
    "TMR (Target Mining RoBERTa)": ("Oxidane/tmr-ai-text-detector", ""),
    "Tabularis ModernBERT R1.1": ("tabularisai/ai-text-detection", ""),
    "DeBERTa-ConPara": ("SES-Lab-OTH/deberta-conpara", ""),
    "GPTTrace Ettin-1B": ("nakas/gpttrace-v8s-1b", "model"),
}
# Models whose own files do not name their architecture, counted as the base model their card or config names (the
# small classification heads they add are left out): SuperAnnotate's config field `pretrain_checkpoint`, and
# DeBERTa-ConPara's card (`base_model`; weights only in a pickled state dict).
BASE_MODELS = {"SuperAnnotate": "FacebookAI/roberta-large", "DeBERTa-ConPara": "microsoft/deberta-v3-large"}


@dataclass(frozen=True)
class Architecture:
    """A Hub model's architecture and size, read from its config and safetensors header."""

    detector: str
    repo: str
    revision: str
    model_type: str
    parameters: int
    non_embedding_parameters: int
    measured: bool


def tensor_sizes(repo: str, revision: str, folder: str, config: dict[str, object]) -> dict[str, int]:
    """Parameter count of every named tensor: from the safetensors header when the repo has one (no weights are
    downloaded), else from the architecture its config builds on PyTorch's meta device."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = [sibling.rfilename for sibling in api.model_info(repo, revision=revision).siblings]
    weights = f"{folder}/model.safetensors" if folder else "model.safetensors"
    if weights in files:
        header = api.parse_safetensors_file_metadata(repo, weights, revision=revision)
        return {name: int(np.prod(tensor.shape)) for name, tensor in header.tensors.items()}
    import torch

    from transformers import AutoConfig, AutoModel

    # The label maps of old configs fail the current config's validation and size only the head the encoder leaves out.
    encoder = {key: value for key, value in config.items() if key not in ("id2label", "label2id", "num_labels")}
    with torch.device("meta"):
        model = AutoModel.from_config(AutoConfig.for_model(**encoder))
    return {name: parameter.numel() for name, parameter in model.named_parameters()}


def architecture(detector: str, repo: str, revision: str | None, *, measured: bool, folder: str = "") -> Architecture:
    """The model's `model_type` from its config and its parameter counts, without downloading its weights."""
    from huggingface_hub import HfApi, hf_hub_download

    counted = BASE_MODELS.get(detector, repo)
    revision = revision or HfApi().model_info(repo).sha
    counted_revision = revision if counted == repo else HfApi().model_info(counted).sha
    config_file = f"{folder}/config.json" if folder and counted == repo else "config.json"
    with open(hf_hub_download(counted, config_file, revision=counted_revision), encoding="utf-8") as handle:
        config = json.load(handle)
    sizes = tensor_sizes(counted, counted_revision, folder if counted == repo else "", config)
    embedding = sum(size for name, size in sizes.items() if any(marker in name for marker in EMBEDDING_MARKERS))
    total = sum(sizes.values())
    return Architecture(detector, repo, revision, str(config["model_type"]), total, total - embedding, measured)


def architectures() -> pd.DataFrame:
    """Every measured classifier and every leaderboard entry with a public model, one row each."""
    rows = [architecture(name, repo, revision, measured=True) for name, (repo, revision) in MEASURED_REPOS.items()]
    rows += [architecture(name, repo, None, measured=False, folder=folder) for name, (repo, folder) in LEADERBOARD_REPOS.items()]
    return pd.DataFrame([asdict(row) for row in rows])


def estimate_throughput(models: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Estimated words and tokens a second of every unmeasured model, scaled from its nearest measured relative.

    `models` is `architectures()`'s table and `costs` the measured cost rows (detector, words_per_second,
    tokens_per_second). Returns one row per unmeasured model, with `estimated` true and the anchor and ratio named,
    or with `excluded` giving why it has no estimate.
    """
    measured = models[models["measured"]].merge(costs[["detector", "words_per_second", "tokens_per_second"]], on="detector")
    rows = []
    for _, target in models[~models["measured"]].iterrows():
        family = measured[measured["model_type"] == target["model_type"]]
        row = {"detector": target["detector"], "repo": target["repo"], "revision": target["revision"],
               "model_type": target["model_type"], "non_embedding_parameters": int(target["non_embedding_parameters"])}
        if family.empty:
            rows.append(row | {"excluded": f"no measured {target['model_type']} model"})
            continue
        ratios = family["non_embedding_parameters"].to_numpy() / target["non_embedding_parameters"]  # (m,)
        nearest = int(np.argmin(np.abs(np.log(ratios))))
        anchor, ratio = family.iloc[nearest], float(ratios[nearest])
        if not 1 / MAX_SCALE <= ratio <= MAX_SCALE:
            size = f"{1 / ratio:.0f} times larger" if ratio < 1 else f"{ratio:.0f} times smaller"
            rows.append(row | {"excluded": f"{size} than {anchor['detector']}, the nearest measured {target['model_type']}"})
            continue
        rows.append(row | {"anchor": anchor["detector"], "ratio": ratio, "words_per_second": anchor["words_per_second"] * ratio,
                           "tokens_per_second": anchor["tokens_per_second"] * ratio, "excluded": ""})
    return pd.DataFrame(rows, columns=["detector", "repo", "revision", "model_type", "non_embedding_parameters", "anchor",
                                       "ratio", "words_per_second", "tokens_per_second", "excluded"])
