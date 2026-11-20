"""Adapters that turn open detectors into scorers (texts in, one score per text out, higher meaning more likely AI), and
the single-core cost measurement behind the accuracy against cost figure.

Each adapter follows its reference implementation: Binoculars the authors' `binoculars/metrics.py` (BSD-3-Clause),
Fast-DetectGPT the analytic criterion of `scripts/fast_detect_gpt.py` (MIT), Desklib the model class on its card, and
sequence classifiers the label index their cards or configs name. Revisions are pinned to the commits read on
2026-11-17. torch and transformers load only when an adapter is built, so the metrics and figures need neither.
"""

from __future__ import annotations

import os
import platform
import time
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from tqdm import tqdm


if TYPE_CHECKING:
    import torch


Scorer = Callable[[list[str]], np.ndarray]
FALCON = ("tiiuae/falcon-7b", "ec89142b67d748a1865ea4451372db8313ada0d8")
FALCON_INSTRUCT = ("tiiuae/falcon-7b-instruct", "8782b5c5d8c9290412416618f36a133653e85285")


@dataclass(frozen=True)
class ClassifierSpec:
    """A sequence classifier on the Hub and how its AI score is read."""

    model_id: str
    revision: str
    ai_index: int
    max_length: int = 512


CLASSIFIERS = {
    "e5-small-lora": ClassifierSpec("MayZhou/e5-small-lora-ai-generated-detector", "483fc4969592dc20e00e5130e7187b5dd25dbcc7", 1),
    "OpenAI RoBERTa-base (GPT-2)": ClassifierSpec("openai-community/roberta-base-openai-detector",
                                                  "6cba99c003b711c7fe94f8a3aa2be35a792cb6fa", 0),
    "RADAR": ClassifierSpec("TrustSafeAI/RADAR-Vicuna-7B", "4ff1f23a69a36aa1df47b0933be6279f1b896c9b", 0),
    "ModernBERT RAID+MAGE (Drayson)": ClassifierSpec("GeorgeDrayson/modernbert-ai-detection-raid-mage",
                                                     "e047d9c56f51a7e5c5ef5b0dd556f519bb9dd624", 1),
    "MAGE Longformer": ClassifierSpec("yaful/MAGE", "0d82ca0fdf6ebef5babb813cc11bd8eb2552c846", 0, 4096),
    "RoBERTa ChatGPT (HC3)": ClassifierSpec("Hello-SimpleAI/chatgpt-detector-roberta", "d2b342c61775d5dd0221808a79983ed3b86ffd86", 1),
}


def device_and_dtype(device: str) -> tuple[str, "torch.dtype"]:
    import torch

    return device, torch.bfloat16 if device.startswith("cuda") else torch.float32


class SequenceClassifierScorer:
    """Softmax probability of the AI class of a Hub sequence classifier, texts truncated to its input limit."""

    def __init__(self, spec: ClassifierSpec, device: str = "cuda") -> None:
        import torch

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.device, dtype = device_and_dtype(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(spec.model_id, revision=spec.revision,
                                                                        torch_dtype=dtype).to(self.device).eval()
        self.torch = torch

    def __call__(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.spec.max_length,
                                 return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits.float()  # (b, c)
        return self.torch.softmax(logits, dim=-1)[:, self.spec.ai_index].cpu().numpy()  # (b,)

    def tokens(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=True, max_length=self.spec.max_length)["input_ids"])


class DesklibScorer:
    """Desklib v1.01: DeBERTa-v3-large, mean pooling over the attention mask, one logit, sigmoid (the card's class)."""

    MODEL = ("desklib/ai-text-detector-v1.01", "5fdea974cd4287c61674951ec78803aa274e2fb7")

    def __init__(self, device: str = "cuda", max_length: int = 768) -> None:
        import torch

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        model_id, revision = self.MODEL
        self.device, dtype = device_and_dtype(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        self.encoder = AutoModel.from_config(config)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        state = load_file(hf_hub_download(model_id, "model.safetensors", revision=revision))
        encoder_state = {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}
        classifier_state = {key.removeprefix("classifier."): value for key, value in state.items() if key.startswith("classifier.")}
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.classifier.load_state_dict(classifier_state, strict=True)
        self.encoder.to(self.device, dtype).eval()
        self.classifier.to(self.device, dtype).eval()
        self.torch = torch

    def __call__(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            hidden = self.encoder(encoded["input_ids"], attention_mask=encoded["attention_mask"])[0]  # (b, l, d)
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (b, l, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (b, d)
            logits = self.classifier(pooled).float().squeeze(-1)  # (b,)
        return self.torch.sigmoid(logits).cpu().numpy()  # (b,)

    def tokens(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"])


def load_causal_lm(model: tuple[str, str], device: str) -> object:
    from transformers import AutoModelForCausalLM

    _, dtype = device_and_dtype(device)
    return AutoModelForCausalLM.from_pretrained(model[0], revision=model[1], torch_dtype=dtype).to(device).eval()


class BinocularsScorer:
    """Binoculars: the performer's log perplexity over the observer-performer cross-perplexity, negated so higher is AI.

    Follows the reference's arithmetic exactly, including its cross-perplexity mask over unshifted positions.
    """

    def __init__(self, observer: tuple[str, str] = FALCON, performer: tuple[str, str] = FALCON_INSTRUCT,
                 device: str = "cuda", max_tokens: int = 512) -> None:
        import torch

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(observer[0], revision=observer[1])
        performer_tokenizer = AutoTokenizer.from_pretrained(performer[0], revision=performer[1])
        assert self.tokenizer.get_vocab() == performer_tokenizer.get_vocab(), "observer and performer share a tokenizer"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.observer, self.performer = load_causal_lm(observer, device), load_causal_lm(performer, device)
        self.device, self.max_tokens, self.torch = device, max_tokens, torch

    def __call__(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        encoded = self.tokenizer(texts, return_tensors="pt", padding="longest" if len(texts) > 1 else False,
                                 truncation=True, max_length=self.max_tokens, return_token_type_ids=False).to(self.device)
        with torch.inference_mode():
            observer_logits = self.observer(**encoded).logits.float()  # (b, l, v)
            performer_logits = self.performer(**encoded).logits.float()  # (b, l, v)
        # Log perplexity of the text under the performer, over the shifted, attended positions.
        shifted = performer_logits[:, :-1]  # (b, l - 1, v)
        targets = encoded["input_ids"][:, 1:]  # (b, l - 1)
        attended = encoded["attention_mask"][:, 1:].float()  # (b, l - 1)
        token_loss = -torch.log_softmax(shifted, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (b, l - 1)
        log_ppl = (token_loss * attended).sum(1) / attended.sum(1)  # (b,)
        # Cross-perplexity: the performer's cross-entropy against the observer's distribution at every position.
        observer_probs = torch.softmax(observer_logits, dim=-1)  # (b, l, v)
        cross = -(observer_probs * torch.log_softmax(performer_logits, dim=-1)).sum(-1)  # (b, l)
        not_pad = (encoded["input_ids"] != self.tokenizer.pad_token_id).float()  # (b, l)
        log_x_ppl = (cross * not_pad).sum(1) / not_pad.sum(1)  # (b,)
        return -(log_ppl / log_x_ppl).cpu().numpy()  # (b,)

    def tokens(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=True, max_length=self.max_tokens)["input_ids"])


class FastDetectGPTScorer:
    """Fast-DetectGPT's analytic sampling discrepancy, one model for sampling and scoring unless a second is given.

    The reference scores one text at a time; here a batch is padded and every sum runs over the attended positions only,
    which gives the reference's value for each text.
    """

    def __init__(self, scoring: tuple[str, str] = FALCON_INSTRUCT, sampling: tuple[str, str] | None = None,
                 device: str = "cuda", max_tokens: int = 512) -> None:
        import torch

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(scoring[0], revision=scoring[1])
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.scoring = load_causal_lm(scoring, device)
        self.sampling = self.scoring if sampling is None else load_causal_lm(sampling, device)
        self.device, self.max_tokens, self.torch = device, max_tokens, torch

    def __call__(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens,
                                 return_token_type_ids=False).to(self.device)
        targets = encoded["input_ids"][:, 1:]  # (b, l - 1)
        attended = encoded["attention_mask"][:, 1:].float()  # (b, l - 1)
        with torch.inference_mode():
            score_logits = self.scoring(**encoded).logits[:, :-1].float()  # (b, l - 1, v)
            reference_logits = score_logits if self.sampling is self.scoring else self.sampling(**encoded).logits[:, :-1].float()
        vocab = min(score_logits.shape[-1], reference_logits.shape[-1])
        log_probs = torch.log_softmax(score_logits[..., :vocab], dim=-1)  # (b, l - 1, v)
        reference = torch.softmax(reference_logits[..., :vocab], dim=-1)  # (b, l - 1, v)
        likelihood = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (b, l - 1)
        mean = (reference * log_probs).sum(-1)  # (b, l - 1)
        variance = (reference * log_probs.square()).sum(-1) - mean.square()  # (b, l - 1)
        discrepancy = ((likelihood - mean) * attended).sum(1) / ((variance * attended).sum(1)).sqrt()  # (b,)
        # A text of one token has no position to score (0 / 0): it reads as no discrepancy, the statistic's null.
        return torch.nan_to_num(discrepancy, nan=0.0).cpu().numpy()  # (b,)

    def tokens(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=True, max_length=self.max_tokens)["input_ids"])


class CheckerScorer:
    """Litmus's predecessor, lhallee/ai-vs-human-checker, through its own package (`ai_text_checker`)."""

    def __init__(self, threads: int | None = None) -> None:
        from ai_text_checker.checker import load_checker

        self.checker = load_checker(threads=threads)

    def __call__(self, texts: list[str]) -> np.ndarray:
        try:
            return np.array([reading.score for reading in self.checker.readings(texts)])  # (b,)
        except AssertionError:  # the checker refuses a text its prose reader leaves empty (code, markup alone)
            return np.array([self.one(text) for text in texts])  # (b,)

    def one(self, text: str) -> float:
        """One text's score, 0.5 for a text the checker refuses, as Litmus scores a text with no tokens."""
        try:
            return float(self.checker.readings([text])[0].score)
        except AssertionError:
            return 0.5

    def tokens(self, text: str) -> int:
        """Tokens of the raw text in the checker's ModernBERT tokenizer (its prose reader may drop a few)."""
        return len(self.checker.tokenizer(text)["input_ids"])


class LitmusScorer:
    """Litmus as a user reads it: a text's verdict from a release, `1 - P(human)` of its calibrated probability.

    Token states come from the trained checkpoint in float32 and everything after them (head, bank, calibration,
    chunks) from the release through the student's NumPy reference, the arithmetic the engine matches. The
    checkpoint and the release folder are named by `LITMUS_CHECKPOINT` and `LITMUS_RELEASE`. A text with no tokens
    scores 0.5.
    """

    def __init__(self, device: str = "cuda") -> None:
        import torch

        from litmus_student.contract import reference
        from litmus_student.evaluate import Reader

        self.reference = reference
        release = Path(os.environ["LITMUS_RELEASE"])
        self.reader = Reader(Path(os.environ["LITMUS_CHECKPOINT"]), release, torch.device(device))
        self.tokenizer = reference.load_tokenizer()
        self.reading = reference.release_reading(release)  # the reading the release was trained in

    def __call__(self, texts: list[str]) -> np.ndarray:
        scores = np.full(len(texts), 0.5)  # (b,)
        ids = [self.reference.token_ids(self.tokenizer, text, self.reading) for text in texts]  # (n_i,) each
        kept = [index for index, each in enumerate(ids) if len(each)]
        if kept:  # every text's chunks in packed forwards, then read against the whole bank together
            scores[kept] = [verdict["probability"] for verdict in self.reader.read_many([ids[index] for index in kept])]

        return scores  # (b,)

    def tokens(self, text: str) -> int:
        return len(self.reference.token_ids(self.tokenizer, text, self.reading))


class EngineScorer:
    """Litmus through its engine as shipped (the `litmus_engine` wheel): the whole reading, windows and cells included,
    from the text, the release folder named by `LITMUS_ENGINE_RELEASE` (model, config and tokenizer). A text with no
    tokens scores 0.5, as `LitmusScorer` scores it."""

    def __init__(self) -> None:
        import litmus_engine

        self.engine = litmus_engine.Engine.from_path(os.environ["LITMUS_ENGINE_RELEASE"])

    def __call__(self, texts: list[str]) -> np.ndarray:
        scores = np.full(len(texts), 0.5)  # (b,)
        for index, text in enumerate(texts):
            if self.engine.count_tokens(text):
                scores[index] = self.engine.score(text)["verdict"]["probability"]
        return scores  # (b,)

    def tokens(self, text: str) -> int:
        return self.engine.count_tokens(text)


def build_scorer(name: str, device: str = "cuda") -> Scorer:
    """The scorer of a detector named in `detectors.csv`'s adapter column or the classifier table."""
    if name in CLASSIFIERS:
        return SequenceClassifierScorer(CLASSIFIERS[name], device)
    builders: dict[str, Callable[[], Scorer]] = {
        "Binoculars": lambda: BinocularsScorer(device=device),
        "Fast-DetectGPT": lambda: FastDetectGPTScorer(device=device),
        "Desklib v1.01": lambda: DesklibScorer(device=device),
        # One thread unless CHECKER_THREADS says more: the cost comparison is single-thread, bulk scoring is not.
        "ai-vs-human-checker": lambda: CheckerScorer(threads=int(os.environ["CHECKER_THREADS"]) if "CHECKER_THREADS" in os.environ else None),
        "Litmus": lambda: LitmusScorer(device=device),
        "Litmus engine": EngineScorer,
    }
    if name not in builders:
        raise KeyError(f"no adapter for {name!r}; known: {sorted([*CLASSIFIERS, *builders])}")
    return builders[name]()


class CountingScorer(Protocol):
    """A scorer that also says how many tokens of a text it reads, in its own tokenizer and truncation."""

    def __call__(self, texts: list[str]) -> np.ndarray: ...

    def tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class CpuCost:
    """One detector's single-core cost on a fixed set of texts: tokens in the detector's own tokenizer as it reads
    them (after its truncation), words split on whitespace."""

    detector: str
    texts: int
    words: int
    tokens: int
    seconds: float
    ms_per_1k_words: float
    words_per_second: float
    tokens_per_second: float
    peak_rss_mb: float
    threads: int
    machine: str


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (Linux and macOS through `resource`, Windows through psutil)."""
    try:
        import resource
    except ImportError:
        import psutil

        return psutil.Process().memory_info().peak_wset / 2**20
    scale = 1 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**20


# An Arm core names itself in /proc/cpuinfo only by implementer and part number; lscpu's names of the server cores.
ARM_PARTS = {"0xd0c": "Neoverse-N1", "0xd40": "Neoverse-V1", "0xd49": "Neoverse-N2", "0xd4f": "Neoverse-V2"}


def processor_name(cpuinfo: Path = Path("/proc/cpuinfo")) -> str:
    """The CPU's model name (Linux `/proc/cpuinfo`, by part number for an Arm core), else what `platform` knows."""
    if cpuinfo.exists():
        lines = cpuinfo.read_text().splitlines()
        for line in lines:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
        fields = {key.strip(): value.strip() for key, value in (line.split(":", 1) for line in lines if ":" in line)}
        if fields.get("CPU implementer") == "0x41":
            part = fields.get("CPU part", "")
            return f"Arm {ARM_PARTS.get(part, f'CPU part {part}')}"
    return platform.processor() or platform.machine()


def measure_cpu_cost(detector: str, scorer: CountingScorer, texts: list[str], *, batch_size: int = 1) -> CpuCost:
    """Wall time of scoring `texts` on one thread, after one warm-up text; run each detector in a fresh process so the
    peak memory is its own. Cost is per 1,000 whitespace words, a unit no detector's tokenizer favors, and per token
    of the detector's own tokenizer."""
    import torch

    torch.set_num_threads(1)
    assert os.environ.get("OMP_NUM_THREADS") == "1", "set OMP_NUM_THREADS=1 before Python starts"
    scorer(texts[:1])
    start = time.perf_counter()
    for begin in tqdm(range(0, len(texts), batch_size), desc=f"{detector} cost", mininterval=30):
        scorer(texts[begin:begin + batch_size])
    seconds = time.perf_counter() - start
    words = sum(len(text.split()) for text in texts)
    tokens = sum(scorer.tokens(text) for text in texts)
    return CpuCost(detector, len(texts), words, tokens, seconds, 1000 * 1000 * seconds / words, words / seconds,
                   tokens / seconds, peak_rss_mb(), 1, f"{processor_name()} / {platform.system()}")
