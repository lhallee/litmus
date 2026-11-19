"""The fixed numbers of a corpus build: tokenizer, language model, lengths, caps and balance.

Every stage reads its limits from here, so a reader can see in one place what the build keeps and why.
"""

from __future__ import annotations

from dataclasses import dataclass


ORIGINS = ("human", "ai", "mixed")
# A benchmark whose labels are hidden gives `unlabeled` rows: they hold copies out of training and are not
# written, so the corpus's labels stay 0, 1 and 2.
UNLABELED = "unlabeled"
ORIGIN_LABELS = {"human": 0, "ai": 1, "mixed": 2, UNLABELED: 3}
LABEL_ORIGINS = {label: origin for origin, label in ORIGIN_LABELS.items()}
FORMATS = ("plain", "markdown", "list", "table", "code", "math", "chat")

FIRST_CORPUS_REPO = "lhallee/many-generator-detection-corpus"
FIRST_CORPUS_REVISION = "7e7c9706f702dc4665cd607a75bf65fe65a977c4"  # main on 2026-11-17
FIRST_CORPUS_SPLITS = ("train", "validation", "test", "unused")  # its own split is redone, so every split is read

TOKENIZER_REPO = "answerdotai/ModernBERT-large"
TOKENIZER_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"  # main on 2026-11-17

# fastText's lid.176 (Joulin et al. 2016, 176 languages, ISO 639-1 labels where one exists). Pinned by hash,
# since the file is served without revisions.
LANGUAGE_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LANGUAGE_MODEL_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
LANGUAGE_CHARACTERS = 2000  # the detector reads the opening this long, which is enough and bounds its cost
MIN_LANGUAGE_PROBABILITY = 0.5  # below this a text's language is `und`
# Languages that are their own balance group; every other language is `other`. These are the most common
# languages in public detection data; a rarer language has too few rows of either class to balance alone.
LANGUAGE_GROUPS = frozenset({"en", "zh", "es", "fr", "de", "ru", "ja", "ko", "pt", "it", "ar", "nl", "id", "vi", "tr", "pl"})

MIN_TOKENS = 8
# The model reads at most 8,192 tokens at once: a longer document is read in chunks, each
# its own text, so no training row is longer. A whole document of up to 131,072 tokens is kept only to evaluate
# that chunked reading (`long_eval`).
MAX_TOKENS = 8_192
LONG_MAX_TOKENS = 131_072
LENGTH_BINS = 10  # [8, 16), [16, 32), ..., [2,048, 4,096), [4,096, 8,192]
LONG_BINS = 4  # (8,192, 16,384), [16,384, 32,768), [32,768, 65,536), [65,536, 131,072]
LENGTH_BIN_LABELS = tuple(
    f"{MIN_TOKENS * 2**index}-{MAX_TOKENS if index == LENGTH_BINS - 1 else MIN_TOKENS * 2 ** (index + 1) - 1}" for index in range(LENGTH_BINS)
)
LONG_BIN_LABELS = tuple(
    f"{MAX_TOKENS + 1 if index == 0 else MAX_TOKENS * 2**index}-{LONG_MAX_TOKENS if index == LONG_BINS - 1 else MAX_TOKENS * 2 ** (index + 1) - 1}"
    for index in range(LONG_BINS)
)


@dataclass(frozen=True)
class BuildConfig:
    """What the build keeps. The defaults are the first corpus's caps, carried over."""

    max_sub_passages: int = 4  # per text, beside the whole text
    ai_cap: int = 150_000  # rows per (source, generator) group
    legacy_ai_cap: int = 40_000  # the same, for a generator of 2023 or earlier
    human_cap: int = 600_000  # rows per (source, domain) group
    mixed_cap: int = 150_000  # rows per (source, mix) group
    # Texts kept per group before passages are cut, as a multiple of the group's row cap. A text gives at
    # least one row, so a group still fills its cap, and the build does not tokenize text it will throw away.
    precap_factor: float = 1.0
    balance_ratio: float = 1.0  # the larger origin of a cell keeps at most this many times the smaller's rows
    thin_share: float = 0.25  # a cell is thin when its smaller origin has less than this share of the larger
    thin_rows: int = 100  # or fewer rows than this
    # The token budget of new training rows (every row but the first corpus's, a benchmark's and a `long_eval`
    # one), counting each row's full length, at most 8,192 tokens. Balance enforces it bin by bin: each length
    # bin may spend `bin_token_shares[bin]` of it, so the longest bin has an explicit share and its few long
    # rows cannot spend the short bins' tokens.
    new_token_budget: float = 3e9
    bin_token_shares: tuple[float, ...] = (0.1,) * LENGTH_BINS
    # Whole documents of 8,193 to 131,072 tokens, kept to evaluate the chunked reading: at most this many per
    # origin, spread evenly over the long bins, and matched between human and model text bin by bin.
    long_eval_cap: int = 3_000
    # Reading is planned at this multiple of the budget (3 to 5 times what survives balance), a source's tokens
    # estimated from its catalog bytes at this rate (parquet bytes over tokens, as SOURCES.md estimates), each
    # text counted at its full length, and split among the origins so.
    read_multiple: float = 4.0
    catalog_bytes_per_token: float = 2.0
    read_origin_shares: tuple[float, ...] = (0.45, 0.45, 0.10)  # human, ai, mixed
    samples_per_cell: int = 20
    shard_rows: int = 50_000
    shard_bytes: int = 256 * 2**20
    read_batch_rows: int = 1_000
    max_read_parts: int = 8  # a source's files are read by at most this many workers at once


def length_bin(tokens: int) -> int:
    """The length bin of a training row: 0 for 8 to 15 tokens, doubling up to 9 for 4,096 to 8,192."""
    assert MIN_TOKENS <= tokens <= MAX_TOKENS, f"{tokens} tokens is outside the binned range"

    return min(tokens.bit_length() - 4, LENGTH_BINS - 1)


def long_bin(tokens: int) -> int:
    """The bin of a long document: 0 for 8,193 to 16,383 tokens, doubling up to 3 for 65,536 to 131,072."""
    assert MAX_TOKENS < tokens <= LONG_MAX_TOKENS, f"{tokens} tokens is not a long document"

    return min(tokens.bit_length() - 14, LONG_BINS - 1)


def language_group(language: str) -> str:
    return language if language in LANGUAGE_GROUPS else "other"
