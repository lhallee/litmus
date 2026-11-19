"""ModernBERT's tokenizer, counted without special tokens, and the places a text may be cut.

Each worker process loads the tokenizer once and encodes in batches. Rust threads inside a worker are turned
off (`TOKENIZERS_PARALLELISM=false`, set by the pipeline) because the pipeline already runs a process per core.
"""

from __future__ import annotations

import re
import numpy as np

from functools import cache
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .config import TOKENIZER_REPO, TOKENIZER_REVISION
from .formats import cuts_at_lines_only


# A sentence starts after a sentence mark, its closing quotes or brackets, and a space; in Chinese and Japanese
# after a full-width mark with no space; and every non-blank line starts one.
SPACED_SENTENCE_PATTERN = re.compile(r"(?<=[.!?…])[\"'”’)\]」』]*[ \t]+(?=\S)")
FULL_WIDTH_SENTENCE_PATTERN = re.compile(r"(?<=[。！？])[」』）”’]*(?=[^\s」』）”’])")
LINE_START_PATTERN = re.compile(r"\n+(?=[^\n])")


def ensure_tokenizer() -> Path:
    """The tokenizer file at its pinned revision, from the Hub cache."""
    return Path(hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION))


@cache
def tokenizer(path: str) -> Tokenizer:
    loaded = Tokenizer.from_file(path)
    loaded.no_truncation()
    loaded.no_padding()

    return loaded


def token_counts(texts: list[str], path: str) -> list[int]:
    """Tokens in each text, without special tokens."""
    return [len(encoding) for encoding in tokenizer(path).encode_batch_fast(texts, add_special_tokens=False)]


def cut_starts(text: str, format_name: str) -> list[int]:
    """Character offsets where a passage may start: 0, each line start, and in prose each sentence start."""
    starts = {0}
    starts.update(match.end() for match in LINE_START_PATTERN.finditer(text))
    if not cuts_at_lines_only(format_name):
        starts.update(match.end() for match in SPACED_SENTENCE_PATTERN.finditer(text))
        starts.update(match.end() for match in FULL_WIDTH_SENTENCE_PATTERN.finditer(text))

    return sorted(start for start in starts if start < len(text))


def encode_with_cuts(texts: list[str], formats: list[str], path: str) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Each text's token count, its cut starts in characters, and the token index at each cut start.

    The token index at a start is the number of tokens that begin before it, from the token offsets, so a
    span between two starts has about the difference in tokens; the exact count of a cut passage is taken
    later by encoding the passage itself.
    """
    encodings = tokenizer(path).encode_batch(texts, add_special_tokens=False)
    totals: list[int] = []
    all_chars: list[list[int]] = []
    all_tokens: list[list[int]] = []
    for text, format_name, encoding in zip(texts, formats, encodings, strict=True):
        starts = cut_starts(text, format_name)
        token_starts = np.fromiter((offset[0] for offset in encoding.offsets), dtype=np.int64, count=len(encoding))  # (t,)
        cut_chars = np.array(starts, dtype=np.int64)  # (s,)
        cut_tokens = np.searchsorted(token_starts, cut_chars, side="left")  # (s,)
        totals.append(len(encoding))
        all_chars.append(starts)
        all_tokens.append(cut_tokens.tolist())

    return totals, all_chars, all_tokens
