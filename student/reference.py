"""NumPy reference of the Litmus student's inference, the answer the Rust engine reproduces (SPEC.md).

It loads a release (`model.safetensors` and `config.json`), reads a text into token ids, cuts them into chunks of at
most `chunk_tokens`, runs the ternary encoder on each chunk as a text of its own (at once or streamed in pieces),
pools each chunk and each heat-map window from running totals, applies the head, scores the results against the
bank, and combines the chunks into the document's verdict: the full reading of SPEC.md, uncertainty and its
explanation included. It also writes random releases and the engine's fixtures. It needs NumPy and `tokenizers`
(the safetensors format is read and written here directly) and the teacher's `litmus_teacher.calibrate`, NumPy only
too, which combines a document's chunks and fits calibration tables.

Required Notice: Copyright Logan Hallee (https://github.com/lhallee/litmus). Licensed under PolyForm Noncommercial
1.0.0 (`../LICENSE`).

    python reference.py init --config s192_l4 --out DIR [--seed 0]
    python reference.py fixtures --out DIR
    python reference.py score --release DIR --text FILE [--feed N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import unicodedata
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict


FORMAT = "litmus-student"
FORMAT_VERSION = 1
INTEGER_ATTENTION_FORMAT_VERSION = 2  # format 1 plus `model.attention` (SPEC.md, Integer attention); float releases stay format 1
INT8_OUTPUT_FORMAT_VERSION = 3  # format 2 plus `model.output: "int8"` (SPEC.md, Int8 output): int8 query and key, int8 bank
ATTENTION_MODES = ("float", "integer")
OUTPUT_MODES = ("float", "int8")  # the head's query and key: float32 unit vectors, or int8 codes
ATTENTION_WEIGHT_LEVELS = 127  # integer attention's weights are whole 127ths, as in model.py's prototype
OUTPUT_CODE_MAX = 127  # an int8 output code lies in [-127, 127]; -128 is never written and fails a load
LICENSE = "PolyForm-Noncommercial-1.0.0"  # Required Notice: Copyright Logan Hallee (https://github.com/lhallee/litmus)
TOKENIZER_REPOSITORY = "answerdotai/ModernBERT-large"
TOKENIZER_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
TOKENIZER_SHA256 = "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1"
ORIGINS = ("human", "ai", "mixed")
CONFIGS_FILE = Path(__file__).with_name("configs.json")
CALIBRATION_FORMAT = "litmus-calibration"  # the teacher's table (calibrate.py), embedded in config.json
CALIBRATION_VERSION = 1
TOP_K = 16  # closeness: the mean of this many highest cosines, the teacher's TOP_K
STRIP_LEVEL = 0.10  # the conformal set whose band the litmus strip draws
MIXED_LEVEL = 0.05  # the family level of the mixed-document signal over a document's chunks
LOG_FLOOR = 1e-12
HEAD_NORM_EPS = 1e-8
NEAREST = 5  # bank entries a reading names
RANDOM_BANK_FIELDS = {"source": ["synthetic-a", "synthetic-b", "synthetic-c"], "generator": ["", "model-x", "model-y"],
                      "domain": ["news", "essays", "code"], "era": ["human", "legacy", "recent"]}
CLAMP_DEVIATIONS = 10.0  # the default head clamp, as the teacher's head and the probe convention have it
# The corpus's standard reading (`../corpus/src/litmus_corpus/standardize.py`), its RE2 patterns in Python's syntax.
CONTROL_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad\u180e\u200b\u200e\u200f\u202a-\u202e"
    r"\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb]"
)
SPACE_SEPARATORS = r" \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000"  # Unicode category Zs, in a character class
# A run of blanks is tried only from its start, so a line holding a long run (code output padded with spaces) reads in
# linear time; tried from inside the run as well, the same matches cost the square of the run's length.
TRAILING_PATTERN = re.compile(rf"(?<![\t{SPACE_SEPARATORS}])[\t{SPACE_SEPARATORS}]+(\n|\Z)")
TEXT_END_PATTERN = re.compile(r"(?<![\t\n\f\r ])[\t\n\f\r ]+\Z")  # RE2's \s
LEADING_BLANK_PATTERN = re.compile(rf"\A(?:[\t{SPACE_SEPARATORS}]*\n)+")
BLANK_RUN_PATTERN = re.compile(r"\n{4,}")
SAFETENSORS_DTYPES: dict[str, np.dtype] = {
    "I8": np.dtype("<i1"), "U8": np.dtype("<u1"), "F16": np.dtype("<f2"), "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"), "U16": np.dtype("<u2"), "I32": np.dtype("<i4"), "I64": np.dtype("<i8"),
}
F32 = np.float32


# ---------------------------------------------------------------- configuration


@dataclass(frozen=True)
class StudentConfig:
    """Every hyperparameter of the arithmetic; `configs.json` names the sweep."""

    vocab_size: int  # V
    embed_width: int  # e
    width: int  # d
    depth: int  # L
    heads: int  # h
    ffn_width: int  # f
    block_size: int  # W
    head_hidden: int  # hh
    head_out: int  # w, the width of the query and of the key
    head_clamp: float = CLAMP_DEVIATIONS  # standardized head inputs are clamped to +-head_clamp
    chunk_tokens: int = 8192  # C_max: the most tokens read as one text; a longer text is read in chunks
    window_width: int = 512  # the heat map's windows
    window_stride: int = 128  # and their stride, which divides window_width
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    act_eps: float = 1e-5
    weight_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError(f"width {self.width} must split into {self.heads} heads of even width")
        if self.block_size % 2:
            raise ValueError("block_size must be even")
        if any(width % 4 for width in (self.embed_width, self.width, self.ffn_width)):
            raise ValueError("every ternary input width must be a multiple of 4 for 2-bit packing")
        if self.window_stride <= 0 or self.window_width % self.window_stride or not self.window_width <= self.chunk_tokens:
            raise ValueError("window_stride must divide window_width, which is at most chunk_tokens")

    @property
    def head_width(self) -> int:
        return self.width // self.heads  # d_h

    @property
    def projected(self) -> bool:
        return self.embed_width != self.width

    def shift(self, layer: int) -> int:
        """The block offset of a layer: 0 on even layers, W/2 on odd ones."""
        return 0 if layer % 2 == 0 else self.block_size // 2

    def to_json(self) -> dict[str, Any]:
        return {"vocab_size": self.vocab_size, "embed_width": self.embed_width, "width": self.width, "depth": self.depth,
                "heads": self.heads, "ffn_width": self.ffn_width, "block_size": self.block_size,
                "rope_base": self.rope_base, "norm_eps": self.norm_eps, "act_eps": self.act_eps,
                "weight_eps": self.weight_eps}

    @classmethod
    def from_release(cls, config: dict[str, Any]) -> StudentConfig:
        reading = config["reading"]
        # the attention is `release_attention`'s and the output `release_output`'s
        model = {key: value for key, value in config["model"].items() if key not in ("attention", "output")}
        return cls(**model, head_hidden=config["head"]["hidden"], head_out=config["head"]["out_width"], head_clamp=config["head"]["clamp"],
                   chunk_tokens=reading["chunk_tokens"], window_width=reading["window_width"], window_stride=reading["window_stride"])


def named_config(name: str) -> StudentConfig:
    """A config of `configs.json` by name (`tiny`, `s192_l4`, ...); its entries override the fixed ones."""
    table = json.loads(CONFIGS_FILE.read_text(encoding="utf-8"))
    if name not in table["configs"]:
        raise KeyError(f"no config {name!r}; known: {sorted(table['configs'])}")

    return StudentConfig(**{**table["fixed"], **table["configs"][name]})


def config_names() -> list[str]:
    return sorted(json.loads(CONFIGS_FILE.read_text(encoding="utf-8"))["configs"])


# ---------------------------------------------------------------- safetensors


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write tensors in the safetensors format, in name order, atomically."""
    # tensors: (...) each, by name
    codes = {dtype: code for code, dtype in SAFETENSORS_DTYPES.items()}
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name])
        dtype = array.dtype.newbyteorder("<")
        if dtype not in codes:
            raise TypeError(f"{name}: dtype {array.dtype} has no safetensors code here")
        blob = array.astype(dtype, copy=False).tobytes()
        header[name] = {"dtype": codes[dtype], "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)

    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for blob in blobs:
            stream.write(blob)
    os.replace(temporary, path)


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Read every tensor of a safetensors file into NumPy arrays (copies, little-endian dtypes)."""
    raw = path.read_bytes()
    (header_length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_length].decode("utf-8"))
    body = memoryview(raw)[8 + header_length :]
    tensors: dict[str, np.ndarray] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        dtype = SAFETENSORS_DTYPES[entry["dtype"]]
        array = np.frombuffer(body[begin:end], dtype=dtype).reshape(entry["shape"]).copy()
        tensors[name] = array

    return tensors  # (...) each, by name, as the header gives


# ---------------------------------------------------------------- ternary packing


def ternarize(W: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.float32]:
    """Absmean ternarization (BitNet b1.58, arXiv 2402.17764, equations 1 to 3): (o, i) float to (o, i) int8, gamma."""
    # W: (o, i)
    W = W.astype(F32)  # (o, i)
    gamma = F32(max(float(np.mean(np.abs(W), dtype=F32)), eps))
    T = np.clip(np.rint(W * (F32(1.0) / gamma)), -1, 1).astype(np.int8)  # (o, i)

    return T, gamma  # (o, i), ()


def pack_ternary(T: np.ndarray) -> np.ndarray:
    """Ternary values to 2-bit codes t + 1, four per byte, lowest bits first: (o, i) int8 to (o, i/4) uint8."""
    # T: (o, i) in {-1, 0, 1}
    rows, columns = T.shape
    assert columns % 4 == 0, "the packed input width must be a multiple of 4"
    assert np.isin(T, (-1, 0, 1)).all(), "ternary values only"
    codes = (T.astype(np.int16) + 1).astype(np.uint8).reshape(rows, columns // 4, 4)  # (o, i/4, 4)

    return codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)  # (o, i/4)


def unpack_ternary(packed: np.ndarray) -> np.ndarray:
    """2-bit codes back to ternary values: (o, i/4) uint8 to (o, i) int8. Code 3 fails."""
    # packed: (o, i/4)
    rows = packed.shape[0]
    codes = (packed[..., None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3  # (o, i/4, 4)
    if (codes == 3).any():
        raise ValueError("packed ternary weights hold the invalid code 3")

    return codes.reshape(rows, -1).astype(np.int8) - 1  # (o, i)


def quantize_embedding(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row absmax int8 of the embedding table: (V, e) float to (V, e) int8 and (V,) float32 scales."""
    # E: (V, e)
    E = E.astype(F32)  # (V, e)
    scale = (np.maximum(np.abs(E).max(axis=1), F32(1e-5)) / F32(127.0)).astype(F32)  # (V,)
    E_q = np.clip(np.rint(E / scale[:, None]), -127, 127).astype(np.int8)  # (V, e)

    return E_q, scale  # (V, e), (V,)


# ---------------------------------------------------------------- text


def standard_reading(text: str) -> str:
    """The corpus's standard reading (SPEC.md, section 1.1)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = CONTROL_PATTERN.sub("", text)
    text = TRAILING_PATTERN.sub(r"\1", text)
    text = TEXT_END_PATTERN.sub("", text)
    text = LEADING_BLANK_PATTERN.sub("", text)

    return BLANK_RUN_PATTERN.sub("\n\n\n", text)


# ---------------------------------------------------------------- canonical reading
#
# The standard reading with every trace of how a text was displayed or collected taken out, so that a model trained and
# read through it cannot use them: HTML entities decoded; typographic quotes, ellipses and unusual spaces as typed on a
# keyboard; in prose, dashes as a spaced hyphen and runs of spaces as one; hard-wrapped prose paragraphs joined onto
# one line; and Markdown rendered as a reader copies it from a rendered page. Code (inside fences or code spans, or a
# line with code's signs) is left as written. Litmus 1.0.0 read every one of these as evidence of origin, by up to 69
# points of the share it called not human (experiments/tooling/quill/2026-11-19_confounders).
#
# Nothing here depends on a Unicode version or on a regex engine's classes, so that another implementation (the
# engine's) gives the same bytes: every pattern reads `\s` and `\d` in ASCII alone, as its case folding (`re.ASCII`); a
# word character is an ASCII letter, digit or underscore, or any non-ASCII character (`WORD`, written `\w` inside a
# class); a letter is an ASCII letter or any non-ASCII character; and within a line, once its typography is typed, the
# only white space is space, tab and a line break a `<br>` left (the standard reading deleted the other controls,
# `TYPED` made every other space separator a space, and `SEPARATORS` broke lines at U+2028 and U+2029).

CANONICAL_READING = "canonical-1"
READINGS = ("standard", "canonical-1")
CANONICAL_PASSES = 8  # every text of a 300,000-text pilot reached its fixed point within this many
WORD = "0-9A-Za-z_\x80-\U0010ffff"  # the word characters, as the inside of a character class
BLANK = " \t\n"  # the white space a line can hold


def compiled(source: str, flags: int = 0) -> re.Pattern[str]:
    r"""A pattern of the canonical reading: `re.ASCII`, and every `\w` (written only inside a class) the `WORD` set."""
    return re.compile(source.replace(r"\w", WORD), re.ASCII | flags)


FENCE_LINE = compiled(r"^[ \t]*(```|~~~)")
RULE_LINE = compiled(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
SETEXT_LINE = compiled(r"^[ \t]{0,3}=+[ \t]*$")
QUOTE_MARKS = compiled(r"^([ \t]{0,3}>[ \t]?)+")
ORDERED_MARK = compiled(r"^[ \t]*(\d{1,3}[.)][ \t]+)(?=\S)")
# Each run of blanks has one way to match, so a line of long runs is read in linear time.
TABLE_SEPARATOR_LINE = compiled(r"^[ \t]*(?:\|[ \t]*)?:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)+(?:\|[ \t]*)?$")
TABLE_ROW_LINE = compiled(r"^[ \t]*\|.*\|[ \t]*$")
# Inline marks. A span's contents never run past the next mark that could open another (a bracket, a pair of stars),
# so no line, however written, takes more than linear time: an unclosed opener stops its search where the next begins.
COMMENT_LINE = compiled(r"^[ \t]*[#*]")  # a comment in code (`# note`, ` * @param`), or a Markdown heading or bullet
QUOTE_PREFIX = compiled(r"(?:[ \t]{0,3}>[ \t]?)+")  # the block marks as `without_marks` finds them, at a position
HEADING_OPEN = compiled(r"[ \t]{0,3}#{1,6}[ \t]+")
BULLET_PREFIX = compiled(r"[ \t]*[-*+•][ \t]+(?=\S)")
CHECKBOX_PREFIX = compiled(r"\[[ xX]\][ \t]+")
CODE_SPAN = compiled(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")  # a run of backticks closed by the next run of its length
IMAGE = compiled(r"!\[([^\[\]\n]*)\]\([^)\[\]\s]*(?:[ \t]+\"[^\"\[\]\n]*\")?\)")
LINK = compiled(r"\[([^\[\]\n]+)\]\([^)\[\]\s]*(?:[ \t]+\"[^\"\[\]\n]*\")?\)")
AUTOLINK = compiled(r"<((?:https?|mailto):[^<>\s]+)>")
BOLD = compiled(r"\*\*(?=\S)((?:[^*\n]|\*(?!\*))+?)(?<=\S)\*\*|__(?=\S)((?:[^_\n]|_(?!_))+?)(?<=\S)__")
ITALIC_STAR = compiled(r"(?<![*\w])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![*\w])")
ITALIC_UNDERSCORE = compiled(r"(?<![_\w])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![_\w])")
STRIKE = compiled(r"~~(?=\S)((?:[^~\n]|~(?!~))+?)(?<=\S)~~")
BREAK_TAG = compiled(r"<br\s*/?>", re.IGNORECASE)
ESCAPE = compiled(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
HELD = "\x00{}\x00"  # a code span's place while the rest of its line is changed; the standard reading removed every \x00
HELD_PATTERN = compiled("\x00(\\d+)\x00")
ENTITY = compiled(r"&(?:(amp|lt|gt|quot|apos|nbsp)|#(\d{1,7})|#[xX]([0-9a-fA-F]{1,6}));")
SEPARATORS = str.maketrans({"\u2028": "\n", "\u2029": "\n\n"})  # line and paragraph separators as line breaks
NAMED_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " "}
TYPED = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "‘": "'", "’": "'",
                       "‚": "'", "‛": "'", "′": "'", "…": "...", " ": " ", " ": " ", " ": " ",
                       " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
                       " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " "})
DASH = compiled(r"(?<![ \t])[ \t]*[—―–][ \t]*")  # em dash, horizontal bar, en dash; a blank run is tried from its start
# A dash typed as hyphens, in prose only: two hyphens, or one or two set apart by spaces, between characters that are
# neither a hyphen nor a backtick, so a run of hyphens (`-----Original Message-----`) and a flag in a code span stay.
TYPED_DASH = compiled(r"(?<=[^\s\-`])--(?=[^\s\-`])|(?<=[^\s\-`]) --? (?=[^\s\-`])")
SPACE_RUN = compiled(r"(?<=\S)[ \t]{2,}(?=\S)")
# Signs of code that prose seldom carries: braces, a closing semicolon, an assignment or comparison operator, empty
# parentheses, and a line opening as code does (a comment, a preprocessor line, an HTML tag, or a keyword in code's own
# shape: `def name(`, `from x import`). A word alone never counts, since wrapped prose starts lines with "from" or "let".
CODE_SIGN = compiled(r"[{}]|;\s*$|\s=\s|[=!<>:+\-*/]=|\(\)|^\s*(?:#include(?![\w])|//|/\*|\*/|def [\w]+\(|class [\w]+[(:]|"
                     r"import [\w.]+(?: as [\w]+)?\s*$|from [\w.]+ import |elif .*:\s*$|lambda [\w]*:|function\s*[\w]*\(|"
                     r"<(?!https?:|mailto:)[A-Za-z/!])")


def entity(match: re.Match[str]) -> str:
    """One HTML entity decoded: the six named here, or a numeric one naming a printable character (else left as is)."""
    if match.group(1):
        return NAMED_ENTITIES[match.group(1)]
    code = int(match.group(2)) if match.group(2) else int(match.group(3), 16)
    printable = code in (9, 10) or (32 <= code < 0x110000 and not 0x7F <= code <= 0x9F and not 0xD800 <= code <= 0xDFFF)
    return chr(code) if printable else match.group(0)


def is_code(line: str) -> bool:
    """A line of code: indented, or with a sign of code once its Markdown quote marks are set aside. Code is found
    line by line, never by fences, so a text reads the same whether its code came fenced (raw Markdown) or not
    (copied from a rendered page)."""
    return line.startswith(("    ", "\t")) or CODE_SIGN.search(QUOTE_MARKS.sub("", line)) is not None


def without_marks(line: str) -> str:
    """A line without its leading quote, heading, bullet and checkbox marks, however many are stacked (`# # #`,
    `+ + +`, `> - [ ]`), so one pass removes them all; once a heading mark has gone, so do its trailing blanks and
    every closing run of `#` set off by a blank (`# Title ##`). Linear in the line, however many marks it stacks."""
    start, stop, heading = 0, len(line), False
    while True:
        before = start
        for mark in (QUOTE_PREFIX, HEADING_OPEN, BULLET_PREFIX):
            found = mark.match(line, start, stop)
            if found:
                start = found.end()
                heading = heading or mark is HEADING_OPEN
                if mark is BULLET_PREFIX and (checkbox := CHECKBOX_PREFIX.match(line, start, stop)):
                    start = checkbox.end()
        if start == before:
            break
    while heading:
        while stop > start and line[stop - 1] in " \t":
            stop -= 1
        hashes = stop
        while hashes > start and line[hashes - 1] == "#":
            hashes -= 1
        heading = start < hashes < stop and line[hashes - 1] in " \t"  # a closing run, set off by a blank
        stop = hashes if heading else stop
    return line[start:stop]


def unmarked(line: str) -> str:
    """A line without its leading Markdown block marks (quotes, headings, bullets and a number), for telling prose."""
    line = without_marks(line)
    ordered = ORDERED_MARK.match(line)
    return line[ordered.end():] if ordered else line


def is_prose(line: str) -> bool:
    """A line of running text: not code, not a table row or tab-separated cells, and at least 60% letters among its
    visible characters (an ASCII letter or any non-ASCII character)."""
    if is_code(line) or "\t" in line or TABLE_ROW_LINE.match(line) or RULE_LINE.match(line):
        return False
    visible = [character for character in line if character not in BLANK]
    letters = sum(character.isascii() and character.isalpha() or not character.isascii() for character in visible)
    return bool(visible) and letters >= 0.6 * len(visible)


def decoded(text: str) -> str:
    """HTML entities decoded until none is left to decode (an entity written as `&amp;lt;` comes out `<`)."""
    for _ in range(4):
        out = ENTITY.sub(entity, text)
        if out == text:
            return out
        text = out
    return text


def typed(line: str) -> str:
    """Typography as typed and dashes as a spaced hyphen; in prose (told without its block mark), a dash typed as
    hyphens as a spaced hyphen too, and runs of spaces as one."""
    line = DASH.sub(" - ", line.translate(TYPED))
    if is_prose(unmarked(line)):
        line = SPACE_RUN.sub(" ", TYPED_DASH.sub(" - ", line))
    return line


def code_lines(lines: list[str]) -> list[bool]:
    """Which lines are code: those with code's signs (`is_code`), and every run of lines opening as a comment does in
    code (`#`, `*`) next to a line of code, so a comment beside code keeps its mark (`# load the data`, ` * @param x`)
    where a Markdown heading or bullet loses it. Linear: each run is looked at once."""
    code = [is_code(line) for line in lines]
    index = 0
    while index < len(lines):
        if code[index] or not COMMENT_LINE.match(lines[index]):
            index += 1
            continue
        end = index
        while end < len(lines) and not code[end] and COMMENT_LINE.match(lines[end]):
            end += 1
        if (index > 0 and code[index - 1]) or (end < len(lines) and code[end]):
            code[index:end] = [True] * (end - index)
        index = end
    return code


def joined(lines: list[str], code: list[bool]) -> list[str]:
    """Each prose paragraph (a run of non-blank lines, every one prose and none code, three words or more) on one
    line."""
    out: list[str] = []
    block: list[str] = []
    prose = True

    def flush() -> None:
        # str.split() splits on more than BLANK, but a line holds no other white space.
        if len(block) > 1 and prose and all(is_prose(line) for line in block) and sum(len(line.split()) for line in block) >= 3:
            out.append(" ".join(line.strip(BLANK) for line in block))
        else:
            out.extend(block)
        block.clear()

    for line, is_code_line in zip(lines, code, strict=True):
        if line.strip(BLANK):
            prose = prose and not is_code_line if block else not is_code_line
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return out


def inline_rendered(line: str) -> str:
    """A line's inline Markdown reduced to its text, code spans to their contents."""
    spans: list[str] = []

    def hold(match: re.Match[str]) -> str:
        spans.append(match.group(2))
        return HELD.format(len(spans) - 1)

    line = CODE_SPAN.sub(hold, line)
    line = AUTOLINK.sub(r"\1", LINK.sub(r"\1", IMAGE.sub(r"\1", line)))
    line = BOLD.sub(r"\1\2", BOLD.sub(r"\1\2", line))  # nested or adjacent bold (the group that did not match is empty)
    line = STRIKE.sub(r"\1", ITALIC_UNDERSCORE.sub(r"\1", ITALIC_STAR.sub(r"\1", line)))
    line = ESCAPE.sub(r"\1", BREAK_TAG.sub("\n", line))
    return HELD_PATTERN.sub(lambda match: spans[int(match.group(1))], line)


def rendered(lines: list[str]) -> tuple[list[str], list[bool]]:
    """Markdown rendered as a reader copies it: fence lines, rules and table separators dropped, table rows as
    tab-separated cells, and on every other line but code (`code_lines`), heading, quote and bullet marks removed,
    ordered numbers kept, and inline marks reduced. Code lines stay as written. Returns the lines and which are code."""
    out: list[str] = []
    code_out: list[bool] = []
    code = code_lines(lines)
    for index, line in enumerate(lines):
        if FENCE_LINE.match(line) or RULE_LINE.match(line) or (SETEXT_LINE.match(line) and index and lines[index - 1].strip(BLANK)):
            continue
        if TABLE_SEPARATOR_LINE.match(line):
            continue
        code_out.append(code[index])
        if TABLE_ROW_LINE.match(line):
            out.append("\t".join(inline_rendered(cell.strip(BLANK)) for cell in line.strip(BLANK).strip("|").split("|")))
            continue
        if code[index]:
            out.append(line)
            continue
        line = without_marks(line)
        if ORDERED_MARK.match(line):
            line = line.lstrip(" \t")
        out.append(inline_rendered(line))
    return out, code_out


def canonical_pass(text: str) -> str:
    """One pass over a text in the standard reading: entities decoded (`decoded`), each line's typography (`typed`),
    Markdown rendered (`rendered`), and prose paragraphs joined (`joined`), then the standard reading again. A pass
    that would leave no letter or digit (a lone rule, say) changes nothing, so no text is emptied."""
    lines = [typed(line) for line in decoded(text).translate(SEPARATORS).split("\n")]
    out = standard_reading("\n".join(joined(*rendered(lines))))
    return out if any(not character.isascii() or character.isalnum() for character in out) else text


def canonical_reading(text: str) -> str:
    """The canonical reading (`CANONICAL_READING`): the standard reading, then `canonical_pass` until a pass changes
    nothing, at most `CANONICAL_PASSES` times. One pass can uncover what the next one reads (an escaped mark becomes a
    mark, a code span's contents become prose, a bullet inside a bullet becomes the first), so the reading is the
    passes' fixed point, and reading its output again changes nothing."""
    out = standard_reading(text)
    for _ in range(CANONICAL_PASSES):
        again = canonical_pass(out)
        if again == out:
            break
        out = again
    return out


def release_reading(release: Path) -> str:
    """The reading a release reads its texts in: its config's `text_reading`, or the standard one if it names none.
    (`reading` is the chunk and window sizes, SPEC.md, section 6.)"""
    reading = json.loads((release / "config.json").read_text(encoding="utf-8")).get("text_reading", "standard")
    if reading not in READINGS:
        raise ValueError(f"{release}: reading {reading!r} is unknown here; known: {', '.join(READINGS)}")
    return reading


def read_as(text: str, reading: str) -> str:
    """A text in one of the `READINGS`."""
    if reading == "standard":
        return standard_reading(text)
    if reading == CANONICAL_READING:
        return canonical_reading(text)
    raise ValueError(f"unknown reading {reading!r}; known: {', '.join(READINGS)}")


def load_tokenizer(path: Path | None = None) -> Any:
    """ModernBERT-large's tokenizer, from a local `tokenizer.json` (checked by sha256) or the pinned Hub revision."""
    from tokenizers import Tokenizer

    if path is None:
        return Tokenizer.from_pretrained(TOKENIZER_REPOSITORY, revision=TOKENIZER_REVISION)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError(f"{path} has sha256 {digest}, not the pinned tokenizer's {TOKENIZER_SHA256}")

    return Tokenizer.from_file(str(path))


def token_ids(tokenizer: Any, text: str, reading: str = "standard") -> np.ndarray:
    """A text in its reading (`READINGS`; a release's `reading`, standard if it names none), tokenized with no special
    tokens added: (n,) int64."""
    return np.asarray(tokenizer.encode(read_as(text, reading), add_special_tokens=False).ids, dtype=np.int64)  # (n,)


# ---------------------------------------------------------------- arithmetic


def rms_norm(X: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    """RMS norm per row with a gain, float32: (n, k), (k,) to (n, k)."""
    # X: (n, k) float32; gain: (k,) float32
    r = F32(1.0) / np.sqrt(np.mean(X * X, axis=-1, keepdims=True, dtype=F32) + F32(eps))  # (n, 1)

    return (X * r) * gain  # (n, k)


def quantize_activations(X: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-token absmax 8-bit quantization: (n, k) float32 to integer values (n, k) float64 and scales s (n,) float32."""
    # X: (n, k) float32
    m = np.maximum(np.abs(X).max(axis=-1), F32(eps))  # (n,)
    s = (F32(127.0) / m).astype(F32)  # (n,)
    X_q = np.clip(np.rint(X * s[:, None]), -128, 127)  # (n, k), integer values in float32

    return X_q.astype(np.float64), s  # (n, k), (n,)


SURPRISE_QUANTILES = 101  # a table's quantiles per origin: 0%, 1%, ..., 100% of its calibration texts' values
POSITIONS_AT_ONCE = 512  # positions whose logits over the vocabulary are held at once: (512, V)


class Surprise(TypedDict):
    n_tokens: int
    mean_log_probability: float  # mean over tokens of log p(token | the text), nats; higher is less surprising
    perplexity: float  # exp(-mean_log_probability)
    percentile: list[float] | None  # human, ai, mixed: the share of that origin's calibration texts at or below it


@dataclass
class LanguageHead:
    """The masked language head (SPEC.md, Surprise): a BitLinear from width d to the embedding width e, an RMS norm,
    and the token embedding's own rows as the output, with a scale and a bias per token."""

    proj_norm: np.ndarray  # (d,)
    proj: Ternary  # T: (e, d)
    norm: np.ndarray  # (e,)
    scale: np.float32
    bias: np.ndarray  # (V,) float32

    @classmethod
    def from_tensors(cls, tensors: dict[str, np.ndarray]) -> LanguageHead:
        return cls(tensors["mlm.proj_norm"], Ternary.from_packed(tensors["mlm.proj.weight"], tensors["mlm.proj.scale"]),
                   tensors["mlm.norm"], F32(tensors["mlm.scale"][0]), tensors["mlm.bias"].astype(F32))

    def log_probabilities(self, H: np.ndarray, ids: np.ndarray, rows: np.ndarray, config: StudentConfig) -> np.ndarray:
        """Each position's log probability of its own token: (c, d) final states, (c,) ids, (V, e) embedding rows
        to (c,) float64. Logits are float32, `scale * (g . row) + bias`; the log-sum-exp is float64."""
        # H: (c, d); ids: (c,); rows: (V, e)
        G = rms_norm(bitlinear(H, self.proj_norm, self.proj, config), self.norm, config.norm_eps)  # (c, e)
        out = np.empty(len(ids))  # (c,)
        for start in range(0, len(ids), POSITIONS_AT_ONCE):
            L = (self.scale * (G[start : start + POSITIONS_AT_ONCE] @ rows.T) + self.bias).astype(np.float64)  # (q, V)
            top = L.max(axis=1)  # (q,)
            total = np.log(np.exp(L - top[:, None]).sum(axis=1)) + top  # (q,)
            out[start : start + len(L)] = L[np.arange(len(L)), ids[start : start + POSITIONS_AT_ONCE]] - total

        return out  # (c,)


def surprise_percentile(quantiles: list[float], value: float) -> float:
    """The share of an origin's calibration texts at or below `value`, from its quantiles: linear between the
    quantiles, 0 below the first, 1 above the last."""
    q = np.asarray(quantiles, dtype=np.float64)  # (Q,)
    assert len(q) == SURPRISE_QUANTILES and (np.diff(q) >= 0).all(), "a surprise table's quantiles rise"
    if value <= q[0]:
        return 0.0
    if value >= q[-1]:
        return 1.0
    upper = int(np.searchsorted(q, value, side="right"))  # q[upper - 1] <= value < q[upper]
    low, high = q[upper - 1], q[upper]
    share = (upper - 1 + (value - low) / (high - low)) / (SURPRISE_QUANTILES - 1)

    return float(share)


def surprise_reading(mean: float, n_tokens: int, table: dict[str, Any] | None) -> Surprise:
    percentile = None if table is None else [surprise_percentile(table["quantiles"][origin], mean) for origin in ORIGINS]

    return {"n_tokens": n_tokens, "mean_log_probability": mean, "perplexity": float(math.exp(-mean)), "percentile": percentile}


def surprise_table(values: np.ndarray, labels: np.ndarray, split: str) -> dict[str, Any]:
    """The release's surprise table: each origin's calibration texts' mean token log probabilities at the quantiles
    0%, 1%, ..., 100%."""
    # values, labels: (n,)
    levels = np.linspace(0.0, 1.0, SURPRISE_QUANTILES)  # (Q,)

    return {"split": split, "texts": [int((labels == origin).sum()) for origin in range(3)],
            "quantiles": {name: np.quantile(values[labels == origin], levels).tolist() for origin, name in enumerate(ORIGINS)}}


@dataclass
class Ternary:
    """One ternary matrix, transposed for products: T (i, o) of -1, 0, 1 as float64, and its scale gamma."""

    T: np.ndarray  # (i, o)
    gamma: np.float32

    @classmethod
    def from_packed(cls, packed: np.ndarray, scale: np.ndarray) -> Ternary:
        # packed: (o, i/4); scale: (1,)
        return cls(T=unpack_ternary(packed).T.astype(np.float64), gamma=F32(scale.reshape(())))  # T: (i, o)


def ternary_product(X_q: np.ndarray, s: np.ndarray, weight: Ternary) -> np.ndarray:
    """Exact integer sums, then one float32 scale per token: (n, i), (n,) to (n, o) float32."""
    # X_q: (n, i) integer values; s: (n,) float32
    acc = X_q @ weight.T  # (n, o), exact: integers under 2^53
    factor = (weight.gamma / s).astype(F32)  # (n,)

    return acc.astype(F32) * factor[:, None]  # (n, o)


def bitlinear(X: np.ndarray, gain: np.ndarray, weight: Ternary, config: StudentConfig) -> np.ndarray:
    """RMS norm, 8-bit quantization and a ternary product: (n, i) to (n, o)."""
    # X: (n, i) float32; gain: (i,)
    X_q, s = quantize_activations(rms_norm(X, gain, config.norm_eps), config.act_eps)  # (n, i), (n,)

    return ternary_product(X_q, s, weight)  # (n, o)


def rope_tables(block_size: int, head_width: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    """Cosines and sines of the rotary angles, computed in float64, stored float32: two (W, d_h/2)."""
    j = np.arange(head_width // 2, dtype=np.float64)  # (d_h/2,)
    frequencies = base ** (-2.0 * j / head_width)  # (d_h/2,)
    angles = np.arange(block_size, dtype=np.float64)[:, None] * frequencies[None, :]  # (W, d_h/2)

    return np.cos(angles).astype(F32), np.sin(angles).astype(F32)  # (W, d_h/2) each


def apply_rope(U: np.ndarray, cosine: np.ndarray, sine: np.ndarray) -> np.ndarray:
    """Rotate halves: (h, n, d_h) with (n, d_h/2) tables to (h, n, d_h)."""
    # U: (h, n, d_h); cosine, sine: (n, d_h/2)
    half = U.shape[-1] // 2
    first, second = U[..., :half], U[..., half:]  # (h, n, d_h/2) each

    return np.concatenate([first * cosine - second * sine, second * cosine + first * sine], axis=-1)  # (h, n, d_h)


def quantize_symmetric(U: np.ndarray, axes: tuple[int, ...], eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Absmax 8-bit quantization with one scale over `axes`, in [-127, 127]: (h, n, d_h) float32 to integer values
    (h, n, d_h) float64 and scales s float32, kept along `axes`."""
    # U: (h, n, d_h) float32
    m = np.maximum(np.abs(U).max(axis=axes, keepdims=True), F32(eps))  # e.g. (h, n, 1)
    s = (F32(127.0) / m).astype(F32)  # e.g. (h, n, 1)
    U_q = np.clip(np.rint(U * s), -127, 127)  # (h, n, d_h), integer values in float32

    return U_q.astype(np.float64), s  # (h, n, d_h), e.g. (h, n, 1)


def quantize_output(X: np.ndarray, eps: float) -> np.ndarray:
    """The int8 codes of the head's query or key (SPEC.md, Int8 output): one absmax scale per vector, `s = 127 / max(max
    |x|, eps)`, `code = clamp(round(x * s), -127, 127)`. Applied to a unit vector of codes it returns the codes."""
    # X: (..., w) float32
    codes, _ = quantize_symmetric(np.asarray(X, dtype=F32), (-1,), eps)  # (..., w) integer values in float64, (..., 1)

    return codes.astype(np.int8)  # (..., w)


def inverse_norms(codes: np.ndarray) -> np.ndarray:
    """`1 / sqrt(sum c^2)` of integer code vectors, float64 (the square root and the division each correctly rounded,
    so every implementation gets the same bits), and 0 for the zero vector (SPEC.md, Int8 output): (..., w) to (...)."""
    # codes: (..., w) integers
    squares = (codes.astype(np.int64) ** 2).sum(axis=-1)  # (...)
    root = np.sqrt(squares.astype(np.float64))  # (...)

    return np.where(squares > 0, 1.0 / np.where(squares > 0, root, 1.0), 0.0)  # (...)


def float32_of_bits(bits: int) -> np.float32:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)[()]


# exp of integer attention (SPEC.md, Integer attention): Cephes expf's reduction and polynomial, constants as float32 bits
EXP_MAGIC = F32(12_582_912.0)  # 1.5 * 2^23: adding and removing it rounds to an integer, ties to even
EXP_LOG2E, EXP_C1, EXP_C2 = float32_of_bits(0x3FB8AA3B), float32_of_bits(0x3F318000), float32_of_bits(0xB95E8083)
EXP_POLYNOMIAL = tuple(float32_of_bits(bits) for bits in (0x39506967, 0x3AB743CE, 0x3C088908, 0x3D2AA9C1, 0x3E2AAAAA, 0x3F000000))


def exp_nonpositive(X: np.ndarray) -> np.ndarray:
    """`exp(x)` for `x <= 0`, float32, one rounding per operation and no fused multiply-add, so every implementation
    gets the same bits (the engine's `dense::exp_nonpositive`): (...) float32 to (...) float32."""
    # X: (...) float32
    X = np.maximum(np.asarray(X, dtype=F32), F32(-87.0))  # (...)
    shifted = X * EXP_LOG2E + EXP_MAGIC  # (...)
    n = shifted - EXP_MAGIC  # (...) round(x / ln 2)
    exponent = shifted.view(np.int32) - EXP_MAGIC.view(np.int32)  # (...)
    r = (X - n * EXP_C1) - n * EXP_C2  # (...)
    p = np.full_like(X, EXP_POLYNOMIAL[0])  # (...)
    for coefficient in EXP_POLYNOMIAL[1:]:
        p = p * r + coefficient
    p = p * (r * r) + r + F32(1.0)  # (...)
    scale = ((exponent + 127) << 23).astype(np.int32).view(F32)  # (...) 2^n

    return p * scale  # (...)


def integer_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, eps: float) -> np.ndarray:
    """Attention of one block whose two products are integer products (SPEC.md, Integer attention): queries and keys
    int8 per token and head, values int8 per block and head, the softmax in float32, weights rounded to 127ths.
    (h, n, d_h) float32 each to (h, n, d_h) float32."""
    # Q, K, V: (h, n, d_h) float32
    d_h = Q.shape[-1]
    Q_q, s_q = quantize_symmetric(Q, (-1,), eps)  # (h, n, d_h), (h, n, 1)
    K_q, s_k = quantize_symmetric(K, (-1,), eps)  # (h, n, d_h), (h, n, 1)
    V_q, s_v = quantize_symmetric(V, (-2, -1), eps)  # (h, n, d_h), (h, 1, 1)
    A = Q_q @ K_q.transpose(0, 2, 1)  # (h, n, n), exact: |A| <= d_h * 127^2 < 2^24
    x = F32(1.0 / math.sqrt(d_h)) / s_q  # (h, n, 1) float32
    y = (F32(1.0) / s_k).transpose(0, 2, 1)  # (h, 1, n) float32
    S = (A.astype(F32) * x) * y  # (h, n, n)
    E = exp_nonpositive(S - S.max(axis=-1, keepdims=True))  # (h, n, n)
    z = E.astype(np.float64).sum(axis=-1, keepdims=True)  # (h, n, 1), float64: its order does not matter
    r = (ATTENTION_WEIGHT_LEVELS / z).astype(F32)  # (h, n, 1)
    P_q = np.rint(E * r)  # (h, n, n), integers in [0, 127]
    B = P_q.astype(np.float64) @ V_q  # (h, n, d_h), exact integers
    t = F32(1.0) / (F32(ATTENTION_WEIGHT_LEVELS) * s_v)  # (h, 1, 1) float32
    O = B.astype(F32) * t  # (h, n, d_h)
    assert O.dtype == F32, "integer attention's float parts are float32"

    return O  # (h, n, d_h)


# ---------------------------------------------------------------- the model


@dataclass
class LayerWeights:
    attn_norm: np.ndarray  # (d,)
    q: Ternary
    k: Ternary
    v: Ternary
    attn_out_norm: np.ndarray  # (d,)
    o: Ternary
    ffn_norm: np.ndarray  # (d,)
    up: Ternary
    ffn_out_norm: np.ndarray  # (f,)
    down: Ternary


class BankReading(TypedDict):
    mean: list[float]  # per origin, the mean same-origin probability against that origin's bank texts
    spread: list[float]  # per origin, their population standard deviation
    closeness: float  # the mean of the top_k highest cosines to any bank vector


class Strip(TypedDict):
    position: float
    band_lo: float
    band_hi: float
    level: float
    unlike_reference: bool


class Reading(TypedDict):
    n_tokens: int
    length_bin: int
    mean: list[float]
    spread: list[float]
    share: list[float]
    score: float
    p_values: list[float]
    sets: dict[str, list[str]]
    not_human: dict[str, bool]
    probability: float
    interval: list[float]
    three_way: list[float]
    spread_percentile: list[float]
    closeness: float
    closeness_p: float
    out_of_reference: bool
    strip: Strip
    explain: NotRequired[Explain]  # added by ReferenceModel.reading


class ChunkReading(Reading):
    start: int  # document positions [start, end)
    end: int


class WindowReading(Reading):
    start: int  # document positions [start, end)
    end: int
    chunk: int


class Cell(TypedDict):
    start: int  # document positions [start, end), one stride of its chunk
    end: int
    position: float  # overlap-weighted mean strip position of the windows covering the cell
    spread: float  # their overlap-weighted standard deviation
    band_lo: float  # the lowest band_lo and the highest band_hi of those windows' strips
    band_hi: float
    windows: int
    unlike_reference: bool  # some covering window is unlike the reference


class Verdict(TypedDict):
    chunks: int
    level: float  # the family level of the mixed-document signal
    p_values: list[float]
    sets: dict[str, list[str]]
    not_human: dict[str, bool]
    probability: float
    interval: list[float]
    disagreement: float
    lowest: float
    highest: float
    three_way: list[float]
    closeness_p: float
    out_of_reference: bool
    strip: Strip
    chunk_level: float
    model_chunks: int
    human_chunks: int
    mixed_document: bool


class DocumentReading(TypedDict):
    n_tokens: int
    chunk_tokens: int
    verdict: Verdict  # the document's, from its chunks
    chunks: list[ChunkReading]  # each with its explanation
    windows: list[WindowReading]  # the heat map's windows, in position order
    cells: list[Cell]  # the heat map, in position order


def chunk_bounds(n_tokens: int, chunk_tokens: int) -> list[tuple[int, int]]:
    """`C = ceil(n / C_max)` chunks; chunk `c` is `[floor(c n / C), floor((c + 1) n / C))` (SPEC.md, Chunks)."""
    if n_tokens <= 0:
        raise ValueError("a text of zero tokens has no reading")
    count = -(-n_tokens // chunk_tokens)
    edges = [index * n_tokens // count for index in range(count + 1)]

    return list(zip(edges[:-1], edges[1:], strict=True))


def window_spans(length: int, width: int, stride: int) -> list[tuple[int, int]]:
    """The windows of a chunk of `length` tokens, chunk positions [start, end) (SPEC.md, Windows and the heat map)."""
    if length <= width:
        return [(0, length)]
    spans = [(start, start + width) for start in range(0, length - width + 1, stride)]
    if spans[-1][1] != length:
        spans.append((length - width, length))

    return spans


def cell_spans(length: int, stride: int) -> list[tuple[int, int]]:
    """The heat map's cells of a chunk: one per stride, the last possibly shorter; chunk positions [start, end)."""
    return [(start, min(start + stride, length)) for start in range(0, length, stride)]


def pooled_statistics(total: np.ndarray, squares: np.ndarray, count: int) -> np.ndarray:
    """float32 concat(mean, variance) from float64 sums: (d,), (d,) to (2d,)."""
    # total, squares: (d,) float64
    assert count > 0, "an empty span has no statistics"
    mean = total / count  # (d,)
    variance = np.maximum(squares / count - mean * mean, 0.0)  # (d,)

    return np.concatenate([mean, variance]).astype(F32)  # (2d,)


@dataclass
class ChunkPool:
    """Running float64 sums of one chunk's token states, recorded at every position a window starts or ends.

    The pooled vector of positions [a, b) is the difference of the totals recorded at b and at a.
    """

    width: int  # d
    length: int  # the chunk's tokens
    window_width: int
    window_stride: int
    total: np.ndarray = field(init=False)  # (d,)
    squares: np.ndarray = field(init=False)  # (d,)
    count: int = 0
    marks: list[int] = field(init=False)
    recorded: dict[int, tuple[np.ndarray, np.ndarray]] = field(init=False)  # position to (d,), (d,)

    def __post_init__(self) -> None:
        self.total = np.zeros(self.width, dtype=np.float64)  # (d,)
        self.squares = np.zeros(self.width, dtype=np.float64)  # (d,)
        spans = window_spans(self.length, self.window_width, self.window_stride)
        self.marks = sorted({position for span in spans for position in span} - {0})
        self.recorded = {0: (self.total.copy(), self.squares.copy())}

    def add(self, H: np.ndarray) -> None:
        # H: (n, d) float32 token states of the next positions, in order
        assert self.count + H.shape[0] <= self.length, "a chunk pool takes only its chunk's tokens"
        first = 0
        while first < H.shape[0]:
            mark = next(mark for mark in self.marks if mark > self.count)
            take = min(mark - self.count, H.shape[0] - first)
            H64 = H[first : first + take].astype(np.float64)  # (t, d)
            self.total += H64.sum(axis=0)  # (d,)
            self.squares += (H64 * H64).sum(axis=0)  # (d,)
            self.count += take
            first += take
            if self.count == mark:
                self.recorded[mark] = (self.total.copy(), self.squares.copy())

    def pooled(self, start: int, end: int) -> np.ndarray:
        """The pooled vector of chunk positions [start, end), both recorded: (2d,) float32."""
        (total_start, squares_start), (total_end, squares_end) = self.recorded[start], self.recorded[end]

        return pooled_statistics(total_end - total_start, squares_end - squares_start, end - start)  # (2d,)


class Calibration:
    """The teacher's calibration table (`litmus-calibration`, version 1), read for one text at a time.

    The arithmetic of one text is `apply_calibration` of `../teacher/src/litmus_teacher/calibrate.py`, which owns the
    table and fits it; `tests/test_reference.py` checks that this reading equals that one. A document's verdict from
    its chunks is that module's `combine_chunks` itself, called here, so it has one implementation.
    """

    def __init__(self, table: dict[str, Any]) -> None:
        if table.get("format") != CALIBRATION_FORMAT or table.get("version") != CALIBRATION_VERSION:
            raise ValueError(f"the calibration table is not {CALIBRATION_FORMAT} version {CALIBRATION_VERSION}")
        self.table = table
        self.upper_edges = np.asarray(table["length_edges"][1:], dtype=np.int64)  # (bins - 1,)
        self.bins = len(table["length_edges"])
        self.levels: list[float] = [float(level) for level in table["levels"]]
        if STRIP_LEVEL not in self.levels or MIXED_LEVEL not in self.levels:
            raise ValueError(f"the table's levels {self.levels} must hold {STRIP_LEVEL} and {MIXED_LEVEL}")
        self.grid = [float(level) for level in table["grid"]]
        self.spread_levels = np.asarray(table["spread_levels"], dtype=np.float64)  # (s,)
        self.top_k = int(table["top_k"])
        self.out_of_reference_level = float(table["out_of_reference_level"])
        for kind in ("probability", "three_way", "spread", "closeness"):
            self._check_groups(table[kind])
        for origin in ORIGINS:
            self._check_groups(table["conformal"][origin])

    def _check_groups(self, entries: list[dict[str, Any]]) -> None:
        members = sorted(index for entry in entries for index in entry["bins"])
        assert members == list(range(self.bins)), "every length bin belongs to exactly one group"

    def length_bin(self, n_tokens: int) -> int:
        """Bin `i` holds counts in [length_edges[i], length_edges[i + 1]); the last is open above."""
        return int(np.searchsorted(self.upper_edges, n_tokens, side="right"))

    @staticmethod
    def group(entries: list[dict[str, Any]], length_bin: int) -> dict[str, Any]:
        return next(entry for entry in entries if length_bin in entry["bins"])

    def grid_p_value(self, value: float, thresholds: list[float | None]) -> float:
        """The smallest grid level `u` whose threshold `t_u` exceeds `value`, else 1."""
        return next((level for level, threshold in zip(self.grid, thresholds, strict=True) if threshold is not None and value < threshold), 1.0)

    def read(self, bank: BankReading, n_tokens: int) -> Reading:
        """Every calibrated output for one text's bank reading (SPEC.md, The reading)."""
        table, b = self.table, self.length_bin(n_tokens)
        mean = np.asarray(bank["mean"], dtype=np.float64)  # (3,)
        share = mean / mean.sum()  # (3,)
        score = float(1.0 - share[0])
        p_values = [self.grid_p_value(float(share[origin]), self.group(table["conformal"][name], b)["thresholds"])
                    for origin, name in enumerate(ORIGINS)]
        sets = {level_key(level): [name for origin, name in enumerate(ORIGINS) if p_values[origin] > level] for level in self.levels}
        not_human = {level_key(level): p_values[0] <= level for level in self.levels}

        venn = self.group(table["probability"], b)
        block = int(np.searchsorted(np.asarray(venn["edges"], dtype=np.float64), score, side="right"))
        p0, p1 = float(venn["p0"][block]), float(venn["p1"][block])
        probability = p1 / (1.0 - p0 + p1)

        softmax = self.group(table["three_way"], b)
        logits = softmax["slope"] * np.log(np.maximum(share, LOG_FLOOR)) + np.asarray(softmax["bias"], dtype=np.float64)  # (3,)
        logits -= logits.max()
        three_way = np.exp(logits) / np.exp(logits).sum()  # (3,)

        spread_group = self.group(table["spread"], b)
        spread_percentile = [float(np.interp(bank["spread"][origin], np.asarray(spread_group[name], dtype=np.float64), self.spread_levels))
                             for origin, name in enumerate(ORIGINS)]
        closeness_p = self.grid_p_value(bank["closeness"], self.group(table["closeness"], b)["thresholds"])
        out_of_reference = closeness_p <= self.out_of_reference_level

        return {
            "n_tokens": n_tokens, "length_bin": b, "mean": bank["mean"], "spread": bank["spread"], "share": share.tolist(),
            "score": score, "p_values": p_values, "sets": sets, "not_human": not_human, "probability": probability,
            "interval": [p0, p1], "three_way": three_way.tolist(), "spread_percentile": spread_percentile,
            "closeness": bank["closeness"], "closeness_p": closeness_p, "out_of_reference": out_of_reference,
            "strip": litmus_strip(three_way.tolist(), sets[level_key(STRIP_LEVEL)], out_of_reference),
        }

    def verdict(self, chunks: list[ChunkReading]) -> Verdict:
        """The document's verdict from its chunks' readings: the teacher's `combine_chunks` at MIXED_LEVEL."""
        calibrate = teacher_calibration()
        verdicts = calibrate.Verdicts(
            np.asarray([chunk["length_bin"] for chunk in chunks]),  # (C,)
            np.asarray([chunk["p_values"] for chunk in chunks], dtype=np.float64),  # (C, 3)
            {level: np.asarray([[name in chunk["sets"][level_key(level)] for name in ORIGINS] for chunk in chunks]) for level in self.levels},
            np.asarray([chunk["probability"] for chunk in chunks]),  # (C,)
            np.asarray([chunk["interval"] for chunk in chunks]),  # (C, 2)
            np.asarray([chunk["three_way"] for chunk in chunks]),  # (C, 3)
            np.asarray([chunk["spread_percentile"] for chunk in chunks]),  # (C, 3)
            np.asarray([chunk["closeness_p"] for chunk in chunks], dtype=np.float64),  # (C,)
            np.asarray([chunk["out_of_reference"] for chunk in chunks]),  # (C,)
        )
        combined = calibrate.combine_chunks(self.table, verdicts, MIXED_LEVEL)
        sets = {level_key(level): [name for origin, name in enumerate(ORIGINS) if combined.sets[level][origin]] for level in self.levels}
        three_way = [float(value) for value in combined.three_way]

        return {
            "chunks": combined.chunks, "level": MIXED_LEVEL, "p_values": [float(value) for value in combined.p_values], "sets": sets,
            "not_human": {level_key(level): combined.not_human(level) for level in self.levels},
            "probability": combined.probability, "interval": [float(value) for value in combined.interval],
            "disagreement": combined.disagreement, "lowest": combined.lowest, "highest": combined.highest, "three_way": three_way,
            "closeness_p": combined.closeness_p_value, "out_of_reference": bool(combined.out_of_reference),
            "strip": litmus_strip(three_way, sets[level_key(STRIP_LEVEL)], bool(combined.out_of_reference)),
            "chunk_level": combined.chunk_level, "model_chunks": combined.model_chunks, "human_chunks": combined.human_chunks,
            "mixed_document": combined.mixed_document,
        }


def level_key(level: float) -> str:
    return f"{level:g}"


STRIP_ANCHORS = {"human": 0.0, "mixed": 0.5, "ai": 1.0}


def litmus_strip(three_way: list[float], conformal_set: list[str], out_of_reference: bool) -> Strip:
    """The strip position from the calibrated three-way probabilities, and the band the conformal set cannot rule out."""
    p_human, p_ai, p_mixed = three_way
    position = p_ai + 0.5 * p_mixed
    anchors = [STRIP_ANCHORS[name] for name in conformal_set]

    return {"position": position, "band_lo": min([position, *anchors]), "band_hi": max([position, *anchors]), "level": STRIP_LEVEL,
            "unlike_reference": not conformal_set or out_of_reference}


def heat_map(windows: list[WindowReading], start: int, length: int, stride: int) -> list[Cell]:
    """The cells of one chunk at document position `start` from its windows (SPEC.md, Windows and the heat map)."""
    cells: list[Cell] = []
    for cell_start, cell_end in cell_spans(length, stride):
        first, last = start + cell_start, start + cell_end
        covering = [(min(last, window["end"]) - max(first, window["start"]), window) for window in windows
                    if window["start"] < last and window["end"] > first]
        assert covering, "every cell lies under at least one window"
        overlap = np.asarray([tokens for tokens, _ in covering], dtype=np.float64)  # (k,)
        position = np.asarray([window["strip"]["position"] for _, window in covering])  # (k,)
        mean = float((overlap * position).sum() / overlap.sum())
        cells.append({
            "start": first, "end": last, "position": mean,
            "spread": float(np.sqrt((overlap * (position - mean) ** 2).sum() / overlap.sum())),
            "band_lo": min(window["strip"]["band_lo"] for _, window in covering),
            "band_hi": max(window["strip"]["band_hi"] for _, window in covering),
            "windows": len(covering), "unlike_reference": any(window["strip"]["unlike_reference"] for _, window in covering),
        })

    return cells


class ConformalCount(TypedDict):
    origin: str
    n: int  # calibration texts of this origin in the text's length group
    count: int  # of them, those whose share of this origin is at most the text's
    p: float  # (1 + count) / (n + 1), the exact split conformal p-value the grid bounds
    bins: list[int]


class ClosenessCount(TypedDict):
    n: int  # calibration texts of every origin in the text's length group
    count: int  # of them, those at most as close to the bank as the text
    p: float
    bins: list[int]


class Explain(TypedDict):
    conformal: list[ConformalCount]
    closeness: ClosenessCount
    nearest: list[dict[str, Any]]  # the nearest bank entries: index, origin, cosine, probability, tokens, metadata


def group_index(entries: list[dict[str, Any]], length_bin: int) -> int:
    return next(index for index, entry in enumerate(entries) if length_bin in entry["bins"])


class Explanation:
    """The calibration counts and the bank's metadata a reading explains itself with (SPEC.md, Explanation).

    The calibration texts' own scores, sorted, per group of the table: `explain.conformal.<origin>.scores` (the
    share of that origin of each calibration text of that origin) and `explain.closeness.scores` (the closeness of
    every calibration text), each with `offsets` marking the groups in the table's order.
    """

    def __init__(self, calibration: Calibration, tensors: dict[str, np.ndarray], fields: dict[str, list[str]]) -> None:
        # tensors: (...) each, by name (SPEC.md, section 6)
        table = calibration.table
        self.conformal = [self._cells(tensors, f"explain.conformal.{name}", table["conformal"][name]) for name in ORIGINS]
        self.closeness = self._cells(tensors, "explain.closeness", table["closeness"])
        self.table = table
        self.origin = tensors["bank.origin"].astype(np.int64)  # (m,)
        self.tokens = tensors["bank.tokens"].astype(np.int64)  # (m,)
        self.fields = fields
        self.metadata = {field: tensors[f"bank.meta.{field}"].astype(np.int64) for field in fields}  # each (m,)
        for field, codes in self.metadata.items():
            assert codes.shape == self.origin.shape and codes.max() < len(fields[field]), f"bank metadata {field} indexes its names"

    @staticmethod
    def _cells(tensors: dict[str, np.ndarray], prefix: str, entries: list[dict[str, Any]]) -> list[np.ndarray]:
        # tensors: (...) each, by name
        scores = tensors[prefix + ".scores"].astype(np.float64)  # (n_c,)
        offsets = tensors[prefix + ".offsets"].astype(np.int64)  # (g + 1,)
        assert len(offsets) == len(entries) + 1 and offsets[-1] == len(scores), f"{prefix} has one cell per group"
        cells = [scores[offsets[index] : offsets[index + 1]] for index in range(len(entries))]  # each (n_g,)
        for cell, entry in zip(cells, entries, strict=True):
            assert len(cell) == entry["n"] and (np.diff(cell) >= 0).all(), f"{prefix}: each cell holds its group's n scores, ascending"

        return cells  # each (n_g,)

    def explain(self, reading: Reading, p: np.ndarray, cosines: np.ndarray) -> Explain:
        """The counts behind the reading's p-values and the nearest bank entries."""
        # p, cosines: (m,), against every bank vector
        b = reading["length_bin"]
        conformal: list[ConformalCount] = []
        for origin, name in enumerate(ORIGINS):
            index = group_index(self.table["conformal"][name], b)
            cell = self.conformal[origin][index]  # (n_g,)
            count = int(np.searchsorted(cell, reading["share"][origin], side="right"))
            conformal.append({"origin": name, "n": len(cell), "count": count, "p": (1 + count) / (len(cell) + 1),
                              "bins": self.table["conformal"][name][index]["bins"]})
        index = group_index(self.table["closeness"], b)
        cell = self.closeness[index]  # (n_g,)
        count = int(np.searchsorted(cell, reading["closeness"], side="right"))
        closeness: ClosenessCount = {"n": len(cell), "count": count, "p": (1 + count) / (len(cell) + 1), "bins": self.table["closeness"][index]["bins"]}
        nearest = []
        for j in np.argsort(-cosines, kind="stable")[:NEAREST]:
            entry = {"index": int(j), "origin": ORIGINS[self.origin[j]], "cosine": float(cosines[j]), "probability": float(p[j]),
                     "tokens": int(self.tokens[j])}
            entry.update({field: self.fields[field][self.metadata[field][j]] for field in self.fields})
            nearest.append(entry)

        return {"conformal": conformal, "closeness": closeness, "nearest": nearest}


def explain_tensors(table: dict[str, Any], mean: np.ndarray, closeness: np.ndarray, labels: np.ndarray, tokens: np.ndarray) -> dict[str, np.ndarray]:
    """The sorted calibration scores behind every group of a fitted table, from the readings it was fitted on."""
    # mean: (n_c, 3); closeness, labels, tokens: (n_c,)
    share = mean / mean.sum(axis=1, keepdims=True)  # (n_c, 3)
    bins = np.searchsorted(np.asarray(table["length_edges"][1:]), tokens, side="right")  # (n_c,)
    tensors: dict[str, np.ndarray] = {}

    def cells(prefix: str, entries: list[dict[str, Any]], values: np.ndarray, selected: np.ndarray) -> None:
        # values, selected: (n_c,)
        parts = [np.sort(values[selected & np.isin(bins, entry["bins"])]).astype(F32) for entry in entries]  # each (n_g,)
        for part, entry in zip(parts, entries, strict=True):
            assert len(part) == entry["n"], f"{prefix}: the readings are the ones the table was fitted on"
        tensors[prefix + ".scores"] = np.concatenate(parts)  # (sum n_g,)
        tensors[prefix + ".offsets"] = np.concatenate([[0], np.cumsum([len(part) for part in parts])]).astype(np.int64)  # (g + 1,)

    for origin, name in enumerate(ORIGINS):
        cells(f"explain.conformal.{name}", table["conformal"][name], share[:, origin], labels == origin)
    cells("explain.closeness", table["closeness"], closeness, np.ones(len(labels), dtype=bool))

    return tensors  # (...) each, by name


class ReferenceModel:
    """A loaded release: the encoder, the head, the bank and the calibration table."""

    def __init__(self, config: StudentConfig, tensors: dict[str, np.ndarray], calibration: dict[str, Any] | None,
                 bank_fields: dict[str, list[str]] | None = None, attention: str = "float", output: str = "float",
                 surprise_table: dict[str, Any] | None = None) -> None:
        # tensors: (...) each, by name (SPEC.md, section 6)
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention {attention!r} is not one of {ATTENTION_MODES}")
        if output not in OUTPUT_MODES:
            raise ValueError(f"output {output!r} is not one of {OUTPUT_MODES}")
        self.config = config
        self.attention = attention  # "float" (SPEC.md, section 3, step 4) or "integer" (Integer attention)
        self.output = output  # "float" (unit vectors, float16 bank) or "int8" (Int8 output: codes, int8 bank)
        self.embed_weight = tensors["embed.weight"]  # (V, e) int8
        self.embed_scale = tensors["embed.scale"].astype(F32)  # (V,)
        assert self.embed_weight.shape == (config.vocab_size, config.embed_width), "embedding shape"
        if config.projected:
            self.proj_norm = tensors["embed.proj_norm"]  # (e,)
            self.proj = Ternary.from_packed(tensors["embed.proj.weight"], tensors["embed.proj.scale"])  # T: (e, d)
        self.layers = [self._layer(tensors, index) for index in range(config.depth)]
        self.final_norm = tensors["final_norm"]  # (d,)
        self.input_mean = tensors["head.input_mean"].astype(F32)  # (2d,)
        self.input_scale = tensors["head.input_scale"].astype(F32)  # (2d,)
        self.fc1_weight = tensors["head.fc1.weight"].astype(F32)  # (hh, 2d)
        self.fc1_bias = tensors["head.fc1.bias"].astype(F32)  # (hh,)
        self.fc2_weight = tensors["head.fc2.weight"].astype(F32)  # (2w, hh)
        self.fc2_bias = tensors["head.fc2.bias"].astype(F32)  # (2w,)
        self.pair_scale, self.pair_bias = (float(value) for value in tensors["head.pair"])
        vectors = tensors["bank.vectors"]  # (m, 2w) float16, or int8 in Int8 output
        assert vectors.dtype == (np.int8 if output == "int8" else np.float16), f"bank.vectors is {vectors.dtype} in a release with {output} output"
        bank = vectors.astype(np.float64)  # (m, 2w), integer values in Int8 output
        self.bank_origin = tensors["bank.origin"].astype(np.int64)  # (m,)
        assert (np.diff(self.bank_origin) >= 0).all() and set(np.unique(self.bank_origin)) == {0, 1, 2}, "bank grouped by origin"
        self.bank_query, self.bank_key = bank[:, : config.head_out], bank[:, config.head_out :]  # (m, w) each
        if output == "int8":
            if vectors.min() < -OUTPUT_CODE_MAX:
                raise ValueError("bank.vectors holds the code -128, which an int8 output never has")
            self.bank_unit = None  # cosines come from integer dot products and `inverse_norms`
            self.bank_inverse_query, self.bank_inverse_key = inverse_norms(self.bank_query), inverse_norms(self.bank_key)  # (m,) each
        else:
            self.bank_unit = bank / np.linalg.norm(bank, axis=1, keepdims=True)  # (m, 2w), for cosines
        self.cosine, self.sine = rope_tables(config.block_size, config.head_width, config.rope_base)  # (W, d_h/2) each
        self.calibration = Calibration(calibration) if calibration is not None else None
        self.explanation = Explanation(self.calibration, tensors, bank_fields or {}) if self.calibration is not None else None
        self.language = LanguageHead.from_tensors(tensors) if "mlm.proj.weight" in tensors else None  # optional (SPEC.md, Surprise)
        self.surprise_table = surprise_table

    @staticmethod
    def _layer(tensors: dict[str, np.ndarray], index: int) -> LayerWeights:
        # tensors: (...) each, by name
        prefix = f"layers.{index}."

        def ternary(name: str) -> Ternary:
            return Ternary.from_packed(tensors[prefix + name + ".weight"], tensors[prefix + name + ".scale"])

        return LayerWeights(
            attn_norm=tensors[prefix + "attn_norm"], q=ternary("attn.q"), k=ternary("attn.k"), v=ternary("attn.v"),
            attn_out_norm=tensors[prefix + "attn_out_norm"], o=ternary("attn.o"), ffn_norm=tensors[prefix + "ffn_norm"],
            up=ternary("ffn.up"), ffn_out_norm=tensors[prefix + "ffn_out_norm"], down=ternary("ffn.down"),
        )

    @classmethod
    def load(cls, directory: Path) -> ReferenceModel:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        attention, output = release_attention(config, str(directory)), release_output(config, str(directory))

        return cls(StudentConfig.from_release(config), read_safetensors(directory / "model.safetensors"), config.get("calibration"),
                   config["bank"].get("fields"), attention, output, config.get("surprise"))

    # ------------------------------------------------------------ encoder

    def embed(self, ids: np.ndarray) -> np.ndarray:
        """Token ids to the first residual stream: (n,) to (n, d) float32."""
        # ids: (n,)
        if ids.size and (ids.min() < 0 or ids.max() >= self.config.vocab_size):
            raise ValueError("token ids lie outside the vocabulary")
        X = self.embed_weight[ids].astype(F32) * self.embed_scale[ids][:, None]  # (n, e)
        if self.config.projected:
            X = bitlinear(X, self.proj_norm, self.proj, self.config)  # (n, d)

        return X  # (n, d)

    def layer_block(self, index: int, X: np.ndarray, first: int) -> np.ndarray:
        """Layer `index` on the inputs of one of its blocks, starting at position `first`: (n, d) to (n, d)."""
        # X: (n, d) float32
        config, layer = self.config, self.layers[index]
        n, h, d_h = X.shape[0], config.heads, config.head_width
        shift, W = config.shift(index), config.block_size
        assert (first + shift) // W == (first + n - 1 + shift) // W, "a block never crosses a block boundary"

        X_q, s = quantize_activations(rms_norm(X, layer.attn_norm, config.norm_eps), config.act_eps)  # (n, d), (n,)
        Q = ternary_product(X_q, s, layer.q).reshape(n, h, d_h).transpose(1, 0, 2)  # (h, n, d_h)
        K = ternary_product(X_q, s, layer.k).reshape(n, h, d_h).transpose(1, 0, 2)  # (h, n, d_h)
        V = ternary_product(X_q, s, layer.v).reshape(n, h, d_h).transpose(1, 0, 2)  # (h, n, d_h)
        r = (np.arange(first, first + n) + shift) % W  # (n,)
        Q = apply_rope(Q, self.cosine[r], self.sine[r])  # (h, n, d_h)
        K = apply_rope(K, self.cosine[r], self.sine[r])  # (h, n, d_h)

        if self.attention == "integer":
            O = integer_attention(Q, K, V, config.act_eps).transpose(1, 0, 2).reshape(n, config.width)  # (n, d)
        else:
            S = (Q @ K.transpose(0, 2, 1)) * F32(1.0 / math.sqrt(d_h))  # (h, n, n)
            E = np.exp(S - S.max(axis=-1, keepdims=True))  # (h, n, n)
            P = E / E.sum(axis=-1, keepdims=True)  # (h, n, n)
            O = (P @ V).transpose(1, 0, 2).reshape(n, config.width)  # (n, d)
        X = X + bitlinear(O, layer.attn_out_norm, layer.o, config)  # (n, d)

        U = bitlinear(X, layer.ffn_norm, layer.up, config)  # (n, f)
        R = np.maximum(U, F32(0.0))  # (n, f)
        X = X + bitlinear(R * R, layer.ffn_out_norm, layer.down, config)  # (n, d)
        assert X.dtype == F32, "the residual stream stays float32"

        return X  # (n, d)

    def block_spans(self, index: int, first: int, end: int) -> list[tuple[int, int]]:
        """The [start, stop) position spans of layer `index`'s blocks over positions [first, end)."""
        W, shift = self.config.block_size, self.config.shift(index)
        spans, start = [], first
        while start < end:
            stop = min(((start + shift) // W + 1) * W - shift, end)
            spans.append((start, stop))
            start = stop

        return spans

    def encode(self, ids: np.ndarray, keep: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
        """The whole-document path: (n,) ids to (n, d) token states, and the residual stream's first `keep` rows
        after the embedding and after each layer."""
        # ids: (n,)
        X = self.embed(ids)  # (n, d)
        kept = [X[:keep].copy()]  # each (keep, d)
        for index in range(self.config.depth):
            X = np.concatenate([self.layer_block(index, X[start:stop], start) for start, stop in self.block_spans(index, 0, len(ids))])  # (n, d)
            kept.append(X[:keep].copy())

        return rms_norm(X, self.final_norm, self.config.norm_eps), kept  # (n, d), each (keep, d)

    # ------------------------------------------------------------ head and reading

    def head(self, pooled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pooled statistics to a query and a key: (2d,) to (w,), (w,). Unit vectors in float32, or, in Int8 output,
        each vector's int8 codes (SPEC.md, Int8 output), which every other method of a release with that output takes."""
        # pooled: (2d,) float32
        clamp = F32(self.config.head_clamp)
        z = np.clip((pooled - self.input_mean) / self.input_scale, -clamp, clamp)  # (2d,)
        a = self.fc1_weight @ z + self.fc1_bias  # (hh,)
        a = a / (F32(1.0) + np.exp(-a))  # (hh,)
        o = self.fc2_weight @ a + self.fc2_bias  # (2w,)
        assert o.dtype == F32, "the head runs in float32"
        q, k = o[: self.config.head_out], o[self.config.head_out :]  # (w,) each
        if self.output == "int8":
            return quantize_output(q, self.config.act_eps), quantize_output(k, self.config.act_eps)  # (w,) int8 each
        q = q / np.maximum(np.linalg.norm(q), F32(HEAD_NORM_EPS))  # (w,)
        k = k / np.maximum(np.linalg.norm(k), F32(HEAD_NORM_EPS))  # (w,)
        assert q.dtype == F32 and k.dtype == F32, "the head runs in float32"

        return q, k  # (w,), (w,)

    def head_output(self, q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """What `head` returns, from the unit query and key the PyTorch model gives: in Int8 output their codes (a
        unit vector of codes carries them exactly, so quantizing it again returns them), else the vectors as they are."""
        # q, k: (w,) float32
        if self.output == "int8":
            return quantize_output(q, self.config.act_eps), quantize_output(k, self.config.act_eps)  # (w,) int8 each

        return q, k  # (w,), (w,)

    def unit_vectors(self, q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The unit query and key of what `head` returns: float64, `code * inverse_norm` in Int8 output."""
        # q, k: (w,) as `head` returns
        if self.output == "int8":
            return q * inverse_norms(q), k * inverse_norms(k)  # (w,) each

        return q.astype(np.float64), k.astype(np.float64)  # (w,) each

    def bank_arrays_int8(self, q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`bank_arrays` in Int8 output (SPEC.md, Int8 output): four integer dot products with every bank row, each
        `float64(dot) * inverse_norm(text vector)) * inverse_norm(bank vector)`, then the float64 steps of the reading."""
        # q, k: (w,) int8 codes
        assert q.dtype == np.int8 and k.dtype == np.int8, "an int8 output reads int8 codes"
        q64, k64 = q.astype(np.float64), k.astype(np.float64)  # (w,) each
        inverse_q, inverse_k = inverse_norms(q), inverse_norms(k)  # () each
        # Sums of at most w products of integers of at most 127, so exact in float64 whatever the order.
        qq, qk = self.bank_query @ q64, self.bank_key @ q64  # (m,) each: q . q_j, q . k_j
        kq, kk = self.bank_query @ k64, self.bank_key @ k64  # (m,) each: k . q_j, k . k_j
        pair = (qk * inverse_q) * self.bank_inverse_key + (kq * inverse_k) * self.bank_inverse_query  # (m,): cos(q, k_j) + cos(k, q_j)
        logits = self.pair_scale * pair / 2.0 + self.pair_bias  # (m,)
        p = 0.5 * (1.0 + np.tanh(0.5 * logits))  # (m,)
        cosines = ((qq * inverse_q) * self.bank_inverse_query + (kk * inverse_k) * self.bank_inverse_key) / 2.0  # (m,)

        return p, cosines  # (m,), (m,)

    def bank_arrays(self, q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The same-origin probability and the cosine of the text against every bank vector, float64: (m,), (m,)."""
        # q, k: (w,) float32 unit vectors, or int8 codes in Int8 output
        if self.output == "int8":
            return self.bank_arrays_int8(q, k)  # (m,), (m,)
        q64, k64 = q.astype(np.float64), k.astype(np.float64)  # (w,) each
        logits = self.pair_scale * (self.bank_key @ q64 + self.bank_query @ k64) / 2.0 + self.pair_bias  # (m,)
        p = 0.5 * (1.0 + np.tanh(0.5 * logits))  # (m,)
        vector = np.concatenate([q64, k64])  # (2w,)
        cosines = self.bank_unit @ (vector / np.linalg.norm(vector))  # (m,)

        return p, cosines  # (m,), (m,)

    def bank_reading(self, q: np.ndarray, k: np.ndarray, top_k: int = TOP_K) -> BankReading:
        """Mean and spread of the same-origin probabilities per origin, and closeness to the bank, in float64."""
        # q, k: (w,) float32
        p, cosines = self.bank_arrays(q, k)  # (m,), (m,)
        mean, spread = [], []
        for origin in range(3):
            p_origin = p[self.bank_origin == origin]  # (m_o,)
            mean.append(float(p_origin.mean()))
            spread.append(float(p_origin.std()))
        closeness = float(np.sort(cosines)[::-1][:top_k].mean())

        return {"mean": mean, "spread": spread, "closeness": closeness}

    def reading(self, q: np.ndarray, k: np.ndarray, n_tokens: int) -> Reading:
        """The full reading of SPEC.md for a span of `n_tokens` tokens, with its explanation."""
        # q, k: (w,) float32
        assert self.calibration is not None and self.explanation is not None, "a reading needs the release's calibration"
        reading = self.calibration.read(self.bank_reading(q, k, self.calibration.top_k), n_tokens)
        p, cosines = self.bank_arrays(q, k)  # (m,), (m,)
        reading["explain"] = self.explanation.explain(reading, p, cosines)

        return reading

    def chunk_reading(self, pool: ChunkPool, start: int, index: int,
                      windowed: bool = True) -> tuple[ChunkReading, list[WindowReading], dict[str, np.ndarray]]:
        """A finished chunk's reading, its windows' readings (none unless `windowed`) and its vectors; `start` is its
        document position."""
        assert self.calibration is not None, "a reading needs the release's calibration table"
        assert pool.count == pool.length, "a chunk is read once all its tokens are pooled"
        config = self.config
        pooled = pool.pooled(0, pool.length)  # (2d,)
        q, k = self.head(pooled)  # (w,) each
        chunk: ChunkReading = {**self.reading(q, k, pool.length), "start": start, "end": start + pool.length}  # type: ignore[typeddict-item]
        windows: list[WindowReading] = []
        window_pooled, window_queries, window_keys = [], [], []
        spans = window_spans(pool.length, config.window_width, config.window_stride) if windowed else []
        for first, last in spans:
            vector = pool.pooled(first, last)  # (2d,)
            window_q, window_k = self.head(vector)  # (w,) each
            reading = self.calibration.read(self.bank_reading(window_q, window_k, self.calibration.top_k), last - first)
            windows.append({**reading, "start": start + first, "end": start + last, "chunk": index})  # type: ignore[typeddict-item]
            window_pooled.append(vector)
            window_queries.append(window_q)
            window_keys.append(window_k)

        stacked = np.stack(window_pooled) if window_pooled else np.zeros((0, 2 * config.width), dtype=F32)  # (v, 2d)
        vectors = {"pooled": pooled, "query": q, "key": k, "window_pooled": stacked}  # (2d,), (w,), (w,), (v, 2d)
        if windows:
            vectors |= {"window_query": np.stack(window_queries), "window_key": np.stack(window_keys)}  # (v, w) each

        return chunk, windows, vectors

    def document(self, pools: list[ChunkPool], windowed: bool = True) -> tuple[DocumentReading, list[dict[str, np.ndarray]]]:
        """The document's reading from its finished chunk pools, in order, and each chunk's vectors; without the heat
        map (no windows, no cells) unless `windowed`."""
        assert self.calibration is not None, "a reading needs the release's calibration table"
        chunks: list[ChunkReading] = []
        windows: list[WindowReading] = []
        cells: list[Cell] = []
        vectors = []
        start = 0
        for index, pool in enumerate(pools):
            chunk, chunk_windows, chunk_vectors = self.chunk_reading(pool, start, index, windowed)
            chunks.append(chunk)
            windows.extend(chunk_windows)
            if windowed:
                cells.extend(heat_map(chunk_windows, start, pool.length, self.config.window_stride))
            vectors.append(chunk_vectors)
            start += pool.length
        reading: DocumentReading = {"n_tokens": start, "chunk_tokens": self.config.chunk_tokens, "verdict": self.calibration.verdict(chunks),
                                    "chunks": chunks, "windows": windows, "cells": cells}

        return reading, vectors  # per chunk: pooled (2d,), query (w,), key (w,), window_pooled (v, 2d)

    def chunk_pool(self, length: int) -> ChunkPool:
        return ChunkPool(self.config.width, length, self.config.window_width, self.config.window_stride)

    def read_states(self, states: list[np.ndarray], windowed: bool = True) -> tuple[DocumentReading, list[dict[str, np.ndarray]]]:
        """A document from each chunk's token states, each chunk encoded as a text of its own (any encoder's states,
        so the PyTorch model's evaluation reads with this same code)."""
        # states: (m_c, d) float32 each, chunks in order
        pools = []
        for H in states:
            pool = self.chunk_pool(H.shape[0])
            pool.add(H)
            pools.append(pool)

        return self.document(pools, windowed)  # per chunk: pooled (2d,), query (w,), key (w,), window_pooled (v, 2d)

    def read_whole(self, ids: np.ndarray, keep: int = 0) -> tuple[DocumentReading, dict[str, Any]]:
        """Score a document, each chunk read at once; `keep` rows of the first chunk's residual streams come back too."""
        # ids: (n,)
        states, kept = [], []
        for index, (start, end) in enumerate(chunk_bounds(len(ids), self.config.chunk_tokens)):
            H, layers = self.encode(ids[start:end], keep)  # (m, d), each (keep, d)
            states.append(H)
            if index == 0:
                kept = [*layers, H[:keep]]
        reading, vectors = self.read_states(states)
        debug = {f"layer_{index}": X for index, X in enumerate(kept[:-1])}

        return reading, {"chunks": vectors, **debug, "token_states": kept[-1]}  # layer_i and token_states (keep, d) of chunk 0

    def read_streaming(self, ids: np.ndarray, piece: int) -> tuple[DocumentReading, dict[str, Any]]:
        """Score a document fed in pieces of `piece` tokens through a StreamingReader."""
        # ids: (n,)
        reader = StreamingReader(self, len(ids))
        for start in range(0, len(ids), piece):
            reader.feed(ids[start : start + piece])
        reading, vectors = reader.finish()

        return reading, {"chunks": vectors}

    # ------------------------------------------------------------ surprise

    def token_log_probabilities(self, ids: np.ndarray) -> np.ndarray:
        """Each token's log probability under the masked language head with nothing masked, every chunk read as a
        text of its own (SPEC.md, Surprise): (n,) ids to (n,) float64."""
        # ids: (n,)
        if self.language is None:
            raise ValueError("this release has no masked language head")
        rows = self.embed_weight.astype(F32) * self.embed_scale[:, None]  # (V, e)
        out = []
        for start, end in chunk_bounds(len(ids), self.config.chunk_tokens):
            H, _ = self.encode(ids[start:end])  # (c, d)
            out.append(self.language.log_probabilities(H, ids[start:end], rows, self.config))

        return np.concatenate(out)  # (n,)

    def surprise(self, ids: np.ndarray) -> Surprise:
        """The text's surprise reading: its mean token log probability (the unmasked pseudo log likelihood), the
        perplexity it gives, and where it falls among each origin's calibration texts when the release has the table."""
        # ids: (n,)
        mean = float(self.token_log_probabilities(ids).mean())

        return surprise_reading(mean, len(ids), self.surprise_table)


def window_context_gap(model: ReferenceModel, ids: np.ndarray, states_of: Any) -> list[dict[str, float]]:
    """Each window of a single-chunk text read two ways: pooled from the chunk's states (as the heat map reads it, its
    tokens having seen neighbors outside the window) and as a standalone text of those tokens (as training reads one).

    `states_of(ids)` gives the token states of a standalone text, (n,) to (n, d) float32, from any encoder. Per
    window: the pooled vector's largest `abs / (1 + abs)` gap, the gaps of the query and key and of the strip
    position, and whether the 0.10 conformal sets and the side of 0.5 agree.
    """
    # ids: (n,)
    assert model.calibration is not None, "a reading needs the release's calibration table"
    config = model.config
    assert len(ids) <= config.chunk_tokens, "the gap is measured inside one chunk"
    pool = model.chunk_pool(len(ids))
    pool.add(states_of(ids))
    rows = []
    for first, last in window_spans(len(ids), config.window_width, config.window_stride):
        if (first, last) == (0, len(ids)):
            continue  # the whole text is its own standalone reading
        alone = model.chunk_pool(last - first)
        alone.add(states_of(ids[first:last]))
        in_context, standalone = pool.pooled(first, last), alone.pooled(0, last - first)  # (2d,) each
        readings = []
        for vector in (in_context, standalone):
            q, k = model.head(vector)  # (w,) each
            readings.append((q, k, model.calibration.read(model.bank_reading(q, k, model.calibration.top_k), last - first)))
        (q1, k1, one), (q2, k2, two) = readings
        key = level_key(STRIP_LEVEL)
        rows.append({
            "start": first, "end": last,
            "pooled_gap": float((np.abs(in_context - standalone) / (1.0 + np.abs(standalone))).max()),
            "query_key_gap": float(max(np.abs(a - b).max() for a, b in zip(model.unit_vectors(q1, k1), model.unit_vectors(q2, k2), strict=True))),
            "position_gap": abs(one["strip"]["position"] - two["strip"]["position"]),
            "same_set": float(one["sets"][key] == two["sets"][key]),
            "same_side": float((one["strip"]["position"] >= 0.5) == (two["strip"]["position"] >= 0.5)),
        })

    return rows


class StreamingReader:
    """A document of any length in bounded memory (SPEC.md, section 5): chunk by chunk, each layer holding only its
    open block. The document's length is known up front (its ids are), so the chunk bounds are too."""

    def __init__(self, model: ReferenceModel, n_tokens: int) -> None:
        self.model = model
        self.bounds = chunk_bounds(n_tokens, model.config.chunk_tokens)
        self.chunk = 0  # the chunk being read
        self.pools: list[ChunkPool] = []  # finished chunks; the engine reads and drops each at once
        self._start_chunk()

    def _start_chunk(self) -> None:
        depth = self.model.config.depth
        start, end = self.bounds[self.chunk]
        self.open: list[list[np.ndarray]] = [[] for _ in range(depth)]  # rows of each layer's open block
        self.open_start = [0] * depth  # chunk position of each open block's first row
        self.received = [0] * depth  # next chunk position each layer receives
        self.fed = 0  # tokens of this chunk fed so far
        self.pool = self.model.chunk_pool(end - start)

    def feed(self, ids: np.ndarray) -> None:
        # ids: (c,)
        first = 0
        while first < ids.size:
            if self.chunk == len(self.bounds):
                raise ValueError("fed more tokens than the document holds")
            start, end = self.bounds[self.chunk]
            take = min(end - start - self.fed, ids.size - first)
            self._push(0, self.model.embed(ids[first : first + take]))  # (t, d)
            self.fed += take
            first += take
            if self.fed == end - start:
                self._end_chunk()

    def _end_chunk(self) -> None:
        for index in range(self.model.config.depth):
            if self.open[index]:
                self._close(index)
        assert self.pool.count == self.pool.length, "every token of the chunk reached the pool"
        self.pools.append(self.pool)
        self.chunk += 1
        if self.chunk < len(self.bounds):
            self._start_chunk()

    def _push(self, index: int, X: np.ndarray) -> None:
        # X: (c, d), the next rows of layer `index` in position order
        config = self.model.config
        W, shift = config.block_size, config.shift(index)
        first = 0
        while first < X.shape[0]:
            position = self.received[index]
            if self.open[index] and (position + shift) % W == 0:
                self._close(index)
            room = W - (position + shift) % W  # rows until the next boundary
            take = X[first : first + room]  # (t, d)
            if not self.open[index]:
                self.open_start[index] = position
            self.open[index].extend(take)
            self.received[index] += take.shape[0]
            first += take.shape[0]

    def _close(self, index: int) -> None:
        block = np.stack(self.open[index])  # (t, d)
        start = self.open_start[index]
        self.open[index] = []
        outputs = self.model.layer_block(index, block, start)  # (t, d)
        if index + 1 < self.model.config.depth:
            self._push(index + 1, outputs)
        else:
            assert start == self.pool.count, "token states reach the pool in position order"
            self.pool.add(rms_norm(outputs, self.model.final_norm, self.model.config.norm_eps))

    def finish(self) -> tuple[DocumentReading, list[dict[str, np.ndarray]]]:
        if self.chunk != len(self.bounds):
            raise ValueError("fed fewer tokens than the document holds")

        return self.model.document(self.pools)  # per chunk: pooled (2d,), query (w,), key (w,), window_pooled (v, 2d)


# ---------------------------------------------------------------- random releases and fixtures


def random_tensors(config: StudentConfig, bank_per_origin: int, seed: int, output: str = "float") -> dict[str, np.ndarray]:
    """A random release's tensors: latent float weights ternarized and packed as training's export does. With int8
    `output` the bank is the same random unit vectors as int8 codes (`quantize_output`), each half with its own scale."""
    generator = np.random.default_rng(seed)
    d, f, e, w = config.width, config.ffn_width, config.embed_width, config.head_out
    tensors: dict[str, np.ndarray] = {}

    def gain(width: int) -> np.ndarray:
        return (1.0 + 0.1 * generator.standard_normal(width)).astype(F32)  # (width,)

    def ternary(name: str, outputs: int, inputs: int) -> None:
        T, gamma = ternarize(generator.standard_normal((outputs, inputs)) / math.sqrt(inputs), config.weight_eps)  # (o, i)
        tensors[name + ".weight"] = pack_ternary(T)  # (o, i/4)
        tensors[name + ".scale"] = np.asarray([gamma], dtype=F32)  # (1,)

    tensors["embed.weight"], tensors["embed.scale"] = quantize_embedding(generator.standard_normal((config.vocab_size, e)))
    if config.projected:
        tensors["embed.proj_norm"] = gain(e)
        ternary("embed.proj", d, e)
    for index in range(config.depth):
        prefix = f"layers.{index}."
        tensors[prefix + "attn_norm"] = gain(d)
        for name in ("q", "k", "v", "o"):
            ternary(prefix + "attn." + name, d, d)
        tensors[prefix + "attn_out_norm"] = gain(d)
        tensors[prefix + "ffn_norm"] = gain(d)
        ternary(prefix + "ffn.up", f, d)
        tensors[prefix + "ffn_out_norm"] = gain(f)
        ternary(prefix + "ffn.down", d, f)
    tensors["final_norm"] = gain(d)
    tensors["head.input_mean"] = np.concatenate([0.1 * generator.standard_normal(d), np.full(d, 0.5)]).astype(F32)  # (2d,)
    tensors["head.input_scale"] = np.exp(0.2 * generator.standard_normal(2 * d)).astype(F32)  # (2d,)
    tensors["head.fc1.weight"] = (generator.standard_normal((config.head_hidden, 2 * d)) / math.sqrt(2 * d)).astype(np.float16)
    tensors["head.fc1.bias"] = (0.1 * generator.standard_normal(config.head_hidden)).astype(np.float16)
    tensors["head.fc2.weight"] = (generator.standard_normal((2 * w, config.head_hidden)) / math.sqrt(config.head_hidden)).astype(np.float16)
    tensors["head.fc2.bias"] = (0.1 * generator.standard_normal(2 * w)).astype(np.float16)
    tensors["head.pair"] = np.asarray([20.0, -0.5], dtype=F32)  # (2,)
    bank = generator.standard_normal((3 * bank_per_origin, 2, w))  # (m, 2, w)
    bank /= np.linalg.norm(bank, axis=-1, keepdims=True)
    if output == "int8":
        codes = quantize_output(bank.astype(F32), config.act_eps)  # (m, 2, w) int8
        tensors["bank.vectors"] = codes.reshape(3 * bank_per_origin, 2 * w)  # (m, 2w) int8
    else:
        tensors["bank.vectors"] = bank.reshape(3 * bank_per_origin, 2 * w).astype(np.float16)  # (m, 2w)
    tensors["bank.origin"] = np.repeat(np.arange(3, dtype=np.uint8), bank_per_origin)  # (m,)
    tensors["bank.tokens"] = generator.integers(1, 5000, 3 * bank_per_origin).astype(np.int32)  # (m,)
    for field, names in RANDOM_BANK_FIELDS.items():
        tensors[f"bank.meta.{field}"] = generator.integers(0, len(names), 3 * bank_per_origin).astype(np.uint16)  # (m,)
    # The masked language head, drawn last so every tensor above is what it was before the head existed.
    tensors["mlm.proj_norm"] = gain(d)
    ternary("mlm.proj", e, d)
    tensors["mlm.norm"] = gain(e)
    tensors["mlm.scale"] = np.asarray([1.0 / math.sqrt(e)], dtype=F32)  # (1,)
    tensors["mlm.bias"] = (0.5 * generator.standard_normal(config.vocab_size)).astype(F32)  # (V,)

    return tensors  # name to array (SPEC.md, section 6)


READABLE_FORMAT_VERSIONS = (FORMAT_VERSION, INTEGER_ATTENTION_FORMAT_VERSION, INT8_OUTPUT_FORMAT_VERSION)


def release_format_version(attention: str, output: str) -> int:
    """The format a release with this attention and output is: 1 for float attention and float output, 2 for integer
    attention with float output, 3 for int8 output (which names both)."""
    if attention not in ATTENTION_MODES:
        raise ValueError(f"attention {attention!r} is not one of {ATTENTION_MODES}")
    if output not in OUTPUT_MODES:
        raise ValueError(f"output {output!r} is not one of {OUTPUT_MODES}")
    if output == "int8":
        return INT8_OUTPUT_FORMAT_VERSION

    return FORMAT_VERSION if attention == "float" else INTEGER_ATTENTION_FORMAT_VERSION


def release_attention(config: dict[str, Any], where: str = "config.json") -> str:
    """The attention a release's `config.json` names (SPEC.md, section 6): format 1 is float attention and has no
    `model.attention`; formats 2 and 3 name it in `model.attention`. Anything else fails."""
    if config["format"] != FORMAT or config["format_version"] not in READABLE_FORMAT_VERSIONS:
        raise ValueError(f"{where} is not a {FORMAT} release of format {', '.join(map(str, READABLE_FORMAT_VERSIONS))}")
    attention = config["model"].get("attention")
    if config["format_version"] == FORMAT_VERSION:
        if attention is not None:
            raise ValueError(f"{where}: model.attention needs format {INTEGER_ATTENTION_FORMAT_VERSION} or later")
        return "float"
    if attention not in ATTENTION_MODES:
        raise ValueError(f"{where}: format {config['format_version']} names model.attention, one of {ATTENTION_MODES}, not {attention!r}")

    return attention


def release_output(config: dict[str, Any], where: str = "config.json") -> str:
    """The head's output a release's `config.json` names: formats 1 and 2 have no `model.output` and are float;
    format 3 names it, and it is `"int8"`. Anything else fails."""
    if config["format"] != FORMAT or config["format_version"] not in READABLE_FORMAT_VERSIONS:
        raise ValueError(f"{where} is not a {FORMAT} release of format {', '.join(map(str, READABLE_FORMAT_VERSIONS))}")
    output = config["model"].get("output")
    if config["format_version"] != INT8_OUTPUT_FORMAT_VERSION:
        if output is not None:
            raise ValueError(f"{where}: model.output needs format {INT8_OUTPUT_FORMAT_VERSION}")
        return "float"
    if output != "int8":
        raise ValueError(f"{where}: format {INT8_OUTPUT_FORMAT_VERSION} names model.output, \"int8\", not {output!r}")

    return output


def release_config(config: StudentConfig, bank_counts: list[int], bank_split: str, bank_fields: dict[str, list[str]],
                   calibration: dict[str, Any] | None, provenance: dict[str, Any], attention: str = "float",
                   output: str = "float") -> dict[str, Any]:
    """The `config.json` of a release (SPEC.md, section 6). A float release is format 1, unchanged; an
    integer-attention release is format 2, which names its attention in `model.attention`; an int8-output release is
    format 3, which names `model.attention` and `model.output`."""
    version = release_format_version(attention, output)
    model = config.to_json()
    if version >= INTEGER_ATTENTION_FORMAT_VERSION:
        model["attention"] = attention
    if version == INT8_OUTPUT_FORMAT_VERSION:
        model["output"] = output

    return {
        "format": FORMAT, "format_version": version,
        "license": LICENSE,
        "tokenizer": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION, "sha256": TOKENIZER_SHA256,
                      "add_special_tokens": False},
        "model": model,
        "head": {"hidden": config.head_hidden, "out_width": config.head_out, "clamp": config.head_clamp},
        "reading": {"chunk_tokens": config.chunk_tokens, "window_width": config.window_width, "window_stride": config.window_stride},
        "bank": {"origins": list(ORIGINS), "counts": bank_counts, "split": bank_split, "fields": bank_fields},
        "calibration": calibration,
        "provenance": provenance,
    }


def write_release(directory: Path, tensors: dict[str, np.ndarray], config_json: dict[str, Any]) -> None:
    # tensors: (...) each, by name (SPEC.md, section 6)
    directory.mkdir(parents=True, exist_ok=True)
    write_safetensors(directory / "model.safetensors", tensors, {"format": FORMAT, "format_version": str(config_json["format_version"])})
    # The config names the weights it belongs to, so a reader of both files can check them against each other.
    weights = {"file": "model.safetensors", "sha256": hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()}
    temporary = directory / "config.json.partial"
    temporary.write_text(json.dumps({**config_json, "weights": weights}, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, directory / "config.json")


def teacher_calibration() -> Any:
    """The teacher's `litmus_teacher.calibrate` (NumPy only), from the sibling folder; only fitting needs it."""
    source = str(Path(__file__).resolve().parent.parent / "teacher" / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import litmus_teacher.calibrate as calibrate

    return calibrate


def fitted_calibration(mean: np.ndarray, spread: np.ndarray, closeness: np.ndarray, tokens: np.ndarray,
                       labels: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The teacher's table fitted on calibration readings, and the sorted scores behind each of its groups."""
    # mean, spread: (n_c, 3); closeness, tokens, labels: (n_c,)
    calibrate = teacher_calibration()
    table = calibrate.fit_calibration(calibrate.Readings(mean, spread, closeness, tokens), labels)

    return table, explain_tensors(table, mean, closeness, labels, tokens)  # the table, and (...) each by name


def synthetic_calibration(model: ReferenceModel, max_tokens: int, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A calibration table for a random model, fitted by the teacher's `fit_calibration` on synthetic readings.

    The readings are the model's own readings of random token sequences, resampled with noise into 3,000 rows of
    every length, each row's mean against its own origin's bank raised by a third, so that p-values, sets and
    intervals vary across a fixture. It tests arithmetic only and calibrates nothing.
    """
    generator = np.random.default_rng(seed)
    lengths = np.unique(np.exp(generator.uniform(0.0, math.log(max_tokens), 16)).astype(np.int64) + 1)  # (r,)
    observed = []
    for n_tokens in lengths:
        H, _ = model.encode(generator.integers(0, model.config.vocab_size, int(n_tokens)))  # (n, d)
        H64 = H.astype(np.float64)  # (n, d)
        pooled = pooled_statistics(H64.sum(axis=0), (H64 * H64).sum(axis=0), len(H))  # (2d,)
        observed.append(model.bank_reading(*model.head(pooled), TOP_K))
    rows = 3000
    pick = generator.integers(0, len(observed), rows)  # (n_c,)
    mean = np.asarray([observed[index]["mean"] for index in pick]) * np.exp(generator.normal(0.0, 0.15, (rows, 3)))  # (n_c, 3)
    labels = generator.integers(0, 3, rows)  # (n_c,)
    mean[np.arange(rows), labels] *= 4.0 / 3.0
    spread = np.asarray([observed[index]["spread"] for index in pick]) * np.exp(generator.normal(0.0, 0.1, (rows, 3)))  # (n_c, 3)
    observed_closeness = np.asarray([reading["closeness"] for reading in observed])  # (r,)
    # Calibration texts sit a little closer to the bank than the fixture's, so some fixture readings are out of reference.
    closeness = observed_closeness[pick] + generator.normal(0.0, 0.5, rows) * observed_closeness.std() + 1.5 * observed_closeness.std()  # (n_c,)
    tokens = np.exp(generator.uniform(0.0, math.log(model.config.chunk_tokens), rows)).astype(np.int64) + 1  # (n_c,) no text is longer than a chunk

    return fitted_calibration(np.clip(mean, 1e-6, 1.0), spread, closeness, tokens, labels)  # the table, and (...) each by name


def synthetic_surprise_table(model: ReferenceModel, seed: int) -> dict[str, Any]:
    """A surprise table for a random model: its mean token log probabilities of a few random sequences, resampled
    with noise and given random origins, each origin's shifted apart, so fixture percentiles vary. Tests arithmetic only."""
    generator = np.random.default_rng(seed)
    observed = np.asarray([model.token_log_probabilities(generator.integers(0, model.config.vocab_size, n)).mean()
                           for n in (3, 17, 64, 200)])  # (4,)
    labels = generator.integers(0, 3, 600)  # (r,)
    values = generator.choice(observed, 600) + generator.normal(0.0, 0.3, 600) + 0.4 * labels  # (r,)

    return surprise_table(values, labels, "random")


def init_release(config: StudentConfig, directory: Path, seed: int, bank_per_origin: int, calibration_tokens: int,
                 attention: str = "float", output: str = "float") -> ReferenceModel:
    """Write a random release with a synthetic calibration table, and return it loaded."""
    tensors = random_tensors(config, bank_per_origin, seed, output)
    provenance = {"kind": "random", "seed": seed}
    uncalibrated = ReferenceModel(config, tensors, None, attention=attention, output=output)
    calibration, explained = synthetic_calibration(uncalibrated, calibration_tokens, seed + 1)
    config_json = release_config(config, [bank_per_origin] * 3, "random", RANDOM_BANK_FIELDS, calibration, provenance, attention, output)
    config_json["surprise"] = synthetic_surprise_table(uncalibrated, seed + 2)
    write_release(directory, {**tensors, **explained}, config_json)

    return ReferenceModel.load(directory)


def fixture_texts(long_words: int) -> list[str]:
    """Texts that exercise the tokenizer, the standard reading, blocks, partial blocks, windows and (in `tiny`) chunks."""
    generator = np.random.default_rng(7)
    words = ["the", "river", "model", "token", "block", "window", "measure", "quiet", "ledger", "signal", "careful", "protein",
             "long", "short", "reading", "section", "bank", "origin", "human", "written", "several", "during", "because",
             "although", "however"]
    sentences = []
    for _ in range(long_words // 12):
        chosen = generator.choice(words, 12)
        sentences.append(" ".join(chosen).capitalize() + ".")

    return [
        "Hello, world.",
        "a",
        ("The committee met on Tuesday to review the proposal. After a long discussion about costs and timelines, "
         "the members agreed to postpone the vote until the next quarter, when more data would be available. "
         "Several members asked for a clearer budget, and the chair promised one within two weeks."),
        "Line one\r\nLine two  \r\n\r\n\r\n\r\n\r\nAfter blank lines\u200b with a zero\u2060width\x07 mark\t \n",
        "Special tokens stay literal text to the reader but not to the tokenizer: [CLS] [SEP] [MASK] <|endoftext|>",
        "日本語の文章です。 Ceci est une phrase en français. هذه جملة عربية. Emoji: \U0001f600 caf\u0065\u0301.",
        "def pooled(states, mask):\n    counts = mask.sum(1)\n    return (states * mask).sum(1) / counts\n",
        " ".join(sentences),
    ]


def fixture_record(model: ReferenceModel, tokenizer: Any, text: str, keep: int) -> dict[str, Any]:
    ids = token_ids(tokenizer, text)  # (n,)
    reading, vectors = model.read_whole(ids, keep)

    chunks = []
    for chunk, values in zip(reading["chunks"], vectors["chunks"], strict=True):
        record = {"start": chunk["start"], "end": chunk["end"], "pooled": values["pooled"].tolist(), "query": values["query"].tolist(),
                  "key": values["key"].tolist(), "window_pooled": values["window_pooled"].tolist()}
        if model.output == "int8":  # each window's codes too, so the engine's test can tell a window whose code a rounding moved
            record |= {"window_query": values["window_query"].tolist(), "window_key": values["window_key"].tolist()}
        chunks.append(record)

    record = {
        "text": text, "standard_reading": standard_reading(text), "ids": ids.tolist(),
        "layer_states": [vectors[f"layer_{index}"].tolist() for index in range(model.config.depth + 1)],
        "token_states": vectors["token_states"].tolist(),
        "chunks": chunks,
        "reading": reading,
    }
    if model.language is not None:  # the surprise reading (SPEC.md, Surprise)
        record["log_probabilities"] = model.token_log_probabilities(ids).tolist()
        record["surprise"] = model.surprise(ids)

    return record


def combination_cases(model: ReferenceModel, seed: int) -> list[dict[str, Any]]:
    """Chunk readings chosen to exercise the verdict's combination, mixed-document signal included, and the verdict
    of each: the engine's check of `combine_chunks` apart from any model."""
    assert model.calibration is not None, "the verdict needs the release's calibration table"
    calibration = model.calibration
    generator = np.random.default_rng(seed)
    grid = [*calibration.grid, 1.0]

    def chunk(p_values: list[float]) -> dict[str, Any]:
        p0, p1 = np.sort(generator.uniform(0.0, 1.0, 2))
        three_way = generator.dirichlet(np.ones(3))
        return {"length_bin": 9, "p_values": p_values,
                "sets": {level_key(level): [name for origin, name in enumerate(ORIGINS) if p_values[origin] > level] for level in calibration.levels},
                "probability": float(p1 / (1.0 - p0 + p1)), "interval": [float(p0), float(p1)], "three_way": three_way.tolist(),
                "spread_percentile": [0.5, 0.5, 0.5], "closeness_p": float(generator.choice(grid)), "out_of_reference": False}

    def random_p() -> list[float]:
        return [float(value) for value in generator.choice(grid, 3)]

    cases = [
        [chunk([0.02, 0.9, 0.5]), chunk([0.8, 0.01, 0.02])],  # a model chunk and a human chunk at 0.05 / 2: mixed
        [chunk([0.03, 0.9, 0.5]), chunk([0.8, 0.01, 0.02])],  # 0.03 > 0.025: not a model chunk, not mixed
        [chunk([0.001, 0.001, 0.001]), chunk([0.5, 0.5, 0.5])],  # one chunk both, no other: not mixed
        [chunk([0.01, 0.6, 0.3]) for _ in range(3)],  # agreeing chunks
        [chunk(random_p())],  # one chunk: its own reading
        [chunk(random_p()) for _ in range(16)],
    ]

    return [{"chunks": chunks, "verdict": calibration.verdict(chunks)} for chunks in cases]  # type: ignore[arg-type]


FIXTURE_SETS = {
    # name: (config, bank texts per origin, longest random sequence read for the synthetic calibration, words of the long text,
    # attention, output). The int8-output sets are the same weights as `tiny` and `s192_l4_integer` in format 3: one with
    # float attention, one with integer attention, so each pairing of attention and output is read.
    "tiny": ("tiny", 8, 200, 300, "float", "float"),
    "s192_l4": ("s192_l4", 64, 1500, 2400, "float", "float"),
    "tiny_integer": ("tiny", 8, 200, 300, "integer", "float"),
    "s192_l4_integer": ("s192_l4", 64, 1500, 2400, "integer", "float"),
    "tiny_int8": ("tiny", 8, 200, 300, "float", "int8"),
    "s192_l4_integer_int8": ("s192_l4", 64, 1500, 2400, "integer", "int8"),
}


def write_fixtures(out: Path, tokenizer: Any) -> None:
    for name, (config_name, bank_per_origin, calibration_tokens, long_words, attention, output) in FIXTURE_SETS.items():
        directory = out / name
        model = init_release(named_config(config_name), directory, 0, bank_per_origin, calibration_tokens, attention, output)
        records = [fixture_record(model, tokenizer, text, keep=4) for text in fixture_texts(long_words)]
        version = release_format_version(attention, output)
        fixture = {"format": FORMAT, "format_version": version, "config": config_name, "keep_rows": 4, "texts": records,
                   "combinations": combination_cases(model, 5)}
        (directory / "fixtures.json").write_text(json.dumps(fixture) + "\n", encoding="utf-8")
        print(f"{directory}: {len(records)} texts, {sum(len(record['ids']) for record in records)} tokens", file=sys.stderr)


# ---------------------------------------------------------------- command line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="write a random release for a config")
    init.add_argument("--config", required=True, help=f"a name in configs.json ({', '.join(config_names())}) or a JSON file of StudentConfig fields")
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--bank-per-origin", type=int, default=64)
    init.add_argument("--attention", choices=ATTENTION_MODES, default="float", help="integer: the optional mode of SPEC.md, Integer attention (format 2)")
    init.add_argument("--output", choices=OUTPUT_MODES, default="float", help="int8: the head's int8 codes and an int8 bank, SPEC.md, Int8 output (format 3)")
    fixtures = commands.add_parser("fixtures", help="write the engine's random releases and fixtures")
    fixtures.add_argument("--out", type=Path, required=True)
    fixtures.add_argument("--tokenizer", type=Path, default=None, help="a local tokenizer.json; default: the pinned Hub revision")
    score = commands.add_parser("score", help="print the reading of one text")
    score.add_argument("--release", type=Path, required=True)
    score.add_argument("--text", type=Path, required=True)
    score.add_argument("--feed", type=int, default=0, help="stream the ids in pieces of this many tokens; 0 reads each chunk at once")
    score.add_argument("--tokenizer", type=Path, default=None)
    arguments = parser.parse_args()

    if arguments.command == "init":
        path = Path(arguments.config)
        config = StudentConfig(**json.loads(path.read_text(encoding="utf-8"))) if path.suffix == ".json" else named_config(arguments.config)
        init_release(config, arguments.out, arguments.seed, arguments.bank_per_origin, 1000, arguments.attention, arguments.output)
    elif arguments.command == "fixtures":
        write_fixtures(arguments.out, load_tokenizer(arguments.tokenizer))
    else:
        model = ReferenceModel.load(arguments.release)
        ids = token_ids(load_tokenizer(arguments.tokenizer), arguments.text.read_text(encoding="utf-8"), release_reading(arguments.release))  # (n,)
        reading, _ = model.read_streaming(ids, arguments.feed) if arguments.feed else model.read_whole(ids)
        print(json.dumps(reading, indent=1))


if __name__ == "__main__":
    main()
