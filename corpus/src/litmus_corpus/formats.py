"""The format stratum of a text, by rules tried in order; the first that holds names the format.

    chat      a chat template's tokens, a reasoning block's tags, or two or more lines that open with a role
              (`User:`, `Assistant:`)
    code      half or more of the characters inside code fences; or, without fences, three or more lines of
              which half or more read as code (a keyword opening, a brace or semicolon closing)
    table     three or more lines of which half or more are table rows (`| a | b |`, or two or more tabs)
    math      LaTeX marks (`$...$`, `\\frac`, `\\begin{equation}`), three or more and at least one per 100 words
    list      two or more list items making half or more of the lines
    markdown  two or more marks of Markdown (headings, emphasis, fences, links, quotes, list items, table rows)
    plain     anything else

Only the first `FORMAT_CHARACTERS` characters are read, which bounds the cost on a book and does not change
the answer for one.
"""

from __future__ import annotations

import re


FORMAT_CHARACTERS = 100_000
CHAT_TEMPLATE_PATTERN = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|start_header_id\|>|<\|eot_id\|>|\[/?INST\]|<start_of_turn>|<end_of_turn>"
    r"|<\|(?:user|assistant|system)\|>|<\|endoftext\|>"
)
REASONING_TAG_PATTERN = re.compile(r"</?(?:think|thinking|reasoning|thought|scratchpad)>|<\|(?:begin|end)_of_thought\|>", re.IGNORECASE)
ROLE_LINE_PATTERN = re.compile(r"^(?:#{1,3}[ \t]*)?(?:user|assistant|human|ai|system|gpt|chatgpt|bot|model)[ \t]*:", re.IGNORECASE | re.MULTILINE)
FENCE_PATTERN = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?(?:^[ \t]*\1[ \t]*$|\Z)", re.DOTALL | re.MULTILINE)
CODE_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:def |class |import |from [\w.]+ import |return\b|if ?\(|for ?\(|while ?\(|#include|#!|public |private |protected "
    r"|static |function\b|const |let |var |package |using |namespace |fn |impl |struct |SELECT |INSERT |CREATE |@\w+|//|/\*|\*/|[{}])"
    r"|[;{}][ \t]*$"
)
TABLE_LINE_PATTERN = re.compile(r"^[ \t]*\|.*\|[ \t]*$|^[^\t\n]*\t[^\t\n]*\t")
MATH_PATTERN = re.compile(
    r"\$\$|\\\[|\\\(|\\begin\{(?:equation|align|gather|eqnarray|math)|\\(?:frac|sum|int|sqrt|alpha|beta|gamma|theta|lambda|cdot|leq|geq|mathbb|mathrm)\b"
    r"|\^\{|_\{|\$[^$\n]{1,80}\$"
)
LIST_LINE_PATTERN = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,3}[.)]|[a-zA-Z][.)])[ \t]+\S")
MARKDOWN_PATTERN = re.compile(
    r"^#{1,6}[ \t]+\S|\*\*[^*\n]+\*\*|__[^_\n]+__|^[ \t]*>[ \t]|\[[^\]\n]+\]\([^)\s]+\)|^[ \t]*(?:```|~~~)"
    r"|^[ \t]*(?:[-*+]|\d{1,3}\.)[ \t]+\S|^[ \t]*\|.*\|[ \t]*$|`[^`\n]+`",
    re.MULTILINE,
)
MIN_CHAT_ROLE_LINES = 2
MIN_STRUCTURED_LINES = 3
STRUCTURED_SHARE = 0.5
MIN_MATH_MARKS = 3
WORDS_PER_MATH_MARK = 100
MIN_LIST_ITEMS = 2
MIN_MARKDOWN_MARKS = 2


def share_of_lines(lines: list[str], pattern: re.Pattern[str]) -> tuple[int, float]:
    hits = sum(1 for line in lines if pattern.search(line))

    return hits, hits / max(len(lines), 1)


def text_format(text: str) -> str:
    """The format stratum of a text."""
    head = text[:FORMAT_CHARACTERS]
    if CHAT_TEMPLATE_PATTERN.search(head) or REASONING_TAG_PATTERN.search(head):
        return "chat"
    if len(ROLE_LINE_PATTERN.findall(head)) >= MIN_CHAT_ROLE_LINES:
        return "chat"
    fenced = sum(match.end() - match.start() for match in FENCE_PATTERN.finditer(head))
    if fenced >= STRUCTURED_SHARE * len(head):
        return "code"
    lines = [line for line in head.split("\n") if line.strip()]
    if fenced == 0 and len(lines) >= MIN_STRUCTURED_LINES:
        _, code_share = share_of_lines(lines, CODE_LINE_PATTERN)
        if code_share >= STRUCTURED_SHARE:
            return "code"
    table_lines, table_share = share_of_lines(lines, TABLE_LINE_PATTERN)
    if table_lines >= MIN_STRUCTURED_LINES and table_share >= STRUCTURED_SHARE:
        return "table"
    math_marks = len(MATH_PATTERN.findall(head))
    if math_marks >= max(MIN_MATH_MARKS, len(head.split()) / WORDS_PER_MATH_MARK):
        return "math"
    list_lines, list_share = share_of_lines(lines, LIST_LINE_PATTERN)
    if list_lines >= MIN_LIST_ITEMS and list_share >= STRUCTURED_SHARE:
        return "list"
    if len(MARKDOWN_PATTERN.findall(head)) >= MIN_MARKDOWN_MARKS:
        return "markdown"

    return "plain"


def cuts_at_lines_only(text_format_name: str) -> bool:
    """Code and tables are cut between lines, never inside one."""
    return text_format_name in ("code", "table")
