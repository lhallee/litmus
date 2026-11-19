"""Class tells: marks a dataset's making left in one class and not the other, removed by named rules.

The text is kept raw; these rules remove only what a writer did not put there. Every rule reads every class
the same way, and the report counts each by source and origin with a few examples of what it removed, so a
reader can see where a rule fell on one class. Benchmark rows keep their text as the benchmark has it.

Rules that drop a text (the rest of it cannot be trusted):

    null_string            the text is a stored null: `None`, `nan`, `[deleted]` and the like
    replacement_character  U+FFFD: bytes were lost in an export
    api_error              a short text that is an API's error message
    refusal                a short text that opens by declining the request
    anonymized             a dataset's anonymization marks (`NAME_1`, `<PERSON>`)
    template_slot          a slot to fill (`[Your Name]`) inside a line of text
    truncated_fragment     cut at a fixed length, with too little left once the cut-off sentence is removed

Rules that rewrite a text:

    mojibake               UTF-8 read as Windows-1252 (`â€™`), decoded again
    export_escapes         a text with no line breaks and literal `\\n` escapes, unescaped
    html_entities          `&quot;` and the like in a text without HTML, unescaped
    html_breaks            `<br>` in a text without other HTML, made a line break
    gutenberg_frame        Project Gutenberg's header and licence around a book
    repeated_header        a first line that 20% or more of a shard's texts share, removed
    repeated_footer        the same for a last line
    response_label         `Answer:` or `Response:` opening a text
    prompt_echo            a reply that opens by repeating its prompt
    template_line          a line that is only a slot to fill (`[Your Name]` under a letter's closing)
    truncation_marker      `[truncated]` or `(continued)` closing a text
    fixed_length_cut       a text at a shard's truncation length (one length held by many texts at the top of
                           the range), cut back to its last sentence or line end
    unfinished_ending      a text whose last line is prose stopping mid-sentence (a generation limit), cut back
                           to its last sentence or line end when that keeps half of it; otherwise left whole
"""

from __future__ import annotations

import html
import re
import numpy as np

from collections import Counter

from .storage import Report


NULL_STRINGS = frozenset({"none", "null", "nan", "n/a", "na", "[deleted]", "[removed]", "undefined", "<empty>"})
API_ERROR_CHARACTERS = 400
API_ERROR_PATTERN = re.compile(
    r"^(?:error|exception|traceback)\b|rate ?limit|context_length_exceeded|maximum context length|internal server error"
    r"|bad gateway|service unavailable|\bhttp(?:error)?:? ?[45]\d\d\b|openai\.error|request timed out|failed to generate"
    r"|^an error (?:has )?occurred",
    re.IGNORECASE,
)
REFUSAL_WORDS = 80
REFUSAL_PATTERN = re.compile(
    r"^(?:i'?m sorry|i am sorry|sorry, (?:but )?i\b|i apologi[sz]e|as an ai\b"
    r"|i can(?:no|')t (?:help|assist|provide|create|write|generate|fulfill|comply|do that|engage|answer)"
    r"|i (?:won'?t|will not|am unable to|'m unable to|am not able to|'m not able to))",
    re.IGNORECASE,
)
ANONYMIZED_PATTERN = re.compile(
    r"\bNAME_\d+\b|<(?:PERSON|NAME|EMAIL|PHONE|URL|ADDRESS|ORGANIZATION)>|\[(?:REDACTED|PERSON|EMAIL_ADDRESS|PHONE_NUMBER)\]"
)
SLOT_PATTERN = re.compile(
    r"\[(?:your|my|the|insert|name|first|last|full|recipient|sender|company|organization|date|address|city|state|phone|email"
    r"|title|position|job|contact|subject|product|brand|author|location|time|number|link|url|website|customer|client|manager"
    r"|employee|school|university|teacher|student|signature|amount|item|order)\b[^\]\n]{0,40}\]",
    re.IGNORECASE,
)
SLOT_LINE_PATTERN = re.compile(r"^[ \t]*(?:" + SLOT_PATTERN.pattern + r")[ \t]*$\n?", re.IGNORECASE | re.MULTILINE)
CP1252_CONTINUATION = "\u0080-¿ŒœŠšŸŽžƒˆ˜–—‘-‚“-„†-•…‰‹›€™"
MOJIBAKE_PATTERN = re.compile(f"[Â-ô][{CP1252_CONTINUATION}]{{1,3}}")
ENTITY_PATTERN = re.compile(r"&(?:quot|amp|lt|gt|nbsp|apos|#\d{1,6}|#x[0-9a-fA-F]{1,6});")
HTML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
GUTENBERG_START = re.compile(r"^.*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$\n?", re.IGNORECASE | re.MULTILINE)
GUTENBERG_END = re.compile(r"^.*(?:\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG|End of (?:the )?Project Gutenberg).*$", re.IGNORECASE | re.MULTILINE)
RESPONSE_LABEL_PATTERN = re.compile(r"^(?:#{1,4}[ \t]*)?(?:response|answer|output|assistant|chatgpt|gpt|bot)[ \t]*:[ \t]*\n?", re.IGNORECASE)
# Read against a text's last characters only: a search of the whole text for `whitespace, marker, end` restarts at every
# character of a whitespace run, which is quadratic in the run (one arena reply held a shard for 22 minutes).
TRUNCATION_MARKER_PATTERN = re.compile(r"(?:\[(?:truncated|\.\.\.|…|continued)\]|\((?:truncated|continued)\)|<truncated>)$", re.IGNORECASE)
TRUNCATION_MARKER_TAIL = 16  # the longest marker is 11 characters
SENTENCE_END_PATTERN = re.compile(r"[.!?。！？…][\"'”’)\]」』]*(?=\s|$)|\n")
TAG_LINE_PATTERN = re.compile(r"^<[^<>]+>$")
LETTER_PATTERN = re.compile(r"[^\W\d_]")
MIN_PROMPT_ECHO_CHARACTERS = 20
REPEATED_LINE_SHARE = 0.2
REPEATED_LINE_MIN_TEXTS = 20
REPEATED_LINE_MAX_CHARACTERS = 200
TRUNCATION_MIN_LENGTH = 32
TRUNCATION_SHARE = 0.02
TRUNCATION_MIN_TEXTS = 10
TRUNCATION_QUANTILE = 0.99
MIN_KEPT_SHARE = 0.5
UNFINISHED_MIN_WORDS = 8
UNFINISHED_LETTER_SHARE = 0.8
UNFINISHED_EXCLUDED_LINE = re.compile(r"^(?:[-*+•>#|]|\d{1,3}[.)]\s)")


def fix_mojibake(text: str) -> str:
    """Each run of UTF-8 bytes shown as Windows-1252 characters, decoded as UTF-8 where that succeeds."""

    def decoded(match: re.Match[str]) -> str:
        try:
            return match.group().encode("cp1252").decode("utf-8")
        except UnicodeError:
            return match.group()

    return MOJIBAKE_PATTERN.sub(decoded, text)


def frequent_lines(lines: list[str]) -> frozenset[str]:
    """Lines that open (or close) at least `REPEATED_LINE_SHARE` of a shard's texts: a header (or footer)."""
    counts = Counter(line for line in lines if line)
    floor = max(REPEATED_LINE_MIN_TEXTS, REPEATED_LINE_SHARE * len(lines))

    return frozenset(
        line
        for line, count in counts.items()
        if count >= floor
        and len(line) <= REPEATED_LINE_MAX_CHARACTERS
        and len(LETTER_PATTERN.findall(line)) >= 3
        and not TAG_LINE_PATTERN.match(line)
    )


def truncation_lengths(texts: list[str], origins: list[str]) -> dict[str, tuple[int | None, int | None]]:
    """Per origin, the character count and the word count at which a shard's texts were cut, or None.

    A dataset that cut its texts at a fixed length leaves many texts at exactly that length, at the top of
    the range. A length held by `TRUNCATION_SHARE` of an origin's texts at or above the 99th percentile is
    taken as such a cut.
    """
    found: dict[str, tuple[int | None, int | None]] = {}
    for origin in set(origins):
        members = [text for text, owner in zip(texts, origins, strict=True) if owner == origin]
        lengths: list[int | None] = []
        for measure in (np.array([len(text) for text in members]), np.array([len(text.split()) for text in members])):  # (m,)
            values, counts = np.unique(measure[measure >= TRUNCATION_MIN_LENGTH], return_counts=True)  # (u,), (u,)
            if len(values) == 0:
                lengths.append(None)
                continue
            top = int(np.argmax(counts))
            held = counts[top] >= max(TRUNCATION_MIN_TEXTS, TRUNCATION_SHARE * len(members))
            at_top = values[top] >= np.quantile(measure, TRUNCATION_QUANTILE)
            lengths.append(int(values[top]) if held and at_top else None)
        found[origin] = (lengths[0], lengths[1])

    return found


def ends_unfinished(text: str) -> bool:
    """Whether a text's last line is prose that stops mid-sentence, as a generation limit leaves a reply.

    The last line must be a run of eight or more words, mostly letters, ending in a letter, a digit or a comma;
    a list item, a heading, a table row, code (a fence anywhere) or a line ending in a mark is not unfinished.
    """
    if "```" in text or "~~~" in text:
        return False
    last = text.rstrip().rpartition("\n")[2].strip()
    if len(last.split()) < UNFINISHED_MIN_WORDS or UNFINISHED_EXCLUDED_LINE.match(last):
        return False
    letters = sum(character.isalpha() or character.isspace() for character in last)

    return (last[-1].isalnum() or last[-1] in ",-–") and letters >= UNFINISHED_LETTER_SHARE * len(last)


def cut_back(text: str) -> str:
    """A text up to its last sentence or line end."""
    ends = [match.end() for match in SENTENCE_END_PATTERN.finditer(text)]

    return text[: ends[-1]].rstrip() if ends else ""


def remove_tells(
    texts: list[str], prompts: list[str | None], origins: list[str], benchmark: list[bool], report: Report
) -> list[str | None]:
    """Each text with its tells removed, or None when a rule drops it. Changes and drops go to `report`."""
    headers = frequent_lines([text.partition("\n")[0].strip() for text in texts])
    footers = frequent_lines([text.rpartition("\n")[2].strip() for text in texts])
    cut_lengths = truncation_lengths(texts, origins)
    cleaned: list[str | None] = []
    for text, prompt, origin, is_benchmark in zip(texts, prompts, origins, benchmark, strict=True):
        if is_benchmark:
            cleaned.append(text)
            continue
        cleaned.append(clean_text(text, prompt, origin, headers, footers, cut_lengths[origin], report))

    return cleaned


def clean_text(
    text: str,
    prompt: str | None,
    origin: str,
    headers: frozenset[str],
    footers: frozenset[str],
    cut_length: tuple[int | None, int | None],
    report: Report,
) -> str | None:
    """One text through every rule in order; None when a rule drops it."""
    stripped = text.strip()
    if stripped.lower() in NULL_STRINGS or not stripped:
        report.drop("null_string", origin)
        return None
    if "\ufffd" in text:
        report.drop("replacement_character", origin)
        return None
    if len(text) <= API_ERROR_CHARACTERS and API_ERROR_PATTERN.search(stripped):
        report.drop("api_error", origin)
        return None
    if REFUSAL_PATTERN.match(stripped) and len(stripped.split()) <= REFUSAL_WORDS:
        report.drop("refusal", origin)
        return None
    if ANONYMIZED_PATTERN.search(text):
        report.drop("anonymized", origin)
        return None
    chars, words = cut_length
    at_cut = (chars is not None and len(text) == chars) or (words is not None and len(text.split()) == words)

    if MOJIBAKE_PATTERN.search(text):
        fixed = fix_mojibake(text)
        if fixed != text:
            report.change("mojibake", origin, ", ".join(sorted(set(MOJIBAKE_PATTERN.findall(text))))[:240])
            text = fixed
    if "\n" not in text and text.count("\\n") >= 2:
        report.change("export_escapes", origin, text[:120])
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
    if "&" in text and len(ENTITY_PATTERN.findall(text)) >= 2 and not HTML_TAG_PATTERN.search(text):
        report.change("html_entities", origin, " ".join(ENTITY_PATTERN.findall(text)[:6]))
        text = html.unescape(text)
    if "<" in text and BREAK_PATTERN.search(text) and not HTML_TAG_PATTERN.search(BREAK_PATTERN.sub("", text)):
        report.change("html_breaks", origin, text[:120])
        text = BREAK_PATTERN.sub("\n", text)
    if "gutenberg" in text.lower():
        start = GUTENBERG_START.search(text)
        end = GUTENBERG_END.search(text, start.end() if start else 0)
        if start or end:
            removed = (text[: start.end()] if start else "") + (text[end.start() :] if end else "")
            report.change("gutenberg_frame", origin, removed)
            text = text[start.end() if start else 0 : end.start() if end else len(text)]
    first, newline, rest = text.strip("\n").partition("\n")
    if newline and first.strip() in headers:
        report.change("repeated_header", origin, first)
        text = rest
    head, newline, last = text.rstrip().rpartition("\n")
    if newline and last.strip() in footers:
        report.change("repeated_footer", origin, last)
        text = head
    label = RESPONSE_LABEL_PATTERN.match(text)
    if label:
        report.change("response_label", origin, label.group())
        text = text[label.end() :]
    if prompt is not None and len(prompt.strip()) >= MIN_PROMPT_ECHO_CHARACTERS and text.lstrip().startswith(prompt.strip()):
        report.change("prompt_echo", origin, prompt.strip())
        text = text.lstrip()[len(prompt.strip()) :].lstrip()
    if "[" in text and SLOT_PATTERN.search(text):
        lines = SLOT_LINE_PATTERN.findall(text)
        if lines:
            report.change("template_line", origin, " | ".join(line.strip() for line in lines))
            text = SLOT_LINE_PATTERN.sub("", text)
        if SLOT_PATTERN.search(text):
            report.drop("template_slot", origin)
            return None
    ending = text.rstrip()
    marker = TRUNCATION_MARKER_PATTERN.search(ending[-TRUNCATION_MARKER_TAIL:])
    if marker:
        report.change("truncation_marker", origin, marker.group())
        text = ending[: len(ending) - len(marker.group())].rstrip(" \t\n")
    if at_cut:
        shorter = cut_back(text)
        if len(shorter) < MIN_KEPT_SHARE * len(text):
            report.drop("truncated_fragment", origin)
            return None
        if shorter != text.rstrip():
            report.change("fixed_length_cut", origin, text[len(shorter) :].strip())
            text = shorter
    elif ends_unfinished(text):
        shorter = cut_back(text)
        if len(shorter) >= MIN_KEPT_SHARE * len(text.rstrip()):
            report.change("unfinished_ending", origin, text[len(shorter) :].strip())
            text = shorter
    if not text.strip():
        report.drop("empty_after_rules", origin)
        return None

    return text
