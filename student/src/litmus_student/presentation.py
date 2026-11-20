"""Training texts presented at random, so that no trace of how a text was displayed or collected tells its origin.

Logan, 2026-11-19: "given every sampling step, all of the sort of confounders would either be removed automatically or
kept ... it should be m dash with spaces, hyphen with spaces, hyphen without spaces, etc. Let's just make sure there
isn't a way that our model can latch on to some of these tells". Litmus 1.0.0 read Markdown, typography, spacing,
line wrapping and how a text ends as evidence of origin (experiments/tooling/quill/2026-11-19_confounders).

Every choice is drawn from a generator seeded anew at every draw of a text (`data.Presenter`; never by label or
content), so a text seen again is seen presented anew, and a presentation says nothing about origin (Logan,
2026-11-19: "everything is random each epoch so can get different versions of each document"):

- Markdown: rendered as a reader copies it (`reference.rendered`), or kept as written, at even odds; then, at any
  origin, a few phrases made bold (0.15) and a short first line made a heading (0.1).
- Prose paragraphs: joined onto one line (0.75), or hard-wrapped at a width drawn from 60 to 100 (0.25). The text's
  own line breaks are never kept: where a text came from shows in them.
- Dashes (an em or en dash, a spaced hyphen, a dash typed as two hyphens): all of a text's in one style of eight,
  `—`, ` — `, `–`, ` – `, ` - `, `-`, `--`, ` -- `, drawn uniformly.
- Quotes curly or straight at even odds; ellipses `…` (0.3) or `...`; one space between sentences (0.85) or two.
- HTML entities decoded and unusual spaces made spaces: always, since they are traces, never styles.
- Synonyms (Logan, 2026-11-19: "5% synonym replacement"): in 5% of texts, each word of four letters or more that the
  synonym table holds (WordNet's commonest sense, `experiments/tooling/quill/2026-11-19_canonical_retrain/scripts/
  synonym_table.py`) is swapped with probability 0.1 for one of its synonyms, its capital kept; never a function word
  of `KEPT_WORDS`, an all-capitals word, or a word in code.

Code (`reference.code_lines`, and code spans) is left as written. Mid-sentence cuts are applied later, on tokens
(`data.Cutter`). The result is in the standard reading and never empty.
"""

from __future__ import annotations

import json
import re
import textwrap
import numpy as np

from pathlib import Path

from .contract import reference


DASH_STYLES = ("—", " — ", "–", " – ", " - ", "-", "--", " -- ")
P_RENDER, P_WRAP, P_CURLY, P_ELLIPSIS, P_DOUBLE_GAP, P_BOLD, P_HEADING = 0.5, 0.25, 0.5, 0.3, 0.15, 0.15, 0.1
P_SYNONYMS, SWAP_RATE = 0.05, 0.1  # of texts; of a chosen text's words the table holds
WRAP_WIDTHS = (60, 100)  # inclusive
# Words WordNet holds whose commonest sense is not the function they mostly serve in prose.
KEPT_WORDS = frozenset("""able about above after again also back been being both does done down even ever from have
here into just keep like made make many more most much must need only over same seem show some such take than that
them then there these they this those very want were what when where which will with would your""".split())
SYNONYM_WORD = re.compile(r"(?<![\w'’\-])[A-Za-z]{4,}(?![\w'’\-])")
# Each a linear scan: an em or en dash with the blanks around it (a run tried from its start), and in prose a dash
# typed as hyphens (two between words, or one or two set apart by spaces), as `reference.TYPED_DASH` finds it.
UNICODE_DASH = reference.compiled(r"(?<![ \t])[ \t]*[—―–][ \t]*")
SPACES = str.maketrans({key: value for key, value in reference.TYPED.items() if value == " "})
STRAIGHT = str.maketrans({key: value for key, value in reference.TYPED.items() if value in ('"', "'")})
DOUBLE_OPEN = re.compile(r'(^|[\s(\[{—–\-/])"')
SINGLE_OPEN = re.compile(r"(^|[\s(\[{—–\-/])'(?=\w)")
SENTENCE_GAP = reference.compiled(r"([.!?][\"')\]”’]?)[ \t]+(?=[\"(“‘]?[A-Z])")
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def choices(generator: np.random.Generator) -> dict:
    """A text's presentation, drawn before its content is read, so every text of every origin draws alike."""
    draws = generator.random(8)  # (8,)
    width = int(generator.integers(WRAP_WIDTHS[0], WRAP_WIDTHS[1] + 1))
    dash = DASH_STYLES[int(generator.integers(len(DASH_STYLES)))]
    return {"render": draws[0] < P_RENDER, "width": width if draws[1] < P_WRAP else 0, "dash": dash, "curly": draws[2] < P_CURLY,
            "ellipsis": "…" if draws[3] < P_ELLIPSIS else "...", "gap": "  " if draws[4] < P_DOUBLE_GAP else " ",
            "bold": draws[5] < P_BOLD, "heading": draws[6] < P_HEADING, "synonyms": draws[7] < P_SYNONYMS}


def load_synonyms(path: Path) -> dict[str, list[str]]:
    """The synonym table `synonym_table.py` writes: each word to its synonyms."""
    return json.loads(path.read_text(encoding="utf-8"))["table"]


def swapped(line: str, table: dict[str, list[str]], generator: np.random.Generator) -> str:
    """Each word the table holds, lower-case or capitalized and not a function word, swapped at `SWAP_RATE` for one
    of its synonyms, its capital kept."""
    def swap(match: re.Match[str]) -> str:
        word = match.group(0)
        options = table.get(word.lower())
        if options is None or word.lower() in KEPT_WORDS or not word[1:].islower() or generator.random() >= SWAP_RATE:
            return word
        choice = options[int(generator.integers(len(options)))]
        return choice.capitalize() if word[0].isupper() else choice

    return SYNONYM_WORD.sub(swap, line)


def curly(line: str) -> str:
    """Straight quotes as a word processor sets them: opening after a space, bracket or dash, else closing."""
    line = DOUBLE_OPEN.sub("\\1“", line).replace('"', "”")
    return SINGLE_OPEN.sub("\\1‘", line).replace("'", "’")


def styled(line: str, choice: dict, table: dict[str, list[str]] | None = None, generator: np.random.Generator | None = None) -> str:
    """A line that is not code in the drawn style, its words perhaps swapped for synonyms (`swapped`, when the text
    drew them and a table is given); its code spans as written."""
    spans: list[str] = []

    def hold(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return reference.HELD.format(len(spans) - 1)

    line = reference.CODE_SPAN.sub(hold, line).translate(STRAIGHT).replace("…", "...")
    if choice.get("synonyms") and table is not None:
        assert generator is not None, "a swap draws from the text's generator"
        line = swapped(line, table, generator)
    line = UNICODE_DASH.sub(choice["dash"], line)
    if reference.is_prose(reference.unmarked(line)):
        line = reference.TYPED_DASH.sub(choice["dash"], reference.SPACE_RUN.sub(" ", line))
        line = SENTENCE_GAP.sub(lambda match: match.group(1) + choice["gap"], line)
    line = line.replace("...", choice["ellipsis"])
    line = curly(line) if choice["curly"] else line
    return reference.HELD_PATTERN.sub(lambda match: spans[int(match.group(1))], line)


def paragraphs(lines: list[str], code: list[bool], width: int) -> list[str]:
    """Each prose paragraph (non-blank lines, every one prose and none code, three words or more) on one line, or
    wrapped at `width` when it is not 0."""
    out: list[str] = []
    block: list[str] = []
    prose = True

    def flush() -> None:
        # Markdown kept as written keeps its block structure: a line with a block mark (a bullet, heading, quote or
        # number) is never joined to another, nor wrapped.
        if block and prose and all(reference.is_prose(line) and reference.unmarked(line) == line for line in block) \
                and sum(len(line.split()) for line in block) >= 3:
            paragraph = " ".join(line.strip(reference.BLANK) for line in block)
            out.extend(textwrap.wrap(paragraph, width, break_long_words=False, break_on_hyphens=False) if width else [paragraph])
        else:
            out.extend(block)
        block.clear()

    for line, is_code in zip(lines, code, strict=True):
        if line.strip(reference.BLANK):
            prose = prose and not is_code if block else not is_code
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return out


def marked(lines: list[str], choice: dict, generator: np.random.Generator) -> list[str]:
    """A few phrases of prose lines made bold, and a short first line made a heading, as drawn."""
    code = reference.code_lines(lines)
    prose = [index for index, line in enumerate(lines) if not code[index] and reference.is_prose(line) and "**" not in line]
    if choice["bold"] and prose:
        for index in generator.choice(prose, size=min(len(prose), int(generator.integers(1, 4))), replace=False):
            words = list(WORD.finditer(lines[index]))
            if len(words) >= 4:
                first = int(generator.integers(0, len(words) - 1))
                last = min(len(words) - 1, first + int(generator.integers(1, 4)))
                start, end = words[first].start(), words[last].end()
                lines[index] = f"{lines[index][:start]}**{lines[index][start:end]}**{lines[index][end:]}"
    first = next((index for index, line in enumerate(lines) if line.strip(reference.BLANK)), None)
    if choice["heading"] and first is not None and first in prose and len(lines[first].split()) <= 10 \
            and not lines[first].rstrip().endswith((".", ":", "?", "!")):
        lines[first] = "#" * int(generator.integers(1, 4)) + " " + lines[first].lstrip()
    return lines


def presented(text: str, generator: np.random.Generator, synonyms: dict[str, list[str]] | None = None) -> str:
    """`text` (in the standard reading) presented as `generator` draws it (see the module's docstring); with no
    `synonyms` table, no word is swapped."""
    choice = choices(generator)
    standard = reference.standard_reading(text)
    lines = reference.decoded(standard).translate(reference.SEPARATORS).translate(SPACES).split("\n")
    lines, code = reference.rendered(lines) if choice["render"] else (lines, reference.code_lines(lines))
    lines = [line if is_code else styled(line, choice, synonyms, generator) for line, is_code in zip(lines, code, strict=True)]
    lines = marked(paragraphs(lines, code, choice["width"]), choice, generator)
    out = reference.standard_reading("\n".join(lines))
    return out if any(not character.isascii() or character.isalnum() for character in out) else standard
