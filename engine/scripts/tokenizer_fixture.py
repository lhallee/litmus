"""Token ids from Hugging Face `tokenizers` for the engine's tokenizer parity test.

Writes `tokenizer_fixture.json`: a list of {"text", "ids"} with ids from `encode(text, add_special_tokens=False)` on the
raw text (no standard reading), so the test isolates the tokenizer. The texts cover English, code, markdown, other
scripts, emoji, whitespace of every kind, literal added tokens, very long lines and seeded random Unicode.

    python scripts/tokenizer_fixture.py --tokenizer PATH/tokenizer.json --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

from pathlib import Path


TOKENIZER_SHA256 = "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1"


def fixed_texts() -> list[str]:
    english = (
        "The committee met on Tuesday to review the proposal. After a long discussion about costs and timelines, "
        "the members agreed to postpone the vote. They'll decide later; we've waited, I'm sure it's fine, you'd agree."
    )
    code = (
        "fn main() {\n    let xs: Vec<u32> = (0..10).map(|x| x * 2).collect();\n\tprintln!(\"{:?}\", xs);\n}\n"
        "def f(a, b=3, *args, **kw):\n    return {'k': a ** b}  # comment\n"
        "SELECT id, COUNT(*) FROM t WHERE x >= 10 AND y <> 'a' GROUP BY id;\n"
        "<div class=\"x\">&nbsp;&amp;</div>\n#include <stdio.h>\nint main(void){return 0;}\n"
    )
    markdown = (
        "# Title\n\n## Sub-title\n\n- item one\n- item **two**\n  - nested `code`\n\n1. first\n2. second\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n> quote\n\n```python\nprint('hi')\n```\n\n[link](https://example.com/a?b=c&d=e)\n"
    )
    scripts = (
        "日本語の文章です。中文句子在这里。한국어 문장입니다. Ceci est une phrase en français, à côté. "
        "Это предложение на русском. هذه جملة عربية. यह हिंदी वाक्य है। Αυτή είναι ελληνική. "
        "ภาษาไทยไม่มีช่องว่าง עברית כאן. Tiếng Việt có dấu."
    )
    emoji = "Emoji: \U0001f600 \U0001f469‍\U0001f4bb \U0001f1fa\U0001f1f8 ❤️ \U0001f44d\U0001f3fd done."
    whitespace = [
        "a  b   c    d     e" + " " * 30 + "f" + " " * 24 + "g" + " " * 25 + "h",
        "tabs\tand\t\ttabs\t \t end",
        "new\nlines\n\nand\n\n\nmore\n\n\n\n\nend\n",
        "crlf\r\nline\r\n\r\nend\r",
        "nbsp here and　ideographic thin en zw​ls ps nel\u0085vt\u000bff\u000cmvs᠎nn end",
        "  lead nbsp,  then a　　　b and  x",
        "  leading spaces",
        " single leading space",
        "trailing spaces   ",
        "\n\n\nleading newlines",
        "   ",
        " ",
        "\n",
        "",
    ]
    added = (
        "Literal tokens: [CLS] [SEP] [MASK] [PAD] [UNK] <|padding|> <|endoftext|> |||IP_ADDRESS||| "
        "|||EMAIL_ADDRESS||| |||PHONE_NUMBER||| [unused0] [unused12] [unused82] [CLS][SEP]x[MASK]y"
    )
    numbers = "1 12 123 1234 12345 3.14159 1e-10 0x1F 1,000,000 ½ ⅔ ² ³ ٣٤٥ ४५६ Ⅻ"
    contractions = "It's they're we've I'm you'll he'd 'S 'T 'quoted' rock'n'roll ''double'' it’s"
    combining = "café é̂ 각 Å ö ñ क़"
    long_line = ("lorem ipsum dolor sit amet, " * 600).strip()
    long_word = "x" * 5000
    long_punct = "-=" * 3000
    long_digits = "0123456789" * 700
    base64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 200
    mixed = "MiXeD CaSe wordsWithCamelCase snake_case_words kebab-case-words URL https://a.b/c_d-e?f=g#h user@example.org"
    return [english, code, markdown, scripts, emoji, *whitespace, added, numbers, contractions, combining, long_line,
            long_word, long_punct, long_digits, base64, mixed]


def random_texts(count: int, seed: int) -> list[str]:
    """Seeded random strings over ASCII, Latin, combining marks, CJK, emoji and whitespace."""
    generator = random.Random(seed)
    ranges = [(0x20, 0x7E), (0x20, 0x7E), (0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF),
              (0x590, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF), (0x2000, 0x206F), (0x3000, 0x30FF),
              (0x4E00, 0x4FFF), (0xAC00, 0xAD00), (0x1F300, 0x1F6FF), (0x9, 0xD)]
    texts = []
    for _ in range(count):
        length = generator.randint(1, 200)
        characters = []
        for _ in range(length):
            low, high = generator.choice(ranges)
            code = generator.randint(low, high)
            if 0xD800 <= code <= 0xDFFF:
                continue
            characters.append(chr(code))
            if generator.random() < 0.15:
                characters.append(" " * generator.randint(1, 4))
        texts.append("".join(characters))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    from tokenizers import Tokenizer

    digest = hashlib.sha256(arguments.tokenizer.read_bytes()).hexdigest()
    assert digest == TOKENIZER_SHA256, f"{arguments.tokenizer} has sha256 {digest}, not the pinned tokenizer's"
    tokenizer = Tokenizer.from_file(str(arguments.tokenizer))
    texts = fixed_texts() + random_texts(400, 11)
    records = [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids} for text in texts]

    arguments.out.mkdir(parents=True, exist_ok=True)
    temporary = arguments.out / "tokenizer_fixture.json.partial"
    temporary.write_text(json.dumps(records, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, arguments.out / "tokenizer_fixture.json")
    print(f"{len(records)} texts, {sum(len(record['ids']) for record in records)} tokens")


if __name__ == "__main__":
    main()
