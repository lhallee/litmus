"""A generator name, as a source writes it, to its maker and the year of its family's release.

Sources name one model many ways (`gpt-3.5-turbo-0613`, `ChatGPT`, `openai/gpt-35`), so the name is matched by
patterns, most specific first. `PATTERNS` is the one table of generator to maker and release year (of the
family, as a source's name allows). A generator of 2023 or earlier has the `legacy` era and a lower cap, as in
the first corpus. A name no pattern matches has the maker `unknown` and the era `unknown`, takes the lower cap
(the cautious choice), and is listed in the build report so the table can grow.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class Maker:
    provider: str
    year: int | None


UNKNOWN = Maker("unknown", None)
HUMAN = Maker("human", None)
LEGACY_LAST_YEAR = 2023

# (pattern, provider, year). Order matters: the first match wins.
PATTERNS: tuple[tuple[str, str, int], ...] = (
    (r"gpt-?oss", "openai", 2025),
    (r"gpt-?5", "openai", 2025),
    (r"gpt-?4\.5|gpt-?4\.1", "openai", 2025),
    (r"gpt-?4o|chatgpt-?4o", "openai", 2024),
    (r"(?<![a-z])o[134](?:-mini|-preview|-pro)?(?![a-z0-9])", "openai", 2024),
    (r"gpt-?4-?(?:turbo|1106|0125|vision)", "openai", 2023),
    (r"gpt-?4", "openai", 2023),
    (r"gpt-?3\.5|gpt-?35|chatgpt|turbo", "openai", 2023),
    (r"text-davinci|davinci|curie|babbage|(?<![a-z])ada(?![a-z])|gpt-?3", "openai", 2022),
    (r"gpt-?2", "openai", 2019),
    (r"claude-?(?:3|4|sonnet-?4|opus-?4|haiku-?4)|claude-(?:sonnet|opus|haiku)", "anthropic", 2024),
    (r"claude", "anthropic", 2023),
    (r"llama-?(?:3|4)|llama3|llama4", "meta", 2024),
    (r"code-?llama|llama-?2|llama2|llama", "meta", 2023),
    (r"(?<![a-z])opt-|galactica", "meta", 2022),
    (r"mistral-?(?:large|small|medium|nemo)|ministral|magistral|codestral|pixtral|mixtral-?8x22", "mistral", 2024),
    (r"mistral|mixtral", "mistral", 2023),
    (r"gemini-?(?:1\.5|2|3|exp|flash)", "google", 2024),
    (r"gemini|palm|bard|bison|unicorn", "google", 2023),
    (r"gemma", "google", 2024),
    (r"flan|(?<![a-z])t5|(?<![a-z])ul2|(?<![a-z])t0(?![a-z0-9])", "google", 2022),
    (r"qwen-?(?:1\.5|2|3)|qwq", "alibaba", 2024),
    (r"qwen", "alibaba", 2023),
    (r"deepseek-?(?:llm|coder-?(?:6|33|1\.3|instruct))", "deepseek", 2023),
    (r"deepseek", "deepseek", 2024),
    (r"command-?(?:r|a)|aya", "cohere", 2024),
    (r"cohere|command", "cohere", 2023),
    (r"phi-?(?:3|4)", "microsoft", 2024),
    (r"phi-?[12]|orca|wizardlm", "microsoft", 2023),
    (r"(?<![a-z])yi-?(?:1\.5|large|lightning)", "01.ai", 2024),
    (r"(?<![a-z])yi-", "01.ai", 2023),
    (r"falcon", "tii", 2023),
    (r"bloom", "bigscience", 2022),
    (r"gpt-?j|gpt-?neo|pythia", "eleutherai", 2022),
    (r"dbrx", "databricks", 2024),
    (r"dolly", "databricks", 2023),
    (r"(?<![a-z])mpt", "mosaicml", 2023),
    (r"vicuna", "lmsys", 2023),
    (r"alpaca", "stanford", 2023),
    (r"koala", "berkeley", 2023),
    (r"stablelm", "stability", 2023),
    (r"starcoder-?2", "bigcode", 2024),
    (r"starcoder|santacoder", "bigcode", 2023),
    (r"grok", "xai", 2024),
    (r"kimi|moonshot", "moonshot", 2024),
    (r"glm-?4|glm-?z1", "zhipu", 2024),
    (r"chatglm|(?<![a-z])glm", "zhipu", 2023),
    (r"baichuan", "baichuan", 2023),
    (r"internlm-?[23]", "shanghai_ai_lab", 2024),
    (r"internlm", "shanghai_ai_lab", 2023),
    (r"nemotron", "nvidia", 2024),
    (r"olmo|tulu-?3", "allenai", 2024),
    (r"tulu", "allenai", 2023),
    (r"smollm", "huggingface", 2024),
    (r"zephyr", "huggingface", 2023),
    (r"openchat", "openchat", 2023),
    (r"hermes|nous", "nous", 2024),
    (r"granite", "ibm", 2024),
    (r"jamba", "ai21", 2024),
    (r"jurassic|(?<![a-z])j2-", "ai21", 2023),
    (r"(?<![a-z])nova-", "amazon", 2024),
    (r"titan", "amazon", 2023),
    (r"ernie", "baidu", 2023),
    (r"hunyuan", "tencent", 2024),
    (r"doubao|seed-", "bytedance", 2024),
    (r"minimax|abab", "minimax", 2024),
    (r"solar", "upstage", 2023),
    (r"reka", "reka", 2024),
    (r"sonar|perplexity|pplx", "perplexity", 2024),
    (r"palmyra", "writer", 2023),
    (r"(?<![a-z])ctrl(?![a-z])", "salesforce", 2019),
    (r"grover", "allenai", 2019),
    (r"xlnet", "google", 2019),
    (r"(?<![a-z])bart(?![a-z])", "meta", 2019),
)
COMPILED = tuple((re.compile(pattern), Maker(provider, year)) for pattern, provider, year in PATTERNS)


@cache
def maker(generator: str) -> Maker:
    """The maker of a generator name, or `UNKNOWN`."""
    lowered = re.sub(r"[\s_]+", "-", generator.strip().lower())
    for pattern, found in COMPILED:
        if pattern.search(lowered):
            return found

    return UNKNOWN


def is_legacy(generator: str) -> bool:
    """Whether a generator gets the lower cap: of 2023 or earlier, or of a year not established."""
    year = maker(generator).year

    return year is None or year <= LEGACY_LAST_YEAR


def era(generator: str, origin: str) -> str:
    """The corpus's `era`: `human`; `legacy` for a generator of 2023 or earlier; `recent`; `unknown` when the
    table above does not establish its year (a mixed text that names no generator is `unknown` too)."""
    if origin == "human":
        return "human"
    year = maker(generator).year if generator else None
    if year is None:
        return "unknown"

    return "legacy" if year <= LEGACY_LAST_YEAR else "recent"
