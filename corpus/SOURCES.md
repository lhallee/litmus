# Litmus source catalog

`sources.csv` lists 312 public sources, one row per source and config: the 102 sources of the
[predecessor's corpus](https://huggingface.co/datasets/lhallee/many-generator-detection-corpus) re-listed, and 210 new
ones that fill the gaps below. Nothing in it is generated; every row is a public dataset as it stood on 2026-11-17,
pinned by `revision`. Of the 102 predecessor sources, 65 are marked to be re-read raw (the predecessor's reader dropped
their formatting, short texts, long texts or mixed rows) and 37 are not; the reason ends each row's `notes`.

## How each row was checked

Every Hub row was read through the Hub API (revision, declared licence, gating) and the datasets-server (`/size` for rows
and parquet bytes, `/first-rows` or `/info` for columns, `/statistics` for the values of label and generator columns).
A builder confirmed that every column a row names exists in its config and split. `verified` says what held:

| verified | Rows | Meaning |
| --- | --- | --- |
| yes | 233 | columns found, rows and bytes from the API, label and generator values seen where a column holds them |
| partial | 77 | reachable but something is unseen: gated (accept terms to read), loading script or archive the viewer cannot read, row count from a partial conversion, label map not stated by the rows; the reason is in the cell |
| no | 2 | MULTITuDE and MultiSocial: Zenodo records restricted to access requests |

`rows` and `bytes` are the whole config and split (`bytes` is the parquet export, or the repository's files where the
viewer has none; a partial conversion gives a lower bound, said in `notes`). 53 rows carry a `benchmark` flag.

## Coverage by kind and origin

Sources / rows (whole sources) / estimated tokens read under the caps below (capped bytes / 2, a rough rate for
compressed text). A source counts under each kind it carries, so columns overlap. `by_column` sources hold both
classes (and some mixed rows) told apart by a label column.

| Kind | human | ai | mixed | by_column |
| --- | --- | --- | --- | --- |
| prose | 39 / 809M / 80B | 24 / 14M / 10B | 9 / 0.2M / 0.2B | 48 / 205M / 14B |
| formatted | 32 / 3,220M / 92B | 113 / 56M / 113B | 0 | 7 / 0.9M / 8B |
| reasoning | 0 | 65 / 20M / 56B | 0 | 0 |
| long | 28 / 17M / 62B | 23 / 8M / 13B | 0 | 12 / 0.1M / 9B |
| code | 17 / 115M / 60B | 25 / 9M / 20B | 0 | 12 / 9M / 3B |
| multilingual | 7 / 123M / 37B | 22 / 10M / 46B | 0 | 15 / 3M / 6B |
| mixed | 0 | 0 | 13 / 0.2M / 0.2B | 23 / 9M / 14B |
| short | 24 / 3,089M / 42B | 13 / 8M / 13B | 1 / 0.01M / 0 | 11 / 1M / 0.5B |

## The gaps, source by source

1. **Raw formatted model output.** Chat logs: WildChat-4.8M and 1M, LMSYS-Chat-1M (gated), WildChat-50m (55 gated repos,
   one per open model), the Arena sets (55k, 100k, 140k, expert-5k, search, webdev 2024 and the 2026 webdev set with 87
   models such as gemini-3.x, gpt-5.5/5.6, claude-opus-4.8/5, kimi-k2.7). Response dumps: AlpacaEval outputs of 228 models
   (GitHub), WildBench outputs of 61 models, Arena-Hard answers, Magpie Generator Zoo (one prompt set, ~20 models), the
   three Nebius regenerations of Infinity-Instruct (Qwen3-235B, gpt-oss-120b, DeepSeek-V3), UltraFeedback, Magpie, smoltalk.
2. **Reasoning and scaffolding kept as written.** Nemotron v1/v2, Llama-Nemotron, smoltalk2 `*_think`, OpenThoughts,
   Mixture-of-Thoughts, AM-team distills, s1K-1.1 (Gemini and R1 traces), Kimi-K2.5 1M, gpt-oss and MiniMax medical sets,
   and the small 2025-2026 distills (TeichAI, Roman1111111, Crownelius, angrygiraffe: Claude 4.5-5, GPT-5.x, Gemini 3.x,
   GLM, Kimi, DeepSeek-V3.2/V4, Grok). Where reasoning lives (a column, `<think>` inline, a nested field) is in `notes`.
3. **Long text.** Model: LongWriter-6k (2k-20k words), DeepResearch Bench reports (four systems, up to 86k chars),
   StoryScope (five generators, 10-30k chars), Kimi K2.5 stories, Opus-4.5 traces, WildBench and Arena-Hard long answers,
   Cosmopedia (Mixtral textbooks), Gutenberg-DPO (human chapter paired with a model chapter). Human: Gutenberg, PG-19, the
   Pile (2020), arXiv full text (ccdv, common-pile LaTeX), GovReport, BigPatent, BookSum full books, OpenReview papers,
   WikiText-103, Pile of Law.
4. **Code.** Human before 2022: CodeSearchNet, CodeNet (dated submissions), CodeContests, APPS, Stack Overflow posts
   (`CreationDate`), GitHub code (BigQuery snapshot), the Pile's GitHub. Model with named generators: DroidCollection
   (46 generators, refined and adversarial), SemEval-2026 Task 13, CodeMirage, HumanVsAICode, AIGCodeSet, the webdev arenas,
   Magicoder, OpenCodeInstruct, self-oss-instruct, GPT-5.5-Codex traces.
5. **Non-English.** Human: mC4 with timestamps, Wiki-40B, Gutenberg in ten languages, XL-Sum, Europarl. Model: WildChat and
   Arena non-English turns, smoltalk multilingual (Qwen3-32B), Multilingual-Thinking, Chinese R1 and Qwen3 distills, Magpie
   Chinese, Kimi MultilingualSTEM, Russian frontier mix. Both: M4 (GitHub), COLING-2025 multilingual, SemEval-2024 A
   multilingual, HC3-Chinese, LLMTrace (en, ru), DetectRL-X, multilingual partial-continuation CSVs.
6. **Mixed text.** How each was mixed is in `mix`: MixSet (polish, complete, rewrite, humanize), Beemo (expert and LLM edits
   of model output), APT-Eval (graded LLM polishing), EditLens (gated), LLMTrace (fill gaps, expand, update spans),
   open-text-detector mix-a (character spans), ARB (human content re-surfaced by an LLM), OpenTuringBench revise and
   continue splits, SemEval-2024 C and RoFT (human prefix, model continuation), coai paraphrases, DroidCollection
   refined code, two humanizer benchmarks (commercial humanizers on model text), PADBen, DetectRL mixing folders,
   CoAuthor sentences, RAID paraphrase attacks.
7. **Human formatted text before 2022.** Reddit comments and submissions (pushshift, `created_utc`), Stack Overflow and
   Stack Exchange, GitHub issues and PRs (common-pile, `created`), Enron email, Ubuntu IRC, Medium, Hacker News, the
   Pile's subsets, WikiText markup, Wikimedia projects, YouTube transcripts, GitHub READMEs, reviews.
8. **Short text.** Titles (arXiv, Reddit, Hacker News, AG News, Amazon), tweets (TweetEval, Sentiment140), comments
   (Civil Comments, Reddit), short answers (HC3, Yahoo), and on the model side short chat replies, TweepFake-style
   synthetic tweets, PADBen sentences, model-written task prompts.

## Where public data is thin, and what it does to balance

- **Long model text.** Only ten priority-1 sources hold long model text, and past ~8k tokens it is LongWriter-6k
  (6,000 rows), about 400 deep-research reports, ~10k StoryScope stories, 655 Kimi stories and a few hundred long traces.
  Human long text is two orders larger. The long bins can be balanced only by cutting human long text to the model count,
  so the longest bins are small, as the dataset card says. Reasoning traces are long but are reasoning, not prose.
- **Formatted human text.** Markdown, code blocks and lists are plentiful in Reddit, Stack Overflow, GitHub issues and
  the Pile, but the assistant style (headings, bold lead-ins, tables, numbered steps in one reply) has almost no pre-2022
  human counterpart. Tables in particular are rare in human text that keeps its markup. A `table` stratum is
  model-heavy unless capped.
- **Mixed text.** 36 sources carry mixed rows but they are small: of the order of 300k mixed rows in all, most from
  DroidCollection (106k refined code), coai (31k paraphrases), OpenTuringBench (86k) and LLMTrace (19k). Many are
  benchmarks. Against tens of millions of human and model texts that is about 1%.
- **Reasoning has no human class.** Thinking blocks are model-only by definition, so a `reasoning` format cell cannot be
  balanced; a detector learns that a thinking block means a model, which is true but cheap.
- **Newest generators.** Most 2026 frontier sets are distillations of under 3,000 rows each; the large 2026 sources are
  the webdev arena (6,390 battles), coai, rasbt, mild-rgb and open-text-detector. 2023-2024 generators dominate by rows.
- **Non-English model text from 2025-2026 generators** is mostly Chinese and the arena languages; other languages come
  from 2023 generators (M4, COLING, SemEval).

## Rejected or not catalogued

| Source | Why |
| --- | --- |
| pangram/WildAI (listed, priority 3), labels of WildAI-monthly | labels are a detector's predictions: circular for training a detector; WildAI-monthly rows of 2021 kept as human by date only |
| LLM-DetectAIve, LAMP corpus | no public data found (LLM-DetectAIve has only a demo Space) |
| M4GT-Bench | on Google Drive, not the Hub; its rows reach the catalog through COLING-2025 (`coling2025_*`) |
| CoAuthor keystroke logs | need replay to rebuild text; only a sentence-level derivative is catalogued; `Artmann/coauthor` is empty |
| Vincent-HKUSTGZ/PADBench, molmohsen arabic polished, DeepResearch-Bench-II, Alibaba-NLP/Open-DeepResearch | empty, PDFs only, or queries without outputs |
| uonlp/CulturaX, statmt/cc100, reciTAL/mlsum, armanc/scientific_papers, deepmind/pg19, Amazon-Reviews-2023 | gated 1.4 TB overlapping mC4, or loading scripts that fetch from other hosts; parquet copies used instead where they exist |
| OASST, Aya, No Robots, twitter100m, Beemo `human_output` | human text written after 2021 |
| Agent trajectory dumps (SWE, terminal-bench, ARC-AGI) | tool-call debris, little natural text |
| FilipT/llm-multitudes | not MULTITuDE: model preference judgments |

## How much of each source is read

Every source is read whole up to 20 GB; larger ones fetch only enough shards (`allow_patterns`) for a cap in GB:
Pile 40; WildChat-50m 30; Stack Overflow, Wiki-40B, GitHub code clean, OpenWebText, Stack Exchange preferences 20;
Reddit comments, GitHub issues, smoltalk2 15; WildChat-4.8M 15; Gutenberg 12; Reddit submissions, mC4 (20 languages),
Nemotron v2, Cosmopedia, Hacker News 10; DeepSeek-R1 other 8; Kimi-K2.5, DetectRL-X 6; 5 for C4 en, FineWeb-Edu 2015, Wikipedia
2020, Nemotron code, AM code, BigPatent, OpenThoughts3, Llama-Nemotron, Superior-Reasoning, Pile of Law, Wikimedia,
peS2o, AM-Qwen3, Stack Exchange 2023, Reddit title-body; 2-4 for the remaining large sets; 0 for GitHub code (unclean) and
WildAI.

| Priority | Rows | Whole size | Read cap |
| --- | --- | --- | --- |
| 1 | 99 | 11.9 TB (mC4 alone is 10.6 TB) | 333 GB |
| 1 + 2 | 244 | 13.0 TB | 543 GB |
| all | 312 | 13.8 TB | 629 GB |

At an assumed 250-500 MB/s from the Hub with parallel downloads, priority 1 takes 11-22 minutes to fetch and priority
1 + 2 18-36 minutes; reading with 26 workers is not the bottleneck. The largest reads are the Pile (40 GB), WildChat-50m (30 GB), Stack
Overflow, Wiki-40B, GitHub code and Stack Exchange preferences (about 20 GB each). Raw downloads can be deleted once
the reader writes its parquet; with nothing deleted, priority 1 + 2 needs about 550 GB of disk. Gated repositories
fail unless their terms were accepted on the account first.

## Notes

- **Benchmark-only data.** APT-Eval exists only as a test split; Beemo, MixSet, EditLens and DeepResearch Bench are
  benchmarks yet among the few sources of mixed and long model text.
- **The predecessor's corpus trained on benchmark tests.** It read MAGE test, Ghostbuster test, HC3 and SemEval-2024
  train/dev, so comparisons of the predecessor checker on those sets are contaminated.
- **Gated and restricted sources** (LMSYS-Chat-1M, Chatbot Arena 33k, WildChat-50m, Nemotron v2, Infinity-Instruct,
  EditLens, one frontier mix) need their terms accepted on a Hub account, and MULTITuDE and MultiSocial need Zenodo
  access requests.
- **Licences.** CodeMirage is CC BY-NC-ND (no derivatives); Pile of Law and WildAI are CC BY-NC-SA. These constrain
  the compiled corpus most; every row keeps its source's terms.
