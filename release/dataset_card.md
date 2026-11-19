---
license: other
license_name: per-row-source-terms-and-cc-by-nc-4.0-compilation
language:
- en
- multilingual
task_categories:
- text-classification
tags:
- ai-text-detection
- machine-generated-text-detection
- mixed-authorship
pretty_name: Litmus corpus
---

<!--
Dataset card of the corpus Litmus trains, calibrates and is evaluated on; it becomes the dataset repository's
README.md. The dataset is private. A count that comes from the build is a slot the release process fills from the release's data. Sources:
corpus/README.md, corpus/SOURCES.md, LICENSING.md and benchmarks/README.md.
-->

# Litmus corpus

The training corpus of Litmus, a free detector of AI written text that says how sure it is.

**This dataset is private.** The compiled corpus is kept in its processed form in a private Hugging Face dataset and is
not released. What is public is its recipe: the source catalog, the pipeline that reads and balances it, and
the split, so that anyone can rebuild it from the sources under each source's own terms.

The corpus holds human, model-written (AI) and mixed texts from public datasets, kept as written, with a format,
language, domain and length on every row, split by clusters of similar texts into the sets that train, choose,
test, and calibrate Litmus (<!--litmus:target.hub.model_repo-->lhallee/litmus<!--/litmus-->).

## Sources

- Public datasets only; no text was generated for this corpus.
- The source catalog (`corpus/sources.csv`, described in `corpus/SOURCES.md`) lists 312 public sources as of
  2026-11-17, each pinned by revision: the 102 sources of the predecessor's corpus,
  [lhallee/many-generator-detection-corpus](https://huggingface.co/datasets/lhallee/many-generator-detection-corpus),
  re-listed, and 210 new ones covering source code, non-English text, mixed and edited text, reasoning traces and chat
  scaffolding, and long and formatted outputs.
- <!--litmus:record.corpus.sources_summary-->The release's build adds 112 sources: the official training splits of nine benchmark families (RAID's with every attack among them), the newest generators' sets, and more mixed and human-domain text; 220 sources hold training rows.<!--/litmus-->

## Origins

| Label | Origin | Rule |
| --- | --- | --- |
| 0 | `human` | Dated before 2022 by a date column or the source's own date |
| 1 | `ai` | Names its generator |
| 2 | `mixed` | Written or edited by both a person and a model; names how it was mixed where the source says |

## Processing

- **One standard reading for every class**: Unicode NFC, line endings to LF, control characters and zero-width marks
  removed, trailing whitespace trimmed, runs of blank lines capped at two. Nothing else is normalized.
- **Class tells removed.** Whatever one class carries that another does not, such as a source's boilerplate, a
  dataset's truncation, or an export's escape marks, is counted and removed.
- **Strata.** Every row carries a format (`plain`, `markdown`, `list`, `table`, `code`, `math`, `chat`), a language,
  a domain and a length bin, and the classes are balanced inside each cell as far as public data allows.
- **Whole texts and passages.** A text gives itself (`whole`) and passages cut at sentence or line boundaries; rows of
  one text share `doc`. Training rows are at most 8,192 tokens; whole documents longer than that are kept only to
  evaluate the chunked reading of long documents.
- **Benchmark holdout.** The official test split of every benchmark in the comparison is held out of every split but
  its own benchmark evaluation, and a row in one carries its name in `benchmark`.

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | int64 | Row id |
| `text` | string | The text, in the standard reading |
| `label` | int8 | 0 human, 1 ai, 2 mixed |
| `source` | string | The source dataset |
| `generator` | string | The model that wrote it, for ai and mixed rows |
| `provider` | string | The generator's provider |
| `era` | string | `human`, `legacy` (generators of 2023 and earlier), `recent`, or `unknown` |
| `domain` | string | Subject domain |
| `format` | string | Format stratum |
| `language` | string | Language |
| `doc` | string | Hash of the whole text the row came from |
| `tokens` | int32 | ModernBERT tokens, no special tokens |
| `whole` | bool | The row is a whole text, not a passage |
| `mix` | string | How a mixed text was made, else empty |
| `benchmark` | string | The benchmark whose test split holds the row, else empty |
| `cluster` | int32 | The e-means cluster, -1 for rows not clustered |
| `split` | string | `train`, `validation`, `test`, `calibration`, `bank` (held-out clusters for the bank; none in this release), `bank_draw` (rows reserved for the bank), `unused`, `benchmark` (a benchmark's held-out test row), or `unembedded` |

## Splits

Clustering is e-means (arXiv 2611.09501) on ModernBERT-base embeddings (the mean of a frozen encoder's last-layer
states), centered and normalized so that distance is cosine. Validation, test, and calibration are each whole
clusters, class-balanced, chosen from English and undetermined-language rows only. `bank_draw` holds mixed rows drawn
from training clusters and reserved for the reference bank, whose human and AI texts are drawn from `train`. The
nearest clusters of every held-out cluster, and every training row that shares a `doc` with a held-out or drawn row,
go to `unused`, so that near-duplicates never cross splits.

| Split | Use | Human | AI | Mixed |
| --- | --- | --- | --- | --- |
| `train` | Training the model; the bank's human and AI texts are drawn from it | <!--litmus:record.splits.rows[set=train].human as int-->6,657,360<!--/litmus--> | <!--litmus:record.splits.rows[set=train].ai as int-->6,355,055<!--/litmus--> | <!--litmus:record.splits.rows[set=train].mixed as int-->478,915<!--/litmus--> |
| `validation` | The readings during training, and choosing the checkpoint | <!--litmus:record.splits.rows[set=validation].human as int-->9,996<!--/litmus--> | <!--litmus:record.splits.rows[set=validation].ai as int-->9,996<!--/litmus--> | <!--litmus:record.splits.rows[set=validation].mixed as int-->2,495<!--/litmus--> |
| `test` | Read once, for the reported results | <!--litmus:record.splits.rows[set=test].human as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=test].ai as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=test].mixed as int-->4,996<!--/litmus--> |
| `bank_draw` | Mixed rows reserved for the reference bank, never trained on | <!--litmus:record.splits.rows[set=bank].human as int-->0<!--/litmus--> | <!--litmus:record.splits.rows[set=bank].ai as int-->0<!--/litmus--> | <!--litmus:record.splits.rows[set=bank].mixed as int-->4,096<!--/litmus--> |
| `calibration` | Fitting the conformal and Venn-Abers tables, by length | <!--litmus:record.splits.rows[set=calibration].human as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=calibration].ai as int-->19,996<!--/litmus--> | <!--litmus:record.splits.rows[set=calibration].mixed as int-->4,995<!--/litmus--> |
| `unused` | Removed to keep the splits disjoint | <!--litmus:record.splits.rows[set=unused].human as int-->430,954<!--/litmus--> | <!--litmus:record.splits.rows[set=unused].ai as int-->689,008<!--/litmus--> | <!--litmus:record.splits.rows[set=unused].mixed as int-->100,527<!--/litmus--> |
| `benchmark` | Benchmark test rows, used only for that benchmark's evaluation | <!--litmus:record.splits.rows[set=benchmark].human as int-->305,626<!--/litmus--> | <!--litmus:record.splits.rows[set=benchmark].ai as int-->2,240,265<!--/litmus--> | <!--litmus:record.splits.rows[set=benchmark].mixed as int-->32,255<!--/litmus--> |
| `unembedded` | Rows without a ModernBERT-base embedding, so not clustered | <!--litmus:record.splits.rows[set=unembedded].human as int-->0<!--/litmus--> | <!--litmus:record.splits.rows[set=unembedded].ai as int-->0<!--/litmus--> | <!--litmus:record.splits.rows[set=unembedded].mixed as int-->0<!--/litmus--> |

<!--litmus:record.split.strata_rows-->Every held-out set holds English and undetermined-language rows only (`--held-out-languages en und`), so a text of another language trains unless its cluster is held out or in a moat. The test set by format, human, AI and mixed: plain 6,426, 6,974, 1,979; code 6,548, 6,807, 1,979; markdown 5,380, 4,702, 786; math 962, 671, 171; list 420, 675, 32; table 249, 132, 49; chat 11, 35, 0. Long texts are thin in every held-out set: test holds 282 human and 282 AI rows of 2,048 to 4,095 tokens, 101 of 4,096 to 8,191 and 26 longer. The split cut the larger origin of each language in training to 1.5 times the smaller (`--train-language-ratio 1.5`), moving 10,734 rows of 70 languages to `unused`.<!--/litmus-->

## Licensing

Every row keeps its source's terms; the compilation is CC BY-NC 4.0. The terms of each source are in the source
catalog. Texts submitted to the Litmus contribute site are never published; they are kept privately and enter no split here
without review.

## Limitations

- Human text predates 2022, so the human class holds no text about later events or in later styles.
- Long model-written text is scarce in public data, and nothing was generated to fill it.
- Mixed texts are described by their sources; how they were mixed varies by source.
- Balance holds only as far as public data allows; <!--litmus:record.corpus.unbalanced_cells-->human and AI training rows are level overall (7,138,302 and 7,094,051) while mixed text is 596,024 rows, 4.0%, and the benchmark families with no dated human rows (DACTYL, Defactify, rasbt 50k, and the multilingual COLING 2025 and SemEval-2024 sets) are AI only<!--/litmus-->.
- Languages other than English hold little text, and no held-out split holds any, so nothing here measures a model
  on them.
