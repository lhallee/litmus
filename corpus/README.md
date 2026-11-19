# Litmus corpus (`litmus_corpus`)

The pipeline that turns public datasets of human, AI, and mixed text into the balanced Parquet corpus the teacher and
the student train on. Nothing is generated: every row comes from a public source pinned by revision. Human text is
dated before 2022, AI text names its generator, and mixed text names how it was mixed where the source says.

| Path | Holds |
| --- | --- |
| `sources.csv` | The source catalog: 312 public sources, one row per source and config, with revision, columns, label map, date evidence, kind, license, priority, and how each was verified |
| [`SOURCES.md`](SOURCES.md) | How the catalog was checked, coverage by kind and origin, the gaps, what was rejected, and how much of each source is read |
| `src/litmus_corpus/` | The stages (below), the catalog check, and `publish.py`, which uploads a written corpus to a private dataset repository only |
| `tests/` | Each stage on small fixtures (`tests/fixtures/`) |

The benchmarks the comparison uses are read from [`../benchmarks/catalog.csv`](../benchmarks/catalog.csv), so their
test splits are held out of every set.

## Commands

From this folder, with `src` on `PYTHONPATH`:

```sh
python -m litmus_corpus build --out OUT --workers N [--priority 1] [--new-token-budget 3e9] [--limit-per-source K] \
    [--skip-benchmark NAME] [--hub-repo OWNER/NAME]
python -m litmus_corpus catalog check [--priority 3] [--out check.json]
```

`build` reads every source of the given priority or lower and every cataloged benchmark, and resumes from finished
shards. `--limit-per-source` is for a small trial run, `--skip-benchmark` names a benchmark the reader cannot open (recorded in
the report), and `--hub-repo` uploads the written corpus to a private dataset repository; a public one is refused.
`--no-first-corpus` leaves out the predecessor's corpus (`lhallee/many-generator-detection-corpus`), for tests and
trial runs. `catalog check` verifies revisions, columns, and row counts against the Hub.

`--family-train` trains on a benchmark family's `train` and `validation` lines (`family_split` in `sources.csv`,
`<family>:<split>`) when the family's test is held out, and reads its `test` lines as hidden-label benchmarks, held out
by name and by content; `in_distribution` in `sources.csv` and in the benchmark catalog says which families are
trained on. A source with a `group_column` (RAID's `source_id`) gives every row of a group one `doc`, so no split
divides it, and is read whole, `group_share` keeping that share of the groups whole.

```sh
python -m litmus_corpus extend --earlier EARLIER/corpus/data --build BUILD --out OUT --workers N [--trained-first-source NAME]
```

`extend` adds a later build to an earlier corpus without moving an earlier row: the earlier rows keep their ids and
texts, are marked anew by the later build's benchmarks, and the later build's new rows follow, minus copies of an
earlier row (`extend.py`). A held-out first-corpus row of a family now trained on (`--trained-first-source`)
returns to training only if it passes the checks its build skipped for a held-out row: not degenerate, and no
copy of another training row's text or opening. `launch/corpus2_chain.sh` builds the second corpus this way: the new sources with
`--family-train --earlier-build` (the sources the first build read are not read again) and `--first-index` (raw
re-reads link to the predecessor's documents), then the extension.

## Stages

Each stage writes one folder under `OUT` with a `report.json` of counts in and out by reason and a `done.json` when
finished.

1. **Read** (`read/`): raw records, with the prompt each answered where the source gives one.
2. **Standardize** (`standardize/`): one standard reading for every origin (Unicode NFC, LF line endings, control
   characters and zero-width marks removed, trailing whitespace trimmed, blank-line runs capped at two), and every
   trait one origin carries that another does not (boilerplate, truncation marks, escapes) counted and removed.
3. **Strata** (`strata/`, `links/`): format (`plain`, `markdown`, `list`, `table`, `code`, `math`, `chat`), language
   (fastText lid.176), document hash, token count (ModernBERT's tokenizer), and cut points; benchmark texts are linked
   to the predecessor's documents.
4. **Passages** (`passages/`): each text whole, if at most 8,192 tokens, and passages cut at sentence or line starts
   to target lengths drawn evenly in the logarithm from 8 tokens to half the text (or 8,192); a whole document of
   8,193 to 131,072 tokens is kept only to evaluate the chunked reading (`long_eval`).
5. **Checks** (`checks_keys/`, `checks/`): empty, letterless, and looping text dropped, human text that is undated
   or dated 2022 or later dropped, and one row kept for each text and each 20-word opening across the corpus.
6. **Hold-out** (`holdout/`, run before the balance): a new row that shares a document, a text, or an opening with a
   benchmark row is dropped, and a predecessor row is marked with the benchmark's name and leaves training.
7. **Balance** (`balance/`): caps per source and generator, domain, or mix, then in each cell of format, language
   group, and length bin the larger of human and AI text cut toward the smaller, matching domains as far as the cell
   allows, within the token budget.
8. **Corpus** (`corpus/`): `data/*.parquet`, `sources.csv`, `report.json`, and samples.

The row contract (`id`, `text`, `label` 0 human, 1 AI, 2 mixed, `source`, `generator`, `provider`, `era`, `domain`,
`format`, `language`, `doc`, `tokens`, `whole`, `long_eval`, `mix`, `benchmark`, `first_corpus_id`,
`text_unchanged`) is `CORPUS_SCHEMA` in `src/litmus_corpus/records.py`. The split into train, validation, test, bank,
and calibration is the teacher's ([`../teacher/`](../teacher/README.md)).

## The corpus

The corpus is built from the priority 1 sources with a budget of 2e9 new training tokens. The benchmarks EditLens,
HC3, and AuTexTification (English and Spanish) are skipped because the reader cannot open them, so their texts are not
held out and the comparison leaves them out. The corpus holds 12,653,367 rows in 589 shards (6.0 GB): 5,110,107
human, 7,186,256 AI, and 357,004 mixed. Three sources fail to read and 35 catalog rows are skipped before reading
because their columns or labels are described in prose; `report.json` lists both. The compiled corpus is a private
dataset; this catalog and pipeline are its public recipe.

## Tests

`python -m pytest` from this folder: the catalog, reading, standardizing, strata, passages, checks, balance, writing,
and the pipeline end to end on fixtures, with no network.
