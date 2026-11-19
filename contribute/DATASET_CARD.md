---
license: other
license_name: private-never-released
pretty_name: Litmus contributions
private: true
---

# Litmus contributions

Texts checked on the Litmus contribute site, each with the model's reading, the model version, the visitor's label
when given, the times and the version of the terms the visitor agreed to. Nothing identifying the visitor is stored.
This dataset is private and is never released: the contribute site's terms promise that submitted texts, and the
readings and labels stored with them, are never published or released in any dataset. It is used to train, calibrate
and evaluate Litmus. A visitor's label is unverified, texts may hold personal data or text the visitor had no right to
share, and no row enters a Litmus training, calibration or evaluation split without review.

## Layout

- `data/<YYYY>/<batch>.parquet`: one file per compaction, rows deduplicated within it by (content hash, model
  version, terms version). A text sent again in a later batch appears again there; sum `count` over `content_hash`
  across batches for its total.
- `removals/<batch>.parquet`: SHA-256 hashes of texts removed on request (never their text). Every row with a listed
  hash has been deleted from every data file, and is left out of every later batch. The compaction that deletes a row
  from a data file also squashes this dataset's history, so no earlier revision keeps the text.

## Columns

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | string | The first submission's id |
| `text` | string | The text, exactly as sent |
| `content_hash` | string | SHA-256 of the text's UTF-8 bytes |
| `reading` | string | The engine's reading, JSON |
| `model_version` | string | The model release that made the reading |
| `annotation` | string | The plurality of the submissions' labels (`ai`, `mixed`, `human`); null when none or tied |
| `annotated_at` | timestamp (UTC) | The latest label time among the labeled submissions |
| `received_at` | timestamp (UTC) | The first submission's time |
| `terms_version` | string | The terms version the visitor agreed to by ticking the consent box |
| `count` | int32 | Submissions of this text under this model and terms version in the batch |
| `annotation_counts` | string | JSON counts of each label, for judging agreement |
