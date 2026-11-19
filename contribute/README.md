# Litmus contribute site

The server of the contribute build of the Litmus app: it serves the page, scores each text on the server with the
engine, stores the text with its reading and the visitor's label on a Modal volume, and compacts the volume to a
private Hub dataset. Visitors consent explicitly: the server takes a text only with `"consent": true`, which the page
sends only after the visitor ticks an "I agree" box. Nothing identifying them is stored, texts are never published,
and anyone can have a text deleted within 30 days through the removal form. It runs on [Modal](https://modal.com);
its terms are the three documents in `legal/`.

## The engine binding this code uses

The PyO3 binding `litmus_engine` is the crate `engine/litmus-py` ([its
interface](../engine/README.md#python-interface)). [`scoring.py`](src/litmus_contribute/scoring.py) uses exactly:

```python
model = litmus_engine.Model.from_path(path)   # a release directory or file
model.version -> str                          # stored with every row
model.count_tokens(text: str) -> int          # checked against the 131,072-token cap before scoring
model.score(text: str) -> dict                # the document reading as JSON-ready dict (student/SPEC.md, "Fields"),
                                              # computed with the GIL released, so one container serves pages while it scores
```

The server stores the reading as the engine returns it, checking only that it is a finite JSON object under 2 MiB.
The reading is what `litmus score --json` prints under `reading`, plus `label`; for a 131,072-token text it is about
1.15 MB with the first trained release. One loaded model is shared by the container's concurrent inputs: `score`
releases the GIL, concurrent calls run in parallel and give the serial readings, and the binding writes no file.
The binding's own tests (`engine/litmus-py/tests`) run this module's `EngineScorer` and `serialize_reading` on it.

The deployment installs the binding from a wheel built for Linux x86_64: `engine/litmus-py/build_wheel.sh` builds
`litmus_engine-<version>-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (abi3: one wheel for CPython
3.10 and later, so 3.12 here) in Docker, and `engine/litmus-py/smoke_test.py` checks it against a release.

## Endpoints

| Route | Does |
| --- | --- |
| `GET /`, `GET /<file>` | The page's files from memory, gzip when it helps, ETags, `immutable` caching for hashed names, `application/wasm` for `.wasm` |
| `POST /api/score` | `{"text", "terms_version", "consent": true, "annotation"?}`: scores, stores a row, returns `{"id", "reading", "model_version", "terms_version", "annotation", "annotation_window_seconds"}`. 400 unless `consent` is the JSON value `true`, 409 if `terms_version` is not the one served now, 413 past 1 MiB or 131,072 tokens; nothing is scored or stored on a refusal |
| `POST /api/contributions/{id}/annotation` | `{"annotation": "ai" \| "mixed" \| "human" \| null}` within 30 minutes of the row (the id carries its time); 202, or 410 after |
| `POST /api/removals` | `{"text"}`: open to anyone, no consent needed. Every stored copy of that exact text is deleted at the next compaction, from the volume and the Hub dataset, whose history is then squashed, and kept out of later batches. Stores the SHA-256 only; idempotent; 202 with `{"content_hash", "removal": "accepted"}` |
| `GET /api/terms` | `{"version", "status", "notice", "consent_label", "links", "limits"}`: the on-page notice and the consent box's label (Markdown, links pointing at the routes below), their version and status, links to the two documents and the private page, and the size limits |
| `GET /legal/terms-of-use`, `GET /legal/privacy-notice` | The documents as text |
| `GET /healthz` | Model and terms versions |

Limits: a body of 4 MiB, a text of 1 MiB and 131,072 tokens, one scoring at a time per one-core container, 300 s per
request. Modal's proxy answers a request still running at 150 s with a 303 to a result URL, which a browser's `fetch`
follows ([webhook timeouts](https://modal.com/docs/guide/webhook-timeouts)); at the assumed 1,000 to 10,000 tokens a
second the longest text takes 13 to 131 s.

## What it keeps, and what stops abuse

Stored per row: the text, its SHA-256, the reading, the model version, the label and its time, the receipt time and
the terms version the visitor agreed to (a row exists only for a request with `"consent": true`, so consent itself is
not a column). Never stored or logged: the address, user agent, cookies (none are set), referrer or any account.
Never published: the texts, readings and labels stay in the volume and the private dataset; the documents allow
publishing only models trained or evaluated with them and figures over many texts that contain no text.
Every API route answers same-origin requests only: no CORS headers, `application/json` required (which forces a
preflight another origin fails), and a foreign `Origin` or a cross-site `Sec-Fetch-Site` refused.

| Control | Setting ([`config.py`](src/litmus_contribute/config.py)) |
| --- | --- |
| Per address (salted HMAC in memory; the salt is drawn after a snapshot restore and dies with the container) | 10 scorings a minute, 4 MiB of text per 10 minutes, 30 labels a minute, 10 removals per 10 minutes |
| Per container, any address | 1 GiB a day written to the volume |
| All containers together (a Modal Dict, one key per container) | 12 scoring core-hours a UTC day, then 503 until the next day |
| Containers | At most 32 (`max_containers`), scaled to zero after 5 s idle (`scaledown_window`) |

What this does not stop: a flood from many addresses still wakes containers, and each one up is billed (at worst 32
containers all day, about $39 a day); poisoned labels from many addresses; and pasted personal or copyrighted text,
which the notice asks visitors not to paste. The backstops are Modal's workspace spend limit, set before the site is
public, and review: labels are unverified, each row carries `annotation_counts` so
agreement across independent submissions can be read, and nothing enters a training or evaluation split without
review. No dataset of contributions is ever released.

## Storage

Modal volumes keep the last committed writer of a file and lose the rest, need `commit()` for others to see a write
and `reload()` to see others' ([Volumes](https://modal.com/docs/guide/volumes), [Volume
reference](https://modal.com/docs/reference/modal.Volume)), so each container appends only to its own files:

```
litmus-contributions (Volume v2)
  shards/<YYYYMMDD>/<window start>_<window s>_<container>_<seq>.jsonl   append-only, one per container and 15-minute window
  receipts/<run>.json                                                    one per compaction
```

A container commits every 30 s when it has written and at shutdown (Modal also commits in the background and at
shutdown). A repeat of a text in the same file is stored without its text. Compaction ([`compact.py`](src/litmus_contribute/compact.py),
run by hand with [`deploy_compact.py`](src/litmus_contribute/deploy_compact.py)) takes contributions from shards
closed for more than an hour (grace, label window and shard window), reads labels and removals from every shard closed
for more than 15 minutes, collapses rows by (content hash, model version, terms version) keeping the count, drops
removed hashes, writes one parquet batch, pushes it with any rewritten earlier batches in one Hub commit, reads every
pushed file back and compares bytes, squashes the dataset's history to one commit when the run rewrote or deleted a
file on the Hub (so no earlier revision keeps a removed text; the dataset is private, so only working history is
lost, and the receipts keep the record), and only then deletes the shards it consumed. A rerun after a stop between
push and delete recognizes its batch; a run that stopped at the squash is finished with `--squash`. The Hub layout and
columns are in [DATASET_CARD.md](DATASET_CARD.md).

The documents promise that a removal reaches the volume, the private dataset with its history, and the maintainers' own copies
within 30 days of the request. Compaction runs by hand, so it must run at least once every four weeks (a removal
waits for the next run, and its text's shard may wait one run more), and copies of the dataset outside the volume and
the Hub (training machines, downloads, caches) must be purged by hand after a run that removed a text.

## Deploying

Both apps deploy with Modal's command line. The web app refuses to deploy until the three documents in `legal/`
carry `status: approved` and no `{{BLANK}}`. The two deploy scripts take their package pins from the maintainers'
environment lock file (`environments/research/lock.txt`, found through `foundry.modal_image`), which this repository
does not hold, so they run as written only in the maintainers' checkout; elsewhere, give `read_pins` a lock file of
your own. From the repository root:

```
PYTHONPATH=contribute/src \
LITMUS_CONTRIBUTE_SITE=<built contribute page> LITMUS_RELEASE=<release folder> \
LITMUS_ENGINE_WHEEL=<litmus_engine-*-abi3-manylinux*_x86_64.whl> LITMUS_PRIVATE_PAGE=<the private page's address> \
modal deploy contribute/src/litmus_contribute/deploy_web.py

PYTHONPATH=contribute/src modal run contribute/src/litmus_contribute/deploy_compact.py \
  --repo-id <the private dataset's repository>      # a dry run; add --no-dry-run to push, squash and delete
```

The web app: `@app.cls` with `cpu=(0.125, 1.0)` (an eighth of a core requested, one at most;
[resources](https://modal.com/docs/guide/resources)), `memory=(512, 1024)`, `enable_memory_snapshot=True` and the
engine loaded in `@modal.enter(snap=True)` ([memory snapshots](https://modal.com/docs/guide/memory-snapshot)),
`min_containers=0`, `max_containers=32`, `scaledown_window=5` (2 s to 20 minutes allowed; [cold starts](https://modal.com/docs/guide/cold-start),
[scaling](https://modal.com/docs/guide/scale)), `@modal.concurrent(max_inputs=8, target_inputs=1)`
([concurrent inputs](https://modal.com/docs/guide/concurrent-inputs)) and `@modal.asgi_app()`
([web endpoints](https://modal.com/docs/guide/webhooks)). It gets no secret; compaction gets `HF_TOKEN` only.

## Cost

Modal's [pricing](https://modal.com/pricing), read 2026-11-17: CPU $0.0000131 per physical core-second, memory
$0.00000222 per GiB-second, volumes free to 1 TiB a month, egress free to 1 TiB a month on Starter, $30 a month of
free compute credits. Modal bills the larger of a container's request and its use. A container here costs
$0.0000142 a second while it scores (1 core, 0.5 GiB) and $0.0000027 a second while it waits (an eighth of a core,
0.5 GiB): $7 a month if one waited all month.

Assumed: 1,000 to 10,000 tokens a second on one core (the Linux wheel read 9,309 a second on one core of a Xeon
Platinum 8358 with an interim model and texts of about 217 tokens; Modal's own processors are not measured), texts
of 2,000 tokens on average; a snapshot cold start of about 2 s (Modal's example) billed at a full core; a visit makes three
requests (the page, a scoring, a label), each of which may start a container that then waits 5 s.

| Month | Compute | Storage and egress | Against $30 |
| --- | --- | --- | --- |
| Idle | $0 (scaled to zero) | Free | $0 |
| 1,000 contributions | About $0.15 | Free | 0.5% |
| 10,000 contributions | About $1.50 | Free | 5% |
| 100,000 contributions (about 2 a minute) | About $8 to $13 | About 2 GB stored, 50 GB egress, free | 25% to 45% |

Waiting, not scoring or writing, is the bill (100,000 texts at 0.2 to 2 s each is 6 to 56 core-hours, $0.30 to
$2.80; volume writes are free and batched every 30 s). A full core requested and 60 s of waiting after each request
would price 1,000 contributions at about $1.75 and 100,000 at $30 to $45; requesting an eighth of a core and
waiting 5 s removes most of it, at the price of a start of about 2 s on most requests and of a scoring possibly
getting less than a full core when the host is busy. The worst flood (32 containers all day,
scoring) is about $39 a day, so a workspace spend limit is the real cap.

## What the contribute page must send and show

For the author of the contribute build of the app (`app/`, feature `contribute`). The server enforces only the request;
everything else here is the page's job, and the documents describe the page this way.

1. **Load the terms.** `GET /api/terms` on load. Keep `version`; render `notice` and `consent_label` as served
   (Markdown: bold, and links already pointing at `/legal/terms-of-use`, `/legal/privacy-notice` and the private
   page). Do not shorten, reorder or restyle them away.
2. **Place the notice.** Directly above the consent box and the Check button, in the page's body type, on every screen
   width, with every link visible without scrolling, underlined and in a contrasting color. Its second paragraph opens
   with the bold sentence "Please do not paste sensitive or personal information, yours or anyone else's."; keep it
   bold and in that place, next to the box.
3. **The consent box.** A real checkbox with `consent_label` as its accessible label ("I agree to the Terms of Use and
   the Privacy Notice, and I am 18 or older.", the two names being links). Unticked on every page load and whenever
   the served `version` changes; never pre-ticked; kept in memory only, never in cookies or browser storage. It may stay
   ticked for later checks in the same page.
4. **The Check button.** Disabled, or sending nothing and pointing at the box, until the box is ticked.
5. **The request.** `POST /api/score`, `Content-Type: application/json`, body
   `{"text": <text>, "terms_version": <version from step 1>, "consent": true}`, plus `"annotation"` when the visitor
   chose a label before checking. `consent` is the JSON `true`, sent only while the box is ticked. Answers: 200 with the
   reading; 400 `{"error"}` if `consent` is missing or not `true`; 409 if the terms changed (fetch `/api/terms` again,
   show the new notice, untick the box); 413 and 429 and 503 with `{"error"}` to show as is.
6. **The removal form.** The notice and the documents promise "the removal form" on the site: a visible form, reachable
   from the notice without leaving the page, with a text box and a Remove button that sends
   `POST /api/removals` with `{"text"}` (no consent box, open to anyone). On 202, say that every stored copy of that
   exact text will be deleted within 30 days and that it must be the exact text that was checked.
7. **Readings.** Show the uncertainty with every reading. The notice's recommendation (a reading is an estimate, not
   proof; we recommend not using one alone to accuse anyone) is enough on this page.
8. **The row reference.** Keep the `id` returned for labels in memory only, for the 30-minute label window, never in
   browser storage, or the Privacy Notice's "no identifier of you in your browser" stops being true.

For each terms version, keep the page's markup and a dated screenshot at phone and desktop widths:
they are the only record of what a visitor agreed to.

## Tests

`py -m pytest` in this folder (41 tests, CPU only, no network, no Modal account): the ASGI app against a local folder
standing in for the volume, validation and caps, explicit consent (every request without `"consent": true` refused
before scoring or storing), same-origin refusals, rate limits, the daily budget, labels in and past the window,
removal by content through compaction to the Hub (mocked) with the history squashed after it, that no address, user
agent, cookie or referrer reaches storage or logs, the served notice, consent label and version against the files, what
the documents must say (texts never published, removal on request, recommendations and not rules), concurrent writers, shard rolling, compaction's deduplication,
verification and rerun, and that the Modal app defines against the pinned SDK and refuses draft documents.

## Limits

- The engine's rate on Modal's processors is not measured. On one laptop core (AVX2) a trained release scores about
  22,600 tokens a second through the binding, above the 1,000 to 10,000 assumed under Cost, so the longest text
  takes about 6 s there; the cost table and the 131,072-token cap rest on the assumed range until it is measured on
  Modal.
- The 30-day removal promise rests on running the site by hand: compaction at least every four weeks, and a purge of
  copies outside the volume and the Hub after a run that removed a text.
- Verified only by a real deployment: snapshot restore with this class (that `asgi_app` runs after restore, so the salt
  and container name are drawn per container, which the code ensures by drawing them lazily anyway), the ASGI
  lifespan's shutdown commit, how Modal's proxy fills the client address, whether Modal's own logs keep addresses, the
  Dict's behavior and price, and cold-start times.
