---
name: litmus-release
description: Cut a Litmus release from a Hugging Face model repository and commit and a version number, with the staged driver release/cut.py. Use when told to release a model as version X, to rebuild the programs and refill every document and number for a new model, to resume or check a half-finished release, or to publish a built release. Never publishes without the person's word.
---

Litmus is a free detector of AI written text that says how sure it is, and this skill cuts one of its releases: given
`<owner/name>@<commit>` and a version, one driver produces a complete, consistent release in which every number comes
from that model, every program is rebuilt with it, and each artifact is ready for where it goes. Read
[RELEASING.md](RELEASING.md) once; this page is the loop.

## Rules

- **Publishing is the person's word.** `publish`, `pages`, and `macos` change things outside this machine and run only with
  `--confirm`. Run `--dry-run`, show what it printed, and wait for a yes in this conversation. A yes for one stage is not
  a yes for the next. Do not commit, push, tag, or create a release on your own either.
- **No typed numbers.** A number in a document is a slot (a `litmus:KEY` comment opening and a `/litmus` comment closing it) filled by the `docs` stage
  from data. If a figure is wrong, fix its source (the record, the template, the key), never the text in the slot.
- **Never guess a missing input.** An unpinned baselines commit, a record without a key, a browser measurement that was
  not made: the driver names it. Report what is needed and from whom; do not substitute.
- **Machines are named by the person.** The driver never connects anywhere. A stage for another platform stops with
  exit code 3 and the exact steps; hand them to the person, then `adopt` what comes back. No GPU is needed.
- Credentials are used, never seen: the Hub and GitHub tokens come from the environment or the person's login, never a
  command line, never printed.

## Loop

1. Start: `python release/cut.py run --work <folder outside the repository> --model <owner/name>@<40-hex> --version X.Y.Z`.
   A model given as a branch or a short hash is refused. `--model local:<folder>` is a trial and is never published.
2. After every stop: `python release/cut.py status --work <folder>`. `stale` says why (a file it read changed); fix the
   cause, then run again, which skips what is done. `--force --only <stage>` reruns one.
3. When it stops, read the message. Fixes by kind:
   - a missing record key: ask the person for the training run's value, or add the key to the record folder on the Hub
     and run again with the new commit (a new `--model` revision needs a new work folder);
   - the browser speed: the person measures it in a real browser (RELEASING.md, step 2), then
     `adopt browser --from <browser.json>`;
   - a build for Windows or Linux: the person runs the steps the message prints on that machine, then
     `adopt build-windows` or `adopt build-linux --from <that machine's work folder>`;
   - an unfinished slot or a bracketed PENDING outside a slot: `slots check --work <folder>` lists each; prose that no longer holds for
     this model (a teacher paragraph in a labels-only build, a size named in words) is edited by hand in the document.
4. Before the release commit: `python release/cut.py version check X.Y.Z` is clean and `slots check` passes.
5. After the person commits and pushes the release commit and builds are made at it: `run --from verify`, then
   `publish --dry-run`, and ask.
6. After yes: `publish --confirm`, ask the person to review and publish the GitHub draft, then `pages --dry-run`, ask,
   `pages --confirm`; `macos --dry-run`, ask, `macos --confirm` when the person wants the macOS programs.
7. Finish with the report: the release's version, the model commit, the stages done, the files and where they went, the
   warnings the receipts recorded (such as a machine different from the baselines' for the cost row), and what the person
   still has to do (the contribute deployment, the ad).

## Where things are

- Driver: `release/cut.py`, code in `release/litmus_release/`, tests with `python tools/run_tests.py release`.
- What it reads: `release/targets.toml` (destinations, asset names, bench document length), `release/baselines.toml` (the
  pinned stored scores), `release/templates.toml` (every sentence and table made from data).
- What it writes: the work folder (`plan.json`, `receipts/`, `model/`, `record/`, `scores/`, `measure/`, `build/`,
  `publish/`), and in the checkout the filled documents, `site/assets/results.json`, `site/assets/links.json`, and
  `release/SHA256SUMS`.
- Assets, versions, and the Hub layout: RELEASING.md, "The assets" and "Versions and the Hub".
