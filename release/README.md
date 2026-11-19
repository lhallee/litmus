# Litmus release

Everything that turns a trained model into a published release. Start with [`RELEASING.md`](RELEASING.md), or, as an
agent, with [`SKILL.md`](SKILL.md): one command, `python release/cut.py run`, takes `<owner/name>@<commit>` and a
version through the stages to a complete release, and `status` says where it stands.

| File | What |
| --- | --- |
| [`cut.py`](cut.py), and `litmus_release/` beside it | The staged driver: receipts, slots, the stages, the gated publishing commands, the record bundle, the version setter |
| [`targets.toml`](targets.toml) | Where a release goes and the facts that do not change with a version: repositories, the Pages address, the asset names, the speed document's length |
| [`baselines.toml`](baselines.toml) | The pinned Hub commit of the other detectors' stored scores and costs, and the rows every detector reads |
| [`templates.toml`](templates.toml) | Each sentence and table that is made from data, written once and used by slot |
| [`model_card.md`](model_card.md) | The model repository's `README.md`: what a reading is, how it works, training data and procedure, evaluation, uncertainty and coverage, speed and memory, limits, use, license |
| [`dataset_card.md`](dataset_card.md) | The corpus repository's `README.md`; the compiled corpus is private, so the card describes its recipe, origins, processing, fields, splits, and licensing |
| [`release_notes.md`](release_notes.md) | The GitHub release's notes |
| [`build_release.py`](build_release.py) | Builds what a release ships from a trained release folder: `litmus-app.exe` and `litmus.exe` with the model built in, the github.io site with the browser page and its model files, the contribute server's page, `manifest.json`, and `SHA256SUMS` |
| [`build_linux.sh`](build_linux.sh) | The Linux programs, built in a pinned image |
| [`SHA256SUMS`](SHA256SUMS) | The SHA-256 of the model's three files, which the release-build workflow and `tools/verify_model.py` check |
| [`BUILD_PROVENANCE.md`](BUILD_PROVENANCE.md) | How release programs are built and how a download is checked |

A number in a card, a guide, or the README is a slot, text between a `litmus:key` comment and a `/litmus` comment, filled by the `docs` stage
from the release's own data files (the record folder on the Hub, the results, and the speed measurements), never typed.
A slot not yet filled holds the word PENDING in square brackets, so `python release/cut.py slots check --work <work folder>` lists what remains.
The cards' opening comments name the files they draw on.
