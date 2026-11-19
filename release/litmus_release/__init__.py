"""The staged release driver: from a trained model on the Hub to a published Litmus release.

`python release/cut.py --help` lists the commands; `release/RELEASING.md` is the runbook and `release/SKILL.md` the
agent's guide. One module per concern: `plan` (what is being released), `receipts` and `stages` (the resumable engine),
`datapath`, `slots`, `values`, and `documents` (every published number comes from a data file), and one module per
group of stages (`hub`, `versions`, `benchmarking`, `building`, `measuring`, `packaging`, `publishing`).
"""
