"""Count what RAID's labeled splits hold, at the revision `corpus/sources.csv` pins, before the corpus reads them.

    python launch/raid_survey.py --out /work/corpus2/raid_survey.json [--limit-rows N]

Downloads `train.csv` and `extra.csv` of `liamdugan/raid` into the Hugging Face cache and writes, per split, the rows by
domain, model, decoding, repetition penalty and attack, the attacks seen on human rows, and how `id`, `source_id` and
`adv_source_id` relate (which row an attacked row came from, and which human document a generation derives from).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import pyarrow as pa
import pyarrow.csv as pacsv

from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPOSITORY = "liamdugan/raid"
REVISION = "865cac74188466cb0c3b7574a10204007b57a459"
FILES = {"train": "train.csv", "extra": "extra.csv"}
KEYS = ("domain", "model", "decoding", "repetition_penalty", "attack")
BLOCK_BYTES = 64 << 20


def survey_split(path: Path, limit: int | None) -> dict[str, object]:
    """Counts of one CSV split, streamed in blocks."""
    options = pacsv.ReadOptions(block_size=BLOCK_BYTES)
    parse = pacsv.ParseOptions(newlines_in_values=True)
    convert = pacsv.ConvertOptions(column_types={name: pa.string() for name in ("id", "adv_source_id", "source_id", *KEYS)},
                                   include_columns=["id", "adv_source_id", "source_id", *KEYS])
    cells: collections.Counter[tuple[str, ...]] = collections.Counter()
    human_attacks: collections.Counter[str] = collections.Counter()
    relations: collections.Counter[str] = collections.Counter()
    human_ids: set[str] = set()
    source_ids: set[str] = set()
    adv_sources: set[str] = set()
    unattacked_ids: set[str] = set()
    rows = 0
    reader = pacsv.open_csv(path, read_options=options, parse_options=parse, convert_options=convert)
    progress = tqdm(desc=path.name, unit="row", mininterval=30, file=sys.stderr)
    for batch in reader:
        table = batch.to_pydict()
        for index in range(batch.num_rows):
            row = {name: table[name][index] for name in table}
            cells[tuple(str(row[key]) for key in KEYS)] += 1
            attack = str(row["attack"])
            if row["model"] == "human":
                human_attacks[attack] += 1
                if attack == "none":
                    human_ids.add(row["id"])
                    relations["human_none: source_id == id" if row["source_id"] == row["id"] else "human_none: source_id != id"] += 1
            if attack == "none":
                unattacked_ids.add(row["id"])
                relations["none: adv_source_id == id" if row["adv_source_id"] == row["id"] else "none: adv_source_id != id"] += 1
            else:
                adv_sources.add(row["adv_source_id"])
            source_ids.add(row["source_id"])
        rows += batch.num_rows
        progress.update(batch.num_rows)
        if limit is not None and rows >= limit:
            break
    progress.close()
    relations["attacked rows whose adv_source_id is an unattacked row"] = len(adv_sources & unattacked_ids)
    relations["attacked rows' distinct adv_source_id"] = len(adv_sources)
    relations["distinct source_id"] = len(source_ids)
    relations["source_id that is a human unattacked row"] = len(source_ids & human_ids)
    by_key: dict[str, dict[str, int]] = {}
    for position, key in enumerate(KEYS):
        totals: collections.Counter[str] = collections.Counter()
        for cell, count in cells.items():
            totals[cell[position]] += count
        by_key[key] = dict(sorted(totals.items()))
    origin_attack: collections.Counter[str] = collections.Counter()
    for cell, count in cells.items():
        origin_attack[f"{'human' if cell[1] == 'human' else 'model'}|{cell[4]}"] += count

    return {"rows": rows, "by": by_key, "origin_attack": dict(sorted(origin_attack.items())), "human_attacks": dict(human_attacks),
            "relations": dict(relations), "cells": {"|".join(cell): count for cell, count in sorted(cells.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-rows", type=int, default=None, help="stop each split after about this many rows (a pilot)")
    arguments = parser.parse_args()
    report: dict[str, object] = {"repository": REPOSITORY, "revision": REVISION}
    for split, name in FILES.items():
        path = Path(hf_hub_download(REPOSITORY, name, repo_type="dataset", revision=REVISION))
        report[split] = survey_split(path, arguments.limit_rows)
        print(json.dumps({split: report[split]["origin_attack"]}), file=sys.stderr, flush=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
