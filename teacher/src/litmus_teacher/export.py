"""Export what distillation needs, in `id` order: the chosen encoder's embeddings, the head, and its query and key.

The student trainer reads `manifest.json`. It names the embedding files (left where the embed step wrote them, float16
`(n, 2d)` mean then variance), the head (`head/pairhead.safetensors` and `head/pairhead.json`, which holds its widths,
scale and bias), and `query_key.npy`, float16 `(n, 2w)`: each row's unit query then unit key, zero for a row the
encoder did not embed (`present.npy` marks the rows that hold vectors). The teacher's pair logits for any two batches
are then `scale * (q_a . k_b + q_b . k_a) / 2 + bias`, computed on the fly (`head.pair_logits`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import numpy as np
import torch

from pathlib import Path

from .bank import chosen_head_name, head_vectors
from .embed import EMBEDDINGS_FILE, META_FILE
from .features import feature_source
from .head import CONFIG_FILE, WEIGHTS_FILE, PairHead
from .strata import write_json, write_summary


CHUNK_ROWS = 262_144
PROGRESS_FILE = "query_key.progress.json"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="split.parquet, named in the manifest")
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="the calibrate step's calibration.json, named in the manifest")
    parser.add_argument("--head", help="a head's directory name under heads/; default the chosen one")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="export only the first this many rows (a smoke run)")


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    device = torch.device("cuda")
    head_name = args.head or chosen_head_name(args.train_dir)
    head_dir = args.train_dir / "heads" / head_name
    record = json.loads((head_dir / "done.json").read_text(encoding="utf-8"))
    if record["features"] == "random":
        raise ValueError("the random-embedding control is not a teacher")
    head = PairHead.load(head_dir / "head").to(device).eval()
    source = feature_source(record["features"], args.embedding_root, record["seed"])
    rows = len(np.load(source.directories[0] / EMBEDDINGS_FILE, mmap_mode="r"))
    rows = min(rows, args.limit) if args.limit else rows
    present = source.embedded(rows)  # (n,)
    width = head.config["output_width"]  # 2w

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(head_dir / "head", args.out / "head", dirs_exist_ok=True)
    np.save(args.out / "present.npy", present)
    path, progress_path = args.out / "query_key.npy", args.out / PROGRESS_FILE
    done = json.loads(progress_path.read_text(encoding="utf-8"))["rows_done"] if progress_path.exists() else 0
    query_key = np.lib.format.open_memmap(path, mode="r+" if done else "w+", dtype=np.float16, shape=(rows, width))  # (n, 2w)
    for begin in range(done, rows, CHUNK_ROWS):
        ids = np.arange(begin, min(begin + CHUNK_ROWS, rows))  # (c,)
        ids = ids[present[ids]]
        if len(ids):
            query_key[ids] = head_vectors(head, source, ids, device).cpu().numpy().astype(np.float16)
        query_key.flush()
        write_json(progress_path, {"rows_done": int(min(begin + CHUNK_ROWS, rows))})

    manifest = {
        "head": {"directory": "head", "weights": WEIGHTS_FILE, "config": CONFIG_FILE, "name": head_name,
                 "features": record["features"], "input_width": head.config["input_width"],
                 "widths": head.config["widths"], "output_width": width, "scale": head.scale.item(),
                 "bias": head.bias.item(),
                 "pair_logit": "scale * (q_a . k_b + q_b . k_a) / 2 + bias; q is the first half of a row, k the second"},
        "embeddings": [{"path": (directory / EMBEDDINGS_FILE).as_posix(),
                        "meta": json.loads((directory / META_FILE).read_text(encoding="utf-8"))}
                       for directory in source.directories],
        "embedding_order": "the head's input is the listed embeddings concatenated, each mean then variance",
        "query_key": {"path": "query_key.npy", "shape": [rows, width], "dtype": "float16", "order": "id",
                      "present": "present.npy"},
        "split": args.split.as_posix(), "bank": (args.bank_dir / "bank.safetensors").as_posix(),
        "calibration": args.calibration.as_posix() if args.calibration else None,
    }
    write_json(args.out / "manifest.json", manifest)
    write_summary(args.out, "export", {"head": head_name, "rows": rows, "present_rows": int(present.sum()),
                                       "seconds": time.perf_counter() - started})
