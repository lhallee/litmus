"""Commands `bundle` and `record-template`: the record a trained model carries to its release.

A release reads two folders of the model's Hub repository: the release folder, and a record folder of the run's own
data files (the evaluation, the teacher's metrics, the corpus and split reports) and `record.toml`, the facts of the run
that no file holds (the recipe, the ablations' outcomes, what the findings were). `bundle` gathers them where the run's
outputs are, names each file as the documents read it, ties `record.toml` to the weights by their hash, and prints the
upload command; it uploads nothing. `record-template` writes `record.toml` with an empty entry for every key the
documents read, so nothing a document needs is forgotten.
"""

from __future__ import annotations

import re
import shutil
import tomllib

from pathlib import Path

from . import receipts, slots
from .commands import StageError
from .datapath import parse_path, split_key
from .documents import markdown_files, read_text
from .hub import RELEASE_FILES


def read_by(files: list[str]) -> str:
    return f"# read by {', '.join(sorted(set(files)))}"


def template(repository: Path) -> str:
    """The text of a `record.toml` with an empty string for each `record.*` key a document reads: a table for each group of
    plain keys, and for a key that selects a row of a list (`record.splits.rows[set=train].human`) one `[[array]]` entry per
    selected row."""
    plain: dict[tuple[str, ...], dict[str, list[str]]] = {}
    arrays: dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], dict[str, list[str]]] = {}
    for path in markdown_files(repository):
        text = read_text(path)
        if "<!--litmus:" not in text:
            continue
        for key in slots.keys(text):
            name = split_key(key)[0]
            if not name.startswith("record."):
                continue
            steps = parse_path(name)[1:]
            where = path.relative_to(repository).as_posix()
            selectors = [position for position, step in enumerate(steps) if isinstance(step, dict)]
            if not selectors:
                plain.setdefault(tuple(map(str, steps[:-1])), {}).setdefault(str(steps[-1]), []).append(where)
            elif len(selectors) == 1 and selectors[0] == len(steps) - 2 and all(isinstance(step, str) for step in steps[:selectors[0]]):
                selector = steps[selectors[0]]
                assert isinstance(selector, dict)
                arrays.setdefault((tuple(map(str, steps[:selectors[0]])), tuple(sorted(selector.items()))), {}).setdefault(str(steps[-1]), []).append(where)
            else:
                raise StageError(f"{key}: a record key is a plain path or one list row selected by fields, then a name")
    lines = ["# The facts of this run that no data file holds. Every entry is read by a document; fill each with what the run did and found.",
             "# Prose is the writer's: state what happened, with its numbers and where they came from. A number is written as a number.", "",
             "[benchmarks]", "# Benchmarks this model trained on (an empty list if none), and row filters such as raid_extra:language=cs,de.",
             "litmus_trained_on = []", "row_filter = []", ""]
    for table, entries in sorted(plain.items()):
        if table:
            lines.append(f"[{'.'.join(table)}]")
        for name, files in sorted(entries.items()):
            lines += [read_by(files), f'{name} = ""']
        lines.append("")
    for (table, selector), entries in sorted(arrays.items()):
        lines.append(f"[[{'.'.join(table)}]]")
        lines += [f'{field} = "{value}"' for field, value in selector]
        for name, files in sorted(entries.items()):
            lines += [read_by(files), f'{name} = ""']
        lines.append("")
    return "\n".join(lines)


def copy_named(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise StageError(f"{source} is not a file")
    shutil.copy2(source, destination)


def bundle(release: Path, out: Path, *, record: Path, evaluation_test: Path, evaluation_validation: Path | None, teacher: Path | None, extra: dict[str, Path]) -> list[Path]:
    """Write `out/release/` and `out/record/` from the run's outputs; returns the files written."""
    missing = [name for name in RELEASE_FILES if not (release / name).is_file()]
    if missing:
        raise StageError(f"{release} lacks {', '.join(missing)}")
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; choose a new folder")
    (out / "release").mkdir(parents=True)
    (out / "record").mkdir()
    written = [Path(shutil.copy2(release / name, out / "release" / name)) for name in RELEASE_FILES]
    weights = receipts.file_digest(release / "model.safetensors")
    text = record.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    stated = (parsed.get("identity") or {}).get("weights_sha256")
    if stated is None:
        text += f'\n[identity]\nweights_sha256 = "{weights}"\n'
    elif stated != weights:
        raise StageError(f"{record} states weights {stated[:12]}, and {release} holds {weights[:12]}")
    (out / "record" / "record.toml").write_text(text, encoding="utf-8", newline="\n")
    written.append(out / "record" / "record.toml")
    folder = evaluation_test if evaluation_test.is_dir() else evaluation_test.parent
    copy_named(folder / "evaluation.json", out / "record" / "evaluation_test.json")
    written.append(out / "record" / "evaluation_test.json")
    if (folder / "evaluation.csv").is_file():
        copy_named(folder / "evaluation.csv", out / "record" / "evaluation_test.csv")
        written.append(out / "record" / "evaluation_test.csv")
    if evaluation_validation is not None:
        copy_named(evaluation_validation / "evaluation.csv", out / "record" / "evaluation_validation.csv")
        written.append(out / "record" / "evaluation_validation.csv")
    if teacher is not None:
        for source, name in (("metrics.csv", "teacher_metrics.csv"), ("uncertainty.csv", "teacher_uncertainty.csv")):
            copy_named(teacher / source, out / "record" / name)
            written.append(out / "record" / name)
    for name, source in extra.items():
        if not re.match(r"^[a-z0-9_]+\.(json|csv|toml)$", name):
            raise StageError(f"{name}: a record file is named in lower case with underscores and ends .json, .csv, or .toml")
        copy_named(source, out / "record" / name)
        written.append(out / "record" / name)
    return written
