"""Stages `bench` and `browser`: the release's speed and memory on one core.

`bench` builds the command line without a model inside it (a build of this checkout's engine at the pinned toolchain, in
the work folder), runs `litmus bench` on the release folder at four document lengths, and writes `measure/bench.json`
with the machine. `browser` takes the browser page's measurement, which needs a real browser and is made by hand
(`release/RELEASING.md`, The browser page) and adopted: the file must name this release's weights and the same document
length as `bench`'s, so a measurement of another model or document is refused.
"""

from __future__ import annotations

import json
import os
import platform
import tomllib

from pathlib import Path
from typing import Any

from . import receipts
from .building import load_tool
from .commands import StageError, run
from .stages import Context, Stage


LENGTHS = (128, 1024, 8192)
# `litmus bench` keeps the fastest of this many readings of each length. Three were too few on a desktop doing other
# work: on 2026-11-21 the document read at 4,531 tokens a second as the fastest of 3 and at 4,906 as the fastest of 7,
# within minutes of each other, where a quiet machine had given 4,971.
REPEATS = 7
BROWSER_KEYS = ("weights_sha256", "document_tokens", "tokens_per_second", "seconds", "browser", "threads", "machine")


def targets(context: Context) -> dict[str, Any]:
    return tomllib.loads((context.repository / "release" / "targets.toml").read_text(encoding="utf-8"))


def measure_folder(context: Context) -> Path:
    folder = context.work / "measure"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def processor_name(context: Context) -> str:
    """The processor's own name as the operating system gives it (such as `AMD Ryzen 7 3800X 8-Core Processor`), which
    the documents print beside the speed; `platform.processor()` names only its family."""
    system = platform.system()
    if system == "Windows":
        command = ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"]
    elif system == "Darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
    else:
        command = ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]
    name = " ".join(run(command, context.work, dict(os.environ), capture=True).split())
    if not name:
        raise StageError(f"the operating system gave no processor name ({' '.join(command)}); the documents name the machine a speed was measured on")
    return name


def bench(context: Context) -> None:
    folder = measure_folder(context)
    tools = load_tool(context.repository, "verify_build")
    tools.put_cargo_on_path()
    target = folder / "target"
    environment = {key: value for key, value in os.environ.items() if key not in ("RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS")}
    environment["CARGO_TARGET_DIR"] = str(target)
    run(["cargo", "build", "--release", "--locked", "-p", "litmus-cli"], context.repository / "engine", environment)
    cli = target / "release" / tools.executable("litmus")
    document_tokens = targets(context)["bench"]["document_tokens"]
    lengths = ",".join(str(length) for length in (*LENGTHS, document_tokens))
    output = run([str(cli), "bench", "--json", "--repeat", str(REPEATS), "--tokens", lengths, "--model", str(context.work / "model")], context.work, environment, capture=True)
    rows = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    if len(rows) != len(LENGTHS) + 1:
        raise StageError(f"`litmus bench` printed {len(rows)} readings for {len(LENGTHS) + 1} lengths:\n{output}")
    document = rows[-1]
    # The app and `litmus score` read a text on every logical processor, so the same document is measured that way too.
    processors = os.cpu_count() or 1
    output = run([str(cli), "bench", "--json", "--repeat", str(REPEATS), "--tokens", str(document_tokens), "--threads", str(processors), "--model", str(context.work / "model")],
                 context.work, environment, capture=True)
    every = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    if len(every) != 1 or every[0].get("threads") != processors:
        raise StageError(f"`litmus bench --threads {processors}` did not print one reading on {processors} threads:\n{output}")
    weights = receipts.file_digest(context.work / "model" / "model.safetensors")
    bench_record = {
        "weights_sha256": weights,
        "machine": {"system": f"{platform.system()} {platform.release()}", "name": processor_name(context), "processor": platform.processor(),
                    "logical_cores": os.cpu_count(), "backend": document["backend"]},
        "repeat": REPEATS,
        "rows": rows,
        "document": document,
        "document_all_processors": every[0],
    }
    (folder / "bench.json").write_text(json.dumps(bench_record, indent=1) + "\n", encoding="utf-8", newline="\n")
    context.note("tokens_per_second_at_document", round(document["text_tokens_per_second"]))
    context.note("tokens_per_second_on_all_processors", round(every[0]["text_tokens_per_second"]))


def bench_outputs(context: Context) -> list[Path]:
    return [context.work / "measure" / "bench.json"]


def browser_adopt(context: Context, source: Path) -> None:
    """Take the browser measurement in `source` (a JSON file, or a folder holding `browser.json`) after checking it."""
    path = source / "browser.json" if source.is_dir() else source
    if not path.is_file():
        raise StageError(f"{path} is not a file; adopt the browser measurement's JSON")
    measured = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in BROWSER_KEYS if key not in measured]
    if missing:
        raise StageError(f"{path} lacks {', '.join(missing)}; the measurement's fields are {', '.join(BROWSER_KEYS)}")
    weights = receipts.file_digest(context.work / "model" / "model.safetensors")
    if measured["weights_sha256"] != weights:
        raise StageError(f"{path} was measured with weights {measured['weights_sha256'][:12]}, and this release's are {weights[:12]}")
    bench_file = context.work / "measure" / "bench.json"
    if bench_file.is_file():
        tokens = json.loads(bench_file.read_text(encoding="utf-8"))["document"]["tokens"]
        if measured["document_tokens"] != tokens:
            raise StageError(f"the browser read {measured['document_tokens']} tokens and `bench` read {tokens}; measure the same document, or rerun bench")
    folder = measure_folder(context)
    (folder / "browser.json").write_text(json.dumps(measured, indent=1) + "\n", encoding="utf-8", newline="\n")


def browser_run(context: Context) -> None:
    raise StageError("the browser page's speed is measured by hand: serve the built page, read the document of `bench.json`'s length in a browser with one worker, "
                     "write measure/browser.json (fields: " + ", ".join(BROWSER_KEYS) + "), then `python release/cut.py adopt browser --work "
                     f"{context.work} --from <that file>` (release/RELEASING.md, The browser page)")


def browser_outputs(context: Context) -> list[Path]:
    return [context.work / "measure" / "browser.json"]


STAGES = (
    Stage("bench", ("fetch",), "speed and memory of the release on one core and on every processor, from `litmus bench`", bench, bench_outputs),
    Stage("browser", ("bench",), "the browser page's speed, measured by hand and adopted", browser_run, browser_outputs, adopt=browser_adopt),
)
