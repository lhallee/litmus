"""GPU use of named detectors reading long documents, measured with `nvidia-smi`, for the ad's speed shot.

    python -m litmus_benchmarks gpu-use --documents /work/sets/text/human /work/sets/text/ai --out /work/gpu_use/gpu_use.json

For each named detector, in a process of its own, the document folders are read in the same order while this process polls
`nvidia-smi` several times a second. The detector's model is loaded and warmed up on one document first, so the sampled
window is the reading alone. One JSON comes out (see `build_report`): the GPU's name and driver, the polling interval, the
document set, an idle baseline taken before anything runs, and per detector the mean and peak GPU utilization over its
reading, the utilization it adds over the idle baseline, the peak GPU memory of its own process, tokens and documents read,
wall seconds and throughput.

Litmus is measured by the same rules and not assumed: the engine as shipped (`Litmus engine`, the `litmus_engine` wheel,
CPU only, pinned to one core) runs under the same polling, and what it adds to the GPU is whatever the polling shows.

Utilization is `nvidia-smi`'s `utilization.gpu`: the share of the last sample period in which any kernel was running on
the GPU, so it is a time share, not a share of the chip's arithmetic. Memory of the detector's own process is the
per-process query (`--query-compute-apps`), matched by process id (the container's id, and the host's id that
`/proc/<pid>/sched` shows); when the machine's driver does not list the process, the report says so and gives the
device's memory in use over the idle baseline instead, never a silent substitute.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import statistics
import subprocess
import sys
import threading
import time

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# What is read from `nvidia-smi`: the device's utilization and memory in use, and every compute process on it.
GPU_QUERY = "--query-gpu=utilization.gpu,memory.used"
APPS_QUERY = "--query-compute-apps=pid,used_memory"
FORMAT = "--format=csv,noheader,nounits"
READY = "READY"
RESULT = "RESULT "
LITMUS_ENGINE = "Litmus engine"
# The open detectors of detectors.csv that have an adapter and run on a GPU, as the benchmark figures name them.
GPU_DETECTORS = (
    "Binoculars",
    "Fast-DetectGPT",
    "Desklib v1.01",
    "RADAR",
    "OpenAI RoBERTa-base (GPT-2)",
    "ModernBERT RAID+MAGE (Drayson)",
    "MAGE Longformer",
    "e5-small-lora",
    "RoBERTa ChatGPT (HC3)",
)
# The ad shows the detectors the benchmark results cover; the rest stay selectable with --detectors. MAGE Longformer does not load
# under the launch image's transformers (its config's id2label fails validation), and Binoculars (two 7B models) needs more than
# a 24 GB GPU: on an A10 its worker died loading, out of memory.
DEFAULT_DETECTORS = ("Fast-DetectGPT", "Desklib v1.01", "ModernBERT RAID+MAGE (Drayson)", "e5-small-lora", "Binoculars", LITMUS_ENGINE)
# Texts to a batch: the two 7B pairs and the 4,096-token Longformer take small ones; the engine reads one text at a time.
BATCH_SIZES = {"Binoculars": 4, "Fast-DetectGPT": 4, "MAGE Longformer": 4, LITMUS_ENGINE: 1}
DEFAULT_BATCH_SIZE = 16
PILOT_DOCUMENTS = 12

Runner = Callable[[Sequence[str]], str]


def run_smi(command: Sequence[str]) -> str:
    """The output of one `nvidia-smi` call; a failing call raises, so a broken measurement is never a quiet zero."""
    done = subprocess.run(list(command), capture_output=True, text=True, timeout=30, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} exited {done.returncode}: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


def parse_number(field_text: str, what: str) -> float | None:
    """One csv field as a number; `[N/A]` and `[Not Supported]` (what a driver prints for what it cannot say) are None."""
    text = field_text.strip()
    if text.startswith("["):
        return None
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"cannot read {what} from {field_text!r}") from error


def parse_gpu_row(output: str) -> tuple[float, float]:
    """Utilization (%) and memory in use (MiB) from `--query-gpu=utilization.gpu,memory.used` of one device."""
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected one device row, got {len(rows)}: {output!r}")
    fields = rows[0].split(",")
    if len(fields) != 2:
        raise ValueError(f"expected utilization and memory, got {rows[0]!r}")
    utilization, memory = parse_number(fields[0], "utilization.gpu"), parse_number(fields[1], "memory.used")
    if utilization is None or memory is None:
        raise ValueError(f"the driver reports no utilization or memory: {rows[0]!r}")
    return utilization, memory


def parse_app_rows(output: str) -> list[tuple[int, float | None]]:
    """Process id and memory (MiB, None when the driver does not say) of every compute process in `--query-compute-apps`
    output; no processes is an empty list (`nvidia-smi` prints nothing, or a sentence, then)."""
    rows: list[tuple[int, float | None]] = []
    for line in output.splitlines():
        if not re.match(r"^\s*\d+\s*,", line):
            continue  # an empty reply or a message such as "No running processes found"
        pid, memory = line.split(",", 1)
        rows.append((int(pid), parse_number(memory, "used_memory")))
    return rows


def parse_device(output: str) -> dict[str, object]:
    """Name, driver and total memory (MiB) from `--query-gpu=name,driver_version,memory.total`."""
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected one device row, got {len(rows)}: {output!r}")
    name, driver, total = (part.strip() for part in rows[0].rsplit(",", 2))
    return {"name": name, "driver": driver, "memory_total_mib": float(total)}


@dataclass(frozen=True)
class Sample:
    """One poll: when (seconds on this process's monotonic clock), the device's utilization and memory in use, and the
    compute processes listed."""

    at: float
    utilization: float
    used_mib: float
    apps: tuple[tuple[int, float | None], ...]


@dataclass
class Sampler:
    """Polls the GPU in a thread, `interval` seconds apart (a poll takes some tens of milliseconds, so the achieved interval
    is a little longer and is reported). Both queries run in the same poll, so a sample's device and process figures
    are one moment."""

    run: Runner
    gpu: int = 0
    interval: float = 0.25
    clock: Callable[[], float] = time.monotonic
    samples: list[Sample] = field(default_factory=list)
    error: BaseException | None = None
    stopping: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def poll(self) -> Sample:
        at = self.clock()
        utilization, used = parse_gpu_row(self.run(["nvidia-smi", f"--id={self.gpu}", GPU_QUERY, FORMAT]))
        apps = parse_app_rows(self.run(["nvidia-smi", f"--id={self.gpu}", APPS_QUERY, FORMAT]))
        return Sample(at, utilization, used, tuple(apps))

    def loop(self) -> None:
        try:
            while not self.stopping.is_set():
                started = self.clock()
                self.samples.append(self.poll())
                self.stopping.wait(max(0.0, self.interval - (self.clock() - started)))
        except Exception as error:  # re-raised by `stop`, so a failed poll fails the measurement
            self.error = error

    def start(self) -> None:
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self) -> list[Sample]:
        self.stopping.set()
        assert self.thread is not None, "start the sampler first"
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("nvidia-smi polling failed") from self.error
        return self.samples


def between(samples: Sequence[Sample], start: float, end: float) -> list[Sample]:
    return [sample for sample in samples if start <= sample.at <= end]


def summarize(samples: Sequence[Sample], pids: set[int], *, allow_empty: bool = False) -> dict[str, object]:
    """Utilization, device memory and process memory over some polls. `pids` are the ids the detector's process goes by;
    every other listed process is reported as a bystander with its peak memory. No polls is an error unless `allow_empty`
    (a pilot of a few documents can be shorter than one poll), which gives None for every figure."""
    if not samples and allow_empty:
        return {"polls": 0, "utilization_mean_percent": None, "utilization_peak_percent": None, "device_memory_used_peak_mib": None,
                "process_rows_seen": 0, "process_memory_peak_mib": None, "other_compute_processes": {}}
    assert samples, "no polls fell in the window: the reading was shorter than the polling interval; read more documents"
    utilization = [sample.utilization for sample in samples]
    own: list[float] = []
    own_rows = 0
    bystanders: dict[int, float] = {}
    for sample in samples:
        for pid, memory in sample.apps:
            if pid in pids:
                own_rows += 1
                if memory is not None:
                    own.append(memory)
            else:
                bystanders[pid] = max(bystanders.get(pid, 0.0), memory or 0.0)
    return {
        "polls": len(samples),
        "utilization_mean_percent": round(statistics.fmean(utilization), 2),
        "utilization_peak_percent": max(utilization),
        "device_memory_used_peak_mib": max(sample.used_mib for sample in samples),
        "process_rows_seen": own_rows,
        "process_memory_peak_mib": max(own) if own else None,
        "other_compute_processes": {str(pid): peak for pid, peak in sorted(bystanders.items())},
    }


def baseline(samples: Sequence[Sample]) -> dict[str, object]:
    """The idle window: nothing of ours is running, so whatever the GPU shows here is the machine's own."""
    assert samples, "no polls fell in the idle window"
    return {
        "polls": len(samples),
        "utilization_mean_percent": round(statistics.fmean(sample.utilization for sample in samples), 2),
        "utilization_peak_percent": max(sample.utilization for sample in samples),
        "device_memory_used_mib": statistics.median(sample.used_mib for sample in samples),
        "compute_processes_listed": sorted({pid for sample in samples for pid, _ in sample.apps}),
    }


def process_memory(own: dict[str, object], idle: dict[str, object]) -> tuple[float | None, str]:
    """The detector's own GPU memory and where it came from. A listed process wins. A process the driver never lists
    gets the device's memory over the idle baseline, labeled as such; `build_report` says whether the per-process query
    worked for any GPU detector, which is what makes an unlisted engine mean no GPU memory."""
    if own["process_memory_peak_mib"] is not None:
        return float(own["process_memory_peak_mib"]), "nvidia-smi --query-compute-apps, matched by process id"
    delta = float(own["device_memory_used_peak_mib"]) - float(idle["device_memory_used_mib"])
    return max(delta, 0.0), "the device's memory in use over the idle baseline (the driver did not list the process)"


def host_pid(pid: int) -> int | None:
    """The process's id as the host sees it, which is what the driver lists: the first line of `/proc/<pid>/sched` reads
    `name (hostpid, #threads: n)`. None where the file is not there."""
    try:
        first = Path(f"/proc/{pid}/sched").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    found = re.search(r"\((\d+),", first)
    return int(found.group(1)) if found else None


def chosen_names(folder: Path) -> set[str] | None:
    """The files a draw chose for this folder, when the folder is `<sets>/text/<kind>` and `<sets>/chosen.json` lists
    `<kind>`'s documents: a sets folder can hold documents of earlier draws, and only the chosen ones are the ad's."""
    manifest = folder.parent.parent / "chosen.json"
    if not manifest.is_file():
        return None
    chosen = json.loads(manifest.read_text(encoding="utf-8"))
    if folder.name not in chosen:
        return None
    return {document["file"] for document in chosen[folder.name]["documents"]}


def document_files(folders: Sequence[Path], limit: int | None) -> list[Path]:
    """The `.txt` documents of the folders, each folder's in name order and the folders in the order given, so every detector
    reads the same documents in the same order; where a draw's `chosen.json` names a folder's documents, only those; the
    first `limit` when a limit is set."""
    files: list[Path] = []
    for folder in folders:
        names = chosen_names(folder)
        files += [path for path in sorted(folder.glob("*.txt")) if names is None or path.name in names]
        assert names is None or {path.name for path in files if path.parent == folder} == names, f"{folder} lacks documents its chosen.json names"
    assert files, f"no .txt documents in {[str(folder) for folder in folders]}"
    return files if limit is None else files[:limit]


def describe_documents(folders: Sequence[Path], files: Sequence[Path], description: str) -> dict[str, object]:
    """The document set's description for the report: counts and lengths, the folders, and, when a `chosen.json` of the
    draw sits two folders up (`<sets>/text/<kind>`), its rule, seed and sources."""
    chars = [len(path.read_text(encoding="utf-8")) for path in files]
    out: dict[str, object] = {
        "description": description,
        "folders": [str(folder) for folder in folders],
        "documents": len(files),
        "words": sum(len(path.read_text(encoding="utf-8").split()) for path in files),
        "chars": {"min": min(chars), "median": statistics.median(chars), "max": max(chars)},
    }
    manifest = folders[0].parent.parent / "chosen.json"
    if manifest.is_file():
        chosen = json.loads(manifest.read_text(encoding="utf-8"))
        out["draw"] = {"seed": chosen.get("seed"), "rule": chosen.get("rule"),
                       "sources": {kind: chosen[kind].get("source") for kind in ("human", "ai") if kind in chosen}}
    return out


def worker_environment(detector: str) -> dict[str, str]:
    """The environment a detector's process starts with: one thread everywhere for the engine (as the cost measurement
    runs it), the container's own for the GPU detectors."""
    env = dict(os.environ)
    if detector == LITMUS_ENGINE:
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    return env


def start_worker(detector: str, folders: Sequence[Path], limit: int | None, batch_size: int, device: str) -> subprocess.Popen[str]:
    command = [sys.executable, "-m", "litmus_benchmarks", "gpu-use-worker", "--detector", detector, "--device", device,
               "--batch-size", str(batch_size), "--documents", *map(str, folders)]
    if limit is not None:
        command += ["--max-documents", str(limit)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=worker_environment(detector))


def read_lines(stream: object, lines: "queue.Queue[str | None]") -> None:
    for line in stream:  # type: ignore[attr-defined]
        lines.put(line.rstrip("\n"))
    lines.put(None)


def wait_for(lines: "queue.Queue[str | None]", prefix: str, timeout: float, detector: str) -> str:
    """The next worker line that starts with `prefix`; a worker that ends first, or is silent for `timeout`, is an error."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{detector}: no {prefix.strip()!r} within {timeout:.0f} s")
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError(f"{detector}: no {prefix.strip()!r} within {timeout:.0f} s") from error
        if line is None:
            raise RuntimeError(f"{detector}: the worker ended before {prefix.strip()!r}; see its messages above")
        if line.startswith(prefix):
            return line


def measure_detector(detector: str, folders: Sequence[Path], limit: int | None, *, run: Runner, gpu: int, interval: float,
                     idle: dict[str, object], idle_seconds: float, load_timeout: float, read_timeout: float,
                     device: str, allow_empty: bool = False) -> dict[str, object]:
    """One detector in a fresh process: a short idle window, then the process loads and warms up, then the reading is
    sampled from the moment it is released to the moment it reports."""
    sampler = Sampler(run, gpu, interval)
    sampler.start()
    time.sleep(idle_seconds)
    spawned = time.monotonic()
    worker = start_worker(detector, folders, limit, BATCH_SIZES.get(detector, DEFAULT_BATCH_SIZE), device)
    lines: "queue.Queue[str | None]" = queue.Queue()
    threading.Thread(target=read_lines, args=(worker.stdout, lines), daemon=True).start()
    try:
        wait_for(lines, READY, load_timeout, detector)
        pids = {worker.pid} | ({found} if (found := host_pid(worker.pid)) is not None else set())
        go = time.monotonic()
        assert worker.stdin is not None
        worker.stdin.write("GO\n")
        worker.stdin.flush()
        result = json.loads(wait_for(lines, RESULT, read_timeout, detector).removeprefix(RESULT))
        done = time.monotonic()
        worker.wait(timeout=60)
    except BaseException:
        worker.kill()
        sampler.stop()
        raise
    samples = sampler.stop()
    before = summarize(between(samples, samples[0].at, spawned), set())
    reading = summarize(between(samples, go, done), pids, allow_empty=allow_empty)
    lifetime = summarize(between(samples, spawned, done), pids)
    memory, source = process_memory(lifetime, idle)
    return {
        "detector": detector,
        "kind": "engine on the CPU, one core" if detector == LITMUS_ENGINE else "model on the GPU",
        "device": "cpu" if detector == LITMUS_ENGINE else device,
        "worker_exit_code": worker.returncode,
        "documents": result["documents"],
        "words": result["words"],
        "tokens_read": result["tokens"],
        "wall_seconds": result["seconds"],
        "tokens_per_second": round(result["tokens"] / result["seconds"], 1),
        "documents_per_second": round(result["documents"] / result["seconds"], 3),
        "batch_size": result["batch_size"],
        "gpu_utilization_mean_percent": reading["utilization_mean_percent"],
        "gpu_utilization_peak_percent": reading["utilization_peak_percent"],
        "gpu_utilization_added_over_idle_percent": None if reading["polls"] == 0 else round(float(reading["utilization_mean_percent"]) - float(idle["utilization_mean_percent"]), 2),
        "gpu_memory_process_peak_mib": memory,
        "gpu_memory_process_source": source,
        "gpu_memory_device_peak_over_idle_mib": max(float(lifetime["device_memory_used_peak_mib"]) - float(idle["device_memory_used_mib"]), 0.0),
        "torch_cuda_max_allocated_mib": result["torch_cuda_max_allocated_mib"],
        "process_ids_matched": sorted(pids),
        "process_rows_seen": lifetime["process_rows_seen"],
        "other_compute_processes": lifetime["other_compute_processes"],
        "idle_before_this_detector": {"polls": before["polls"], "utilization_mean_percent": before["utilization_mean_percent"],
                                      "device_memory_used_peak_mib": before["device_memory_used_peak_mib"]},
        "polls_during_reading": reading["polls"],
    }


def build_report(*, device: dict[str, object], interval: float, achieved: float, documents: dict[str, object],
                 idle: dict[str, object], detectors: list[dict[str, object]], pilot: bool) -> dict[str, object]:
    """The JSON the ad reads. `per_process_query_works` is true when some GPU detector's process was listed, which is what
    lets a Litmus row with no listed process mean "no GPU memory" and not "not measured"."""
    gpu_rows = [row for row in detectors if row["device"] != "cpu"]
    works = any(int(row["process_rows_seen"]) > 0 for row in gpu_rows)
    for row in detectors:
        if row["device"] == "cpu":
            row["gpu_memory_process_note"] = (
                "no process of the engine was listed on the GPU, which the per-process query shows for the GPU detectors" if works
                else "the per-process query listed no process on this machine, so this row's memory is the device's, over idle")
    return {
        "schema": 1,
        "pilot": pilot,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {**device, "cpu": platform.processor() or platform.machine(), "system": platform.system()},
        "sampling": {
            "tool": "nvidia-smi, polled: --query-gpu=utilization.gpu,memory.used and --query-compute-apps=pid,used_memory",
            "interval_requested_seconds": interval,
            "interval_achieved_seconds": round(achieved, 3),
            "utilization_means": "the share of each sample period in which any kernel ran on the GPU (nvidia-smi's utilization.gpu), averaged over the reading",
        },
        "documents": documents,
        "idle_baseline": idle,
        "per_process_query_works": works,
        "detectors": detectors,
    }


def achieved_interval(samples: Sequence[Sample]) -> float:
    gaps = [later.at - earlier.at for earlier, later in zip(samples, samples[1:])]
    return statistics.fmean(gaps) if gaps else float("nan")


def measure(*, folders: Sequence[Path], detectors: Sequence[str], out: Path, interval: float, idle_seconds: float, limit: int | None,
            description: str, gpu: int, device: str, load_timeout: float, read_timeout: float, pilot: bool,
            run: Runner = run_smi) -> dict[str, object]:
    """Idle baseline first, then each detector in its own process in the order given; the report is written once at the end
    and again after each detector, so a long run that stops keeps what it measured."""
    files = document_files(folders, limit)
    documents = describe_documents(folders, files, description)
    machine = parse_device(run(["nvidia-smi", f"--id={gpu}", "--query-gpu=name,driver_version,memory.total", FORMAT]))
    sampler = Sampler(run, gpu, interval)
    sampler.start()
    time.sleep(idle_seconds)
    idle_samples = sampler.stop()
    idle = baseline(idle_samples)
    rows: list[dict[str, object]] = []
    achieved = [achieved_interval(idle_samples)]
    for detector in detectors:
        print(f"== {datetime.now().strftime('%H:%M:%S')} {detector}", file=sys.stderr, flush=True)
        rows.append(measure_detector(detector, folders, limit, run=run, gpu=gpu, interval=interval, idle=idle,
                                     idle_seconds=min(idle_seconds, 3.0), load_timeout=load_timeout, read_timeout=read_timeout,
                                     device=device, allow_empty=pilot))
        report = build_report(device=machine, interval=interval, achieved=statistics.fmean(achieved), documents=documents,
                              idle=idle, detectors=rows, pilot=pilot)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def work(detector: str, folders: Sequence[Path], limit: int | None, batch_size: int, device: str) -> dict[str, object]:
    """The detector's process: load and warm up on the first document, say READY, wait for GO on standard input, read every
    document in batches, and report one RESULT line. Everything else it prints goes to standard error."""
    from .detectors import build_scorer

    files = document_files(folders, limit)
    texts = [path.read_text(encoding="utf-8") for path in files]
    if detector == LITMUS_ENGINE:
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cores[0]})  # one core: the engine uses no threads, and nothing else may share its process
    scorer = build_scorer(detector, "cpu" if detector == LITMUS_ENGINE else device)
    scorer(texts[:1])
    cuda = None
    if detector != LITMUS_ENGINE:
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cuda = torch
    print(READY, flush=True)
    assert sys.stdin.readline().strip() == "GO", "the parent sends GO"
    started = time.perf_counter()
    for begin in range(0, len(texts), batch_size):
        scorer(texts[begin:begin + batch_size])
    if cuda is not None:
        cuda.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "detector": detector,
        "documents": len(texts),
        "words": sum(len(text.split()) for text in texts),
        "tokens": sum(scorer.tokens(text) for text in texts),  # type: ignore[attr-defined]
        "seconds": seconds,
        "batch_size": batch_size,
        "torch_cuda_max_allocated_mib": None if cuda is None else cuda.cuda.max_memory_allocated() / 2**20,
    }
    print(RESULT + json.dumps(result), flush=True)
    return result
