"""GPU-use parsing and aggregation against canned `nvidia-smi` output; nothing here needs a GPU."""

from __future__ import annotations

import json
import time
import pytest

from litmus_benchmarks import gpu_use
from litmus_benchmarks.gpu_use import (Sample, Sampler, baseline, between, build_report, document_files, parse_app_rows,
                                       parse_device, parse_gpu_row, process_memory, summarize)


def test_the_device_row_parses_and_a_driver_that_says_nothing_is_an_error() -> None:
    assert parse_gpu_row("37, 1234\n") == (37.0, 1234.0)
    assert parse_gpu_row(" 0 , 540 \n") == (0.0, 540.0)
    with pytest.raises(ValueError):
        parse_gpu_row("[N/A], 1234\n")
    with pytest.raises(ValueError):
        parse_gpu_row("")
    with pytest.raises(ValueError):
        parse_gpu_row("37, 1234\n12, 99\n")  # one device was asked for


def test_the_process_rows_parse_and_no_processes_is_an_empty_list() -> None:
    assert parse_app_rows("4312, 15234\n4400, 540\n") == [(4312, 15234.0), (4400, 540.0)]
    assert parse_app_rows("") == []
    assert parse_app_rows("No running processes found\n") == []
    assert parse_app_rows("4312, [N/A]\n") == [(4312, None)]  # a driver that lists a process but not its memory


def test_the_device_description_parses_with_commas_in_the_name() -> None:
    parsed = parse_device("NVIDIA GH200 480GB, 570.124.06, 97871\n")
    assert parsed == {"name": "NVIDIA GH200 480GB", "driver": "570.124.06", "memory_total_mib": 97871.0}
    assert parse_device("NVIDIA H100, PCIe, 550.54, 81559\n")["name"] == "NVIDIA H100, PCIe"


def poll(at: float, utilization: float, used: float, *apps: tuple[int, float | None]) -> Sample:
    return Sample(at, utilization, used, tuple(apps))


def test_summary_takes_the_own_process_by_id_and_names_the_bystanders() -> None:
    samples = [poll(1.0, 80, 20000, (500, 15000), (900, 3000)), poll(1.25, 100, 21000, (500, 15500), (900, 3100)),
               poll(1.5, 60, 20500)]
    out = summarize(samples, {500, 77})
    assert out["utilization_mean_percent"] == 80.0
    assert out["utilization_peak_percent"] == 100
    assert out["device_memory_used_peak_mib"] == 21000
    assert out["process_memory_peak_mib"] == 15500
    assert out["process_rows_seen"] == 2
    assert out["other_compute_processes"] == {"900": 3100.0}


def test_a_process_the_driver_never_lists_has_no_process_memory_and_falls_back_loudly() -> None:
    idle = baseline([poll(0.0, 0, 540), poll(0.25, 0, 560)])
    assert idle["device_memory_used_mib"] == 550
    own = summarize([poll(1.0, 0, 560), poll(1.25, 0, 575)], {500})
    assert own["process_memory_peak_mib"] is None
    memory, source = process_memory(own, idle)
    assert memory == 25.0 and "idle baseline" in source
    listed, listed_source = process_memory(summarize([poll(1.0, 50, 9000, (500, 8400))], {500}), idle)
    assert listed == 8400.0 and "matched by process id" in listed_source


def test_an_empty_window_is_an_error_unless_a_pilot_allows_it() -> None:
    with pytest.raises(AssertionError):
        summarize([], {1})
    assert summarize([], {1}, allow_empty=True)["polls"] == 0


def test_the_window_is_the_polls_between_two_times() -> None:
    samples = [poll(t / 4, 0, 0) for t in range(10)]
    assert [sample.at for sample in between(samples, 0.5, 1.0)] == [0.5, 0.75, 1.0]


def canned(command: list[str]) -> str:
    """What `nvidia-smi` prints for the two polls."""
    if command[2].startswith("--query-gpu"):
        return "42, 1000\n"
    return "123, 800\n"


def test_a_poll_reads_both_queries_and_the_sampler_collects_polls_then_stops() -> None:
    sampler = Sampler(canned, gpu=0, interval=0.01)
    one = sampler.poll()
    assert (one.utilization, one.used_mib, one.apps) == (42.0, 1000.0, ((123, 800.0),))
    sampler.start()
    time.sleep(0.15)
    samples = sampler.stop()
    assert len(samples) >= 3 and all(sample.apps == ((123, 800.0),) for sample in samples)
    assert gpu_use.achieved_interval(samples) < 0.1


def test_a_failing_poll_fails_the_measurement_instead_of_reading_zero() -> None:
    def broken(command: list[str]) -> str:
        raise RuntimeError("nvidia-smi exited 9")

    sampler = Sampler(broken, interval=0.01)
    sampler.start()
    time.sleep(0.05)
    with pytest.raises(RuntimeError, match="polling failed"):
        sampler.stop()


def detector_row(name: str, device: str, rows_seen: int) -> dict[str, object]:
    return {"detector": name, "device": device, "process_rows_seen": rows_seen}


def test_the_engine_row_says_whether_the_per_process_query_was_shown_to_work() -> None:
    meta = dict(device={"name": "g"}, interval=0.25, achieved=0.3, documents={}, idle={}, pilot=False)
    works = build_report(detectors=[detector_row("Binoculars", "cuda:0", 12), detector_row("Litmus engine", "cpu", 0)], **meta)
    assert works["per_process_query_works"] is True
    assert "no process of the engine was listed" in works["detectors"][1]["gpu_memory_process_note"]
    blind = build_report(detectors=[detector_row("Binoculars", "cuda:0", 0), detector_row("Litmus engine", "cpu", 0)], **meta)
    assert blind["per_process_query_works"] is False
    assert "listed no process on this machine" in blind["detectors"][1]["gpu_memory_process_note"]
    json.dumps(works)  # the report is plain JSON


def test_documents_come_in_folder_order_then_name_order_and_a_limit_takes_the_first(tmp_path_factory) -> None:
    first, second = tmp_path_factory.mktemp("human"), tmp_path_factory.mktemp("ai")
    for folder, names in ((first, ("b.txt", "a.txt")), (second, ("z.txt", "c.txt"))):
        for name in names:
            (folder / name).touch()
    (first / "notes.md").touch()
    assert [path.name for path in document_files([first, second], None)] == ["a.txt", "b.txt", "c.txt", "z.txt"]
    assert [path.name for path in document_files([first, second], 3)] == ["a.txt", "b.txt", "c.txt"]
    with pytest.raises(AssertionError):
        document_files([first.parent / "missing"], None)


def test_only_the_documents_a_draw_chose_are_read(tmp_path_factory, monkeypatch) -> None:
    folder = tmp_path_factory.mktemp("ai")
    for name in ("chosen.txt", "older_draw.txt", "also_chosen.txt"):
        (folder / name).touch()
    monkeypatch.setattr(gpu_use, "chosen_names", lambda path: {"chosen.txt", "also_chosen.txt"})
    assert [path.name for path in document_files([folder], None)] == ["also_chosen.txt", "chosen.txt"]
    monkeypatch.setattr(gpu_use, "chosen_names", lambda path: {"chosen.txt", "missing.txt"})
    with pytest.raises(AssertionError, match="lacks documents"):
        document_files([folder], None)
    monkeypatch.setattr(gpu_use, "chosen_names", lambda path: None)
    assert len(document_files([folder], None)) == 3


def test_the_engine_and_the_gpu_detectors_get_their_own_batch_sizes_and_environment() -> None:
    assert gpu_use.BATCH_SIZES["Litmus engine"] == 1
    assert gpu_use.worker_environment("Litmus engine")["OMP_NUM_THREADS"] == "1"
    assert gpu_use.LITMUS_ENGINE in gpu_use.DEFAULT_DETECTORS and gpu_use.DEFAULT_DETECTORS[-1] == gpu_use.LITMUS_ENGINE
    assert set(gpu_use.DEFAULT_DETECTORS[:-1]) <= set(gpu_use.GPU_DETECTORS), "every default is a detector the module knows"


FAKE_WORKER = (
    "import json, sys, time\n"
    "print('READY', flush=True)\n"
    "assert sys.stdin.readline().strip() == 'GO'\n"
    "time.sleep(0.4)\n"
    "print('RESULT ' + json.dumps({'documents': 5, 'words': 500, 'tokens': 800, 'seconds': 0.4, 'batch_size': 2,"
    " 'torch_cuda_max_allocated_mib': None}), flush=True)\n"
)


def test_one_detector_end_to_end_with_a_stand_in_process_and_canned_polls(monkeypatch) -> None:
    import queue
    import subprocess
    import sys

    def stand_in(detector, folders, limit, batch_size, device):
        return subprocess.Popen([sys.executable, "-c", FAKE_WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    monkeypatch.setattr(gpu_use, "start_worker", stand_in)
    idle = baseline([poll(0.0, 1, 500), poll(0.25, 3, 500)])
    row = gpu_use.measure_detector("Binoculars", [], None, run=canned, gpu=0, interval=0.02, idle=idle, idle_seconds=0.1,
                                   load_timeout=30, read_timeout=30, device="cuda:0")
    assert row["documents"] == 5 and row["tokens_read"] == 800 and row["tokens_per_second"] == 2000.0
    assert row["gpu_utilization_mean_percent"] == 42.0 and row["gpu_utilization_added_over_idle_percent"] == 40.0
    assert row["gpu_memory_process_peak_mib"] is None or row["gpu_memory_process_source"]
    assert row["other_compute_processes"] == {"123": 800.0}  # the canned process is not the stand-in's id
    assert row["polls_during_reading"] >= 3
    lines: queue.Queue = queue.Queue()
    lines.put("loading model")
    lines.put("READY")
    assert gpu_use.wait_for(lines, "READY", 1.0, "x") == "READY"
    with pytest.raises(TimeoutError):
        gpu_use.wait_for(lines, "RESULT ", 0.05, "x")
    lines.put(None)
    with pytest.raises(RuntimeError, match="ended before"):
        gpu_use.wait_for(lines, "RESULT ", 1.0, "x")
