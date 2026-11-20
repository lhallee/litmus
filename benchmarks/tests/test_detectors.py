from __future__ import annotations

from pathlib import Path

from litmus_benchmarks.detectors import processor_name


def test_processor_name_reads_an_x86_model_name(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz\n")
    assert processor_name(cpuinfo) == "Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz"


def test_processor_name_names_an_arm_core_by_its_part(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    core = "processor\t: {}\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU part\t: {}\n\n"
    cpuinfo.write_text(core.format(0, "0xd4f") + core.format(1, "0xd4f"))
    assert processor_name(cpuinfo) == "Arm Neoverse-V2"
    cpuinfo.write_text(core.format(0, "0xfff"))
    assert processor_name(cpuinfo) == "Arm CPU part 0xfff"
