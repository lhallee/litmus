"""A small checkout and work folder with synthetic data, for the release driver's tests.

The numbers are made up for the tests and say nothing about any model.
"""

from __future__ import annotations

import json
import sys
import pytest

from pathlib import Path


RELEASE_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_FOLDER))

from litmus_release.plan import Plan  # noqa: E402
from litmus_release.stages import Context, Options  # noqa: E402

WEIGHTS = b"synthetic weights"
CONFIG = {
    "format": "litmus-student",
    "format_version": 3,
    "model": {"width": 192, "depth": 4, "heads": 3, "ffn_width": 768, "embed_width": 128, "attention": "float", "output": "float"},
    "head": {"hidden": 256, "out_width": 64, "clamp": 30.0},
    "reading": {"window_stride": 128},
    "bank": {"origins": ["human", "ai", "mixed"], "counts": [4096, 4096, 4093]},
    "calibration": {"calibration_rows": 67484, "length_bins": ["a", "b", "c"]},
    "provenance": {"step": 4999},
}
TARGETS = """
[github]
repo = "owner/litmus"
pages_branch = "gh-pages"
pages_url = "https://owner.github.io/litmus/"
training_guide = "https://example.org/guide"
[hub]
model_repo = "owner/litmus"
dataset_repo = "owner/litmus-corpus"
[contribute]
site_url = ""
[programs]
platforms = ["windows", "linux"]
signing_note = "the programs are not code-signed"
[programs.assets]
windows_app = "litmus-app.exe"
windows_cli = "litmus.exe"
linux_app = "litmus-app-linux-x86_64"
linux_cli = "litmus-linux-x86_64"
macos_arm64_app = "litmus-app-macos-arm64"
macos_x86_64_app = "litmus-app-macos-x86_64"
[bench]
document_tokens = 1000
[citation]
text = ""
"""
TEMPLATES = """
size = "`{model.config_name}` ({model.ternary_weights as m2} ternary weights)"
contribute = { if = "target.contribute.site_url", then = "{target.contribute.site_url}", else = "The contribute site, once it is live," }
table = '''
| Reader | PR AUC |
| --- | --- |
| Litmus | {results.rows[benchmark=test,detector=Litmus,task=ai_vs_human].pr_auc as f3} |
@if teacher.rows | Teacher | {teacher.rows[model=head].pr_auc as f3} |
'''
"""
RESULTS = {
    "model": {"name": "Litmus", "version": "1.0.0", "date": "2026-01-01"},
    "texts_per_benchmark": 3000,
    "benchmarks": [{"id": "test", "name": "Test", "kind": "two_class"}, {"id": "mage", "name": "MAGE", "kind": "two_class"}],
    "rows": [
        {"benchmark": "test", "detector": "Litmus", "task": "ai_vs_human", "stratum": "all", "pr_auc": 0.93456, "roc_auc": 0.9, "in_distribution": False},
        {"benchmark": "mage", "detector": "Litmus", "task": "ai_vs_human", "stratum": "all", "pr_auc": 0.8, "roc_auc": 0.81234, "in_distribution": True},
        {"benchmark": "mage", "detector": "Other", "task": "ai_vs_human", "stratum": "all", "pr_auc": 0.7, "roc_auc": 0.7, "in_distribution": False},
    ],
    "costs": [{"detector": "Litmus engine", "seconds_per_1k_words": 0.1321, "tokens_per_second": 23104.0, "measured": True, "machine": "Xeon / Linux"},
              {"detector": "Other", "seconds_per_1k_words": 3.5, "tokens_per_second": 353.2, "measured": True, "machine": "Xeon / Linux"}],
    "comparison": {"unseen_benchmarks": ["mage"],
                   "mean_pr_auc_unseen": [{"detector": "Litmus", "pr_auc": 0.8}, {"detector": "Other", "pr_auc": 0.7}]},
    "release": {"version": "", "date": "", "app_exe_bytes": None},
}
PAGE = """<html><body>
<p>about <span data-value="results.release.app_tokens_per_second" data-format="int">24,000</span> tokens a second</p>
<p>size <span class="tbd" data-value="results.release.app_exe_bytes" data-format="mb">being finalized</span></p>
</body></html>
"""
DOCUMENT = """# Title

The model is <!--litmus:t:size-->[PENDING]<!--/litmus--> and has <!--litmus:model.bank.human as int-->[PENDING]<!--/litmus--> human bank texts.

<!--litmus:t:table-->
[PENDING]
<!--/litmus-->

Written by hand, never touched.
"""


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "release").mkdir(parents=True)
    (root / "release" / "targets.toml").write_text(TARGETS, encoding="utf-8")
    (root / "release" / "templates.toml").write_text(TEMPLATES, encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "page.md").write_text(DOCUMENT, encoding="utf-8", newline="\n")
    (root / "site" / "assets").mkdir(parents=True)
    (root / "site" / "assets" / "results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    (root / "site" / "index.html").write_text(PAGE, encoding="utf-8", newline="\n")
    return root


@pytest.fixture
def plan() -> Plan:
    return Plan(version="1.0.0", repo="owner/model", revision="a" * 40, release_path="release", record_path="record", date="2026-01-01")


@pytest.fixture
def context(tmp_path: Path, repository: Path, plan: Plan) -> Context:
    work = tmp_path / "work"
    (work / "model").mkdir(parents=True)
    (work / "model" / "model.safetensors").write_bytes(WEIGHTS)
    (work / "model" / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (work / "model" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (work / "record").mkdir()
    (work / "receipts").mkdir()
    # The fetch receipt's weights hash is how a measurement is tied to the release's weights.
    (work / "receipts" / "fetch.json").write_text(json.dumps({"details": {"weights_sha256": __import__("hashlib").sha256(WEIGHTS).hexdigest()}}), encoding="utf-8")
    return Context(plan=plan, work=work, repository=repository, options=Options(), system="linux")
