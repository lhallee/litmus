from __future__ import annotations

import hashlib
import json
import pytest

from litmus_release import documents as docs
from litmus_release.commands import StageError
from litmus_release.data import documents
from litmus_release.stages import Stage
from tests.conftest import WEIGHTS


def measured(context, *, browser: bool = True) -> None:
    sha = hashlib.sha256(WEIGHTS).hexdigest()
    folder = context.work / "measure"
    folder.mkdir()
    row = {"tokens": 1000, "text_tokens_per_second": 24012.6, "seconds": 0.04, "backend": "avx2", "process_peak_bytes": 66_000_000}
    (folder / "bench.json").write_text(json.dumps({"weights_sha256": sha, "document": row, "rows": [row], "machine": {"name": "n", "processor": "p", "system": "s", "backend": "avx2"}}), encoding="utf-8")
    if browser:
        (folder / "browser.json").write_text(json.dumps({"weights_sha256": sha, "document_tokens": 1000, "tokens_per_second": 5712.2, "seconds": 0.2, "browser": "Edge", "threads": 1, "machine": "m"}),
                                             encoding="utf-8")


def test_fill_writes_every_number_from_the_data_and_leaves_the_prose(context) -> None:
    measured(context)
    docs.fill(context)
    text = (context.repository / "docs" / "page.md").read_text(encoding="utf-8")
    assert "<!--litmus:t:size-->`s192_l4` (1.79M ternary weights)<!--/litmus-->" in text
    assert "<!--litmus:model.bank.human as int-->4,096<!--/litmus-->" in text
    assert "| Litmus | 0.935 |" in text and "Teacher" not in text
    assert text.endswith("Written by hand, never touched.\n")
    results = json.loads((context.repository / "site" / "assets" / "results.json").read_text(encoding="utf-8"))
    assert results["release"]["version"] == "1.0.0" and results["release"]["app_tokens_per_second"] == 24013
    assert results["release"]["browser_tokens_per_second"] == 5712 and results["release"]["speed_document_tokens"] == 1000
    page = (context.repository / "site" / "index.html").read_text(encoding="utf-8")
    assert ">24,013</span>" in page and "being finalized" in page
    links = json.loads((context.repository / "site" / "assets" / "links.json").read_text(encoding="utf-8"))
    assert links["download_linux_app"] == "https://github.com/owner/litmus/releases/download/v1.0.0/litmus-app-linux-x86_64"
    assert links["download_macos_app"] == "" and links["download_macos_intel_app"] == ""
    sums = (context.repository / "release" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert sums[0] == f"{hashlib.sha256(WEIGHTS).hexdigest()}  model.safetensors" and len(sums) == 3
    assert docs.unfinished(context) == []


def test_the_speed_on_every_processor_reaches_the_pages_once_it_is_measured(context) -> None:
    # A measurement made before `bench` read on every processor: the release block and the README's line say nothing of it.
    measured(context)
    available = documents(context, context.repository / "release" / "targets.toml")
    block = docs.release_block(available, {"app_tokens_per_second_all_processors": 1, "app_processors": 2})
    assert "app_tokens_per_second_all_processors" not in block and "app_processors" not in block
    assert block["app_tokens_per_second"] == 24013
    # With it, both carry the speed and the number of processors it was measured on.
    bench = context.work / "measure" / "bench.json"
    held = json.loads(bench.read_text(encoding="utf-8"))
    held["document_all_processors"] = {"tokens": 1000, "text_tokens_per_second": 161234.4, "seconds": 0.0062, "threads": 16, "backend": "avx2"}
    bench.write_text(json.dumps(held), encoding="utf-8")
    block = docs.release_block(documents(context, context.repository / "release" / "targets.toml"), {})
    assert (block["app_tokens_per_second_all_processors"], block["app_processors"], block["app_document_seconds_all_processors"]) == (161234, 16, 0.0)


def test_the_range_between_sources_reaches_the_site_and_draws_its_example_strips(context) -> None:
    measured(context)
    page = context.repository / "site" / "reading.html"
    strip = ('<div class="strip" data-chance="0.5" role="img"><div class="strip-bar"></div><div class="strip-band" style="--lo: 0.1; --hi: 0.2"></div><div class="strip-mark" style="--x: 0.5"></div></div>\n'
             '<div class="strip" data-chance="0.63" data-interval="0.42,0.83"><div class="strip-bar"></div><div class="strip-band" style="--lo: 0; --hi: 1"></div><div class="strip-mark" style="--x: 0.63"></div></div>\n'
             '<p>a chance of 50% reads <span data-value="results.release.chance_range.at_50_low" data-format="int">being finalized</span>% at its low end</p>\n')
    page.write_text(strip, encoding="utf-8", newline="\n")
    # A release whose table has no shift between sources: no block, and a band is the interval the strip states.
    docs.fill(context)
    assert "chance_range" not in json.loads((context.repository / "site" / "assets" / "results.json").read_text(encoding="utf-8"))["release"]
    text = page.read_text(encoding="utf-8")
    assert "--lo: 0.500; --hi: 0.500" in text and "--lo: 0.420; --hi: 0.830" in text and "being finalized" in text
    # With it, the block holds the ranges the programs print, and each band is widened by the shift on both sides.
    config = context.work / "model" / "config.json"
    held = json.loads(config.read_text(encoding="utf-8"))
    held["calibration"]["between_sources"] = {"level": 0.9, "log_odds": 1.431902, "sources": {"human": 37, "not_human": 75}}
    config.write_text(json.dumps(held), encoding="utf-8")
    docs.fill(context)
    block = json.loads((context.repository / "site" / "assets" / "results.json").read_text(encoding="utf-8"))["release"]["chance_range"]
    assert block == documents(context, context.repository / "release" / "targets.toml")["ranges"], "the Markdown documents read the same ranges"
    assert (block["level_percent"], block["odds"], block["human_sources"], block["not_human_sources"]) == (90, 4.2, 37, 75)
    assert (block["at_50_low"], block["at_50_high"], block["at_99_low"], block["at_99_high"], block["at_5_low"]) == (19, 81, 96, 99, 1)
    text = page.read_text(encoding="utf-8")
    assert "--lo: 0.193; --hi: 0.807" in text and "--lo: 0.147; --hi: 0.953" in text and ">19</span>% at its low end" in text
    # A mark that is not at the strip's chance is an error, never drawn.
    page.write_text(strip.replace("--x: 0.5", "--x: 0.6"), encoding="utf-8", newline="\n")
    with pytest.raises(StageError, match="mark is at 0.6 and its chance is 0.5"):
        docs.fill(context)


def test_the_site_links_both_macos_programs_once_the_platform_is_listed(context) -> None:
    measured(context)
    targets = context.repository / "release" / "targets.toml"
    targets.write_text(targets.read_text(encoding="utf-8").replace('platforms = ["windows", "linux"]', 'platforms = ["windows", "linux", "macos"]'), encoding="utf-8")
    docs.fill(context)
    links = json.loads((context.repository / "site" / "assets" / "links.json").read_text(encoding="utf-8"))
    base = "https://github.com/owner/litmus/releases/download/v1.0.0/"
    assert links["download_macos_app"] == base + "litmus-app-macos-arm64" and links["download_macos_intel_app"] == base + "litmus-app-macos-x86_64"


def test_fill_twice_changes_nothing(context) -> None:
    measured(context)
    docs.fill(context)
    before = {path: path.read_bytes() for path in [context.repository / "docs" / "page.md", context.repository / "site" / "index.html", context.repository / "site" / "assets" / "results.json"]}
    docs.fill(context)
    assert before == {path: path.read_bytes() for path in before}


def test_fill_fails_while_a_value_has_no_data_and_names_it(context) -> None:
    measured(context)
    page = context.repository / "docs" / "page.md"
    page.write_text(page.read_text(encoding="utf-8") + "\nMore: <!--litmus:record.corpus.rows as int-->[PENDING]<!--/litmus-->.\n", encoding="utf-8", newline="\n")
    with pytest.raises(Exception, match="record"):
        docs.fill(context)


def test_unfinished_reports_an_unfilled_slot_a_stray_marker_and_a_stale_value(context) -> None:
    measured(context)
    page = context.repository / "docs" / "page.md"
    assert any("not filled" in item for item in docs.unfinished(context))
    docs.fill(context)
    page.write_text(page.read_text(encoding="utf-8").replace("4,096", "4,097") + "\nThis is [PENDING: a number].\n", encoding="utf-8", newline="\n")
    left = docs.unfinished(context)
    assert any("stale" in item for item in left) and any("not a slot" in item for item in left)


def test_a_typed_number_on_the_site_with_no_data_is_an_error(context) -> None:
    measured(context, browser=False)
    page = context.repository / "site" / "index.html"
    page.write_text(page.read_text(encoding="utf-8").replace("results.release.app_tokens_per_second", "results.release.browser_tokens_per_second"), encoding="utf-8", newline="\n")
    with pytest.raises(StageError, match="typed number"):
        docs.fill(context)


def test_a_measurement_of_other_weights_is_refused(context) -> None:
    measured(context)
    path = context.work / "measure" / "bench.json"
    path.write_text(path.read_text(encoding="utf-8").replace(hashlib.sha256(WEIGHTS).hexdigest(), "0" * 64), encoding="utf-8")
    with pytest.raises(StageError, match="weights"):
        documents(context, context.repository / "release" / "targets.toml")["bench"]


def test_a_crlf_document_with_slots_is_refused_rather_than_mixed(context) -> None:
    page = context.repository / "docs" / "page.md"
    page.write_bytes(page.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(StageError, match="CRLF"):
        docs.unfinished(context)


def test_the_record_stage_names_what_the_documents_ask_of_it(context) -> None:
    page = context.repository / "docs" / "page.md"
    page.write_text(page.read_text(encoding="utf-8") + "\n<!--litmus:record.training.seed-->[PENDING]<!--/litmus-->\n", encoding="utf-8", newline="\n")
    (context.work / "record" / "record.toml").write_text(f'[identity]\nweights_sha256 = "{hashlib.sha256(WEIGHTS).hexdigest()}"\n[benchmarks]\nlitmus_trained_on = []\n', encoding="utf-8")
    with pytest.raises(StageError, match="record.training.seed"):
        docs.check_record(context)
    (context.work / "record" / "record.toml").write_text(f'[identity]\nweights_sha256 = "{hashlib.sha256(WEIGHTS).hexdigest()}"\n[benchmarks]\nlitmus_trained_on = []\n[training]\nseed = 0\n', encoding="utf-8")
    docs.check_record(context)
    assert (context.work / "data" / "record_check.json").is_file()


def test_a_record_of_other_weights_is_refused(context) -> None:
    (context.work / "record" / "record.toml").write_text('[identity]\nweights_sha256 = "' + "0" * 64 + '"\n', encoding="utf-8")
    with pytest.raises(StageError, match="another model"):
        docs.check_record(context)


def test_the_docs_stage_needs_the_four_stages_that_make_its_data() -> None:
    stage: Stage = docs.STAGES[1]
    assert stage.name == "docs" and set(stage.needs) == {"record", "results", "bench", "browser"}
