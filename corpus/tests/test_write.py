from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataclasses import dataclass
from pathlib import Path
from litmus_corpus import catalog_check, storage
from litmus_corpus.catalog_check import check_source
from litmus_corpus.publish import PublicRepositoryError, publish

from conftest import spec_of


def test_row_groups_are_sized_by_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "MAX_ROW_GROUP_BYTES", 10_000)
    texts = ["x" * 3_000] * 10 + ["short"] * 100
    storage.write_parquet(pa.table({"text": texts}), tmp_path / "shard.parquet")

    metadata = pq.ParquetFile(tmp_path / "shard.parquet").metadata
    sizes = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]

    assert sum(sizes) == 110 and sizes[0] == 4  # four long rows pass 10,000 bytes; many short rows share a group
    assert pq.read_table(tmp_path / "shard.parquet").column("text").to_pylist() == texts


@dataclass
class Info:
    private: bool


@dataclass
class Commit:
    oid: str


class FakeApi:
    """Stands in for `HfApi`: a repository that exists with the given visibility, and the calls made."""

    def __init__(self, private: bool) -> None:
        self.private = private
        self.calls: list[str] = []

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool) -> None:
        assert private and exist_ok and repo_type == "dataset"
        self.calls.append("create")

    def repo_info(self, repo_id: str, repo_type: str) -> Info:
        return Info(self.private)

    def upload_folder(self, folder_path: str, repo_id: str, repo_type: str, commit_message: str) -> Commit:
        self.calls.append("upload")
        return Commit("abc")

    def dataset_info(self, repo_id: str, revision: str) -> None:
        if revision == "missing":
            raise FileNotFoundError(f"{repo_id}@{revision}")


def test_publish_goes_to_a_private_repository_only(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")

    private = FakeApi(private=True)
    assert publish(tmp_path, "owner/litmus-corpus", private) == "abc" and private.calls == ["create", "upload"]
    public = FakeApi(private=False)
    with pytest.raises(PublicRepositoryError):
        publish(tmp_path, "owner/litmus-corpus", public)
    assert "upload" not in public.calls


def test_catalog_check_finds_missing_columns_revisions_and_row_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog_check, "server_features", lambda spec, split: {"text", "model"} if split == "train" else None)
    monkeypatch.setattr(catalog_check, "server_rows", lambda spec: (2_000, False))
    spec = spec_of(name="checked", origin="ai", text_column="text", generator_column="model_name", split="train|test", rows="1000")

    found = check_source(spec, FakeApi(private=True), features_of=lambda spec, split: {"text"})

    assert found["problems"] == [
        "split train: columns not found: model_name (has model, text)",
        "split test: columns not found: model_name (has text)",
        "rows: the Hub has 2,000, the catalog 1,000",
    ]
    missing = spec_of(name="gone", origin="ai", text_column="text", generator="gpt-4o", revision="missing")
    assert check_source(missing, FakeApi(private=True))["problems"][0].startswith("revision missing")
