from __future__ import annotations

import json
from pathlib import Path

from factor_forge.data import DataVersionRepository


def test_publish_is_versioned_and_hashes_files(tmp_path, panel):
    repository = DataVersionRepository(tmp_path / "data", tmp_path / "metadata.sqlite3")
    version = repository.publish(panel, source="test")
    resolved, loaded = repository.load_panel("latest")
    assert resolved == version
    assert len(loaded) == len(panel)
    manifest_path = tmp_path / "data" / "versions" / version / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_version"] == version
    assert "curated/stock_daily_panel.parquet" in manifest["files"]
    assert manifest["version_kind"] == "complete"


def test_latest_resolves_complete_version_not_newer_increment(tmp_path, panel):
    repository = DataVersionRepository(tmp_path / "data", tmp_path / "metadata.sqlite3")
    complete = repository.publish(panel, source="test", version_kind="complete")
    increment = repository.publish(
        panel.tail(1), source="test", version_kind="incremental"
    )
    assert repository.resolve("latest_any") == increment
    assert repository.resolve("latest") == complete

