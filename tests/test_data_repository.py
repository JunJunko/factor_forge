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

