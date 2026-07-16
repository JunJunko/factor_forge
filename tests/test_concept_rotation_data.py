from pathlib import Path

import pandas as pd

from scripts.concept_rotation_data import fetch_one_concept


def test_member_fetch_paginates_and_deduplicates(monkeypatch, tmp_path: Path):
    pages = {
        0: pd.DataFrame({
            "trade_date": ["20250101", "20250101"], "ts_code": ["C", "C"],
            "con_code": ["A", "A"], "name": ["a", "a"],
        }),
    }
    monkeypatch.setattr(
        "scripts.concept_rotation_data.query_with_retry",
        lambda endpoint, **kwargs: pages.get(kwargs["offset"], pd.DataFrame()),
    )
    rows, count = fetch_one_concept("C", "20250101", "20250102", tmp_path)
    assert rows == 1
    assert count == 1
    assert (tmp_path / "C.parquet").exists()
