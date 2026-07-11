from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from conftest import make_panel
from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.data.freshness import FreshnessPolicy, MarketDataFreshnessService


class CalendarProvider:
    def __init__(self, dates: list[str]):
        self.dates = dates

    def query(self, endpoint: str, **kwargs):
        assert endpoint == "trade_cal"
        start = str(kwargs.get("start_date", "00000000"))
        end = str(kwargs.get("end_date", "99999999"))
        dates = [date for date in self.dates if start <= date <= end]
        return pd.DataFrame({
            "exchange": ["SSE"] * len(dates),
            "cal_date": dates,
            "is_open": [1] * len(dates),
            "pretrade_date": [None] * len(dates),
        })


def _project(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump({
        "project_name": "freshness_test", "timezone": "Asia/Shanghai",
        "paths": {
            "data_root": str(tmp_path / "data"),
            "metadata_db": str(tmp_path / "metadata.sqlite3"),
            "artifacts_root": str(tmp_path / "artifacts"),
        },
    }), encoding="utf-8")
    return load_project(path)


def _policy():
    return FreshnessPolicy(
        data_ready_after="18:00", min_last_day_rows=10,
        min_last_day_tradeable=10, min_last_day_liquid=10,
        max_required_missing_rate=0.05,
    )


def test_expected_latest_uses_previous_open_day_before_data_ready_time(tmp_path):
    project = _project(tmp_path)
    service = MarketDataFreshnessService(
        project, CalendarProvider(["20240212", "20240213"]), policy=_policy()
    )
    timezone = ZoneInfo("Asia/Shanghai")
    before = service.expected_latest_trade_date(datetime(2024, 2, 13, 17, 0, tzinfo=timezone))
    after = service.expected_latest_trade_date(datetime(2024, 2, 13, 19, 0, tzinfo=timezone))
    assert before == "2024-02-12"
    assert after == "2024-02-13"


def test_freshness_audit_blocks_stale_or_incomplete_tail(tmp_path):
    project = _project(tmp_path)
    panel = make_panel(days=30, stocks=20)
    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    repository.publish(panel, source="test", version_kind="complete")
    service = MarketDataFreshnessService(
        project, CalendarProvider([]), policy=_policy()
    )
    end = pd.Timestamp(panel["trade_date"].max())
    current = service.audit(end.strftime("%Y-%m-%d"))
    stale = service.audit((end + pd.offsets.BDay(1)).strftime("%Y-%m-%d"))
    assert current.status == "CURRENT"
    assert current.last_day_rows == 20
    assert stale.status == "STALE_OR_INCOMPLETE"
    assert stale.failures == (f"data_end<{(end + pd.offsets.BDay(1)):%Y-%m-%d}",)


def test_freshness_gate_rejects_today_before_configured_ready_time(tmp_path):
    project = _project(tmp_path)
    panel = make_panel(days=30, stocks=20)
    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    repository.publish(panel, source="test", version_kind="complete")
    service = MarketDataFreshnessService(project, CalendarProvider([]), policy=_policy())
    end = pd.Timestamp(panel["trade_date"].max())
    result = service.audit((end - pd.offsets.BDay(1)).strftime("%Y-%m-%d"))
    assert result.status == "STALE_OR_INCOMPLETE"
    assert "before_ready_cutoff" in result.failures[0]


def test_auto_sync_publishes_increment_then_new_complete_version(tmp_path, monkeypatch):
    project = _project(tmp_path)
    full = make_panel(days=31, stocks=20)
    cutoff = pd.Timestamp(full["trade_date"].max())
    base = full.loc[full["trade_date"].lt(cutoff)].copy()
    increment_panel = full.loc[full["trade_date"].eq(cutoff)].copy()
    repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    base_version = repository.publish(base, source="test", version_kind="complete")
    increment_version = repository.publish(
        increment_panel, source="test", version_kind="incremental"
    )

    class FakeIngestor:
        def __init__(self, *args, **kwargs):
            pass

        def ingest(self, start, end, *, version_kind):
            assert start == end == cutoff.strftime("%Y%m%d")
            assert version_kind == "incremental"
            return increment_version

    monkeypatch.setattr("factor_forge.data.freshness.TushareIngestor", FakeIngestor)
    provider = CalendarProvider([
        (cutoff - pd.offsets.BDay(1)).strftime("%Y%m%d"), cutoff.strftime("%Y%m%d")
    ])
    service = MarketDataFreshnessService(project, provider, policy=_policy())
    result = service.ensure_current(
        datetime(cutoff.year, cutoff.month, cutoff.day, 19, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    assert result.status == "CURRENT"
    assert result.synchronized is True
    assert result.incremental_versions == (increment_version,)
    assert result.data_version not in {base_version, increment_version}
    assert repository.resolve("latest") == result.data_version
