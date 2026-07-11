from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from factor_forge.config import ProjectConfig

from .ingestion import TushareIngestor
from .repository import DataVersionRepository


@dataclass(frozen=True)
class FreshnessPolicy:
    data_ready_after: str = "18:00"
    min_last_day_rows: int = 1_000
    min_last_day_tradeable: int = 500
    min_last_day_liquid: int = 500
    max_required_missing_rate: float = 0.05


@dataclass(frozen=True)
class FreshnessResult:
    status: str
    expected_latest_trade_date: str
    data_end_date: str
    data_version: str
    version_kind: str
    synchronized: bool
    incremental_versions: tuple[str, ...]
    last_day_rows: int
    last_day_tradeable: int
    last_day_liquid: int
    required_missing_rates: dict[str, float]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["incremental_versions"] = list(self.incremental_versions)
        payload["failures"] = list(self.failures)
        return payload


class MarketDataFreshnessService:
    REQUIRED_LAST_DAY_FIELDS = ("adj_open", "adj_close", "amount_cny", "turnover_rate")

    def __init__(
        self,
        project: ProjectConfig,
        provider,
        *,
        policy: FreshnessPolicy | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ):
        self.project = project
        self.provider = provider
        self.policy = policy or FreshnessPolicy()
        self.progress = progress
        self.repository = DataVersionRepository(
            project.paths.data_root, project.paths.metadata_db
        )

    def expected_latest_trade_date(self, now: datetime | None = None) -> str:
        timezone = ZoneInfo(self.project.timezone)
        local_now = now.astimezone(timezone) if now is not None else datetime.now(timezone)
        start = (pd.Timestamp(local_now.date()) - pd.Timedelta(days=35)).strftime("%Y%m%d")
        end = pd.Timestamp(local_now.date()).strftime("%Y%m%d")
        calendar = self.provider.query("trade_cal", exchange="SSE", start_date=start, end_date=end)
        open_dates = self._open_dates(calendar)
        ready = time.fromisoformat(self.policy.data_ready_after)
        today = local_now.strftime("%Y%m%d")
        if local_now.time() < ready:
            open_dates = [date for date in open_dates if date < today]
        else:
            open_dates = [date for date in open_dates if date <= today]
        if not open_dates:
            raise RuntimeError("trade calendar returned no completed open date")
        return pd.Timestamp(open_dates[-1]).strftime("%Y-%m-%d")

    def ensure_current(self, now: datetime | None = None) -> FreshnessResult:
        expected = self.expected_latest_trade_date(now)
        resolved, manifest = self.repository.load_manifest("latest")
        increments: list[str] = []
        if pd.Timestamp(manifest["end_date"]) < pd.Timestamp(expected):
            calendar = self.provider.query(
                "trade_cal", exchange="SSE",
                start_date=(pd.Timestamp(manifest["end_date"]) + pd.Timedelta(days=1)).strftime("%Y%m%d"),
                end_date=pd.Timestamp(expected).strftime("%Y%m%d"),
            )
            missing = self._open_dates(calendar)
            for start, end in self._contiguous_ranges(missing):
                increment = TushareIngestor(
                    self.project, self.provider, progress=self.progress
                ).ingest(start, end, version_kind="incremental")
                increments.append(increment)
                resolved = self._merge_increment(resolved, increment)
            resolved, manifest = self.repository.load_manifest("latest")
        return self.audit(expected, synchronized=bool(increments), increments=increments)

    def audit(
        self,
        expected_trade_date: str,
        *,
        synchronized: bool = False,
        increments: list[str] | None = None,
    ) -> FreshnessResult:
        resolved, manifest = self.repository.load_manifest("latest")
        end_date = str(manifest["end_date"])
        failures: list[str] = []
        if pd.Timestamp(end_date) < pd.Timestamp(expected_trade_date):
            failures.append(f"data_end<{expected_trade_date}")
        elif pd.Timestamp(end_date) > pd.Timestamp(expected_trade_date):
            failures.append(f"data_end>{expected_trade_date}_before_ready_cutoff")
        panel_path = (
            self.project.paths.data_root / "versions" / resolved
            / "curated" / "stock_daily_panel.parquet"
        )
        columns = [
            "trade_date", "ts_code", "is_tradeable", "is_liquid",
            *self.REQUIRED_LAST_DAY_FIELDS,
        ]
        day = pd.read_parquet(
            panel_path, columns=columns,
            filters=[("trade_date", "=", pd.Timestamp(end_date))],
        )
        rows = int(len(day))
        tradeable = int(day["is_tradeable"].fillna(False).astype(bool).sum())
        liquid = int(day["is_liquid"].fillna(False).astype(bool).sum())
        missing_rates = {
            field: float(day[field].isna().mean()) if rows else 1.0
            for field in self.REQUIRED_LAST_DAY_FIELDS
        }
        if rows < self.policy.min_last_day_rows:
            failures.append(f"last_day_rows<{self.policy.min_last_day_rows}")
        if tradeable < self.policy.min_last_day_tradeable:
            failures.append(f"last_day_tradeable<{self.policy.min_last_day_tradeable}")
        if liquid < self.policy.min_last_day_liquid:
            failures.append(f"last_day_liquid<{self.policy.min_last_day_liquid}")
        for field, rate in missing_rates.items():
            if rate > self.policy.max_required_missing_rate:
                failures.append(
                    f"{field}_missing_rate>{self.policy.max_required_missing_rate:.2%}"
                )
        status = "CURRENT" if not failures else "STALE_OR_INCOMPLETE"
        return FreshnessResult(
            status=status,
            expected_latest_trade_date=pd.Timestamp(expected_trade_date).strftime("%Y-%m-%d"),
            data_end_date=end_date,
            data_version=resolved,
            version_kind=str(manifest.get("version_kind") or "legacy_complete"),
            synchronized=synchronized,
            incremental_versions=tuple(increments or []),
            last_day_rows=rows,
            last_day_tradeable=tradeable,
            last_day_liquid=liquid,
            required_missing_rates=missing_rates,
            failures=tuple(failures),
        )

    def _merge_increment(self, base_version: str, increment_version: str) -> str:
        _, base = self.repository.load_panel(base_version)
        _, increment = self.repository.load_panel(increment_version)
        panel = pd.concat([base, increment], ignore_index=True)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = (
            panel.sort_values(["trade_date", "ts_code"])
            .drop_duplicates(["trade_date", "ts_code"], keep="last")
            .sort_values(["ts_code", "trade_date"])
            .reset_index(drop=True)
        )
        self._recompute_flags(panel)
        return self.repository.publish(
            panel, raw_datasets=None, source="tushare_append_auto_sync",
            version_kind="complete",
        )

    def _recompute_flags(self, panel: pd.DataFrame) -> None:
        grouped = panel.groupby("ts_code", sort=False)
        panel["listing_trade_days"] = grouped.cumcount() + 1
        panel["is_tradeable"] = (
            panel["raw_open"].notna() & panel["adj_open"].notna()
            & ~panel["is_suspended"].fillna(True).astype(bool)
        )
        panel["is_factor_eligible"] = (
            panel["is_tradeable"].fillna(False).astype(bool)
            & ~panel["is_st"].fillna(False).astype(bool)
            & ~panel["is_delisting_period"].fillna(False).astype(bool)
            & panel["listing_trade_days"].ge(self.project.data.listing_age_days)
        )
        liquidity = self.project.data.liquidity
        rolling_amount = (
            panel["amount_cny"].where(panel["amount_cny"] > 0)
            .groupby(panel["ts_code"], sort=False)
            .rolling(liquidity.window, min_periods=liquidity.min_traded_days)
            .mean().reset_index(level=0, drop=True).reindex(panel.index)
        )
        rank = rolling_amount.where(panel["is_tradeable"]).groupby(
            panel["trade_date"], sort=False
        ).rank(method="first", ascending=False)
        panel["is_liquid"] = rank.le(1000).fillna(False)

    @staticmethod
    def _open_dates(calendar: pd.DataFrame) -> list[str]:
        return sorted(
            calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"]
            .dropna().astype(str).unique().tolist()
        )

    @staticmethod
    def _contiguous_ranges(dates: list[str]) -> list[tuple[str, str]]:
        if not dates:
            return []
        # Tushare accepts a closed date interval; weekends inside the interval are harmless.
        ranges: list[tuple[str, str]] = []
        start = previous = pd.Timestamp(dates[0])
        for raw in dates[1:]:
            current = pd.Timestamp(raw)
            if (current - previous).days <= 4:
                previous = current
                continue
            ranges.append((start.strftime("%Y%m%d"), previous.strftime("%Y%m%d")))
            start = previous = current
        ranges.append((start.strftime("%Y%m%d"), previous.strftime("%Y%m%d")))
        return ranges
