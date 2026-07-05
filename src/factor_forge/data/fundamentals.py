from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .tushare_provider import TushareProvider


INCOME_FIELDS = [
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "total_revenue", "revenue", "n_income_attr_p", "update_flag",
]
BALANCE_FIELDS = [
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int", "update_flag",
]
PIT_FIELDS = [
    "revenue_ttm", "net_assets", "roe_ttm", "revenue_growth_yoy",
    "roe_change_yoy", "debt_to_assets", "net_profit_ttm",
]


@dataclass(frozen=True)
class FundamentalBuildResult:
    output_path: Path
    rows: int
    securities: int
    first_available_date: str | None
    last_available_date: str | None


def quarter_periods(start_year: int, end_date: str | pd.Timestamp) -> list[str]:
    end = pd.Timestamp(end_date)
    periods = []
    for year in range(start_year, end.year + 1):
        for month_day in ["0331", "0630", "0930", "1231"]:
            period = pd.Timestamp(f"{year}{month_day}")
            if period <= end:
                periods.append(period.strftime("%Y%m%d"))
    return periods


def _effective_announcement(frame: pd.DataFrame) -> pd.Series:
    announced = pd.to_datetime(frame.get("ann_date"), errors="coerce")
    final_announced = pd.to_datetime(frame.get("f_ann_date"), errors="coerce")
    # Using the later date is conservative when the scheduled and actual fields differ.
    return pd.concat([announced, final_announced], axis=1).max(axis=1)


def _normalize_statement(frame: pd.DataFrame, value_fields: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["event_date"] = _effective_announcement(result)
    result["end_date"] = pd.to_datetime(result["end_date"], errors="coerce")
    if "report_type" in result:
        consolidated = result["report_type"].astype(str).eq("1")
        if consolidated.any():
            result = result.loc[consolidated]
    result = result.dropna(subset=["ts_code", "event_date", "end_date"])
    result["_updated"] = pd.to_numeric(result.get("update_flag", 0), errors="coerce").fillna(0)
    result["_completeness"] = result.reindex(columns=value_fields).notna().sum(axis=1)
    result = result.sort_values(
        ["ts_code", "end_date", "event_date", "_updated", "_completeness"]
    ).drop_duplicates(["ts_code", "end_date", "event_date"], keep="last")
    return result.drop(columns=["_updated", "_completeness"])


def _row_number(row: pd.Series | dict | None, fields: list[str]) -> float:
    if row is None:
        return np.nan
    for field in fields:
        value = row.get(field)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return np.nan


def _prior_period(period: pd.Timestamp, years: int = 1) -> pd.Timestamp:
    return period - pd.DateOffset(years=years)


def _ttm_value(
    statements: dict[pd.Timestamp, pd.Series], period: pd.Timestamp, fields: list[str]
) -> float:
    current = _row_number(statements.get(period), fields)
    if not np.isfinite(current):
        return np.nan
    if period.month == 12:
        return current
    previous_fy = pd.Timestamp(year=period.year - 1, month=12, day=31)
    previous_same = _prior_period(period)
    fy_value = _row_number(statements.get(previous_fy), fields)
    previous_value = _row_number(statements.get(previous_same), fields)
    if not np.isfinite(fy_value) or not np.isfinite(previous_value):
        return np.nan
    return current + fy_value - previous_value


def _snapshot(
    income: dict[pd.Timestamp, pd.Series],
    balance: dict[pd.Timestamp, pd.Series],
    period: pd.Timestamp,
) -> dict[str, float]:
    revenue_ttm = _ttm_value(income, period, ["total_revenue", "revenue"])
    profit_ttm = _ttm_value(income, period, ["n_income_attr_p"])
    net_assets = _row_number(
        balance.get(period),
        ["total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int"],
    )
    previous_period = _prior_period(period)
    previous_assets = _row_number(
        balance.get(previous_period),
        ["total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int"],
    )
    average_assets = np.nanmean([net_assets, previous_assets]) if (
        np.isfinite(net_assets) or np.isfinite(previous_assets)
    ) else np.nan
    roe_ttm = profit_ttm / average_assets if np.isfinite(average_assets) and average_assets > 0 else np.nan

    previous_revenue = _ttm_value(income, previous_period, ["total_revenue", "revenue"])
    revenue_growth = (
        revenue_ttm / previous_revenue - 1
        if np.isfinite(revenue_ttm) and np.isfinite(previous_revenue) and previous_revenue > 0
        else np.nan
    )
    previous_profit = _ttm_value(income, previous_period, ["n_income_attr_p"])
    two_year_assets = _row_number(
        balance.get(_prior_period(period, 2)),
        ["total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int"],
    )
    previous_average_assets = np.nanmean([previous_assets, two_year_assets]) if (
        np.isfinite(previous_assets) or np.isfinite(two_year_assets)
    ) else np.nan
    previous_roe = (
        previous_profit / previous_average_assets
        if np.isfinite(previous_profit) and np.isfinite(previous_average_assets)
        and previous_average_assets > 0 else np.nan
    )
    roe_change = roe_ttm - previous_roe if np.isfinite(roe_ttm) and np.isfinite(previous_roe) else np.nan
    total_assets = _row_number(balance.get(period), ["total_assets"])
    total_liabilities = _row_number(balance.get(period), ["total_liab"])
    debt_to_assets = (
        total_liabilities / total_assets
        if np.isfinite(total_liabilities) and np.isfinite(total_assets) and total_assets > 0
        else np.nan
    )
    return {
        "revenue_ttm": revenue_ttm,
        "net_assets": net_assets,
        "roe_ttm": roe_ttm,
        "revenue_growth_yoy": revenue_growth,
        "roe_change_yoy": roe_change,
        "debt_to_assets": debt_to_assets,
        "net_profit_ttm": profit_ttm,
    }


def _next_session(event_date: pd.Timestamp, sessions: np.ndarray) -> pd.Timestamp | pd.NaT:
    position = sessions.searchsorted(np.datetime64(event_date), side="right")
    return pd.Timestamp(sessions[position]) if position < len(sessions) else pd.NaT


def build_pit_fundamentals(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    trading_dates: pd.Series | pd.Index,
    securities: set[str] | None = None,
) -> pd.DataFrame:
    """Reconstruct statement revisions as they became knowable through time."""
    income = _normalize_statement(income, ["total_revenue", "revenue", "n_income_attr_p"])
    balance = _normalize_statement(
        balance,
        ["total_assets", "total_liab", "total_hldr_eqy_exc_min_int", "total_hldr_eqy_inc_min_int"],
    )
    if securities is not None:
        income = income.loc[income["ts_code"].isin(securities)]
        balance = balance.loc[balance["ts_code"].isin(securities)]
    sessions = pd.to_datetime(pd.Index(trading_dates).dropna().unique()).sort_values().to_numpy()
    rows: list[dict] = []
    income_by_code = {code: frame for code, frame in income.groupby("ts_code", sort=False)}
    balance_by_code = {code: frame for code, frame in balance.groupby("ts_code", sort=False)}
    codes = sorted(set(income_by_code).intersection(balance_by_code))
    for code in codes:
        stock_income = income_by_code[code]
        stock_balance = balance_by_code[code]
        income_events = {
            pd.Timestamp(event): frame.to_dict("records")
            for event, frame in stock_income.groupby("event_date", sort=False)
        }
        balance_events = {
            pd.Timestamp(event): frame.to_dict("records")
            for event, frame in stock_balance.groupby("event_date", sort=False)
        }
        events = sorted(set(income_events).union(balance_events))
        known_income: dict[pd.Timestamp, pd.Series] = {}
        known_balance: dict[pd.Timestamp, pd.Series] = {}
        previous_payload: tuple | None = None
        for event in events:
            for row in income_events.get(event, []):
                known_income[pd.Timestamp(row["end_date"])] = row
            for row in balance_events.get(event, []):
                known_balance[pd.Timestamp(row["end_date"])] = row
            common_periods = set(known_income).intersection(known_balance)
            if not common_periods:
                continue
            period = max(common_periods)
            values = _snapshot(known_income, known_balance, period)
            payload = (period, *(values[field] for field in PIT_FIELDS))
            if previous_payload is not None and all(
                (left == right) or (pd.isna(left) and pd.isna(right))
                for left, right in zip(payload, previous_payload)
            ):
                continue
            available_date = _next_session(pd.Timestamp(event), sessions)
            if pd.isna(available_date):
                continue
            rows.append({
                "ts_code": code,
                "available_date": available_date,
                "source_event_date": pd.Timestamp(event),
                "report_period": period,
                **values,
            })
            previous_payload = payload
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[
            "ts_code", "available_date", "source_event_date", "report_period", *PIT_FIELDS
        ])
    # Several weekend announcements can map to the same next session.  The last
    # source event is the complete information set available at that session.
    result = result.sort_values(["ts_code", "available_date", "source_event_date"])
    result = result.drop_duplicates(["ts_code", "available_date"], keep="last")
    return result.reset_index(drop=True)


class TushareFundamentalIngestor:
    def __init__(
        self,
        provider: TushareProvider,
        data_root: str | Path = "data",
        progress=None,
    ):
        self.provider = provider
        self.data_root = Path(data_root)
        self.progress = progress

    def ingest(
        self,
        *,
        start_year: int,
        end_date: str,
        trading_dates: pd.Series | pd.Index,
        securities: set[str],
        output_path: str | Path,
    ) -> FundamentalBuildResult:
        periods = quarter_periods(start_year, end_date)
        datasets = {
            "income_vip": (INCOME_FIELDS, []),
            "balancesheet_vip": (BALANCE_FIELDS, []),
        }
        total = len(periods) * len(datasets)
        completed = 0
        cache_root = self.data_root / "staging" / "fundamentals_tushare"
        for endpoint, (fields, frames) in datasets.items():
            for period_position, period in enumerate(periods):
                path = cache_root / endpoint / f"period={period}.parquet"
                complete_marker = path.with_suffix(".complete")
                if path.exists():
                    frame = pd.read_parquet(path)
                    # Legacy first-pass caches were fetched without pagination.
                    # A frame exactly at the API cap is potentially truncated.
                    suspected_cap = 7000 if endpoint == "balancesheet_vip" else 9000
                    refresh_recent = period_position >= len(periods) - 2
                    if len(frame) == suspected_cap or frame.empty or refresh_recent:
                        frame = self._fetch(endpoint, period, fields)
                        temporary = path.with_suffix(".parquet.tmp")
                        frame.to_parquet(temporary, index=False)
                        temporary.replace(path)
                        complete_marker.write_text("complete\n", encoding="utf-8")
                    elif not complete_marker.exists():
                        complete_marker.write_text("complete\n", encoding="utf-8")
                else:
                    frame = self._fetch(endpoint, period, fields)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".parquet.tmp")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                    complete_marker.write_text("complete\n", encoding="utf-8")
                frames.append(frame.reindex(columns=fields))
                completed += 1
                if self.progress:
                    self.progress(completed, total, endpoint, period, len(frame))
        income = pd.concat(
            [frame for frame in datasets["income_vip"][1] if not frame.empty],
            ignore_index=True,
        )
        balance = pd.concat(
            [frame for frame in datasets["balancesheet_vip"][1] if not frame.empty],
            ignore_index=True,
        )
        pit = build_pit_fundamentals(income, balance, trading_dates, securities)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        pit.to_parquet(temporary, index=False)
        temporary.replace(output)
        quality = {
            "rows": len(pit),
            "securities": int(pit["ts_code"].nunique()),
            "first_available_date": self._date_text(pit["available_date"].min()),
            "last_available_date": self._date_text(pit["available_date"].max()),
            "field_coverage": {field: float(pit[field].notna().mean()) for field in PIT_FIELDS},
            "future_date_violations": int((pit["source_event_date"] >= pit["available_date"]).sum()),
            "duplicate_keys": int(pit.duplicated(["ts_code", "available_date"]).sum()),
        }
        output.with_suffix(".quality.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return FundamentalBuildResult(
            output, len(pit), int(pit["ts_code"].nunique()),
            quality["first_available_date"], quality["last_available_date"],
        )

    def _fetch(self, endpoint: str, period: str, fields: list[str]) -> pd.DataFrame:
        # VIP statement endpoints have different hard caps (observed 7,000 for
        # balance sheets and 9,000 for income). Stay below both caps.
        page_size = 5000
        pages = []
        offset = 0
        while True:
            page = None
            for attempt in range(5):
                try:
                    page = self.provider.query(
                        endpoint, period=period, fields=",".join(fields),
                        limit=page_size, offset=offset,
                    )
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 12))
            if page is None or page.empty:
                break
            pages.append(page)
            if len(page) < page_size:
                break
            offset += page_size
        return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(columns=fields)

    @staticmethod
    def _date_text(value) -> str | None:
        return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()
