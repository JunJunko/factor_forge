from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.data.tushare_provider import TushareProvider


Progress = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class TimingIngestResult:
    output_dir: str
    start_date: str
    end_date: str
    files: dict[str, dict]


class TushareTimingIngestor:
    """Fetch local raw tables needed by the timing feature builder.

    The ingestor writes simple parquet files under ``data/timing``. It keeps the
    files intentionally close to Tushare's raw schema so feature-definition code
    remains the only place that imposes modelling semantics.
    """

    def __init__(
        self,
        provider: TushareProvider,
        output_dir: str | Path = "data/timing",
        *,
        request_sleep: float = 0.12,
        progress: Progress | None = None,
    ):
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.request_sleep = request_sleep
        self.progress = progress

    def ingest(
        self,
        *,
        start_date: str,
        end_date: str,
        index_code: str = "000300.SH",
        project_config: str | Path = "configs/project.yaml",
        data_version: str = "latest",
        include_options: bool = True,
        include_futures: bool = True,
        include_moneyflow: bool = True,
        overwrite: bool = False,
    ) -> TimingIngestResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        calendar = self._trade_dates(start_date, end_date)
        files: dict[str, dict] = {}

        tasks: list[tuple[str, Callable[[], pd.DataFrame]]] = [
            ("index_daily", lambda: self._query_range("index_daily", ts_code=index_code, start_date=start_date, end_date=end_date)),
            ("index_dailybasic", lambda: self._query_range("index_dailybasic", ts_code=index_code, start_date=start_date, end_date=end_date)),
            ("bond_yield", lambda: self._safe_query_range("yc_cb", start_date=start_date, end_date=end_date)),
            ("margin", lambda: self._query_by_trade_date("margin", calendar)),
            ("cpi", lambda: self._with_macro_available_date(
                self._safe_query_range("cn_cpi", start_m=start_date[:6], end_m=end_date[:6]),
                kind="cpi",
            )),
            ("pmi", lambda: self._with_macro_available_date(
                self._safe_query_range("cn_pmi", start_m=start_date[:6], end_m=end_date[:6]),
                kind="pmi",
            )),
            ("stock_daily", lambda: self._stock_daily_from_version(project_config, data_version, start_date, end_date)),
        ]
        if include_options:
            tasks.extend([
                ("option_basic", self._option_basic),
                ("option_daily", lambda: self._query_by_trade_date("opt_daily", calendar)),
            ])
        if include_futures:
            tasks.extend([
                ("futures_basic", lambda: self._safe_query_range("fut_basic", exchange="CFFEX", fut_type="1")),
                ("futures_daily", lambda: self._query_by_trade_date("fut_daily", calendar, exchange="CFFEX")),
                ("futures_holding", lambda: self._query_by_trade_date("fut_holding", calendar, exchange="CFFEX")),
            ])
        if include_moneyflow:
            tasks.append(("moneyflow", lambda: self._query_by_trade_date("moneyflow_mkt_dc", calendar)))

        for position, (name, fetcher) in enumerate(tasks, start=1):
            path = self.output_dir / f"{name}.parquet"
            if path.exists() and not overwrite:
                frame = pd.read_parquet(path)
                files[name] = self._file_info(path, frame, skipped=True)
                self._emit(name, position, len(tasks), f"skip existing rows={len(frame)}")
                continue
            self._emit(name, position, len(tasks), "fetch")
            try:
                frame = fetcher()
            except Exception as exc:
                frame = pd.DataFrame({"error": [str(exc)]})
                error_path = self.output_dir / f"{name}.error.json"
                error_path.write_text(json.dumps({"endpoint": name, "error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
                files[name] = {"path": str(error_path), "rows": 0, "status": "FAILED", "error": str(exc)}
                self._emit(name, position, len(tasks), f"failed: {exc}")
                continue
            frame.to_parquet(path, index=False)
            files[name] = self._file_info(path, frame, skipped=False)
            self._emit(name, position, len(tasks), f"saved rows={len(frame)}")

        manifest = {
            "start_date": start_date,
            "end_date": end_date,
            "index_code": index_code,
            "files": files,
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return TimingIngestResult(str(self.output_dir), start_date, end_date, files)

    def _trade_dates(self, start_date: str, end_date: str) -> list[str]:
        calendar = self.provider.query("trade_cal", exchange="SSE", start_date=start_date, end_date=end_date)
        return calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"].astype(str).tolist()

    def _query_range(self, endpoint: str, **kwargs) -> pd.DataFrame:
        frame = self.provider.query(endpoint, **kwargs)
        return frame if frame is not None else pd.DataFrame()

    def _safe_query_range(self, endpoint: str, **kwargs) -> pd.DataFrame:
        try:
            return self._query_range(endpoint, **kwargs)
        except Exception:
            # Some Tushare endpoints use different parameter names across product
            # tiers. Keep the run alive; the feature builder tolerates empty
            # optional blocks.
            return pd.DataFrame()

    def _query_by_trade_date(self, endpoint: str, dates: list[str], **kwargs) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        staging = self.output_dir / "staging" / endpoint
        staging.mkdir(parents=True, exist_ok=True)
        total = len(dates)
        report_every = max(total // 25, 1)
        for position, date in enumerate(dates, start=1):
            path = staging / f"trade_date={date}.parquet"
            from_cache = path.exists()
            if from_cache:
                frame = pd.read_parquet(path)
            else:
                try:
                    frame = self.provider.query(endpoint, trade_date=date, **kwargs)
                except Exception:
                    frame = pd.DataFrame()
                if frame is None:
                    frame = pd.DataFrame()
                if "trade_date" not in frame:
                    frame["trade_date"] = date
                temporary = path.with_suffix(".parquet.tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
            if frame is not None and not frame.empty:
                frames.append(frame)
            if self.progress and (position == 1 or position % report_every == 0 or position == total):
                self.progress(endpoint, position, total, date)
            if self.request_sleep > 0 and not from_cache:
                time.sleep(self.request_sleep)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _option_basic(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for exchange in ["SSE", "SZSE", "CFFEX"]:
            try:
                frame = self.provider.query("opt_basic", exchange=exchange)
            except Exception:
                frame = pd.DataFrame()
            if frame is not None and not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()

    @staticmethod
    def _with_macro_available_date(frame: pd.DataFrame, *, kind: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        result = frame.copy()
        month_col = None
        for candidate in ["month", "MONTH"]:
            if candidate in result:
                month_col = candidate
                break
        if month_col is None:
            return result
        month_start = pd.to_datetime(result[month_col].astype(str), format="%Y%m", errors="coerce")
        if kind == "cpi":
            # CPI is normally released in the following month. Use a conservative
            # deterministic availability date when no official release timestamp
            # is present in the table.
            available = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=10)
        elif kind == "pmi":
            # Manufacturing PMI is usually available at month-end or the next
            # calendar day. MonthEnd+1 avoids same-month lookahead in daily joins.
            available = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        else:
            available = month_start
        result["available_date"] = available.dt.strftime("%Y%m%d")
        return result

    def _stock_daily_from_version(
        self,
        project_config: str | Path,
        data_version: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        project = load_project(Path(project_config))
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        _, panel = repository.load_panel(data_version)
        data = panel.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        start, end = pd.to_datetime(start_date, format="%Y%m%d"), pd.to_datetime(end_date, format="%Y%m%d")
        data = data.loc[data["trade_date"].between(start, end)]
        columns = [
            column for column in [
                "trade_date", "ts_code", "adj_close", "close", "pre_close", "pct_chg",
                "amount", "amount_cny", "is_tradeable", "is_liquid",
            ] if column in data.columns
        ]
        result = data[columns].copy()
        if "pct_chg" not in result and "adj_close" in result:
            result = result.sort_values(["ts_code", "trade_date"])
            result["pct_chg"] = result.groupby("ts_code")["adj_close"].pct_change(fill_method=None)
        if "amount" not in result and "amount_cny" in result:
            result["amount"] = result["amount_cny"]
        return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def _emit(self, stage: str, done: int, total: int, detail: str) -> None:
        if self.progress:
            self.progress(stage, done, total, detail)

    @staticmethod
    def _file_info(path: Path, frame: pd.DataFrame, *, skipped: bool) -> dict:
        dates = None
        if "trade_date" in frame and not frame.empty:
            parsed = pd.to_datetime(frame["trade_date"], errors="coerce")
            dates = {
                "start": str(parsed.min().date()) if parsed.notna().any() else None,
                "end": str(parsed.max().date()) if parsed.notna().any() else None,
            }
        return {
            "path": str(path),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "date_range": dates,
            "status": "SKIPPED" if skipped else "SAVED",
        }
