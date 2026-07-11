from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from collections.abc import Callable
import shutil

import pandas as pd

from factor_forge.config import ProjectConfig
from .metadata import MetadataStore
from .panel import DailyPanelBuilder
from .repository import DataVersionRepository
from .tushare_provider import TushareProvider


class TushareIngestor:
    """Date-oriented ingestion; backtests never call this class."""

    def __init__(
        self, project: ProjectConfig, provider: TushareProvider,
        progress: Callable[[int, int, str], None] | None = None,
    ):
        self.project = project
        self.provider = provider
        self.metadata = MetadataStore(project.paths.metadata_db)
        self.repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        self.progress = progress

    def check_permissions(self) -> list[dict]:
        report = self.provider.permission_report()
        with self.metadata.connect() as connection:
            for item in report:
                connection.execute(
                    "INSERT OR REPLACE INTO meta_api_permission(endpoint,status,checked_at,detail) VALUES (?,?,?,?)",
                    (item["endpoint"], item["status"], item["checked_at"], item["detail"]),
                )
        return report

    def ingest(
        self, start_date: str, end_date: str, *, version_kind: str = "complete"
    ) -> str:
        ingestion_id = f"ingest_{uuid4().hex[:12]}"
        started = datetime.now(timezone.utc).isoformat()
        with self.metadata.connect() as connection:
            connection.execute(
                "INSERT INTO meta_ingestion_run(ingestion_id,started_at,start_date,end_date,status) VALUES (?,?,?,?,?)",
                (ingestion_id, started, start_date, end_date, "RUNNING"),
            )
        try:
            datasets = self._fetch(start_date, end_date)
            with pd.option_context("mode.copy_on_write", True):
                panel = DailyPanelBuilder(self.project).build(datasets)
                version = self.repository.publish(panel, datasets, version_kind=version_kind)
            self._persist_dimensions(datasets, version)
            self._cleanup_staging(start_date, end_date)
            with self.metadata.connect() as connection:
                connection.execute(
                    "UPDATE meta_ingestion_run SET finished_at=?, status=? WHERE ingestion_id=?",
                    (datetime.now(timezone.utc).isoformat(), "SUCCESS", ingestion_id),
                )
            return version
        except Exception as exc:
            with self.metadata.connect() as connection:
                connection.execute(
                    "UPDATE meta_ingestion_run SET finished_at=?, status=?, error_message=? WHERE ingestion_id=?",
                    (datetime.now(timezone.utc).isoformat(), "FAILED", str(exc), ingestion_id),
                )
            raise

    def _fetch(self, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        stocks = pd.concat([
            self.provider.query("stock_basic", exchange="", list_status=status,
                                fields="ts_code,symbol,name,market,exchange,list_status,list_date,delist_date")
            for status in ["L", "D", "P"]
        ], ignore_index=True).drop_duplicates("ts_code")
        stocks = stocks[stocks["exchange"].isin(self.project.data.exchanges)]
        stocks = stocks[stocks["market"].isin(["主板", "创业板", "科创板"])]
        calendar = self.provider.query("trade_cal", exchange="SSE", start_date=start_date, end_date=end_date)
        open_dates = calendar.loc[calendar["is_open"].astype(int) == 1, "cal_date"].tolist()
        dataset_names = ["daily", "adj_factor", "daily_basic", "stk_limit", "suspend", "st_status"]
        staging = self._staging_dir(start_date, end_date)
        staging.mkdir(parents=True, exist_ok=True)
        endpoint_map = {
            "daily": "daily", "adj_factor": "adj_factor", "daily_basic": "daily_basic",
            "stk_limit": "stk_limit", "suspend": "suspend_d", "st_status": "stock_st",
        }
        st_checked_dates: list[str] = []
        for position, date in enumerate(open_dates, start=1):
            for name, endpoint in endpoint_map.items():
                path = staging / name / f"trade_date={date}.parquet"
                if not path.exists():
                    frame = self.provider.query(endpoint, trade_date=date)
                    if "trade_date" not in frame:
                        frame["trade_date"] = date
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".parquet.tmp")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                if name == "st_status":
                    st_checked_dates.append(date)
            if self.progress and (position == 1 or position % 20 == 0 or position == len(open_dates)):
                self.progress(position, len(open_dates), date)
        result = {}
        required_columns = {
            "daily": ["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "vol", "amount", "pct_chg"],
            "adj_factor": ["trade_date", "ts_code", "adj_factor"],
            "daily_basic": ["trade_date", "ts_code", "total_mv", "circ_mv", "turnover_rate"],
            "stk_limit": ["trade_date", "ts_code", "up_limit", "down_limit"],
            "suspend": ["trade_date", "ts_code"],
            "st_status": ["trade_date", "ts_code", "name", "type"],
        }
        for name in dataset_names:
            files = sorted((staging / name).glob("trade_date=*.parquet"))
            frames = []
            for path in files:
                # A partition may be a degenerate empty file (only the trade_date
                # partition column, zero rows) when an upstream query returned
                # nothing for that day. Skip it — empty partitions contribute no
                # rows; reindex guarantees the required column set otherwise.
                frame = pd.read_parquet(path)
                if frame.empty:
                    continue
                frames.append(frame.reindex(columns=required_columns[name]))
            result[name] = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=required_columns[name])
            )
        result["stock_basic"] = stocks
        result["trade_calendar"] = calendar
        result["st_status_coverage"] = pd.DataFrame({"trade_date": st_checked_dates})
        try:
            result.update(self._fetch_industry_reference())
        except Exception:
            result["industry_classification"] = pd.DataFrame()
            result["industry_membership"] = pd.DataFrame()
        try:
            result["index_daily"] = self.provider.query(
                "index_daily", ts_code="000852.SH", start_date=start_date, end_date=end_date
            )
        except Exception:
            result["index_daily"] = pd.DataFrame()
        return result

    def _fetch_industry_reference(self) -> dict[str, pd.DataFrame]:
        level = self.project.data.industry_level.upper()
        member_code_field = f"{level.lower()}_code"
        member_name_field = f"{level.lower()}_name"
        industries = self.provider.query(
            "index_classify", level=level, src=self.project.data.industry_standard
        )
        member_frames = []
        for code in industries["index_code"].dropna().unique():
            for is_new in ["Y", "N"]:
                frame = self.provider.query(
                    "index_member_all", **{member_code_field: code, "is_new": is_new}
                )
                if not frame.empty:
                    member_frames.append(frame)
        members = pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame()
        name_map = industries.set_index("index_code")["industry_name"].to_dict()
        members = members.rename(
            columns={member_code_field: "industry_code", member_name_field: "industry_name"}
        )
        if "industry_name" not in members:
            members["industry_name"] = members["industry_code"].map(name_map)
        members = members.drop_duplicates(
            ["ts_code", "industry_code", "in_date", "out_date"], keep="last"
        )
        return {
            "industry_classification": industries,
            "industry_membership": members,
        }

    def _staging_dir(self, start_date: str, end_date: str):
        return self.project.paths.data_root / "staging" / f"tushare_{start_date}_{end_date}"

    def _cleanup_staging(self, start_date: str, end_date: str) -> None:
        path = self._staging_dir(start_date, end_date)
        root = (self.project.paths.data_root / "staging").resolve()
        resolved = path.resolve()
        if path.exists() and resolved.parent == root:
            shutil.rmtree(path)

    def _persist_dimensions(self, datasets: dict[str, pd.DataFrame], version: str) -> None:
        stocks = datasets["stock_basic"]
        calendar = datasets["trade_calendar"]
        industries = datasets.get("industry_classification", pd.DataFrame())
        members = datasets.get("industry_membership", pd.DataFrame())
        with self.metadata.connect() as connection:
            for row in stocks.to_dict("records"):
                connection.execute(
                    "INSERT OR REPLACE INTO dim_security(ts_code,symbol,name,exchange,market,list_date,delist_date) VALUES (?,?,?,?,?,?,?)",
                    tuple(row.get(key) for key in ["ts_code", "symbol", "name", "exchange", "market", "list_date", "delist_date"]),
                )
            for row in calendar.to_dict("records"):
                connection.execute(
                    "INSERT OR REPLACE INTO dim_trade_calendar(exchange,cal_date,is_open,pretrade_date) VALUES (?,?,?,?)",
                    (row.get("exchange", "SSE"), row.get("cal_date"), int(row.get("is_open", 0)), row.get("pretrade_date")),
                )
            for row in industries.to_dict("records"):
                code = row.get("index_code") or row.get("industry_code")
                name = row.get("industry_name") or row.get("name")
                if code and name:
                    connection.execute(
                        "INSERT OR REPLACE INTO dim_industry(industry_code,industry_name,level,standard) VALUES (?,?,?,?)",
                        (code, name, row.get("level", self.project.data.industry_level), self.project.data.industry_standard),
                    )
            for row in members.to_dict("records"):
                code = row.get("industry_code") or row.get("l1_code")
                in_date = row.get("in_date")
                if row.get("ts_code") and code and in_date:
                    connection.execute(
                        "INSERT OR REPLACE INTO bridge_security_industry_history(ts_code,industry_code,industry_name,in_date,out_date,source_version) VALUES (?,?,?,?,?,?)",
                        (row.get("ts_code"), code, row.get("industry_name") or row.get("l1_name"), in_date, row.get("out_date"), version),
                    )
