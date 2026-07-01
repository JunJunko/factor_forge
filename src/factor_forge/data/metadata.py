from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta_contract_version (
    contract_name TEXT NOT NULL,
    version TEXT NOT NULL,
    installed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (contract_name, version)
);
CREATE TABLE IF NOT EXISTS meta_dataset_registry (
    dataset_name TEXT PRIMARY KEY,
    storage_kind TEXT NOT NULL,
    source_name TEXT NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta_data_version (
    data_version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    quality_status TEXT NOT NULL CHECK (quality_status IN ('PASSED', 'FAILED'))
);
CREATE TABLE IF NOT EXISTS meta_ingestion_run (
    ingestion_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS meta_api_permission (
    endpoint TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS meta_quality_issue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_version TEXT,
    rule_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS dim_security (
    ts_code TEXT PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    exchange TEXT,
    market TEXT,
    list_date TEXT,
    delist_date TEXT
);
CREATE TABLE IF NOT EXISTS dim_trade_calendar (
    exchange TEXT NOT NULL,
    cal_date TEXT NOT NULL,
    is_open INTEGER NOT NULL,
    pretrade_date TEXT,
    PRIMARY KEY (exchange, cal_date)
);
CREATE TABLE IF NOT EXISTS dim_industry (
    industry_code TEXT PRIMARY KEY,
    industry_name TEXT NOT NULL,
    level TEXT NOT NULL,
    standard TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bridge_security_industry_history (
    ts_code TEXT NOT NULL,
    industry_code TEXT NOT NULL,
    industry_name TEXT,
    in_date TEXT NOT NULL,
    out_date TEXT,
    source_version TEXT NOT NULL,
    PRIMARY KEY (ts_code, industry_code, in_date, source_version)
);
"""


class MetadataStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO meta_contract_version(contract_name, version) VALUES (?, ?)",
                ("data_contract", "1.0.0"),
            )
            datasets = [
                ("stock_daily_panel", "PARQUET", "curated", "Standard daily factor panel"),
                ("daily", "PARQUET", "tushare", "Unadjusted daily OHLCV"),
                ("adj_factor", "PARQUET", "tushare", "Corporate-action adjustment factors"),
                ("daily_basic", "PARQUET", "tushare", "Daily market capitalization and turnover"),
                ("stk_limit", "PARQUET", "tushare", "Daily price limits"),
                ("suspend", "PARQUET", "tushare", "Point-in-time suspensions"),
                ("st_status", "PARQUET", "tushare", "Point-in-time risk-warning status"),
                ("industry_membership", "PARQUET", "tushare", "Point-in-time SW industry membership"),
                ("index_daily", "PARQUET", "tushare", "Market benchmark daily prices"),
            ]
            connection.executemany(
                "INSERT OR IGNORE INTO meta_dataset_registry(dataset_name,storage_kind,source_name,description) VALUES (?,?,?,?)",
                datasets,
            )

    def latest_version(self) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT data_version FROM meta_data_version WHERE quality_status='PASSED' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
