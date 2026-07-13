from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from factor_forge.config import ProjectConfig
from factor_forge.exceptions import DataQualityError

from .repository import DataVersionRepository
from .tushare_provider import TushareProvider


@dataclass(frozen=True)
class MoneyflowEnrichmentResult:
    base_data_version: str
    start_date: str
    end_date: str
    mainboard_rows: int
    moneyflow_rows: int
    coverage: float
    amount_bound_violation_ratio: float
    published_data_version: str | None
    staging_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


class MainboardMoneyflowEnricher:
    """Add Tushare daily active moneyflow to an immutable daily panel.

    The builder reuses an existing immutable market panel, fetches only the
    missing moneyflow endpoint, filters through the Tushare security master,
    and publishes to a separate main-board repository. Staging partitions are
    resumable and are retained after audit-only runs.
    """

    def __init__(
        self,
        base_project: ProjectConfig,
        target_project: ProjectConfig,
        provider: TushareProvider,
    ):
        self.base_project = base_project
        self.target_project = target_project
        self.provider = provider
        self.base_repository = DataVersionRepository(
            base_project.paths.data_root, base_project.paths.metadata_db
        )
        self.target_repository = DataVersionRepository(
            target_project.paths.data_root, target_project.paths.metadata_db
        )

    def run(
        self,
        *,
        base_version: str,
        start_date: str,
        end_date: str,
        publish: bool = False,
        min_coverage: float = 0.90,
        progress=None,
    ) -> MoneyflowEnrichmentResult:
        resolved, manifest = self.base_repository.load_manifest(base_version)
        panel_path = (
            self.base_project.paths.data_root / "versions" / resolved
            / "curated" / "stock_daily_panel.parquet"
        )
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        base = pq.read_table(
            panel_path,
            filters=[("trade_date", ">=", start.to_pydatetime()),
                     ("trade_date", "<=", end.to_pydatetime())],
        ).to_pandas()
        if base.empty:
            raise DataQualityError("Base panel has no rows in the requested enrichment interval")

        securities = self._mainboard_security_master()
        allowed = set(securities["ts_code"].dropna().astype(str))
        base = base.loc[base["ts_code"].astype(str).isin(allowed)].copy()
        if base.empty:
            raise DataQualityError("No main-board rows remain after security-master filtering")

        dates = sorted(pd.to_datetime(base["trade_date"]).dt.strftime("%Y%m%d").unique())
        staging = (
            self.target_project.paths.data_root / "staging"
            / f"moneyflow_order_size_v2_{resolved}_{dates[0]}_{dates[-1]}"
        )
        staging.mkdir(parents=True, exist_ok=True)
        for position, date in enumerate(dates, start=1):
            path = staging / f"trade_date={date}.parquet"
            if not path.exists():
                frame = self.provider.query("moneyflow", trade_date=date)
                frame = frame.reindex(columns=[
                    "trade_date", "ts_code", "net_mf_amount",
                    "buy_sm_amount", "sell_sm_amount", "buy_lg_amount", "sell_lg_amount",
                    "buy_elg_amount", "sell_elg_amount",
                ])
                frame = frame.loc[frame["ts_code"].astype(str).isin(allowed)].copy()
                temporary = path.with_suffix(".parquet.tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
            if progress and (position == 1 or position % 50 == 0 or position == len(dates)):
                progress(position, len(dates), date)

        frames = [pd.read_parquet(path) for path in sorted(staging.glob("trade_date=*.parquet"))]
        moneyflow = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if moneyflow.duplicated(["trade_date", "ts_code"]).any():
            raise DataQualityError("Moneyflow staging contains duplicate trade_date + ts_code keys")
        enriched, audit = self._merge_and_audit(base, moneyflow, min_coverage=min_coverage)

        published = None
        if publish:
            published = self.target_repository.publish(
                enriched,
                {"moneyflow": moneyflow, "stock_basic": securities},
                source=f"tushare_moneyflow_enriched_from_{resolved}",
                version_kind="complete",
            )
            shutil.rmtree(staging)
        return MoneyflowEnrichmentResult(
            base_data_version=resolved,
            start_date=str(pd.Timestamp(enriched["trade_date"].min()).date()),
            end_date=str(pd.Timestamp(enriched["trade_date"].max()).date()),
            mainboard_rows=len(enriched),
            moneyflow_rows=len(moneyflow),
            coverage=audit["coverage"],
            amount_bound_violation_ratio=audit["amount_bound_violation_ratio"],
            published_data_version=published,
            staging_dir=str(staging),
        )

    def _mainboard_security_master(self) -> pd.DataFrame:
        frames = [
            self.provider.query(
                "stock_basic", exchange="", list_status=status,
                fields="ts_code,symbol,name,market,exchange,list_status,list_date,delist_date",
            )
            for status in ["L", "D", "P"]
        ]
        securities = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
        return securities.loc[
            securities["exchange"].isin(self.target_project.data.exchanges)
            & securities["market"].eq("主板")
        ].copy()

    def _merge_and_audit(
        self, base: pd.DataFrame, moneyflow: pd.DataFrame, *, min_coverage: float
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        flow = moneyflow.copy()
        flow["trade_date"] = pd.to_datetime(flow["trade_date"], errors="coerce")
        source_columns = [
            "net_mf_amount", "buy_sm_amount", "sell_sm_amount", "buy_lg_amount",
            "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
        ]
        for column in source_columns:
            flow[f"{column}_cny"] = pd.to_numeric(flow[column], errors="coerce") * 10_000.0
        cny_columns = [f"{column}_cny" for column in source_columns]
        enriched = base.drop(columns=cny_columns, errors="ignore").merge(
            flow[["trade_date", "ts_code", *cny_columns]],
            on=["trade_date", "ts_code"], how="left", validate="one_to_one",
        )
        eligible_amount = pd.to_numeric(enriched["amount_cny"], errors="coerce").gt(0)
        covered = enriched["net_mf_amount_cny"].notna() & eligible_amount
        denominator = int(eligible_amount.sum())
        coverage = float(covered.sum() / denominator) if denominator else 0.0
        if coverage < min_coverage:
            raise DataQualityError(
                f"Moneyflow coverage {coverage:.4f} is below required {min_coverage:.4f}"
            )
        bound_violations = covered & (
            enriched["net_mf_amount_cny"].abs() > enriched["amount_cny"] * 1.001
        )
        bound_ratio = float(bound_violations.sum() / covered.sum()) if covered.any() else 0.0
        if bound_ratio > 0.001:
            raise DataQualityError(
                f"Moneyflow/amount unit audit failed: violation_ratio={bound_ratio:.6f}"
            )
        age_ok = enriched["listing_trade_days"].ge(self.target_project.data.listing_age_days)
        enriched["is_factor_eligible"] = enriched["is_factor_eligible"].fillna(False) & age_ok
        enriched["is_tradeable"] = enriched["is_tradeable"].fillna(False) & age_ok
        enriched["is_liquid"] = enriched["is_liquid"].fillna(False) & age_ok
        return enriched, {
            "coverage": coverage,
            "amount_bound_violation_ratio": bound_ratio,
        }
