from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.config import ProjectConfig


def _dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


class DailyPanelBuilder:
    """Convert source-shaped frames into the versioned standard daily panel."""

    def __init__(self, project: ProjectConfig):
        self.project = project

    def build(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        daily = _dates(datasets["daily"], ["trade_date"])
        daily = daily.rename(columns={
            "open": "raw_open", "high": "raw_high", "low": "raw_low",
            "close": "raw_close", "vol": "source_volume_hands",
            "amount": "source_amount_thousand_cny", "pct_chg": "pct_change",
        })
        suspensions = _dates(datasets.get("suspend", pd.DataFrame()), ["trade_date"])
        if not suspensions.empty:
            keys = suspensions[["trade_date", "ts_code"]].drop_duplicates()
            base = pd.concat([daily, keys], ignore_index=True).drop_duplicates(
                ["trade_date", "ts_code"], keep="first"
            )
        else:
            base = daily
        panel = base.merge(
            _dates(datasets["adj_factor"], ["trade_date"])[["trade_date", "ts_code", "adj_factor"]],
            on=["trade_date", "ts_code"], how="left",
        )
        basic = _dates(datasets.get("daily_basic", pd.DataFrame()), ["trade_date"])
        if not basic.empty:
            keep = [c for c in ["trade_date", "ts_code", "total_mv", "circ_mv", "turnover_rate"] if c in basic]
            panel = panel.merge(basic[keep], on=["trade_date", "ts_code"], how="left")
        limits = _dates(datasets.get("stk_limit", pd.DataFrame()), ["trade_date"])
        if not limits.empty:
            limits = limits.rename(columns={"up_limit": "limit_up_price", "down_limit": "limit_down_price"})
            panel = panel.merge(
                limits[["trade_date", "ts_code", "limit_up_price", "limit_down_price"]],
                on=["trade_date", "ts_code"], how="left",
            )
        panel["is_suspended"] = panel["raw_open"].isna()
        panel["volume_shares"] = panel.get("source_volume_hands", np.nan) * 100.0
        panel["amount_cny"] = panel.get("source_amount_thousand_cny", np.nan) * 1000.0
        panel["total_mv_cny"] = panel.get("total_mv", np.nan) * 10_000.0
        panel["circ_mv_cny"] = panel.get("circ_mv", np.nan) * 10_000.0
        panel["log_total_mv"] = np.log(panel["total_mv_cny"].where(panel["total_mv_cny"] > 0))
        panel["log_circ_mv"] = np.log(panel["circ_mv_cny"].where(panel["circ_mv_cny"] > 0))
        for price in ["open", "high", "low", "close"]:
            panel[f"adj_{price}"] = panel[f"raw_{price}"] * panel["adj_factor"]
        panel = self._attach_security(panel, datasets["stock_basic"])
        panel = self._attach_industry(panel, datasets.get("industry_membership", pd.DataFrame()))
        panel = self._attach_st(
            panel, datasets.get("st_status", pd.DataFrame()),
            datasets.get("st_status_coverage", pd.DataFrame()),
        )
        if "limit_up_price" not in panel:
            panel["limit_up_price"] = np.nan
        if "limit_down_price" not in panel:
            panel["limit_down_price"] = np.nan
        panel["is_limit_up_open"] = (
            panel["raw_open"].notna() & panel["limit_up_price"].notna()
            & (panel["raw_open"] >= panel["limit_up_price"] - 0.001)
        )
        panel["is_limit_down_open"] = (
            panel["raw_open"].notna() & panel["limit_down_price"].notna()
            & (panel["raw_open"] <= panel["limit_down_price"] + 0.001)
        )
        panel = panel.sort_values(["ts_code", "trade_date"])
        panel["listing_trade_days"] = panel.groupby("ts_code").cumcount() + 1
        panel["is_factor_eligible"] = (
            (panel["listing_trade_days"] >= self.project.data.listing_age_days)
            & panel["adj_close"].notna() & ~panel["is_suspended"]
        )
        panel["is_tradeable"] = (
            panel["is_factor_eligible"] & ~panel["is_st"] & ~panel["is_delisting_period"]
        )
        liquidity = self.project.data.liquidity
        rolling = panel.groupby("ts_code", sort=False)["amount_cny"].rolling(
            liquidity.window, min_periods=liquidity.window
        )
        avg_amount = rolling.mean().droplevel(0).reindex(panel.index)
        traded_days = rolling.count().droplevel(0).reindex(panel.index)
        panel["is_liquid"] = (
            panel["is_tradeable"] & (avg_amount >= liquidity.min_avg_amount_cny)
            & (traded_days >= liquidity.min_traded_days)
        )
        industry_level = self.project.data.industry_level.lower()
        industry_columns = [f"industry_{industry_level}_code", f"industry_{industry_level}_name"]
        ordered = [
            "trade_date", "ts_code", "raw_open", "raw_high", "raw_low", "raw_close", "pre_close",
            "adj_factor", "adj_open", "adj_high", "adj_low", "adj_close", "volume_shares", "amount_cny",
            "pct_change", "total_mv_cny", "circ_mv_cny", "log_total_mv", "log_circ_mv", "turnover_rate",
            *industry_columns, "limit_up_price", "limit_down_price", "is_suspended",
            "is_limit_up_open", "is_limit_down_open", "is_st", "is_delisting_period", "listing_trade_days",
            "is_factor_eligible", "is_tradeable", "is_liquid", "st_status_known",
        ]
        for column in ordered:
            if column not in panel:
                panel[column] = np.nan
        return panel.sort_values(["trade_date", "ts_code"])[ordered].reset_index(drop=True)

    @staticmethod
    def _attach_security(panel: pd.DataFrame, securities: pd.DataFrame) -> pd.DataFrame:
        # A final delist_date is not a point-in-time signal. Delisting status is
        # attached from the daily risk-status snapshot in _attach_st instead.
        panel["is_delisting_period"] = False
        return panel

    def _attach_industry(self, panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
        level = self.project.data.industry_level.lower()
        output_code = f"industry_{level}_code"
        output_name = f"industry_{level}_name"
        panel[output_code] = None
        panel[output_name] = None
        if membership.empty:
            return panel
        members = _dates(membership, ["in_date", "out_date"])
        code_col = "industry_code" if "industry_code" in members else f"{level}_code"
        name_col = "industry_name" if "industry_name" in members else f"{level}_name"
        for ts_code, indexes in panel.groupby("ts_code").groups.items():
            intervals = members[members["ts_code"] == ts_code].sort_values("in_date")
            if intervals.empty:
                continue
            dates = panel.loc[indexes, "trade_date"]
            for _, row in intervals.iterrows():
                active = dates.ge(row["in_date"]) & (row["out_date"] is pd.NaT or pd.isna(row["out_date"]) or dates.lt(row["out_date"]))
                active_indexes = dates.index[active]
                panel.loc[active_indexes, output_code] = row[code_col]
                panel.loc[active_indexes, output_name] = row.get(name_col)
        return panel

    @staticmethod
    def _attach_st(panel: pd.DataFrame, status: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
        panel["is_st"] = False
        checked_dates = set(pd.to_datetime(coverage.get("trade_date", pd.Series(dtype=str)), errors="coerce").dropna())
        if status.empty:
            panel["st_status_known"] = panel["trade_date"].isin(checked_dates)
            return panel
        st = _dates(status, ["trade_date"])
        text = st.get("name", pd.Series("", index=st.index)).fillna("").astype(str)
        if "type" in st:
            text = text + st["type"].fillna("").astype(str)
        st["is_st"] = True
        st["is_delisting_period"] = text.str.contains("退市|退市整理", regex=True)
        keys = st[["trade_date", "ts_code", "is_st", "is_delisting_period"]].drop_duplicates(
            ["trade_date", "ts_code"]
        ).set_index(["trade_date", "ts_code"])
        panel_keys = pd.MultiIndex.from_frame(panel[["trade_date", "ts_code"]])
        panel["is_st"] = panel_keys.isin(keys.index)
        delisting_keys = keys.index[keys["is_delisting_period"].eq(True)]
        panel["is_delisting_period"] = panel_keys.isin(delisting_keys)
        panel["st_status_known"] = panel["trade_date"].isin(checked_dates) if checked_dates else True
        return panel
