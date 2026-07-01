from __future__ import annotations

import pandas as pd


class IndustryContextBuilder:
    """Validate and expose signal-date PIT SW L1 membership from the stock panel."""

    def build(self, panel: pd.DataFrame) -> pd.DataFrame:
        required = {"trade_date", "ts_code", "industry_l1_code", "industry_l1_name"}
        missing = sorted(required - set(panel.columns))
        if missing:
            raise ValueError(f"PIT industry membership missing columns: {missing}")
        result = panel.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        duplicate = result.dropna(subset=["industry_l1_code"]).duplicated(["trade_date", "ts_code"])
        if duplicate.any():
            raise ValueError("duplicate_stock_industry_mapping")
        coverage = float(result["industry_l1_code"].notna().mean()) if len(result) else 0.0
        if coverage == 0:
            raise ValueError("PIT industry membership missing for all stock-days")
        return result.rename(columns={
            "industry_l1_code": "industry_code", "industry_l1_name": "industry_name"
        })
