from __future__ import annotations

import pandas as pd

from factor_forge.config import ProjectConfig
from factor_forge.data.panel import DailyPanelBuilder


def test_columnless_empty_adj_factor_does_not_raise_key_error():
    daily = pd.DataFrame({
        "trade_date": ["20240102"], "ts_code": ["000001.SZ"],
        "open": [10.0], "high": [10.2], "low": [9.9], "close": [10.1],
        "pre_close": [9.8], "vol": [123.0], "amount": [456.0], "pct_chg": [3.0],
    })
    datasets = {
        "daily": daily,
        "adj_factor": pd.DataFrame(),
        "daily_basic": pd.DataFrame(), "stk_limit": pd.DataFrame(), "suspend": pd.DataFrame(),
        "stock_basic": pd.DataFrame({"ts_code": ["000001.SZ"]}),
        "industry_membership": pd.DataFrame(),
        "st_status": pd.DataFrame(), "st_status_coverage": pd.DataFrame({"trade_date": ["20240102"]}),
    }
    project = ProjectConfig.model_validate({"data": {"listing_age_days": 1,
        "liquidity": {"window": 1, "min_avg_amount_cny": 1, "min_traded_days": 1}}})

    panel = DailyPanelBuilder(project).build(datasets)

    assert pd.isna(panel.loc[0, "adj_factor"])
    assert pd.isna(panel.loc[0, "adj_close"])
