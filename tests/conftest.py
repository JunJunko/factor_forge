from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_panel(days: int = 8, stocks: int = 4) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for stock in range(stocks):
        code = f"{stock:06d}.SZ"
        for index, date in enumerate(dates):
            price = 10.0 + stock + index * (0.1 + stock * 0.01)
            rows.append({
                "trade_date": date, "ts_code": code,
                "raw_open": price, "raw_high": price * 1.01, "raw_low": price * 0.99,
                "raw_close": price, "pre_close": price - 0.1,
                "adj_factor": 1.0, "adj_open": price, "adj_high": price * 1.01,
                "adj_low": price * 0.99, "adj_close": price,
                "volume_shares": 1_000_000.0, "amount_cny": 50_000_000.0,
                "pct_change": 1.0, "total_mv_cny": 1e9 * (stock + 1),
                "circ_mv_cny": 8e8 * (stock + 1), "log_total_mv": np.log(1e9 * (stock + 1)),
                "log_circ_mv": np.log(8e8 * (stock + 1)), "turnover_rate": 1.0,
                "industry_l1_code": f"I{stock % 2}", "industry_l1_name": f"Industry {stock % 2}",
                "limit_up_price": price * 1.1, "limit_down_price": price * 0.9,
                "is_suspended": False, "is_limit_up_open": False, "is_limit_down_open": False,
                "is_st": False, "is_delisting_period": False, "listing_trade_days": 100 + index,
                "is_factor_eligible": True, "is_tradeable": True, "is_liquid": True,
                "st_status_known": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return make_panel()

