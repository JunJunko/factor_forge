from __future__ import annotations

import numpy as np
import pandas as pd


class IndustryResidualReturnBuilder:
    """Build open_t1-to-open_t(h+1) stock and signal-date PIT leave-one-out targets."""

    def build(self, stocks: pd.DataFrame, horizons: list[int]) -> dict[str, dict[int, pd.Series]]:
        frame = stocks.sort_values(["ts_code", "trade_date"]).copy()
        g = frame.groupby("ts_code", sort=False)["adj_open"]
        targets = {"stock_return": {}, "stock_minus_sw_l1_return": {}}
        for horizon in horizons:
            stock_return = g.shift(-(horizon + 1)) / g.shift(-1) - 1
            valid = stock_return.notna() & frame["sw_l1_industry_code"].notna()
            key = [frame["trade_date"], frame["sw_l1_industry_code"]]
            total = stock_return.where(valid).groupby(key).transform("sum")
            count = stock_return.where(valid).groupby(key).transform("count")
            benchmark = (total - stock_return) / (count - 1).replace(0, np.nan)
            targets["stock_return"][horizon] = stock_return.reindex(stocks.index)
            targets["stock_minus_sw_l1_return"][horizon] = (stock_return - benchmark).reindex(stocks.index)
        return targets
