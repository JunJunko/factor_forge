from __future__ import annotations

import numpy as np
import pandas as pd


class IndustryFeatureBuilder:
    def build(self, stocks: pd.DataFrame, *, short_ema=3, long_ema=10,
              breadth_change_window=5, minimum_industry_members=8) -> pd.DataFrame:
        frame = stocks.sort_values(["ts_code", "trade_date"]).copy()
        g = frame.groupby("ts_code", sort=False)
        frame["stock_return_1d"] = g["adj_close"].pct_change(fill_method=None)
        frame["ma20"] = g["adj_close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
        frame["amount_ma5"] = g["amount_cny"].transform(lambda x: x.rolling(5, min_periods=5).mean())
        frame["amount_ma20"] = g["amount_cny"].transform(lambda x: x.rolling(20, min_periods=20).mean())
        valid = frame.get("is_tradeable", True)
        valid = pd.Series(valid, index=frame.index).fillna(False).astype(bool)
        base = frame.loc[valid & frame["industry_code"].notna()].copy()
        base["market_return_1d_equal_weight"] = base.groupby("trade_date")["stock_return_1d"].transform("mean")
        base["positive"] = base["stock_return_1d"] > 0
        base["outperform"] = base["stock_return_1d"] > base["market_return_1d_equal_weight"]
        base["above_ma20"] = base["adj_close"] > base["ma20"]
        base["volume_improving"] = base["amount_ma5"] > base["amount_ma20"]
        keys = ["trade_date", "industry_code"]
        industry = base.groupby(keys, as_index=False).agg(
            industry_name=("industry_name", "first"), member_count=("ts_code", "nunique"),
            industry_return_1d_equal_weight=("stock_return_1d", "mean"),
            industry_return_1d_median=("stock_return_1d", "median"),
            market_return_1d_equal_weight=("market_return_1d_equal_weight", "first"),
            positive_return_ratio=("positive", "mean"), outperform_market_ratio=("outperform", "mean"),
            above_ma20_ratio=("above_ma20", "mean"), volume_improving_ratio=("volume_improving", "mean"),
            industry_amount=("amount_cny", "sum"),
        )
        industry["eligible_industry"] = industry["member_count"] >= minimum_industry_members
        industry["total_market_amount"] = industry.groupby("trade_date")["industry_amount"].transform("sum")
        industry["industry_amount_share"] = industry["industry_amount"] / industry["total_market_amount"]
        industry["log_industry_amount_share"] = np.log(industry["industry_amount_share"].where(industry["industry_amount_share"] > 0))
        industry["industry_excess_return_1d"] = industry["industry_return_1d_equal_weight"] - industry["market_return_1d_equal_weight"]
        industry = industry.sort_values(["industry_code", "trade_date"])
        ig = industry.groupby("industry_code", sort=False)
        for window in (5, 20):
            industry[f"industry_excess_return_{window}d"] = ig["industry_excess_return_1d"].transform(
                lambda x, w=window: (1 + x).rolling(w, min_periods=w).apply(np.prod, raw=True) - 1
            )
        industry["relative_strength_level"] = industry.groupby("trade_date")["industry_excess_return_20d"].rank(pct=True)
        short = ig["industry_excess_return_1d"].transform(lambda x: x.ewm(span=short_ema, adjust=False, min_periods=short_ema).mean())
        long = ig["industry_excess_return_1d"].transform(lambda x: x.ewm(span=long_ema, adjust=False, min_periods=long_ema).mean())
        industry["relative_strength_velocity"] = short - long
        industry["relative_strength_acceleration"] = industry["relative_strength_velocity"] - ig["relative_strength_velocity"].shift(3)
        industry["breadth_level"] = (.30 * industry["positive_return_ratio"] + .35 * industry["outperform_market_ratio"] +
                                     .20 * industry["above_ma20_ratio"] + .15 * industry["volume_improving_ratio"])
        industry["breadth_velocity"] = industry["breadth_level"] - ig["breadth_level"].shift(breadth_change_window)
        industry["industry_amount_share_change_5d"] = industry["industry_amount_share"] - ig["industry_amount_share"].shift(5)
        return industry.sort_values(["trade_date", "industry_code"]).reset_index(drop=True)
