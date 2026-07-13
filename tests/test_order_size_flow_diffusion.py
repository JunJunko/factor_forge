from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.combinations import FactorCombinationEngine
from factor_forge.config import load_factor, load_factor_combination
from factor_forge.factors.engine import FactorEngine


ROOT = Path(__file__).resolve().parents[1]


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=130)
    codes = [f"{600000 + item:06d}.SH" for item in range(120)]
    index = pd.MultiIndex.from_product([dates, codes], names=["trade_date", "ts_code"])
    frame = index.to_frame(index=False)
    day = frame["trade_date"].map({value: idx for idx, value in enumerate(dates)}).astype(float)
    stock = frame["ts_code"].str[:6].astype(int).sub(600000).astype(float)
    close = 10.0 + day * 0.01 + stock * 0.001 + np.sin((day + stock) / 9.0) * 0.05
    frame["adj_close"] = close
    frame["adj_high"] = close + 0.1
    frame["adj_low"] = close - 0.1
    frame["amount_cny"] = 50_000_000.0 + stock * 100_000.0 + (day % 7) * 1_000_000.0
    frame["net_mf_amount_cny"] = frame["amount_cny"] * np.sin(day / 7.0 + stock / 11.0) * 0.02
    frame["buy_sm_amount_cny"] = frame["amount_cny"] * (0.20 + 0.01 * np.sin(day / 5.0))
    frame["sell_sm_amount_cny"] = frame["amount_cny"] * (0.19 + 0.01 * np.cos(stock / 7.0))
    frame["buy_lg_amount_cny"] = frame["amount_cny"] * (0.18 + 0.01 * np.sin((day + stock) / 8.0))
    frame["sell_lg_amount_cny"] = frame["amount_cny"] * (0.17 + 0.01 * np.cos(day / 9.0))
    frame["buy_elg_amount_cny"] = frame["amount_cny"] * (0.08 + 0.005 * np.sin(day / 6.0))
    frame["sell_elg_amount_cny"] = frame["amount_cny"] * (0.07 + 0.005 * np.cos(stock / 6.0))
    frame["turnover_rate"] = 1.0 + 0.2 * np.cos(day / 7.0 + stock / 11.0)
    frame["log_total_mv"] = np.log(1_000_000_000.0 + stock * 10_000_000.0)
    frame["industry_l1_code"] = (stock.astype(int) % 12).map(lambda value: f"I{value:02d}")
    frame["is_factor_eligible"] = True
    frame["is_tradeable"] = True
    frame["is_liquid"] = True
    return frame


def test_order_size_factors_and_combinations_compute(tmp_path):
    panel = _panel()
    factors = sorted((ROOT / "configs/factors").glob("*order*_v1.yaml"))
    factors += [ROOT / "configs/factors/large_order_lead_3d_v1.yaml"]
    engine = FactorEngine()
    for path in factors:
        result = engine.compute(panel, load_factor(path))
        latest = result.loc[result["trade_date"].eq(result["trade_date"].max()), "factor_value"]
        assert latest.notna().sum() == 120, path.name

    for path in sorted((ROOT / "configs/combinations").glob("order_size_*.yaml")):
        load_factor_combination(path)
        result = FactorCombinationEngine(tmp_path / path.stem).run(panel, path)
        latest = result.factor_values.loc[
            result.factor_values["trade_date"].eq(result.factor_values["trade_date"].max()),
            "factor_value",
        ]
        assert latest.notna().sum() == 120, path.name
