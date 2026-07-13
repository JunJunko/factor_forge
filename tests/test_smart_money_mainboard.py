from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.combinations import FactorCombinationEngine
from factor_forge.config import load_experiment, load_factor, load_factor_combination, load_project
from factor_forge.factors.engine import FactorEngine
from factor_forge.data.moneyflow_enrichment import MainboardMoneyflowEnricher


ROOT = Path(__file__).resolve().parents[1]


def _smart_money_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=130)
    codes = [f"{600000 + item:06d}.SH" for item in range(120)]
    index = pd.MultiIndex.from_product([dates, codes], names=["trade_date", "ts_code"])
    frame = index.to_frame(index=False)
    day = frame["trade_date"].map({value: idx for idx, value in enumerate(dates)}).astype(float)
    stock = frame["ts_code"].str[:6].astype(int).sub(600000).astype(float)
    close = 10.0 + day * 0.01 + stock * 0.001 + np.sin((day + stock) / 9.0) * 0.05
    frame["adj_close"] = close
    frame["adj_high"] = close + 0.10 + (stock % 3) * 0.001
    frame["adj_low"] = close - 0.10 - (stock % 5) * 0.001
    frame["amount_cny"] = 50_000_000.0 + stock * 100_000.0 + (day % 7) * 1_000_000.0
    frame["net_mf_amount_cny"] = frame["amount_cny"] * (
        0.02 * np.sin(day / 5.0 + stock / 13.0) + 0.0001 * stock
    )
    frame["turnover_rate"] = 1.0 + 0.2 * np.cos(day / 7.0 + stock / 11.0)
    frame["log_total_mv"] = np.log(1_000_000_000.0 + stock * 10_000_000.0)
    frame["industry_l1_code"] = (stock.astype(int) % 12).map(lambda value: f"I{value:02d}")
    frame["is_factor_eligible"] = True
    frame["is_tradeable"] = True
    frame["is_liquid"] = True
    return frame


def test_mainboard_project_and_all_smart_money_configs_validate_and_compute(tmp_path):
    project = load_project(ROOT / "configs/project_mainboard_moneyflow.yaml")
    assert project.data.boards == ["main"]
    assert project.data.include_moneyflow is True

    factor_paths = sorted((ROOT / "configs/factors").glob("smart_flow_*.yaml")) + [
        ROOT / "configs/factors/price_acceptance_industry_1d_v1.yaml",
        ROOT / "configs/factors/close_location_1d_v1.yaml",
        ROOT / "configs/factors/turnover_abnormal_120d_v1.yaml",
        ROOT / "configs/factors/prior_industry_relative_return_20d_v1.yaml",
        ROOT / "configs/factors/volatility_20d_v1.yaml",
    ]
    panel = _smart_money_panel()
    engine = FactorEngine()
    for path in factor_paths:
        spec = load_factor(path)
        result = engine.compute(panel, spec)
        latest = result.loc[result["trade_date"].eq(result["trade_date"].max()), "factor_value"]
        assert latest.notna().sum() == 120, path.name

    for path in sorted((ROOT / "configs/combinations").glob("smart_money_*.yaml")):
        load_factor_combination(path)
        result = FactorCombinationEngine(tmp_path / path.stem).run(panel, path)
        latest = result.factor_values.loc[
            result.factor_values["trade_date"].eq(result.factor_values["trade_date"].max()),
            "factor_value",
        ]
        assert latest.notna().sum() == 120, path.name

    for path in sorted((ROOT / "configs/experiments").glob("smart_money_*.yaml")):
        spec = load_experiment(path)
        assert spec.project_config == Path("configs/project_mainboard_moneyflow.yaml")
        assert spec.stage_l1.forward_horizons == [5]


def test_moneyflow_enrichment_converts_units_checks_coverage_and_applies_listing_age():
    enricher = MainboardMoneyflowEnricher.__new__(MainboardMoneyflowEnricher)
    enricher.target_project = load_project(ROOT / "configs/project_mainboard_moneyflow.yaml")
    base = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "ts_code": ["600000.SH", "000001.SZ"],
        "amount_cny": [1_000_000.0, 2_000_000.0],
        "listing_trade_days": [119, 120],
        "is_factor_eligible": [True, True],
        "is_tradeable": [True, True],
        "is_liquid": [True, True],
    })
    moneyflow = pd.DataFrame({
        "trade_date": ["20240102", "20240102"],
        "ts_code": ["600000.SH", "000001.SZ"],
        "net_mf_amount": [10.0, -20.0],
        "buy_sm_amount": [30.0, 40.0], "sell_sm_amount": [20.0, 50.0],
        "buy_lg_amount": [50.0, 20.0], "sell_lg_amount": [10.0, 30.0],
        "buy_elg_amount": [15.0, 10.0], "sell_elg_amount": [5.0, 20.0],
    })

    enriched, audit = enricher._merge_and_audit(base, moneyflow, min_coverage=1.0)

    assert enriched["net_mf_amount_cny"].tolist() == [100_000.0, -200_000.0]
    assert enriched["buy_sm_amount_cny"].tolist() == [300_000.0, 400_000.0]
    assert enriched["sell_elg_amount_cny"].tolist() == [50_000.0, 200_000.0]
    assert audit == {"coverage": 1.0, "amount_bound_violation_ratio": 0.0}
    assert not enriched.loc[0, "is_factor_eligible"]
    assert enriched.loc[1, "is_factor_eligible"]
