import numpy as np
import pandas as pd

from factor_forge.timing import (
    TimingFeatureConfig,
    TimingInputData,
    build_option_atm_iv,
    build_timing_dataset,
)
from factor_forge.timing.transforms import black_scholes_price


def _dates(n=90):
    return pd.bdate_range("2024-01-02", periods=n)


def _index_daily(dates):
    close = 3000 * np.exp(np.linspace(0, 0.08, len(dates)))
    return pd.DataFrame({
        "trade_date": dates,
        "ts_code": "000300.SH",
        "close": close,
    })


def _stock_daily(dates):
    rows = []
    for i in range(8):
        price = 10 + i
        for pos, date in enumerate(dates):
            ret = 0.01 * np.sin(pos / 5 + i)
            price *= 1 + ret
            rows.append({
                "trade_date": date,
                "ts_code": f"{i:06d}.SZ",
                "close": price,
                "pre_close": price / (1 + ret),
                "pct_chg": ret,
                "amount": 1_000_000 + i * 1000,
            })
    return pd.DataFrame(rows)


def test_timing_dataset_builds_lagged_normalized_features_and_20d_label():
    dates = _dates()
    index_daily = _index_daily(dates)
    stock_daily = _stock_daily(dates)
    dailybasic = pd.DataFrame({
        "trade_date": dates,
        "ts_code": "000300.SH",
        "pe_ttm": np.linspace(14, 11, len(dates)),
    })
    bond = pd.DataFrame({
        "trade_date": dates,
        "curve_term": 10,
        "yield": np.linspace(2.6, 2.8, len(dates)),
    })
    margin = pd.DataFrame({
        "trade_date": dates,
        "rzmre": np.linspace(10_000_000, 20_000_000, len(dates)),
        "rzye": np.linspace(200_000_000, 230_000_000, len(dates)),
    })
    pmi = pd.DataFrame({
        "available_date": [dates[5], dates[25], dates[45]],
        "pmi": [49.8, 50.3, 50.8],
    })
    inputs = TimingInputData(
        index_daily=index_daily,
        stock_daily=stock_daily,
        index_dailybasic=dailybasic,
        bond_yield=bond,
        margin=margin,
        pmi=pmi,
    )
    cfg = TimingFeatureConfig(data_lag=1, z_window=20, pct_window=40, horizon=20, horizons=(5, 10, 20))
    result = build_timing_dataset(inputs, cfg)

    assert "erp" in result.feature_names
    assert "erp_z_20" in result.feature_names
    assert "up_ratio_pct_40" in result.feature_names
    assert "rzmre_ratio_high_90" in result.feature_names
    assert result.label_name == "label_20d_excess_return"
    assert {"label_5d_excess_return", "label_10d_excess_return", "label_20d_excess_return"} <= set(result.dataset.columns)
    first_label = result.dataset.loc[0, result.label_name]
    expected = index_daily.loc[20, "close"] / index_daily.loc[0, "close"] - 1
    assert np.isclose(first_label, expected)
    expected_5d = index_daily.loc[5, "close"] / index_daily.loc[0, "close"] - 1
    assert np.isclose(result.dataset.loc[0, "label_5d_excess_return"], expected_5d)
    # The PMI release at dates[5] becomes usable after one daily lag and is then
    # forward-filled until the next release.
    assert pd.isna(result.dataset.loc[5, "pmi"])
    assert result.dataset.loc[6, "pmi"] == 49.8
    assert result.dataset.loc[24, "pmi"] == 49.8


def test_option_atm_iv_recovers_black_scholes_volatility():
    trade_date = pd.Timestamp("2024-01-02")
    maturity = pd.Timestamp("2024-03-01")
    price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        rate=0.02,
        time_to_expiry=(maturity - trade_date).days / 365,
        volatility=0.25,
        option_type="C",
    )
    option_basic = pd.DataFrame({
        "ts_code": ["OPT1"],
        "call_put": ["C"],
        "exercise_price": [100.0],
        "maturity_date": [maturity],
    })
    option_daily = pd.DataFrame({
        "trade_date": [trade_date],
        "ts_code": ["OPT1"],
        "close": [price],
        "amount": [1000.0],
    })
    spot = pd.DataFrame({"trade_date": [trade_date], "index_close": [100.0]})
    iv = build_option_atm_iv(option_basic, option_daily, spot, TimingFeatureConfig())

    assert np.isclose(iv.loc[0, "iv_atm"], 0.25, atol=1e-3)
