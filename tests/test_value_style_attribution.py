"""风格归因数学原语的手工可核对测试。

只测两个静态方法：因子收益构造（符号 + T+2 时序映射）与 OLS 回收（已知 beta）。
不测全链路，避免依赖数据版本。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_forge.ml.value_style_attribution import StyleAttributionRunner


def _factor_return_panel() -> pd.DataFrame:
    """9 只股票 × 5 个交易日。0-4 号为小盘（size 信号=1），5-8 号为大盘（信号=0）。

    小盘每日 +10%（adj_open: 10,10,11,12.1,13.31），大盘每日 -10%（10,10,9,8.1,7.29）。
    任意信号日 T 的前向收益（adj_open[T+2]/adj_open[T+1]-1）：小盘=+0.1，大盘=-0.1。
    因此 top tercile（小盘）- bottom tercile（大盘）= 0.2，映射到实现日 T+2。
    """
    dates = pd.bdate_range("2024-01-02", periods=5)
    rows = []
    small_path = [10.0, 10.0, 11.0, 12.1, 13.31]
    large_path = [10.0, 10.0, 9.0, 8.1, 7.29]
    for code in range(9):
        is_small = code < 5
        path = small_path if is_small else large_path
        signal = 1.0 if is_small else 0.0
        for di, date in enumerate(dates):
            rows.append({
                "trade_date": date, "ts_code": f"{code:06d}.SZ",
                "adj_open": path[di], "is_liquid": True, "size": signal,
            })
    return pd.DataFrame(rows)


def test_factor_return_sign_and_timing():
    panel = _factor_return_panel()
    result = StyleAttributionRunner._build_style_factor_returns(panel, ["size"], "is_liquid")

    dates = sorted(panel["trade_date"].unique())
    expected_realized = {dates[2], dates[3], dates[4]}  # 信号日 d0,d1,d2 → 实现 d2,d3,d4

    assert set(result["trade_date"]) == expected_realized, "收益必须映射到信号日 +2 个交易日"
    assert list(result["factor"].unique()) == ["size"]
    # 小盘（做多腿）跑赢大盘（做空腿）→ 正收益，符合 size 方向“做多小盘”
    assert (result["factor_return"] > 0).all()
    # 三条记录都应精确等于 0.2（每组内 fwd_ret 相同）
    np.testing.assert_allclose(result["factor_return"].to_numpy(), 0.2, atol=1e-9)


def test_factor_return_universe_filter():
    """非 universe 成员必须被排除。"""
    panel = _factor_return_panel()
    panel.loc[panel["ts_code"] == "000000.SZ", "is_liquid"] = False  # 剔除一只小盘
    result = StyleAttributionRunner._build_style_factor_returns(panel, ["size"], "is_liquid")
    # 仍应为正（剩余 2 只小盘 vs 3 只大盘）
    assert (result["factor_return"] > 0).all()


def test_regression_recovers_known_beta():
    """构造 excess = 0.5*factor + noise，OLS 应回收 beta≈0.5、alpha≈0。"""
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.bdate_range("2024-01-02", periods=n)
    factor = rng.normal(0, 0.01, size=n)
    excess = 0.5 * factor + rng.normal(0, 1e-6, size=n)  # 极小噪声，确保回收精确
    aligned = pd.DataFrame({"trade_date": dates, "excess_return": excess, "momentum": factor})

    reg = StyleAttributionRunner._fit_regression(aligned, ["momentum"], nw_lag=5, blocks=200, seed=1)

    assert reg["betas"]["momentum"] == pytest.approx(0.5, abs=1e-4)
    assert reg["alpha_annual"] == pytest.approx(0.0, abs=1e-3)
    assert reg["r_squared"] > 0.99
    # bootstrap CI 应紧贴 0（alpha 真值为 0）
    assert reg["alpha_bootstrap_ci_low"] < 0 < reg["alpha_bootstrap_ci_high"] or abs(reg["alpha_annual"]) < 1e-3


def test_verdict_classification_real_alpha():
    regs = {"full_sample": {
        "n_obs": 600, "r_squared": 0.3,
        "alpha_annual": 0.08, "alpha_t_nw": 3.0, "alpha_p_nw": 0.001,
        "alpha_bootstrap_ci_low": 0.04, "alpha_bootstrap_ci_high": 0.12,
        "betas": {"size": 0.1}, "tvalues": {"size": 1.0}, "pvalues": {"size": 0.3},
        "alpha_daily": 0.08 / 244,
    }, "year_2024": {"alpha_annual": 0.07}, "year_2025": {"alpha_annual": 0.09}, "year_2026": {"alpha_annual": 0.06}}
    verdict = StyleAttributionRunner._classify_verdict(regs, pd.DataFrame(), ["size"])
    assert verdict["class"] == "REAL_ALPHA"


def test_verdict_classification_style_beta_disguised():
    regs = {"full_sample": {
        "n_obs": 600, "r_squared": 0.6,
        "alpha_annual": 0.001, "alpha_t_nw": 0.2, "alpha_p_nw": 0.84,
        "alpha_bootstrap_ci_low": -0.02, "alpha_bootstrap_ci_high": 0.02,
        "betas": {"size": 0.5}, "tvalues": {"size": 4.0}, "pvalues": {"size": 0.001},
        "alpha_daily": 0.001 / 244,
    }}
    verdict = StyleAttributionRunner._classify_verdict(regs, pd.DataFrame(), ["size"])
    assert verdict["class"] == "STYLE_BETA_DISGUISED"
