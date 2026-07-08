from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .transforms import (
    add_changes,
    add_extreme_flags,
    add_rolling_normalizations,
    clean_numeric,
    first_existing,
    implied_volatility_bisection,
    lag_non_date_columns,
    rolling_mad_zscore,
    rolling_percentile,
    to_datetime_series,
)


@dataclass(frozen=True)
class TimingFeatureConfig:
    index_code: str = "000300.SH"
    benchmark_code: str | None = None
    future_prefix: str | None = None
    fallback_bond_10y_yield: float | None = None
    horizon: int = 20
    horizons: tuple[int, ...] | None = None
    data_lag: int = 1
    z_window: int = 252
    pct_window: int = 756
    change_windows: tuple[int, ...] = (5, 20, 60)
    boll_windows: tuple[int, ...] = (20, 60)
    annualization_days: int = 365
    basis_roll_days: int = 5
    option_min_days_to_expiry: int = 7
    option_atm_moneyness: float = 0.05
    option_min_amount: float = 0.0
    option_price_field: str = "close"
    option_risk_free_rate: float = 0.02
    iv_bounds: tuple[float, float] = (0.01, 1.50)
    z_clip: float = 3.0
    pct_clip: tuple[float, float] = (0.01, 0.99)
    extreme_low: tuple[float, ...] = (0.05, 0.10)
    extreme_high: tuple[float, ...] = (0.90, 0.95)


@dataclass
class TimingInputData:
    index_daily: pd.DataFrame
    stock_daily: pd.DataFrame | None = None
    index_dailybasic: pd.DataFrame | None = None
    bond_yield: pd.DataFrame | None = None
    margin: pd.DataFrame | None = None
    option_basic: pd.DataFrame | None = None
    option_daily: pd.DataFrame | None = None
    option_iv_daily: pd.DataFrame | None = None
    futures_basic: pd.DataFrame | None = None
    futures_daily: pd.DataFrame | None = None
    futures_holding: pd.DataFrame | None = None
    moneyflow: pd.DataFrame | None = None
    cpi: pd.DataFrame | None = None
    pmi: pd.DataFrame | None = None
    epu: pd.DataFrame | None = None


@dataclass(frozen=True)
class TimingFeatureResult:
    dataset: pd.DataFrame
    feature_names: list[str]
    label_name: str
    feature_groups: dict[str, list[str]] = field(default_factory=dict)


def build_timing_dataset(
    inputs: TimingInputData,
    config: TimingFeatureConfig | None = None,
) -> TimingFeatureResult:
    """Build a daily timing feature table for ML models.

    The builder accepts Tushare-style raw tables, lags non-price feature blocks by
    ``data_lag`` sessions, and creates raw, normalized, change, extreme-state and
    selected interaction features. It deliberately does not prescribe factor
    directions.
    """
    cfg = config or TimingFeatureConfig()
    base = _base_index_frame(inputs.index_daily, cfg.index_code)
    dataset = base[["trade_date", "index_close"]].copy()
    feature_groups: dict[str, list[str]] = {}

    builders = [
        ("valuation", _valuation_features(inputs.index_dailybasic, inputs.bond_yield, cfg)),
        ("market_breadth", _market_breadth_features(inputs.stock_daily, inputs.index_daily, cfg)),
        ("leverage_funding", _margin_features(inputs.margin, inputs.stock_daily, cfg)),
        ("option_sentiment", _option_features(inputs, base, cfg)),
        ("futures_sentiment", _futures_features(inputs, base, cfg)),
        ("main_moneyflow", _moneyflow_features(inputs.moneyflow, inputs.stock_daily, cfg)),
        ("macro", _macro_features(inputs.cpi, inputs.pmi, inputs.epu, cfg)),
    ]
    for group, frame in builders:
        if frame.empty:
            feature_groups[group] = []
            continue
        if group == "macro":
            aligned = pd.merge_asof(
                dataset[["trade_date"]].sort_values("trade_date"),
                frame.sort_values("trade_date"),
                on="trade_date",
                direction="backward",
            )
            columns = [column for column in aligned.columns if column != "trade_date"]
        else:
            columns = [column for column in frame.columns if column != "trade_date"]
            aligned = dataset[["trade_date"]].merge(frame, on="trade_date", how="left")
        aligned = lag_non_date_columns(aligned, cfg.data_lag)
        dataset = dataset.merge(aligned, on="trade_date", how="left")
        feature_groups[group] = columns

    risk_features = _risk_state_features(dataset, cfg)
    feature_groups["risk_state"] = risk_features
    interaction_features = _interaction_features(dataset)
    feature_groups["interaction"] = interaction_features

    horizons = tuple(sorted(set(cfg.horizons or (cfg.horizon,))))
    label_names: list[str] = []
    for horizon in horizons:
        label_name = f"label_{horizon}d_excess_return"
        dataset[label_name] = _future_excess_return(base, inputs.index_daily, cfg, horizon=horizon)
        label_names.append(label_name)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).sort_values("trade_date").reset_index(drop=True)
    feature_names = [
        column
        for group in feature_groups.values()
        for column in group
        if column in dataset.columns
    ]
    feature_names = list(dict.fromkeys(feature_names))
    preferred_label = f"label_{cfg.horizon}d_excess_return"
    label_name = preferred_label if preferred_label in label_names else label_names[-1]
    return TimingFeatureResult(dataset=dataset, feature_names=feature_names, label_name=label_name, feature_groups=feature_groups)


def build_option_atm_iv(
    option_basic: pd.DataFrame,
    option_daily: pd.DataFrame,
    spot_daily: pd.DataFrame,
    config: TimingFeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute a simple near-month ATM option IV series from option quotes."""
    cfg = config or TimingFeatureConfig()
    if option_basic is None or option_basic.empty or option_daily is None or option_daily.empty:
        return pd.DataFrame(columns=["trade_date", "iv_atm"])
    basic = option_basic.copy()
    daily = option_daily.copy()
    spot = spot_daily[["trade_date", "index_close"]].copy()
    basic_code = first_existing(basic, ["ts_code", "opt_code", "code"])
    daily_code = first_existing(daily, ["ts_code", "opt_code", "code"])
    strike_col = first_existing(basic, ["exercise_price", "strike_price", "strike"])
    maturity_col = first_existing(basic, ["maturity_date", "expire_date", "delist_date", "last_ddate"])
    type_col = first_existing(basic, ["call_put", "option_type", "opt_type"])
    price_col = cfg.option_price_field if cfg.option_price_field in daily else first_existing(daily, ["settle", "close"])
    amount_col = first_existing(daily, ["amount", "amt"])
    if not all([basic_code, daily_code, strike_col, maturity_col, type_col, price_col]):
        return pd.DataFrame(columns=["trade_date", "iv_atm"])
    basic = basic.rename(columns={basic_code: "option_code"})
    daily = daily.rename(columns={daily_code: "option_code"})
    basic["maturity_date"] = to_datetime_series(basic[maturity_col])
    basic["strike"] = clean_numeric(basic[strike_col])
    basic["option_type"] = basic[type_col].astype("string")
    daily["trade_date"] = to_datetime_series(daily["trade_date"])
    daily["option_price"] = clean_numeric(daily[price_col])
    daily["option_amount"] = clean_numeric(daily[amount_col]) if amount_col else np.nan
    data = daily.merge(
        basic[["option_code", "maturity_date", "strike", "option_type"]],
        on="option_code",
        how="left",
    ).merge(spot, on="trade_date", how="left")
    data["days_to_expiry"] = (data["maturity_date"] - data["trade_date"]).dt.days
    data["moneyness"] = np.log(data["strike"] / data["index_close"])
    usable = (
        data["days_to_expiry"].gt(cfg.option_min_days_to_expiry)
        & data["moneyness"].abs().le(cfg.option_atm_moneyness)
        & data["option_price"].gt(0)
    )
    if amount_col:
        usable &= data["option_amount"].fillna(0).ge(cfg.option_min_amount)
    data = data.loc[usable].copy()
    if data.empty:
        return pd.DataFrame(columns=["trade_date", "iv_atm"])
    data["time_to_expiry"] = data["days_to_expiry"] / 365.0
    low, high = cfg.iv_bounds
    data["iv"] = [
        implied_volatility_bisection(
            price=row.option_price,
            spot=row.index_close,
            strike=row.strike,
            rate=cfg.option_risk_free_rate,
            time_to_expiry=row.time_to_expiry,
            option_type=str(row.option_type),
            low=low,
            high=high,
        )
        for row in data.itertuples(index=False)
    ]
    data = data.dropna(subset=["iv"])
    if data.empty:
        return pd.DataFrame(columns=["trade_date", "iv_atm"])
    nearest = data.groupby("trade_date")["days_to_expiry"].transform("min")
    data = data.loc[data["days_to_expiry"].eq(nearest)]
    return data.groupby("trade_date", as_index=False)["iv"].mean().rename(columns={"iv": "iv_atm"})


def _base_index_frame(index_daily: pd.DataFrame, index_code: str) -> pd.DataFrame:
    if index_daily is None or index_daily.empty:
        raise ValueError("index_daily is required")
    data = index_daily.copy()
    data["trade_date"] = to_datetime_series(data["trade_date"])
    if "ts_code" in data:
        selected = data["ts_code"].astype(str).eq(index_code)
        if selected.any():
            data = data.loc[selected].copy()
    close_col = first_existing(data, ["close", "adj_close", "index_close"])
    if close_col is None:
        raise ValueError("index_daily must contain close or adj_close")
    data["index_close"] = clean_numeric(data[close_col])
    return data[["trade_date", "index_close"]].dropna().sort_values("trade_date").reset_index(drop=True)


def _valuation_features(
    dailybasic: pd.DataFrame | None,
    bond_yield: pd.DataFrame | None,
    cfg: TimingFeatureConfig,
) -> pd.DataFrame:
    if dailybasic is None or dailybasic.empty:
        return pd.DataFrame()
    data = dailybasic.copy()
    data["trade_date"] = to_datetime_series(data["trade_date"])
    pe_col = first_existing(data, ["pe_ttm", "pe"])
    if pe_col is None:
        return pd.DataFrame()
    data["pe_ttm"] = clean_numeric(data[pe_col])
    result = data.groupby("trade_date", as_index=False)["pe_ttm"].last()
    result["earnings_yield"] = 1.0 / result["pe_ttm"].where(result["pe_ttm"] > 0)
    if bond_yield is not None and not bond_yield.empty:
        bond = bond_yield.copy()
        date_col = first_existing(bond, ["trade_date", "date", "cal_date"])
        if date_col:
            bond["trade_date"] = to_datetime_series(bond[date_col])
            if "curve_term" in bond:
                bond = bond.loc[clean_numeric(bond["curve_term"]).round(6).eq(10.0)]
            yield_col = first_existing(bond, ["yield", "yield_rate", "curve_value", "value"])
            if yield_col:
                bond["bond_10y_yield"] = clean_numeric(bond[yield_col]) / 100.0
                result = result.merge(
                    bond.groupby("trade_date", as_index=False)["bond_10y_yield"].last(),
                    on="trade_date",
                    how="left",
                )
    if "bond_10y_yield" not in result:
        result["bond_10y_yield"] = np.nan
    if cfg.fallback_bond_10y_yield is not None:
        result["bond_10y_yield"] = result["bond_10y_yield"].fillna(cfg.fallback_bond_10y_yield)
    result["erp"] = result["earnings_yield"] - result["bond_10y_yield"]
    features = ["pe_ttm", "earnings_yield", "bond_10y_yield", "erp"]
    return _decorate_daily(result[["trade_date", *features]], features, cfg, extremes=["erp", "pe_ttm"])


def _market_breadth_features(
    stock_daily: pd.DataFrame | None,
    index_daily: pd.DataFrame,
    cfg: TimingFeatureConfig,
) -> pd.DataFrame:
    base = _base_index_frame(index_daily, cfg.index_code)
    result = base.copy()
    result["index_log_ret"] = np.log(result["index_close"]).diff()
    result["index_ret_20d"] = result["index_close"].pct_change(20)
    result["index_ret_60d"] = result["index_close"].pct_change(60)
    result["index_vol_20d"] = result["index_log_ret"].rolling(20, min_periods=10).std()
    result["index_drawdown_60d"] = result["index_close"] / result["index_close"].rolling(60, min_periods=20).max() - 1
    for window in (20, 60):
        ma = result["index_close"].rolling(window, min_periods=max(5, window // 2)).mean()
        result[f"index_above_ma{window}"] = result["index_close"].gt(ma).astype(float).where(ma.notna())
    features = ["index_ret_20d", "index_ret_60d", "index_vol_20d", "index_drawdown_60d", "index_above_ma20", "index_above_ma60"]
    if stock_daily is not None and not stock_daily.empty:
        stock = stock_daily.copy()
        stock["trade_date"] = to_datetime_series(stock["trade_date"])
        pct_col = first_existing(stock, ["pct_chg", "pct_change"])
        if pct_col:
            stock["pct_chg"] = clean_numeric(stock[pct_col])
        else:
            close_col = first_existing(stock, ["close", "adj_close"])
            pre_col = first_existing(stock, ["pre_close"])
            if close_col and pre_col:
                stock["pct_chg"] = clean_numeric(stock[close_col]) / clean_numeric(stock[pre_col]) - 1
            else:
                stock["pct_chg"] = np.nan
        counts = stock.groupby("trade_date")["pct_chg"].agg(
            up_count=lambda item: item.gt(0).sum(),
            down_count=lambda item: item.lt(0).sum(),
        ).reset_index()
        counts["up_ratio"] = counts["up_count"] / (counts["up_count"] + counts["down_count"]).replace(0, np.nan)
        counts["adv_dec_log"] = np.log((counts["up_count"] + 0.5) / (counts["down_count"] + 0.5))
        counts["up_ratio_ma5"] = counts["up_ratio"].rolling(5, min_periods=3).mean()
        counts["up_ratio_ma20"] = counts["up_ratio"].rolling(20, min_periods=10).mean()
        counts["breadth_thrust"] = counts["up_ratio_ma5"] - counts["up_ratio_ma20"]
        result = result.merge(counts[["trade_date", "up_ratio", "adv_dec_log", "up_ratio_ma5", "up_ratio_ma20", "breadth_thrust"]], on="trade_date", how="left")
        features.extend(["up_ratio", "adv_dec_log", "up_ratio_ma5", "up_ratio_ma20", "breadth_thrust"])
    return _decorate_daily(result[["trade_date", *features]], features, cfg, extremes=["up_ratio", "adv_dec_log"])


def _margin_features(
    margin: pd.DataFrame | None,
    stock_daily: pd.DataFrame | None,
    cfg: TimingFeatureConfig,
) -> pd.DataFrame:
    if margin is None or margin.empty:
        return pd.DataFrame()
    data = margin.copy()
    data["trade_date"] = to_datetime_series(data["trade_date"])
    buy_col = first_existing(data, ["rzmre", "margin_buy", "fin_buy"])
    bal_col = first_existing(data, ["rzye", "margin_balance", "fin_balance"])
    if buy_col is None and bal_col is None:
        return pd.DataFrame()
    result = data.groupby("trade_date", as_index=False).agg(
        rzmre=(buy_col, "sum") if buy_col else ("trade_date", "size"),
        rzye=(bal_col, "sum") if bal_col else ("trade_date", "size"),
    )
    if buy_col is None:
        result["rzmre"] = np.nan
    if bal_col is None:
        result["rzye"] = np.nan
    if stock_daily is not None and not stock_daily.empty:
        turnover = _daily_market_amount(stock_daily)
        result = result.merge(turnover, on="trade_date", how="left")
        result["rzmre_ratio"] = clean_numeric(result["rzmre"]) / result["market_amount"].replace(0, np.nan)
    else:
        result["rzmre_ratio"] = clean_numeric(result["rzmre"])
    result["rzye_chg_5d"] = result["rzye"].pct_change(5)
    result["rzye_chg_20d"] = result["rzye"].pct_change(20)
    result["rzye_chg_60d"] = result["rzye"].pct_change(60)
    features = ["rzmre_ratio", "rzye_chg_5d", "rzye_chg_20d", "rzye_chg_60d"]
    for window in cfg.boll_windows:
        ma = result["rzmre_ratio"].rolling(window, min_periods=max(5, window // 2)).mean()
        std = result["rzmre_ratio"].rolling(window, min_periods=max(5, window // 2)).std()
        pos = f"rzmre_boll_pos_{window}"
        pct = f"rzmre_boll_pct_{window}"
        result[pos] = (result["rzmre_ratio"] - ma) / std.replace(0, np.nan)
        result[pct] = (result["rzmre_ratio"] - (ma - 2 * std)) / (4 * std).replace(0, np.nan)
        features.extend([pos, pct])
    return _decorate_daily(result[["trade_date", *features]], features, cfg, extremes=["rzmre_ratio"])


def _option_features(
    inputs: TimingInputData,
    base: pd.DataFrame,
    cfg: TimingFeatureConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if inputs.option_basic is not None and inputs.option_daily is not None and not inputs.option_daily.empty:
        frames.append(_put_call_features(inputs.option_basic, inputs.option_daily))
    if inputs.option_iv_daily is not None and not inputs.option_iv_daily.empty:
        iv = inputs.option_iv_daily.copy()
        iv["trade_date"] = to_datetime_series(iv["trade_date"])
        iv_col = first_existing(iv, ["iv_atm", "iv", "implied_volatility"])
        if iv_col:
            frames.append(iv.groupby("trade_date", as_index=False).agg(iv_atm=(iv_col, "mean")))
    elif inputs.option_basic is not None and inputs.option_daily is not None:
        frames.append(build_option_atm_iv(inputs.option_basic, inputs.option_daily, base, cfg))
    if not frames:
        return pd.DataFrame()
    result = _merge_feature_frames(frames)
    features = [column for column in ["put_call_log", "iv_atm"] if column in result]
    if "iv_atm" in result:
        rv = base.copy()
        rv["realized_vol_20d"] = np.log(rv["index_close"]).diff().rolling(20, min_periods=10).std() * np.sqrt(252)
        result = result.merge(rv[["trade_date", "realized_vol_20d"]], on="trade_date", how="left")
        result["iv_realized_spread"] = result["iv_atm"] - result["realized_vol_20d"]
        features.extend(["iv_realized_spread"])
    return _decorate_daily(result[["trade_date", *features]], features, cfg, extremes=features)


def _put_call_features(option_basic: pd.DataFrame, option_daily: pd.DataFrame) -> pd.DataFrame:
    basic = option_basic.copy()
    daily = option_daily.copy()
    basic_code = first_existing(basic, ["ts_code", "opt_code", "code"])
    daily_code = first_existing(daily, ["ts_code", "opt_code", "code"])
    type_col = first_existing(basic, ["call_put", "option_type", "opt_type"])
    amount_col = first_existing(daily, ["amount", "amt"])
    if not all([basic_code, daily_code, type_col, amount_col]):
        return pd.DataFrame()
    basic = basic.rename(columns={basic_code: "option_code"})
    daily = daily.rename(columns={daily_code: "option_code"})
    daily["trade_date"] = to_datetime_series(daily["trade_date"])
    daily["amount"] = clean_numeric(daily[amount_col])
    basic["option_type"] = basic[type_col].astype("string").str.lower()
    data = daily.merge(basic[["option_code", "option_type"]], on="option_code", how="left")
    data["is_put"] = data["option_type"].isin(["p", "put", "认沽"])
    data["is_call"] = data["option_type"].isin(["c", "call", "认购"])
    result = pd.DataFrame({
        "trade_date": sorted(data["trade_date"].dropna().unique()),
    })
    put = data.loc[data["is_put"]].groupby("trade_date")["amount"].sum()
    call = data.loc[data["is_call"]].groupby("trade_date")["amount"].sum()
    result = result.merge(put.rename("put_amount").reset_index(), on="trade_date", how="left")
    result = result.merge(call.rename("call_amount").reset_index(), on="trade_date", how="left")
    result["put_call_log"] = np.log((result["put_amount"].fillna(0) + 1.0) / (result["call_amount"].fillna(0) + 1.0))
    return result[["trade_date", "put_call_log"]]


def _futures_features(inputs: TimingInputData, base: pd.DataFrame, cfg: TimingFeatureConfig) -> pd.DataFrame:
    prefix = _future_prefix(cfg)
    frames: list[pd.DataFrame] = []
    if inputs.futures_holding is not None and not inputs.futures_holding.empty:
        holding = inputs.futures_holding.copy()
        holding["trade_date"] = to_datetime_series(holding["trade_date"])
        symbol_col = first_existing(holding, ["symbol", "ts_code"])
        if symbol_col:
            holding = holding.loc[holding[symbol_col].astype(str).str.startswith(prefix)].copy()
        long_col = first_existing(holding, ["long_hld", "long_holding"])
        short_col = first_existing(holding, ["short_hld", "short_holding"])
        if long_col and short_col:
            agg = holding.groupby("trade_date", as_index=False).agg(
                fut_long_hld=(long_col, "sum"),
                fut_short_hld=(short_col, "sum"),
            )
            agg["fut_ls_log"] = np.log((agg["fut_short_hld"] + 1.0) / (agg["fut_long_hld"] + 1.0))
            frames.append(agg[["trade_date", "fut_ls_log"]])
    if inputs.futures_daily is not None and inputs.futures_basic is not None:
        basis = _basis_features(inputs.futures_daily, inputs.futures_basic, base, cfg, prefix=prefix)
        if not basis.empty:
            frames.append(basis)
    if not frames:
        return pd.DataFrame()
    result = _merge_feature_frames(frames)
    features = [column for column in result.columns if column != "trade_date"]
    return _decorate_daily(result, features, cfg, extremes=features)


def _basis_features(
    futures_daily: pd.DataFrame,
    futures_basic: pd.DataFrame,
    base: pd.DataFrame,
    cfg: TimingFeatureConfig,
    *,
    prefix: str,
) -> pd.DataFrame:
    daily = futures_daily.copy()
    basic = futures_basic.copy()
    daily_code = first_existing(daily, ["ts_code", "symbol"])
    basic_code = first_existing(basic, ["ts_code", "symbol"])
    close_col = first_existing(daily, ["close", "settle"])
    maturity_col = first_existing(basic, ["maturity_date", "delist_date", "last_ddate"])
    if not all([daily_code, basic_code, close_col, maturity_col]):
        return pd.DataFrame()
    daily = daily.rename(columns={daily_code: "future_code"})
    basic = basic.rename(columns={basic_code: "future_code"})
    daily_symbol = daily["future_code"].astype(str).str.extract(r"^([A-Z]{1,3}\d{4})", expand=False)
    daily = daily.loc[daily_symbol.fillna("").str.startswith(prefix)].copy()
    basic_symbol_col = first_existing(basic, ["symbol", "name", "future_code"])
    if basic_symbol_col:
        basic = basic.loc[basic[basic_symbol_col].astype(str).str.match(fr"^{prefix}\d{{4}}", na=False)].copy()
    if daily.empty or basic.empty:
        return pd.DataFrame()
    daily["trade_date"] = to_datetime_series(daily["trade_date"])
    daily["future_price"] = clean_numeric(daily[close_col])
    basic["maturity_date"] = to_datetime_series(basic[maturity_col])
    data = daily.merge(basic[["future_code", "maturity_date"]], on="future_code", how="left").merge(base, on="trade_date", how="left")
    data["days_to_maturity"] = (data["maturity_date"] - data["trade_date"]).dt.days
    data = data.loc[data["days_to_maturity"].gt(cfg.basis_roll_days)].copy()
    data["basis_ann"] = (
        (data["future_price"] - data["index_close"])
        / data["index_close"].replace(0, np.nan)
        * cfg.annualization_days
        / data["days_to_maturity"].replace(0, np.nan)
    )
    data = data.dropna(subset=["basis_ann"])
    if data.empty:
        return pd.DataFrame()
    data["rank"] = data.groupby("trade_date")["days_to_maturity"].rank(method="first")
    near = data.loc[data["rank"].eq(1)].groupby("trade_date", as_index=False)["basis_ann"].mean().rename(columns={"basis_ann": "fut_near_basis_ann"})
    next_contract = data.loc[data["rank"].eq(2)].groupby("trade_date", as_index=False)["basis_ann"].mean().rename(columns={"basis_ann": "fut_next_basis_ann"})
    return near.merge(next_contract, on="trade_date", how="outer")


def _future_prefix(cfg: TimingFeatureConfig) -> str:
    if cfg.future_prefix:
        return cfg.future_prefix.upper()
    return {
        "000300.SH": "IF",
        "000016.SH": "IH",
        "000905.SH": "IC",
        "000852.SH": "IM",
    }.get(cfg.index_code, "IF")


def _moneyflow_features(
    moneyflow: pd.DataFrame | None,
    stock_daily: pd.DataFrame | None,
    cfg: TimingFeatureConfig,
) -> pd.DataFrame:
    if moneyflow is None or moneyflow.empty:
        return pd.DataFrame()
    data = moneyflow.copy()
    data["trade_date"] = to_datetime_series(data["trade_date"])
    main_col = first_existing(data, ["main_net_inflow", "net_mf_amount", "main_net_amount", "net_amount"])
    if main_col is None:
        return pd.DataFrame()
    result = data.groupby("trade_date", as_index=False).agg(main_net=(main_col, "sum"))
    if stock_daily is not None and not stock_daily.empty:
        turnover = _daily_market_amount(stock_daily)
        result = result.merge(turnover, on="trade_date", how="left")
        result["main_net_ratio"] = clean_numeric(result["main_net"]) / result["market_amount"].replace(0, np.nan)
    else:
        result["main_net_ratio"] = clean_numeric(result["main_net"])
    result["main_net_ratio_ma5"] = result["main_net_ratio"].rolling(5, min_periods=3).mean()
    result["main_net_ratio_ma20"] = result["main_net_ratio"].rolling(20, min_periods=10).mean()
    result["main_net_ratio_sum_5d"] = result["main_net_ratio"].rolling(5, min_periods=3).sum()
    result["main_net_ratio_sum_20d"] = result["main_net_ratio"].rolling(20, min_periods=10).sum()
    features = ["main_net_ratio", "main_net_ratio_ma5", "main_net_ratio_ma20", "main_net_ratio_sum_5d", "main_net_ratio_sum_20d"]
    return _decorate_daily(result[["trade_date", *features]], features, cfg, extremes=["main_net_ratio"])


def _macro_features(
    cpi: pd.DataFrame | None,
    pmi: pd.DataFrame | None,
    epu: pd.DataFrame | None,
    cfg: TimingFeatureConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cpi_frame = _macro_single(cpi, "cpi_yoy", ["cpi_yoy", "nt_yoy", "value"])
    if not cpi_frame.empty:
        cpi_frame = cpi_frame.sort_values("trade_date")
        cpi_frame["cpi_yoy_chg_1m"] = cpi_frame["cpi_yoy"].diff(1)
        cpi_frame["cpi_yoy_chg_3m"] = cpi_frame["cpi_yoy"].diff(3)
        frames.append(cpi_frame)
    pmi_frame = _macro_single(pmi, "pmi", ["pmi010000", "pmi", "value"])
    if not pmi_frame.empty:
        pmi_frame = pmi_frame.sort_values("trade_date")
        pmi_frame["pmi_above_50"] = pmi_frame["pmi"].gt(50).astype(float).where(pmi_frame["pmi"].notna())
        pmi_frame["pmi_chg_1m"] = pmi_frame["pmi"].diff(1)
        pmi_frame["pmi_chg_3m"] = pmi_frame["pmi"].diff(3)
        frames.append(pmi_frame)
    epu_frame = _macro_single(epu, "epu", ["epu", "value", "index"])
    if not epu_frame.empty:
        epu_frame = epu_frame.sort_values("trade_date")
        epu_frame["epu_log"] = np.log(epu_frame["epu"].where(epu_frame["epu"] > 0))
        epu_frame["epu_log_chg_1m"] = epu_frame["epu_log"].diff(1)
        frames.append(epu_frame[["trade_date", "epu_log", "epu_log_chg_1m"]])
    if not frames:
        return pd.DataFrame()
    result = _merge_feature_frames(frames)
    macro_columns = [column for column in result.columns if column != "trade_date"]
    result[macro_columns] = result[macro_columns].ffill()
    features: list[str] = []
    if "cpi_yoy" in result:
        features.extend(["cpi_yoy", "cpi_yoy_chg_1m", "cpi_yoy_chg_3m"])
    if "pmi" in result:
        features.extend(["pmi", "pmi_above_50", "pmi_chg_1m", "pmi_chg_3m"])
    if "epu_log" in result:
        features.extend(["epu_log", "epu_log_chg_1m"])
    result["macro_growth_score"] = result.get("pmi", pd.Series(np.nan, index=result.index)) + result.get("pmi_chg_3m", pd.Series(np.nan, index=result.index))
    result["macro_inflation_score"] = result.get("cpi_yoy", pd.Series(np.nan, index=result.index)) + result.get("cpi_yoy_chg_3m", pd.Series(np.nan, index=result.index))
    result["macro_uncertainty_score"] = result.get("epu_log", pd.Series(np.nan, index=result.index))
    features.extend(["macro_growth_score", "macro_inflation_score", "macro_uncertainty_score"])
    return _decorate_daily(result[["trade_date", *features]], features, cfg, extremes=[item for item in ["pmi", "cpi_yoy", "epu_log"] if item in result])


def _macro_single(source: pd.DataFrame | None, value_name: str, candidates: list[str]) -> pd.DataFrame:
    if source is None or source.empty:
        return pd.DataFrame()
    data = source.copy()
    date_col = first_existing(data, ["available_date", "ann_date", "trade_date", "month", "date"])
    value_col = first_existing(data, candidates)
    if date_col is None or value_col is None:
        return pd.DataFrame()
    data["trade_date"] = to_datetime_series(data[date_col])
    data[value_name] = clean_numeric(data[value_col])
    return data.groupby("trade_date", as_index=False)[value_name].last()


def _risk_state_features(dataset: pd.DataFrame, cfg: TimingFeatureConfig) -> list[str]:
    names: list[str] = []
    for column in ["index_vol_20d", "index_drawdown_60d"]:
        if column in dataset:
            names.extend(add_rolling_normalizations(dataset, column, z_window=cfg.z_window, pct_window=cfg.pct_window, z_clip=cfg.z_clip, pct_clip=cfg.pct_clip))
    return names


def _interaction_features(dataset: pd.DataFrame) -> list[str]:
    definitions = {
        "cheap_and_panic": ("erp_pct_756", "iv_atm_pct_756"),
        "cheap_and_breadth_repair": ("erp_pct_756", "breadth_thrust"),
        "panic_and_breadth_repair": ("iv_atm_pct_756", "breadth_thrust"),
        "high_leverage_and_weak_breadth": ("rzmre_ratio_pct_756", "up_ratio_pct_756"),
        "basis_discount_and_panic": ("fut_near_basis_ann_pct_756", "iv_atm_pct_756"),
        "pmi_down_and_epu_high": ("pmi_chg_3m", "epu_log_pct_756"),
    }
    names: list[str] = []
    for name, (left, right) in definitions.items():
        if left not in dataset or right not in dataset:
            continue
        if name == "high_leverage_and_weak_breadth":
            dataset[name] = dataset[left] * (1.0 - dataset[right])
        elif name == "basis_discount_and_panic":
            dataset[name] = (1.0 - dataset[left]) * dataset[right]
        elif name == "pmi_down_and_epu_high":
            dataset[name] = dataset[left].clip(upper=0).abs() * dataset[right]
        else:
            dataset[name] = dataset[left] * dataset[right]
        names.append(name)
    return names


def _future_excess_return(base: pd.DataFrame, index_daily: pd.DataFrame, cfg: TimingFeatureConfig, *, horizon: int) -> pd.Series:
    close = base["index_close"]
    future = close.shift(-horizon) / close - 1
    if not cfg.benchmark_code:
        return future
    benchmark = _base_index_frame(index_daily, cfg.benchmark_code)
    benchmark = base[["trade_date"]].merge(benchmark, on="trade_date", how="left")["index_close"]
    return future - (benchmark.shift(-horizon) / benchmark - 1)


def _decorate_daily(
    frame: pd.DataFrame,
    features: list[str],
    cfg: TimingFeatureConfig,
    *,
    extremes: list[str],
) -> pd.DataFrame:
    result = frame.sort_values("trade_date").copy()
    final_features = list(features)
    for column in list(features):
        if column not in result:
            continue
        final_features.extend(add_rolling_normalizations(result, column, z_window=cfg.z_window, pct_window=cfg.pct_window, z_clip=cfg.z_clip, pct_clip=cfg.pct_clip))
        final_features.extend(add_changes(result, column, tuple(window for window in cfg.change_windows if window <= 60)))
    for column in extremes:
        pct_column = f"{column}_pct_{cfg.pct_window}"
        if pct_column in result:
            final_features.extend(add_extreme_flags(result, pct_column, prefix=column, low=cfg.extreme_low, high=cfg.extreme_high))
    return result[["trade_date", *dict.fromkeys(final_features)]]


def _daily_market_amount(stock_daily: pd.DataFrame) -> pd.DataFrame:
    data = stock_daily.copy()
    data["trade_date"] = to_datetime_series(data["trade_date"])
    amount_col = first_existing(data, ["amount_cny", "amount", "amt"])
    if amount_col is None:
        return pd.DataFrame({"trade_date": data["trade_date"].drop_duplicates(), "market_amount": np.nan})
    data["market_amount"] = clean_numeric(data[amount_col])
    return data.groupby("trade_date", as_index=False)["market_amount"].sum()


def _merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    result = valid[0].copy()
    for frame in valid[1:]:
        result = result.merge(frame, on="trade_date", how="outer")
    return result.sort_values("trade_date").reset_index(drop=True)
