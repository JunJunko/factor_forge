"""Dataset assembly for the supply-contraction pipeline: panel -> feature matrix + label.

This is the inline-pandas analogue of :mod:`factor_forge.ml.dataset` -- it bypasses the
DSL and calls the pure primitives in :mod:`factor_forge.ml.supply_features` directly,
applies the document's per-day winsorize + cross-sectional z-score (sec. 3.4), the sample
filter (sec. 16), and builds the forward industry-neutral label (sec. 15).  The output is
the canonical Qlib ``(datetime, instrument)`` frame plus the A/B feature-group registry
(document sec. 17).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import supply_features as sf
from .supply_config import SupplyFeatureConfig, SupplyLabelConfig


# Document sec. 17 ablation split.  Model A = controls (size / liquidity /
# microstructure / volatility / environment); Model B = Model A + supply core
# (the price-strength + conditional-volume-residual family whose incremental alpha is
# the whole point of the study).
FEATURE_GROUP_REGISTRY: dict[str, list[str]] = {
    "controls": [
        "log_raw_price",
        "tick_noise",
        "log_float_market_cap",
        "log_avg_amount_20",
        "amihud_illiquidity_20",
        "zero_return_days_20",
        "volatility_20",
        "turnover_zscore_20",
        "turnover_zscore_60",
        "amount_zscore_20",
        "industry_return_5",
        "industry_breadth",
        "market_breadth",
        "intraday_range",
        "close_location",
        "upper_shadow_ratio",
        "body_ratio",
        "gap_return",
        "price_impact_1",
        "price_impact_5",
    ],
    "supply_core": [
        "excess_ret_5",
        "excess_ret_1",
        "excess_ret_3",
        "excess_ret_10",
        "risk_adjusted_ret_5",
        "volume_residual",
        "scarcity",
        "volume_residual_5d_mean",
        "scarcity_days_ratio_5",
        "scarcity_slope_5",
        "up_days_ratio_5",
        "effective_ticks_5",
        "tick_return",
    ],
    # Interaction terms (document sec. 13); trees can learn these but the document
    # recommends keeping both composites and their raw components (sec. 13.5).
    "composite": [
        "simple_low_volume_rise",
        "conditional_scarcity_factor",
        "close_quality_scarcity_factor",
        "persistent_scarcity_factor",
        "price_adjusted_scarcity_factor",
    ],
    # V2: stable turnover baseline + recent no-volume rise (handoff doc sec. 3 / 6.4).
    # The three independent legs -- baseline stability, 2-day price strength, and the
    # un-activated-volume z -- live here RAW and separate; the dataset never collapses
    # them into a single composite (handoff 3.1 / 4.2.3).  The manual composites
    # (no_volume_activation_score, stable_no_volume_rise) are deferred to task 9 and only
    # built after the raw legs pass the univariate / conditional checks.
    "baseline_structure": [
        "baseline_turnover_mean_28",
        "baseline_turnover_std_28",
        "baseline_amount_mean_28",
        "turnover_vol_rank_28",
        "turnover_stability_28",
        "excess_ret_2",
        "price_strength_2",
        "recent_volume_z_t1_raw",
        "recent_volume_z_t_raw",
        "recent_volume_z_mean_2_raw",
        "recent_volume_z_mean_2_clip",
        "recent_volume_z_max_2_raw",
        "recent_volume_z_max_2_clip",
        "effective_ticks_2",
    ],
}


# V2 daily cross-section percentile-rank features in [0, 1]; they bypass the winsor+zscore
# pass so their interpretability is preserved (handoff doc 3.5).  Other V2 fields (baseline
# levels, the z family, price_strength_2, effective_ticks_2) go through the same per-day
# winsor + cross-sectional zscore as the V1 features.
CROSS_SECTION_RANK_FEATURES = frozenset({"turnover_vol_rank_28", "turnover_stability_28"})


REQUIRED_PANEL_COLUMNS = {
    "trade_date", "ts_code",
    "raw_open", "raw_high", "raw_low", "raw_close", "pre_close",
    "adj_open", "adj_close",
    "turnover_rate", "amount_cny", "circ_mv_cny",
    "industry_l1_code",
    "is_st", "is_delisting_period", "is_suspended", "is_tradeable", "listing_trade_days",
}


def _forward_industry_neutral_label(
    adj_open: pd.Series,
    adj_close: pd.Series,
    stocks: pd.Series,
    dates: pd.Series,
    industries: pd.Series,
    horizon: int,
    method: str,
) -> pd.Series:
    """Forward industry-neutral return (document sec. 15).

    ``open_to_open``: ``adj_open[t+h+1] / adj_open[t+1] - 1`` (matches the project
    BacktestEngine's adj_open mark-to-market and Qlib's ``deal_price=("$open","$open")``).
    ``open_to_close``: ``adj_close[t+h] / adj_open[t+1] - 1`` (the document's literal
    sec. 15.1 definition).  The industry benchmark is the leave-one-out mean of members'
    forward returns, so a stock never benchmarks against itself.
    """
    if method == "open_to_open":
        fwd = (
            adj_open.groupby(stocks, sort=False).shift(-(horizon + 1))
            / adj_open.groupby(stocks, sort=False).shift(-1)
            - 1.0
        )
    else:  # open_to_close
        fwd = (
            adj_close.groupby(stocks, sort=False).shift(-horizon)
            / adj_open.groupby(stocks, sort=False).shift(-1)
            - 1.0
        )
    industry_fwd = sf._industry_loo_mean(fwd, dates, industries)
    return fwd - industry_fwd


def build_supply_dataset(
    panel: pd.DataFrame,
    index_daily: pd.DataFrame | None,
    features: SupplyFeatureConfig,
    label: SupplyLabelConfig,
    sample_weight_train: tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(dataset_df, feature_names)`` in Qlib ``(datetime, instrument)`` shape.

    ``dataset_df`` carries every computed feature column plus ``label``; the feature
    *group* a model uses is selected later via :data:`FEATURE_GROUP_REGISTRY`.  Excluded
    samples (document sec. 16) are NaN-masked, NOT dropped -- Qlib's backtest calendar
    needs the full date x instrument grid.
    """
    missing = REQUIRED_PANEL_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing supply-contraction fields: {sorted(missing)}")

    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    stocks = data["ts_code"]
    dates = data["trade_date"]
    industries = data["industry_l1_code"]
    valid_mask = (
        data["is_tradeable"].eq(True)
        & ~data["is_suspended"]
        & ~data["is_st"]
        & ~data["is_delisting_period"]
        & data["listing_trade_days"].ge(features.min_listing_days)
        & industries.notna()
    )

    adj_close = data["adj_close"]
    adj_open = data["adj_open"]
    log_return = sf.log_returns(adj_close, stocks)

    # --- returns & volatility ---
    excess = sf.excess_returns(
        adj_close, stocks, dates, industries, features.excess_return_windows
    )
    industry_ret = {
        n: sf._industry_loo_mean(
            adj_close.groupby(stocks, sort=False).pct_change(n, fill_method=None),
            dates,
            industries,
        )
        for n in features.excess_return_windows
    }
    vol20 = sf.volatility(log_return, stocks, features.volatility_window, ddof=features.volatility_ddof)
    abs_excess_ret_1 = excess[features.excess_return_windows[0]].abs() if features.excess_return_windows else log_return.abs()

    # --- activity ---
    tz = {w: sf.log_turnover_zscore(data["turnover_rate"], stocks, w) for w in features.turnover_windows}
    amt_z = sf.amount_zscore(data["amount_cny"], stocks, features.amount_window)
    log_amt = sf.log_avg_amount(data["amount_cny"], stocks, features.log_amount_window)

    # --- activity-regime regressors ---
    mkt_tz = sf.market_turnover_z(data["turnover_rate"], dates, stocks, window=60)
    ind_tz = sf.industry_turnover_z(data["turnover_rate"], dates, industries, stocks, window=60)

    # --- K-line (raw prices) ---
    intraday = sf.intraday_range(data["raw_high"], data["raw_low"], data["pre_close"])
    close_loc = sf.close_location(data["raw_close"], data["raw_low"], data["raw_high"])
    upper_shadow = sf.upper_shadow_ratio(
        data["raw_high"], data["raw_open"], data["raw_close"], data["raw_low"]
    )
    body = sf.body_ratio(data["raw_open"], data["raw_close"], data["raw_high"], data["raw_low"])
    gap = sf.gap_return(data["raw_open"], data["pre_close"], stocks)

    # --- conditional volume residual (the signature feature) ---
    vres = sf.volume_residual(
        data["turnover_rate"], abs_excess_ret_1, intraday, mkt_tz, ind_tz, vol20,
        stocks, features.volume_residual_window, features.volume_residual_min_periods,
    )
    vres_5d = sf.volume_residual_mean(vres, stocks, features.persistence_window)
    scar = sf.scarcity(vres)
    scar_days = sf.scarcity_days_ratio(vres, stocks, features.persistence_window)
    scar_slope = sf.scarcity_slope(vres, stocks, features.persistence_window)

    # --- price impact ---
    pi1 = sf.price_impact_1(excess[1] if 1 in excess else excess[features.excess_return_windows[0]],
                            data["turnover_rate"], features.price_impact_denom_floor)
    pi5 = sf.price_impact_5(excess.get(5, excess[features.excess_return_windows[0]]),
                            data["turnover_rate"], stocks, features.price_impact_denom_floor)
    up_days = sf.up_days_ratio(excess[1] if 1 in excess else excess[features.excess_return_windows[0]],
                               stocks, features.persistence_window)

    # --- microstructure (raw prices) ---
    log_rp = sf.log_raw_price(data["raw_close"])
    tick_r = sf.tick_return(data["raw_close"], features.tick_size)
    tick_n = sf.tick_noise(data["raw_close"], vol20, features.tick_size)
    eff_ticks = sf.effective_ticks(data["raw_close"], stocks, features.effective_ticks_window, features.tick_size)

    # --- liquidity & size ---
    log_fmv = sf.log_float_market_cap(data["circ_mv_cny"])
    amihud = sf.amihud_illiquidity(log_return.abs(), data["amount_cny"], stocks, features.amihud_window)
    zero_days = sf.zero_return_days(log_return, stocks, features.zero_return_window)

    # --- environment ---
    mkt_breadth = sf.market_breadth(log_return, dates, valid_mask)
    ind_breadth = sf.industry_breadth(log_return, dates, industries, valid_mask)

    # --- V2: stable turnover baseline + recent no-volume rise (handoff doc sec. 3) ---
    bw, ew = features.baseline_window, features.event_window
    lt = sf._log_turnover(data["turnover_rate"])
    baseline_t_mean = sf.baseline_window_stat(lt, stocks, bw, ew, method="mean")
    baseline_t_std = sf.baseline_window_stat(lt, stocks, bw, ew, method="std", ddof=0)
    baseline_amt_mean = sf.baseline_window_stat(
        data["amount_cny"].where(data["amount_cny"] > 0), stocks, bw, ew, method="mean"
    )
    # Volatility ending at t-2 so the last two event-day returns do not pollute it (handoff 3.6).
    vol20_prior = sf.volatility_prior(
        log_return, stocks, features.volatility_window, ew, ddof=features.volatility_ddof
    )
    excess_2 = sf.excess_returns(adj_close, stocks, dates, industries, windows=[ew])[ew]
    price_str_2 = sf.price_strength_2(excess_2, vol20_prior)
    eff_ticks_2 = sf.effective_ticks(data["raw_close"], stocks, ew, features.tick_size)
    # std_floor: daily cross-section quantile of the baseline std, valid stocks only
    # (handoff 3.7 method 1; train_period_fixed is a phase-2 option, not yet wired).
    if features.std_floor_method == "cross_section_quantile":
        std_floor = (
            baseline_t_std.where(valid_mask)
            .groupby(dates, sort=False)
            .transform(lambda s: s.quantile(features.std_floor_quantile))
        )
    else:
        raise NotImplementedError(
            f"std_floor_method={features.std_floor_method!r} not implemented in phase 1"
        )
    z_t1_raw, z_t_raw = sf.recent_volume_z(lt, baseline_t_mean, baseline_t_std, std_floor, stocks)
    z_agg = sf.recent_volume_z_aggregates(
        z_t1_raw, z_t_raw, features.z_clip_lower, features.z_clip_upper
    )
    # Daily cross-section percentile rank of the baseline std (valid stocks only); kept in
    # [0,1] and exempted from the winsor+zscore pass below (handoff 3.5).
    turnover_vol_rank_28 = baseline_t_std.where(valid_mask).groupby(dates, sort=False).rank(pct=True)
    turnover_stability_28 = 1.0 - turnover_vol_rank_28

    # --- assemble ---
    excess_ret_5_raw = excess.get(5, excess[features.excess_return_windows[0]])
    rar5 = sf.risk_adjusted_ret_5(excess_ret_5_raw, vol20)
    turnover_z_60 = tz.get(60, tz[features.turnover_windows[-1]])
    feature_columns: dict[str, pd.Series] = {
        "volatility_20": vol20,
        "log_raw_price": log_rp,
        "tick_return": tick_r,
        "tick_noise": tick_n,
        "effective_ticks_5": eff_ticks,
        "log_float_market_cap": log_fmv,
        "log_avg_amount_20": log_amt,
        "amihud_illiquidity_20": amihud,
        "zero_return_days_20": zero_days,
        "turnover_zscore_20": tz.get(20, tz[features.turnover_windows[0]]),
        "turnover_zscore_60": turnover_z_60,
        "amount_zscore_20": amt_z,
        "intraday_range": intraday,
        "close_location": close_loc,
        "upper_shadow_ratio": upper_shadow,
        "body_ratio": body,
        "gap_return": gap,
        "industry_return_5": industry_ret.get(5, industry_ret[features.excess_return_windows[-1]]),
        "industry_breadth": ind_breadth,
        "market_breadth": mkt_breadth,
        "price_impact_1": pi1,
        "price_impact_5": pi5,
        "risk_adjusted_ret_5": rar5,
        "volume_residual": vres,
        "scarcity": scar,
        "volume_residual_5d_mean": vres_5d,
        "scarcity_days_ratio_5": scar_days,
        "scarcity_slope_5": scar_slope,
        "up_days_ratio_5": up_days,
        # V2 baseline_structure (handoff doc sec. 3 / 6.4) -- three independent RAW legs.
        "baseline_turnover_mean_28": baseline_t_mean,
        "baseline_turnover_std_28": baseline_t_std,
        "baseline_amount_mean_28": baseline_amt_mean,
        "turnover_vol_rank_28": turnover_vol_rank_28,
        "turnover_stability_28": turnover_stability_28,
        "excess_ret_2": excess_2,
        "price_strength_2": price_str_2,
        "recent_volume_z_t1_raw": z_t1_raw,
        "recent_volume_z_t_raw": z_t_raw,
        "recent_volume_z_mean_2_raw": z_agg["recent_volume_z_mean_2_raw"],
        "recent_volume_z_mean_2_clip": z_agg["recent_volume_z_mean_2_clip"],
        "recent_volume_z_max_2_raw": z_agg["recent_volume_z_max_2_raw"],
        "recent_volume_z_max_2_clip": z_agg["recent_volume_z_max_2_clip"],
        "effective_ticks_2": eff_ticks_2,
    }
    for n in features.excess_return_windows:
        feature_columns[f"excess_ret_{n}"] = excess[n]

    # --- composite factors (document sec. 13), built from RAW component values ---
    simple = sf.simple_low_volume_rise(excess_ret_5_raw, turnover_z_60, stocks)
    conditional = sf.conditional_scarcity_factor(rar5, vres)
    close_quality = sf.close_quality_scarcity_factor(conditional, close_loc, upper_shadow)
    persistent = sf.persistent_scarcity_factor(conditional, scar_days, up_days)
    # price_adjusted uses the raw price_weight (training sample weight, sec. 8.6) as a
    # multiplier per the document; computed here from the same raw tick_noise.
    pw_raw = sf.price_weight(tick_n, features.price_weight_lambda)
    price_adjusted = sf.price_adjusted_scarcity_factor(persistent, pw_raw)
    feature_columns.update({
        "simple_low_volume_rise": simple,
        "conditional_scarcity_factor": conditional,
        "close_quality_scarcity_factor": close_quality,
        "persistent_scarcity_factor": persistent,
        "price_adjusted_scarcity_factor": price_adjusted,
    })

    feature_names = list(feature_columns.keys())
    out = pd.DataFrame({"datetime": dates.to_numpy(), "instrument": stocks.to_numpy()})
    for name, series in feature_columns.items():
        out[name] = series.to_numpy()

    # --- label ---
    out["label"] = _forward_industry_neutral_label(
        adj_open, adj_close, stocks, dates, industries,
        horizon=label.horizon, method=label.label_method,
    ).to_numpy()

    # --- document sec. 3.4: per-day 1%/99% winsorize then cross-sectional z-score.
    # V2 percentile-rank features in [0,1] bypass this pass to keep their interpretability
    # (handoff 3.5); everything else follows the document's sec. 3.4 recipe. ---
    scale_targets = [n for n in feature_names if n not in CROSS_SECTION_RANK_FEATURES]
    out[scale_targets] = out[scale_targets].replace([np.inf, -np.inf], np.nan)
    if features.winsor_quantile:
        q = features.winsor_quantile
        grouped = out.groupby("datetime")
        lower = grouped[scale_targets].transform(lambda s: s.quantile(q))
        upper = grouped[scale_targets].transform(lambda s: s.quantile(1 - q))
        out[scale_targets] = out[scale_targets].clip(lower, upper)
    if features.cross_sectional_zscore:
        grouped = out.groupby("datetime")[scale_targets]
        mean = grouped.transform("mean")
        std = grouped.transform("std", ddof=0)
        out[scale_targets] = (out[scale_targets] - mean) / std.replace(0, np.nan)

    # --- sample filter (sec. 16): NaN out excluded rows, keep the grid ---
    exclude = ~valid_mask.to_numpy()
    out.loc[exclude, feature_names + ["label"]] = np.nan

    # --- sample_weight (document sec. 8.6 / 9.5 / 9.6): a TRAINING WEIGHT, not a model
    # input.  liquidity_weight thresholds (A_low / A_full) are estimated on the training
    # segment ONLY to avoid look-ahead (sec. 9.5).  Carried as a column so the runner can
    # fold it into the LightGBM reweighter; it is excluded from feature_names on purpose. ---
    if features.use_sample_weight and sample_weight_train is not None:
        train_lo, train_hi = pd.Timestamp(sample_weight_train[0]), pd.Timestamp(sample_weight_train[1])
        train_mask = dates.between(train_lo, train_hi)
        train_vals = log_amt[train_mask].dropna()
        if len(train_vals):
            a_low = float(train_vals.quantile(features.liquidity_weight_low_quantile))
            a_full = float(train_vals.quantile(features.liquidity_weight_full_quantile))
        else:
            a_low = a_full = float("nan")
        liw = sf.liquidity_weight(log_amt, a_low, a_full)
        sw = sf.sample_weight(pw_raw, liw)
        out["sample_weight"] = sw.to_numpy()
        out.loc[exclude, "sample_weight"] = np.nan
    else:
        out["sample_weight"] = np.nan
    return out, feature_names


def features_for_groups(groups: list[str]) -> list[str]:
    """Flatten a list of feature-group names into a concrete feature list."""
    names: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUP_REGISTRY:
            raise ValueError(f"unknown feature group {group!r}; known: {sorted(FEATURE_GROUP_REGISTRY)}")
        for name in FEATURE_GROUP_REGISTRY[group]:
            if name not in names:
                names.append(name)
    return names


def features_for_model(spec) -> list[str]:
    """Resolve a :class:`ModelSpec` to a concrete feature list (groups + explicit)."""
    names = features_for_groups(spec.feature_groups)
    for name in spec.features:
        if name not in names:
            names.append(name)
    return names


def to_qlib_frame(dataset: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Qlib canonical MultiIndex ``(datetime, instrument)`` with ``(feature, name)`` columns."""
    frame = dataset.set_index(["datetime", "instrument"])[feature_names + ["label"]].sort_index()
    frame.columns = pd.MultiIndex.from_tuples(
        [("feature", c) for c in feature_names] + [("label", "LABEL0")]
    )
    return frame
