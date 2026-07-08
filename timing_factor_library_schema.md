# Timing Factor Library

This module builds daily A-share market-timing features for machine-learning
position models. The public entry point is:

```python
from factor_forge.timing import TimingInputData, TimingFeatureConfig, build_timing_dataset
```

Or from local parquet/csv/xlsx tables:

```powershell
python -m factor_forge.cli ml timing-build configs/ml/timing_factor_library_v1.yaml
```

## Point-In-Time Convention

Features are computed on the source observation date and then shifted by
`data_lag` trading rows before they are exposed to the model. The default lag is
1 day. Monthly macro data should use `available_date` when available; the builder
daily-aligns it, forward-fills it after release, and then applies the same lag.

The default labels are configured by `horizons`, for example `[5, 10, 20]`:

```text
label_5d_excess_return = close[t+5] / close[t] - 1
label_10d_excess_return = close[t+10] / close[t] - 1
label_20d_excess_return = close[t+20] / close[t] - 1
```

If `benchmark_code` is configured, the benchmark's same-horizon return is
subtracted.

When 10-year bond yield data is unavailable, set
`fallback_bond_10y_yield`, for example `0.025`, to keep ERP features usable:

```text
erp = 1 / pe_ttm - fallback_bond_10y_yield
```

## Input Tables

Only `index_daily` is required. Other tables are optional; missing blocks simply
produce fewer feature groups.

| Config key | Typical Tushare source | Minimum useful columns |
|---|---|---|
| `index_daily` | `index_daily` | `trade_date`, `ts_code`, `close` |
| `stock_daily` | `daily` / adjusted panel | `trade_date`, `ts_code`, `pct_chg` or `close` + `pre_close`, `amount` |
| `index_dailybasic` | `index_dailybasic` | `trade_date`, `pe_ttm` or `pe` |
| `bond_yield` | `yc_cb` | `trade_date`, `curve_term`, `yield` |
| `margin` | `margin` | `trade_date`, `rzmre`, `rzye` |
| `option_basic` | `opt_basic` | option code, call/put type, strike, maturity date |
| `option_daily` | `opt_daily` | `trade_date`, option code, `close` or `settle`, `amount` |
| `option_iv_daily` | precomputed | `trade_date`, `iv_atm` or `iv` |
| `futures_basic` | `fut_basic` | futures code, maturity date |
| `futures_daily` | `fut_daily` | `trade_date`, futures code, `close` or `settle` |
| `futures_holding` | `fut_holding` | `trade_date`, `long_hld`, `short_hld` |
| `moneyflow` | money-flow endpoint | `trade_date`, main net flow column |
| `cpi` | `cn_cpi` | `available_date` or date column, CPI value |
| `pmi` | `cn_pmi` | `available_date` or date column, PMI value |
| `epu` | external | `available_date` or date column, EPU value |

Column aliases are accepted for common naming differences.

For futures features, `future_prefix` controls which stock-index futures family
is used. Defaults are inferred from `index_code`: `000300.SH -> IF`,
`000016.SH -> IH`, `000905.SH -> IC`, `000852.SH -> IM`.

## Feature Shape

The builder keeps three layers:

1. Raw economic metrics, such as `erp`, `up_ratio`, `rzmre_ratio`,
   `put_call_log`, `iv_atm`, `fut_ls_log`, `fut_near_basis_ann`, `pmi`.
2. Rolling representations:
   - `*_z_<window>` uses rolling median/MAD z-score and clips to `[-z_clip, z_clip]`.
   - `*_pct_<window>` uses rolling historical percentile and clips to
     `pct_clip`.
   - `*_chg_5d`, `*_chg_20d`, `*_chg_60d` capture state changes.
3. State and interaction features:
   - `*_low_5`, `*_low_10`, `*_high_90`, `*_high_95`
   - `cheap_and_panic`
   - `cheap_and_breadth_repair`
   - `panic_and_breadth_repair`
   - `high_leverage_and_weak_breadth`
   - `basis_discount_and_panic`
   - `pmi_down_and_epu_high`

The returned `TimingFeatureResult` includes:

```text
dataset: daily feature table
feature_names: ML input columns
label_name: target column
feature_groups: grouped feature-name mapping
```

## IV

If `option_iv_daily` is supplied, it is used directly. Otherwise the builder
tries to infer a simple near-month ATM IV from `option_basic`, `option_daily`,
and the configured index close using Black-Scholes bisection. This is intended
as a stable first version, not a full volatility surface.
