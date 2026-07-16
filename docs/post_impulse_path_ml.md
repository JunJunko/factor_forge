# Post-impulse path ML

This pipeline turns an abnormal-rise event into a point-in-time event snapshot and a retained
T+1..T+3 path. It is an ML diagnostic pipeline and does not enter the Factor scoring Gate.

## Sample clock

```text
event close T0       detect the impulse
T+1..T+3             observe pressure and price response
signal close T+3     freeze every predictor
next open T+4        label/trading clock starts
T+4..T+13            ten-trading-day outcome window
```

One modeling row is keyed by `event_id`, `ts_code`, `event_date`, and `signal_date`. The raw
path remains in `event_path.parquet` so later feature work does not need to reconstruct or
silently redefine the event sequence.

## Feature blocks

| Prefix | Role |
|---|---|
| `coord__` | Size, beta, volatility and liquidity risk coordinates |
| `event__` | Impulse strength and contemporaneous event context |
| `pressure__` | Pressure prerequisite level, component count and path |
| `absorb__` | Conditional price-impact, low, close and range response |
| `regime__` | Market trend, breadth, style, volatility and industry breadth |
| `interaction__` | Small pre-defined mechanism interactions |

`pressure__present` defines the default modeling population and is excluded from predictors.
Absorption measurements remain missing when the pressure prerequisite fails. A no-down-close
window is never encoded as perfect price-impact resilience.

## Fixed ablation

```text
M0 coordinate baseline
M1 + event
M2 + pressure
M3 + absorption path
M4 + regime
M5 + mechanism interactions
```

All arms use the same pressure-qualified samples. Ridge/logistic imputation, missing indicators,
and scaling are fit on training rows only. Optional LightGBM receives numeric PIT values and NaN,
without whole-dataset scaling. The final score is residualized each day against industry, size,
beta, volatility, and liquidity before the risk-neutral ranking metrics are calculated.

Train and validation tails are purged by `label_horizon + 1` trading days. Random splits and
target encoding are not used.

## Labels

- `label__industry_excess_10d`: next-open to horizon-open return minus the same-date industry
  benchmark.
- `label__quality_atr`: future MFE minus a configured MAE penalty.
- `label__success`: future breakout plus configured MFE and MAE conditions.

## Run

```powershell
python -m factor_forge.cli ml post-impulse-run configs/ml/post_impulse_path_ml_v1.yaml
```

The run writes the event dataset, retained path, feature manifest, split audit, fitted models,
predictions, and M0-M5 metrics to a content-addressed directory under
`artifacts/post_impulse_ml_runs/`.

## Fixed M3 diagnostic

After the M2/M3 ablation has been run, the absorption block can be decomposed without changing
the event sample, thresholds, windows, or model hyperparameters:

```powershell
python -m factor_forge.cli ml post-impulse-m3 configs/ml/post_impulse_m3_diagnostic_v1.yaml
```

The fixed variants are impact, low path, close acceptance, range contraction, their path-core
combination, the precomputed summary, and the full M3 block. Both industry-excess regression and
second-wave classification are compared with M2, and standardized Ridge/Logit coefficients are
saved for diagnosis. This is a post-hoc decomposition, not a new clean hold-out decision.

## Minimal M3 walk-forward Gate

The next stage is frozen to C0 (M2 Logistic) versus C1 (C0 plus four minimal path features), with
four expanding folds and an 11-trading-day purge:

```powershell
python -m factor_forge.cli ml post-impulse-m3-walkforward \
  configs/ml/post_impulse_m3_minimal_walkforward_v1.yaml
```

The run saves OOF probabilities, fold and aggregate metrics, calibration bins, standardized
coefficient signs, and one deterministic next action. Failure of a pre-registered mechanism sign
cannot be repaired by flipping it after inspecting the result; that would be a different market
hypothesis.

## M1 versus M2 OOF executable backtest

After stopping M3 expansion, the event and pressure blocks can be tested directly under an
executable portfolio clock. The comparison keeps the pressure-qualified event pool fixed: M1
uses coordinate and event features, while M2 adds only pressure features.

```powershell
python -m factor_forge.cli ml post-impulse-m2-backtest `
  configs/ml/post_impulse_m1_m2_oof_backtest_v1.yaml
```

The experiment uses four expanding folds, an 11-trading-day purge, fixed Ridge alpha, T+1-open
execution, ten-day sleeves, Top 5/10/20 portfolios, and 20/40/60 bps cost scenarios. Its Gate
requires M2 to improve OOF neutral IC and executable returns, survive cost stress, remain stable
across portfolio breadth and years, and pass stock/month contribution concentration checks. The
saved action is deterministic: forward observation only after every Gate condition passes;
otherwise stop the post-impulse event strategy instead of tuning on the inspected history.

## M2.1 compressed pressure reranker

The M2.1 stage separates incremental reranking value from standalone-strategy viability. Risk
coordinates remain regression controls but their coefficient contributions are excluded from the
trading score. The redundant raw pressure block is compared with three pre-defined mechanisms:
shock intensity, pressure persistence, and pressure resolution.

```powershell
python -m factor_forge.cli ml post-impulse-m21 `
  configs/ml/post_impulse_m21_compressed_reranker_v1.yaml
```

The executable audit divides sleeve cash by the actual buyable selection count and builds a
matched event-cohort portfolio with the same T+1-open entry, ten-day holding, constraints, and
cost. It uses CNY 100 million for every arm so 100-share lot rounding does not leave the broad
cohort structurally underinvested. Signals without enough remaining market dates for a complete holding period are excluded.
Top 5 at 40 bps is the frozen primary development specification; because Top 5 was selected after
the prior history was inspected, only later unseen data can authorize deployment. A historical
reranker pass keeps raw M2 as the reference and places M2.1 in shadow observation; it does not
authorize replacing M2 from the inspected return difference alone.

## M2-PATH realization diagnostic

M2-PATH reuses the frozen OOF C0/M2/M2.1 scores and traces daily Top 5 selections from the T+1
entry open through 1/2/3/5/7/10-trading-day exits. It does not refit a model. Planned exits obey
suspension and limit-down constraints by deferring until the first sellable open.

```powershell
python -m factor_forge.cli ml post-impulse-m2-path `
  configs/ml/post_impulse_m2_path_v1.yaml
```

The attribution separates names common to two variants from names added and dropped by the
pressure block. A horizon can only become a forward hypothesis when added-minus-dropped return is
positive in at least three folds and three years, survives removal of the best one percent of
events, remains positive at 40 bps, and has support from an adjacent horizon. Because all history
through the configured cutoff has already been inspected, even a passing horizon remains shadow
only until later data confirms it.
