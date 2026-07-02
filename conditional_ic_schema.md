# Conditional IC in L1

`stage_l1.conditional_ic` evaluates whether the main factor predicts returns differently across states defined by a conditioning factor. It is diagnostic evidence and does not change the existing L1 pass/fail gate.

```yaml
stage_l1:
  forward_horizons: [1, 3, 5, 10, 15]
  min_cross_section: 100
  universes: [tradeable, liquid]
  conditional_ic:
    enabled: true
    conditioning_factor: configs/factors/deviation_60d.yaml
    quantile_groups: 5
    min_group_size: 20
    store_daily_ic: true
```

`conditioning_factor` accepts:

- `main_factor`: split the main factor itself into daily quantiles, then measure its within-quantile IC.
- A path to an atomic factor or factor-combination YAML: split on that factor and measure the main factor's IC inside each state.

Horizons and universes are inherited from the parent L1 configuration. Quantiles are assigned independently each day within the selected universe. Ties remain in the same quantile; consequently a discrete conditioning factor can leave some quantiles empty. A daily bucket is included only when it contains at least `min_group_size` evaluable stocks.

Summary results are stored under `conditional_ic` in `l1_predictive_power.json` and as a flat table in `l1_conditional_ic_summary.csv`. Every result contains the condition quantile, observations, daily coverage, mean future return, mean Rank IC, ICIR, positive ratio, ordinary t statistic, and Newey-West t/p values. `significance_rank` ranks the five intervals by absolute Newey-West t value within each variant/universe/horizon context. `fdr_q` applies Benjamini-Hochberg correction across the conditional tests in the run. `strongest_by_context` points to the leading interval in each context, while `strongest_result` points to the leading row in the whole run; these are convenience pointers, not independent tests.

When `store_daily_ic` is true, the complete daily series is saved to `l1_conditional_ic_daily.parquet`. If the condition comes from another YAML, the run also stores `inputs/conditioning_factor.yaml` and `conditioning_factor_values.parquet`. The conditioning YAML content participates in the run ID hash, and the conditioning factor must pass the temporal-consistency audit.

## Conditional L2 portfolio

L2 can reuse the exact L1 conditioning factor and quantile definition as a point-in-time candidate filter:

```yaml
stage_l2:
  condition_filter:
    enabled: true
    source: stage_l1_conditional_ic
    include_quantiles: [5]
    min_cross_section: 100
    benchmark: condition_equal_weight
```

Enabling the filter requires `stage_l1.conditional_ic.enabled: true`. Only stocks in the included daily condition quantiles can enter the main factor's TopN selection. The primary `benchmark_return` and `annualized_excess_return` use the condition-eligible equal-weight portfolio. The daily and metric artifacts also retain `universe_benchmark_return`, `universe_benchmark_annualized_return`, and `annualized_excess_return_vs_universe` for comparison with the unfiltered universe.

Selected point-in-time memberships are stored in `l2_condition_membership.parquet`, with coverage diagnostics in `l2_condition_filter_summary.csv`. Trades include `condition_quantile` for execution audit.
