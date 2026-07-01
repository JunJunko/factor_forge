# Factor Combination Architecture

The stable CLI dispatches by `kind`. Missing `kind` and `kind: factor` retain the original atomic path. Only `kind: factor_combination` enters `FactorCombinationEngine`.

The atomic engine computes raw standard factor rows. The combination engine resolves YAML references, reuses content-addressed atomic caches, applies the experiment scope, preprocesses each daily cross-section, aligns direction, combines scores, and applies filters. It emits `trade_date`, `ts_code`, `factor_value`, and `valid_flag`. The existing L0/L1 evaluator, industry-slice evaluator, scorer, and backtest engine consume that table unchanged.

For industry slices, the selector is built first and each requested scope calls the combination builder again with its own mask. Therefore Top2/Top5/Top10/Bottom5 normalization is not inherited from the all-market score.
