# Backward Compatibility Report

- Stable two-YAML CLI: unchanged.
- Atomic YAML without `kind`: defaults to `factor`.
- Atomic YAML with `kind: factor`: follows the original runner and `FactorEngine` path.
- Existing experiment YAML: unchanged and accepted.
- Atomic factor output/evaluation/backtest behavior: no atomic execution code was modified.
- Combination YAML directly references existing atomic YAML and needs no third config.

Full regression result is recorded in `test_report.md`.
