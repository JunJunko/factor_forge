# Mamba-state to LightGBM pilot

This pilot tests one narrow question: does a causal state representation of recent PIT features and frozen anomaly-event channels add stable out-of-sample information to the existing cross-sectional LightGBM model?

It does not implement Graph-Mamba, explicit feature gates, causal factor sensitivity, or LLM hypothesis generation.

## Architecture

```text
Immutable PIT panel
  -> existing cross-sectional raw features
  -> eight frozen Radar templates as dense historical channels
  -> on-demand [stock, date, lookback, feature] sequences
  -> causal reference selective-state encoder
  -> frozen state_00..state_N embeddings
  -> Raw / State / Raw+State LightGBM ablation
  -> existing backtest and artifact index
```

Radar event channels are restricted to `eligible`, `event`, `severity`, and `valid`. The channel API cannot return arbitrary panel columns, and event severity is zero on valid non-event rows. `ObservationCard` remains unchanged and label-free.

## Install

The reference encoder is an optional dependency:

```powershell
python -m pip install -e ".[ml,mamba]"
```

On a Windows CPU research machine, install the official CPU PyTorch wheel first if needed:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The `torch_reference` backend is a small, causal selective-state implementation for validating the research contract. It is not the optimized `mamba-ssm` CUDA kernel. A production `mamba-ssm` backend should run in a separately pinned Linux/NVIDIA environment.

## Run

```powershell
python -m factor_forge.cli ml mamba-state-run `
  configs/ml/mamba_state_lightgbm_pilot_v1.yaml
```

The frozen production pilot uses:

- 60 trading-day sequences;
- a minimum of 40 valid input days;
- eight explicitly listed Radar templates;
- a 16-dimensional state embedding;
- masked reconstruction without future-return labels;
- three frozen random seeds;
- identical samples and LightGBM parameters across all ablation arms.

## Artifacts

```text
artifacts/mamba_state_runs/<run_id>/
  checkpoints/encoder_seed_<seed>.pt
  config.yaml
  encoder_training_history.parquet
  manifest.json
  model_comparison.csv
  modeling_dataset.parquet
  raw/
  raw_state/
  report.md
  sequence_index.parquet
  state/
  state_embeddings.parquet
  state_schema.json
  summary.json
  temporal_audit.json
```

`state_schema.json` freezes the raw feature list, event channel list, Radar definition hashes, encoder checkpoint hashes, state column names, and seed aggregation rule. SQLite indexes only the manifest; it does not store sequence tensors or embeddings.

## Interpretation boundary

State embeddings are predictive representations, not causal factor sensitivities. Promotion requires `raw_state` to improve over `raw` on the common test sample across RankIC, cost-adjusted TopN behavior, temporal stability, regime slices, and seed stability. One favorable aggregate return is insufficient.

If this pilot fails to improve the raw baseline, stop. Do not add Graph, feature gates, more layers, or threshold variants to rescue the result.

## Bounded anomaly-driven demonstration

To demonstrate the chain on one frozen anomaly scan without launching the 11-million-row production run:

```powershell
python -m factor_forge.cli ml mamba-anomaly-demo `
  --scan-summary artifacts/market_anomaly_scans/<scan_id>/scan_summary.json
```

This command uses only quality-passing templates with current events, builds historical event sequences from their immutable event artifacts, reserves the latest mature dates for a test segment, and ranks the current event pool. It is a diagnostic demo over one discovery window, not a promotion experiment.
