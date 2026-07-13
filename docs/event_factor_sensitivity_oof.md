# Event-Mamba factor sensitivity with chronological OOF stacking

This ML family treats every frozen Radar anomaly Episode as the modeling unit. It does not
pool ordinary stock-days and it does not ask an LLM to invent numerical factors.

```text
frozen rule anomaly events
  -> anchored Event Episodes
  -> 60-day causal Event-Mamba sequence
  -> named beta sensitivities + beta-times-factor + residual embedding
  -> chronological OOF blocks
  -> LightGBM event-pool cross-sectional stacking
  -> OOS gate
  -> forward observation before any trading promotion
```

## Frozen named factor basis

1. `short_reversal`: negative five-day return.
2. `trend_acceleration`: five-day return minus one quarter of twenty-day return.
3. `volume_price_efficiency`: five-day absolute displacement per log average amount.
4. `volatility_compression`: negative short/long realized-volatility ratio deviation.
5. `industry_relative_return`: five-day return minus same-date industry mean.
6. `liquidity_displacement`: five-day absolute displacement per log turnover.
7. `intraday_rejection`: close location within the current adjusted bar, centered at zero.

All axes are winsorized and standardized within date. Their definitions are fixed before
validation and are built by the strict ML Feature Builder.

## Event-Mamba outputs

For factor `k`, Event-Mamba emits a bounded named sensitivity `beta_k` and the exact gated
feature `beta_k * factor_k`. A four-dimensional residual event embedding is retained only for
the E3 incremental arm. The model is conditioned on the frozen event-template index and event
severity.

## Strict OOF contract

Each historical block is encoded by an Event-Mamba checkpoint trained and validated strictly
before that block, with a six-trading-day embargo for the five-day T+1-open label. LightGBM
evaluation for block `b` can train only on OOF rows from blocks `< b` whose labels were already
available at the start of `b`.

- E0: controls, severity and template identity.
- E1: E0 plus the seven raw named factors.
- E2: E1 plus OOF beta, gated factors and Event-Mamba auxiliary prediction.
- E3: E2 plus residual OOF embedding.

The primary metric is paired daily `RankIC(E2) - RankIC(E1)`. `E3 - E2` is secondary and cannot
rescue a failed primary gate.

## LLM boundary

An LLM may translate the frozen beta table into explanations such as “this event state weakens
short reversal and strengthens industry-relative return.” It does not read raw market tables,
change factor formulas, choose directions, or contribute to the numerical score.

## Run

```powershell
python -m factor_forge.cli ml event-factor-sensitivity-run `
  configs/ml/event_rankers/event_factor_sensitivity_oof_v1.yaml
```

Historical validation consumes the family's only validation peek. A passed gate permits
forward observation only; it is not authorization for live capital.
