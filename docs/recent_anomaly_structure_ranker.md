# Recent anomaly structure ranker

This ML research family asks whether the market's latest frozen anomaly events contain a
point-in-time, recently effective conditional structure. It is distinct from the prior
single-template Event Episode study and from the pooled full-market Mamba pilot.

```text
seven quality-passing frozen Radar templates
  -> per-template anchored event Episodes
  -> T+1-open to T+6-open 5D universe-excess target
  -> label_available_date = event date + 6 trading days
  -> PIT template efficacy and conditional factor-IC state
  -> causal 60-day selective-state embedding
  -> rolling 126D train / 40D validation / 20D OOS folds
  -> static LightGBM versus adaptive LightGBM
```

## Frozen hypotheses

- Preferred: recent matured template efficacy plus causal pre-event state improves daily
  cross-sectional event ranking beyond the static model.
- Template-mix null: template identity, frequency, market context, prior return, volatility,
  liquidity, size and industry explain the apparent effect.
- Recency-variance null: recent samples are too sparse, so adaptation raises estimation
  variance instead of improving OOS ranking.

## Measurements

The research plan groups the inputs into eight measurements: event severity, template
identity, raw price/volume state, Mamba sequence state, 20/60/120-day mature event efficacy,
conditional factor IC, direction stability, and known exposure/context controls. These are
implemented in the strict ML feature builder; no Factor DSL or ObservationCard is modified.

The event source is the immutable 252-trading-day history already stored in each selected
Observation artifact; the runner does not rescan a longer full-market history. For every
signal date, recent efficacy uses only event targets whose
`label_available_date <= signal_date`. Today's event targets are absent and today's rows are
never fitting rows. Multiple template triggers for one stock are retained during conditional
modeling, but OOS daily IC and live ranks aggregate back to one stock-date.

## Model comparison and gate

- Static model: severity + raw PIT features + frozen template identity.
- Adaptive model: static inputs + Mamba embedding + PIT recent efficacy/factor state.
- Primary metric: paired daily `RankIC(adaptive) - RankIC(static)` on identical rolling OOS
  stock-dates.
- Gate: Newey-West t >= 2.0, positive-day ratio > 0.50, and at least 20 OOS dates.

Conditional weights are exported from OOS LightGBM SHAP contributions. They are diagnostics
of what the model used, not independently validated Factors or Alpha.

## Run

```powershell
python -m factor_forge.cli ml recent-anomaly-structure-run `
  configs/ml/event_rankers/recent_anomaly_structure_v1.yaml
```

This run consumes the family's single historical validation peek. A failed gate must not be
used to retune the same OOS folds. A passed gate permits forward observation only.
