# Single-template Event Episode ranker

The Event Episode pipeline asks whether one frozen short-horizon anomaly contains repeatable conditional behavior. It does not train on ordinary stock-days and does not mix event mechanisms.

```text
frozen PIT event template
  -> raw triggers
  -> anchored five-trading-day Episode deduplication
  -> mature same-date/same-industry full-control matching
  -> matched excess labels
  -> pre-event 60-day selective-state representation
  -> E0 severity / E1 raw / E2 state / E3 raw+state
  -> same-template live ranking
```

## Run

```powershell
python -m factor_forge.cli ml event-episode-run `
  configs/ml/event_rankers/price_drop_without_volume_episode_v1.yaml
```

The first contract is frozen to `price_drop_without_volume_confirmation_v1`. Historical events use a 504-trading-day window, labels use T+1-open execution, and the primary target is the five-day event return minus the mean return of mature nearest controls matched on date, industry, prior return, volatility, liquidity, and size.

All raw same-template triggers are excluded from the control pool, including triggers suppressed by Episode deduplication. Only anchor events enter supervised training. The live scan date is scored after all models and encoder checkpoints are frozen and contributes zero fitting rows.

## Interpretation

The primary Mamba increment is the paired daily difference:

```text
daily RankIC(E3 Raw+State) - daily RankIC(E1 Raw)
```

Its Newey-West t-value must reach the frozen gate before the result can move to forward observation. A positive unpaired mean, State-only result, or current live ranking is insufficient. Graph and explicit feature gates remain out of scope.
