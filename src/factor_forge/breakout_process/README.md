# Frozen-box breakout process

This package is intentionally independent from `factor_forge.factors`, the DSL, and the
backtest engine. It accepts an OHLCV `DataFrame` and returns three point-in-time tables:

- `boxes`: one row per frozen box and its terminal lifecycle state;
- `daily_features`: end-of-day pre-breakout snapshots while a box is active;
- `events`: one immutable snapshot per confirmed close breakout.

## Minimal use

```python
from factor_forge.breakout_process import BreakoutProcessEngine

result = BreakoutProcessEngine().run(panel)
boxes = result.boxes
daily_features = result.daily_features
events = result.events
```

The default input columns are `trade_date`, `ts_code`, `adj_open`, `adj_high`,
`adj_low`, `adj_close`, and `volume_shares`. Pass a `ColumnMap` to use other names.

## Timing contract

At date `t`, a new box is built only from rows ending at `t-1`. Its upper, lower, and
ATR scale remain immutable for the box lifetime. A breakout event at `t` contains:

- setup factors computed from the box source window;
- pre-breakout factors whose `pre_window_end` is `t-1`;
- breakout factors available after the close at `t`.

Consequently, an event snapshot is suitable for a decision no earlier than after the
close at `t`; execution timing remains the caller's responsibility.

## State contract

The lifecycle has only `active`, `triggered`, and `closed` states. Approach and
acceleration are continuous factors, not lifecycle states. An active box closes because
of `breakout`, `downside_failure`, or `expired`; it is never silently rebased.

The numerical defaults in `BreakoutConfig` are research defaults, not calibrated trading
parameters. In particular, box qualification thresholds should be validated on the
intended universe before use.
