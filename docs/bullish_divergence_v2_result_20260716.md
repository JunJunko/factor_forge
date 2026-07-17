# Bullish Divergence v2 Corrected E1 Result — 2026-07-16

## Status

`STOP_OR_REVISE_BEFORE_CONDITIONAL_MATRIX`

This v2 experiment was designed after inspecting v1 history. Results through
2026-07-10 are diagnostic only and are not an unseen validation.

Artifacts:

- Feature run: `artifacts/bullish_divergence_v2_runs/bullish_divergence_v2_20260716T134747Z`
- Corrected event study: `artifacts/bullish_divergence_v2_event_studies/divergence_v2_e1_20260716T142458Z`

## Event construction

- Full-market coverage: 100% median and minimum daily stock coverage.
- Strict A-B separation: 20-60 trading days.
- Price geometry eligibility: second low between -0.25 and +1.00 ATR relative
  to the first low.
- Both troughs require a negative three-day return into the trough.
- Gap, reliability, lower-low depth and intervening rebound are excluded from
  the score.
- RSI, MACD/ATR and downside-velocity improvement are ranked separately before
  combination.
- Score quintiles are calculated inside the event pool on each signal date.
- Date contrasts use only dates on which both compared states are observed.

The full run produced:

- 404,423 geometry-candidate stock-days.
- 355,163 divergence-candidate stock-days.
- 49,260 geometry-placebo stock-days with no oscillator improvement.
- 83,099 deduplicated origin episodes.
- 70,343 first post-signal retest events within ten trading days.

## Origin event versus geometry placebo

The primary comparator required the same date, same SW L1 industry and a valid
20-60 day price geometry, but no positive RSI, MACD/ATR or downside-velocity
improvement.

- Mature origin events: 82,063.
- Matched events: 28,749.
- Match rate: 35.03%.
- Maximum absolute SMD: 0.8210.

The strict placebo pool was therefore too sparse and structurally different to
support a clean causal estimate. The largest residual imbalance was decline
into A, followed by lower-low depth, trough gap and intervening rebound.

Nevertheless, the intended positive score ordering did not appear:

| Event-pool score quintile | 10-day net industry excess | Matched excess |
|---|---:|---:|
| Q1 | -0.6128% | +0.1041% |
| Q2 | -0.5460% | +0.0860% |
| Q3 | -0.5509% | -0.0574% |
| Q4 | -0.5182% | -0.2653% |
| Q5 | -0.5625% | +0.0864% |

The complete-date `Q5-Q1` matched contrast was -0.1583%, with a 90% moving
date-block interval of [-0.5373%, +0.2355%].

Historical pre-B support also failed to show independent incremental value:

- `pre-B support - no support`: +0.0022%.
- 90% interval: [-0.2781%, +0.3428%].

The origin comparison is inconclusive as a causal estimate because its
coverage and balance Gates failed, but it provides no evidence that the
corrected score is positively monotonic.

## True post-signal retest

The retest clock begins after the origin signal. The first subsequent candle
that intersects the frozen B anchor becomes a new signal date, and its future
return starts at the next open.

- Mature retest events: 69,390.
- Matched events: 68,603.
- Match rate: 98.87%.
- Maximum absolute SMD: 0.1982.

This comparison passed the pre-registered coverage and balance thresholds.

Primary matched results:

- Overall retest effect: -0.3599%.
- 90% interval: [-0.5104%, -0.2234%].
- Q5-Q1: +0.0701%.
- 90% interval: [-0.1555%, +0.3119%].

The score therefore did not identify a statistically reliable better retest
subset.

Retest-state contrasts:

| State | Events | Net industry excess | Matched excess |
|---|---:|---:|---:|
| No reclaim | 28,702 | -0.5207% | -0.2413% |
| Reclaim without false break | 19,346 | -0.6253% | -0.3427% |
| False-break reclaim | 21,077 | -0.5536% | -0.2118% |

On same-date complete cases:

- `reclaim - no reclaim`: -0.2917%, 90% interval
  [-0.5129%, -0.0904%].
- `false-break reclaim - no reclaim`: -0.2073%, 90% interval
  [-0.4483%, +0.0006%].

Waiting for the first retest did not improve the event. A simple close reclaim
was significantly worse than a touch without reclaim on the inspected
history. The false-break variant was also non-positive and nearly
significantly negative.

## Decision

Do not proceed to the regime/concept/momentum conditional matrix using this
bottom-divergence event as the base long signal.

The next scientifically coherent alternatives are new hypotheses, not further
v2 tuning:

1. Test the event as a short-term avoidance/risk factor rather than a long
   Alpha.
2. Replace two-point oscillator differences with a path-based exhaustion or
   selling-pressure model.
3. Require independent evidence of demand arrival, rather than interpreting
   a price-level reclaim as demand.
4. Accumulate genuinely unseen forward data before authorizing any revised
   long-signal specification.

