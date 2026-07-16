# Bullish Divergence E1/E1.1 Result — 2026-07-16

## Outcome

The pre-registered E1/E1.1 Gate failed. The current bullish-divergence definition and score must not proceed directly to regime/concept/momentum ML.

Artifacts:

- Feature run: `artifacts/bullish_divergence_runs/bullish_divergence_20260716T061045Z`
- Event study: `artifacts/bullish_divergence_event_studies/divergence_e1_20260716T063113Z`

## Sample and execution clock

- Signal interval: 2021-01-01 through 2026-07-10.
- Full-market daily feature coverage: median 100%, minimum 100%.
- Stock-day rows: 6,860,497 across 5,954 stocks.
- Raw event-candidate rows: 752,718.
- Ten-day deduplicated episodes: 125,401.
- Mature 10-day episodes: 122,597.
- Matched episodes: 120,853; match rate 98.58%.
- Matching: same date and SW L1 industry, three nearest controls, eight PIT controls.
- Primary label: T+1-open to T+11-open industry LOO excess; raw event summaries deduct 40 bps.

## Primary findings

### Divergence score

The divergence score was inversely related to future payoff rather than monotonically positive.

| Score group | 10-day net industry excess | Matched excess |
|---|---:|---:|
| Q1 | -0.3810% | -0.0371% |
| Q2 | -0.5194% | -0.1838% |
| Q3 | -0.5105% | -0.2273% |
| Q4 | -0.6225% | -0.3230% |
| Q5 | -0.6324% | -0.3849% |

`Q5 - Q1` matched excess was -0.3528%, with a 90% date-block-bootstrap interval of [-0.5470%, -0.1561%]. This rejects the intended positive score monotonicity on the inspected history.

### Touch/retest factor

| Touch state | Episodes | 10-day net industry excess | Matched excess |
|---|---:|---:|---:|
| U0 no touch | 81,252 | -0.5296% | -0.2337% |
| U1 touch, no reclaim | 446 | -0.1227% | -0.1288% |
| U2 touch and reclaim | 33,131 | -0.5131% | -0.1767% |
| U3 false break and reclaim | 7,282 | -0.5499% | -0.3072% |

`U2 - U0` was +0.0752%, but its 90% bootstrap interval was [-0.1122%, +0.2699%]. It was positive in 4 of 6 calendar years, then negative in 2025 and 2026. The current touch feature therefore has suggestive conditional information but no independently validated Alpha.

`U3 - U0` was only +0.0070%, with a 90% interval of [-0.2860%, +0.2859%], and was positive in only 2 of 6 years.

### Overall matched effect

The overall matched 10-day excess was -0.2212%, with a 90% bootstrap interval of [-0.3456%, -0.0976%]. This is negative before relying on portfolio-model selection.

## Balance-first audit

The full matcher passed the match-rate requirement but failed the maximum absolute SMD requirement:

- 60-day drawdown SMD: -0.3175.
- 20-day return SMD: -0.2737.
- All other controls had absolute SMD below 0.12.

The matched effect is therefore not used as a clean causal estimate. However, the raw score monotonicity, bootstrap contrasts, yearly attribution, and balance failure all point against promoting the current specification.

## Interpretation

The current score rewards the magnitude of the lower low. In practice, higher scores also select more severe recent price damage and deeper drawdowns. The result is consistent with a falling-knife severity effect overwhelming oscillator improvement.

The scientifically justified revision is not to invert the inspected score. A new hypothesis should:

1. Move lower-low depth, 20-day return and 60-day drawdown from positive Alpha components into matching/risk controls.
2. Treat price geometry as an eligibility band rather than rewarding a deeper lower low.
3. Define the candidate Alpha as oscillator/sell-pressure improvement conditional on matched price damage.
4. Keep U2 touch-and-reclaim as a potential modifier, not a standalone entry rule.
5. Freeze the revision as a new version and authorize it only for shadow/forward observation because all history through 2026-07-10 has now been inspected.

## Decision

`STOP_OR_REVISE_BEFORE_ML`

Do not fit the planned regime/concept/momentum ML on this event definition. Doing so would allow a flexible model to hide the failure of the underlying event and would convert regime discovery into post-hoc rescue.
