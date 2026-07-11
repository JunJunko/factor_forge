# Research workflows

## Contents

1. Idea route
2. Radar route
3. Promotion checklist
4. Scheduling policy

## Idea route

### 1. Frame the question

Record:

- observed phenomenon;
- falsifiable thesis;
- expected horizon;
- behavior family;
- expected direction only if specified before viewing validation results.

Create an Idea and activate it. Add competing explanations such as ordinary reversal, liquidity, industry movement, size, or known Regime.

### 2. Map hypotheses to Features

Use three buckets, at most eight measurements total:

| Bucket | Purpose |
|---|---|
| Descriptor | Measure how strongly the phenomenon occurred |
| Discriminator | Separate preferred and competing explanations |
| Context/control | Detect known exposure or conditional behavior |

For every Feature record: semantic meaning, formula/implementation reference, PIT availability, lookback, missing behavior, and whether larger values have a hypothesized direction.

### 3. Decide execution path

- DSL-expressible single score → Factor YAML.
- Several existing Factors with fixed combination → factor-combination YAML.
- Nonlinear incremental test with registered feature builder → ML config.
- Missing operator/field → stop and implement the contract before research execution.

### 4. Run minimal experiments

Use one primary metric. A recommended progression is:

```text
M0 known baseline
M1 baseline + descriptor
M2 + discriminator
M3 + context/control
```

Do not generate a Cartesian grid. Register every run, including invalid and failed runs.

### 5. Decide

- `reject`: no independent information or invalid data.
- `revise_one_hypothesis`: one identified mechanism or matching issue needs a single revision.
- `observe_forward`: validation passed; freeze direction/config and begin forward evidence.
- `promote_candidate`: use only after required forward/paper evidence, costs, and trading checks.

## Radar route

### 1. Daily scan

Run the two frozen templates after the latest immutable data version has been published. Scanning is label-free and can be automated safely.

### 2. Triage ObservationCards

Check:

- temporal audit passed;
- event count and coverage are adequate;
- event concentration is not dominated by one stock/industry;
- recent event rate differs meaningfully from history;
- definition is not a duplicate;
- the event is not explained by data errors, corporate actions, limit events, or stale data.

Frequency change only determines research priority.

### 3. Run matched Event Study

Create a config referencing the immutable Observation directory. Do not edit the card. Validate the config, then run it once under the validation budget.

Interpret in this order:

1. E0 frozen artifact and label-maturity audit;
2. matched coverage;
3. full-control SMD balance;
4. pre-registered 5D daily-equal-weight effect;
5. FDR and direction consistency;
6. Severity and Regime diagnostics.

Stop before step 4 if balance fails.

### 4. Convert supported relationship to Features

Translate the mechanism, not the backtest winner. Example:

```text
Observation: large price drop with weak volume confirmation
Potential continuous Features:
  return distance from normal
  volume conditional residual given return
  intraday close location
  liquidity/context controls
```

Do not default to `event_flag`. Preserve baseline controls and test incremental information.

## Promotion checklist

Require all relevant checks:

- source data and Feature PIT-safe;
- formula expressible and lookback inferred;
- no validation budget violation;
- no unmatched/common-support failure;
- independent increment beyond known exposures;
- OOS/forward evidence not used to regenerate the same hypothesis;
- T+1 execution, costs, concentration, turnover, and capacity evaluated;
- immutable configs and artifacts linked to the Trial.

## Scheduling policy

Recommended cadence:

```text
daily after data publication: Radar scan only
weekly: human/Skill triage and duplicate review
on explicit selection: one Phase 3 Event Study
after observe_forward: forward monitor
```

Do not automatically run Event Studies or generate Factors for every daily Observation. That creates multiple-testing explosion.

On Windows, schedule the deterministic script rather than a model invocation. Example action:

```text
python C:\Users\Junko\.codex\skills\factor-forge-alpha-research\scripts\run_radar_cycle.py --workspace D:\pyworkspace\factor_forge --data-version latest
```

Creating or modifying a Windows Scheduled Task is an external state change. Show the command and obtain explicit user confirmation before executing it.
