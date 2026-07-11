---
name: factor-forge-alpha-research
description: Drive Factor Forge research from a natural-language market idea or a scheduled Radar anomaly scan into tracked hypotheses, measurable Features, declarative Factor YAML, staged evaluation/backtests, and an auditable decision. Use when working in the Factor Forge repository to capture an Idea, map Idea-to-Feature-to-Factor, run or schedule label-free anomaly scans, interpret ObservationCards, run matched Event Studies, generate factor/experiment configs, or decide whether a research branch may proceed to forward observation or backtesting.
---

# Factor Forge Alpha Research

Use Factor Forge as the deterministic engine. Use the model only to formulate hypotheses, choose existing measurements, and explain evidence.

## Start safely

1. Locate the Factor Forge root by finding `pyproject.toml`, `configs/contracts/`, and `src/factor_forge/`.
2. Read the repository `CLAUDE.md`, then read [references/contracts.md](references/contracts.md).
3. Run `python -m factor_forge.cli research init` if the research control database is absent.
4. Inspect `git status --short`. Preserve unrelated changes.
5. Choose exactly one route:
   - user supplied a market idea -> Idea route;
   - user requested anomaly discovery/monitoring or supplied an ObservationCard -> Radar route.

## Non-negotiable boundaries

- Never let an LLM read raw market tables for numerical discovery; use Factor Forge commands and compact JSON/CSV summaries.
- Never add forward returns, IC, Sharpe, targets, or labels to an ObservationCard or its event file.
- Never overwrite immutable data, Observation, Event Study, or experiment artifacts.
- Count every attempted validation, including failures and implementation-invalid runs.
- Do not convert an anomaly to a Factor merely because event frequency increased.
- Stop when the deterministic Gate returns `reject` or `revise_one_hypothesis`. Only `observe_forward` permits promotion work.
- Use sealed data only through the explicit approval/audit command.
- Use `apply_patch` for repository edits. Validate every generated YAML before running it.

## Idea route

Follow [references/workflows.md](references/workflows.md#idea-route).

1. Capture one falsifiable Idea with `research idea-create`; activate it.
2. Add 2-4 competing hypotheses. Do not record only the preferred story.
3. Propose at most eight measurements:
   - anomaly descriptors;
   - measurements that distinguish competing explanations;
   - known exposure/regime controls.
4. Search existing factor configs, ML feature code, and operator contracts before proposing anything new.
5. Pass the expression Gate:
   - existing DSL can express it -> put reusable intermediates in `calculation.features`;
   - DSL cannot express it -> do not invent syntax; choose an audited approximation, extend the DSL with tests, or route to the ML feature pipeline.
6. Create one ExperimentPlan with one primary metric and no more than five trials.
7. Generate Factor and Experiment YAML from repository templates. Keep data source, horizons, TopN, and costs out of Factor YAML.
8. Validate, run, register the external run as a trial, and save exactly one decision.

## Radar route

Follow [references/workflows.md](references/workflows.md#radar-route).

1. For a one-off scan, use `radar scan` with a frozen template.
2. For a daily cycle, run:

   ```powershell
   python <skill-dir>/scripts/run_radar_cycle.py --workspace <factor-forge-root> --data-version latest
   ```

3. Inspect each ObservationCard for identity, event count, recent/history frequency, concentration, and temporal audit. Do not inspect forward returns at this stage.
4. Triage before Event Study. Skip duplicates and data-quality artifacts.
5. Create a Phase 3 config referencing the frozen Observation directory; run `event-study validate` then `event-study run`.
6. Require adequate maturity, match rate, covariate balance, and the pre-registered full-controls/5D Gate.
7. If the Gate says `observe_forward`, map the supported continuous relationship--not merely the binary event flag--to Features, then continue through the Idea route at the expression Gate.

## Feature to Factor rules

- Treat a Feature as a measurable variable, not automatically as a trading direction.
- In current Factor Forge, executable DSL Features live under `calculation.features`; there is no separate executable Feature Registry.
- Prefer continuous measurements such as relation residual, distance, severity, recovery strength, or conditional percentile over a brittle event boolean.
- Define a Factor only when the Feature combination produces one sortable score with a stated direction and expected shape.
- For an event-derived Factor, compare against the baseline exposure that defines the event. The Factor must add information beyond prior return, volatility, liquidity, size, industry, and known Regime effects.
- Use LightGBM only for incremental comparison: baseline model versus baseline plus new Features.

## Finish every run

Report:

- Idea/Observation ID;
- Feature measurements selected and why;
- expression route used: DSL, DSL extension, or ML;
- Factor and Experiment config paths;
- Trial ID, data role, data version, and artifact path;
- primary metric and deterministic Gate;
- the single saved next action;
- remaining Idea and Family budgets.

Do not claim Alpha before forward evidence.
