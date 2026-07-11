# Factor Forge contracts

## Contents

1. Object mapping
2. Repository sources of truth
3. Commands
4. Expression Gate
5. Artifact boundaries

## Object mapping

```text
Idea
  -> competing Hypotheses
  -> measurable Features
  -> one sortable Factor or ML score
  -> ExperimentPlan / Trial
  -> deterministic Gate / Decision
```

Radar adds a prior stage:

```text
Frozen label-free Observation
  -> matched Event Study
  -> supported or rejected relationship
  -> Feature proposal
```

An Observation is not a Factor. Event frequency and event severity are contemporaneous evidence, not future-return evidence.

## Repository sources of truth

Read these only as needed:

- `configs/contracts/operator_registry_v1.yaml`: allowed DSL operators and semantics.
- `configs/contracts/field_dictionary_v1.yaml`: valid fields and aliases.
- `configs/factors/_factor_template.yaml`: Factor YAML shape.
- `configs/experiments/_experiment_template.yaml`: staged experiment shape.
- `configs/radar/*.yaml`: frozen label-free anomaly templates.
- `configs/event_studies/*.yaml`: matched Event Study configs.
- `docs/research_control_phase0_1.md`: lineage and budget commands.
- `docs/radar_phase2.md`: Observation contract.
- `docs/event_study_phase3.md`: label, matching, inference, and Gate contract.

Search before adding:

```powershell
rg -n "<concept>|<field>|<operator>" configs src/factor_forge tests
```

## Commands

Initialize and inspect:

```powershell
python -m factor_forge.cli research init
python -m factor_forge.cli research index-artifacts --artifacts-root artifacts
python -m factor_forge.cli research artifact-summary
```

Idea lineage:

```powershell
python -m factor_forge.cli research idea-create --title "..." --thesis "..." --family-id family --target-horizon 5
python -m factor_forge.cli research idea-status IDEA_ID --status active
python -m factor_forge.cli research hypothesis-add IDEA_ID --statement "..."
python -m factor_forge.cli research plan-create IDEA_ID --name plan_v1 --primary-metric metric_name --hypothesis-id HYP_ID
python -m factor_forge.cli research trial-record PLAN_ID --data-role discovery --status success --external-run-id RUN_ID --artifact-path PATH
python -m factor_forge.cli research decision-save TRIAL_ID --action observe_forward --reason "..." --decided-by NAME
python -m factor_forge.cli research idea-show IDEA_ID
```

Factor evaluation:

```powershell
python -m factor_forge.cli factor validate configs/factors/<factor>.yaml
python -m factor_forge.cli experiment run configs/experiments/<experiment>.yaml
```

Radar and Event Study:

```powershell
python -m factor_forge.cli radar validate-template configs/radar/<template>.yaml
python -m factor_forge.cli radar scan --template configs/radar/<template>.yaml --data-version latest --as-of YYYYMMDD
python -m factor_forge.cli event-study validate configs/event_studies/<study>.yaml
python -m factor_forge.cli event-study run configs/event_studies/<study>.yaml
```

## Expression Gate

Current executable Feature locations:

- Factor DSL: `calculation.features` and `calculation.formula`.
- Factor combination: component Factor YAML plus fixed preprocessing.
- ML: named Python feature builders and strict ML config.

Historical rolling percentile is used by Radar but is not automatically a Factor DSL operator. Before translating Radar evidence, inspect the operator registry. If an operator is missing:

1. Prefer a semantically defensible existing expression such as rolling robust Z/distance only when it measures the same hypothesis.
2. Otherwise implement a new operator, update the operator contract, static lookback inference, and temporal-consistency tests.
3. Route to ML only when the measurement belongs to an existing strict feature builder.

Never place pseudo-functions in YAML.

## Artifact boundaries

- `artifacts/radar_observations`: label-free and immutable.
- `artifacts/radar_event_studies`: labels allowed; never copied back to Observation.
- `artifacts/runs`: Factor evaluation/backtest.
- `data/research.sqlite3`: lineage, budgets, decisions, observation/event-study registry.

The primary Event Study metric is pre-registered `full_controls / 5D / daily equal-weight paired excess`. Severity and Regime tables are diagnostics only. Absolute SMD above the configured threshold blocks interpretation even when t-statistics look attractive.
