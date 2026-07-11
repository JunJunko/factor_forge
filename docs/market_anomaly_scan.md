# Latest market anomaly scan

The market anomaly scan is a discovery-only bundle over eight frozen stock-event templates and two frozen market relationship-drift templates.

## Run

```powershell
python -m factor_forge.cli anomaly-scan latest `
  --config configs/radar/latest_market_scan_v1.yaml `
  --data-version latest
```

The command resolves `latest` to an immutable data version and writes:

```text
artifacts/market_anomaly_scans/<scan_id>/
  manifest.json
  report.md
  scan_summary.json
```

Each event template independently writes an immutable, label-free `ObservationCard` and `events.parquet`. Each drift template writes an independent `DriftCard` and relationship series. The batch report is only an index and ranking layer; it does not mutate source cards.

## Contract boundaries

- Event triggers use only information available at or before the event date.
- `ObservationCard` rejects forward-return, target, IC, Sharpe, and other future-label fields.
- Feature-to-return drift uses only mature labels and records its effective date.
- A quality-gate failure is retained for audit but excluded from highlights.
- A detected drift means a monitored relationship changed; it is not a profitable direction or trading signal.
- The scan never launches Event Study, factor generation, backtesting, or promotion automatically.

## Skill

Use `$scan-market-anomalies` to run the fixed bundle and summarize the structured report. The Skill reads `scan_summary.json` and `report.md`, not raw K-line rows or large event/relationship files.

## Performance

The first run on a new data version computes all PIT rolling features and can take several minutes. Repeating the same frozen configuration, data version, and as-of date is cached. The next performance step is a shared PIT feature cache and single panel load; threshold relaxation is not an acceptable optimization.
