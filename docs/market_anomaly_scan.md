# Latest market anomaly scan

The market anomaly scan is a discovery-only bundle that dynamically discovers every frozen event template matching `configs/radar/*_v1.yaml` (excluding the batch config) and every frozen relationship-drift template matching `configs/drift/*_v1.yaml`.

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

## Automatic synchronization and freshness

For an unpinned latest scan, the CLI now performs this gate before loading templates:

```text
query exchange trading calendar
  -> determine the latest completed/data-ready trading day
  -> resolve the latest complete full-history version
  -> fetch missing open dates as incremental versions
  -> merge each increment into a new immutable complete version
  -> audit last-day row, tradeable, liquid, and required-field coverage
  -> run the anomaly scan only when freshness_status=CURRENT
```

The default data-ready cutoff is `18:00 Asia/Shanghai`. Before that time on an open day, the expected date remains the previous open day. This avoids treating an incomplete same-day snapshot as end-of-day data.

`latest` now means the most recently published quality-passing **complete** version. A newer incremental version is never selected by research runners. Use `latest_any` only for ingestion diagnostics.

To diagnose without fetching, use `--no-sync`; the freshness check still runs and blocks stale data. A pinned `--data-version <id>` or historical `--as-of` is marked `PINNED` and never triggers network synchronization.

Every batch summary records:

```text
expected_latest_trade_date
data_end_date
freshness status and failures
resolved complete data_version
whether synchronization occurred
incremental versions consumed
last-day coverage metrics
```

## Contract boundaries

- Event triggers use only information available at or before the event date.
- `ObservationCard` rejects forward-return, target, IC, Sharpe, and other future-label fields.
- Feature-to-return drift uses only mature labels and records its effective date.
- A quality-gate failure is retained for audit but excluded from highlights.
- A detected drift means a monitored relationship changed; it is not a profitable direction or trading signal.
- A stale, ahead-of-ready-cutoff, or incomplete last day blocks the scan.
- The scan never launches Event Study, factor generation, backtesting, or promotion automatically.

## Skill

Use `$scan-market-anomalies` to run the fixed bundle and summarize the structured report. The Skill reads `scan_summary.json` and `report.md`, not raw K-line rows or large event/relationship files.

For installation, commands, freshness semantics, output fields, interpretation, and troubleshooting, see [scan_market_anomalies_skill_guide.md](scan_market_anomalies_skill_guide.md).

## Performance

The first run on a new data version computes all PIT rolling features and can take several minutes. Repeating the same frozen configuration, data version, and as-of date is cached. The next performance step is a shared PIT feature cache and single panel load; threshold relaxation is not an acceptable optimization.
