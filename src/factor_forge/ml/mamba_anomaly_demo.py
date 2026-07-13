from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.experiments.artifacts import json_default

from .config import FeatureConfig, LabelConfig
from .dataset import build_dataset
from .mamba_state_config import EncoderConfig, EncoderTrainingConfig
from .mamba_state_dataset import build_sequence_store
from .mamba_state_runner import MambaStateLightGBMRunner
from .mamba_state_trainer import encode_sequences, fit_reference_encoder


DEMO_VERSION = "today_anomaly_state_demo_v1"


class TodayAnomalyStateDemoRunner:
    """Bounded CPU demo driven by a frozen latest-market anomaly scan."""

    def run(
        self,
        scan_summary_path: str | Path,
        *,
        project_config: str | Path = "configs/project.yaml",
        output_root: str | Path = "artifacts/mamba_anomaly_demos",
    ) -> dict:
        started = datetime.now(timezone.utc)
        scan_summary_path = Path(scan_summary_path)
        scan = json.loads(scan_summary_path.read_text(encoding="utf-8"))
        selected = [
            row for row in scan["events"]
            if row["quality_gate_passed"] and row["scan_date_event_count"] > 0
        ]
        if not selected:
            raise ValueError("scan contains no quality-passing current event templates")
        scan_date = pd.Timestamp(scan["as_of_date"])
        project = load_project(project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, _ = repository.load_manifest(scan["data_version"])
        if version != scan["data_version"]:
            raise ValueError("scan data version did not resolve immutably")

        params = {
            "sequence_length": 60, "min_valid_days": 40,
            "encoder": {"d_model": 16, "d_state": 8, "layers": 1, "embedding_dim": 8},
            "epochs": 5, "seed": 17, "lightgbm_estimators": 200,
            "selected_templates": [row["template_id"] for row in selected],
        }
        identity = hashlib.sha256(
            json.dumps({
                "demo_version": DEMO_VERSION, "scan_id": scan["scan_id"], "params": params,
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        run_id = f"mamba_anomaly_demo_{identity}"
        output = Path(output_root) / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            return {**cached, "cached": True}

        panel_path = project.paths.data_root / "versions" / version / "curated" / "stock_daily_panel.parquet"
        start = scan_date - pd.Timedelta(days=520)
        panel_columns = [
            "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny",
            "is_tradeable", "is_liquid", "log_total_mv", "log_circ_mv", "turnover_rate",
        ]
        panel = pd.read_parquet(
            panel_path, columns=panel_columns,
            filters=[("trade_date", ">=", start), ("trade_date", "<=", scan_date)],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        feature_cfg = FeatureConfig(windows=[5, 10, 20, 60], winsor_quantile=0.01, cross_sectional_zscore=True)
        label_cfg = LabelConfig(horizon=5, price="adj_open", excess_over_universe=True)
        flat, raw_features = build_dataset(panel, feature_cfg, label_cfg)
        event_names, severity_names = [], []
        candidate_templates: dict[str, set[str]] = {}
        for row in selected:
            template_id = row["template_id"]
            events = pd.read_parquet(row["events_path"], columns=["trade_date", "ts_code", "severity"])
            events["trade_date"] = pd.to_datetime(events["trade_date"])
            events = events.loc[events["trade_date"].le(scan_date)].copy()
            event_name, severity_name = f"event__{template_id}", f"severity__{template_id}"
            events[event_name] = 1.0
            events = events.rename(columns={"severity": severity_name})
            flat = flat.merge(
                events.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})[
                    ["datetime", "instrument", event_name, severity_name]
                ],
                on=["datetime", "instrument"], how="left", validate="one_to_one",
            )
            flat[event_name] = flat[event_name].fillna(0.0).astype("float32")
            flat[severity_name] = pd.to_numeric(flat[severity_name], errors="coerce").fillna(0.0).astype("float32")
            event_names.append(event_name)
            severity_names.append(severity_name)
            current_codes = set(events.loc[events["trade_date"].eq(scan_date), "ts_code"].astype(str))
            for code in current_codes:
                candidate_templates.setdefault(code, set()).add(template_id)
        state_inputs = [*raw_features, *event_names, *severity_names]
        flat = flat.sort_values(["instrument", "datetime"], kind="mergesort").reset_index(drop=True)
        store = build_sequence_store(
            flat, state_inputs, length=params["sequence_length"],
            min_valid_days=params["min_valid_days"], validity_feature_names=raw_features,
        )
        end_rows = store.samples["end_row"].to_numpy(dtype=np.int64)
        endpoint_events = store.frame.iloc[end_rows][event_names].sum(axis=1).to_numpy()
        event_positions = np.flatnonzero(endpoint_events > 0)
        event_dates = pd.to_datetime(store.samples.iloc[event_positions]["datetime"])
        mature = pd.to_numeric(store.samples.iloc[event_positions]["label"], errors="coerce").notna().to_numpy()
        historical_positions = event_positions[mature & event_dates.lt(scan_date).to_numpy()]
        live_positions = event_positions[event_dates.eq(scan_date).to_numpy()]
        if len(live_positions) == 0:
            raise ValueError("today's event candidates are missing from the sequence store")
        unique_dates = np.array(sorted(pd.to_datetime(store.samples.iloc[historical_positions]["datetime"]).unique()))
        if len(unique_dates) < 60:
            raise ValueError(f"demo needs at least 60 mature historical event dates, found {len(unique_dates)}")
        train_cut = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.60) - 1])
        valid_cut = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.80) - 1])
        hist_dates = pd.to_datetime(store.samples.iloc[historical_positions]["datetime"])
        train_positions = historical_positions[hist_dates.le(train_cut).to_numpy()]
        valid_positions = historical_positions[hist_dates.gt(train_cut).to_numpy() & hist_dates.le(valid_cut).to_numpy()]
        test_positions = historical_positions[hist_dates.gt(valid_cut).to_numpy()]

        encoder_cfg = EncoderConfig(
            d_model=16, d_state=8, layers=1, embedding_dim=8, dropout=0.10, mask_probability=0.20
        )
        training_cfg = EncoderTrainingConfig(
            epochs=5, batch_size=256, learning_rate=0.002, weight_decay=1e-4,
            patience=3, validation_fraction=0.15, max_train_samples=50_000,
            max_valid_samples=20_000, random_seeds=[17], device="cpu", num_workers=0,
        )
        output.mkdir(parents=True, exist_ok=False)
        fit = fit_reference_encoder(
            store, train_positions, valid_positions, encoder_cfg, training_cfg,
            seed=17, checkpoint_path=output / "encoder_seed_17.pt",
        )
        all_positions = np.sort(np.unique(np.concatenate([
            train_positions, valid_positions, test_positions, live_positions,
        ])))
        state_values = encode_sequences(
            fit.model, store, all_positions, batch_size=training_cfg.batch_size, device=fit.device,
        )
        state_names = [f"state_{index:02d}" for index in range(state_values.shape[1])]
        modeling = MambaStateLightGBMRunner._modeling_frame(
            store, all_positions, raw_features, state_values, state_names
        )
        position_to_row = {int(position): index for index, position in enumerate(all_positions)}
        row_sets = {
            "train": np.array([position_to_row[int(position)] for position in train_positions]),
            "valid": np.array([position_to_row[int(position)] for position in valid_positions]),
            "test": np.array([position_to_row[int(position)] for position in test_positions]),
            "live": np.array([position_to_row[int(position)] for position in live_positions]),
        }
        common_features = [*raw_features, *state_names]
        usable = (
            modeling[common_features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            & modeling["is_liquid"].eq(True)
        )
        for name in ("train", "valid", "test"):
            rows = row_sets[name]
            row_sets[name] = rows[usable.iloc[rows].to_numpy() & pd.to_numeric(
                modeling.iloc[rows]["label"], errors="coerce"
            ).notna().to_numpy()]
        row_sets["live"] = row_sets["live"][usable.iloc[row_sets["live"]].to_numpy()]
        if any(len(row_sets[name]) == 0 for name in row_sets):
            raise ValueError("a demo split is empty after applying the common sample intersection")

        feature_sets = {
            "raw": raw_features, "state": state_names, "raw_state": [*raw_features, *state_names],
        }
        comparisons, live = [], modeling.iloc[row_sets["live"]][["datetime", "instrument"]].copy()
        for variant, features in feature_sets.items():
            model = _fit_lightgbm(modeling, row_sets, features)
            test = modeling.iloc[row_sets["test"]]
            score = model.predict(test[features])
            daily = pd.DataFrame({
                "date": test["datetime"].to_numpy(), "score": score,
                "label": test["label"].to_numpy(),
            }).groupby("date").apply(
                lambda group: group["score"].corr(group["label"], method="spearman"),
                include_groups=False,
            )
            live[f"score_{variant}"] = model.predict(modeling.iloc[row_sets["live"]][features])
            live[f"rank_{variant}"] = live[f"score_{variant}"].rank(ascending=False, method="min").astype(int)
            comparisons.append({
                "variant": variant, "test_rows": len(test), "test_dates": int(test["datetime"].nunique()),
                "rank_ic_mean": float(daily.mean()),
                "rank_ic_ir": float(daily.mean() / daily.std(ddof=0))
                if np.isfinite(daily.std(ddof=0)) and daily.std(ddof=0) > 0 else None,
            })
        live["templates_triggered"] = live["instrument"].map(
            lambda code: ",".join(sorted(candidate_templates.get(str(code), set())))
        )
        live = live.sort_values("rank_raw_state").reset_index(drop=True)
        comparison = pd.DataFrame(comparisons)
        comparison.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
        live.to_parquet(output / "live_ranking.parquet", index=False)
        live.to_csv(output / "live_ranking.csv", index=False, encoding="utf-8-sig")
        fit.history.to_parquet(output / "encoder_training_history.parquet", index=False)
        modeling[["datetime", "instrument", *state_names]].to_parquet(
            output / "event_state_embeddings.parquet", index=False
        )
        summary = {
            "run_id": run_id, "demo_version": DEMO_VERSION, "scan_id": scan["scan_id"],
            "data_version": version, "as_of_date": scan["as_of_date"],
            "selected_templates": params["selected_templates"],
            "historical_event_rows": int(len(historical_positions)),
            "train_rows": int(len(row_sets["train"])), "valid_rows": int(len(row_sets["valid"])),
            "test_rows": int(len(row_sets["test"])), "live_candidates": int(len(live)),
            "date_splits": {
                "train_end": train_cut.strftime("%Y-%m-%d"),
                "valid_end": valid_cut.strftime("%Y-%m-%d"),
                "test_end": pd.Timestamp(unique_dates[-1]).strftime("%Y-%m-%d"),
            },
            "comparison": comparisons,
            "top_live": live.head(20).to_dict(orient="records"),
            "checkpoint_sha256": fit.checkpoint_sha256,
            "interpretation_boundary": (
                "CPU demo on one frozen discovery window; rankings are not validated Alpha or trading advice."
            ),
            "run_dir": str(output.resolve()), "cached": False,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
        )
        (output / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "runner_type": "mamba_anomaly_demos", "status": "COMPLETED",
            "data_version": version, "scan_id": scan["scan_id"],
            "checkpoint_sha256": fit.checkpoint_sha256,
            "contains_future_labels_in_event_inputs": False,
            "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        return summary


def _fit_lightgbm(modeling, rows, features):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        objective="regression", learning_rate=0.03, num_leaves=31, max_depth=-1,
        n_estimators=200, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        modeling.iloc[rows["train"]][features], modeling.iloc[rows["train"]]["label"],
        eval_set=[(modeling.iloc[rows["valid"]][features], modeling.iloc[rows["valid"]]["label"])],
    )
    return model


def _report(summary):
    lines = [
        "# Today Anomaly-Driven State Demo", "",
        f"- Scan: `{summary['scan_id']}` as of `{summary['as_of_date']}`",
        f"- Historical event rows: `{summary['historical_event_rows']}`",
        f"- Live candidates: `{summary['live_candidates']}`", "",
        "## Frozen comparison", "", "|Variant|Test rows|Dates|RankIC|ICIR|", "|---|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]:
        lines.append(
            f"|{row['variant']}|{row['test_rows']}|{row['test_dates']}|{row['rank_ic_mean']}|{row['rank_ic_ir']}|"
        )
    lines.extend(["", "## Current Raw+State ranking", ""])
    for row in summary["top_live"][:10]:
        lines.append(
            f"- `{row['instrument']}` rank={row['rank_raw_state']} "
            f"score={row['score_raw_state']:.6g} events={row['templates_triggered']}"
        )
    lines.extend(["", "> " + summary["interpretation_boundary"], ""])
    return "\n".join(lines)
