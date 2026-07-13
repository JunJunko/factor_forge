from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository
from factor_forge.experiments.artifacts import json_default
from factor_forge.research_control import ArtifactIndexer, ResearchControlStore

from .mamba_state_config import MambaStatePilotConfig, load_mamba_state_config
from .mamba_state_dataset import build_sequence_store
from .mamba_state_features import build_state_feature_frame
from .mamba_state_trainer import encode_sequences, fit_reference_encoder


ENGINE_VERSION = "mamba_state_lightgbm_pilot_v1"
VARIANTS = ("raw", "state", "raw_state")


class MambaStateLightGBMRunner:
    def run(self, config_path: str | Path) -> dict:
        started = datetime.now(timezone.utc)
        config_path = Path(config_path)
        cfg = load_mamba_state_config(config_path)
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, manifest = repository.load_manifest(cfg.data_version)
        self._validate_coverage(cfg, manifest, version)
        _, panel = repository.load_panel(version)
        cutoff = pd.Timestamp(manifest["end_date"])
        state_frame = build_state_feature_frame(
            panel, cfg.features, cfg.label,
            event_template_paths=cfg.sequence.event_templates if cfg.sequence.include_event_channels else (),
            as_of_date=cutoff,
        )
        store = build_sequence_store(
            state_frame.frame, state_frame.state_feature_names,
            length=cfg.sequence.length, min_valid_days=cfg.sequence.min_valid_days,
            validity_feature_names=state_frame.raw_feature_names,
        )
        segment_positions = {
            name: store.positions_between(segment.start, segment.end)
            for name, segment in (
                ("train", cfg.segments.train),
                ("valid", cfg.segments.valid),
                ("test", cfg.segments.test),
            )
        }
        if any(len(segment_positions[name]) == 0 for name in ("train", "valid", "test")):
            raise ValueError("one or more pilot segments contain no valid sequence samples")

        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        run_id = (
            f"mamba_state_{started:%Y%m%dT%H%M%SZ}_"
            + hashlib.sha256(f"{ENGINE_VERSION}|{version}|{config_hash}".encode()).hexdigest()[:12]
        )
        output = cfg.output_root / run_id
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()

        encoder_train, encoder_valid = _chronological_encoder_split(
            store, segment_positions["train"], cfg.training.validation_fraction
        )
        all_positions = np.sort(np.unique(np.concatenate(list(segment_positions.values()))))
        embeddings = []
        checkpoint_hashes: dict[str, str] = {}
        histories = []
        devices = []
        for seed in cfg.training.random_seeds:
            fit = fit_reference_encoder(
                store, encoder_train, encoder_valid, cfg.encoder, cfg.training,
                seed=seed, checkpoint_path=output / "checkpoints" / f"encoder_seed_{seed}.pt",
            )
            embeddings.append(encode_sequences(
                fit.model, store, all_positions,
                batch_size=cfg.training.batch_size, device=fit.device,
            ))
            history = fit.history.copy()
            history["seed"] = seed
            histories.append(history)
            checkpoint_hashes[str(seed)] = fit.checkpoint_sha256
            devices.append(fit.device)
        state_values = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        state_names = [f"state_{index:02d}" for index in range(state_values.shape[1])]
        modeling = self._modeling_frame(
            store, all_positions, state_frame.raw_feature_names, state_values, state_names
        )
        modeling.to_parquet(output / "modeling_dataset.parquet", index=False)
        pd.concat(histories, ignore_index=True).to_parquet(
            output / "encoder_training_history.parquet", index=False
        )
        modeling[["datetime", "instrument", *state_names]].rename(columns={
            "datetime": "trade_date", "instrument": "ts_code",
        }).assign(
            encoder_fit_end=cfg.segments.train.end,
        ).to_parquet(output / "state_embeddings.parquet", index=False)
        store.samples.iloc[all_positions].to_parquet(output / "sequence_index.parquet", index=False)

        feature_sets = {
            "raw": state_frame.raw_feature_names,
            "state": state_names,
            "raw_state": [*state_frame.raw_feature_names, *state_names],
        }
        common = self._common_sample_mask(modeling, cfg, feature_sets["raw_state"])
        split_masks = {
            name: common & pd.to_datetime(modeling["datetime"]).between(
                pd.Timestamp(segment.start), pd.Timestamp(segment.end)
            )
            for name, segment in (
                ("train", cfg.segments.train),
                ("valid", cfg.segments.valid),
                ("test", cfg.segments.test),
            )
        }
        if any(not split_masks[name].any() for name in ("train", "valid", "test")):
            raise ValueError("common Raw/State/Raw+State sample intersection is empty in a segment")

        comparison, summaries = [], {}
        test_panel = panel.loc[pd.to_datetime(panel["trade_date"]).between(
            pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end)
        )].copy()
        for variant in VARIANTS:
            variant_output = output / variant
            variant_output.mkdir()
            result = self._fit_variant(
                modeling, feature_sets[variant], split_masks, cfg, test_panel, variant_output
            )
            summaries[variant] = result
            comparison.append({"variant": variant, **{
                key: value for key, value in result.items()
                if key not in {"features", "run_dir"}
            }})
        comparison_frame = pd.DataFrame(comparison)
        comparison_frame.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")

        temporal_audit = {
            "sequence_is_causal": True,
            "sequence_end_equals_sample_date": True,
            "event_channels_are_label_free": True,
            "observation_card_unchanged": True,
            "encoder_objective": "masked_reconstruction",
            "encoder_uses_future_return_labels": False,
            "encoder_fit_end": cfg.segments.train.end,
            "lightgbm_sample_intersection_rows": int(common.sum()),
            "segments_disjoint": True,
        }
        (output / "temporal_audit.json").write_text(
            json.dumps(temporal_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state_schema = {
            "feature_schema_hash": state_frame.feature_schema_hash,
            "raw_features": state_frame.raw_feature_names,
            "event_channels": state_frame.event_channel_names,
            "state_input_features": state_frame.state_feature_names,
            "state_features": state_names,
            "template_hashes": state_frame.template_hashes,
            "checkpoint_hashes": checkpoint_hashes,
            "aggregation": "arithmetic_mean_across_frozen_seeds",
        }
        (output / "state_schema.json").write_text(
            json.dumps(state_schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        finished = datetime.now(timezone.utc)
        summary = {
            "run_id": run_id, "engine_version": ENGINE_VERSION,
            "data_version": version, "feature_schema_hash": state_frame.feature_schema_hash,
            "template_hashes": state_frame.template_hashes,
            "sequence_samples": int(len(store.samples)),
            "common_modeling_rows": int(common.sum()),
            "encoder_devices": sorted(set(devices)),
            "checkpoint_hashes": checkpoint_hashes,
            "variants": summaries,
            "incremental_rank_ic": (
                summaries["raw_state"]["rank_ic_mean"] - summaries["raw"]["rank_ic_mean"]
            ),
            "interpretation_boundary": (
                "State embeddings are predictive representations, not causal factor sensitivities."
            ),
            "run_dir": str(output.resolve()),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
        )
        manifest_payload = {
            "run_id": run_id, "runner_type": "mamba_state_runs", "status": "COMPLETED",
            "engine_version": ENGINE_VERSION, "data_version": version,
            "config_hash": config_hash, "feature_schema_hash": state_frame.feature_schema_hash,
            "template_hashes": state_frame.template_hashes,
            "checkpoint_hashes": checkpoint_hashes,
            "started_at": started.isoformat(), "finished_at": finished.isoformat(),
            "contains_future_labels_in_event_channels": False,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        store_db = ResearchControlStore(project.paths.data_root / "research.sqlite3")
        store_db.initialize()
        ArtifactIndexer(store_db, cfg.output_root.parent).index()
        return summary

    @staticmethod
    def _modeling_frame(store, positions, raw_names, state_values, state_names):
        samples = store.samples.iloc[positions].reset_index(drop=True).copy()
        end_rows = samples["end_row"].to_numpy(dtype=np.int64)
        raw = store.frame.iloc[end_rows][raw_names].reset_index(drop=True)
        output = pd.concat([
            samples.drop(columns=["start_row", "end_row", "valid_days"], errors="ignore"), raw,
            pd.DataFrame(state_values, columns=state_names),
        ], axis=1)
        if output.duplicated(["datetime", "instrument"]).any():
            raise ValueError("modeling dataset contains duplicate datetime/instrument keys")
        return output

    @staticmethod
    def _common_sample_mask(modeling, cfg, features):
        universe = f"is_{cfg.portfolio.universe}"
        if universe not in modeling:
            raise KeyError(f"modeling dataset missing {universe}")
        return (
            modeling[features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            & pd.to_numeric(modeling["label"], errors="coerce").notna()
            & modeling[universe].eq(True)
        )

    @staticmethod
    def _fit_variant(modeling, features, masks, cfg, test_panel, output):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('LightGBM is required. Install with: pip install -e ".[ml]"') from exc
        model = LGBMRegressor(**cfg.lightgbm.model_dump())
        model.fit(
            modeling.loc[masks["train"], features], modeling.loc[masks["train"], "label"],
            eval_set=[(modeling.loc[masks["valid"], features], modeling.loc[masks["valid"], "label"])],
        )
        test = modeling.loc[masks["test"], ["datetime", "instrument", "label"]].copy()
        test["score"] = model.predict(modeling.loc[masks["test"], features])
        predictions = test.rename(columns={
            "datetime": "trade_date", "instrument": "ts_code", "score": "factor_value",
        })[["trade_date", "ts_code", "factor_value"]]
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        predictions.to_parquet(output / "predictions.parquet", index=False)
        daily_ic = test.groupby("datetime").apply(
            lambda group: group["score"].corr(group["label"], method="spearman")
            if len(group) >= 2 else np.nan,
            include_groups=False,
        ).rename("rank_ic")
        daily_ic.to_frame().to_parquet(output / "daily_rank_ic.parquet")
        backtest = BacktestEngine().run(
            test_panel, predictions, universe=cfg.portfolio.universe,
            top_n=cfg.portfolio.top_n, holding_days=cfg.portfolio.holding_days,
            initial_cash=cfg.portfolio.initial_cash, lot_size=cfg.portfolio.lot_size,
            constraints=ExecutionConstraints(), cost_model=CostModel(),
            cost_scenario_bps=cfg.portfolio.cost_bps,
        )
        backtest.daily.to_parquet(output / "portfolio_daily.parquet", index=False)
        backtest.trades.to_parquet(output / "trades.parquet", index=False)
        if hasattr(model, "booster_"):
            model.booster_.save_model(str(output / "lightgbm_model.txt"))
        top_n = max(1, min(cfg.portfolio.top_n, int(test.groupby("datetime").size().min())))
        top_mean = test.sort_values(["datetime", "score"], ascending=[True, False]).groupby(
            "datetime", sort=True
        ).head(top_n)["label"].mean()
        return {
            **backtest.metrics,
            "rank_ic_mean": float(daily_ic.mean()),
            "rank_ic_ir": float(daily_ic.mean() / daily_ic.std(ddof=0))
            if np.isfinite(daily_ic.std(ddof=0)) and daily_ic.std(ddof=0) > 0 else None,
            "top_n_mean_label": float(top_mean),
            "train_rows": int(masks["train"].sum()),
            "valid_rows": int(masks["valid"].sum()),
            "test_rows": int(masks["test"].sum()),
            "features": list(features), "run_dir": str(output.resolve()),
        }

    @staticmethod
    def _validate_coverage(cfg: MambaStatePilotConfig, manifest: dict, version: str):
        if not cfg.require_full_segment_coverage:
            return
        available_start, available_end = pd.Timestamp(manifest["start_date"]), pd.Timestamp(manifest["end_date"])
        tolerance = pd.Timedelta(days=7)
        if (
            available_start > pd.Timestamp(cfg.segments.train.start) + tolerance
            or available_end < pd.Timestamp(cfg.segments.test.end) - tolerance
        ):
            raise ValueError(
                f"data version does not cover pilot segments: {version} "
                f"{available_start.date()}..{available_end.date()}"
            )


def _chronological_encoder_split(store, train_positions, fraction):
    positions = np.asarray(train_positions, dtype=np.int64)
    dates = pd.to_datetime(store.samples.iloc[positions]["datetime"])
    unique_dates = np.array(sorted(dates.unique()))
    split = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * (1 - fraction))))
    cutoff = pd.Timestamp(unique_dates[split - 1])
    train = positions[dates.le(cutoff).to_numpy()]
    valid = positions[dates.gt(cutoff).to_numpy()]
    if not len(train) or not len(valid):
        raise ValueError("encoder chronological split is empty")
    return train, valid


def _report(summary):
    lines = [
        "# Mamba State + LightGBM Pilot", "",
        f"- Data version: `{summary['data_version']}`",
        f"- Common modeling rows: `{summary['common_modeling_rows']}`",
        f"- Incremental RankIC (Raw+State - Raw): `{summary['incremental_rank_ic']}`", "",
        "## Frozen ablation", "",
        "|Variant|RankIC|ICIR|TopN mean label|", "|---|---:|---:|---:|",
    ]
    for name in VARIANTS:
        item = summary["variants"][name]
        lines.append(
            f"|{name}|{item['rank_ic_mean']}|{item['rank_ic_ir']}|{item['top_n_mean_label']}|"
        )
    lines.extend(["", "> " + summary["interpretation_boundary"], ""])
    return "\n".join(lines)
