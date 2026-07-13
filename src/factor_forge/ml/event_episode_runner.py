from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.evaluation.l1 import _newey_west_stats
from factor_forge.experiments.artifacts import json_default
from factor_forge.radar.models import ObservationCard
from factor_forge.radar.templates import filter_required_fields, load_radar_template, required_trading_rows

from .event_episode_config import load_event_episode_config
from .event_episode_dataset import build_event_episode_dataset
from .mamba_state_runner import MambaStateLightGBMRunner
from .mamba_state_trainer import encode_sequences, fit_reference_encoder


ENGINE_VERSION = "event_episode_ranker_v2"
ARMS = ("e0_severity", "e1_raw", "e2_state", "e3_raw_state")


class EventEpisodeRankerRunner:
    def run(self, config_path: str | Path) -> dict:
        started = datetime.now(timezone.utc)
        config_path = Path(config_path)
        cfg = load_event_episode_config(config_path)
        template = load_radar_template(cfg.template)
        if template.id != "price_drop_without_volume_confirmation_v1":
            raise ValueError("first Event Episode pilot is frozen to price_drop_without_volume_confirmation_v1")
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, manifest = repository.load_manifest(cfg.data_version)
        as_of = pd.Timestamp(cfg.as_of_date or manifest["end_date"])
        card, source_events, source_manifest = _load_source_observation(cfg.source_observation_dir)
        if (
            card.definition.id != template.id
            or card.definition.definition_hash != template.definition_hash()
            or card.data_version != version
            or card.as_of_date != as_of.strftime("%Y-%m-%d")
        ):
            raise ValueError("source ObservationCard identity differs from Episode config")

        warmup = required_trading_rows(template) + cfg.episode.history_trading_days
        calendar_days = math.ceil(warmup * 7 / 5) + 120
        start = as_of - pd.Timedelta(days=calendar_days)
        panel_path = project.paths.data_root / "versions" / version / "curated" / "stock_daily_panel.parquet"
        columns = list(dict.fromkeys([
            "trade_date", "ts_code", "adj_open", "adj_high", "adj_low", "adj_close",
            "volume_shares", "amount_cny", "turnover_rate", "log_total_mv", "log_circ_mv",
            "industry_l1_code", "is_tradeable", "is_liquid",
            *filter_required_fields(template), *template.data.required_fields,
        ]))
        panel = pd.read_parquet(
            panel_path, columns=columns,
            filters=[("trade_date", ">=", start), ("trade_date", "<=", as_of)],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        dataset = build_event_episode_dataset(panel, template, cfg, as_of_date=as_of)
        _validate_live_source(dataset.live_events, source_events, as_of)

        primary = f"matched_excess_{cfg.primary_horizon}"
        mature = dataset.episodes.loc[
            dataset.episodes[primary].notna() & dataset.episodes["trade_date"].lt(as_of)
        ].copy()
        if len(mature) < cfg.minimum_matched_episodes:
            raise ValueError(
                f"matched mature episodes {len(mature)} < minimum {cfg.minimum_matched_episodes}"
            )
        split = _split_and_purge(mature, panel, cfg)
        train_positions = split.loc[split["split"].eq("train"), "sample_position"].to_numpy(int)
        valid_positions = split.loc[split["split"].eq("valid"), "sample_position"].to_numpy(int)
        test_positions = split.loc[split["split"].eq("test"), "sample_position"].to_numpy(int)
        live_positions = dataset.live_events["sample_position"].to_numpy(int)

        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        identity = hashlib.sha256(
            f"{ENGINE_VERSION}|{version}|{template.definition_hash()}|{config_hash}".encode()
        ).hexdigest()[:16]
        run_id = f"event_episode_{template.id}_{identity}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()

        embeddings, checkpoint_hashes, histories = [], {}, []
        for seed in cfg.training.random_seeds:
            fit = fit_reference_encoder(
                dataset.store, train_positions, valid_positions,
                cfg.encoder, cfg.training, seed=seed,
                checkpoint_path=output / "checkpoints" / f"encoder_seed_{seed}.pt",
            )
            all_positions = np.sort(np.unique(np.concatenate([
                train_positions, valid_positions, test_positions, live_positions,
            ])))
            embeddings.append(encode_sequences(
                fit.model, dataset.store, all_positions,
                batch_size=cfg.training.batch_size, device=fit.device,
            ))
            history = fit.history.copy()
            history["seed"] = seed
            histories.append(history)
            checkpoint_hashes[str(seed)] = fit.checkpoint_sha256
        state_values = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
        state_names = [f"event_state_{index:02d}" for index in range(state_values.shape[1])]
        base = MambaStateLightGBMRunner._modeling_frame(
            dataset.store, all_positions, dataset.raw_feature_names, state_values, state_names
        )
        historical_meta = split[[
            "episode_id", "trade_date", "ts_code", "severity", primary, "split", "sample_weight",
        ]].rename(columns={"trade_date": "datetime", "ts_code": "instrument", primary: "target"})
        new_anchor_keys = set(map(tuple, dataset.episodes.loc[
            dataset.episodes["trade_date"].eq(as_of), ["trade_date", "ts_code"]
        ].to_numpy()))
        live_meta = dataset.live_events[["trade_date", "ts_code", "severity"]].copy()
        live_meta["episode_id"] = live_meta.apply(
            lambda row: "live_" + hashlib.sha256(
                f"{template.definition_hash()}|{row.ts_code}|{as_of:%Y-%m-%d}".encode()
            ).hexdigest()[:16], axis=1,
        )
        live_meta["target"], live_meta["split"], live_meta["sample_weight"] = np.nan, "live", np.nan
        live_meta["episode_is_new"] = [
            (pd.Timestamp(date), str(code)) in new_anchor_keys
            for date, code in live_meta[["trade_date", "ts_code"]].itertuples(index=False, name=None)
        ]
        live_meta = live_meta.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
        metadata = pd.concat([historical_meta, live_meta], ignore_index=True, sort=False)
        modeling = base.merge(metadata, on=["datetime", "instrument"], how="inner", validate="one_to_one")

        arm_features = {
            "e0_severity": ["severity"],
            "e1_raw": dataset.raw_feature_names,
            "e2_state": state_names,
            "e3_raw_state": [*dataset.raw_feature_names, *state_names],
        }
        required_features = ["severity", *dataset.raw_feature_names, *state_names]
        historical_common = (
            modeling["split"].isin(["train", "valid", "test"])
            & modeling[required_features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            & modeling["target"].notna()
            & modeling["is_liquid"].eq(True)
        )
        live_common = (
            modeling["split"].eq("live")
            & modeling[required_features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            & modeling["is_liquid"].eq(True)
        )
        common_ids = set(modeling.loc[historical_common, "episode_id"])
        if len(common_ids) < cfg.minimum_matched_episodes:
            raise ValueError("common E0-E3 sample fell below minimum matched episodes")
        models, comparison = {}, []
        for arm in ARMS:
            arm_output = output / arm
            arm_output.mkdir()
            result, model = _fit_arm(
                modeling, historical_common, arm_features[arm], cfg, arm_output
            )
            models[arm] = model
            comparison.append({"arm": arm, **result})
        comparison_frame = pd.DataFrame(comparison)
        comparison_frame.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
        paired_deltas = {
            "e3_minus_e1": _paired_ic_delta(output, "e3_raw_state", "e1_raw", cfg.primary_horizon),
            "e2_minus_e1": _paired_ic_delta(output, "e2_state", "e1_raw", cfg.primary_horizon),
        }
        (output / "paired_ic_deltas.json").write_text(json.dumps(
            paired_deltas, ensure_ascii=False, indent=2, default=json_default
        ), encoding="utf-8")

        live = modeling.loc[live_common, [
            "episode_id", "datetime", "instrument", "severity", "episode_is_new",
        ]].copy()
        for arm in ARMS:
            live[f"score_{arm}"] = models[arm].predict(
                modeling.loc[live_common, arm_features[arm]]
            )
            live[f"rank_{arm}"] = live[f"score_{arm}"].rank(
                ascending=False, method="min"
            ).astype(int)
        live = live.sort_values("rank_e3_raw_state").reset_index(drop=True)
        all_live_keys = source_events.loc[
            pd.to_datetime(source_events["trade_date"]).eq(as_of), ["trade_date", "ts_code"]
        ].copy()
        scored_keys = set(live["instrument"].astype(str))
        exclusions = all_live_keys.loc[~all_live_keys["ts_code"].astype(str).isin(scored_keys)].copy()
        exclusions["reason"] = "missing_complete_sequence_or_raw_feature"

        dataset.episodes.to_parquet(output / "episodes.parquet", index=False)
        dataset.matched_pairs.to_parquet(output / "matched_controls.parquet", index=False)
        modeling.to_parquet(output / "modeling_dataset.parquet", index=False)
        pd.concat(histories, ignore_index=True).to_parquet(
            output / "encoder_training_history.parquet", index=False
        )
        live.to_parquet(output / "live_ranking.parquet", index=False)
        live.to_csv(output / "live_ranking.csv", index=False, encoding="utf-8-sig")
        exclusions.to_parquet(output / "live_exclusions.parquet", index=False)
        delta_ic = (
            comparison_frame.set_index("arm").loc["e3_raw_state", "rank_ic_mean"]
            - comparison_frame.set_index("arm").loc["e1_raw", "rank_ic_mean"]
        )
        summary = {
            "run_id": run_id, "engine_version": ENGINE_VERSION,
            "data_version": version, "as_of_date": as_of.strftime("%Y-%m-%d"),
            "template_id": template.id, "definition_hash": template.definition_hash(),
            "raw_episode_count": int(len(dataset.episodes)),
            "matched_mature_episode_count": int(len(mature)),
            "primary_match_rate": dataset.match_rate_primary,
            "common_episode_count": len(common_ids),
            "split_counts": modeling.loc[historical_common, "split"].value_counts().to_dict(),
            "comparison": comparison,
            "delta_rank_ic_e3_vs_e1": float(delta_ic),
            "paired_ic_deltas": paired_deltas,
            "live_source_count": int(len(all_live_keys)),
            "live_scored_count": int(len(live)), "live_excluded_count": int(len(exclusions)),
            "top_live": live.head(20).to_dict(orient="records"),
            "checkpoint_hashes": checkpoint_hashes,
            "decision": (
                "EVIDENCE_PENDING_FORWARD_VALIDATION"
                if paired_deltas["e3_minus_e1"].get("nw_t_value") is not None
                and paired_deltas["e3_minus_e1"]["nw_t_value"] >= 2.0
                else "NO_VALIDATED_STATE_INCREMENT"
            ),
            "interpretation_boundary": (
                "Single-template recent Episode study; positive ranking evidence is not yet promoted Alpha."
            ),
            "run_dir": str(output.resolve()), "cached": False,
        }
        summary_path.write_text(json.dumps(
            summary, ensure_ascii=False, indent=2, default=json_default
        ), encoding="utf-8")
        temporal = {
            "observation_identity_verified": True,
            "raw_non_anchor_events_excluded_from_controls": True,
            "matching_filters_label_maturity_before_neighbor_selection": True,
            "sequence_end_equals_episode_anchor": True,
            "future_labels_excluded_from_encoder_inputs": True,
            "split_purge_trading_days": max(cfg.horizons) + 1,
            "live_rows_used_for_fit": 0,
            "common_episode_id_hash": _id_hash(common_ids),
        }
        (output / "temporal_audit.json").write_text(json.dumps(
            temporal, ensure_ascii=False, indent=2
        ), encoding="utf-8")
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        (output / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "runner_type": "event_episode_runs", "status": "COMPLETED",
            "engine_version": ENGINE_VERSION, "data_version": version,
            "template_id": template.id, "definition_hash": template.definition_hash(),
            "config_hash": config_hash, "source_observation_id": card.observation_id,
            "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
            "checkpoint_hashes": checkpoint_hashes,
            "contains_future_labels_in_inputs": False, "live_rows_used_for_fit": 0,
            "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        return summary


def _split_and_purge(episodes, panel, cfg):
    data = episodes.sort_values(["trade_date", "ts_code"]).copy()
    dates = np.array(sorted(pd.to_datetime(data["trade_date"]).unique()))
    if len(dates) < 30:
        raise ValueError("Episode study requires at least 30 mature event dates")
    train_index = max(1, int(len(dates) * cfg.split.train_fraction))
    valid_index = max(train_index + 1, int(len(dates) * (cfg.split.train_fraction + cfg.split.valid_fraction)))
    valid_index = min(valid_index, len(dates) - 1)
    train_cut, valid_cut = pd.Timestamp(dates[train_index - 1]), pd.Timestamp(dates[valid_index - 1])
    valid_start, test_start = pd.Timestamp(dates[train_index]), pd.Timestamp(dates[valid_index])
    calendar = pd.Index(sorted(pd.to_datetime(panel["trade_date"]).unique()))
    ordinal = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    purge = max(cfg.horizons) + 1
    data["split"] = np.select([
        data["trade_date"].le(train_cut),
        data["trade_date"].between(valid_start, valid_cut),
        data["trade_date"].ge(test_start),
    ], ["train", "valid", "test"], default="drop")
    event_ord = data["trade_date"].map(ordinal)
    data = data.loc[~(
        (data["split"].eq("train") & event_ord.gt(int(ordinal[valid_start]) - purge - 1))
        | (data["split"].eq("valid") & event_ord.gt(int(ordinal[test_start]) - purge - 1))
    )].copy()
    train = data["split"].eq("train")
    reference = int(data.loc[train, "trade_date"].map(ordinal).max())
    age = reference - data["trade_date"].map(ordinal)
    decay = np.power(0.5, age / cfg.episode.decay_half_life_days)
    date_count = data.groupby("trade_date")["episode_id"].transform("count")
    data["sample_weight"] = decay / date_count
    data.loc[train, "sample_weight"] /= data.loc[train, "sample_weight"].mean()
    return data


def _fit_arm(modeling, common, features, cfg, output):
    from lightgbm import LGBMRegressor

    masks = {name: common & modeling["split"].eq(name) for name in ("train", "valid", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("an Event Episode model split is empty on the common sample")
    params = cfg.lightgbm.model_dump()
    model = LGBMRegressor(**params, verbosity=-1)
    model.fit(
        modeling.loc[masks["train"], features], modeling.loc[masks["train"], "target"],
        sample_weight=modeling.loc[masks["train"], "sample_weight"],
        eval_set=[(modeling.loc[masks["valid"], features], modeling.loc[masks["valid"], "target"])],
    )
    test = modeling.loc[masks["test"], [
        "episode_id", "datetime", "instrument", "target",
    ]].copy()
    test["score"] = model.predict(modeling.loc[masks["test"], features])
    daily_ic, daily_spread = [], []
    for date, group in test.groupby("datetime", sort=True):
        if len(group) >= 3:
            daily_ic.append({"datetime": date, "rank_ic": group["score"].corr(group["target"], method="spearman")})
        if len(group) >= 5:
            size = max(1, int(math.ceil(len(group) * 0.20)))
            ordered = group.sort_values("score")
            daily_spread.append({
                "datetime": date,
                "top_bottom_spread": ordered.tail(size)["target"].mean() - ordered.head(size)["target"].mean(),
            })
    ic = pd.DataFrame(daily_ic)
    spread = pd.DataFrame(daily_spread)
    ic.to_parquet(output / "daily_rank_ic.parquet", index=False)
    spread.to_parquet(output / "daily_top_bottom.parquet", index=False)
    test.to_parquet(output / "predictions.parquet", index=False)
    model.booster_.save_model(str(output / "lightgbm_model.txt"))
    ic_stats = _newey_west_stats(ic["rank_ic"].dropna(), max_lags=cfg.primary_horizon - 1) if len(ic) else {}
    spread_stats = _newey_west_stats(spread["top_bottom_spread"].dropna(), max_lags=cfg.primary_horizon - 1) if len(spread) else {}
    result = {
        "features": features, "train_rows": int(masks["train"].sum()),
        "valid_rows": int(masks["valid"].sum()), "test_rows": int(masks["test"].sum()),
        "test_dates": int(test["datetime"].nunique()),
        "rank_ic_mean": float(ic["rank_ic"].mean()) if len(ic) else None,
        "rank_ic_nw_t": ic_stats.get("nw_t_value"),
        "top_bottom_mean": float(spread["top_bottom_spread"].mean()) if len(spread) else None,
        "top_bottom_nw_t": spread_stats.get("nw_t_value"),
    }
    (output / "metrics.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2, default=json_default
    ), encoding="utf-8")
    return result, model


def _load_source_observation(path):
    path = Path(path)
    card_path, events_path, manifest_path = path / "observation_card.json", path / "events.parquet", path / "manifest.json"
    card = ObservationCard.model_validate_json(card_path.read_text(encoding="utf-8"))
    return card, pd.read_parquet(events_path), manifest_path.read_bytes()


def _validate_live_source(live, source_events, as_of):
    expected = set(source_events.loc[
        pd.to_datetime(source_events["trade_date"]).eq(as_of), "ts_code"
    ].astype(str))
    actual = set(live["ts_code"].astype(str))
    if expected != actual:
        raise ValueError(f"dense live events differ from frozen source: expected={len(expected)} actual={len(actual)}")


def _id_hash(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def _paired_ic_delta(output, left_arm, right_arm, horizon):
    left = pd.read_parquet(output / left_arm / "daily_rank_ic.parquet").set_index("datetime")["rank_ic"]
    right = pd.read_parquet(output / right_arm / "daily_rank_ic.parquet").set_index("datetime")["rank_ic"]
    delta = (left - right).dropna()
    return {
        "dates": int(len(delta)), "mean": float(delta.mean()) if len(delta) else None,
        "positive_ratio": float((delta > 0).mean()) if len(delta) else None,
        **(_newey_west_stats(delta, max_lags=horizon - 1) if len(delta) else {}),
    }


def _report(summary):
    lines = [
        "# Single-template Event Episode Ranker", "",
        f"- Template: `{summary['template_id']}`",
        f"- As of: `{summary['as_of_date']}`",
        f"- Matched mature episodes: `{summary['matched_mature_episode_count']}`",
        f"- E3 minus E1 RankIC: `{summary['delta_rank_ic_e3_vs_e1']}`", "",
        "|Arm|RankIC|NW t|Top-Bottom|NW t|", "|---|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]:
        lines.append(
            f"|{row['arm']}|{row['rank_ic_mean']}|{row['rank_ic_nw_t']}|"
            f"{row['top_bottom_mean']}|{row['top_bottom_nw_t']}|"
        )
    lines.extend(["", "## Live E3 ranking", ""])
    for row in summary["top_live"][:10]:
        lines.append(
            f"- `{row['instrument']}` rank={row['rank_e3_raw_state']} "
            f"score={row['score_e3_raw_state']:.6g} severity={row['severity']:.4g}"
        )
    lines.extend(["", "> " + summary["interpretation_boundary"], ""])
    return "\n".join(lines)
