from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.radar.templates import RadarTemplate

from .config import LabelConfig
from .dataset import build_dataset
from .event_episode_dataset import deduplicate_event_episodes
from .mamba_state_dataset import SequenceStore, build_sequence_store
from .recent_anomaly_structure import (
    attach_label_available_date,
    build_pit_recent_structure_features,
)


@dataclass(frozen=True)
class RecentAnomalyDataset:
    store: SequenceStore
    episodes: pd.DataFrame
    live_events: pd.DataFrame
    raw_feature_names: list[str]
    recent_feature_names: list[str]
    template_feature_names: list[str]
    state_input_names: list[str]
    template_hashes: dict[str, str]
    calendar: pd.DatetimeIndex


def build_recent_anomaly_dataset(
    panel: pd.DataFrame,
    templates: list[RadarTemplate],
    config,
    *,
    as_of_date: str | pd.Timestamp,
    source_events: dict[str, pd.DataFrame],
) -> RecentAnomalyDataset:
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    as_of = pd.Timestamp(as_of_date)
    flat, raw_features = build_dataset(
        panel, config.features,
        LabelConfig(horizon=config.primary_horizon, price="adj_open", excess_over_universe=True),
    )
    event_codes = set().union(*(
        set(frame["ts_code"].astype(str)) for frame in source_events.values()
    ))
    flat = flat.loc[flat["instrument"].astype(str).isin(event_codes)].copy()
    base_channels = panel.loc[
        panel["ts_code"].astype(str).isin(event_codes), ["trade_date", "ts_code"]
    ].copy()
    base_channels["ts_code"] = base_channels["ts_code"].astype(str)
    event_frame = base_channels.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
    channel_names, template_hashes, measured_by_template = [], {}, {}
    for template in templates:
        if template.id not in source_events:
            raise ValueError(f"missing immutable source events for {template.id}")
        source = source_events[template.id].copy()
        source["trade_date"] = pd.to_datetime(source["trade_date"])
        source["ts_code"] = source["ts_code"].astype(str)
        if "is_event" in source:
            source = source.loc[source["is_event"].eq(True)]
        source = source[["trade_date", "ts_code", "severity"]]
        if source.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"source events contain duplicate keys for {template.id}")
        event_col, severity_col = f"{template.id}__event", f"{template.id}__severity"
        measured = base_channels.merge(source, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        measured[event_col] = measured["severity"].notna().astype("float32")
        measured[severity_col] = pd.to_numeric(measured.pop("severity"), errors="coerce").fillna(0.0).astype("float32")
        measured_by_template[template.id] = measured
        event_frame = event_frame.merge(
            measured.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})[
                ["datetime", "instrument", event_col, severity_col]
            ], on=["datetime", "instrument"], how="left", validate="one_to_one",
        )
        channel_names.extend([event_col, severity_col])
        template_hashes[template.id] = template.definition_hash()
    flat = flat.merge(event_frame, on=["datetime", "instrument"], how="left", validate="one_to_one")
    state_inputs = [*raw_features, *channel_names]
    store = build_sequence_store(
        flat, state_inputs, length=config.recent_structure.sequence_length,
        min_valid_days=config.recent_structure.min_valid_days,
        validity_feature_names=raw_features,
    )
    lookup = store.samples.reset_index().set_index(["datetime", "instrument"])["index"]
    endpoint = flat[[
        "datetime", "instrument", "label", "is_liquid", *raw_features,
    ]].rename(columns={"datetime": "trade_date", "instrument": "ts_code", "label": "target"})
    calendar = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    history_dates = set(calendar[-config.recent_structure.history_trading_days:])

    episode_frames, live_frames = [], []
    for template in templates:
        _, anchors = deduplicate_event_episodes(
            measured_by_template[template.id], template_id=template.id,
            definition_hash=template.definition_hash(),
            dedup_trading_days=config.recent_structure.dedup_trading_days,
        )
        anchors = anchors.loc[anchors["trade_date"].isin(history_dates)].copy()
        anchors["template_id"] = template.id
        episode_frames.append(anchors)
        event_col, severity_col = f"{template.id}__event", f"{template.id}__severity"
        measured = measured_by_template[template.id]
        live = measured.loc[
            measured["trade_date"].eq(as_of) & measured[event_col].eq(1),
            ["trade_date", "ts_code", severity_col],
        ].rename(columns={severity_col: "severity"})
        live["template_id"] = template.id
        live["episode_id"] = [f"live_{template.id}_{code}" for code in live["ts_code"].astype(str)]
        live_frames.append(live)
    episodes = pd.concat(episode_frames, ignore_index=True)
    live = pd.concat(live_frames, ignore_index=True) if live_frames else pd.DataFrame()
    episodes = episodes.merge(endpoint, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    live = live.merge(endpoint, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    combined = pd.concat([episodes, live], ignore_index=True, sort=False)
    combined["sample_position"] = [
        lookup.get((pd.Timestamp(date), str(code)), np.nan)
        for date, code in combined[["trade_date", "ts_code"]].itertuples(index=False, name=None)
    ]
    combined = combined.dropna(subset=["sample_position"]).copy()
    combined["sample_position"] = combined["sample_position"].astype(int)
    combined = attach_label_available_date(combined, calendar, horizon=config.primary_horizon)
    combined, recent_features = build_pit_recent_structure_features(
        combined, calendar=calendar,
        windows=config.recent_structure.efficacy_windows,
        target_col="target", factor_columns=config.recent_structure.factor_columns,
        minimum_mature_events=config.recent_structure.minimum_mature_events,
    )
    template_features = []
    for template in templates:
        name = f"template__{template.id}"
        combined[name] = combined["template_id"].eq(template.id).astype("float32")
        template_features.append(name)
    is_live = combined["episode_id"].astype(str).str.startswith("live_")
    episodes = combined.loc[~is_live].sort_values(["trade_date", "template_id", "ts_code"]).reset_index(drop=True)
    live = combined.loc[is_live].sort_values(["template_id", "ts_code"]).reset_index(drop=True)
    return RecentAnomalyDataset(
        store=store, episodes=episodes, live_events=live,
        raw_feature_names=raw_features, recent_feature_names=recent_features,
        template_feature_names=template_features, state_input_names=state_inputs,
        template_hashes=template_hashes, calendar=calendar,
    )
