from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_forge.radar.templates import RadarTemplate

from .config import LabelConfig
from .dataset import build_dataset
from .event_episode_dataset import deduplicate_event_episodes
from .event_factor_basis import build_event_factor_basis
from .event_factor_sensitivity_config import FACTOR_BASIS
from .mamba_state_dataset import SequenceStore, build_sequence_store
from .recent_anomaly_structure import attach_label_available_date


@dataclass(frozen=True)
class EventFactorDataset:
    store: SequenceStore
    episodes: pd.DataFrame
    live_events: pd.DataFrame
    raw_feature_names: list[str]
    factor_names: list[str]
    template_feature_names: list[str]
    state_input_names: list[str]
    template_ids: list[str]
    template_hashes: dict[str, str]
    calendar: pd.DatetimeIndex


def build_event_factor_dataset(
    panel: pd.DataFrame,
    templates: list[RadarTemplate],
    config,
    *,
    as_of_date: str | pd.Timestamp,
    source_events: dict[str, pd.DataFrame],
) -> EventFactorDataset:
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["ts_code"] = panel["ts_code"].astype(str)
    as_of = pd.Timestamp(as_of_date)
    flat, raw_features = build_dataset(
        panel, config.features,
        LabelConfig(horizon=config.primary_horizon, price="adj_open", excess_over_universe=True),
    )
    factors = build_event_factor_basis(panel).rename(
        columns={"trade_date": "datetime", "ts_code": "instrument"}
    )
    flat = flat.merge(factors, on=["datetime", "instrument"], how="left", validate="one_to_one")
    event_codes = set().union(*(set(frame["ts_code"].astype(str)) for frame in source_events.values()))
    flat = flat.loc[flat["instrument"].astype(str).isin(event_codes)].copy()
    base = panel.loc[panel["ts_code"].isin(event_codes), ["trade_date", "ts_code"]].copy()
    event_frame = base.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
    measured_by_template, channel_names, template_hashes = {}, [], {}
    for template in templates:
        source = source_events[template.id].copy()
        source["trade_date"] = pd.to_datetime(source["trade_date"])
        source["ts_code"] = source["ts_code"].astype(str)
        if "is_event" in source:
            source = source.loc[source["is_event"].eq(True)]
        source = source[["trade_date", "ts_code", "severity"]]
        if source.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"source events contain duplicate keys for {template.id}")
        event_col, severity_col = f"{template.id}__event", f"{template.id}__severity"
        measured = base.merge(source, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        measured[event_col] = measured["severity"].notna().astype("float32")
        measured[severity_col] = pd.to_numeric(
            measured.pop("severity"), errors="coerce"
        ).fillna(0.0).astype("float32")
        measured_by_template[template.id] = measured
        event_frame = event_frame.merge(
            measured.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})[
                ["datetime", "instrument", event_col, severity_col]
            ], on=["datetime", "instrument"], how="left", validate="one_to_one",
        )
        channel_names.extend([event_col, severity_col])
        template_hashes[template.id] = template.definition_hash()
    flat = flat.merge(event_frame, on=["datetime", "instrument"], how="left", validate="one_to_one")
    state_inputs = [*raw_features, *FACTOR_BASIS, *channel_names]
    store = build_sequence_store(
        flat, state_inputs, length=config.event.sequence_length,
        min_valid_days=config.event.min_valid_days,
        validity_feature_names=[*raw_features, *FACTOR_BASIS],
    )
    lookup = store.samples.reset_index().set_index(["datetime", "instrument"])["index"]
    endpoint = flat[[
        "datetime", "instrument", "label", "is_liquid", *raw_features, *FACTOR_BASIS,
    ]].rename(columns={"datetime": "trade_date", "instrument": "ts_code", "label": "target"})
    calendar = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    output_dates = set(calendar[-config.event.history_trading_days:])
    episode_frames, live_frames = [], []
    for template_index, template in enumerate(templates):
        measured = measured_by_template[template.id]
        _, anchors = deduplicate_event_episodes(
            measured, template_id=template.id, definition_hash=template.definition_hash(),
            dedup_trading_days=config.event.dedup_trading_days,
        )
        anchors = anchors.loc[anchors["trade_date"].isin(output_dates)].copy()
        anchors["template_id"], anchors["template_index"] = template.id, template_index
        episode_frames.append(anchors)
        event_col, severity_col = f"{template.id}__event", f"{template.id}__severity"
        live = measured.loc[
            measured["trade_date"].eq(as_of) & measured[event_col].eq(1),
            ["trade_date", "ts_code", severity_col],
        ].rename(columns={severity_col: "severity"})
        live["template_id"], live["template_index"] = template.id, template_index
        live["episode_id"] = [f"live_{template.id}_{code}" for code in live["ts_code"]]
        live_frames.append(live)
    episodes = pd.concat(episode_frames, ignore_index=True).merge(
        endpoint, on=["trade_date", "ts_code"], how="left", validate="many_to_one"
    )
    live = pd.concat(live_frames, ignore_index=True).merge(
        endpoint, on=["trade_date", "ts_code"], how="left", validate="many_to_one"
    )
    combined = pd.concat([episodes, live], ignore_index=True, sort=False)
    combined["sample_position"] = [
        lookup.get((pd.Timestamp(date), str(code)), np.nan)
        for date, code in combined[["trade_date", "ts_code"]].itertuples(index=False, name=None)
    ]
    combined = combined.dropna(subset=["sample_position"]).copy()
    combined["sample_position"] = combined["sample_position"].astype(int)
    combined = attach_label_available_date(combined, calendar, horizon=config.primary_horizon)
    template_features = []
    for template in templates:
        name = f"template__{template.id}"
        combined[name] = combined["template_id"].eq(template.id).astype("float32")
        template_features.append(name)
    is_live = combined["episode_id"].astype(str).str.startswith("live_")
    return EventFactorDataset(
        store=store,
        episodes=combined.loc[~is_live].sort_values(
            ["trade_date", "template_id", "ts_code"]
        ).reset_index(drop=True),
        live_events=combined.loc[is_live].sort_values(
            ["template_id", "ts_code"]
        ).reset_index(drop=True),
        raw_feature_names=raw_features, factor_names=list(FACTOR_BASIS),
        template_feature_names=template_features, state_input_names=state_inputs,
        template_ids=[template.id for template in templates],
        template_hashes=template_hashes, calendar=calendar,
    )
