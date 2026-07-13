from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import MATCHING_CONTROLS, MatchStage, MatchingConfig


def mark_frozen_events(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    keys = events[["trade_date", "ts_code", "severity"]].copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    if keys.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("frozen observation events contain duplicate keys")
    result = panel.merge(keys, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    result["is_frozen_event"] = result["severity"].notna()
    return result


def match_all_stages(
    data: pd.DataFrame,
    config: MatchingConfig,
    horizons: list[int],
) -> dict[str, pd.DataFrame]:
    event_dates = data.loc[data["is_frozen_event"], "trade_date"].unique()
    candidates = data.loc[
        data["trade_date"].isin(event_dates)
        & data["is_liquid"].fillna(False).astype(bool)
    ].copy()
    output = {}
    for stage in config.stages:
        output[stage.id] = _match_stage(candidates, stage, config, horizons)
    return output


def match_episode_anchors(
    data: pd.DataFrame,
    config: MatchingConfig,
    horizons: list[int],
) -> pd.DataFrame:
    """Match deduped episode anchors while excluding every raw same-template trigger.

    Unlike the Phase-3 compatibility matcher, maturity is applied before neighbor
    selection so an episode receives up to ``neighbors`` mature controls per horizon.
    """
    required = {"is_raw_event", "is_episode_anchor", *MATCHING_CONTROLS}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"episode matching data missing columns: {sorted(missing)}")
    stage = config.stages[-1]
    rows: list[dict] = []
    event_dates = data.loc[data["is_episode_anchor"], "trade_date"].unique()
    candidates = data.loc[
        data["trade_date"].isin(event_dates)
        & data["is_liquid"].fillna(False).astype(bool)
    ].copy()
    for (trade_date, industry), group in candidates.groupby(
        ["trade_date", "industry_l1_code"], dropna=False, sort=True
    ):
        if pd.isna(industry):
            continue
        events = group.loc[group["is_episode_anchor"]].copy()
        controls_base = group.loc[~group["is_raw_event"]].copy()
        if events.empty or controls_base.empty:
            continue
        centers, scales = _robust_location_scale(group, stage.controls)
        for horizon in horizons:
            event_return = f"forward_return_{horizon}"
            mature = f"label_mature_{horizon}"
            controls = controls_base.loc[
                controls_base[mature].fillna(False).astype(bool)
            ].dropna(subset=[*stage.controls, event_return])
            if controls.empty:
                continue
            lookup = controls.set_index("ts_code", drop=False)
            for event in events.itertuples(index=False):
                item = event._asdict()
                if (
                    not bool(item.get(mature, False))
                    or pd.isna(item.get(event_return))
                    or any(pd.isna(item.get(field)) for field in stage.controls)
                ):
                    continue
                selected = _select_controls(
                    item, controls, stage.controls, centers, scales,
                    config.neighbors, config.caliper, trade_date,
                )
                for control_code, distance in selected:
                    control = lookup.loc[control_code]
                    if isinstance(control, pd.DataFrame):
                        control = control.iloc[0]
                    rows.append({
                        "trade_date": pd.Timestamp(trade_date),
                        "industry_l1_code": industry,
                        "event_code": str(item["ts_code"]),
                        "control_code": str(control_code),
                        "horizon": int(horizon),
                        "match_distance": float(distance),
                        "severity": float(item["severity"]),
                        "event_return": float(item[event_return]),
                        "control_return": float(control[event_return]),
                        **{
                            f"event_{field}": item.get(field)
                            for field in MATCHING_CONTROLS
                        },
                        **{
                            f"control_{field}": control.get(field)
                            for field in MATCHING_CONTROLS
                        },
                    })
    return pd.DataFrame(rows)


def _match_stage(
    candidates: pd.DataFrame,
    stage: MatchStage,
    config: MatchingConfig,
    horizons: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    group_fields = ["trade_date", "industry_l1_code"]
    for (trade_date, industry), group in candidates.groupby(group_fields, dropna=False, sort=True):
        event_rows = group.loc[group["is_frozen_event"]].copy()
        control_rows = group.loc[~group["is_frozen_event"]].copy()
        if event_rows.empty or control_rows.empty or pd.isna(industry):
            continue
        if stage.controls:
            usable_controls = control_rows.dropna(subset=stage.controls)
        else:
            usable_controls = control_rows
        if usable_controls.empty:
            continue
        control_lookup = usable_controls.set_index("ts_code", drop=False)
        centers, scales = _robust_location_scale(group, stage.controls)
        for event in event_rows.itertuples(index=False):
            event_dict = event._asdict()
            if stage.controls and any(pd.isna(event_dict[field]) for field in stage.controls):
                continue
            selected = _select_controls(
                event_dict, usable_controls, stage.controls, centers, scales,
                config.neighbors, config.caliper, trade_date,
            )
            for control_code, distance in selected:
                control = control_lookup.loc[control_code]
                if isinstance(control, pd.DataFrame):
                    control = control.iloc[0]
                row = {
                    "stage": stage.id,
                    "trade_date": pd.Timestamp(trade_date),
                    "industry_l1_code": industry,
                    "event_code": event_dict["ts_code"],
                    "control_code": control_code,
                    "match_distance": float(distance),
                    "severity": float(event_dict["severity"]),
                }
                for field in MATCHING_CONTROLS:
                    row[f"event_{field}"] = event_dict.get(field)
                    row[f"control_{field}"] = control.get(field)
                for horizon in horizons:
                    row[f"event_return_{horizon}"] = event_dict.get(f"forward_return_{horizon}")
                    row[f"control_return_{horizon}"] = control.get(f"forward_return_{horizon}")
                    row[f"event_label_mature_{horizon}"] = bool(
                        event_dict.get(f"label_mature_{horizon}", False)
                    )
                    row[f"control_label_mature_{horizon}"] = bool(
                        control.get(f"label_mature_{horizon}", False)
                    )
                rows.append(row)
    columns = [
        "stage", "trade_date", "industry_l1_code", "event_code", "control_code",
        "match_distance", "severity",
        *[f"{side}_{field}" for field in MATCHING_CONTROLS for side in ("event", "control")],
        *[
            field
            for horizon in horizons
            for field in (
                f"event_return_{horizon}", f"control_return_{horizon}",
                f"event_label_mature_{horizon}", f"control_label_mature_{horizon}",
            )
        ],
    ]
    return pd.DataFrame(rows, columns=columns)


def _robust_location_scale(group: pd.DataFrame, controls: list[str]) -> tuple[dict, dict]:
    centers, scales = {}, {}
    for field in controls:
        values = pd.to_numeric(group[field], errors="coerce").dropna()
        center = float(values.median()) if len(values) else 0.0
        mad = float((values - center).abs().median()) if len(values) else 0.0
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(values.std(ddof=0)) if len(values) else 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        centers[field], scales[field] = center, scale
    return centers, scales


def _select_controls(
    event: dict,
    controls: pd.DataFrame,
    fields: list[str],
    centers: dict,
    scales: dict,
    neighbors: int,
    caliper: float,
    trade_date,
) -> list[tuple[str, float]]:
    if not fields:
        ordered = sorted(controls["ts_code"].astype(str).unique())
        digest = hashlib.sha256(f"{trade_date}|{event['ts_code']}".encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16) % len(ordered)
        rotated = ordered[offset:] + ordered[:offset]
        return [(code, 0.0) for code in rotated[:neighbors]]
    matrix = controls[fields].to_numpy(dtype=float)
    event_values = np.array([event[field] for field in fields], dtype=float)
    center = np.array([centers[field] for field in fields], dtype=float)
    scale = np.array([scales[field] for field in fields], dtype=float)
    standardized_controls = (matrix - center) / scale
    standardized_event = (event_values - center) / scale
    distance = np.sqrt(np.mean((standardized_controls - standardized_event) ** 2, axis=1))
    ranked = pd.DataFrame({
        "ts_code": controls["ts_code"].astype(str).to_numpy(),
        "distance": distance,
    }).sort_values(["distance", "ts_code"], kind="mergesort")
    ranked = ranked.loc[ranked["distance"].le(caliper)].head(neighbors)
    return list(ranked.itertuples(index=False, name=None))
