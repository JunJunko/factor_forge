"""Point-in-time validation for the post-impulse sell-absorption hypothesis.

This is deliberately a mechanism study, not a trading model.  An impulse is
identified from information available at its close.  Three subsequent trading
days are observed and the resulting snapshot is scored only at that close;
returns and MFE/MAE are then calculated from the next open onward.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository
from factor_forge.breakout_process.research import _newey_west


EPS = 1e-12


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(StrictModel):
    start: str
    end: str


class AbsorptionFeatureConfig(StrictModel):
    history_window: int = Field(default=252, ge=60)
    atr_window: int = Field(default=20, ge=5)
    impulse_return_window: int = Field(default=3, ge=2)
    impulse_percentile: float = Field(default=0.90, gt=0.5, lt=1.0)
    industry_percentile: float = Field(default=0.80, gt=0.5, lt=1.0)
    observation_days: int = Field(default=3, ge=1, le=10)
    event_cooldown_days: int = Field(default=5, ge=1, le=30)
    process_window: int = Field(default=3, ge=2, le=10)
    forward_horizon: int = Field(default=10, ge=2, le=30)
    min_listing_days: int = Field(default=60, ge=1)


class AbsorptionContinuationConfig(StrictModel):
    version: int = 1
    name: str = "post_impulse_sell_absorption_v1"
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    history_start_date: str = "2019-01-01"
    sample_start_date: str = "2021-01-01"
    sample_end_date: str | None = None
    features: AbsorptionFeatureConfig = Field(default_factory=AbsorptionFeatureConfig)
    segments: dict[str, Segment] = Field(
        default_factory=lambda: {
            "discovery": Segment(start="2021-01-01", end="2023-12-31"),
            "validation": Segment(start="2024-01-01", end="2024-12-31"),
            "held_out": Segment(start="2025-01-01", end="2026-12-31"),
        }
    )
    output_root: Path = Path("artifacts/absorption_continuation_runs")
    minimum_daily_events: int = Field(default=5, ge=3)
    inspect_held_out: bool = False


def load_absorption_continuation_config(path: str | Path) -> AbsorptionContinuationConfig:
    return AbsorptionContinuationConfig.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    )


def _ts_rank(value: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    """Current value's percentile in its own trailing history (including today)."""
    ranked = value.groupby(codes, sort=False).rolling(window, min_periods=window).rank(pct=True)
    return ranked.reset_index(level=0, drop=True).reindex(value.index)


def _lag(value: pd.Series, codes: pd.Series, periods: int = 1) -> pd.Series:
    return value.groupby(codes, sort=False).shift(periods)


def _rolling_mean(value: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    result = value.groupby(codes, sort=False).rolling(window, min_periods=window).mean()
    return result.reset_index(level=0, drop=True).reindex(value.index)


def _rolling_std(value: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    result = value.groupby(codes, sort=False).rolling(window, min_periods=window).std(ddof=0)
    return result.reset_index(level=0, drop=True).reindex(value.index)


def _slope(value: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    """OLS slope of the last ``window`` observations with a fixed time grid."""
    x = np.arange(window, dtype=float)
    weights = (x - x.mean()) / np.square(x - x.mean()).sum()
    result = pd.Series(0.0, index=value.index)
    for position, weight in enumerate(weights):
        result += weight * _lag(value, codes, window - 1 - position)
    observed = value.groupby(codes, sort=False).rolling(window, min_periods=1).count()
    observed = observed.reset_index(level=0, drop=True).reindex(value.index)
    return result.where(observed.eq(window))


def _future_extreme(value: pd.Series, codes: pd.Series, horizon: int, method: str) -> pd.Series:
    """Extreme of T+1 ... T+horizon, aligned to T without using it as a feature."""
    next_value = _lag(value, codes, -1)
    reverse_value = next_value.iloc[::-1]
    reverse_codes = codes.iloc[::-1]
    rolled = reverse_value.groupby(reverse_codes, sort=False).rolling(
        horizon, min_periods=horizon
    ).agg(method)
    rolled = rolled.reset_index(level=0, drop=True)
    return rolled.iloc[::-1].reindex(value.index)


def _daily_neutral_residual(
    frame: pd.DataFrame, target: str, controls: list[str], minimum: int = 60
) -> pd.Series:
    """Cross-sectional industry/size/liquidity residual known at the same close."""
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    required = [target, "industry_l1_code", *controls]
    for _, indexes in frame.groupby("trade_date", sort=False).groups.items():
        sample = frame.loc[indexes].dropna(subset=required)
        if len(sample) < minimum:
            continue
        dummies = pd.get_dummies(sample["industry_l1_code"], drop_first=True, dtype=float)
        design = pd.concat(
            [pd.Series(1.0, index=sample.index, name="intercept"), sample[controls], dummies], axis=1
        )
        if len(sample) <= design.shape[1] + 5:
            continue
        x = design.to_numpy(dtype=float)
        y = sample[target].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        output.loc[sample.index] = y - x @ beta
    return output


def _daily_rank_ic(events: pd.DataFrame, factor: str, label: str, minimum: int) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in events[["signal_date", factor, label]].dropna().groupby("signal_date", sort=True):
        if len(group) >= minimum and group[factor].nunique() > 1 and group[label].nunique() > 1:
            values[date] = group[factor].corr(group[label], method="spearman")
    return pd.Series(values, dtype=float)


def _top_bottom_spread(events: pd.DataFrame, factor: str, label: str, minimum: int) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in events[["signal_date", factor, label]].dropna().groupby("signal_date", sort=True):
        if len(group) < minimum or group[factor].nunique() < 2:
            continue
        count = max(1, int(math.ceil(len(group) * 0.2)))
        ordered = group.sort_values(factor)
        values[date] = float(ordered.tail(count)[label].mean() - ordered.head(count)[label].mean())
    return pd.Series(values, dtype=float)


class AbsorptionContinuationRunner:
    """Run the pre-specified first-pass mechanism validation."""

    REQUIRED_COLUMNS = [
        "trade_date", "ts_code", "raw_open", "raw_high", "raw_low", "raw_close",
        "adj_open", "adj_high", "adj_low", "adj_close", "volume_shares", "amount_cny",
        "turnover_rate", "circ_mv_cny", "industry_l1_code", "is_liquid", "is_tradeable",
        "is_suspended", "is_st", "is_delisting_period", "listing_trade_days",
    ]

    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_absorption_continuation_config(config_path)
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version = repository.resolve(cfg.data_version)
        panel_path = Path(project.paths.data_root) / "versions" / data_version / "curated" / "stock_daily_panel.parquet"
        panel = pd.read_parquet(
            panel_path,
            columns=self.REQUIRED_COLUMNS,
            filters=[("trade_date", ">=", pd.Timestamp(cfg.history_start_date))],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = panel.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
        features = self._build_panel_features(panel, cfg.features)
        events = self._event_snapshots(features, cfg)
        metrics, daily = self._evaluate(events, cfg)

        digest = hashlib.sha256(json.dumps(raw_config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
        run_id = f"absorption_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{digest}"
        output = cfg.output_root / run_id
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_text(
            yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        events.to_parquet(output / "event_snapshots.parquet", index=False)
        metrics.to_csv(output / "factor_metrics.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(output / "daily_factor_statistics.csv", index=False, encoding="utf-8-sig")
        report = self._report(cfg, data_version, events, metrics)
        (output / "report.md").write_text(report, encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_version": data_version,
            "event_count": int(len(events)),
            "event_start": str(events["event_time"].min().date()) if len(events) else None,
            "event_end": str(events["signal_date"].max().date()) if len(events) else None,
            "held_out_inspected": cfg.inspect_held_out,
            "output_path": str(output.resolve()),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _build_panel_features(panel: pd.DataFrame, spec: AbsorptionFeatureConfig) -> pd.DataFrame:
        data = panel.copy()
        codes = data["ts_code"]
        close, high, low = (data[name].astype(float) for name in ("adj_close", "adj_high", "adj_low"))
        prev_close = _lag(close, codes)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        data["atr"] = _rolling_mean(tr, codes, spec.atr_window)
        atr = data["atr"].where(data["atr"] > EPS)
        daily_return = close / prev_close - 1.0
        data["return_3d"] = close / _lag(close, codes, spec.impulse_return_window) - 1.0
        data["return_3d_ts_rank"] = _ts_rank(data["return_3d"], codes, spec.history_window)
        data["return_3d_industry_rank"] = data.groupby(
            ["trade_date", "industry_l1_code"], sort=False
        )["return_3d"].rank(pct=True)
        data["displacement_slope"] = _ts_rank(
            _slope(data["return_3d_ts_rank"], codes, spec.process_window), codes, spec.history_window
        )
        data["profit_pressure"] = _ts_rank(close / _lag(close, codes, 10) - 1.0, codes, spec.history_window)

        turnover = pd.to_numeric(data["turnover_rate"], errors="coerce").clip(lower=0.0)
        data["turnover_rank"] = _ts_rank(_rolling_mean(turnover, codes, spec.process_window), codes, spec.history_window)
        down_turnover = turnover.where(daily_return < 0.0, 0.0)
        down_return = (-daily_return.clip(upper=0.0)).fillna(0.0)
        down_turn = _rolling_mean(down_turnover, codes, spec.process_window)
        down_move = _rolling_mean(down_return, codes, spec.process_window)
        data["sell_pressure_rank"] = _ts_rank(down_turn, codes, spec.history_window)
        downside_impact = down_move / down_turn.where(down_turn > EPS)
        # A no-selling window is not assigned an artificially good absorption score.
        downside_impact = downside_impact.where(down_turn > EPS, 0.0)
        data["downside_impact_rank"] = _ts_rank(downside_impact, codes, spec.history_window)
        data["impact_resilience"] = 1.0 - data["downside_impact_rank"]
        data["absorption_gap"] = data["sell_pressure_rank"] - data["downside_impact_rank"]

        high_lookback = data.groupby(codes, sort=False)["adj_high"].rolling(
            spec.process_window + 2, min_periods=spec.process_window + 2
        ).max().reset_index(level=0, drop=True).reindex(data.index)
        drawdown_atr = (high_lookback - close).clip(lower=0.0) / atr
        data["drawdown_resilience"] = 1.0 - _ts_rank(drawdown_atr, codes, spec.history_window)
        range_atr = (high - low) / atr
        data["range_contraction"] = 1.0 - _ts_rank(
            _slope(np.log(range_atr.where(range_atr > EPS)), codes, spec.process_window),
            codes, spec.history_window,
        )
        data["low_slope"] = _ts_rank(_slope(low / atr, codes, spec.process_window), codes, spec.history_window)
        raw_range = (data["raw_high"] - data["raw_low"]).where(lambda x: x > EPS)
        close_location = (data["raw_close"] - data["raw_low"]) / raw_range
        daily_vwap = data["amount_cny"] / data["volume_shares"].where(data["volume_shares"] > EPS)
        close_vwap = data["raw_close"] / daily_vwap.where(daily_vwap > EPS) - 1.0
        data["close_acceptance"] = (
            _ts_rank(close_location, codes, spec.history_window)
            + _ts_rank(close_vwap, codes, spec.history_window)
        ) / 2.0
        industry_mean = data.groupby(["trade_date", "industry_l1_code"], sort=False)["return_3d"].transform("mean")
        data["relative_strength"] = (data["return_3d"] - industry_mean).groupby(
            data["trade_date"], sort=False
        ).rank(pct=True)
        data["volatility_rank"] = _ts_rank(_rolling_std(daily_return, codes, spec.atr_window), codes, spec.history_window)
        data["log_circ_mv"] = np.log(data["circ_mv_cny"].where(data["circ_mv_cny"] > 0))

        score_parts = [
            "sell_pressure_rank", "impact_resilience", "drawdown_resilience",
            "close_acceptance", "range_contraction", "relative_strength",
        ]
        data["absorption_score"] = data[score_parts].mean(axis=1, skipna=False)
        data["absorption_score_neutral"] = _daily_neutral_residual(
            data, "absorption_score", ["log_circ_mv", "turnover_rank", "volatility_rank"]
        )

        entry = _lag(data["adj_open"].astype(float), codes, -1)
        for horizon in (5, spec.forward_horizon):
            data[f"forward_return_{horizon}d"] = _lag(data["adj_open"].astype(float), codes, -(horizon + 1)) / entry - 1.0
        future_high = _future_extreme(data["adj_high"].astype(float), codes, spec.forward_horizon, "max")
        future_low = _future_extreme(data["adj_low"].astype(float), codes, spec.forward_horizon, "min")
        data["mfe_atr"] = (future_high / entry - 1.0) / (atr / close)
        data["mae_atr"] = (entry / future_low - 1.0) / (atr / close)
        data["quality_target_atr"] = data["mfe_atr"] - 0.5 * data["mae_atr"]
        data["is_valid_universe"] = (
            data["is_liquid"].fillna(False).astype(bool)
            & data["is_tradeable"].fillna(False).astype(bool)
            & ~data["is_suspended"].fillna(True).astype(bool)
            & ~data["is_st"].fillna(True).astype(bool)
            & ~data["is_delisting_period"].fillna(True).astype(bool)
            & data["listing_trade_days"].ge(spec.min_listing_days)
        )
        return data

    @staticmethod
    def _event_snapshots(data: pd.DataFrame, cfg: AbsorptionContinuationConfig) -> pd.DataFrame:
        spec = cfg.features
        codes = data["ts_code"]
        qualified = (
            data["is_valid_universe"]
            & data["return_3d_ts_rank"].ge(spec.impulse_percentile)
            & data["return_3d_industry_rank"].ge(spec.industry_percentile)
        )
        prior_qualified = _rolling_mean(
            qualified.groupby(codes, sort=False).shift(1).eq(True).astype(float),
            codes, spec.event_cooldown_days,
        ).fillna(0.0)
        starts = qualified & prior_qualified.eq(0.0)
        source = data.loc[starts, ["ts_code", "trade_date"]].rename(columns={"trade_date": "event_time"})
        source["signal_date"] = source.groupby("ts_code", sort=False)["event_time"].shift(-0)
        # The source frame is sparse, so map the actual trading-date offset from the full panel.
        source = source.merge(
            data[["ts_code", "trade_date"]].assign(
                signal_date=data.groupby("ts_code", sort=False)["trade_date"].shift(-spec.observation_days)
            ),
            left_on=["ts_code", "event_time"], right_on=["ts_code", "trade_date"], how="left", validate="one_to_one",
        ).drop(columns=["trade_date", "signal_date_x"]).rename(columns={"signal_date_y": "signal_date"})
        snapshots = source.merge(
            data, left_on=["ts_code", "signal_date"], right_on=["ts_code", "trade_date"], how="left", validate="one_to_one"
        ).drop(columns=["trade_date"])
        snapshots = snapshots.loc[
            snapshots["signal_date"].ge(pd.Timestamp(cfg.sample_start_date))
            & (snapshots["signal_date"] <= pd.Timestamp(cfg.sample_end_date) if cfg.sample_end_date else True)
        ].copy()
        return snapshots.sort_values(["signal_date", "ts_code"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _evaluate(events: pd.DataFrame, cfg: AbsorptionContinuationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        factors = [
            "absorption_score", "absorption_score_neutral", "absorption_gap", "sell_pressure_rank",
            "impact_resilience", "drawdown_resilience", "close_acceptance", "range_contraction",
            "relative_strength", "low_slope", "displacement_slope",
        ]
        labels = ["forward_return_5d", f"forward_return_{cfg.features.forward_horizon}d", "quality_target_atr"]
        rows: list[dict] = []
        daily_rows: list[dict] = []
        segments = dict(cfg.segments)
        if not cfg.inspect_held_out:
            segments.pop("held_out", None)
        for segment_name, segment in segments.items():
            sample = events.loc[events["signal_date"].between(pd.Timestamp(segment.start), pd.Timestamp(segment.end))]
            for factor in factors:
                for label in labels:
                    ic = _daily_rank_ic(sample, factor, label, cfg.minimum_daily_events)
                    spread = _top_bottom_spread(sample, factor, label, cfg.minimum_daily_events)
                    ic_t, ic_p = _newey_west(ic, max(cfg.features.forward_horizon - 1, 0))
                    spread_t, spread_p = _newey_west(spread, max(cfg.features.forward_horizon - 1, 0))
                    rows.append({
                        "segment": segment_name, "factor": factor, "label": label,
                        "event_count": int(sample[[factor, label]].dropna().shape[0]),
                        "daily_ic_count": int(len(ic)), "rank_ic_mean": float(ic.mean()) if len(ic) else np.nan,
                        "rank_ic_positive_ratio": float((ic > 0).mean()) if len(ic) else np.nan,
                        "rank_ic_nw_t": ic_t, "rank_ic_nw_p": ic_p,
                        "top_minus_bottom_daily_count": int(len(spread)),
                        "top_minus_bottom_mean": float(spread.mean()) if len(spread) else np.nan,
                        "top_minus_bottom_nw_t": spread_t, "top_minus_bottom_nw_p": spread_p,
                    })
                    daily_rows.extend(
                        {"segment": segment_name, "factor": factor, "label": label, "statistic": "rank_ic", "signal_date": date, "value": value}
                        for date, value in ic.items()
                    )
                    daily_rows.extend(
                        {"segment": segment_name, "factor": factor, "label": label, "statistic": "top_minus_bottom", "signal_date": date, "value": value}
                        for date, value in spread.items()
                    )
        return pd.DataFrame(rows), pd.DataFrame(daily_rows)

    @staticmethod
    def _report(cfg: AbsorptionContinuationConfig, data_version: str, events: pd.DataFrame, metrics: pd.DataFrame) -> str:
        primary_label = f"forward_return_{cfg.features.forward_horizon}d"
        primary = metrics.loc[
            (metrics["factor"].eq("absorption_score_neutral")) & metrics["label"].eq(primary_label)
        ].copy()
        lines = [
            "# Post-impulse sell-absorption mechanism validation",
            "",
            f"- Data version: `{data_version}`",
            f"- Event: 3-day return is >= its own {cfg.features.impulse_percentile:.0%} history percentile and >= its industry {cfg.features.industry_percentile:.0%} percentile.",
            f"- Snapshot: close after {cfg.features.observation_days} post-impulse trading days; entry is next open.",
            f"- Events: {len(events):,}; held-out inspected: `{cfg.inspect_held_out}`.",
            "- Primary score: sell-pressure percentile + low downside-impact percentile + drawdown resilience + close acceptance + range contraction + relative strength; then residualized daily against industry, size, turnover and volatility.",
            "- Primary test: daily cross-sectional Rank IC and daily top-minus-bottom quintile spread, with Newey-West inference.",
            "",
            "## Primary metric",
            "",
        ]
        if primary.empty:
            lines.append("No primary-statistic rows were available.")
        else:
            lines.extend([
                "|segment|events|IC days|Rank IC|IC NW p|spread days|top-bottom|spread NW p|",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in primary.itertuples():
                lines.append(
                    f"|{row.segment}|{row.event_count}|{row.daily_ic_count}|{row.rank_ic_mean:.4f}|{row.rank_ic_nw_p:.4g}|"
                    f"{row.top_minus_bottom_daily_count}|{row.top_minus_bottom_mean:.4%}|{row.top_minus_bottom_nw_p:.4g}|"
                )
        lines += [
            "",
            "## Interpretation boundary",
            "",
            "This run tests the stated mechanism before any model fitting. A positive discovery result is not an alpha claim; the validation result must agree in sign and remain economically meaningful before the sealed held-out period may be inspected.",
        ]
        return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate post-impulse sell-absorption continuation")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(AbsorptionContinuationRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
