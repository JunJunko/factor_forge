from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from factor_forge.config import load_project
from factor_forge.data import DataVersionRepository

from .fast import BreakoutEventBuilder
from .models import BreakoutConfig


DEFAULT_FACTORS = (
    "range_compactness",
    "volatility_contraction",
    "trend_flatness",
    "approach_velocity",
    "pre_acceleration",
    "direction_persistence",
    "breakout_strength",
    "breakout_velocity",
    "breakout_acceleration",
    "relative_volume",
    "continuous_move",
    "consolidation_age",
)

PAIR_FACTORS = (
    "range_compactness",
    "volatility_contraction",
    "trend_flatness",
    "approach_velocity",
    "pre_acceleration",
    "direction_persistence",
    "breakout_strength",
    "breakout_acceleration",
    "relative_volume",
    "continuous_move",
)

NAMED_COMBINATIONS = {
    "box_quality": ("range_compactness", "volatility_contraction", "trend_flatness"),
    "pre_dynamics": ("approach_velocity", "pre_acceleration", "direction_persistence"),
    "breakout_impulse": (
        "breakout_strength",
        "breakout_velocity",
        "breakout_acceleration",
        "relative_volume",
    ),
    "acceleration_continuation": (
        "pre_acceleration",
        "breakout_acceleration",
        "direction_persistence",
    ),
    "quality_acceleration": (
        "range_compactness",
        "volatility_contraction",
        "trend_flatness",
        "pre_acceleration",
    ),
    "full_process": (
        "range_compactness",
        "volatility_contraction",
        "trend_flatness",
        "approach_velocity",
        "pre_acceleration",
        "direction_persistence",
        "breakout_strength",
        "breakout_acceleration",
        "relative_volume",
        "continuous_move",
    ),
}


@dataclass(frozen=True)
class BreakoutResearchConfig:
    project_config: str = "configs/project.yaml"
    data_version: str = "latest"
    sample_start_date: str | None = None
    sample_end_date: str | None = None
    universe: str = "liquid"
    forward_horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    min_cross_section: int = 8
    min_ic_days: int = 40
    include_pairs: bool = True
    output_root: str = "artifacts/breakout_runs"
    workers: int = 1
    breakout: BreakoutConfig = BreakoutConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BreakoutResearchConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        breakout = BreakoutConfig(**(raw.pop("breakout", {}) or {}))
        if "forward_horizons" in raw:
            raw["forward_horizons"] = tuple(int(value) for value in raw["forward_horizons"])
        return cls(**raw, breakout=breakout)

    def __post_init__(self) -> None:
        if self.universe not in {"eligible", "tradeable", "liquid"}:
            raise ValueError("universe must be eligible, tradeable, or liquid")
        if not self.forward_horizons or any(value <= 0 for value in self.forward_horizons):
            raise ValueError("forward_horizons must contain positive integers")
        if self.min_cross_section < 3:
            raise ValueError("min_cross_section must be at least 3")
        if self.min_ic_days < 2:
            raise ValueError("min_ic_days must be at least 2")


def _newey_west(values: pd.Series, lags: int) -> tuple[float | None, float | None]:
    clean = values.dropna().to_numpy(dtype=float)
    count = len(clean)
    if count < 2:
        return None, None
    lags = min(max(int(lags), 0), count - 1)
    centered = clean - clean.mean()
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * float(
            np.dot(centered[lag:], centered[:-lag]) / count
        )
    variance = max(long_run_variance, 0.0) / count
    if variance <= 0:
        return None, None
    t_value = float(clean.mean() / math.sqrt(variance))
    return t_value, math.erfc(abs(t_value) / math.sqrt(2))


def _apply_fdr(frame: pd.DataFrame) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame["nw_p_value"].dropna().sort_values()
    count = len(valid)
    running = 1.0
    for position in range(count - 1, -1, -1):
        index = valid.index[position]
        running = min(running, float(valid.iloc[position]) * count / (position + 1))
        output.loc[index] = running
    return output


class BreakoutResearchRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = BreakoutResearchConfig.from_yaml(config_path)
        project = load_project(config.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version = repository.resolve(config.data_version)
        panel_path = (
            Path(project.paths.data_root)
            / "versions"
            / data_version
            / "curated"
            / "stock_daily_panel.parquet"
        )
        panel = self._load_panel(panel_path)
        builder_input = panel[
            [
                "trade_date",
                "ts_code",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
                "volume_shares",
            ]
        ]
        process = BreakoutEventBuilder(config.breakout, workers=config.workers).run(builder_input)
        events = self._enrich_events(process.events, panel, config)
        score_specs, score_values = self._build_scores(events, config.include_pairs)
        events = pd.concat([events, score_values], axis=1)
        conditions = self._conditions(events)
        results, daily_ic = self._evaluate(
            events,
            score_specs,
            conditions,
            config.forward_horizons,
            config.min_cross_section,
            config.min_ic_days,
        )

        digest = hashlib.sha256(
            json.dumps(raw_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:8]
        run_id = f"breakout_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{digest}"
        output = Path(config.output_root) / run_id
        output.mkdir(parents=True, exist_ok=False)
        config_copy = output / "config.yaml"
        config_copy.write_text(
            yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        process.boxes.to_parquet(output / "boxes.parquet", index=False)
        events.to_parquet(output / "events_with_scores.parquet", index=False)
        results.to_csv(output / "ic_results.csv", index=False, encoding="utf-8-sig")
        daily_ic.to_parquet(output / "daily_ic.parquet", index=False)
        promising = results.loc[results["promising"]].copy()
        promising.to_csv(output / "promising_results.csv", index=False, encoding="utf-8-sig")
        report = self._report(data_version, events, process.boxes, results, score_specs, conditions)
        (output / "report.md").write_text(report, encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_version": data_version,
            "event_count": int(len(events)),
            "box_count": int(len(process.boxes)),
            "score_count": len(score_specs),
            "condition_count": len(conditions),
            "test_count": int(len(results)),
            "promising_count": int(results["promising"].sum()),
            "inference": "daily event cross-sectional Spearman IC; Newey-West HAC; BH-FDR",
            "output_path": str(output.resolve()),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _load_panel(path: Path) -> pd.DataFrame:
        available = set(pq.ParquetFile(path).schema.names)
        required = [
            "trade_date",
            "ts_code",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "volume_shares",
            "log_total_mv",
            "is_factor_eligible",
            "is_tradeable",
            "is_liquid",
        ]
        optional = ["industry_l2_code", "pct_change"]
        columns = required + [column for column in optional if column in available]
        panel = pd.read_parquet(path, columns=columns)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        return panel.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)

    def _enrich_events(
        self,
        raw_events: pd.DataFrame,
        panel: pd.DataFrame,
        config: BreakoutResearchConfig,
    ) -> pd.DataFrame:
        if raw_events.empty:
            raise ValueError("breakout process generated no events")
        grouped_open = panel.groupby("ts_code", sort=False)["adj_open"]
        entry = grouped_open.shift(-1)
        labels = panel[["trade_date", "ts_code"]].copy()
        for horizon in config.forward_horizons:
            labels[f"forward_return_{horizon}"] = grouped_open.shift(-(horizon + 1)) / entry - 1.0

        metadata_columns = [
            "trade_date",
            "ts_code",
            "log_total_mv",
            "is_factor_eligible",
            "is_tradeable",
            "is_liquid",
        ]
        if "industry_l2_code" in panel:
            metadata_columns.append("industry_l2_code")
        metadata = panel[metadata_columns].merge(labels, on=["trade_date", "ts_code"], how="left")
        events = raw_events.rename(columns={"event_time": "trade_date"}).merge(
            metadata, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
        )
        universe_column = {
            "eligible": "is_factor_eligible",
            "tradeable": "is_tradeable",
            "liquid": "is_liquid",
        }[config.universe]
        events = events.loc[events[universe_column].fillna(False).astype(bool)].copy()
        if config.sample_start_date:
            events = events.loc[events["trade_date"] >= pd.Timestamp(config.sample_start_date)]
        if config.sample_end_date:
            events = events.loc[events["trade_date"] <= pd.Timestamp(config.sample_end_date)]

        market_mask = panel["is_factor_eligible"].fillna(False).astype(bool)
        if "pct_change" in panel:
            market_return = (
                panel.loc[market_mask, ["trade_date", "pct_change"]]
                .assign(market_component_return=lambda value: value["pct_change"] / 100.0)
                .groupby("trade_date")["market_component_return"]
                .mean()
            )
        else:
            market_components = panel.loc[
                market_mask, ["trade_date", "ts_code", "adj_close"]
            ].copy()
            market_components["market_component_return"] = market_components.groupby(
                "ts_code", sort=False
            )["adj_close"].pct_change(fill_method=None)
            market_return = market_components.groupby("trade_date")["market_component_return"].mean()
        market = market_return.to_frame()
        market["market_trend_20"] = market["market_component_return"].rolling(20).mean()
        market["market_volatility_20"] = market["market_component_return"].rolling(20).std(ddof=0)
        market["market_volatility_reference"] = (
            market["market_volatility_20"].expanding(60).median().shift(1)
        )
        events = events.merge(market.reset_index(), on="trade_date", how="left")
        events["continuous_move"] = -events["gap_atr"].abs()
        return events.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    @staticmethod
    def _build_scores(
        events: pd.DataFrame, include_pairs: bool
    ) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
        specs: dict[str, tuple[str, ...]] = {
            f"single:{factor}": (factor,) for factor in DEFAULT_FACTORS
        }
        if include_pairs:
            for left_index, left in enumerate(PAIR_FACTORS):
                for right in PAIR_FACTORS[left_index + 1 :]:
                    specs[f"pair:{left}+{right}"] = (left, right)
        specs.update({f"named:{name}": components for name, components in NAMED_COMBINATIONS.items()})

        ranked = pd.DataFrame(index=events.index)
        for factor in DEFAULT_FACTORS:
            ranked[factor] = events.groupby("trade_date")[factor].rank(method="average", pct=True)
        scores = pd.DataFrame(index=events.index)
        for name, components in specs.items():
            scores[name] = ranked[list(components)].mean(axis=1, skipna=False)
        return specs, scores

    @staticmethod
    def _conditions(events: pd.DataFrame) -> dict[str, pd.Series]:
        daily_compactness = events.groupby("trade_date")["range_compactness"].transform("median")
        daily_strength = events.groupby("trade_date")["breakout_strength"].transform("median")
        return {
            "all": pd.Series(True, index=events.index),
            "pre_accelerating": events["pre_acceleration"] > 0,
            "pre_decelerating": events["pre_acceleration"] <= 0,
            "continuous_breakout": events["gap_atr"].abs() <= 0.5,
            "gap_led_breakout": events["gap_atr"] > 0.5,
            "volume_confirmed": events["relative_volume"] > 0,
            "volume_weak": events["relative_volume"] <= 0,
            "volatility_contracting": events["volatility_contraction"] > 0,
            "volatility_expanding": events["volatility_contraction"] <= 0,
            "compact_box": events["range_compactness"] >= daily_compactness,
            "wide_box": events["range_compactness"] < daily_compactness,
            "strong_breakout": events["breakout_strength"] >= daily_strength,
            "shallow_breakout": events["breakout_strength"] < daily_strength,
            "market_up": events["market_trend_20"] >= 0,
            "market_down": events["market_trend_20"] < 0,
            "market_high_vol": (
                events["market_volatility_20"] >= events["market_volatility_reference"]
            ),
            "market_low_vol": (
                events["market_volatility_20"] < events["market_volatility_reference"]
            ),
        }

    def _evaluate(
        self,
        events: pd.DataFrame,
        score_specs: dict[str, tuple[str, ...]],
        conditions: dict[str, pd.Series],
        horizons: tuple[int, ...],
        min_cross_section: int,
        min_ic_days: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        score_names = list(score_specs)
        result_rows: list[dict] = []
        daily_rows: list[pd.DataFrame] = []
        for condition_name, condition_mask in conditions.items():
            for horizon in horizons:
                return_column = f"forward_return_{horizon}"
                sample = events.loc[condition_mask.fillna(False), ["trade_date", return_column, *score_names]]
                per_day: list[pd.Series] = []
                for trade_date, daily in sample.groupby("trade_date", sort=True):
                    valid_return = daily[return_column].notna()
                    daily = daily.loc[valid_return]
                    if len(daily) < min_cross_section:
                        continue
                    score_ranks = daily[score_names].rank(method="average")
                    return_rank = daily[return_column].rank(method="average")
                    correlations = score_ranks.corrwith(return_rank)
                    counts = daily[score_names].notna().sum()
                    correlations[counts < min_cross_section] = np.nan
                    correlations.name = trade_date
                    per_day.append(correlations)
                daily_matrix = pd.DataFrame(per_day)
                if daily_matrix.empty:
                    continue
                daily_matrix.index.name = "trade_date"
                long_daily = daily_matrix.stack().rename("rank_ic").reset_index()
                long_daily = long_daily.rename(columns={"level_1": "score"})
                long_daily["condition"] = condition_name
                long_daily["horizon"] = horizon
                daily_rows.append(long_daily)

                for score in score_names:
                    values = daily_matrix[score].dropna()
                    if len(values) < 2:
                        continue
                    cutoff = values.index.sort_values()[int(len(values) * 0.8)] if len(values) >= 5 else None
                    oos = values.loc[values.index >= cutoff] if cutoff is not None else pd.Series(dtype=float)
                    std = float(values.std(ddof=1))
                    nw_t, nw_p = _newey_west(values, max(horizon - 1, 0))
                    result_rows.append(
                        {
                            "score": score,
                            "components": "+".join(score_specs[score]),
                            "component_count": len(score_specs[score]),
                            "condition": condition_name,
                            "horizon": horizon,
                            "ic_days": int(len(values)),
                            "rank_ic_mean": float(values.mean()),
                            "rank_ic_std": std,
                            "rank_ic_ir_annualized": float(values.mean() / std * math.sqrt(252)) if std else np.nan,
                            "positive_ratio": float((values > 0).mean()),
                            "oos_rank_ic_mean": float(oos.mean()) if len(oos) else np.nan,
                            "nw_t_value": nw_t,
                            "nw_p_value": nw_p,
                        }
                    )
        results = pd.DataFrame(result_rows)
        if results.empty:
            raise ValueError("no IC result met the minimum daily cross-section")
        results["fdr_q"] = _apply_fdr(results)
        results["promising"] = (
            (results["ic_days"] >= min_ic_days)
            & (results["rank_ic_mean"] >= 0.02)
            & (results["nw_t_value"] >= 2.0)
            & (results["oos_rank_ic_mean"] > 0)
            & (results["fdr_q"] <= 0.10)
        )
        results["exploratory_score"] = (
            results["rank_ic_mean"]
            * np.sqrt(results["ic_days"])
            * np.where(results["oos_rank_ic_mean"] > 0, 1.0, 0.5)
        )
        results = results.sort_values(
            ["promising", "exploratory_score"], ascending=[False, False]
        ).reset_index(drop=True)
        daily_ic = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
        return results, daily_ic

    @staticmethod
    def _report(
        data_version: str,
        events: pd.DataFrame,
        boxes: pd.DataFrame,
        results: pd.DataFrame,
        score_specs: dict[str, tuple[str, ...]],
        conditions: dict[str, pd.Series],
    ) -> str:
        eligible = results.loc[results["ic_days"] >= 40].copy()
        top = eligible.head(30)
        lines = [
            "# 盘整突破过程因子条件 IC 报告",
            "",
            f"- 数据版本：`{data_version}`",
            f"- 有效突破事件：{len(events):,}",
            f"- 箱体记录：{len(boxes):,}",
            f"- 因子/组合：{len(score_specs)}",
            f"- 条件切片：{len(conditions)}",
            f"- 通过预设 promising 门槛：{int(results['promising'].sum())}",
            "",
            "IC 为突破事件当日截面的 Spearman IC；收益从下一交易日开盘开始。",
            "显著性使用 Newey–West HAC，FDR 为全部条件搜索上的 BH 校正。",
            "本轮属于探索性组合搜索，结果仍需固定规则后做真正样本外确认。",
            "",
            "## 排名前 30 的结果",
            "",
            "|组合|条件|周期|IC天数|Mean IC|OOS IC|NW t|FDR q|",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in top.itertuples():
            lines.append(
                f"|{row.score}|{row.condition}|{row.horizon}|{row.ic_days}|"
                f"{row.rank_ic_mean:.4f}|{row.oos_rank_ic_mean:.4f}|"
                f"{row.nw_t_value:.2f}|{row.fdr_q:.4f}|"
            )
        return "\n".join(lines) + "\n"
