from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository


FROZEN_SCORE_NAME = "approach_velocity_plus_low_gap_v1"


@dataclass(frozen=True)
class FrozenValidationConfig:
    project_config: str
    data_version: str
    source_run: str
    output_root: str = "artifacts/breakout_validations"
    first_test_year: int = 2022
    horizons: tuple[int, ...] = (10, 20)
    min_cross_section: int = 8
    backtest_top_n: int = 10
    initial_cash: float = 1_000_000.0
    lot_size: int = 100

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FrozenValidationConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "horizons" in raw:
            raw["horizons"] = tuple(int(value) for value in raw["horizons"])
        return cls(**raw)


def frozen_score(events: pd.DataFrame) -> pd.Series:
    """The immutable V1 formula: equal-weight daily percentile ranks."""
    approach = events.groupby("trade_date")["approach_velocity"].rank(
        method="average", pct=True
    )
    low_gap = (-events["gap_atr"].abs()).groupby(events["trade_date"]).rank(
        method="average", pct=True
    )
    return (approach + low_gap) / 2.0


def exposure_residual(
    frame: pd.DataFrame,
    value_column: str,
    *,
    minimum_residual_degrees: int = 3,
) -> pd.Series:
    """Daily OLS residual against current L2 industry dummies and log market cap."""
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, indexes in frame.groupby("trade_date", sort=False).groups.items():
        group = frame.loc[indexes]
        valid = group[[value_column, "log_total_mv", "industry_l2_code"]].notna().all(axis=1)
        sample = group.loc[valid]
        if len(sample) < 8:
            continue
        size = sample["log_total_mv"].astype(float)
        size_std = float(size.std(ddof=0))
        size_z = (size - size.mean()) / size_std if size_std > 0 else size * 0.0
        dummies = pd.get_dummies(sample["industry_l2_code"], dtype=float, drop_first=True)
        design = pd.concat(
            [
                pd.Series(1.0, index=sample.index, name="intercept"),
                size_z.rename("size"),
                dummies,
            ],
            axis=1,
        )
        if len(sample) <= design.shape[1] + minimum_residual_degrees:
            continue
        x = design.to_numpy(dtype=float)
        y = sample[value_column].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        output.loc[sample.index] = y - x @ coefficients
    return output


def _newey_west(values: pd.Series, lags: int) -> tuple[float | None, float | None]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return None, None
    lags = min(max(lags, 0), len(clean) - 1)
    centered = clean - clean.mean()
    long_run_variance = float(np.dot(centered, centered) / len(clean))
    for lag in range(1, lags + 1):
        long_run_variance += 2 * (1 - lag / (lags + 1)) * float(
            np.dot(centered[lag:], centered[:-lag]) / len(clean)
        )
    variance = max(long_run_variance, 0.0) / len(clean)
    if variance <= 0:
        return None, None
    t_value = float(clean.mean() / math.sqrt(variance))
    return t_value, math.erfc(abs(t_value) / math.sqrt(2))


class FrozenBreakoutValidationRunner:
    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        config = FrozenValidationConfig.from_yaml(config_path)
        source = Path(config.source_run)
        events = pd.read_parquet(source / "events_with_scores.parquet")
        events["trade_date"] = pd.to_datetime(events["trade_date"])
        events["frozen_score"] = frozen_score(events)

        stored = "pair:approach_velocity+continuous_move"
        formula_max_difference = (
            float((events["frozen_score"] - events[stored]).abs().max())
            if stored in events
            else None
        )
        if formula_max_difference is not None and formula_max_difference > 1e-12:
            raise ValueError("recomputed frozen score does not match the source research score")

        events["neutral_score"] = exposure_residual(events, "frozen_score")
        ic_summary, daily_ic, walk_forward = self._evaluate(events, config)

        project = load_project(config.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version, panel = repository.load_panel(config.data_version)
        start = events["trade_date"].min()
        panel = panel.loc[pd.to_datetime(panel["trade_date"]) >= start].copy()
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        factor_values = events[["trade_date", "ts_code", "frozen_score"]].rename(
            columns={"frozen_score": "factor_value"}
        )

        constraints = ExecutionConstraints(
            exclude_suspended=True,
            cannot_buy_limit_up=True,
            cannot_sell_limit_down=True,
            exclude_st=True,
            exclude_delisting_period=True,
            min_listing_days=60,
        )
        costs = CostModel(
            commission_bps_per_side=3,
            slippage_bps_per_side=5,
            stamp_duty_bps_sell=5,
        )
        backtests: list[dict] = []
        backtest_details: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for horizon in config.horizons:
            result = BacktestEngine().run(
                panel,
                factor_values,
                universe="liquid",
                top_n=config.backtest_top_n,
                holding_days=horizon,
                initial_cash=config.initial_cash,
                lot_size=config.lot_size,
                constraints=constraints,
                cost_model=costs,
            )
            metrics = {"holding_days": horizon, **result.metrics}
            backtests.append(metrics)
            backtest_details[horizon] = (result.daily, result.trades)
            del result
            gc.collect()

        run_id = f"frozen_breakout_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        output = Path(config.output_root) / run_id
        output.mkdir(parents=True, exist_ok=False)
        config_copy = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        (output / "config.yaml").write_text(
            yaml.safe_dump(config_copy, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        events[
            ["trade_date", "ts_code", "frozen_score", "neutral_score"]
        ].to_parquet(output / "frozen_factor_values.parquet", index=False)
        ic_summary.to_csv(output / "ic_summary.csv", index=False, encoding="utf-8-sig")
        daily_ic.to_parquet(output / "daily_ic.parquet", index=False)
        walk_forward.to_csv(output / "walk_forward_by_year.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(backtests).to_csv(output / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        for horizon, (daily, trades) in backtest_details.items():
            daily.to_parquet(output / f"backtest_{horizon}d_daily.parquet", index=False)
            trades.to_parquet(output / f"backtest_{horizon}d_trades.parquet", index=False)
        report = self._report(
            data_version,
            formula_max_difference,
            events,
            ic_summary,
            walk_forward,
            pd.DataFrame(backtests),
        )
        (output / "report.md").write_text(report, encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "status": "COMPLETED",
            "data_version": data_version,
            "source_run": str(source.resolve()),
            "formula": "0.5 * daily_pct_rank(approach_velocity) + 0.5 * daily_pct_rank(-abs(gap_atr))",
            "formula_max_difference_vs_discovery_run": formula_max_difference,
            "event_count": int(len(events)),
            "output_path": str(output.resolve()),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def _evaluate(
        self, events: pd.DataFrame, config: FrozenValidationConfig
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary_rows: list[dict] = []
        daily_rows: list[dict] = []
        variants = {
            "raw": ("frozen_score", None),
            "industry_size_neutral_factor": ("neutral_score", None),
        }
        for horizon in config.horizons:
            return_column = f"forward_return_{horizon}"
            neutral_return_column = f"neutral_return_{horizon}"
            events[neutral_return_column] = exposure_residual(events, return_column)
            horizon_variants = {
                **variants,
                "industry_size_neutral_factor_and_return": (
                    "neutral_score",
                    neutral_return_column,
                ),
            }
            for variant, (score_column, target_override) in horizon_variants.items():
                target = target_override or return_column
                sample = events[["trade_date", score_column, target]].dropna()
                counts = sample.groupby("trade_date").size()
                valid_dates = counts[counts >= config.min_cross_section].index
                sample = sample.loc[sample["trade_date"].isin(valid_dates)]
                daily = sample.groupby("trade_date").apply(
                    lambda group: group[score_column].corr(group[target], method="spearman"),
                    include_groups=False,
                ).dropna()
                nw_t, nw_p = _newey_west(daily, horizon - 1)
                summary_rows.append(
                    {
                        "variant": variant,
                        "horizon": horizon,
                        "observations": int(len(sample)),
                        "ic_days": int(len(daily)),
                        "mean_rank_ic": float(daily.mean()),
                        "positive_ratio": float((daily > 0).mean()),
                        "nw_t_value": nw_t,
                        "nw_p_value": nw_p,
                    }
                )
                for date, value in daily.items():
                    daily_rows.append(
                        {
                            "trade_date": date,
                            "variant": variant,
                            "horizon": horizon,
                            "rank_ic": value,
                        }
                    )
        daily_ic = pd.DataFrame(daily_rows)
        walk = daily_ic.loc[
            pd.to_datetime(daily_ic["trade_date"]).dt.year >= config.first_test_year
        ].copy()
        walk["test_year"] = pd.to_datetime(walk["trade_date"]).dt.year
        walk_forward = (
            walk.groupby(["variant", "horizon", "test_year"])["rank_ic"]
            .agg(ic_days="count", mean_rank_ic="mean", positive_ratio=lambda value: (value > 0).mean())
            .reset_index()
        )
        return pd.DataFrame(summary_rows), daily_ic, walk_forward

    @staticmethod
    def _report(
        data_version: str,
        max_difference: float | None,
        events: pd.DataFrame,
        ic: pd.DataFrame,
        walk: pd.DataFrame,
        backtests: pd.DataFrame,
    ) -> str:
        lines = [
            "# 冻结突破过程因子验证",
            "",
            f"- 数据版本：`{data_version}`",
            f"- 事件数：{len(events):,}",
            "- 冻结公式：`0.5 × Rank(逼近速度) + 0.5 × Rank(-|Gap/ATR|)`",
            f"- 与发现阶段最大数值差：{max_difference}",
            "- 声明：历史数据已被发现阶段查看，本报告是冻结后的历史走步重放，不是真正盲样本。",
            "",
            "## IC与中性化",
            "",
            ic.to_markdown(index=False),
            "",
            "## 逐年测试段",
            "",
            walk.to_markdown(index=False),
            "",
            "## 可交易回测",
            "",
            backtests.to_markdown(index=False),
            "",
        ]
        return "\n".join(lines)
