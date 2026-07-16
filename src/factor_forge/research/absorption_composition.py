"""Constrained post-impulse factor-composition search and executable backtest."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(StrictModel):
    start: str
    end: str


class CompositionConfig(StrictModel):
    version: int = 1
    name: str = "post_impulse_composition_ridge_v1"
    source_run: Path
    project_config: Path = Path("configs/project.yaml")
    data_version: str
    train: Segment
    selection: Segment
    test: Segment
    ridge_alphas: list[float] = Field(default_factory=lambda: [10.0, 100.0, 1000.0, 3000.0])
    top_n: list[int] = Field(default_factory=lambda: [5, 10, 20])
    holding_days: int = Field(default=10, ge=1)
    cost_bps: float = Field(default=20.0, ge=0)
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    lot_size: int = Field(default=100, ge=1)
    timing_position_path: Path | None = None
    timing_position_column: str = "executed_position"
    timing_missing_position_policy: Literal["error", "carry_previous"] = "error"
    output_root: Path = Path("artifacts/absorption_composition_runs")


def load_composition_config(path: str | Path) -> CompositionConfig:
    return CompositionConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


PROCESS_FEATURES = [
    "sell_pressure_rank", "impact_resilience", "drawdown_resilience", "close_acceptance",
    "range_contraction", "relative_strength", "low_slope", "displacement_slope",
    "profit_pressure", "return_3d_ts_rank", "turnover_rank",
]
CONTROL_FEATURES = ["log_circ_mv", "volatility_rank"]
LABEL = "forward_return_10d"


def _daily_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        value = pd.to_numeric(frame[column], errors="coerce")
        mean = value.groupby(frame["signal_date"]).transform("mean")
        std = value.groupby(frame["signal_date"]).transform("std", ddof=0)
        output[column] = (value - mean) / std.replace(0, np.nan)
    return output


def _daily_rank_ic(frame: pd.DataFrame, score: pd.Series) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    sample = frame[["signal_date", LABEL]].copy()
    sample["score"] = score
    for date, group in sample.dropna().groupby("signal_date", sort=True):
        if len(group) >= 5 and group["score"].nunique() > 1 and group[LABEL].nunique() > 1:
            values[date] = group["score"].corr(group[LABEL], method="spearman")
    return pd.Series(values, dtype=float)


def _top_bottom(frame: pd.DataFrame, score: pd.Series) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    sample = frame[["signal_date", LABEL]].copy()
    sample["score"] = score
    for date, group in sample.dropna().groupby("signal_date", sort=True):
        if len(group) < 5 or group["score"].nunique() < 2:
            continue
        count = max(1, math.ceil(len(group) * 0.2))
        ordered = group.sort_values("score")
        values[date] = float(ordered.tail(count)[LABEL].mean() - ordered.head(count)[LABEL].mean())
    return pd.Series(values, dtype=float)


class AbsorptionCompositionRunner:
    """Fit weights only before the test interval, then run the shared backtester."""

    PANEL_COLUMNS = [
        "trade_date", "ts_code", "raw_open", "adj_open", "adj_close", "is_liquid",
        "is_suspended", "is_limit_up_open", "is_limit_down_open", "is_st",
        "is_delisting_period", "listing_trade_days", "industry_l1_code",
    ]

    def run(self, config_path: str | Path) -> dict:
        from sklearn.linear_model import Ridge

        path = Path(config_path)
        cfg = load_composition_config(path)
        output = cfg.output_root / f"{cfg.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(path.read_bytes())
        events = pd.read_parquet(cfg.source_run / "event_snapshots.parquet")
        events["signal_date"] = pd.to_datetime(events["signal_date"])
        design, score_design = self._design(events)
        train = events["signal_date"].between(pd.Timestamp(cfg.train.start), pd.Timestamp(cfg.train.end))
        selection = events["signal_date"].between(pd.Timestamp(cfg.selection.start), pd.Timestamp(cfg.selection.end))
        test = events["signal_date"].between(pd.Timestamp(cfg.test.start), pd.Timestamp(cfg.test.end))
        train_mask = train & design.notna().all(axis=1) & events[LABEL].notna()
        sample_weight = 1.0 / events.loc[train_mask].groupby("signal_date")["signal_date"].transform("count")

        candidates: list[dict] = []
        selected_model = None
        best_score = -np.inf
        for alpha in cfg.ridge_alphas:
            model = Ridge(alpha=alpha).fit(
                design.loc[train_mask], events.loc[train_mask, LABEL], sample_weight=sample_weight
            )
            score = self._process_score(model, design, score_design)
            rank_ic = _daily_rank_ic(events.loc[selection], score.loc[selection])
            candidates.append({
                "alpha": alpha, "selection_ic_days": len(rank_ic),
                "selection_rank_ic": float(rank_ic.mean()),
                "selection_positive_ratio": float((rank_ic > 0).mean()),
            })
            if float(rank_ic.mean()) > best_score:
                best_score, selected_model = float(rank_ic.mean()), model
        assert selected_model is not None
        selected_alpha = float(next(item["alpha"] for item in candidates if item["selection_rank_ic"] == best_score))
        fit_mask = (train | selection) & design.notna().all(axis=1) & events[LABEL].notna()
        fit_weight = 1.0 / events.loc[fit_mask].groupby("signal_date")["signal_date"].transform("count")
        model = Ridge(alpha=selected_alpha).fit(
            design.loc[fit_mask], events.loc[fit_mask, LABEL], sample_weight=fit_weight
        )
        score = self._process_score(model, design, score_design)
        weights = pd.DataFrame({"feature": PROCESS_FEATURES, "weight": model.coef_[:len(PROCESS_FEATURES)]})
        weights.to_csv(output / "selected_process_weights.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(candidates).to_csv(output / "alpha_selection.csv", index=False, encoding="utf-8-sig")

        test_events = events.loc[test, ["signal_date", "ts_code", LABEL]].copy()
        test_events["factor_value"] = score.loc[test].to_numpy()
        test_events = test_events.rename(columns={"signal_date": "trade_date"})
        test_events.to_parquet(output / "test_scores.parquet", index=False)
        statistics = self._test_statistics(events.loc[test], score.loc[test])
        (output / "test_statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")

        project = load_project(cfg.project_config)
        panel_path = Path(project.paths.data_root) / "versions" / cfg.data_version / "curated" / "stock_daily_panel.parquet"
        panel = pd.read_parquet(
            panel_path, columns=self.PANEL_COLUMNS,
            filters=[("trade_date", ">=", pd.Timestamp(cfg.test.start)), ("trade_date", "<=", pd.Timestamp(cfg.test.end))],
        )
        membership = test_events[["trade_date", "ts_code"]].copy()
        membership["selection_eligible"] = True
        membership["condition_quantile"] = 5
        variants: list[tuple[str, pd.Series | None]] = [("base", None)]
        timing_filled_dates: list[str] = []
        if cfg.timing_position_path is not None:
            timing = pd.read_csv(cfg.timing_position_path)
            required = {"trade_date", cfg.timing_position_column}
            if not required <= set(timing.columns):
                raise ValueError(
                    "timing position file missing columns: " + ", ".join(sorted(required - set(timing.columns)))
                )
            timing["trade_date"] = pd.to_datetime(timing["trade_date"])
            multiplier = pd.Series(
                pd.to_numeric(timing[cfg.timing_position_column], errors="coerce").to_numpy(),
                index=timing["trade_date"], name="timing_multiplier",
            ).groupby(level=0).last().clip(0.0, 1.0)
            needed = pd.Index(pd.to_datetime(panel["trade_date"].unique())).sort_values()
            aligned = multiplier.reindex(needed)
            missing = aligned.index[aligned.isna()]
            if len(missing) and cfg.timing_missing_position_policy == "carry_previous":
                aligned = aligned.ffill()
                timing_filled_dates = [date.strftime("%Y-%m-%d") for date in missing]
            if aligned.isna().any():
                raise ValueError(
                    "timing position coverage is incomplete; missing="
                    + ", ".join(date.strftime("%Y-%m-%d") for date in aligned.index[aligned.isna()])
                )
            multiplier = aligned
            variants.append(("timing_overlay", multiplier))
        rows: list[dict] = []
        for variant, multiplier in variants:
            for top_n in cfg.top_n:
                result = BacktestEngine().run(
                    panel, test_events, universe="liquid", top_n=top_n, holding_days=cfg.holding_days,
                    initial_cash=cfg.initial_cash, lot_size=cfg.lot_size,
                    constraints=ExecutionConstraints(), cost_model=CostModel(), cost_scenario_bps=cfg.cost_bps,
                    selection_membership=membership, position_multiplier=multiplier,
                )
                result.daily.to_csv(output / f"{variant}_backtest_top{top_n}_daily.csv", index=False, encoding="utf-8-sig")
                result.trades.to_csv(output / f"{variant}_backtest_top{top_n}_trades.csv", index=False, encoding="utf-8-sig")
                rows.append({"variant": variant, "top_n": top_n, **result.metrics})
        backtests = pd.DataFrame(rows)
        backtests.to_csv(output / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        (output / "report.md").write_text(self._report(cfg, selected_alpha, weights, statistics, backtests), encoding="utf-8")
        manifest = {
            "status": "COMPLETED", "output_path": str(output.resolve()), "source_run": str(cfg.source_run.resolve()),
            "selected_alpha": selected_alpha, "fit_end": cfg.selection.end, "test_start": cfg.test.start,
            "test_period_opened": True, "timing_position_path": str(cfg.timing_position_path) if cfg.timing_position_path else None,
            "timing_missing_position_policy": cfg.timing_missing_position_policy,
            "timing_filled_dates": timing_filled_dates,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _design(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        numeric = list(dict.fromkeys(PROCESS_FEATURES + CONTROL_FEATURES))
        normalized = _daily_zscore(events, numeric)
        industries = pd.get_dummies(events["industry_l1_code"].astype("string"), prefix="industry", dtype=float)
        return pd.concat([normalized, industries], axis=1), normalized[PROCESS_FEATURES]

    @staticmethod
    def _process_score(model, design: pd.DataFrame, process_design: pd.DataFrame) -> pd.Series:
        coefficients = pd.Series(model.coef_[:len(PROCESS_FEATURES)], index=PROCESS_FEATURES)
        return process_design.fillna(0.0).dot(coefficients)

    @staticmethod
    def _test_statistics(events: pd.DataFrame, score: pd.Series) -> dict:
        rank_ic = _daily_rank_ic(events, score)
        spread = _top_bottom(events, score)
        return {
            "event_count": int(len(events)), "rank_ic_days": int(len(rank_ic)),
            "rank_ic_mean": float(rank_ic.mean()), "rank_ic_positive_ratio": float((rank_ic > 0).mean()),
            "top_minus_bottom_days": int(len(spread)), "top_minus_bottom_mean": float(spread.mean()),
        }

    @staticmethod
    def _report(cfg: CompositionConfig, alpha: float, weights: pd.DataFrame, statistics: dict, backtests: pd.DataFrame) -> str:
        lines = [
            "# Post-impulse composition search and backtest",
            "",
            f"- Weight fit: {cfg.train.start} to {cfg.train.end}; alpha selected on {cfg.selection.start} to {cfg.selection.end}.",
            f"- Selected Ridge alpha: {alpha:g}. The score contains process-feature contributions only; size, volatility and industry were fit as controls.",
            f"- Test period opened: {cfg.test.start} to {cfg.test.end}. This is now an inspected period and cannot serve as a future clean hold-out.",
            "",
            "## Test event ranking",
            "",
            f"- Events: {statistics['event_count']:,}; Rank IC: {statistics['rank_ic_mean']:.4f} over {statistics['rank_ic_days']} days; top-minus-bottom: {statistics['top_minus_bottom_mean']:.4%}.",
            "",
            "## Executable backtest (T+1 open, 10-day holding, 20 bps round trip)",
            "",
            "|Variant|Top N|Annual return|Event excess|Universe excess|Sharpe|Max drawdown|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in backtests.itertuples():
            lines.append(
                f"|{row.variant}|{row.top_n}|{row.annualized_return:.2%}|{row.annualized_excess_return:.2%}|"
                f"{row.annualized_excess_return_vs_universe:.2%}|{row.sharpe:.2f}|{row.max_drawdown:.2%}|"
            )
        lines += ["", "## Learned process weights", "", weights.to_markdown(index=False, floatfmt=".6f"), ""]
        return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Search post-impulse factor weights then backtest")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(AbsorptionCompositionRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
