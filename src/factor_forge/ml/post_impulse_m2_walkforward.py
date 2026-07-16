from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import Field, model_validator

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data import DataVersionRepository

from .config import StrictModel
from .post_impulse_m3_walkforward import (
    WalkForwardFold,
    _calendar_ordinals,
    _purged_train_mask,
)
from .post_impulse_runner import (
    REGRESSION_TARGET,
    _daily_equal_weights,
    _linear_regressor,
    _neutralize_score,
    _ranking_metrics,
)


ENGINE_VERSION = "post_impulse_m1_m2_oof_backtest_v1"
PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "adj_open", "adj_close", "is_liquid",
    "is_suspended", "is_limit_up_open", "is_limit_down_open", "is_st",
    "is_delisting_period", "listing_trade_days", "industry_l1_code",
]


class M2BacktestGate(StrictModel):
    primary_top_n: Literal[10] = 10
    primary_cost_bps: Literal[20.0] = 20.0
    stress_cost_bps: Literal[40.0] = 40.0
    minimum_positive_ic_folds: int = Field(default=3, ge=1, le=4)
    require_positive_oof_ic_delta: bool = True
    require_positive_m2_minus_m1_return: bool = True
    require_positive_event_excess: bool = True
    require_positive_universe_excess: bool = True
    require_positive_stress_event_excess: bool = True
    require_all_topn_m2_minus_m1_positive: bool = True
    minimum_profitable_years: int = Field(default=3, ge=1, le=4)
    max_top10_stock_abs_pnl_share: float = Field(default=0.35, gt=0, le=1)
    max_top10_month_abs_pnl_share: float = Field(default=0.70, gt=0, le=1)
    require_positive_trimmed_trade_pnl: bool = True


class PostImpulseM2WalkForwardConfig(StrictModel):
    version: Literal[1] = 1
    name: str = "post_impulse_m1_m2_oof_backtest_v1"
    source_run: Path
    project_config: Path = Path("configs/project.yaml")
    purge_trading_days: Literal[11] = 11
    ridge_alpha: Literal[1000.0] = 1000.0
    folds: list[WalkForwardFold]
    top_n: list[Literal[5, 10, 20]] = Field(default_factory=lambda: [5, 10, 20])
    cost_bps: list[Literal[20.0, 40.0, 60.0]] = Field(
        default_factory=lambda: [20.0, 40.0, 60.0]
    )
    holding_days: Literal[10] = 10
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    lot_size: Literal[100] = 100
    minimum_train_events: int = Field(default=300, ge=20)
    minimum_daily_events: int = Field(default=5, ge=3)
    gate: M2BacktestGate = Field(default_factory=M2BacktestGate)
    output_root: Path = Path("artifacts/post_impulse_m2_walkforward_runs")

    @model_validator(mode="after")
    def frozen_space(self):
        if len(self.folds) != 4:
            raise ValueError("M1/M2 OOF backtest is frozen to four folds")
        if self.top_n != [5, 10, 20]:
            raise ValueError("top_n must remain [5, 10, 20]")
        if self.cost_bps != [20.0, 40.0, 60.0]:
            raise ValueError("cost_bps must remain [20, 40, 60]")
        for previous, current in zip(self.folds, self.folds[1:]):
            if current.train_start != previous.train_start or current.train_end <= previous.train_end:
                raise ValueError("folds must use one expanding training origin")
        return self


def load_post_impulse_m2_walkforward_config(
    path: str | Path,
) -> PostImpulseM2WalkForwardConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PostImpulseM2WalkForwardConfig.model_validate(yaml.safe_load(handle) or {})


def _feature_sets(events: pd.DataFrame) -> dict[str, list[str]]:
    m1 = sorted(
        column for column in events.columns
        if column.startswith(("coord__", "event__"))
    )
    pressure = sorted(
        column for column in events.columns
        if column.startswith("pressure__") and column != "pressure__present"
    )
    return {"m1": m1, "m2": [*m1, *pressure]}


def _pair_closed_trades(trades: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    keys = ["signal_date", "entry_date", "sleeve_id", "ts_code"]
    frame = trades.copy()
    for column in ["signal_date", "entry_date", "trade_date"]:
        frame[column] = pd.to_datetime(frame[column])
    buys = frame.loc[frame["side"].eq("BUY")].rename(columns={
        "gross_value": "buy_value", "cost": "buy_cost", "trade_date": "buy_date",
    })[keys + ["buy_date", "buy_value", "buy_cost"]]
    sells = frame.loc[frame["side"].eq("SELL")].rename(columns={
        "gross_value": "sell_value", "cost": "sell_cost", "trade_date": "sell_date",
    })[keys + ["sell_date", "sell_value", "sell_cost"]]
    closed = buys.merge(sells, on=keys, how="inner", validate="one_to_one")
    closed["net_pnl"] = (
        closed["sell_value"] - closed["sell_cost"]
        - closed["buy_value"] - closed["buy_cost"]
    )
    meta = metadata.copy()
    meta["signal_date"] = pd.to_datetime(meta["signal_date"])
    meta = meta.drop_duplicates(["signal_date", "ts_code"])
    closed = closed.merge(meta, on=["signal_date", "ts_code"], how="left", validate="many_to_one")
    closed["exit_year"] = closed["sell_date"].dt.year
    closed["exit_month"] = closed["sell_date"].dt.to_period("M").astype(str)
    return closed


def _concentration_audit(closed: pd.DataFrame, daily: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    if closed.empty:
        return {}, {}
    total_abs = float(closed["net_pnl"].abs().sum())
    stock = closed.groupby("ts_code", as_index=False)["net_pnl"].sum()
    stock["abs_pnl"] = stock["net_pnl"].abs()
    month = closed.groupby("exit_month", as_index=False)["net_pnl"].sum()
    month["abs_pnl"] = month["net_pnl"].abs()
    industry = closed.groupby("industry_l1_code", dropna=False, as_index=False)["net_pnl"].sum()
    size = closed.groupby("size_bucket", dropna=False, as_index=False)["net_pnl"].sum()
    count_to_drop = max(1, math.ceil(len(closed) * 0.01))
    trimmed = closed.sort_values("net_pnl", ascending=False).iloc[count_to_drop:]

    daily_frame = daily.copy()
    daily_frame["trade_date"] = pd.to_datetime(daily_frame["trade_date"])
    yearly = daily_frame.groupby(daily_frame["trade_date"].dt.year).agg(
        total_return=("return", lambda value: float((1.0 + value).prod() - 1.0)),
        event_excess=("excess_return", lambda value: float((1.0 + value).prod() - 1.0)),
        universe_return=(
            "universe_benchmark_return", lambda value: float((1.0 + value).prod() - 1.0)
        ),
    ).reset_index(names="year")
    summary = {
        "closed_trade_count": int(len(closed)),
        "net_closed_trade_pnl": float(closed["net_pnl"].sum()),
        "top1pct_trimmed_trade_pnl": float(trimmed["net_pnl"].sum()),
        "top10_stock_abs_pnl_share": (
            float(stock.nlargest(10, "abs_pnl")["abs_pnl"].sum() / total_abs)
            if total_abs > 0 else np.nan
        ),
        "top10_month_abs_pnl_share": (
            float(month.nlargest(10, "abs_pnl")["abs_pnl"].sum() / total_abs)
            if total_abs > 0 else np.nan
        ),
        "profitable_years": int(yearly["total_return"].gt(0).sum()),
        "year_count": int(len(yearly)),
    }
    return summary, {
        "closed_trades": closed,
        "stock": stock.sort_values("abs_pnl", ascending=False),
        "month": month.sort_values("abs_pnl", ascending=False),
        "industry": industry.sort_values("net_pnl", ascending=False),
        "size": size.sort_values("net_pnl", ascending=False),
        "yearly": yearly,
    }


class PostImpulseM2WalkForwardRunner:
    """Fixed M1 versus M2 OOF Ridge and executable T+1-open backtest."""

    def run(self, config_path: str | Path) -> dict:
        import joblib

        config_path = Path(config_path)
        cfg = load_post_impulse_m2_walkforward_config(config_path)
        source_summary = json.loads((cfg.source_run / "summary.json").read_text(encoding="utf-8"))
        events = pd.read_parquet(cfg.source_run / "event_dataset.parquet")
        events["signal_date"] = pd.to_datetime(events["signal_date"])
        events = events.loc[events["pressure__present"].eq(1.0)].copy()
        features = _feature_sets(events)

        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        data_version = source_summary["data_version"]
        panel_path = (
            Path(project.paths.data_root) / "versions" / data_version
            / "curated" / "stock_daily_panel.parquet"
        )
        calendar = _calendar_ordinals(
            panel_path, cfg.folds[0].train_start, source_summary["data_end"]
        )

        digest = hashlib.sha256(
            config_path.read_bytes()
            + source_summary["run_id"].encode()
            + ENGINE_VERSION.encode()
        ).hexdigest()[:16]
        run_id = f"post_impulse_m2_walkforward_{digest}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "models").mkdir()
        (output / "backtests").mkdir()
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        oof_rows, fold_metrics, coefficient_rows, fold_audit = [], [], [], []
        for fold in cfg.folds:
            base_train, audit = _purged_train_mask(
                events, fold, calendar, cfg.purge_trading_days
            )
            train = base_train & events[REGRESSION_TARGET].notna()
            test = events["signal_date"].between(
                pd.Timestamp(fold.test_start), pd.Timestamp(fold.test_end)
            )
            if train.sum() < cfg.minimum_train_events:
                raise ValueError(f"fold {fold.id} has only {train.sum()} mature train events")
            audit.update({
                "fold": fold.id,
                "mature_train_events": int(train.sum()),
                "test_events": int(test.sum()),
                "mature_test_events": int((test & events[REGRESSION_TARGET].notna()).sum()),
            })
            fold_audit.append(audit)
            weights = _daily_equal_weights(events.loc[train, "signal_date"])
            for variant, columns in features.items():
                x = events[columns].apply(pd.to_numeric, errors="coerce").astype(float)
                x = x.mask(~np.isfinite(x))
                model = _linear_regressor(cfg.ridge_alpha)
                model.fit(
                    x.loc[train], events.loc[train, REGRESSION_TARGET],
                    model__sample_weight=weights,
                )
                joblib.dump(model, output / "models" / f"{fold.id}_{variant}.joblib")
                score = pd.Series(model.predict(x.loc[test]), index=events.index[test])
                neutral = _neutralize_score(events.loc[test], score)
                mature_test = test & events[REGRESSION_TARGET].notna()
                raw_metrics = _ranking_metrics(
                    events.loc[mature_test], score.reindex(events.index[mature_test]),
                    REGRESSION_TARGET, minimum_daily_events=cfg.minimum_daily_events,
                )
                neutral_metrics = _ranking_metrics(
                    events.loc[mature_test], neutral.reindex(events.index[mature_test]),
                    REGRESSION_TARGET, minimum_daily_events=cfg.minimum_daily_events,
                )
                fold_metrics.append({
                    "fold": fold.id, "variant": variant,
                    "raw_rank_ic": raw_metrics["rank_ic_mean"],
                    "raw_top_bottom": raw_metrics["top_bottom_mean"],
                    "neutral_rank_ic": neutral_metrics["rank_ic_mean"],
                    "neutral_top_bottom": neutral_metrics["top_bottom_mean"],
                    "rank_ic_days": neutral_metrics["rank_ic_days"],
                })
                names = model.named_steps["imputer"].get_feature_names_out(columns)
                coefficients = np.asarray(model.named_steps["model"].coef_, dtype=float)
                coefficient_rows.extend(
                    {
                        "fold": fold.id, "variant": variant, "feature": name,
                        "standardized_coefficient": value,
                    }
                    for name, value in zip(names, coefficients)
                )
                oof_rows.append(pd.DataFrame({
                    "event_id": events.loc[test, "event_id"],
                    "trade_date": events.loc[test, "signal_date"],
                    "ts_code": events.loc[test, "ts_code"],
                    "industry_l1_code": events.loc[test, "industry_l1_code"],
                    "coord__log_circ_mv": events.loc[test, "coord__log_circ_mv"],
                    "fold": fold.id, "variant": variant,
                    "target": events.loc[test, REGRESSION_TARGET],
                    "score_raw": score, "factor_value": neutral,
                }))

        oof = pd.concat(oof_rows, ignore_index=True)
        fold_frame = self._fold_deltas(pd.DataFrame(fold_metrics))
        aggregate = self._aggregate_metrics(oof, cfg.minimum_daily_events)
        predictions = oof.loc[oof["factor_value"].notna()].copy()
        predictions["size_pct"] = predictions.groupby(["variant", "trade_date"], sort=False)[
            "coord__log_circ_mv"
        ].rank(pct=True)
        predictions["size_bucket"] = pd.cut(
            predictions["size_pct"], bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["small", "mid", "large"], include_lowest=True,
        ).astype("string")

        panel = pd.read_parquet(
            panel_path,
            columns=PANEL_COLUMNS,
            filters=[
                ("trade_date", ">=", pd.Timestamp(cfg.folds[0].test_start)),
                ("trade_date", "<=", pd.Timestamp(source_summary["data_end"])),
            ],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        membership = predictions[["trade_date", "ts_code"]].drop_duplicates().copy()
        membership["selection_eligible"] = True
        membership["condition_quantile"] = 5
        backtest_rows, primary_result = [], None
        for variant in ["m1", "m2"]:
            values = predictions.loc[
                predictions["variant"].eq(variant), ["trade_date", "ts_code", "factor_value"]
            ]
            for top_n in cfg.top_n:
                for cost_bps in cfg.cost_bps:
                    result = BacktestEngine().run(
                        panel, values, universe="liquid", top_n=top_n,
                        holding_days=cfg.holding_days, initial_cash=cfg.initial_cash,
                        lot_size=cfg.lot_size, constraints=ExecutionConstraints(),
                        cost_model=CostModel(), cost_scenario_bps=cost_bps,
                        selection_membership=membership,
                    )
                    stem = f"{variant}_top{top_n}_cost{int(cost_bps)}"
                    result.daily.to_csv(
                        output / "backtests" / f"{stem}_daily.csv",
                        index=False, encoding="utf-8-sig",
                    )
                    result.trades.to_csv(
                        output / "backtests" / f"{stem}_trades.csv",
                        index=False, encoding="utf-8-sig",
                    )
                    backtest_rows.append({
                        "variant": variant, "top_n": top_n, "cost_bps": cost_bps,
                        **result.metrics,
                    })
                    if (
                        variant == "m2" and top_n == cfg.gate.primary_top_n
                        and cost_bps == cfg.gate.primary_cost_bps
                    ):
                        primary_result = result
        assert primary_result is not None
        backtests = pd.DataFrame(backtest_rows)

        meta = predictions.loc[predictions["variant"].eq("m2"), [
            "trade_date", "ts_code", "industry_l1_code", "size_bucket",
        ]].rename(columns={"trade_date": "signal_date"})
        closed = _pair_closed_trades(primary_result.trades, meta)
        concentration, tables = _concentration_audit(closed, primary_result.daily)
        gate = self._gate(cfg, fold_frame, aggregate, backtests, concentration)

        oof.to_parquet(output / "oof_predictions.parquet", index=False)
        fold_frame.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        aggregate.to_csv(output / "aggregate_ic.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(coefficient_rows).to_csv(
            output / "standardized_coefficients.csv", index=False, encoding="utf-8-sig"
        )
        backtests.to_csv(output / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        for name, table in tables.items():
            table.to_csv(output / f"contribution_{name}.csv", index=False, encoding="utf-8-sig")
        (output / "fold_audit.json").write_text(
            json.dumps(fold_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "concentration_summary.json").write_text(
            json.dumps(concentration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "gate.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "report.md").write_text(
            self._report(fold_frame, aggregate, backtests, concentration, gate), encoding="utf-8"
        )
        summary = {
            "run_id": run_id,
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": source_summary["run_id"],
            "data_version": data_version,
            "oof_event_count": int(len(predictions.loc[predictions["variant"].eq("m1")])),
            "gate_passed": bool(gate["passed"]),
            "saved_next_action": gate["next_action"],
            "historical_diagnostic_only": True,
            "output_path": str(output.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _fold_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
        baseline = metrics.loc[metrics["variant"].eq("m1")].set_index("fold")
        result = metrics.copy()
        for column in [
            "raw_rank_ic", "raw_top_bottom", "neutral_rank_ic", "neutral_top_bottom"
        ]:
            result[f"delta_{column}_vs_m1"] = [
                value - baseline.loc[fold, column]
                for fold, value in result[["fold", column]].itertuples(index=False, name=None)
            ]
        return result

    @staticmethod
    def _aggregate_metrics(oof: pd.DataFrame, minimum_daily_events: int) -> pd.DataFrame:
        rows = []
        for variant, group in oof.groupby("variant", sort=False):
            sample = group.rename(columns={"trade_date": "signal_date", "target": REGRESSION_TARGET})
            raw = _ranking_metrics(
                sample, sample["score_raw"], REGRESSION_TARGET,
                minimum_daily_events=minimum_daily_events,
            )
            neutral = _ranking_metrics(
                sample, sample["factor_value"], REGRESSION_TARGET,
                minimum_daily_events=minimum_daily_events,
            )
            rows.append({
                "variant": variant, "event_count": int(len(group)),
                "raw_rank_ic": raw["rank_ic_mean"],
                "raw_top_bottom": raw["top_bottom_mean"],
                "neutral_rank_ic": neutral["rank_ic_mean"],
                "neutral_top_bottom": neutral["top_bottom_mean"],
                "rank_ic_days": neutral["rank_ic_days"],
            })
        result = pd.DataFrame(rows)
        baseline = result.loc[result["variant"].eq("m1")].iloc[0]
        for column in [
            "raw_rank_ic", "raw_top_bottom", "neutral_rank_ic", "neutral_top_bottom"
        ]:
            result[f"delta_{column}_vs_m1"] = result[column] - baseline[column]
        return result

    @staticmethod
    def _gate(
        cfg: PostImpulseM2WalkForwardConfig,
        folds: pd.DataFrame,
        aggregate: pd.DataFrame,
        backtests: pd.DataFrame,
        concentration: dict,
    ) -> dict:
        candidate_folds = folds.loc[folds["variant"].eq("m2")]
        candidate_ic = aggregate.loc[aggregate["variant"].eq("m2")].iloc[0]
        primary = backtests.loc[
            backtests["variant"].eq("m2")
            & backtests["top_n"].eq(cfg.gate.primary_top_n)
            & backtests["cost_bps"].eq(cfg.gate.primary_cost_bps)
        ].iloc[0]
        primary_m1 = backtests.loc[
            backtests["variant"].eq("m1")
            & backtests["top_n"].eq(cfg.gate.primary_top_n)
            & backtests["cost_bps"].eq(cfg.gate.primary_cost_bps)
        ].iloc[0]
        stress = backtests.loc[
            backtests["variant"].eq("m2")
            & backtests["top_n"].eq(cfg.gate.primary_top_n)
            & backtests["cost_bps"].eq(cfg.gate.stress_cost_bps)
        ].iloc[0]
        topn = backtests.loc[
            backtests["cost_bps"].eq(cfg.gate.primary_cost_bps)
        ].pivot(index="top_n", columns="variant", values="annualized_return")
        topn_delta = topn["m2"] - topn["m1"]
        checks = {
            "positive_ic_folds": int(candidate_folds["delta_neutral_rank_ic_vs_m1"].gt(0).sum()),
            "required_positive_ic_folds": cfg.gate.minimum_positive_ic_folds,
            "oof_delta_neutral_rank_ic": float(candidate_ic["delta_neutral_rank_ic_vs_m1"]),
            "primary_m2_minus_m1_annualized_return": float(
                primary["annualized_return"] - primary_m1["annualized_return"]
            ),
            "primary_event_excess": float(primary["annualized_excess_return"]),
            "primary_universe_excess": float(primary["annualized_excess_return_vs_universe"]),
            "stress_40bps_event_excess": float(stress["annualized_excess_return"]),
            "all_topn_m2_minus_m1_positive": bool(topn_delta.gt(0).all()),
            "profitable_years": int(concentration.get("profitable_years", 0)),
            "required_profitable_years": cfg.gate.minimum_profitable_years,
            "top10_stock_abs_pnl_share": float(
                concentration.get("top10_stock_abs_pnl_share", np.nan)
            ),
            "max_top10_stock_abs_pnl_share": cfg.gate.max_top10_stock_abs_pnl_share,
            "top10_month_abs_pnl_share": float(
                concentration.get("top10_month_abs_pnl_share", np.nan)
            ),
            "max_top10_month_abs_pnl_share": cfg.gate.max_top10_month_abs_pnl_share,
            "top1pct_trimmed_trade_pnl": float(
                concentration.get("top1pct_trimmed_trade_pnl", np.nan)
            ),
        }
        passed = (
            checks["positive_ic_folds"] >= cfg.gate.minimum_positive_ic_folds
            and (not cfg.gate.require_positive_oof_ic_delta or checks["oof_delta_neutral_rank_ic"] > 0)
            and (
                not cfg.gate.require_positive_m2_minus_m1_return
                or checks["primary_m2_minus_m1_annualized_return"] > 0
            )
            and (not cfg.gate.require_positive_event_excess or checks["primary_event_excess"] > 0)
            and (
                not cfg.gate.require_positive_universe_excess
                or checks["primary_universe_excess"] > 0
            )
            and (
                not cfg.gate.require_positive_stress_event_excess
                or checks["stress_40bps_event_excess"] > 0
            )
            and (
                not cfg.gate.require_all_topn_m2_minus_m1_positive
                or checks["all_topn_m2_minus_m1_positive"]
            )
            and checks["profitable_years"] >= cfg.gate.minimum_profitable_years
            and checks["top10_stock_abs_pnl_share"] <= cfg.gate.max_top10_stock_abs_pnl_share
            and checks["top10_month_abs_pnl_share"] <= cfg.gate.max_top10_month_abs_pnl_share
            and (
                not cfg.gate.require_positive_trimmed_trade_pnl
                or checks["top1pct_trimmed_trade_pnl"] > 0
            )
        )
        return {
            "passed": bool(passed),
            "checks": checks,
            "next_action": (
                "observe_m2_forward_on_new_data"
                if passed else "stop_post_impulse_event_strategy"
            ),
        }

    @staticmethod
    def _report(
        folds: pd.DataFrame,
        aggregate: pd.DataFrame,
        backtests: pd.DataFrame,
        concentration: dict,
        gate: dict,
    ) -> str:
        m2_folds = folds.loc[folds["variant"].eq("m2")]
        primary_rows = backtests.loc[
            backtests["cost_bps"].eq(20.0) & backtests["top_n"].isin([5, 10, 20])
        ]
        lines = [
            "# M1 versus M2 OOF executable backtest",
            "",
            "- Same pressure-qualified event pool; M2 adds only pressure features to M1.",
            "- Four expanding folds, 11-day purge, fixed Ridge alpha=1000.",
            "- Execution: T+1 open, ten-day sleeves, limit/suspension constraints.",
            "- Historical diagnostic only; covered dates were previously inspected.",
            "",
            "## Fold IC delta, M2 minus M1", "",
            "|Fold|Delta neutral IC|Delta top-bottom|",
            "|---|---:|---:|",
        ]
        for row in m2_folds.itertuples():
            lines.append(
                f"|{row.fold}|{row.delta_neutral_rank_ic_vs_m1:+.4f}|"
                f"{row.delta_neutral_top_bottom_vs_m1:+.4%}|"
            )
        lines += [
            "", "## Aggregate OOF IC", "", aggregate.to_markdown(index=False, floatfmt=".6f"),
            "", "## 20 bps executable portfolios", "",
            "|Variant|Top N|Annual return|Event excess|Universe excess|Sharpe|Max drawdown|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in primary_rows.itertuples():
            lines.append(
                f"|{row.variant}|{row.top_n}|{row.annualized_return:.2%}|"
                f"{row.annualized_excess_return:.2%}|"
                f"{row.annualized_excess_return_vs_universe:.2%}|"
                f"{row.sharpe:.2f}|{row.max_drawdown:.2%}|"
            )
        lines += [
            "", "## Primary M2 concentration", "", "```json",
            json.dumps(concentration, ensure_ascii=False, indent=2), "```",
            "", "## Deterministic Gate", "",
            f"- Passed: `{gate['passed']}`.",
            f"- Saved next action: `{gate['next_action']}`.", "",
        ]
        return "\n".join(lines)
