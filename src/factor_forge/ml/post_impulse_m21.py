from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import Field, model_validator

from factor_forge.backtest import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project

from .config import StrictModel
from .post_impulse_m2_walkforward import _concentration_audit, _pair_closed_trades
from .post_impulse_m3_walkforward import WalkForwardFold, _calendar_ordinals, _purged_train_mask
from .post_impulse_runner import (
    REGRESSION_TARGET,
    _daily_equal_weights,
    _linear_regressor,
    _neutralize_score,
    _ranking_metrics,
)


ENGINE_VERSION = "post_impulse_m21_compressed_reranker_v2"
COMPRESSED_FEATURES = [
    "m21__shock_intensity",
    "m21__pressure_persistence",
    "m21__pressure_resolution",
]
SHOCK_COMPONENTS = [
    "pressure__close_below_vwap_level",
    "pressure__high_to_close_level",
    "pressure__down_turnover_level",
]
PERSISTENCE_COMPONENTS = [
    "pressure__component_count",
    "pressure__active_days",
    "pressure__slope",
]
PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "adj_open", "adj_close", "is_liquid",
    "is_suspended", "is_limit_up_open", "is_limit_down_open", "is_st",
    "is_delisting_period", "listing_trade_days", "industry_l1_code",
]


class M21Gate(StrictModel):
    primary_top_n: Literal[5] = 5
    primary_cost_bps: Literal[40.0] = 40.0
    minimum_positive_ic_folds: int = Field(default=3, ge=1, le=4)
    minimum_positive_delta_years: int = Field(default=3, ge=1, le=4)
    minimum_profitable_years: int = Field(default=3, ge=1, le=4)
    maximum_drawdown: float = Field(default=-0.35, ge=-1.0, lt=0.0)


class PostImpulseM21Config(StrictModel):
    version: Literal[1] = 1
    name: str = "post_impulse_m21_compressed_reranker_v1"
    source_run: Path
    project_config: Path = Path("configs/project.yaml")
    purge_trading_days: Literal[11] = 11
    ridge_alpha: Literal[1000.0] = 1000.0
    folds: list[WalkForwardFold]
    top_n: list[Literal[5, 10]] = Field(default_factory=lambda: [5, 10])
    cost_bps: list[Literal[20.0, 40.0, 60.0]] = Field(
        default_factory=lambda: [20.0, 40.0, 60.0]
    )
    holding_days: Literal[10] = 10
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    lot_size: Literal[100] = 100
    minimum_train_events: int = Field(default=300, ge=20)
    minimum_daily_events: int = Field(default=5, ge=3)
    gate: M21Gate = Field(default_factory=M21Gate)
    output_root: Path = Path("artifacts/post_impulse_m21_runs")

    @model_validator(mode="after")
    def frozen_space(self):
        if len(self.folds) != 4:
            raise ValueError("M2.1 is frozen to four folds")
        if self.top_n != [5, 10]:
            raise ValueError("top_n must remain [5, 10]")
        if self.cost_bps != [20.0, 40.0, 60.0]:
            raise ValueError("cost_bps must remain [20, 40, 60]")
        for previous, current in zip(self.folds, self.folds[1:]):
            if current.train_start != previous.train_start or current.train_end <= previous.train_end:
                raise ValueError("folds must use one expanding training origin")
        return self


def load_post_impulse_m21_config(path: str | Path) -> PostImpulseM21Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PostImpulseM21Config.model_validate(yaml.safe_load(handle) or {})


def build_compressed_pressure_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build three same-close cross-sectional mechanisms from the redundant pressure block."""
    missing = set(SHOCK_COMPONENTS + PERSISTENCE_COMPONENTS) - set(events.columns)
    if missing:
        raise ValueError(f"missing pressure inputs: {', '.join(sorted(missing))}")
    frame = events.copy()

    def daily_rank(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.groupby(frame["signal_date"], sort=False).rank(
            pct=True, method="average"
        )

    shock_ranks = pd.concat(
        [daily_rank(frame[column]).rename(column) for column in SHOCK_COMPONENTS], axis=1
    )
    component_rank = daily_rank(frame["pressure__component_count"])
    active_rank = daily_rank(frame["pressure__active_days"])
    positive_slope = pd.to_numeric(frame["pressure__slope"], errors="coerce").clip(lower=0.0)
    slope_rank = positive_slope.groupby(frame["signal_date"], sort=False).rank(
        pct=True, method="average"
    )
    declining_rank = (-pd.to_numeric(frame["pressure__slope"], errors="coerce")).groupby(
        frame["signal_date"], sort=False
    ).rank(pct=True, method="average")

    frame["m21__shock_intensity"] = shock_ranks.mean(axis=1)
    frame["m21__pressure_persistence"] = pd.concat(
        [component_rank, active_rank, slope_rank], axis=1
    ).mean(axis=1)
    frame["m21__pressure_resolution"] = (
        frame["m21__shock_intensity"] * declining_rank
    )
    return frame


def _feature_sets(events: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    controls = sorted(column for column in events if column.startswith("coord__"))
    event = sorted(column for column in events if column.startswith("event__"))
    pressure = sorted(
        column for column in events
        if column.startswith("pressure__") and column != "pressure__present"
    )
    return {
        "c0_event": {"fit": [*controls, *event], "alpha": event},
        "c1_raw_pressure": {
            "fit": [*controls, *event, *pressure], "alpha": [*event, *pressure],
        },
        "c2_compressed_pressure": {
            "fit": [*controls, *event, *COMPRESSED_FEATURES],
            "alpha": [*event, *COMPRESSED_FEATURES],
        },
    }


def _alpha_only_score(model, x: pd.DataFrame, alpha_columns: list[str]) -> pd.Series:
    """Score only alpha columns while risk coordinates remain regression controls."""
    imputed = model.named_steps["imputer"].transform(x)
    transformed = model.named_steps["scaler"].transform(imputed)
    names = np.asarray(model.named_steps["imputer"].get_feature_names_out(x.columns), dtype=str)
    alpha = set(alpha_columns)
    mask = np.asarray([
        name in alpha
        or (name.startswith("missingindicator_") and name.removeprefix("missingindicator_") in alpha)
        for name in names
    ])
    coefficients = np.asarray(model.named_steps["model"].coef_, dtype=float)
    values = transformed[:, mask] @ coefficients[mask]
    return pd.Series(np.asarray(values).reshape(-1), index=x.index, dtype=float)


def _fold_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[metrics["variant"].eq("c0_event")].set_index("fold")
    result = metrics.copy()
    for column in ["raw_rank_ic", "raw_top_bottom", "neutral_rank_ic", "neutral_top_bottom"]:
        result[f"delta_{column}_vs_c0"] = [
            value - baseline.loc[fold, column]
            for fold, value in result[["fold", column]].itertuples(index=False, name=None)
        ]
    return result


def _aggregate_metrics(oof: pd.DataFrame, minimum_daily_events: int) -> pd.DataFrame:
    rows = []
    for variant, group in oof.groupby("variant", sort=False):
        sample = group.rename(columns={"trade_date": "signal_date", "target": REGRESSION_TARGET})
        mature = sample[REGRESSION_TARGET].notna()
        raw = _ranking_metrics(
            sample.loc[mature], sample.loc[mature, "score_raw"], REGRESSION_TARGET,
            minimum_daily_events=minimum_daily_events,
        )
        neutral = _ranking_metrics(
            sample.loc[mature], sample.loc[mature, "factor_value"], REGRESSION_TARGET,
            minimum_daily_events=minimum_daily_events,
        )
        rows.append({
            "variant": variant, "event_count": int(len(group)),
            "raw_rank_ic": raw["rank_ic_mean"], "raw_top_bottom": raw["top_bottom_mean"],
            "neutral_rank_ic": neutral["rank_ic_mean"],
            "neutral_top_bottom": neutral["top_bottom_mean"],
            "rank_ic_days": neutral["rank_ic_days"],
        })
    result = pd.DataFrame(rows)
    baseline = result.loc[result["variant"].eq("c0_event")].iloc[0]
    for column in ["raw_rank_ic", "raw_top_bottom", "neutral_rank_ic", "neutral_top_bottom"]:
        result[f"delta_{column}_vs_c0"] = result[column] - baseline[column]
    return result


def _yearly_comparison(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    frame = left[["trade_date", "return"]].merge(
        right[["trade_date", "return"]], on="trade_date", suffixes=("_left", "_right")
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    rows = []
    for year, group in frame.groupby(frame["trade_date"].dt.year):
        left_return = float((1.0 + group["return_left"]).prod() - 1.0)
        right_return = float((1.0 + group["return_right"]).prod() - 1.0)
        rows.append({
            "year": int(year), "candidate_return": left_return,
            "baseline_return": right_return, "candidate_minus_baseline": left_return - right_return,
        })
    return pd.DataFrame(rows)


class PostImpulseM21Runner:
    """Compressed pressure reranker with corrected executable cohort comparison."""

    def run(self, config_path: str | Path) -> dict:
        import joblib

        config_path = Path(config_path)
        cfg = load_post_impulse_m21_config(config_path)
        source_summary = json.loads((cfg.source_run / "summary.json").read_text(encoding="utf-8"))
        events = pd.read_parquet(cfg.source_run / "event_dataset.parquet")
        events["signal_date"] = pd.to_datetime(events["signal_date"])
        events = events.loc[events["pressure__present"].eq(1.0)].copy()
        events = build_compressed_pressure_features(events)
        features = _feature_sets(events)

        project = load_project(cfg.project_config)
        data_version = source_summary["data_version"]
        panel_path = (
            Path(project.paths.data_root) / "versions" / data_version
            / "curated" / "stock_daily_panel.parquet"
        )
        calendar = _calendar_ordinals(
            panel_path, cfg.folds[0].train_start, source_summary["data_end"]
        )

        digest = hashlib.sha256(
            config_path.read_bytes() + source_summary["run_id"].encode() + ENGINE_VERSION.encode()
        ).hexdigest()[:16]
        run_id = f"post_impulse_m21_{digest}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "models").mkdir()
        (output / "backtests").mkdir()
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        oof_rows, metric_rows, coefficient_rows, fold_audit = [], [], [], []
        for fold in cfg.folds:
            base_train, audit = _purged_train_mask(events, fold, calendar, cfg.purge_trading_days)
            train = base_train & events[REGRESSION_TARGET].notna()
            test = events["signal_date"].between(
                pd.Timestamp(fold.test_start), pd.Timestamp(fold.test_end)
            )
            if train.sum() < cfg.minimum_train_events:
                raise ValueError(f"fold {fold.id} has only {train.sum()} mature train events")
            audit.update({
                "fold": fold.id, "mature_train_events": int(train.sum()),
                "test_events": int(test.sum()),
                "mature_test_events": int((test & events[REGRESSION_TARGET].notna()).sum()),
            })
            fold_audit.append(audit)
            weights = _daily_equal_weights(events.loc[train, "signal_date"])
            for variant, specification in features.items():
                columns = specification["fit"]
                x = events[columns].apply(pd.to_numeric, errors="coerce").astype(float)
                x = x.mask(~np.isfinite(x))
                model = _linear_regressor(cfg.ridge_alpha)
                model.fit(
                    x.loc[train], events.loc[train, REGRESSION_TARGET],
                    model__sample_weight=weights,
                )
                joblib.dump(model, output / "models" / f"{fold.id}_{variant}.joblib")
                raw = _alpha_only_score(model, x.loc[test], specification["alpha"])
                neutral = _neutralize_score(events.loc[test], raw)
                mature = test & events[REGRESSION_TARGET].notna()
                raw_metrics = _ranking_metrics(
                    events.loc[mature], raw.reindex(events.index[mature]), REGRESSION_TARGET,
                    minimum_daily_events=cfg.minimum_daily_events,
                )
                neutral_metrics = _ranking_metrics(
                    events.loc[mature], neutral.reindex(events.index[mature]), REGRESSION_TARGET,
                    minimum_daily_events=cfg.minimum_daily_events,
                )
                metric_rows.append({
                    "fold": fold.id, "variant": variant,
                    "raw_rank_ic": raw_metrics["rank_ic_mean"],
                    "raw_top_bottom": raw_metrics["top_bottom_mean"],
                    "neutral_rank_ic": neutral_metrics["rank_ic_mean"],
                    "neutral_top_bottom": neutral_metrics["top_bottom_mean"],
                    "rank_ic_days": neutral_metrics["rank_ic_days"],
                })
                names = model.named_steps["imputer"].get_feature_names_out(columns)
                coefficients = np.asarray(model.named_steps["model"].coef_, dtype=float)
                coefficient_rows.extend({
                    "fold": fold.id, "variant": variant, "feature": name,
                    "role": (
                        "alpha" if name in specification["alpha"]
                        or name.removeprefix("missingindicator_") in specification["alpha"]
                        else "control"
                    ),
                    "standardized_coefficient": value,
                } for name, value in zip(names, coefficients))
                oof_rows.append(pd.DataFrame({
                    "event_id": events.loc[test, "event_id"],
                    "trade_date": events.loc[test, "signal_date"],
                    "ts_code": events.loc[test, "ts_code"],
                    "industry_l1_code": events.loc[test, "industry_l1_code"],
                    "coord__log_circ_mv": events.loc[test, "coord__log_circ_mv"],
                    "fold": fold.id, "variant": variant,
                    "target": events.loc[test, REGRESSION_TARGET],
                    "score_raw": raw, "factor_value": neutral,
                }))

        oof = pd.concat(oof_rows, ignore_index=True)
        folds = _fold_deltas(pd.DataFrame(metric_rows))
        aggregate = _aggregate_metrics(oof, cfg.minimum_daily_events)

        panel = pd.read_parquet(
            panel_path, columns=PANEL_COLUMNS,
            filters=[
                ("trade_date", ">=", pd.Timestamp(cfg.folds[0].test_start)),
                ("trade_date", "<=", pd.Timestamp(source_summary["data_end"])),
            ],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel_dates = pd.Index(panel["trade_date"].drop_duplicates().sort_values())
        last_mature_signal = panel_dates[-(cfg.holding_days + 2)]
        predictions = oof.loc[
            oof["factor_value"].notna() & oof["trade_date"].le(last_mature_signal)
        ].copy()
        predictions["size_pct"] = predictions.groupby(
            ["variant", "trade_date"], sort=False
        )["coord__log_circ_mv"].rank(pct=True)
        predictions["size_bucket"] = pd.cut(
            predictions["size_pct"], bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["small", "mid", "large"], include_lowest=True,
        ).astype("string")
        membership = predictions[["trade_date", "ts_code"]].drop_duplicates().copy()
        membership["selection_eligible"] = True
        membership["condition_quantile"] = 5

        common_events = oof.loc[oof["trade_date"].le(last_mature_signal), [
            "trade_date", "ts_code",
        ]].drop_duplicates()
        common_events["factor_value"] = 0.0
        common_membership = common_events[["trade_date", "ts_code"]].copy()
        common_membership["selection_eligible"] = True
        common_membership["condition_quantile"] = 5
        cohort_top_n = int(common_events.groupby("trade_date").size().max())

        cohort_results, cohort_rows = {}, []
        for cost_bps in cfg.cost_bps:
            result = BacktestEngine().run(
                panel, common_events, universe="liquid", top_n=cohort_top_n,
                holding_days=cfg.holding_days, initial_cash=cfg.initial_cash,
                lot_size=cfg.lot_size, constraints=ExecutionConstraints(),
                cost_model=CostModel(), cost_scenario_bps=cost_bps,
                selection_membership=common_membership, fully_invest_selected=True,
            )
            cohort_results[cost_bps] = result
            result.daily.to_csv(
                output / "backtests" / f"matched_cohort_cost{int(cost_bps)}_daily.csv",
                index=False, encoding="utf-8-sig",
            )
            result.trades.to_csv(
                output / "backtests" / f"matched_cohort_cost{int(cost_bps)}_trades.csv",
                index=False, encoding="utf-8-sig",
            )
            cohort_rows.append({"cost_bps": cost_bps, **result.metrics})

        backtest_rows, results = [], {}
        for variant in features:
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
                        selection_membership=membership, fully_invest_selected=True,
                    )
                    cohort = cohort_results[cost_bps]
                    stem = f"{variant}_top{top_n}_cost{int(cost_bps)}"
                    result.daily.to_csv(
                        output / "backtests" / f"{stem}_daily.csv", index=False,
                        encoding="utf-8-sig",
                    )
                    result.trades.to_csv(
                        output / "backtests" / f"{stem}_trades.csv", index=False,
                        encoding="utf-8-sig",
                    )
                    results[(variant, top_n, cost_bps)] = result
                    backtest_rows.append({
                        "variant": variant, "top_n": top_n, "cost_bps": cost_bps,
                        **result.metrics,
                        "matched_cohort_annualized_return": cohort.metrics["annualized_return"],
                        "annualized_excess_vs_matched_cohort": (
                            result.metrics["annualized_return"] - cohort.metrics["annualized_return"]
                        ),
                    })
        backtests = pd.DataFrame(backtest_rows)

        key = ("c2_compressed_pressure", cfg.gate.primary_top_n, cfg.gate.primary_cost_bps)
        primary = results[key]
        cohort = cohort_results[cfg.gate.primary_cost_bps]
        corrected_daily = primary.daily.copy()
        corrected_daily["excess_return"] = (
            corrected_daily["return"].to_numpy() - cohort.daily["return"].to_numpy()
        )
        meta = predictions.loc[predictions["variant"].eq("c2_compressed_pressure"), [
            "trade_date", "ts_code", "industry_l1_code", "size_bucket",
        ]].rename(columns={"trade_date": "signal_date"})
        closed = _pair_closed_trades(primary.trades, meta)
        concentration, tables = _concentration_audit(closed, corrected_daily)
        yearly_delta = _yearly_comparison(
            primary.daily,
            results[("c0_event", cfg.gate.primary_top_n, cfg.gate.primary_cost_bps)].daily,
        )
        gate = self._gate(cfg, folds, aggregate, backtests, concentration, yearly_delta)

        oof.to_parquet(output / "oof_predictions.parquet", index=False)
        folds.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        aggregate.to_csv(output / "aggregate_ic.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(coefficient_rows).to_csv(
            output / "standardized_coefficients.csv", index=False, encoding="utf-8-sig"
        )
        backtests.to_csv(output / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(cohort_rows).to_csv(
            output / "matched_cohort_summary.csv", index=False, encoding="utf-8-sig"
        )
        yearly_delta.to_csv(output / "yearly_delta.csv", index=False, encoding="utf-8-sig")
        for name, table in tables.items():
            table.to_csv(output / f"contribution_{name}.csv", index=False, encoding="utf-8-sig")
        for filename, payload in [
            ("fold_audit.json", fold_audit),
            ("concentration_summary.json", concentration),
            ("gate.json", gate),
        ]:
            (output / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (output / "report.md").write_text(
            self._report(folds, aggregate, backtests, concentration, yearly_delta, gate),
            encoding="utf-8",
        )
        summary = {
            "run_id": run_id, "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": source_summary["run_id"], "data_version": data_version,
            "last_mature_signal": str(pd.Timestamp(last_mature_signal).date()),
            "oof_scored_events": int(len(predictions.loc[
                predictions["variant"].eq("c2_compressed_pressure")
            ])),
            "reranker_gate_passed": bool(gate["reranker_passed"]),
            "standalone_gate_passed": bool(gate["standalone_passed"]),
            "saved_next_action": gate["next_action"],
            "historical_development_only": True, "output_path": str(output.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _gate(cfg, folds, aggregate, backtests, concentration, yearly_delta) -> dict:
        candidate_folds = folds.loc[folds["variant"].eq("c2_compressed_pressure")]
        candidate_ic = aggregate.loc[
            aggregate["variant"].eq("c2_compressed_pressure")
        ].iloc[0]
        primary = backtests.loc[
            backtests["variant"].eq("c2_compressed_pressure")
            & backtests["top_n"].eq(cfg.gate.primary_top_n)
            & backtests["cost_bps"].eq(cfg.gate.primary_cost_bps)
        ].iloc[0]
        baseline = backtests.loc[
            backtests["variant"].eq("c0_event")
            & backtests["top_n"].eq(cfg.gate.primary_top_n)
            & backtests["cost_bps"].eq(cfg.gate.primary_cost_bps)
        ].iloc[0]
        raw = backtests.loc[
            backtests["variant"].eq("c1_raw_pressure")
            & backtests["top_n"].eq(cfg.gate.primary_top_n)
            & backtests["cost_bps"].eq(cfg.gate.primary_cost_bps)
        ].iloc[0]
        checks = {
            "positive_ic_folds": int(candidate_folds["delta_neutral_rank_ic_vs_c0"].gt(0).sum()),
            "required_positive_ic_folds": cfg.gate.minimum_positive_ic_folds,
            "oof_delta_neutral_rank_ic": float(candidate_ic["delta_neutral_rank_ic_vs_c0"]),
            "compressed_minus_event_annualized_return": float(
                primary["annualized_return"] - baseline["annualized_return"]
            ),
            "compressed_minus_raw_annualized_return": float(
                primary["annualized_return"] - raw["annualized_return"]
            ),
            "positive_delta_years": int(yearly_delta["candidate_minus_baseline"].gt(0).sum()),
            "required_positive_delta_years": cfg.gate.minimum_positive_delta_years,
            "primary_annualized_return": float(primary["annualized_return"]),
            "matched_cohort_excess": float(primary["annualized_excess_vs_matched_cohort"]),
            "universe_excess": float(primary["annualized_excess_return_vs_universe"]),
            "max_drawdown": float(primary["max_drawdown"]),
            "maximum_drawdown": cfg.gate.maximum_drawdown,
            "profitable_years": int(concentration.get("profitable_years", 0)),
            "required_profitable_years": cfg.gate.minimum_profitable_years,
            "top1pct_trimmed_trade_pnl": float(
                concentration.get("top1pct_trimmed_trade_pnl", np.nan)
            ),
        }
        reranker = (
            checks["positive_ic_folds"] >= cfg.gate.minimum_positive_ic_folds
            and checks["oof_delta_neutral_rank_ic"] > 0
            and checks["compressed_minus_event_annualized_return"] > 0
            and checks["compressed_minus_raw_annualized_return"] >= 0
            and checks["positive_delta_years"] >= cfg.gate.minimum_positive_delta_years
        )
        standalone = (
            reranker
            and checks["primary_annualized_return"] > 0
            and checks["matched_cohort_excess"] > 0
            and checks["universe_excess"] > 0
            and checks["max_drawdown"] >= cfg.gate.maximum_drawdown
            and checks["profitable_years"] >= cfg.gate.minimum_profitable_years
            and checks["top1pct_trimmed_trade_pnl"] > 0
        )
        if standalone:
            action = "forward_observe_m21_standalone_on_new_data"
        elif reranker:
            action = "retain_raw_m2_and_shadow_m21_as_reranker"
        else:
            action = "retain_raw_m2_only_and_stop_pressure_expansion"
        return {
            "reranker_passed": bool(reranker), "standalone_passed": bool(standalone),
            "checks": checks, "next_action": action,
        }

    @staticmethod
    def _report(folds, aggregate, backtests, concentration, yearly_delta, gate) -> str:
        primary = backtests.loc[
            backtests["top_n"].eq(5) & backtests["cost_bps"].eq(40.0)
        ]
        lines = [
            "# M2.1 compressed pressure reranker", "",
            "- Risk coordinates are fit as controls but excluded from trading-score contributions.",
            "- The matched event cohort uses the same T+1-open, ten-day sleeves, constraints and cost.",
            "- Capital is divided by the actual executable selection count.",
            "- Historical development diagnostic only; Top 5 was selected after prior inspection.",
            "", "## Aggregate OOF IC", "",
            aggregate.to_markdown(index=False, floatfmt=".6f"),
            "", "## Top 5 at 40 bps", "",
            "|Variant|Annual return|Matched cohort excess|Universe excess|Sharpe|Max drawdown|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in primary.itertuples():
            lines.append(
                f"|{row.variant}|{row.annualized_return:.2%}|"
                f"{row.annualized_excess_vs_matched_cohort:.2%}|"
                f"{row.annualized_excess_return_vs_universe:.2%}|"
                f"{row.sharpe:.2f}|{row.max_drawdown:.2%}|"
            )
        lines += [
            "", "## C2 versus C0 yearly return", "",
            yearly_delta.to_markdown(index=False, floatfmt=".4f"),
            "", "## Primary concentration", "", "```json",
            json.dumps(concentration, ensure_ascii=False, indent=2), "```",
            "", "## Deterministic Gate", "",
            f"- Reranker passed: `{gate['reranker_passed']}`.",
            f"- Standalone passed: `{gate['standalone_passed']}`.",
            f"- Saved next action: `{gate['next_action']}`.", "",
        ]
        return "\n".join(lines)
