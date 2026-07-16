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

from factor_forge.config import ExecutionConstraints, load_project

from .config import StrictModel


ENGINE_VERSION = "post_impulse_m2_path_v2"
VARIANTS = ["c0_event", "c1_raw_pressure", "c2_compressed_pressure"]
PAIRS = [
    ("c1_raw_pressure", "c0_event", "raw_pressure_vs_event"),
    ("c2_compressed_pressure", "c1_raw_pressure", "compressed_vs_raw"),
]
PANEL_COLUMNS = [
    "trade_date", "ts_code", "raw_open", "adj_open", "adj_high", "adj_low",
    "is_liquid", "is_tradeable", "is_suspended", "is_limit_up_open",
    "is_limit_down_open", "is_st", "is_delisting_period", "listing_trade_days",
    "industry_l1_code",
]


class M2PathGate(StrictModel):
    primary_pair: Literal["raw_pressure_vs_event"] = "raw_pressure_vs_event"
    primary_cost_bps: Literal[40.0] = 40.0
    minimum_positive_folds: int = Field(default=3, ge=1, le=4)
    minimum_positive_years: int = Field(default=3, ge=1, le=4)
    trim_best_fraction: Literal[0.01] = 0.01
    require_adjacent_support: bool = True


class PostImpulseM2PathConfig(StrictModel):
    version: Literal[1] = 1
    name: str = "post_impulse_m2_path_v1"
    score_run: Path
    event_run: Path
    project_config: Path = Path("configs/project.yaml")
    horizons: list[Literal[1, 2, 3, 5, 7, 10]] = Field(
        default_factory=lambda: [1, 2, 3, 5, 7, 10]
    )
    top_n: Literal[5] = 5
    cost_bps: list[Literal[20.0, 40.0, 60.0]] = Field(
        default_factory=lambda: [20.0, 40.0, 60.0]
    )
    gate: M2PathGate = Field(default_factory=M2PathGate)
    output_root: Path = Path("artifacts/post_impulse_m2_path_runs")

    @model_validator(mode="after")
    def frozen_space(self):
        if self.horizons != [1, 2, 3, 5, 7, 10]:
            raise ValueError("horizons must remain [1, 2, 3, 5, 7, 10]")
        if self.cost_bps != [20.0, 40.0, 60.0]:
            raise ValueError("cost_bps must remain [20, 40, 60]")
        return self


def load_post_impulse_m2_path_config(path: str | Path) -> PostImpulseM2PathConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PostImpulseM2PathConfig.model_validate(yaml.safe_load(handle) or {})


def _net_return(gross_return: float | pd.Series, cost_bps: float):
    half_cost = cost_bps / 2.0 / 10_000.0
    return (1.0 + gross_return) * (1.0 - half_cost) - 1.0 - half_cost


def _break_even_cost_bps(gross_return: float) -> float:
    if not np.isfinite(gross_return) or gross_return <= 0:
        return 0.0
    return float(20_000.0 * gross_return / (2.0 + gross_return))


def _trim_best_mean(values: pd.Series, fraction: float) -> float:
    sample = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    if sample.empty:
        return np.nan
    drop = max(1, math.ceil(len(sample) * fraction))
    retained = sample.iloc[drop:]
    return float(retained.mean()) if len(retained) else np.nan


def _newey_west_mean_test(values: pd.Series, lags: int = 10) -> tuple[float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(sample) < 2:
        return np.nan, np.nan
    residual = sample - sample.mean()
    long_run_variance = float(np.dot(residual, residual) / len(sample))
    for lag in range(1, min(lags, len(sample) - 1) + 1):
        covariance = float(np.dot(residual[lag:], residual[:-lag]) / len(sample))
        long_run_variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / len(sample))
    if standard_error == 0:
        return np.nan, np.nan
    statistic = float(sample.mean() / standard_error)
    p_value = float(math.erfc(abs(statistic) / math.sqrt(2.0)))
    return statistic, p_value


def _is_buyable(row: pd.Series, constraints: ExecutionConstraints) -> bool:
    return bool(
        np.isfinite(row.get("raw_open", np.nan))
        and (not constraints.exclude_suspended or not row.get("is_suspended", True))
        and (not constraints.cannot_buy_limit_up or not row.get("is_limit_up_open", False))
        and (not constraints.exclude_st or not row.get("is_st", False))
        and (
            not constraints.exclude_delisting_period
            or not row.get("is_delisting_period", False)
        )
        and row.get("listing_trade_days", 0) >= constraints.min_listing_days
    )


def _is_sellable(row: pd.Series, constraints: ExecutionConstraints) -> bool:
    return bool(
        np.isfinite(row.get("raw_open", np.nan))
        and (not constraints.exclude_suspended or not row.get("is_suspended", True))
        and (
            not constraints.cannot_sell_limit_down
            or not row.get("is_limit_down_open", False)
        )
    )


def _top_selections(oof: pd.DataFrame, top_n: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = oof.loc[
        oof["variant"].isin(VARIANTS)
        & oof["factor_value"].notna()
        & pd.to_datetime(oof["trade_date"]).le(cutoff)
    ].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["selection_rank"] = frame.groupby(
        ["variant", "trade_date"], sort=False
    )["factor_value"].rank(ascending=False, method="first")
    return frame.loc[frame["selection_rank"].le(top_n)].copy()


def _industry_forward_returns(
    panel: pd.DataFrame, horizons: list[int], constraints: ExecutionConstraints
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    frame = panel.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    codes = frame["ts_code"]
    opens = pd.to_numeric(frame["adj_open"], errors="coerce")
    entry = opens.groupby(codes, sort=False).shift(-1)
    valid = (
        frame["is_liquid"].fillna(False).astype(bool)
        & frame["is_tradeable"].fillna(False).astype(bool)
        & ~frame["is_suspended"].fillna(True).astype(bool)
        & ~frame["is_st"].fillna(True).astype(bool)
        & ~frame["is_delisting_period"].fillna(True).astype(bool)
        & frame["listing_trade_days"].ge(constraints.min_listing_days)
    )
    benchmarks = {}
    for horizon in horizons:
        exit_open = opens.groupby(codes, sort=False).shift(-(horizon + 1))
        column = f"planned_return_h{horizon}"
        frame[column] = exit_open / entry - 1.0
        benchmarks[horizon] = frame[column].where(valid).groupby(
            [frame["trade_date"], frame["industry_l1_code"]], sort=False
        ).mean()
    return frame, benchmarks


def _evaluate_selected_paths(
    selections: pd.DataFrame,
    panel: pd.DataFrame,
    benchmarks: dict[int, pd.Series],
    horizons: list[int],
    costs: list[float],
) -> pd.DataFrame:
    constraints = ExecutionConstraints()
    grouped = {}
    for code, group in panel.groupby("ts_code", sort=False):
        ordered = group.sort_values("trade_date", kind="stable").reset_index(drop=True)
        grouped[code] = (
            ordered,
            {pd.Timestamp(date): index for index, date in enumerate(ordered["trade_date"])},
        )
    rows = []
    for item in selections.itertuples(index=False):
        stock = grouped.get(item.ts_code)
        if stock is None:
            continue
        history, date_to_index = stock
        signal_index = date_to_index.get(pd.Timestamp(item.trade_date))
        if signal_index is None or signal_index + 1 >= len(history):
            continue
        entry_index = signal_index + 1
        entry = history.iloc[entry_index]
        if not _is_buyable(entry, constraints):
            continue
        entry_open = float(entry["adj_open"])
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        industry = item.industry_l1_code
        for horizon in horizons:
            planned_index = signal_index + horizon + 1
            if planned_index >= len(history):
                continue
            planned = history.iloc[planned_index]
            planned_open = float(planned["adj_open"])
            if not np.isfinite(planned_open):
                continue
            exit_index = planned_index
            while exit_index < len(history) and not _is_sellable(
                history.iloc[exit_index], constraints
            ):
                exit_index += 1
            if exit_index >= len(history):
                continue
            actual = history.iloc[exit_index]
            actual_open = float(actual["adj_open"])
            gross = actual_open / entry_open - 1.0
            planned_gross = planned_open / entry_open - 1.0
            window = history.iloc[entry_index:planned_index + 1]
            mfe = float(pd.to_numeric(window["adj_high"], errors="coerce").max() / entry_open - 1.0)
            mae = float(pd.to_numeric(window["adj_low"], errors="coerce").min() / entry_open - 1.0)
            benchmark = benchmarks[horizon].get(
                (pd.Timestamp(item.trade_date), industry), np.nan
            )
            row = {
                "event_id": item.event_id, "variant": item.variant,
                "fold": item.fold, "trade_date": pd.Timestamp(item.trade_date),
                "ts_code": item.ts_code, "industry_l1_code": industry,
                "selection_rank": item.selection_rank, "horizon": horizon,
                "entry_date": pd.Timestamp(entry["trade_date"]),
                "planned_exit_date": pd.Timestamp(planned["trade_date"]),
                "actual_exit_date": pd.Timestamp(actual["trade_date"]),
                "exit_delay_trading_days": int(exit_index - planned_index),
                "gross_return": gross, "planned_gross_return": planned_gross,
                "industry_benchmark_return": benchmark,
                "industry_excess": planned_gross - benchmark,
                "mfe": mfe, "mae": mae,
            }
            for cost in costs:
                row[f"net_return_{int(cost)}bps"] = _net_return(gross, cost)
            rows.append(row)
    return pd.DataFrame(rows)


def _variant_summaries(paths: pd.DataFrame, costs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_columns = [
        "gross_return", "industry_excess", "mfe", "mae", "exit_delay_trading_days",
        *[f"net_return_{int(cost)}bps" for cost in costs],
    ]
    daily = paths.groupby(
        ["variant", "fold", "trade_date", "horizon"], as_index=False
    )[value_columns].mean()
    rows = []
    for (variant, horizon), group in daily.groupby(["variant", "horizon"], sort=False):
        gross = float(group["gross_return"].mean())
        row = {
            "variant": variant, "horizon": int(horizon),
            "event_count": int(len(paths.loc[
                paths["variant"].eq(variant) & paths["horizon"].eq(horizon)
            ])),
            "daily_count": int(len(group)), "mean_gross_return": gross,
            "median_daily_gross_return": float(group["gross_return"].median()),
            "daily_win_rate": float(group["gross_return"].gt(0).mean()),
            "mean_industry_excess": float(group["industry_excess"].mean()),
            "mean_mfe": float(group["mfe"].mean()), "mean_mae": float(group["mae"].mean()),
            "mean_exit_delay": float(group["exit_delay_trading_days"].mean()),
            "break_even_cost_bps": _break_even_cost_bps(gross),
        }
        for cost in costs:
            column = f"net_return_{int(cost)}bps"
            row[f"mean_net_return_{int(cost)}bps"] = float(group[column].mean())
        rows.append(row)
    yearly = daily.copy()
    yearly["year"] = yearly["trade_date"].dt.year
    yearly = yearly.groupby(["variant", "year", "horizon"], as_index=False)[
        value_columns
    ].mean()
    folds = daily.groupby(["variant", "fold", "horizon"], as_index=False)[
        value_columns
    ].mean()
    return pd.DataFrame(rows), yearly, folds


def _selection_assignments(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = {
        variant: group.groupby("trade_date", sort=False)
        for variant, group in selections.groupby("variant", sort=False)
    }
    for candidate, baseline, pair in PAIRS:
        candidate_dates = {date: group for date, group in indexed[candidate]}
        baseline_dates = {date: group for date, group in indexed[baseline]}
        for date in sorted(set(candidate_dates) | set(baseline_dates)):
            candidate_group = candidate_dates.get(date, pd.DataFrame())
            baseline_group = baseline_dates.get(date, pd.DataFrame())
            candidate_ids = set(candidate_group.get("event_id", pd.Series(dtype=str)))
            baseline_ids = set(baseline_group.get("event_id", pd.Series(dtype=str)))
            roles = [
                (candidate_ids & baseline_ids, "common"),
                (candidate_ids - baseline_ids, "added"),
                (baseline_ids - candidate_ids, "dropped"),
            ]
            source = pd.concat([candidate_group, baseline_group], ignore_index=True).drop_duplicates(
                "event_id"
            ).set_index("event_id")
            for event_ids, role in roles:
                for event_id in event_ids:
                    item = source.loc[event_id]
                    rows.append({
                        "pair": pair, "candidate": candidate, "baseline": baseline,
                        "trade_date": pd.Timestamp(date), "fold": item["fold"],
                        "event_id": event_id, "role": role,
                    })
    return pd.DataFrame(rows)


def _swap_summaries(
    assignments: pd.DataFrame,
    paths: pd.DataFrame,
    costs: list[float],
    gate: M2PathGate,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_paths = paths.sort_values("variant").drop_duplicates(["event_id", "horizon"])
    columns = [
        "event_id", "horizon", "gross_return", "industry_excess",
        *[f"net_return_{int(cost)}bps" for cost in costs],
    ]
    attributed = assignments.merge(unique_paths[columns], on="event_id", how="inner")
    value_columns = columns[2:]
    role_daily = attributed.groupby(
        ["pair", "candidate", "baseline", "fold", "trade_date", "horizon", "role"],
        as_index=False,
    )[value_columns].mean()
    primary_column = f"net_return_{int(gate.primary_cost_bps)}bps"
    pivot = role_daily.pivot_table(
        index=["pair", "candidate", "baseline", "fold", "trade_date", "horizon"],
        columns="role", values=primary_column,
    ).reset_index()
    for role in ["added", "dropped", "common"]:
        if role not in pivot:
            pivot[role] = np.nan
    pivot["swap_alpha"] = pivot["added"] - pivot["dropped"]
    pivot["year"] = pivot["trade_date"].dt.year

    rows = []
    for (pair, horizon), group in pivot.groupby(["pair", "horizon"], sort=False):
        event_sample = attributed.loc[
            attributed["pair"].eq(pair) & attributed["horizon"].eq(horizon)
        ]
        added = event_sample.loc[event_sample["role"].eq("added"), primary_column]
        dropped = event_sample.loc[event_sample["role"].eq("dropped"), primary_column]
        fold_means = group.groupby("fold")["swap_alpha"].mean()
        year_means = group.groupby("year")["swap_alpha"].mean()
        nw_t, nw_p = _newey_west_mean_test(group["swap_alpha"])
        rows.append({
            "pair": pair, "horizon": int(horizon),
            "daily_count": int(group["swap_alpha"].notna().sum()),
            "mean_added_net": float(group["added"].mean()),
            "mean_dropped_net": float(group["dropped"].mean()),
            "mean_common_net": float(group["common"].mean()),
            "mean_swap_alpha": float(group["swap_alpha"].mean()),
            "newey_west_t": nw_t, "newey_west_p": nw_p,
            "positive_folds": int(fold_means.gt(0).sum()),
            "positive_years": int(year_means.gt(0).sum()),
            "trimmed_added_minus_dropped": (
                _trim_best_mean(added, gate.trim_best_fraction)
                - _trim_best_mean(dropped, gate.trim_best_fraction)
            ),
        })
    return pd.DataFrame(rows), pivot, attributed


class PostImpulseM2PathRunner:
    """Diagnose when pressure reranking earns its incremental return after entry."""

    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_post_impulse_m2_path_config(config_path)
        score_summary = json.loads((cfg.score_run / "summary.json").read_text(encoding="utf-8"))
        event_summary = json.loads((cfg.event_run / "summary.json").read_text(encoding="utf-8"))
        if score_summary["data_version"] != event_summary["data_version"]:
            raise ValueError("score and event runs use different data versions")
        digest = hashlib.sha256(
            config_path.read_bytes() + score_summary["run_id"].encode() + ENGINE_VERSION.encode()
        ).hexdigest()[:16]
        run_id = f"post_impulse_m2_path_{digest}"
        output = cfg.output_root / run_id
        summary_path = output / "summary.json"
        if summary_path.exists():
            return {**json.loads(summary_path.read_text(encoding="utf-8")), "cached": True}
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())

        oof = pd.read_parquet(cfg.score_run / "oof_predictions.parquet")
        cutoff = pd.Timestamp(score_summary["last_mature_signal"])
        selections = _top_selections(oof, cfg.top_n, cutoff)
        assignments = _selection_assignments(selections)

        project = load_project(cfg.project_config)
        panel_path = (
            Path(project.paths.data_root) / "versions" / score_summary["data_version"]
            / "curated" / "stock_daily_panel.parquet"
        )
        panel = pd.read_parquet(
            panel_path, columns=PANEL_COLUMNS,
            filters=[
                ("trade_date", ">=", selections["trade_date"].min()),
                ("trade_date", "<=", pd.Timestamp(event_summary["data_end"])),
            ],
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        prepared, benchmarks = _industry_forward_returns(
            panel, cfg.horizons, ExecutionConstraints()
        )
        paths = _evaluate_selected_paths(
            selections, prepared, benchmarks, cfg.horizons, cfg.cost_bps
        )
        variant_summary, yearly, fold_summary = _variant_summaries(paths, cfg.cost_bps)
        swap_summary, swap_daily, attributed = _swap_summaries(
            assignments, paths, cfg.cost_bps, cfg.gate
        )
        gate = self._gate(cfg, variant_summary, swap_summary)

        selections.to_parquet(output / "top5_selections.parquet", index=False)
        paths.to_parquet(output / "event_paths.parquet", index=False)
        assignments.to_csv(output / "selection_assignments.csv", index=False, encoding="utf-8-sig")
        variant_summary.to_csv(output / "path_summary.csv", index=False, encoding="utf-8-sig")
        yearly.to_csv(output / "path_yearly.csv", index=False, encoding="utf-8-sig")
        fold_summary.to_csv(output / "path_folds.csv", index=False, encoding="utf-8-sig")
        swap_summary.to_csv(output / "swap_summary.csv", index=False, encoding="utf-8-sig")
        swap_daily.to_csv(output / "swap_daily.csv", index=False, encoding="utf-8-sig")
        attributed.to_parquet(output / "swap_attributed_events.parquet", index=False)
        (output / "gate.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "report.md").write_text(
            self._report(variant_summary, swap_summary, gate), encoding="utf-8"
        )
        summary = {
            "run_id": run_id, "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "score_run": score_summary["run_id"], "data_version": score_summary["data_version"],
            "signal_cutoff": str(cutoff.date()), "selected_rows": int(len(selections)),
            "path_rows": int(len(paths)), "mechanism_gate_passed": bool(gate["mechanism_passed"]),
            "standalone_horizon_found": bool(gate["standalone_horizon_found"]),
            "proposed_horizon": gate["proposed_horizon"],
            "saved_next_action": gate["next_action"],
            "historical_path_diagnostic_only": True, "output_path": str(output.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _gate(cfg, variant_summary: pd.DataFrame, swap_summary: pd.DataFrame) -> dict:
        primary = swap_summary.loc[swap_summary["pair"].eq(cfg.gate.primary_pair)].copy()
        candidate = variant_summary.loc[
            variant_summary["variant"].eq("c1_raw_pressure")
        ].set_index("horizon")
        primary["candidate_net"] = [
            candidate.loc[horizon, f"mean_net_return_{int(cfg.gate.primary_cost_bps)}bps"]
            for horizon in primary["horizon"]
        ]
        primary["mechanism_ok"] = (
            primary["mean_swap_alpha"].gt(0)
            & primary["positive_folds"].ge(cfg.gate.minimum_positive_folds)
            & primary["positive_years"].ge(cfg.gate.minimum_positive_years)
            & primary["trimmed_added_minus_dropped"].gt(0)
        )
        ordered = list(cfg.horizons)
        adjacent = []
        mechanism_by_horizon = primary.set_index("horizon")["mechanism_ok"].to_dict()
        for horizon in primary["horizon"]:
            index = ordered.index(int(horizon))
            neighbors = []
            if index > 0:
                neighbors.append(ordered[index - 1])
            if index + 1 < len(ordered):
                neighbors.append(ordered[index + 1])
            adjacent.append(any(mechanism_by_horizon.get(value, False) for value in neighbors))
        primary["adjacent_support"] = adjacent
        primary["standalone_ok"] = (
            primary["mechanism_ok"]
            & primary["candidate_net"].gt(0)
            & (
                primary["adjacent_support"]
                if cfg.gate.require_adjacent_support else True
            )
        )
        qualified = sorted(primary.loc[primary["standalone_ok"], "horizon"].astype(int))
        mechanism = bool(primary["mechanism_ok"].any())
        proposed = qualified[0] if qualified else None
        if proposed is not None:
            action = f"freeze_h{proposed}_for_forward_shadow_only"
        elif mechanism:
            action = "retain_m2_reranker_without_standalone_exit"
        else:
            action = "stop_post_impulse_path_expansion"
        return {
            "mechanism_passed": mechanism,
            "standalone_horizon_found": proposed is not None,
            "proposed_horizon": proposed,
            "horizon_checks": primary.to_dict(orient="records"),
            "next_action": action,
        }

    @staticmethod
    def _report(variant_summary, swap_summary, gate) -> str:
        primary = swap_summary.loc[swap_summary["pair"].eq("raw_pressure_vs_event")]
        lines = [
            "# M2-PATH pressure reranking realization path", "",
            "- Existing OOF scores only; no refit and no feature or threshold search.",
            "- Top 5 enters at T+1 open and exits after 1/2/3/5/7/10 trading days.",
            "- Buy/sell constraints and deferred exits are applied; primary cost is 40 bps.",
            "- Historical mechanism diagnostic only; any proposed horizon requires later unseen data.",
            "", "## Variant path summary", "",
            variant_summary.to_markdown(index=False, floatfmt=".6f"),
            "", "## Raw-pressure swap attribution", "",
            primary.to_markdown(index=False, floatfmt=".6f"),
            "", "## Deterministic decision", "",
            f"- Mechanism passed: `{gate['mechanism_passed']}`.",
            f"- Standalone horizon found: `{gate['standalone_horizon_found']}`.",
            f"- Proposed horizon: `{gate['proposed_horizon']}`.",
            f"- Saved next action: `{gate['next_action']}`.", "",
        ]
        return "\n".join(lines)
