from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    r4_dir = Path(cfg["r4_artifact"])
    timing_dir = Path(cfg["timing_artifact"])
    r4 = pd.read_parquet(r4_dir / "scenario_daily_nav.parquet")
    r4["return_date"] = pd.to_datetime(r4["return_date"])
    r4 = r4.loc[
        r4["variant"].eq(cfg["r4_variant"])
        & r4["roundtrip_cost_bps"].isin(cfg["base_roundtrip_cost_bps"])
    ].copy()
    if r4.empty:
        raise RuntimeError("the frozen R7 artifact has no requested baseline rows")

    timing_raw = pd.read_csv(timing_dir / "timing_position_daily.csv")
    timing_raw["trade_date"] = pd.to_datetime(timing_raw["trade_date"])
    timing_summary = json.loads((timing_dir / "summary.json").read_text(encoding="utf-8"))
    stable_cfg = yaml.safe_load(Path(cfg["timing_stable_factor_config"]).read_text(encoding="utf-8"))
    timing_dataset_cfg = yaml.safe_load(
        (Path(timing_summary["dataset_path"]).parent / "config.yaml").read_text(encoding="utf-8")
    )
    causality = audit_timing_signal(
        timing_raw, timing_summary, stable_cfg, timing_dataset_cfg, cfg
    )

    # Deliberately discard labels, contemporaneous returns and model predictions.
    timing = timing_raw[["trade_date", "target_position", "executed_position"]].copy()
    timing = timing.rename(columns={"trade_date": "return_date"})
    primary_start = pd.Timestamp(cfg["primary_sample"]["start"])
    coverage_base = r4.loc[
        r4["roundtrip_cost_bps"].eq(cfg["base_roundtrip_cost_bps"][0])
        & r4["return_date"].ge(primary_start)
    ]
    internal_gap_dates = coverage_base.loc[
        coverage_base["return_date"].le(timing["return_date"].max())
        & ~coverage_base["return_date"].isin(timing["return_date"]),
        "return_date",
    ]
    trailing_dates = coverage_base.loc[
        coverage_base["return_date"].gt(timing["return_date"].max()), "return_date"
    ]
    causality.update({
        "internal_missing_inference_dates_held_constant": [
            str(item.date()) for item in internal_gap_dates
        ],
        "internal_missing_inference_count": int(len(internal_gap_dates)),
        "trailing_r4_dates_excluded_count": int(len(trailing_dates)),
        "trailing_r4_start": (
            str(trailing_dates.min().date()) if not trailing_dates.empty else None
        ),
        "stale_signal_extended_beyond_artifact_end": False,
    })

    all_daily: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    sample_specs = [
        {
            "name": cfg["primary_sample"]["name"],
            "start": pd.Timestamp(cfg["primary_sample"]["start"]),
            "end": timing["return_date"].max(),
            "is_primary": True,
        },
        {
            "name": cfg["diagnostic_sample"]["name"],
            "start": pd.Timestamp(cfg["diagnostic_sample"]["start"]),
            "end": pd.Timestamp(cfg["diagnostic_sample"]["end"]),
            "is_primary": False,
        },
    ]
    for sample in sample_specs:
        for base_cost in cfg["base_roundtrip_cost_bps"]:
            base = r4.loc[
                r4["roundtrip_cost_bps"].eq(base_cost)
                & r4["return_date"].between(sample["start"], sample["end"])
            ].copy()
            expected = base.loc[base["return_date"].le(timing["return_date"].max())]
            aligned = expected.merge(timing, on="return_date", how="left", validate="one_to_one")
            aligned["timing_signal_observed"] = aligned["executed_position"].notna()
            # An isolated failed inference means holding yesterday's already
            # executed position.  The sample still stops at the artifact end,
            # so this cannot turn into an unbounded stale-signal forward fill.
            aligned[["target_position", "executed_position"]] = aligned[
                ["target_position", "executed_position"]
            ].ffill()
            if aligned[["target_position", "executed_position"]].isna().any().any():
                raise RuntimeError(f"leading timing signal gap: {sample['name']} cost={base_cost}")
            if aligned.empty:
                raise RuntimeError(f"empty aligned sample: {sample['name']} cost={base_cost}")

            previous_timing = timing.loc[timing["return_date"].lt(aligned["return_date"].min())].tail(1)
            previous_position = (
                float(previous_timing["executed_position"].iloc[0])
                if not previous_timing.empty
                else float(aligned["executed_position"].iloc[0])
            )
            for policy in cfg["policies"]:
                day = apply_overlay(
                    aligned,
                    policy,
                    previous_position=previous_position,
                    timing_cost_bps=float(cfg["timing_one_way_cost_bps"]),
                )
                day["sample"] = sample["name"]
                day["is_primary"] = bool(sample["is_primary"])
                day["policy"] = policy["name"]
                day["base_roundtrip_cost_bps"] = int(base_cost)
                all_daily.append(day)
                summary_rows.append({
                    "sample": sample["name"],
                    "is_primary": bool(sample["is_primary"]),
                    "policy": policy["name"],
                    "base_roundtrip_cost_bps": int(base_cost),
                    **performance_metrics(day),
                })

    daily = pd.concat(all_daily, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    summary = add_deltas_and_pass_flags(summary)
    primary = summary.loc[summary["is_primary"]].copy()
    decision = make_decision(primary, cfg)
    monthly = period_performance(daily.loc[daily["is_primary"]], "M")
    yearly = period_performance(daily.loc[daily["is_primary"]], "Y")
    drawdowns = drawdown_episodes(daily.loc[daily["is_primary"]])
    position_diagnostics = timing_position_diagnostics(daily.loc[daily["is_primary"]])
    prediction_diagnostics = timing_prediction_diagnostics(
        timing_raw, pd.Timestamp(cfg["primary_sample"]["start"])
    )

    output_root = Path(cfg["output_root"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"timing_overlay_r8_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    daily.to_parquet(output / "daily_performance.parquet", index=False)
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_performance.csv", index=False, encoding="utf-8-sig")
    drawdowns.to_csv(output / "drawdown_episodes.csv", index=False, encoding="utf-8-sig")
    position_diagnostics.to_csv(
        output / "timing_position_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    prediction_diagnostics.to_csv(
        output / "timing_prediction_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    (output / "causality_audit.json").write_text(
        json.dumps(causality, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(
            summary,
            monthly,
            yearly,
            drawdowns,
            position_diagnostics,
            prediction_diagnostics,
            causality,
            decision,
            cfg,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output),
        "primary_summary": primary.to_dict("records"),
        "decision": decision,
        "causality": causality,
    }, ensure_ascii=False, indent=2, default=json_default))


def audit_timing_signal(
    timing: pd.DataFrame,
    summary: dict,
    stable_cfg: dict,
    dataset_cfg: dict,
    experiment_cfg: dict,
) -> dict:
    required = {"trade_date", "target_position", "executed_position"}
    if missing := required - set(timing.columns):
        raise ValueError(f"timing artifact missing columns: {sorted(missing)}")
    ordered = timing.sort_values("trade_date").reset_index(drop=True)
    if ordered["trade_date"].duplicated().any():
        raise ValueError("timing artifact has duplicate trade dates")
    expected_execution = ordered["target_position"].shift(1).fillna(0.0).clip(0.0, 1.0)
    execution_lag_error = float((ordered["executed_position"] - expected_execution).abs().max())
    if execution_lag_error > 1e-12:
        raise RuntimeError(f"timing execution is not target_position.shift(1): {execution_lag_error}")
    oos_start = pd.Timestamp(experiment_cfg["primary_sample"]["start"])
    test_start = pd.Timestamp(summary["test_start"])
    train_end = pd.Timestamp(summary["train_end"])
    selection_end = pd.Timestamp(stable_cfg["selection_end_date"])
    checks = {
        "oos_not_before_model_test": bool(oos_start >= test_start),
        "train_ends_before_oos": bool(train_end < oos_start),
        "factor_selection_ends_before_oos": bool(selection_end < oos_start),
        "feature_data_lag_at_least_one_session": bool(
            int(dataset_cfg["features"]["data_lag"]) >= 1
        ),
        "executed_position_is_one_session_lagged": bool(execution_lag_error <= 1e-12),
        "positions_within_zero_one": bool(
            ordered[["target_position", "executed_position"]].ge(0).all().all()
            and ordered[["target_position", "executed_position"]].le(1).all().all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"timing causality audit failed: {checks}")
    oos = ordered.loc[ordered["trade_date"].ge(oos_start)]
    return {
        "status": "PASS",
        "checks": checks,
        "model_train_end": summary["train_end"],
        "model_test_start": summary["test_start"],
        "stable_factor_selection_end": stable_cfg["selection_end_date"],
        "feature_data_lag_sessions": int(dataset_cfg["features"]["data_lag"]),
        "timing_index_code": dataset_cfg["features"]["index_code"],
        "timing_benchmark_code": dataset_cfg["features"].get("benchmark_code"),
        "timing_label": summary["label_column"],
        "timing_artifact_start": str(ordered["trade_date"].min().date()),
        "timing_artifact_end": str(ordered["trade_date"].max().date()),
        "strict_oos_rows": int(len(oos)),
        "strict_oos_average_executed_position": float(oos["executed_position"].mean()),
        "maximum_execution_lag_error": execution_lag_error,
        "overlay_input_columns": ["trade_date", "target_position", "executed_position"],
        "labels_or_same_day_returns_used_by_overlay": False,
    }


def policy_exposure(position: pd.Series, policy: dict) -> pd.Series:
    signal = pd.to_numeric(position, errors="raise").clip(0.0, 1.0)
    if policy["kind"] == "constant":
        return pd.Series(float(policy["exposure"]), index=position.index)
    if policy["kind"] == "linear_floor":
        floor = float(policy["floor"])
        return floor + (1.0 - floor) * signal
    if policy["kind"] == "tail_guard":
        return pd.Series(
            np.where(
                signal.le(float(policy["weak_signal_max"])),
                float(policy["defensive_exposure"]),
                float(policy["normal_exposure"]),
            ),
            index=position.index,
            dtype=float,
        )
    raise ValueError(f"unknown timing policy kind: {policy['kind']}")


def apply_overlay(
    aligned: pd.DataFrame,
    policy: dict,
    *,
    previous_position: float,
    timing_cost_bps: float,
) -> pd.DataFrame:
    day = aligned.sort_values("return_date").reset_index(drop=True).copy()
    day["exposure"] = policy_exposure(day["executed_position"], policy)
    previous_exposure = float(policy_exposure(pd.Series([previous_position]), policy).iloc[0])
    day["timing_turnover"] = day["exposure"].diff().abs()
    day.loc[0, "timing_turnover"] = abs(float(day.loc[0, "exposure"]) - previous_exposure)
    day["timing_cost_drag"] = day["timing_turnover"] * timing_cost_bps / 10_000.0
    # Treat R4 as the underlying strategy sleeve: its already-costed return is
    # scaled by the timing allocation, while timing allocation changes pay an
    # additional one-way overlay cost.
    day["overlay_return"] = day["exposure"] * day["net_return"] - day["timing_cost_drag"]
    day["overlay_nav"] = (1.0 + day["overlay_return"]).cumprod()
    return day


def performance_metrics(day: pd.DataFrame) -> dict:
    ret = day["overlay_return"].astype(float)
    nav = (1.0 + ret).cumprod()
    dd = nav / nav.cummax() - 1.0
    total = float(nav.iloc[-1] - 1.0)
    annual = float(nav.iloc[-1] ** (252.0 / len(day)) - 1.0)
    volatility = float(ret.std(ddof=1) * np.sqrt(252.0))
    maximum_drawdown = float(dd.min())
    return {
        "start_date": day["return_date"].min(),
        "end_date": day["return_date"].max(),
        "days": int(len(day)),
        "total_return": total,
        "annualized_return": annual,
        "annualized_volatility": volatility,
        "sharpe_zero_rate": float(annual / volatility) if volatility > 0 else np.nan,
        "maximum_drawdown": maximum_drawdown,
        "calmar": float(annual / abs(maximum_drawdown)) if maximum_drawdown < 0 else np.nan,
        "mean_exposure": float(day["exposure"].mean()),
        "mean_daily_timing_turnover": float(day["timing_turnover"].mean()),
        "timing_cost_drag_sum": float(day["timing_cost_drag"].sum()),
        "positive_day_fraction": float(ret.gt(0).mean()),
    }


def add_deltas_and_pass_flags(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    baseline = result.loc[result["policy"].eq("no_timing"), [
        "sample", "base_roundtrip_cost_bps", "total_return", "annualized_return",
        "maximum_drawdown", "sharpe_zero_rate", "calmar",
    ]].rename(columns={
        "total_return": "baseline_total_return",
        "annualized_return": "baseline_annualized_return",
        "maximum_drawdown": "baseline_maximum_drawdown",
        "sharpe_zero_rate": "baseline_sharpe_zero_rate",
        "calmar": "baseline_calmar",
    })
    result = result.merge(baseline, on=["sample", "base_roundtrip_cost_bps"], validate="many_to_one")
    result["total_return_delta"] = result["total_return"] - result["baseline_total_return"]
    result["annualized_return_delta"] = result["annualized_return"] - result["baseline_annualized_return"]
    result["maximum_drawdown_delta"] = result["maximum_drawdown"] - result["baseline_maximum_drawdown"]
    result["sharpe_delta"] = result["sharpe_zero_rate"] - result["baseline_sharpe_zero_rate"]
    result["simultaneous_return_drawdown_improvement"] = (
        result["total_return_delta"].gt(0) & result["maximum_drawdown_delta"].gt(0)
    )
    return result


def make_decision(primary: pd.DataFrame, cfg: dict) -> dict:
    policies = [item["name"] for item in cfg["policies"] if item["name"] != "no_timing"]
    policy_results = {}
    for policy in policies:
        rows = primary.loc[primary["policy"].eq(policy)].sort_values("base_roundtrip_cost_bps")
        passed = bool(
            len(rows) == len(cfg["base_roundtrip_cost_bps"])
            and rows["simultaneous_return_drawdown_improvement"].all()
        )
        policy_results[policy] = {
            "pass": passed,
            "cost_scenarios_passed": int(rows["simultaneous_return_drawdown_improvement"].sum()),
            "cost_scenarios_required": int(len(cfg["base_roundtrip_cost_bps"])),
        }
    accepted = [name for name, value in policy_results.items() if value["pass"]]
    return {
        "status": "TIMING_ACCEPTED" if accepted else "TIMING_REJECTED",
        "accepted_policies": accepted,
        "policy_results": policy_results,
        "primary_rule": "strict 2025+ OOS total return and MDD must both improve at 20bp and 40bp R4 costs",
        "timing_model_refit": False,
        "r4_model_refit": False,
        "parameter_search_performed": False,
    }


def period_performance(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    data = daily.copy()
    if frequency == "M":
        data["period"] = data["return_date"].dt.to_period("M").astype(str)
    elif frequency == "Y":
        data["period"] = data["return_date"].dt.year.astype(str)
    else:
        raise ValueError(frequency)
    rows = []
    keys = ["sample", "policy", "base_roundtrip_cost_bps", "period"]
    for key, group in data.groupby(keys, observed=True):
        nav = (1.0 + group["overlay_return"]).cumprod()
        rows.append({
            **dict(zip(keys, key, strict=True)),
            "return": float(nav.iloc[-1] - 1.0),
            "maximum_drawdown": float((nav / nav.cummax() - 1.0).min()),
            "mean_exposure": float(group["exposure"].mean()),
            "timing_turnover": float(group["timing_turnover"].sum()),
            "days": int(len(group)),
        })
    return pd.DataFrame(rows)


def drawdown_episodes(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["sample", "policy", "base_roundtrip_cost_bps"]
    for key, group in daily.groupby(keys, observed=True):
        group = group.sort_values("return_date").reset_index(drop=True)
        nav = (1.0 + group["overlay_return"]).cumprod()
        running_peak = nav.cummax()
        dd = nav / running_peak - 1.0
        trough_idx = int(dd.idxmin())
        peak_idx = int(nav.iloc[: trough_idx + 1].idxmax())
        recovered = np.flatnonzero(nav.iloc[trough_idx + 1 :].to_numpy() >= nav.iloc[peak_idx])
        recovery_date = (
            group.iloc[trough_idx + 1 + int(recovered[0])]["return_date"]
            if len(recovered)
            else pd.NaT
        )
        rows.append({
            **dict(zip(keys, key, strict=True)),
            "peak_date": group.iloc[peak_idx]["return_date"],
            "trough_date": group.iloc[trough_idx]["return_date"],
            "recovery_date": recovery_date,
            "maximum_drawdown": float(dd.iloc[trough_idx]),
        })
    return pd.DataFrame(rows)


def timing_position_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    base = daily.loc[
        daily["policy"].eq("no_timing")
        & daily["base_roundtrip_cost_bps"].eq(20)
    ].copy()
    base["year"] = base["return_date"].dt.year
    rows = []
    for (year, position), group in base.groupby(
        ["year", "executed_position"], observed=True
    ):
        selected_nav = float((1.0 + group["net_return"]).prod())
        rows.append({
            "year": int(year),
            "executed_position": float(position),
            "days": int(len(group)),
            "year_day_fraction": float(len(group) / (base["year"].eq(year).sum())),
            "mean_r4_net_return": float(group["net_return"].mean()),
            "sum_r4_net_return": float(group["net_return"].sum()),
            "compounded_selected_day_r4_return": selected_nav - 1.0,
        })
    return pd.DataFrame(rows)


def timing_prediction_diagnostics(timing: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    data = timing.loc[timing["trade_date"].ge(oos_start)].copy()
    data["year"] = data["trade_date"].dt.year
    return data.groupby("year", as_index=False).agg(
        days=("trade_date", "size"),
        mean_prediction=("prediction", "mean"),
        std_prediction=("prediction", "std"),
        minimum_prediction=("prediction", "min"),
        maximum_prediction=("prediction", "max"),
        mean_executed_position=("executed_position", "mean"),
        zero_position_fraction=("executed_position", lambda item: float(item.eq(0.0).mean())),
    )


def render_report(
    summary,
    monthly,
    yearly,
    drawdowns,
    position_diagnostics,
    prediction_diagnostics,
    causality,
    decision,
    cfg,
) -> str:
    primary = summary.loc[summary["is_primary"]].copy()
    columns = [
        "policy", "base_roundtrip_cost_bps", "total_return", "annualized_return",
        "maximum_drawdown", "sharpe_zero_rate", "calmar", "mean_exposure",
        "total_return_delta", "maximum_drawdown_delta",
        "simultaneous_return_drawdown_improvement",
    ]
    monthly_primary = monthly.loc[monthly["base_roundtrip_cost_bps"].eq(20)].copy()
    worst_month = monthly_primary.sort_values("return").groupby("policy", observed=True).head(1)
    return "\n".join([
        "# R8 R7 + Timing Overlay",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Accepted policies: `{decision['accepted_policies']}`",
        "- R7 and timing artifacts were frozen; no model refit or threshold search was performed.",
        "- Primary evidence begins at 2025-01-01, after the timing model and factor selection cutoffs.",
        "",
        "## Strict OOS comparison",
        "",
        primary[columns].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Yearly performance",
        "",
        yearly.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Worst month at 20bp R4 cost",
        "",
        worst_month[["policy", "period", "return", "maximum_drawdown", "mean_exposure"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Maximum drawdown episode",
        "",
        drawdowns.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Timing calibration diagnostics",
        "",
        prediction_diagnostics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "R4 return conditional on the timing model's executed position (20bp R4 cost):",
        "",
        position_diagnostics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Causality audit",
        "",
        f"- Status: `{causality['status']}`",
        f"- Timing train end: `{causality['model_train_end']}`; test start: `{causality['model_test_start']}`.",
        f"- Stable-factor selection end: `{causality['stable_factor_selection_end']}`.",
        f"- Feature availability lag: `{causality['feature_data_lag_sessions']}` session.",
        f"- Original target: `{causality['timing_index_code']}` / `{causality['timing_label']}`; benchmark: `{causality['timing_benchmark_code']}`.",
        f"- Signal coverage ends: `{causality['timing_artifact_end']}`.",
        f"- Internal missing inference sessions held at the prior executed position: `{causality['internal_missing_inference_count']}`.",
        f"- Later R4 sessions excluded rather than stale-filled: `{causality['trailing_r4_dates_excluded_count']}`.",
        "- The overlay reads only date, target position, and one-session-lagged executed position; labels and same-day market returns are not inputs.",
        "",
        "## Interpretation limits",
        "",
        "- Timing is applied to R4 as an underlying strategy sleeve. R4 net returns are scaled by exposure and timing changes pay the configured one-way overlay cost.",
        "- The pre-2025 result is training-overlap diagnostic evidence only and is excluded from the decision.",
        "- The timing artifact currently ends before the R7 artifact; dates after timing coverage are not silently forward-filled.",
        "- This is a single fixed OOS path, not sufficient by itself for live-capital authorization.",
        "",
        f"Config: `{cfg['name']}`",
        "",
    ])


def json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/research/index_backed_r4_timing_overlay_v1.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
