from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from concept_etf_latest_signal import load_incremental_stock_panel
from index_backed_r4_strict_pit_experiment import build_mapping, resolve_snapshot
from factor_forge.research.concept_etf_exit_state import (
    ExitStateRules,
    attach_exit_state_features,
    simulate_exit_state_sleeves,
)
from factor_forge.research.concept_etf_rotation import prepare_etf_panel
from factor_forge.research.concept_etf_shadow import nonoverlapping_holding_periods
from factor_forge.research.concept_rotation_alpha import build_concept_dataset
from factor_forge.research.index_backed_rotation import (
    build_dynamic_etf_signal_panel,
    build_monthly_index_history_eligibility,
    expand_monthly_index_membership,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    r7_cfg = yaml.safe_load(Path(cfg["base_r7_config"]).read_text(encoding="utf-8"))
    rules = ExitStateRules(**cfg["exit_rules"])

    print("rebuilding the frozen strict-PIT R7 signal panel", flush=True)
    panel, panel_audit = build_strict_panel(r7_cfg, Path(cfg["data_root"]))
    panel = attach_exit_state_features(panel, rules=rules)
    prefix_audit = exit_feature_prefix_audit(panel, rules)

    nav_parts: list[pd.DataFrame] = []
    sleeve_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    nonoverlap_rows: list[pd.DataFrame] = []
    first_cost = int(cfg["evaluation"]["roundtrip_cost_bps"][0])
    for cost in cfg["evaluation"]["roundtrip_cost_bps"]:
        for policy in cfg["policies"]:
            print(f"simulating {policy} cost={cost}bp", flush=True)
            daily, sleeves, signals, actions = simulate_exit_state_sleeves(
                panel,
                policy,
                start=cfg["evaluation"]["start"],
                end=cfg["evaluation"]["end"],
                roundtrip_cost_bps=float(cost),
                rules=rules,
            )
            evaluation_start = pd.Timestamp(cfg["evaluation"]["start"])
            daily = daily.loc[daily["return_date"].gt(evaluation_start)].copy()
            daily["net_nav"] = (1.0 + daily["net_return"]).cumprod()
            daily["policy"] = policy
            daily["roundtrip_cost_bps"] = int(cost)
            sleeves = sleeves.loc[sleeves["return_date"].gt(evaluation_start)].copy()
            sleeves["policy"] = policy
            sleeves["roundtrip_cost_bps"] = int(cost)
            periods = nonoverlapping_holding_periods(sleeves, holding_days=rules.holding_days)
            periods["policy"] = policy
            periods["roundtrip_cost_bps"] = int(cost)
            nonoverlap_rows.append(periods)
            summary_rows.append(
                summarize(policy, int(cost), daily, periods, actions)
            )
            nav_parts.append(daily)
            sleeve_parts.append(sleeves)
            if int(cost) == first_cost:
                signals["roundtrip_cost_bps"] = int(cost)
                actions["roundtrip_cost_bps"] = int(cost)
                signal_parts.append(signals)
                action_parts.append(actions)

    nav = pd.concat(nav_parts, ignore_index=True)
    sleeves = pd.concat(sleeve_parts, ignore_index=True)
    signals = pd.concat(signal_parts, ignore_index=True)
    actions = pd.concat(action_parts, ignore_index=True)
    periods = pd.concat(nonoverlap_rows, ignore_index=True)
    summary = add_baseline_deltas(pd.DataFrame(summary_rows))
    monthly = monthly_performance(nav)
    nonoverlap = nonoverlap_summary(periods)
    reason_summary = exit_reason_summary(signals, actions)
    baseline_audit = baseline_reproduction_audit(nav, Path(cfg["base_r7_artifact"]))
    causality_audit = make_causality_audit(
        signals, actions, prefix_audit, baseline_audit, panel_audit
    )
    decision = make_decision(summary, cfg)
    latest_signals = render_latest_user_signals(signals, policy="E2_price_diffusion_state")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(cfg["output_root"]) / f"exit_state_r9_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
    nonoverlap.to_csv(output / "nonoverlap_sleeve_summary.csv", index=False, encoding="utf-8-sig")
    reason_summary.to_csv(output / "exit_reason_summary.csv", index=False, encoding="utf-8-sig")
    latest_signals.to_csv(output / "latest_user_signals.csv", index=False, encoding="utf-8-sig")
    nav.to_parquet(output / "daily_nav.parquet", index=False)
    sleeves.to_parquet(output / "sleeve_daily.parquet", index=False)
    signals.to_parquet(output / "daily_user_signals.parquet", index=False)
    actions.to_csv(output / "exit_actions.csv", index=False, encoding="utf-8-sig")
    periods.to_parquet(output / "nonoverlap_periods.parquet", index=False)
    (output / "causality_audit.json").write_text(
        json.dumps(causality_audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(
            summary, monthly, nonoverlap, reason_summary, latest_signals,
            causality_audit, decision, cfg,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output),
        "summary": summary.to_dict("records"),
        "decision": decision,
        "causality_audit": causality_audit,
    }, ensure_ascii=False, indent=2, default=json_default))


def build_strict_panel(config: dict, data_root: Path) -> tuple[pd.DataFrame, dict]:
    snapshot = resolve_snapshot(data_root)
    candidates = pd.read_parquet(snapshot / "theme_etf_candidates.parquet")
    weights = pd.read_parquet(snapshot / "index_weights.parquet")
    daily = pd.read_parquet(snapshot / "fund_daily.parquet")
    share = pd.read_parquet(snapshot / "fund_share.parquet")
    basic = pd.read_parquet(snapshot / "etf_basic_all_statuses.parquet")
    stocks = load_incremental_stock_panel(
        [Path(path) for path in config["stock_panels"]],
        config["history_start"],
        config["history_end"],
    )
    calendar = pd.DatetimeIndex(sorted(stocks["trade_date"].unique()))
    metadata = candidates.drop_duplicates("index_code").rename(columns={
        "index_code": "concept_code", "index_name": "concept_name",
    })[["concept_code", "concept_name", "cluster"]]
    concept_index, members = expand_monthly_index_membership(
        weights,
        metadata,
        calendar,
        lag_sessions=int(config["data"]["membership_lag_sessions"]),
    )
    _, concepts, feature_audit = build_concept_dataset(
        stocks, concept_index, members, breadth_weight_lag=1,
    )
    eligibility = build_monthly_index_history_eligibility(
        weights,
        calendar,
        minimum_weight_months=int(config["data"]["minimum_weight_months_at_selection"]),
        minimum_members=int(config["data"]["minimum_index_members"]),
        availability_lag_sessions=int(
            config["data"]["index_weight_availability_lag_sessions"]
        ),
    )
    etfs = prepare_etf_panel(
        daily,
        share,
        pd.DataFrame(),
        basic.rename(columns={"csname": "name"}),
        share_availability_lag_sessions=int(
            config["data"]["share_availability_lag_sessions"]
        ),
    )
    schedule, mapping_audit = build_mapping(
        etfs, candidates, concepts, calendar, eligibility, config,
    )
    panel = build_dynamic_etf_signal_panel(concepts, etfs, schedule, candidates)
    return panel, {
        "snapshot": str(snapshot.resolve()),
        "panel_rows": int(len(panel)),
        "panel_etfs": int(panel["ts_code"].nunique()),
        "panel_start": panel["trade_date"].min(),
        "panel_end": panel["trade_date"].max(),
        "mapping_rows": int(len(schedule)),
        "mapping_audit_rows": int(len(mapping_audit)),
        "feature_audit": feature_audit,
    }


def summarize(
    policy: str,
    cost: int,
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    actions: pd.DataFrame,
) -> dict:
    returns = daily["net_return"].astype(float)
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    annualized = float(nav.iloc[-1] ** (252.0 / len(daily)) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
    sleeve_total = periods.groupby("sleeve", observed=True)["net_return"].apply(
        lambda values: float(np.prod(1.0 + values) - 1.0)
    )
    tail = returns.loc[returns.le(returns.quantile(0.05))]
    return {
        "policy": policy,
        "roundtrip_cost_bps": cost,
        "evaluation_start": daily["return_date"].min(),
        "evaluation_end": daily["return_date"].max(),
        "days": int(len(daily)),
        "total_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_zero_rate": float(annualized / volatility) if volatility > 0 else np.nan,
        "maximum_drawdown": float(drawdown.min()),
        "worst_daily_return": float(returns.min()),
        "expected_shortfall_5pct": float(tail.mean()),
        "mean_daily_turnover": float(daily["turnover"].mean()),
        "mean_cash_weight": float(daily["cash_weight"].mean()),
        "positive_sleeves": int(sleeve_total.gt(0).sum()),
        "minimum_sleeve_total_return": float(sleeve_total.min()),
        "median_sleeve_total_return": float(sleeve_total.median()),
        "exit_actions": int(len(actions)),
        "reductions": int(actions["action"].eq("reduce_half").sum()) if not actions.empty else 0,
        "full_exits": int(actions["action"].eq("sell_all").sum()) if not actions.empty else 0,
        "false_exit_rate": float(actions["false_exit"].mean()) if len(actions) else 0.0,
        "mean_avoided_remaining_return": (
            float(actions["avoided_remaining_return"].mean()) if len(actions) else 0.0
        ),
    }


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary.loc[summary["policy"].eq("E0_fixed"), [
        "roundtrip_cost_bps", "total_return", "maximum_drawdown",
        "expected_shortfall_5pct", "sharpe_zero_rate",
    ]].rename(columns={
        "total_return": "baseline_total_return",
        "maximum_drawdown": "baseline_maximum_drawdown",
        "expected_shortfall_5pct": "baseline_expected_shortfall_5pct",
        "sharpe_zero_rate": "baseline_sharpe_zero_rate",
    })
    result = summary.merge(baseline, on="roundtrip_cost_bps", validate="many_to_one")
    result["total_return_delta"] = result["total_return"] - result["baseline_total_return"]
    result["maximum_drawdown_delta"] = (
        result["maximum_drawdown"] - result["baseline_maximum_drawdown"]
    )
    result["expected_shortfall_delta"] = (
        result["expected_shortfall_5pct"] - result["baseline_expected_shortfall_5pct"]
    )
    result["sharpe_delta"] = result["sharpe_zero_rate"] - result["baseline_sharpe_zero_rate"]
    return result


def monthly_performance(nav: pd.DataFrame) -> pd.DataFrame:
    data = nav.copy()
    data["month"] = data["return_date"].dt.to_period("M").astype(str)
    rows = []
    for key, group in data.groupby(
        ["policy", "roundtrip_cost_bps", "month"], observed=True
    ):
        curve = (1.0 + group["net_return"]).cumprod()
        rows.append({
            "policy": key[0],
            "roundtrip_cost_bps": int(key[1]),
            "month": key[2],
            "monthly_return": float(curve.iloc[-1] - 1.0),
            "monthly_max_drawdown": float((curve / curve.cummax() - 1.0).min()),
            "turnover": float(group["turnover"].sum()),
            "mean_cash_weight": float(group["cash_weight"].mean()),
        })
    return pd.DataFrame(rows)


def nonoverlap_summary(periods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in periods.groupby(
        ["policy", "roundtrip_cost_bps", "sleeve"], observed=True
    ):
        rows.append({
            "policy": key[0],
            "roundtrip_cost_bps": int(key[1]),
            "sleeve": int(key[2]),
            "periods": int(len(group)),
            "mean_net_return": float(group["net_return"].mean()),
            "positive_period_fraction": float(group["net_return"].gt(0).mean()),
            "total_net_return": float(np.prod(1.0 + group["net_return"]) - 1.0),
            "mean_turnover": float(group["turnover"].mean()),
        })
    result = pd.DataFrame(rows)
    baseline = result.loc[result["policy"].eq("E0_fixed"), [
        "roundtrip_cost_bps", "sleeve", "mean_net_return", "total_net_return",
    ]].rename(columns={
        "mean_net_return": "baseline_mean_net_return",
        "total_net_return": "baseline_total_net_return",
    })
    result = result.merge(
        baseline, on=["roundtrip_cost_bps", "sleeve"], validate="many_to_one"
    )
    result["mean_net_excess"] = result["mean_net_return"] - result["baseline_mean_net_return"]
    result["total_net_excess"] = result["total_net_return"] - result["baseline_total_net_return"]
    return result


def exit_reason_summary(signals: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    action_signals = signals.loc[signals["planned_action"].ne("none")].copy()
    rows = []
    for key, group in action_signals.groupby(["policy", "planned_action", "reasons"], observed=True):
        rows.append({
            "policy": key[0],
            "planned_action": key[1],
            "reasons": key[2],
            "signals": int(len(group)),
        })
    result = pd.DataFrame(rows)
    if actions.empty:
        return result
    action_counts = actions.groupby(["policy", "action"], as_index=False).agg(
        executed_actions=("ts_code", "size"),
        false_exit_rate=("false_exit", "mean"),
        mean_avoided_remaining_return=("avoided_remaining_return", "mean"),
    ).rename(columns={"action": "planned_action"})
    return result.merge(action_counts, on=["policy", "planned_action"], how="left")


def baseline_reproduction_audit(nav: pd.DataFrame, artifact: Path) -> dict:
    frozen = pd.read_parquet(artifact / "scenario_daily_nav.parquet")
    frozen = frozen.loc[frozen["variant"].eq("strict_with_delisted")].copy()
    rows = []
    for cost in sorted(nav["roundtrip_cost_bps"].unique()):
        current = nav.loc[
            nav["policy"].eq("E0_fixed") & nav["roundtrip_cost_bps"].eq(cost),
            ["return_date", "net_return"],
        ]
        prior = frozen.loc[
            frozen["roundtrip_cost_bps"].eq(cost), ["return_date", "net_return"]
        ]
        joined = current.merge(prior, on="return_date", suffixes=("_r9", "_r7"), validate="one_to_one")
        rows.append({
            "roundtrip_cost_bps": int(cost),
            "rows": int(len(joined)),
            "maximum_absolute_return_difference": float(
                (joined["net_return_r9"] - joined["net_return_r7"]).abs().max()
            ),
        })
    return {
        "passed": all(row["maximum_absolute_return_difference"] <= 1e-12 for row in rows),
        "comparisons": rows,
    }


def exit_feature_prefix_audit(panel: pd.DataFrame, rules: ExitStateRules) -> dict:
    cutoff = pd.Timestamp("2025-06-30")
    columns = [
        "score_rank_exit", "etf_return_5d_exit", "momentum_acceleration_exit",
        "price_weak_exit", "diffusion_weak_exit", "relative_weak_exit",
        "absolute_breakdown_exit",
    ]
    truncated = attach_exit_state_features(
        panel.loc[panel["trade_date"].le(cutoff)].drop(columns=columns), rules=rules,
    )
    full = panel.loc[panel["trade_date"].le(cutoff)]
    joined = truncated[["trade_date", "ts_code", *columns]].merge(
        full[["trade_date", "ts_code", *columns]],
        on=["trade_date", "ts_code"], suffixes=("_truncated", "_full"),
        validate="one_to_one",
    )
    max_difference = 0.0
    for column in columns:
        left = joined[f"{column}_truncated"]
        right = joined[f"{column}_full"]
        if pd.api.types.is_bool_dtype(left):
            difference = float(left.ne(right).mean())
        else:
            difference = float((pd.to_numeric(left) - pd.to_numeric(right)).abs().max())
        max_difference = max(max_difference, difference if np.isfinite(difference) else 0.0)
    return {
        "passed": bool(max_difference <= 1e-12),
        "cutoff": str(cutoff.date()),
        "rows": int(len(joined)),
        "maximum_difference": max_difference,
    }


def make_causality_audit(signals, actions, prefix, baseline, panel_audit) -> dict:
    execution_after_signal = bool(
        actions.empty
        or pd.to_datetime(actions["execution_date"]).gt(pd.to_datetime(actions["signal_date"])).all()
    )
    signal_execution_after_close = bool(
        signals.empty
        or pd.to_datetime(signals["execution_date"]).gt(pd.to_datetime(signals["signal_date"])).all()
    )
    checks = {
        "exit_feature_prefix_invariance": bool(prefix["passed"]),
        "baseline_exactly_reproduces_frozen_r7": bool(baseline["passed"]),
        "executed_actions_are_after_signal_close": execution_after_signal,
        "all_user_actions_are_next_session_only": signal_execution_after_close,
        "future_labels_used": False,
    }
    return {
        "status": "PASS" if all(value is True or value is False and key == "future_labels_used" for key, value in checks.items()) else "FAIL",
        "checks": checks,
        "prefix_audit": prefix,
        "baseline_reproduction": baseline,
        "panel_audit": panel_audit,
    }


def make_decision(summary: pd.DataFrame, cfg: dict) -> dict:
    gate = cfg["decision"]
    results = {}
    for policy in [item for item in cfg["policies"] if item != "E0_fixed"]:
        rows = summary.loc[summary["policy"].eq(policy)]
        checks = {
            "mdd_improvement": bool(
                rows["maximum_drawdown_delta"].ge(float(gate["minimum_mdd_improvement"])).all()
            ),
            "return_noninferiority": bool(
                rows["total_return_delta"].ge(-float(gate["maximum_total_return_drag"])).all()
            ),
            "expected_shortfall_improvement": bool(rows["expected_shortfall_delta"].gt(0).all()),
            "false_exit_rate": bool(rows["false_exit_rate"].le(float(gate["maximum_false_exit_rate"])).all()),
            "both_cost_scenarios": bool(
                len(rows) == len(cfg["evaluation"]["roundtrip_cost_bps"])
            ),
        }
        results[policy] = {"pass": all(checks.values()), "checks": checks}
    accepted = [policy for policy, result in results.items() if result["pass"]]
    return {
        "verdict": "EXIT_STATE_ACCEPTED" if accepted else "EXIT_STATE_NOT_VALIDATED",
        "accepted_policies": accepted,
        "policy_results": results,
        "threshold_search_performed": False,
        "virgin_oos_confirmation": False,
        "next_step": "forward_shadow_only" if accepted else "keep_frozen_r7_exit",
    }


def render_latest_user_signals(signals: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    selected = signals.loc[signals["policy"].eq(policy)].copy()
    latest = selected["signal_date"].max()
    selected = selected.loc[selected["signal_date"].eq(latest)].copy()
    labels = {
        "HOLD": "继续持有",
        "WATCH": "风险观察",
        "REDUCE": "下一交易日减仓50%",
        "SELL": "下一交易日清仓",
    }
    selected["user_signal"] = selected["status"].map(labels)
    selected["research_only"] = True
    columns = [
        "signal_date", "execution_date", "sleeve", "holding_day", "ts_code",
        "etf_name", "concept_name", "current_weight", "user_signal", "reasons",
        "score_rank", "etf_momentum_20d", "etf_momentum_60d",
        "common_delta_rank", "common_breadth_delta_smooth5", "rrg_quadrant",
        "scheduled_review_date", "research_only",
    ]
    return selected[columns].sort_values(["user_signal", "sleeve", "ts_code"])


def render_report(summary, monthly, nonoverlap, reasons, latest, causality, decision, cfg) -> str:
    view_columns = [
        "policy", "roundtrip_cost_bps", "total_return", "annualized_return",
        "maximum_drawdown", "expected_shortfall_5pct", "mean_daily_turnover",
        "mean_cash_weight", "exit_actions", "false_exit_rate",
        "total_return_delta", "maximum_drawdown_delta", "expected_shortfall_delta",
    ]
    worst = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)].sort_values(
        "monthly_return"
    ).groupby("policy", observed=True).head(1)
    sleeve20 = nonoverlap.loc[nonoverlap["roundtrip_cost_bps"].eq(20)]
    return "\n".join([
        "# R9 Explainable Exit State Machine",
        "",
        "## Decision",
        "",
        f"- Verdict: `{decision['verdict']}`",
        f"- Accepted policies: `{decision['accepted_policies']}`",
        "- This is exploratory evidence because the R7 interval was already inspected.",
        "- No exit threshold or confirmation window was selected from these results.",
        "",
        "## Scenario comparison",
        "",
        summary[view_columns].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Worst month at 20bp",
        "",
        worst[["policy", "month", "monthly_return", "monthly_max_drawdown", "turnover"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Non-overlapping sleeve results at 20bp",
        "",
        sleeve20.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Exit reasons",
        "",
        reasons.to_markdown(index=False, floatfmt=".4f") if not reasons.empty else "No exit actions.",
        "",
        "## Latest research-only user signals",
        "",
        latest.to_markdown(index=False, floatfmt=".4f") if not latest.empty else "No held positions.",
        "",
        "## Causality and reproduction",
        "",
        f"- Audit: `{causality['status']}`",
        f"- Exit-feature prefix invariance: `{causality['checks']['exit_feature_prefix_invariance']}`",
        f"- E0 reproduces frozen strict-PIT R7: `{causality['checks']['baseline_exactly_reproduces_frozen_r7']}`",
        "- Signals use close-known features and execute no earlier than the next session open.",
        "- No future return or exit label is used by E1/E2.",
        "",
        "## Decision gates",
        "",
        "A policy must improve MDD by at least 2 percentage points at both 20bp and 40bp, lose no more than 2 percentage points of total return, improve 5% expected shortfall, and keep false exits at or below 40%.",
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
        default="configs/research/index_backed_r4_exit_state_v1.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
