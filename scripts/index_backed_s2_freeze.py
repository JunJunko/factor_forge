from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from concept_etf_latest_signal import load_incremental_stock_panel
from factor_forge.research.concept_etf_rotation import build_etf_signal_panel, prepare_etf_panel
from factor_forge.research.concept_etf_shadow import (
    CASH,
    simulate_staggered_sleeves,
    staggered_target_weights,
)
from factor_forge.research.concept_first_rotation import (
    CONCEPT_FEATURES,
    build_concept_first_features,
)
from factor_forge.research.concept_rotation_alpha import build_concept_dataset
from factor_forge.research.concept_state_residual_rotation import (
    StateResidualRules,
    attach_state_residual_scores_to_etfs,
    fit_frozen_s2_model,
    fit_state_residual_walk_forward,
    score_frozen_s2_model,
    within_state_oof_diagnostics,
)
from factor_forge.research.index_backed_rotation import expand_monthly_index_membership


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    snapshot = resolve_snapshot(Path(args.data_root), config["history_end"])
    mapping = pd.read_parquet(snapshot / "exact_index_etf_mapping.parquet")
    weights = pd.read_parquet(snapshot / "index_weights.parquet")
    daily = pd.read_parquet(snapshot / "fund_daily.parquet")
    share = pd.read_parquet(snapshot / "fund_share.parquet")
    etf_basic = pd.read_parquet(snapshot / "etf_basic.parquet")

    print("loading 720+ session stock history", flush=True)
    stocks = load_incremental_stock_panel(
        [Path(path) for path in config["stock_panels"]],
        config["history_start"], config["history_end"],
    )
    calendar = pd.DatetimeIndex(sorted(stocks["trade_date"].unique()))
    print("expanding lagged monthly index memberships", flush=True)
    concept_index, members = expand_monthly_index_membership(
        weights, mapping, calendar,
        lag_sessions=int(config["data"]["membership_lag_sessions"]),
    )
    print(
        f"daily memberships dates={members['trade_date'].nunique()} rows={len(members)}",
        flush=True,
    )
    print("building causal diffusion and state-residual features", flush=True)
    _, concept_raw, feature_audit = build_concept_dataset(
        stocks, concept_index, members, breadth_weight_lag=1,
    )
    concepts = build_concept_first_features(concept_raw)
    rules = rules_from_config(config)
    history_days = int(concepts["trade_date"].nunique())
    minimum_history = int(config["freeze"]["minimum_history_days"])
    if history_days < minimum_history:
        raise RuntimeError(f"history gate failed: {history_days} < {minimum_history}")

    print("running development-only walk-forward S0/S2 comparison", flush=True)
    oof_scores, _, _, fold_audit = fit_state_residual_walk_forward(
        concepts, start=str(concepts["trade_date"].min().date()),
        end=config["training_cutoff"], rules=rules,
    )
    if oof_scores.empty:
        raise RuntimeError("index-backed S2 walk-forward returned no OOF scores")
    etf_metadata = etf_basic.rename(columns={"csname": "name"})
    etfs = prepare_etf_panel(daily, share, pd.DataFrame(), etf_metadata)
    panel, mapping_audit = build_etf_signal_panel(
        concept_raw, etfs, mapping,
        selection_cutoff=config["selection_cutoff"],
        minimum_mapping_correlation=float(config["data"]["minimum_mapping_correlation"]),
    )
    mapping_pass = mapping_audit["mapping_pass"].astype("boolean").fillna(False).astype(bool)
    passed_etfs = set(mapping_audit.loc[mapping_pass, "ts_code"].astype(str))
    frozen_mapping = mapping.loc[mapping["etf_code"].astype(str).isin(passed_etfs)].copy()
    if frozen_mapping.empty:
        raise RuntimeError("no ETF mapping passed the exact-index correlation gate")
    panel = attach_state_residual_scores_to_etfs(
        panel, oof_scores, concept_overlay_weight=rules.concept_overlay_weight,
    )
    evaluation_start = pd.Timestamp(oof_scores["trade_date"].min())
    historical_summary, historical_daily = historical_comparison(
        panel, evaluation_start, config["training_cutoff"],
        config["execution"]["roundtrip_cost_bps"],
    )
    within_ic, within_buckets = within_state_oof_diagnostics(
        concepts, oof_scores,
        policies={"R2_within_nonlinear_5d": "score_R2_within_nonlinear_5d"},
    )

    print("fitting and sealing final S2 estimator", flush=True)
    frozen_model, model_audit = fit_frozen_s2_model(
        concepts, training_cutoff=config["training_cutoff"], rules=rules,
    )
    frozen_scores = score_frozen_s2_model(concepts, frozen_model)
    forward_panel = attach_frozen_scores(panel_without_oof(panel), frozen_scores, rules.concept_overlay_weight)
    forward_panel["volatility_20d"] = forward_panel.groupby(
        "ts_code", sort=False,
    )["etf_return_1d"].transform(lambda values: values.rolling(20, min_periods=18).std(ddof=0))
    signal_date = pd.Timestamp(config["training_cutoff"])
    latest = forward_panel.loc[forward_panel["trade_date"].eq(signal_date)].copy()
    mapping_flag = latest["mapping_pass"].astype("boolean").fillna(False).astype(bool)
    concept_flag = latest["eligible_concept"].astype("boolean").fillna(False).astype(bool)
    latest = latest.loc[mapping_flag & concept_flag]
    if latest.empty:
        raise RuntimeError(f"no eligible frozen S2 scores on {signal_date.date()}")
    target = staggered_target_weights(
        latest, "R4_rank_buffer", previous_holdings=set(),
        score_column="score_S2_nonlinear_overlay",
    )
    recommendations = phased_initial_recommendation(
        latest, target, signal_date, pd.Timestamp(config["oos_start"]),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"frozen_s2_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "frozen_s2_model.joblib"
    joblib.dump(frozen_model, model_path)
    concepts.to_parquet(output / "index_concept_features.parquet", index=False)
    oof_scores.to_parquet(output / "development_oof_scores.parquet", index=False)
    fold_audit.to_csv(output / "fold_audit.csv", index=False, encoding="utf-8-sig")
    frozen_mapping.to_csv(
        output / "frozen_universe_mapping.csv", index=False, encoding="utf-8-sig",
    )
    mapping_audit.to_csv(output / "mapping_correlation_audit.csv", index=False, encoding="utf-8-sig")
    historical_summary.to_csv(output / "development_historical_summary.csv", index=False, encoding="utf-8-sig")
    historical_daily.to_parquet(output / "development_historical_daily.parquet", index=False)
    within_ic.to_csv(output / "development_within_state_ic.csv", index=False, encoding="utf-8-sig")
    within_buckets.to_csv(output / "development_within_state_buckets.csv", index=False, encoding="utf-8-sig")
    latest.sort_values("score_S2_nonlinear_overlay", ascending=False).to_csv(
        output / "first_oos_ranking.csv", index=False, encoding="utf-8-sig",
    )
    recommendations.to_csv(output / "forward_ledger.csv", index=False, encoding="utf-8-sig")
    data_audit = {
        "history_days": history_days,
        "minimum_history_days": minimum_history,
        "history_gate_passed": history_days >= minimum_history,
        "history_start": str(concepts["trade_date"].min().date()),
        "history_end": str(concepts["trade_date"].max().date()),
        "concepts": int(concepts["concept_code"].nunique()),
        "concept_rows": int(len(concepts)),
        "daily_member_rows": int(len(members)),
        "mapping_candidates": int(mapping["etf_code"].nunique()),
        "mapped_etfs": int(frozen_mapping["etf_code"].nunique()),
        "mapping_pass": int(mapping_audit["mapping_pass"].sum()),
        "clusters": int(frozen_mapping["cluster"].nunique()),
        "oof_start": str(evaluation_start.date()),
        "oof_end": str(oof_scores["trade_date"].max().date()),
        "folds": int(fold_audit["fold"].nunique()),
        "feature_audit": feature_audit,
    }
    freeze_manifest = {
        "status": "FROZEN_FORWARD_ONLY_NO_REAL_MONEY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "data_snapshot": str(snapshot.resolve()),
        "data_manifest_sha256": sha256(snapshot / "manifest.json"),
        "model_sha256": sha256(model_path),
        "mapping_sha256": sha256(output / "frozen_universe_mapping.csv"),
        "training_cutoff": config["training_cutoff"],
        "oos_start": config["oos_start"],
        "feature_columns": list(CONCEPT_FEATURES),
        "model_audit": model_audit,
        "immutable_after_freeze": config["freeze"]["immutable_after_freeze"],
        "restart_rule": "any model, feature, universe, cost, or execution change restarts OOS clock",
        "historical_results_are_untouched_oos": False,
        "orders_authorized": False,
    }
    (output / "data_audit.json").write_text(
        json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(data_audit, model_audit, historical_summary, within_ic, recommendations),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output), "data_audit": data_audit,
        "freeze_manifest": freeze_manifest,
        "first_oos_targets": recommendations.to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str))


def rules_from_config(config: dict) -> StateResidualRules:
    model = config["model"]
    return StateResidualRules(
        hgb_learning_rate=float(model["hgb_learning_rate"]),
        hgb_max_iter=int(model["hgb_max_iter"]),
        hgb_max_depth=int(model["hgb_max_depth"]),
        hgb_l2_regularization=float(model["hgb_l2_regularization"]),
        minimum_train_days=int(model["minimum_train_days"]),
        validation_days=int(model["validation_days"]),
        test_days=int(model["test_days"]),
        embargo_days=int(model["embargo_days"]),
        minimum_train_rows=int(model["minimum_train_rows"]),
        concept_overlay_weight=float(model["concept_overlay_weight"]),
        seed=int(model["seed"]),
    )


def historical_comparison(panel, start, end, costs) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, daily_parts = [], []
    policies = {
        "S0_etf_r4": "score_S0_etf_r4",
        "S2_nonlinear_overlay": "score_S2_nonlinear_overlay",
    }
    for cost in costs:
        for policy, score_column in policies.items():
            daily, _, _ = simulate_staggered_sleeves(
                panel, "R4_rank_buffer", start=str(start.date()), end=end,
                roundtrip_cost_bps=float(cost), score_column=score_column,
            )
            daily["policy"] = policy
            daily["roundtrip_cost_bps"] = int(cost)
            drawdown = daily["net_nav"] / daily["net_nav"].cummax() - 1
            rows.append({
                "policy": policy, "roundtrip_cost_bps": int(cost),
                "start": str(start.date()), "end": str(daily["return_date"].max().date()),
                "total_return": float(daily["net_nav"].iloc[-1] - 1),
                "maximum_drawdown": float(drawdown.min()),
                "mean_daily_turnover": float(daily["turnover"].mean()),
                "mean_cash_weight": float(daily["cash_weight"].mean()),
            })
            daily_parts.append(daily)
    return pd.DataFrame(rows), pd.concat(daily_parts, ignore_index=True)


def panel_without_oof(panel: pd.DataFrame) -> pd.DataFrame:
    remove = [column for column in panel if column.startswith("score_S") or column.startswith("score_R")]
    remove += [
        column for column in panel
        if column.startswith("within_") or column.startswith("state_prior_") or column == "fold"
    ]
    return panel.drop(columns=list(dict.fromkeys(remove)), errors="ignore")


def attach_frozen_scores(panel, scores, overlay_weight) -> pd.DataFrame:
    result = panel.merge(
        scores[["trade_date", "concept_code", "score_R2_within_nonlinear_5d"]],
        on=["trade_date", "concept_code"], how="left", validate="many_to_one",
    )
    result["price_momentum_z"] = result.groupby("trade_date", sort=False)[
        "score_etf_momentum"
    ].transform(zscore)
    result["score_S0_etf_r4"] = result["price_momentum_z"]
    result["score_S2_nonlinear_overlay"] = (
        (1 - overlay_weight) * result["price_momentum_z"]
        + overlay_weight * result["score_R2_within_nonlinear_5d"]
    )
    return result


def phased_initial_recommendation(latest, target, signal_date, execute_date) -> pd.DataFrame:
    metadata = latest.drop_duplicates("ts_code").set_index("ts_code")
    rows = []
    for code, sleeve_weight in sorted(target.items(), key=lambda item: item[1], reverse=True):
        if code == CASH or sleeve_weight <= 1e-12:
            continue
        rows.append({
            "signal_date": signal_date, "execute_date": execute_date,
            "oos_day": 1, "sleeve": 0, "initialization": "phase_in_from_cash",
            "ts_code": code,
            "etf_name": "现金" if code == CASH else metadata.loc[code, "etf_name"],
            "concept_name": "现金" if code == CASH else metadata.loc[code, "concept_name"],
            "cluster": "cash" if code == CASH else metadata.loc[code, "cluster"],
            "sleeve_target_weight": sleeve_weight,
            "portfolio_target_weight": sleeve_weight / 5,
            "status": "FROZEN_FORWARD_ONLY_NO_REAL_MONEY",
        })
    invested = sum(row["portfolio_target_weight"] for row in rows if row["ts_code"] != CASH)
    rows.append({
        "signal_date": signal_date, "execute_date": execute_date,
        "oos_day": 1, "sleeve": -1, "initialization": "phase_in_from_cash",
        "ts_code": CASH, "etf_name": "现金", "concept_name": "现金", "cluster": "cash",
        "sleeve_target_weight": 0.0, "portfolio_target_weight": 1 - invested,
        "status": "FROZEN_FORWARD_ONLY_NO_REAL_MONEY",
    })
    return pd.DataFrame(rows)


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    return (values - values.mean()) / std if pd.notna(std) and std > 0 else pd.Series(0.0, index=values.index)


def resolve_snapshot(root: Path, history_end: str) -> Path:
    candidates = []
    for path in root.glob("index_backed_*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("history_end") == history_end:
            candidates.append((path.stat().st_mtime, path.parent))
    if not candidates:
        raise FileNotFoundError(f"no index-backed snapshot ending {history_end}")
    return max(candidates)[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_report(data_audit, model_audit, summary, within_ic, recommendations) -> str:
    display = summary.copy()
    for column in ["total_return", "maximum_drawdown", "mean_daily_turnover", "mean_cash_weight"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    ic_display = within_ic.copy()
    for column in ["mean_within_state_rank_ic", "positive_group_rate"]:
        ic_display[column] = ic_display[column].map(lambda value: f"{value:.2%}")
    return f"""# 指数成分支持版 S2：冻结与前向复验

状态：**FROZEN_FORWARD_ONLY_NO_REAL_MONEY**。

## 数据与冻结

- 历史交易日：{data_audit['history_days']}（门槛 {data_audit['minimum_history_days']}，通过={data_audit['history_gate_passed']}）
- 精确ETF映射：{data_audit['mapped_etfs']}；映射相关性通过：{data_audit['mapping_pass']}；聚类：{data_audit['clusters']}
- 模型训练截止：{model_audit['training_cutoff']}；成熟训练样本：{model_audit['mature_train_rows']}
- 真正未触碰样本外从下一交易日开始。任何模型、特征、宇宙、成本或执行规则变化都会重启样本外时钟。

## 开发期走步结果（不是未触碰样本外）

{display.to_markdown(index=False)}

## S2状态内IC

{ic_display.to_markdown(index=False)}

## 首个样本外影子目标

采用五袖套从现金逐日建仓，首日仅启用一个20%袖套。

{recommendations.to_markdown(index=False)}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze index-backed S2 and start OOS ledger")
    parser.add_argument("--config", default="configs/research/index_backed_s2_forward_v1.yaml")
    parser.add_argument("--data-root", default="data/index_backed_s2")
    parser.add_argument("--output-root", default="artifacts/index_backed_s2_forward")
    return parser.parse_args()


if __name__ == "__main__":
    main()
