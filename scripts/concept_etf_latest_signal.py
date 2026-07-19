from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.data.tushare_provider import TushareProvider
from factor_forge.research.concept_etf_rotation import build_etf_signal_panel, prepare_etf_panel
from factor_forge.research.concept_etf_shadow import (
    CASH,
    simulate_staggered_sleeves,
    staggered_target_weights,
)
from factor_forge.research.concept_rotation_alpha import build_concept_dataset, load_dc_snapshot


PANEL_COLUMNS = [
    "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny", "circ_mv_cny",
    "is_suspended", "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable",
]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stock_paths = [Path(args.base_panel), *[Path(path) for path in args.increment_panels]]
    stocks = load_incremental_stock_panel(stock_paths, config["history_start"], args.as_of)
    concept_index, members = load_combined_concepts(
        [Path(path) for path in args.concept_roots], stocks["trade_date"].unique()
    )
    print("building latest lagged-weight concept features", flush=True)
    _, features, feature_audit = build_concept_dataset(
        stocks, concept_index, members, breadth_weight_lag=1,
    )

    etf_root = resolve_etf_root(Path(args.etf_data_root), args.as_of)
    basic = pd.read_parquet(etf_root / "fund_basic.parquet")
    daily = pd.read_parquet(etf_root / "fund_daily.parquet")
    share = pd.read_parquet(etf_root / "fund_share.parquet")
    nav = pd.read_parquet(etf_root / "fund_nav.parquet")
    mapping = pd.read_parquet(etf_root / "concept_etf_mapping.parquet")
    etfs = prepare_etf_panel(daily, share, nav, basic)
    signal_panel, mapping_audit = build_etf_signal_panel(
        features, etfs, mapping, selection_cutoff=config["selection_cutoff"],
        minimum_mapping_correlation=float(config["minimum_mapping_correlation"]),
    )
    signal_panel["volatility_20d"] = signal_panel.groupby("ts_code", sort=False)["etf_return_1d"].transform(
        lambda values: values.rolling(20, min_periods=18).std(ddof=0)
    )
    signal_date = pd.Timestamp(args.as_of)
    next_trade_date = resolve_next_trade_date(signal_date)
    audit = data_gate(signal_date, stocks, concept_index, members, signal_panel, mapping_audit, feature_audit)
    failures = gate_failures(audit)
    if failures:
        raise RuntimeError("latest signal data gate failed: " + "; ".join(failures))

    ranking = latest_ranking(signal_panel, signal_date)
    recommendations = build_recommendations(signal_panel, signal_date, next_trade_date)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"live_signal_{signal_date:%Y%m%d}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    ranking.to_csv(output / "latest_ranking.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(output / "shadow_recommendations.csv", index=False, encoding="utf-8-sig")
    signal_panel.to_parquet(output / "latest_signal_panel.parquet", index=False)
    features.to_parquet(output / "latest_concept_features.parquet", index=False)
    (output / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": "SHADOW_ONLY_NO_REAL_MONEY", "created_at": datetime.now(timezone.utc).isoformat(),
        "signal_date": str(signal_date.date()), "next_trade_date": str(next_trade_date.date()),
        "execute_at": "next_trade_open", "etf_snapshot": str(etf_root.resolve()),
        "stock_versions": [str(path.resolve()) for path in stock_paths],
        "concept_snapshots": [str(Path(path).resolve()) for path in args.concept_roots],
        "important": "Non-overlapping validation failed; these are model shadow targets, not investment advice or authorized orders.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "signal_report.md").write_text(
        render_report(ranking, recommendations, audit, manifest), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), **manifest}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the latest concept ETF shadow recommendation")
    parser.add_argument("--as-of", default="20260717")
    parser.add_argument("--config", default="configs/research/concept_etf_rotation_v1.yaml")
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panels", nargs="+", default=[
        "data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet",
        "data/versions/data_v1_20260717T142755Z_7042ceab/curated/stock_daily_panel.parquet",
    ])
    parser.add_argument("--concept-roots", nargs="+", default=[
        "data/concept_rotation/dc_20241230_20250627_by_date",
        "data/concept_rotation/dc_20250630_20260714",
        "data/concept_rotation/dc_20260715_20260717_by_date_concept",
    ])
    parser.add_argument("--etf-data-root", default="data/concept_etf_rotation")
    parser.add_argument("--output-root", default="artifacts/concept_etf_live_signal")
    return parser.parse_args()


def load_incremental_stock_panel(paths: list[Path], start: str, end: str) -> pd.DataFrame:
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    accumulated = pd.read_parquet(
        paths[0], columns=PANEL_COLUMNS, filters=[("trade_date", ">=", start_date)]
    )
    accumulated["trade_date"] = pd.to_datetime(accumulated["trade_date"])
    accumulated = accumulated.loc[accumulated["trade_date"].le(end_date)]
    for path in paths[1:]:
        increment = pd.read_parquet(path, columns=PANEL_COLUMNS)
        increment["trade_date"] = pd.to_datetime(increment["trade_date"])
        increment = increment.loc[increment["trade_date"].between(start_date, end_date)]
        increment = increment.loc[~increment.set_index(["trade_date", "ts_code"]).index.isin(
            accumulated.set_index(["trade_date", "ts_code"]).index
        )].copy()
        last_listing = accumulated.groupby("ts_code", observed=True)["listing_trade_days"].max()
        increment = increment.sort_values(["ts_code", "trade_date"])
        increment["listing_trade_days"] = (
            increment.groupby("ts_code", observed=True).cumcount() + 1
            + increment["ts_code"].map(last_listing).fillna(0)
        )
        increment["is_tradeable"] = (
            increment["listing_trade_days"].ge(60)
            & ~increment["is_suspended"].fillna(True)
            & ~increment["is_st"].fillna(False)
            & ~increment["is_delisting_period"].fillna(False)
            & increment["adj_open"].notna() & increment["adj_close"].notna()
        )
        accumulated = pd.concat([accumulated, increment], ignore_index=True)
    return accumulated.drop_duplicates(["trade_date", "ts_code"], keep="last").sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)


def load_combined_concepts(roots: list[Path], trade_dates) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices, relations = [], []
    for root in roots:
        index, members = load_dc_snapshot(root, trade_dates=trade_dates)
        indices.append(index)
        relations.append(members)
    index = pd.concat(indices, ignore_index=True).drop_duplicates(["trade_date", "concept_code"], keep="last")
    members = pd.concat(relations, ignore_index=True).drop_duplicates(
        ["trade_date", "concept_code", "ts_code"], keep="last"
    )
    return index.sort_values(["trade_date", "concept_code"]), members.sort_values(
        ["trade_date", "concept_code", "ts_code"]
    )


def resolve_etf_root(root: Path, as_of: str) -> Path:
    candidates = []
    for manifest_path in root.glob("tushare_*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("history_end", "") >= as_of and (manifest_path.parent / "candidate_selection_audit.parquet").exists():
            candidates.append((manifest.get("history_end", ""), manifest_path.stat().st_mtime, manifest_path.parent))
    if not candidates:
        raise FileNotFoundError(f"no complete ETF snapshot through {as_of}")
    return max(candidates)[2]


def resolve_next_trade_date(signal_date: pd.Timestamp) -> pd.Timestamp:
    provider = TushareProvider()
    calendar = provider.query(
        "trade_cal", exchange="SSE", start_date=(signal_date + pd.Timedelta(days=1)).strftime("%Y%m%d"),
        end_date=(signal_date + pd.Timedelta(days=10)).strftime("%Y%m%d"),
    )
    open_dates = pd.to_datetime(calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"])
    return pd.Timestamp(open_dates.min())


def data_gate(signal_date, stocks, concept_index, members, signal_panel, mapping_audit, feature_audit) -> dict:
    latest = signal_panel.loc[signal_panel["trade_date"].eq(signal_date)]
    return {
        "signal_date": str(signal_date.date()),
        "stock_latest_date": str(stocks["trade_date"].max().date()),
        "concept_latest_date": str(concept_index["trade_date"].max().date()),
        "member_latest_date": str(members["trade_date"].max().date()),
        "etf_latest_date": str(signal_panel["trade_date"].max().date()),
        "latest_concepts": int(concept_index.loc[concept_index["trade_date"].eq(signal_date), "concept_code"].nunique()),
        "latest_member_rows": int(members["trade_date"].eq(signal_date).sum()),
        "latest_etfs": int(latest["ts_code"].nunique()),
        "mapping_pass": int(mapping_audit["mapping_pass"].sum()),
        "latest_missing_scores": int(latest["score_etf_momentum"].isna().sum()),
        "latest_missing_nav": int(latest["unit_nav"].isna().sum()),
        "max_abs_etf_return_1d": float(signal_panel["etf_return_1d"].abs().max()),
        "max_latest_volatility_20d": float(latest["volatility_20d"].max()),
        "feature_audit": feature_audit,
    }


def gate_failures(audit: dict) -> list[str]:
    date = audit["signal_date"]
    failures = []
    for key in ("stock_latest_date", "concept_latest_date", "member_latest_date", "etf_latest_date"):
        if audit[key] != date:
            failures.append(f"{key}={audit[key]}")
    if audit["latest_concepts"] < 400:
        failures.append(f"latest_concepts={audit['latest_concepts']}")
    if audit["latest_member_rows"] < 50_000:
        failures.append(f"latest_member_rows={audit['latest_member_rows']}")
    if audit["latest_etfs"] != 11 or audit["mapping_pass"] != 11:
        failures.append(f"ETF/mapping coverage={audit['latest_etfs']}/{audit['mapping_pass']}")
    if audit["latest_missing_scores"] or audit["latest_missing_nav"]:
        failures.append("latest score or NAV missing")
    if audit["max_abs_etf_return_1d"] > 0.25:
        failures.append(f"abnormal adjusted ETF return={audit['max_abs_etf_return_1d']:.2%}")
    if audit["max_latest_volatility_20d"] > 0.10:
        failures.append(f"abnormal latest ETF volatility={audit['max_latest_volatility_20d']:.2%}")
    return failures


def latest_ranking(panel: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    day = panel.loc[panel["trade_date"].eq(signal_date)].copy()
    day["momentum_rank"] = day["score_etf_momentum"].rank(ascending=False, method="min")
    columns = [
        "momentum_rank", "ts_code", "etf_name", "concept_name", "cluster", "match_type",
        "etf_momentum_20d", "etf_momentum_60d", "score_etf_momentum", "volatility_20d",
        "rrg_quadrant", "common_breadth_delta_smooth5", "rs_momentum_5d", "amount_cny", "aum_cny",
    ]
    return day[columns].sort_values("momentum_rank")


def build_recommendations(
    panel: pd.DataFrame,
    signal_date: pd.Timestamp,
    next_trade_date: pd.Timestamp,
) -> pd.DataFrame:
    day = panel.loc[panel["trade_date"].eq(signal_date)]
    active_dates = sorted(panel.loc[panel["trade_date"].between(pd.Timestamp("2025-07-01"), signal_date), "trade_date"].unique())
    signal_sleeve = active_dates.index(signal_date.to_datetime64()) % 5
    scenarios = {
        "R1_base": ("R1_staggered_momentum", "all", set()),
        "R4_base": ("R4_rank_buffer", "all", set()),
        "R4_no_proxy": ("R4_rank_buffer", "no_proxy", set()),
    }
    rows = []
    for scenario, (variant, universe, excluded) in scenarios.items():
        _, sleeves, _ = simulate_staggered_sleeves(
            panel, variant, start="2025-07-01", end=str(signal_date.date()),
            roundtrip_cost_bps=20, universe=universe, excluded_etfs=excluded,
        )
        sleeve_targets = {}
        previous_holdings = set()
        for sleeve in range(5):
            latest = sleeves.loc[sleeves["sleeve"].eq(sleeve)].sort_values("return_date").iloc[-1]
            weights = parse_weights(latest["target_weights"])
            sleeve_targets[sleeve] = weights
            if sleeve == signal_sleeve:
                previous_holdings = {code for code, weight in weights.items() if code != CASH and weight > 0}
        new_target = staggered_target_weights(
            day, variant, previous_holdings=previous_holdings,
            universe=universe, excluded_etfs=excluded,
        )
        sleeve_targets[signal_sleeve] = new_target
        composite = {}
        for weights in sleeve_targets.values():
            for code, weight in weights.items():
                composite[code] = composite.get(code, 0.0) + weight / 5
        metadata = day.set_index("ts_code")
        for code, weight in sorted(composite.items(), key=lambda item: item[1], reverse=True):
            if weight <= 1e-10:
                continue
            rows.append({
                "scenario": scenario, "signal_date": signal_date, "execute_date": next_trade_date,
                "rebalancing_sleeve": signal_sleeve, "ts_code": code,
                "etf_name": "现金" if code == CASH else metadata.loc[code, "etf_name"],
                "concept_name": "现金" if code == CASH else metadata.loc[code, "concept_name"],
                "composite_shadow_weight": weight,
                "rebalanced_sleeve_weight": new_target.get(code, 0.0),
                "status": "SHADOW_ONLY_NO_REAL_MONEY",
            })
    return pd.DataFrame(rows)


def parse_weights(value: str) -> dict[str, float]:
    return {item.split(":")[0]: float(item.split(":")[1]) for item in value.split(";") if item}


def render_report(ranking, recommendations, audit, manifest) -> str:
    ranking_display = ranking.copy()
    for column in ("etf_momentum_20d", "etf_momentum_60d", "volatility_20d"):
        ranking_display[column] = ranking_display[column].map(lambda value: f"{value:.2%}")
    recommendation_display = recommendations.copy()
    recommendation_display["composite_shadow_weight"] = recommendation_display["composite_shadow_weight"].map(
        lambda value: f"{value:.2%}"
    )
    recommendation_display["rebalanced_sleeve_weight"] = recommendation_display["rebalanced_sleeve_weight"].map(
        lambda value: f"{value:.2%}"
    )
    return f"""# 下一交易日概念ETF影子信号

状态：**{manifest['status']}**。信号日 {manifest['signal_date']} 收盘，计划观察日 {manifest['next_trade_date']} 开盘。非重叠验证未通过，因此不构成实盘或投资建议。

## 数据闸门

- 最新股票/概念/成员/ETF日期：{audit['stock_latest_date']} / {audit['concept_latest_date']} / {audit['member_latest_date']} / {audit['etf_latest_date']}
- 最新概念数：{audit['latest_concepts']}；成员行：{audit['latest_member_rows']}；ETF与映射：{audit['latest_etfs']}/{audit['mapping_pass']}

## 最新动量排名

{ranking_display.to_markdown(index=False)}

## 五袖组合影子目标

本次只重平衡一个20%袖子；组合权重是五袖模型目标的合计，实际开盘漂移需在开盘价出现后才能精确计算。

{recommendation_display.to_markdown(index=False)}
"""


if __name__ == "__main__":
    main()
