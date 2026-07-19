from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.research.concept_etf_rotation import (
    build_etf_signal_panel,
    evaluate_etf_signals,
    evaluate_signal_ic,
    prepare_etf_panel,
)
from factor_forge.research.concept_rotation_alpha import build_concept_dataset, load_dc_snapshot


PANEL_COLUMNS = [
    "trade_date", "ts_code", "adj_open", "adj_close", "amount_cny", "circ_mv_cny",
    "is_suspended", "is_st", "is_delisting_period", "listing_trade_days", "is_tradeable",
]
SPLITS = {
    "overall": ("2025-07-01", "2026-07-14"),
    "2025_h2": ("2025-07-01", "2025-12-31"),
    "2026_q1": ("2026-01-01", "2026-03-31"),
    "2026_q2": ("2026-04-01", "2026-06-30"),
    "shadow_2026_jul": ("2026-07-01", "2026-07-14"),
}


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = resolve_data_root(Path(args.etf_data_root))
    require_snapshot(data_root)
    print("loading stock and concept point-in-time data", flush=True)
    stocks = load_stock_panel(Path(args.base_panel), Path(args.increment_panel), config["history_start"], config["history_end"])
    concept_index, members = load_combined_concepts([Path(value) for value in args.concept_roots], stocks["trade_date"].unique())
    if args.concept_features:
        print("reusing explicitly supplied lagged-weight concept features", flush=True)
        concept_features = pd.read_parquet(args.concept_features)
        concept_audit = feature_audit(concept_features, members)
    else:
        print("building lagged-weight breadth and RRG features", flush=True)
        _, concept_features, concept_audit = build_concept_dataset(
            stocks, concept_index, members, breadth_weight_lag=1,
        )

    basic = pd.read_parquet(data_root / "fund_basic.parquet")
    daily = pd.read_parquet(data_root / "fund_daily.parquet")
    share = pd.read_parquet(data_root / "fund_share.parquet")
    nav = pd.read_parquet(data_root / "fund_nav.parquet")
    mapping = pd.read_parquet(data_root / "concept_etf_mapping.parquet")
    etfs = prepare_etf_panel(daily, share, nav, basic)
    signal_panel, mapping_audit = build_etf_signal_panel(
        concept_features, etfs, mapping,
        selection_cutoff=config["selection_cutoff"],
        minimum_mapping_correlation=float(config["minimum_mapping_correlation"]),
    )
    data_audit = audit_data(config, stocks, concept_index, members, etfs, mapping_audit, concept_audit)
    failures = gate_failures(data_audit)
    if failures:
        raise RuntimeError("ETF rotation data gate failed: " + "; ".join(failures))

    print("running E0-E5, five weekly offsets and cost stress", flush=True)
    summary, periods, paired = evaluate_etf_signals(signal_panel, splits=SPLITS)
    signal_ic = evaluate_signal_ic(
        signal_panel, start=config["validation_start"], end=config["history_end"]
    )
    run_id = datetime.now(timezone.utc).strftime("concept_etf_rotation_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    concept_features.to_parquet(output / "concept_daily_features_lagged.parquet", index=False)
    signal_panel.to_parquet(output / "etf_signal_panel.parquet", index=False)
    mapping_audit.to_csv(output / "mapping_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    periods.to_parquet(output / "rebalance_periods.parquet", index=False)
    paired.to_csv(output / "incremental_e3_vs_e2.csv", index=False, encoding="utf-8-sig")
    signal_ic.to_csv(output / "signal_rank_ic.csv", index=False, encoding="utf-8-sig")
    (output / "data_audit.json").write_text(json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    decision = make_decision(summary, paired, mapping_audit)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(data_root, data_audit, mapping_audit, summary, signal_ic, decision), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concept breadth/RRG to ETF implementation validation")
    parser.add_argument("--config", default="configs/research/concept_etf_rotation_v1.yaml")
    parser.add_argument("--etf-data-root", default="data/concept_etf_rotation")
    parser.add_argument("--concept-roots", nargs="+", default=[
        "data/concept_rotation/dc_20241230_20250627_by_date",
        "data/concept_rotation/dc_20250630_20260714",
    ])
    parser.add_argument("--base-panel", default="data/versions/data_v1_20260710T115133Z_208e9fc8/curated/stock_daily_panel.parquet")
    parser.add_argument("--increment-panel", default="data/versions/data_v1_20260715T053637Z_77138057/curated/stock_daily_panel.parquet")
    parser.add_argument("--concept-features", help="explicit previously generated lagged feature parquet")
    parser.add_argument("--output-root", default="artifacts/concept_etf_rotation")
    return parser.parse_args()


def resolve_data_root(root: Path) -> Path:
    if (root / "manifest.json").exists():
        return root
    candidates = [path.parent for path in root.glob("tushare_*/manifest.json")]
    if not candidates:
        raise FileNotFoundError(f"no complete ETF snapshot below {root}")
    return max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime)


def require_snapshot(root: Path) -> None:
    required = [
        "manifest.json", "fund_basic.parquet", "fund_daily.parquet", "fund_share.parquet",
        "fund_nav.parquet", "concept_etf_mapping.parquet", "candidate_selection_audit.parquet",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"incomplete ETF snapshot {root}: {missing}")


def load_stock_panel(base_path: Path, increment_path: Path, start: str, end: str) -> pd.DataFrame:
    start_date, end_date = pd.to_datetime(start), pd.to_datetime(end)
    base = pd.read_parquet(base_path, columns=PANEL_COLUMNS, filters=[("trade_date", ">=", start_date)])
    increment = pd.read_parquet(increment_path, columns=PANEL_COLUMNS)
    panel = pd.concat([base, increment], ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return panel.loc[panel["trade_date"].between(start_date, end_date)].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def load_combined_concepts(roots: list[Path], trade_dates) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices, relations = [], []
    for root in roots:
        index, members = load_dc_snapshot(root, trade_dates=trade_dates)
        indices.append(index)
        relations.append(members)
    index = pd.concat(indices, ignore_index=True).drop_duplicates(["trade_date", "concept_code"], keep="last")
    members = pd.concat(relations, ignore_index=True).drop_duplicates(["trade_date", "concept_code", "ts_code"], keep="last")
    return index.sort_values(["trade_date", "concept_code"]), members.sort_values(["trade_date", "concept_code", "ts_code"])


def feature_audit(features: pd.DataFrame, members: pd.DataFrame) -> dict:
    return {
        "start_date": str(pd.to_datetime(features["trade_date"]).min().date()),
        "end_date": str(pd.to_datetime(features["trade_date"]).max().date()),
        "index_rows": len(features), "member_rows": len(members),
        "concepts": int(features["concept_code"].nunique()),
        "eligible_concept_days": int(features["eligible_concept"].fillna(False).sum()),
        "reused_explicit_feature_artifact": True,
    }


def audit_data(config, stocks, concept_index, members, etfs, mapping_audit, concept_audit) -> dict:
    validation_start, end = pd.Timestamp(config["validation_start"]), pd.Timestamp(config["history_end"])
    stock_dates = set(stocks.loc[stocks["trade_date"].between(validation_start, end), "trade_date"])
    concept_dates = set(concept_index.loc[concept_index["trade_date"].between(validation_start, end), "trade_date"])
    selected_codes = set(mapping_audit["ts_code"])
    coverage = etfs.loc[etfs["trade_date"].between(validation_start, end) & etfs["ts_code"].isin(selected_codes)].groupby("ts_code")["trade_date"].nunique()
    return {
        "history_start": str(stocks["trade_date"].min().date()), "history_end": str(stocks["trade_date"].max().date()),
        "validation_trade_dates": len(stock_dates), "missing_concept_trade_dates": len(stock_dates - concept_dates),
        "concepts": int(concept_index["concept_code"].nunique()), "member_rows": len(members),
        "selected_mapping_rows": len(mapping_audit), "mapping_pass_rows": int(mapping_audit["mapping_pass"].sum()),
        "minimum_mapping_correlation": float(config["minimum_mapping_correlation"]),
        "minimum_etf_date_coverage": float(coverage.min() / len(stock_dates)) if len(stock_dates) and len(coverage) else 0,
        "duplicate_etf_keys": int(etfs.duplicated(["trade_date", "ts_code"]).sum()),
        "nav_coverage": float(etfs["unit_nav"].notna().mean()) if "unit_nav" in etfs else 0,
        "concept_feature_audit": concept_audit,
    }


def gate_failures(audit: dict) -> list[str]:
    failures = []
    if audit["missing_concept_trade_dates"]:
        failures.append(f"missing_concept_trade_dates={audit['missing_concept_trade_dates']}")
    if audit["duplicate_etf_keys"]:
        failures.append(f"duplicate_etf_keys={audit['duplicate_etf_keys']}")
    if audit["mapping_pass_rows"] < 6:
        failures.append(f"only {audit['mapping_pass_rows']} mappings passed correlation gate")
    if audit["minimum_etf_date_coverage"] < 0.95:
        failures.append(f"minimum_etf_date_coverage={audit['minimum_etf_date_coverage']:.3f}")
    return failures


def make_decision(summary: pd.DataFrame, paired: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    overall = paired.loc[paired["split"].eq("overall")]
    subperiods = paired.loc[paired["split"].isin(["2025_h2", "2026_q1", "2026_q2"])]
    incremental = float(overall.iloc[0]["incremental_net_excess"]) if len(overall) else None
    t_value = float(overall.iloc[0]["nw_t"]) if len(overall) else None
    positive_subperiods = int(subperiods["incremental_net_excess"].gt(0).sum())
    placebo = summary.loc[(summary["split"].eq("overall")) & (summary["signal"].eq("E5_placebo_mapping")), "net_excess_20bps"]
    e3 = summary.loc[(summary["split"].eq("overall")) & (summary["signal"].eq("E3_rrg_breadth")), "net_excess_20bps"]
    beats_placebo = bool(len(placebo) and len(e3) and e3.iloc[0] > placebo.iloc[0])
    keep = bool(incremental is not None and incremental > 0 and t_value is not None and t_value > 1.0 and positive_subperiods >= 2 and beats_placebo)
    return {
        "verdict": "KEEP_FOR_FORWARD_CONFIRMATION" if keep else "DO_NOT_KEEP_AS_ALPHA",
        "incremental_e3_vs_e2_net_excess_20bps": incremental, "incremental_nw_t": t_value,
        "positive_subperiods": positive_subperiods, "beats_placebo_mapping": beats_placebo,
        "mapping_pass": int(mapping["mapping_pass"].sum()),
        "important_caveat": "The historical interval has already been inspected; any keep verdict still requires a new forward confirmation window.",
    }


def render_report(data_root, audit, mapping, summary, signal_ic, decision) -> str:
    mapping_lines = "\n".join(
        f"- {row.concept_name} → {row.etf_name} ({row.ts_code}), pre-cutoff corr={row.mapping_correlation:.3f}, {'PASS' if row.mapping_pass else 'FAIL'}"
        for row in mapping.itertuples()
    )
    overall = summary.loc[summary["split"].eq("overall")].set_index("signal")
    e1_ic = signal_ic.loc[(signal_ic["signal"].eq("E1_etf_momentum")) & signal_ic["horizon"].eq(5)].iloc[0]
    return f"""# 概念轮动到ETF的实施验证

## 结论

**{decision['verdict']}**。E3（RRG+共同成分扩散）相对 E2（仅RRG）的20bps成本后增量为 {decision['incremental_e3_vs_e2_net_excess_20bps']!s}，NW t值为 {decision['incremental_nw_t']!s}。

本次验证将《轮动.txt》的“扩散度 + RRG + 市场环境”思想落到ETF可交易标的；市场环境保留为后续条件分析，不作为这次样本内新增自由度。历史区间已经被查看，不能再称为未触碰样本。

20bps完整往返成本下，每5日平均超额：ETF自身动量 E1={overall.loc['E1_etf_momentum', 'net_excess_20bps']:.3%}，概念RRG E2={overall.loc['E2_concept_rrg', 'net_excess_20bps']:.3%}，RRG+扩散 E3={overall.loc['E3_rrg_breadth', 'net_excess_20bps']:.3%}，错配安慰剂 E5={overall.loc['E5_placebo_mapping', 'net_excess_20bps']:.3%}。E1的5日截面Rank IC={e1_ic['mean_rank_ic']:.3f}（NW t={e1_ic['rank_ic_nw_t']:.2f}），是本轮最值得单独前瞻确认的基线，但它不支持“扩散带来新增信息”的原假设。

## 数据完整性

- 冻结ETF快照：`{data_root}`
- 概念数：{audit['concepts']}；验证交易日：{audit['validation_trade_dates']}；概念缺失交易日：{audit['missing_concept_trade_dates']}
- 映射通过：{audit['mapping_pass_rows']}/{audit['selected_mapping_rows']}；ETF最小日期覆盖率：{audit['minimum_etf_date_coverage']:.2%}
- ETF净值覆盖率：{audit['nav_coverage']:.2%}；重复ETF键：{audit['duplicate_etf_keys']}

## 映射准入

{mapping_lines}

语义匹配和 2025-06-30 前的规模/流动性用于选基金；同期概念收益与ETF收益相关性用于二次准入。低空经济因无合格ETF、工业母机因规模/流动性不足而排除。

## 实验

- E0：合格ETF等权；E1：ETF自身20/60日动量；E2：概念RRG；E3：RRG+共同成分扩散残差；E4：原方案硬阈值；E5：错配概念的安慰剂。
- 信号用T日收盘数据，收益按T+1开盘到T+6开盘；Top 3等权，同簇最多1只；覆盖五个周度起始偏移。
- 往返成本压力为10/20/40bps。E4不足3只时保留现金暴露差，并单独报告平均暴露。

## 决策规则

E3只有同时满足：相对E2增量为正、NW t>1、三个子区间至少两个为正、并优于错配安慰剂，才进入前瞻确认。当前通过子区间数：{decision['positive_subperiods']}；优于安慰剂：{decision['beats_placebo_mapping']}。
"""


if __name__ == "__main__":
    main()
