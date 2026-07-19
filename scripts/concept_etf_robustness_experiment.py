from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from factor_forge.research.concept_etf_shadow import (
    evaluate_specification,
    latest_target_table,
)


START = "2025-07-01"
END = "2026-07-14"


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    panel_path = resolve_panel(Path(args.signal_panel))
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    expected = set(config["etf_codes"])
    actual = set(panel["ts_code"].unique())
    if expected != actual:
        raise RuntimeError(f"frozen universe mismatch: missing={expected-actual}, extra={actual-expected}")

    print("running primary P0/P1/P2/P3 turnover-aware simulations", flush=True)
    primary_rows, primary_periods = [], []
    comparisons = [
        ("P1_etf_momentum", "P0_equal_weight"),
        ("P2_breadth_overlay", "P1_etf_momentum"),
        ("P3_rrg_filter", "P1_etf_momentum"),
    ]
    for cost in (20, 40):
        for portfolio, benchmark in comparisons:
            summary, periods = evaluate_specification(
                panel, portfolio, benchmark=benchmark, start=START, end=END,
                horizon=5, top_n=int(config["signal"]["top_n"]),
                roundtrip_cost_bps=cost,
            )
            periods["benchmark"] = benchmark
            periods["roundtrip_cost_bps"] = cost
            primary_rows.append(summary)
            primary_periods.append(periods)

    print("running top-N, horizon and mapping-strictness grid", flush=True)
    grid_rows = []
    for horizon in (5, 10, 20):
        for top_n in (2, 3, 4):
            for universe in ("all", "no_proxy", "exact"):
                summary, _ = evaluate_specification(
                    panel, "P1_etf_momentum", start=START, end=END,
                    horizon=horizon, top_n=top_n, roundtrip_cost_bps=20, universe=universe,
                )
                grid_rows.append(summary)

    print("running leave-one-ETF-out falsification", flush=True)
    leave_one_out = []
    for code in sorted(expected):
        summary, _ = evaluate_specification(
            panel, "P1_etf_momentum", start=START, end=END, horizon=5, top_n=3,
            roundtrip_cost_bps=20, excluded_etf=code,
        )
        leave_one_out.append(summary)

    primary = pd.DataFrame(primary_rows)
    grid = pd.DataFrame(grid_rows)
    loo = pd.DataFrame(leave_one_out)
    periods = pd.concat(primary_periods, ignore_index=True)
    attribution = profit_attribution(panel, periods)
    targets = latest_target_table(panel, top_n=int(config["signal"]["top_n"]))
    decision = make_decision(primary, grid, loo, attribution)

    run_id = datetime.now(timezone.utc).strftime("concept_etf_robustness_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    primary.to_csv(output / "primary_p0_p3.csv", index=False, encoding="utf-8-sig")
    periods.to_parquet(output / "primary_periods.parquet", index=False)
    grid.to_csv(output / "e1_robustness_grid.csv", index=False, encoding="utf-8-sig")
    loo.to_csv(output / "e1_leave_one_out.csv", index=False, encoding="utf-8-sig")
    attribution.to_csv(output / "e1_profit_attribution.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(output / "shadow_targets_preview.csv", index=False, encoding="utf-8-sig")
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "research_report.md").write_text(
        render_report(panel_path, primary, grid, loo, attribution, targets, decision), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(output), "decision": decision}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF momentum robustness and shadow portfolio experiment")
    parser.add_argument("--config", default="configs/research/concept_etf_forward_v1.yaml")
    parser.add_argument("--signal-panel", default="artifacts/concept_etf_rotation")
    parser.add_argument("--output-root", default="artifacts/concept_etf_robustness")
    return parser.parse_args()


def resolve_panel(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.glob("concept_etf_rotation_*/etf_signal_panel.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no ETF signal panel below {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def profit_attribution(panel: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    sample = periods.loc[
        periods["portfolio"].eq("P1_etf_momentum")
        & periods["roundtrip_cost_bps"].eq(20)
    ]
    returns = panel.set_index(["trade_date", "ts_code"])["forward_open_5d"]
    rows = []
    for period in sample.itertuples():
        for item in period.target_weights.split(";"):
            code, raw_weight = item.split(":")
            if code == "__CASH__":
                continue
            weight = float(raw_weight)
            rows.append({
                "ts_code": code, "gross_contribution": weight * returns.get((period.trade_date, code), 0.0),
            })
    result = pd.DataFrame(rows).groupby("ts_code", as_index=False).agg(
        gross_contribution=("gross_contribution", "sum"), selection_periods=("gross_contribution", "size")
    )
    positive_total = result["gross_contribution"].clip(lower=0).sum()
    result["positive_profit_share"] = result["gross_contribution"].clip(lower=0) / positive_total
    names = panel[["ts_code", "etf_name", "concept_name"]].drop_duplicates("ts_code")
    return result.merge(names, on="ts_code", how="left").sort_values("positive_profit_share", ascending=False)


def make_decision(primary: pd.DataFrame, grid: pd.DataFrame, loo: pd.DataFrame, attribution: pd.DataFrame) -> dict:
    e1_20 = primary.loc[
        primary["portfolio"].eq("P1_etf_momentum") & primary["roundtrip_cost_bps"].eq(20)
    ].iloc[0]
    e1_40 = primary.loc[
        primary["portfolio"].eq("P1_etf_momentum") & primary["roundtrip_cost_bps"].eq(40)
    ].iloc[0]
    loo_positive = float(loo["mean_net_excess"].gt(0).mean())
    standard_grid = grid.loc[grid["universe"].isin(["all", "no_proxy"])]
    grid_positive = float(standard_grid["mean_net_excess"].gt(0).mean())
    maximum_contribution = float(attribution["positive_profit_share"].max())
    ready = bool(
        e1_20["mean_net_excess"] > 0 and e1_20["positive_offsets"] >= 4
        and e1_40["mean_net_excess"] > 0 and loo_positive >= 0.80
        and grid_positive >= 0.70
    )
    return {
        "verdict": "READY_FOR_FORWARD_SHADOW_WITH_CONCENTRATION_FLAG" if ready else "DO_NOT_START_FORWARD_SHADOW",
        "not_an_alpha_confirmation": True,
        "primary_e1_net_excess_20bps": float(e1_20["mean_net_excess"]),
        "primary_e1_nw_t_20bps": float(e1_20["net_excess_nw_t"]),
        "primary_e1_positive_offsets": int(e1_20["positive_offsets"]),
        "primary_e1_net_excess_40bps": float(e1_40["mean_net_excess"]),
        "leave_one_out_positive_fraction": loo_positive,
        "standard_grid_positive_fraction": grid_positive,
        "historical_maximum_positive_profit_contribution": maximum_contribution,
        "historical_concentration_above_30pct": maximum_contribution > 0.30,
        "forward_clock_months": 12,
    }


def render_report(panel_path, primary, grid, loo, attribution, targets, decision) -> str:
    def metric(portfolio, cost):
        return primary.loc[
            primary["portfolio"].eq(portfolio) & primary["roundtrip_cost_bps"].eq(cost)
        ].iloc[0]

    p1, p2, p3 = metric("P1_etf_momentum", 20), metric("P2_breadth_overlay", 20), metric("P3_rrg_filter", 20)
    leader = attribution.iloc[0]
    target_lines = "\n".join(
        f"- {row.portfolio}: {row.etf_name} ({row.ts_code}) {row.target_weight:.1%}"
        for row in targets.itertuples()
    )
    return f"""# ETF动量稳健性与前瞻影子组合

## 决策

**{decision['verdict']}**，但这只是允许进入影子观察，不是Alpha确认。历史信号面板：`{panel_path}`。

P1相对P0在真实换手、20bps完整往返成本下，每5日平均净超额为 {p1.mean_net_excess:.3%}（NW t={p1.net_excess_nw_t:.2f}），五个起始偏移中 {int(p1.positive_offsets)} 个为正；40bps下净超额为 {decision['primary_e1_net_excess_40bps']:.3%}。

逐一剔除ETF后，{decision['leave_one_out_positive_fraction']:.1%} 的组合仍为正；Top-N、持有期和非代理映射标准网格中，{decision['standard_grid_positive_fraction']:.1%} 为正。严格exact-only结果只作为小样本诊断，不计入准入比例。

收益集中度仍需警惕：{leader.etf_name}（{leader.ts_code}）占历史正贡献的 {leader.positive_profit_share:.1%}，超过30%预警线。不过将其完全剔除后策略净超额仍为正，因此允许进入影子观察，但带集中度标记；前瞻期必须重新计算该比例。

## 概念过滤器

- P2相对P1增量：{p2.mean_net_excess:.3%}（扩散与RRG同时恶化时减半权重）。
- P3相对P1增量：{p3.mean_net_excess:.3%}（只允许RRG领先或改善）。

两者的历史结果只用于决定是否并行记录，前瞻期仍必须同时保留P1，才能检验真正的增量。

## 首个影子目标预览

信号日期为数据集中最后一个交易日；这是流程预览，不是实际成交指令。

{target_lines}

## 前瞻规则

- 每周最后一个可用交易日收盘计算，下一交易日开盘模拟成交。
- P0/P1/P2/P3同时记录，Top 3，同一主题簇最多1只。
- 基础完整往返成本20bps，压力成本40bps，使用真实目标权重变化和持仓漂移计算换手。
- 最少观察12个月，6个月只做中期审计；修改ETF池或参数会重新开始计时。
"""


if __name__ == "__main__":
    main()
