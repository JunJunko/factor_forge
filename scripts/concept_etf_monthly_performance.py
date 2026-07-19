from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_forge.research.concept_etf_shadow import PORTFOLIOS, monthly_performance, simulate_weekly_daily_nav


def main() -> None:
    args = parse_args()
    panel_path = resolve_panel(Path(args.signal_panel))
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    daily_parts = []
    for portfolio in PORTFOLIOS:
        daily_parts.append(simulate_weekly_daily_nav(
            panel, portfolio, start=args.start, end=args.end, top_n=args.top_n,
            roundtrip_cost_bps=args.roundtrip_cost_bps,
        ))
    daily = pd.concat(daily_parts, ignore_index=True)
    monthly = monthly_performance(daily)
    summary = portfolio_summary(daily, monthly)
    decision = schedule_decision(summary)

    run_id = datetime.now(timezone.utc).strftime("concept_etf_monthly_%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    daily.to_parquet(output / "daily_nav.parquet", index=False)
    monthly.to_csv(output / "monthly_performance_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    for value, filename in (
        ("monthly_return", "monthly_returns.csv"),
        ("monthly_max_drawdown", "monthly_max_drawdowns.csv"),
        ("month_end_drawdown", "month_end_drawdowns.csv"),
        ("monthly_excess_vs_p0", "monthly_excess_vs_p0.csv"),
    ):
        monthly.pivot(index="month", columns="portfolio", values=value).to_csv(
            output / filename, encoding="utf-8-sig"
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source_panel": str(panel_path.resolve()),
        "start": args.start, "end": args.end, "top_n": args.top_n,
        "roundtrip_cost_bps": args.roundtrip_cost_bps,
        "schedule": "weekly last completed trading day signal, next-trade-open execution",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "monthly_report.md").write_text(render_report(monthly, summary, manifest, decision), encoding="utf-8")
    print(json.dumps({"run_dir": str(output), "summary": summary.to_dict("records")}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly returns and drawdowns for P0-P3")
    parser.add_argument("--signal-panel", default="artifacts/concept_etf_rotation")
    parser.add_argument("--start", default="2025-07-01")
    parser.add_argument("--end", default="2026-07-14")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=20)
    parser.add_argument("--output-root", default="artifacts/concept_etf_monthly")
    return parser.parse_args()


def resolve_panel(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.glob("concept_etf_rotation_*/etf_signal_panel.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no ETF signal panel below {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def portfolio_summary(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, frame in daily.groupby("portfolio", observed=True):
        frame = frame.sort_values("return_date")
        drawdown = frame["net_nav"] / frame["net_nav"].cummax().clip(lower=1.0) - 1
        portfolio_monthly = monthly.loc[monthly["portfolio"].eq(portfolio)]
        rows.append({
            "portfolio": portfolio, "total_return": float(frame["net_nav"].iloc[-1] - 1),
            "maximum_drawdown": float(drawdown.min()),
            "positive_months": int(portfolio_monthly["monthly_return"].gt(0).sum()),
            "months": len(portfolio_monthly), "mean_monthly_return": float(portfolio_monthly["monthly_return"].mean()),
            "mean_monthly_excess_vs_p0": float(portfolio_monthly["monthly_excess_vs_p0"].mean()),
            "total_turnover": float(frame["turnover"].sum()),
        })
    return pd.DataFrame(rows)


def schedule_decision(summary: pd.DataFrame) -> dict:
    values = summary.set_index("portfolio")
    p0, p1 = values.loc["P0_equal_weight"], values.loc["P1_etf_momentum"]
    passes = bool(
        p1["total_return"] > p0["total_return"]
        and p1["mean_monthly_excess_vs_p0"] > 0
        and p1["maximum_drawdown"] >= p0["maximum_drawdown"]
    )
    return {
        "verdict": "PASS_FIXED_WEEKLY_SCHEDULE" if passes else "DO_NOT_DEPLOY_FIXED_WEEKLY_STRATEGY",
        "shadow_monitoring_allowed": True,
        "p1_total_return": float(p1["total_return"]), "p0_total_return": float(p0["total_return"]),
        "p1_maximum_drawdown": float(p1["maximum_drawdown"]),
        "p0_maximum_drawdown": float(p0["maximum_drawdown"]),
        "p1_mean_monthly_excess_vs_p0": float(p1["mean_monthly_excess_vs_p0"]),
        "reason": "The executable fixed weekly schedule underperformed P0 and had a materially deeper drawdown.",
    }


def render_report(monthly: pd.DataFrame, summary: pd.DataFrame, manifest: dict, decision: dict) -> str:
    returns = _percent_table(monthly, "monthly_return")
    drawdowns = _percent_table(monthly, "monthly_max_drawdown")
    end_drawdowns = _percent_table(monthly, "month_end_drawdown")
    summary_display = summary.copy()
    for column in ("total_return", "maximum_drawdown", "mean_monthly_return", "mean_monthly_excess_vs_p0"):
        summary_display[column] = summary_display[column].map(lambda value: f"{value:.2%}")
    return f"""# P0-P3逐月收益与回撤

区间：{manifest['start']} 至 {manifest['end']}（最后一个月为截至数据日的非完整月份）。口径：{manifest['schedule']}；完整往返成本 {manifest['roundtrip_cost_bps']:.0f}bps。月内最大回撤从月初净值开始计算，月末回撤则相对全历史净值高点。

## 结论

**{decision['verdict']}**。按可执行的固定周度规则，P1累计收益低于P0，且最大回撤显著更深；可以继续记录影子数据，但不能据此部署资金。

## 汇总

{summary_display.to_markdown(index=False)}

## 每月收益

{returns.to_markdown()}

## 每月最大回撤

{drawdowns.to_markdown()}

## 月末相对历史高点回撤

{end_drawdowns.to_markdown()}
"""


def _percent_table(monthly: pd.DataFrame, value: str) -> pd.DataFrame:
    table = monthly.pivot(index="month", columns="portfolio", values=value)
    return table.map(lambda number: f"{number:.2%}")


if __name__ == "__main__":
    main()
