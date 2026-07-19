from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factor_forge.research.concept_etf_shadow import (
    monthly_performance,
    nonoverlapping_holding_periods,
    simulate_staggered_sleeves,
)
from factor_forge.research.concept_rotation_alpha import (
    block_bootstrap_mean,
    newey_west_mean,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(config["source_artifact"])
    panel_path = source / "pit_signal_panel.parquet"
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = attach_robustness_momentum_scores(panel, config)
    specs = build_ofat_specifications(config)
    start = pd.Timestamp(config["evaluation"]["start"])
    end = pd.Timestamp(config["evaluation"]["end"])

    print(f"running {len(specs)} frozen R4 OFAT specifications", flush=True)
    summary_parts, sleeve_parts, nav_parts = [], [], []
    baseline_sleeves = None
    baseline_attribution = None
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec['specification']}", flush=True)
        _, raw_sleeves, attribution = simulate_staggered_sleeves(
            panel,
            "R4_rank_buffer",
            start=str(start.date()),
            end=str(end.date()),
            roundtrip_cost_bps=0,
            score_column=spec["score_column"],
            holding_days=int(spec["holding_days"]),
            execution_delay_days=int(spec["execution_delay_days"]),
            r4_selection_count=int(spec["selection_count"]),
            r4_retention_rank=int(spec["retention_rank"]),
            r4_maximum_etf_weight=float(spec["maximum_etf_weight"]),
            r4_absolute_momentum_column=spec["absolute_momentum_column"],
        )
        if spec["specification"] == "baseline":
            baseline_sleeves = raw_sleeves.copy()
            baseline_attribution = attribution.copy()
        for cost in (20, 40):
            daily, sleeves = reprice_sleeves(raw_sleeves, cost, spec["specification"])
            summary_parts.append(summarize_specification(daily, sleeves, spec, cost))
            nav_parts.append(daily)
            if spec["specification"] == "baseline":
                sleeve_parts.append(sleeve_absolute_statistics(sleeves, spec, cost))

    if baseline_sleeves is None or baseline_attribution is None:
        raise RuntimeError("baseline specification did not run")
    summary = pd.DataFrame(summary_parts)
    baseline_sleeve_stats = pd.concat(sleeve_parts, ignore_index=True)
    daily_nav = pd.concat(nav_parts, ignore_index=True)
    cost_stress, cost_nav = run_cost_stress(baseline_sleeves, config, start)
    monthly = baseline_monthly_performance(cost_nav)
    yearly = calendar_year_performance(cost_nav)
    rolling = rolling_window_performance(cost_nav, window=252)
    month_concentration = monthly_profit_concentration(monthly)
    decision = make_decision(
        summary,
        baseline_sleeve_stats,
        yearly,
        month_concentration,
        config["acceptance"],
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_root) / f"r4_r6_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "ofat_robustness_summary.csv", index=False, encoding="utf-8-sig")
    baseline_sleeve_stats.to_csv(
        output / "baseline_nonoverlap_sleeves.csv", index=False, encoding="utf-8-sig",
    )
    cost_stress.to_csv(output / "cost_stress.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "baseline_monthly_performance.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "baseline_calendar_years.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(output / "baseline_rolling_12m.csv", index=False, encoding="utf-8-sig")
    daily_nav.to_parquet(output / "ofat_daily_nav.parquet", index=False)
    cost_nav.to_parquet(output / "baseline_cost_daily_nav.parquet", index=False)
    baseline_attribution.to_csv(
        output / "baseline_etf_attribution_gross.csv", index=False, encoding="utf-8-sig",
    )
    manifest = {
        "source_artifact": str(source.resolve()),
        "source_panel": str(panel_path.resolve()),
        "evaluation_start": str(start.date()),
        "evaluation_end": str(end.date()),
        "specifications": len(specs),
        "policy": "R4_rank_buffer_dynamic_PIT",
        "s2_used": False,
        "frozen_forward_experiment_changed": False,
        "parameter_selection_performed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        render_report(summary, cost_stress, baseline_sleeve_stats, yearly, rolling, monthly, decision),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(output),
        "decision": decision,
        "baseline_cost_stress": cost_stress.to_dict("records"),
    }, ensure_ascii=False, indent=2))


def attach_robustness_momentum_scores(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = panel.sort_values(["ts_code", "trade_date"]).copy()
    lookbacks = {(20, 60)}
    for short, long in config["stress"]["one_factor_at_a_time"]["momentum_lookbacks"]:
        lookbacks.add((int(short), int(long)))
    for lookback in sorted({value for pair in lookbacks for value in pair}):
        column = f"robust_momentum_{lookback}d"
        result[column] = result.groupby("ts_code", sort=False)["adj_close"].pct_change(
            lookback, fill_method=None,
        )
    mapping = result["mapping_pass"].astype("boolean").fillna(False).astype(bool)
    for short, long in sorted(lookbacks):
        for short_weight in sorted({0.60, *map(
            float, config["stress"]["one_factor_at_a_time"]["short_weight"],
        )}):
            column = score_column(short, long, short_weight)
            short_z = result.loc[mapping].groupby("trade_date")[
                f"robust_momentum_{short}d"
            ].transform(cross_sectional_zscore)
            long_z = result.loc[mapping].groupby("trade_date")[
                f"robust_momentum_{long}d"
            ].transform(cross_sectional_zscore)
            result[column] = np.nan
            result.loc[mapping, column] = short_weight * short_z + (1 - short_weight) * long_z
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    standard_deviation = numeric.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / standard_deviation


def score_column(short: int, long: int, short_weight: float) -> str:
    weight = int(round(short_weight * 100))
    return f"robust_score_{short}_{long}_w{weight}"


def build_ofat_specifications(config: dict) -> list[dict]:
    base = config["baseline"]
    baseline = {
        "specification": "baseline",
        "family": "baseline",
        "holding_days": int(base["holding_days"]),
        "execution_delay_days": int(base["execution_delay_days"]),
        "selection_count": int(base["selection_count"]),
        "retention_rank": int(base["retention_rank"]),
        "maximum_etf_weight": float(base["maximum_etf_weight"]),
        "short_lookback": int(base["short_lookback"]),
        "long_lookback": int(base["long_lookback"]),
        "short_weight": float(base["short_weight"]),
    }
    specs = [baseline]
    stress = config["stress"]["one_factor_at_a_time"]
    simple_parameters = (
        "holding_days", "execution_delay_days", "selection_count",
        "retention_rank", "maximum_etf_weight", "short_weight",
    )
    for parameter in simple_parameters:
        for value in stress[parameter]:
            spec = baseline.copy()
            spec.update({
                "specification": f"{parameter}_{str(value).replace('.', 'p')}",
                "family": parameter,
                parameter: value,
            })
            if parameter == "selection_count":
                spec["retention_rank"] = int(value) + 2
            specs.append(spec)
    for short, long in stress["momentum_lookbacks"]:
        spec = baseline.copy()
        spec.update({
            "specification": f"momentum_{short}_{long}",
            "family": "momentum_lookbacks",
            "short_lookback": int(short),
            "long_lookback": int(long),
        })
        specs.append(spec)
    for spec in specs:
        spec["score_column"] = score_column(
            int(spec["short_lookback"]),
            int(spec["long_lookback"]),
            float(spec["short_weight"]),
        )
        spec["absolute_momentum_column"] = (
            f"robust_momentum_{int(spec['long_lookback'])}d"
        )
    return specs


def reprice_sleeves(
    raw_sleeves: pd.DataFrame,
    cost_bps: float,
    specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sleeves = raw_sleeves.copy()
    sleeves["cost_drag"] = sleeves["turnover"] * float(cost_bps) / 10_000
    sleeves["net_return"] = sleeves["gross_return"] - sleeves["cost_drag"]
    sleeves["net_nav"] = sleeves.groupby("sleeve", sort=False)["net_return"].transform(
        lambda values: (1 + values).cumprod(),
    )
    sleeves["nav_before"] = sleeves.groupby("sleeve", sort=False)["net_nav"].shift(
        fill_value=1.0,
    )
    sleeves["specification"] = specification
    sleeves["roundtrip_cost_bps"] = int(cost_bps)
    rows = []
    previous_nav = 1.0
    for date, day in sleeves.groupby("return_date", sort=True):
        nav = float(day["net_nav"].mean())
        capital = day["nav_before"] / day["nav_before"].sum()
        rows.append({
            "return_date": date,
            "portfolio": specification,
            "specification": specification,
            "roundtrip_cost_bps": int(cost_bps),
            "net_nav": nav,
            "net_return": nav / previous_nav - 1,
            "gross_return": float((capital * day["gross_return"]).sum()),
            "turnover": float((capital * day["turnover"]).sum()),
            "cost_drag": float((capital * day["cost_drag"]).sum()),
            "cash_weight": float((capital * day["cash_weight"]).sum()),
            "is_rebalance": bool(day["is_rebalance"].any()),
        })
        previous_nav = nav
    return pd.DataFrame(rows), sleeves


def summarize_specification(daily, sleeves, spec, cost) -> dict:
    drawdown = daily["net_nav"] / daily["net_nav"].cummax().clip(lower=1.0) - 1
    periods = nonoverlapping_holding_periods(
        sleeves, holding_days=int(spec["holding_days"]),
    )
    totals = periods.groupby("sleeve")["net_return"].apply(
        lambda values: float(np.prod(1 + values) - 1),
    )
    years = max((daily["return_date"].max() - daily["return_date"].min()).days / 365.25, 1 / 252)
    return {
        **spec,
        "roundtrip_cost_bps": int(cost),
        "total_return": float(daily["net_nav"].iloc[-1] - 1),
        "annualized_return": float(daily["net_nav"].iloc[-1] ** (1 / years) - 1),
        "maximum_drawdown": float(drawdown.min()),
        "mean_daily_turnover": float(daily["turnover"].mean()),
        "mean_cash_weight": float(daily["cash_weight"].mean()),
        "positive_sleeves": int(totals.gt(0).sum()),
        "sleeves": int(len(totals)),
        "minimum_sleeve_total_return": float(totals.min()),
        "median_sleeve_total_return": float(totals.median()),
    }


def sleeve_absolute_statistics(sleeves, spec, cost) -> pd.DataFrame:
    periods = nonoverlapping_holding_periods(
        sleeves, holding_days=int(spec["holding_days"]),
    )
    rows = []
    for sleeve, group in periods.groupby("sleeve", observed=True):
        statistics = newey_west_mean(group["net_return"], 0)
        low, high = block_bootstrap_mean(
            group["net_return"].to_numpy(float),
            block=4,
            samples=2_000,
            seed=20260718 + int(sleeve),
        )
        nav = (1 + group["net_return"]).cumprod()
        drawdown = nav / nav.cummax().clip(lower=1.0) - 1
        rows.append({
            "roundtrip_cost_bps": int(cost),
            "sleeve": int(sleeve),
            "periods": len(group),
            "mean_period_return": statistics["mean"],
            "mean_return_t": statistics["t_value"],
            "bootstrap_95_low": low,
            "bootstrap_95_high": high,
            "positive_period_rate": float(group["net_return"].gt(0).mean()),
            "total_return": float(nav.iloc[-1] - 1),
            "maximum_drawdown": float(drawdown.min()),
        })
    return pd.DataFrame(rows)


def run_cost_stress(raw_sleeves, config, start):
    summaries, navs = [], []
    for cost in config["stress"]["roundtrip_cost_bps"]:
        daily, sleeves = reprice_sleeves(raw_sleeves, cost, "baseline")
        daily = daily.loc[daily["return_date"].gt(start)].copy()
        daily["net_nav"] = (1 + daily["net_return"]).cumprod()
        drawdown = daily["net_nav"] / daily["net_nav"].cummax().clip(lower=1.0) - 1
        summaries.append({
            "roundtrip_cost_bps": int(cost),
            "total_return": float(daily["net_nav"].iloc[-1] - 1),
            "maximum_drawdown": float(drawdown.min()),
            "mean_daily_turnover": float(daily["turnover"].mean()),
            "total_cost_drag": float(daily["cost_drag"].sum()),
        })
        navs.append(daily)
    return pd.DataFrame(summaries), pd.concat(navs, ignore_index=True)


def baseline_monthly_performance(cost_nav):
    parts = []
    for cost, daily in cost_nav.groupby("roundtrip_cost_bps", observed=True):
        daily = daily.copy()
        daily["portfolio"] = "baseline"
        month = monthly_performance(daily, benchmark_portfolio="baseline")
        month["roundtrip_cost_bps"] = int(cost)
        parts.append(month)
    return pd.concat(parts, ignore_index=True)


def calendar_year_performance(cost_nav):
    rows = []
    for cost, daily in cost_nav.groupby("roundtrip_cost_bps", observed=True):
        frame = daily.sort_values("return_date").copy()
        frame["year"] = frame["return_date"].dt.year
        for year, group in frame.groupby("year"):
            nav = (1 + group["net_return"]).cumprod()
            drawdown = nav / nav.cummax().clip(lower=1.0) - 1
            rows.append({
                "roundtrip_cost_bps": int(cost),
                "year": int(year),
                "trading_days": len(group),
                "return": float(nav.iloc[-1] - 1),
                "maximum_drawdown": float(drawdown.min()),
            })
    return pd.DataFrame(rows)


def rolling_window_performance(cost_nav, *, window):
    rows = []
    for cost, daily in cost_nav.groupby("roundtrip_cost_bps", observed=True):
        frame = daily.sort_values("return_date").reset_index(drop=True)
        for end_index in range(window - 1, len(frame), 21):
            sample = frame.iloc[end_index - window + 1:end_index + 1]
            rows.append({
                "roundtrip_cost_bps": int(cost),
                "window_end": sample["return_date"].iloc[-1],
                "trading_days": len(sample),
                "return": float(np.prod(1 + sample["net_return"]) - 1),
            })
    return pd.DataFrame(rows)


def monthly_profit_concentration(monthly):
    sample = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)].copy()
    positive = sample["monthly_return"].clip(lower=0)
    return float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0


def make_decision(summary, sleeve_stats, yearly, month_concentration, acceptance):
    neighborhood = summary.loc[summary["specification"].ne("baseline")]
    positive_20 = float(neighborhood.loc[
        neighborhood["roundtrip_cost_bps"].eq(20), "total_return",
    ].gt(0).mean())
    positive_40 = float(neighborhood.loc[
        neighborhood["roundtrip_cost_bps"].eq(40), "total_return",
    ].gt(0).mean())
    baseline_sleeves = sleeve_stats.loc[sleeve_stats["roundtrip_cost_bps"].eq(20)]
    years20 = yearly.loc[
        yearly["roundtrip_cost_bps"].eq(20) & yearly["trading_days"].ge(100)
    ]
    year_fraction = float(years20["return"].gt(0).mean())
    checks = {
        "positive_neighborhood_fraction_20bps": positive_20 >= float(
            acceptance["minimum_positive_neighborhood_fraction_20bps"]
        ),
        "positive_neighborhood_fraction_40bps": positive_40 >= float(
            acceptance["minimum_positive_neighborhood_fraction_40bps"]
        ),
        "positive_baseline_sleeves": int(baseline_sleeves["total_return"].gt(0).sum())
        >= int(acceptance["minimum_positive_baseline_sleeves"]),
        "positive_calendar_year_fraction": year_fraction >= float(
            acceptance["minimum_positive_calendar_year_fraction"]
        ),
        "monthly_profit_concentration": month_concentration <= float(
            acceptance["maximum_monthly_positive_profit_share"]
        ),
    }
    passed = all(checks.values())
    return {
        "verdict": "R4_ROBUSTNESS_PASSED_KEEP_AS_SHADOW_BENCHMARK" if passed
        else "R4_ROBUSTNESS_FAILED_RESEARCH_ONLY",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "positive_neighborhood_fraction_20bps": positive_20,
            "positive_neighborhood_fraction_40bps": positive_40,
            "positive_baseline_sleeves_20bps": int(
                baseline_sleeves["total_return"].gt(0).sum()
            ),
            "positive_calendar_year_fraction_20bps": year_fraction,
            "monthly_positive_profit_share_20bps": month_concentration,
        },
        "original_baseline_remains_frozen": True,
        "parameter_selection_performed": False,
        "s2_used": False,
        "frozen_forward_experiment_changed": False,
        "not_real_money_authorization": True,
    }


def render_report(summary, cost, sleeves, yearly, rolling, monthly, decision):
    baseline = summary.loc[summary["specification"].eq("baseline")].copy()
    display = baseline[[
        "roundtrip_cost_bps", "total_return", "annualized_return", "maximum_drawdown",
        "mean_daily_turnover", "mean_cash_weight", "positive_sleeves", "sleeves",
    ]].copy()
    display = format_percent(display, [
        "total_return", "annualized_return", "maximum_drawdown",
        "mean_daily_turnover", "mean_cash_weight",
    ])
    family = summary.loc[
        summary["roundtrip_cost_bps"].eq(20) & summary["specification"].ne("baseline")
    ].groupby("family", as_index=False).agg(
        specifications=("specification", "size"),
        minimum_total_return=("total_return", "min"),
        median_total_return=("total_return", "median"),
        maximum_total_return=("total_return", "max"),
        worst_drawdown=("maximum_drawdown", "min"),
    )
    family = format_percent(family, [
        "minimum_total_return", "median_total_return", "maximum_total_return", "worst_drawdown",
    ])
    cost_display = format_percent(cost.copy(), [
        "total_return", "maximum_drawdown", "mean_daily_turnover", "total_cost_drag",
    ])
    sleeve_display = sleeves.loc[sleeves["roundtrip_cost_bps"].eq(20)].copy()
    sleeve_display = format_percent(sleeve_display, [
        "mean_period_return", "bootstrap_95_low", "bootstrap_95_high",
        "positive_period_rate", "total_return", "maximum_drawdown",
    ])
    year_display = format_percent(
        yearly.loc[yearly["roundtrip_cost_bps"].eq(20)].copy(),
        ["return", "maximum_drawdown"],
    )
    rolling20 = rolling.loc[rolling["roundtrip_cost_bps"].eq(20), "return"]
    monthly20 = monthly.loc[monthly["roundtrip_cost_bps"].eq(20)]
    return f"""# R6 动态PIT R4稳健性复验

## 结论

**{decision['verdict']}**。本轮是单因素邻域检验，不选择最优参数；R4原始5日规则保持不变，S2未参与，冻结前瞻实验未修改。

## 原始基线

{display.to_markdown(index=False)}

## 单因素参数邻域（20bp）

{family.to_markdown(index=False)}

邻域正收益比例：20bp为 {decision['metrics']['positive_neighborhood_fraction_20bps']:.1%}，40bp为 {decision['metrics']['positive_neighborhood_fraction_40bps']:.1%}。

## 成本压力

{cost_display.to_markdown(index=False)}

## 非重叠持仓袖套（20bp）

{sleeve_display.to_markdown(index=False)}

## 时间分段（20bp）

{year_display.to_markdown(index=False)}

- 252日滚动窗口数量：{len(rolling20)}；正收益比例：{rolling20.gt(0).mean():.1%}；最差滚动收益：{rolling20.min():.2%}。
- 正收益月份占比：{monthly20['monthly_return'].gt(0).mean():.1%}。
- 最大单月正收益贡献占比：{decision['metrics']['monthly_positive_profit_share_20bps']:.1%}。

## 使用边界

通过稳健性门槛只允许将动态PIT R4作为影子基准继续观察，不构成实盘授权，也不允许根据本表挑选历史最优参数。
"""


def format_percent(frame, columns):
    result = frame.copy()
    for column in columns:
        result[column] = result[column].map(lambda value: f"{value:.2%}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R6 dynamic PIT R4 robustness experiment")
    parser.add_argument(
        "--config", default="configs/research/index_backed_r4_robustness_v1.yaml",
    )
    parser.add_argument("--output-root", default="artifacts/index_backed_r4_robustness")
    return parser.parse_args()


if __name__ == "__main__":
    main()
