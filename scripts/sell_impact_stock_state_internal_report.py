from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sell_impact_factor_attribution as attr
import sell_impact_sorting_repair as base


SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_score_band_walkforward_20260708T091419Z")
ATTRIBUTION_RUN = Path("artifacts/strategy_reviews/sell_impact_factor_attribution_20260708T134826Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
START_DATE = "2024-01-01"

INTERNAL_COMPONENTS = {
    "stock_state_low_vol": {
        "source": "volatility_20_z",
        "formula": "-volatility_20_z",
        "name": "低波动暴露",
        "interpretation": "值越高，表示20日波动率越低。",
    },
    "stock_state_small_size": {
        "source": "log_circ_mv_z",
        "formula": "-log_circ_mv_z",
        "name": "小市值暴露",
        "interpretation": "值越高，表示流通市值越小。",
    },
}


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_stock_state_internal_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)

    dataset = load_dataset()
    picks = load_top5_picks()
    enriched_picks = enrich_picks(picks, dataset)

    component_map = component_definition()
    ic = build_ic_report(dataset)
    yearly_ic = build_yearly_ic(dataset)
    deciles = build_deciles(dataset)
    corr = build_corr(dataset)
    top5 = build_top5_exposure(enriched_picks)
    quadrant = build_quadrant_payoff(enriched_picks)
    allocation = build_contribution_allocation(enriched_picks)

    component_map.to_csv(output / "stock_state_component_definition.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(output / "stock_state_internal_ic.csv", index=False, encoding="utf-8-sig")
    yearly_ic.to_csv(output / "stock_state_internal_yearly_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(output / "stock_state_internal_decile_returns.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(output / "stock_state_internal_correlation.csv", encoding="utf-8-sig")
    enriched_picks.to_csv(output / "top5_stock_state_internal_detail.csv", index=False, encoding="utf-8-sig")
    top5.to_csv(output / "top5_stock_state_internal_summary.csv", index=False, encoding="utf-8-sig")
    quadrant.to_csv(output / "top5_stock_state_quadrant_payoff.csv", index=False, encoding="utf-8-sig")
    allocation.to_csv(output / "top5_stock_state_contribution_allocation.csv", index=False, encoding="utf-8-sig")

    plot_deciles(deciles, output / "stock_state_internal_decile_returns.png")
    write_report(output, component_map, ic, yearly_ic, deciles, corr, top5, quadrant, allocation)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "attribution_run": str(ATTRIBUTION_RUN),
                "start_date": START_DATE,
                "note": "cluster_stock_state = mean(-volatility_20_z, -log_circ_mv_z). Contribution allocation is an economic approximation.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done -> {output}")


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_parquet(SOURCE_RUN / "walkforward_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    dataset = dataset.loc[dataset["trade_date"].ge(pd.Timestamp(START_DATE))].copy()
    dataset = dataset.loc[dataset["ts_code"].map(attr.permission_eligible)].copy()
    dataset["stock_state_low_vol"] = -pd.to_numeric(dataset["volatility_20_z"], errors="coerce")
    dataset["stock_state_small_size"] = -pd.to_numeric(dataset["log_circ_mv_z"], errors="coerce")
    dataset["stock_state_rebuilt"] = dataset[["stock_state_low_vol", "stock_state_small_size"]].mean(axis=1)
    return dataset


def load_top5_picks() -> pd.DataFrame:
    picks = pd.read_csv(ATTRIBUTION_RUN / "top5_pick_attribution.csv")
    picks["trade_date"] = pd.to_datetime(picks["trade_date"])
    return picks


def enrich_picks(picks: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trade_date",
        "ts_code",
        "cluster_stock_state",
        "stock_state_low_vol",
        "stock_state_small_size",
        "volatility_20_z",
        "log_circ_mv_z",
    ]
    out = picks.merge(dataset[cols], on=["trade_date", "ts_code"], how="left", suffixes=("", "_dataset"))
    out["stock_state_internal_sum_abs"] = out[["stock_state_low_vol", "stock_state_small_size"]].abs().sum(axis=1)
    denom = out["stock_state_internal_sum_abs"].replace(0, np.nan)
    out["low_vol_abs_share"] = out["stock_state_low_vol"].abs() / denom
    out["small_size_abs_share"] = out["stock_state_small_size"].abs() / denom
    out["stock_state_internal_dominant"] = np.where(
        out["stock_state_low_vol"].abs().ge(out["stock_state_small_size"].abs()),
        "stock_state_low_vol",
        "stock_state_small_size",
    )
    out["stock_state_internal_quadrant"] = np.select(
        [
            out["stock_state_low_vol"].gt(0) & out["stock_state_small_size"].gt(0),
            out["stock_state_low_vol"].gt(0) & out["stock_state_small_size"].le(0),
            out["stock_state_low_vol"].le(0) & out["stock_state_small_size"].gt(0),
        ],
        ["both_low_vol_and_small_size", "low_vol_only", "small_size_only"],
        default="neither",
    )
    out["approx_low_vol_model_contribution"] = out["cluster_stock_state"] * out["low_vol_abs_share"].fillna(0.5)
    out["approx_small_size_model_contribution"] = out["cluster_stock_state"] * out["small_size_abs_share"].fillna(0.5)
    return out


def component_definition() -> pd.DataFrame:
    rows = []
    for key, meta in INTERNAL_COMPONENTS.items():
        rows.append({"component": key, **meta})
    rows.append(
        {
            "component": "cluster_stock_state",
            "source": "stock_state_low_vol + stock_state_small_size",
            "formula": "mean(stock_state_low_vol, stock_state_small_size)",
            "name": "个股状态聚合",
            "interpretation": "值越高，表示更偏低波动和/或更偏小市值。",
        }
    )
    return pd.DataFrame(rows)


def build_ic_report(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in factor_cols():
        daily = daily_ic(dataset, factor)
        rows.append(ic_summary(daily["rank_ic"], factor, "2024_2026"))
    return pd.DataFrame(rows).sort_values("rank_ic_mean", ascending=False)


def build_yearly_ic(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in factor_cols():
        daily = daily_ic(dataset, factor)
        if daily.empty:
            continue
        daily["year"] = daily["trade_date"].dt.year
        for year, frame in daily.groupby("year"):
            rows.append(ic_summary(frame["rank_ic"], factor, int(year)))
    return pd.DataFrame(rows).sort_values(["factor", "window"])


def build_deciles(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in factor_cols():
        dec = decile_return(dataset, factor)
        dec["factor"] = factor
        rows.append(dec)
    return pd.concat(rows, ignore_index=True)


def build_corr(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[factor_cols()].replace([np.inf, -np.inf], np.nan).corr(method="spearman")


def build_top5_exposure(picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in picks.groupby(["fold", "stock_state_internal_dominant"], dropna=False):
        fold, dominant = keys
        labels = pd.to_numeric(frame["label"], errors="coerce")
        rows.append(
            {
                "fold": fold,
                "dominant_component": dominant,
                "pick_count": int(len(frame)),
                "pick_ratio": float(len(frame) / len(picks.loc[picks["fold"].eq(fold)])),
                "mean_forward_return": float(labels.mean()),
                "hit_rate": float((labels > 0).mean()),
                "mean_low_vol": float(frame["stock_state_low_vol"].mean()),
                "mean_small_size": float(frame["stock_state_small_size"].mean()),
                "mean_low_vol_abs_share": float(frame["low_vol_abs_share"].mean()),
                "mean_small_size_abs_share": float(frame["small_size_abs_share"].mean()),
                "mean_cluster_stock_state_contribution": float(frame["cluster_stock_state"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "pick_ratio"], ascending=[True, False])


def build_quadrant_payoff(picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in picks.groupby(["fold", "stock_state_internal_quadrant"], dropna=False):
        fold, quadrant = keys
        labels = pd.to_numeric(frame["label"], errors="coerce")
        rows.append(
            {
                "fold": fold,
                "quadrant": quadrant,
                "pick_count": int(len(frame)),
                "pick_ratio": float(len(frame) / len(picks.loc[picks["fold"].eq(fold)])),
                "mean_forward_return": float(labels.mean()),
                "hit_rate": float((labels > 0).mean()),
                "mean_cluster_stock_state_contribution": float(frame["cluster_stock_state"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "pick_count"], ascending=[True, False])


def build_contribution_allocation(picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, frame in picks.groupby("fold"):
        labels = pd.to_numeric(frame["label"], errors="coerce")
        for component, contribution_col, share_col in [
            ("stock_state_low_vol", "approx_low_vol_model_contribution", "low_vol_abs_share"),
            ("stock_state_small_size", "approx_small_size_model_contribution", "small_size_abs_share"),
        ]:
            weights = pd.to_numeric(frame[share_col], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "fold": fold,
                    "component": component,
                    "component_name": INTERNAL_COMPONENTS[component]["name"],
                    "mean_abs_share": float(weights.mean()),
                    "mean_approx_model_contribution": float(frame[contribution_col].mean()),
                    "weighted_forward_return": weighted_mean(labels, weights),
                }
            )
    return pd.DataFrame(rows)


def factor_cols() -> list[str]:
    return ["cluster_stock_state", "stock_state_low_vol", "stock_state_small_size"]


def daily_ic(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("trade_date"):
        data = group[[factor, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 30 or data[factor].nunique() < 3 or data["label"].nunique() < 3:
            continue
        rows.append(
            {
                "trade_date": pd.Timestamp(date),
                "rank_ic": float(data[factor].corr(data["label"], method="spearman")),
                "pearson_ic": float(data[factor].corr(data["label"], method="pearson")),
                "n": int(len(data)),
            }
        )
    return pd.DataFrame(rows, columns=["trade_date", "rank_ic", "pearson_ic", "n"])


def ic_summary(values: pd.Series, factor: str, window: Any) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else np.nan
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return {
        "factor": factor,
        "window": window,
        "days": int(len(values)),
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "icir": float(mean / std * np.sqrt(252)) if std and np.isfinite(std) and std > 0 else np.nan,
        "positive_ratio": float((values > 0).mean()) if len(values) else np.nan,
    }


def decile_return(frame: pd.DataFrame, factor: str, bins: int = 10) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("trade_date"):
        data = group[[factor, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < bins * 10 or data[factor].nunique() < bins:
            continue
        data = data.assign(decile=pd.qcut(data[factor].rank(method="first"), bins, labels=False) + 1)
        for decile, local in data.groupby("decile"):
            rows.append(
                {
                    "trade_date": pd.Timestamp(date),
                    "decile": int(decile),
                    "mean_forward_return": float(local["label"].mean()),
                    "count": int(len(local)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["decile", "mean_forward_return", "count", "days"])
    data = pd.DataFrame(rows)
    return (
        data.groupby("decile", as_index=False)
        .agg(mean_forward_return=("mean_forward_return", "mean"), count=("count", "mean"), days=("trade_date", "nunique"))
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any() or w[mask].sum() <= 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def plot_deciles(deciles: pd.DataFrame, path: Path) -> None:
    if deciles.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for factor in factor_cols():
        local = deciles.loc[deciles["factor"].eq(factor)].sort_values("decile")
        ax.plot(local["decile"], local["mean_forward_return"], marker="o", label=factor)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("cluster_stock_state internal decile returns")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Mean 10d forward return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    component_map: pd.DataFrame,
    ic: pd.DataFrame,
    yearly_ic: pd.DataFrame,
    deciles: pd.DataFrame,
    corr: pd.DataFrame,
    top5: pd.DataFrame,
    quadrant: pd.DataFrame,
    allocation: pd.DataFrame,
) -> None:
    lines = [
        "# cluster_stock_state Internal Factor Decomposition",
        "",
        "## Scope",
        f"- Source dataset: `{SOURCE_RUN}`",
        f"- Top5 attribution source: `{ATTRIBUTION_RUN}`",
        "- Stock pool: main-board permission pool, excluding STAR, ChiNext and Beijing.",
        "- Important: the production LightGBM receives `cluster_stock_state` and its regime interactions, not the two internal components directly.",
        "",
        "## Component Definition",
        md_table(component_map),
        "",
        "## IC Summary",
        md_table(ic),
        "",
        "## Yearly IC",
        md_table(yearly_ic, 30),
        "",
        "## Correlation",
        md_table(corr.reset_index().rename(columns={"index": "factor"})),
        "",
        "## Top5 Dominant Component",
        md_table(top5, 30),
        "",
        "## Top5 Quadrant Payoff",
        md_table(quadrant, 30),
        "",
        "## Approximate Model Contribution Allocation",
        "This allocation splits the model-level `cluster_stock_state` contribution by the absolute internal component shares.",
        md_table(allocation, 30),
        "",
        "## Files",
        "- `stock_state_component_definition.csv`",
        "- `stock_state_internal_ic.csv`",
        "- `stock_state_internal_yearly_ic.csv`",
        "- `stock_state_internal_decile_returns.csv` and `stock_state_internal_decile_returns.png`",
        "- `stock_state_internal_correlation.csv`",
        "- `top5_stock_state_internal_detail.csv`",
        "- `top5_stock_state_internal_summary.csv`",
        "- `top5_stock_state_quadrant_payoff.csv`",
        "- `top5_stock_state_contribution_allocation.csv`",
    ]
    (output / "cluster_stock_state_internal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
