from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import reliability_feature_columns
from .labels import future_diagnostic_columns, reliability_label_columns


def write_reliability_dataset_report(
    *,
    output: Path,
    dataset: pd.DataFrame,
    source_path: str,
    cost_buffer: float,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = reliability_label_columns(horizons)
    diagnostics = future_diagnostic_columns(horizons)
    features = reliability_feature_columns(dataset)
    label_summary = _label_summary(dataset, labels)
    bucket_summary = _bucket_summary(dataset, horizons)
    lines = [
        "# Factor Reliability Dataset Report",
        "",
        "## Scope",
        f"- Source health file: `{source_path}`",
        f"- Date range: `{dataset['date'].min()}` to `{dataset['date'].max()}`",
        f"- Rows: `{len(dataset)}`",
        f"- Cost buffer: `{cost_buffer}`",
        "- Objective: build short-horizon factor reliability features and labels only; no model training is run in this stage.",
        "",
        "## Feature Columns",
        "\n".join(f"- `{col}`" for col in features),
        "",
        "## Label Columns",
        "\n".join(f"- `{col}`" for col in labels),
        "",
        "## Future Diagnostic Columns",
        "\n".join(f"- `{col}`" for col in diagnostics),
        "",
        "## Label Summary",
        _md_table(label_summary),
        "",
        "## Raw Health Bucket Test",
        "This is not model output. It checks whether current rolling health already separates future spread.",
        _md_table(bucket_summary),
        "",
        "## Leakage Guard",
        "- Feature columns are computed from current and trailing windows only.",
        "- Future spread and future rank IC are present only as labels/diagnostics.",
        "- Do not include `future_*` or `validity_label_*` columns in model features.",
    ]
    (output / "factor_reliability_dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reliability_model_report(
    *,
    output: Path,
    dataset: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    reliability_daily: pd.DataFrame,
    simulation: pd.DataFrame,
    source_path: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Factor Reliability Model v1 Report",
        "",
        "## Dataset Summary",
        f"- Source dataset: `{source_path}`",
        f"- Date range: `{dataset['date'].min()}` to `{dataset['date'].max()}`",
        f"- Rows: `{len(dataset)}`",
        "- Split: train 2024-01-02..2025-06-30, validation 2025-07-01..2025-12-31, test 2026-01-01..2026-06-30.",
        "- Objective: predict short-horizon future factor spread, not stock return or portfolio return.",
        "",
        "## Model Comparison",
        _md_table(results["metrics"]),
        "",
        "## Rank IC Comparison",
        _md_table(results["metrics"][["horizon", "model", "sample", "rank_ic", "q5_gt_q1", "rows"]]),
        "",
        "## Bucket Monotonicity",
        _md_table(results["bucket_test"]),
        "",
        "## Calibration",
        _md_table(results["calibration"]),
        "",
        "## Stability",
        _md_table(results["stability"]),
        "",
        "## Feature Importance",
        _md_table(results["feature_importance"].head(80)),
        "",
        "## Reliability Distribution",
        _md_table(_reliability_distribution(reliability_daily)),
        "",
        "## Dynamic Factor Weighting Simulation",
        "Simulation uses factor-level spread only: fixed spread vs spread weighted by reliability score. It does not change position count, cash exposure, or portfolio execution.",
        _md_table(simulation),
        "",
        "## Leakage Guard",
        "- Model features exclude `future_*` and `validity_label_*` columns.",
        "- Reliability percentile mapping is calibrated on train+validation predictions only.",
        "- LightGBM uses shallow parameters only: max_depth=3, num_leaves=8, learning_rate=0.05, n_estimators=240.",
    ]
    (output / "factor_reliability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dynamic_weighting_simulation(
    dataset: pd.DataFrame,
    reliability_daily: pd.DataFrame,
    *,
    horizon: int = 10,
) -> pd.DataFrame:
    spread_col = f"future_top_bottom_spread_{horizon}d"
    reliability_col = f"reliability_{horizon}d"
    merged = dataset[["date", "factor_name", spread_col]].merge(
        reliability_daily[["date", "factor_name", "sample", reliability_col]],
        on=["date", "factor_name"],
        how="inner",
    ).dropna(subset=[spread_col, reliability_col])
    rows = []
    for sample, group in merged.groupby("sample", sort=False):
        fixed = group[spread_col]
        weighted = group[spread_col] * group[reliability_col]
        rows.append(_simulation_row(sample, "baseline_fixed_weight", fixed))
        rows.append(_simulation_row(sample, "reliability_weighted", weighted))
    return pd.DataFrame(rows)


def _simulation_row(sample: str, variant: str, series: pd.Series) -> dict:
    std = series.std(ddof=1)
    curve = series.cumsum()
    drawdown = curve - curve.cummax()
    return {
        "sample": sample,
        "variant": variant,
        "mean_spread": float(series.mean()),
        "spread_std": float(std),
        "icir_like": float(series.mean() / std) if std and std > 0 else pd.NA,
        "max_drawdown_like": float(drawdown.min()) if len(drawdown) else pd.NA,
        "alpha_decay": float(series.tail(max(5, len(series) // 5)).mean() - series.head(max(5, len(series) // 5)).mean()) if len(series) >= 10 else pd.NA,
        "turnover_proxy": 0.0 if variant == "baseline_fixed_weight" else float(series.diff().abs().mean()),
        "rows": int(len(series)),
    }


def _reliability_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in frame.columns if c.startswith("reliability_")]:
        sample = frame[col].dropna()
        rows.append(
            {
                "reliability": col,
                "rows": int(len(sample)),
                "mean": float(sample.mean()) if len(sample) else pd.NA,
                "std": float(sample.std(ddof=1)) if len(sample) > 1 else pd.NA,
                "p10": float(sample.quantile(0.10)) if len(sample) else pd.NA,
                "p50": float(sample.quantile(0.50)) if len(sample) else pd.NA,
                "p90": float(sample.quantile(0.90)) if len(sample) else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _label_summary(dataset: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        sample = dataset[label].dropna()
        rows.append(
            {
                "label": label,
                "rows": int(len(sample)),
                "positive_ratio": float(sample.mean()) if len(sample) else pd.NA,
                "positive_count": int(sample.sum()) if len(sample) else 0,
                "negative_count": int((sample == 0).sum()) if len(sample) else 0,
            }
        )
    return pd.DataFrame(rows)


def _bucket_summary(dataset: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    if "spread_20" not in dataset.columns:
        return pd.DataFrame()
    frame = dataset.copy()
    try:
        frame["health_bucket"] = pd.qcut(frame["spread_20"].rank(method="first"), 5, labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    except ValueError:
        return pd.DataFrame()
    rows = []
    for horizon in horizons:
        spread_col = f"future_top_bottom_spread_{horizon}d"
        label_col = f"validity_label_{horizon}d"
        if spread_col not in frame or label_col not in frame:
            continue
        for bucket, group in frame.dropna(subset=[spread_col, label_col]).groupby("health_bucket", observed=True):
            rows.append(
                {
                    "horizon": horizon,
                    "health_bucket": str(bucket),
                    "future_spread_mean": float(group[spread_col].mean()),
                    "win_ratio": float(group[label_col].mean()),
                    "rows": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def _md_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.round(6).to_markdown(index=False)
