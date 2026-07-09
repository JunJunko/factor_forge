from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sell_impact_low_vol_regime_experiment as low_vol
import sell_impact_model_weight_attribution_experiment as attr
import sell_impact_ranker_timing_compare as timing_compare
import sell_impact_score_band_walkforward as wf
import sell_impact_sorting_repair as base


ATTRIBUTION_RUN = Path("artifacts/strategy_reviews/sell_impact_model_weight_attribution_20260709T062805Z")
OUTPUT_ROOT = Path("artifacts/strategy_reviews")
HOLDING_DAYS = 10
LOOKBACKS = [40, 80, 120]
MIN_OBS = 15


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_online_group_reweight_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"loading attribution run={ATTRIBUTION_RUN}")
    scored = pd.read_parquet(ATTRIBUTION_RUN / "score_group_contributions.parquet")
    scored["trade_date"] = pd.to_datetime(scored["trade_date"])
    group_cols = [col for col in scored.columns if col.startswith("group__")]
    groups = [col.removeprefix("group__") for col in group_cols]
    log(f"rows={len(scored):,} groups={len(groups)}")

    dataset = low_vol.load_dataset(log)
    version, panel = base.load_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    market_benchmark = timing_compare.load_market_benchmark(version)
    position_multiplier = timing_compare.load_position_multiplier(timing_compare.TIMING_DAILY)

    prediction_rows: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    for fold in wf.FOLDS:
        fold_name = fold["fold"]
        fold_frame = scored.loc[scored["fold"].eq(fold_name)].copy()
        log(f"{fold_name}: online reweight rows={len(fold_frame):,}")
        raw_test = fold_frame.loc[fold_frame["sample"].eq("test"), ["trade_date", "ts_code", "label", "raw_model"]].rename(
            columns={"raw_model": "score"}
        )
        raw_test["score_variant"] = "raw_model"
        raw_test["fold"] = fold_name
        prediction_rows.append(raw_test)

        for lookback in LOOKBACKS:
            variant = f"online_group_weighted_lb{lookback}"
            pred, weights = build_online_score(fold_frame, groups, lookback)
            pred["score_variant"] = variant
            pred["fold"] = fold_name
            prediction_rows.append(pred)
            for row in weights:
                row["score_variant"] = variant
                row["fold"] = fold_name
            weight_rows.extend(weights)

        for pred in prediction_rows_for_fold(prediction_rows, fold_name):
            variant = str(pred["score_variant"].iloc[0])
            ic_rows.append(attr.ic_summary(low_vol.daily_rank_ic(pred), variant, fold_name, "test"))
            decile_rows.extend(attr.decile_spread(pred, variant, fold_name))
            portfolio_rows.append(
                attr.run_portfolio(
                    panel=panel,
                    dataset=dataset,
                    pred=pred,
                    fold=fold,
                    score_variant=variant,
                    market_benchmark=market_benchmark,
                    position_multiplier=position_multiplier,
                    log=log,
                )
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    ic = pd.DataFrame(ic_rows).drop_duplicates(["score_variant", "fold", "sample"])
    deciles = pd.DataFrame(decile_rows)
    portfolio = pd.DataFrame(portfolio_rows).drop_duplicates(["score_variant", "fold"])
    comparison = attr.build_comparison(ic, portfolio, deciles)

    predictions.to_parquet(output / "online_group_scores.parquet", index=False)
    weights.to_csv(output / "online_group_weights_by_date.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(output / "online_group_rank_ic.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(output / "online_group_decile_spread.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "online_group_portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "online_group_comparison.csv", index=False, encoding="utf-8-sig")
    write_report(output, comparison, ic, portfolio, weights)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "attribution_run": str(ATTRIBUTION_RUN),
                "data_version": version,
                "lookbacks": LOOKBACKS,
                "min_obs": MIN_OBS,
                "holding_days_label_lag": HOLDING_DAYS + 1,
                "rule": (
                    "At each test date, reweight LightGBM SHAP factor-cluster contributions using only "
                    "train/valid history plus test signal dates whose 10-day labels would already be known."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def prediction_rows_for_fold(frames: list[pd.DataFrame], fold: str) -> list[pd.DataFrame]:
    return [frame for frame in frames if "fold" in frame.columns and str(frame["fold"].iloc[0]) == fold]


def build_online_score(
    fold_frame: pd.DataFrame,
    groups: list[str],
    lookback: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    test = fold_frame.loc[fold_frame["sample"].eq("test")].copy()
    test_dates = list(pd.Index(test["trade_date"].unique()).sort_values())
    daily_ic = build_daily_group_ic(fold_frame, groups)
    last_known_pretest_date = daily_ic.loc[daily_ic["sample"].isin(["train", "valid"]), "trade_date"].max()
    by_test_date = {date: frame.copy() for date, frame in test.groupby("trade_date")}
    rows = []
    weight_rows = []
    for i, date in enumerate(test_dates):
        matured_idx = i - (HOLDING_DAYS + 1)
        if matured_idx >= 0:
            cutoff = test_dates[matured_idx]
        else:
            cutoff = last_known_pretest_date
        recent = daily_ic.loc[daily_ic["trade_date"].le(cutoff)].sort_values("trade_date").tail(lookback)
        weights = payoff_weights_from_daily_ic(recent, groups)
        today = by_test_date[date].copy()
        today["score"] = 0.0
        for group in groups:
            z = attr.cs_zscore_series(today[f"group__{group}"]).fillna(0.0)
            today["score"] += z * weights.get(group, 0.0)
            weight_rows.append(
                {
                    "trade_date": date,
                    "lookback": lookback,
                    "feature_group": group,
                    "weight": weights.get(group, 0.0),
                    "history_start": recent["trade_date"].min() if not recent.empty else pd.NaT,
                    "history_end": recent["trade_date"].max() if not recent.empty else pd.NaT,
                    "history_days": int(recent["trade_date"].nunique()),
                }
            )
        rows.append(today[["trade_date", "ts_code", "label", "score"]])
    return pd.concat(rows, ignore_index=True), weight_rows


def build_daily_group_ic(fold_frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for date, day in fold_frame.groupby("trade_date", sort=True):
        row = {"trade_date": date, "sample": str(day["sample"].iloc[0])}
        label = pd.to_numeric(day["label"], errors="coerce")
        for group in groups:
            score = pd.to_numeric(day[f"group__{group}"], errors="coerce")
            if len(day) < 30 or score.nunique(dropna=True) < 2 or label.nunique(dropna=True) < 2:
                value = np.nan
            else:
                value = score.corr(label, method="spearman")
            row[group] = float(value) if pd.notna(value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def payoff_weights_from_daily_ic(daily_ic: pd.DataFrame, groups: list[str]) -> dict[str, float]:
    if daily_ic["trade_date"].nunique() < MIN_OBS:
        return {group: 1.0 / len(groups) for group in groups}
    scores = {}
    for group in groups:
        values = pd.to_numeric(daily_ic[group], errors="coerce").dropna()
        mean_ic = float(values.mean()) if len(values) else 0.0
        scores[group] = max(mean_ic, 0.0)
    total = sum(scores.values())
    if total <= 0:
        return {group: 1.0 / len(groups) for group in groups}
    return {group: scores[group] / total for group in groups}


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(
    output: Path,
    comparison: pd.DataFrame,
    ic: pd.DataFrame,
    portfolio: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    last_weights = (
        weights.sort_values("trade_date")
        .groupby(["score_variant", "feature_group"], as_index=False)
        .tail(1)
        .sort_values(["score_variant", "weight"], ascending=[True, False])
    )
    lines = [
        "# Online Factor-Cluster Reweighting",
        "",
        "## Method",
        "- Keep the trained LightGBM model and its SHAP factor-cluster contributions.",
        "- Reweight clusters from recently realized RankIC only.",
        "- Test labels are lagged by 11 trading days, matching the 10-day holding label.",
        "",
        "## Comparison",
        md_table(comparison, 20),
        "",
        "## RankIC",
        md_table(ic.sort_values(["score_variant", "fold"]), 80),
        "",
        "## Portfolio",
        md_table(
            portfolio[
                [
                    "score_variant",
                    "fold",
                    "annualized_return",
                    "annualized_excess_return_vs_csi1000",
                    "sharpe",
                    "max_drawdown",
                    "annualized_turnover",
                    "execution_rate",
                ]
            ].sort_values(["score_variant", "fold"]),
            80,
        ),
        "",
        "## Last Available Weights",
        md_table(last_weights, 80),
    ]
    (output / "online_group_reweight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
