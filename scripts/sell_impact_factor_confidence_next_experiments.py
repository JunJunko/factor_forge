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

import sell_impact_factor_confidence_v0 as v0


OUTPUT_ROOT = Path("artifacts/strategy_reviews")
SOURCE_RUN = v0.SOURCE_RUN
MODEL = v0.MODEL
COST_BUFFER = 0.002
LABEL_DELAY_TRADING_DAYS = v0.LABEL_DELAY_TRADING_DAYS
RECENT_WINDOW = v0.RECENT_WINDOW
RECENT_STABILITY_WINDOW = v0.RECENT_STABILITY_WINDOW
PRIOR_STRENGTH_K = v0.PRIOR_STRENGTH_K
TEST_START = pd.Timestamp(v0.TEST_START)
TEST_END = pd.Timestamp("2026-06-12")


def main() -> None:
    output = OUTPUT_ROOT / f"sell_impact_factor_confidence_next_experiments_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"source_run={SOURCE_RUN}")
    dataset = pd.read_parquet(SOURCE_RUN / "recent_halfyear_dataset.parquet")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    selected = pd.read_csv(SOURCE_RUN / "selected_condition_interactions.csv")
    log(f"dataset rows={len(dataset):,} dates={dataset['trade_date'].nunique():,}")

    scored = v0.build_band_scores(dataset, selected, log)
    daily_metrics = v0.daily_factor_metrics(scored)
    ic_health = v0.build_factor_health_daily(daily_metrics).rename(columns={"confidence": "confidence_ic"})
    spread_health = build_spread_confidence_daily(daily_metrics)
    health = ic_health.merge(
        spread_health[
            [
                "trade_date",
                "confidence_spread",
                "posterior_spread",
                "posterior_spread_std",
                "posterior_spread_z",
                "prior_spread_scope",
                "prior_spread_obs",
                "recent_spread_20_spread_model",
                "future_spread_binary",
            ]
        ],
        on="trade_date",
        how="left",
    )
    health.to_csv(output / "factor_confidence_daily_next.csv", index=False, encoding="utf-8-sig")

    score_spread = experiment_score_scaling(scored, health, "confidence_ic")
    score_spread.to_csv(output / "experiment1_score_scaling_decile_spread.csv", index=False, encoding="utf-8-sig")

    target_compare = confidence_target_comparison(health)
    target_compare.to_csv(output / "experiment2_confidence_target_comparison.csv", index=False, encoding="utf-8-sig")

    expanded = expanded_history_validation(health)
    expanded.to_csv(output / "experiment3_expanded_history_bucket_validation.csv", index=False, encoding="utf-8-sig")

    write_report(output, score_spread, target_compare, expanded)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "source_run": str(SOURCE_RUN),
                "model": MODEL,
                "experiments": [
                    "final_score = band_score vs band_score * daily confidence",
                    "IC confidence target vs future top-bottom spread > cost buffer",
                    "expanded history validation with explicit clean/test split labels",
                ],
                "cost_buffer": COST_BUFFER,
                "label_delay_trading_days": LABEL_DELAY_TRADING_DAYS,
                "note": "The expanded-history scorer is a fixed shadow scorer trained with the tactical model specification; pre-2026 rows are diagnostic and not clean OOS.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def build_spread_confidence_daily(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    m = metrics.sort_values("trade_date").reset_index(drop=True)
    for i, row in m.iterrows():
        available = m.iloc[: max(0, i - LABEL_DELAY_TRADING_DAYS)].copy()
        recent20 = available.tail(RECENT_WINDOW)
        recent60 = available.tail(RECENT_STABILITY_WINDOW)
        prior = select_spread_prior(available, str(row.get("regime")), str(row.get("coarse_regime")))
        recent_spread = float(recent20["top_bottom_spread"].mean()) if len(recent20) else np.nan
        recent_std = float(recent60["top_bottom_spread"].std(ddof=1)) if len(recent60) > 1 else np.nan
        prior_mu = prior["prior_spread_mean"]
        prior_std = prior["prior_spread_std"]
        if not np.isfinite(prior_std) or prior_std <= 1e-6:
            prior_std = float(available["top_bottom_spread"].std(ddof=1)) if len(available) > 1 else 0.03
        if not np.isfinite(recent_std) or recent_std <= 1e-6:
            recent_std = prior_std
        n = int(recent20["top_bottom_spread"].notna().sum())
        w = n / (n + PRIOR_STRENGTH_K) if n > 0 else 0.0
        posterior_mu = w * nz(recent_spread, prior_mu) + (1.0 - w) * nz(prior_mu, 0.0)
        posterior_std = np.sqrt(max(1e-8, w * nz(recent_std, prior_std) ** 2 + (1.0 - w) * nz(prior_std, 0.03) ** 2))
        posterior_z = (posterior_mu - COST_BUFFER) / posterior_std if posterior_std > 0 else 0.0
        confidence_raw = float(np.clip(v0.sigmoid(1.5 * posterior_z), 0.20, 1.0))
        rows.append(
            {
                "trade_date": row["trade_date"],
                "factor": "band_score",
                "regime": row.get("regime"),
                "coarse_regime": row.get("coarse_regime"),
                "prior_spread_scope": prior["prior_scope"],
                "prior_spread_obs": prior["prior_obs"],
                "prior_spread_mean": prior_mu,
                "prior_spread_std": prior_std,
                "recent_spread_20_spread_model": recent_spread,
                "recent_spread_std_60": recent_std,
                "posterior_spread": posterior_mu,
                "posterior_spread_std": posterior_std,
                "posterior_spread_z": posterior_z,
                "confidence_spread_raw": confidence_raw,
                "future_spread": row.get("top_bottom_spread"),
                "future_rank_ic": row.get("rank_ic"),
                "future_spread_binary": bool(row.get("top_bottom_spread", np.nan) > COST_BUFFER),
            }
        )
    out = pd.DataFrame(rows)
    out["confidence_spread"] = out["confidence_spread_raw"].ewm(span=5, adjust=False).mean().clip(0.20, 1.0)
    return out


def select_spread_prior(available: pd.DataFrame, regime: str, coarse_regime: str) -> dict[str, Any]:
    for scope, mask in [
        ("regime", available["regime"].eq(regime) if "regime" in available else pd.Series(False, index=available.index)),
        (
            "coarse_regime",
            available["coarse_regime"].eq(coarse_regime)
            if "coarse_regime" in available
            else pd.Series(False, index=available.index),
        ),
        ("global", pd.Series(True, index=available.index)),
    ]:
        group = available.loc[mask].dropna(subset=["top_bottom_spread"])
        if len(group) >= v0.PRIOR_MIN_OBS or scope == "global":
            return {
                "prior_scope": scope,
                "prior_obs": int(len(group)),
                "prior_spread_mean": float(group["top_bottom_spread"].mean()) if len(group) else 0.0,
                "prior_spread_std": float(group["top_bottom_spread"].std(ddof=1)) if len(group) > 1 else 0.03,
            }
    raise RuntimeError("unreachable prior fallback")


def experiment_score_scaling(scored: pd.DataFrame, health: pd.DataFrame, confidence_col: str) -> pd.DataFrame:
    frame = scored.merge(health[["trade_date", confidence_col]], on="trade_date", how="left")
    frame[confidence_col] = frame[confidence_col].fillna(0.50)
    frame["score_band"] = frame["band_score"]
    frame["score_band_times_confidence"] = frame["band_score"] * frame[confidence_col]
    rows = []
    for sample_name, mask in sample_masks(frame):
        sample = frame.loc[mask].copy()
        for score_col in ["score_band", "score_band_times_confidence"]:
            daily = daily_decile_spread(sample, score_col)
            rows.append(
                {
                    "sample": sample_name,
                    "score": score_col,
                    "days": int(len(daily)),
                    "mean_top_bottom_spread": float(daily["top_bottom_spread"].mean()),
                    "positive_spread_ratio": float(daily["top_bottom_spread"].gt(0).mean()),
                    "mean_rank_ic": float(daily["rank_ic"].mean()),
                    "same_top_decile_as_band_ratio": same_top_decile_ratio(sample, score_col),
                }
            )
    return pd.DataFrame(rows)


def confidence_target_comparison(health: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_name, mask in sample_masks(health):
        sample = health.loc[mask].dropna(subset=["future_spread", "future_rank_ic"]).copy()
        for confidence_col, label in [
            ("confidence_ic", "ic_posterior_target"),
            ("confidence_spread", "spread_gt_cost_target"),
        ]:
            rows.extend(bucket_validation(sample, confidence_col, sample_name, label))
    return pd.DataFrame(rows)


def expanded_history_validation(health: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = health.dropna(subset=["confidence_ic", "confidence_spread", "future_spread", "future_rank_ic"]).copy()
    samples = {
        "expanded_after_warmup": frame["prior_obs"].fillna(0).ge(60),
        "history_2024_2025q3_shadow_in_sample": frame["trade_date"].between(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-09-30")),
        "valid_2025q4_shadow": frame["trade_date"].between(pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")),
        "clean_2026h1": frame["trade_date"].between(TEST_START, TEST_END),
    }
    for sample_name, mask in samples.items():
        sample = frame.loc[mask].copy()
        for confidence_col in ["confidence_ic", "confidence_spread"]:
            rows.extend(bucket_validation(sample, confidence_col, sample_name, confidence_col))
    return pd.DataFrame(rows)


def bucket_validation(sample: pd.DataFrame, confidence_col: str, sample_name: str, target_name: str) -> list[dict[str, Any]]:
    if len(sample) < 20 or sample[confidence_col].nunique(dropna=True) < 5:
        return []
    sample = sample.copy()
    sample["bucket"] = pd.qcut(
        sample[confidence_col].rank(method="first"),
        5,
        labels=["q1_low", "q2", "q3", "q4", "q5_high"],
    )
    rows = []
    for bucket, group in sample.groupby("bucket", observed=True):
        rows.append(
            {
                "sample": sample_name,
                "target": target_name,
                "confidence": confidence_col,
                "bucket": str(bucket),
                "days": int(len(group)),
                "confidence_mean": float(group[confidence_col].mean()),
                "future_spread_mean": float(group["future_spread"].mean()),
                "future_spread_gt_cost_ratio": float(group["future_spread"].gt(COST_BUFFER).mean()),
                "future_rank_ic_mean": float(group["future_rank_ic"].mean()),
                "future_rank_ic_positive_ratio": float(group["future_rank_ic"].gt(0).mean()),
            }
        )
    return rows


def daily_decile_spread(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("trade_date", sort=True):
        data = group[[score_col, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 50:
            continue
        data = data.copy()
        data["decile"] = pd.qcut(data[score_col].rank(method="first"), 10, labels=False) + 1
        avg = data.groupby("decile")["label"].mean()
        rows.append(
            {
                "trade_date": date,
                "top_bottom_spread": float(avg.get(10, np.nan) - avg.get(1, np.nan)),
                "rank_ic": float(data[score_col].corr(data["label"], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def same_top_decile_ratio(frame: pd.DataFrame, score_col: str) -> float:
    if score_col == "score_band":
        return 1.0
    ratios = []
    for _date, group in frame.groupby("trade_date", sort=True):
        data = group[["ts_code", "score_band", score_col]].dropna()
        if len(data) < 50:
            continue
        n = max(1, int(np.ceil(len(data) * 0.10)))
        base_top = set(data.nlargest(n, "score_band")["ts_code"])
        test_top = set(data.nlargest(n, score_col)["ts_code"])
        ratios.append(len(base_top & test_top) / n)
    return float(np.mean(ratios)) if ratios else np.nan


def sample_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    dates = pd.to_datetime(frame["trade_date"])
    return [
        ("clean_2026h1", dates.between(TEST_START, TEST_END)),
        ("expanded_after_warmup", pd.Series(True, index=frame.index)),
    ]


def nz(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


def write_report(output: Path, score_spread: pd.DataFrame, target_compare: pd.DataFrame, expanded: pd.DataFrame) -> None:
    lines = [
        "# Factor Confidence Next Experiments",
        "",
        "## Experiment 1: Score Scaling",
        "Daily scalar confidence is multiplied by `band_score`. Because the multiplier is positive and identical for all stocks on a date, same-day cross-sectional ranks should not change.",
        md_table(score_spread, 20),
        "",
        "## Experiment 2: Confidence Target",
        f"Spread-target confidence uses future top-bottom spread above `{COST_BUFFER:.4f}` as the validation target and builds posterior evidence from spread, not IC.",
        md_table(target_compare, 60),
        "",
        "## Experiment 3: Expanded History",
        "Expanded rows increase statistical power, but pre-2026 rows are diagnostic because the fixed shadow scorer is trained on part of that history.",
        md_table(expanded, 80),
        "",
        "## Files",
        "- `factor_confidence_daily_next.csv`",
        "- `experiment1_score_scaling_decile_spread.csv`",
        "- `experiment2_confidence_target_comparison.csv`",
        "- `experiment3_expanded_history_bucket_validation.csv`",
    ]
    (output / "factor_confidence_next_experiments_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
