"""Generate live-style sell-impact ranker candidates.

Signal convention:
- Features use data available at signal-date close.
- Entries are intended for the next trading-day open.
- The selector is the validated cluster_stock_state + low-vol LightGBM ranker, direct top.
- Timing overlay is applied as the new-entry cash multiplier.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sell_impact_sorting_repair as base
from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository


OUTPUT_ROOT = Path("artifacts/sell_impact_ranker_live_signals")
PROJECT_CONFIG = "configs/project.yaml"
DEFAULT_TIMING_DAILY = Path(
    "artifacts/timing_position_models/"
    "timing_position_model_v1_20260708T025521Z_181c72c6/"
    "timing_position_daily.csv"
)
MODEL_VARIANT = "cluster_stock_state_plus_low_vol_ranker"
MODEL_VERSION = "sell_impact_cluster_low_vol_ranker_direct_top_v2"
TOP_N = 5
TOP_CANDIDATES = 100
TRAIN_START = "20220101"


def main(signal_date: str | None = None) -> None:
    version, panel = load_latest_panel()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    signal_ts = (
        pd.Timestamp(signal_date).normalize()
        if signal_date
        else pd.Timestamp(panel["trade_date"].max()).normalize()
    )
    output = OUTPUT_ROOT / f"ranker_direct_top_signal_{signal_ts:%Y%m%d}_{datetime.now():%Y%m%dT%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    log = log_factory(output)
    log(f"data_version={version} signal_date={signal_ts.date()}")

    if signal_ts not in set(panel["trade_date"]):
        raise ValueError(f"signal_date {signal_ts.date()} not found in latest panel")

    work = panel.loc[panel["trade_date"].ge(pd.Timestamp("2022-01-01"))].copy()
    old_train_start, old_test_end = base.TRAIN_START, base.TEST_END
    base.TRAIN_START = TRAIN_START
    base.TEST_END = signal_ts.strftime("%Y%m%d")
    try:
        dataset = base.build_dataset(work, log, include_unlabeled_date=signal_ts)
    finally:
        base.TRAIN_START, base.TEST_END = old_train_start, old_test_end
    dataset = dataset.loc[dataset["ts_code"].map(permission_eligible)].copy()
    dataset = add_live_model_features(dataset)
    dataset.to_parquet(output / "live_dataset.parquet", index=False)
    log(f"dataset rows={len(dataset):,} signal_candidates={dataset['trade_date'].eq(signal_ts).sum():,}")

    model, train, valid, features = fit_ranker(dataset, signal_ts, log)
    signal_rows = dataset.loc[dataset["trade_date"].eq(signal_ts)].dropna(subset=features).copy()
    if signal_rows.empty:
        raise ValueError(f"no predictable candidates for {signal_ts.date()}")
    signal_rows["score"] = model.predict(signal_rows[features])

    enriched = enrich_candidates(signal_rows, panel, signal_ts)
    entry_date = next_trade_date(panel, signal_ts)
    timing_info = timing_for_entry(entry_date, signal_ts)
    final_exposure = float(np.clip(timing_info["target_position"], 0.0, 1.0))

    top100 = enriched.sort_values(["score", "ts_code"], ascending=[False, True]).head(TOP_CANDIDATES).copy()
    top100["rank"] = np.arange(1, len(top100) + 1)
    top100["factor_value"] = top100["score"]
    top100["raw_factor_value"] = top100["score"]
    top100["score_direction"] = 1.0
    top100["target_weight"] = 0.0
    if final_exposure > 0 and len(top100):
        top100.loc[top100["rank"].le(TOP_N), "target_weight"] = final_exposure / TOP_N

    ordered_cols = [
        "rank",
        "trade_date",
        "ts_code",
        "name",
        "industry_l1_name",
        "factor_value",
        "raw_factor_value",
        "score_direction",
        "target_weight",
        "raw_close",
        "amount_cny",
        "listing_trade_days",
        "is_tradeable",
        "is_st",
        "is_delisting_period",
        "is_suspended",
        "is_limit_up_open",
        "is_limit_down_open",
        *base.CLUSTER_COLS,
        "stock_state_low_vol",
        *base.REGIME_COLS,
    ]
    existing_cols = [c for c in ordered_cols if c in top100.columns]
    top100[existing_cols].to_csv(output / "top100_candidates.csv", index=False, encoding="utf-8-sig")
    top100.head(TOP_N)[existing_cols].to_csv(output / "top_recommendations.csv", index=False, encoding="utf-8-sig")

    gain = pd.DataFrame(
        {
            "feature": features,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain_importance", ascending=False)
    gain.to_csv(output / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")

    summary = {
        "signal_date": signal_ts,
        "intended_execution": "next_trade_day_open",
        "entry_date_for_timing": entry_date,
        "data_version": version,
        "model": "LightGBM LambdaRank cluster_stock_state + stock_state_low_vol",
        "signal_algorithm": MODEL_VERSION,
        "frozen_model_version": MODEL_VERSION,
        "selector": "ranker_direct_top",
        "top_n": TOP_N,
        "timing_model": {
            "enabled": True,
            "position_column": "target_position",
            **timing_info,
        },
        "final_exposure": final_exposure,
        "target_position": final_exposure,
        "permission_filter": {
            "enabled": True,
            "excluded_boards": ["STAR 688/689.SH", "ChiNext 300/301/302.SZ", "Beijing *.BJ"],
        },
        "train_start_requested": TRAIN_START,
        "train_start_actual": train["trade_date"].min(),
        "train_end_actual": train["trade_date"].max(),
        "valid_start_actual": valid["trade_date"].min(),
        "valid_end_actual": valid["trade_date"].max(),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "features": features,
        "factor_clusters": base.CLUSTER_COLS,
        "stock_state_augmented_features": ["stock_state_low_vol"],
        "regime_features": base.REGIME_COLS,
        "predictable_candidates": int(len(signal_rows)),
        "shown_candidates": int(len(top100)),
        "pit_universe": "latest panel is_liquid top1000 + tradeable + condition Q5 + permission filter",
        "next_day_fillability_note": "Next-day suspension/limit-up/open fill must be rechecked when entry-date data is available.",
    }
    (output / "signal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_markdown_report(output, summary, top100, gain)
    log(f"done output={output}")


def load_latest_panel() -> tuple[str, pd.DataFrame]:
    project = load_project(PROJECT_CONFIG)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest("latest")
    _, panel = repo.load_panel(version)
    return version, panel


def fit_ranker(dataset: pd.DataFrame, signal_ts: pd.Timestamp, log):
    import lightgbm as lgb

    features = features_for_live_model(dataset)
    year = signal_ts.year
    train_start = pd.Timestamp(f"{max(2022, year - 4)}-01-01")
    train_end = pd.Timestamp(f"{year - 2}-12-31")
    valid_start = pd.Timestamp(f"{year - 1}-01-01")
    valid_end = pd.Timestamp(f"{year - 1}-12-31")

    realized = dataset.loc[
        dataset["label"].notna()
        & dataset["exit_date"].notna()
        & pd.to_datetime(dataset["exit_date"]).le(signal_ts)
    ].copy()
    train = realized.loc[realized["trade_date"].between(train_start, train_end)].dropna(subset=features + ["label"])
    valid = realized.loc[realized["trade_date"].between(valid_start, valid_end)].dropna(subset=features + ["label"])
    if len(train) < 20_000 or len(valid) < 5_000:
        fallback_start = pd.Timestamp("2022-01-01")
        fallback_end = signal_ts - pd.Timedelta(days=120)
        train = realized.loc[realized["trade_date"].between(fallback_start, fallback_end)].dropna(subset=features + ["label"])
        valid = realized.loc[realized["trade_date"].gt(fallback_end)].dropna(subset=features + ["label"])
    if len(train) < 20_000 or len(valid) < 1_000:
        raise ValueError(f"not enough realized samples train={len(train):,} valid={len(valid):,}")

    train = train.sort_values(["trade_date", "ts_code"])
    valid = valid.sort_values(["trade_date", "ts_code"])
    log(
        f"training {MODEL_VARIANT} features={len(features)} "
        f"train={len(train):,} {train['trade_date'].min().date()}..{train['trade_date'].max().date()} "
        f"valid={len(valid):,} {valid['trade_date'].min().date()}..{valid['trade_date'].max().date()}"
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=250,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(
        train[features],
        base.relevance_labels(train),
        group=train.groupby("trade_date").size().to_list(),
        eval_set=[(valid[features], base.relevance_labels(valid))],
        eval_group=[valid.groupby("trade_date").size().to_list()],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model, train, valid, features


def add_live_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    out["stock_state_low_vol"] = -pd.to_numeric(out["volatility_20_z"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def features_for_live_model(dataset: pd.DataFrame) -> list[str]:
    original_interactions = [
        c
        for c in dataset.columns
        if "__x__regime_" in c and any(c.startswith(f"{cluster}__x__") for cluster in base.CLUSTER_COLS)
    ]
    features = [*base.CLUSTER_COLS, *base.REGIME_COLS, *original_interactions, "stock_state_low_vol"]
    missing = [feature for feature in features if feature not in dataset.columns]
    if missing:
        raise ValueError(f"live model missing features: {missing}")
    return features


def enrich_candidates(candidates: pd.DataFrame, panel: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.DataFrame:
    cols = [
        "trade_date",
        "ts_code",
        "raw_close",
        "amount_cny",
        "listing_trade_days",
        "industry_l1_name",
        "is_tradeable",
        "is_st",
        "is_delisting_period",
        "is_suspended",
        "is_limit_up_open",
        "is_limit_down_open",
    ]
    day = panel.loc[panel["trade_date"].eq(signal_ts), [c for c in cols if c in panel.columns]].copy()
    out = candidates.merge(day, on=["trade_date", "ts_code"], how="left", suffixes=("", "_panel"))
    names = stock_name_map()
    out["name"] = out["ts_code"].map(names).fillna("")
    return out


def stock_name_map() -> dict[str, str]:
    for path in sorted(Path("data/versions").glob("*/raw/tushare/stock_basic.parquet"), reverse=True):
        try:
            data = pd.read_parquet(path, columns=["ts_code", "name"])
        except Exception:
            continue
        if not data.empty:
            return dict(zip(data["ts_code"].astype(str), data["name"].astype(str)))
    return {}


def next_trade_date(panel: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.Timestamp | None:
    dates = pd.Index(pd.to_datetime(panel["trade_date"].drop_duplicates()).sort_values())
    future = dates[dates > signal_ts]
    if len(future):
        return pd.Timestamp(future[0]).normalize()
    timing = load_timing()
    future_timing = timing.index[timing.index > signal_ts]
    return pd.Timestamp(future_timing[0]).normalize() if len(future_timing) else None


def load_timing() -> pd.Series:
    path = latest_timing_daily_path()
    if path is None:
        return pd.Series(dtype=float, name="target_position")
    frame = pd.read_csv(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    values = pd.to_numeric(frame["target_position"], errors="coerce").clip(0.0, 1.0)
    out = pd.Series(values.to_numpy(), index=frame["trade_date"], name="target_position").sort_index()
    out.attrs["source"] = str(path)
    return out


def latest_timing_daily_path() -> Path | None:
    root = Path("artifacts/timing_position_models")
    candidates = [p for p in root.glob("timing_position_model_v1_*/timing_position_daily.csv") if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return DEFAULT_TIMING_DAILY if DEFAULT_TIMING_DAILY.exists() else None


def timing_for_entry(entry_date: pd.Timestamp | None, signal_ts: pd.Timestamp) -> dict[str, object]:
    timing = load_timing()
    if timing.empty:
        return {
            "target_position": 1.0,
            "timing_date": None,
            "source": None,
            "fallback_reason": "missing_timing_file",
        }
    lookup_date = entry_date if entry_date is not None else signal_ts
    if lookup_date in timing.index:
        return {
            "target_position": float(timing.loc[lookup_date]),
            "timing_date": lookup_date,
            "source": timing.attrs.get("source"),
        }
    prior = timing.loc[timing.index <= lookup_date]
    if not prior.empty:
        return {
            "target_position": float(prior.iloc[-1]),
            "timing_date": pd.Timestamp(prior.index[-1]),
            "source": timing.attrs.get("source"),
            "fallback_reason": "used_latest_prior_timing_date",
        }
    return {
        "target_position": 1.0,
        "timing_date": None,
        "source": timing.attrs.get("source"),
        "fallback_reason": "no_prior_timing_date",
    }


def permission_eligible(ts_code: str) -> bool:
    code = str(ts_code)
    if code.endswith(".BJ"):
        return False
    if code.endswith(".SH") and code[:3] in {"688", "689"}:
        return False
    if code.endswith(".SZ") and code[:3] in {"300", "301", "302"}:
        return False
    return True


def write_markdown_report(output: Path, summary: dict, top100: pd.DataFrame, gain: pd.DataFrame) -> None:
    top = top100.head(TOP_N)[
        ["rank", "ts_code", "name", "industry_l1_name", "factor_value", "target_weight", "raw_close", "amount_cny"]
    ].copy()
    lines = [
        "# Sell Impact Ranker Live Signal",
        "",
        f"- Signal date: `{pd.Timestamp(summary['signal_date']).date()}`",
        f"- Intended execution: `{summary['intended_execution']}`",
        f"- Final exposure from timing model: `{summary['final_exposure']:.2%}`",
        f"- Predictable candidates: `{summary['predictable_candidates']}`",
        "",
        "## Top Recommendations",
        top.round(6).to_markdown(index=False),
        "",
        "## Top Feature Gain",
        gain.head(20).round(6).to_markdown(index=False),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_factory(output: Path):
    log_path = output / "run.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
