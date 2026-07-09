from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factor_forge.factor_state import (  # noqa: E402
    FactorHealthConfig,
    FactorStateModelConfig,
    build_factor_health_daily,
    build_factor_state_labels,
    build_factor_state_output,
    run_factor_state_model,
)


DEFAULT_SOURCE_RUN = Path("artifacts/strategy_reviews/sell_impact_recent_halfyear_tactical_20260709T100107Z")
OUTPUT_ROOT = Path("artifacts/factor_state")
MARKET_COLUMNS = [
    "market_ret_20",
    "market_ret_60",
    "market_vol_20",
    "market_breadth_20",
    "market_xsec_vol_20",
    "market_turnover_chg_5_20",
]


def main() -> None:
    args = parse_args()
    output = args.output or OUTPUT_ROOT / f"factor_state_transition_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("loading factor observations")
    observations, input_meta = load_factor_observations(args)
    observations.to_parquet(output / "factor_observation_input.parquet", index=False)
    observations.to_csv(output / "factor_observation_input_sample.csv", index=False, encoding="utf-8-sig")
    log(f"observations rows={len(observations):,} dates={observations['trade_date'].nunique():,}")

    health_cfg = FactorHealthConfig(
        factor_col=args.score_column,
        return_col=args.return_column,
        factor_name=args.factor_name,
        min_obs_per_day=args.min_obs_per_day,
    )
    health = build_factor_health_daily(observations, health_cfg)
    health.to_parquet(output / "factor_health_daily.parquet", index=False)
    health.to_csv(output / "factor_health_daily.csv", index=False, encoding="utf-8-sig")
    log(f"health rows={len(health):,}")

    labels = build_factor_state_labels(
        health,
        forward_window=args.forward_window,
        spread_threshold=args.spread_threshold,
        monotonicity_threshold=args.monotonicity_threshold,
    )
    labels.to_csv(output / "factor_state_label.csv", index=False, encoding="utf-8-sig")
    log(f"labels rows={labels['state'].notna().sum():,}")

    model_results = {}
    model_error = ""
    try:
        model_results = run_factor_state_model(
            labels,
            output_dir=output / "model",
            config=FactorStateModelConfig(
                train_start=args.train_start,
                train_end=args.train_end,
                valid_start=args.valid_start,
                valid_end=args.valid_end,
                test_start=args.test_start,
                test_end=args.test_end,
            ),
        )
        interface = build_factor_state_output(
            model_results["predictions"],
            model_name=args.output_model,
            sample=args.output_sample,
            output_path=output / "factor_state_daily.csv",
        )
        log(f"state output rows={len(interface):,}")
    except Exception as exc:  # noqa: BLE001 - report diagnostics instead of hiding short-sample failure.
        model_error = str(exc)
        log(f"model skipped: {model_error}")

    write_report(output, input_meta, health, labels, model_results, model_error, args)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(output),
                "input": input_meta,
                "factor_name": args.factor_name,
                "score_column": args.score_column,
                "return_column": args.return_column,
                "model_error": model_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Factor State Transition Model artifacts.")
    parser.add_argument("--input", type=Path, default=None, help="CSV/parquet with trade_date, ts_code, score and label columns.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--walkforward-run", type=Path, default=None, help="sell_impact_score_band_walkforward run directory.")
    parser.add_argument("--walkforward-variant", default="regime_aware_cluster_ranker")
    parser.add_argument("--walkforward-sample", default="test", choices=["test", "valid", "all"])
    parser.add_argument("--use-selected-band", action="store_true", help="Transform score to score-band factor using band_selection.csv.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--factor-name", default="band_score")
    parser.add_argument("--score-column", default="band_score")
    parser.add_argument("--return-column", default="label")
    parser.add_argument("--min-obs-per-day", type=int, default=30)
    parser.add_argument("--forward-window", type=int, default=60)
    parser.add_argument("--spread-threshold", type=float, default=0.0)
    parser.add_argument("--monotonicity-threshold", type=float, default=0.0)
    parser.add_argument("--train-start", default="20180101")
    parser.add_argument("--train-end", default="20221231")
    parser.add_argument("--valid-start", default="20230101")
    parser.add_argument("--valid-end", default="20231231")
    parser.add_argument("--test-start", default="20240101")
    parser.add_argument("--test-end", default="20261231")
    parser.add_argument("--output-model", default="logistic_regression")
    parser.add_argument("--output-sample", default=None)
    return parser.parse_args()


def load_factor_observations(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    if args.walkforward_run is not None:
        return load_walkforward_observations(args)
    source = args.input or latest_factor_exposure_file()
    frame = read_frame(source)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if args.score_column not in frame.columns:
        raise ValueError(f"score column {args.score_column!r} not found in {source}")
    if args.return_column not in frame.columns:
        raise ValueError(f"return column {args.return_column!r} not found in {source}")
    frame = merge_market_context(frame, args.source_run)
    keep = ["trade_date", "ts_code", args.score_column, args.return_column, *[col for col in MARKET_COLUMNS if col in frame.columns]]
    if "market_turnover_change" in frame.columns:
        keep.append("market_turnover_change")
    out = frame[[col for col in dict.fromkeys(keep) if col in frame.columns]].copy()
    return out, {
        "source_file": str(source),
        "source_run": str(args.source_run),
        "rows": int(len(out)),
        "start_date": str(out["trade_date"].min().date()),
        "end_date": str(out["trade_date"].max().date()),
        "dates": int(out["trade_date"].nunique()),
    }


def load_walkforward_observations(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    run = args.walkforward_run
    if not run.exists():
        raise FileNotFoundError(f"walkforward run not found: {run}")
    files = sorted(run.glob(f"predictions_test_*_{args.walkforward_variant}.parquet"))
    if not files:
        raise FileNotFoundError(f"no predictions_test_*_{args.walkforward_variant}.parquet under {run}")
    selected_bands = load_selected_bands(run) if args.use_selected_band else {}
    frames = []
    for path in files:
        fold = "_".join(path.stem.split("_")[1:3])
        frame = pd.read_parquet(path)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["fold"] = fold
        if args.walkforward_sample != "all":
            frame = frame.loc[frame["sample"].eq(args.walkforward_sample)].copy()
        if args.use_selected_band:
            band = selected_bands.get((fold, args.walkforward_variant))
            if band is None:
                raise ValueError(f"selected band missing for fold={fold} variant={args.walkforward_variant}")
            frame["score_pct"] = frame.groupby("trade_date")["score"].rank(pct=True, method="first")
            frame[args.score_column] = -(frame["score_pct"] - float(band)).abs()
            frame["selected_band"] = float(band)
            frame = frame.drop(columns=["score_pct"])
        elif args.score_column != "score":
            frame[args.score_column] = frame["score"]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["trade_date", "ts_code", "fold"]).drop_duplicates(["trade_date", "ts_code"], keep="last")
    out = merge_market_context(out, run)
    keep = [
        "trade_date",
        "ts_code",
        args.score_column,
        args.return_column,
        "fold",
        "sample",
        "selected_band",
        *[col for col in MARKET_COLUMNS if col in out.columns],
        "market_turnover_change",
    ]
    out = out[[col for col in dict.fromkeys(keep) if col in out.columns]].copy()
    return out, {
        "source_file": str(run),
        "source_run": str(run),
        "walkforward_variant": args.walkforward_variant,
        "walkforward_sample": args.walkforward_sample,
        "use_selected_band": bool(args.use_selected_band),
        "rows": int(len(out)),
        "start_date": str(out["trade_date"].min().date()),
        "end_date": str(out["trade_date"].max().date()),
        "dates": int(out["trade_date"].nunique()),
    }


def load_selected_bands(run: Path) -> dict[tuple[str, str], float]:
    path = run / "band_selection.csv"
    if not path.exists():
        raise FileNotFoundError(f"band_selection.csv not found: {path}")
    frame = pd.read_csv(path)
    selected = frame.loc[frame["selected"].astype(str).str.lower().eq("true")].copy()
    return {
        (str(row.fold), str(row.variant)): float(row.band)
        for row in selected.itertuples(index=False)
    }


def latest_factor_exposure_file() -> Path:
    candidates = sorted(Path("artifacts/strategy_reviews").glob("sell_impact_full_model_evaluation_param068_*/factor_exposures_2026h1.parquet"))
    if not candidates:
        raise FileNotFoundError("no factor_exposures_2026h1.parquet found under artifacts/strategy_reviews")
    return candidates[-1]


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def merge_market_context(frame: pd.DataFrame, source_run: Path) -> pd.DataFrame:
    dataset_path = source_run / "recent_halfyear_dataset.parquet"
    if not dataset_path.exists():
        dataset_path = source_run / "walkforward_dataset.parquet"
    if not dataset_path.exists():
        return frame
    missing = [col for col in MARKET_COLUMNS if col not in frame.columns]
    if not missing:
        return frame
    dataset = pd.read_parquet(dataset_path, columns=["trade_date", "ts_code", *missing])
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    return frame.merge(dataset, on=["trade_date", "ts_code"], how="left")


def write_report(
    output: Path,
    input_meta: dict,
    health: pd.DataFrame,
    labels: pd.DataFrame,
    model_results: dict,
    model_error: str,
    args: argparse.Namespace,
) -> None:
    label_counts = labels["state_name"].value_counts(dropna=False).rename_axis("state").reset_index(name="rows")
    lines = [
        "# Factor State Transition Model Report",
        "",
        "## Scope",
        f"- Factor: `{args.factor_name}`",
        f"- Score column: `{args.score_column}`",
        f"- Return column for factor payoff measurement: `{args.return_column}`",
        f"- Input: `{input_meta['source_file']}`",
        f"- Date range: `{input_meta['start_date']}` to `{input_meta['end_date']}`, {input_meta['dates']} trading days.",
        "- Objective: factor health and lifecycle state probability, not stock return prediction or portfolio optimization.",
        "",
        "## Output Files",
        "- `factor_observation_input.parquet`",
        "- `factor_health_daily.parquet` / `factor_health_daily.csv`",
        "- `factor_state_label.csv`",
        "- `model/*.csv`",
        "- `factor_state_daily.csv` when a model can be trained",
        "",
        "## Feature Layer",
        md_table(health.tail(10), 10),
        "",
        "## State Label Distribution",
        md_table(label_counts, 20),
        "",
    ]
    if model_error:
        lines += [
            "## Model Status",
            f"Model training skipped or incomplete: `{model_error}`",
            "",
        ]
    else:
        lines += [
            "## State Prediction Accuracy",
            md_table(model_results["metrics"], 30),
            "",
            "## Confusion Matrix",
            md_table(model_results["confusion_matrix"], 80),
            "",
            "## Probability Calibration",
            md_table(model_results["calibration"], 80),
            "",
            "## State Transition Matrix",
            md_table(model_results["transition_matrix"], 40),
            "",
            "## Early Warning",
            md_table(model_results["early_warning"], 40),
            "",
            "## Feature Importance",
            "### Logistic Regression",
            md_table(model_results.get("logistic_regression_feature_importance", pd.DataFrame()), 30),
            "",
            "### LightGBM Shallow",
            md_table(model_results.get("lightgbm_shallow_feature_importance", pd.DataFrame()), 30),
            "",
        ]
    lines += [
        "## Caveats",
        "- 当前默认输入使用现有 artifacts 中的 band_score 暴露；如果历史 band_score 只有 2026H1，模型会自动 fallback 到短样本时间切分，结论只能作为流程验证。",
        "- 正式版应接入 2018 起逐日历史 factor_score / future_return / market regime 数据，再使用 2018-2022 / 2023 / 2024-2026 固定切分。",
        "- future spread / future IC 只用于标签，不在输入特征中使用。",
    ]
    (output / "factor_state_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    return frame.head(max_rows).round(6).to_markdown(index=False)


if __name__ == "__main__":
    main()
