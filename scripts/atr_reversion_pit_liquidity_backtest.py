"""PIT rolling-liquidity Top1000 ATR backtest.

Strictness goals:
- Daily universe is selected from past rolling amount, not full-sample liquidity.
- Model training/prediction rows are restricted to each day's PIT universe.
- Benchmark uses the same PIT universe.
- Signals are T close; trades execute at T+1 open.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_forge.config import load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.atr_reversion_config import load_atr_reversion_config
from factor_forge.ml.atr_reversion_dataset import (
    FEATURE_GROUPS,
    NO_CROSS_SECTION_ZSCORE,
    build_atr_reversion_dataset,
)

from atr_reversion_benchmark import csi1000_open_to_open_returns
from atr_reversion_small_portfolio_backtest import (
    LOT_SIZE,
    INITIAL_CASH,
    _can_sell_at_open,
    _json_default,
    _metrics,
    _position_value,
)


TOP_NS = [5, 10]
REBALANCE_DAYS = [10]
COST_BPS = [10, 20]
LIQUIDITY_WINDOW = 20
LIQUIDITY_MIN_PERIODS = 18
PIT_TOP_N = 1000


def main(config_path: str = "configs/ml/atr_reversion_lightgbm_v1.yaml") -> None:
    cfg0 = load_atr_reversion_config(config_path)
    cfg = cfg0.model_copy(
        update={
            "universe_top_n": None,
            "features": cfg0.features.model_copy(
                update={"cross_sectional_zscore": False, "winsor_quantile": 0.0}
            ),
            "label": cfg0.label.model_copy(update={"cross_sectional_rank_label": False}),
        }
    )
    output = cfg0.output_root / f"{cfg0.name}_pit_liquidity_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"start config={config_path}")
    version, panel = _load_panel(cfg)
    log(f"loaded panel rows={len(panel):,} version={version}")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    log("building PIT rolling-liquidity top1000 flags")
    pit = _pit_liquidity_flags(panel)
    pit.to_parquet(output / "pit_liquidity_flags.parquet", index=False)
    log(f"PIT flags rows={len(pit):,}; active rows={int(pit['pit_top1000'].sum()):,}")

    # Keep full histories for stocks that ever enter the PIT universe during the experiment
    # so rolling ATR/percentile features have enough past context.
    start = pd.Timestamp(cfg.segments.train.start)
    end = pd.Timestamp(cfg.segments.test.end)
    ever = pit.loc[
        pit["trade_date"].between(start, end) & pit["pit_top1000"], "ts_code"
    ].unique()
    work_panel = panel[panel["ts_code"].isin(ever)].copy()
    log(f"feature panel rows={len(work_panel):,} stocks={len(ever):,}")

    cache = cfg0.output_root / "atr_reversion_pit_liquidity_dataset_w20_top1000.parquet"
    if cfg0.cache_dataset and cache.exists():
        log(f"loading cached raw dataset {cache}")
        dataset = pd.read_parquet(cache)
        features = FEATURE_GROUPS["all"]
    else:
        log("building raw ATR dataset for PIT universe histories")
        dataset, features = build_atr_reversion_dataset(work_panel, cfg.features, cfg.label)
        dataset.to_parquet(cache, index=False)
        log(f"cached raw dataset -> {cache}")

    log("attaching PIT flags and applying PIT-only preprocessing")
    dataset = _attach_and_preprocess_pit(dataset, pit, features, cfg0.features.winsor_quantile)
    dataset.to_parquet(output / "pit_model_dataset.parquet", index=False)
    log(f"PIT model rows={len(dataset):,}; usable={int(dataset['pit_top1000'].sum()):,}")

    predictions = _train_predict(dataset, features, cfg0, log)
    pred_path = output / "predictions_pit_all_features.parquet"
    predictions.to_parquet(pred_path, index=False)
    log(f"wrote predictions -> {pred_path}")

    test_panel = panel[
        panel["trade_date"].between(pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end))
    ].merge(pit, on=["trade_date", "ts_code"], how="left")
    test_panel["pit_top1000"] = test_panel["pit_top1000"].fillna(False).astype(bool)
    rows = []
    for top_n in TOP_NS:
        for rebalance_days in REBALANCE_DAYS:
            for cost in COST_BPS:
                log(f"backtest top_n={top_n} rebalance={rebalance_days} cost={cost}")
                daily, trades = _run_pit_backtest(
                    test_panel,
                    predictions,
                    top_n=top_n,
                    rebalance_days=rebalance_days,
                    cost_bps=cost,
                )
                tag = f"top{top_n}_rebalance{rebalance_days}_cost{cost}"
                daily.to_parquet(output / f"daily_{tag}.parquet", index=False)
                trades.to_parquet(output / f"trades_{tag}.parquet", index=False)
                metrics = _metrics(daily, trades)
                metrics.update({
                    "top_n": top_n,
                    "rebalance_days": rebalance_days,
                    "cost_bps": cost,
                    "avg_holding_count": float(daily["holding_count"].mean()),
                    "avg_cash_ratio": float(daily["cash_ratio"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                })
                rows.append(metrics)
                log(
                    f"done ann={metrics['annualized_return']:.2%} "
                    f"excess={metrics['annualized_excess_return']:.2%} "
                    f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_drawdown']:.2%}"
                )

    metrics_df = pd.DataFrame(rows).sort_values(
        ["annualized_excess_return", "sharpe"], ascending=False
    )
    metrics_df.to_csv(output / "pit_backtest_metrics.csv", index=False, encoding="utf-8-sig")
    yearly = _yearly_tables(output)
    yearly.to_csv(output / "pit_yearly_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "data_version": version,
                "run_dir": str(output),
                "liquidity_window": LIQUIDITY_WINDOW,
                "liquidity_min_periods": LIQUIDITY_MIN_PERIODS,
                "pit_top_n": PIT_TOP_N,
                "best": metrics_df.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(metrics_df, yearly, version), encoding="utf-8")
    log("wrote metrics/yearly/report")
    log("done")
    print(f"run_dir={output}")


def _load_panel(cfg):
    project = load_project(cfg.project_config)
    repo = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
    version, _manifest = repo.load_manifest(cfg.data_version)
    _, panel = repo.load_panel(version)
    return version, panel


def _pit_liquidity_flags(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["ts_code", "trade_date"]).copy()
    rolling_amount = (
        data["amount_cny"].where(data["amount_cny"] > 0)
        .groupby(data["ts_code"], sort=False)
        .rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_MIN_PERIODS)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(data.index)
    )
    data["rolling_amount_20"] = rolling_amount
    eligible = (
        data["is_tradeable"].fillna(False).astype(bool)
        & data["rolling_amount_20"].notna()
    )
    rank = data["rolling_amount_20"].where(eligible).groupby(data["trade_date"], sort=False).rank(
        method="first", ascending=False
    )
    data["pit_top1000"] = rank.le(PIT_TOP_N)
    return data[["trade_date", "ts_code", "rolling_amount_20", "pit_top1000"]].copy()


def _attach_and_preprocess_pit(
    dataset: pd.DataFrame,
    pit: pd.DataFrame,
    features: list[str],
    winsor_quantile: float,
) -> pd.DataFrame:
    out = dataset.merge(
        pit.rename(columns={"trade_date": "datetime", "ts_code": "instrument"}),
        on=["datetime", "instrument"],
        how="left",
    )
    out["pit_top1000"] = out["pit_top1000"].fillna(False).astype(bool)
    label_cols = [c for c in out.columns if c == "label" or c.startswith("label_")]
    out.loc[~out["pit_top1000"], features + label_cols + ["sample_weight"]] = np.nan
    scale_targets = [name for name in features if name not in NO_CROSS_SECTION_ZSCORE]
    out[scale_targets] = out[scale_targets].replace([np.inf, -np.inf], np.nan)
    if winsor_quantile:
        q = winsor_quantile
        valid = out["pit_top1000"]
        grouped = out.loc[valid].groupby("datetime")
        lower = grouped[scale_targets].transform(lambda s: s.quantile(q))
        upper = grouped[scale_targets].transform(lambda s: s.quantile(1 - q))
        clipped = out.loc[valid, scale_targets].clip(lower, upper)
        out.loc[valid, scale_targets] = clipped
    valid = out["pit_top1000"]
    grouped = out.loc[valid].groupby("datetime")[scale_targets]
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0)
    out.loc[valid, scale_targets] = (
        out.loc[valid, scale_targets] - mean
    ) / std.replace(0, np.nan)
    for col in label_cols:
        ranked = out.loc[valid].groupby("datetime")[col].rank(pct=True) - 0.5
        out.loc[valid, col] = ranked.where(out.loc[valid, col].notna())
    return out


def _train_predict(dataset: pd.DataFrame, features: list[str], cfg, log) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    cols = ["datetime", "instrument", *features, "label", "sample_weight", "pit_top1000"]
    train = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.train.start), pd.Timestamp(cfg.segments.train.end))
        & dataset["pit_top1000"],
        cols,
    ].dropna(subset=[*features, "label"])
    valid = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.valid.start), pd.Timestamp(cfg.segments.valid.end))
        & dataset["pit_top1000"],
        cols,
    ].dropna(subset=[*features, "label"])
    test = dataset.loc[
        dataset["datetime"].between(pd.Timestamp(cfg.segments.test.start), pd.Timestamp(cfg.segments.test.end))
        & dataset["pit_top1000"],
        cols,
    ].dropna(subset=features)
    log(f"model slices train={len(train):,} valid={len(valid):,} test_predictable={len(test):,}")
    params = cfg.model.model_dump()
    params.pop("objective", None)
    params.setdefault("verbosity", -1)
    params.setdefault("force_col_wise", True)
    model = LGBMRegressor(objective="regression", **params)
    fit_kwargs = {}
    if "sample_weight" in train:
        fit_kwargs["sample_weight"] = train["sample_weight"].fillna(1.0)
    eval_set = [(valid[features], valid["label"])] if len(valid) else None
    log("fitting LightGBM on PIT rows")
    model.fit(train[features], train["label"], eval_set=eval_set, **fit_kwargs)
    log("predicting PIT test rows")
    out = test[["datetime", "instrument"]].copy()
    out["factor_value"] = model.predict(test[features])
    return out.rename(columns={"datetime": "trade_date", "instrument": "ts_code"})


def _run_pit_backtest(panel, pred, *, top_n: int, rebalance_days: int, cost_bps: float):
    data = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = list(pd.Index(data["trade_date"].unique()).sort_values())
    by_date = {d: g.set_index("ts_code") for d, g in data.groupby("trade_date")}
    pred_by_date = {d: g.set_index("ts_code") for d, g in pred.groupby("trade_date")}
    cash = INITIAL_CASH
    positions: dict[str, dict] = {}
    daily_rows, trade_rows = [], []
    half_cost = cost_bps / 2.0 / 10_000.0

    for i, date in enumerate(dates):
        today = by_date[date]
        turnover = cost = 0.0
        buys = sells = 0
        if i > 0 and (i - 1) % rebalance_days == 0:
            signal_date = dates[i - 1]
            for code, pos in list(positions.items()):
                if code not in today.index or not _can_sell_at_open(today.loc[code]):
                    continue
                value = _position_value(pos, today.loc[code])
                sell_cost = value * half_cost
                cash += value - sell_cost
                turnover += value
                cost += sell_cost
                sells += 1
                trade_rows.append({"trade_date": date, "ts_code": code, "side": "SELL", "value": value, "cost": sell_cost})
                del positions[code]
            signals = pred_by_date.get(signal_date)
            if signals is not None and cash > 0:
                eligible = signals.join(today[[
                    "raw_open", "adj_open", "is_tradeable", "is_suspended",
                    "is_st", "is_delisting_period", "listing_trade_days", "is_limit_up_open",
                ]], how="inner")
                mask = (
                    eligible["factor_value"].notna()
                    & eligible["is_tradeable"].fillna(False).astype(bool)
                    & ~eligible["is_suspended"].fillna(True).astype(bool)
                    & ~eligible["is_st"].fillna(True).astype(bool)
                    & ~eligible["is_delisting_period"].fillna(True).astype(bool)
                    & eligible["listing_trade_days"].ge(60)
                    & ~eligible["is_limit_up_open"].fillna(False).astype(bool)
                )
                picks = eligible[mask].sort_values("factor_value", ascending=False).head(top_n)
                target_cash = cash / max(len(picks), 1)
                for code, row in picks.iterrows():
                    price = float(row["raw_open"])
                    shares = int(target_cash // (price * LOT_SIZE)) * LOT_SIZE
                    if shares <= 0:
                        continue
                    gross = shares * price
                    buy_cost = gross * half_cost
                    if gross + buy_cost > cash:
                        continue
                    cash -= gross + buy_cost
                    turnover += gross
                    cost += buy_cost
                    buys += 1
                    positions[code] = {
                        "shares": shares,
                        "entry_raw_open": price,
                        "entry_adj_open": float(row["adj_open"]),
                    }
                    trade_rows.append({"trade_date": date, "ts_code": code, "side": "BUY", "value": gross, "cost": buy_cost})
        pos_value = 0.0
        for code, pos in positions.items():
            pos_value += _position_value(pos, today.loc[code]) if code in today.index else pos["shares"] * pos["entry_raw_open"]
        nav = cash + pos_value
        daily_rows.append({
            "trade_date": date,
            "nav": nav,
            "cash": cash,
            "cash_ratio": cash / nav if nav > 0 else np.nan,
            "holding_count": len(positions),
            "turnover": turnover / (daily_rows[-1]["nav"] if daily_rows else INITIAL_CASH),
            "transaction_cost": cost,
            "executed_buys": buys,
            "executed_sells": sells,
        })
    daily = pd.DataFrame(daily_rows)
    daily["benchmark_return"] = csi1000_open_to_open_returns(dates)
    daily["return"] = daily["nav"].pct_change().fillna(0.0)
    daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    return daily, pd.DataFrame(trade_rows)


def _benchmark_return(data, dates, i):
    if i == 0:
        return 0.0
    signal_date = dates[i - 1]
    today = data[data["trade_date"].eq(dates[i])]
    signal = data[
        data["trade_date"].eq(signal_date)
        & data["pit_top1000"].fillna(False)
        & data["is_tradeable"].fillna(False)
    ]
    joined = signal[["ts_code", "adj_open"]].merge(
        today[["ts_code", "adj_open", "is_tradeable"]],
        on="ts_code",
        suffixes=("_prev", "_cur"),
    )
    joined = joined[joined["is_tradeable"].fillna(False)]
    ret = joined["adj_open_cur"] / joined["adj_open_prev"] - 1.0
    return float(ret.replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)


def _yearly_tables(out_dir: Path):
    rows = []
    for path in out_dir.glob("daily_top*_rebalance*_cost*.parquet"):
        parts = path.stem.removeprefix("daily_top").split("_")
        top_n = int(parts[0])
        rebalance = int(parts[1].removeprefix("rebalance"))
        cost = int(parts[2].removeprefix("cost"))
        df = pd.read_parquet(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for year, g in df.groupby(df["trade_date"].dt.year):
            if len(g) < 2:
                continue
            total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
            bench = (1 + g["benchmark_return"]).prod() - 1
            dd = g["nav"] / g["nav"].cummax() - 1
            rows.append({
                "top_n": top_n,
                "rebalance_days": rebalance,
                "cost_bps": cost,
                "year": int(year),
                "return": float(total),
                "benchmark_return": float(bench),
                "excess_return": float(total - bench),
                "max_drawdown": float(dd.min()),
            })
    return pd.DataFrame(rows)


def _report(metrics, yearly, version):
    show = metrics[[
        "top_n", "rebalance_days", "cost_bps", "annualized_return",
        "benchmark_annualized_return", "annualized_excess_return", "sharpe",
        "max_drawdown", "avg_daily_turnover", "avg_holding_count",
    ]].copy()
    for col in ["annualized_return", "benchmark_annualized_return", "annualized_excess_return", "max_drawdown", "avg_daily_turnover"]:
        show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly.copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown"]:
        y[col] = y[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return "\n".join([
        "# ATR Reversion PIT Rolling Liquidity Top1000 Backtest",
        "",
        f"- data version: `{version}`",
        "- universe: daily top1000 by 20d rolling amount, min 18 observations",
        "- model rows and benchmark are restricted to the same PIT universe",
        "",
        "## Overall",
        "",
        show.to_markdown(index=False),
        "",
        "## Yearly",
        "",
        y.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/ml/atr_reversion_lightgbm_v1.yaml")
