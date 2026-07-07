"""PPO position overlay for the frozen fit-quality flip rule.

The stock selection, next-open execution, tradeability filters, HMM gate, and
fit-quality score flip are already baked into the base backtest files.  This
script only learns a rebalance-cycle exposure multiplier, so it can be audited
as an execution-layer overlay rather than a new stock selector.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atr_reversion_small_portfolio_backtest import _json_default, _metrics


BASE_DIR = Path(
    "artifacts/atr_reversion_runs/"
    "atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z/"
    "fit_quality_sensitivity_20260707T031019Z"
)
OUTPUT_ROOT = Path("artifacts/atr_reversion_runs/atr_lower_shadow_reversion_v1_pit_liquidity_20260706T091843Z")
LOOKBACK = 40
MIN_OBS = 15
POLICY = "fit_quality_flip_only"
TOP_N = 5
COSTS = [20]
REBALANCE_DAYS = 10
INITIAL_CASH = 1_000_000.0
ACTION_SETS = {
    "ppo_defensive": np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float),
    "ppo_micro": np.array([0.50, 0.75, 1.0], dtype=float),
}
ALL_FOLDS = ["test_2022", "test_2023", "test_2024", "test_2025", "test_2026h1"]
TEST_FOLDS = ["test_2025", "test_2026h1"]
TRAIN_FOLDS = {
    "test_2025": ["test_2022", "test_2023", "test_2024"],
    "test_2026h1": ["test_2022", "test_2023", "test_2024", "test_2025"],
}

FEATURE_COLUMNS = [
    "fit_obs",
    "rank_ic_rolling",
    "decile_spread_rolling",
    "top5_excess_rolling",
    "top5_hit_rolling",
    "score_direction",
    "flip_signal",
    "strategy_ret_5",
    "strategy_ret_10",
    "strategy_ret_20",
    "strategy_excess_10",
    "strategy_excess_20",
    "strategy_vol_20",
    "strategy_drawdown_20",
    "benchmark_ret_5",
    "benchmark_ret_20",
    "benchmark_vol_20",
    "benchmark_drawdown_20",
    "base_exposure_at_signal",
    "base_turnover_10",
    "prev_cycle_return",
    "prev_cycle_drawdown",
    "prev_action",
]


@dataclass
class PPOModel:
    actions: np.ndarray
    policy_w: np.ndarray
    policy_b: np.ndarray
    value_w: np.ndarray
    value_b: float
    mean: np.ndarray
    std: np.ndarray


def main() -> None:
    output = OUTPUT_ROOT / f"ppo_frozen_rule_overlay_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    gate_scores = _load_gate_scores()
    cycle_frames = []
    for cost in COSTS:
        for fold in ALL_FOLDS:
            cycles = _cycles_for_fold(fold, cost, gate_scores)
            cycle_frames.append(cycles)
            log(f"loaded cycles fold={fold} cost={cost} cycles={len(cycles)}")
    cycles_all = pd.concat(cycle_frames, ignore_index=True)
    cycles_all.to_csv(output / "ppo_cycle_dataset.csv", index=False, encoding="utf-8-sig")

    metrics_rows = []
    yearly_rows = []
    action_frames = []
    for cost in COSTS:
        for test_fold in TEST_FOLDS:
            train = cycles_all[
                cycles_all["cost_bps"].eq(cost) & cycles_all["fold"].isin(TRAIN_FOLDS[test_fold])
            ].copy()
            test = cycles_all[
                cycles_all["cost_bps"].eq(cost) & cycles_all["fold"].eq(test_fold)
            ].copy()
            base_daily, base_trades = _load_fold(test_fold, cost)
            base_metrics = _metrics(base_daily, base_trades)
            base_metrics.update(
                {
                    "policy": "frozen_rule_base",
                    "fold": test_fold,
                    "cost_bps": cost,
                    "top_n": TOP_N,
                    "rebalance_days": REBALANCE_DAYS,
                    "avg_exposure": float(base_daily["exposure"].mean()),
                    "avg_daily_turnover": float(base_daily["turnover"].mean()),
                }
            )
            metrics_rows.append(base_metrics)
            yearly_rows.extend(_yearly_rows(base_daily, "frozen_rule_base", test_fold, cost))

            for policy_name, actions in ACTION_SETS.items():
                log(
                    f"training {policy_name} test={test_fold} cost={cost} "
                    f"train_cycles={len(train)} test_cycles={len(test)} actions={actions.tolist()}"
                )
                model = _train_ppo(train, seed=20260707 + cost + len(test_fold) + len(actions), actions=actions)
                acted = _predict_actions(model, test)
                acted = _smooth_actions(acted, actions=actions)
                acted["policy"] = policy_name
                acted.to_csv(output / f"{policy_name}_actions_{test_fold}_cost{cost}.csv", index=False, encoding="utf-8-sig")
                action_frames.append(acted)
                ppo_daily = _daily_with_overlay(test_fold, cost, acted)
                ppo_daily.to_parquet(output / f"daily_{policy_name}_{test_fold}_top{TOP_N}_cost{cost}.parquet", index=False)
                ppo_metrics = _metrics(ppo_daily, pd.DataFrame())
                ppo_metrics.update(
                    {
                        "policy": policy_name,
                        "fold": test_fold,
                        "cost_bps": cost,
                        "top_n": TOP_N,
                        "rebalance_days": REBALANCE_DAYS,
                        "avg_exposure": float(ppo_daily["exposure"].mean()),
                        "avg_daily_turnover": float(ppo_daily["turnover"].mean()),
                    }
                )
                metrics_rows.append(ppo_metrics)
                yearly_rows.extend(_yearly_rows(ppo_daily, policy_name, test_fold, cost))
                log(
                    f"{test_fold} cost={cost} {policy_name} base_ann={base_metrics['annualized_return']:.2%} "
                    f"ppo_ann={ppo_metrics['annualized_return']:.2%} "
                    f"base_dd={base_metrics['max_drawdown']:.2%} ppo_dd={ppo_metrics['max_drawdown']:.2%}"
                )

    metrics_df = pd.DataFrame(metrics_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    actions_df = pd.concat(action_frames, ignore_index=True)
    comparison = _comparison(metrics_df)
    action_summary = _action_summary(actions_df)
    metrics_df.to_csv(output / "ppo_overlay_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "ppo_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    actions_df.to_csv(output / "ppo_overlay_actions.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(output / "ppo_action_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "ppo_overlay_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "base_dir": str(BASE_DIR),
                "run_dir": str(output),
                "lookback": LOOKBACK,
                "min_obs": MIN_OBS,
                "action_sets": {k: v.tolist() for k, v in ACTION_SETS.items()},
                "test_folds": TEST_FOLDS,
                "train_folds": TRAIN_FOLDS,
                "comparison": comparison.to_dict("records"),
                "action_summary": action_summary.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(comparison, yearly_df, action_summary), encoding="utf-8")
    log(f"done output={output}")
    print(f"run_dir={output}")


def _load_gate_scores() -> pd.DataFrame:
    scores = pd.read_csv(BASE_DIR / "sensitivity_gate_scores.csv")
    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    mask = (
        scores["lookback"].eq(LOOKBACK)
        & scores["min_obs"].eq(MIN_OBS)
        & scores["policy"].eq(POLICY)
        & scores["cost_bps"].isin(COSTS)
    )
    out = scores.loc[mask].copy()
    out["flip_signal"] = out["score_direction"].lt(0.0).astype(float)
    return out.sort_values(["fold", "cost_bps", "trade_date"]).reset_index(drop=True)


def _tag(fold: str, cost: int) -> str:
    return f"lookback{LOOKBACK}_minobs{MIN_OBS}_{fold}_{POLICY}_top{TOP_N}_cost{cost}"


def _load_fold(fold: str, cost: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(BASE_DIR / f"daily_{_tag(fold, cost)}.parquet")
    trades = pd.read_parquet(BASE_DIR / f"trades_{_tag(fold, cost)}.parquet")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    return daily.sort_values("trade_date").reset_index(drop=True), trades.sort_values("trade_date").reset_index(drop=True)


def _cycles_for_fold(fold: str, cost: int, gate_scores: pd.DataFrame) -> pd.DataFrame:
    daily, _ = _load_fold(fold, cost)
    scores = gate_scores[gate_scores["fold"].eq(fold) & gate_scores["cost_bps"].eq(cost)].copy()
    scores = scores.set_index("trade_date")
    dates = daily["trade_date"].tolist()
    rows = []
    prev_cycle_return = 0.0
    prev_cycle_drawdown = 0.0
    prev_action = 1.0
    for signal_idx in range(0, len(dates) - 1, REBALANCE_DAYS):
        start = signal_idx + 1
        end = min(signal_idx + REBALANCE_DAYS, len(dates) - 1)
        segment = daily.iloc[start : end + 1].copy()
        hist = daily.iloc[: signal_idx + 1].copy()
        if segment.empty or hist.empty:
            continue
        signal_date = dates[signal_idx]
        score = _score_at(scores, signal_date)
        base_returns = segment["return"].fillna(0.0).to_numpy(float)
        wealth = np.cumprod(1.0 + base_returns)
        cycle_return = float(wealth[-1] - 1.0)
        cycle_drawdown = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
        record = {
            "fold": fold,
            "cost_bps": cost,
            "signal_date": signal_date,
            "start_date": segment["trade_date"].iloc[0],
            "end_date": segment["trade_date"].iloc[-1],
            "base_cycle_return": cycle_return,
            "base_cycle_drawdown": cycle_drawdown,
            "base_cycle_benchmark_return": float((1.0 + segment["benchmark_return"].fillna(0.0)).prod() - 1.0),
            "base_cycle_exposure": float(segment["exposure"].mean()),
            "prev_cycle_return": prev_cycle_return,
            "prev_cycle_drawdown": prev_cycle_drawdown,
            "prev_action": prev_action,
            **_score_features(score),
            **_history_features(hist),
        }
        rows.append(record)
        prev_cycle_return = cycle_return
        prev_cycle_drawdown = cycle_drawdown
        prev_action = 1.0
    out = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in out:
            out[col] = np.nan
    return out


def _score_at(scores: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    if date in scores.index:
        row = scores.loc[date]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row
    eligible = scores[scores.index <= date]
    if eligible.empty:
        return pd.Series(dtype=float)
    return eligible.iloc[-1]


def _score_features(score: pd.Series) -> dict:
    return {
        "fit_obs": float(score.get("fit_obs", 0.0)),
        "rank_ic_rolling": float(score.get("rank_ic_rolling", np.nan)),
        "decile_spread_rolling": float(score.get("decile_spread_rolling", np.nan)),
        "top5_excess_rolling": float(score.get("top5_excess_rolling", np.nan)),
        "top5_hit_rolling": float(score.get("top5_hit_rolling", np.nan)),
        "score_direction": float(score.get("score_direction", 1.0)),
        "flip_signal": float(score.get("flip_signal", 0.0)),
    }


def _history_features(hist: pd.DataFrame) -> dict:
    ret = hist["return"].fillna(0.0)
    bench = hist["benchmark_return"].fillna(0.0)
    nav20 = (1.0 + ret.tail(20)).cumprod()
    bench20 = (1.0 + bench.tail(20)).cumprod()
    return {
        "strategy_ret_5": _compound(ret.tail(5)),
        "strategy_ret_10": _compound(ret.tail(10)),
        "strategy_ret_20": _compound(ret.tail(20)),
        "strategy_excess_10": _compound(ret.tail(10)) - _compound(bench.tail(10)),
        "strategy_excess_20": _compound(ret.tail(20)) - _compound(bench.tail(20)),
        "strategy_vol_20": float(ret.tail(20).std(ddof=0)),
        "strategy_drawdown_20": float((nav20 / nav20.cummax() - 1.0).min()) if len(nav20) else 0.0,
        "benchmark_ret_5": _compound(bench.tail(5)),
        "benchmark_ret_20": _compound(bench.tail(20)),
        "benchmark_vol_20": float(bench.tail(20).std(ddof=0)),
        "benchmark_drawdown_20": float((bench20 / bench20.cummax() - 1.0).min()) if len(bench20) else 0.0,
        "base_exposure_at_signal": float(hist["exposure"].iloc[-1]),
        "base_turnover_10": float(hist["turnover"].tail(10).sum()),
    }


def _compound(s: pd.Series) -> float:
    return float((1.0 + s.fillna(0.0)).prod() - 1.0) if len(s) else 0.0


def _standardize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    med = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(med)
    x_test = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(med)
    mean = x_train.mean().to_numpy(float)
    std = x_train.std(ddof=0).replace(0.0, 1.0).to_numpy(float)
    return (x_train.to_numpy(float) - mean) / std, (x_test.to_numpy(float) - mean) / std, mean, std


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _log_probs(probs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs[np.arange(len(actions)), actions], 1e-12, 1.0))


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def _train_ppo(
    cycles: pd.DataFrame,
    *,
    seed: int,
    actions: np.ndarray,
    epochs: int = 260,
    rollout_repeats: int = 10,
) -> PPOModel:
    rng = np.random.default_rng(seed)
    x, _, mean, std = _standardize(cycles, cycles)
    n, d = x.shape
    k = len(actions)
    policy_w = rng.normal(0.0, 0.015, size=(d, k))
    policy_b = np.linspace(-0.04, 0.06, k, dtype=float)
    value_w = np.zeros(d)
    value_b = 0.0
    gamma = 0.94
    clip_eps = 0.18
    lr_pi = 0.010
    lr_v = 0.018
    entropy_coef = 0.003

    for _ in range(epochs):
        obs = []
        acts = []
        rewards = []
        old_logps = []
        prev_action = 1.0
        for _repeat in range(rollout_repeats):
            for i in range(n):
                xi = x[i : i + 1]
                probs = _softmax(xi @ policy_w + policy_b)[0]
                action_idx = int(rng.choice(k, p=probs))
                action = float(actions[action_idx])
                row = cycles.iloc[i]
                cycle_ret = float(row["base_cycle_return"])
                cycle_dd = abs(float(row["base_cycle_drawdown"]))
                base_exposure = float(row["base_cycle_exposure"])
                excess_ret = cycle_ret - float(row["base_cycle_benchmark_return"])
                action_change = abs(action - prev_action)
                reward = (
                    action * cycle_ret
                    + 0.25 * action * excess_ret
                    - 0.18 * action * cycle_dd
                    - 0.004 * action_change
                    - 0.0015 * max(0.0, action - base_exposure)
                )
                obs.append(x[i])
                acts.append(action_idx)
                rewards.append(reward)
                old_logps.append(math.log(max(probs[action_idx], 1e-12)))
                prev_action = action

        obs_arr = np.asarray(obs, dtype=float)
        act_arr = np.asarray(acts, dtype=int)
        reward_arr = np.asarray(rewards, dtype=float)
        old_logp_arr = np.asarray(old_logps, dtype=float)
        returns = _discounted_returns(reward_arr, gamma)
        values = obs_arr @ value_w + value_b
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std(ddof=0) + 1e-8)

        probs = _softmax(obs_arr @ policy_w + policy_b)
        new_logp = _log_probs(probs, act_arr)
        ratio = np.exp(new_logp - old_logp_arr)
        active = ((adv >= 0.0) & (ratio <= 1.0 + clip_eps)) | ((adv < 0.0) & (ratio >= 1.0 - clip_eps))
        coeff = np.where(active, adv * ratio, 0.0) / len(obs_arr)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(act_arr)), act_arr] = 1.0
        grad_logits = coeff[:, None] * (onehot - probs)
        entropy_grad = -entropy_coef * probs * (np.log(np.clip(probs, 1e-12, 1.0)) + 1.0) / len(obs_arr)
        grad_logits += entropy_grad
        policy_w += lr_pi * obs_arr.T @ grad_logits
        policy_b += lr_pi * grad_logits.sum(axis=0)

        value_pred = obs_arr @ value_w + value_b
        v_err = returns - value_pred
        value_w += lr_v * (obs_arr.T @ v_err) / len(obs_arr)
        value_b += lr_v * float(v_err.mean())

    return PPOModel(actions, policy_w, policy_b, value_w, value_b, mean, std)


def _predict_actions(model: PPOModel, cycles: pd.DataFrame) -> pd.DataFrame:
    x = cycles[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    center = pd.Series(model.mean, index=FEATURE_COLUMNS)
    x = x.fillna(center)
    x_arr = (x.to_numpy(float) - model.mean) / model.std
    probs = _softmax(x_arr @ model.policy_w + model.policy_b)
    action_idx = probs.argmax(axis=1)
    out = cycles.copy()
    out["ppo_raw_action_idx"] = action_idx
    out["ppo_raw_multiplier"] = model.actions[action_idx]
    for idx, action in enumerate(model.actions):
        out[f"prob_{action:.2f}"] = probs[:, idx]
    return out


def _smooth_actions(cycles: pd.DataFrame, *, actions: np.ndarray, max_step: int = 2) -> pd.DataFrame:
    out = cycles.sort_values("signal_date").copy()
    smoothed_idx = []
    prev_idx = len(actions) - 1
    for raw_idx in out["ppo_raw_action_idx"].astype(int):
        lo = max(0, prev_idx - max_step)
        hi = min(len(actions) - 1, prev_idx + max_step)
        idx = int(np.clip(raw_idx, lo, hi))
        smoothed_idx.append(idx)
        prev_idx = idx
    out["ppo_action_idx"] = smoothed_idx
    out["ppo_multiplier"] = actions[out["ppo_action_idx"].to_numpy(int)]
    return out


def _daily_with_overlay(fold: str, cost: int, cycles: pd.DataFrame) -> pd.DataFrame:
    daily, _ = _load_fold(fold, cost)
    out = daily.copy()
    out["base_return"] = out["return"]
    out["base_nav"] = out["nav"]
    out["ppo_multiplier"] = 1.0
    out["return"] = 0.0
    for _, cycle in cycles.iterrows():
        mask = out["trade_date"].between(pd.Timestamp(cycle["start_date"]), pd.Timestamp(cycle["end_date"]))
        multiplier = float(cycle["ppo_multiplier"])
        out.loc[mask, "ppo_multiplier"] = multiplier
        out.loc[mask, "return"] = out.loc[mask, "base_return"].fillna(0.0) * multiplier
    out["nav"] = INITIAL_CASH * (1.0 + out["return"].fillna(0.0)).cumprod()
    out["excess_return"] = out["return"] - out["benchmark_return"]
    out["exposure"] = out["exposure"].fillna(0.0) * out["ppo_multiplier"]
    out["turnover"] = out["turnover"].fillna(0.0) * out["ppo_multiplier"]
    out["transaction_cost"] = out["transaction_cost"].fillna(0.0) * out["ppo_multiplier"]
    out["cash_ratio"] = 1.0 - out["exposure"].clip(0.0, 1.0)
    return out


def _yearly_rows(daily: pd.DataFrame, label: str, fold: str, cost: int) -> list[dict]:
    rows = []
    for year, g in daily.groupby(daily["trade_date"].dt.year):
        if len(g) < 2:
            continue
        total = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0
        bench = (1.0 + g["benchmark_return"]).prod() - 1.0
        dd = g["nav"] / g["nav"].cummax() - 1.0
        rows.append(
            {
                "policy": label,
                "fold": fold,
                "year": int(year),
                "cost_bps": cost,
                "return": float(total),
                "benchmark_return": float(bench),
                "excess_return": float(total - bench),
                "max_drawdown": float(dd.min()),
                "avg_exposure": float(g["exposure"].mean()),
            }
        )
    return rows


def _comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["policy"].eq("frozen_rule_base")].copy()
    overlays = metrics[metrics["policy"].ne("frozen_rule_base")].copy()
    keep = ["fold", "cost_bps", "annualized_return", "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure"]
    left = base[keep].rename(
        columns={
            "annualized_return": "base_ann",
            "annualized_excess_return": "base_excess",
            "sharpe": "base_sharpe",
            "max_drawdown": "base_maxdd",
            "avg_exposure": "base_exposure",
        }
    )
    right = overlays[["policy", *keep]].rename(
            columns={
                "annualized_return": "ppo_ann",
                "annualized_excess_return": "ppo_excess",
                "sharpe": "ppo_sharpe",
                "max_drawdown": "ppo_maxdd",
                "avg_exposure": "ppo_exposure",
            }
    )
    out = left.merge(
        right,
        on=["fold", "cost_bps"],
        how="inner",
    )
    out["ann_delta"] = out["ppo_ann"] - out["base_ann"]
    out["excess_delta"] = out["ppo_excess"] - out["base_excess"]
    out["maxdd_delta"] = out["ppo_maxdd"] - out["base_maxdd"]
    out["exposure_delta"] = out["ppo_exposure"] - out["base_exposure"]
    return out


def _action_summary(actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, fold, cost), g in actions.groupby(["policy", "fold", "cost_bps"]):
        counts = g["ppo_multiplier"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "policy": policy,
                "fold": fold,
                "cost_bps": int(cost),
                "cycles": int(len(g)),
                "avg_multiplier": float(g["ppo_multiplier"].mean()),
                "zero_ratio": float(counts.get(0.0, 0.0)),
                "half_or_less_ratio": float(sum(v for k, v in counts.items() if k <= 0.5)),
                "full_ratio": float(counts.get(1.0, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    return out


def _report(comparison: pd.DataFrame, yearly: pd.DataFrame, action_summary: pd.DataFrame) -> str:
    comp = _fmt_pct(
        comparison,
        [
            "base_ann",
            "base_excess",
            "base_maxdd",
            "base_exposure",
            "ppo_ann",
            "ppo_excess",
            "ppo_maxdd",
            "ppo_exposure",
            "ann_delta",
            "excess_delta",
            "maxdd_delta",
            "exposure_delta",
        ],
    )
    for col in ["base_sharpe", "ppo_sharpe"]:
        comp[col] = comp[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = _fmt_pct(yearly.copy(), ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"])
    a = _fmt_pct(action_summary.copy(), ["avg_multiplier", "zero_ratio", "half_or_less_ratio", "full_ratio"])
    return "\n".join(
        [
            "# PPO Frozen Rule Position Overlay",
            "",
            "Base strategy: fit_quality_flip_only, lookback=40, min_obs=15, Top5, 10-day rebalance, cost=20bps.",
            "PPO changes only the rebalance-cycle exposure multiplier.",
            "",
            "## OOS Comparison",
            "",
            comp.to_markdown(index=False),
            "",
            "## Yearly",
            "",
            y.to_markdown(index=False),
            "",
            "## Action Summary",
            "",
            a.to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    main()
