"""PPO-style position overlay for ATR reversion execution.

This experiment keeps stock selection and hard gates fixed.  The policy only
chooses a rebalance-cycle exposure multiplier from {0, 25%, 50%, 75%, 100%}.
It is intentionally small and dependency-free so it can be audited before any
larger RL stack is introduced.
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
    "training_window_defensive_gate_rolling_2y_20260706T133808Z"
)
OUTPUT_ROOT = Path("artifacts/atr_reversion_ppo_overlay")
VARIANT = "rolling_2y"
POLICY = "risk_kill_only"
TOP_N = 5
REBALANCE_DAYS = 10
INITIAL_CASH = 1_000_000.0
ACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
TEST_FOLDS = ["test_2025", "test_2026h1"]
TRAIN_FOLDS = {
    "test_2025": ["test_2022", "test_2023", "test_2024"],
    "test_2026h1": ["test_2022", "test_2023", "test_2024", "test_2025"],
}
FEATURE_COLUMNS = [
    "market_ret_20",
    "market_ret_60",
    "market_drawdown_60",
    "market_vol_20",
    "market_breadth_20",
    "xsec_vol_20",
    "turnover_chg_5_20",
    "momentum_minus_reversal_20",
    "lower_shadow_style_20",
    "repair_style_20",
    "core_signal_style_20",
    "pred_score_mean",
    "pred_score_std",
    "pred_score_p90",
    "top_score_mean",
    "top_score_spread",
    "top_industry_hhi",
    "signal_ic_20",
    "signal_ic_60",
    "selected_momentum_rank",
    "selected_industry_strength",
    "selected_hot_industry_ratio",
    "selected_industry_hhi",
    "top5_excess_5round",
    "top5_winrate_5round",
    "gate_score",
    "strategy_gate",
    "base_exposure",
    "prev_cycle_return",
    "prev_action",
]


@dataclass
class PPOModel:
    policy_w: np.ndarray
    policy_b: np.ndarray
    value_w: np.ndarray
    value_b: float
    mean: np.ndarray
    std: np.ndarray


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


def _standardize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    med = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(med)
    x_test = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(med)
    mean = x_train.mean().to_numpy(float)
    std = x_train.std(ddof=0).replace(0.0, 1.0).to_numpy(float)
    return (
        (x_train.to_numpy(float) - mean) / std,
        (x_test.to_numpy(float) - mean) / std,
        mean,
        std,
    )


def _train_ppo(cycles: pd.DataFrame, *, seed: int = 42, epochs: int = 220, rollout_repeats: int = 6) -> PPOModel:
    rng = np.random.default_rng(seed)
    x, _, mean, std = _standardize(cycles, cycles)
    n, d = x.shape
    k = len(ACTIONS)
    policy_w = rng.normal(0.0, 0.02, size=(d, k))
    policy_b = np.zeros(k)
    value_w = np.zeros(d)
    value_b = 0.0
    gamma = 0.96
    clip_eps = 0.2
    lr_pi = 0.015
    lr_v = 0.02
    entropy_coef = 0.002

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
                action = ACTIONS[action_idx]
                row = cycles.iloc[i]
                cycle_ret = float(row["base_cycle_return"])
                cycle_dd = abs(float(row["base_cycle_drawdown"]))
                hard_gate = float(row["base_exposure"])
                turnover_penalty = 0.0008 * abs(action - prev_action)
                reward = hard_gate * action * cycle_ret - 0.15 * hard_gate * action * cycle_dd - turnover_penalty
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

    return PPOModel(policy_w, policy_b, value_w, value_b, mean, std)


def _predict_actions(model: PPOModel, cycles: pd.DataFrame) -> pd.DataFrame:
    x = cycles[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    train_center = pd.Series(model.mean, index=FEATURE_COLUMNS)
    x = x.fillna(train_center)
    x_arr = (x.to_numpy(float) - model.mean) / model.std
    probs = _softmax(x_arr @ model.policy_w + model.policy_b)
    action_idx = probs.argmax(axis=1)
    out = cycles.copy()
    out["ppo_action_idx"] = action_idx
    out["ppo_multiplier"] = ACTIONS[action_idx]
    for idx, action in enumerate(ACTIONS):
        out[f"prob_{action:.2f}"] = probs[:, idx]
    return out


def _load_fold(fold: str, cost: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag = f"{VARIANT}_{fold}_{POLICY}_top{TOP_N}_cost{cost}"
    daily = pd.read_parquet(BASE_DIR / f"daily_{tag}.parquet")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    scores = pd.read_csv(BASE_DIR / f"gate_scores_{VARIANT}_{fold}_{POLICY}_cost{cost}.csv")
    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    return daily.sort_values("trade_date").reset_index(drop=True), scores.sort_values("trade_date").reset_index(drop=True)


def _cycles_for_fold(fold: str, cost: int) -> pd.DataFrame:
    daily, scores = _load_fold(fold, cost)
    score_by_date = scores.set_index("trade_date")
    rows = []
    dates = daily["trade_date"].tolist()
    prev_cycle_return = 0.0
    prev_action = 1.0
    for signal_idx in range(0, len(dates) - 1, REBALANCE_DAYS):
        start = signal_idx + 1
        end = min(signal_idx + REBALANCE_DAYS, len(dates) - 1)
        segment = daily.iloc[start : end + 1].copy()
        if segment.empty:
            continue
        signal_date = dates[signal_idx]
        if signal_date in score_by_date.index:
            row = score_by_date.loc[signal_date].copy()
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
        else:
            row = scores.iloc[min(signal_idx, len(scores) - 1)].copy()
        base_daily = segment["return"].fillna(0.0).to_numpy(float)
        wealth = np.cumprod(1.0 + base_daily)
        base_cycle_return = float(wealth[-1] - 1.0)
        base_cycle_drawdown = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
        record = {col: row.get(col, np.nan) for col in FEATURE_COLUMNS if col not in {"base_exposure", "prev_cycle_return", "prev_action"}}
        record.update({
            "fold": fold,
            "cost_bps": cost,
            "signal_date": signal_date,
            "start_date": segment["trade_date"].iloc[0],
            "end_date": segment["trade_date"].iloc[-1],
            "base_cycle_return": base_cycle_return,
            "base_cycle_drawdown": base_cycle_drawdown,
            "base_exposure": float(segment["exposure"].mean()),
            "prev_cycle_return": prev_cycle_return,
            "prev_action": prev_action,
        })
        rows.append(record)
        prev_cycle_return = base_cycle_return
        prev_action = 1.0
    out = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in out:
            out[col] = np.nan
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
        rows.append({
            "policy": label,
            "fold": fold,
            "year": int(year),
            "cost_bps": cost,
            "return": float(total),
            "benchmark_return": float(bench),
            "excess_return": float(total - bench),
            "max_drawdown": float(dd.min()),
            "avg_exposure": float(g["exposure"].mean()),
        })
    return rows


def main() -> None:
    output = OUTPUT_ROOT / f"ppo_position_overlay_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    t0 = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    all_cycles = []
    for cost in [10, 20]:
        for fold in ["test_2022", "test_2023", "test_2024", "test_2025", "test_2026h1"]:
            c = _cycles_for_fold(fold, cost)
            all_cycles.append(c)
            log(f"loaded cycles fold={fold} cost={cost} cycles={len(c)}")
    cycles_all = pd.concat(all_cycles, ignore_index=True)
    cycles_all.to_csv(output / "cycle_dataset.csv", index=False, encoding="utf-8-sig")

    metrics_rows = []
    yearly_rows = []
    action_frames = []
    for cost in [10, 20]:
        for test_fold in TEST_FOLDS:
            train = cycles_all[
                cycles_all["cost_bps"].eq(cost) & cycles_all["fold"].isin(TRAIN_FOLDS[test_fold])
            ].copy()
            test = cycles_all[
                cycles_all["cost_bps"].eq(cost) & cycles_all["fold"].eq(test_fold)
            ].copy()
            log(f"training PPO cost={cost} test={test_fold} train_cycles={len(train)} test_cycles={len(test)}")
            model = _train_ppo(train, seed=1000 + cost + len(test_fold))
            acted = _predict_actions(model, test)
            acted.to_csv(output / f"ppo_actions_{test_fold}_cost{cost}.csv", index=False, encoding="utf-8-sig")
            action_frames.append(acted)
            daily_ppo = _daily_with_overlay(test_fold, cost, acted)
            daily_ppo.to_parquet(output / f"daily_ppo_overlay_{test_fold}_top{TOP_N}_cost{cost}.parquet", index=False)
            base_daily, _ = _load_fold(test_fold, cost)
            empty_trades = pd.DataFrame()
            ppo_metrics = _metrics(daily_ppo, empty_trades)
            base_metrics = _metrics(base_daily, empty_trades)
            for name, metrics, daily in [
                ("risk_kill_only", base_metrics, base_daily),
                ("ppo_overlay", ppo_metrics, daily_ppo),
            ]:
                metrics.update({
                    "policy": name,
                    "fold": test_fold,
                    "cost_bps": cost,
                    "top_n": TOP_N,
                    "rebalance_days": REBALANCE_DAYS,
                    "avg_exposure": float(daily["exposure"].mean()),
                    "avg_daily_turnover": float(daily["turnover"].mean()),
                })
                metrics_rows.append(metrics)
                yearly_rows.extend(_yearly_rows(daily, name, test_fold, cost))
            log(
                f"{test_fold} cost={cost} base_ann={base_metrics['annualized_return']:.2%} "
                f"ppo_ann={ppo_metrics['annualized_return']:.2%} "
                f"ppo_avg_exposure={daily_ppo['exposure'].mean():.2%}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    actions_df = pd.concat(action_frames, ignore_index=True)
    comparison = _comparison(metrics_df)
    metrics_df.to_csv(output / "ppo_overlay_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output / "ppo_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    actions_df.to_csv(output / "ppo_overlay_actions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "ppo_overlay_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "base_dir": str(BASE_DIR),
                "run_dir": str(output),
                "actions": ACTIONS.tolist(),
                "test_folds": TEST_FOLDS,
                "train_folds": TRAIN_FOLDS,
                "metrics": metrics_df.to_dict("records"),
                "comparison": comparison.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(comparison, yearly_df, actions_df), encoding="utf-8")
    log(f"done output={output}")
    print(f"run_dir={output}")


def _comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["policy"].eq("risk_kill_only")].copy()
    ppo = metrics[metrics["policy"].eq("ppo_overlay")].copy()
    keep = ["fold", "cost_bps", "annualized_return", "annualized_excess_return", "sharpe", "max_drawdown", "avg_exposure"]
    out = base[keep].rename(columns={
        "annualized_return": "base_ann",
        "annualized_excess_return": "base_excess",
        "sharpe": "base_sharpe",
        "max_drawdown": "base_maxdd",
        "avg_exposure": "base_exposure",
    }).merge(
        ppo[keep].rename(columns={
            "annualized_return": "ppo_ann",
            "annualized_excess_return": "ppo_excess",
            "sharpe": "ppo_sharpe",
            "max_drawdown": "ppo_maxdd",
            "avg_exposure": "ppo_exposure",
        }),
        on=["fold", "cost_bps"],
        how="inner",
    )
    out["ann_delta"] = out["ppo_ann"] - out["base_ann"]
    out["excess_delta"] = out["ppo_excess"] - out["base_excess"]
    out["maxdd_delta"] = out["ppo_maxdd"] - out["base_maxdd"]
    out["exposure_delta"] = out["ppo_exposure"] - out["base_exposure"]
    return out


def _report(comparison: pd.DataFrame, yearly: pd.DataFrame, actions: pd.DataFrame) -> str:
    comp = comparison.copy()
    for col in [c for c in comp.columns if c.endswith(("ann", "excess", "maxdd", "exposure", "delta"))]:
        comp[col] = comp[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    comp["base_sharpe"] = comp["base_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    comp["ppo_sharpe"] = comp["ppo_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    y = yearly[yearly["year"].isin([2025, 2026])].copy()
    for col in ["return", "benchmark_return", "excess_return", "max_drawdown", "avg_exposure"]:
        y[col] = y[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    a = actions.groupby(["fold", "cost_bps", "ppo_multiplier"]).size().reset_index(name="cycles")
    return "\n".join([
        "# PPO Position Overlay",
        "",
        "## OOS Comparison",
        "",
        comp.to_markdown(index=False),
        "",
        "## Yearly 2025/2026",
        "",
        y.to_markdown(index=False),
        "",
        "## Action Counts",
        "",
        a.to_markdown(index=False),
        "",
    ])


if __name__ == "__main__":
    main()
