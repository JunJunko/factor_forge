"""风格归因：把价值策略超额收益拆成风格因子暴露 + 残差 alpha。

本模块是 Day 1 判决门。判决结果（真实 alpha / 纯风格 beta / 2025 regime / 混合）决定后续
做特征分流、风格中性化还是 regime 研究。

时序约定（必须与 ``backtest/engine.py`` 的开盘计价节奏一致，否则 beta 被隔夜跳空污染）：
- 引擎用 ``adj_open`` 标记头寸（``engine.py:_position_value``），基准用
  ``adj_open.shift(-2)/adj_open.shift(-1)-1`` 并映射到 T+2（``engine.py:_benchmark_returns``）。
- 因此风格因子收益也用同一约定：T 收盘知信号/universe → T+1 开盘进 → T+2 开盘出，
  映射到 T+2，与 ``daily.excess_return`` 同轴。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.config import CostModel, ExecutionConstraints, load_project
from factor_forge.data.repository import DataVersionRepository
from factor_forge.ml.config import PortfolioConfig
from factor_forge.ml.value_dataset import _rolling_by_stock, attach_point_in_time_fundamentals


# 风格因子方向约定：signal 越大 = 做多腿。factor_return = top_tercile - bottom_tercile。
# - size   : -log_total_mv            （小盘溢价，做多小盘）
# - value  : net_assets / total_mv_cny （高 B/P，做多价值）
# - momentum: 100d 残差动量 skip 20    （做多动量）
# - volatility: -20d 波动              （低波 anomaly，做多低波）
# - liquidity: -log(turnover_rate)     （流动性溢价，做多低流动性）
STYLE_SIGNAL_DIRECTIONS: dict[str, str] = {
    "size": "small_cap (long small)",
    "value": "high_book_to_market (long value)",
    "momentum": "100d_winner (long momentum)",
    "volatility": "low_volatility (long low-vol)",
    "liquidity": "illiquid (long low-turnover)",
}

TRADING_DAYS = 244


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StyleAttributionConfig(_StrictModel):
    version: int = 1
    name: str = "value_style_attribution_v1"
    project_config: Path = Path("configs/project.yaml")
    data_version: str = "latest"
    fundamentals_path: Path
    predictions_path: Path
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    gross_cost_bps: float = Field(default=0.0, ge=0)
    net_cost_bps: float = Field(default=20.0, ge=0)
    style_factors: list[str] = Field(default_factory=lambda: ["size", "value", "momentum", "volatility", "liquidity"])
    factor_universe: Literal["liquid", "tradeable"] = "liquid"
    nw_lag: int = Field(default=15, ge=0)
    bootstrap_blocks: int = Field(default=1000, ge=1)
    bootstrap_seed: int = 71
    by_year: bool = True
    output_root: Path = Path("artifacts/value_style_attribution")


def load_style_attribution_config(path: str | Path) -> StyleAttributionConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return StyleAttributionConfig.model_validate(yaml.safe_load(handle) or {})


class StyleAttributionRunner:
    """风格归因诊断 runner，骨架照搬 ``ValueDiagnosticsRunner``。"""

    def run(self, config_path: str | Path) -> dict:
        config_path = Path(config_path)
        cfg = load_style_attribution_config(config_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
        output = cfg.output_root / f"{cfg.name}_{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_bytes(config_path.read_bytes())
        logger = self._run_logger(output, run_id)
        started = time.perf_counter()
        state = {"stage": "initializing"}

        def stage(name: str, detail: str = "") -> None:
            state["stage"] = name
            logger.info("progress stage=%s detail=%s elapsed_seconds=%.1f", name, detail, time.perf_counter() - started)

        try:
            result = self._execute(cfg, output, logger, stage)
            logger.info("run_completed elapsed_seconds=%.1f", time.perf_counter() - started)
            return result
        except Exception as exc:
            logger.exception("run_failed stage=%s", state["stage"])
            (output / "error.json").write_text(json.dumps({
                "status": "FAILED", "stage": state["stage"],
                "error_type": type(exc).__name__, "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise

    def _execute(self, cfg: StyleAttributionConfig, output: Path, logger: logging.Logger, stage) -> dict:
        stage("load_data")
        project = load_project(cfg.project_config)
        repository = DataVersionRepository(project.paths.data_root, project.paths.metadata_db)
        version, panel = repository.load_panel(cfg.data_version)
        panel = panel.copy()
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])

        stage("attach_fundamentals")
        fundamentals = pd.read_parquet(cfg.fundamentals_path)
        working = attach_point_in_time_fundamentals(panel, fundamentals)

        stage("load_predictions")
        predictions = pd.read_parquet(cfg.predictions_path)
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        if "prediction_blend" not in predictions.columns:
            raise ValueError("predictions.parquet must contain prediction_blend")

        # 只在测试期 + 回看窗口内算风格信号，避免对整段 10 年面板做滚动。
        test_start = predictions["trade_date"].min()
        lookback_floor = test_start - pd.Timedelta(days=365)
        signal_panel = working.loc[working["trade_date"] >= lookback_floor].copy()

        stage("build_style_signals")
        universe_col = f"is_{cfg.factor_universe}"
        if universe_col not in signal_panel.columns:
            raise ValueError(f"panel has no {universe_col} column for factor_universe={cfg.factor_universe}")
        signals = self._build_style_signals(signal_panel, cfg.style_factors, universe_col)
        signals_wide = signals.drop(columns=["adj_open"]).copy()
        signals_wide.to_parquet(output / "style_signals.parquet", index=False)

        stage("build_factor_returns")
        factor_returns = self._build_style_factor_returns(signals, cfg.style_factors, universe_col)
        factor_returns.to_csv(output / "style_factor_returns.csv", index=False, encoding="utf-8-sig")
        factor_wide = factor_returns.pivot(index="trade_date", columns="factor", values="factor_return").sort_index()

        stage("factor_correlation")
        corr = factor_wide.corr()
        corr.to_csv(output / "factor_return_correlation.csv", encoding="utf-8-sig")

        # 双成本回归：gross(0) 与 net(net_cost_bps)
        results_by_cost: dict[str, dict] = {}
        yearly_rows: list[dict] = []
        tilt_rows: list[dict] = []
        for label, cost_bps in [("gross", cfg.gross_cost_bps), ("net", cfg.net_cost_bps)]:
            stage("cell_backtest", f"cost={label}")
            backtest = self._run_cell_backtest(panel, predictions, cfg.portfolio, cost_bps)
            daily = backtest.daily[["trade_date", "return", "benchmark_return", "excess_return"]].copy()
            daily["trade_date"] = pd.to_datetime(daily["trade_date"])
            if label == "net":
                backtest.daily.to_parquet(output / "portfolio_daily.parquet", index=False)

            stage("holdings_tilt", f"cost={label}")
            tilt = self._intended_topn_tilt(predictions, signals_wide, cfg.portfolio.top_n, universe_col)
            tilt["cost"] = label
            tilt_rows.append(tilt)
            if label == "net":
                tilt.to_csv(output / "holdings_style_exposure.csv", index=False, encoding="utf-8-sig")

            stage("regression", f"cost={label}")
            aligned = daily.merge(factor_wide, on="trade_date", how="inner").dropna(subset=["excess_return"])
            full = self._fit_regression(aligned, cfg.style_factors, cfg.nw_lag, cfg.bootstrap_blocks, cfg.bootstrap_seed)
            results_by_cost[label] = {"full_sample": full, "n_obs": int(len(aligned))}

            if cfg.by_year:
                aligned["year"] = aligned["trade_date"].dt.year
                for year, year_frame in aligned.groupby("year"):
                    if len(year_frame) <= len(cfg.style_factors) + 5:
                        continue
                    reg = self._fit_regression(year_frame, cfg.style_factors, cfg.nw_lag, cfg.bootstrap_blocks, cfg.bootstrap_seed)
                    results_by_cost[label][f"year_{year}"] = reg
                    explained = sum(float(reg["betas"][f]) * float(year_frame[f].mean()) for f in cfg.style_factors) * TRADING_DAYS
                    yearly_rows.append({
                        "cost": label, "year": int(year), "n_obs": int(len(year_frame)),
                        "realized_excess_annual": float(year_frame["excess_return"].mean() * TRADING_DAYS),
                        "explained_annual": float(explained),
                        "residual_alpha_annual": float(reg["alpha_annual"]),
                        "r_squared": float(reg["r_squared"]),
                    })

        if yearly_rows:
            pd.DataFrame(yearly_rows).to_csv(output / "yearly_decomposition.csv", index=False, encoding="utf-8-sig")

        stage("regression_summary")
        summary_rows = []
        for label, regs in results_by_cost.items():
            for window, reg in regs.items():
                if window == "n_obs":
                    continue
                row = {"cost": label, "window": window, "n_obs": int(reg["n_obs"]), "r_squared": reg["r_squared"],
                       "alpha_annual": reg["alpha_annual"], "alpha_t_nw": reg["alpha_t_nw"], "alpha_p_nw": reg["alpha_p_nw"],
                       "alpha_bootstrap_ci_low": reg["alpha_bootstrap_ci_low"], "alpha_bootstrap_ci_high": reg["alpha_bootstrap_ci_high"]}
                for f in cfg.style_factors:
                    row[f"beta_{f}"] = reg["betas"].get(f)
                    row[f"t_{f}"] = reg["tvalues"].get(f)
                summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output / "regression_summary.csv", index=False, encoding="utf-8-sig")

        verdict = self._classify_verdict(results_by_cost.get("net", results_by_cost.get("gross", {})), corr, cfg.style_factors)
        summary = {
            "data_version": version, "predictions_path": str(cfg.predictions_path),
            "portfolio": cfg.portfolio.model_dump(), "style_factors": cfg.style_factors,
            "verdict": verdict,
            "regression": {label: {k: v for k, v in regs.items() if k != "n_obs"} for label, regs in results_by_cost.items()},
            "factor_return_correlation_max_offdiag": self._max_offdiag_corr(corr, cfg.style_factors),
            "holdings_tilt_net": tilt_rows[1].to_dict("records") if len(tilt_rows) > 1 else [],
            "run_dir": str(output),
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (output / "report.md").write_text(self._report(summary_df, corr, tilt_rows, verdict, cfg), encoding="utf-8")
        return summary

    # ---- 核心数学 ----

    @staticmethod
    def _build_style_signals(panel: pd.DataFrame, factors: list[str], universe_col: str) -> pd.DataFrame:
        """按 (trade_date, ts_code) 产出 adj_open + universe + 风格信号列。所有信号只用 ≤ trade_date 的数据。"""
        data = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        stocks = data["ts_code"]
        close = pd.to_numeric(data["adj_close"], errors="coerce").where(lambda x: x > 0)
        log_return = np.log(close).groupby(stocks, sort=False).diff()
        total_mv = pd.to_numeric(data["total_mv_cny"], errors="coerce").where(lambda x: x > 0)
        net_assets = pd.to_numeric(data["net_assets"], errors="coerce")
        turnover = pd.to_numeric(data["turnover_rate"], errors="coerce").where(lambda x: x > 0)

        out = data[["trade_date", "ts_code", "adj_open", universe_col]].copy()
        # size: 做多小盘 → 取负 log_total_mv
        out["size"] = -pd.to_numeric(data["log_total_mv"], errors="coerce")
        # value: 高 B/P
        out["value"] = (net_assets / total_mv).where((total_mv > 0) & (net_assets > 0))
        # momentum: 100d 求和 skip 20
        out["momentum"] = _rolling_by_stock(log_return, stocks, 100, "sum", shift=20)
        # volatility: 做多低波 → 取负 20d 标准差
        out["volatility"] = -_rolling_by_stock(log_return, stocks, 20, "std")
        # liquidity: 做多低流动性 → 取负 log turnover
        out["liquidity"] = -np.log(turnover)
        return out

    @staticmethod
    def _build_style_factor_returns(signals: pd.DataFrame, factors: list[str], universe_col: str) -> pd.DataFrame:
        """镜像 _benchmark_returns 的时序：T 收盘知信号 → T+1 开盘进 → T+2 开盘出，映射到 T+2。

        ``signals`` 必须含 trade_date/ts_code/adj_open/universe_col + 各风格信号列。
        factor_return = top_tercile_equal_weight_fwd - bottom_tercile_equal_weight_fwd
        """
        merged = signals.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        grp = merged.groupby("ts_code", sort=False)["adj_open"]
        # 进入 T+1 开盘，退出 T+2 开盘
        merged["fwd_ret"] = grp.shift(-2) / grp.shift(-1) - 1
        merged = merged.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

        rows: list[dict] = []
        for factor in factors:
            frame = merged.dropna(subset=[factor, "fwd_ret"]).copy()
            elig = frame[universe_col].fillna(False).astype(bool)
            frame = frame.loc[elig].copy()
            if frame.empty:
                continue
            frame["rank"] = frame.groupby("trade_date")[factor].rank(pct=True)
            top = frame.loc[frame["rank"] >= 2.0 / 3.0]
            bot = frame.loc[frame["rank"] <= 1.0 / 3.0]
            top_mean = top.groupby("trade_date")["fwd_ret"].mean()
            bot_mean = bot.groupby("trade_date")["fwd_ret"].mean()
            diff = (top_mean - bot_mean).rename(factor).reset_index().rename(columns={factor: "factor_return"})
            diff["factor"] = factor
            rows.append(diff)
        if not rows:
            return pd.DataFrame(columns=["trade_date", "factor", "factor_return"])
        fwd = pd.concat(rows, ignore_index=True)

        # 信号日 T → 实现日 T+2（沿交易日后移两个交易日）
        all_dates = pd.Index(merged["trade_date"].drop_duplicates().sort_values())
        pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
        fwd["realized_date"] = fwd["trade_date"].map(
            lambda d: all_dates[int(pos.loc[d]) + 2] if pd.notna(d) and int(pos.loc[d]) + 2 < len(all_dates) else pd.NaT
        )
        fwd = fwd.dropna(subset=["realized_date"])
        return fwd[["realized_date", "factor", "factor_return"]].rename(columns={"realized_date": "trade_date"})

    @staticmethod
    def _fit_regression(aligned: pd.DataFrame, factors: list[str], nw_lag: int, blocks: int, seed: int) -> dict:
        """OLS: excess_return ~ const + Σ factor。Newey-West HAC + 残差 alpha 的 block bootstrap。"""
        try:
            import statsmodels.api as sm
        except ImportError as exc:
            raise RuntimeError("statsmodels is required for style attribution. Install with: python -m pip install statsmodels") from exc
        y = aligned["excess_return"].to_numpy(dtype=float)
        X = sm.add_constant(aligned[factors].to_numpy(dtype=float))
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag})
        betas = dict(zip(factors, model.params[1:].tolist()))
        tvalues = dict(zip(factors, model.tvalues[1:].tolist()))
        pvalues = dict(zip(factors, model.pvalues[1:].tolist()))
        alpha_daily = float(model.params[0])
        alpha_t = float(model.tvalues[0])
        alpha_p = float(model.pvalues[0])

        ci_low, ci_high = StyleAttributionRunner._block_bootstrap_alpha(X, y, blocks, max(1, nw_lag), seed)
        return {
            "n_obs": int(len(y)), "r_squared": float(model.rsquared),
            "alpha_daily": alpha_daily, "alpha_annual": alpha_daily * TRADING_DAYS,
            "alpha_t_nw": alpha_t, "alpha_p_nw": alpha_p,
            "alpha_bootstrap_ci_low": ci_low * TRADING_DAYS, "alpha_bootstrap_ci_high": ci_high * TRADING_DAYS,
            "betas": betas, "tvalues": tvalues, "pvalues": pvalues,
        }

    @staticmethod
    def _block_bootstrap_alpha(X: np.ndarray, y: np.ndarray, blocks: int, block_len: int, seed: int) -> tuple[float, float]:
        """移动 block bootstrap 估计截距（日频 alpha）的分位置信区间，保留自相关。"""
        rng = np.random.default_rng(seed)
        n = len(y)
        if n <= block_len + 1:
            return (np.nan, np.nan)
        n_blocks = max(n // block_len, 1)
        starts = rng.integers(0, n - block_len, size=(blocks, n_blocks))
        alphas = np.empty(blocks)
        for b in range(blocks):
            idx = (starts[b][:, None] + np.arange(block_len)[None, :]).ravel()
            try:
                import statsmodels.api as sm
                fit = sm.OLS(y[idx], X[idx]).fit()
                alphas[b] = fit.params[0]
            except Exception:
                alphas[b] = np.nan
        alphas = alphas[np.isfinite(alphas)]
        if len(alphas) < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5)))

    @staticmethod
    def _run_cell_backtest(panel: pd.DataFrame, predictions: pd.DataFrame, portfolio: PortfolioConfig, cost_bps: float):
        test_panel = panel.loc[panel["trade_date"].between(predictions["trade_date"].min(), predictions["trade_date"].max())].copy()
        signals = predictions[["trade_date", "ts_code", "prediction_blend"]].rename(columns={"prediction_blend": "factor_value"})
        return BacktestEngine().run(
            test_panel, signals,
            universe=portfolio.universe, top_n=portfolio.top_n, holding_days=portfolio.holding_days,
            initial_cash=portfolio.initial_cash, lot_size=portfolio.lot_size,
            constraints=ExecutionConstraints(), cost_model=CostModel(),
            cost_scenario_bps=cost_bps,
        )

    @staticmethod
    def _intended_topn_tilt(predictions: pd.DataFrame, signals_wide: pd.DataFrame, top_n: int, universe_col: str) -> pd.DataFrame:
        """模型目标 top_n 选股的平均风格暴露 vs universe 平均，作为回归 beta 的符号交叉验证。"""
        factors = [c for c in signals_wide.columns if c not in {"trade_date", "ts_code", universe_col, "adj_open"}]
        m = predictions[["trade_date", "ts_code", "prediction_blend"]].merge(
            signals_wide[["trade_date", "ts_code", universe_col, *factors]],
            on=["trade_date", "ts_code"], how="inner",
        )
        elig = m[universe_col].fillna(False).astype(bool) & m["prediction_blend"].notna()
        m = m.loc[elig].copy()
        rows: list[dict] = []
        for f in factors:
            valid = m.dropna(subset=[f])
            if valid.empty:
                continue
            valid["is_top"] = valid.groupby("trade_date")["prediction_blend"].rank(
                method="first", ascending=False
            ) <= top_n
            top_mean = valid.loc[valid["is_top"], f].mean()
            universe_mean = valid[f].mean()
            rows.append({
                "factor": f, "top_n_mean": float(top_mean), "universe_mean": float(universe_mean),
                "tilt": float(top_mean - universe_mean),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _max_offdiag_corr(corr: pd.DataFrame, factors: list[str]) -> float:
        if corr.empty:
            return float("nan")
        mask = ~np.eye(len(corr), dtype=bool)
        return float(np.abs(corr.to_numpy()[mask]).max()) if mask.any() else float("nan")

    @staticmethod
    def _classify_verdict(regs: dict, corr: pd.DataFrame, factors: list[str]) -> dict:
        """根据 net（若无则 gross）的全样本 + 分年度残差 alpha，给出四类判决之一。"""
        full = regs.get("full_sample")
        if not full:
            return {"class": "UNKNOWN", "reason": "no full-sample regression available"}
        alpha_annual = full["alpha_annual"]
        alpha_t = full["alpha_t_nw"]
        ci_low = full["alpha_bootstrap_ci_low"]
        ci_high = full["alpha_bootstrap_ci_high"]
        nw_signif = abs(alpha_t) >= 2.0
        boot_signif = np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0 or ci_high < 0)
        alpha_signif = nw_signif and boot_signif

        year_alphas = [v["alpha_annual"] for k, v in regs.items() if k.startswith("year_") and np.isfinite(v["alpha_annual"])]
        positive_years = sum(1 for a in year_alphas if a > 0)
        stable = len(year_alphas) >= 2 and positive_years >= max(2, len(year_alphas) - 1) and min(year_alphas) > -0.02

        dominant_beta = None
        if full["betas"]:
            ranked = sorted(full["betas"].items(), key=lambda kv: abs(kv[1]), reverse=True)
            dominant_beta = {"factor": ranked[0][0], "beta": ranked[0][1]}
        big_beta = any(abs(b) >= 0.3 for b in full["betas"].values())

        if alpha_signif and stable:
            verdict_class = "REAL_ALPHA"
            reason = "残差 alpha 跨年稳定且显著；问题在执行/换手，不在选股"
        elif alpha_signif and not stable:
            verdict_class = "REGIME_SPECIFIC"
            reason = "残差 alpha 显著但不跨年稳定；集中在特定 regime（如 2025）"
        elif not alpha_signif and big_beta:
            verdict_class = "STYLE_BETA_DISGUISED"
            reason = "残差 alpha 不显著且风格 beta 大；模型本质是风格 beta 伪装"
        else:
            verdict_class = "MIXED"
            reason = "残差 alpha 与风格 beta 均不突出或并存；建议做特征分流"

        return {
            "class": verdict_class, "reason": reason,
            "alpha_annual": alpha_annual, "alpha_t_nw": alpha_t, "alpha_p_nw": full["alpha_p_nw"],
            "alpha_bootstrap_ci": [ci_low, ci_high],
            "nw_signif": bool(nw_signif), "bootstrap_signif": bool(boot_signif),
            "year_alphas": year_alphas, "positive_year_ratio": (positive_years / len(year_alphas)) if year_alphas else None,
            "dominant_beta": dominant_beta,
        }

    @staticmethod
    def _report(summary_df: pd.DataFrame, corr: pd.DataFrame, tilt_rows: list[pd.DataFrame], verdict: dict, cfg: StyleAttributionConfig) -> str:
        lines: list[str] = []
        lines.append("# 价值策略风格归因")
        lines.append("")
        lines.append("## 判决")
        lines.append("")
        lines.append(f"- **{verdict.get('class', 'UNKNOWN')}**：{verdict.get('reason', '')}")
        if np.isfinite(verdict.get("alpha_annual", float("nan"))):
            ci = verdict.get("alpha_bootstrap_ci", [float("nan"), float("nan")])
            lines.append(f"- 残差 alpha 年化：{verdict['alpha_annual']:.2%}（NW t={verdict['alpha_t_nw']:.2f}，bootstrap 95% CI [{ci[0]:.2%}, {ci[1]:.2%}]）")
            lines.append(f"- 分年度 alpha：{[f'{a:.2%}' for a in verdict.get('year_alphas', [])]}（正年份占比 {verdict.get('positive_year_ratio')})")
        dom = verdict.get("dominant_beta")
        if dom:
            lines.append(f"- 主导风格 beta：{dom['factor']} = {dom['beta']:.3f}")
        lines.append("")
        lines.append("## 回归明细（全样本 + 分年度，gross vs net）")
        lines.append("")
        lines.append(summary_df.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append("## 风格因子收益相关矩阵（|ρ|>0.6 提示共线性）")
        lines.append("")
        lines.append(corr.to_markdown(floatfmt=".3f"))
        lines.append("")
        if tilt_rows:
            tilt = pd.concat(tilt_rows, ignore_index=True)
            lines.append("## 目标 top_n 选股的风格暴露 vs universe（回归 beta 的符号交叉验证）")
            lines.append("")
            lines.append(tilt.to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
        lines.append("## 方向约定")
        lines.append("")
        for f, d in STYLE_SIGNAL_DIRECTIONS.items():
            if f in cfg.style_factors:
                lines.append(f"- **{f}**：{d}")
        lines.append("")
        lines.append("时序镜像 `_benchmark_returns`：T 收盘知信号 → T+1 开盘进 → T+2 开盘出，映射到 T+2。")
        return "\n".join(lines)

    @staticmethod
    def _run_logger(output: Path, run_id: str) -> logging.Logger:
        logger = logging.getLogger(f"factor_forge.value_style_attribution.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.handlers[:] = [file_handler, stream_handler]
        return logger
