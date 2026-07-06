# 仓库架构映射 — 低量上涨 / 供给收缩研究管线

> 对应交接说明任务 1（sec. 6.2 建议逻辑模块 → 真实文件）。
> **v1** = `scarcity` / `volume_residual` 族（已工程闭环，结论为 ML 无增量）；**v2** = 稳定基线 + 无量上涨族（本回合奠基：特征 + 防泄漏测试）。

## 1. 逻辑模块 → 真实文件

| 交接建议模块 | 真实路径 | 状态 |
|---|---|---|
| data/market_data_loader | [panel.py](src/factor_forge/data/panel.py) `PanelBuilder` + [repository.py](src/factor_forge/data/repository.py) `DataVersionRepository` | v1/v2 共用 |
| data/industry_mapping_loader | panel.py `_attach_industry`（point-in-time） | 共用 |
| data/tradability_loader | panel.py `_attach_st` 等（is_tradeable / is_suspended / is_st / is_limit_*_open / listing_trade_days） | 共用 |
| features/scarcity_features | [supply_features.py](src/factor_forge/ml/supply_features.py)（volume_residual / scarcity / scarcity_slope_5 / scarcity_days_ratio_5） | v1 |
| features/stable_volume_baseline_features | supply_features.py（`baseline_window_stat` / `volatility_prior` / `recent_volume_z` 等）+ [supply_dataset.py](src/factor_forge/ml/supply_dataset.py) | **v2 本回合新增** |
| features/price_microstructure_features | supply_features.py（tick_return / tick_noise / effective_ticks） | v1；`effective_ticks_2` 本回合复用 |
| features/liquidity_control_features | supply_features.py（amihud / log_avg_amount / liquidity_weight / sample_weight） | v1 |
| features/market_regime_features | supply_features.py（market_breadth / industry_breadth / market_turnover_z / industry_turnover_z） | v1；外部 `regime_score` 未接入 |
| datasets/feature_dataset_builder | [supply_dataset.py](src/factor_forge/ml/supply_dataset.py) `build_supply_dataset` | v1+v2 共用 |
| datasets/qlib_dataset_adapter | [supply_qlib_bin.py](src/factor_forge/ml/supply_qlib_bin.py) + `supply_dataset.to_qlib_frame` | v1 |
| datasets/label_builder | supply_dataset `_forward_industry_neutral_label` | 共用 |
| models/lgbm_control/core/scarcity/full | [supply_runner.py](src/factor_forge/ml/supply_runner.py) `qlib.contrib.model.gbdt.LGBModel` + `AblationSpec`（Model A/B） | v1；Model C/D/E 待任务 10 |
| experiments/univariate_analysis | [supply_ic.py](src/factor_forge/ml/supply_ic.py) + scripts/supply_ic_look.py / supply_diagnostics.py | v1；**v2 字段未分析（任务 5 待做）** |
| experiments/conditional_matrix_analysis | supply_ic.`quantile_2d_sort` + scripts | v1；v2 字段待任务 6 |
| experiments/neutralization_analysis | scripts/supply_neutral_ic.py | v1；v2 待任务 8 |
| experiments/walk_forward_training | — | **未实现（任务 11）**；breakout 管线有 `breakout_qlib_walkforward.py` 可参考 |
| experiments/topn_backtest | supply_runner `_qlib_backtest` + `_crosscheck`（BacktestEngine） | v1 单配置；TopN 全网格 + 双引擎待任务 12 |
| reports/stable_no_volume_rise_report | [供给收缩因子研究报告.md](供给收缩因子研究报告.md)（v1 最终）+ supply_*.md | v1 完整；v2 待任务 13 |

## 2. 配置与命令
- 配置：[configs/ml/supply_contraction_qlib_v1.yaml](configs/ml/supply_contraction_qlib_v1.yaml)（范本）、`supply_run_top1000.yaml`（实跑主消融）、`supply_run_top1000_minimal.yaml`（精简消融）。
- CLI：`python -m factor_forge.cli ml supply-run <yaml>`。
- 产物：`artifacts/supply_contraction_runs/<name>_<id>/`（predictions / portfolio_daily / report.md / summary.json 含 `incremental_alpha`）。

## 3. 特征组注册（`supply_dataset.FEATURE_GROUP_REGISTRY`）
- `controls`（20）：规模 / 流动性 / 微观 / 波动 / 环境。
- `supply_core`（13）：v1 价格强度 + 条件成交残差族（excess_ret_{1,3,5,10} / risk_adjusted_ret_5 / volume_residual / scarcity / scarcity_*）。
- `composite`（5）：v1 人工组合（simple_low_volume_rise 等）。
- **`baseline_structure`（14）：v2 本回合新增**，三条独立原始腿。

## 4. 回测双引擎现状与 v2 任务 12 口径
- 主：Qlib `TopkDropoutStrategy` + `AShareExchange`（`supply_runner._qlib_backtest`）。
- 辅：项目 `BacktestEngine`（`supply_runner._crosscheck`，单配置 sanity）。
- 对账：[scripts/qlib_vs_be_diagnose.py](scripts/qlib_vs_be_diagnose.py) 验证毛收益差 <7%。
- **v2 任务 12（用户决策）**：双引擎并行全报，TopN 全网格 Top2/5/10/20 × 持有 1/3/5/10/15 日 × 10/20bps + 暴露分析 + Model A–E 增量。

## 5. 横向缺口（v1/v2 均未做）
- 任务 11 walk-forward 滚动（5y/1y/1y，年滚）。
- 任务 12 正式 TopN 网格 + 暴露 + 集中度。
- 任务 8 子样本鲁棒性（剔除低流动 5%/10%、高 tick_noise 10%、按价格/市值/流动性分组）。
- v2 的单变量 / 条件矩阵 / 三维联合 / 人工组合 / Model C-D-E / 最终报告（任务 5–10、13）均阻塞于 v2 字段——本回合已解除该阻塞。
