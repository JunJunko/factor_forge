# V2 特征 Schema — 稳定成交基线 + 无量上涨结构

> 对应 `Codex交接说明_稳定成交基线后的无量上涨因子研究.md` sec. 3 / 6.4。
> 实现：时序原语在 [supply_features.py](src/factor_forge/ml/supply_features.py)，横截面/组装在 [supply_dataset.py](src/factor_forge/ml/supply_dataset.py)，参数契约在 [supply_config.py](src/factor_forge/ml/supply_config.py)。
> 三条独立原始腿（基线稳定度 / 2 日价格强度 / 未激活成交 z）必须分别保留，**禁止合并成单一分数**（交接 3.1 / 4.2.3）。人工组合延后到任务 9，且只作附加字段。

## 1. 时间窗口约定（交接 3.2 / 4.1）
- 信号日 `t`：收盘生成信号，最早 `t+1` 开盘成交。
- **基线窗口** `t-29 ~ t-2`（28 根）；**事件窗口** `t-1 ~ t`（2 根）。
- 事件两日**禁止**进入基线 mean/std（实现：`_rolling(..., shift=event_window)`）。
- 波动率用**截止 t-2** 的 20 日（`volatility_prior gap=event_window`），避免事件两日污染（交接 3.6）。
- 所有特征只用 `t` 及之前数据；横截面操作只用当日有效股。

## 2. 分层
| 层 | 位置 | 职责 |
|---|---|---|
| 时序原语 | `supply_features.py` | per-stock 原始值（baseline_window_stat / volatility_prior / price_strength_2 / recent_volume_z / effective_ticks） |
| 横截面 | `supply_dataset.py` | 日级 `turnover_vol_rank_28` / `turnover_stability_28` / `std_floor` |
| 归一化 | `supply_dataset.py` | rank 类字段保留 [0,1]；其余走每日 1%/99% winsor + 横截面 zscore（规范 sec. 3.4） |

## 3. 字段表（`baseline_structure` 组，14 字段）

| 字段 | 公式 | 窗口 | 预期方向 | 依赖面板字段 | 缺失/最少样本 | 归一化 |
|---|---|---|---|---|---|---|
| baseline_turnover_mean_28 | Mean(log1p(turnover/100)) | t-29..t-2 | 控制变量（成交水平） | turnover_rate | min_periods=21（75%×28） | winsor+zscore |
| baseline_turnover_std_28 | Std(log1p(turnover/100), ddof=0) | t-29..t-2 | **原始方向，不预设**（交接 3.5） | turnover_rate | min_periods=21 | winsor+zscore |
| baseline_amount_mean_28 | Mean(amount_cny) | t-29..t-2 | 控制变量（成交额水平） | amount_cny | min_periods=21 | winsor+zscore |
| turnover_vol_rank_28 | PercentileRank_cross(baseline_turnover_std_28) | 当日截面 | 越高=越波动；LightGBM 优先输入（3.5） | baseline_std_28 | 仅当日有效股 | **保留[0,1]，跳过** |
| turnover_stability_28 | 1 − turnover_vol_rank_28 | 当日截面 | 越高=越稳定（解释用） | turnover_vol_rank_28 | 仅当日有效股 | **保留[0,1]，跳过** |
| excess_ret_2 | ret_stock_2 − ret_industry_LOO_2 | 2 根 | 正向（上涨强度） | adj_close, industry_l1_code | 行业内需 ≥2 股 | winsor+zscore |
| price_strength_2 | excess_ret_2 / (vol20_[..t-2]·√2 + ε) | 事件2 / 波动20截止t-2 | 正向 | excess_ret_2, vol20_prior | ε=1e-12 | winsor+zscore |
| recent_volume_z_t1_raw | (x[t-1] − baseline_mean_28) / max(baseline_std_28, std_floor) | 基线28 + 事件1 | 负向（越低越未激活） | turnover_rate, baseline_mean/std, std_floor | std_floor 防除零 | winsor+zscore |
| recent_volume_z_t_raw | 同上，用 x[t] | 同 | 负向 | 同 | 同 | winsor+zscore |
| recent_volume_z_mean_2_raw | (z_t1 + z_t) / 2 | 事件2 | 负向 | z_t1, z_t | — | winsor+zscore |
| recent_volume_z_mean_2_clip | clip(mean_2_raw, −3, 3) | 事件2 | 负向（主口径之一） | mean_2_raw | z_clip=±3 | winsor+zscore |
| recent_volume_z_max_2_raw | max(z_t1, z_t) | 事件2 | 负向，**优先**（3.7/5.5） | z_t1, z_t | — | winsor+zscore |
| recent_volume_z_max_2_clip | clip(max_2_raw, −3, 3) | 事件2 | 负向（主口径） | max_2_raw | z_clip=±3 | winsor+zscore |
| effective_ticks_2 | (raw_close[t] − raw_close[t-2]) / tick_size | 事件2 | 诊断（低价股跳价） | raw_close | tick_size=0.01 | winsor+zscore |

> `x = log1p(turnover_rate/100)`。`std_floor` 见下。

## 4. 关键派生量（dataset 层，非独立特征）
- **std_floor**：当日有效股 `baseline_turnover_std_28` 的 10% 分位（交接 3.7 方案1，用户裁决点4）；防 `baseline_std_28→0` 除零。`train_period_fixed` 方案为 phase-2 选项，当前 `supply_config.std_floor_method="cross_section_quantile"`，其余值抛 `NotImplementedError`。
- **volatility_prior_20**：log_return 的 20 日标准差，`shift=event_window`（截止 t-2），ddof 沿用 `volatility_ddof`（默认 1，规范 4.5 样本标准差）。

## 5. 暂未实现（后续任务）
- `turnover_vol_self_rank_252`（自身历史分位，交接 3.5 可选）。
- `no_volume_activation_score_2_l05/l10`（交接 3.9，任务 9）。
- `stable_no_volume_rise_l05/l10`（交接 3.10，任务 9）。
- `liquidity_weight / price_weight / sample_weight`：v1 已实现，v2 直接复用，不重写。

## 6. 防泄漏要点（详见 [data_leakage_check.md](data_leakage_check.md)）
- 基线/事件窗口严格分离（`_rolling shift` 参数）。
- 横截面 rank/std_floor 只用当日有效股。
- 改 `t+1` 之后数据不影响 `t` 日特征（单测 `test_future_data_does_not_change_prior_features`）。

## 7. 默认参数（交接 §9 待消融项的 phase-1 取值）
| 参数 | 默认 | 来源 |
|---|---|---|
| baseline_window / event_window | 28 / 2 | 交接 3.2 |
| z_clip | [−3, 3]（[−4,4] 为敏感性变体） | 交接 3.8 |
| std_floor_method | cross_section_quantile, q=0.10 | 交接 3.7 方案1（裁决点4） |
| volatility_ddof | 1 | 规范 4.5 |
| 行业基准 | leave-one-out | 裁决点3 |
| 最少有效样本 | 75% × 窗口 | 规范 3.3 |
