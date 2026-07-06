# 防泄漏检查（data leakage check）

> 对应交接说明任务 2 / 4。逐条核对 sec. 4.1 与 sec. 19 的强制要求，标注实现位置与单测覆盖。

## 1. 窗口严格（交接 4.1.1–3）
- 基线 [t-29, t-2]（28 根），事件 [t-1, t]（2 根），事件两日不进基线。
- 实现：[supply_features.py](src/factor_forge/ml/supply_features.py) `_rolling(..., shift=event_window)`；`baseline_window_stat` 固定 `shift=event_window`。
- 单测：`test_baseline_window_excludes_event_days`。

## 2. 只用 t 及之前（交接 4.1.4）
- 所有特征 PIT。改 `t+1` 及之后数据不影响 `t` 日特征。
- 单测：`test_future_data_does_not_change_prior_features`（验证 baseline_mean / z_t1 全长度不变，z_t 在 bar ≤38 不变；z_t[t] 读 x[t]，bar 自身变化不是泄漏）。

## 3. 信号时点（交接 4.1.5–6）
- t 收盘生成信号，t+1 开盘成交；标签 `_forward_industry_neutral_label` 从 `adj_open[t+1]` 起算（open_to_open 默认，对齐 BacktestEngine 与 Qlib deal_price）。

## 4. 标准化 / 阈值口径（交接 4.1.6–7）
- 横截面 rank / std_floor / winsor / zscore **当日**；训练期阈值（`liquidity_weight`）仅 train 段。
- 禁全样本分位数。v2 `std_floor` 用当日截面 10% 分位（方案 1）。

## 5. 时间切分（交接 4.1.8 / 4.2.8）
- train < valid < test 严格不重叠有序（[ml/config.py](src/factor_forge/ml/config.py) `Segment` / `Segments`）。禁止随机打乱。
- **walk-forward 滚动未实现**（任务 11）；当前为固定单切分。

## 6. 行业 PIT（交接 4.1.9）
- [panel.py](src/factor_forge/data/panel.py) `_attach_industry` 历史映射；`_industry_loo_mean` 排除自身。
- 单测：`test_industry_loo_mean_excludes_self`（v1）；v2 `excess_ret_2` 复用同机制。

## 7. 除零防护（交接 4.3.4）
- `baseline_std_28 → 0` 时 `std_floor` 兜底（`max(baseline_std, std_floor)`）。
- 单测：`test_recent_volume_z_no_div_by_zero_when_baseline_std_zero`。

## 8. 单日激活识别（交接 5.5）
- `recent_volume_z_max_2` 优先于 `mean_2`，避免一天放量一天缩量被均值掩盖。
- 单测：`test_max_z_catches_single_day_activation` / `test_max_z_beats_mean_when_one_day_spike_one_day_shrink`。

## 9. 横截面 rank 仅当日有效股（交接 4.8）
- `baseline_std.where(valid_mask).groupby(dates).rank(pct=True)`；`std_floor` 同样 `.where(valid_mask)` 后取分位。
- 单测：`test_cross_section_rank_uses_only_valid_stocks`。

## 10. 波动率不被事件日污染（交接 3.6）
- `price_strength_2` 用 vol20 截止 t-2（`volatility_prior gap=2`）。
- 单测：`test_volatility_prior_excludes_event_days`（事件两日大波动不进入 vol 基准）。

## 11. 回测时点匹配（交接 19.10）
- v1：Qlib `deal_price=("$open","$open")` + `AShareExchange`（涨停买不进 / 跌停卖顺延），`BacktestEngine` 交叉核对。
- v2 任务 12：双引擎全网格沿用同口径。

## 12. effective_ticks 单位（交接 4.6）
- `effective_ticks_2 = (raw_close[t] − raw_close[t-2]) / tick_size`，用未复权价。
- 单测：`test_effective_ticks_2_counts_tick_steps`。

## 结论
v2 基础特征的防泄漏不变量已由 9 个单测覆盖（[tests/test_supply_v2_features.py](tests/test_supply_v2_features.py)），叠加 v1 既有的截尾不变性测试（`test_features_have_no_future_leakage`）与端到端 smoke（`test_supply_runner`）。**未发现泄漏**。

后续阶段需补的防泄漏：
- 任务 11 walk-forward：训练期阈值/模型参数冻结，fold 间不回看。
- 任务 12 TopN 网格：测试集一次性评估，不在测试集调参（交接 10.5）。
- 任务 10 Model E（sample_weight）：有权重 vs 无权重对照，确认权重不引入未来（A_low/A_full 已限 train 段）。
