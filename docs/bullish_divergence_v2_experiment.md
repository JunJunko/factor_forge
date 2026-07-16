# 底背离机制对齐实验 v2

v2 是在查看 v1 全历史结果后注册的新假设，因此历史结果只能用于诊断，不能作为未见样本验证或直接部署依据。

## 修正目标

v2 只回答一个问题：

> 在相同日期、相同行业和相似价格损伤/双低点几何结构下，振荡器或下跌速度改善是否提供未来 10 日增量收益？

主要修正：

1. A 必须位于 B 前 20–60 个交易日。
2. A、B 前 3 日收益均为负。
3. 第二低点相对第一低点限制在 `[-0.25, +1.00] ATR`。
4. gap、reliability、lower-low 深度和中间反弹幅度不再获得评分奖励。
5. RSI、MACD/ATR、下跌速度分别做当日截面 rank，再组合。
6. 主对照是“价格几何合格但三个振荡器均未改善”的 placebo。
7. score quintile 在同日事件池内重新分组。
8. Q5-Q1、状态差异只使用同日两组均存在的 complete-case 日期。

## v2 分数

```text
oscillator_strength = max(
    q(rsi14_B-rsi14_A),
    q((macd_hist_B-macd_hist_A)/ATR_A),
    q(downside_velocity_B-downside_velocity_A)
)

confirmation_strength = mean(
    q(close_location_T),
    q(lower_shadow_T/ATR_T),
    q(-(down_volume_B-down_volume_A))
)

div_v2_score = 100 * (
    0.75 * oscillator_strength
    + 0.25 * confirmation_strength
)
```

## 两个触碰时钟

- `support_v2__pre_b_present`：在原始底背离信号 T 判断 B 前是否已有价格聚集，只解释为历史支撑。
- post-signal retest：从原始信号之后开始观察，首次触碰 B 锚定价的当天成为新的事件 T2；T2 收盘冻结，T2+1 开盘成交。

后者不再把 B 前历史触碰解释成“底背离形成后的回踩”。

## 运行

```powershell
python scripts/bullish_divergence_v2_features.py `
  --panel data/versions/<version>/curated/stock_daily_panel.parquet `
  --start-date 2021-01-01 `
  --end-date 2026-07-10
```

```powershell
python scripts/bullish_divergence_v2_event_study.py `
  --panel data/versions/<version>/curated/stock_daily_panel.parquet `
  --features artifacts/bullish_divergence_v2_runs/<run>/stock_day_features.parquet `
  --episodes artifacts/bullish_divergence_v2_runs/<run>/divergence_episodes.parquet `
  --retests artifacts/bullish_divergence_v2_runs/<run>/post_signal_retest_events.parquet
```

