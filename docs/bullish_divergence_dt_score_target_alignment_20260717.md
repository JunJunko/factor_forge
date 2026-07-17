# DT_SCORE 目标对齐 OOF 实验（2026-07-17）

## 结论

本轮没有得到可确认的、可交易的 DT_SCORE 增量 alpha。

- 长样本中，`lgb_lambdarank + Top10` 的 DT_SCORE−DT_BASE 为 **+13.03bp/事件日**，10 日 block-bootstrap 95% CI 为 **[+2.89, +23.76]bp**，20 次日内标签 placebo 的单侧 p=0.0476；但 Top5 为 -2.86bp、Top20 仅 +0.78bp，组合宽度不稳健。
- 同一个 `lgb_lambdarank + Top10` 在 PIT 概念语境样本中反向为 **-13.47bp**，最后一折为约 -62.09bp；长样本命中没有迁移。
- PIT 中 `logit_top_decile + Top5` 为 **+18.60bp**，4/4 折为正且 placebo p=0.0476；但 bootstrap CI 为 [-4.73, +38.44]bp，长样本仅 +2.23bp、p=0.2381，且 Top10/20 增量迅速衰减。
- 两个 p=0.0476 的命中来自不同模型、不同组合。全实验有 4 个目标 × 7 个组合 × 2 个样本 = 56 个主要比较；Benjamini–Hochberg 校正后两者 q 值均约 0.944，均不显著。
- 排序改善没有稳定转化为绝对收益。长样本 `lgb_lambdarank + DT_SCORE + Top10` 的 10 日行业超额收益仍为 **-20.50bp**，按 40bp 单边成本后为 **-60.50bp**。它只是比 DT_BASE 少亏约 13bp。

因此，本轮结果支持“DT_SCORE 会局部改变尾部排序”，但不支持“当前 DT_SCORE 构造能够稳定提升可交易收益”。

## 实验口径

### OOF 与防泄漏

- 预测标签：事件后 10 日行业超额收益 `label__industry_excess_10d`。
- 每个测试期只使用 `label_available_date < test_start` 的训练样本。
- DT_BASE 使用 X（控制）+ D（背离）+ T（触碰）特征；DT_SCORE 只新增：
  - `structure__double_divergence_present`
  - `structure__double_divergence_trend_score`
  - `structure__triple_history_available`
- 两个特征臂使用完全相同的折次、随机种子与行采样；LightGBM `feature_fraction=1.0`，避免特征数变化改变随机抽列结果。
- placebo 在每个交易日内打乱训练标签，保留日期、横截面规模与标签分布。

### 对齐目标

| objective | 训练目标 |
|---|---|
| `lgb_regression` | 连续 10 日行业超额收益基线 |
| `lgb_lambdarank` | 每日收益排序映射为 0–9 relevance grade，优化 LambdaRank/NDCG |
| `lgb_top_decile` | 每日收益最高 `ceil(10% × N)` 的二分类目标 |
| `logit_top_decile` | 同一 top-decile 标签的正则化逻辑回归交叉检查 |

分类样本权重先令每天总权重相等，再令当天正负类别权重各占 50%。

### 组合与成本

- 固定持仓数：Top5、Top10、Top20。
- 自适应持仓数：每日 Top10%、Top20%。
- 连续权重：rank-weighted long-only、rank-weighted long-short。
- 成本敏感性：20/40/60bp；主报告使用 40bp。long-short 按双边成本处理。
- 每日事件数中位数为 38；约 33% 的长样本交易日不足 20 个事件，因此 Top20 在不少日期接近全选，必须与分位数组合共同解释。

## DT_SCORE 增量结果

下表为 DT_SCORE−DT_BASE 的 `portfolio return − 当日全部事件平均收益`，单位 bp/事件日；括号为 placebo 单侧 p 值。

| 样本 / objective | Top5 | Top10 | Top20 | Top10% | Top20% | Rank-long | Rank-LS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 长样本 regression | +2.08 (.429) | -0.42 (.571) | -1.74 (.857) | -13.71 (.952) | -2.38 (.810) | -0.10 (.667) | -1.47 (.667) |
| 长样本 LambdaRank | -2.86 (.810) | **+13.03 (.048)** | +0.78 (.476) | +3.54 (.381) | +4.72 (.381) | +1.71 (.238) | +6.69 (.286) |
| 长样本 LGB top-decile | -3.03 (.619) | -2.93 (.810) | +1.90 (.190) | -4.79 (.571) | -4.72 (.762) | -0.35 (.571) | -1.68 (.476) |
| 长样本 logit top-decile | +2.23 (.238) | -1.00 (.524) | -0.15 (.524) | +4.70 (.190) | +1.19 (.571) | -0.23 (.667) | -1.10 (.667) |
| PIT regression | +6.98 (.286) | -0.94 (.524) | +0.05 (.571) | +6.34 (.333) | -1.14 (.714) | -0.71 (1.000) | -3.00 (.952) |
| PIT LambdaRank | -15.62 (.905) | **-13.47 (.905)** | -4.73 (1.000) | -22.80 (.810) | -8.15 (.810) | +0.39 (.476) | +1.77 (.476) |
| PIT LGB top-decile | +10.66 (.190) | +1.76 (.476) | -5.38 (.952) | +16.29 (.190) | +1.39 (.619) | +0.53 (.524) | +2.48 (.524) |
| PIT logit top-decile | **+18.60 (.048)** | +2.59 (.238) | +0.10 (.429) | -5.09 (.714) | +1.84 (.476) | +0.14 (.810) | -2.33 (.810) |

## 排序与收益为何没有同步

DT_SCORE 的平均 Rank IC 增量都很小，且 bootstrap 区间覆盖零：

| objective | 长样本 ΔRank IC | PIT ΔRank IC |
|---|---:|---:|
| regression | +0.00108 | -0.00119 |
| LambdaRank | +0.00138 | -0.00133 |
| LGB top-decile | +0.00171 | -0.00093 |
| logit top-decile | +0.00015 | +0.00207 |

这说明两个显著命中更像局部尾部重排，而不是整条排序曲线改善。长样本 LambdaRank Top10 在 5 折中 4 折为正，但 Top5 只有 2 折为正；PIT 的 LambdaRank Top10 在后期失效。PIT logit Top5 具有折次一致性，但没有跨样本和持仓宽度一致性。

绝对表现上，连续回归仍提供更高且更稳定的总体 Rank IC（长样本 DT_SCORE 约 3.34%，PIT 约 2.5%），但在当前成本和组合定义下也未形成净正收益。Top-decile 树模型在长样本的总体 Rank IC 为负，说明其概率输出存在明显非单调/时期不稳；不能仅凭局部 TopN 收益采用。

## 特征使用情况

- 树模型实际主要使用 `structure__double_divergence_trend_score`。
- `structure__double_divergence_present` 有较小的增量使用。
- `structure__triple_history_available` 在三个树目标中的 gain importance 为 0。
- PIT 第一折只有 206 个训练事件；在 `min_child_samples=50` 下，树模型的 DT_BASE/DT_SCORE 基本无法产生结构差异，因此 PIT 树模型主要应按后 3 折解释。

## 判定与下一轮建议

当前假设不应直接进入实盘候选。下一轮应继续检验“结构趋势只在特定状态下有效”，而不是继续增加相近的全局模型：

1. 把 DT_SCORE 拆成方向与强度，分别测试 `present × trend_sign`、趋势绝对值、前后两次背离间隔和价格/振荡器改善比例，避免单一压缩分数掩盖非线性。
2. 用训练折内确定阈值，做两阶段模型：先估计是否进入可恢复状态，再在状态内做收益排序；状态可由市场 regime、概念轮动强度、触碰/收复形态共同定义。
3. 将 TopN 选择规则也纳入训练期选择并锁定到下一折，避免事后从 Top5/10/20 中挑最优；至少使用 nested walk-forward 或独立 holdout。
4. 增加 placebo 至 100–200 次，并把最终候选限定为：跨长/PIT 同号、Top5/10/20 或 10%/20% 至少三项同号、折次多数为正、FDR 后仍显著、40bp 成本后净收益为正。

在上述门槛下，本轮没有候选通过。
