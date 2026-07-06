# Codex 交接说明：稳定成交基线后的无量上涨因子研究与 Qlib/LightGBM 实验

## 0. 交接目标

本交接用于让 Codex 在现有量化研究/回测项目中，实现并验证一个新的量价结构假设：

> 个股过去 28 个交易日的成交状态较稳定；最近 2 个交易日价格相对行业上涨，但成交量没有明显放大，说明上涨没有触发大量卖盘释放，可能存在“卖方供给收缩、惜售或筹码锁定”结构，并对后续收益具有预测能力。

本任务的目标不是证明“机构提前知道消息”或“存在消息泄露”，而是验证可观察的量价结构是否具有样本外 Alpha。

研究必须保持可解释性。不得把价格、流动性、市值、成交稳定度、无量上涨、K 线质量、Regime 等全部直接乘成一个最终权重后只看回测结果。

---

# 1. 项目背景

项目现有研究方向是“低量上涨/供给收缩”类因子，核心变量包括：

- `scarcity`
- `scarcity_slope_5`
- `scarcity_days_ratio_5`
- `turnover_zscore_60`
- 波动率、市值、流动性等控制变量

已有统计结果如下：

| 因子 | 中性化控制变量 | RankIC | ICIR | Newey t |
|---|---|---:|---:|---:|
| scarcity | 原始 | +0.0119 | 2.79 | +4.32 |
| scarcity | + volatility_20 | +0.0092 | 2.26 | +3.66 |
| scarcity | + size | +0.0099 | 2.70 | +4.64 |
| scarcity | + liquidity | +0.0083 | 2.28 | +3.92 |
| scarcity | + turnover_zscore_60 | +0.0091 | 2.83 | +4.74 |
| scarcity_slope_5 | 原始 | +0.0058 | 1.57 | +3.37 |
| scarcity_slope_5 | + vol20 | +0.0061 | 1.69 | +3.58 |
| scarcity_slope_5 | + vol20+size | +0.0058 | 1.70 | +3.63 |
| scarcity_slope_5 | + vol20+size+liq | +0.0055 | 1.64 | +3.48 |
| scarcity_slope_5 | + 全部 | +0.0054 | 1.69 | +3.53 |
| scarcity_days_ratio_5 | 原始 | +0.0100 | 2.79 | +4.05 |
| scarcity_days_ratio_5 | + vol20+size | +0.0092 | 2.89 | +4.39 |
| scarcity_days_ratio_5 | + vol20+size+liq | +0.0063 | 2.03 | +3.14 |
| scarcity_days_ratio_5 | + 全部（含 turnover_z） | +0.0039 | 1.36 | +2.15 |

当前判断：

1. `scarcity` 是主要有效信号，属于弱但较稳定的 Alpha，适合作为组合模型核心特征。
2. `scarcity_slope_5` 单因子强度较弱，但控制后相对稳定，可能具有少量正交信息。
3. `scarcity_days_ratio_5` 很大一部分信息与流动性和持续低换手重合，独立增量较弱。
4. 下一步不应只继续堆叠因子，而应验证一个更具体的结构：
   - 前期成交稳定；
   - 最近价格上涨；
   - 上涨未触发成交放大。

项目计划引入 Qlib 和 LightGBM，但 Qlib 只负责数据集切分、训练、预测和实验管理，现有回测引擎仍负责正式的 TopN、固定持有期、成本和可交易性回测。

---

# 2. 当前问题

需要回答以下研究问题：

1. 在最近 2 日价格上涨的股票中，“未放量”是否比“明显放量”具有更高的未来收益？
2. 前 28 日成交波动率越低，是否会增强“上涨未放量”信号？
3. 成交稳定度与未来收益的关系是单调、倒 U、U 形，还是没有增量作用？
4. 极低成交波动是否只是“长期无人交易”造成的假稳定？
5. 结果是否由低价股、最小跳价噪声、小市值、低流动性、高波动、普通动量、行业或市场 Regime 驱动？
6. 新结构相对于现有 `scarcity` 系列是否提供独立的样本外增量？
7. 将新结构加入 LightGBM 后，能否稳定改善 OOS RankIC、TopN 扣费收益、最大回撤、年度稳定性和收益集中度？

---

# 3. 已确定的设计和结论

## 3.1 核心经济假设

研究对象不是普通 Alpha95，也不是简单的“成交量越稳定越好”。

核心假设是：

> 前 28 日存在稳定的成交基线，最近 2 日价格明显上移，但成交没有相对基线显著放大，表示价格上涨未触发明显卖盘释放。

核心结构由三个独立部分组成：

1. `baseline_volume_stability_28`
2. `price_strength_2`
3. `recent_no_volume_activation_2`

必须分别保存、分别检验，不得只保存最终组合值。

## 3.2 时间窗口

以信号日为 `t`：

- 历史成交基准窗口：`t-29 ~ t-2`，共 28 个交易日。
- 最近事件窗口：`t-1 ~ t`，共 2 个交易日。
- 信号在 `t` 日收盘后生成，最早在 `t+1` 开盘交易。
- 验证持有期：`1、3、5、10、15 日`。

最近两日不得进入前 28 日稳定度、均值或历史回归参数的计算。

## 3.3 基础变量

优先使用换手率而不是原始成交量：

\[
x_i=\log(1+turnover\_rate_i)
\]

原因：

- 原始成交量受总股本和流通股本影响；
- 换手率更适合跨股票比较；
- 对数变换降低极端放量日影响。

收益率使用统一复权价格计算。真实价格水平、最小跳价和涨跌停判断必须使用未复权价格。

## 3.4 前 28 日成交波动率与成交水平

\[
baseline\_turnover\_vol_{28,t}
=
Std(x_{t-29:t-2})
\]

\[
baseline\_turnover\_mean_{28,t}
=
Mean(x_{t-29:t-2})
\]

\[
baseline\_amount\_mean_{28,t}
=
Mean(amount_{t-29:t-2})
\]

必须同时保留“波动程度”和“成交水平”，因为低波动可能只是长期无成交。

## 3.5 成交波动率转换为 0~1

每天对有效股票做横截面百分位排名：

\[
turnover\_vol\_rank_{28,t}
=
PercentileRank_{cross\ section}
(baseline\_turnover\_vol_{28,t})
\]

取值：

- 接近 0：成交波动较低；
- 接近 1：成交波动较高。

便于解释的稳定度：

\[
turnover\_stability_{28,t}
=
1-turnover\_vol\_rank_{28,t}
\]

但在 LightGBM 中优先输入原始方向的 `turnover_vol_rank_28`，不要提前强制假设“越稳定越好”。

可选增加个股自身历史分位数：

\[
turnover\_vol\_self\_rank_{252,t}
=
PercentileRank
\left(
baseline\_turnover\_vol_{28,t},
baseline\_turnover\_vol_{28,t-251:t}
\right)
\]

## 3.6 最近两日价格上涨强度

\[
excess\_ret_{2,t}
=
ret_{stock,2,t}
-
ret_{industry,2,t}
\]

\[
price\_strength_{2,t}
=
\frac{
excess\_ret_{2,t}
}{
volatility_{20,t-2}\sqrt{2}+\epsilon
}
\]

要求：

- 波动率窗口截止到 `t-2`；
- 避免最近两日上涨污染波动率基准；
- `epsilon` 建议为 `1e-6`。

低价股诊断字段：

\[
effective\_ticks_{2,t}
=
\frac{
raw\_close_t-raw\_close_{t-2}
}{
tick\_size_t
}
\]

## 3.7 最近两日是否放量

前 28 日基准：

\[
baseline\_mean_{28}
=
Mean(x_{t-29:t-2})
\]

\[
baseline\_std_{28}
=
Std(x_{t-29:t-2})
\]

最近两天分别计算：

\[
z_{t-1}
=
\frac{
x_{t-1}-baseline\_mean_{28}
}{
\max(baseline\_std_{28},std\_floor)
}
\]

\[
z_t
=
\frac{
x_t-baseline\_mean_{28}
}{
\max(baseline\_std_{28},std\_floor)
}
\]

生成：

\[
recent\_volume\_z\_mean_2
=
\frac{z_{t-1}+z_t}{2}
\]

\[
recent\_volume\_z\_max_2
=
\max(z_{t-1},z_t)
\]

核心假设更接近“两天都没有明显放量”，所以 `recent_volume_z_max_2` 优先级高于两日均值。

`std_floor` 推荐实现两种配置化方案：

1. 当日横截面 `baseline_std_28` 的 10% 分位数；
2. 仅使用训练期估计的固定下限。

不得使用全样本分位数。

## 3.8 最近两日放量数据的极值处理

必须保留原始值和截断值：

```text
recent_volume_z_mean_2_raw
recent_volume_z_mean_2_clip
recent_volume_z_max_2_raw
recent_volume_z_max_2_clip
```

主口径：

\[
z^{clip}=clip(z,-3,3)
\]

`[-4,4]` 可作为敏感性测试。

## 3.9 0~1 未放量得分

人工组合实验可生成：

\[
no\_volume\_activation\_score_2
=
\exp
\left[
-\lambda
\max(recent\_volume\_z\_max_2^{clip},0)
\right]
\]

建议测试：

\[
\lambda \in \{0.5,1.0\}
\]

该字段只属于人工组合候选，不能替代原始 Z 特征。

## 3.10 可选人工核心组合

只有在独立变量和条件分组验证成立后，才能生成：

\[
stable\_no\_volume\_rise
=
turnover\_stability_{28}
\times
\max(price\_strength_2,0)
\times
no\_volume\_activation\_score_2
\]

必须同时保留三个原始组成变量。

## 3.11 低价、流动性、市值的处理边界

以下字段不是核心 Alpha，只能作为控制变量、样本权重或极端过滤条件：

```text
log_raw_price
tick_return
tick_noise
effective_ticks_2
log_float_market_cap
baseline_amount_mean_28
baseline_turnover_mean_28
amihud_illiquidity_20
volatility_20
turnover_zscore_60
```

\[
tick\_return_t
=
\frac{tick\_size_t}{raw\_close_t}
\]

\[
tick\_noise_t
=
\frac{
tick\_return_t
}{
volatility_{20,t}+\epsilon
}
\]

这些变量不得直接全部乘入核心因子。

## 3.12 LightGBM 样本权重

可选训练样本权重：

\[
price\_weight
=
\frac{1}{1+\lambda_p \cdot tick\_noise}
\]

\[
liquidity\_weight
=
clip
\left(
\frac{
\log(baseline\_amount\_mean_{28})-\log(A_{low})
}{
\log(A_{full})-\log(A_{low})
},
0,1
\right)
\]

\[
sample\_weight
=
clip(
price\_weight \times liquidity\_weight,
0.1,1
)
\]

要求：

- `A_low`、`A_full` 只能由训练期确定；
- 样本权重不能覆盖或修改核心因子值；
- 必须对“有权重”和“无权重”模型做消融对比。

## 3.13 Qlib 的职责边界

Qlib 用于：

- `DatasetH` 或等价数据集封装；
- 时间序列训练/验证/测试切分；
- LightGBM 训练；
- 预测分数输出；
- 实验记录；
- 可选辅助回测。

现有回测引擎用于：

- Top2、Top5、Top10、Top20；
- 1、3、5、10、15 日持有期；
- 10bps、20bps 成本；
- `t` 收盘信号、`t+1` 开盘成交；
- ST、停牌、涨跌停、上市天数、无法成交等交易约束；
- 正式绩效验收。

Qlib 不得直接替换现有正式回测口径。

---

# 4. 不允许改变的约束

## 4.1 时间与未来数据约束

1. 前 28 日基准窗口固定为 `t-29 ~ t-2`。
2. 最近事件窗口固定为 `t-1 ~ t`。
3. 最近两日不得进入前 28 日稳定度和均值计算。
4. 所有特征只能使用 `t` 及之前数据。
5. 信号在 `t` 收盘后生成，最早 `t+1` 开盘执行。
6. 所有标准化、分位数、阈值、模型参数只能由训练期或当日横截面确定。
7. 禁止使用全样本均值、标准差、分位数或回归参数。
8. 禁止随机打乱股票日样本进行训练/验证/测试切分。
9. 行业分类必须使用当时可获得的历史映射，禁止未来行业分类回填。

## 4.2 因子解释约束

1. 核心 Alpha 只能表达：前期成交稳定、最近价格上涨、最近成交未被激活。
2. 低价、流动性、市值、波动率、K 线质量、Regime 不得全部乘入核心因子。
3. 必须保留所有核心原始特征，不能只输出最终组合分数。
4. 控制变量必须和核心变量分层存储。
5. 人工组合因子只能作为附加字段和消融实验，不得替代原始特征。
6. LightGBM 必须通过模型消融证明核心结构的增量，不允许只报告最终模型总收益。

## 4.3 数据与执行约束

1. 收益率使用统一复权价格。
2. 真实价格、最小跳价、涨跌停使用未复权价格。
3. 原始成交量不是主口径，优先使用换手率。
4. 比值类特征必须防止分母接近 0。
5. 最近两日成交 Z 值必须保留 raw 和 clipped 两个版本。
6. 必须记录每个样本被过滤或降权的原因。
7. 不得用固定“股价低于 5 元全部删除”作为主方案。
8. 流动性硬过滤只允许剔除极端不可交易样本，不得把股票池人为收缩为高流动性大盘股。

---

# 5. 已否决或不允许直接采用的方案

## 5.1 直接使用原始 Alpha95

否决主公式：

\[
STD(AMOUNT,20)
\]

原因：

- 受市值、价格和成交规模影响；
- 不能表达“前 28 日稳定、最近 2 日上涨未放量”；
- 容易重新引入规模和流动性暴露。

可作为对照特征，但不能作为核心实现。

## 5.2 所有变量直接相乘成最终权重

否决示例：

```text
上涨强度
× 成交稳定度
× 未放量得分
× 价格权重
× 流动性权重
× 市值权重
× K线质量
× Regime
```

原因：

- 失去经济解释；
- 无法判断收益来源；
- 偏离“卖盘未激活”假设；
- 不利于消融和 OOS 诊断。

## 5.3 只使用一个最终组合字段训练 LightGBM

否决：

```text
X = [final_weighted_factor]
```

应使用独立特征，并将人工组合字段只作为附加输入。

## 5.4 简单认为成交波动越低越好

尚未验证，不能预设单调方向。必须先判断单调、倒 U、U 形或无效。

## 5.5 用最近两日平均成交掩盖单日放量

仅使用两日均值可能掩盖“一天严重放量、一天极度缩量”。必须同时计算并优先分析 `recent_volume_z_max_2`。

## 5.6 用硬价格阈值处理低价股

否决主方案：

```text
raw_close < 5 元，全部删除
```

主方案使用 `tick_noise`、`effective_ticks_2`、子样本诊断、样本权重，并只对极端异常做硬过滤。

## 5.7 用全样本计算去极值阈值

禁止使用全样本 1%/99% 分位数、全样本标准差下限或全样本归一化参数。

## 5.8 一开始直接加入完整 Alpha158

第一版不允许“核心特征 + 完整 Alpha158”，否则无法判断收益来源并容易过拟合。

## 5.9 让 Qlib 完全接管正式回测

Qlib 默认策略未必等同于现有固定持有期、交易成本和可交易性口径。正式结果必须由现有回测引擎验收。

---

# 6. 涉及的文件、模块和数据结构

Codex 必须先扫描仓库，将以下逻辑模块映射到实际路径，不得假设仓库已有固定文件名。

## 6.1 已有说明文件

已有研究规范文件：

```text
低量上涨供给收缩因子与LightGBM特征计算规范.md
```

Codex 应先读取该文件，并将本交接说明作为增量设计。

## 6.2 建议逻辑模块

```text
data/
  market_data_loader
  industry_mapping_loader
  tradability_loader

features/
  scarcity_features
  stable_volume_baseline_features
  price_microstructure_features
  liquidity_control_features
  market_regime_features

datasets/
  feature_dataset_builder
  qlib_dataset_adapter
  label_builder

models/
  lgbm_control_model
  lgbm_core_model
  lgbm_scarcity_model
  lgbm_full_model

experiments/
  univariate_analysis
  conditional_matrix_analysis
  neutralization_analysis
  walk_forward_training
  topn_backtest

reports/
  stable_no_volume_rise_report
```

如仓库已有同类模块，应在原模块内扩展，不要重复创建平行实现。

## 6.3 输入数据结构

最低需要：

| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | date | 交易日 |
| ts_code | string | 股票代码 |
| raw_open/raw_high/raw_low/raw_close | float | 未复权 OHLC |
| adj_open/adj_close | float | 复权价格 |
| amount | float | 成交额，单位统一 |
| turnover_rate | float | 换手率，单位统一 |
| float_market_cap | float | 流通市值 |
| industry_code | string | 当日行业代码 |
| tick_size | float | 最小报价单位 |
| is_st | bool | 是否 ST |
| is_suspended | bool | 是否停牌 |
| list_days | int | 上市天数 |
| limit_up_price/limit_down_price | float | 当日涨跌停价 |
| tradable_next_open | bool | 次日开盘是否可交易 |
| scarcity | float | 已有因子 |
| scarcity_slope_5 | float | 已有因子 |
| scarcity_days_ratio_5 | float | 已有因子 |
| turnover_zscore_60 | float | 已有控制变量 |
| regime_score | float | 可选市场状态评分 |

字段名称不同的，在适配层统一映射，不要在多个模块重复改名。

## 6.4 新增特征表结构

主键：

```text
(trade_date, ts_code)
```

核心特征：

```text
baseline_turnover_mean_28
baseline_turnover_std_28
baseline_amount_mean_28
turnover_vol_rank_28
turnover_stability_28
turnover_vol_self_rank_252
price_strength_2
excess_ret_2
effective_ticks_2
recent_volume_z_t1_raw
recent_volume_z_t_raw
recent_volume_z_mean_2_raw
recent_volume_z_mean_2_clip
recent_volume_z_max_2_raw
recent_volume_z_max_2_clip
no_volume_activation_score_2_l05
no_volume_activation_score_2_l10
stable_no_volume_rise_l05
stable_no_volume_rise_l10
```

控制特征：

```text
log_raw_price
tick_return
tick_noise
log_float_market_cap
amihud_illiquidity_20
volatility_20
turnover_zscore_60
liquidity_weight
price_weight
sample_weight
```

质量与执行字段：

```text
close_location_2
upper_shadow_ratio_2
tradable_flag
filter_reason
data_quality_flag
```

标签：

```text
future_excess_ret_1
future_excess_ret_3
future_excess_ret_5
future_excess_ret_10
future_excess_ret_15
future_rank_label_5
future_rank_label_10
```

## 6.5 Qlib 预测输出结构

```text
trade_date
ts_code
pred_score
model_name
model_version
target_horizon
train_start_date
train_end_date
feature_version
sample_weight_enabled
```

`pred_score` 交由现有回测引擎注册为组合因子，例如 `qlib_stable_no_volume_rise_score`。

## 6.6 配置结构

所有窗口、截断和参数必须配置化，至少包含：

```yaml
baseline_window: 28
event_window: 2
volatility_window: 20
self_rank_window: 252
z_clip_lower: -3.0
z_clip_upper: 3.0
std_floor_method: cross_section_quantile
std_floor_quantile: 0.10
activation_lambda:
  - 0.5
  - 1.0
holding_periods:
  - 1
  - 3
  - 5
  - 10
  - 15
top_n:
  - 2
  - 5
  - 10
  - 20
cost_bps:
  - 10
  - 20
signal_time: close_t
entry_price: open_t1
```

如现有配置模板不允许新增字段，应放入已有允许结构或模块常量文件，不得破坏现有校验协议。

---

# 7. 需要 Codex 完成的具体任务

## 任务 1：仓库结构检查

1. 扫描当前仓库。
2. 找出行情数据入口、因子计算、行业数据、Regime、中性化、IC 分析、LightGBM/Qlib、TopN 回测、实验配置和报告模块。
3. 输出本交接逻辑模块到真实文件的映射。
4. 未检查已有实现前不得新增重复模块。

交付：`architecture_mapping.md`

## 任务 2：数据口径审计

检查并记录：

- 复权价格口径；
- 换手率和成交额单位；
- 历史行业映射；
- 停牌、ST、涨跌停处理；
- 次日开盘可交易标记；
- 上市天数；
- 最小报价单位；
- 未来数据回填；
- 全样本归一化。

交付：

```text
data_definition_audit.md
data_leakage_check.md
```

发现问题时必须修复或阻断实验，不得静默继续。

## 任务 3：实现新特征

实现第 6.4 节字段。

要求：

- 优先向量化；
- 窗口严格；
- 支持缺失值和最少有效样本；
- 保存 raw 和 clipped；
- 生成字段级公式、窗口、方向和依赖说明。

交付：

```text
feature_schema.md
feature_sample.parquet 或 feature_sample.csv
```

## 任务 4：单元测试和防泄漏测试

至少覆盖：

1. 最近两日未进入基准窗口；
2. 修改 `t+1` 以后数据不会改变 `t` 日特征；
3. `baseline_std_28` 接近 0 时不会除零；
4. 单日极端放量时 `max_z_2` 能识别；
5. 一天放量、一天缩量时均值和最大值符合预期；
6. 低价股 `effective_ticks_2` 和 `tick_noise` 正确；
7. 停牌、ST、一字板和次日不可交易样本正确标记；
8. 横截面 rank 只使用当日有效股票；
9. 训练期阈值不会读取验证期和测试期；
10. 行业收益映射不使用未来行业分类。

交付按仓库规范组织测试文件。

## 任务 5：独立变量统计分析

分别分析：

```text
turnover_vol_rank_28
turnover_stability_28
price_strength_2
recent_volume_z_mean_2_clip
recent_volume_z_max_2_clip
baseline_turnover_mean_28
baseline_amount_mean_28
```

每个变量输出：

- Pearson IC；
- RankIC；
- ICIR；
- Newey-West t；
- 1、3、5、10、15 日衰减；
- 五分位和十分位收益；
- 分年度表现；
- 样本数和覆盖率；
- 行业、价格、市值、流动性暴露。

Newey-West 滞后至少覆盖标签重叠：

- 5 日标签：至少 4；
- 10 日标签：至少 9；
- 15 日标签：至少 14。

交付：

```text
univariate_factor_report.md
univariate_factor_metrics.csv
factor_decay.csv
factor_yearly_ic.csv
```

## 任务 6：二维条件实验

### 6.1 上涨强度 × 最近放量

将 `price_strength_2` 和 `recent_volume_z_max_2_clip` 分别做五分位。

重点报告：

- 在相同上涨强度中，未放量组是否优于放量组；
- 在价格上涨最高 Q5 内，成交 Z 从低到高是否呈收益下降；
- 样本数和年度一致性。

### 6.2 成交稳定度 × 最近放量

在 `price_strength_2` 最高 20% 样本中，做：

```text
turnover_vol_rank_28 × recent_volume_z_max_2_clip
```

判断关系是单调、倒 U、U 形或无规律。

交付：

```text
conditional_matrix_price_volume.csv
conditional_matrix_stability_activation.csv
conditional_matrix_report.md
```

## 任务 7：三维联合验证

比较至少四组：

| 稳定 | 上涨 | 未放量 | 未来收益 |
|---|---|---|---:|
| 否 | 是 | 是 |  |
| 是 | 否 | 是 |  |
| 是 | 是 | 否 |  |
| 是 | 是 | 是 |  |

报告：

- 未来 1/3/5/10/15 日收益；
- 各组样本数；
- 分年度结果；
- 控制前后差异。

交付：

```text
three_way_interaction_report.md
three_way_interaction_metrics.csv
```

## 任务 8：混杂控制和中性化

依次加入：

1. `volatility_20`
2. `log_float_market_cap`
3. 流动性指标
4. `turnover_zscore_60`
5. `log_raw_price`
6. `tick_noise`
7. 行业哑变量
8. 全部控制

要求：

- 每次只增加明确控制；
- 报告 RankIC 保留比例；
- 报告 Newey-West t；
- 报告是否方向反转；
- 不得只报告最终“全部控制”。

子样本：

- 剔除流动性最低 5%；
- 剔除流动性最低 10%；
- 剔除 `tick_noise` 最高 10%；
- 按价格、市值、流动性分组。

交付：

```text
neutralization_report.md
neutralization_metrics.csv
subsample_robustness.csv
```

## 任务 9：人工组合因子实验

仅在任务 5~8 支持核心假设后，测试：

```text
stable_no_volume_rise_l05
stable_no_volume_rise_l10
```

比较：

- 三个原始组件；
- 人工乘积；
- 中性化后的人工乘积；
- 不同价格和流动性子样本。

若关系为倒 U，不得强行使用线性稳定度乘法。

## 任务 10：接入 Qlib 和 LightGBM

建立模型：

### Model A：控制模型

```text
log_raw_price
tick_noise
effective_ticks_2
log_float_market_cap
baseline_amount_mean_28
baseline_turnover_mean_28
amihud_illiquidity_20
volatility_20
turnover_zscore_60
industry_strength
market_breadth
regime_score
```

### Model B：A + 新核心结构

```text
turnover_vol_rank_28
turnover_vol_self_rank_252
price_strength_2
recent_volume_z_mean_2_clip
recent_volume_z_max_2_clip
```

### Model C：B + 现有 scarcity 系列

```text
scarcity
scarcity_slope_5
scarcity_days_ratio_5
```

### Model D：C + 人工组合字段

```text
stable_no_volume_rise_l05
stable_no_volume_rise_l10
```

### Model E：样本权重敏感性

分别训练无 `sample_weight` 和有 `sample_weight` 版本。

必须比较：

- `B - A`：新结构增量；
- `C - B`：现有 scarcity 系列增量；
- `D - C`：人工组合增量；
- 有权重与无权重：低价和低流动性降权效果。

不得只报告最终模型总收益。

## 任务 11：时间切分和滚动训练

禁止随机切分。

采用：

```text
训练 5 年
验证 1 年
测试 1 年
每次向前滚动 1 年
```

数据不足时可缩短训练窗口，但必须保持时间顺序。测试窗口只能用于冻结后的最终评估。

交付：

```text
walk_forward_config.yaml
walk_forward_summary.csv
model_fold_metrics.csv
```

## 任务 12：正式 TopN 回测

Qlib 输出 `pred_score`，交由现有回测引擎执行：

- Top2 / Top5 / Top10 / Top20
- 持有 1 / 3 / 5 / 10 / 15 日
- 成本 10bps / 20bps
- `t` 收盘生成信号，`t+1` 开盘买入

报告：

- 年化收益和超额收益；
- Sharpe、最大回撤、Calmar；
- 换手率和成本拖累；
- 分年度收益；
- 行业、市值、价格和流动性暴露；
- 收益贡献集中度；
- Model A/B/C/D/E 增量。

## 任务 13：最终研究报告

最终报告必须明确回答：

1. 上涨未放量是否有效？
2. 前 28 日成交稳定度是否增强该信号？
3. 稳定度关系是单调还是非线性？
4. 极端稳定是否代表无人交易？
5. 控制价格、最小跳价、流动性、市值后还剩多少？
6. 新结构相对 `scarcity` 是否有独立增量？
7. 人工组合是否优于原始组件？
8. LightGBM 是否真正利用了新结构？
9. 样本权重是否改善 OOS，而不是仅改变股票池？
10. 最终结论属于放弃、辅助特征、条件因子或正式组合模型。

---

# 8. 验收标准

## 8.1 实现验收

1. 所有新增字段可从原始数据完全复算。
2. 窗口严格符合 `t-29 ~ t-2` 和 `t-1 ~ t`。
3. raw、clip、rank、score 字段命名清晰。
4. 每个字段有公式、依赖、方向和缺失处理说明。
5. 所有测试通过。
6. 无未来数据泄漏。
7. 同一输入可重复生成完全一致结果。
8. Qlib 预测输出可直接被现有回测引擎读取。
9. 不破坏现有 scarcity 因子和原有实验流程。

## 8.2 研究验收

至少完成：

- 单变量分析；
- 二维条件矩阵；
- 三维联合验证；
- 中性化；
- 低价和低流动性子样本；
- Model A/B/C/D 消融；
- 有/无 sample weight 对比；
- walk-forward OOS；
- TopN 成本后回测。

缺失任何一个环节，都不能直接宣称因子有效。

## 8.3 有效性判断

### 可进入 LightGBM 核心候选池

满足多数条件：

1. 在相同 `price_strength_2` 下，较低 `recent_volume_z_max_2` 未来收益更高；
2. 多年份和多个 OOS fold 方向一致；
3. 控制 size、liquidity、tick_noise、volatility 后不明显反转；
4. Model B 相对 Model A 有稳定增量；
5. 增量不是少数股票或单一年份贡献；
6. Top10/Top20 至少在部分合理持有期和成本下稳定改善。

### 只能作为辅助特征

- 单因子 IC 弱；
- 条件分组存在局部有效区间；
- LightGBM 消融有小幅稳定增量；
- 人工乘积无稳定优势。

### 应放弃成交稳定度

- 五分组和条件矩阵无规律；
- 极端稳定组收益完全由低流动性驱动；
- 加入稳定度后 Model B 不优于 A；
- OOS 方向频繁反转；
- 控制流动性后效果消失。

### 应放弃整个新假设

- 相同上涨强度下，不放量与放量无差异；
- 不放量上涨未来收益更差；
- OOS 持续反转；
- 结果完全来自无法交易、低价或极低流动性股票。

---

# 9. 尚未确定的问题

以下问题必须通过实验决定，Codex 不得自行预设答案：

1. 成交稳定度是否越高越好？
2. 是否存在中等偏高稳定度最优的倒 U 形？
3. 横截面稳定度和个股自身历史稳定度哪一个更有效？
4. `recent_volume_z_max_2` 是否明显优于 `recent_volume_z_mean_2`？
5. `z_clip` 使用 `[-3,3]` 还是 `[-4,4]` 更稳健？
6. `std_floor` 使用当日横截面 10% 分位数还是训练期固定阈值？
7. `no_volume_activation_score` 的 `lambda=0.5` 还是 `1.0` 更合理？
8. 是否需要最近两日都为正超额收益，还是只看累计两日超额收益？
9. 是否需要加入最近两日收盘位置和上影线作为辅助特征？
10. 新结构是否只在特定行业强度、市场宽度或 Regime 下有效？
11. 人工组合是否比独立原始特征更好？
12. `sample_weight` 是否改善 OOS，还是损失有效小盘样本？
13. 最佳预测标签是未来 5 日还是 10 日行业超额收益？
14. LightGBM 目标应使用原始超额收益、横截面 rank，还是稳健损失？
15. 新结构相对于现有 `scarcity` 的增量是否足够进入正式模型？

---

# 10. Codex 执行原则

1. 先检查仓库，再修改代码。
2. 优先复用现有模块和配置。
3. 每个阶段输出结果和问题，再进入下一阶段。
4. 不得为追求更好回测结果擅自更改窗口、标签、交易时间或股票池。
5. 不得在测试集反复调参。
6. 不得只输出最终收益，必须保留中间统计证据。
7. 所有重要参数必须配置化并记录版本。
8. 所有报告必须包含样本数、覆盖率和异常样本说明。
9. 任何无法确认的项目口径必须在报告中标注，不得猜测。
10. 即使模型无效，也要明确失败发生在哪个环节。

---

# 11. 最终交付清单

```text
architecture_mapping.md
data_definition_audit.md
data_leakage_check.md
feature_schema.md
feature_sample.parquet 或 feature_sample.csv
univariate_factor_report.md
univariate_factor_metrics.csv
factor_decay.csv
factor_yearly_ic.csv
conditional_matrix_price_volume.csv
conditional_matrix_stability_activation.csv
conditional_matrix_report.md
three_way_interaction_report.md
three_way_interaction_metrics.csv
neutralization_report.md
neutralization_metrics.csv
subsample_robustness.csv
manual_composite_report.md
manual_composite_metrics.csv
walk_forward_config.yaml
walk_forward_summary.csv
model_fold_metrics.csv
topn_backtest_report.md
topn_backtest_metrics.csv
yearly_performance.csv
cost_sensitivity.csv
holding_period_comparison.csv
contribution_concentration.csv
stable_no_volume_rise_final_report.md
```

并提交：

- 新增或修改的源代码；
- 单元测试；
- 实验配置；
- 可复现命令；
- 依赖变更；
- 变更说明；
- 已知限制。
