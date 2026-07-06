# 低量上涨供给收缩因子与 LightGBM 特征计算规范

## 1. 研究目标

本研究希望识别以下交易结构：

> 个股价格相对行业或市场明显上涨，但实际成交活跃度低于该股票自身历史正常水平，说明价格上涨可能来自卖方供给收缩、惜售或筹码锁定，而不是单纯依靠成交量放大推动。

需要特别说明：

- 本因子不能证明存在消息泄露或内幕交易。
- 本因子仅用于刻画“较少成交量推动价格上移”的量价结构。
- 所有特征必须只使用信号日及之前的数据。
- 信号在交易日 `t` 收盘后计算，最早只能在 `t+1` 执行交易。

---

# 2. 基础数据要求

至少需要以下日频字段：

| 字段 | 含义 | 口径 |
|---|---|---|
| trade_date | 交易日期 | 自然交易日 |
| ts_code | 股票代码 | 唯一证券标识 |
| raw_open | 未复权开盘价 | 当日真实成交价格 |
| raw_high | 未复权最高价 | 当日真实成交价格 |
| raw_low | 未复权最低价 | 当日真实成交价格 |
| raw_close | 未复权收盘价 | 当日真实成交价格 |
| adj_open | 复权开盘价 | 建议后复权或统一收益率口径 |
| adj_high | 复权最高价 | 与收益率计算一致 |
| adj_low | 复权最低价 | 与收益率计算一致 |
| adj_close | 复权收盘价 | 用于收益率计算 |
| volume | 成交量 | 股或手，需统一单位 |
| amount | 成交额 | 建议统一为人民币元 |
| turnover_rate | 换手率 | 成交量 / 流通股本 |
| float_market_cap | 流通市值 | 当日可流通市值 |
| total_market_cap | 总市值 | 当日总市值 |
| industry_code | 行业代码 | 建议使用申万一级或二级行业 |
| is_st | 是否ST | 当日状态 |
| list_days | 上市天数 | 截至当日 |
| limit_up_price | 涨停价 | 使用当日真实涨停价 |
| limit_down_price | 跌停价 | 使用当日真实跌停价 |
| tick_size | 最小报价单位 | A股通常为0.01元，仍应保留字段 |

建议同时准备：

- 市场指数日收益率；
- 行业指数日收益率；
- 市场上涨股票比例；
- 行业上涨股票比例；
- 市场成交额；
- 行业成交额；
- 已有的连续型 Regime Score。

---

# 3. 通用计算规则

## 3.1 收益率口径

收益率统一使用复权价格计算：

\[
ret_{N,t}=\frac{adj\_close_t}{adj\_close_{t-N}}-1
\]

其中：

- `ret_1`：1日收益；
- `ret_3`：3日累计收益；
- `ret_5`：5日累计收益；
- `ret_10`：10日累计收益。

禁止使用未复权价格直接计算跨除权日收益率。

---

## 3.2 真实价格口径

以下特征必须使用未复权价格：

- `log_raw_price`
- `tick_return`
- `tick_noise`
- `effective_ticks_3`
- `effective_ticks_5`
- 涨跌停判断
- 跳空收益
- K线实体、上影线和收盘位置

原因是复权价格不代表当时真实报价水平，会破坏低价股与最小跳价噪声的识别。

---

## 3.3 滚动窗口要求

所有滚动统计均采用：

- 仅使用 `t` 及之前的数据；
- 建议设置 `min_periods`；
- 不允许使用全样本均值和标准差；
- 不允许使用未来数据回填；
- 行业横截面统计必须在当日截面内完成。

建议：

| 窗口 | 最少有效样本 |
|---|---:|
| 20日窗口 | 15 |
| 60日窗口 | 40 |
| 120日窗口 | 80 |

---

## 3.4 去极值

所有极端比值类特征建议在每个交易日横截面内：

1. 按1%和99%分位数缩尾；
2. 再进行标准化或分位数排名。

推荐处理顺序：

\[
raw \rightarrow winsorize \rightarrow zscore/rank
\]

---

# 4. 核心价格特征

## 4.1 excess_ret_1

### 含义

个股最近1日相对行业的超额收益。

### 公式

\[
excess\_ret\_1
=
ret_{stock,1}
-
ret_{industry,1}
\]

### 计算口径

- 个股收益使用复权收盘价；
- 行业收益使用对应行业指数；
- 行业映射使用当日可获得的行业分类；
- 不得使用未来行业分类回填历史。

---

## 4.2 excess_ret_3

\[
excess\_ret\_3
=
ret_{stock,3}
-
ret_{industry,3}
\]

其余口径与 `excess_ret_1` 一致。

---

## 4.3 excess_ret_5

\[
excess\_ret\_5
=
ret_{stock,5}
-
ret_{industry,5}
\]

这是第一版模型最重要的价格强度特征之一。

---

## 4.4 excess_ret_10

\[
excess\_ret\_10
=
ret_{stock,10}
-
ret_{industry,10}
\]

用于区分：

- 刚刚开始上涨；
- 已经连续上涨较长时间；
- 可能已经进入过热阶段。

---

## 4.5 volatility_20

### 含义

个股过去20日的日收益波动率。

### 公式

\[
volatility\_20
=
Std(ret_{1,t-19:t})
\]

### 计算口径

- 使用复权收盘价日收益；
- 使用样本标准差；
- 最少15个有效交易日；
- 停牌日不填充为0收益，建议保留缺失或单独标记。

---

## 4.6 risk_adjusted_ret_5

### 含义

最近5日行业超额收益相对于个股自身正常波动的强度。

### 公式

\[
risk\_adjusted\_ret\_5
=
\frac{excess\_ret\_5}
{volatility\_20\sqrt{5}+\epsilon}
\]

### 参数

\[
\epsilon=10^{-6}
\]

### 建议处理

截断到：

\[
[-3,3]
\]

该特征可以减少低价高波动股票因几分钱上涨而获得过高分数的问题。

---

# 5. 成交活跃度特征

## 5.1 turnover_mean_20

### 含义

过去20日平均换手率。

### 公式

\[
turnover\_mean\_20
=
Mean(turnover\_rate_{t-19:t})
\]

---

## 5.2 turnover_std_20

### 含义

过去20日换手率标准差。

### 公式

\[
turnover\_std\_20
=
Std(turnover\_rate_{t-19:t})
\]

---

## 5.3 turnover_zscore_20

### 含义

当日换手率相对过去20日历史水平的标准化偏离。

### 公式

\[
turnover\_zscore\_20
=
\frac{
\log(1+turnover\_rate_t)
-
Mean[\log(1+turnover\_rate)]_{20}
}{
Std[\log(1+turnover\_rate)]_{20}
+\epsilon
}
\]

### 解释

- 小于0：当日换手低于自身近期正常水平；
- 大于0：当日换手高于自身近期正常水平。

---

## 5.4 turnover_zscore_60

\[
turnover\_zscore\_60
=
\frac{
\log(1+turnover\_rate_t)
-
Mean[\log(1+turnover\_rate)]_{60}
}{
Std[\log(1+turnover\_rate)]_{60}
+\epsilon
}
\]

用于判断缩量是否不仅相对短期异常，也相对中期异常。

---

## 5.5 amount_zscore_20

### 含义

当日成交额相对过去20日的异常程度。

### 公式

\[
amount\_zscore\_20
=
\frac{
\log(1+amount_t)
-
Mean[\log(1+amount)]_{20}
}{
Std[\log(1+amount)]_{20}
+\epsilon
}
\]

成交额可辅助识别：

- 低换手但高市值股票；
- 原始成交量单位不同导致的问题；
- 流通股本变化造成的换手率异常。

---

## 5.6 avg_amount_20

\[
avg\_amount\_20
=
Mean(amount_{t-19:t})
\]

建议模型输入：

\[
log\_avg\_amount\_20
=
\log(1+avg\_amount\_20)
\]

---

# 6. 条件成交量残差特征

## 6.1 volume_residual

### 含义

在当前涨跌幅、振幅、市场活跃度和行业活跃度条件下，实际换手率相对于模型预期值的偏离。

### 推荐模型

对每只股票使用过去120个交易日进行滚动回归：

\[
\log(1+turnover\_rate_t)
=
\alpha
+\beta_1|excess\_ret\_1|
+\beta_2 intraday\_range
+\beta_3 market\_turnover\_z
+\beta_4 industry\_turnover\_z
+\beta_5 volatility\_20
+\epsilon_t
\]

预测值：

\[
expected\_log\_turnover_t
=
\hat{\alpha}
+\sum_{i=1}^{5}\hat{\beta_i}X_{i,t}
\]

成交量残差：

\[
volume\_residual_t
=
\log(1+turnover\_rate_t)
-
expected\_log\_turnover_t
\]

### 解释

- `volume_residual < 0`：实际成交活跃度低于正常预期；
- `volume_residual > 0`：实际成交活跃度高于正常预期。

### 防泄漏要求

- 回归参数只能使用 `t-1` 及之前的数据拟合；
- 当日 `t` 的自变量可以代入已经拟合好的模型；
- 不允许用全样本回归后再生成历史残差。

---

## 6.2 scarcity

### 含义

供给收缩强度。

### 公式

\[
scarcity_t=-volume\_residual_t
\]

通常可进一步取正值部分：

\[
scarcity\_positive_t
=
\max(-volume\_residual_t,0)
\]

---

## 6.3 volume_residual_3d_mean

\[
volume\_residual\_3d\_mean
=
Mean(volume\_residual_{t-2:t})
\]

用于判断异常低成交是否连续存在。

---

## 6.4 volume_residual_5d_mean

\[
volume\_residual\_5d\_mean
=
Mean(volume\_residual_{t-4:t})
\]

值越低，说明最近5日持续低于正常成交预期。

---

## 6.5 scarcity_days_ratio_5

### 含义

最近5日中，实际成交低于模型预期的天数比例。

### 公式

\[
scarcity\_days\_ratio\_5
=
\frac{
\sum_{k=0}^{4}
I(volume\_residual_{t-k}<0)
}{5}
\]

取值范围：

\[
[0,1]
\]

---

## 6.6 scarcity_slope_5

### 含义

最近5日供给收缩强度的趋势。

### 计算方法

对以下序列进行一元线性回归：

\[
scarcity_{t-4:t}
=
a+b\times[0,1,2,3,4]
\]

取斜率：

\[
scarcity\_slope\_5=b
\]

### 解释

- 大于0：供给收缩程度逐渐增强；
- 小于0：成交量正在恢复或放大。

---

# 7. 价格推动效率特征

## 7.1 price_impact_1

### 含义

单位换手率推动的1日行业超额收益。

### 公式

\[
price\_impact\_1
=
\frac{excess\_ret\_1}
{turnover\_rate_t+\epsilon}
\]

### 注意

该比值在极低换手率时容易爆炸，必须：

- 设置最低换手率；
- 横截面1%和99%缩尾；
- 同时输入流动性特征；
- 不建议单独作为选股因子。

---

## 7.2 price_impact_5

### 含义

最近5日累计换手所推动的行业超额收益。

### 公式

\[
price\_impact\_5
=
\frac{
excess\_ret\_5
}{
\sum_{k=0}^{4}turnover\_rate_{t-k}
+\epsilon
}
\]

### 建议

先对分母设置下限：

\[
denominator
=
\max(
\sum turnover\_rate,
0.005
)
\]

具体下限需结合换手率单位确认。

---

## 7.3 price_impact_slope_5

### 含义

最近5日价格推动效率是否持续增强。

### 计算

先计算每天的 `price_impact_1`，再对过去5日做线性回归：

\[
price\_impact_{t-4:t}
=
a+b\times[0,1,2,3,4]
\]

取：

\[
price\_impact\_slope\_5=b
\]

---

## 7.4 up_days_ratio_5

### 含义

最近5日中，个股取得正行业超额收益的天数比例。

### 公式

\[
up\_days\_ratio\_5
=
\frac{
\sum_{k=0}^{4}
I(excess\_ret_{1,t-k}>0)
}{5}
\]

---

## 7.5 positive_excess_days_5

\[
positive\_excess\_days\_5
=
\sum_{k=0}^{4}
I(excess\_ret_{1,t-k}>0)
\]

取值范围为0至5。

---

# 8. 低价股与最小跳价噪声特征

## 8.1 log_raw_price

### 含义

真实未复权股价的对数。

### 公式

\[
log\_raw\_price
=
\log(raw\_close_t)
\]

### 用途

用于让模型识别低价股，但不建议仅依靠该字段降权。

---

## 8.2 tick_return

### 含义

一个最小报价单位对应的价格收益率。

### 公式

\[
tick\_return_t
=
\frac{tick\_size_t}{raw\_close_t}
\]

### 示例

- 股价1元，最小报价0.01元，`tick_return=1%`；
- 股价20元，最小报价0.01元，`tick_return=0.05%`。

值越高，价格离散化噪声越严重。

---

## 8.3 tick_noise

### 含义

最小跳价收益相对于个股正常日波动的比例。

### 公式

\[
tick\_noise_t
=
\frac{
tick\_return_t
}{
volatility\_20+\epsilon
}
\]

### 解释

- 值越高，少量最小跳价就可能制造较大的百分比收益；
- 适合作为样本权重和模型特征；
- 通常比单纯使用股价更合理。

---

## 8.4 effective_ticks_3

### 含义

最近3日价格实际移动了多少个最小报价单位。

### 公式

\[
effective\_ticks\_3
=
\frac{
raw\_close_t-raw\_close_{t-3}
}{
tick\_size_t
}
\]

---

## 8.5 effective_ticks_5

\[
effective\_ticks\_5
=
\frac{
raw\_close_t-raw\_close_{t-5}
}{
tick\_size_t
}
\]

### 用途

用于避免：

- 股价只上涨几分钱；
- 百分比收益看起来很高；
- 实际只移动了极少价格档位。

---

## 8.6 price_weight

### 含义

基于跳价噪声的训练样本降权系数。

### 公式

\[
price\_weight_t
=
\frac{
1
}{
1+\lambda \times tick\_noise_t
}
\]

建议测试：

\[
\lambda \in \{1,2,3\}
\]

建议最终截断：

\[
price\_weight
=
clip(price\_weight,0.1,1)
\]

该字段通常用于 `sample_weight`，不一定作为模型输入。

---

# 9. 流动性与规模控制特征

## 9.1 log_float_market_cap

\[
log\_float\_market\_cap
=
\log(1+float\_market\_cap_t)
\]

用于控制小市值暴露。

---

## 9.2 log_total_market_cap

\[
log\_total\_market\_cap
=
\log(1+total\_market\_cap_t)
\]

与流通市值存在较强相关性，第一版可二选一，优先使用流通市值。

---

## 9.3 amihud_illiquidity_20

### 含义

单位成交额对应的绝对价格波动。

### 公式

\[
amihud\_illiquidity\_20
=
Mean\left(
\frac{|ret_{1,t}|}
{amount_t+\epsilon}
\right)_{20}
\]

### 推荐缩放

成交额单位较大时，可以乘以常数：

\[
amihud\_scaled
=
10^8\times amihud
\]

缩放不影响排序，只方便数值稳定。

---

## 9.4 zero_return_days_20

### 含义

过去20日中，复权收盘价收益为0的交易日数量。

### 公式

\[
zero\_return\_days\_20
=
\sum_{k=0}^{19}
I(|ret_{1,t-k}|<10^{-8})
\]

该值较高时，股票可能存在流动性不足、停牌或报价不连续。

---

## 9.5 liquidity_weight

### 含义

基于20日平均成交额的样本权重。

### 公式

设：

- `A_low`：最低有效成交额阈值；
- `A_full`：达到完整权重的成交额阈值。

\[
liquidity\_weight
=
clip
\left(
\frac{
\log(avg\_amount\_20)-\log(A_{low})
}{
\log(A_{full})-\log(A_{low})
},
0,
1
\right)
\]

建议根据样本分布设置，例如：

- `A_low` 使用全市场20日成交额的10%分位数；
- `A_full` 使用50%分位数；
- 阈值只能使用训练期估计，不能用全样本。

---

## 9.6 sample_weight

### 含义

LightGBM训练时的最终样本权重。

### 公式

\[
sample\_weight
=
clip(
price\_weight
\times
liquidity\_weight,
0.1,
1
)
\]

可选增强：

\[
sample\_weight
=
price\_weight
\times
liquidity\_weight
\times
tradability\_weight
\]

其中不可交易样本应优先直接剔除，而不是仅降权。

---

# 10. K线结构特征

## 10.1 close_location

### 含义

收盘价在当日最高价和最低价区间中的位置。

### 公式

\[
close\_location
=
\frac{
raw\_close_t-raw\_low_t
}{
raw\_high_t-raw\_low_t+\epsilon
}
\]

取值通常在0到1之间：

- 接近1：收盘靠近最高价；
- 接近0：收盘靠近最低价。

当最高价等于最低价时，建议设为0.5，并增加一字板标记。

---

## 10.2 upper_shadow_ratio

### 含义

上影线占当日振幅的比例。

### 公式

\[
upper\_shadow\_ratio
=
\frac{
raw\_high_t-\max(raw\_open_t,raw\_close_t)
}{
raw\_high_t-raw\_low_t+\epsilon
}
\]

上影线过长可能表示：

- 上方抛压较大；
- 拉升后回落；
- 惜售结构不稳定。

---

## 10.3 body_ratio

### 含义

K线实体占当日振幅的比例。

### 公式

\[
body\_ratio
=
\frac{
|raw\_close_t-raw\_open_t|
}{
raw\_high_t-raw\_low_t+\epsilon
}
\]

---

## 10.4 intraday_range

### 含义

日内振幅。

### 公式

\[
intraday\_range
=
\frac{
raw\_high_t-raw\_low_t
}{
raw\_close_{t-1}+\epsilon
}
\]

必须使用前一交易日未复权收盘价作为分母。

---

## 10.5 gap_return

### 含义

当日开盘相对前一日收盘的跳空幅度。

### 公式

\[
gap\_return
=
\frac{
raw\_open_t
}{
raw\_close_{t-1}
}
-1
\]

用于区分：

- 盘中逐渐上涨；
- 单纯由跳空高开贡献的上涨。

---

## 10.6 limit_up_flag

### 含义

当日是否收于涨停价或接近涨停价。

### 推荐定义

\[
limit\_up\_flag
=
I(
raw\_close_t
\ge
limit\_up\_price_t-\delta
)
\]

其中：

\[
\delta=0.5\times tick\_size
\]

### 注意

涨停样本可能：

- 无法在收盘价买入；
- 次日开盘仍不可成交；
- 成交量受价格限制机制影响。

建议作为特征保留，但回测必须单独做可交易性判断。

---

# 11. 趋势与位置辅助特征

## 11.1 close_above_ma5

\[
close\_above\_ma5
=
\frac{
adj\_close_t
}{
MA(adj\_close,5)_t
}
-1
\]

---

## 11.2 close_above_ma20

\[
close\_above\_ma20
=
\frac{
adj\_close_t
}{
MA(adj\_close,20)_t
}
-1
\]

用于区分：

- 刚刚突破；
- 已经远离均线；
- 可能已经过热。

---

# 12. 市场与行业环境特征

## 12.1 market_return_1

\[
market\_return\_1
=
\frac{
market\_index\_close_t
}{
market\_index\_close_{t-1}
}
-1
\]

---

## 12.2 market_return_5

\[
market\_return\_5
=
\frac{
market\_index\_close_t
}{
market\_index\_close_{t-5}
}
-1
\]

---

## 12.3 industry_return_1

\[
industry\_return\_1
=
\frac{
industry\_index\_close_t
}{
industry\_index\_close_{t-1}
}
-1
\]

---

## 12.4 industry_return_5

\[
industry\_return\_5
=
\frac{
industry\_index\_close_t
}{
industry\_index\_close_{t-5}
}
-1
\]

---

## 12.5 industry_strength_rank

### 含义

当日各行业5日收益的横截面分位数排名。

### 公式

\[
industry\_strength\_rank
=
PercentileRank_{industries}
(
industry\_return\_5
)
\]

取值范围：

\[
[0,1]
\]

越接近1，行业相对越强。

---

## 12.6 market_breadth

### 含义

全市场上涨股票占比。

### 公式

\[
market\_breadth
=
\frac{
N(ret_{1}>0)
}{
N(valid\ stocks)
}
\]

建议剔除：

- 停牌；
- 上市时间不足；
- 无有效价格；
- 明显异常证券。

---

## 12.7 industry_breadth

### 含义

股票所属行业内上涨股票占比。

### 公式

\[
industry\_breadth
=
\frac{
N_{industry}(ret_{1}>0)
}{
N_{industry}(valid\ stocks)
}
\]

---

## 12.8 market_volatility

### 含义

市场指数过去20日波动率。

### 公式

\[
market\_volatility
=
Std(market\_return_{1,t-19:t})
\]

---

## 12.9 regime_score

### 含义

已有市场状态模块输出的连续型评分。

### 要求

- 不建议直接使用未来定义的牛市、熊市标签；
- 优先使用当日可计算的连续评分；
- 所有阈值和参数必须仅使用训练期确定；
- 需要保留评分的具体组成，便于解释模型。

---

# 13. 推荐组合特征

## 13.1 simple_low_volume_rise

### 含义

最简单的低量上涨交互特征。

### 公式

先计算：

\[
ret\_z
=
ZScore(excess\_ret\_5,60)
\]

\[
turnover\_z
=
turnover\_zscore\_60
\]

最终：

\[
simple\_low\_volume\_rise
=
\max(ret\_z,0)
\times
\max(-turnover\_z,0)
\]

---

## 13.2 conditional_scarcity_factor

### 含义

风险调整后的上涨强度与条件成交量残差的交互。

### 公式

\[
conditional\_scarcity\_factor
=
\max(risk\_adjusted\_ret\_5,0)
\times
\max(-volume\_residual,0)
\]

---

## 13.3 close_quality_scarcity_factor

### 含义

在供给收缩基础上，要求收盘质量较高。

### 公式

\[
close\_quality\_scarcity\_factor
=
conditional\_scarcity\_factor
\times
close\_location
\times
(1-upper\_shadow\_ratio)
\]

---

## 13.4 persistent_scarcity_factor

### 含义

持续性低量上涨结构。

### 公式

\[
persistent\_scarcity\_factor
=
conditional\_scarcity\_factor
\times
scarcity\_days\_ratio\_5
\times
up\_days\_ratio\_5
\]

---

## 13.5 price_adjusted_scarcity_factor

### 含义

对低价股最小跳价噪声降权后的供给收缩因子。

### 公式

\[
price\_adjusted\_scarcity\_factor
=
persistent\_scarcity\_factor
\times
price\_weight
\]

该字段可用于传统因子排序，但在LightGBM中建议同时保留各原始组成特征。

---

# 14. 第一版 LightGBM 推荐特征集

建议第一版控制在以下特征，不要直接加入大量通用技术指标。

## 14.1 核心量价特征

1. `excess_ret_1`
2. `excess_ret_3`
3. `excess_ret_5`
4. `excess_ret_10`
5. `risk_adjusted_ret_5`
6. `turnover_zscore_20`
7. `turnover_zscore_60`
8. `amount_zscore_20`
9. `volume_residual`
10. `volume_residual_3d_mean`
11. `volume_residual_5d_mean`
12. `scarcity_days_ratio_5`
13. `scarcity_slope_5`
14. `price_impact_1`
15. `price_impact_5`
16. `price_impact_slope_5`
17. `up_days_ratio_5`

## 14.2 低价与微观结构特征

18. `log_raw_price`
19. `tick_return`
20. `tick_noise`
21. `effective_ticks_3`
22. `effective_ticks_5`

## 14.3 流动性和风险控制

23. `log_float_market_cap`
24. `log_avg_amount_20`
25. `turnover_mean_20`
26. `turnover_std_20`
27. `amihud_illiquidity_20`
28. `zero_return_days_20`
29. `volatility_20`

## 14.4 K线结构

30. `close_location`
31. `upper_shadow_ratio`
32. `body_ratio`
33. `intraday_range`
34. `gap_return`
35. `limit_up_flag`

## 14.5 市场和行业环境

36. `market_return_5`
37. `industry_return_5`
38. `industry_strength_rank`
39. `market_breadth`
40. `industry_breadth`
41. `market_volatility`
42. `regime_score`

第一版可以先使用约25至30个特征，后续通过消融实验逐步增加。

---

# 15. 训练标签

## 15.1 未来5日行业超额收益

信号日为 `t`，交易从 `t+1` 开始：

\[
future\_ret\_5
=
\frac{
adj\_close_{t+5}
}{
adj\_open_{t+1}
}
-1
\]

行业同期收益：

\[
future\_industry\_ret\_5
=
\frac{
industry\_close_{t+5}
}{
industry\_open_{t+1}
}
-1
\]

标签：

\[
y_5
=
future\_ret\_5
-
future\_industry\_ret\_5
\]

---

## 15.2 未来10日行业超额收益

\[
y_{10}
=
future\_ret_{10}
-
future\_industry\_ret_{10}
\]

建议5日和10日分别训练独立模型。

---

## 15.3 横截面排名标签

每天对 `y_5` 做横截面百分位排名：

\[
y\_rank_5
=
PercentileRank_{date}(y_5)
\]

适用于：

- 回归模型预测每日截面排名；
- 降低极端收益值对训练的影响；
- 与最终TopN选股目标更一致。

---

# 16. 样本过滤规则

训练和回测前至少剔除：

1. ST和退市整理股票；
2. 上市不足60至120个交易日的股票；
3. 停牌或无有效价格数据的样本；
4. 日均成交额低于最低流动性门槛的股票；
5. 信号日一字涨停且无法买入的股票；
6. `t+1` 开盘无法成交的股票；
7. 除权数据异常；
8. 价格、成交额或换手率明显错误；
9. 行业分类缺失；
10. 未来标签区间内停牌导致收益无法可靠计算的样本。

低价股不建议仅凭固定价格全部剔除，优先通过：

- `tick_noise`
- `effective_ticks`
- `price_weight`
- 流动性门槛

进行控制。

---

# 17. 消融实验设计

## 模型A：仅控制变量

输入：

- `log_raw_price`
- `tick_noise`
- `log_float_market_cap`
- `log_avg_amount_20`
- `amihud_illiquidity_20`
- `volatility_20`
- `industry_return_5`
- `market_breadth`
- `regime_score`

目的：测量模型仅依靠市值、价格、流动性和市场环境可以获得多少效果。

---

## 模型B：控制变量 + 低量上涨核心特征

在模型A基础上增加：

- `excess_ret_5`
- `risk_adjusted_ret_5`
- `volume_residual`
- `volume_residual_5d_mean`
- `scarcity_days_ratio_5`
- `price_impact_5`
- `up_days_ratio_5`
- `close_location`
- `upper_shadow_ratio`

目的：验证低量上涨结构是否存在独立增量Alpha。

---

## 模型C：模型B + 环境增强

增加：

- `industry_strength_rank`
- `industry_breadth`
- `market_volatility`
- 更完整的 `regime_score`
- `scarcity_slope_5`
- `price_impact_slope_5`

目的：验证低量上涨是否只在特定市场和行业环境中有效。

---

# 18. 评价指标

模型必须在样本外数据上评价：

1. OOS Pearson IC；
2. OOS Rank IC；
3. IC均值、标准差和ICIR；
4. Top2、Top5、Top10、Top20收益；
5. 1、3、5、10、15日持有期；
6. 10bps和20bps交易成本；
7. 年化收益；
8. 超额收益；
9. Sharpe；
10. 最大回撤；
11. Calmar；
12. 分年度表现；
13. 行业暴露；
14. 市值和股价暴露；
15. 股票收益贡献集中度；
16. 特征重要性稳定性；
17. SHAP方向稳定性；
18. 模型B相对模型A的样本外增量。

最关键结论不是模型总收益，而是：

\[
Incremental\ Alpha
=
Performance(Model\ B)
-
Performance(Model\ A)
\]

如果模型B没有稳定优于模型A，则低量上涨结构未提供可靠的独立Alpha。

---

# 19. 防止数据泄漏的强制要求

1. 所有滚动均值、标准差和回归模型只能使用当日及以前数据；
2. 生成 `volume_residual_t` 的模型必须使用 `t-1` 以前样本拟合；
3. 标准化参数只能在训练窗口内估计；
4. 行业分类必须使用历史当时可获得版本；
5. 不允许用未来退市、ST或成分股信息回填；
6. 训练集、验证集和测试集必须按时间顺序切分；
7. 不允许随机打乱时间序列后切分；
8. 调参只能使用训练集和验证集；
9. 测试集只能在模型冻结后使用一次；
10. 回测成交价格必须与信号时间严格匹配。

---

# 20. 推荐第一轮实现顺序

## 阶段一：先验证原始结构

实现：

- `excess_ret_5`
- `risk_adjusted_ret_5`
- `turnover_zscore_20`
- `turnover_zscore_60`
- `volume_residual`
- `price_impact_5`
- `tick_noise`
- `effective_ticks_5`
- `close_location`
- `upper_shadow_ratio`

先做二维分组：

\[
excess\_ret\_5
\times
volume\_residual
\]

观察在相同上涨强度下，成交量残差越低的股票未来收益是否越高。

---

## 阶段二：加入持续性

增加：

- `volume_residual_5d_mean`
- `scarcity_days_ratio_5`
- `scarcity_slope_5`
- `up_days_ratio_5`
- `price_impact_slope_5`

---

## 阶段三：训练LightGBM

先运行模型A和模型B的消融实验，再决定是否加入完整市场和Regime特征。

不要在第一版加入大量MACD、RSI、KDJ、均线交叉等通用技术指标，否则很难判断模型是否真正验证了“低量上涨供给收缩”假设。
