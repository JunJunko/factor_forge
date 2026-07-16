# 底背离 × Regime × 概念轮动 × 个股动量：Alpha 实验设计 v1

## 1. 研究问题与主假设

研究对象不是“出现底背离就买”，而是检验一个条件 Alpha：

> 当个股出现可在收盘时确认的底背离，并且市场处于修复而非退潮、所属概念处于扩散和相对强度改善阶段、个股自身从负动量转为早期恢复时，未来 10 个交易日的行业中性收益及路径质量是否显著优于普通底背离。

预注册的最佳组合候选是：

- 市场：`repair`，或 PIT HMM 的风险改善概率较高；排除 `retreat`，降低 `overheat` 权重。
- 概念：`improving -> leading`，共同成分扩散上升、成员 churn 低、RRG 相对动量为正。
- 个股：20/60 日仍处低位，但 3/5 日相对动量、价格加速度和收盘接受度转正，即“早期恢复”，而不是已经过度上涨。
- 底背离：价格第二低点更低或接近前低，但 RSI/MACD/卖压至少一个维度形成更高低点，且第二低点后的确认没有使用未来数据。

这是待证伪假设。`retreat + lagging + 动量继续恶化` 是预注册负对照。

## 2. 两条样本线

### 2.1 长历史机制样本

使用申万一级/二级行业 PIT 映射，覆盖尽可能长的历史。它负责回答：市场状态、板块相对强弱和个股动量是否真的对底背离有条件增益。

### 2.2 真实概念版本样本

只使用当日可见的概念成员快照。当前仓库中的真实概念版本大约从 2024 年末开始，样本不足以独立授权复杂 ML 上线，只用于：

- 验证行业层面的机制能否迁移到概念；
- 比较 current membership 与 common membership 的扩散口径；
- 生成 OOF/shadow 预测并进入前向观察。

禁止用最新概念成员向历史回填作为正式结果；它只能作为 look-ahead placebo。

## 3. 样本时钟与事件定义

```text
历史低点 A          第二低点 B（位于 T-3..T）       信号收盘 T       成交 T+1 开盘
只使用当时数据      B 之后最多仅使用截至 T 的确认    冻结全部入参       标签从此开始
```

主事件采用因果双低点定义：

1. 在 `T-4..T` 找最低点 B，记录 `b_age`。
2. 在 B 之前的 `20..60` 个交易日内寻找候选低点 A；A 与 B 至少间隔 5 日。
3. A、B 都要求此前存在下降段；A 到 B 之间存在至少 `0.5 ATR` 的反弹，避免把连续下跌中的相邻两天误当双底。
4. T 收盘时计算背离和确认；不等待 T+1 数据确认 B。
5. 同一股票 10 日内的重复信号合并为一个 episode，主样本使用 episode 首个满足条件的信号日；峰值日仅作诊断。

主窗口固定为 `lookback=60, separation=5, current_window=5`。20/40/80 日多尺度值可以作为 ML 特征，但不得在 sealed test 上选择“最好窗口”。

## 4. 底背离评分

### 4.1 保留的原子特征

底背离绝不能只保存一个最终分数。ML 数据集必须保留下列原子量：

| 组 | 特征示例 | 定义方向 |
|---|---|---|
| 双低点几何 | `div__price_lower_low_atr` | `(low_A-low_B)/ATR20_A`，越高表示第二低点越低 |
| 双低点几何 | `div__low_similarity_atr` | `-abs(low_B-low_A)/ATR20_A` |
| 间隔与形态 | `div__trough_gap_days`, `div__intervening_rebound_atr`, `div__b_age` | 可靠性坐标，不预设线性方向 |
| RSI 背离 | `div__rsi14_higher_low`, `div__rsi6_higher_low` | `RSI_B-RSI_A` |
| MACD 背离 | `div__macd_hist_higher_low`, `div__macd_slope_b` | 第二低点动能改善 |
| 收益动能背离 | `div__ret5_higher_low`, `div__downside_velocity_change` | 下跌速度是否放缓 |
| 卖压背离 | `div__down_volume_change`, `div__turnover_change`, `div__main_sell_ratio_change` | 价格创新低时卖压是否收缩 |
| 波动/区间 | `div__range_contraction`, `div__natr_change` | 第二低点是否缩量缩波 |
| 当日接受 | `div__close_location`, `div__lower_shadow_atr`, `div__reclaim_ma5_atr` | 截至 T 的反转确认 |
| 可靠性 | `div__history_valid_ratio`, `div__indicator_agreement_count` | 缺失和信号一致性 |

所有 oscillator 在 A、B 的取值都必须使用各自时点已经可见的历史。若使用复权价，必须通过“修改未来行情不改变 T 特征”的 prefix-invariance 测试。

### 4.2 可解释基准分

先将原子量按当日股票截面转成 `[0,1]` robust percentile，定义：

```text
core = geometric_mean(
    q(price_lower_low_atr),
    max(q(rsi14_higher_low), q(macd_hist_higher_low), q(downside_velocity_change))
)

confirmation = mean(
    q(intervening_rebound_atr), q(close_location),
    q(lower_shadow_atr), q(-down_volume_change)
)

divergence_score = 100 * reliability * (0.70 * core + 0.30 * confirmation)
```

`reliability` 由历史完整度、低点间隔和异常停牌情况构成，范围 `[0,1]`。这个分数只用于可解释基准、分层和事件门槛；LightGBM/Ridge 仍读取原子特征，防止人工加权掩盖非线性。

### 4.3 底背离价位触碰/回踩因子

把第二低点 B 定义为当前底背离的锚定价位。必须同时保存“真实价格”和“可跨股票比较的标准化价格”：

```text
touch__level_raw       = 把 B 的 PIT 复权价换算到 T 日原始价格空间后的等价支撑价
touch__level_raw_origin = B 当日原始最低价 raw_low_B
touch__level_adj_pit   = PIT_adjusted_low_B
touch__level_to_close  = touch__level_adj_pit / adjusted_close_T - 1
touch__zone_width      = max(2 * tick_size, 0.15 * ATR20_T)
touch_zone             = [level - zone_width, level + zone_width]
```

`touch__level_raw` 是 T 日价格空间中可以直接展示或用于订单参考的实际价格，例如 `9.83 元`；`touch__level_raw_origin` 用于审计历史 B 当日的原始最低价。两者都是 diagnostic/order 字段，不直接进入 ML，防止模型把股票绝对价格当 Alpha。ML 使用 `level_to_close`、ATR 距离和百分比距离。

对最近 10 根 K 线 `j in [T-9, T]`，排除形成锚点的 B 本身。一根 K 线触碰价位的定义为其完整价格区间与触碰带相交：

```text
touch_j = low_j <= level + zone_width AND high_j >= level - zone_width

distance_j_atr = max(low_j - level, level - high_j, 0) / ATR20_T
```

排除 B 非常重要，否则每一个底背离样本的 `touch_occurred` 都恒为 1。分别统计 B 之前的“历史支撑聚集”和 B 之后的“形成后回踩”：

| 字段 | 含义 |
|---|---|
| `touch__occurred_10d` | 排除 B 后，最近 10 日是否至少有一根 K 线触碰 |
| `touch__count_10d` | 触碰次数；保留连续值，不假设越多越好 |
| `touch__nearest_distance_atr_10d` | 最近 K 线区间到锚定价的最小 ATR 距离 |
| `touch__age_days` | 最近一次触碰距 T 的交易日数 |
| `touch__pre_b_count` | B 之前对同一价位的触碰次数，衡量历史支撑聚集 |
| `touch__post_b_count` | B 之后的回踩次数，衡量形成后的再确认 |
| `touch__last_penetration_atr` | `(level-low_last_touch)/ATR`，正值表示盘中跌破深度 |
| `touch__last_close_reclaim_atr` | `(close_last_touch-level)/ATR`，正值表示收盘重新站回 |
| `touch__post_touch_return_to_t` | 最近触碰收盘到 T 收盘的收益 |
| `touch__false_break_reclaim` | 最近触碰日盘中跌破、收盘站回的 0/1 标识 |
| `touch__post_b_observable` | B 早于 T 时为 1；B=T 时形成后回踩尚不可观测 |

可解释的触碰承接分为：

```text
touch_acceptance_score = 100 * touch_occurred * mean(
    exp(-touch_age_days / 5),
    q(-nearest_distance_atr),
    q(last_close_reclaim_atr),
    q(post_touch_return_to_t)
)
```

`touch__count_10d` 不进入人工分数：第一次/第二次回踩可能强化支撑，过多测试也可能消耗买盘，这种非单调关系留给树模型检验。若 `B=T`，`post_b_*` 字段保留 NaN 并设置可观测标志，不能错误编码成“没有回踩”。触碰带宽主规格固定为 `0.15 ATR + 2 ticks floor`；`0.10/0.25 ATR` 仅作敏感性分析。

## 5. ML 特征分组

每一列都在 manifest 中记录 `name / group / role / clock / window / transform / missing_policy / expected_sign / source_version`。

### D：底背离机制

使用第 4 节全部原子特征，并增加：

- 20/40/80 日三个固定尺度的低点几何和 RSI 背离；
- `div__score`, `div__score_rank`, `div__episode_age`；
- `div__rsi_macd_agree`, `div__price_sell_pressure_agree`。

### T：底背离价位触碰

使用第 4.3 节全部 `touch__` 原子量。`touch__level_raw` 和 `touch__level_adj_pit` 作为展示/订单字段；模型只读取相对价格、ATR 距离、次数、时间、穿透、收盘站回和触碰后反应。线性模型需为不可观测的 `post_b_*` 增加 missing indicator，LightGBM 保留 NaN。

### R：市场 Regime

连续状态优先于硬标签：

- 趋势：指数 5/20/60 日收益、MA20/60 距离、60 日回撤；
- 广度：上涨比例、MA20/60 以上比例、涨跌家数、`breadth_thrust`；
- 风险：指数波动率、下行波动、横截面离散度、跌停/涨停比例；
- 流动性与情绪：成交额 5/20 比、换手、主力净流、融资变化；
- 风格：小盘减大盘 5/10/20 日收益；
- 规则状态 one-hot：`repair/overheat/retreat/divergence/neutral`；
- PIT HMM：`regime__p_state_0..K`、状态熵、状态持续天数和概率变化。

HMM 必须 walk-forward 重训并输出 filtered probability，禁止用全样本拟合或 smoothed probability。

### C：概念版本与轮动

复用现有 `concept_rotation_alpha.py` 和 `concept_rotation_ml.py` 的 PIT 口径：

- 相对强弱：概念 1/5/20/60 日收益、`rs_20d`、`rs_momentum_5d`、RRG quadrant；
- 扩散：自由流通市值加权 breadth、5 日变化、common-membership breadth 变化与平滑值；
- 生命周期：new/persistent improving、confirmed/persistent leading、weakening/lagging、leading age；
- 成员版本质量：成员数、匹配覆盖率、概念年龄、5 日 membership churn；
- 资金：概念成交额、全市场占比、成交额排名和加速度；
- 拥挤与重叠：股票所属概念数、概念对 Jaccard、概念收益离散度。

一只股票可能属于多个概念，禁止直接展开成数百个 concept ID。按当日 PIT 成员关系聚合为：

- `concept__best_*`：所属概念中最强的值；
- `concept__top3_mean_*`：Top3 均值；
- `concept__support_count_*`：满足 improving/leading、breadth 上升的概念数；
- `concept__dispersion_*`：所属概念状态离散度；
- `concept__primary_*`：按“概念轮动分 × 1/sqrt(成员数) × 1/sqrt(个股概念度数)”选择主概念后的值。

概念代码、概念名称和股票代码只作为 key/诊断字段，不能作为模型入参，避免模型记忆实体。

### M：个股动量与位置

- 绝对动量：1/3/5/10/20/60/120 日收益；
- 相对动量：相对申万行业、主概念、全市场的 5/20/60 日收益；
- 加速度：`ret_5 - ret_20/4`、`ret_10-ret_60/6`、RSI/MACD 斜率；
- 位置：距 20/60/120 日高低点、MA5/10/20/60 距离、60 日回撤；
- 量价确认：成交额 5/20 比、换手变化、上涨日/下跌日量比；
- 动量状态（仅报告/交互）：继续恶化、早期恢复、趋势确认、过热。

“早期恢复”的预注册定义是：`ret20` 截面分位低于 40%，同时 `ret3/ret5`、加速度或 MA5 reclaim 至少两项转正。模型读取连续量，状态标签不替代原子量。

### X：风险坐标与可交易性

- 申万一级/二级行业、log 流通市值、beta20/60、波动率20/60；
- ADV20、换手水平、上市天数、价格、涨跌停距离；
- ST、停牌、退市期、开盘涨停/跌停均为 filter 或 execution 字段，不作为 Alpha 来源。

风险坐标用于匹配、残差化和模型控制。必须分别报告“完整预测分”和“对行业/规模/beta/波动/流动性日截面残差化后的 Alpha 分”。

### I：少量预注册交互

只人工生成以下交互，其余非线性交给树模型：

- `div_score × regime_repair_probability`
- `div_score × market_breadth_delta_5d`
- `div_score × concept_best_rs_momentum`
- `div_score × concept_top3_common_breadth_delta`
- `div_score × stock_momentum_acceleration`
- `div_score × early_recovery_flag`
- `concept_rs_momentum × stock_relative_momentum`
- `concept_breadth_delta × stock_amount_ratio_5_20`

## 6. 标签设计

信号在 T 收盘冻结，统一从 T+1 可成交开盘开始。

主标签：

```text
label__industry_excess_10d =
    stock open(T+11) / open(T+1) - 1
    - SW_L1_LOO_benchmark_return(T+1, T+11)
```

辅助标签：

- `label__industry_excess_5d`、`label__industry_excess_20d`；
- `label__concept_excess_10d`，仅概念 PIT 样本使用；
- `label__quality_atr_10d = MFE_ATR - 1.5 * abs(MAE_ATR)`；
- `label__success_10d`：10 日行业超额扣 40 bps 后为正，且 MAE 不低于 `-1.5 ATR`；
- `label__time_to_mfe`、`label__mae_atr`、`label__mfe_atr` 仅做路径诊断。

训练可以同时比较回归、LambdaRank 和 success 分类，但主决策指标固定为 10 日行业中性 OOF Rank IC 与可执行 Top10 收益。

## 7. 实验阶梯与消融

### E0：数据和因果 Gate

- PIT 行业、PIT 概念、退市股票、停牌和上市状态完整；
- 改写 T+1 以后数据不改变 T 的任何特征；
- latest-membership 回填版本只能出现在 placebo 报告；
- 重复事件合并、标签成熟日和 purge/embargo 审计通过。

### E1：底背离本体是否存在 Alpha

同日、同行业匹配 3 个非背离对照，控制：5/20 日收益、60 日回撤、规模、波动率、ADV20、换手和 beta。比较 score 五分组的未来超额、MFE/MAE 和成功率。

若底背离本体不呈单调性，或相对匹配对照无增益，停止组合优化，避免靠 Regime/概念特征包装一个无效事件。

### E1.1：触碰因子的独立增量

在完全相同的底背离事件池内，先比较 `M1a = X + D` 与 `M1b = X + D + T`。同时用二维分组区分：未触碰、触碰但收盘未站回、触碰且收盘站回、假跌破后站回。主检验是 `M1b` 对 `M1a` 的 OOF Rank IC 和 Top10 净收益增量，而不是单独挑选表现最好的触碰带宽。

### E2：条件矩阵，不训练模型

固定底背离事件池，报告：

- score quintile × 市场规则状态；
- score quintile × 概念生命周期；
- score quintile × 个股动量状态；
- 市场 × 概念 × 个股三维组合，但只检验预注册的正组合和负对照。

统计按 signal date 聚类，并使用按日期 block bootstrap；重叠持有期不能按股票行数计算普通 t 值。

### E3：固定样本的 ML 消融

| Arm | 输入 | 要回答的问题 |
|---|---|---|
| M0 | X 风险坐标 | 仅靠暴露能做到什么 |
| M1a | X + D | 底背离是否超越风险坐标 |
| M1b | X + D + T | 价位触碰/回踩是否有独立增量 |
| M2 | X + D + T + R | 市场状态的增量 |
| M3 | X + D + T + C | 概念轮动的增量 |
| M4 | X + D + T + M | 个股动量的增量 |
| M5 | X + D + T + R + C + M | 三块联合增量 |
| M6 | M5 + I | 预注册机制交互是否再增益 |

所有 Arm 使用完全相同的事件样本、fold、标签和超参数预算。M2/M3/M4 的顺序不表示允许逐步挑最好结果；三者都要报告。触碰带宽和回看窗口由配置冻结，不能因为 M1b 结果不佳再重新挑选。

模型顺序：

1. Ridge/Logistic 作为符号和可解释性基线；
2. LightGBM regression；
3. LightGBM LambdaRank，按 signal date 分组；
4. 不在 v1 引入神经网络、自动特征搜索或贝叶斯调参。

概念样本短时优先使用浅树、小叶节点、强正则，并把模型权重限制为基准分的 20%；只有后续 unseen 数据才允许提高权重。

### E4：可执行 OOF 回测

- T 收盘信号，T+1 开盘成交；
- 主规格：每天 Top10、持有 10 日、10 个等资金 staggered sleeves；
- 辅助：Top5/20，持有 5/20 日；
- 主成本 40 bps round-trip，压力测试 20/60 bps；
- 涨停买不进、跌停卖出顺延、停牌顺延、100 股整数手；
- 单票最大 10%，单行业最大 25%，成交不超过 ADV20 的 5%；
- 概念重叠按股票合并，不能因为一只股票属于多个概念而重复加权。

## 8. 切分、标准化与防泄漏

- 只允许 expanding walk-forward；禁止随机切分。
- purge 至少为 `label_horizon + 1 = 11` 个交易日，valid/test 之间同样 embargo。
- 线性模型的缺失填充、winsor、标准化只在训练段拟合；LightGBM 使用原值/NaN。
- 当日截面 rank 可以使用当日全部可投资股票；历史 percentile 必须 strict-prior。
- 每日事件数不均衡时使用 `1 / 当日事件数` 作为统计/训练权重，避免极端行情日支配损失。
- 标签、MFE、MAE、未来可交易性字段禁止进入 predictor manifest。
- 不把硬状态和它的完全冗余反向字段同时输入线性模型；one-hot 留一个基准类。
- 特征选择、early stopping 和超参数选择仅使用 train/validation；sealed test 只开一次。

## 9. Placebo 与稳健性

必须运行：

- 在同日截面内随机打乱 divergence score；
- 打乱概念成员但保持每日概念规模分布；
- latest-membership backfill look-ahead placebo；
- 用 weakening/lagging 概念替换 improving/leading；
- 信号日期随机平移 ±20 日（不跨股票历史边界）；
- 去掉收益最好的 1% events、最好的月份和最好的年份；
- 大小盘、行业、年份、规则 Regime 和 HMM state 分层；
- current membership breadth 与 common-membership breadth 对照。

多窗口、多标签和多个条件格子的显著性统一做 Benjamini-Hochberg FDR；主标签和主组合不参与事后改名。

## 10. 决策 Gate

进入 shadow candidate 必须同时满足：

1. E1 的 score 五分组大体单调，Q5-Q1 的 10 日匹配超额为正，date-block bootstrap 90% CI 下界大于 0；
2. M1b 相对 M1a 的 OOF Rank IC 或可执行净收益至少一项有正增量，且另一项不得显著恶化；
3. M5/M6 相对 M1b 的 OOF Rank IC 增量为正，至少 3/4 folds 同方向；
4. 主规格 40 bps 后净超额为正，且相对 M1b 在至少 60% 年度或半年度窗口胜出；
5. 60 bps、去掉最佳 1% events、去掉最佳月份后仍为正；
6. Alpha 不能由单一行业、单一概念或少数股票贡献超过 30%；
7. placebo 不产生同等级结果；
8. 概念版本样本只可授权 shadow，至少积累 6 个月未见前向数据后再决定是否部署。

任一 Gate 失败，不允许通过改窗口、改 Regime 命名、改 TopN 或删除差年份来“修复”同一次试验；修改后必须登记为新 hypothesis/version。

## 11. 推荐实现路径

1. 新建 `bullish_divergence_features.py`，负责因果双低点、D 特征和 T 触碰特征；先做 prefix-invariance 与“排除锚点 B”单测。
2. 新建 stock-day feature assembler，把现有 timing regime、concept rotation、stock momentum 聚合到 `(trade_date, ts_code)`。
3. 同时产出 daily panel 与 episode table；前者便于横截面 IC，后者用于事件研究和 ML。
4. 先执行 E0/E1/E2 并冻结报告；通过后才执行 E3。
5. E3 产出 OOF prediction 后调用现有 T+1 开盘执行引擎完成 E4，禁止使用 in-sample score 回测。

建议的关键产物：

```text
artifacts/bullish_divergence_runs/<run_id>/
  manifest.json
  feature_manifest.json
  stock_day_features.parquet
  divergence_episodes.parquet
  matched_event_study.parquet
  conditional_matrix.csv
  split_audit.json
  oof_predictions.parquet
  ablation_metrics.csv
  trades.parquet
  nav.parquet
  robustness.csv
  decision.json
  research_report.md
```

### 当前实现命令

先构建全市场 D/T 特征（正式 E1 禁止使用 `--max-stocks` 或局部 `--stock-code`）：

```powershell
python scripts/bullish_divergence_features.py `
  --panel data/versions/<version>/curated/stock_daily_panel.parquet `
  --start-date 2021-01-01 `
  --end-date 2026-07-10
```

再使用生成目录中的 stock-day 和 episode 文件运行 E1/E1.1：

```powershell
python scripts/bullish_divergence_event_study.py `
  --panel data/versions/<version>/curated/stock_daily_panel.parquet `
  --features artifacts/bullish_divergence_runs/<run>/stock_day_features.parquet `
  --episodes artifacts/bullish_divergence_runs/<run>/divergence_episodes.parquet
```

E1 脚本要求特征表每日股票覆盖率中位数至少 95%；局部样本只有显式加入 `--allow-partial-universe` 才能运行，且输出决策固定为 diagnostic only。
