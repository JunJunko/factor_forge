# `$scan-market-anomalies` 使用手册

## 目录

1. [用途与边界](#1-用途与边界)
2. [运行条件](#2-运行条件)
3. [最快使用方法](#3-最快使用方法)
4. [“最新市场”的严格定义](#4-最新市场的严格定义)
5. [完整执行流程](#5-完整执行流程)
6. [命令参数与常见运行方式](#6-命令参数与常见运行方式)
7. [十个冻结模板](#7-十个冻结模板)
8. [输出目录与文件](#8-输出目录与文件)
9. [如何阅读扫描结果](#9-如何阅读扫描结果)
10. [Skill 会怎样回答](#10-skill-会怎样回答)
11. [失败与故障排查](#11-失败与故障排查)
12. [研究边界与后续流程](#12-研究边界与后续流程)
13. [配置与维护规则](#13-配置与维护规则)
14. [常见问题](#14-常见问题)

## 1. 用途与边界

`$scan-market-anomalies` 是 Factor Forge 的最新市场异常扫描 Skill。它使用固定的十模板组合，从最新完整日线数据中产生：

- 8 类个股事件异常；
- 2 类市场关系漂移；
- 当前事件发生率、历史比较和覆盖情况；
- 数据同步、完整性和 freshness 审计结果；
- 最多 5 个通过质量门的当前研究线索。

它的目标是：

> 比人工浏览 K 线更系统地发现值得进一步研究的异常，而不是自动产生交易信号。

Skill 不会自动执行：

- 买卖方向判断；
- 因子生成；
- Event Study；
- 回测；
- 模拟盘或实盘；
- 阈值搜索或自动调参。

扫描结果只能称为“Observation”“异常候选”或“关系漂移”，不能直接称为 Alpha。

## 2. 运行条件

### 2.1 工作目录

默认项目目录：

```text
D:\pyworkspace\factor_forge
```

Skill 会优先使用当前包含以下文件的工作目录：

```text
pyproject.toml
configs/radar/latest_market_scan_v1.yaml
src/factor_forge/
```

### 2.2 Skill 安装位置

```text
C:\Users\Junko\.codex\skills\scan-market-anomalies
```

主要文件：

```text
scan-market-anomalies/
  SKILL.md
  agents/openai.yaml
  references/template-catalog.md
```

### 2.3 Python 环境

在项目根目录应能运行：

```powershell
python -m factor_forge.cli anomaly-scan latest --help
```

### 2.4 数据权限

扫描“最新市场”时，即使本地数据已经完整，也需要查询交易日历来确定现实世界中最新已经完成且可用的交易日。

环境需要能够初始化 `TushareProvider`。如果需要补数，还必须具备以下数据端点权限：

- 日线行情；
- 复权因子；
- 日线基本指标；
- 涨跌停价格；
- 停牌；
- ST 状态；
- 交易日历。

通常需要设置：

```powershell
$env:TUSHARE_TOKEN = "你的 token"
```

## 3. 最快使用方法

### 3.1 自然语言触发

在 Codex 中直接输入：

```text
扫描最新市场异常
```

当上下文明确指向 Factor Forge 时，Skill 应自动触发。

### 3.2 显式指定 Skill

推荐使用：

```text
$scan-market-anomalies 扫描最新市场异常
```

其他示例：

```text
$scan-market-anomalies 看一下最新市场有没有异常的量价关系
```

```text
$scan-market-anomalies 扫描最新市场，只汇报通过质量门的异常和关系漂移
```

```text
$scan-market-anomalies 检查数据是否最新，然后运行十模板扫描
```

### 3.3 直接运行 CLI

```powershell
python -m factor_forge.cli anomaly-scan latest `
  --config configs/radar/latest_market_scan_v1.yaml `
  --data-version latest
```

`--sync` 默认开启，因此通常不需要显式填写。

## 4. “最新市场”的严格定义

“最新”不是简单读取数据库里发布时间最晚的文件。

系统现在将最新市场定义为：

```text
交易所日历已经确认开市
+ 当前时间已经超过数据就绪时间
+ 必要日线端点已经补齐
+ 末日覆盖质量通过
+ 已合并进完整历史版本
= 可以扫描的最新交易日
```

### 4.1 默认数据就绪时间

默认配置：

```yaml
data_ready_after: "18:00"
timezone: Asia/Shanghai
```

含义：

| 当前场景 | 预期最新交易日 |
|---|---|
| 交易日 18:00 前 | 上一个已完成交易日 |
| 交易日 18:00 后 | 当日 |
| 周末 | 最近一个交易日 |
| 节假日 | 节前最近一个交易日 |

18:00 是日线数据就绪保护时间，不是交易所收盘时间。其目的是避免把收盘后尚未完整更新的数据当成完整日线。

### 4.2 complete 与 incremental

系统区分两种数据版本：

```text
incremental：只包含新补的一个或多个交易日
complete：包含研究所需完整历史的数据版本
```

`latest` 只解析质量通过的最新 `complete` 版本。

即使一个 `incremental` 版本发布时间更晚，也不会被扫描器当成 `latest`。增量数据必须先与此前完整版本合并，生成新的不可变 `complete` 版本。

`latest_any` 仅用于数据摄取诊断，不应用于研究扫描。

### 4.3 freshness 状态

批次报告中的 `freshness.status` 可能为：

| 状态 | 含义 | 是否继续扫描 |
|---|---|---:|
| `CURRENT` | 预期交易日、数据截止日和末日质量全部一致 | 是 |
| `STALE_OR_INCOMPLETE` | 数据滞后、提前使用未就绪日或末日覆盖不合格 | 否 |
| `PINNED` | 用户固定了历史日期或数据版本 | 是，但不代表现实最新 |

## 5. 完整执行流程

最新扫描的实际执行顺序如下：

```text
用户调用 $scan-market-anomalies
        │
        ▼
读取 latest_market_scan_v1.yaml
        │
        ▼
查询交易所交易日历
        │
        ▼
按 Asia/Shanghai + 18:00 计算 expected_latest_trade_date
        │
        ▼
解析最新 complete 数据版本
        │
        ├── 数据落后 ──► 拉取缺失交易日
        │                  │
        │                  ▼
        │             发布 incremental
        │                  │
        │                  ▼
        │             合并完整历史
        │                  │
        │                  ▼
        │             发布新 complete
        │
        ▼
检查末日行数、可交易数、流动池和关键字段缺失率
        │
        ├── freshness 失败 ──► 阻断并报告原因
        │
        ▼
运行 8 个无标签事件模板
        │
        ▼
运行 2 个关系漂移模板
        │
        ▼
质量门过滤 + 当前事件 Jaccard 去重 + 优先级排序
        │
        ▼
生成 scan_summary.json 和 report.md
        │
        ▼
Skill 只读取结构化摘要并汇报一个下一步动作
```

自动同步是“扫描触发时同步”，不是操作系统后台定时任务。没有调用扫描命令时，它不会自行运行。

## 6. 命令参数与常见运行方式

### 6.1 查看帮助

```powershell
python -m factor_forge.cli anomaly-scan latest --help
```

### 6.2 默认最新扫描

```powershell
python -m factor_forge.cli anomaly-scan latest
```

等价于：

```powershell
python -m factor_forge.cli anomaly-scan latest `
  --config configs/radar/latest_market_scan_v1.yaml `
  --data-version latest `
  --sync
```

用途：日常最新市场扫描。

### 6.3 不补数，只检查 freshness

```powershell
python -m factor_forge.cli anomaly-scan latest --no-sync
```

注意：`--no-sync` 不等于完全离线。系统仍会查询交易日历来判断数据库是否过期。

如果数据库滞后，命令会被 freshness gate 阻断，而不是静默扫描旧数据。

用途：

- 检查本地数据是否已由其他任务同步；
- 诊断同步和扫描的责任边界；
- 避免当前命令主动拉行情数据。

### 6.4 固定数据版本

```powershell
python -m factor_forge.cli anomaly-scan latest `
  --data-version data_v1_20260710T115133Z_208e9fc8
```

固定版本时：

- 不自动同步；
- freshness 状态为 `PINNED`；
- 输出可重复；
- 适合审计或复现实验。

### 6.5 历史截止日扫描

```powershell
python -m factor_forge.cli anomaly-scan latest `
  --data-version data_v1_20260710T115133Z_208e9fc8 `
  --as-of 20260630
```

历史扫描时：

- 只使用 `2026-06-30` 及之前的数据；
- 不触发实时补数；
- freshness 标记为 `PINNED`；
- PIT rolling percentile 仍只使用事件日前历史。

推荐同时固定数据版本，避免未来数据版本变化影响历史复现入口。

### 6.6 自定义批次配置

```powershell
python -m factor_forge.cli anomaly-scan latest `
  --config configs/radar/latest_market_scan_v1.yaml
```

生产扫描不应临时修改模板或阈值。需要修改时，应复制为新模板、产生新的定义哈希，并作为独立研究版本验证。

## 7. 十个冻结模板

### 7.1 八个无未来标签事件模板

| 模板 | 核心定义 | 研究问题 |
|---|---|---|
| `price_drop_without_volume_confirmation_v1` | 3 日收益历史分位 ≤ 5%，成交量分位 ≤ 50% | 抛压有限还是流动性跳跃？ |
| `volume_surge_without_price_impact_v1` | 成交量分位 ≥ 95%，绝对收益分位 ≤ 50% | 大量换手为什么没有推动价格？ |
| `high_turnover_low_displacement_v1` | 换手分位 ≥ 90%，条件位移残差分位 ≤ 10% | 承接吸收、派发还是正常高流动性？ |
| `low_liquidity_large_displacement_v1` | 绝对收益分位 ≥ 95%，成交额分位 ≤ 30% | 流动性真空还是信息冲击？ |
| `long_lower_wick_strong_close_v1` | 下影分位 ≥ 90%，收盘位置 ≥ 0.70，近期收益偏弱 | 是否存在盘中承接？ |
| `stock_industry_divergence_v1` | 行业、规模、波动控制后残差处于双尾 5% | 独立信息还是暂时偏离行业？ |
| `volatility_compression_breakout_v1` | ATR 和近期振幅分位 ≤ 10%，突破强度分位 ≥ 90% | 压缩后的突破是延续还是假突破？ |
| `trend_exhaustion_v1` | 长周期趋势极端但短周期速度明显衰减或改善 | 趋势是否正在失速或修复？ |

这些模板只输出当时可观察到的事件及严重度。`ObservationCard` 明确禁止：

```text
forward_return
future_return
target
label
rank_ic
icir
sharpe
```

### 7.2 Feature—未来收益关系漂移

模板：`feature_return_relation_drift_v1`

监控关系：

```text
lower_shadow_ratio → forward_return_5
volume_price_efficiency → forward_return_5
industry_relative_return_5d → forward_return_10
```

窗口：

```text
recent  = 60 个有效交易日
medium  = 252 个有效交易日
baseline = 756 个有效交易日
```

检测规则：

```text
robust_delta_zscore 阈值 = 2.5
最小持续天数 = 20
最近最少有效天数 = 40
日截面最少样本 = 500
```

该模板允许使用未来收益，但只使用已经成熟的标签。报告中的 `effective_as_of_date` 会落后于扫描日 5 或 10 个交易日，未成熟尾部不会进入关系判断。

同时控制：

- 市场方向；
- 市场波动；
- 市场宽度；
- 流动性状态。

### 7.3 变量之间的关系漂移

模板：`variable_relation_drift_v1`

监控关系：

```text
turnover_rate ↔ abs_return_1d
stock_return_1d ↔ industry_return_1d
volatility_20d ↔ short_reversal_1d
```

窗口：

```text
recent  = 60
medium  = 252
baseline = 504
```

检测规则：

```text
CUSUM 阈值 = 2.5
最小持续天数 = 15
最近最少有效天数 = 40
日截面最少样本 = 500
```

变量关系漂移使用同期变量，不依赖未来标签。

## 8. 输出目录与文件

### 8.1 批次总报告

```text
artifacts/market_anomaly_scans/<scan_id>/
  manifest.json
  report.md
  scan_summary.json
```

文件用途：

| 文件 | 用途 |
|---|---|
| `scan_summary.json` | Skill 和程序读取的机器可读总摘要 |
| `report.md` | 人工快速阅读报告 |
| `manifest.json` | 运行身份、数据版本和不可变产物登记 |

### 8.2 事件 ObservationCard

```text
artifacts/radar_observations/<observation_id>/
  observation_card.json
  events.parquet
  manifest.json
```

`events.parquet` 包含事件日、股票、严重度及模板测量字段，但不包含未来收益。

### 8.3 关系 DriftCard

```text
artifacts/radar_drifts/<drift_id>/
  drift_card.json
  relation_series.parquet
  manifest.json
```

### 8.4 缓存语义

相同的以下组合会复用不可变产物：

```text
模板定义哈希
+ 完整 data_version
+ as_of_date
```

即使命中缓存，最新扫描仍会先执行 freshness 检查，避免因为旧缓存而忽略数据库已经落后。

## 9. 如何阅读扫描结果

### 9.1 先看 freshness

必须首先确认：

```json
{
  "status": "CURRENT",
  "expected_latest_trade_date": "2026-07-10",
  "data_end_date": "2026-07-10",
  "synchronized": false,
  "failures": []
}
```

需要重点理解的字段：

| 字段 | 含义 |
|---|---|
| `expected_latest_trade_date` | 按现实交易日历和数据就绪时间应当具备的交易日 |
| `data_end_date` | 完整版本实际覆盖的最后日期 |
| `synchronized` | 本次扫描前是否真的执行了补数 |
| `incremental_versions` | 本次补数产生并被合并的增量版本 |
| `last_day_rows` | 最后交易日股票行数 |
| `last_day_tradeable` | 最后交易日可交易股票数 |
| `last_day_liquid` | 最后交易日流动池数量 |
| `required_missing_rates` | 关键字段最后交易日缺失率 |
| `failures` | freshness 阻断原因 |

### 9.2 再看事件质量门

事件模板主要检查：

- 总事件数；
- 唯一股票数；
- 唯一行业数；
- 最大行业占比；
- 最大单股贡献；
- 测量字段缺失率；
- PIT 时间审计。

`quality_gate_passed=false` 的模板会保留在报告中用于审计，但不会进入 highlights。

质量门失败不等于程序运行失败。例如：

```text
low_liquidity_large_displacement_v1: event_count<300
```

它表示历史样本不足以支持当前模板进入重点研究，而不是数据同步失败。

### 9.3 当前事件字段

| 字段 | 含义 |
|---|---|
| `scan_date_event_count` | 最新交易日触发股票数 |
| `scan_date_event_rate` | 最新交易日触发比例 |
| `rolling_event_rate_zscore` | 当前事件发生率相对滚动历史的偏离程度 |
| `recent_event_rate` | 最近窗口事件发生率 |
| `historical_event_rate` | 较长历史窗口事件发生率 |
| `event_rate_ratio` | 最近发生率 ÷ 历史发生率 |
| `unique_stocks` | 历史事件涉及股票数量 |
| `unique_industries` | 历史事件涉及行业数量 |
| `severity_p90` | 严重度的 90% 分位 |

不要只看 `event_rate_ratio`。历史发生率很低的时候，少量事件也可能产生很大的倍数。

### 9.4 highlights 排序

当前优先级综合使用：

```text
abs(rolling_event_rate_zscore)
abs(log(event_rate_ratio))
```

取两者较大值，并且只保留：

- 最新交易日事件数大于 0；
- 质量门通过；
- 没有被当前事件集合去重的模板。

### 9.5 当前事件去重

系统比较不同模板在最新交易日触发的股票集合。

若 Jaccard 相似度达到：

```yaml
dedup_jaccard_threshold: 0.70
```

较低优先级模板会标记 `duplicate_of`，避免同一批股票因为相近定义重复占据 highlights。

### 9.6 如何阅读关系漂移

重要字段：

| 字段 | 含义 |
|---|---|
| `baseline_mean` | 历史基准关系均值 |
| `medium_mean` | 中期关系均值 |
| `recent_mean` | 最近关系均值 |
| `delta` | 最近减去基准 |
| `robust_delta_zscore` | 稳健标准化后的变化强度 |
| `cusum_score` | 累积变化强度 |
| `persistence_days` | 变化连续满足条件的天数 |
| `valid_days_recent` | 最近窗口有效样本天数 |
| `is_drift` | 是否同时通过阈值、持续性和质量要求 |
| `direction` | 关系增强或减弱，不是股票涨跌方向 |
| `effective_as_of_date` | 关系证据真正成熟到的日期 |

即使 z-score 很高，只要持续性不足，`is_drift` 仍应为 `false`。

## 10. Skill 会怎样回答

Skill 的固定回答结构是：

1. 扫描身份和 freshness；
2. 最多 5 个通过质量门的当前异常；
3. 质量门失败模板；
4. 被确认的关系漂移；
5. 研究边界与唯一下一步。

示例格式：

```text
扫描完成
- 预期最新交易日：2026-07-10
- 数据截止日：2026-07-10
- freshness：CURRENT
- 数据版本：data_v1_...
- 本次是否补数：否

当前异常
1. 急跌未放量：64 个事件，近期/历史发生率 2.67
2. 高换手低位移：44 个事件，滚动 z-score 0.66

质量门失败
- 低成交大位移：总事件数不足

关系漂移
- 换手率—绝对收益关系减弱，持续 45 日

边界：以上是研究线索，不是 Alpha 或交易建议。
下一步：检查换手—价格冲击漂移的竞争性解释。
```

## 11. 失败与故障排查

### 11.1 `No complete published data version exists`

含义：数据库中没有可供研究使用的完整历史版本。

处理：先完成全历史数据初始化和质量发布，再运行最新同步。不要把单日增量强行标记为完整版本。

### 11.2 `freshness gate blocked anomaly scan: data_end<...`

含义：数据库最后日期落后于交易日历预期日期，并且自动同步没有成功补齐。

检查：

- Tushare Token；
- 数据端点权限；
- 网络连接；
- 最新 ingestion run 的错误信息；
- 缺失日期对应的 staging 分区。

### 11.3 `data_end>..._before_ready_cutoff`

含义：当前版本已经包含今天数据，但按配置时间今天的数据还不应被视为完整日线。

处理：

- 等待 18:00 后重新扫描；或
- 固定到上一个完整交易日进行历史扫描。

不要通过关闭 freshness gate 来使用疑似未完成数据。

### 11.4 `last_day_rows<1000`

含义：最后交易日股票行数异常偏少。

可能原因：

- 单个端点只返回部分市场；
- 当日分区未完成；
- 合并版本引用错误；
- 数据供应商尚未完成更新。

### 11.5 关键字段缺失率超过 5%

默认检查：

```text
adj_open
adj_close
amount_cny
turnover_rate
```

任何字段最后交易日缺失率超过 `0.05` 都会阻断扫描。

### 11.6 `event_count<...`

这是模板质量门失败，不是扫描故障。

正确处理：保留记录但不高亮，不应为了让它通过而临时降低阈值。

### 11.7 `cached=true`

表示相同模板、数据版本和截止日期已经生成过不可变产物。

它不表示跳过 freshness；freshness 仍在读取缓存前检查。

### 11.8 首次扫描耗时较长

新数据版本首次运行需要计算逐股票 PIT rolling percentile 和截面残差，可能耗时数分钟。

相同版本重复扫描会复用缓存。不要为了加速而：

- 缩短必要历史窗口；
- 降低质量门；
- 使用未来数据预计算；
- 把增量版本直接当完整版本。

## 12. 研究边界与后续流程

扫描阶段：

```text
最新完整数据
→ 无标签 ObservationCard
→ 异常频率与覆盖检查
→ 人工选择研究问题
```

如果某个事件值得继续，下一阶段应单独申请：

```text
冻结 ObservationCard
→ 匹配对照
→ 3/5/10 日成熟标签
→ 递进 Event Study
→ 决策 Gate
```

只有 Event Study 支持后，才讨论：

```text
异常机制
→ 可测量 Feature
→ 连续 Factor 或模型增量
→ 回测
→ 前向观察
```

Skill 不会自动跨越这些阶段。

关系漂移尤其需要注意：DriftCard 描述的是市场级关系变化，不天然对应一组个股事件，因此不能机械地送入个股 Event Study。

## 13. 配置与维护规则

主配置：

```text
configs/radar/latest_market_scan_v1.yaml
```

关键配置：

```yaml
dedup_jaccard_threshold: 0.70
max_highlights: 5

freshness:
  auto_sync: true
  require_current: true
  data_ready_after: "18:00"
  min_last_day_rows: 1000
  min_last_day_tradeable: 500
  min_last_day_liquid: 500
  max_required_missing_rate: 0.05
```

维护原则：

1. 日常扫描不得临时改阈值。
2. 修改模板必须生成新的定义哈希。
3. 不得覆盖旧 ObservationCard、DriftCard 或数据版本。
4. 新增模板时必须明确它属于事件异常还是关系漂移。
5. ObservationCard 永远不能加入未来标签。
6. Feature—收益漂移必须执行标签成熟截断。
7. freshness gate 失败时不得回退到任意旧版本继续“最新扫描”。

Skill 文件：

```text
C:\Users\Junko\.codex\skills\scan-market-anomalies\SKILL.md
```

修改 Skill 后运行：

```powershell
$env:PYTHONUTF8 = "1"
python C:\Users\Junko\.codex\skills\.system\skill-creator\scripts\quick_validate.py `
  C:\Users\Junko\.codex\skills\scan-market-anomalies
```

## 14. 常见问题

### Q1：直接说“扫描最新市场异常”可以吗？

可以。显式写 `$scan-market-anomalies` 更稳定，也更容易确认使用了指定 Skill。

### Q2：Skill 会把原始 K 线发给模型吗？

不会。确定性 Python 代码计算异常，Skill 只读取 `scan_summary.json` 和 `report.md`。

### Q3：数据库已经最新，还会访问数据供应商吗？

会查询交易日历以确认现实应有交易日；只有发现缺口时才拉取缺失行情端点。

### Q4：`--no-sync` 能完全离线运行吗？

不能。它仍需要交易日历来做 freshness 判断。完全固定复现应指定明确 `--data-version` 和 `--as-of`。

### Q5：异常发生率上涨就代表可以买入吗？

不代表。发生率只说明这种市场现象变多，没有说明未来收益方向。

### Q6：关系变弱意味着做空吗？

不意味着。“strengthening/weakening”描述的是统计关系变化，不是未来价格方向。

### Q7：为什么 Feature—收益关系的有效日期早于扫描日期？

因为未来 5/10 日收益需要等待标签成熟。未成熟尾部被明确排除。

### Q8：可以自动把最显著异常送去 Event Study 吗？

当前不可以。必须由人明确授权，避免自循环不断污染验证样本。

### Q9：如何查看最近一次报告？

在 `artifacts/market_anomaly_scans/` 下找到最新的 `<scan_id>` 目录，优先阅读：

```text
report.md
scan_summary.json
```

### Q10：如何确认 Skill 本身有效？

```powershell
$env:PYTHONUTF8 = "1"
python C:\Users\Junko\.codex\skills\.system\skill-creator\scripts\quick_validate.py `
  C:\Users\Junko\.codex\skills\scan-market-anomalies
```

预期输出：

```text
Skill is valid!
```
