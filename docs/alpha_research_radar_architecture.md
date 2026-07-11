# Factor Forge 异常驱动研究架构审视与落地方案

版本：0.1  
日期：2026-07-11  
状态：设计提案，不改变现有 Factor Forge V1 语义

## 1. 执行结论

对话中的总体方向合理，但需要收缩目标和调整实施顺序。

建议把系统定义为：

> 一个受约束的 Alpha Research Radar：用确定性程序发现值得研究的异常和关系漂移，用结构化实验证伪解释，用 LLM 降低编排和总结成本。

不建议把系统定义为“自主发现 Alpha 的 Agent”。后者会把最难的问题——多重检验、样本污染、非平稳性和交易可行性——错误地包装成模型能力问题。

建议保留并复用现有 Factor Forge 的数据版本、PIT 约束、DSL、评估、回测、评分和产物体系，只新增一层薄的研究控制面。第一阶段不需要 LLM，不需要知识图谱，不需要微服务，也不需要重构现有 Factor YAML。

最终目标分三层：

1. **研究雷达**：发现关系异常、事件异常和近期关系漂移。
2. **研究控制面**：记录 Idea、Hypothesis、Trial、Decision 和搜索预算，生成可复现实验。
3. **AI 适配层**：用便宜模型对异常卡片分类、提出竞争假设、映射已有 Feature、总结结果；高级模型只做少量升级审查。

## 2. 对话内容的有效压缩

整段讨论可以压缩为一条研究链：

```text
市场数据
  -> 确定性观察（异常 / 背离 / 漂移）
  -> 可证伪问题
  -> 竞争假设
  -> 区分性 Feature
  -> 递进实验
  -> 规则 Gate
  -> Reject / Observe / Candidate
  -> 发现日之后的前向验证
```

各对象的职责应明确区分：

| 对象 | 定义 | 不应承担的职责 |
|---|---|---|
| Observation | 当期数据相对历史、同业或条件分布的偏离 | 不直接声称未来有收益 |
| Drift | 某个分布、关系或预测性能相对历史基线发生持续变化 | 不等同于已知 Regime |
| Hypothesis | 对 Observation/Drift 的可证伪解释 | 不只是一个漂亮故事 |
| Feature | 描述异常或区分竞争解释的可测变量 | 不必天然具有交易方向 |
| Factor | 输出可排序交易分数的规则或模型 | 不负责保存所有研究知识 |
| Experiment | 在冻结配置下验证一个具体问题 | 不应无边界搜索参数 |
| Decision | 对一次研究分支的唯一下一步判断 | 不由 Sharpe 单指标决定 |

这里最重要的修正是：

> 不是“用因子拟合异常”，而是“用 Feature 区分异常背后的机制，再检验其中是否存在对未来超额收益的独立增量”。

## 3. 哪些判断合理

### 3.1 合理且应保留

- 传统技术变量没有消失，但更适合作为原料、条件和上下文，而非直接交易规则。
- Feature 研究和模型组合不是二选一；模型不能创造输入中不存在的信息。
- 原始市场数据的统计计算应由 Python 完成，LLM 只读取压缩后的结构化证据。
- 分位适合作为异常检测的统一第一层表示，尤其适合跨股票、跨时期比较。
- 关系异常通常比单变量极值更值得研究，例如“在这种跌幅下，成交量为何异常低”。
- 异常必须有竞争假设、匹配对照和已知暴露控制。
- 近期才出现的关系需要 Change/Drift 流程和发现日之后的顺序验证，不能用全历史平均掩盖。
- Skill 应定义固定协议，Research Core 应提供有限工具，LLM 不直接操作数据库和大文件。
- 成功结果详细保存，失败结果只保存足以避免重复踩坑的结论。

### 3.2 公开方法依据

这不是纯粹的架构想象。它组合了已有的事件研究、条件收益、异常检测、概念漂移、公式因子搜索和确定性回测方法。自动公式搜索、质量多样性、受限 DSL、家族去重、正负经验反馈等方向已有公开研究；金融概念漂移和回测多重检验也有成熟警告与方法。

但“异常扫描 -> LLM 竞争解释 -> Feature 映射 -> 自动实验 -> 长期实盘 Alpha”的完整闭环还不能视为行业已验证标准。它应当作为一个待验证的研究生产率假设，而不是收益假设。

## 4. 对话中需要修正或降级的部分

### 4.1 不能在异常发现阶段使用未来收益评分

对话中的某些异常卡片把 `forward_return`、对照组收益差和 OOS 效果直接放进异常发现结果。如果扫描器据此排序异常，就已经把预测标签用于发现，形成选择偏差。

必须分成两个产物：

```text
ObservationCard（无未来标签）
  -> 注册并冻结 definition_hash / discovered_at
  -> EventStudyResult（允许读取未来标签）
```

ObservationCard 只能包含当时可见的信息：异常定义、强度、频率、覆盖度、集中度、数据质量和相对基准。未来收益只属于后续实验。

### 4.2 “分位异常”不是完整方法

边际分位只能回答“这个值相对自身历史是否极端”，不能回答“在给定其他变量后是否反常”。推荐三层递进：

1. 边际分位：`pct(x | own_history)`；
2. 同类分位：`pct(x | date, industry, size/liquidity bucket)`；
3. 条件残差：`x - E[x | return, volatility, liquidity, regime]`。

第一版使用滚动经验分位和简单分箱残差即可，不必马上上 Isolation Forest、KDE 或深度异常检测。

### 4.3 Regime 与近期漂移必须分流

应维护两条不同研究流：

- **Event Research**：某类股票事件在不同日期反复出现，研究条件收益。
- **Relation Drift**：变量关系本身近期发生变化，研究结构突变和持续性。

Drift 先用已知 Regime 模型解释，只有其残差仍持续异常，才能称为“未知状态无法解释的关系变化”。即便如此，也只能进入前向观察，不能立即当作长期静态因子。

### 4.4 “Feature 永远保存原始值”过于绝对

现有 Factor Forge 的 Factor YAML 已经同时包含中间 Feature、公式、winsorize 和 standardize；组合层也有独立预处理。立刻强制拆出全局 Feature Store 会产生大规模迁移，却未必提高研究质量。

建议采用渐进边界：

- 具有稳定金融语义、可跨项目复用的计算，注册为 `Measurement`；
- 去极值、截面 Rank、ZScore、中性化、缺失值策略保留在 Experiment/Factor pipeline；
- 只有多次复用且语义稳定的派生量，才提升为注册 Measurement；
- 现有 Factor YAML 继续作为可执行事实来源，后续由 adapter 引用 Measurement，而不是一次性重写。

### 4.5 “Feature Budget 每个 Behavior 最多 20 个”不应写死

固定 20 没有统计依据。真正需要控制的是有效搜索自由度：

- 每个 Idea 首轮最多 8 个输入 Measurement；
- 最多 5 个递进实验；
- 最多 2 次基于验证结果的修改；
- 每次修改必须记录新增自由度和理由；
- 预算耗尽后必须归档或由人工升级。

### 4.6 知识图谱、向量库和多 Agent 均不应进入第一阶段

当前最缺的不是语义检索，而是统一的运行登记、实验谱系、搜索次数和冻结边界。SQLite 的规范化表、JSON 产物和全文检索足够支撑第一版。只有当人工搜索真实成为瓶颈时，再增加 embedding；图数据库没有近期必要性。

### 4.7 便宜模型不应决定统计结论

便宜模型适合分类、映射、压缩和提出候选解释，不适合：

- 计算显著性；
- 判断数据泄漏；
- 决定实盘准入；
- 自由生成复杂公式；
- 根据重复回测无限优化。

模型输出必须经过 JSON Schema 校验；最终 Gate 由确定性代码执行。

## 5. 对现有 Factor Forge 的审视

### 5.1 已有能力，应直接复用

当前仓库已经包含：

- 不可变数据版本、manifest 和 PIT 行业/状态数据；
- YAML + 受限 DSL 的因子定义和 lookback 静态检查；
- L0 质量、L1 IC/分层/OOS/BH-FDR、L2 T+1 可交易回测、L3 稳健性；
- 固定执行约束、成本网格、贡献集中度和暴露诊断；
- 因子组合、条件 IC、ML、Regime、factor state/reliability 等专项管线；
- 不可变 run artifact 和 manifest；
- Web dashboard 与大量专项研究脚本。

因此，不能另起一套 Backtest Engine、Feature Engine 或 Artifact Store。新架构应通过 adapter 调用这些能力。

### 5.2 当前核心缺口

1. **研究运行没有统一索引**：产物分散在多个 `artifacts/*` 目录，manifest 结构不完全一致。
2. **缺少全局 Trial Ledger**：单次运行记录了组合数，但没有按 Idea/Family 累积所有尝试。
3. **缺少不可绕过的封存边界**：配置可声明 train/valid/test，但没有统一权限层阻止生成器读取 sealed 数据。
4. **Idea/Hypothesis/Decision 没有一等对象**：研究逻辑主要存在于 YAML 描述、报告和文件名中。
5. **研究能力碎片化**：专项脚本和 ML 子管线难以被统一发现和编排。
6. **缺少 Observation/Drift 标准产物**：目前擅长验证一个给定因子，不擅长产出无标签异常卡片。
7. **缺少发现时点**：对“近期新生关系”，必须知道系统第一次看到它的时间，才能定义真正前向期。

### 5.3 不建议现在重构的部分

- 不改 Factor Forge V1 的 DSL 和因子评估语义；
- 不合并 Factor 与 ML 两条现有管线；
- 不迁移已有 artifact；
- 不把行情和特征矩阵写入 SQLite；
- 不先建立统一大而全 Feature Store；
- 不把专项研究脚本一次性改造成插件。

## 6. 优化后的目标架构

```text
Immutable Market Data / PIT Panel
              |
              v
  +---------------------------+
  | Deterministic Radar       |
  | percentile / residual /   |
  | event / relation / drift  |
  +-------------+-------------+
                |
                v
       ObservationCard (no y)
                |
        register + freeze
                |
                v
  +---------------------------+
  | Research Control Plane    |
  | Idea / Hypothesis / Trial |
  | Budget / Lineage / Gate   |
  +------+------+-------------+
         |      |
         |      +------> Cheap LLM Skill Adapter
         |               classify / explain / map
         v
  Existing Factor Forge Adapters
  factor / combination / ML / backtest
         |
         v
  Validation artifacts + deterministic Gate
         |
         v
  Reject / Watch / Candidate / Forward Monitor
```

### 6.1 新增模块建议

```text
src/factor_forge/
  research_control/
    models.py              # Idea/Hypothesis/Trial/Decision schema
    store.py               # 独立 research.sqlite3
    lineage.py             # run/artifact 统一索引
    budget.py              # 搜索预算和状态机
    gates.py               # 确定性准入规则
    adapters.py            # 调用现有 factor/ml/专项 runner

  radar/
    models.py              # ObservationCard/DriftCard schema
    percentiles.py         # PIT rolling percentile
    residuals.py           # 条件/同类残差
    event_scanner.py       # 事件型异常
    relation_monitor.py    # rolling IC/beta/conditional rate
    change_detection.py    # robust z/CUSUM/persistence
    dedup.py               # 定义相似度 + 事件重叠度
    writer.py              # 不可变卡片产物

  research_api/
    service.py             # Python service facade，先不建 HTTP
    schemas.py             # 稳定输入输出
```

Skill 放在 `.claude/skills`、Codex skill 或其他宿主层，不进入数值核心。先提供 CLI/Python 工具，确有跨进程需求再加 HTTP/MCP。

### 6.2 独立控制库

研究控制元数据建议使用 `data/research.sqlite3`，不要混入当前 `metadata.sqlite3`。后者是数据版本和 PIT 维表的控制面，生命周期和权限不同。

最小表：

```text
research_idea
research_hypothesis
observation_card
experiment_plan
trial_run
research_decision
research_budget
artifact_index
sealed_access_audit
```

最小关系：

```text
Idea 1---N Hypothesis
Idea 1---N ObservationCard
Hypothesis 1---N ExperimentPlan
ExperimentPlan 1---N TrialRun
TrialRun 1---1 Decision
Idea/Family 1---1 Budget
```

`trial_run` 只索引现有 artifact，不复制大型结果。

## 7. 三类雷达，不应混成一个算法

### 7.1 Event Anomaly Radar

目标：寻找个股或行业在当日相对历史/同类的反常事件。

第一版输入：OHLCV、行业、市值、流动性、可交易状态。

第一版只做 8 个关系模板：

1. 急跌但成交确认不足；
2. 放量但价格冲击不足；
3. 缩量上涨；
4. 极端下影与强/弱收盘；
5. 个股相对行业强弱背离；
6. 波动扩张但收盘位置反常；
7. 连续行情中速度衰减；
8. 流动性投入与价格结果不匹配。

模板必须预注册，阈值只能来自小规模固定集合。

### 7.2 Relation Monitor

目标：持续监控固定关系的滚动变化，而非自由搜索所有变量对。

首批关系：

- Feature -> forward residual return 的 rolling RankIC；
- price ↔ volume 的滚动条件残差；
- stock ↔ industry 的残差强度；
- volatility ↔ reversal 的条件收益；
- 已上线因子的 rolling IC、TopN excess 和 turnover health。

所有 target 关系只能在验证模块计算，Radar 的实时告警使用截至当时已经完全成熟的标签，必须按 horizon 延迟。

### 7.3 Emerging Relation / Drift Radar

目标：识别近期持续变化，并排除已知 Regime 解释。

第一版流程：

```text
long baseline vs recent window
  -> robust delta z-score
  -> persistence filter
  -> cross-sectional coverage
  -> known-regime expected relation
  -> residual drift
  -> placebo historical start dates
  -> register discovered_at
  -> forward-only confirmation
```

第一版只使用 rolling difference、稳健 Z、CUSUM 和持续性过滤。BOCPD/HMM 可作为后续对照，不应成为 MVP 依赖。

## 8. 两张卡片，禁止混用

### 8.1 ObservationCard：无未来标签

```yaml
schema_version: 1
observation_id: obs_...
definition_id: relation_price_volume_v1
definition_hash: sha256:...
discovered_at: 2026-07-11T...
data_version: data_v1_...
as_of_date: 2026-07-10
observation_type: relation_anomaly
entity_scope: stock
conditions:
  return_3d_pct_252_lte: 0.05
  volume_residual_pct_252_lte: 0.20
evidence:
  event_count: 317
  unique_stocks: 241
  industry_coverage: 0.81
  recent_event_rate: 0.014
  historical_event_rate: 0.006
  severity_median: 2.3
quality:
  stale_data: false
  corporate_action_issue: false
  limit_event_share: 0.04
status: registered
```

禁止包含：forward return、Sharpe、IC、对照组未来收益差。

### 8.2 ResearchCard：允许验证标签

```yaml
schema_version: 1
idea_id: idea_...
observation_id: obs_...
hypotheses:
  - id: limited_selling_pressure
  - id: liquidity_vacuum
  - id: ordinary_reversal
candidate_measurements:
  - down_speed_3d
  - volume_given_return_residual
  - lower_shadow_ratio
  - amihud_20
  - industry_relative_return
experiment_sequence:
  - event_vs_matched_control
  - severity_monotonicity
  - control_reversal
  - control_liquidity
  - regime_interaction
budget:
  max_trials: 5
  max_revisions: 2
status: ready
```

LLM 可以建议 ResearchCard，但不能修改 ObservationCard 的冻结证据。

## 9. 实验设计：每一步只回答一个问题

### E0 数据与定义审计

- 除权复权、停牌、涨跌停、新股、ST、数据缺失；
- 事件定义 PIT；
- 事件样本不使用未来可得信息；
- 标签成熟延迟正确。

失败立即停止。

### E1 事件相对匹配对照是否有差异

匹配优先级：同日期、同行业、相似前期收益、波动、流动性和市值。输出均值、中位数、胜率、MAE/MFE、分布差和贡献集中度。

### E2 强度是否具有宽阈值单调性

比较预注册的 3 档强度或 Q1-Q5，不允许看到结果后任意移动阈值。若只在一个狭窄点有效，降级。

### E3 竞争解释的递进控制

按假设顺序一次只加一组控制：

```text
baseline prior return
  -> + relation residual
  -> + price-action confirmation
  -> + liquidity
  -> + industry/market context
```

重点是增量与解释消失点，不是完整模型的绝对表现。

### E4 Regime 主效应与交互

比较：

```text
y ~ regime
y ~ regime + anomaly
y ~ regime + anomaly + anomaly:regime
```

输出必须区分跨状态 Alpha、条件 Alpha、纯状态现象。

### E5 交易验证

只有 E1-E4 通过才进入现有 L2/L3：TopN、T+1、成本、换手、容量、年度、贡献集中和稳定性。

### E6 ML 增量

最后比较：

```text
known baseline model
vs
known baseline + anomaly measurements
```

不以完整 LightGBM 的表现替代新信息增量。

## 10. 防止系统自我欺骗的硬约束

### 10.1 全局 Trial Ledger

必须累计记录：

- Idea 和 factor family 的所有试验；
- 阈值、窗口、方向、股票池、horizon、pipeline 的每次变化；
- 谁/哪个模型看过哪些结果；
- 哪个数据区间参与过生成或修改；
- 失败和取消的运行，不能只登记成功结果。

现有 run manifest 中的 `total_combinations_tested` 应汇入全局账本，但不能替代它。

### 10.2 数据角色必须由基础设施授权

```text
discovery      可供扫描和生成
validation     可供有限修改，访问即计入预算
sealed_test    生成器不可访问，只能由人工批准的一次性 gate 读取
forward        发现日之后自然累积，不回填成历史 OOS
```

仅在 YAML 写日期不够。DataView 应按角色裁剪列和日期；sealed access 要有审计记录。

### 10.3 多重检验按“研究家族”计算

现有 BH-FDR 对单次实验有价值，但自动研究需要跨运行的 family-level 试验计数。第一版至少做到：

- family_id；
- cumulative_trials；
- validation_peeks；
- 预注册 primary metric；
- BH-FDR/White Reality Check/DSR 等结果可后续逐步加入；
- 报告永远展示搜索宽度，不能只展示获胜配置。

### 10.4 唯一下一步动作

Gate 只允许：

```text
reject
observe_forward
revise_one_hypothesis
promote_candidate
retire
```

不得由 LLM 一次展开十几个后续实验。

## 11. Skill 与模型路由

### 11.1 第一版只需要两个 Skill

`observation_to_research`：

1. 读取一张冻结 ObservationCard；
2. 搜索已有 Measurement/Factor/Idea；
3. 输出 2-4 个竞争假设；
4. 每个假设引用已有 Measurement；
5. 只在确实缺失时提出一个新 Measurement 草案；
6. 生成不超过 5 个递进实验；
7. 不运行 sealed test。

`review_research_result`：

1. 读取统一 summary，而非明细数据；
2. 先应用确定性 Gate；
3. 总结哪项假设被削弱/保留；
4. 输出一个下一步动作；
5. 保存人工待确认 Decision 草案。

### 11.2 工具接口

第一版保持 8 个以内：

```text
search_research_assets
get_observation_card
create_idea
save_hypotheses
create_experiment_plan
run_trial
get_trial_summary
propose_decision
```

写接口应幂等并带 `expected_version`。模型不直接执行 SQL、不读取 parquet、不编辑正式 YAML。

### 11.3 模型升级规则

- Level 0，无模型：扫描、统计、匹配、Gate、报表、去重；
- Level 1，便宜模型：分类、枚举映射、摘要、相似研究召回；
- Level 2，中等模型：竞争假设、最小实验序列；
- Level 3，高级模型：新 Measurement 设计、矛盾结果审查、进入 paper trade 前复核。

触发升级必须满足至少一个条件：低置信度、无已有 Measurement 可表达、竞争解释结果矛盾、候选准入。模型成本要按 Idea 和通过候选分别核算。

## 12. 分阶段落地计划

### Phase 0：基线与设计冻结（3-5 天）

交付：

- 统计当前 artifact 类型、runner 和 manifest 差异；
- 定义 `ResearchRunEnvelope`，统一索引而不改原产物；
- 冻结 8 个异常模板、4 个关系监控模板；
- 定义数据角色和首个 sealed boundary；
- 写清基线对照：人工研究 vs 随机模板 vs 雷达推荐。

验收：能够回答“当前有哪些研究运行、来自哪个配置和数据版本”。

### Phase 1：Research Ledger（1-2 周）

交付：

- `research_control/models.py` 与 SQLite schema；
- artifact indexer，扫描现有 `artifacts/*/manifest.json`；
- Idea/Hypothesis/Trial/Decision CLI；
- family budget 和 validation peek 记录；
- 只读 Web 页面展示研究谱系。

验收：任一新实验都可追溯到 Idea、实验计划、配置 hash、数据版本、代码版本和累计试验数；失败运行同样登记。

停止条件：如果无法统一索引现有多类 manifest，先修索引契约，不进入雷达开发。

### Phase 2：无标签异常雷达 MVP（2-3 周）

交付：

- PIT rolling percentile；
- peer percentile 与简单条件残差；
- 8 个预注册关系模板；
- ObservationCard schema、冻结 hash、数据质量审计；
- 定义去重：definition similarity + event Jaccard；
- 每周候选报告。

验收：

- 同一 data version/config 重跑字节级或数值级可复现；
- 所有卡片不含未来标签；
- 每周最多输出 10 张非重复卡片；
- 人工抽查至少 90% 可解释且无明显数据问题；
- 从卡片到原始事件样本可审计。

停止条件：连续 4 周大多数卡片只是数据错误、涨跌停或同一现象改名，则先改模板和数据质量，不接 LLM。

### Phase 3：确定性 Event Study 与 Gate（2-3 周）

交付：

- 匹配对照；
- horizon 3/5/10 的成熟标签；
- E0-E4 递进实验；
- family-level trial 计数；
- 统一 `trial_summary.json`；
- adapter 接入现有 ExperimentRunner/ML runner。

验收：每张进入研究的卡片只有一个预注册 primary metric，每个实验回答一个问题；系统可以明确输出“普通反转解释掉”“流动性解释掉”“仅 Regime 交互”“仍有独立增量”。

### Phase 4：关系漂移雷达（2-3 周）

交付：

- 4 个固定 relation monitor；
- baseline/recent delta、robust z、CUSUM、persistence；
- known-regime residual；
- 历史 placebo start；
- `discovered_at` 和 forward monitor。

验收：回放历史时告警只使用当时可见且已成熟的数据；发现后的确认期与历史回溯证据分开展示。

### Phase 5：便宜模型 + 两个 Skill（1-2 周）

交付：

- 固定 JSON Schema；
- `observation_to_research` 与 `review_research_result`；
- prompt/skill/model/input hash 缓存；
- 低置信度升级和成本日志；
- 人工确认写操作。

验收：

- 结构化输出有效率 > 98%；
- 首轮实验计划不超过预算；
- 至少 80% 使用已有 Measurement；
- 相对人工流程，卡片到可运行实验的时间下降 50% 以上；
- 便宜模型不能绕过 Gate 或 sealed 权限。

### Phase 6：受控对照评估（至少 8-12 周）

三组使用相同数据、模板和试验预算：

- A：随机/规则模板选择；
- B：雷达 + Skill；
- C：人工研究。

主指标不是最高 Sharpe，而是：

- 每 100 个 trial 的 validation 通过数；
- 候选与现有因子的独立增量；
- forward 衰减；
- 重复率；
- 人工小时；
- 每个合格候选的 token/算力成本；
- 数据泄漏和预算违规数。

扩大投入的必要条件：B 在候选通过率或人工时间上显著优于 A，并且不劣于 C 的 forward 质量。否则保留 Ledger/Radar 作为人工辅助，不继续做自主 Loop。

## 13. MVP 的明确边界

包含：

- A 股日频 Top1000；
- OHLCV、行业、市值、流动性和交易状态；
- 8 个关系异常模板；
- 4 个关系漂移监控；
- 3/5/10 日标签；
- SQLite 控制面 + Parquet/JSON 产物；
- 现有 Factor Forge runner adapter；
- 最后阶段才接便宜模型。

不包含：

- L2/逐笔数据；
- 任意公式生成；
- 多 Agent 自循环；
- 自动实盘；
- 知识图谱；
- 向量数据库；
- 在线强化学习；
- 自动组合优化；
- 大规模自由参数搜索。

## 14. 推荐的第一批实现任务

按优先级排序：

1. `ResearchRunEnvelope` 与 artifact indexer；
2. research SQLite schema 和 trial ledger；
3. data-role/sealed-access facade；
4. ObservationCard schema；
5. PIT rolling percentile 的防泄漏单元测试；
6. 两个关系模板：急跌未放量、放量价格不动；
7. 卡片去重和每周 top-k；
8. 匹配对照 event study；
9. 递进控制和唯一动作 Gate；
10. 历史回放测试；
11. 4 个 relation monitor；
12. 最后才实现 Skill 和模型路由。

## 15. 最终评价

按不同目标评价：

| 目标 | 评价 |
|---|---|
| 提升研究可复现性 | 高可行 |
| 减少重复研究和专项脚本碎片 | 高可行 |
| 更系统地产生值得研究的问题 | 可行，需用对照实验验证效率 |
| 用便宜模型降低编排和总结成本 | 高可行，但必须结构化和限权 |
| 自动发现稳定实盘 Alpha | 不确定，不应作为立项承诺 |
| 完全自主闭环 | 当前不建议 |

这个项目最合理的商业/研究价值不是“替代研究员”，而是把研究纪律固化进基础设施：所有异常有发现时点，所有假设有竞争解释，所有实验有预算，所有结果有谱系，所有候选都经过真正前向期。

一句话结论：

> 架构值得做，但应先建设 Research Ledger 和无标签 Radar，再做实验 Gate，最后才接 Skill；如果顺序反过来，系统会更快地产生故事和回测，而不是更可靠地产生知识。

## 16. 参考依据

- Bailey et al., *The Probability of Backtest Overfitting*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey and López de Prado, *The Deflated Sharpe Ratio*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Zhang et al., *AutoAlpha*: https://arxiv.org/abs/2002.08245
- Neri, *Domain Specific Concept Drift Detectors for Predicting Financial Time Series*: https://arxiv.org/abs/2103.14079
- Shi et al., *Hubble: An LLM-Driven Agentic Framework for Safe, Diverse, and Reproducible Alpha Factor Discovery*: https://arxiv.org/abs/2604.09601

