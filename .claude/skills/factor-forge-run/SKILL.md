---
name: factor-forge-run
description: >
  用自然语言驱动 factor_forge 的 CLI，把"我写好了因子 YAML 和实验 YAML，帮我跑某段日期的回测"
  这类请求自动编排成一条完整的命令链：定位配置 → 检查数据版本覆盖 → 廉价预检 →
  跑分级门控回测 → 解读分级与产出路径。只要用户提到 factor_forge / factor-forge CLI、
  因子回测、跑因子、跑实验、回测某段时间、validate 因子/实验 YAML、查看 run 评分或分级、
  或贴出 configs/factors/*.yaml 与 configs/experiments/*.yaml 想执行，就启用本 skill。
  即使用户没显式说"用 skill"，只要意图是"按这两份配置执行回测/校验"，也应当启用。
---

# factor-forge-run：用自然语言驱动 factor_forge 回测链路

这个 skill 把一句自然语言（"用这两个 yaml 跑 2021~2025 回测"）翻译成 factor_forge CLI 的完整命令链，并把结果用中文讲清楚。它的价值不在"敲命令"本身，而在于：

- 帮你**正确处理日期区间**——CLI 没有 `--start/--end`，区间只能落到数据版本或实验 YAML 上，这个坑很容易踩。
- **如实解读分级门控的结果**——`factor validate` 只校验 schema，真正的公式/lookback/数据门控在 `experiment run` 里，失败原因要会读。
- **不动你的原始 YAML**——需要收窄日期区间时，生成临时副本注入 `sample_start/end_date`，原文件保持干净（版本可控）。

## 何时启用

用户给了（或上下文里有）因子 YAML 和/或实验 YAML，并且想**执行**它们——校验、回测、看评分、看卡在哪。典型触发：
- "用 `sw_l2_industry_rank_acceleration` 这个因子跑 2021~2025 回测"
- "帮我 validate 这两个 yaml 然后回测"
- "刚才那个因子 backtest 跑出来什么分级"
- "@configs/factors/x.yaml @configs/experiments/x.yaml 跑一下"

如果用户只是想**编写/修改** YAML（不是执行），不要用本 skill——那是普通编辑任务。

## 关键事实（决定 skill 行为的硬约束）

这些是 CLI 的真实行为，编排时必须遵守，不能臆造命令或参数：

1. **调用方式**：始终用 `python -m factor_forge.cli ...`。`factor-forge` 短命令依赖 PATH，不可靠。
2. **`factor validate <path>`**：位置参数（不是 `--config`）。**只做 Pydantic schema 校验**（字段名规范、枚举、结构），**不**检查 DSL 公式语法、lookback 天数、字段是否存在。成功打印 `VALID: <name> (contract v1)`。它是廉价预检，**不是**"能跑"的保证。
3. **`experiment run <path>`**：位置参数，**没有任何** `--start/--end/--dry-run/--data-version` flag。它内部会：加载并校验因子 spec → 算因子值（这里才会暴露 DSL/lookback/字段错误）→ L0 质量门 → 数据覆盖门 → 方向是否冻结 → L1 预测力门 → L2 全网格回测 → 评分。
4. **日期区间来源**（CLI 层无法传）：
   - 默认取数据版本的**整段跨度**；
   - 或由实验 YAML 的 `sample_start_date` / `sample_end_date`（格式 `"YYYY-MM-DD"`）收窄。
5. **`run_id` 可复用**：由「数据版本 + 四份配置原文哈希」决定。配置没变，再次 `experiment run` 会命中已有产出目录、不重算——告诉用户这点，避免他以为白跑了。
6. **没有 dry-run**。"先校验"的语义＝`factor validate`（秒级 schema 预检）＋ 用户确认是否进入昂贵的 `experiment run`。

## 编排工作流

按顺序执行。每一步给出**真实命令**，不要编造参数。

### 1. 解析用户意图

从请求里提取：
- **因子 YAML 路径** 与 **实验 YAML 路径**：支持 @提及、文件名片段、完整路径。片段不全时，用 Glob 在 `configs/factors/*.yaml`、`configs/experiments/*.yaml` 里匹配并和用户确认。若只给了因子没给实验，问用户用哪个实验（或是否复用某个模板）。
- **回测区间**：如"2021~2025"→ `2021-01-01` 至 `2025-12-31`；"去年"→换算成绝对日期再确认。**没有**区间就不收窄，跑数据版本整段。
- **执行模式**：默认"预检通过→自动跑回测"。若用户说"先只校验/别真跑/dry"，则停在第 4 步之后等确认。

如果实验 YAML 里 `factor_config` 指向的因子与用户给的因子 YAML 不一致，**指出来**并问以哪个为准（实验 YAML 内嵌的 `factor_config` 才是真正生效的）。

### 2. 检查数据底子（决定区间怎么落地）

读 `data/versions/*/manifest.json`（每个已发布版本目录下都有），取 `quality_status=="PASSED"` 中 `created_at` 最新的那个，记下它的 `start_date` / `end_date` 跨度。这是 `data_version: latest` 实际会解析到的版本。

> 若 `data/versions/` 不存在或为空：说明还没拉过数据。直接跳到第 3 步的"未覆盖"分支。

把【请求区间】和【数据版本跨度】比较，按下表决策：

| 情况 | 处理 |
|---|---|
| 请求区间 ⊆ 数据跨度，且**等于**整段 | 直接用原实验 YAML 跑，无需改日期 |
| 请求区间 ⊆ 数据跨度，但数据**更宽** | 生成实验 YAML 的**临时副本**（如 `configs/experiments/<name>.tmp_daterange.yaml`），注入 `sample_start_date` / `sample_end_date`，对副本跑。**原文件不动。** |
| 请求区间 **超出** 数据跨度 | **停下，不要跑**。告诉用户缺哪段，给出可直接复制的命令：`python -m factor_forge.cli data ingest --config configs/project_sw_l2.yaml --start <YYYYMMDD> --end <YYYYMMDD>`（项目配置按实验 YAML 里 `project_config` 字段取，通常是 `configs/project_sw_l2.yaml`） |
| 没有请求区间 | 直接用原实验 YAML 跑整段 |

注入 sample 日期时，副本里同时把 `data_version` 保持不变（仍是 `latest` 或原值），只加两个日期字段。

### 3. 廉价预检

```
python -m factor_forge.cli factor validate <因子YAML绝对路径>
```

- 失败（Pydantic 报错 / 非零退出）→ 把错误讲成人话，停。常见：因子名不符合 `^[a-z][a-z0-9_]*$`、枚举值拼错、缺必填字段。
- 成功 → 如实说明："schema 校验通过，但 DSL 公式/lookback/字段缺失只在回测里才会真实校验。"

### 4. 跑回测（分级门控即真实校验）

```
python -m factor_forge.cli experiment run <实验YAML或临时副本绝对路径>
```

这一步可能较慢（全网格回测）。建议用后台运行或合理 timeout。它向 stdout 打印一段 JSON：`{"run_id", "run_dir", "status", "assessment": {...}}`。

### 5. 解读结果

以 `run_dir` 下的文件为准（比 stdout 更可靠，stdout 偶尔被日志污染）：

- 读 `<run_dir>/manifest.json` 的 `status` 字段——这是唯一真相。
- 若 `status=="SUCCESS"`：读 `<run_dir>/alpha_assessment.json`，取 `classification`（APPROVED / CANDIDATE / WATCHLIST / REJECTED / INVALID）和 `total_score`（0–100）。
- 其它 status：按下表定位失败文件并讲清楚。

| status | 含义 | 看哪个文件定位 |
|---|---|---|
| `SUCCESS` | 跑完并评分 | `alpha_assessment.json`、`report.md`（中文摘要）、`l2_summary.json` |
| `STOPPED_L0` | 因子质量门没过（覆盖率/缺失/截面唯一性不足） | `l0_quality.json` |
| `INVALID_DATA_COVERAGE` | 数据覆盖阻碍命中（命中样本太少） | `manifest.json` 的 details |
| `STOPPED_DIRECTION_UNFROZEN` | 因子 `direction: unknown`，无法评分 | 因子 YAML 的 `direction` 字段 |
| `STOPPED_L1` | 预测力门没过（IC/分位组不达标） | `l1_predictive_power.json` |
| `FAILED` | 抛了未捕获异常 | `error.log` |

`report.md`（中文）和 `l2_summary.json` 只在成功路径才存在——早停时不会有，别去找。

### 6. 汇报（中文，简洁）

给用户三件事：
1. **结论**：分级 + 总分（或卡在哪一步、为什么）。
2. **产出位置**：`run_dir` 绝对路径，并提示"配置没变再跑会复用这个目录"。
3. **下一步建议**：例如分级是 WATCHLIST → 建议看 robustness；卡在 L0 → 建议查覆盖率/数据版本；公式报错 → 指到具体算子。

## 命令速查（全部已核实）

```bash
# 数据（仅在数据版本不覆盖请求区间时才需要）
python -m factor_forge.cli data init --config configs/project_sw_l2.yaml
python -m factor_forge.cli data check-permissions --config configs/project_sw_l2.yaml
python -m factor_forge.cli data ingest --config configs/project_sw_l2.yaml --start 20210101 --end 20251231

# 廉价 schema 预检（位置参数，秒级）
python -m factor_forge.cli factor validate configs/factors/sw_l2_industry_rank_acceleration.yaml

# 全流程分级回测（位置参数，无 --start/--end/--dry-run）
python -m factor_forge.cli experiment run configs/experiments/sw_l2_rank_acceleration_v1.yaml
```

## 易错点提醒

- **别给 `experiment run` 加 `--start/--end`**——它不认。日期只能通过 YAML 或数据版本。
- **别把 `factor validate` 当"能跑"的保证**——它不碰 DSL。用户问"这因子对不对"时，要跑 `experiment run` 才知道公式层有没有问题。
- **临时副本用完别留垃圾**：跑完可以保留（便于复现），但明确告诉用户它是注入了日期的副本、原文件没动。
- **项目配置别搞混**：L2 因子用 `configs/project_sw_l2.yaml`（`industry_level: L2`），别用默认的 `configs/project.yaml`（L1）。以实验 YAML 里 `project_config` 字段为准。
- **data_version: latest 按 created_at 取最新 PASSED 版本**，不是按 end_date——新拉一个更窄的版本会让 latest 指向它。
