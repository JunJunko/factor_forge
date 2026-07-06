# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

Factor Forge 是一个面向 **A 股日频**的因子评估与回测平台。V1 的边界是硬性约束，改动任何模块前都要先确认没有越界：

- 仅沪深主板/创业板/科创板，仅日频；T 日收盘产生信号，T+1 开盘成交。
- 因子**只能是 YAML + 受限 DSL**，禁止自由 Python、分钟/Tick、自动挑参、任意策略逻辑、组合优化器。**因子评估链路内部禁止机器学习**；ML 有独立的研究/诊断管线（`src/factor_forge/ml/`），复用数据与回测底层，但有自己的配置空间与时序规约，且**不进入因子评估的评分门控链路**（见下文"研究/诊断层"）。
- 因子实验的回测空间被 Pydantic 锁死：持有期 ∈ {1,3,5,10,15}，TopN ∈ {2,5,10,20}（见 [config.py](src/factor_forge/config.py) 的 `ExperimentSpec.fixed_first_version_space`）。**此约束仅适用于因子评估链路**；ML 研究链路用 [ml/config.py](src/factor_forge/ml/config.py) 的 `PortfolioConfig`，是更宽的另一套空间。

## 常用命令

开发环境未把 Scripts 放到 PATH 时，统一用模块调用方式：

```powershell
python -m pip install -e ".[tushare,test]"          # 安装（含可选依赖）
$env:TUSHARE_TOKEN = "你的 token"                     # 数据拉取需要

python -m pytest -q                                   # 全量测试
python -m pytest tests/test_dsl.py -q                 # 单个文件
python -m pytest tests/test_backtest_timing.py::test_limit_down_sell_is_deferred  # 单个用例

python -m factor_forge.cli data init --config configs/project.yaml
python -m factor_forge.cli data check-permissions --config configs/project.yaml
python -m factor_forge.cli data ingest --config configs/project.yaml --start 20240101 --end 20241231
python -m factor_forge.cli factor validate configs/factors/momentum_20d.yaml
python -m factor_forge.cli experiment run configs/experiments/momentum_20d.yaml
```

PATH 配置好后 `python -m factor_forge.cli` 可简写为 `factor-forge`。运行时无网络/无 token 时，跑测试与实验（用本地数据版本）不需要 Tushare。

## 整体架构：一条带质量门的不变链路

整个平台围绕两个原则组织：**数据版本不可变**，以及**分阶段硬性门控**。理解 [experiments/runner.py](src/factor_forge/experiments/runner.py) 的 `ExperimentRunner.run` 就理解了整条链路。

**数据层** (`data/`，CLI 的 `data` 子命令)：Tushare 按交易日分区落地到 `data/staging/` → [panel.py](src/factor_forge/data/panel.py) 的 `DailyPanelBuilder` 拼成标准面板 → [quality.py](src/factor_forge/data/quality.py) 的质量门 → [repository.py](src/factor_forge/data/repository.py) 的 `DataVersionRepository.publish` 写入**不可变**、内容哈希命名的 `data/versions/<data_v1_...>/`（含 `manifest.json` + curated Parquet + 原始 raw 数据），并把时点维表（证券/交易日历/行业历史）写入 `data/metadata.sqlite3`。**数据版本与运行目录一律不覆盖**——重发相同内容会抛 `DataQualityError`。

**因子层** (`factors/`)：[dsl.py](src/factor_forge/factors/dsl.py) 用 Python AST 求值一个**刻意很小**的公式语言，遇到白名单外的语法直接抛 `DSLValidationError`。[engine.py](src/factor_forge/factors/engine.py) 的 `FactorEngine.compute` 顺序是：校验字段 → 逐个算 `features`（不允许覆盖标准字段名）→ 算 `formula` → 方向取反 → winsorize/standardize → universe 掩码。其中 `infer_lookback` 会静态推导公式实际需要的回看天数，**超过声明的 `lookback_days` 就报错**——这是防未来数据的关键守卫。

**评估层** (`evaluation/`)：[l0.py](src/factor_forge/evaluation/l0.py)（覆盖率/缺失/截面唯一性）、[l1.py](src/factor_forge/evaluation/l1.py)（rank/pearson IC、分位组、OOS、BH-FDR）、[neutralization.py](src/factor_forge/evaluation/neutralization.py)（按日 OLS 残差，构造 raw/规模/行业中性变体）、[robustness.py](src/factor_forge/evaluation/robustness.py)（逐年/滚动/walk-forward/TopN 曲线/持仓衰减/成本敏感度/参数邻域）。

**回测层** (`backtest/`，[engine.py](src/factor_forge/backtest/engine.py))：固定语义——T 收盘信号、T+1 开盘成交；用 `holding_days` 个**独立资金袖套**实现重叠持仓；硬约束：涨停开盘买不进（资金闲置、不替换）、跌停开盘卖不出（顺延到下一可卖日）、排除停牌/ST/退市整理期。每个组合跑 `成本场景 bps` 网格。

**评分层** (`scoring/`，[engine.py](src/factor_forge/scoring/engine.py))：六维合计 100 分（OOS 有效性/可交易表现/稳定性/TopN 结构/统计证据/独立性）+ 硬否决标志，输出 `INVALID/REJECTED/WATCHLIST/CANDIDATE/APPROVED`。**主测量组合固定为 universe=liquid, top_n=2, cost_bps=20**（见 `configs/contracts/alpha_scoring_v1.yaml`），评分各维度都以它为基准。缺失证据一律记 0 分，绝不臆造。

`ExperimentRunner` 的门控顺序：L0 不过 → INVALID 停；数据覆盖阻碍命中 → INVALID 停；`direction: unknown` → WATCHLIST 停；L1 不过 → REJECTED 停；否则进 L2 全网格、可选 L3，最后评分。每次运行产出落到 `artifacts/runs/<run_id>/`（`run_id` 由数据版本 + 四份配置原文的哈希决定，配置不变即复用）。

**研究/诊断层** (`ml/`，CLI 的 `ml` 子命令)：独立于评分链路。[MLExperimentRunner](src/factor_forge/ml/runner.py) 按严格不重叠的 train<valid<test 分段训练 LightGBM，`ValueRegressionRunner`、`StyleAttributionRunner`（[ml/value_style_attribution.py](src/factor_forge/ml/value_style_attribution.py)）做价值归因诊断。它们从不可变数据版本出发，再用**同一个** `BacktestEngine` 评估——因此复用底层数据/时序/回测不变量（T+1 开盘、收益映射到 T+2、涨停买不进/跌停卖顺延），但**不走 L0/L1/评分门控**，产出落在 `artifacts/ml_runs/`、`artifacts/value_*` 等独立目录。可选依赖 lightgbm/statsmodels 缺失时给出明确安装提示。其回测空间与字段规约见 [ml/config.py](src/factor_forge/ml/config.py) 的 `MLExperimentConfig`/`PortfolioConfig`，是比因子层更宽的研究专用空间——**不要与因子评估链路的空间混淆，也不要让 ML 产物回流进因子评估链路**。

## 关键不变量（改动时务必保持）

- **因子禁止自由 Python**：DSL 白名单见 `configs/contracts/operator_registry_v1.yaml` 与 [dsl.py](src/factor_forge/factors/dsl.py) 的 `_eval`/`_call`。算子语义不能漂移——例如 [operators.py](src/factor_forge/factors/operators.py) 里 `ts_std` 显式用 `ddof=0`（总体标准差），偏离 pandas 默认，是有意为之。
- **字段别名**：DSL 里写 `open/close/amount/market_cap/industry`，分别映射到面板规范列 `adj_open/adj_close/amount_cny/total_mv_cny/industry_l1_code`（见 `FIELD_ALIASES`）。新增算子/字段要同时更新别名表。
- **单位换算**：[panel.py](src/factor_forge/data/panel.py) 把 Tushare 源单位转成规范单位（手→股 ×100，千元→元 ×1000，万元→元 ×10000）。
- **点在时间（point-in-time）正确性**：ST 状态、行业归属必须是时点的；`delist_date` **不能**当信号（退市状态改用日频快照判定）。`future_data_violations` 必须恒为 0——靠 DSL 只暴露当前/过去算子来保证。
- **契约是事实来源**：`configs/contracts/*.yaml`（data/factor/operator/scoring/backtest/artifact）是版本化规约，代码负责执行它们，改语义要同步改契约文件。

## 配置体系

- Pydantic 模型集中在 [config.py](src/factor_forge/config.py)，YAML 经 `load_project/load_factor/load_experiment` 校验。
- 新增因子：复制 [configs/factors/momentum_20d.yaml](configs/factors/momentum_20d.yaml)，只改元数据、`required_fields`、`lookback_days`、`features`、`formula`、`parameters`；**不要**在因子里写数据源、TopN、持有期、成本。
- 因子需要行业内截面时，复制 `configs/project_sw_l2.yaml` 把 `industry_level` 设为 L2（对应已有 sw_l2 因子/实验样例）。

## 测试约定

测试都是**人工可核对的小样本**（README 有清单：DSL 滞后语义、禁止任意 Python、T+1 成交、涨停买不进、跌停卖顺延、版本哈希、端到端链路）。新增功能要延续这个风格——构造可手工验证的小面板（参考 [tests/conftest.py](tests/conftest.py) 的 `make_panel` 与 `tests/test_experiment_runner.py` 的合成漂移面板），不要依赖外部数据。
