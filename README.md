# Factor Forge

Factor Forge 是一个面向 A 股日频研究的可复用因子评估与回测平台。V1 的边界固定为：沪深主板、创业板、科创板；日频；T 日收盘后产生信号；T+1 开盘成交；Top2/5/10/20；持有 1/3/5/10/15 个交易日。

它刻意把职责拆开：

- `data`：Tushare 只负责落地原始数据；质量门通过后发布不可变 Parquet 数据版本，SQLite 保存控制面和时点维表。
- `factors`：YAML + 受限 DSL。因子只读标准面板，只输出 `trade_date / ts_code / factor_value`。
- `evaluation`：L0 数据质量、L1 IC/分组/OOS、L3 稳健性与暴露诊断。
- `backtest`：固定的 T+1 开盘执行、独立资金袖套、涨跌停/停牌约束和成本情景。
- `scoring`：硬性否决、六维 100 分评分和 `INVALID/REJECTED/WATCHLIST/CANDIDATE/APPROVED` 分级。

V1 不支持分钟/Tick、自由 Python 因子、机器学习、自动挑最优参数、任意策略逻辑或组合优化器。

## 安装

```powershell
python -m pip install -e ".[tushare,test]"
$env:TUSHARE_TOKEN = "你的 token"
```

如果当前 Python 的 Scripts 目录不在 `PATH`，使用下面这种模块调用方式最稳妥：

```powershell
python -m factor_forge.cli data init --config configs/project.yaml
python -m factor_forge.cli data check-permissions --config configs/project.yaml
python -m factor_forge.cli data ingest --config configs/project.yaml --start 20240101 --end 20241231
python -m factor_forge.cli factor validate configs/factors/momentum_20d.yaml
python -m factor_forge.cli experiment run configs/experiments/momentum_20d.yaml
```

安装环境已配置 Scripts 路径时，可把 `python -m factor_forge.cli` 简写为 `factor-forge`。

## 新增因子

复制 `configs/factors/momentum_20d.yaml`，只修改元数据、字段依赖、中间特征和公式；不要在因子里写数据源、TopN、持有期或成本。公式可用算子及精确定义见 `configs/contracts/operator_registry_v1.yaml`。

实验产物保存到 `artifacts/runs/<run_id>/`，包含输入配置原文、数据/代码版本、因子值、分阶段指标、持仓、交易、评分和报告。数据版本与运行目录均不覆盖旧结果。

## 验证

```powershell
python -m pytest -q
```

测试包含人工可核对的小样本：DSL 滞后语义、禁止任意 Python、T+1 成交、涨停买入失败、跌停卖出顺延、版本哈希，以及从 YAML 到最终评分产物的完整链路。
# Factor combinations

The existing two-YAML command also accepts `kind: factor_combination`:

```bash
factor-forge run --factor configs/combinations/short_term_alpha_combo_v1.yaml --experiment configs/experiments/short_term_alpha_combo_l1.yaml
```

Atomic factor YAML files remain valid with no `kind` field. Combination YAML files reference them by path; the platform computes or reuses their raw cache automatically.

## Conditional IC

L1 can split the cross-section by a conditioning factor and calculate the main factor's Rank IC inside each daily quantile. Enable `stage_l1.conditional_ic` and set `conditioning_factor` to `main_factor` or another factor YAML. See [conditional_ic_schema.md](conditional_ic_schema.md) for the configuration and artifact contract.
