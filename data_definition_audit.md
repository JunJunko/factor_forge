# 数据口径审计（data definition audit）

> 对应交接说明任务 2。盘点复权 / 单位 / 行业 / 可交易性 / tick / 未来数据口径，逐项标注代码依据与 v2 兼容性。

## 1. 复权口径（交接 4.3.1 / 4.3.2）
- 收益率与波动率用**复权价** `adj_close` / `adj_open`（[panel.py](src/factor_forge/data/panel.py)：`raw × adj_factor`）。
- 真实价格、最小跳价、涨跌停、K 线用**未复权** `raw_*`。
- v2 检查：`price_strength_2` 用 adj 收益 + adj 波动率；`effective_ticks_2` 用 `raw_close`。✓ 符合。

## 2. 单位
| 字段 | 面板单位 | 转换 | v2 用法 |
|---|---|---|---|
| turnover_rate | 百分比（1.23 = 1.23%） | `_log_turnover` 做 `/100 → log1p` | baseline / z 全部经此转换 |
| amount_cny | 元 | 无 | `baseline_amount_mean_28` 用原始元均值（交接 3.4 字面），winsor+zscore 归一 |
| circ_mv_cny | 元 | panel 由 `circ_mv × 10000` 派生 | log_float_market_cap（v1，v2 复用） |
| tick_size | 常量 0.01 | 不入面板（[supply_config.py](src/factor_forge/ml/supply_config.py)） | effective_ticks_2 / tick_return 共用 |

## 3. 行业映射（交接 4.1.9）
- [panel.py](src/factor_forge/data/panel.py) `_attach_industry` 按 `industry_level` 配置（L1/L2），**point-in-time**（历史映射，禁止未来回填）。
- v2 `excess_ret_2` 沿用 v1 **leave-one-out** 行业基准（用户裁决点 3），`_industry_loo_mean` 排除自身。
- 待确认：交接 3.6 字面为"行业指数收益"，项目无外部行业指数加载，统一用 LOO。

## 4. ST / 停牌 / 涨跌停 / 退市（交接 16）
- panel 派生 flags：`is_st` / `is_delisting_period` / `is_suspended` / `is_tradeable` / `is_limit_up_open` / `is_limit_down_open` / `listing_trade_days`。
- `supply_dataset.valid_mask = is_tradeable & ¬suspended & ¬st & ¬delisting & listing≥min_listing_days & industry notna`。
- 无效样本 **NaN 屏蔽**（不删除，保留日期×证券网格供 Qlib 日历）。

## 5. 上市天数（交接 16.2）
- `listing_trade_days`（panel 派生，截至当日）；`min_listing_days` 默认 60。

## 6. tick_size
- 0.01 常量（A 股现代统一档位）。交接 6.3 列为面板字段，但项目实测统一，作为配置常量；如需按板位差异化，改 `supply_config.SupplyFeatureConfig.tick_size`。

## 7. 未来数据 / 全样本归一化（交接 4.1.7）
- **禁止**全样本均值 / 标准差 / 分位数 / 回归参数。
- v2：横截面 rank / std_floor / winsor / zscore **当日**计算；baseline 窗口严格 [t-29, t-2]。
- 训练期阈值（`liquidity_weight` 的 A_low / A_full）只用 train 段（交接 9.5，`supply_dataset` 已实现）。

## 8. 限制与待确认
- `regime_score`：交接 6.3 / 任务 10 Model A 列为可选输入；项目无现成 regime 模块，v1 用 market_breadth / industry_breadth / market_turnover_z 近似。v2 Model A 是否引入外部 regime_score 留任务 10 决定。
- 行业指数：见 §3，v2 沿用 LOO。
- v2 `baseline_amount_mean_28` 用原始 amount 均值（非 log）；交接 3.4 字面如此。如训练数值不稳，可在任务 5 消融时试 log 版本。

## 9. 结论
v2 所需全部原始字段在面板齐备，口径与交接说明一致；未发现口径冲突。`tick_size` 为常量、行业用 LOO 两处与交接字面略有出入，已由用户裁决确认。
