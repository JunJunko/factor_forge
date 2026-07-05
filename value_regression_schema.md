# 价值修复 LightGBM 数据契约

运行入口：

```powershell
python -m factor_forge.cli ml value-run configs/ml/value_regression_lightgbm_v1.yaml
```

## PIT 财务快照

`fundamentals_path` 指向 Parquet 文件。主键为 `ts_code + available_date`，其中
`available_date` 是财务信息首次可用于交易决策的交易日，不能使用报告期结束日代替。

必需字段：

- `ts_code`
- `available_date`
- `revenue_ttm`
- `net_assets`
- `roe_ttm`
- `revenue_growth_yoy`
- `roe_change_yoy`
- `debt_to_assets`
- `net_profit_ttm`

可通过以下命令从 Tushare VIP 财务报表构建：

```powershell
python -m factor_forge.cli data ingest-fundamentals --start-year 2014
```

季度累计利润表字段转换为 TTM 时使用“本期累计 + 上年年报 − 上年同期累计”；
ROE 使用 TTM 归母净利润除以期初期末平均归母净资产。所有修订按新公告事件追加，
不会改写较早的 PIT 快照。

管线只做向后 as-of 合并，并检查 `available_date <= trade_date`。财务数据缺失时不会
用未来值回填。

## 八个模型特征

1. `liquidity_adjusted_value_gap`：剔除流动性折价后的行业内基本面价值缺口。
2. `fundamental_revision_20d`：用 T-20 已知回归系数度量基本面修正。
3. `residual_price_dislocation_20_5`：最近 20 至 5 日行业相对回撤。
4. `industry_relative_strength_5d`：最近 5 日行业相对修复确认。
5. `amihud_improvement_5_20`：最近 5 日相对此前 20 日的价格冲击改善。
6. `abnormal_attention_5_20`：剔除收益、波动和规模后的换手提升。
7. `price_delay_improvement_20d`：行业信息响应延迟的 20 日改善。
8. `residual_cross_sectional_momentum_120_20`：T-120 至 T-20 的行业相对动量。

价格窗口按“中期趋势 → 回撤 → 最近确认”排列，不共享收益观测区间。各特征完成经济
风险控制后再做当日 MAD 缩尾和 Z-score。

## 标签与防泄漏

模型分别预测从 T+1 开盘开始的 5、10、20 日全市场超额收益。训练集和验证集尾部按
各自标签期限执行 purge；三个预测先做当日截面排名，再按配置权重融合。

产物包含模型、各期限预测、组合回测、特征重要性和逐日 Spearman 相关性审计。
默认独立性门槛为日截面绝对 Spearman 的中位数不超过 0.15、90% 分位数不超过
0.35；任意特征对越界时训练会在拟合模型前停止。

运行目录会从任务启动时立即创建，并持续写入 `run.log`。日志记录阶段、分组进度、
累计耗时和进程内存；任何未处理异常都会同时写入 `error.json`，包含失败阶段、异常
类型、消息和完整 traceback。
