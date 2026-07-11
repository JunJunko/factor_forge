# Event Study Phase 3

Phase 3 对冻结 ObservationCard 做匹配对照和成熟标签检验。源卡片与源事件文件保持不可变、无标签；所有未来收益只存在于独立 Event Study 产物。

## 标签语义

观察发生在 T 日收盘，标签与 Factor Forge 保持一致：

```text
forward_return_h = adj_open(T+h+1) / adj_open(T+1) - 1
```

分别计算 3、5、10 日。若 T+1 入场价或 T+h+1 退出价在标签数据版本中尚不存在，该事件记为 censored，不参与该 horizon 的估计。

## 固定匹配序列

```text
M0 同日 + 同申万一级行业
M1 M0 + 过去5日收益
M2 M1 + 20日波动率
M3 M2 + 20日平均成交额 + 总市值
```

每个事件最多匹配 3 个控制股。控制股不能是同模板当日事件。连续变量使用同日同行业内的稳健标准化距离，完整控制的 caliper 固定为 3.0。

## 推断与 Gate

- 先把一个事件的多个控制股收益求均值，形成一个配对收益差；
- 再按交易日聚合配对差，使用 `horizon-1` 阶 Newey–West；
- 同时报告事件加权均值与日期等权均值；预注册主指标和方向判断只使用日期等权均值；
- 4 个匹配阶段 × 3 个 horizon 统一报告 BH-FDR；
- 唯一预注册主指标是 `full_controls / 5D / daily mean paired excess`；
- Severity 和 Regime 仅为诊断，不能替代主指标；
- Gate 只输出 `reject`、`revise_one_hypothesis` 或 `observe_forward`。
- 完整匹配任一控制变量的绝对 SMD 超过 0.20 时，不允许解释收益差异，固定返回 `INSUFFICIENT_BALANCE`。

## 命令

```powershell
python -m factor_forge.cli event-study validate `
  configs/event_studies/price_drop_without_volume_phase3_v1.yaml

python -m factor_forge.cli event-study run `
  configs/event_studies/price_drop_without_volume_phase3_v1.yaml
```

产物：

```text
artifacts/radar_event_studies/<run_id>/
  e0_audit.json
  summary.json
  e1_e3_progressive_controls.csv
  e2_severity_monotonicity.csv
  e4_regime_diagnostics.csv
  matched_pairs/*.parquet
  paired_events/*.parquet
  report.md
  manifest.json
```

每次新研究自动创建或复用 Idea/Hypothesis/Plan，并以 validation trial 登记，因此会消耗 Idea 和 Family 的 trial/peek 预算。缓存复用不会重复消耗预算。
