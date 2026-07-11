# Radar Phase 2

Phase 2 只负责生成无未来标签的市场观察，不评价收益，不生成因子，也不做 Alpha 准入。

## 硬边界

- 当前值只和严格早于当前行的历史比较；
- ObservationCard 禁止 `forward_return`、`target`、`label`、IC、ICIR 和 Sharpe；
- 事件明细只包含当期测量、历史分位、严重度和事件身份；
- 同一模板、数据版本和 as-of 日期得到确定性的 Observation ID；
- 产物不可覆盖，重复运行只复用语义相同的产物；
- 两个模板均为冻结配置，不开放任意公式。

## 模板

```text
price_drop_without_volume_confirmation_v1
volume_surge_without_price_impact_v1
```

第一个模板寻找“急跌但成交确认不足”；第二个寻找“放量但价格冲击不足”。它们只说明量价关系偏离正常，不说明后续应该反转或延续。

## 命令

```powershell
python -m factor_forge.cli radar validate-template `
  configs/radar/price_drop_without_volume_confirmation_v1.yaml

python -m factor_forge.cli radar scan `
  --template configs/radar/price_drop_without_volume_confirmation_v1.yaml `
  --data-version latest `
  --as-of 20260710
```

产物写入：

```text
artifacts/radar_observations/<observation_id>/
  observation_card.json
  events.parquet
  manifest.json
```

同时在 `data/research.sqlite3` 的 `observation_card` 表登记定义 hash、发现时点、数据版本和产物路径。

## 卡片证据的含义

- `recent_event_rate`：最近固定交易窗口的事件行占比；
- `historical_event_rate`：发现窗口中、排除最近窗口后的事件行占比；
- `event_rate_ratio`：近期发生率相对历史部分的变化；
- `severity_*`：偏离模板阈值的程度，不是收益分数；
- `industry_coverage`：事件覆盖行业占可用行业的比例；
- `max_entity_share/max_industry_share`：事件集中度；
- `temporal_audit_passed`：追加未来行不会改变历史 PIT 分位。

后续 Phase 3 才能把冻结卡片送入匹配对照和 forward-return Event Study。
