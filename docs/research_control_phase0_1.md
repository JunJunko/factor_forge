# Research Control Phase 0/1

本模块是现有 Factor Forge 上方的薄控制面，不改变因子 DSL、数据版本、评估、回测或 ML 语义。

## 初始化与回填

```powershell
python -m factor_forge.cli research init
python -m factor_forge.cli research index-artifacts --artifacts-root artifacts
python -m factor_forge.cli research artifact-summary
```

默认研究库为 `data/research.sqlite3`。它只保存控制元数据和 artifact 路径，不复制行情、特征或回测明细。

## 最小研究闭环

```powershell
python -m factor_forge.cli research idea-create `
  --title "急跌未放量" `
  --thesis "未被成交确认的急跌可能包含不同于普通反转的信息" `
  --family-id price_volume_divergence `
  --target-horizon 5

python -m factor_forge.cli research hypothesis-add IDEA_ID `
  --statement "持续卖压有限，价格可能均值回归"

python -m factor_forge.cli research plan-create IDEA_ID `
  --name matched_control_v1 `
  --primary-metric matched_forward_5d_excess `
  --hypothesis-id HYPOTHESIS_ID

python -m factor_forge.cli research trial-record PLAN_ID `
  --data-role validation `
  --status success `
  --external-run-id EXISTING_RUN_ID `
  --artifact-path artifacts/runs/EXISTING_RUN_ID

python -m factor_forge.cli research decision-save TRIAL_ID `
  --action observe_forward `
  --reason "验证期增量存在，但发现后证据不足" `
  --decided-by Junko

python -m factor_forge.cli research idea-show IDEA_ID
```

每个 trial 都同时消耗 Idea 和 Family 预算，失败 trial 也消耗预算。`validation` 自动计为一次 peek。普通命令禁止登记 `sealed_test`；封存访问必须使用独立审批命令并保留审计记录。

Phase 0 的冻结边界、首批模板和默认预算见 `configs/research/phase0_protocol_v1.yaml`。
