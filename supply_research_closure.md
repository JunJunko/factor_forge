# 低量上涨 / 供给收缩因子研究 —— 结案与经验沉淀

> 覆盖 **v1**（`scarcity` / `volume_residual` 滚动 OLS）与 **v2**（稳定成交基线 + 无量上涨三段式）两条线。
> 结案日期 2026-07-06。状态：**暂停**——alpha 真实但太薄，扣成本为负，ML 无增量；不作为独立纯多头 alpha 推进。
> 本文档供以后重启或迁移经验时复用；细节不重复，给指针。

---

## 1. 一句话结论

「无量上涨 / 供给收缩」结构在 A 股 top-1000 是**统计上真实、但经济上太薄**的弱 alpha：单因子 RankIC 0.005~0.012，扣 20bps 成本后净超额为负，LightGBM 消融无增量。**不值得作为独立纯多头因子上线。**

## 2. 核心证据（top 1000, 2017-01 ~ 2026-06）

### v1（`scarcity` 族，120d 条件成交收缩 OLS 残差）
| 指标 | 数值 | 来源 |
|---|---|---|
| scarcity RankIC / ICIR / Newey-t | +0.0119 / 2.79 / +4.32 | [supply_ic_report.md](supply_ic_report.md) |
| 中性化后存留（+vol/size/liq/turn_z） | 76~83% | [supply_neutral_ic.md](supply_neutral_ic.md) |
| Top-100 净超额（20bps） | **−11.76 bps / 5 日** | [供给收缩因子研究报告.md](供给收缩因子研究报告.md) |
| ML Model B−A 增量 | −0.0024 IC、−8.98% 年化（无增量） | `artifacts/supply_contraction_runs/` |
| 多空价差 | 10.98 bps / 5 日（**65% 空头腿**） | [supply_diagnostics.md](supply_diagnostics.md) |
| 2026 月度 IC | 翻负（Jun −0.058） | [supply_2026_probe.md](supply_2026_probe.md) |

### v2（稳定基线 + 无量上涨，2 日事件窗口）
| 因子 | RankIC | t | 判读 |
|---|---|---|---|
| `price_strength_2` | −0.0140 | −4.80 | **2 日反转**，非假设的正向 |
| `recent_volume_z_max_2_clip` | −0.0052 | −2.05 | 弱；max ≈ mean |
| `turnover_stability_28` | +0.0450 | +9.70 | 强但**倒 U**，与 vol20 共线 0.67 |

详见 [univariate_factor_report.md](univariate_factor_report.md) / [factor_decay.csv](factor_decay.csv) / [factor_yearly_ic.csv](factor_yearly_ic.csv)。

### 窗口扫描转折（v2 假设的"正确形态"）
| P（日） | excess_ret_P RankIC | × scarcity 二维交互 |
|---|---|---|
| 2~20 | 负（反转，P=20 最负 −0.020） | 杂乱 |
| 60 | −0.010（衰减） | — |
| **120** | +0.001（t 0.17，无单独 alpha） | **清晰对角线，右下角 +0.00144 / 5 日** |

中期窗口下「无量 × 上涨」交互浮现（右下角 = 强 120d 上涨 × 强 120d 收缩 = 全表最高 +14.4 bps），但**扣 20bps 仍为负**。详见 [window_scan_report.md](window_scan_report.md)。

## 3. 为什么暂停
1. **IC 量级不够**：A 股纯多头因子一般要 RankIC ≥ 0.03 / ICIR ≥ 1.5；这里最好才 0.012。
2. **扣成本为负**：scarcity 与中期交互的毛超额都 < 20bps 往返成本。
3. **ML 无增量**：v1 Model B vs A 增量为负；精简消融（仅 scarcity+scarcity_slope_5）至多持平。
4. **regime 依赖**：2026 翻负（大盘动量 + 高波动 regime 协同翻转，非稀缺专属崩塌）。

---

## 4. 可复用的发现（"以后有机会用"的核心）

> 这部分比结论更有价值——是花了很多算力换来的、对**别的因子**也适用的经验。

### 4.1 量价交互因子的"腿窗口必须匹配"
v1 ML 无增量的根因：`supply_core` 用 `excess_ret_5`（短期反转腿）配 `scarcity`（中期 120d 无量腿），**两腿时间尺度错配**，反转腿把交互拖负。v2 窗口扫描证实：价格上涨腿必须和成交收缩腿**同一时间尺度**才有交互信号。
→ **教训**：设计任何「量 × 价」交互因子，两腿窗口要对齐；短期反转腿会拖负中期结构腿。这也解释了 v1 报告里"组合因子被 5d 反转腿拖负"。

### 4.2 A 股事件窗口的收益结构
- 2~20 日：行业超额是**反转**（IC 负，P=20 最负 −0.020）。
- 60 日：衰减为 0。
- 120 日：**无动量**（IC ~0，t 0.17）——A 股半年线既不反转也不动量。
→ **教训**：A 股不要在 120 日尺度单押"动量/反转"；中期 alpha 必须靠交互或条件结构，不是单腿价格趋势。

### 4.3「成交稳定度」是低波动异象的再包装
`turnover_stability_28` IC +0.045/t 9.70 看似最强，但：十分位**倒 U**（中等稳定最好，极端稳定反而差）；与 `volatility_20` 截面 spearman **0.67**、与市值 **0.60**；Q10（最稳定）收益为负 = "长期无人交易的假稳定"。
→ **教训**：任何「低成交波动/稳定」信号都要先中性化 vol20+size，否则只是重复低波动异象。倒 U 形态意味着不可线性相乘（交接 §9 明示，本数据证实）。

### 4.4 短期无量 = 低换手；要独立须用条件残差
`recent_volume_z_max_2` 与 `turnover_zscore_60` 相关 0.76~0.79；而 v1 `scarcity`（控制涨跌幅/振幅/市场活跃度后的成交残差）与 `turnover_zscore_60` 仅 −0.37——条件化才独立。
→ **教训**：刻画"成交异常低"用**条件残差**（回归后的偏离），不要用原始换手 z-score——后者高度共线于低换手。

### 4.5 max vs mean
`recent_volume_z_max_2` ≈ `recent_volume_z_mean_2`（IC −0.0052 vs −0.0055），交接"max 优先"未被支持。→ 短窗口的"两天都未放量"假设在这个数据里没有比"平均未放量"更强的效力。

### 4.6 工程不变量（指针）
- 双引擎对账：Qlib TopkDropout vs 项目 BacktestEngine 毛收益差 <7%，净差来自换手机制（TopkDropout 接近日换手 0.21 vs BE 固定袖套 0.37）。详见 [scripts/qlib_vs_be_diagnose.py](scripts/qlib_vs_be_diagnose.py)。
- 防泄漏：`volume_residual` 滚动 OLS 窗口 `[t-window, t-1]` 永不含 t；横截面操作当日；训练期阈值仅 train 段。详见 [data_leakage_check.md](data_leakage_check.md)。
- Qlib Windows 坑见 memory `[[supply-contraction-qlib-pipeline]]`。

---

## 5. 如果以后重启：触发条件与第一步

**值得重启的触发条件**（任一满足即可考虑）：
- **成本结构下降**（降到 5bps 以下，或用期货/ETF 实现使换手成本可忽略）。
- **改多空对冲**：多空价差 10.98 bps/5 日、**65% 来自空头腿**——纯多头用不上，多空可能复活。
- **低换手框架重构**：降换手以减成本拖累（v1 报告建议之一）。
- **找到 regime 闸门**：v1 regime 守卫 OOS 未跑通；若找到稳定开关（仅在特定市场宽度/波动下启用），非 2026 类环境可能可交易。

**重启的第一步（最小判决，1–2 小时，不要从头跑任务 6–13）**：
> 中期交互右下角（强 120d 上涨 × 强 120d 成交收缩）扣 20bps 后的净超额，在持有期 5/10/15 日是否为正且分年度稳定。

复用：`supply_v2_ic_dataset.parquet`（已缓存，含 `excess_ret_120` + `scarcity`）+ `BacktestEngine`。
- 为正且稳 → 复活，从任务 7 三维联合继续。
- 为负 → 彻底确认放弃。

**重启时的窗口设定**：事件窗口用 **60~120 日**（中期），不要用交接原设的 2 日；基线窗口在事件窗口之前等长或更长。

---

## 6. 产物索引

**代码**
- v1+v2 特征/数据集/训练/回测：[src/factor_forge/ml/supply_*.py](src/factor_forge/ml/)
- v2 单变量 + 窗口扫描：[scripts/supply_v2_univariate.py](scripts/supply_v2_univariate.py)、[scripts/supply_v2_window_scan.py](scripts/supply_v2_window_scan.py)
- v1 诊断脚本：`scripts/supply_{baseline_health,diagnostics,ic_look,neutral_ic,regime_guard,regime_oos,2026_probe}.py`、[scripts/qlib_vs_be_diagnose.py](scripts/qlib_vs_be_diagnose.py)

**数据**
- 数据集缓存（top 1000, raw, 含 v2 全字段 + `excess_ret_{20,60,120}`）：`supply_v2_ic_dataset.parquet`
- 数据版本 `data_v1_20260704T074315Z_88d001e2`（2016-01 ~ 2026-06, 6021 股）

**报告**
- v1 最终：[供给收缩因子研究报告.md](供给收缩因子研究报告.md)
- v1 诊断：supply_ic_report.md / supply_diagnostics.md / supply_neutral_ic.md / supply_baseline_health.md / supply_2026_probe.md
- v2 phase-1：[feature_schema.md](feature_schema.md) / [architecture_mapping.md](architecture_mapping.md) / [data_definition_audit.md](data_definition_audit.md) / [data_leakage_check.md](data_leakage_check.md)
- v2 分析：[univariate_factor_report.md](univariate_factor_report.md) / [window_scan_report.md](window_scan_report.md)
- 本结案：[supply_research_closure.md](supply_research_closure.md)

**配置**：[configs/ml/supply_contraction_qlib_v1.yaml](configs/ml/supply_contraction_qlib_v1.yaml)（范本）、`supply_run_top1000.yaml`（实跑）、`supply_run_top1000_minimal.yaml`（精简）

**测试**：[tests/test_supply_features.py](tests/test_supply_features.py)（v1）/ [tests/test_supply_v2_features.py](tests/test_supply_v2_features.py)（v2 防泄漏）/ test_supply_dataset.py / test_supply_runner.py

**Memory**：`[[supply-contraction-qlib-pipeline]]`（v1 工程 + Qlib 坑）、`[[supply-v2-stable-baseline]]`（v2 立项 + 任务5 + 窗口扫描发现）
