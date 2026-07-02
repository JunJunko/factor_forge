# 盘整突破过程因子条件 IC 报告

- 数据版本：`data_v1_20260701T095408Z_c7b9995d`
- 有效突破事件：69,292
- 箱体记录：329,203
- 因子/组合：63
- 条件切片：17
- 通过预设 promising 门槛：721

IC 为突破事件当日截面的 Spearman IC；收益从下一交易日开盘开始。
显著性使用 Newey–West HAC，FDR 为全部条件搜索上的 BH 校正。
本轮属于探索性组合搜索，结果仍需固定规则后做真正样本外确认。

## 排名前 30 的结果

|组合|条件|周期|IC天数|Mean IC|OOS IC|NW t|FDR q|
|---|---|---:|---:|---:|---:|---:|---:|
|single:continuous_move|pre_accelerating|10|1165|0.0636|0.0428|7.97|0.0000|
|single:continuous_move|wide_box|10|1126|0.0645|0.0455|7.99|0.0000|
|pair:approach_velocity+continuous_move|compact_box|20|1137|0.0627|0.0437|6.53|0.0000|
|pair:approach_velocity+continuous_move|all|20|1260|0.0592|0.0398|7.88|0.0000|
|pair:approach_velocity+continuous_move|volatility_contracting|20|1260|0.0592|0.0398|7.88|0.0000|
|single:continuous_move|pre_accelerating|20|1158|0.0611|0.0393|6.31|0.0000|
|pair:approach_velocity+continuous_move|all|10|1271|0.0578|0.0327|8.62|0.0000|
|pair:approach_velocity+continuous_move|volatility_contracting|10|1271|0.0578|0.0327|8.62|0.0000|
|pair:approach_velocity+continuous_move|pre_accelerating|10|1165|0.0601|0.0421|7.90|0.0000|
|pair:approach_velocity+continuous_move|compact_box|10|1144|0.0592|0.0421|6.94|0.0000|
|pair:approach_velocity+continuous_move|strong_breakout|10|1144|0.0591|0.0569|8.11|0.0000|
|single:continuous_move|all|10|1271|0.0559|0.0309|7.90|0.0000|
|single:continuous_move|volatility_contracting|10|1271|0.0559|0.0309|7.90|0.0000|
|pair:approach_velocity+continuous_move|volume_confirmed|20|1249|0.0563|0.0370|7.93|0.0000|
|pair:range_compactness+continuous_move|wide_box|10|1126|0.0593|0.0442|7.19|0.0000|
|single:continuous_move|wide_box|20|1120|0.0594|0.0242|6.64|0.0000|
|pair:approach_velocity+continuous_move|pre_accelerating|20|1158|0.0584|0.0528|6.74|0.0000|
|pair:direction_persistence+continuous_move|pre_accelerating|20|1158|0.0582|0.0610|6.68|0.0000|
|single:continuous_move|strong_breakout|10|1144|0.0583|0.0470|7.58|0.0000|
|pair:approach_velocity+continuous_move|strong_breakout|20|1137|0.0584|0.0469|7.88|0.0000|
|pair:direction_persistence+continuous_move|pre_accelerating|10|1165|0.0565|0.0446|7.27|0.0000|
|pair:approach_velocity+continuous_move|volume_confirmed|10|1260|0.0543|0.0304|8.29|0.0000|
|pair:direction_persistence+continuous_move|all|10|1271|0.0540|0.0216|7.81|0.0000|
|pair:direction_persistence+continuous_move|volatility_contracting|10|1271|0.0540|0.0216|7.81|0.0000|
|pair:direction_persistence+continuous_move|wide_box|10|1126|0.0563|0.0279|7.17|0.0000|
|pair:approach_velocity+continuous_move|wide_box|20|1120|0.0563|0.0345|6.70|0.0000|
|pair:direction_persistence+continuous_move|all|20|1260|0.0528|0.0379|6.21|0.0000|
|pair:direction_persistence+continuous_move|volatility_contracting|20|1260|0.0528|0.0379|6.21|0.0000|
|single:continuous_move|volume_confirmed|10|1260|0.0525|0.0317|7.72|0.0000|
|pair:range_compactness+continuous_move|wide_box|20|1120|0.0551|0.0238|5.92|0.0000|
