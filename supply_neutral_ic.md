# scarcity 家族独立性检验（横截面中性化后 IC）

- universe: 流动性 top 1000；窗口 2017-01-01 ~ 2026-06-30
- 每日横截面 OLS 残差化因子对控制变量，再算残差对 5 日行业中性标签的日频 RankIC。

|    | factor                | spec                              |   rank_ic_mean |   rank_ic_ir |   rank_ic_newey_t |   rank_ic_positive_ratio |
|---:|:----------------------|:----------------------------------|---------------:|-------------:|------------------:|-------------------------:|
|  0 | scarcity              | raw                               |         0.0119 |       2.7883 |            4.3206 |                   0.5758 |
|  1 | scarcity              | neut: vol20                       |         0.0092 |       2.2580 |            3.6590 |                   0.5575 |
|  2 | scarcity              | neut: vol20 + size                |         0.0099 |       2.7034 |            4.6426 |                   0.5710 |
|  3 | scarcity              | neut: vol20 + size + liq          |         0.0083 |       2.2774 |            3.9216 |                   0.5588 |
|  4 | scarcity              | neut: vol20 + size + liq + turn_z |         0.0091 |       2.8254 |            4.7416 |                   0.5732 |
|  5 | scarcity_days_ratio_5 | raw                               |         0.0100 |       2.7914 |            4.0520 |                   0.5819 |
|  6 | scarcity_days_ratio_5 | neut: vol20                       |         0.0096 |       2.8064 |            4.2396 |                   0.5697 |
|  7 | scarcity_days_ratio_5 | neut: vol20 + size                |         0.0092 |       2.8945 |            4.3850 |                   0.5854 |
|  8 | scarcity_days_ratio_5 | neut: vol20 + size + liq          |         0.0063 |       2.0254 |            3.1407 |                   0.5549 |
|  9 | scarcity_days_ratio_5 | neut: vol20 + size + liq + turn_z |         0.0039 |       1.3568 |            2.1524 |                   0.5357 |
| 10 | scarcity_slope_5      | raw                               |         0.0058 |       1.5747 |            3.3682 |                   0.5375 |
| 11 | scarcity_slope_5      | neut: vol20                       |         0.0061 |       1.6863 |            3.5843 |                   0.5401 |
| 12 | scarcity_slope_5      | neut: vol20 + size                |         0.0058 |       1.7039 |            3.6303 |                   0.5488 |
| 13 | scarcity_slope_5      | neut: vol20 + size + liq          |         0.0055 |       1.6378 |            3.4803 |                   0.5436 |
| 14 | scarcity_slope_5      | neut: vol20 + size + liq + turn_z |         0.0054 |       1.6858 |            3.5312 |                   0.5492 |
| 15 | volume_residual       | raw                               |        -0.0119 |      -2.7883 |           -4.3206 |                   0.4242 |
| 16 | volume_residual       | neut: vol20                       |        -0.0092 |      -2.2580 |           -3.6590 |                   0.4425 |
| 17 | volume_residual       | neut: vol20 + size                |        -0.0099 |      -2.7034 |           -4.6426 |                   0.4290 |
| 18 | volume_residual       | neut: vol20 + size + liq          |        -0.0083 |      -2.2774 |           -3.9216 |                   0.4412 |
| 19 | volume_residual       | neut: vol20 + size + liq + turn_z |        -0.0091 |      -2.8254 |           -4.7416 |                   0.4268 |

## 读法

- 沿每一行从上到下：若 IC 随控制变量加入而**坍缩到 ~0/不显著**，说明该因子只是控制变量的代理；
- 若 IC **保持显著且方向不变**，则该因子对控制变量有**独立增量**。