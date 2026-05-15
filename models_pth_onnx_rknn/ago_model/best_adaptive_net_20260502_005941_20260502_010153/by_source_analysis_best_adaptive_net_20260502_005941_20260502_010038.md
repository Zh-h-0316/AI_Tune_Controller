# By-Source 问题榜单与调参建议

- 输入文件: D:\Huace_Work\AI_Control\AI_Tune\models\pretrain_diagnostics_by_source_20260502_010034.csv
- 样本源数量: 150
- comp_ratio(mean/median/p90): 0.9506 / 0.9980 / 1.0232
- abs_gap_mean(mean/p90): 0.001367 / 0.001952

## 1) 低补偿比榜单 (Top-K)
| split | source_id | sample_count | abs_residual_mean | abs_model_comp_mean | comp_ratio | abs_gap_mean | issue_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0249532 | 0.00448268 | 0.179643 | 0.0228197 | UnderComp\|TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_22\csv_data\Conventional_flatland_no_22_Excellent_Excellent_Seg2.csv | 473 | 0.00231893 | 0.00043381 | 0.187073 | 0.00254907 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0226879 | 0.00441488 | 0.194592 | 0.0206158 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0204333 | 0.00478418 | 0.234136 | 0.018696 | UnderComp\|TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0239999 | 0.00611967 | 0.254987 | 0.0212416 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0183362 | 0.00476735 | 0.259996 | 0.0167508 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0219256 | 0.00573267 | 0.261461 | 0.019303 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_21\csv_data\Conventional_flatland_no_21_Excellent_Excellent_Seg1.csv | 182 | 0.00364215 | 0.000967909 | 0.265752 | 0.00454963 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0202185 | 0.00614646 | 0.304002 | 0.0178381 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.021434 | 0.00681621 | 0.31801 | 0.01671 | TailRisk |

## 2) 高缺口榜单 (Top-K)
| split | source_id | sample_count | abs_gap_mean | abs_gap_p99 | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0228197 | 0.0496071 | 0.179643 | UnderComp\|TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0212416 | 0.0544962 | 0.254987 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0206158 | 0.04695 | 0.194592 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.019303 | 0.0493362 | 0.261461 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.018696 | 0.0539006 | 0.234136 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0178381 | 0.0495632 | 0.304002 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0171289 | 0.0525015 | 0.387676 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0167508 | 0.0483725 | 0.259996 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.01671 | 0.0461539 | 0.31801 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg1.csv | 152 | 0.00525478 | 0.0141416 | 2.6001 | TailRisk |

## 3) 高尾部风险榜单 (Top-K)
| split | source_id | sample_count | abs_gap_p99 | abs_gap_mean | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0544962 | 0.0212416 | 0.254987 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0539006 | 0.018696 | 0.234136 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0525015 | 0.0171289 | 0.387676 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0496071 | 0.0228197 | 0.179643 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0495632 | 0.0178381 | 0.304002 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0493362 | 0.019303 | 0.261461 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0483725 | 0.0167508 | 0.259996 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.04695 | 0.0206158 | 0.194592 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0461539 | 0.01671 | 0.31801 | TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg2.csv | 700 | 0.0184127 | 0.00496214 | 1.52451 | TailRisk |

## 4) 调参建议
1. 补偿幅值整体可用：重点检查方向偏置与长尾风险。
2. 长尾风险较高：建议提高困难样本占比或按 source 重采样。
3. 存在方向偏置：检查误差符号定义、补偿符号以及特征归一化。

## 5) 标签说明
1. UnderComp: comp_ratio < 0.30，补偿量明显不足。
2. TailRisk: abs_gap_p99 > 2 * abs_gap_mean，尾部风险偏高。
3. DirectionBias: residual_mean 与 comp_mean 符号相反。
4. GeneralizationRisk: 同源 val 相比 train 的 abs_gap_mean 恶化超过 20%。
