# By-Source 问题榜单与调参建议

- 输入文件: D:\Huace_Work\AI_Control\AI_Tune\models\pretrain_diagnostics_by_source_20260318_182604.csv
- 样本源数量: 150
- comp_ratio(mean/median/p90): 1.1741 / 1.2178 / 1.4330
- abs_gap_mean(mean/p90): 0.001611 / 0.002496

## 1) 低补偿比榜单 (Top-K)
| split | source_id | sample_count | abs_residual_mean | abs_model_comp_mean | comp_ratio | abs_gap_mean | issue_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0249532 | 0.00452238 | 0.181234 | 0.0228131 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0226879 | 0.00445281 | 0.196264 | 0.0206137 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0204333 | 0.00483699 | 0.236721 | 0.0187055 | UnderComp\|TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0239999 | 0.00618807 | 0.257837 | 0.021231 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0183362 | 0.00480998 | 0.262321 | 0.0167505 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0219256 | 0.00580456 | 0.264739 | 0.0192903 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0202185 | 0.00620057 | 0.306678 | 0.0178329 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.021434 | 0.00688761 | 0.321341 | 0.0166693 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0161003 | 0.00633255 | 0.393317 | 0.0171556 | TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_22\csv_data\Conventional_flatland_no_22_Excellent_Excellent_Seg2.csv | 473 | 0.00231893 | 0.00111692 | 0.481655 | 0.00330734 | TailRisk\|DirectionBias |

## 2) 高缺口榜单 (Top-K)
| split | source_id | sample_count | abs_gap_mean | abs_gap_p99 | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0228131 | 0.0496883 | 0.181234 | UnderComp\|TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.021231 | 0.0547187 | 0.257837 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0206137 | 0.0470097 | 0.196264 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0192903 | 0.0494766 | 0.264739 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0187055 | 0.0540537 | 0.236721 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0178329 | 0.0496431 | 0.306678 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0171556 | 0.0526218 | 0.393317 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0167505 | 0.0484333 | 0.262321 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0166693 | 0.0462047 | 0.321341 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_21\csv_data\Conventional_flatland_no_21_Excellent_Excellent_Seg1.csv | 182 | 0.00647861 | 0.0129994 | 0.786107 | TailRisk\|DirectionBias |

## 3) 高尾部风险榜单 (Top-K)
| split | source_id | sample_count | abs_gap_p99 | abs_gap_mean | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0547187 | 0.021231 | 0.257837 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0540537 | 0.0187055 | 0.236721 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0526218 | 0.0171556 | 0.393317 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0496883 | 0.0228131 | 0.181234 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0496431 | 0.0178329 | 0.306678 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0494766 | 0.0192903 | 0.264739 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0484333 | 0.0167505 | 0.262321 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0470097 | 0.0206137 | 0.196264 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0462047 | 0.0166693 | 0.321341 | TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg2.csv | 700 | 0.0183432 | 0.00494528 | 1.52207 | TailRisk |

## 4) 调参建议
1. 补偿幅值整体可用：重点检查方向偏置与长尾风险。
2. 长尾风险较高：建议提高困难样本占比或按 source 重采样。
3. 存在方向偏置：检查误差符号定义、补偿符号以及特征归一化。

## 5) 标签说明
1. UnderComp: comp_ratio < 0.30，补偿量明显不足。
2. TailRisk: abs_gap_p99 > 2 * abs_gap_mean，尾部风险偏高。
3. DirectionBias: residual_mean 与 comp_mean 符号相反。
4. GeneralizationRisk: 同源 val 相比 train 的 abs_gap_mean 恶化超过 20%。
