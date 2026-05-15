# By-Source 问题榜单与调参建议

- 输入文件: D:\Huace_Work\AI_Control\AI_Tune\models\pretrain_diagnostics_by_source_20260502_011146.csv
- 样本源数量: 150
- comp_ratio(mean/median/p90): 0.9547 / 0.9823 / 1.0573
- abs_gap_mean(mean/p90): 0.001332 / 0.001101

## 1) 低补偿比榜单 (Top-K)
| split | source_id | sample_count | abs_residual_mean | abs_model_comp_mean | comp_ratio | abs_gap_mean | issue_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0249532 | 0.00446372 | 0.178883 | 0.0228348 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0226879 | 0.00439112 | 0.193545 | 0.0206318 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0204333 | 0.00476546 | 0.23322 | 0.0187258 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_21\csv_data\Conventional_flatland_no_21_Excellent_Excellent_Seg1.csv | 182 | 0.00364215 | 0.000871941 | 0.239403 | 0.00277577 | UnderComp\|TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0239999 | 0.00608855 | 0.25369 | 0.0212654 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0183362 | 0.00474476 | 0.258764 | 0.0167588 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0219256 | 0.00569981 | 0.259962 | 0.0193261 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0202185 | 0.00611303 | 0.302349 | 0.0178503 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_22\csv_data\Conventional_flatland_no_22_Excellent_Excellent_Seg1.csv | 301 | 0.00146071 | 0.00045784 | 0.313436 | 0.00107101 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.021434 | 0.00674957 | 0.314901 | 0.0167486 | TailRisk |

## 2) 高缺口榜单 (Top-K)
| split | source_id | sample_count | abs_gap_mean | abs_gap_p99 | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0228348 | 0.0496077 | 0.178883 | UnderComp\|TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0212654 | 0.0545472 | 0.25369 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0206318 | 0.0469887 | 0.193545 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0193261 | 0.0493661 | 0.259962 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0187258 | 0.0539611 | 0.23322 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0178503 | 0.0495879 | 0.302349 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0171474 | 0.0525457 | 0.388435 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0167588 | 0.0483606 | 0.258764 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0167486 | 0.0460793 | 0.314901 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg1.csv | 152 | 0.00512399 | 0.0137498 | 2.55429 | TailRisk |

## 3) 高尾部风险榜单 (Top-K)
| split | source_id | sample_count | abs_gap_p99 | abs_gap_mean | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0545472 | 0.0212654 | 0.25369 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0539611 | 0.0187258 | 0.23322 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0525457 | 0.0171474 | 0.388435 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0496077 | 0.0228348 | 0.178883 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0495879 | 0.0178503 | 0.302349 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0493661 | 0.0193261 | 0.259962 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0483606 | 0.0167588 | 0.258764 | UnderComp\|TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0469887 | 0.0206318 | 0.193545 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0460793 | 0.0167486 | 0.314901 | TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg2.csv | 700 | 0.0181614 | 0.00493043 | 1.50473 | TailRisk |

## 4) 调参建议
1. 补偿幅值整体可用：重点检查方向偏置与长尾风险。
2. 长尾风险较高：建议提高困难样本占比或按 source 重采样。
3. 存在方向偏置：检查误差符号定义、补偿符号以及特征归一化。

## 5) 标签说明
1. UnderComp: comp_ratio < 0.30，补偿量明显不足。
2. TailRisk: abs_gap_p99 > 2 * abs_gap_mean，尾部风险偏高。
3. DirectionBias: residual_mean 与 comp_mean 符号相反。
4. GeneralizationRisk: 同源 val 相比 train 的 abs_gap_mean 恶化超过 20%。
