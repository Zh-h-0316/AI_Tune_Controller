# By-Source 问题榜单与调参建议

- 输入文件: D:\Huace_Work\AI_Control\AI_Tune\models_pth_onnx_rknn\pretrain_diagnostics_by_source_20260509_094803.csv
- 样本源数量: 150
- comp_ratio(mean/median/p90): 1.0855 / 0.9960 / 1.0139
- abs_gap_mean(mean/p90): 0.001362 / 0.000255

## 1) 低补偿比榜单 (Top-K)
| split | source_id | sample_count | abs_residual_mean | abs_model_comp_mean | comp_ratio | abs_gap_mean | issue_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0239622 | 0.00598424 | 0.249736 | 0.0222382 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0215358 | 0.00632649 | 0.293766 | 0.0205501 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0166189 | 0.00526949 | 0.317078 | 0.0149728 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0227322 | 0.00753957 | 0.331669 | 0.0197534 | TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0189836 | 0.00634797 | 0.334393 | 0.0171497 | TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0207869 | 0.007032 | 0.338291 | 0.01783 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0175049 | 0.00598289 | 0.341784 | 0.0165029 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0187657 | 0.00787445 | 0.41962 | 0.0163407 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0198525 | 0.00910551 | 0.458659 | 0.0150889 | TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_8\csv_data\Conventional_flatland_no_8_Excellent_Excellent_Seg6.csv | 1535 | 0.000585761 | 0.000562942 | 0.961044 | 4.13331e-05 | TailRisk |

## 2) 高缺口榜单 (Top-K)
| split | source_id | sample_count | abs_gap_mean | abs_gap_p99 | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0222382 | 0.0520238 | 0.249736 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0205501 | 0.0526197 | 0.293766 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0197534 | 0.0504222 | 0.331669 | TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.01783 | 0.051267 | 0.338291 | TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0171497 | 0.0483504 | 0.334393 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0165029 | 0.0495712 | 0.341784 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0163407 | 0.0513398 | 0.41962 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0150889 | 0.049842 | 0.458659 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0149728 | 0.0463415 | 0.317078 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg2.csv | 700 | 0.0143675 | 0.0461254 | 6.889 | TailRisk\|DirectionBias |

## 3) 高尾部风险榜单 (Top-K)
| split | source_id | sample_count | abs_gap_p99 | abs_gap_mean | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 243 | 0.0526197 | 0.0205501 | 0.293766 | UnderComp\|TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 623 | 0.0520238 | 0.0222382 | 0.249736 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0513398 | 0.0163407 | 0.41962 | TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.051267 | 0.01783 | 0.338291 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0504222 | 0.0197534 | 0.331669 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.049842 | 0.0150889 | 0.458659 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 957 | 0.0495712 | 0.0165029 | 0.341784 | TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2192 | 0.0483504 | 0.0171497 | 0.334393 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 72 | 0.0463415 | 0.0149728 | 0.317078 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg2.csv | 700 | 0.0461254 | 0.0143675 | 6.889 | TailRisk\|DirectionBias |

## 4) 调参建议
1. 补偿幅值整体可用：重点检查方向偏置与长尾风险。
2. 长尾风险较高：建议提高困难样本占比或按 source 重采样。
3. 存在方向偏置：检查误差符号定义、补偿符号以及特征归一化。

## 5) 标签说明
1. UnderComp: comp_ratio < 0.30，补偿量明显不足。
2. TailRisk: abs_gap_p99 > 2 * abs_gap_mean，尾部风险偏高。
3. DirectionBias: residual_mean 与 comp_mean 符号相反。
4. GeneralizationRisk: 同源 val 相比 train 的 abs_gap_mean 恶化超过 20%。
