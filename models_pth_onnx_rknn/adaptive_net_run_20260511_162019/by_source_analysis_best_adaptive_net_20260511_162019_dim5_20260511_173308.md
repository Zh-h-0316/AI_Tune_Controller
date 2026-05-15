# By-Source 问题榜单与调参建议

- 输入文件: D:\Huace_Work\AI_Control\AI_Tune\models_pth_onnx_rknn\adaptive_net_run_20260511_162019\pretrain_diagnostics_by_source_20260511_173302.csv
- 样本源数量: 147
- comp_ratio(mean/median/p90): 1.0739 / 0.9953 / 1.0069
- abs_gap_mean(mean/p90): 0.001373 / 0.000105

## 1) 低补偿比榜单 (Top-K)
| split | source_id | sample_count | abs_residual_mean | abs_model_comp_mean | comp_ratio | abs_gap_mean | issue_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 583 | 0.023932 | 0.00601212 | 0.251216 | 0.0221537 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 229 | 0.0214744 | 0.00634791 | 0.295603 | 0.0205827 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 68 | 0.0168779 | 0.00520496 | 0.308389 | 0.0149483 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0227323 | 0.00746785 | 0.328513 | 0.0197679 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2180 | 0.0190044 | 0.00631716 | 0.332406 | 0.0171914 | TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0207868 | 0.00694301 | 0.33401 | 0.0178571 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 935 | 0.0176231 | 0.00595144 | 0.337706 | 0.0165827 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0187656 | 0.00782075 | 0.416761 | 0.0163443 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0198525 | 0.00893797 | 0.450219 | 0.0151794 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Conventional_flatland_no_21\csv_data\Conventional_flatland_no_21_Excellent_Excellent_Seg2.csv | 47 | 0.0022846 | 0.00179963 | 0.787722 | 0.000487551 | TailRisk |

## 2) 高缺口榜单 (Top-K)
| split | source_id | sample_count | abs_gap_mean | abs_gap_p99 | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 583 | 0.0221537 | 0.0520322 | 0.251216 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 229 | 0.0205827 | 0.0529039 | 0.295603 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0197679 | 0.0503859 | 0.328513 | TailRisk\|DirectionBias |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0178571 | 0.0512013 | 0.33401 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2180 | 0.0171914 | 0.0484094 | 0.332406 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 935 | 0.0165827 | 0.0496335 | 0.337706 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0163443 | 0.0512407 | 0.416761 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0151794 | 0.0496853 | 0.450219 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 68 | 0.0149483 | 0.0462916 | 0.308389 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg2.csv | 700 | 0.0142599 | 0.0455422 | 6.82816 | TailRisk\|DirectionBias |

## 3) 高尾部风险榜单 (Top-K)
| split | source_id | sample_count | abs_gap_p99 | abs_gap_mean | comp_ratio | issue_tags |
| --- | --- | --- | --- | --- | --- | --- |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg7.csv | 229 | 0.0529039 | 0.0205827 | 0.295603 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg6.csv | 583 | 0.0520322 | 0.0221537 | 0.251216 | UnderComp\|TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg4.csv | 153 | 0.0512407 | 0.0163443 | 0.416761 | TailRisk |
| val | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg1.csv | 2606 | 0.0512013 | 0.0178571 | 0.33401 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg3.csv | 2334 | 0.0503859 | 0.0197679 | 0.328513 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg2.csv | 441 | 0.0496853 | 0.0151794 | 0.450219 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg5.csv | 935 | 0.0496335 | 0.0165827 | 0.337706 | TailRisk\|DirectionBias |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg9.csv | 2180 | 0.0484094 | 0.0171914 | 0.332406 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_2\csv_data\Heavy_load_no_2_Excellent_Excellent_Seg8.csv | 68 | 0.0462916 | 0.0149483 | 0.308389 | TailRisk |
| train | D:\Huace_Work\AI_Control\Excellent_Data\Process_Data\Heavy_load_no_6\csv_data\Heavy_load_no_6_Excellent_Excellent_Seg2.csv | 700 | 0.0455422 | 0.0142599 | 6.82816 | TailRisk\|DirectionBias |

## 4) 调参建议
1. 补偿幅值整体可用：重点检查方向偏置与长尾风险。
2. 长尾风险较高：建议提高困难样本占比或按 source 重采样。
3. 存在方向偏置：检查误差符号定义、补偿符号以及特征归一化。

## 5) 标签说明
1. UnderComp: comp_ratio < 0.30，补偿量明显不足。
2. TailRisk: abs_gap_p99 > 2 * abs_gap_mean，尾部风险偏高。
3. DirectionBias: residual_mean 与 comp_mean 符号相反。
4. GeneralizationRisk: 同源 val 相比 train 的 abs_gap_mean 恶化超过 20%。
