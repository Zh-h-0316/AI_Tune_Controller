# Adaptive LQR 网络设计文档（中文版）

## 1. 设计目标

本项目通过监督学习拟合“历史状态序列 -> 最优前轮转角”映射关系，形成可在线实时调用的自适应控制网络，与 LQR 控制器协同工作。核心目标如下：

1. 提升横向路径跟踪精度（减小横向误差与航向误差）。
2. 降低传统 LQR 参数反复调参成本。
3. 在速度变化、轨迹变化、轻微建模误差下保持鲁棒性。
4. 满足执行器物理约束（转角限幅）并抑制控制抖动。

---

## 2. 总体方案

控制框架采用“模型控制 + 数据驱动补偿”的方式：

- 基线控制：LQR 根据当前误差计算基础转角。
- 网络补偿：网络根据历史误差序列与标量状态输出增益修正量（Mode A）或附加转角（Mode D）。
- 输出融合：得到最终转角命令并进行物理限幅。

该方案在保持经典控制器可解释性与稳定性的同时，引入数据驱动自适应能力。

---

## 3. 数据设计

### 3.1 输入特征

- 时间序列特征（默认 5 维，可配置）：
  - 横向误差 `e_y`
  - 航向误差 `e_psi`
  - 横滚角 `roll`
  - 俯仰角 `pitch`（可选）
  - 车身航向变化率 `omega`（航向角速度）
- 可选差分特征（额外 +2 维）：
  - `Δe_y`
  - `Δe_psi`
- 标量特征（2 维）：
  - 车速 `v`
  - 轴距 `L`

### 3.2 监督标签
- 监督目标为离线“最优前轮转角” `delta_opt`（由优质历史数据给出）。

### 3.3 样本构造

- 使用滑动窗口构造长度为 `SEQ_LEN` 的序列样本：
  - 输入：历史 `SEQ_LEN` 帧状态序列 + 当前标量状态
  - 输出：当前时刻最优转角 `delta_opt`
- `omega` 的来源：
  - 训练数据：CSV 的 `I` 列（角速度）并统一转为 `rad/s`
  - 在线推理：由状态估计/传感器模块直接提供 `omega` 数值
- `pitch` 的开关策略：
  - `USE_PITCH_FEATURE=True`：`pitch` 及其历史序列参与模型输入
  - `USE_PITCH_FEATURE=False`：`pitch` 不作为输入特征，不引入模型数据源

### 3.4 数据划分策略（已优化）

- **Group Split**：按源文件/轨迹分组，避免同一轨迹窗口同时进入训练集与验证集（防泄漏）。

- **速度分层划分**：按速度区间分层选择验证组，降低 train/val 速度分布偏差。
- **归一化无泄漏**：仅用训练集统计量（mean/std）归一化训练与验证数据。

---


### 4.1 主体结构
- 时序编码器：LSTM（支持多层、dropout）。
- 可选时间注意力：对各时刻隐状态加权汇聚。
- 融合层：时序特征与标量特征拼接后进入 MLP。
- 残差连接：融合特征到输出空间的投影残差，提升收敛稳定性。

### 4.2 输出模式

- Mode A（推荐）：输出增益修正参数
  - `alpha_e, beta_e, alpha_th, beta_th`
  - 自适应调整 LQR 增益后再计算转角
- Mode D：直接输出附加转角 `delta_add`

---

## 5. 数学形式

### 5.1 基线控制

$$
\delta_{base} = -k_e e_y - k_{\theta} e_{\psi}
$$

### 5.2 Mode A 自适应增益

$$
k_e' = \alpha_e k_e + \beta_e, \quad k_{\theta}' = \alpha_{\theta} k_{\theta} + \beta_{\theta}
$$

$$
\delta_{pred} = -k_e' e_y - k_{\theta}' e_{\psi}
$$

### 5.3 总损失函数（已实现）

1) 跟踪损失（限幅一致性后计算）

$$
\mathcal{L}_{track} = \text{SmoothL1}(\text{clip}(\delta_{pred}),\ \text{clip}(\delta_{opt}))
$$

2) 越界惩罚（抑制超出执行器边界）

$$
\mathcal{L}_{limit} = \mathbb{E}[\max(0, \delta_{pred}-\delta_{max}) + \max(0, \delta_{min}-\delta_{pred})]
$$

3) 变化率惩罚（抑制抖动）

$$
\mathcal{L}_{rate} = \mathbb{E}[|\text{clip}(\delta_{pred}) - \text{clip}(\delta_{prev})|]
$$

4) 组合目标

$$
\mathcal{L} = \mathcal{L}_{track} + \lambda_{limit}\mathcal{L}_{limit} + \lambda_{rate}\mathcal{L}_{rate}
$$

---

## 6. 训练策略（当前版本）

- 优化器：Adam（含权重衰减）。
- 学习率策略：CosineAnnealingLR。
- 损失：SmoothL1（Huber）。
- 稳定化：梯度裁剪（`clip_grad_norm_`）。
- 正则化：LSTM dropout + MLP dropout + 权重衰减。
- 早停：按验证损失监控并保存 best model。

### 6.1 训练日志与评估

每个 epoch 记录：

- `train/val loss`
- `train/val overall MAE`
- `train/val 各速度箱 MAE`
- `各速度箱样本数`
- `学习率`

并输出至 `training_metrics_*.csv`，用于离线分析泛化能力。

---

## 7. 在线控制与部署逻辑

1. 读取当前状态并维护历史窗口。
2. 计算 LQR 基线控制量。
3. 网络前向得到自适应修正（Mode A 或 Mode D）。
4. 合成最终转角命令。
5. 对最终转角执行物理限幅 `STEERING_LIMIT_MIN/MAX`。

该流程保证在线推理与训练目标在“约束边界”上保持一致，降低部署偏差。

---

## 8. 与功能目标的对应关系

### 8.1 提升跟踪效果

- 时序特征（含 `omega`）+ 注意力提升对动态误差演化的建模能力。
- 速度分层评估可定位高速工况短板并定向优化。

### 8.2 减少调参成本

- 通过监督学习自动学习增益修正规律，减少人工调 `Q/R` 频次。

### 8.3 提升鲁棒性

- 分组切分 + 分层切分减少过拟合假象。
- 限幅一致性损失与在线限幅增强物理可执行性。
- 变化率惩罚降低输出抖动，提升控制稳定性。

---

## 9. 推荐参数起点（可按数据规模微调）

- `SEQ_LEN = 10~20`
- `HIDDEN_SIZE = 64~128`
- `LSTM_LAYERS = 2`
- `LSTM_DROPOUT = 0.2~0.3`
- `MLP_HIDDEN = [128, 64]`
- `WEIGHT_DECAY = 1e-4`
- `HUBER_BETA = 0.02~0.05`
- `GRAD_CLIP_NORM = 1.0`
- `STEER_LIMIT_LOSS_WEIGHT = 0.05~0.2`
- `RATE_LOSS_WEIGHT = 0.02~0.1`

说明：若引入 `omega` 后噪声偏大，可适度提高 `SEQ_LEN` 或在数据侧增加角速度低通滤波。

---

## 10. 下一步可扩展方向

1. 不确定度估计与门控融合：按置信度融合网络输出与 LQR，异常场景自动保守。
2. OOD 检测与回退机制：速度/误差超分布时回退纯 LQR。
3. 多目标训练：联合优化 `e_y`、`e_psi` 与控制能量指标。
4. 轻量化部署：模型蒸馏或低秩压缩，降低车载推理延迟。

---

## 11. 结论

当前网络方案已从“纯拟合最优转角”升级为“带物理约束、分层评估、抗抖动”的工程化训练框架，能够更好支撑实时横向控制应用。通过持续引入不确定度融合与 OOD 回退机制，可进一步提升复杂工况下的稳定性与鲁棒性。

---

## 12. 实现版（代码落地说明）

本章面向研发实现，按当前代码结构给出“模块职责 -> 输入输出 -> 关键逻辑 -> 可调参数”的对应说明。

### 12.1 文件与模块职责

- `Adaptive Network.py`
  - 数据加载、样本构造、数据集定义
  - 自适应网络结构定义
  - 训练与验证流程（含分层 MAE 日志）
  - 在线控制器封装与对比仿真
- `Config_Para.py`
  - 车辆参数、执行器约束、仿真相关参数

### 12.2 关键对象与接口

1) `load_samples(root_dir, seq_len)`

- 输入：数据根目录、时间窗长度。
- 输出：`samples(list[dict])`。
- 样本字段：
  - `time_series`: `(SEQ_LEN, time_dim)`
  - `scalar`: `[v, wheelbase]`
  - `e`, `theta`, `delta_opt`, `wheelbase`, `source_id`
- 说明：
  - `source_id` 用于 Group Split 防泄漏。
  - `time_dim` 由开关动态决定：
    - `USE_PITCH_FEATURE=True`：基础维度 5（`e_y/e_psi/roll/pitch/omega`）
    - `USE_PITCH_FEATURE=False`：基础维度 4（`e_y/e_psi/roll/omega`）
    - 开启 `USE_DIFF_FEATURE` 后额外 +2（`Δe_y/Δe_psi`）
  - 支持差分特征开关 `USE_DIFF_FEATURE`。

2) `ControlDataset(samples, stats=None, augment=False)`

- 输入：样本、可选统计量、增强开关。
- 输出：训练可直接迭代的张量元组：
  - `norm_time, norm_scalar, raw_scalar, e, theta, delta_opt, wheelbase`
- 说明：
  - `stats=None` 时按当前数据计算归一化统计。
  - 验证集应传入训练集统计量，避免信息泄漏。

3) `AdaptiveNetwork(...).forward(time_seq, scalar)`

- 输入：
  - `time_seq`: `(B, T, time_dim)`
  - `scalar`: `(B, 2)`
- 输出：
  - Mode A：`alpha_e, beta_e, alpha_th, beta_th`
  - Mode D：`delta_add`
- 说明：
  - 时序分支：LSTM +（可选）注意力。
  - 标量分支：MLP。
  - 融合后经 MLP 输出并加残差投影。

4) `train_network()`

- 功能：端到端训练 + 验证 + 模型保存 + CSV 日志导出。
- 关键步骤：
  - 构造 `samples`。
  - Group Split + 速度分层划分 train/val。
  - 训练集统计归一化并复用于验证集。
  - 训练时计算组合损失（见 12.4）。
  - 每个 epoch 记录 `loss/MAE/速度箱 MAE/箱样本数/lr`。
  - 保存 `best_model`、`final_model`、`config`、`training_metrics.csv`。

5) `AdaptiveLQRController.get_control(state, e_y, e_psi, omega)`

- 功能：在线控制接口。
- 输入：当前车辆状态、误差与外部直接提供的 `omega`。
- 输出：`(delta_final, delta_lqr)`。
- 关键逻辑：
  - 接收外部输入 `omega`，并与 `e_y/e_psi` 一起维护等长历史窗口。
  - 先计算 `delta_lqr`。
  - 网络估计自适应修正（Mode A/D）。
  - 生成 `delta_final` 并做物理限幅。

### 12.3 数据划分实现细节

1. 计算 `val_size = max(1, int(N * VAL_SPLIT))`。
2. 以 `source_id` 聚合样本为组，保证同组不跨 train/val。
3. 若启用 `USE_SPEED_STRATIFIED_SPLIT`：
   - 先按组均速映射到速度箱；
   - 各速度箱内按比例抽取验证组；
   - 若验证样本不足，再从剩余组补齐。
4. 打印并记录 train/val 各速度箱样本数，确认分布合理。

### 12.4 损失函数实现（训练与验证一致）

设执行器转角边界为 $[\delta_{min}, \delta_{max}]$。

1) 跟踪损失（限幅一致性）

$$
\mathcal{L}_{track}=\text{SmoothL1}(\text{clip}(\delta_{pred}),\text{clip}(\delta_{opt}))
$$

2) 越界惩罚

$$
\mathcal{L}_{limit}=\mathbb{E}[\max(0,\delta_{pred}-\delta_{max})+\max(0,\delta_{min}-\delta_{pred})]
$$

3) 变化率惩罚

$$
\mathcal{L}_{rate}=\mathbb{E}[|\text{clip}(\delta_{pred})-\text{clip}(\delta_{prev})|]
$$

4) 总损失

$$
\mathcal{L}=\mathcal{L}_{track}+\lambda_{limit}\mathcal{L}_{limit}+\lambda_{rate}\mathcal{L}_{rate}
$$

其中：

- $\lambda_{limit}$ 对应 `STEER_LIMIT_LOSS_WEIGHT`
- $\lambda_{rate}$ 对应 `RATE_LOSS_WEIGHT`

### 12.5 训练日志输出规范

每个 epoch 输出与记录：

- `train_loss`, `val_loss`
- `train_mae`, `val_mae`
- `train_mae_bin_i`, `val_mae_bin_i`
- `train_count_bin_i`, `val_count_bin_i`
- `lr`

CSV 文件命名：`training_metrics_YYYYMMDD_HHMMSS.csv`。

### 12.6 运行与调用

1. 直接运行 `Adaptive Network.py`：
   - 先执行 `train_network()`。
   - 训练成功后执行 `run_comparison(model_path, config_path)`。

2. 在线集成最小接口：
   - 初始化：`AdaptiveLQRController(model_path, config_path)`
  - 循环调用：`get_control(state, e_y, e_psi, omega)`

### 12.7 关键配置项（实现侧）

- 数据与切分：
  - `DATA_ROOT`, `VAL_SPLIT`, `USE_SPEED_STRATIFIED_SPLIT`, `SPEED_BINS_MPS`
  - `DATA_FILTER_ENABLED`, `INCLUDE_DIR_KEYWORDS`, `EXCLUDE_DIR_KEYWORDS`, `INCLUDE_FILE_KEYWORDS`, `EXCLUDE_FILE_KEYWORDS`
- 模型：
  - `SEQ_LEN`, `HIDDEN_SIZE`, `LSTM_LAYERS`, `LSTM_DROPOUT`, `USE_ATTENTION`, `MLP_HIDDEN`, `MLP_DROPOUT`, `USE_PITCH_FEATURE`
- 优化：
  - `LR`, `WEIGHT_DECAY`, `HUBER_BETA`, `GRAD_CLIP_NORM`, `EARLY_STOP_PATIENCE`
- 约束与平滑：
  - `STEER_LIMIT_LOSS_WEIGHT`, `RATE_LOSS_WEIGHT`

### 12.8 验收标准（建议）

1. 训练过程：
   - `val_loss` 随 epoch 整体下降或稳定。
   - 高速箱 `val_mae_bin` 不明显劣化。

2. 仿真对比：
   - Adaptive LQR 在 `MAE Lateral` 与 `MAE Heading` 上优于或不劣于 Pure LQR。
   - 转角输出无明显越界与高频抖动。

3. 工程一致性：
  - 训练与推理输入特征定义一致（时序含 `omega`，`pitch` 由 `USE_PITCH_FEATURE` 控制是否启用，标量为 `[v, L]`）。
   - 推理端执行最终转角限幅。

### 12.9 可选数据过滤使用示例

- 目标：仅训练 `flatland` 数据，排除 `rain` 场景。
- 配置示例：
  - `DATA_FILTER_ENABLED = True`
  - `INCLUDE_DIR_KEYWORDS = ['flatland']`
  - `EXCLUDE_DIR_KEYWORDS = ['rain']`
  - `INCLUDE_FILE_KEYWORDS = []`
  - `EXCLUDE_FILE_KEYWORDS = []`

说明：过滤逻辑在递归发现 `csv_data` 目录和读取 `.csv` 文件两个阶段同时生效。默认关闭过滤，不影响全量训练。

网络模型未将以下三方面囊括其中：
定位数据源、执行器特性、前轮转角估计
因此执行器特性、前轮转角估计对实际作业效果会产生交叉影响
即预测的目标前轮转角优良，实际作业效果也可能较差