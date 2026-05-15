# Android Studio 端侧部署指导（无代码版）

本文档用于指导将本项目训练得到的模型集成到 Android Studio 终端侧应用中，重点说明：
- 数据预处理
- NPU 推理调用流程
- 推理结果解析
- 与基准 LQR 控制融合得到最终控制量

说明：本文件仅给出工程化流程与接口约束，不提供代码实现。

## 1. 部署目标与推荐产物

推荐部署产物（由 model_conversion_package 生成）：
- `adaptive_net_android.rknn`：NPU 推理模型
- `adaptive_net_android.fixed.onnx`：用于离线一致性校验（可选）
- `model_deploy_meta.json`：部署元数据（输入维度、归一化参数、控制模式等）

推荐输入来源：
- 优先使用训练后自动生成并拷贝到 `input_models/` 的 `*deploy_checkpoint_*.pt`
- 该文件同时包含 `state_dict + config + stats`，便于追溯与复现实验

## 2. 端侧工程模块划分建议

建议拆分为 6 个模块：
1. `SensorPipeline`：采集并对齐车辆状态与轨迹信息
2. `FeatureBuilder`：构建时序输入与标量输入
3. `Normalizer`：按 `model_deploy_meta.json` 做标准化
4. `NpuInferenceEngine`：封装 RKNN Runtime 调用
5. `ControlFusion`：将模型输出与 LQR 基准融合
6. `SafetyLimiter`：方向盘角度限幅与变化率约束

## 3. 输入输出契约（必须严格一致）

以导出模型为准：
- 输入 1：`time_features`，形状 `[1, SEQ_LEN, time_dim]`
- 输入 2：`scalar_features`，形状 `[1, 2]`
- 输出：`output`，形状 `[1, 4]`（当前导出头部固定 4 维）

其中：
- `SEQ_LEN`、`time_dim` 必须读取 `model_deploy_meta.json`
- 批大小固定为 1（端侧实时控制推荐）

## 4. 数据预处理流程

### 4.1 特征组织

时序特征窗口（长度 `SEQ_LEN`）建议包含：
- 横向误差 `e_y`
- 航向误差 `e_psi`
- 横滚角/俯仰角/角速度（与训练一致）

标量特征 2 维：
- 车速 `v`
- 轴距 `L`

关键要求：
- 特征顺序必须与训练时完全一致
- 单位必须一致（角度统一为弧度）
- 每个控制周期滑窗更新 1 帧，不可跳帧

### 4.2 标准化

使用 `model_deploy_meta.json` 中的统计量：
- `time_mean`、`time_std`
- `scalar_mean`、`scalar_std`

标准化公式：
- `x_norm = (x - mean) / (std + 1e-8)`

注意事项：
- 禁止在端侧重新估计 mean/std
- 若出现 std 极小值，必须按导出元数据与公式处理

## 5. NPU 推理调用流程（RKNN）

建议流程：
1. 应用启动时初始化 RKNN Runtime
2. 加载 `adaptive_net_android.rknn`
3. 预分配输入/输出缓冲区（避免频繁申请内存）
4. 每个控制周期写入标准化后的输入
5. 执行一次推理，读取输出张量
6. 交给 `ControlFusion` 计算最终控制量

性能建议：
- 推理线程与控制主线程解耦
- 为推理设置超时与降级策略
- 记录推理耗时 P50/P95/P99

### 5.1 Java/Kotlin -> JNI -> C -> RKNN 数据流

端侧实际数据链路建议按下图理解：

```mermaid
flowchart LR
		A[Java/Kotlin 业务层\n采集车辆状态与轨迹信息] --> B[FeatureBuilder\n构造 time_features 与 scalar_features 原始值]
		B --> C[JNI 接口层\nfloat[] time_array\nfloat[] scalar_array]
		C --> D[rknn_bridge_jni.c\nGetFloatArrayElements]
		D --> E[adaptive_net_rknn_infer_raw]
		E --> F[adaptive_net_rknn_normalize\n按 time_mean/time_std\nscalar_mean/scalar_std 标准化]
		F --> G[adaptive_net_rknn_infer_normalized]
		G --> H[rknn_inputs_set\n输入0: time_features\n输入1: scalar_features]
		H --> I[rknn_run\nNPU 执行推理]
		I --> J[rknn_outputs_get\n得到 output[4]]
		J --> K[C 层输出数组 output4]
		K --> L[JNI 返回 jfloatArray]
		L --> M[Java/Kotlin 控制层\n解析输出并与 LQR 融合]
		M --> N[SafetyLimiter\n限幅/变化率约束/异常回退]
```

各层职责如下：

1. Java/Kotlin 层
- 从传感器、定位、轨迹跟踪模块拿到原始状态
- 组织长度为 `SEQ_LEN` 的时序窗口
- 准备 2 维标量输入：`v`、`L`

2. JNI 层
- 把 Java 的 `float[]` 转成 native 可读指针
- 调用 C 接口，不直接参与模型计算

3. C 封装层
- 校验输入长度
- 对原始输入做标准化
- 调用 RKNN Runtime 执行推理
- 将输出拷贝到固定长度的结果数组

4. RKNN Runtime / NPU
- 接收两个输入张量
- 在 NPU 上完成前向推理
- 返回长度为 4 的输出张量

5. 控制融合层
- 按 `MODE` 解释模型输出
- 与 LQR 基准控制量融合
- 做安全限制后输出最终转角

### 5.2 与归档代码的对应关系

归档代码中，这条链路已经有一版可运行实现：

- [归档/cpp/rknn_bridge_jni.c](归档/cpp/rknn_bridge_jni.c)
	负责 Java/Kotlin 与 native C 的桥接

- [归档/cpp/adaptive_net_rknn.c](归档/cpp/adaptive_net_rknn.c)
	负责标准化、设置输入、执行 RKNN 推理、拷贝输出

- [归档/cpp/include/adaptive_net_rknn.h](归档/cpp/include/adaptive_net_rknn.h)
	负责定义固定输入输出维度与 C 接口声明

这部分归档代码证明了端侧推理链路本身是成立的；当前源文件的改进方向，是把原先硬编码在 C 文件里的统计量逐步替换为由导出流程自动生成的 `model_deploy_meta.json`。

## 6. 模型输出解析与控制融合

### 6.1 先计算 LQR 基准量

基准控制：
$$
\delta_{lqr} = -(K_e e_y + K_\theta e_\psi)
$$

其中 `K_e`、`K_theta` 来自现有 LQR 参数表或在线调度结果。

### 6.2 根据控制模式融合

从 `model_deploy_meta.json -> model_config.MODE` 读取模式。

#### 模式 A（增益调度）

模型输出解释为：
- `alpha_e_raw, beta_e_raw, alpha_th_raw, beta_th_raw`

融合逻辑：
1. 将 `alpha` 约束到 `MODE_A_ALPHA_RANGE`
2. 将 `beta` 约束到 `[-MODE_A_BETA_SCALE, MODE_A_BETA_SCALE]`
3. 计算新增益：
$$
K_e' = \alpha_e K_e + \beta_e, \quad
K_\theta' = \alpha_\theta K_\theta + \beta_\theta
$$
4. 最终控制：
$$
\delta = -(K_e' e_y + K_\theta' e_\psi)
$$

#### 模式 D（直接补偿）

模型输出第 1 维作为补偿项 `\Delta\delta`（按配置缩放）
$$
\delta = \delta_{lqr} + \Delta\delta
$$

说明：当前导出头部固定 4 维，若使用模式 D，仅消费约定维度，其余忽略。

## 7. 安全与鲁棒性策略

对最终控制量 `\delta` 必做三层约束：
1. 绝对限幅：`[STEERING_LIMIT_MIN, STEERING_LIMIT_MAX]`
2. 变化率限制：`|delta_t - delta_{t-1}| <= rate_limit`
3. 异常回退：推理失败/超时时回退到纯 LQR

建议额外加入：
- 输入健康检查（NaN/Inf/突变）
- 模型输出健康检查（越界/常量漂移）
- 回退状态打点与告警

## 8. 一致性验证（上线前必做）

至少做 3 类一致性：
1. `PyTorch -> ONNX` 一致性（离线）
2. `ONNX -> RKNN` 一致性（离线）
3. 端侧在线一致性（同一段日志回放）

验收建议：
- 控制量 MAE / 峰值误差 / 延迟
- 极端工况（急弯、低速抖动、坡道）
- 回退触发率与恢复时间

## 9. 文件交付清单（Android 集成最小集）

必需：
- `adaptive_net_android.rknn`
- `model_deploy_meta.json`
- 版本说明（模型时间戳、训练来源、模式）

可选：
- `adaptive_net_android.fixed.onnx`（用于问题定位）
- 离线回放样例数据（用于联调）

## 10. 常见问题排查

1. 推理输出接近常量
- 检查输入特征顺序/单位是否与训练一致
- 检查标准化是否使用正确 stats
- 检查滑窗是否正确滚动

2. 与离线结果偏差大
- 检查 NPU 输入 dtype/shape
- 检查模式选择（A/D）与参数缩放
- 检查控制后处理（限幅、rate limit）是否一致

3. 车辆出现振荡
- 降低模型补偿权重或收紧补偿上限
- 增加变化率约束
- 扩大回退条件并提高回退优先级

---

如需下一步，我可以基于本指导文档继续提供：
- Android 侧接口定义清单（不含实现）
- 联调检查清单（逐项可打勾）
- 上线验收模板（指标阈值版）
