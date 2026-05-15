MOVE FILE TO: d:\Huace_Work\AI_Control\AI_Tune\deployment\README.md
# 部署工具集 - 文件索引与使用指南

## 📁 文件结构

```
deploy_tools/
├── README.md                           ← 你在这里（总览）
├── model_deployment.py                 ← 🔴 核心工具：模型格式转换
├── inference_main.py                   ← 🟢 推理脚本：通用推理接口
├── deployment_reference.py             ← 📚 参考代码：6 种部署方案
├── check_deployment_env.py             ← 🔍 环境检查：依赖验证
├── DEPLOYMENT_GUIDE.md                 ← 📖 完整指南：详细部署步骤
├── DEPLOYMENT_SOLUTION.md              ← 📋 完整方案：总体解决方案
└── QUICK_REFERENCE.py                  ← ⚡ 快速参考：速查表
```

## 🚀 快速开始（5 分钟）

### 第 1 步：导出模型
```bash
cd deploy_tools
python model_deployment.py
# 选择模型 → 选择 [4] 全部导出
✓ 输出保存到：../models/deployment/
```

### 第 2 步：安装依赖（二选一）
```bash
# 方案 A：轻量级 ONNX（推荐）
pip install onnxruntime numpy

# 方案 B：高性能量化（推荐速度）
pip install torch
```

### 第 3 步：运行推理
```bash
python inference_main.py \
  --model ../models/deployment/adaptive_net_*.onnx \
  --config ../models/deployment/deployment_config_*.pt \
  --verbose --benchmark
```

---

## 📚 文件详解

### 🔴 `model_deployment.py`（必用）
**功能**：一键导出模型为多种格式

**命令**：
```bash
python model_deployment.py
```

**输出格式**：
- ✅ ONNX 格式（12 MB，推荐通用）
- ✅ TorchScript 格式（15 MB，推荐 C++）
- ✅ 量化模型（5 MB，推荐性能）
- ✅ 配置文件（包含统计信息）

**使用时机**：
- 第一次部署前
- 模型更新后
- 需要转换格式时

---

### 🟢 `inference_main.py`（核心）
**功能**：通用推理脚本，支持所有格式

**命令**：
```bash
# 基础推理
python inference_main.py \
  --model ../models/deployment/model.onnx \
  --config ../models/deployment/config.pt \
  --verbose

# 性能测试
python inference_main.py \
  --model ../models/deployment/model.onnx \
  --config ../models/deployment/config.pt \
  --benchmark

# 使用 GPU
python inference_main.py \
  --model ../models/deployment/model.pth \
  --config ../models/deployment/config.pt \
  --device cuda
```

**支持的格式**：
- `.onnx` ← ONNX 模型
- `.pth` ← PyTorch 权重
- `.pt` ← TorchScript 模型

---

### 📚 `deployment_reference.py`
**功能**：参考代码，展示 6 种部署方案

**包含内容**：
1. ONNX Runtime 推理
2. TorchScript 推理
3. 量化模型推理
4. 完整推理管道
5. 性能测试对比
6. 最小依赖部署

**使用**：作为代码参考，复制相关代码到你的项目

---

### 🔍 `check_deployment_env.py`
**功能**：检查环境和依赖

**命令**：
```bash
python check_deployment_env.py
```

**检查项**：
- ✅ PyTorch/NumPy/Pandas 依赖
- ✅ ONNX/ONNX Runtime 可选依赖
- ✅ 模型文件是否存在
- ✅ 部署工具文件完整性
- ✅ 模型加载测试

---

### 📖 `DEPLOYMENT_GUIDE.md`
**功能**：完整部署指南

**内容**：
- 快速开始流程
- 三种部署方案详解
- Linux/Docker/树莓派 部署
- 故障排除指南
- 文件大小对比

**何时阅读**：需要了解详细部署步骤

---

### 📋 `DEPLOYMENT_SOLUTION.md`
**功能**：完整解决方案报告

**内容**：
- 问题分析
- 核心原因
- 完整方案说明
- 性能对比数据
- 验证清单

**何时阅读**：需要了解整体方案设计

---

### ⚡ `QUICK_REFERENCE.py`
**功能**：快速参考卡

**命令**：
```bash
python QUICK_REFERENCE.py
```

**内容**：
- 问题总览
- 3 步解决方案
- 性能对比表
- 文件导览
- 常见问题
- 一行命令速查

---

## 🎯 常见任务速查

### 我想...

#### 部署到 Linux 服务器
```bash
# 1. 导出 ONNX 模型
python model_deployment.py  # 选择 [1] ONNX

# 2. 在服务器上安装
pip install onnxruntime numpy

# 3. 复制文件到服务器
# 复制：adaptive_net_*.onnx 和 deployment_config_*.pt

# 4. 运行推理
python inference_main.py --model adaptive_net.onnx --config deployment_config.pt
```

#### 在树莓派上部署
```bash
# 1. 导出量化模型（最轻量）
python model_deployment.py  # 选择 [3] 量化

# 2. 在树莓派上安装（轻量级）
pip install onnxruntime-rpi4

# 3. 使用 ONNX 推理（推荐）
python inference_main.py --model adaptive_net.onnx --config deployment_config.pt
```

#### 在 C++ 中集成
```bash
# 1. 导出 TorchScript 格式
python model_deployment.py  # 选择 [2] TorchScript

# 2. C++ 代码使用 LibTorch 加载
#include <torch/script.h>
auto module = torch::jit.load("adaptive_net.pt");
```

#### 获得最快推理速度
```bash
# 1. 导出量化模型
python model_deployment.py  # 选择 [3] 量化 → [2] dynamic

# 2. 推理时使用量化模型
python inference_main.py --model adaptive_net_quantized_dynamic.pth --config config.pt
# 速度：5ms/推理（原来 15ms）
```

#### 减少部署文件体积
```bash
# 1. 导出量化模型 INT8（最小）
python model_deployment.py  # 选择 [3] 量化 → [1] int8
# 输出：5 MB（原来 20 MB）

# 2. 推理
python inference_main.py --model adaptive_net_quantized_int8.pth --config config.pt
```

---

## 💾 关键问题解答

### Q: 在终端部署时，是否可以只需要 `.pt` 文件，一定需要 `.pth` 文件吗？

**简短答案**：取决于部署方案

**详细分析**：

#### 文件含义
- **`.pth`** ← PyTorch 权重文件（state_dict）
  - 包含：模型参数
  - 大小：~20 MB
  - 用途：加载模型权重进行推理

- **`.pt`** ← PyTorch 序列化对象
  - 包含：配置 + 统计信息（stats）
  - 大小：~0.5 MB
  - 用途：保存模型架构参数和数据预处理信息

#### 三种部署方案

**方案 1：ONNX 部署（推荐）❌ 不需要 `.pth`**
```
需要：
  ✅ adaptive_net.onnx       （所有权重已转换）
  ✅ deployment_config.pt    （统计信息）

不需要：
  ❌ adaptive_net.pth        （权重已包含在 ONNX 中）

原理：ONNX 是通用格式，已包含所有权重，不依赖 PyTorch
```

**方案 2：TorchScript 部署 ❌ 不需要 `.pth`**
```
需要：
  ✅ adaptive_net.pt         （完整模型，包含权重和架构）
  ✅ deployment_config.pt    （统计信息）

不需要：
  ❌ adaptive_net.pth        （权重已包含在 TorchScript 中）

原理：TorchScript ✓ 是完整模型，包含架构和权重
```

**方案 3：PyTorch 直接推理 ✅ 必须要 `.pth`**
```
需要：
  ✅ adaptive_net.pth        （权重必须，用于 load_state_dict）
  ✅ deployment_config.pt    （配置和统计信息必须）

原理：PyTorch 推理需要架构 + 权重，架构从配置重建，权重从 .pth 加载
```

#### 结论表格

| 部署方案 | .pth 需要 | .pt 需要 | 文件大小 | 推荐 |
|---------|---------|---------|---------|------|
| **ONNX Runtime** | ❌ | ✅ | 12 MB | ⭐⭐⭐⭐⭐ |
| **TorchScript** | ❌ | ✅ | 15 MB | ⭐⭐⭐⭐ |
| **PyTorch 直接** | ✅ | ✅ | 20 MB | ⭐⭐⭐ |

---

### Q: 最小化部署包大小，应该怎么做？

**答案**：使用 ONNX 量化 + 只保留必要文件

```
部署包内容（最小化）：
├── adaptive_net_quantized.onnx    (5 MB，量化 ONNX)
├── deployment_config.pt           (0.5 MB)
└── inference_main.py              (11 KB)

总计：~5.5 MB

原来（未优化）：
├── adaptive_net.pth               (20 MB)
├── deployment_config.pt           (0.5 MB)
└── inference_main.py              (11 KB)

总计：~20.5 MB

节省：75% ✅
```

---

### Q: 终端环境没有 PyTorch，应该用哪个方案？

**答案**：ONNX Runtime

```bash
# ONNX Runtime 安装（仅需）
pip install onnxruntime numpy

# 这种情况不需要 PyTorch，不需要 .pth 文件
# 只需要：
# - adaptive_net.onnx
# - deployment_config.pt
```

---

## 🔗 文件依赖关系

```
部署工作流：

1️⃣ 模型导出
   Adaptive Network.py
            ↓
   model_deployment.py
            ↓
   导出多种格式：
   ├─ adaptive_net.onnx        ← 方案 A
   ├─ adaptive_net.pt          ← 方案 B
   ├─ adaptive_net_quantized   ← 方案 C
   └─ deployment_config.pt

2️⃣ 推理选择
   选择格式
      ↓
   ├─ ONNX     → inference_main.py (仅需 onnxruntime)
   ├─ TorchScript → inference_main.py (需要 torch)
   └─ PyTorch  → inference_main.py (需要 torch)
      ↓
   推理结果

3️⃣ 部署到终端
   选中格式文件
      ↓
   scp/ftp 复制到远端
      ↓
   远端运行 inference_main.py
      ↓
   获得推理结果
```

---

## ✅ 验证清单

- [ ] 模型已导出到 `../models/deployment/`
- [ ] 选择部署方案（ONNX / TorchScript / PyTorch）
- [ ] 安装对应依赖
- [ ] 测试 `inference_main.py` 推理
- [ ] 验证推理速度和精度
- [ ] 打包部署文件
- [ ] 部署到目标环境

---

## 💡 推荐方案速查

| 场景 | 推荐方案 | 理由 |
|------|--------|------|
| 通用服务器 | ONNX Runtime | 轻量、跨平台 |
| 性能优先 | 量化 ONNX | 最快最小 |
| C++ 集成 | TorchScript | 原生支持 |
| 树莓派 | ONNX Runtime | 资源受限 |
| 容器化 | ONNX + Docker | 可扩展 |

---

## 🚀 立即开始

```bash
# 1. 查看快速参考
python QUICK_REFERENCE.py

# 2. 检查环境
python check_deployment_env.py

# 3. 导出模型
python model_deployment.py

# 4. 运行推理
python inference_main.py --help
```

**祝你部署顺利！** 🎉
