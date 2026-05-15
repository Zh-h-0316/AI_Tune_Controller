# 模型转换独立包

这个目录是专门给"把当前 checkpoint 转成 ONNX / RKNN"的独立交付包。

适合直接交给别人使用，不依赖当前项目其它目录。

## 目录结构

- `input_models/` 原始输入模型文件
- `output_models/` 当前已经生成好的 ONNX / RKNN 输出文件
- `scripts/` 转换脚本
- `docker/rknn/` RKNN Toolkit2 的 Docker 环境
- `docs/` 中文方法说明

## 支持的输入格式

### 新格式（推荐）- 部署合并文件

训练完成后自动生成 `*deploy_checkpoint_*.pt` 文件并复制到 `input_models/`。
该文件包含 state_dict + config + stats，无需自定义加载器，标准 `torch.load` 即可读取。

### 旧格式 - 分离文件对

- `.pth` 权重文件 + `.pt` 配置文件
- 需要 `torch_zip_checkpoint_loader.py` 来正确解析

## 你最常用的文件

- `scripts/export_to_onnx.py` 从 checkpoint 导出 ONNX（自动检测新/旧格式）
- `scripts/convert_to_rknn.py` 从 ONNX 转换 RKNN
- `scripts/torch_zip_checkpoint_loader.py` 自定义 checkpoint 读取器（仅旧格式需要）
- `output_models/adaptive_net_android.rknn` 已生成好的 RKNN 模型
- `output_models/model_deploy_meta.json` 模型部署元数据（输入维度、归一化参数等，供 Android 端使用）

## 推荐使用方式

如果只想快速使用当前已有结果，直接拿:

- `output_models/adaptive_net_android.onnx`
- `output_models/adaptive_net_android.fixed.onnx`
- `output_models/adaptive_net_android.rknn`
- `output_models/model_deploy_meta.json`

如果想重新跑完整转换流程，优先看: `docs/使用方法_中文.md`

## 为什么旧格式需要自定义 checkpoint 读取器

旧版 `.pth` 权重文件不能直接完全依赖标准 `torch.load` 来正确恢复大矩阵权重。
使用新格式（部署合并文件）则不存在此问题。
