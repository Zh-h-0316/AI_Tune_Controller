MOVE FILE TO: d:\Huace_Work\AI_Control\AI_Tune\deployment\deployment_reference.py
"""
部署参考代码：如何在终端环境中使用导出的各种格式模型
"""

# ============================================================
# 1. ONNX 部署（推荐 - 轻量级，无需 PyTorch）
# ============================================================

def deploy_with_onnx():
    """
    使用 ONNX Runtime 部署
    
    优势：
    - 不需要完整的 PyTorch
    - 文件大小小
    - 跨平台兼容
    - 推理速度快
    
    安装：
        pip install onnxruntime
    
    如果是在资源有限的终端：
        pip install onnxruntime-gpu  # 如果有 GPU
    """
    import onnxruntime as rt
    import numpy as np
    
    # 加载 ONNX 模型
    sess = rt.InferenceSession("adaptive_net_20260320_120000.onnx")
    
    # 准备输入数据
    seq_len = 10
    time_dim = 5
    batch_size = 1
    
    time_series = np.random.randn(batch_size, seq_len, time_dim).astype(np.float32)
    scalar = np.random.randn(batch_size, 2).astype(np.float32)
    
    # 运行推理
    outputs = sess.run(None, {
        'time_series': time_series,
        'scalar': scalar
    })
    
    prediction = outputs[0]
    print(f"预测输出: {prediction}")
    return prediction


# ============================================================
# 2. PyTorch 轻量级部署（仅需要 torch，无需其他依赖）
# ============================================================

def deploy_with_torchscript():
    """
    使用 TorchScript 部署
    
    优势：
    - PyTorch 原生格式
    - 可在各种环境运行
    - 支持 C++ 推理
    
    最小依赖：
        pip install torch
    """
    import torch
    
    # 加载 TorchScript 模型
    model = torch.jit.load("adaptive_net_20260320_120000.pt")
    model.eval()
    
    # 准备输入
    time_series = torch.randn(1, 10, 5)  # (batch, seq_len, time_dim)
    scalar = torch.randn(1, 2)           # (batch, scalar_dim)
    
    # 运行推理
    with torch.no_grad():
        output = model(time_series, scalar)
    
    print(f"预测输出: {output}")
    return output


# ============================================================
# 3. 量化模型部署（文件最小，推理最快）
# ============================================================

def deploy_with_quantized():
    """
    使用量化模型部署
    
    优势：
    - 文件大小最小（原来的 25%）
    - 推理速度最快（2-4 倍）
    - 仅需 PyTorch
    
    文件大小对比：
    - 原始模型: ~20 MB
    - TorchScript: ~15 MB
    - ONNX: ~12 MB
    - 量化模型: ~5 MB
    """
    import torch
    
    # 加载量化权重
    model_dict = torch.load("adaptive_net_quantized_dynamic_20260320_120000.pth")
    
    # 需要先创建模型架构（从配置文件加载）
    config_data = torch.load("deployment_config_20260320_120000.pt")
    stats = config_data['stats']
    config = config_data['config']
    
    # 重建模型架构并加载量化权重
    # 参考 Adaptive Network.py 中的模型定义
    from Adaptive Network import AdaptiveNetwork, ControlMode
    
    model = AdaptiveNetwork(
        mode=ControlMode(config['MODE']),
        time_dim=config_data['time_dim'],
        scalar_dim=2,
        **{k: config[k] for k in [
            'HIDDEN_SIZE', 'LSTM_LAYERS', 'LSTM_DROPOUT',
            'USE_ATTENTION', 'MLP_HIDDEN', 'MLP_DROPOUT',
            'MODE_A_ALPHA_RANGE', 'MODE_A_BETA_SCALE', 'MODE_D_DELTA_SCALE'
        ] if k in config}
    )
    
    model.load_state_dict(model_dict)
    model.eval()
    
    # 运行推理
    time_series = torch.randn(1, 10, 5)
    scalar = torch.randn(1, 2)
    
    with torch.no_grad():
        output = model(time_series, scalar)
    
    print(f"预测输出: {output}")
    return output


# ============================================================
# 4. 完整推理管道（带数据预处理和后处理）
# ============================================================

def inference_pipeline_onnx(model_path, config_path, input_data):
    """
    完整的 ONNX 推理管道
    
    Args:
        model_path: ONNX 模型文件路径
        config_path: 配置文件路径
        input_data: 输入数据字典 {
            'time_series': np.array (batch, seq_len, time_dim),
            'scalar': np.array (batch, 2)
        }
    
    Returns:
        输出字典 {'output': np.array}
    """
    import onnxruntime as rt
    import torch
    import numpy as np
    
    # 1. 加载配置和统计信息
    config_data = torch.load(config_path, map_location='cpu', weights_only=False)
    stats = config_data['stats']
    
    # 2. 数据标准化（使用训练时的统计信息）
    normalized_data = {}
    
    # 时间序列标准化
    time_series = input_data['time_series']  # np.array
    time_mean = stats['time_mean']
    time_std = stats['time_std']
    normalized_data['time_series'] = (
        (time_series - time_mean) / (time_std + 1e-8)
    ).astype(np.float32)
    
    # 标量标准化
    scalar = input_data['scalar']  # np.array
    scalar_mean = stats['scalar_mean']
    scalar_std = stats['scalar_std']
    normalized_data['scalar'] = (
        (scalar - scalar_mean) / (scalar_std + 1e-8)
    ).astype(np.float32)
    
    # 3. 加载模型并运行推理
    sess = rt.InferenceSession(model_path)
    outputs = sess.run(None, normalized_data)
    
    # 4. 返回结果
    return {'output': outputs[0]}


# ============================================================
# 5. 性能测试对比
# ============================================================

def benchmark_deployment_formats():
    """
    对比不同部署格式的性能
    
    输出：
    - 加载时间
    - 推理时间
    - 内存占用
    - 文件大小
    """
    import torch
    import time
    import os
    
    print("\n性能对比测试")
    print("=" * 60)
    
    test_files = {
        'ONNX': "adaptive_net_20260320_120000.onnx",
        'TorchScript': "adaptive_net_20260320_120000.pt",
        'Quantized': "adaptive_net_quantized_dynamic_20260320_120000.pth",
        'Original': "adaptive_net_20260320_120000.pth",
    }
    
    results = {}
    
    for fmt, path in test_files.items():
        if not os.path.exists(path):
            print(f"✗ {fmt}: 文件不存在 ({path})")
            continue
        
        # 文件大小
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # 加载时间
        start = time.time()
        if fmt == 'ONNX':
            import onnxruntime as rt
            model = rt.InferenceSession(path)
        else:
            model = torch.jit.load(path) if fmt == 'TorchScript' else torch.load(path)
        load_time = time.time() - start
        
        # 推理时间（10 次平均）
        times = []
        for _ in range(10):
            time_series = torch.randn(1, 10, 5) if fmt != 'ONNX' else None
            scalar = torch.randn(1, 2) if fmt != 'ONNX' else None
            
            start = time.time()
            if fmt == 'ONNX':
                import onnxruntime as rt
                import numpy as np
                model.run(None, {
                    'time_series': np.random.randn(1, 10, 5).astype(np.float32),
                    'scalar': np.random.randn(1, 2).astype(np.float32)
                })
            else:
                with torch.no_grad():
                    model(time_series, scalar)
            times.append(time.time() - start)
        
        avg_infer_time = sum(times) / len(times)
        
        results[fmt] = {
            'file_size_mb': file_size_mb,
            'load_time_ms': load_time * 1000,
            'infer_time_ms': avg_infer_time * 1000,
        }
        
        print(f"\n{fmt}:")
        print(f"  文件大小: {file_size_mb:.2f} MB")
        print(f"  加载时间: {load_time*1000:.2f} ms")
        print(f"  推理时间: {avg_infer_time*1000:.2f} ms")
    
    print("\n" + "=" * 60)
    print("推荐用途：")
    print("- ONNX: 最轻量，跨平台，推荐首选")
    print("- TorchScript: PyTorch 原生，支持 C++")
    print("- Quantized: 文件最小，速度最快")
    print("- Original: 精度最高，但文件最大")
    
    return results


# ============================================================
# 6. 在资源受限的终端环境中使用最小依赖部署
# ============================================================

def minimal_deployment_setup():
    """
    在资源受限的终端中，最小化依赖的部署方案
    
    推荐方案：
    1. 转换为 ONNX 格式
    2. 使用 onnxruntime（比 PyTorch 轻量 10 倍）
    3. 在终端上仅需要：
       - Python 3.7+
       - onnxruntime
       - numpy
    
    安装步骤：
        # 最小依赖（约 100 MB）
        pip install onnxruntime numpy
        
        # 对比：PyTorch 完整安装（约 500+ MB）
        pip install torch
    
    部署代码示例：
    """
    code_example = '''
import onnxruntime as rt
import numpy as np
import torch

# 1. 加载统计信息和配置
config_data = torch.load("deployment_config.pt", map_location='cpu', weights_only=False)
stats = config_data['stats']

# 2. 加载 ONNX 模型（轻量级）
session = rt.InferenceSession("adaptive_net.onnx", providers=['CPUExecutionProvider'])

# 3. 准备输入数据并标准化
def preprocess(time_series_raw, scalar_raw):
    time_series = (time_series_raw - stats['time_mean']) / (stats['time_std'] + 1e-8)
    scalar = (scalar_raw - stats['scalar_mean']) / (stats['scalar_std'] + 1e-8)
    return time_series.astype(np.float32), scalar.astype(np.float32)

# 4. 运行推理
time_series, scalar = preprocess(data_time_series, data_scalar)
outputs = session.run(None, {'time_series': time_series, 'scalar': scalar})

# 5. 获取结果
prediction = outputs[0]
print(f"预测: {prediction}")
    '''
    
    print("最小依赖部署方案")
    print("=" * 60)
    print("\n部署文件大小对比：")
    print("- PyTorch 完整: ~500+ MB")
    print("- ONNX Runtime: ~50 MB")
    print("- 模型文件: 5-20 MB（取决于量化）")
    print("\n总体大小: 55-70 MB vs 500+ MB（节省 85-90%）")
    print(f"\n示例代码：\n{code_example}")


if __name__ == "__main__":
    print("\n部署参考文档")
    print("=" * 60)
    print("\n请根据你的部署场景选择合适的方案：")
    print("\n1. 跨平台部署 → 使用 ONNX")
    print("2. CPU 服务器 → 使用量化模型 + ONNX Runtime")
    print("3. C++ 环境 → 使用 TorchScript")
    print("4. 开发调试 → 使用 PyTorch 原模型")
    print("5. 资源受限（树莓派等） → 使用 ONNX + 量化")
    
    # minimal_deployment_setup()
