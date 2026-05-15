MOVE FILE TO: d:\Huace_Work\AI_Control\AI_Tune\deployment\inference_main.py
"""
推理脚本模板：直接使用导出的模型进行部署推理
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


def inference_with_onnx(model_path, config_path, time_series, scalar, verbose=False):
    """
    使用 ONNX 模型进行推理（最轻量级）
    
    Args:
        model_path: ONNX 模型文件路径
        config_path: 配置文件路径 (.pt)
        time_series: 时间序列输入 (batch, seq_len, time_dim)
        scalar: 标量输入 (batch, 2)
        verbose: 是否打印详细信息
        
    Returns:
        推理结果
    """
    try:
        import onnxruntime as rt
        import numpy as np
        import torch
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("   请运行: pip install onnxruntime numpy")
        return None
    
    try:
        # 加载配置和统计信息
        if verbose:
            print(f"📋 加载配置文件: {config_path}")
        config_data = torch.load(config_path, map_location='cpu', weights_only=False)
        stats = config_data['stats']
        
        # 加载 ONNX 模型
        if verbose:
            print(f"📦 加载 ONNX 模型: {model_path}")
        session = rt.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # 数据预处理（标准化）
        if verbose:
            print("🔄 执行数据预处理...")
        
        time_series_np = np.asarray(time_series, dtype=np.float32)
        scalar_np = np.asarray(scalar, dtype=np.float32)
        
        time_mean = np.asarray(stats['time_mean'], dtype=np.float32)
        time_std = np.asarray(stats['time_std'], dtype=np.float32)
        scalar_mean = np.asarray(stats['scalar_mean'], dtype=np.float32)
        scalar_std = np.asarray(stats['scalar_std'], dtype=np.float32)
        
        time_series_normalized = (time_series_np - time_mean) / (time_std + 1e-8)
        scalar_normalized = (scalar_np - scalar_mean) / (scalar_std + 1e-8)
        
        # 运行推理
        if verbose:
            print("⚙️ 运行模型推理...")
        outputs = session.run(None, {
            'time_series': time_series_normalized,
            'scalar': scalar_normalized
        })
        
        result = outputs[0]
        
        if verbose:
            print(f"✅ 推理完成，输出形状: {result.shape}")
        
        return result
        
    except Exception as e:
        print(f"❌ ONNX 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def inference_with_pytorch(model_path, config_path, time_series, scalar, device='cpu', verbose=False):
    """
    使用 PyTorch 模型进行推理
    
    Args:
        model_path: PyTorch 模型权重路径 (.pth) 或 TorchScript 路径 (.pt)
        config_path: 配置文件路径 (.pt)
        time_series: 时间序列输入
        scalar: 标量输入
        device: 'cpu' 或 'cuda'
        verbose: 是否打印详细信息
        
    Returns:
        推理结果
    """
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("   请运行: pip install torch")
        return None
    
    try:
        device = torch.device(device)
        
        # 加载配置
        if verbose:
            print(f"📋 加载配置文件: {config_path}")
        config_data = torch.load(config_path, map_location='cpu', weights_only=False)
        stats = config_data['stats']
        config = config_data['config']
        
        # 判断是 TorchScript 还是状态字典
        if verbose:
            print(f"📦 加载模型: {model_path}")
        
        if model_path.endswith('.pt'):
            # TorchScript 格式
            model = torch.jit.load(model_path, map_location=device)
        else:
            # 状态字典格式（.pth）
            import importlib.util
            import os
            # 动态加载 Adaptive Network.py
            base_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.join(base_dir, "..", "Adaptive Network.py")
            spec = importlib.util.spec_from_file_location("adaptive_network_merged", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载模块: {module_path}")
            adaptive_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adaptive_module)
            AdaptiveNetwork = adaptive_module.AdaptiveNetwork
            ControlMode = adaptive_module.ControlMode
            
            time_dim = config_data.get('time_dim') or stats['time_mean'].shape[0]
            mode_value = config['MODE']
            if isinstance(mode_value, str):
                mode_value = ControlMode(mode_value)
            
            model = AdaptiveNetwork(
                mode=mode_value,
                time_dim=time_dim,
                scalar_dim=2,
                hidden_size=config.get('HIDDEN_SIZE', 64),
                lstm_layers=config.get('LSTM_LAYERS', 2),
                lstm_dropout=config.get('LSTM_DROPOUT', 0.3),
                use_attention=config.get('USE_ATTENTION', True),
                mlp_hidden=config.get('MLP_HIDDEN', [128, 64]),
                mlp_dropout=config.get('MLP_DROPOUT', 0.2),
                mode_a_alpha_range=config.get('MODE_A_ALPHA_RANGE', (0.5, 1.5)),
                mode_a_beta_scale=config.get('MODE_A_BETA_SCALE', 0.1),
                mode_d_delta_scale=config.get('MODE_D_DELTA_SCALE', 0.1)
            ).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
        
        model.eval()
        
        # 数据预处理
        if verbose:
            print("🔄 执行数据预处理...")
        
        time_series_np = np.asarray(time_series, dtype=np.float32)
        scalar_np = np.asarray(scalar, dtype=np.float32)
        
        time_mean = torch.from_numpy(np.asarray(stats['time_mean'], dtype=np.float32)).to(device)
        time_std = torch.from_numpy(np.asarray(stats['time_std'], dtype=np.float32)).to(device)
        scalar_mean = torch.from_numpy(np.asarray(stats['scalar_mean'], dtype=np.float32)).to(device)
        scalar_std = torch.from_numpy(np.asarray(stats['scalar_std'], dtype=np.float32)).to(device)
        
        time_series_tensor = torch.from_numpy(time_series_np).to(device)
        scalar_tensor = torch.from_numpy(scalar_np).to(device)
        
        time_series_normalized = (time_series_tensor - time_mean) / (time_std + 1e-8)
        scalar_normalized = (scalar_tensor - scalar_mean) / (scalar_std + 1e-8)
        
        # 运行推理
        if verbose:
            print("⚙️ 运行模型推理...")
        
        with torch.no_grad():
            output = model(time_series_normalized, scalar_normalized)
        
        result = output.cpu().numpy()
        
        if verbose:
            print(f"✅ 推理完成，输出形状: {result.shape}")
        
        return result
        
    except Exception as e:
        print(f"❌ PyTorch 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_sample_input(seq_len=10, time_dim=5, batch_size=1):
    """创建示例输入数据"""
    import numpy as np
    
    time_series = np.random.randn(batch_size, seq_len, time_dim).astype(np.float32)
    scalar = np.random.randn(batch_size, 2).astype(np.float32)
    
    return time_series, scalar


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自适应网络模型推理脚本')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径 (.onnx, .pt, .pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='推理设备')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批大小')
    parser.add_argument('--seq-len', type=int, default=10,
                       help='时间序列长度')
    parser.add_argument('--time-dim', type=int, default=5,
                       help='时间序列维度')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能测试')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    print("\n" + "="*60)
    print("自适应网络推理")
    print("="*60)
    
    # 创建示例输入
    if args.verbose:
        print(f"\n📊 创建示例输入数据:")
        print(f"   批大小: {args.batch_size}")
        print(f"   序列长度: {args.seq_len}")
        print(f"   时间维度: {args.time_dim}")
    
    time_series, scalar = create_sample_input(
        seq_len=args.seq_len,
        time_dim=args.time_dim,
        batch_size=args.batch_size
    )
    
    print(f"\n📥 输入数据形状:")
    print(f"   时间序列: {time_series.shape}")
    print(f"   标量: {scalar.shape}")
    
    # 选择推理方案
    model_ext = os.path.splitext(args.model)[1].lower()
    
    if args.verbose:
        print(f"\n🔍 检测到模型格式: {model_ext}")
    
    if model_ext == '.onnx':
        print(f"\n🚀 使用 ONNX Runtime 推理...")
        result = inference_with_onnx(
            args.model,
            args.config,
            time_series,
            scalar,
            verbose=args.verbose
        )
    else:
        print(f"\n🚀 使用 PyTorch 推理...")
        result = inference_with_pytorch(
            args.model,
            args.config,
            time_series,
            scalar,
            device=args.device,
            verbose=args.verbose
        )
    
    if result is not None:
        print(f"\n📤 推理结果:")
        print(f"   输出形状: {result.shape}")
        print(f"   输出样本: {result[0]}")
        
        if args.benchmark:
            import time
            print(f"\n⏱️ 性能测试 (10 次运算平均)...")
            times = []
            for _ in range(10):
                start = time.time()
                if model_ext == '.onnx':
                    inference_with_onnx(args.model, args.config, time_series, scalar)
                else:
                    inference_with_pytorch(args.model, args.config, time_series, scalar, device=args.device)
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            print(f"   平均推理时间: {avg_time*1000:.2f} ms")
            print(f"   吞吐量: {args.batch_size/avg_time:.1f} 样本/秒")
        
        print("\n✅ 推理成功!")
    else:
        print("\n❌ 推理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
