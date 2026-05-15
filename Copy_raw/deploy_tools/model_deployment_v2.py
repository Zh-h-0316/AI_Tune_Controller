"""
模型部署工具 v2 - 改进版本
- 自动处理路径问题
- 支持从任意目录运行
- 更好的错误提示
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import importlib.util

# 动态加载 Adaptive Network 模块
def _load_adaptive_module():
    """从正确的路径加载 Adaptive Network 模块"""
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "Adaptive Network.py"),  # deploy_tools/../Adaptive Network.py
        os.path.join(os.path.dirname(__file__), "Adaptive Network.py"),        # deploy_tools/Adaptive Network.py
        os.path.join(os.getcwd(), "Adaptive Network.py"),                      # 当前工作目录
        r"D:\Huace_Work\AI_Control\AI_Tune\Adaptive Network.py",              # 完整路径
    ]
    
    for module_path in possible_paths:
        if os.path.exists(module_path):
            print(f"📦 加载模块: {module_path}")
            spec = importlib.util.spec_from_file_location("adaptive_network_merged", module_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    
    raise ImportError(
        f"无法加载 'Adaptive Network.py'\n"
        f"已尝试位置:\n" + 
        "\n".join(f"  - {p}" for p in possible_paths) +
        f"\n请确保在工作目录下有 'Adaptive Network.py' 文件"
    )

try:
    _adaptive_module = _load_adaptive_module()
    AdaptiveNetwork = _adaptive_module.AdaptiveNetwork
    ControlMode = _adaptive_module.ControlMode
    TRAIN_CONFIG = _adaptive_module.TRAIN_CONFIG
except ImportError as e:
    print(f"❌ 错误: {e}")
    sys.exit(1)


class ModelDeployer:
    """模型部署工具类"""
    
    def __init__(self, model_path, config_path, device='cpu'):
        """
        初始化部署工具
        
        Args:
            model_path: 权重文件路径 (.pth)
            config_path: 配置文件路径 (.pt)
            device: 推理设备 ('cpu' 或 'cuda')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device(device)
        self.model = None
        self.config = None
        self.stats = None
        self._load_model()
        
    def _load_model(self):
        """加载模型和配置"""
        print(f"📋 加载配置文件: {self.config_path}")
        saved_data = torch.load(self.config_path, map_location='cpu', weights_only=False)
        self.stats = saved_data['stats']
        self.config = saved_data['config']
        
        # 转换 MODE 字符串为枚举
        mode_value = self.config.get('MODE', ControlMode.A)
        if isinstance(mode_value, str):
            self.config['MODE'] = ControlMode(mode_value)
        
        # 构建模型
        time_dim = self.stats['time_mean'].shape[0]
        self.model = AdaptiveNetwork(
            mode=self.config['MODE'],
            time_dim=time_dim,
            scalar_dim=2,
            hidden_size=self.config.get('HIDDEN_SIZE', 64),
            lstm_layers=self.config.get('LSTM_LAYERS', 2),
            lstm_dropout=self.config.get('LSTM_DROPOUT', 0.3),
            use_attention=self.config.get('USE_ATTENTION', True),
            mlp_hidden=self.config.get('MLP_HIDDEN', [128, 64]),
            mlp_dropout=self.config.get('MLP_DROPOUT', 0.2),
            mode_a_alpha_range=self.config.get('MODE_A_ALPHA_RANGE', (0.5, 1.5)),
            mode_a_beta_scale=self.config.get('MODE_A_BETA_SCALE', 0.1),
            mode_d_delta_scale=self.config.get('MODE_D_DELTA_SCALE', 0.1)
        ).to(self.device)
        
        print(f"📦 加载权重文件: {self.model_path}")
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True),
            strict=True
        )
        self.model.eval()
        print("✓ 模型加载完成")
    
    def export_to_onnx(self, output_path=None):
        """导出 ONNX 格式（最轻量级）"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                os.path.dirname(self.model_path),
                f"adaptive_net_{timestamp}.onnx"
            )
        
        print(f"\n📤 导出 ONNX 模型到: {output_path}")
        
        # 创建虚拟输入
        seq_len = self.config['SEQ_LEN']
        time_dim = self.stats['time_mean'].shape[0]
        dummy_time_series = torch.randn(1, seq_len, time_dim, device=self.device)
        dummy_scalar = torch.randn(1, 2, device=self.device)
        
        try:
            torch.onnx.export(
                self.model,
                (dummy_time_series, dummy_scalar),
                output_path,
                input_names=['time_series', 'scalar'],
                output_names=['output'],
                opset_version=12,
                dynamic_axes={
                    'time_series': {0: 'batch_size'},
                    'scalar': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )
            
            # 验证导出的 ONNX 模型
            try:
                import onnx
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
            except ImportError:
                pass  # ONNX 未安装，但导出成功
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ ONNX 导出成功，文件大小: {file_size_mb:.2f} MB")
            print(f"   推荐: pip install onnxruntime")
            return output_path
            
        except Exception as e:
            print(f"❌ ONNX 导出失败: {e}")
            return None
    
    def export_to_torchscript(self, output_path=None):
        """导出 TorchScript 格式"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                os.path.dirname(self.model_path),
                f"adaptive_net_{timestamp}.pt"
            )
        
        print(f"\n📤 导出 TorchScript 模型到: {output_path}")
        
        try:
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ TorchScript 导出成功，文件大小: {file_size_mb:.2f} MB")
            return output_path
            
        except Exception as e:
            print(f"❌ TorchScript 导出失败: {e}")
            return None
    
    def export_to_quantized(self, output_path=None, quantization_type='int8'):
        """导出量化模型"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                os.path.dirname(self.model_path),
                f"adaptive_net_quantized_{quantization_type}_{timestamp}.pth"
            )
        
        print(f"\n📤 导出量化模型到: {output_path}")
        print(f"   量化类型: {quantization_type}")
        
        try:
            if quantization_type == 'dynamic':
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
                )
            else:
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            torch.save(quantized_model.state_dict(), output_path)
            
            original_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            quantized_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            compression_ratio = (1 - quantized_size_mb / original_size_mb) * 100
            
            print(f"✅ 量化导出成功")
            print(f"   原始: {original_size_mb:.2f} MB → 量化后: {quantized_size_mb:.2f} MB")
            print(f"   压缩率: {compression_ratio:.1f}%")
            return output_path
            
        except Exception as e:
            print(f"❌ 量化导出失败: {e}")
            return None
    
    def export_config_and_stats(self, output_dir=None):
        """导出配置文件"""
        if output_dir is None:
            output_dir = os.path.dirname(self.model_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_output_path = os.path.join(output_dir, f"deployment_config_{timestamp}.pt")
        
        print(f"\n📤 导出配置文件到: {config_output_path}")
        
        try:
            deployment_config = {
                'stats': self.stats,
                'config': self.config,
                'time_dim': self.stats['time_mean'].shape[0],
            }
            torch.save(deployment_config, config_output_path)
            print(f"✅ 配置文件导出成功")
            return config_output_path
            
        except Exception as e:
            print(f"❌ 配置文件导出失败: {e}")
            return None
    
    def export_all_formats(self, output_dir=None):
        """一键导出所有格式"""
        if output_dir is None:
            output_dir = os.path.dirname(self.model_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("开始导出所有部署格式...")
        print("="*60)
        
        results = {}
        
        # 导出各种格式
        results['onnx'] = self.export_to_onnx()
        results['torchscript'] = self.export_to_torchscript()
        results['quantized_int8'] = self.export_to_quantized(quantization_type='int8')
        results['quantized_dynamic'] = self.export_to_quantized(quantization_type='dynamic')
        results['config'] = self.export_config_and_stats(output_dir)
        
        # 打印总结
        print("\n" + "="*60)
        print("导出总结:")
        print("="*60)
        for fmt, path in results.items():
            status = "✓" if path else "✗"
            print(f"{status} {fmt}: {path if path else '导出失败'}")
        
        return results


def select_model_for_deployment(model_dir):
    """选择要部署的模型"""
    models = sorted([
        f for f in os.listdir(model_dir)
        if (f.startswith("adaptive_net_") or f.startswith("best_adaptive_net_")) 
        and f.endswith(".pth")
    ], reverse=True)
    
    if not models:
        print("❌ 未找到任何模型文件")
        return None, None
    
    print("\n📦 可用模型列表:")
    for idx, m in enumerate(models, 1):
        model_path = os.path.join(model_dir, m)
        size = os.path.getsize(model_path) / (1024*1024)
        print(f"[{idx}] {m} ({size:.1f} MB)")
    
    while True:
        try:
            sel = input("\n请输入模型编号 (默认 1): ").strip()
            idx = int(sel) - 1 if sel else 0
            if 0 <= idx < len(models):
                model_name = models[idx]
                model_path = os.path.join(model_dir, model_name)
                
                # 查找对应的配置文件
                for config_prefix in ['best_config_', 'config_']:
                    config_name = model_name.replace(
                        model_name.split('_')[0] + '_' + model_name.split('_')[1],
                        config_prefix.rstrip('_'),
                        1
                    ).replace('.pth', '.pt')
                    config_path = os.path.join(model_dir, config_name)
                    if os.path.exists(config_path):
                        return model_path, config_path
                
                print(f"❌ 未找到对应的配置文件")
                continue
        except (ValueError, IndexError):
            print("❌ 输入无效，请输入数字")


def main():
    print("\n" + "="*60)
    print("自适应网络模型部署工具 v2")
    print("="*60)
    
    # 确定模型目录
    model_dir = TRAIN_CONFIG['MODEL_DIR']
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return
    
    # 选择模型
    model_path, config_path = select_model_for_deployment(model_dir)
    if not model_path or not config_path:
        print("❌ 模型选择失败")
        return
    
    # 初始化部署工具
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  使用设备: {device}")
    deployer = ModelDeployer(model_path, config_path, device=device)
    
    # 导出选项
    print("\n📤 部署格式选择:")
    print("[1] ONNX（推荐用于跨平台部署）")
    print("[2] TorchScript（推荐用于 C++ 部署）")
    print("[3] 量化模型（推荐用于减小文件大小和加速）")
    print("[4] 全部导出（推荐）")
    
    choice = input("\n请选择 [1-4]，默认 4: ").strip() or '4'
    
    output_dir = os.path.join(model_dir, "deployment")
    os.makedirs(output_dir, exist_ok=True)
    
    if choice == '1':
        deployer.export_to_onnx(os.path.join(output_dir, "adaptive_net.onnx"))
        deployer.export_config_and_stats(output_dir)
    elif choice == '2':
        deployer.export_to_torchscript(os.path.join(output_dir, "adaptive_net.pt"))
        deployer.export_config_and_stats(output_dir)
    elif choice == '3':
        deployer.export_to_quantized(os.path.join(output_dir, "adaptive_net_quantized.pth"), 'dynamic')
        deployer.export_config_and_stats(output_dir)
    else:
        deployer.export_all_formats(output_dir)
    
    print(f"\n✅ 导出完成，文件位置:")
    print(f"   {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ 程序已中止")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
