MOVE FILE TO: d:\Huace_Work\AI_Control\AI_Tune\deployment\check_deployment_env.py
"""
验证脚本：检查部署环境和工具是否正常工作
"""

import os
import sys
import importlib

def check_dependencies():
    """检查必要的依赖"""
    print("\n📋 检查依赖环境...")
    print("="*60)
    
    dependencies = {
        'torch': '必需',
        'numpy': '必需',
        'pandas': '必需',
    }
    
    optional_deps = {
        'onnx': '用于 ONNX 导出',
        'onnxruntime': '用于 ONNX 推理（推荐）',
    }
    
    missing_required = []
    missing_optional = []
    
    # 检查必需依赖
    for pkg, desc in dependencies.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', '已安装')
            print(f"✅ {pkg:20} {version:30} ({desc})")
        except ImportError:
            print(f"❌ {pkg:20} {'未安装':30} ({desc})")
            missing_required.append(pkg)
    
    print("\n📦 可选依赖:")
    for pkg, desc in optional_deps.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', '已安装')
            print(f"✅ {pkg:20} {version:30} ({desc})")
        except ImportError:
            print(f"⚠️  {pkg:20} {'未安装':30} ({desc})")
            missing_optional.append(pkg)
    
    return missing_required, missing_optional


def check_model_files():
    """检查模型文件是否存在"""
    print("\n📁 检查模型文件...")
    print("="*60)
    
    model_dir = r"D:\Huace_Work\AI_Control\AI_Tune\models_pth_onnx_rknn"
    
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return False
    
    print(f"✅ 模型目录: {model_dir}")
    
    # 查找模型文件
    model_files = [
        f for f in os.listdir(model_dir)
        if (f.startswith("adaptive_net_") or f.startswith("best_adaptive_net_")) 
        and f.endswith(".pth")
    ]
    
    config_files = [
        f for f in os.listdir(model_dir)
        if (f.startswith("config_") or f.startswith("best_config_")) 
        and f.endswith(".pt")
    ]
    
    print(f"\n📦 模型文件: {len(model_files)} 个")
    if model_files:
        latest_model = sorted(model_files, reverse=True)[0]
        model_size = os.path.getsize(os.path.join(model_dir, latest_model)) / (1024*1024)
        print(f"   最新模型: {latest_model} ({model_size:.2f} MB)")
    
    print(f"\n⚙️  配置文件: {len(config_files)} 个")
    if config_files:
        latest_config = sorted(config_files, reverse=True)[0]
        print(f"   最新配置: {latest_config}")
    
    return len(model_files) > 0 and len(config_files) > 0


def check_deployment_tools():
    """检查部署工具文件"""
    print("\n🛠️  检查部署工具...")
    print("="*60)
    
    base_dir = r"D:\Huace_Work\AI_Control\AI_Tune"
    
    tools = {
        'model_deployment.py': '模型转换工具',
        'deployment_reference.py': '部署参考代码',
        'inference_main.py': '推理脚本',
        'DEPLOYMENT_GUIDE.md': '部署指南',
    }
    
    all_present = True
    for filename, desc in tools.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024
            print(f"✅ {filename:30} {size:8.1f} KB  ({desc})")
        else:
            print(f"❌ {filename:30} {'不存在':8}  ({desc})")
            all_present = False
    
    return all_present


def recommend_deployment():
    """推荐部署方案"""
    print("\n💡 推荐部署方案...")
    print("="*60)
    
    strategies = [
        {
            'name': 'ONNX Runtime (推荐)',
            'pros': ['轻量级', '跨平台', '推理快'],
            'size': '~100 MB',
            'command': 'pip install onnxruntime numpy',
        },
        {
            'name': '量化模型 (最快)',
            'pros': ['最小文件', '推理最快', 'PyTorch 原生'],
            'size': '~200 MB',
            'command': 'pip install torch',
        },
        {
            'name': 'TorchScript (工业级)',
            'pros': ['C++ 支持', 'PyTorch 原生', '完全兼容'],
            'size': '~200 MB',
            'command': 'pip install torch',
        },
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   优点: {', '.join(strategy['pros'])}")
        print(f"   依赖大小: {strategy['size']}")
        print(f"   安装: {strategy['command']}")


def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型加载...")
    print("="*60)
    
    try:
        import torch
        print("✅ PyTorch 加载成功")
        
        model_dir = r"D:\Huace_Work\AI_Control\AI_Tune\models_pth_onnx_rknn"
        model_files = [
            f for f in os.listdir(model_dir)
            if (f.startswith("best_adaptive_net_") or f.startswith("adaptive_net_")) 
            and f.endswith(".pth")
        ]
        
        if model_files:
            latest_model = sorted(model_files, reverse=True)[0]
            model_path = os.path.join(model_dir, latest_model)
            
            print(f"\n📦 加载模型: {latest_model}")
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # 统计参数量
            total_params = sum(p.numel() for p in state_dict.values())
            total_size_mb = sum(p.numel() * 4 / (1024*1024) for p in state_dict.values())
            
            print(f"✅ 模型加载成功")
            print(f"   参数数量: {total_params:,} ({total_params/1e6:.1f}M)")
            print(f"   内存占用: {total_size_mb:.2f} MB")
        else:
            print("⚠️  未找到模型文件")
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")


def print_quick_start():
    """打印快速开始指南"""
    print("\n🚀 快速开始指南")
    print("="*60)
    
    quick_start = """
步骤 1: 导出模型（5分钟）
    python model_deployment.py
    选择最新模型 → 选择 [4] 全部导出

步骤 2: 安装推理依赖
    # 方案 A: ONNX Runtime (推荐，最轻量级)
    pip install onnxruntime numpy
    
    # 或方案 B: PyTorch (功能完整)
    pip install torch

步骤 3: 运行推理
    # ONNX 推理
    python inference_main.py \\
      --model models/deployment/adaptive_net_*.onnx \\
      --config models/deployment/deployment_config_*.pt \\
      --verbose
    
    # 或 PyTorch 推理
    python inference_main.py \\
      --model models/deployment/adaptive_net_*.pth \\
      --config models/deployment/deployment_config_*.pt \\
      --verbose

步骤 4: 性能测试
    python inference_main.py \\
      --model models/deployment/adaptive_net_*.onnx \\
      --config models/deployment/deployment_config_*.pt \\
      --verbose --benchmark

详见：DEPLOYMENT_GUIDE.md
    """
    print(quick_start)


def main():
    """主检查流程"""
    print("\n" + "="*60)
    print("模型部署环境检查工具")
    print("="*60)
    
    # 1. 检查依赖
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"\n❌ 缺失必需依赖: {', '.join(missing_required)}")
        print("请运行: pip install -r requirements.txt")
    
    if missing_optional:
        print(f"\n⚠️  建议安装可选依赖以获得完整功能: {', '.join(missing_optional)}")
    
    # 2. 检查文件
    has_models = check_model_files()
    has_tools = check_deployment_tools()
    
    # 3. 测试加载
    if has_models:
        test_model_loading()
    
    # 4. 推荐方案
    recommend_deployment()
    
    # 5. 快速开始
    print_quick_start()
    
    # 总结
    print("\n" + "="*60)
    print("✅ 检查完成！")
    print("="*60)
    
    if missing_required:
        print("\n⚠️  需要安装必需依赖后才能使用部署工具")
        return 1
    elif has_models and has_tools:
        print("\n✅ 环境已准备好，可以开始部署！")
        print("   运行: python model_deployment.py")
        return 0
    else:
        print("\n⚠️  部分文件缺失，请检查")
        return 1


if __name__ == "__main__":
    sys.exit(main())
