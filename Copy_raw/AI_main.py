import os
import shutil
import importlib.util
from datetime import datetime

# Local imports

def _load_adaptive_module():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(base_dir, "Adaptive Network.py")
    spec = importlib.util.spec_from_file_location("adaptive_network_merged", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载合并后的模块: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_adaptive_module = _load_adaptive_module()
TRAIN_CONFIG = _adaptive_module.TRAIN_CONFIG
ControlMode = _adaptive_module.ControlMode
train_network = _adaptive_module.train_network
run_comparison = _adaptive_module.run_comparison
fine_tune_with_rl = _adaptive_module.fine_tune_with_rl


def _select_test_trajectory():
    print("\n=== 测试轨迹选择 ===")
    print("[1] sine (默认)")
    print("[2] straight")
    print("[3] circle")
    print("[4] lane_change")
    mapping = {
        '1': 'sine',
        '2': 'straight',
        '3': 'circle',
        '4': 'lane_change',
        'sine': 'sine',
        'straight': 'straight',
        'circle': 'circle',
        'lane_change': 'lane_change'
    }
    while True:
        value = input("请选择测试轨迹 [1/2/3/4 或名称，回车默认sine]: ").strip().lower()
        if value == '':
            return 'sine'
        if value in mapping:
            return mapping[value]
        print("输入无效，请重新输入。")


def _select_rl_trajectory(default_path='sine'):
    print("\n=== RL微调轨迹选择 ===")
    print("[1] sine")
    print("[2] straight")
    print("[3] circle")
    print("[4] lane_change")
    mapping = {'1': 'sine', '2': 'straight', '3': 'circle', '4': 'lane_change'}
    inv = {v: k for k, v in mapping.items()}
    default_key = inv.get(default_path, '1')
    value = input(f"请选择RL训练轨迹 [1/2/3/4，回车默认{mapping[default_key]}]: ").strip().lower()
    if value == '':
        return mapping[default_key]
    return mapping.get(value, mapping[default_key])


def _configure_training_options():
    print("\n=== 训练参数选择 ===")
    print(f"当前模式: {TRAIN_CONFIG['MODE'].value if hasattr(TRAIN_CONFIG['MODE'], 'value') else TRAIN_CONFIG['MODE']}")
    print(f"当前俯仰角输入: {'启用' if TRAIN_CONFIG.get('USE_PITCH_FEATURE', True) else '禁用'}")

    # 训练模式选择
    mode_map = {
        'a': ControlMode.A,
        'b': ControlMode.B,
        'c': ControlMode.C,
        'd': ControlMode.D
    }
    while True:
        mode_in = input("选择训练模式 [A/B/C/D] (回车保持当前): ").strip().lower()
        if mode_in == '':
            break
        if mode_in in mode_map:
            TRAIN_CONFIG['MODE'] = mode_map[mode_in]
            if mode_in in ['b', 'c']:
                print("提示: 当前代码对 B/C 模式为基础实现，推荐优先使用 A 或 D。")
            break
        print("输入无效，请输入 A/B/C/D 或直接回车。")

    # 俯仰角特征开关
    while True:
        pitch_in = input("是否引入俯仰角及其历史作为输入? [Y/N] (回车保持当前): ").strip().lower()
        if pitch_in == '':
            break
        if pitch_in in ['y', 'yes']:
            TRAIN_CONFIG['USE_PITCH_FEATURE'] = True
            break
        if pitch_in in ['n', 'no']:
            TRAIN_CONFIG['USE_PITCH_FEATURE'] = False
            break
        print("输入无效，请输入 Y/N 或直接回车。")

    mode_show = TRAIN_CONFIG['MODE'].value if hasattr(TRAIN_CONFIG['MODE'], 'value') else TRAIN_CONFIG['MODE']
    print(f"训练配置确认 -> MODE: {mode_show}, USE_PITCH_FEATURE: {TRAIN_CONFIG.get('USE_PITCH_FEATURE', True)}")


def _select_model_and_config(model_dir, models, allow_default_latest=False):
    def _resolve_config_path(model_name):
        ts = model_name
        for prefix in ("adaptive_net_", "best_adaptive_net_", "RL_adaptive_net_"):
            if ts.startswith(prefix):
                ts = ts.replace(prefix, "", 1)
                break
        ts = ts.replace(".pth", "")

        candidates = [
            os.path.join(model_dir, f"best_config_{ts}.pt"),
            os.path.join(model_dir, f"RL_config_{ts}.pt"),
            os.path.join(model_dir, f"config_{ts}.pt")
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    while True:
        try:
            prompt = "请输入模型编号: "
            if allow_default_latest:
                prompt = "请输入模型编号（回车默认最新模型[1]）: "
            raw = input(prompt).strip()
            if allow_default_latest and raw == '':
                midx = 0
            else:
                midx = int(raw) - 1
            if midx < 0 or midx >= len(models):
                print("编号超出范围，请重新输入。")
                continue
            model_name = models[midx]
            model_path = os.path.join(model_dir, model_name)
            config_path = _resolve_config_path(model_name)
            if not os.path.exists(config_path):
                print("Error: 对应的配置文件丢失，请选择其他模型。")
                continue
            return model_path, config_path
        except ValueError:
            print("输入无效，请输入数字编号。")


def _backup_model_and_config(model_path, config_path):
    backup_dir = os.path.join(os.path.dirname(model_path), "preserved_models")
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_bak = os.path.join(backup_dir, f"preserved_{ts}_{os.path.basename(model_path)}")
    config_bak = os.path.join(backup_dir, f"preserved_{ts}_{os.path.basename(config_path)}")
    shutil.copy2(model_path, model_bak)
    shutil.copy2(config_path, config_bak)
    print(f"已备份历史模型: {model_bak}")
    print(f"已备份历史配置: {config_bak}")


def _maybe_rl_after_test(metrics, model_path, config_path, path_type):
    mae_lat = float(metrics.get('adaptive', {}).get('mae_lateral', 1.0)) if metrics else 1.0
    imp_lat = float(metrics.get('improvement', {}).get('mae_lateral', -1.0)) if metrics else -1.0
    suggest = (mae_lat > TRAIN_CONFIG.get('RL_TRIGGER_LATERAL_MAE', 0.10)) or (imp_lat <= 0.0)
    if suggest:
        print("测试结果建议执行RL微调。")
    ans = input("是否执行RL微调? [y/N]: ").strip().lower()
    if ans in ('y', 'yes'):
        rl_path = _select_rl_trajectory(default_path=path_type)
        rl_model, rl_config = fine_tune_with_rl(model_path, config_path, path_type=rl_path)
        if rl_model and rl_config:
            run_comparison(rl_model, rl_config, path_type=rl_path)

# ==================== 主程序 ====================
def main():
    model_dir = TRAIN_CONFIG['MODEL_DIR']
    
    # 1. Check for existing models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    models = sorted([
        f for f in os.listdir(model_dir)
        if (f.startswith("adaptive_net_") or f.startswith("best_adaptive_net_") or f.startswith("RL_adaptive_net_")) and f.endswith(".pth")
    ], reverse=True)
    
    choice = 'n'
    selected_model = None
    selected_config = None
    resume_model_path = None
    
    if models:
        print("\n发现历史训练模型:")
        for idx, m in enumerate(models):
            print(f"[{idx+1}] {m}")
        
        while True:
            resp = input("\n请选择: [N]ew Train (重新训练), [S]elect Test (选择并测试), [R]etrain (重新训练并保存新版), [F]inetune RL (强化学习微调): ").lower()
            if resp in ['n', 's', 'r', 'f']:
                choice = resp
                break
    
    if choice == 's' and models:
        try:
            selected_model, selected_config = _select_model_and_config(model_dir, models)
            selected_path = _select_test_trajectory()
            metrics = run_comparison(selected_model, selected_config, path_type=selected_path, num_runs=5)
            _maybe_rl_after_test(metrics, selected_model, selected_config, selected_path)
            return
        except:
            print("输入无效，默认重新训练。")
            choice = 'n'

    if choice == 'f' and models:
        try:
            selected_model, selected_config = _select_model_and_config(model_dir, models, allow_default_latest=True)
            selected_path = _select_rl_trajectory(default_path='sine')
            rl_model, rl_config = fine_tune_with_rl(selected_model, selected_config, path_type=selected_path)
            if rl_model and rl_config:
                run_comparison(rl_model, rl_config, path_type=selected_path)
            return
        except:
            print("强化学习微调流程输入无效，退出。")
            return

    if choice == 'r':
        if not models:
            print("未找到历史模型，R 将退化为 N（全新训练）。")
        else:
            print("\n=== 选择继续训练的基础模型 ===")
            for idx, m in enumerate(models):
                print(f"[{idx+1}] {m}")
            selected_model, selected_config = _select_model_and_config(model_dir, models, allow_default_latest=True)
            _backup_model_and_config(selected_model, selected_config)
            resume_model_path = selected_model

    if choice in ['n', 'r']:
        _configure_training_options()
        print("\n=== 开始训练流程 ===")
        m_path, c_path = train_network(resume_model_path=resume_model_path)
        
        if m_path and c_path:
            print("\n训练完成，正在进入对比测试...")
            selected_path = _select_test_trajectory()
            metrics = run_comparison(m_path, c_path, path_type=selected_path, num_runs=5)
            _maybe_rl_after_test(metrics, m_path, c_path, selected_path)
        else:
            print("训练未成功完成。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序终止")
