import os
import re
import shutil
import importlib.util
import sys
import torch
from datetime import datetime

# Local imports

def _load_adaptive_module():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(base_dir, "Adaptive Network.py")
    module_name = "adaptive_network_merged"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        return existing_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载合并后的模块: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_adaptive_module = _load_adaptive_module()
TRAIN_CONFIG = _adaptive_module.TRAIN_CONFIG
ControlMode = _adaptive_module.ControlMode
train_network = _adaptive_module.train_network
run_comparison = _adaptive_module.run_comparison
fine_tune_with_rl = _adaptive_module.fine_tune_with_rl
generate_deployment_files = _adaptive_module.generate_deployment_files
bundle_model_related_files = _adaptive_module.bundle_model_related_files
evaluate_selected_model_prediction_performance = _adaptive_module.evaluate_selected_model_prediction_performance
get_baseline_cache_path = _adaptive_module._get_baseline_cache_path
prepare_training_run_dir = _adaptive_module._prepare_training_run_dir


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


def _collect_model_entries(search_root):
    entries = []
    candidate_dirs = [search_root]

    try:
        for name in os.listdir(search_root):
            sub_path = os.path.join(search_root, name)
            if os.path.isdir(sub_path) and 'adaptive_net' in name.lower():
                candidate_dirs.append(sub_path)
    except Exception as exc:
        print(f"Warning: 列举子目录失败: {exc}")

    for current_root in candidate_dirs:
        try:
            files = os.listdir(current_root)
        except Exception as exc:
            print(f"Warning: 读取目录失败: {current_root} | {exc}")
            continue

        for file_name in files:
            full_path = os.path.join(current_root, file_name)
            if not os.path.isfile(full_path) or not file_name.endswith('.pth'):
                continue

            has_expected_prefix = (
                file_name.startswith('adaptive_net_')
                or file_name.startswith('best_adaptive_net_')
                or file_name.startswith('RL_adaptive_net_')
            )
            if not has_expected_prefix:
                continue

            entries.append({
                'model_name': file_name,
                'model_path': full_path,
                'display_path': os.path.relpath(full_path, search_root),
                'mtime': os.path.getmtime(full_path),
            })

    entries.sort(key=lambda item: item['mtime'], reverse=True)
    return entries


def _resolve_config_for_model_path(model_path, model_dir):
    model_name = os.path.basename(model_path)
    model_parent_dir = os.path.dirname(model_path)
    ts = model_name
    for prefix in ('adaptive_net_', 'best_adaptive_net_', 'RL_adaptive_net_'):
        if ts.startswith(prefix):
            ts = ts.replace(prefix, '', 1)
            break
    ts = ts.replace('.pth', '')
    ts_base = re.sub(r'_dim\d+$', '', ts)

    if model_name.startswith('best_adaptive_net_'):
        candidates = [
            os.path.join(model_parent_dir, f"best_config_{ts}.pt"),
            os.path.join(model_parent_dir, f"best_config_{ts_base}.pt"),
            os.path.join(model_parent_dir, f"config_{ts}.pt"),
            os.path.join(model_parent_dir, f"config_{ts_base}.pt"),
            os.path.join(model_dir, f"best_config_{ts}.pt"),
            os.path.join(model_dir, f"best_config_{ts_base}.pt"),
            os.path.join(model_dir, f"config_{ts}.pt"),
            os.path.join(model_dir, f"config_{ts_base}.pt")
        ]
    elif model_name.startswith('RL_adaptive_net_'):
        candidates = [
            os.path.join(model_parent_dir, f"RL_config_{ts}.pt"),
            os.path.join(model_parent_dir, f"RL_config_{ts_base}.pt"),
            os.path.join(model_parent_dir, f"config_{ts}.pt"),
            os.path.join(model_parent_dir, f"config_{ts_base}.pt"),
            os.path.join(model_parent_dir, f"best_config_{ts}.pt"),
            os.path.join(model_parent_dir, f"best_config_{ts_base}.pt"),
            os.path.join(model_dir, f"RL_config_{ts}.pt"),
            os.path.join(model_dir, f"RL_config_{ts_base}.pt"),
            os.path.join(model_dir, f"config_{ts}.pt"),
            os.path.join(model_dir, f"config_{ts_base}.pt"),
            os.path.join(model_dir, f"best_config_{ts}.pt"),
            os.path.join(model_dir, f"best_config_{ts_base}.pt")
        ]
    else:
        candidates = [
            os.path.join(model_parent_dir, f"config_{ts}.pt"),
            os.path.join(model_parent_dir, f"config_{ts_base}.pt"),
            os.path.join(model_parent_dir, f"best_config_{ts}.pt"),
            os.path.join(model_parent_dir, f"best_config_{ts_base}.pt"),
            os.path.join(model_dir, f"config_{ts}.pt"),
            os.path.join(model_dir, f"config_{ts_base}.pt"),
            os.path.join(model_dir, f"best_config_{ts}.pt"),
            os.path.join(model_dir, f"best_config_{ts_base}.pt")
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _print_model_entries(model_entries):
    print("\n发现历史训练模型:")
    for idx, entry in enumerate(model_entries, start=1):
        print(f"[{idx}] {entry['display_path']}")


def _select_model_entry(model_entries, allow_default_latest=False):
    while True:
        try:
            prompt = "请输入模型编号: "
            if allow_default_latest:
                prompt = "请输入模型编号（回车默认最新模型[1]）: "
            raw = input(prompt).strip()
            if allow_default_latest and raw == '':
                selected_index = 0
            else:
                selected_index = int(raw) - 1
            if selected_index < 0 or selected_index >= len(model_entries):
                print("编号超出范围，请重新输入。")
                continue
            return model_entries[selected_index]
        except ValueError:
            print("输入无效，请输入数字编号。")


def _configure_training_options():
    print("\n=== 训练参数选择 ===")
    print(f"当前模式: {TRAIN_CONFIG['MODE'].value if hasattr(TRAIN_CONFIG['MODE'], 'value') else TRAIN_CONFIG['MODE']}")
    print("Android cpp 工程固定输入: 10帧 x 5维 [e_y, e_psi, roll, pitch, omega]，输出为 Mode A 原始值。")
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

    # Android 端输入结构固定为 5 维时序特征；模式允许 A/D，其余模式仅保留本地实验用途。
    TRAIN_CONFIG['USE_PITCH_FEATURE'] = True
    TRAIN_CONFIG['USE_DIFF_FEATURE'] = False
    if TRAIN_CONFIG['MODE'] in (ControlMode.A, ControlMode.D):
        print(
            f"已按 Android cpp 工程固定训练配置: MODE={TRAIN_CONFIG['MODE'].value}, "
            "USE_PITCH_FEATURE=True, USE_DIFF_FEATURE=False"
        )
    else:
        print(
            f"已按本地训练固定输入配置: MODE={TRAIN_CONFIG['MODE'].value}, "
            "USE_PITCH_FEATURE=True, USE_DIFF_FEATURE=False"
        )
        print("提示: Android 部署端当前仅支持 A / D 模式。")

    mode_show = TRAIN_CONFIG['MODE'].value if hasattr(TRAIN_CONFIG['MODE'], 'value') else TRAIN_CONFIG['MODE']
    print(
        f"训练配置确认 -> MODE: {mode_show}, "
        f"USE_PITCH_FEATURE: {TRAIN_CONFIG.get('USE_PITCH_FEATURE', True)}, "
        f"USE_DIFF_FEATURE: {TRAIN_CONFIG.get('USE_DIFF_FEATURE', False)}"
    )

    baseline_cache_path = get_baseline_cache_path()
    rebuild_baseline = input(
        "数据集是否已变更(若变更则需要重做基础LQR预计算)? [Y/N]\n"
        f"Y: 重新计算并保存到本次训练目录\n"
        f"N: 沿用历史预计算文件；若未找到则自动计算并保存到本次训练目录\n"
        f"本次训练目录缓存文件: {baseline_cache_path}\n> "
    ).strip().lower()
    TRAIN_CONFIG['REBUILD_BASELINE_CACHE'] = rebuild_baseline in ('y', 'yes')
    if TRAIN_CONFIG['REBUILD_BASELINE_CACHE']:
        print("本次训练将重建基础LQR查表缓存。")
    else:
        print("本次训练将优先复用已有基础LQR查表缓存；若缓存不存在则自动创建。")


def _backup_model_and_config(model_path, config_path=None):
    backup_dir = os.path.join(os.path.dirname(model_path), "preserved_models")
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_bak = os.path.join(backup_dir, f"preserved_{ts}_{os.path.basename(model_path)}")
    shutil.copy2(model_path, model_bak)
    print(f"已备份历史模型: {model_bak}")
    if config_path and os.path.exists(config_path):
        config_bak = os.path.join(backup_dir, f"preserved_{ts}_{os.path.basename(config_path)}")
        shutil.copy2(config_path, config_bak)
        print(f"已备份历史配置: {config_bak}")
    elif config_path:
        print(f"Warning: 配置文件不存在，跳过备份: {config_path}")


def _compute_expected_time_dim_from_config(config):
    return _adaptive_module._get_expected_time_dim_from_config(config)


def _can_resume_with_current_config(config_path):
    saved_data = torch.load(config_path, map_location='cpu', weights_only=False)
    saved_stats = saved_data.get('stats', {})
    saved_config = saved_data.get('config', {})

    saved_time_dim = None
    if 'time_mean' in saved_stats:
        saved_time_dim = int(saved_stats['time_mean'].shape[0])
    if saved_time_dim is None:
        saved_time_dim = _compute_expected_time_dim_from_config(saved_config)

    current_time_dim = _compute_expected_time_dim_from_config(TRAIN_CONFIG)
    compatible = saved_time_dim == current_time_dim
    return compatible, saved_time_dim, current_time_dim, saved_config


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
            bundle_model_related_files(
                model_path=rl_model,
                config_path=rl_config,
                model_dir=TRAIN_CONFIG['MODEL_DIR'],
                ensure_diagnostics=True
            )


def _maybe_rl_without_test(model_path, config_path, default_path='sine'):
    ans = input("是否执行RL微调? [y/N]: ").strip().lower()
    if ans not in ('y', 'yes'):
        print("已跳过RL微调。")
        return None, None

    rl_path = _select_rl_trajectory(default_path=default_path)
    rl_model, rl_config = fine_tune_with_rl(model_path, config_path, path_type=rl_path)
    if rl_model and rl_config:
        run_comparison(rl_model, rl_config, path_type=rl_path)
        bundle_model_related_files(
            model_path=rl_model,
            config_path=rl_config,
            model_dir=TRAIN_CONFIG['MODEL_DIR'],
            ensure_diagnostics=True
        )
    return rl_model, rl_config


def _run_training_flow(resume_model_path=None, configure_options=True):
    prepare_training_run_dir(TRAIN_CONFIG)
    if configure_options:
        _configure_training_options()
    print("\n=== 开始训练流程 ===")
    model_path, config_path = train_network(resume_model_path=resume_model_path)

    if not (model_path and config_path):
        print("训练未成功完成。")
        return None, None, None, None

    print("训练收尾步骤已完成，下面进入训练后交互选项。")
    run_test = input("\n训练完成，是否立即进行对比测试? [y/N]: ").strip().lower()
    if run_test not in ('y', 'yes'):
        print("已跳过测试。")
        _maybe_rl_without_test(model_path, config_path, default_path='sine')
        return model_path, config_path, None, None

    print("\n训练完成，正在进入对比测试...")
    selected_path = _select_test_trajectory()
    metrics = run_comparison(model_path, config_path, path_type=selected_path, num_runs=1)
    _maybe_rl_after_test(metrics, model_path, config_path, selected_path)
    return model_path, config_path, selected_path, metrics


def _require_config_path(config_path):
    if os.path.exists(config_path):
        return True
    print(f"Error: 对应的配置文件丢失: {config_path}")
    return False

# ==================== 主程序 ====================
def main():
    model_dir = TRAIN_CONFIG['MODEL_DIR']

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_entries = _collect_model_entries(model_dir)

    if not model_entries:
        print("未检测到训练好的模型，开始全新训练...")
        _run_training_flow(resume_model_path=None, configure_options=True)
        return

    _print_model_entries(model_entries)
    while True:
        resp = input(
            "\n请选择: [N]ew Train (重新训练), [S]elect Test (选择并测试), [E]valuate Diagnose (完整拟合效果评估), [R]etrain (继续训练), [F]inetune RL (强化学习微调), [D]eploy (生成部署文件): "
        ).strip().lower()
        if resp in ['n', 's', 'e', 'r', 'f', 'd']:
            choice = resp
            break

    if choice == 'n':
        _run_training_flow(resume_model_path=None, configure_options=True)
        return

    if choice == 's':
        selected_entry = _select_model_entry(model_entries)
        selected_model = selected_entry['model_path']
        selected_config = _resolve_config_for_model_path(selected_model, model_dir)
        if not _require_config_path(selected_config):
            return
        selected_path = _select_test_trajectory()
        metrics = run_comparison(selected_model, selected_config, path_type=selected_path, num_runs=1)
        _maybe_rl_after_test(metrics, selected_model, selected_config, selected_path)
        return

    if choice == 'e':
        model_files = [entry['display_path'] for entry in model_entries]
        evaluate_selected_model_prediction_performance(
            model_dir=model_dir,
            model_files=model_files,
            resolve_config_for_model_name=lambda model_name: _resolve_config_for_model_path(
                os.path.join(model_dir, model_name),
                model_dir,
            ),
        )
        return

    if choice == 'f':
        selected_entry = _select_model_entry(model_entries, allow_default_latest=True)
        selected_model = selected_entry['model_path']
        selected_config = _resolve_config_for_model_path(selected_model, model_dir)
        if not _require_config_path(selected_config):
            return
        selected_path = _select_rl_trajectory(default_path='sine')
        rl_model, rl_config = fine_tune_with_rl(selected_model, selected_config, path_type=selected_path)
        if rl_model and rl_config:
            run_comparison(rl_model, rl_config, path_type=selected_path)
            bundle_model_related_files(
                model_path=rl_model,
                config_path=rl_config,
                model_dir=model_dir,
                ensure_diagnostics=True
            )
        return

    if choice == 'd':
        selected_entry = _select_model_entry(model_entries, allow_default_latest=True)
        selected_model = selected_entry['model_path']
        selected_config = _resolve_config_for_model_path(selected_model, model_dir)
        if not _require_config_path(selected_config):
            return
        generate_deployment_files(selected_model, selected_config)
        return

    selected_entry = _select_model_entry(model_entries, allow_default_latest=True)
    selected_model = selected_entry['model_path']
    selected_config = _resolve_config_for_model_path(selected_model, model_dir)
    if not _require_config_path(selected_config):
        return
    _backup_model_and_config(selected_model, selected_config)

    _configure_training_options()
    can_resume, saved_time_dim, current_time_dim, saved_config = _can_resume_with_current_config(selected_config)
    resume_model_path = selected_model
    if not can_resume:
        saved_pitch = saved_config.get('USE_PITCH_FEATURE', 'unknown')
        saved_diff = saved_config.get('USE_DIFF_FEATURE', 'unknown')
        print(
            "Warning: 当前交互配置与所选模型结构不兼容，将按新训练处理，不加载旧权重。"
        )
        print(
            f"  所选模型输入维度: dim{saved_time_dim} "
            f"(USE_PITCH_FEATURE={saved_pitch}, USE_DIFF_FEATURE={saved_diff})"
        )
        print(
            f"  当前交互配置维度: dim{current_time_dim} "
            f"(USE_PITCH_FEATURE={TRAIN_CONFIG.get('USE_PITCH_FEATURE', True)}, "
            f"USE_DIFF_FEATURE={TRAIN_CONFIG.get('USE_DIFF_FEATURE', False)})"
        )
        resume_model_path = None
    else:
        print(f"当前交互配置与所选模型兼容，将继续训练: dim{current_time_dim}")

    model_path, config_path, _selected_path, _metrics = _run_training_flow(
        resume_model_path=resume_model_path,
        configure_options=False
    )
    if model_path and config_path:
        bundle_model_related_files(
            model_path=model_path,
            config_path=config_path,
            model_dir=model_dir,
            ensure_diagnostics=False
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序终止")
