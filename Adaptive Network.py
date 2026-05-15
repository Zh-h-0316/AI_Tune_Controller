import os
import shlex
import shutil
from functools import lru_cache
from enum import Enum
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import Config_Para as cfg
import RL_finetune as rl_tuner
from data_structures import State, Control
from vehicle_model import VehicleModel, PathTracker
from path_generator import PathGenerator
from LQR_ratio import LQR_car
from by_source_analyzer import analyze_by_source_file


# ==================== 配置与枚举 ====================
class ControlMode(Enum):
    A = 'A'  # Gain scheduling (Alpha * K + Beta)
    B = 'B'  # Gain scaling (Alpha * K)
    C = 'C'  # Gain bias (K + Beta)
    D = 'D'  # Direct control modification (u_LQR + u_net)


TRAIN_CONFIG = {
    'SEQ_LEN': 10,
    'BATCH_SIZE': 64,
    'EPOCHS': 100,
    'LR': 1e-3,
    'HIDDEN_SIZE': 64,
    'MODE': ControlMode.A,
    'DATA_ROOT': r"D:\Huace_Work\AI_Control\Excellent_Data\Process_Data",
    'MODEL_DIR': r"D:\Huace_Work\AI_Control\AI_Tune\models_pth_onnx_rknn",
    'DATA_FILTER_ENABLED': False,          # 是否启用数据过滤
    'INCLUDE_DIR_KEYWORDS': [],            # 仅保留路径中包含任一关键词的csv_data目录（如['flatland']）
    'EXCLUDE_DIR_KEYWORDS': [],            # 排除路径中包含任一关键词的csv_data目录
    'INCLUDE_FILE_KEYWORDS': [],           # 仅保留文件名包含任一关键词的csv
    'EXCLUDE_FILE_KEYWORDS': [],           # 排除文件名包含任一关键词的csv

    # 新增训练配置
    'LSTM_LAYERS': 2,                # LSTM层数
    'LSTM_DROPOUT': 0.3,             # LSTM层间dropout（仅当layers>1时有效）
    'USE_ATTENTION': True,           # 是否使用时间注意力
    'MLP_HIDDEN': [128, 64],         # 融合MLP各层维度
    'MLP_DROPOUT': 0.2,              # MLP dropout率
    'WEIGHT_DECAY': 1e-4,            # 权重衰减
    'GRAD_CLIP_NORM': 1.0,           # 梯度裁剪阈值
    'HUBER_BETA': 0.03,              # SmoothL1(Huber) beta
    'SMOOTH_LOSS_WEIGHT': 0.2,       # 控制量平滑损失权重（当前未启用）
    'STEER_LIMIT_LOSS_WEIGHT': 0.03,  # 转角越界惩罚权重：抑制过量补偿导致的执行器顶边运行
    'RATE_LOSS_WEIGHT': 0.03,        # 控制变化率惩罚权重：抑制实车左右摆动
    'COMP_LOSS_WEIGHT': 0.6,         # 残差对齐损失权重：保留补偿学习，但不过度追求大幅补偿
    'COMP_FOCUS_GAMMA': 1.0,         # 残差越大权重越高，重点学习大补偿样本
    'UNDER_COMP_LOSS_WEIGHT': 0.08,  # 补偿不足惩罚：保留轻度推动，避免整体偏保守
    'OVER_COMP_LOSS_WEIGHT': 0.25,   # 补偿过量惩罚：抑制实车中因过补偿导致的左右摆动
    'MODE_A_ALPHA_RANGE': (0.3, 1.8),  # 模式A中alpha输出范围
    'MODE_A_BETA_SCALE': 0.25,       # 模式A中beta输出范围[-scale, scale]
    'MODE_D_DELTA_SCALE': 0.30,      # 模式D补偿量缩放上限（原先0.1偏小）
    'MODE_D_USE_TANH_BOUND': True,   # 模式D输出使用tanh限幅，确保DELTA_SCALE真正作为补偿上限
    'DATA_AUG_NOISE': 0.01,          # 数据增强噪声标准差（0表示关闭）
    'VAL_SPLIT': 0.2,                 # 验证集比例
    'USE_SPEED_STRATIFIED_SPLIT': True,  # 是否按速度区间分层划分（与Group Split结合）
    'SPEED_BINS_MPS': [1.0 / 3.6, 4.0 / 3.6, 7.0 / 3.6, 10.0 / 3.6, 15.0 / 3.6, 20.0 / 3.6, 30.0 / 3.6],
    # 速度分层边界(m/s)，按真实作业车速划分：
    # [0,1km/h), [1,4km/h), [4,7km/h), [7,10km/h), [10,15km/h), [15,20km/h), [20,30km/h), [30km/h,+inf)
    'SPEED_BIN_MIN_RATIO_FILTER_ENABLED': False,  # 少样本速度段不再丢弃，仅保留统计输出
    'SPEED_BIN_MIN_RATIO': 0.0,      # 仅保留兼容字段，不再用于训练与评估过滤
    'LOW_SPEED_THRESHOLD_MPS': 1.0 / 3.6,  # 低速重点区间上界，用于损失加权与结果解读
    'LOW_SPEED_LOSS_BOOST': 1.0,     # 取消低速额外权重放大，避免低速样本主导补偿学习
    'SPEED_LOSS_MIN_WEIGHT': 0.9,    # 分速度段损失权重下限，避免牺牲其他速度段
    'SPEED_LOSS_MAX_WEIGHT': 2.4,    # 分速度段损失权重上限，避免低速权重过大挤压其他速度段
    'SPEED_FEATURE_GAIN': 1.8,       # 速度特征增强系数，提升模型对速度变化的敏感性
    'EARLY_STOP_PATIENCE': 20,        # 早停耐心值（连续20轮无提升则提前结束）
    'USE_PITCH_FEATURE': True,        # 是否使用俯仰角及其历史序列作为时序输入
    'USE_DIFF_FEATURE': False,        # 是否在时间序列中加入差分特征（会改变输入维度）

    # GPU / training performance
    'DEVICE': 'auto',                 # 'auto'|'cpu'|'cuda'
    'NUM_WORKERS': 4,                 # DataLoader worker count
    'PIN_MEMORY': True,               # DataLoader pin_memory (useful for CUDA)
    'USE_AMP': False,                 # 使用混合精度训练 (自动混合精度)
    'BASELINE_SPEED_MAX_KMH': 25.0,   # 基础LQR查表最高速度 (km/h)
    'BASELINE_SPEED_STEP_KMH': 0.1,   # 基础LQR查表速度步长 (km/h)
    'BASELINE_CACHE_FILENAME': 'android_lqr_baseline_lookup.pt',  # 基础LQR查表缓存文件名
    'REBUILD_BASELINE_CACHE': False,  # 训练前是否强制重建基础LQR查表缓存
    'WHEELBASE_KEY_DECIMALS': 6,      # wheelbase离散键保留小数位
    'RKNN_PYTHON_EXE': '',            # 可选：本地 Windows RKNN Toolkit2 Python 解释器路径
    'RKNN_WSL_ENABLED': True,         # 是否优先尝试通过 WSL 中的 RKNN Toolkit2 生成 .rknn
    'RKNN_WSL_DISTRO': '',            # 可选：指定 WSL 发行版名称，为空时自动选择默认可用发行版
    'RKNN_WSL_PYTHON': 'python3',     # WSL 中用于执行 convert_to_rknn.py 的 Python 命令或绝对路径

    # RL fine-tune (PPO)
    'RL_ENABLED_DEFAULT': False,      # 拟合+测试后，默认是否继续做RL微调
    'RL_EPISODES': 100,               # RL训练episode数
    'RL_STEPS_PER_EPISODE': 300,      # 每个episode最大步数
    'RL_PPO_EPOCHS': 5,               # PPO每轮更新轮数
    'RL_MINIBATCH_SIZE': 128,         # PPO小批量大小
    'RL_LR': 3e-4,                    # PPO学习率
    'RL_GAMMA': 0.99,                 # 折扣因子
    'RL_GAE_LAMBDA': 0.95,            # GAE系数
    'RL_CLIP_EPS': 0.2,               # PPO裁剪阈值
    'RL_ENTROPY_COEF': 0.01,          # 熵奖励系数
    'RL_VALUE_COEF': 0.5,             # Value损失权重
    'RL_MAX_GRAD_NORM': 1.0,          # 梯度裁剪
    'RL_INIT_LOG_STD': -1.2,          # 连续动作初始log std
    'RL_TRIGGER_LATERAL_MAE': 0.10,   # 建议启用RL微调的横向误差阈值
    'RL_EARLY_STOP_PATIENCE': 15,     # RL微调自动早停耐心值（连续15轮无提升则提前结束）
    'RL_EARLY_STOP_MIN_DELTA': 1e-3,  # RL微调最小提升阈值

    # 训练前诊断导出
    'DIAG_EXPORT_ENABLED': True,       # 是否在训练前导出诊断CSV
    'DIAG_BATCH_SIZE': 4096,           # 诊断推理批大小
    'DIAG_MAX_SAMPLES': 0,             # 单split最大样本数，0表示全部
}

ANDROID_TIME_FEATURE_NAMES = ('e_y', 'e_psi', 'roll', 'pitch', 'omega')
ANDROID_EXPECTED_TIME_DIM = len(ANDROID_TIME_FEATURE_NAMES)

if not os.path.exists(TRAIN_CONFIG['MODEL_DIR']):
    os.makedirs(TRAIN_CONFIG['MODEL_DIR'])


def _get_model_output_root(config=None):
    config = TRAIN_CONFIG if config is None else config
    return os.path.abspath(config.get('MODEL_ROOT_DIR', config['MODEL_DIR']))


def _prepare_training_run_dir(config=None, run_label=None):
    config = TRAIN_CONFIG if config is None else config
    model_root_dir = _get_model_output_root(config)
    os.makedirs(model_root_dir, exist_ok=True)

    current_run_dir = config.get('CURRENT_RUN_DIR')
    if current_run_dir and os.path.isdir(current_run_dir):
        config['MODEL_ROOT_DIR'] = model_root_dir
        config['MODEL_DIR'] = current_run_dir
        return current_run_dir

    if not run_label:
        run_label = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_run_dir = os.path.join(model_root_dir, f"adaptive_net_run_{run_label}")
    run_dir = base_run_dir
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = f"{base_run_dir}_{suffix}"
        suffix += 1

    os.makedirs(run_dir, exist_ok=True)
    config['MODEL_ROOT_DIR'] = model_root_dir
    config['CURRENT_RUN_DIR'] = run_dir
    config['MODEL_DIR'] = run_dir
    print(f"本次训练输出目录: {run_dir}")
    return run_dir


def _get_expected_time_dim_from_config(config):
    base_time_dim = 5 if config.get('USE_PITCH_FEATURE', True) else 4
    return base_time_dim + (2 if config.get('USE_DIFF_FEATURE', False) else 0)


def _get_speed_bin_edges(config=None):
    config = TRAIN_CONFIG if config is None else config
    default_edges = [1.5 / 3.6, 4.0 / 3.6, 6.0 / 3.6, 10.0 / 3.6]
    raw_edges = config.get('SPEED_BINS_MPS', default_edges)
    return sorted(float(edge) for edge in raw_edges)


def _get_speed_bin_labels(speed_bins=None):
    speed_bins = _get_speed_bin_edges() if speed_bins is None else list(speed_bins)
    labels = []
    lower_kmh = 0.0
    for edge_mps in speed_bins:
        upper_kmh = edge_mps * 3.6
        labels.append(f"[{lower_kmh:.1f},{upper_kmh:.1f}) km/h")
        lower_kmh = upper_kmh
    labels.append(f"[{lower_kmh:.1f},+inf) km/h")
    return labels


def _speed_bin_id(speed_mps, speed_bins=None):
    speed_bins = _get_speed_bin_edges() if speed_bins is None else speed_bins
    return int(np.digitize(float(speed_mps), speed_bins, right=False))


def _format_speed_bin_count_report(counts, speed_bins=None):
    labels = _get_speed_bin_labels(speed_bins)
    return ', '.join(f"{label}:{int(count)}" for label, count in zip(labels, counts))


def _format_speed_bin_value_report(values, speed_bins=None, value_fmt='.3f'):
    labels = _get_speed_bin_labels(speed_bins)
    chunks = []
    for label, value in zip(labels, values):
        chunks.append(f"{label}:{format(float(value), value_fmt)}")
    return ', '.join(chunks)


def _upgrade_legacy_scalar_net_state_dict(state_dict, model_state_dict):
    upgraded = dict(state_dict)
    legacy_w1 = upgraded.get('scalar_net.0.weight')
    legacy_b1 = upgraded.get('scalar_net.0.bias')
    legacy_w2 = upgraded.get('scalar_net.2.weight')
    legacy_b2 = upgraded.get('scalar_net.2.bias')

    target_w1 = model_state_dict.get('scalar_net.0.weight')
    target_b1 = model_state_dict.get('scalar_net.0.bias')
    target_w2 = model_state_dict.get('scalar_net.2.weight')
    target_b2 = model_state_dict.get('scalar_net.2.bias')

    if any(item is None for item in (legacy_w1, legacy_b1, legacy_w2, legacy_b2, target_w1, target_b1, target_w2, target_b2)):
        return upgraded, False

    is_legacy_layout = (
        tuple(legacy_w1.shape) == (32, 2)
        and tuple(legacy_b1.shape) == (32,)
        and tuple(legacy_w2.shape) == (32, 32)
        and tuple(legacy_b2.shape) == (32,)
        and tuple(target_w1.shape) == (64, 7)
        and tuple(target_b1.shape) == (64,)
        and tuple(target_w2.shape) == (32, 64)
        and tuple(target_b2.shape) == (32,)
    )
    if not is_legacy_layout:
        return upgraded, False

    new_w1 = target_w1.clone()
    new_b1 = target_b1.clone()
    new_w2 = target_w2.clone()
    new_b2 = target_b2.clone()

    new_w1.zero_()
    new_b1.zero_()
    new_w2.zero_()
    new_b2.copy_(legacy_b2)

    new_w1[:32, :2].copy_(legacy_w1)
    new_b1[:32].copy_(legacy_b1)
    new_w2[:, :32].copy_(legacy_w2)

    upgraded['scalar_net.0.weight'] = new_w1
    upgraded['scalar_net.0.bias'] = new_b1
    upgraded['scalar_net.2.weight'] = new_w2
    upgraded['scalar_net.2.bias'] = new_b2
    return upgraded, True


def _load_adaptive_network_state(model, state_or_path, *, map_location=None, strict=True, load_label='checkpoint'):
    if isinstance(state_or_path, str):
        checkpoint = torch.load(state_or_path, map_location=map_location, weights_only=True)
    else:
        checkpoint = state_or_path

    model_state = model.state_dict()
    adapted_checkpoint, upgraded_legacy = _upgrade_legacy_scalar_net_state_dict(checkpoint, model_state)

    try:
        model.load_state_dict(adapted_checkpoint, strict=strict)
        if upgraded_legacy:
            print(f"检测到旧版 scalar_net 权重布局，已自动升级后加载 {load_label}。")
        return {'legacy_upgraded': upgraded_legacy, 'loaded': True}
    except RuntimeError:
        raise


def _build_speed_bin_loss_weights(speed_values, config=None, *, device=None):
    config = TRAIN_CONFIG if config is None else config
    speed_bins = _get_speed_bin_edges(config)
    speed_tensor = torch.as_tensor(speed_values, dtype=torch.float32, device=device)

    if speed_tensor.numel() == 0:
        return torch.ones(len(speed_bins) + 1, dtype=torch.float32, device=device)

    bins_tensor = torch.as_tensor(speed_bins, dtype=torch.float32, device=speed_tensor.device)
    bin_ids = torch.bucketize(speed_tensor.reshape(-1), bins_tensor, right=False)
    counts = torch.bincount(bin_ids, minlength=len(speed_bins) + 1).to(dtype=torch.float32)

    weights = torch.ones_like(counts)
    valid_mask = counts > 0
    if torch.any(valid_mask):
        mean_count = counts[valid_mask].mean()
        weights[valid_mask] = torch.sqrt(mean_count / counts[valid_mask])

    weights[0] = weights[0] * max(1.0, float(config.get('LOW_SPEED_LOSS_BOOST', 1.0)))
    weights = torch.clamp(
        weights,
        min=float(config.get('SPEED_LOSS_MIN_WEIGHT', 0.9)),
        max=float(config.get('SPEED_LOSS_MAX_WEIGHT', 2.4)),
    )
    return weights


def _filter_samples_by_speed_bin_ratio(samples, config=None):
    config = TRAIN_CONFIG if config is None else config
    if not samples:
        return samples

    speed_bins = _get_speed_bin_edges(config)
    num_bins = len(speed_bins) + 1
    counts = [0] * num_bins
    for sample in samples:
        speed_val = float(sample['scalar'][0])
        counts[_speed_bin_id(speed_val, speed_bins)] += 1

    if not bool(config.get('SPEED_BIN_MIN_RATIO_FILTER_ENABLED', False)):
        print(
            "Speed-bin ratio filter disabled; all samples are retained. "
            f"bin_counts[{_format_speed_bin_count_report(counts, speed_bins)}]"
        )
        return samples

    min_ratio = float(config.get('SPEED_BIN_MIN_RATIO', 0.0))
    if min_ratio <= 0.0:
        print(
            "Speed-bin ratio filter threshold <= 0; all samples are retained. "
            f"bin_counts[{_format_speed_bin_count_report(counts, speed_bins)}]"
        )
        return samples

    sample_bin_ids = []
    for sample in samples:
        speed_val = float(sample['scalar'][0])
        bin_id = _speed_bin_id(speed_val, speed_bins)
        sample_bin_ids.append(bin_id)
    print(
        "Speed-bin ratio filtering has been retired; keeping all samples. "
        f"requested_threshold={min_ratio:.2%}, bin_ratios[{_format_speed_bin_value_report([count / max(1, len(samples)) for count in counts], speed_bins)}]"
    )
    return samples


def _compute_compensation_balance_losses(model_comp, residual, speed_sample_weight, config=None):
    config = TRAIN_CONFIG if config is None else config
    residual_abs = torch.abs(residual).detach()
    residual_norm = residual_abs.mean().detach() + 1e-6
    focus_weight = 1.0 + float(config.get('COMP_FOCUS_GAMMA', 1.0)) * (residual_abs / residual_norm)
    sample_weight = speed_sample_weight * focus_weight
    abs_model_comp = torch.abs(model_comp)
    abs_residual = torch.abs(residual)
    loss_comp = (sample_weight * torch.abs(model_comp - residual)).mean()
    loss_under_comp = (speed_sample_weight * torch.relu(abs_residual - abs_model_comp)).mean()
    loss_over_comp = (speed_sample_weight * torch.relu(abs_model_comp - abs_residual)).mean()
    return loss_comp, loss_under_comp, loss_over_comp


def _extract_source_group(source_id, data_root=None):
    source_text = str(source_id) if source_id is not None else 'unknown_source'
    if not source_text:
        return 'unknown_group'

    norm_source = os.path.normpath(source_text)
    if data_root:
        try:
            norm_root = os.path.normpath(data_root)
            common = os.path.commonpath([os.path.abspath(norm_source), os.path.abspath(norm_root)])
            if common == os.path.abspath(norm_root):
                rel_path = os.path.relpath(norm_source, norm_root)
                first_part = rel_path.split(os.sep)[0]
                if first_part and first_part not in ('.', '..'):
                    return first_part
        except Exception:
            pass

    path_parts = [part for part in norm_source.split(os.sep) if part and part not in ('.', '..')]
    if len(path_parts) >= 2:
        return path_parts[-2]
    if path_parts:
        return path_parts[-1]
    return 'unknown_group'


def _attach_diagnostic_slice_columns(diag_df, speed_bins=None):
    speed_bins = _get_speed_bin_edges() if speed_bins is None else list(speed_bins)
    speed_edges = [0.0] + speed_bins + [np.inf]
    speed_labels = _get_speed_bin_labels(speed_bins)
    diag_df = diag_df.copy()

    diag_df['speed_bin'] = pd.cut(
        diag_df['speed_mps'],
        bins=speed_edges,
        right=False,
        include_lowest=True,
        labels=speed_labels,
    )

    diag_df['abs_roll_deg'] = np.rad2deg(np.abs(diag_df['roll_last_rad']))
    diag_df['abs_pitch_deg'] = np.rad2deg(np.abs(diag_df['pitch_last_rad']))
    diag_df['source_group'] = diag_df['source_id'].map(
        lambda value: _extract_source_group(value, TRAIN_CONFIG.get('DATA_ROOT'))
    )

    diag_df['roll_slice'] = pd.cut(
        diag_df['abs_roll_deg'],
        bins=[0.0, 5.0, 15.0, 25.0, np.inf],
        right=False,
        include_lowest=True,
        labels=['[0,5) deg', '[5,15) deg', '[15,25) deg', '[25,+inf) deg'],
    )
    diag_df['pitch_slice'] = pd.cut(
        diag_df['abs_pitch_deg'],
        bins=[0.0, 5.0, 15.0, 25.0, np.inf],
        right=False,
        include_lowest=True,
        labels=['[0,5) deg', '[5,15) deg', '[15,25) deg', '[25,+inf) deg'],
    )
    return diag_df


def _sanitize_source_file_label(source_id):
    source_name = os.path.splitext(os.path.basename(str(source_id)))[0]
    safe_chars = []
    for ch in source_name:
        if ch.isalnum() or ch in ('-', '_'):
            safe_chars.append(ch)
        else:
            safe_chars.append('_')
    safe_name = ''.join(safe_chars).strip('_')
    return safe_name or 'source'


def _export_random_source_fit_reports(diag_df, bundle_dir, sample_count=5):
    if diag_df is None or len(diag_df) == 0 or 'source_id' not in diag_df.columns:
        return []

    unique_sources = [src for src in pd.Series(diag_df['source_id']).dropna().astype(str).unique().tolist() if src]
    if not unique_sources:
        return []

    sample_count = max(1, int(sample_count))
    rng = np.random.default_rng()
    selected_count = min(sample_count, len(unique_sources))
    selected_sources = rng.choice(unique_sources, size=selected_count, replace=False).tolist()

    exported = []
    eps = 1e-12
    for idx, source_id in enumerate(selected_sources, start=1):
        source_df = diag_df[diag_df['source_id'].astype(str) == str(source_id)].copy()
        if len(source_df) == 0:
            continue

        source_df = source_df.sort_values(['split', 'sample_index']).reset_index(drop=True)
        source_df['speed_kmh'] = source_df['speed_mps'] * 3.6
        source_df['actual_comp_rad'] = source_df['residual_rad']
        source_df['model_minus_actual_comp_rad'] = source_df['model_comp_rad'] - source_df['actual_comp_rad']
        source_df['actual_minus_model_comp_rad'] = source_df['actual_comp_rad'] - source_df['model_comp_rad']
        source_df['abs_comp_gap_rad'] = np.abs(source_df['actual_minus_model_comp_rad'])
        source_df['model_to_actual_ratio_signed'] = source_df['model_comp_rad'] / np.where(
            np.abs(source_df['actual_comp_rad']) > eps,
            source_df['actual_comp_rad'],
            np.nan,
        )
        source_df['model_to_actual_ratio_abs'] = np.abs(source_df['model_comp_rad']) / np.maximum(
            np.abs(source_df['actual_comp_rad']),
            eps,
        )

        export_columns = [
            'split',
            'sample_index',
            'source_id',
            'speed_mps',
            'speed_kmh',
            'e_y',
            'e_psi',
            'delta_opt_rad',
            'delta_lqr_base_rad',
            'actual_comp_rad',
            'model_comp_rad',
            'actual_minus_model_comp_rad',
            'model_minus_actual_comp_rad',
            'abs_comp_gap_rad',
            'model_to_actual_ratio_signed',
            'model_to_actual_ratio_abs',
            'delta_pred_from_model_rad',
            'abs_err_lqr_to_opt_rad',
            'abs_err_pred_to_opt_rad',
            'improvement_abs_rad',
            'speed_bin',
        ]
        export_df = source_df[export_columns]

        file_stem = _sanitize_source_file_label(source_id)
        export_path = os.path.join(bundle_dir, f"fit_detail_random_source_{idx:02d}_{file_stem}.csv")
        export_df.to_csv(export_path, index=False, encoding='utf-8-sig', float_format='%.9g')

        ratio_series = source_df['model_to_actual_ratio_abs'].replace([np.inf, -np.inf], np.nan).dropna()
        exported.append({
            'index': idx,
            'source_id': str(source_id),
            'sample_count': int(len(export_df)),
            'csv_path': export_path,
            'mae_pred_to_opt': float(source_df['abs_err_pred'].mean()) if 'abs_err_pred' in source_df.columns else float(source_df['abs_err_pred_to_opt_rad'].mean()),
            'mae_lqr_to_opt': float(source_df['abs_err_lqr'].mean()) if 'abs_err_lqr' in source_df.columns else float(source_df['abs_err_lqr_to_opt_rad'].mean()),
            'mean_actual_comp': float(source_df['actual_comp_rad'].mean()),
            'mean_model_comp': float(source_df['model_comp_rad'].mean()),
            'mean_abs_comp_gap': float(source_df['abs_comp_gap_rad'].mean()),
            'mean_ratio_abs': float(ratio_series.mean()) if len(ratio_series) > 0 else np.nan,
        })

    return exported


def validate_android_training_config(config, *, actual_time_dim=None):
    mode_value = config.get('MODE', ControlMode.A)
    mode_name = mode_value.value if isinstance(mode_value, ControlMode) else str(mode_value)
    issues = []

    if mode_name not in (ControlMode.A.value, ControlMode.D.value):
        issues.append(f"MODE 必须为 A 或 D，当前为 {mode_name}")
    if not config.get('USE_PITCH_FEATURE', True):
        issues.append("USE_PITCH_FEATURE 必须为 True")
    if config.get('USE_DIFF_FEATURE', False):
        issues.append("USE_DIFF_FEATURE 必须为 False")

    expected_time_dim = _get_expected_time_dim_from_config(config)
    if expected_time_dim != ANDROID_EXPECTED_TIME_DIM:
        issues.append(
            f"时序输入维度必须为 {ANDROID_EXPECTED_TIME_DIM} ({', '.join(ANDROID_TIME_FEATURE_NAMES)})，"
            f"当前配置推导为 {expected_time_dim}"
        )

    if actual_time_dim is not None and int(actual_time_dim) != ANDROID_EXPECTED_TIME_DIM:
        issues.append(
            f"当前模型/统计量的 time_dim={actual_time_dim}，但 Android 工程固定需要 {ANDROID_EXPECTED_TIME_DIM}"
        )

    return issues


@lru_cache(maxsize=4096)
def _solve_android_lqr_gains_cached(speed_mps_key: float, wheelbase_m_key: float) -> tuple[float, float]:
    lqr = LQR_car(dt=cfg.VEHICLE_DT)
    lqr.update_car_state(0.0, 0.0, 0.0, speed_mps_key)
    lqr.Update_A_B_matrix(wheelbase_m_key)
    lqr.Update_Q_R_matrix(
        q11=cfg.ANDROID_LQR_Q1,
        q22=cfg.ANDROID_LQR_Q2,
        r00=cfg.ANDROID_LQR_R,
        r11=cfg.ANDROID_LQR_R11,
    )
    K, _ = lqr.Solve()
    return float(K[0, 1]), float(K[0, 0])


def _compute_android_lqr_gains(speed_tensor, wheelbase_tensor, *, device, dtype):
    speed_values = speed_tensor.detach().cpu().numpy().reshape(-1)
    wheelbase_values = wheelbase_tensor.detach().cpu().numpy().reshape(-1)

    k_e_values = []
    k_th_values = []
    for speed_mps, wheelbase_m in zip(speed_values, wheelbase_values):
        k_e_base, k_th_base = _solve_android_lqr_gains_cached(
            round(float(speed_mps), 6),
            round(float(wheelbase_m), 6),
        )
        k_e_values.append(k_e_base)
        k_th_values.append(k_th_base)

    return (
        torch.tensor(k_e_values, dtype=dtype, device=device),
        torch.tensor(k_th_values, dtype=dtype, device=device),
    )


def _get_baseline_cache_path(config=None):
    config = TRAIN_CONFIG if config is None else config
    return os.path.join(
        config['MODEL_DIR'],
        config.get('BASELINE_CACHE_FILENAME', 'android_lqr_baseline_lookup.pt')
    )


def _collect_baseline_cache_candidates(config=None):
    config = TRAIN_CONFIG if config is None else config
    cache_filename = config.get('BASELINE_CACHE_FILENAME', 'android_lqr_baseline_lookup.pt')
    search_root = _get_model_output_root(config)
    candidate_dirs = [search_root]

    try:
        for name in os.listdir(search_root):
            sub_path = os.path.join(search_root, name)
            if os.path.isdir(sub_path) and 'adaptive_net' in name.lower():
                candidate_dirs.append(sub_path)
    except Exception as exc:
        print(f"Warning: 列举基础LQR缓存候选目录失败: {exc}")

    candidate_paths = []
    seen = set()
    for current_root in candidate_dirs:
        candidate_path = os.path.join(current_root, cache_filename)
        if not os.path.isfile(candidate_path):
            continue
        norm_path = os.path.normcase(os.path.abspath(candidate_path))
        if norm_path in seen:
            continue
        seen.add(norm_path)
        candidate_paths.append(candidate_path)

    candidate_paths.sort(key=os.path.getmtime, reverse=True)
    return candidate_paths


def _normalize_wheelbase_key(wheelbase_value, decimals=None):
    if decimals is None:
        decimals = int(TRAIN_CONFIG.get('WHEELBASE_KEY_DECIMALS', 6))
    return round(float(wheelbase_value), int(decimals))


def _collect_unique_wheelbases_from_samples(samples):
    wheelbase_map = {}
    for sample in samples:
        raw_wheelbase = float(sample['wheelbase'])
        wheelbase_key = _normalize_wheelbase_key(raw_wheelbase)
        if wheelbase_key not in wheelbase_map:
            wheelbase_map[wheelbase_key] = raw_wheelbase
    return [wheelbase_map[key] for key in sorted(wheelbase_map.keys())]


def _build_android_lqr_lookup_cache(samples, cache_path=None):
    cache_path = _get_baseline_cache_path() if cache_path is None else cache_path
    speed_step_kmh = float(TRAIN_CONFIG.get('BASELINE_SPEED_STEP_KMH', 0.1))
    speed_max_kmh = float(TRAIN_CONFIG.get('BASELINE_SPEED_MAX_KMH', 20.0))
    if speed_step_kmh <= 0:
        raise ValueError("BASELINE_SPEED_STEP_KMH 必须大于 0")
    if speed_max_kmh <= 0:
        raise ValueError("BASELINE_SPEED_MAX_KMH 必须大于 0")

    wheelbase_values = _collect_unique_wheelbases_from_samples(samples)
    if not wheelbase_values:
        raise ValueError("未从样本中提取到任何 wheelbase，无法构建基础LQR查表缓存")

    speed_grid_kmh = np.round(
        np.arange(0.0, speed_max_kmh + speed_step_kmh * 0.5, speed_step_kmh, dtype=np.float64),
        6,
    )
    speed_grid_mps = speed_grid_kmh / 3.6

    k_e_table = np.zeros((len(wheelbase_values), len(speed_grid_mps)), dtype=np.float32)
    k_th_table = np.zeros((len(wheelbase_values), len(speed_grid_mps)), dtype=np.float32)

    for wheelbase_idx, wheelbase_value in enumerate(wheelbase_values):
        wheelbase_key = _normalize_wheelbase_key(wheelbase_value)
        for speed_idx, speed_mps in enumerate(speed_grid_mps):
            k_e_base, k_th_base = _solve_android_lqr_gains_cached(
                round(float(speed_mps), 6),
                wheelbase_key,
            )
            k_e_table[wheelbase_idx, speed_idx] = np.float32(k_e_base)
            k_th_table[wheelbase_idx, speed_idx] = np.float32(k_th_base)

    payload = {
        'metadata': {
            'created_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'data_root': str(TRAIN_CONFIG.get('DATA_ROOT', '')),
            'sample_count': int(len(samples)),
            'wheelbase_count': int(len(wheelbase_values)),
            'speed_step_kmh': speed_step_kmh,
            'speed_max_kmh': speed_max_kmh,
            'wheelbase_key_decimals': int(TRAIN_CONFIG.get('WHEELBASE_KEY_DECIMALS', 6)),
        },
        'speed_grid_kmh': torch.tensor(speed_grid_kmh, dtype=torch.float32),
        'speed_grid_mps': torch.tensor(speed_grid_mps, dtype=torch.float32),
        'wheelbase_values': torch.tensor(wheelbase_values, dtype=torch.float32),
        'k_e_table': torch.tensor(k_e_table, dtype=torch.float32),
        'k_th_table': torch.tensor(k_th_table, dtype=torch.float32),
    }
    torch.save(payload, cache_path)
    print(
        f"基础LQR查表缓存已重建: {cache_path} | "
        f"wheelbase={len(wheelbase_values)} 个, speed_grid={len(speed_grid_kmh)} 个"
    )
    return payload


def _load_android_lqr_lookup_cache(cache_path=None):
    if cache_path is None:
        candidate_paths = _collect_baseline_cache_candidates()
        if not candidate_paths:
            return None, None
        cache_path = candidate_paths[0]
    elif not os.path.exists(cache_path):
        return None, None

    payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    required_keys = {'metadata', 'speed_grid_kmh', 'wheelbase_values', 'k_e_table', 'k_th_table'}
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"基础LQR查表缓存文件缺少字段: {missing_keys}")
    return payload, cache_path


def _resolve_android_lqr_lookup_cache(samples, rebuild=False):
    current_cache_path = _get_baseline_cache_path()
    payload = None
    if not rebuild:
        payload, loaded_cache_path = _load_android_lqr_lookup_cache()
        if payload is not None:
            cached_wheelbases = payload['wheelbase_values'].tolist()
            cached_keys = {_normalize_wheelbase_key(value) for value in cached_wheelbases}
            required_keys = {
                _normalize_wheelbase_key(sample['wheelbase'])
                for sample in samples
            }
            missing_keys = sorted(required_keys - cached_keys)
            if missing_keys:
                print(
                    "已找到历史基础LQR查表缓存，但未完全覆盖当前数据集 wheelbase。"
                    f"缺失示例: {missing_keys[:5]}。"
                    "将自动重建并保存新的预计算表。"
                )
                return _build_android_lqr_lookup_cache(samples, current_cache_path)

            print(f"已复用基础LQR查表缓存: {loaded_cache_path}")
            return payload

        print(
            "未找到历史基础LQR预计算表文件。"
            f"已自动开启重计算，并将保存到: {current_cache_path}"
        )
        return _build_android_lqr_lookup_cache(samples, current_cache_path)

    return _build_android_lqr_lookup_cache(samples, current_cache_path)


def _lookup_precomputed_android_lqr_gains(speed_tensor, wheelbase_tensor, lookup_payload):
    speed_step_kmh = float(lookup_payload['metadata'].get('speed_step_kmh', 0.1))
    speed_grid_kmh = lookup_payload['speed_grid_kmh'].detach().cpu().numpy().reshape(-1)
    wheelbase_values = lookup_payload['wheelbase_values'].detach().cpu().numpy().reshape(-1)
    k_e_table = lookup_payload['k_e_table'].detach().cpu().numpy()
    k_th_table = lookup_payload['k_th_table'].detach().cpu().numpy()

    speed_kmh = speed_tensor.detach().cpu().numpy().reshape(-1) * 3.6
    speed_indices = np.rint(speed_kmh / speed_step_kmh).astype(np.int64)
    speed_indices = np.clip(speed_indices, 0, len(speed_grid_kmh) - 1)

    wheelbase_map = {
        _normalize_wheelbase_key(value): idx
        for idx, value in enumerate(wheelbase_values.tolist())
    }
    wheelbase_array = wheelbase_tensor.detach().cpu().numpy().reshape(-1)
    wheelbase_indices = np.empty(len(wheelbase_array), dtype=np.int64)
    missing_keys = []
    for idx, wheelbase_value in enumerate(wheelbase_array):
        wheelbase_key = _normalize_wheelbase_key(wheelbase_value)
        table_idx = wheelbase_map.get(wheelbase_key)
        if table_idx is None:
            missing_keys.append(wheelbase_key)
            continue
        wheelbase_indices[idx] = table_idx

    if missing_keys:
        raise RuntimeError(
            "基础LQR查表缓存未覆盖当前 batch 的 wheelbase。"
            f"缺失示例: {sorted(set(missing_keys))[:5]}。"
            "请重建基础LQR查表缓存。"
        )

    k_e_values = k_e_table[wheelbase_indices, speed_indices].astype(np.float32, copy=False)
    k_th_values = k_th_table[wheelbase_indices, speed_indices].astype(np.float32, copy=False)
    return torch.from_numpy(k_e_values), torch.from_numpy(k_th_values)


# ==================== 数据处理 ====================
def load_samples(root_dir, seq_len=10):
    """
    加载CSV数据并生成滑动窗口样本。
    """
    samples = []
    if not os.path.exists(root_dir):
        print(f"Warning: Data directory {root_dir} not found.")
        return []

    print(f"Loading data from {root_dir} (recursive csv_data search)...")

    def contains_any_keyword(text, keywords):
        if not keywords:
            return False
        text_lower = text.lower()
        return any(str(k).lower() in text_lower for k in keywords)

    csv_dirs = []
    for current_root, dir_names, _ in os.walk(root_dir):
        if 'csv_data' in dir_names:
            csv_dir_path = os.path.join(current_root, 'csv_data')

            if TRAIN_CONFIG.get('DATA_FILTER_ENABLED', False):
                include_dir_keywords = TRAIN_CONFIG.get('INCLUDE_DIR_KEYWORDS', [])
                exclude_dir_keywords = TRAIN_CONFIG.get('EXCLUDE_DIR_KEYWORDS', [])

                if include_dir_keywords and not contains_any_keyword(csv_dir_path, include_dir_keywords):
                    continue
                if exclude_dir_keywords and contains_any_keyword(csv_dir_path, exclude_dir_keywords):
                    continue

            csv_dirs.append(csv_dir_path)

    if not csv_dirs:
        print(f"Warning: No csv_data directories found under {root_dir}.")
        return []

    print(f"Found {len(csv_dirs)} csv_data directories.")

    if TRAIN_CONFIG.get('DATA_FILTER_ENABLED', False):
        print(
            "Data filter enabled | "
            f"include_dir={TRAIN_CONFIG.get('INCLUDE_DIR_KEYWORDS', [])}, "
            f"exclude_dir={TRAIN_CONFIG.get('EXCLUDE_DIR_KEYWORDS', [])}, "
            f"include_file={TRAIN_CONFIG.get('INCLUDE_FILE_KEYWORDS', [])}, "
            f"exclude_file={TRAIN_CONFIG.get('EXCLUDE_FILE_KEYWORDS', [])}"
        )

    for csv_dir in csv_dirs:
        for fname in os.listdir(csv_dir):
            if not fname.endswith('.csv'):
                continue

            if TRAIN_CONFIG.get('DATA_FILTER_ENABLED', False):
                include_file_keywords = TRAIN_CONFIG.get('INCLUDE_FILE_KEYWORDS', [])
                exclude_file_keywords = TRAIN_CONFIG.get('EXCLUDE_FILE_KEYWORDS', [])

                if include_file_keywords and not contains_any_keyword(fname, include_file_keywords):
                    continue
                if exclude_file_keywords and contains_any_keyword(fname, exclude_file_keywords):
                    continue

            fpath = os.path.join(csv_dir, fname)
            try:
                df = pd.read_csv(fpath)

                def pick_col(candidates):
                    for col_name in candidates:
                        if col_name in df.columns:
                            return col_name
                    return None

                col_v = pick_col(['D', 'Obs_V_kmh', 'Est_V_kmh'])
                col_roll = pick_col(['G', 'Roll_Deg'])
                col_pitch = pick_col(['H', 'Pitch_Deg'])
                col_omega = pick_col(['I', 'Omega_DegS'])
                col_heading = pick_col(['R', 'xHeading_Deg', 'Obs_Heading_Deg', 'Est_Heading_Deg'])
                col_lateral = pick_col(['S', 'xTrack_m'])
                col_delta_opt = pick_col(['F', 'Demand_WheelAngle_Deg', 'Motor_TargetAngle_Deg', 'Obs_WheelAngle_Deg'])
                col_wheelbase = pick_col(['J', 'WheelBase_m'])

                required_cols = {
                    'v': col_v,
                    'roll': col_roll,
                    'omega': col_omega,
                    'heading': col_heading,
                    'lateral_error': col_lateral,
                    'delta_opt': col_delta_opt,
                    'wheelbase': col_wheelbase
                }
                if TRAIN_CONFIG.get('USE_PITCH_FEATURE', True):
                    required_cols['pitch'] = col_pitch

                missing = [name for name, col_name in required_cols.items() if col_name is None]
                if missing:
                    print(f"Skip {fname}: missing required columns {missing}")
                    continue

                v_mps = df[col_v].to_numpy(dtype=float) / 3.6
                roll_rad = np.deg2rad(df[col_roll].to_numpy(dtype=float))
                pitch_rad = None
                if TRAIN_CONFIG.get('USE_PITCH_FEATURE', True):
                    pitch_rad = np.deg2rad(df[col_pitch].to_numpy(dtype=float))
                omega_rad_s = np.deg2rad(df[col_omega].to_numpy(dtype=float))
                heading_rad = np.deg2rad(df[col_heading].to_numpy(dtype=float))
                lateral_error = df[col_lateral].to_numpy(dtype=float)
                delta_opt_rad = np.deg2rad(df[col_delta_opt].to_numpy(dtype=float))
                wheelbase = df[col_wheelbase].to_numpy(dtype=float)

                n = len(v_mps)
                if n <= seq_len:
                    continue

                for t in range(seq_len, n):
                    # 基础时间序列特征
                    base_features = [
                        lateral_error[t - seq_len:t],
                        heading_rad[t - seq_len:t],
                        roll_rad[t - seq_len:t]
                    ]
                    if TRAIN_CONFIG.get('USE_PITCH_FEATURE', True) and pitch_rad is not None:
                        base_features.append(pitch_rad[t - seq_len:t])
                    base_features.append(omega_rad_s[t - seq_len:t])
                    base_seq = np.column_stack(base_features).astype(np.float32)

                    # 可选：加入差分特征（一阶差分）
                    if TRAIN_CONFIG['USE_DIFF_FEATURE']:
                        diff_e = np.diff(lateral_error[t - seq_len:t], prepend=lateral_error[t - seq_len])
                        diff_theta = np.diff(heading_rad[t - seq_len:t], prepend=heading_rad[t - seq_len])
                        diff_seq = np.column_stack([diff_e, diff_theta]).astype(np.float32)
                        time_seq = np.concatenate([base_seq, diff_seq], axis=1)  # 变为7维
                    else:
                        time_seq = base_seq

                    scalar_feat = np.array([v_mps[t], wheelbase[t]], dtype=np.float32)

                    samples.append({
                        'time_series': time_seq,
                        'scalar': scalar_feat,
                        'e': np.float32(lateral_error[t]),
                        'theta': np.float32(heading_rad[t]),
                        'delta_opt': np.float32(delta_opt_rad[t]),
                        'wheelbase': np.float32(wheelbase[t]),
                        'source_id': fpath
                    })
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue

    print(f"Total samples loaded before speed-bin ratio filter: {len(samples)}")
    samples = _filter_samples_by_speed_bin_ratio(samples, TRAIN_CONFIG)
    print(f"Total samples retained for training/evaluation: {len(samples)}")
    return samples


class ControlDataset(Dataset):
    def __init__(self, samples, stats=None, augment=False, baseline_lookup=None):
        self.samples = samples
        self.augment = augment  # 是否启用数据增强

        if len(samples) > 0:
            all_time = np.array([s['time_series'] for s in samples])
            all_scalar = np.array([s['scalar'] for s in samples])
        else:
            # 默认形状，需根据USE_DIFF_FEATURE调整
            base_time_dim = 5 if TRAIN_CONFIG.get('USE_PITCH_FEATURE', True) else 4
            time_dim = base_time_dim + (2 if TRAIN_CONFIG['USE_DIFF_FEATURE'] else 0)
            all_time = np.zeros((1, TRAIN_CONFIG['SEQ_LEN'], time_dim))
            all_scalar = np.zeros((1, 2))

        if stats is None:
            self.stats = {
                'scalar_mean': torch.FloatTensor(all_scalar.mean(axis=0)),
                'scalar_std': torch.FloatTensor(all_scalar.std(axis=0) + 1e-6),
                'time_mean': torch.FloatTensor(all_time.mean(axis=(0, 1))),
                'time_std': torch.FloatTensor(all_time.std(axis=(0, 1)) + 1e-6)
            }
        else:
            self.stats = stats

        self.time_mean = self.stats['time_mean'].view(1, 1, -1)
        self.time_std = self.stats['time_std'].view(1, 1, -1)
        self.scalar_mean = self.stats['scalar_mean'].view(1, 2)
        self.scalar_std = self.stats['scalar_std'].view(1, 2)

        self.raw_time = torch.FloatTensor(all_time)
        self.raw_scalar = torch.FloatTensor(all_scalar)
        self.e = torch.FloatTensor([s['e'] for s in samples])
        self.theta = torch.FloatTensor([s['theta'] for s in samples])
        self.delta_opt = torch.FloatTensor([s['delta_opt'] for s in samples])
        self.wheelbase = torch.FloatTensor([s['wheelbase'] for s in samples])
        if len(samples) > 0:
            if baseline_lookup is None:
                self.k_e_base, self.k_th_base = _compute_android_lqr_gains(
                    self.raw_scalar[:, 0],
                    self.wheelbase,
                    device=torch.device('cpu'),
                    dtype=self.e.dtype,
                )
            else:
                self.k_e_base, self.k_th_base = _lookup_precomputed_android_lqr_gains(
                    self.raw_scalar[:, 0],
                    self.wheelbase,
                    baseline_lookup,
                )
        else:
            self.k_e_base = torch.zeros(0, dtype=torch.float32)
            self.k_th_base = torch.zeros(0, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 基础归一化
        norm_time = (self.raw_time[idx] - self.time_mean.squeeze(0)) / self.time_std.squeeze(0)
        norm_scalar = (self.raw_scalar[idx] - self.scalar_mean.squeeze(0)) / self.scalar_std.squeeze(0)

        # 数据增强（仅在训练时）
        if self.augment and TRAIN_CONFIG['DATA_AUG_NOISE'] > 0:
            noise_std = TRAIN_CONFIG['DATA_AUG_NOISE']
            # 对时间序列加高斯噪声
            norm_time = norm_time + torch.randn_like(norm_time) * noise_std
            # 对标量车速加噪声（第二维是占位符0，不加噪）
            scalar_noise = torch.randn(1) * noise_std
            norm_scalar[0] = norm_scalar[0] + scalar_noise

        return (
            norm_time,
            norm_scalar,
            self.raw_scalar[idx],  # 原始标量（可能用于调试）
            self.e[idx],
            self.theta[idx],
            self.delta_opt[idx],
            self.k_e_base[idx],
            self.k_th_base[idx]
        )


# ==================== 网络模型（改进版） ====================
class AdaptiveNetwork(nn.Module):
    def __init__(self, mode: ControlMode, time_dim=5, scalar_dim=2, hidden_size=64,
                 lstm_layers=2, lstm_dropout=0.3, use_attention=True,
                 mlp_hidden=[128, 64], mlp_dropout=0.2,
                 mode_a_alpha_range=(0.5, 1.5), mode_a_beta_scale=0.1,
                 mode_d_delta_scale=0.1, mode_d_use_tanh_bound=False, speed_feature_gain=1.8):
        super().__init__()
        self.mode = mode
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.mode_a_alpha_min = float(mode_a_alpha_range[0])
        self.mode_a_alpha_max = float(mode_a_alpha_range[1])
        if self.mode_a_alpha_max <= self.mode_a_alpha_min:
            self.mode_a_alpha_min, self.mode_a_alpha_max = 0.5, 1.5
        self.mode_a_beta_scale = float(mode_a_beta_scale)
        self.mode_d_delta_scale = float(mode_d_delta_scale)
        self.mode_d_use_tanh_bound = bool(mode_d_use_tanh_bound)
        self.speed_feature_gain = float(speed_feature_gain)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=time_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )

        # 注意力机制（可选）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

        # 标量特征网络：在保持外部输入维度不变的前提下，增强速度相关表达能力
        scalar_feature_dim = 7
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 融合MLP骨干（带BatchNorm和Dropout）
        fusion_dim = hidden_size + 32
        layers = []
        in_dim = fusion_dim
        for out_dim in mlp_hidden:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))
            in_dim = out_dim
        self.mlp_backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # 输出层（4维，对应alpha_e, beta_e, alpha_th, beta_th 或 直接修正量）
        self.output_head = nn.Linear(in_dim, 4)

        # 残差连接：在隐藏特征空间进行，保证维度一致
        self.residual_proj = nn.Identity() if fusion_dim == in_dim else nn.Linear(fusion_dim, in_dim)

    def _build_speed_aware_scalar_features(self, scalar):
        speed = scalar[:, 0:1]
        wheelbase = scalar[:, 1:2]
        speed_sq = speed * speed
        speed_wheelbase = speed * wheelbase
        speed_tanh = torch.tanh(speed * self.speed_feature_gain)
        low_speed_focus = torch.exp(-torch.abs(speed) * self.speed_feature_gain)
        speed_ratio = speed / (1.0 + torch.abs(speed))
        return torch.cat(
            [speed, wheelbase, speed_sq, speed_wheelbase, speed_tanh, low_speed_focus, speed_ratio],
            dim=1,
        )

    def forward(self, time_seq, scalar):
        # LSTM提取时序特征
        lstm_out, (h_n, c_n) = self.lstm(time_seq)  # lstm_out: (batch, seq_len, hidden)

        if self.use_attention:
            # 计算注意力权重
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            # 加权求和
            seq_feat = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        else:
            # 取最后时刻输出
            seq_feat = lstm_out[:, -1, :]  # (batch, hidden)

        # 标量特征
        scalar_features = self._build_speed_aware_scalar_features(scalar)
        scal_feat = self.scalar_net(scalar_features)  # (batch, 32)

        # 融合
        combined = torch.cat([seq_feat, scal_feat], dim=1)  # (batch, hidden+32)

        # MLP骨干前向
        hidden_feat = self.mlp_backbone(combined)

        # 残差连接（在隐藏特征空间）
        if self.residual_proj is not None:
            residual = self.residual_proj(combined)
            hidden_feat = hidden_feat + residual

        # 输出层
        mlp_out = self.output_head(hidden_feat)

        # 根据模式返回不同输出
        if self.mode == ControlMode.A:
            # 更精细的输出约束
            alpha_e_raw, beta_e_raw, alpha_th_raw, beta_th_raw = mlp_out[:, 0], mlp_out[:, 1], mlp_out[:, 2], mlp_out[:, 3]
            alpha_span = self.mode_a_alpha_max - self.mode_a_alpha_min
            alpha_e = self.mode_a_alpha_min + alpha_span * torch.sigmoid(alpha_e_raw)
            beta_e = self.mode_a_beta_scale * torch.tanh(beta_e_raw)
            alpha_th = self.mode_a_alpha_min + alpha_span * torch.sigmoid(alpha_th_raw)
            beta_th = self.mode_a_beta_scale * torch.tanh(beta_th_raw)
            return alpha_e, beta_e, alpha_th, beta_th

        elif self.mode == ControlMode.D:
            if self.mode_d_use_tanh_bound:
                delta_add = torch.tanh(mlp_out[:, 0]) * self.mode_d_delta_scale
            else:
                delta_add = mlp_out[:, 0] * self.mode_d_delta_scale
            return delta_add

        else:
            # 其他模式暂不实现详细改进，返回原始输出
            return mlp_out


def export_pretrain_diagnostics(model, train_dataset, val_dataset, model_dir, mode):
    """
    训练前诊断：导出delta_opt与LQR基线残差、模型补偿等逐样本数据，并打印统计摘要。
    """
    if not TRAIN_CONFIG.get('DIAG_EXPORT_ENABLED', True):
        return None

    model_device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    diag_batch = int(TRAIN_CONFIG.get('DIAG_BATCH_SIZE', 4096))
    max_samples_cfg = int(TRAIN_CONFIG.get('DIAG_MAX_SAMPLES', 0))

    def _collect_for_dataset(ds, split_name):
        n_total = len(ds)
        n_use = n_total if max_samples_cfg <= 0 else min(n_total, max_samples_cfg)

        rows = {
            'split': [],
            'sample_index': [],
            'source_id': [],
            'speed_mps': [],
            'wheelbase_m': [],
            'e_y': [],
            'e_psi': [],
            'roll_last_rad': [],
            'pitch_last_rad': [],
            'omega_last_rad_s': [],
            'delta_opt_rad': [],
            'delta_lqr_base_rad': [],
            'residual_rad': [],
            'model_comp_rad': [],
            'residual_minus_comp_rad': [],
            'delta_pred_from_model_rad': [],
            'abs_residual_rad': [],
            'abs_model_comp_rad': [],
            'abs_residual_minus_comp_rad': [],
            'abs_err_lqr_to_opt_rad': [],
            'abs_err_pred_to_opt_rad': [],
            'improvement_abs_rad': [],
            'comp_to_residual_ratio': [],
        }

        with torch.no_grad():
            for start in range(0, n_use, diag_batch):
                end = min(start + diag_batch, n_use)

                raw_time = ds.raw_time[start:end].to(model_device)
                raw_scalar_dev = ds.raw_scalar[start:end].to(model_device)
                norm_time = (raw_time - ds.time_mean.to(model_device)) / ds.time_std.to(model_device)
                norm_scalar = (raw_scalar_dev - ds.scalar_mean.to(model_device)) / ds.scalar_std.to(model_device)
                raw_e = ds.e[start:end].to(model_device)
                raw_theta = ds.theta[start:end].to(model_device)
                raw_delta_opt = ds.delta_opt[start:end].to(model_device)
                raw_scalar = ds.raw_scalar[start:end]
                k_e_base = ds.k_e_base[start:end].to(model_device)
                k_th_base = ds.k_th_base[start:end].to(model_device)

                delta_lqr_base = -(k_e_base * raw_e + k_th_base * raw_theta)

                if mode == ControlMode.A:
                    alpha_e, beta_e, alpha_th, beta_th = model(norm_time, norm_scalar)
                    k_e_final = alpha_e * k_e_base + beta_e
                    k_th_final = alpha_th * k_th_base + beta_th
                    delta_pred = -k_e_final * raw_e - k_th_final * raw_theta
                    model_comp = delta_pred - delta_lqr_base
                elif mode == ControlMode.D:
                    delta_add = model(norm_time, norm_scalar).view(-1)
                    delta_pred = delta_lqr_base + delta_add
                    model_comp = delta_add
                else:
                    delta_pred = delta_lqr_base
                    model_comp = torch.zeros_like(delta_pred)

                residual = raw_delta_opt - delta_lqr_base
                residual_minus_comp = residual - model_comp

                speed_np = raw_scalar[:, 0].cpu().numpy()
                wheelbase_np = raw_scalar[:, 1].cpu().numpy()
                e_np = raw_e.cpu().numpy()
                th_np = raw_theta.cpu().numpy()
                delta_opt_np = raw_delta_opt.cpu().numpy()
                delta_lqr_np = delta_lqr_base.cpu().numpy()
                residual_np = residual.cpu().numpy()
                comp_np = model_comp.cpu().numpy()
                diff_np = residual_minus_comp.cpu().numpy()
                delta_pred_np = delta_pred.cpu().numpy()
                roll_np = raw_time[:, -1, 2].cpu().numpy() if raw_time.shape[-1] > 2 else np.zeros(end - start)
                pitch_np = raw_time[:, -1, 3].cpu().numpy() if raw_time.shape[-1] > 3 else np.zeros(end - start)
                omega_np = raw_time[:, -1, 4].cpu().numpy() if raw_time.shape[-1] > 4 else np.zeros(end - start)
                abs_err_lqr_np = np.abs(delta_opt_np - delta_lqr_np)
                abs_err_pred_np = np.abs(delta_opt_np - delta_pred_np)
                improvement_abs_np = abs_err_lqr_np - abs_err_pred_np
                comp_ratio_np = np.abs(comp_np) / np.maximum(np.abs(residual_np), 1e-12)

                rows['split'].extend([split_name] * (end - start))
                rows['sample_index'].extend(list(range(start, end)))
                rows['source_id'].extend([s.get('source_id', 'unknown_source') for s in ds.samples[start:end]])
                rows['speed_mps'].extend(speed_np.tolist())
                rows['wheelbase_m'].extend(wheelbase_np.tolist())
                rows['e_y'].extend(e_np.tolist())
                rows['e_psi'].extend(th_np.tolist())
                rows['roll_last_rad'].extend(roll_np.tolist())
                rows['pitch_last_rad'].extend(pitch_np.tolist())
                rows['omega_last_rad_s'].extend(omega_np.tolist())
                rows['delta_opt_rad'].extend(delta_opt_np.tolist())
                rows['delta_lqr_base_rad'].extend(delta_lqr_np.tolist())
                rows['residual_rad'].extend(residual_np.tolist())
                rows['model_comp_rad'].extend(comp_np.tolist())
                rows['residual_minus_comp_rad'].extend(diff_np.tolist())
                rows['delta_pred_from_model_rad'].extend(delta_pred_np.tolist())
                rows['abs_residual_rad'].extend(np.abs(residual_np).tolist())
                rows['abs_model_comp_rad'].extend(np.abs(comp_np).tolist())
                rows['abs_residual_minus_comp_rad'].extend(np.abs(diff_np).tolist())
                rows['abs_err_lqr_to_opt_rad'].extend(abs_err_lqr_np.tolist())
                rows['abs_err_pred_to_opt_rad'].extend(abs_err_pred_np.tolist())
                rows['improvement_abs_rad'].extend(improvement_abs_np.tolist())
                rows['comp_to_residual_ratio'].extend(comp_ratio_np.tolist())

        return pd.DataFrame(rows)

    train_df = _collect_for_dataset(train_dataset, 'train')
    val_df = _collect_for_dataset(val_dataset, 'val')
    diag_df = pd.concat([train_df, val_df], ignore_index=True)
    speed_bins = _get_speed_bin_edges(TRAIN_CONFIG)
    diag_df = _attach_diagnostic_slice_columns(diag_df, speed_bins)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(model_dir, f"pretrain_diagnostics_{timestamp}.csv")
    diag_df.to_csv(csv_path, index=False, encoding='utf-8-sig', float_format='%.9g')

    # 二号汇总文件：按数据源(source_id)与split聚合，便于快速定位问题数据
    group_cols = ['split', 'source_id']
    summary_df = (
        diag_df
        .groupby(group_cols, as_index=False)
        .agg(
            sample_count=('sample_index', 'count'),
            speed_mps_mean=('speed_mps', 'mean'),
            speed_mps_std=('speed_mps', 'std'),
            abs_residual_mean=('abs_residual_rad', 'mean'),
            abs_residual_p90=('abs_residual_rad', lambda x: np.percentile(x, 90)),
            abs_residual_p99=('abs_residual_rad', lambda x: np.percentile(x, 99)),
            abs_model_comp_mean=('abs_model_comp_rad', 'mean'),
            abs_model_comp_p90=('abs_model_comp_rad', lambda x: np.percentile(x, 90)),
            abs_model_comp_p99=('abs_model_comp_rad', lambda x: np.percentile(x, 99)),
            abs_gap_mean=('abs_residual_minus_comp_rad', 'mean'),
            abs_gap_p90=('abs_residual_minus_comp_rad', lambda x: np.percentile(x, 90)),
            abs_gap_p99=('abs_residual_minus_comp_rad', lambda x: np.percentile(x, 99)),
            residual_mean=('residual_rad', 'mean'),
            comp_mean=('model_comp_rad', 'mean')
        )
    )
    summary_df['speed_mps_std'] = summary_df['speed_mps_std'].fillna(0.0)

    summary_csv_path = os.path.join(model_dir, f"pretrain_diagnostics_by_source_{timestamp}.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig', float_format='%.9g')

    slice_frames = []
    slice_columns = [
        ('speed_bin', 'speed_bin'),
        ('roll_slice', 'roll_slice'),
        ('pitch_slice', 'pitch_slice'),
        ('source_id', 'source_id'),
        ('source_group', 'source_group'),
    ]
    for slice_type, slice_col in slice_columns:
        sub_df = diag_df.dropna(subset=[slice_col])
        if len(sub_df) == 0:
            continue
        grouped = (
            sub_df
            .groupby(['split', slice_col], as_index=False, observed=True)
            .agg(
                sample_count=('sample_index', 'count'),
                speed_mps_mean=('speed_mps', 'mean'),
                abs_residual_mean=('abs_residual_rad', 'mean'),
                abs_model_comp_mean=('abs_model_comp_rad', 'mean'),
                abs_gap_mean=('abs_residual_minus_comp_rad', 'mean'),
                mae_lqr_to_opt=('abs_err_lqr_to_opt_rad', 'mean'),
                mae_pred_to_opt=('abs_err_pred_to_opt_rad', 'mean'),
                improvement_abs=('improvement_abs_rad', 'mean'),
                comp_to_residual_ratio_mean=('comp_to_residual_ratio', 'mean'),
            )
        )
        grouped.insert(1, 'slice_type', slice_type)
        grouped = grouped.rename(columns={slice_col: 'slice_value'})
        slice_frames.append(grouped)

    slice_summary_df = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    slice_summary_csv_path = os.path.join(model_dir, f"pretrain_diagnostics_by_slice_{timestamp}.csv")
    slice_summary_df.to_csv(slice_summary_csv_path, index=False, encoding='utf-8-sig', float_format='%.9g')

    def _summarize(df, name):
        abs_res = df['abs_residual_rad'].to_numpy()
        abs_comp = df['abs_model_comp_rad'].to_numpy()
        abs_gap = df['abs_residual_minus_comp_rad'].to_numpy()
        res = df['residual_rad'].to_numpy()
        comp = df['model_comp_rad'].to_numpy()

        corr = np.nan
        if len(res) > 1 and np.std(res) > 1e-12 and np.std(comp) > 1e-12:
            corr = float(np.corrcoef(res, comp)[0, 1])

        print(
            f"[DIAG:{name}] samples={len(df)} | "
            f"|res| mean={abs_res.mean():.6f}, p90={np.percentile(abs_res, 90):.6f}, p99={np.percentile(abs_res, 99):.6f} | "
            f"|comp| mean={abs_comp.mean():.6f}, p90={np.percentile(abs_comp, 90):.6f}, p99={np.percentile(abs_comp, 99):.6f} | "
            f"|res-comp| mean={abs_gap.mean():.6f}, p90={np.percentile(abs_gap, 90):.6f}, p99={np.percentile(abs_gap, 99):.6f} | "
            f"corr(res,comp)={corr:.4f}"
        )

    print("\n========== 训练前诊断摘要 ==========")
    _summarize(diag_df, 'all')
    _summarize(diag_df[diag_df['split'] == 'train'], 'train')
    _summarize(diag_df[diag_df['split'] == 'val'], 'val')
    top_k = 10
    top_worst = summary_df.sort_values('abs_gap_mean', ascending=False).head(top_k)
    print(f"\n[DIAG] abs(residual-comp) 均值最高的 Top-{top_k} 数据源:")
    for _, r in top_worst.iterrows():
        print(
            f"  split={r['split']}, source={r['source_id']}, n={int(r['sample_count'])}, "
            f"|gap|mean={r['abs_gap_mean']:.6f}, |res|mean={r['abs_residual_mean']:.6f}, "
            f"|comp|mean={r['abs_model_comp_mean']:.6f}"
        )
    print(f"诊断CSV已保存至: {csv_path}")
    print(f"按source汇总CSV已保存至: {summary_csv_path}")
    if len(slice_summary_df) > 0:
        top_slice = slice_summary_df.sort_values('abs_gap_mean', ascending=False).head(top_k)
        print(f"\n[DIAG] abs(residual-comp) 均值最高的 Top-{top_k} 切片:")
        for _, r in top_slice.iterrows():
            print(
                f"  split={r['split']}, slice={r['slice_type']}:{r['slice_value']}, n={int(r['sample_count'])}, "
                f"|gap|mean={r['abs_gap_mean']:.6f}, improvement={r['improvement_abs']:.6f}, "
                f"comp_ratio_mean={r['comp_to_residual_ratio_mean']:.4f}"
            )
        print(f"按切片汇总CSV已保存至: {slice_summary_csv_path}")

    if model_was_training:
        model.train()

    return csv_path


# ==================== 部署文件生成工具 ====================
def print_model_file_info(file_path, label=""):
    """打印模型文件中包含的信息摘要"""
    if not os.path.exists(file_path):
        print(f"[INFO] 文件不存在: {file_path}")
        return
    file_size = os.path.getsize(file_path)
    print(f"\n{'='*60}")
    print(f"📦 模型文件信息{f' ({label})' if label else ''}: {os.path.basename(file_path)}")
    print(f"   路径: {file_path}")
    print(f"   大小: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            if 'state_dict' in data and 'config' in data:
                # 部署用合并文件
                print(f"   类型: 部署合并文件 (权重+配置+统计量)")
                sd = data['state_dict']
                print(f"   权重层数: {len(sd)} 个参数张量")
                total_params = sum(v.numel() for v in sd.values())
                print(f"   总参数量: {total_params:,}")
                cfg = data['config']
                print(f"   控制模式: {cfg.get('MODE', '?')}")
                print(f"   网络结构: LSTM({cfg.get('LSTM_LAYERS',2)}层, hidden={cfg.get('HIDDEN_SIZE',64)}) + MLP{cfg.get('MLP_HIDDEN',[128,64])}")
                print(f"   注意力: {'是' if cfg.get('USE_ATTENTION', True) else '否'}")
                print(f"   SEQ_LEN: {cfg.get('SEQ_LEN', 10)}")
                if 'stats' in data:
                    stats = data['stats']
                    print(f"   输入维度: time_dim={stats['time_mean'].shape[0]}, scalar_dim={stats['scalar_mean'].shape[0]}")
                if data.get('config', {}).get('RL_FINETUNED'):
                    print(f"   RL微调: 是 (源模型: {cfg.get('RL_SOURCE_MODEL','?')})")
            elif 'stats' in data and 'config' in data:
                # 纯配置文件
                print(f"   类型: 配置+统计量文件")
                cfg = data['config']
                stats = data['stats']
                print(f"   控制模式: {cfg.get('MODE', '?')}")
                print(f"   time_dim: {stats['time_mean'].shape[0]}")
                print(f"   包含字段: {list(data.keys())}")
            else:
                # state_dict
                print(f"   类型: 权重文件 (state_dict)")
                print(f"   权重层数: {len(data)} 个参数张量")
                total_params = sum(v.numel() for v in data.values() if hasattr(v, 'numel'))
                print(f"   总参数量: {total_params:,}")
                for k, v in list(data.items())[:5]:
                    print(f"     {k}: {list(v.shape)}")
                if len(data) > 5:
                    print(f"     ... (共 {len(data)} 个)")
        else:
            print(f"   类型: {type(data).__name__}")
    except Exception as e:
        print(f"   解析失败: {e}")
    print(f"{'='*60}")


def save_deployment_checkpoint(model, stats, config, model_dir, tag="", copy_to_conversion=True):
    """
    保存部署用合并checkpoint文件。
    该文件包含 state_dict + config + stats，既可用于部署转换，也可用于继续训练/测试。
    
    Args:
        model: 训练好的 AdaptiveNetwork 模型
        stats: 数据集统计量 dict (time_mean, time_std, scalar_mean, scalar_std)
        config: 序列化后的训练配置 dict
        model_dir: 模型保存目录
        tag: 文件名标签前缀 (如 "best_", "RL_")
        copy_to_conversion: 是否自动复制到 model_conversion_package/input_models/
    
    Returns:
        部署文件路径
    """
    import shutil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保config中的枚举类型已序列化
    deploy_config = dict(config)
    if isinstance(deploy_config.get('MODE'), ControlMode):
        deploy_config['MODE'] = deploy_config['MODE'].value
    
    # 合并为一个文件: state_dict + config + stats
    deploy_data = {
        'state_dict': model.state_dict(),
        'config': deploy_config,
        'stats': {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in stats.items()},
        'time_dim': stats['time_mean'].shape[0],
        'scalar_dim': stats['scalar_mean'].shape[0],
        'format_version': 2,  # 标记为合并格式
        'saved_at': timestamp,
    }

    dim_tag = f"_dim{int(stats['time_mean'].shape[0])}"
    deploy_filename = f"{tag}deploy_checkpoint_{timestamp}{dim_tag}.pt"
    deploy_path = os.path.join(model_dir, deploy_filename)
    torch.save(deploy_data, deploy_path)
    
    print_model_file_info(deploy_path, label="部署用合并文件")
    
    # 自动复制到 model_conversion_package/input_models/
    if copy_to_conversion:
        conversion_input_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model_conversion_package", "model_conversion_package", "input_models"
        )
        if os.path.isdir(conversion_input_dir):
            dst = os.path.join(conversion_input_dir, deploy_filename)
            shutil.copy2(deploy_path, dst)
            print(f"✅ 已自动复制到转换包输入目录: {dst}")
        else:
            print(f"⚠️  model_conversion_package/input_models/ 目录不存在，跳过自动复制。")
    
    return deploy_path


def _copy_generated_outputs_to_model_bundle(model_path, output_dir, generated_files, generated_dir_names=None):
    """
    将 output_models 下本次生成的文件复制到源模型所在目录，并按目录名统一重命名。
    仅当模型位于 MODEL_DIR 的某个子目录中时才执行。
    """
    source_model_path = os.path.abspath(model_path)
    source_dir = os.path.dirname(source_model_path)
    model_root_dir = os.path.abspath(TRAIN_CONFIG['MODEL_DIR'])

    if os.path.normcase(source_dir) == os.path.normcase(model_root_dir):
        return []

    if not os.path.isdir(source_dir):
        print(f"[DEPLOY_COPY] 源模型目录不存在，跳过回拷: {source_dir}")
        return []

    bundle_name = os.path.basename(source_dir.rstrip(os.sep))
    if not bundle_name:
        print("[DEPLOY_COPY] 无法解析源模型目录名，跳过回拷。")
        return []

    if bundle_name.startswith('best_adaptive_net_'):
        renamed_base = bundle_name.replace('best_adaptive_net_', 'best_adaptive_net_onnx_rknn_', 1)
    elif bundle_name.startswith('adaptive_net_'):
        renamed_base = bundle_name.replace('adaptive_net_', 'adaptive_net_onnx_rknn_', 1)
    elif bundle_name.startswith('RL_adaptive_net_'):
        renamed_base = bundle_name.replace('RL_adaptive_net_', 'RL_adaptive_net_onnx_rknn_', 1)
    else:
        renamed_base = f"{bundle_name}_onnx_rknn"

    copied_paths = []
    for src in generated_files:
        if not src or not os.path.isfile(src):
            continue

        src_name = os.path.basename(src)
        if src_name == 'adaptive_net_android.onnx':
            new_name = f"{renamed_base}.onnx"
        elif src_name == 'adaptive_net_android.fixed.onnx':
            new_name = f"{renamed_base}.fixed.onnx"
        elif src_name == 'adaptive_net_android.rknn':
            new_name = f"{renamed_base}.rknn"
        elif src_name == 'model_deploy_meta.json':
            new_name = f"{renamed_base}.json"
        else:
            new_name = f"{renamed_base}_{src_name}"

        dst = os.path.join(source_dir, new_name)
        shutil.copy2(src, dst)
        copied_paths.append(dst)
        print(f"[DEPLOY_COPY] 已复制: {dst}")

    for dir_name in generated_dir_names or []:
        if not dir_name:
            continue
        src_dir = os.path.join(output_dir, dir_name)
        if not os.path.isdir(src_dir):
            continue

        src_dir_name = os.path.basename(src_dir.rstrip(os.sep))
        if src_dir_name.endswith('_terminal_deploy'):
            new_dir_name = f"{renamed_base}_terminal_deploy"
        else:
            new_dir_name = f"{renamed_base}_{src_dir_name}"

        dst_dir = os.path.join(source_dir, new_dir_name)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        copied_paths.append(dst_dir)
        print(f"[DEPLOY_COPY] 已复制目录: {dst_dir}")

    return copied_paths


def generate_deployment_files(model_path, config_path):
    """
    从已训练的模型文件生成部署文件（ONNX + 元数据），输出到 model_conversion_package/output_models/。

    Args:
        model_path: 权重文件路径 (.pth)
        config_path: 配置文件路径 (.pt)
    """
    print(f"\n{'='*60}")
    print("生成部署文件")
    print(f"{'='*60}")
    print(f"  模型文件: {model_path}")
    print(f"  配置文件: {config_path}")

    # 1. 加载配置和权重
    saved_data = torch.load(config_path, map_location='cpu', weights_only=False)
    stats = saved_data['stats']
    train_config = saved_data['config']

    mode_value = train_config.get('MODE', ControlMode.A)
    if isinstance(mode_value, str):
        train_config['MODE'] = ControlMode(mode_value)

    time_dim = stats['time_mean'].shape[0]
    android_compat_issues = validate_android_training_config(train_config, actual_time_dim=time_dim)
    if android_compat_issues:
        print("⚠️  当前模型配置与 Android cpp 工程约定不一致，已停止导出。")
        for issue in android_compat_issues:
            print(f"   - {issue}")
        print(
            "   Android 工程固定输入为 10 帧 × 5 维时序特征 "
            f"({', '.join(ANDROID_TIME_FEATURE_NAMES)})，且使用 Mode A / D 原始输出后处理。"
        )
        return

    model = AdaptiveNetwork(
        mode=train_config['MODE'],
        time_dim=time_dim,
        scalar_dim=2,
        hidden_size=train_config.get('HIDDEN_SIZE', 64),
        lstm_layers=train_config.get('LSTM_LAYERS', 2),
        lstm_dropout=train_config.get('LSTM_DROPOUT', 0.3),
        use_attention=train_config.get('USE_ATTENTION', True),
        mlp_hidden=train_config.get('MLP_HIDDEN', [128, 64]),
        mlp_dropout=train_config.get('MLP_DROPOUT', 0.2),
        mode_a_alpha_range=train_config.get('MODE_A_ALPHA_RANGE', (0.5, 1.5)),
        mode_a_beta_scale=train_config.get('MODE_A_BETA_SCALE', 0.1),
        mode_d_delta_scale=train_config.get('MODE_D_DELTA_SCALE', 0.1),
        mode_d_use_tanh_bound=train_config.get('MODE_D_USE_TANH_BOUND', False),
        speed_feature_gain=train_config.get('SPEED_FEATURE_GAIN', 1.8)
    )
    _load_adaptive_network_state(
        model,
        model_path,
        map_location='cpu',
        strict=True,
        load_label=f"deployment source model {model_path}",
    )
    model.eval()
    print("✅ 模型加载成功")

    # 2. 生成 deploy_checkpoint 并复制到 input_models
    serializable_config = dict(train_config)
    if isinstance(serializable_config.get('MODE'), ControlMode):
        serializable_config['MODE'] = serializable_config['MODE'].value

    deploy_path = save_deployment_checkpoint(
        model=model,
        stats=stats,
        config=serializable_config,
        model_dir=TRAIN_CONFIG['MODEL_DIR'],
        tag="best_",
        copy_to_conversion=True
    )
    print(f"✅ 部署合并文件: {deploy_path}")

    # 3. 调用 export_to_onnx 脚本生成 ONNX 及终端部署包
    script_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model_conversion_package", "model_conversion_package", "scripts"
    )
    export_script = os.path.join(script_dir, "export_to_onnx.py")
    rknn_script = os.path.join(script_dir, "convert_to_rknn.py")
    package_dir = os.path.dirname(script_dir)
    windows_build_docker_script = os.path.join(package_dir, "build_rknn_docker_image.ps1")
    windows_run_docker_script = os.path.join(package_dir, "run_model_conversion_in_docker.ps1")
    generated_output_files = []
    generated_output_dirs = []
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model_conversion_package", "model_conversion_package", "output_models"
    )

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    if model_stem.startswith('best_adaptive_net_'):
        output_stem = model_stem.replace('best_adaptive_net_', 'best_', 1)
    elif model_stem.startswith('adaptive_net_'):
        output_stem = model_stem.replace('adaptive_net_', '', 1)
    elif model_stem.startswith('RL_adaptive_net_'):
        output_stem = model_stem.replace('RL_adaptive_net_', 'RL_', 1)
    else:
        output_stem = model_stem

    expected_terminal_deploy_dir = f"{output_stem}_terminal_deploy"
    expected_terminal_rknn_path = os.path.join(output_dir, expected_terminal_deploy_dir, "adaptive_net_android.rknn")

    def _remove_stale_rknn_outputs():
        stale_paths = [
            os.path.join(output_dir, "adaptive_net_android.rknn"),
            expected_terminal_rknn_path,
        ]
        for stale_path in stale_paths:
            if os.path.isfile(stale_path):
                try:
                    os.remove(stale_path)
                    print(f"[DEPLOY] 已清理旧RKNN文件: {stale_path}")
                except Exception as exc:
                    print(f"[DEPLOY] 清理旧RKNN文件失败: {stale_path} | {exc}")

    def _iter_rknn_python_candidates(current_python):
        candidates = []
        configured_python = str(TRAIN_CONFIG.get('RKNN_PYTHON_EXE', '') or '').strip()
        env_python = str(os.environ.get('RKNN_PYTHON_EXE', '') or '').strip()

        for candidate in [configured_python, env_python, current_python]:
            if not candidate:
                continue
            normalized = os.path.abspath(candidate) if os.path.exists(candidate) else candidate
            if normalized in candidates:
                continue
            candidates.append(normalized)
        return candidates

    def _windows_path_to_wsl(path_text):
        normalized = os.path.abspath(path_text).replace('\\', '/')
        drive, tail = os.path.splitdrive(normalized)
        if not drive:
            return normalized
        drive_letter = drive.rstrip(':').lower()
        return f"/mnt/{drive_letter}{tail}"

    def _resolve_wsl_distro_name(wsl_exe):
        def _sanitize_distro_name(raw_value):
            return str(raw_value or '').replace('\x00', '').strip()

        configured_distro = _sanitize_distro_name(TRAIN_CONFIG.get('RKNN_WSL_DISTRO', ''))
        env_distro = _sanitize_distro_name(os.environ.get('RKNN_WSL_DISTRO', ''))
        for candidate in (configured_distro, env_distro):
            if candidate:
                return candidate

        import subprocess

        query = subprocess.run(
            [wsl_exe, "-l", "-q"],
            cwd=package_dir,
            capture_output=True, text=False
        )
        if query.returncode != 0:
            return None

        stdout_bytes = query.stdout or b''
        stdout_text = ''
        for encoding in ('utf-16-le', 'utf-8', 'gbk'):
            try:
                stdout_text = stdout_bytes.decode(encoding)
                if stdout_text:
                    break
            except UnicodeDecodeError:
                continue
        if not stdout_text:
            stdout_text = stdout_bytes.decode(errors='ignore')

        for line in stdout_text.splitlines():
            distro = line.strip().lstrip('*').strip()
            if distro:
                distro = distro.replace('\x00', '').strip()
                if distro:
                    return distro
        return None

    def _python_can_import_rknn(python_exe):
        import subprocess

        if os.path.exists(python_exe) and not os.path.isfile(python_exe):
            return False, f"不是有效的 Python 可执行文件: {python_exe}"

        probe = subprocess.run(
            [python_exe, "-c", "from rknn.api import RKNN; print('RKNN_OK')"],
            cwd=script_dir,
            capture_output=True, text=True
        )
        if probe.returncode == 0:
            return True, None

        err_text = (probe.stderr or probe.stdout or '').strip()
        return False, err_text or "未知错误"

    def _print_docker_setup_hint():
        print("请先在 Windows 安装 Docker Desktop，并确认以下命令可在 PowerShell 中执行成功:")
        print("  docker --version")
        print("  docker info")
        print("若刚安装完成，请启动 Docker Desktop，等待 Engine Running 后再重新执行部署流程。")

    def _print_wsl_setup_hint():
        print("请确认 WSL 发行版中已安装可导入 rknn.api 的 Python 环境。")
        print("可通过 TRAIN_CONFIG['RKNN_WSL_DISTRO'] / TRAIN_CONFIG['RKNN_WSL_PYTHON'] 指定发行版与 Python。")

    def _run_wsl_subprocess(wsl_exe, distro_name, shell_cmd, cwd):
        import subprocess

        return subprocess.run(
            [wsl_exe, "-d", distro_name, "--", "bash", "-lc", shell_cmd],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

    def _try_rknn_via_wsl():
        import subprocess

        if os.name != 'nt':
            return False
        if not bool(TRAIN_CONFIG.get('RKNN_WSL_ENABLED', True)):
            return False

        wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
        if not wsl_exe:
            print("⚠️  当前系统未检测到 wsl 命令，无法执行 WSL RKNN 转换。")
            return False

        distro_name = _resolve_wsl_distro_name(wsl_exe)
        if not distro_name:
            print("⚠️  未解析到可用的 WSL 发行版，无法执行 WSL RKNN 转换。")
            _print_wsl_setup_hint()
            return False

        wsl_python = str(TRAIN_CONFIG.get('RKNN_WSL_PYTHON', 'python3') or 'python3').strip()
        env_wsl_python = str(os.environ.get('RKNN_WSL_PYTHON', '') or '').strip()
        if env_wsl_python:
            wsl_python = env_wsl_python

        wsl_script_dir = _windows_path_to_wsl(script_dir)
        wsl_rknn_script = _windows_path_to_wsl(rknn_script)
        probe_cmd = (
            f"cd {shlex.quote(wsl_script_dir)} ; "
            f"{shlex.quote(wsl_python)} -c \"from rknn.api import RKNN; print('RKNN_OK')\""
        )
        probe = _run_wsl_subprocess(wsl_exe, distro_name, probe_cmd, package_dir)
        if probe.returncode != 0:
            print(f"⚠️  WSL RKNN 环境检测失败: distro={distro_name}, python={wsl_python}")
            probe_err = (probe.stderr or probe.stdout or '').strip()
            if probe_err:
                print(f"WSL 探测错误信息:\n{probe_err}")
            _print_wsl_setup_hint()
            return False

        print(f"\n优先尝试通过 WSL 生成 RKNN... distro={distro_name}, python={wsl_python}")
        run_cmd = f"cd {shlex.quote(wsl_script_dir)} ; {shlex.quote(wsl_python)} {shlex.quote(wsl_rknn_script)}"
        run_result = _run_wsl_subprocess(wsl_exe, distro_name, run_cmd, package_dir)
        if run_result.stdout:
            print(run_result.stdout)
        if run_result.returncode != 0:
            print("⚠️  WSL RKNN 转换失败。")
            combined_output = (run_result.stderr or run_result.stdout or '').strip()
            if combined_output:
                print(f"WSL 转换错误信息:\n{combined_output}")
            return False

        generated_rknn_path = os.path.join(package_dir, "output_models", "adaptive_net_android.rknn")
        if not os.path.isfile(generated_rknn_path):
            print("⚠️  WSL RKNN 转换命令返回成功，但未检测到 .rknn 文件。")
            print(f"缺失文件: {generated_rknn_path}")
            return False

        print("✅ WSL RKNN 转换成功，.rknn 文件已生成。")
        return True

    def _print_docker_runtime_hint(output_text):
        lowered = (output_text or '').lower()
        if not output_text:
            return
        if "docker desktop" in lowered or "engine is not running" in lowered or "docker daemon" in lowered:
            print("提示: Docker 已安装，但 Docker Desktop 可能未启动。请先启动 Docker Desktop 再重试。")
        elif "access is denied" in lowered:
            print("提示: Docker 当前权限不足。请确认当前账户具备 Docker 使用权限，或尝试以管理员身份启动 Docker Desktop。")
        elif "cannot connect to the docker daemon" in lowered:
            print("提示: 当前无法连接到 Docker daemon。请确认 Docker Desktop 已完全启动，且 WSL2/Hyper-V 运行正常。")

    def _try_rknn_via_docker():
        import subprocess, sys

        if os.name != 'nt':
            print("⚠️  当前未配置非 Windows 的自动 Docker fallback，请手动执行 RKNN 转换脚本。")
            return False

        if not (os.path.isfile(windows_build_docker_script) and os.path.isfile(windows_run_docker_script)):
            print("⚠️  未找到 Windows Docker 转换脚本，无法自动生成 RKNN。")
            return False

        docker_exe = shutil.which("docker")
        if not docker_exe:
            print("⚠️  当前系统未检测到 docker 命令，无法执行 RKNN Docker fallback。")
            _print_docker_setup_hint()
            return False

        docker_info_result = subprocess.run(
            [docker_exe, "info"],
            cwd=package_dir,
            capture_output=True, text=True
        )
        if docker_info_result.returncode != 0:
            print("⚠️  检测到 Docker 已安装，但当前不可用。")
            combined_output = (docker_info_result.stderr or docker_info_result.stdout or '').strip()
            if combined_output:
                print(f"Docker 状态信息:\n{combined_output}")
                _print_docker_runtime_hint(combined_output)
            _print_docker_setup_hint()
            return False

        powershell_exe = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell_exe:
            print("⚠️  未找到 PowerShell，无法自动执行 Docker RKNN 转换。")
            return False

        print("\n优先尝试通过 Docker 自动生成 RKNN...")

        build_result = subprocess.run(
            [powershell_exe, "-ExecutionPolicy", "Bypass", "-File", windows_build_docker_script],
            cwd=package_dir,
            capture_output=True, text=True
        )
        if build_result.stdout:
            print(build_result.stdout)
        if build_result.returncode != 0:
            print("⚠️  Docker RKNN 镜像构建失败。")
            combined_output = (build_result.stderr or build_result.stdout or '').strip()
            if combined_output:
                print(f"Docker 构建错误信息:\n{combined_output}")
                _print_docker_runtime_hint(combined_output)
            return False

        run_result = subprocess.run(
            [powershell_exe, "-ExecutionPolicy", "Bypass", "-File", windows_run_docker_script],
            cwd=package_dir,
            capture_output=True, text=True
        )
        if run_result.stdout:
            print(run_result.stdout)
        if run_result.returncode != 0:
            print("⚠️  Docker RKNN 转换失败。")
            combined_output = (run_result.stderr or run_result.stdout or '').strip()
            if combined_output:
                print(f"Docker 转换错误信息:\n{combined_output}")
                _print_docker_runtime_hint(combined_output)
            return False

        generated_rknn_path = os.path.join(package_dir, "output_models", "adaptive_net_android.rknn")
        if not os.path.isfile(generated_rknn_path):
            print("⚠️  Docker RKNN 转换命令返回成功，但未检测到 .rknn 文件。")
            print(f"缺失文件: {generated_rknn_path}")
            return False

        print("✅ Docker RKNN 转换成功，.rknn 文件已生成。")
        return True

    if os.path.isfile(export_script):
        _remove_stale_rknn_outputs()
        print(f"\n正在执行 ONNX 导出脚本...")
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, export_script],
            cwd=script_dir,
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"⚠️  ONNX 导出脚本返回错误:\n{result.stderr}")
        else:
            generated_output_files.extend([
                os.path.join(output_dir, "adaptive_net_android.onnx"),
                os.path.join(output_dir, "adaptive_net_android.fixed.onnx"),
                os.path.join(output_dir, "model_deploy_meta.json")
            ])
            generated_output_dirs.append(expected_terminal_deploy_dir)
            print(f"✅ 部署文件已生成，输出目录: {output_dir}")

            if os.path.isfile(rknn_script):
                print(f"\n正在执行 RKNN 转换脚本...")
                rknn_success = False
                last_rknn_error = None
                wsl_success = _try_rknn_via_wsl()
                if wsl_success:
                    generated_output_files.append(os.path.join(output_dir, "adaptive_net_android.rknn"))
                    rknn_success = True
                else:
                    docker_success = _try_rknn_via_docker()
                    if docker_success:
                        generated_output_files.append(os.path.join(output_dir, "adaptive_net_android.rknn"))
                        rknn_success = True
                    else:
                        print("⚠️  WSL / Docker 自动 RKNN 转换未成功，回退到本地 RKNN Toolkit2 Python 环境。")
                        for python_exe in _iter_rknn_python_candidates(sys.executable):
                            can_import, reason = _python_can_import_rknn(python_exe)
                            if not can_import:
                                print(f"- 跳过 RKNN Python 环境: {python_exe}")
                                if reason:
                                    print(f"  原因: {reason}")
                                last_rknn_error = reason
                                continue

                            print(f"- 使用以下 Python 执行 RKNN 转换: {python_exe}")
                            rknn_result = subprocess.run(
                                [python_exe, rknn_script],
                                cwd=script_dir,
                                capture_output=True, text=True
                            )
                            if rknn_result.stdout:
                                print(rknn_result.stdout)
                            if rknn_result.returncode == 0:
                                generated_output_files.append(os.path.join(output_dir, "adaptive_net_android.rknn"))
                                print("✅ RKNN 文件已生成并写入终端部署目录。")
                                rknn_success = True
                                break

                            last_rknn_error = (rknn_result.stderr or rknn_result.stdout or '').strip()
                            print(f"⚠️  RKNN 转换执行失败: {python_exe}")
                            if last_rknn_error:
                                print(f"RKNN 转换错误信息:\n{last_rknn_error}")

                if not rknn_success:
                    print(
                        "⚠️  RKNN 转换未成功完成。"
                        "这通常表示 WSL / Docker 自动转换失败，"
                        "且当前 Python 环境未安装 RKNN Toolkit2，"
                        "并且未配置可用的 Windows 本地 RKNN Python 环境。"
                    )
                    configured_wsl_distro = str(TRAIN_CONFIG.get('RKNN_WSL_DISTRO', '') or '').strip()
                    configured_wsl_python = str(TRAIN_CONFIG.get('RKNN_WSL_PYTHON', '') or '').strip()
                    if configured_wsl_distro or configured_wsl_python:
                        print(
                            "当前配置的 WSL RKNN 参数: "
                            f"distro={configured_wsl_distro or '<auto>'}, "
                            f"python={configured_wsl_python or 'python3'}"
                        )
                    configured_python = str(TRAIN_CONFIG.get('RKNN_PYTHON_EXE', '') or '').strip()
                    if configured_python:
                        print(f"当前配置的 RKNN_PYTHON_EXE: {configured_python}")
                    elif os.name == 'nt':
                        print(
                            "可在 TRAIN_CONFIG['RKNN_WSL_DISTRO'] / TRAIN_CONFIG['RKNN_WSL_PYTHON'] "
                            "指定 WSL RKNN 环境，或在 TRAIN_CONFIG['RKNN_PYTHON_EXE'] / 环境变量 RKNN_PYTHON_EXE "
                            "中指定本地 RKNN Python。"
                        )
                    if last_rknn_error:
                        print(f"最后一次 RKNN 错误:\n{last_rknn_error}")
                elif not os.path.isfile(expected_terminal_rknn_path):
                    print("⚠️  RKNN 转换流程报告成功，但当前 terminal_deploy 目录中缺少 .rknn 文件。")
                    print(f"   期望文件: {expected_terminal_rknn_path}")
            else:
                print(f"⚠️  未找到 RKNN 转换脚本: {rknn_script}")

            if not os.path.isfile(expected_terminal_rknn_path):
                print("❌ 本次部署未生成新的 terminal_deploy RKNN 文件。")
                print(f"   缺失文件: {expected_terminal_rknn_path}")
                print("   注意: output_models 根目录若仍有 adaptive_net_android.rknn，可能只是历史残留文件，不代表本次转换成功。")

            copied_outputs = _copy_generated_outputs_to_model_bundle(
                model_path=model_path,
                output_dir=output_dir,
                generated_files=generated_output_files,
                generated_dir_names=generated_output_dirs
            )
            if copied_outputs:
                print(f"✅ 已自动回拷部署产物到源模型目录，共 {len(copied_outputs)} 项。")
    else:
        print(f"⚠️  未找到 ONNX 导出脚本: {export_script}")
        print("   请手动执行 model_conversion_package 中的转换工具。")

    print(f"{'='*60}")


# ==================== 训练流程（改进版） ====================
def train_network(resume_model_path=None):
    print(">>> 开始训练自适应网络（改进版）...")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _prepare_training_run_dir(TRAIN_CONFIG, run_label=run_timestamp)

    # 设备选择：支持 auto / cpu / cuda
    cfg_device = str(TRAIN_CONFIG.get('DEVICE', 'auto')).lower()
    if cfg_device in ('cuda', 'gpu'):
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU.")
            cfg_device = 'cpu'
    if cfg_device == 'auto':
        cfg_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg_device)
    print(f"使用的设备: {device}")

    if device.type == 'cuda':
        device_id = device.index
        device_name = torch.cuda.get_device_name(device_id)
        props = torch.cuda.get_device_properties(device_id)
        total_memory_gb = props.total_memory / (1024**3)
        print(f"CUDA 设备 ID: {device_id}")
        print(f"CUDA 设备名称: {device_name}")
        print(f"CUDA 总显存: {total_memory_gb:.2f} GB")
        print(f"CUDA 计算能力: {props.major}.{props.minor}")

    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
    use_amp = bool(TRAIN_CONFIG.get('USE_AMP', False)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    samples = load_samples(TRAIN_CONFIG['DATA_ROOT'], seq_len=TRAIN_CONFIG['SEQ_LEN'])
    if not samples:
        print("Error: No training data found. Cannot proceed to train.")
        return None, None

    print(f"Dataset size: {len(samples)}")
    if len(samples) < 2:
        print("Error: Need at least 2 samples to split train/val datasets.")
        return None, None

    # 按来源文件分组划分，避免同一轨迹窗口同时出现在训练/验证中导致泄漏
    val_size = max(1, int(len(samples) * TRAIN_CONFIG['VAL_SPLIT']))
    if val_size >= len(samples):
        val_size = len(samples) - 1

    rng = np.random.default_rng(42)
    group_to_indices = {}
    for idx, sample in enumerate(samples):
        source_id = sample.get('source_id', 'unknown_source')
        if source_id not in group_to_indices:
            group_to_indices[source_id] = []
        group_to_indices[source_id].append(idx)

    group_keys = list(group_to_indices.keys())

    speed_bins = _get_speed_bin_edges(TRAIN_CONFIG)

    group_speed_mean = {
        g: float(np.mean([float(samples[i]['scalar'][0]) for i in idx_list]))
        for g, idx_list in group_to_indices.items()
    }
    group_sample_count = {g: len(idx_list) for g, idx_list in group_to_indices.items()}

    val_group_set = set()

    if TRAIN_CONFIG.get('USE_SPEED_STRATIFIED_SPLIT', False):
        # 先按组的平均速度分层，再在每层选取验证组（近似满足各层val比例）
        bin_to_groups = {}
        for g in group_keys:
            b = _speed_bin_id(group_speed_mean[g], speed_bins)
            if b not in bin_to_groups:
                bin_to_groups[b] = []
            bin_to_groups[b].append(g)

        for b, groups_in_bin in bin_to_groups.items():
            rng.shuffle(groups_in_bin)
            bin_total = sum(group_sample_count[g] for g in groups_in_bin)
            target_bin_val = int(round(bin_total * TRAIN_CONFIG['VAL_SPLIT']))

            current_bin_val = 0
            for g in groups_in_bin:
                if current_bin_val >= target_bin_val:
                    break
                val_group_set.add(g)
                current_bin_val += group_sample_count[g]

        # 若分层后验证样本不足，则补充剩余组
        if sum(group_sample_count[g] for g in val_group_set) < val_size:
            remaining_groups = [g for g in group_keys if g not in val_group_set]
            rng.shuffle(remaining_groups)
            for g in remaining_groups:
                val_group_set.add(g)
                if sum(group_sample_count[x] for x in val_group_set) >= val_size:
                    break
    else:
        rng.shuffle(group_keys)
        current_val = 0
        for g in group_keys:
            val_group_set.add(g)
            current_val += group_sample_count[g]
            if current_val >= val_size:
                break

    val_indices = []
    for g in val_group_set:
        val_indices.extend(group_to_indices[g])

    val_index_set = set(val_indices)
    train_indices = [i for i in range(len(samples)) if i not in val_index_set]

    # 极端情况下兜底，确保两侧至少有1个样本
    if len(train_indices) == 0:
        move_idx = val_indices.pop()
        train_indices.append(move_idx)
    if len(val_indices) == 0:
        move_idx = train_indices.pop()
        val_indices.append(move_idx)

    def bin_count_report(index_list):
        counts = [0] * (len(speed_bins) + 1)
        for idx in index_list:
            v = float(samples[idx]['scalar'][0])
            counts[_speed_bin_id(v, speed_bins)] += 1
        return counts

    train_bin_counts = bin_count_report(train_indices)
    val_bin_counts = bin_count_report(val_indices)
    print(f"Split -> train: {len(train_indices)}, val: {len(val_indices)}, groups: {len(group_keys)}")
    print(
        f"Speed bins (m/s edges={speed_bins}) -> "
        f"train[{_format_speed_bin_count_report(train_bin_counts, speed_bins)}], "
        f"val[{_format_speed_bin_count_report(val_bin_counts, speed_bins)}]"
    )

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    # 仅使用训练集统计量做归一化，避免验证信息泄漏
    if bool(TRAIN_CONFIG.get('REBUILD_BASELINE_CACHE', False)):
        print("开始重建基础LQR预计算表，完成后自动进入训练...")
    else:
        print("开始检查基础LQR预计算表，若缓存可用将直接复用...")

    baseline_lookup = _resolve_android_lqr_lookup_cache(
        samples,
        rebuild=bool(TRAIN_CONFIG.get('REBUILD_BASELINE_CACHE', False)),
    )
    print("基础LQR预计算表已就绪，开始训练主循环...")

    train_dataset = ControlDataset(train_samples, augment=True, baseline_lookup=baseline_lookup)
    val_dataset = ControlDataset(val_samples, stats=train_dataset.stats, augment=False, baseline_lookup=baseline_lookup)

    # DataLoader配置：若使用CUDA则可以开启pin_memory以加速Host->Device的拷贝
    num_workers = int(TRAIN_CONFIG.get('NUM_WORKERS', 0))
    pin_memory = bool(TRAIN_CONFIG.get('PIN_MEMORY', True)) and device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 计算时间序列维度
    time_dim = train_dataset[0][0].shape[-1]

    model = AdaptiveNetwork(
        mode=TRAIN_CONFIG['MODE'],
        time_dim=time_dim,
        scalar_dim=2,
        hidden_size=TRAIN_CONFIG['HIDDEN_SIZE'],
        lstm_layers=TRAIN_CONFIG['LSTM_LAYERS'],
        lstm_dropout=TRAIN_CONFIG['LSTM_DROPOUT'],
        use_attention=TRAIN_CONFIG['USE_ATTENTION'],
        mlp_hidden=TRAIN_CONFIG['MLP_HIDDEN'],
        mlp_dropout=TRAIN_CONFIG['MLP_DROPOUT'],
        mode_a_alpha_range=TRAIN_CONFIG.get('MODE_A_ALPHA_RANGE', (0.5, 1.5)),
        mode_a_beta_scale=TRAIN_CONFIG.get('MODE_A_BETA_SCALE', 0.1),
        mode_d_delta_scale=TRAIN_CONFIG.get('MODE_D_DELTA_SCALE', 0.1),
        mode_d_use_tanh_bound=TRAIN_CONFIG.get('MODE_D_USE_TANH_BOUND', False),
        speed_feature_gain=TRAIN_CONFIG.get('SPEED_FEATURE_GAIN', 1.8),
    ).to(device)

    if resume_model_path:
        try:
            _load_adaptive_network_state(
                model,
                resume_model_path,
                map_location=device,
                strict=True,
                load_label=f"resume checkpoint {resume_model_path}",
            )
            print(f"Resume training from checkpoint: {resume_model_path}")
        except Exception as e:
            print(f"Warning: failed to load resume checkpoint {resume_model_path}, fallback to fresh training. Error: {e}")

    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['LR'], weight_decay=TRAIN_CONFIG['WEIGHT_DECAY'])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['EPOCHS'])
    speed_bins = _get_speed_bin_edges(TRAIN_CONFIG)
    speed_bins_tensor = torch.tensor(speed_bins, dtype=torch.float32, device=device)
    num_speed_bins = len(speed_bins) + 1
    steer_min = float(cfg.STEERING_LIMIT_MIN)
    steer_max = float(cfg.STEERING_LIMIT_MAX)
    train_speed_bin_weights = _build_speed_bin_loss_weights(train_dataset.raw_scalar[:, 0], TRAIN_CONFIG, device=device)
    val_speed_bin_weights = _build_speed_bin_loss_weights(val_dataset.raw_scalar[:, 0], TRAIN_CONFIG, device=device)

    time_mean_device = train_dataset.stats['time_mean'].to(device)
    time_std_device = train_dataset.stats['time_std'].to(device)

    print(
        f"Speed-loss weights train[{_format_speed_bin_value_report(train_speed_bin_weights.detach().cpu().tolist(), speed_bins)}]"
    )
    print(
        f"Speed-loss weights val[{_format_speed_bin_value_report(val_speed_bin_weights.detach().cpu().tolist(), speed_bins)}]"
    )

    print(f"Training for {TRAIN_CONFIG['EPOCHS']} epochs...")
    # 固定本次训练的时间戳：用于best路径覆盖保存，避免单次训练产生多个best文件

    # 在训练开始前准备可序列化的配置（用于保存到磁盘）
    serializable_config = dict(TRAIN_CONFIG)
    if isinstance(serializable_config.get('MODE'), ControlMode):
        serializable_config['MODE'] = serializable_config['MODE'].value

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = None
    best_config_path = None
    last_epoch_is_best = False
    epoch_metrics = []

    for epoch in range(TRAIN_CONFIG['EPOCHS']):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_abs_sum = 0.0
        train_count = 0
        train_bin_abs_sum = [0.0] * num_speed_bins
        train_bin_count = [0] * num_speed_bins
        for batch in train_loader:
            norm_time, norm_scalar, raw_scalar, e, theta, delta_opt, k_e_base, k_th_base = [x.to(device) for x in batch]

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                if TRAIN_CONFIG['MODE'] == ControlMode.A:
                    alpha_e, beta_e, alpha_th, beta_th = model(norm_time, norm_scalar)

                    k_e_final = alpha_e * k_e_base + beta_e
                    k_th_final = alpha_th * k_th_base + beta_th

                    delta_base = -k_e_base * e - k_th_base * theta
                    delta_pred = -k_e_final * e - k_th_final * theta
                    model_comp = delta_pred - delta_base

                elif TRAIN_CONFIG['MODE'] == ControlMode.D:
                    delta_add = model(norm_time, norm_scalar)
                    delta_base = -k_e_base * e - k_th_base * theta
                    delta_pred = delta_base + delta_add
                    model_comp = delta_add
                else:
                    delta_base = -k_e_base * e - k_th_base * theta
                    delta_pred = -k_e_base * e - k_th_base * theta
                    model_comp = torch.zeros_like(delta_pred)

                # 转角限幅一致性：用限幅后的预测与标签计算主损失
                delta_pred_clip = torch.clamp(delta_pred, min=steer_min, max=steer_max)
                delta_opt_clip = torch.clamp(delta_opt, min=steer_min, max=steer_max)
                speed_bin_ids = torch.bucketize(raw_scalar[:, 0].contiguous(), speed_bins_tensor, right=False)
                speed_sample_weight = train_speed_bin_weights.index_select(0, speed_bin_ids)
                loss_track_raw = F.smooth_l1_loss(
                    delta_pred_clip,
                    delta_opt_clip,
                    beta=TRAIN_CONFIG['HUBER_BETA'],
                    reduction='none'
                )
                loss_track = (speed_sample_weight * loss_track_raw).mean()

                # 残差对齐：显式学习“应补偿多少”而不仅是最终角度误差
                residual = delta_opt - delta_base
                loss_comp, loss_under_comp, loss_over_comp = _compute_compensation_balance_losses(
                    model_comp,
                    residual,
                    speed_sample_weight,
                    TRAIN_CONFIG,
                )

                # 越界惩罚：抑制模型输出超出执行器物理边界
                overflow_upper = torch.relu(delta_pred - steer_max)
                overflow_lower = torch.relu(steer_min - delta_pred)
                loss_limit = (overflow_upper + overflow_lower).mean()

                # 变化率惩罚：使用窗口末端历史误差近似上一时刻转角，抑制抖动
                prev_e = norm_time[:, -1, 0] * time_std_device[0] + time_mean_device[0]
                prev_theta = norm_time[:, -1, 1] * time_std_device[1] + time_mean_device[1]
                delta_prev = -k_e_base * prev_e - k_th_base * prev_theta
                delta_prev_clip = torch.clamp(delta_prev, min=steer_min, max=steer_max)
                loss_rate = torch.abs(delta_pred_clip - delta_prev_clip).mean()

                loss = (
                    loss_track
                    + TRAIN_CONFIG['STEER_LIMIT_LOSS_WEIGHT'] * loss_limit
                    + TRAIN_CONFIG['RATE_LOSS_WEIGHT'] * loss_rate
                    + TRAIN_CONFIG.get('COMP_LOSS_WEIGHT', 0.0) * loss_comp
                    + TRAIN_CONFIG.get('UNDER_COMP_LOSS_WEIGHT', 0.0) * loss_under_comp
                    + TRAIN_CONFIG.get('OVER_COMP_LOSS_WEIGHT', 0.0) * loss_over_comp
                )

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['GRAD_CLIP_NORM'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['GRAD_CLIP_NORM'])
                optimizer.step()

            total_train_loss += loss.item() * len(e)

            abs_err = torch.abs(delta_pred_clip.detach() - delta_opt_clip)
            train_abs_sum += abs_err.sum().item()
            train_count += abs_err.numel()

            for bin_idx in range(num_speed_bins):
                mask = speed_bin_ids == bin_idx
                if torch.any(mask):
                    train_bin_abs_sum[bin_idx] += abs_err[mask].sum().item()
                    train_bin_count[bin_idx] += int(mask.sum().item())

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_train_mae = train_abs_sum / max(1, train_count)
        train_bin_mae = [
            (train_bin_abs_sum[i] / train_bin_count[i]) if train_bin_count[i] > 0 else np.nan
            for i in range(num_speed_bins)
        ]

        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_abs_sum = 0.0
        val_count = 0
        val_bin_abs_sum = [0.0] * num_speed_bins
        val_bin_count = [0] * num_speed_bins
        with torch.no_grad():
            for batch in val_loader:
                norm_time, norm_scalar, raw_scalar, e, theta, delta_opt, k_e_base, k_th_base = [x.to(device) for x in batch]

                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    if TRAIN_CONFIG['MODE'] == ControlMode.A:
                        alpha_e, beta_e, alpha_th, beta_th = model(norm_time, norm_scalar)
                        k_e_final = alpha_e * k_e_base + beta_e
                        k_th_final = alpha_th * k_th_base + beta_th
                        delta_base = -k_e_base * e - k_th_base * theta
                        delta_pred = -k_e_final * e - k_th_final * theta
                        model_comp = delta_pred - delta_base
                    elif TRAIN_CONFIG['MODE'] == ControlMode.D:
                        delta_add = model(norm_time, norm_scalar)
                        delta_base = -k_e_base * e - k_th_base * theta
                        delta_pred = delta_base + delta_add
                        model_comp = delta_add
                    else:
                        delta_base = -k_e_base * e - k_th_base * theta
                        delta_pred = -k_e_base * e - k_th_base * theta
                        model_comp = torch.zeros_like(delta_pred)

                    delta_pred_clip = torch.clamp(delta_pred, min=steer_min, max=steer_max)
                    delta_opt_clip = torch.clamp(delta_opt, min=steer_min, max=steer_max)

                    speed_bin_ids = torch.bucketize(raw_scalar[:, 0].contiguous(), speed_bins_tensor, right=False)
                    speed_sample_weight = val_speed_bin_weights.index_select(0, speed_bin_ids)
                    loss_track_raw = F.smooth_l1_loss(
                        delta_pred_clip,
                        delta_opt_clip,
                        beta=TRAIN_CONFIG['HUBER_BETA'],
                        reduction='none'
                    )
                    loss_track = (speed_sample_weight * loss_track_raw).mean()
                    residual = delta_opt - delta_base
                    loss_comp, loss_under_comp, loss_over_comp = _compute_compensation_balance_losses(
                        model_comp,
                        residual,
                        speed_sample_weight,
                        TRAIN_CONFIG,
                    )
                    overflow_upper = torch.relu(delta_pred - steer_max)
                    overflow_lower = torch.relu(steer_min - delta_pred)
                    loss_limit = (overflow_upper + overflow_lower).mean()

                    prev_e = norm_time[:, -1, 0] * time_std_device[0] + time_mean_device[0]
                    prev_theta = norm_time[:, -1, 1] * time_std_device[1] + time_mean_device[1]
                    delta_prev = -k_e_base * prev_e - k_th_base * prev_theta
                    delta_prev_clip = torch.clamp(delta_prev, min=steer_min, max=steer_max)
                    loss_rate = torch.abs(delta_pred_clip - delta_prev_clip).mean()

                    loss = (
                        loss_track
                        + TRAIN_CONFIG['STEER_LIMIT_LOSS_WEIGHT'] * loss_limit
                        + TRAIN_CONFIG['RATE_LOSS_WEIGHT'] * loss_rate
                        + TRAIN_CONFIG.get('COMP_LOSS_WEIGHT', 0.0) * loss_comp
                        + TRAIN_CONFIG.get('UNDER_COMP_LOSS_WEIGHT', 0.0) * loss_under_comp
                        + TRAIN_CONFIG.get('OVER_COMP_LOSS_WEIGHT', 0.0) * loss_over_comp
                    )

                total_val_loss += loss.item() * len(e)

                abs_err = torch.abs(delta_pred_clip - delta_opt_clip)
                val_abs_sum += abs_err.sum().item()
                val_count += abs_err.numel()

                for bin_idx in range(num_speed_bins):
                    mask = speed_bin_ids == bin_idx
                    if torch.any(mask):
                        val_bin_abs_sum[bin_idx] += abs_err[mask].sum().item()
                        val_bin_count[bin_idx] += int(mask.sum().item())

        avg_val_loss = total_val_loss / len(val_dataset)
        avg_val_mae = val_abs_sum / max(1, val_count)
        val_bin_mae = [
            (val_bin_abs_sum[i] / val_bin_count[i]) if val_bin_count[i] > 0 else np.nan
            for i in range(num_speed_bins)
        ]

        scheduler.step()  # 更新学习率

        def fmt_bin_mae(mae_list):
            return '[' + ', '.join('nan' if np.isnan(v) else f'{v:.5f}' for v in mae_list) + ']'

        print(
            f"Epoch {epoch + 1}/{TRAIN_CONFIG['EPOCHS']} | "
            f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
            f"Train MAE: {avg_train_mae:.6f}, Val MAE: {avg_val_mae:.6f}"
        )
        print(f"  Speed-bin MAE train={fmt_bin_mae(train_bin_mae)} | val={fmt_bin_mae(val_bin_mae)}")

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_mae': avg_train_mae,
            'val_mae': avg_val_mae,
            'lr': optimizer.param_groups[0]['lr']
        }
        for bin_idx in range(num_speed_bins):
            epoch_record[f'train_mae_bin_{bin_idx}'] = train_bin_mae[bin_idx]
            epoch_record[f'val_mae_bin_{bin_idx}'] = val_bin_mae[bin_idx]
            epoch_record[f'train_count_bin_{bin_idx}'] = train_bin_count[bin_idx]
            epoch_record[f'val_count_bin_{bin_idx}'] = val_bin_count[bin_idx]
        epoch_metrics.append(epoch_record)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            last_epoch_is_best = True
            # 时间戳命名 + 覆盖保存：单次训练只保留一个“当前最优”best文件
            dim_tag = f"_dim{time_dim}"
            best_model_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"best_adaptive_net_{run_timestamp}{dim_tag}.pth")
            best_config_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"best_config_{run_timestamp}{dim_tag}.pt")
            torch.save(model.state_dict(), best_model_path)
            # 同步保存配置文件（用于后续测试/加载）
            best_config_payload = {
                'stats': train_dataset.stats,
                'config': serializable_config
            }
            torch.save(best_config_payload, best_config_path)
            print(f"  -> Best model saved (val loss: {best_val_loss:.6f})")
        else:
            last_epoch_is_best = False
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['EARLY_STOP_PATIENCE']:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 保存最终模型和配置（可选）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_tag = f"_dim{time_dim}"
    final_model_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"adaptive_net_{timestamp}{dim_tag}.pth")
    config_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"config_{timestamp}{dim_tag}.pt")
    metrics_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"training_metrics_{timestamp}.csv")

    if last_epoch_is_best and best_model_path and best_config_path:
        # 最后一轮就是最佳模型时，避免重复保存同一份权重与配置
        final_model_path = best_model_path
        config_path = best_config_path
        print("最后一轮即最佳模型，已跳过重复保存最终模型。")
    else:
        torch.save(model.state_dict(), final_model_path)
        serializable_config = dict(TRAIN_CONFIG)
        if isinstance(serializable_config.get('MODE'), ControlMode):
            serializable_config['MODE'] = serializable_config['MODE'].value

        torch.save({
            'stats': train_dataset.stats,
            'config': serializable_config
        }, config_path)

    pd.DataFrame(epoch_metrics).to_csv(metrics_path, index=False)
    print("训练主循环已结束，开始生成训练后诊断与部署文件...")

    # 训练后诊断：优先基于最佳权重导出，确保诊断结果对应最终可用模型
    diag_model_path = best_model_path if best_model_path and os.path.exists(best_model_path) else final_model_path
    diag_path = None
    if diag_model_path and os.path.exists(diag_model_path):
        try:
            print("开始导出训练后诊断文件...")
            _load_adaptive_network_state(
                model,
                diag_model_path,
                map_location=device,
                strict=True,
                load_label=f"diagnostic model {diag_model_path}",
            )
            diag_path = export_pretrain_diagnostics(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                model_dir=TRAIN_CONFIG['MODEL_DIR'],
                mode=TRAIN_CONFIG['MODE'],
            )
        except Exception as e:
            print(f"Warning: 训练后诊断导出失败: {e}")
    else:
        print("Warning: 训练后诊断导出跳过，未找到可用模型权重文件。")

    if diag_path and os.path.exists(diag_path):
        by_source_diag_path = diag_path.replace('pretrain_diagnostics_', 'pretrain_diagnostics_by_source_')
        if os.path.exists(by_source_diag_path):
            try:
                summarize_latest_compensation_ratio(TRAIN_CONFIG['MODEL_DIR'], diag_path=by_source_diag_path)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_stem = os.path.splitext(os.path.basename(diag_model_path))[0]
                analysis_md_path = os.path.join(
                    TRAIN_CONFIG['MODEL_DIR'],
                    f"by_source_analysis_{model_stem}_{ts}.md"
                )
                analysis_result = analyze_by_source_file(
                    by_source_csv_path=by_source_diag_path,
                    output_md_path=analysis_md_path,
                    top_k=10
                )
                print(f"训练后 by_source 源诊断文档已保存至: {analysis_result['output_md_path']}")
            except Exception as e:
                print(f"Warning: 训练后 by_source 源诊断分析失败: {e}")
        else:
            print(f"Warning: 未找到 by_source 诊断CSV: {by_source_diag_path}")

    # 打印已保存文件信息
    print_model_file_info(final_model_path, label="最终权重")
    print_model_file_info(config_path, label="配置文件")

    # 生成部署用合并checkpoint（自动复制到 model_conversion_package/input_models/）
    try:
        print("开始生成部署用合并checkpoint...")
        deploy_model = model
        deploy_model_path = diag_model_path if diag_model_path and os.path.exists(diag_model_path) else final_model_path
        _load_adaptive_network_state(
            deploy_model,
            deploy_model_path,
            map_location=device,
            strict=True,
            load_label=f"deploy model {deploy_model_path}",
        )
        deploy_tag = "best_" if (best_model_path and deploy_model_path == best_model_path) else ""
        deploy_path = save_deployment_checkpoint(
            model=deploy_model,
            stats=train_dataset.stats,
            config=serializable_config,
            model_dir=TRAIN_CONFIG['MODEL_DIR'],
            tag=deploy_tag,
            copy_to_conversion=True
        )
        print(f"✅ 部署合并文件已保存至: {deploy_path}")
    except Exception as e:
        print(f"Warning: 部署合并文件生成失败: {e}")

    print(f"最终模型已保存至: {final_model_path}")
    print(f"最佳模型（若早停触发）: {best_model_path}")
    print(f"训练指标日志已保存至: {metrics_path}")
    print(f"本次训练输出目录: {TRAIN_CONFIG['MODEL_DIR']}")
    return final_model_path, config_path


# ==================== 仿真对比（与原始版本相同，仅需加载新模型） ====================
class AdaptiveLQRController:
    def __init__(self, model_path, config_path):
        print(f"Loading Adaptive LQR Model from {model_path}")
        # 这里需要加载包含自定义配置的 dict，因此需要 weights_only=False（显式避免未来默认变更）
        saved_data = torch.load(config_path, map_location='cpu', weights_only=False)
        self.stats = saved_data['stats']
        self.config = saved_data['config']
        mode_value = self.config.get('MODE', ControlMode.A)
        if isinstance(mode_value, str):
            self.config['MODE'] = ControlMode(mode_value)

        # 推理设备选择（可与训练保持一致）
        cfg_device = str(self.config.get('DEVICE', 'auto')).lower()
        if cfg_device in ('cuda', 'gpu') and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU.")
            cfg_device = 'cpu'
        if cfg_device == 'auto':
            cfg_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(cfg_device)

        # 根据配置构建模型（需与训练时一致），并加载权重
        time_dim = self.stats['time_mean'].shape[0]
        self.model = AdaptiveNetwork(
            mode=self.config['MODE'],
            time_dim=time_dim,
            scalar_dim=2,
            hidden_size=self.config['HIDDEN_SIZE'],
            lstm_layers=self.config.get('LSTM_LAYERS', 2),
            lstm_dropout=self.config.get('LSTM_DROPOUT', 0.3),
            use_attention=self.config.get('USE_ATTENTION', True),
            mlp_hidden=self.config.get('MLP_HIDDEN', [128, 64]),
            mlp_dropout=self.config.get('MLP_DROPOUT', 0.2),
            mode_a_alpha_range=self.config.get('MODE_A_ALPHA_RANGE', (0.5, 1.5)),
            mode_a_beta_scale=self.config.get('MODE_A_BETA_SCALE', 0.1),
            mode_d_delta_scale=self.config.get('MODE_D_DELTA_SCALE', 0.1),
            mode_d_use_tanh_bound=self.config.get('MODE_D_USE_TANH_BOUND', False),
            speed_feature_gain=self.config.get('SPEED_FEATURE_GAIN', 1.8)
        ).to(self.device)
        # 使用 weights_only=True 只加载 tensor 权重，避免 pickle 反序列化风险
        _load_adaptive_network_state(
            self.model,
            model_path,
            map_location=self.device,
            strict=True,
            load_label=f"inference model {model_path}",
        )
        self.model.eval()

        self.lqr = LQR_car(dt=cfg.VEHICLE_DT)

        self.seq_len = self.config['SEQ_LEN']
        self.use_pitch_feature = self.config.get('USE_PITCH_FEATURE', True)
        self.history = {
            'lat_error': [0.0] * self.seq_len,
            'heading_error': [0.0] * self.seq_len,
            'roll': [0.0] * self.seq_len,
            'omega': [0.0] * self.seq_len
        }
        if self.use_pitch_feature:
            self.history['pitch'] = [0.0] * self.seq_len

    def update_history(self, e_y, e_psi, omega, roll=0.0, pitch=0.0):
        self.history['lat_error'].append(e_y)
        self.history['heading_error'].append(e_psi)
        self.history['roll'].append(roll)
        if self.use_pitch_feature:
            self.history['pitch'].append(pitch)
        self.history['omega'].append(omega)

        if len(self.history['lat_error']) > self.seq_len:
            self.history['lat_error'].pop(0)
            self.history['heading_error'].pop(0)
            self.history['roll'].pop(0)
            if self.use_pitch_feature:
                self.history['pitch'].pop(0)
            self.history['omega'].pop(0)

    def get_control(self, state: State, e_y: float, e_psi: float, omega: float, roll: float | None = None, pitch: float | None = None):
        if roll is None:
            roll = float(getattr(state, 'roll', 0.0))
        if pitch is None:
            pitch = float(getattr(state, 'pitch', 0.0))

        self.update_history(e_y, e_psi, omega, roll=roll, pitch=pitch)

        self.lqr.update_car_state(state.x, state.y, state.psi, state.v)
        self.lqr.Update_A_B_matrix(cfg.VEHICLE_L)
        self.lqr.Update_Q_R_matrix(
            q11=cfg.ANDROID_LQR_Q1,
            q22=cfg.ANDROID_LQR_Q2,
            r00=cfg.ANDROID_LQR_R,
            r11=cfg.ANDROID_LQR_R11,
        )
        K, _ = self.lqr.Solve()

        delta_lqr = -(K[0, 0] * e_psi + K[0, 1] * e_y)

        # 构建时间序列（注意如果训练时用了差分，这里也需要对应调整）
        base_features = [
            self.history['lat_error'],
            self.history['heading_error'],
            self.history['roll']
        ]
        if self.use_pitch_feature:
            base_features.append(self.history['pitch'])
        base_features.append(self.history['omega'])
        base_seq = np.column_stack(base_features).astype(np.float32)

        if self.config.get('USE_DIFF_FEATURE', False):
            diff_e = np.diff(self.history['lat_error'], prepend=self.history['lat_error'][0])
            diff_theta = np.diff(self.history['heading_error'], prepend=self.history['heading_error'][0])
            diff_seq = np.column_stack([diff_e, diff_theta]).astype(np.float32)
            time_seq = np.concatenate([base_seq, diff_seq], axis=1)
        else:
            time_seq = base_seq

        scalar_feat = np.array([state.v, cfg.VEHICLE_L], dtype=np.float32)

        t_mean = self.stats['time_mean'].numpy().squeeze()
        t_std = self.stats['time_std'].numpy().squeeze()
        s_mean = self.stats['scalar_mean'].numpy().squeeze()
        s_std = self.stats['scalar_std'].numpy().squeeze()

        norm_time = torch.FloatTensor((time_seq - t_mean) / t_std).unsqueeze(0).to(self.device)
        norm_scalar = torch.FloatTensor((scalar_feat - s_mean) / s_std).unsqueeze(0).to(self.device)

        delta_final = delta_lqr

        with torch.no_grad():
            if self.config['MODE'] == ControlMode.A:
                alpha_e, beta_e, alpha_th, beta_th = self.model(norm_time, norm_scalar)

                K_psi, K_y = K[0, 0], K[0, 1]

                K_psi_new = float(alpha_th) * K_psi + float(beta_th)
                K_y_new = float(alpha_e) * K_y + float(beta_e)

                delta_final = -(K_psi_new * e_psi + K_y_new * e_y)

            elif self.config['MODE'] == ControlMode.D:
                delta_add = self.model(norm_time, norm_scalar)
                delta_final = delta_lqr + float(delta_add)

        delta_final = float(np.clip(delta_final, cfg.STEERING_LIMIT_MIN, cfg.STEERING_LIMIT_MAX))
        return delta_final, delta_lqr


def run_comparison(model_path, config_path, path_type='sine', num_runs=1):
    num_runs = int(max(1, num_runs))
    print(f"\n>>> 开始对比仿真测试 (Adaptive LQR vs Pure LQR), path={path_type}, runs={num_runs}...")

    v_ref = cfg.DEFAULT_V_REF
    dt = cfg.VEHICLE_DT
    path_type = str(path_type).lower().strip()
    run_metrics = []

    for run_idx in range(num_runs):
        if path_type == 'straight':
            path = PathGenerator.generate_straight_path(num_points=600, v_ref=v_ref, length=60.0)
        elif path_type == 'circle':
            path = PathGenerator.generate_circle_path(num_points=600, v_ref=v_ref, radius=20.0)
        elif path_type == 'lane_change':
            path = PathGenerator.generate_lane_change_path(num_points=600, v_ref=v_ref, lane_width=3.5)
        else:
            if path_type != 'sine':
                print(f"Unknown path_type '{path_type}', fallback to 'sine'.")
            path = PathGenerator.generate_sine_path(num_points=600, v_ref=v_ref, amplitude=1.0, frequency=0.3)

        # 起始点固定为该轨迹的标准起点，确保各轮一致且在线上
        init_x, init_y, init_psi = PathGenerator.get_initial_state_for_path(path_type)

        veh_lqr = VehicleModel(state=State(x=init_x, y=init_y, psi=init_psi, v=v_ref), dt=dt)
        lqr_pure = LQR_car(dt=dt)
        tracker_lqr = PathTracker()
        tracker_lqr.set_path(path)

        veh_apt = VehicleModel(state=State(x=init_x, y=init_y, psi=init_psi, v=v_ref), dt=dt)
        ctrl_apt = AdaptiveLQRController(model_path, config_path)
        tracker_apt = PathTracker()
        tracker_apt.set_path(path)

        log_lqr = {'x': [], 'y': [], 'e_y': [], 'e_psi': [], 'delta': []}
        log_apt = {'x': [], 'y': [], 'e_y': [], 'e_psi': [], 'delta': [], 'delta_lqr_pre': []}

        steps = 400
        for _ in range(steps):
            ref_lqr = path[tracker_lqr.find_nearest_point(veh_lqr.state.x, veh_lqr.state.y)[0]]
            err_lqr = veh_lqr.calc_errors(ref_lqr)

            lqr_pure.update_car_state(veh_lqr.state.x, veh_lqr.state.y, veh_lqr.state.psi, veh_lqr.state.v)
            lqr_pure.Update_A_B_matrix(cfg.VEHICLE_L)
            lqr_pure.Update_Q_R_matrix(q11=100.0, q22=100.0, r00=10.0, r11=0.01)
            k_lqr, _ = lqr_pure.Solve()
            cmd_lqr = -(k_lqr[0, 0] * err_lqr.e_psi + k_lqr[0, 1] * err_lqr.e_y)

            veh_lqr.update(Control(delta_target=cmd_lqr))

            log_lqr['x'].append(veh_lqr.state.x)
            log_lqr['y'].append(veh_lqr.state.y)
            log_lqr['e_y'].append(err_lqr.e_y)
            log_lqr['e_psi'].append(err_lqr.e_psi)
            log_lqr['delta'].append(cmd_lqr)

            ref_apt = path[tracker_apt.find_nearest_point(veh_apt.state.x, veh_apt.state.y)[0]]
            err_apt = veh_apt.calc_errors(ref_apt)

            cmd_apt, cmd_apt_lqr_pre = ctrl_apt.get_control(veh_apt.state, err_apt.e_y, err_apt.e_psi, veh_apt.state.omega)
            veh_apt.update(Control(delta_target=cmd_apt))

            log_apt['x'].append(veh_apt.state.x)
            log_apt['y'].append(veh_apt.state.y)
            log_apt['e_y'].append(err_apt.e_y)
            log_apt['e_psi'].append(err_apt.e_psi)
            log_apt['delta'].append(cmd_apt)
            log_apt['delta_lqr_pre'].append(cmd_apt_lqr_pre)

        plt.figure(figsize=(12, 12))

        plt.subplot(4, 1, 1)
        plt.plot(np.degrees(log_apt['delta_lqr_pre']), 'g', label='LQR (Uncompensated, same run)')
        plt.plot(np.degrees(log_apt['delta']), 'r', label='AI_LQR (Compensated)')
        plt.title(f"Run {run_idx + 1}/{num_runs} - Front Wheel Angle (Before vs After Compensation)")
        plt.ylabel("Angle (deg)")
        plt.legend()
        plt.grid()

        plt.subplot(4, 1, 2)
        ref_x = [p.x for p in path[:steps]]
        ref_y = [p.y for p in path[:steps]]
        plt.plot(ref_x, ref_y, 'k--', label='Reference', alpha=0.5)
        plt.plot(log_lqr['x'], log_lqr['y'], 'b', label='Pure LQR')
        plt.plot(log_apt['x'], log_apt['y'], 'r', label='Adaptive LQR')
        plt.title("Trajectory Comparison")
        plt.legend()
        plt.grid()

        plt.subplot(4, 1, 3)
        plt.plot(log_lqr['e_y'], 'b', label='Pure LQR')
        plt.plot(log_apt['e_y'], 'r', label='Adaptive LQR')
        plt.title("Lateral Error")
        plt.ylabel("Error (m)")
        plt.grid()

        plt.subplot(4, 1, 4)
        plt.plot(log_lqr['e_psi'], 'b', label='Pure LQR')
        plt.plot(log_apt['e_psi'], 'r', label='Adaptive LQR')
        plt.title("Heading Error")
        plt.ylabel("Error (rad)")
        plt.xlabel("Step")
        plt.grid()

        plt.tight_layout()
        plt.show()

        mae_y_lqr = float(np.mean(np.abs(log_lqr['e_y'])))
        mae_psi_lqr = float(np.mean(np.abs(log_lqr['e_psi'])))
        mae_y_apt = float(np.mean(np.abs(log_apt['e_y'])))
        mae_psi_apt = float(np.mean(np.abs(log_apt['e_psi'])))

        run_item = {
            'run_index': run_idx + 1,
            'initial_state': {'x': init_x, 'y': init_y, 'psi': init_psi},
            'lqr': {'mae_lateral': mae_y_lqr, 'mae_heading': mae_psi_lqr},
            'adaptive': {'mae_lateral': mae_y_apt, 'mae_heading': mae_psi_apt},
            'improvement': {
                'mae_lateral': mae_y_lqr - mae_y_apt,
                'mae_heading': mae_psi_lqr - mae_psi_apt
            }
        }
        run_metrics.append(run_item)

        print(f"\n--- 第 {run_idx + 1} 轮统计 ---")
        print(f"MAE Lateral (m): Pure={mae_y_lqr:.4f}, Adaptive={mae_y_apt:.4f}, Diff={mae_y_lqr - mae_y_apt:.4f}")
        print(f"MAE Heading (rad): Pure={mae_psi_lqr:.4f}, Adaptive={mae_psi_apt:.4f}, Diff={mae_psi_lqr - mae_psi_apt:.4f}")

    lqr_lat_arr = np.array([x['lqr']['mae_lateral'] for x in run_metrics], dtype=float)
    lqr_head_arr = np.array([x['lqr']['mae_heading'] for x in run_metrics], dtype=float)
    apt_lat_arr = np.array([x['adaptive']['mae_lateral'] for x in run_metrics], dtype=float)
    apt_head_arr = np.array([x['adaptive']['mae_heading'] for x in run_metrics], dtype=float)

    mean_lqr_lat, std_lqr_lat = float(lqr_lat_arr.mean()), float(lqr_lat_arr.std())
    mean_lqr_head, std_lqr_head = float(lqr_head_arr.mean()), float(lqr_head_arr.std())
    mean_apt_lat, std_apt_lat = float(apt_lat_arr.mean()), float(apt_lat_arr.std())
    mean_apt_head, std_apt_head = float(apt_head_arr.mean()), float(apt_head_arr.std())

    imp_lat = mean_lqr_lat - mean_apt_lat
    imp_head = mean_lqr_head - mean_apt_head
    better_lat_count = int(np.sum(apt_lat_arr < lqr_lat_arr))
    better_head_count = int(np.sum(apt_head_arr < lqr_head_arr))

    if imp_lat > 0 and imp_head > 0:
        conclusion = "综合判断：Adaptive LQR 在横向与航向误差两项上整体优于 Pure LQR，建议优先使用。"
    elif imp_lat > 0 or imp_head > 0:
        conclusion = "综合判断：Adaptive LQR 在部分指标上有提升，建议结合业务侧权重进一步评估。"
    else:
        conclusion = "综合判断：Adaptive LQR 暂未体现稳定优势，建议继续数据扩充或启用RL微调。"

    print("\n========== 综合统计 ==========")
    print(f"Runs: {num_runs}, Path: {path_type}")
    print(f"Lateral MAE  | Pure={mean_lqr_lat:.4f}±{std_lqr_lat:.4f}, Adaptive={mean_apt_lat:.4f}±{std_apt_lat:.4f}, Diff={imp_lat:.4f}")
    print(f"Heading MAE  | Pure={mean_lqr_head:.4f}±{std_lqr_head:.4f}, Adaptive={mean_apt_head:.4f}±{std_apt_head:.4f}, Diff={imp_head:.4f}")
    print(f"Adaptive 优于 Pure 次数 | Lateral: {better_lat_count}/{num_runs}, Heading: {better_head_count}/{num_runs}")
    print(conclusion)

    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.dirname(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    comparison_csv_path = os.path.join(report_dir, f"comparison_runs_{model_stem}_{report_timestamp}.csv")
    comparison_md_path = os.path.join(report_dir, f"comparison_summary_{model_stem}_{report_timestamp}.md")

    pd.DataFrame([
        {
            'run_index': item['run_index'],
            'lqr_mae_lateral': item['lqr']['mae_lateral'],
            'adaptive_mae_lateral': item['adaptive']['mae_lateral'],
            'improvement_lateral': item['improvement']['mae_lateral'],
            'lqr_mae_heading': item['lqr']['mae_heading'],
            'adaptive_mae_heading': item['adaptive']['mae_heading'],
            'improvement_heading': item['improvement']['mae_heading'],
        }
        for item in run_metrics
    ]).to_csv(comparison_csv_path, index=False)

    comparison_lines = [
        "# 对比仿真测试结果",
        f"- 模型文件: {os.path.basename(model_path)}",
        f"- 配置文件: {os.path.basename(config_path)}",
        f"- 路径类型: {path_type}",
        f"- 测试轮数: {num_runs}",
        f"- Pure LQR 横向 MAE: {mean_lqr_lat:.6f}",
        f"- Adaptive LQR 横向 MAE: {mean_apt_lat:.6f}",
        f"- Pure LQR 航向 MAE: {mean_lqr_head:.6f}",
        f"- Adaptive LQR 航向 MAE: {mean_apt_head:.6f}",
        f"- 结论: {conclusion}",
        "",
        "说明: 该文件反映闭环轨迹跟踪效果；模型拟合精度验证请优先查看 E 评估生成的 fit_validation_report_* 文件。",
    ]
    with open(comparison_md_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(comparison_lines) + '\n')
    print(f"对比测试CSV已保存至: {comparison_csv_path}")
    print(f"对比测试摘要已保存至: {comparison_md_path}")

    return {
        'path_type': path_type,
        'num_runs': num_runs,
        'runs': run_metrics,
        'lqr': {
            'mae_lateral': mean_lqr_lat,
            'mae_lateral_std': std_lqr_lat,
            'mae_heading': mean_lqr_head,
            'mae_heading_std': std_lqr_head
        },
        'adaptive': {
            'mae_lateral': mean_apt_lat,
            'mae_lateral_std': std_apt_lat,
            'mae_heading': mean_apt_head,
            'mae_heading_std': std_apt_head
        },
        'improvement': {
            'mae_lateral': imp_lat,
            'mae_heading': imp_head,
            'better_lateral_count': better_lat_count,
            'better_heading_count': better_head_count
        },
        'conclusion': conclusion
    }


def _suggest_rl_finetune(metrics):
    return rl_tuner.suggest_rl_finetune(
        metrics,
        lateral_threshold=TRAIN_CONFIG.get('RL_TRIGGER_LATERAL_MAE', 0.10)
    )


def _select_rl_trajectory(default_path='sine'):
    return rl_tuner.select_rl_trajectory(default_path=default_path)


def fine_tune_with_rl(model_path, config_path, path_type='sine'):
    return rl_tuner.fine_tune_with_rl(
        model_path=model_path,
        config_path=config_path,
        controller_cls=AdaptiveLQRController,
        train_config=TRAIN_CONFIG,
        path_type=path_type
    )


def summarize_latest_compensation_ratio(model_dir, top_k=10, diag_path=None):
    """
    汇总最新 pretrain_diagnostics_by_source_*.csv 的补偿比统计。
    ratio = abs_model_comp_mean / abs_residual_mean
    """
    if diag_path is None:
        diag_files = [
            f for f in os.listdir(model_dir)
            if f.startswith('pretrain_diagnostics_by_source_') and f.endswith('.csv')
        ]
        if not diag_files:
            print("[AUTO] 未找到 pretrain_diagnostics_by_source_*.csv，跳过补偿比汇总。")
            return None
        diag_files.sort(reverse=True)
        latest_path = os.path.join(model_dir, diag_files[0])
    else:
        latest_path = diag_path
        if not os.path.exists(latest_path):
            print(f"[AUTO] 指定诊断文件不存在: {latest_path}")
            return None

    try:
        df = pd.read_csv(latest_path)
    except Exception as e:
        print(f"[AUTO] 读取诊断文件失败: {latest_path} | {e}")
        return None

    required_cols = {'abs_model_comp_mean', 'abs_residual_mean', 'split', 'source_id'}
    if not required_cols.issubset(set(df.columns)):
        print(f"[AUTO] 诊断文件列缺失，期望列: {required_cols}")
        return None

    eps = 1e-12
    ratio = df['abs_model_comp_mean'].to_numpy(dtype=float) / np.maximum(df['abs_residual_mean'].to_numpy(dtype=float), eps)

    mean_ratio = float(np.mean(ratio))
    median_ratio = float(np.median(ratio))
    p90_ratio = float(np.percentile(ratio, 90))

    out_df = df.copy()
    out_df['comp_to_residual_ratio'] = ratio
    low_ratio_df = out_df.sort_values('comp_to_residual_ratio', ascending=True).head(max(1, int(top_k)))

    print("\n========== [AUTO] 补偿比汇总 ==========")
    print(f"诊断文件: {latest_path}")
    print(f"rows={len(out_df)} | ratio_mean={mean_ratio:.4f} | ratio_median={median_ratio:.4f} | ratio_p90={p90_ratio:.4f}")
    print(f"ratio最低 Top-{min(top_k, len(low_ratio_df))} 数据源:")
    for _, r in low_ratio_df.iterrows():
        print(
            f"  split={r['split']}, ratio={float(r['comp_to_residual_ratio']):.4f}, "
            f"|res|={float(r['abs_residual_mean']):.6f}, |comp|={float(r['abs_model_comp_mean']):.6f}, "
            f"source={r['source_id']}"
        )

    return {
        'diag_path': latest_path,
        'ratio_mean': mean_ratio,
        'ratio_median': median_ratio,
        'ratio_p90': p90_ratio,
    }


def _build_eval_datasets_by_config(saved_config, saved_stats):
    """
    使用指定模型配置重建评估数据集，用于导出诊断文件。
    """
    seq_len = int(saved_config.get('SEQ_LEN', TRAIN_CONFIG['SEQ_LEN']))
    data_root = saved_config.get('DATA_ROOT', TRAIN_CONFIG['DATA_ROOT'])

    config_keys = [
        'DATA_FILTER_ENABLED', 'INCLUDE_DIR_KEYWORDS', 'EXCLUDE_DIR_KEYWORDS',
        'INCLUDE_FILE_KEYWORDS', 'EXCLUDE_FILE_KEYWORDS',
        'USE_PITCH_FEATURE', 'USE_DIFF_FEATURE',
        'VAL_SPLIT', 'USE_SPEED_STRATIFIED_SPLIT', 'SPEED_BINS_MPS'
    ]
    backup = {k: TRAIN_CONFIG.get(k) for k in config_keys}

    for k in config_keys:
        if k in saved_config:
            TRAIN_CONFIG[k] = saved_config[k]

    try:
        samples = load_samples(data_root, seq_len=seq_len)
    finally:
        for k in config_keys:
            TRAIN_CONFIG[k] = backup[k]

    if not samples:
        print("[EVAL] 未加载到样本，无法导出诊断文件。")
        return None, None

    if len(samples) < 2:
        print("[EVAL] 样本数量不足，无法划分训练/验证用于诊断。")
        return None, None

    val_split = float(saved_config.get('VAL_SPLIT', TRAIN_CONFIG['VAL_SPLIT']))
    use_speed_stratified = bool(saved_config.get('USE_SPEED_STRATIFIED_SPLIT', TRAIN_CONFIG['USE_SPEED_STRATIFIED_SPLIT']))
    speed_bins = _get_speed_bin_edges(saved_config)

    val_size = max(1, int(len(samples) * val_split))
    if val_size >= len(samples):
        val_size = len(samples) - 1

    rng = np.random.default_rng(42)
    group_to_indices = {}
    for idx, sample in enumerate(samples):
        source_id = sample.get('source_id', 'unknown_source')
        if source_id not in group_to_indices:
            group_to_indices[source_id] = []
        group_to_indices[source_id].append(idx)

    group_keys = list(group_to_indices.keys())

    group_speed_mean = {
        g: float(np.mean([float(samples[i]['scalar'][0]) for i in idx_list]))
        for g, idx_list in group_to_indices.items()
    }
    group_sample_count = {g: len(idx_list) for g, idx_list in group_to_indices.items()}

    val_group_set = set()

    if use_speed_stratified:
        bin_to_groups = {}
        for g in group_keys:
            b = _speed_bin_id(group_speed_mean[g], speed_bins)
            if b not in bin_to_groups:
                bin_to_groups[b] = []
            bin_to_groups[b].append(g)

        for _, groups_in_bin in bin_to_groups.items():
            rng.shuffle(groups_in_bin)
            bin_total = sum(group_sample_count[g] for g in groups_in_bin)
            target_bin_val = int(round(bin_total * val_split))

            current_bin_val = 0
            for g in groups_in_bin:
                if current_bin_val >= target_bin_val:
                    break
                val_group_set.add(g)
                current_bin_val += group_sample_count[g]

        if sum(group_sample_count[g] for g in val_group_set) < val_size:
            remaining_groups = [g for g in group_keys if g not in val_group_set]
            rng.shuffle(remaining_groups)
            for g in remaining_groups:
                val_group_set.add(g)
                if sum(group_sample_count[x] for x in val_group_set) >= val_size:
                    break
    else:
        rng.shuffle(group_keys)
        current_val = 0
        for g in group_keys:
            val_group_set.add(g)
            current_val += group_sample_count[g]
            if current_val >= val_size:
                break

    val_indices = []
    for g in val_group_set:
        val_indices.extend(group_to_indices[g])

    val_index_set = set(val_indices)
    train_indices = [i for i in range(len(samples)) if i not in val_index_set]

    if len(train_indices) == 0:
        move_idx = val_indices.pop()
        train_indices.append(move_idx)
    if len(val_indices) == 0:
        move_idx = train_indices.pop()
        val_indices.append(move_idx)

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    train_dataset = ControlDataset(train_samples, stats=saved_stats, augment=False)
    val_dataset = ControlDataset(val_samples, stats=saved_stats, augment=False)
    return train_dataset, val_dataset


def evaluate_selected_model_prediction_performance(model_dir, model_files, resolve_config_for_model_name):
    """
    第4项：仅评估拟合预测性能（不重训）
    - 自选模型
    - 导出 pretrain_diagnostics 文件
    - 导出性能分析文件
    - 输出补偿比汇总
    """
    if not model_files:
        print("[EVAL] 未检测到模型文件。")
        return None

    print("\n[EVAL] 请选择用于预测性能评估的模型:")
    for idx, f in enumerate(model_files, start=1):
        print(f"  {idx}. {f}")

    sel = input("输入模型编号 (默认 1): ").strip()
    try:
        sel_idx = int(sel) - 1 if sel else 0
        sel_idx = max(0, min(sel_idx, len(model_files) - 1))
    except ValueError:
        sel_idx = 0

    selected_model_file = model_files[sel_idx]
    selected_model_path = os.path.join(model_dir, selected_model_file)
    selected_config_path = resolve_config_for_model_name(selected_model_file)

    if not os.path.exists(selected_config_path):
        print(f"[EVAL] 缺少配置文件: {selected_config_path}")
        return None

    print(f"[EVAL] 已选择模型: {selected_model_file}")

    saved_data = torch.load(selected_config_path, map_location='cpu', weights_only=False)
    saved_stats = saved_data.get('stats')
    saved_config = saved_data.get('config', {})
    mode_value = saved_config.get('MODE', ControlMode.A)
    if isinstance(mode_value, str):
        mode_value = ControlMode(mode_value)

    train_dataset, val_dataset = _build_eval_datasets_by_config(saved_config, saved_stats)
    if train_dataset is None or val_dataset is None:
        print("[EVAL] 诊断数据集构建失败。")
        return None

    time_dim = saved_stats['time_mean'].shape[0]
    eval_model = AdaptiveNetwork(
        mode=mode_value,
        time_dim=time_dim,
        scalar_dim=2,
        hidden_size=saved_config.get('HIDDEN_SIZE', TRAIN_CONFIG['HIDDEN_SIZE']),
        lstm_layers=saved_config.get('LSTM_LAYERS', 2),
        lstm_dropout=saved_config.get('LSTM_DROPOUT', 0.3),
        use_attention=saved_config.get('USE_ATTENTION', True),
        mlp_hidden=saved_config.get('MLP_HIDDEN', [128, 64]),
        mlp_dropout=saved_config.get('MLP_DROPOUT', 0.2),
        mode_a_alpha_range=saved_config.get('MODE_A_ALPHA_RANGE', (0.5, 1.5)),
        mode_a_beta_scale=saved_config.get('MODE_A_BETA_SCALE', 0.1),
        mode_d_delta_scale=saved_config.get('MODE_D_DELTA_SCALE', 0.1),
        mode_d_use_tanh_bound=saved_config.get('MODE_D_USE_TANH_BOUND', False),
        speed_feature_gain=saved_config.get('SPEED_FEATURE_GAIN', 1.8)
    )
    _load_adaptive_network_state(
        eval_model,
        selected_model_path,
        map_location='cpu',
        strict=True,
        load_label=f"evaluation model {selected_model_path}",
    )
    eval_model.eval()

    diag_path = export_pretrain_diagnostics(
        model=eval_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_dir=model_dir,
        mode=mode_value,
    )

    if not diag_path or not os.path.exists(diag_path):
        print("[EVAL] 诊断文件导出失败，无法继续性能分析。")
        return None

    diag_df = pd.read_csv(diag_path)

    required_cols = {
        'split', 'speed_mps', 'delta_opt_rad', 'delta_lqr_base_rad',
        'delta_pred_from_model_rad', 'abs_residual_rad', 'abs_model_comp_rad'
    }
    if not required_cols.issubset(set(diag_df.columns)):
        print(f"[EVAL] 诊断文件缺少必要列: {required_cols}")
        return None

    eps = 1e-12
    diag_df['abs_err_pred'] = np.abs(diag_df['delta_pred_from_model_rad'] - diag_df['delta_opt_rad'])
    diag_df['abs_err_lqr'] = np.abs(diag_df['delta_lqr_base_rad'] - diag_df['delta_opt_rad'])
    diag_df['improvement_abs'] = diag_df['abs_err_lqr'] - diag_df['abs_err_pred']
    diag_df['improvement_ratio'] = diag_df['improvement_abs'] / np.maximum(diag_df['abs_err_lqr'], eps)
    diag_df['comp_to_residual_ratio'] = diag_df['abs_model_comp_rad'] / np.maximum(diag_df['abs_residual_rad'], eps)

    def _agg(df, scope_name):
        mae_lqr = float(df['abs_err_lqr'].mean())
        mae_pred = float(df['abs_err_pred'].mean())
        imp_abs = float(df['improvement_abs'].mean())
        imp_ratio = float(df['improvement_ratio'].mean())
        comp_ratio_mean = float(df['comp_to_residual_ratio'].mean())
        comp_ratio_median = float(df['comp_to_residual_ratio'].median())
        comp_ratio_p90 = float(np.percentile(df['comp_to_residual_ratio'], 90))
        return {
            'scope': scope_name,
            'sample_count': int(len(df)),
            'mae_lqr_to_opt': mae_lqr,
            'mae_pred_to_opt': mae_pred,
            'improvement_abs': imp_abs,
            'improvement_ratio_mean': imp_ratio,
            'comp_to_residual_ratio_mean': comp_ratio_mean,
            'comp_to_residual_ratio_median': comp_ratio_median,
            'comp_to_residual_ratio_p90': comp_ratio_p90,
        }

    summary_rows = [_agg(diag_df, 'all')]
    for split_name in ['train', 'val']:
        split_df = diag_df[diag_df['split'] == split_name]
        if len(split_df) > 0:
            summary_rows.append(_agg(split_df, split_name))

    speed_bins = _get_speed_bin_edges(saved_config)
    diag_df = _attach_diagnostic_slice_columns(diag_df, speed_bins)
    speed_labels = _get_speed_bin_labels(speed_bins)
    bin_rows = []
    for split_name in ['train', 'val']:
        split_df = diag_df[diag_df['split'] == split_name]
        if len(split_df) == 0:
            continue
        for label in speed_labels:
            sub = split_df[split_df['speed_bin'] == label]
            if len(sub) == 0:
                continue
            bin_rows.append({
                'split': split_name,
                'speed_bin': str(label),
                'sample_count': int(len(sub)),
                'mae_lqr_to_opt': float(sub['abs_err_lqr'].mean()),
                'mae_pred_to_opt': float(sub['abs_err_pred'].mean()),
                'improvement_abs': float(sub['improvement_abs'].mean()),
                'improvement_ratio_mean': float(sub['improvement_ratio'].mean()),
                'comp_to_residual_ratio_mean': float(sub['comp_to_residual_ratio'].mean()),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(0, 'model_file', selected_model_file)
    summary_df.insert(1, 'config_file', os.path.basename(selected_config_path))

    for _, row in summary_df.iterrows():
        print(
            f"[EVAL] scope={row['scope']}, n={int(row['sample_count'])}, "
            f"MAE(pred->opt)={float(row['mae_pred_to_opt']):.6f}, "
            f"MAE(LQR->opt)={float(row['mae_lqr_to_opt']):.6f}, "
            f"improvement={float(row['improvement_abs']):.6f}, "
            f"comp_ratio_mean={float(row['comp_to_residual_ratio_mean']):.4f}"
        )

    print("[EVAL] 分速度段分析(仅控制台输出):")
    for item in bin_rows:
        print(
            f"  split={item['split']}, speed_bin={item['speed_bin']}, n={item['sample_count']}, "
            f"MAE(pred->opt)={item['mae_pred_to_opt']:.6f}, MAE(LQR->opt)={item['mae_lqr_to_opt']:.6f}, "
            f"improvement={item['improvement_abs']:.6f}, comp_ratio_mean={item['comp_to_residual_ratio_mean']:.4f}"
        )

    print(f"[EVAL] 诊断文件: {diag_path}")
    print("[EVAL] 性能分析文件已关闭，不再生成 performance_analysis*.csv。")

    by_source_diag_path = diag_path.replace('pretrain_diagnostics_', 'pretrain_diagnostics_by_source_')
    ratio_stats = summarize_latest_compensation_ratio(model_dir, diag_path=by_source_diag_path)

    # 读取 by_source 生成问题榜单 + 调参建议 markdown
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_stem = os.path.splitext(os.path.basename(selected_model_file))[0]
    bundle_dir = os.path.join(model_dir, f"{selected_stem}_{ts}")
    os.makedirs(bundle_dir, exist_ok=True)

    summary_csv_path = os.path.join(bundle_dir, f"fit_validation_summary_{ts}.csv")
    speed_csv_path = os.path.join(bundle_dir, f"fit_validation_speed_bins_{ts}.csv")
    report_md_path = os.path.join(bundle_dir, f"fit_validation_report_{ts}.md")
    random_source_summary_csv_path = os.path.join(bundle_dir, f"fit_validation_random_sources_{ts}.csv")

    summary_df.to_csv(summary_csv_path, index=False)
    pd.DataFrame(bin_rows).to_csv(speed_csv_path, index=False)
    random_source_exports = _export_random_source_fit_reports(diag_df, bundle_dir, sample_count=5)
    pd.DataFrame(random_source_exports).to_csv(random_source_summary_csv_path, index=False, encoding='utf-8-sig')

    overall_row = summary_df[summary_df['scope'] == 'all'].iloc[0]
    low_speed_label = speed_labels[0] if speed_labels else 'N/A'
    low_speed_rows = [item for item in bin_rows if item['speed_bin'] == low_speed_label]
    low_speed_rows.sort(key=lambda item: (item['split'] != 'val', -item['sample_count']))
    worst_speed_rows = sorted(bin_rows, key=lambda item: item['mae_pred_to_opt'], reverse=True)[:3]

    report_lines = [
        "# 拟合效果验证报告",
        f"- 模型文件: {selected_model_file}",
        f"- 配置文件: {os.path.basename(selected_config_path)}",
        f"- 总样本数: {int(overall_row['sample_count'])}",
        f"- 拟合 MAE(pred->opt): {float(overall_row['mae_pred_to_opt']):.6f}",
        f"- 基线 MAE(LQR->opt): {float(overall_row['mae_lqr_to_opt']):.6f}",
        f"- 平均提升量: {float(overall_row['improvement_abs']):.6f}",
        f"- 补偿/残差均值比: {float(overall_row['comp_to_residual_ratio_mean']):.4f}",
        "",
        "## 重点结论",
    ]

    if float(overall_row['comp_to_residual_ratio_mean']) < 0.95:
        report_lines.append(
            f"- 当前模型整体补偿仍偏保守，comp_to_residual_ratio_mean={float(overall_row['comp_to_residual_ratio_mean']):.4f}，说明输出相对最优值普遍偏低。"
        )
    else:
        report_lines.append(
            f"- 当前模型整体补偿量已接近残差尺度，comp_to_residual_ratio_mean={float(overall_row['comp_to_residual_ratio_mean']):.4f}。"
        )

    if low_speed_rows:
        report_lines.append(f"- 低速重点区间 {low_speed_label} 的拟合结果如下：")
        for item in low_speed_rows:
            report_lines.append(
                f"  {item['split']}: n={item['sample_count']}, MAE(pred->opt)={item['mae_pred_to_opt']:.6f}, "
                f"MAE(LQR->opt)={item['mae_lqr_to_opt']:.6f}, improvement={item['improvement_abs']:.6f}, "
                f"comp_ratio_mean={item['comp_to_residual_ratio_mean']:.4f}"
            )

    if worst_speed_rows:
        report_lines.append("- 当前拟合最差的速度段 Top3：")
        for item in worst_speed_rows:
            report_lines.append(
                f"  split={item['split']}, speed_bin={item['speed_bin']}, n={item['sample_count']}, "
                f"MAE(pred->opt)={item['mae_pred_to_opt']:.6f}, improvement={item['improvement_abs']:.6f}"
            )

    if random_source_exports:
        report_lines.append(f"- 已随机抽取 {len(random_source_exports)} 个原始文件输出逐点拟合明细。")
        report_lines.append(
            "  关键字段说明: actual_comp_rad=实际补偿量, model_comp_rad=模型补偿量, actual_minus_model_comp_rad=实际-模型, model_to_actual_ratio_abs=补偿占比绝对值。"
        )
        for item in random_source_exports:
            report_lines.append(
                f"  #{item['index']}: source={item['source_id']}, n={item['sample_count']}, "
                f"mean_abs_gap={item['mean_abs_comp_gap']:.6f}, mean_ratio_abs={item['mean_ratio_abs']:.4f}, "
                f"file={os.path.basename(item['csv_path'])}"
            )

    report_lines.extend([
        "",
        "## 输出文件",
        f"- 汇总CSV: {os.path.basename(summary_csv_path)}",
        f"- 分速度段CSV: {os.path.basename(speed_csv_path)}",
        f"- 随机源文件汇总CSV: {os.path.basename(random_source_summary_csv_path)}",
        f"- 诊断CSV: {os.path.basename(diag_path)}",
    ])

    with open(report_md_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(report_lines) + '\n')

    analysis_md_path = os.path.join(bundle_dir, f"by_source_analysis_{ts}.md")
    analysis_result = None
    if os.path.exists(by_source_diag_path):
        analysis_result = analyze_by_source_file(
            by_source_csv_path=by_source_diag_path,
            output_md_path=analysis_md_path,
            top_k=10
        )
        print(f"[EVAL] by_source 分析文档: {analysis_result['output_md_path']}")
    else:
        print(f"[EVAL] 未找到 by_source 诊断文件: {by_source_diag_path}")

    # 将该模型相关文件归档到同一文件夹（迁移而非复制，避免外部重复）
    files_to_copy = [
        selected_model_path,
        selected_config_path,
        diag_path,
        by_source_diag_path,
    ]
    copied_files = []
    for src in files_to_copy:
        if src and os.path.exists(src):
            dst = os.path.join(bundle_dir, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            copied_files.append(dst)

    print(f"[EVAL] 模型相关文件已归档到: {bundle_dir}")
    print(f"[EVAL] 拟合效果汇总CSV: {summary_csv_path}")
    print(f"[EVAL] 分速度段CSV: {speed_csv_path}")
    if random_source_exports:
        print(f"[EVAL] 随机源文件汇总CSV: {random_source_summary_csv_path}")
        for item in random_source_exports:
            print(f"[EVAL] 随机源文件逐点明细: {item['csv_path']}")
    print(f"[EVAL] 拟合效果报告: {report_md_path}")

    return {
        'diag_path': diag_path,
        'ratio_stats': ratio_stats,
        'analysis_md_path': analysis_md_path if analysis_result else None,
        'summary_csv_path': summary_csv_path,
        'speed_csv_path': speed_csv_path,
        'random_source_summary_csv_path': random_source_summary_csv_path,
        'random_source_detail_paths': [item['csv_path'] for item in random_source_exports],
        'report_md_path': report_md_path,
        'bundle_dir': bundle_dir,
        'copied_files': copied_files,
    }


def _export_diagnostics_for_specific_model(model_path, config_path, model_dir):
    """
    基于指定模型与配置导出诊断文件，并生成 by_source 分析文档。
    """
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        return None

    try:
        saved_data = torch.load(config_path, map_location='cpu', weights_only=False)
        saved_stats = saved_data.get('stats')
        saved_config = saved_data.get('config', {})
        mode_value = saved_config.get('MODE', ControlMode.A)
        if isinstance(mode_value, str):
            mode_value = ControlMode(mode_value)

        train_dataset, val_dataset = _build_eval_datasets_by_config(saved_config, saved_stats)
        if train_dataset is None or val_dataset is None:
            return None

        time_dim = saved_stats['time_mean'].shape[0]
        eval_model = AdaptiveNetwork(
            mode=mode_value,
            time_dim=time_dim,
            scalar_dim=2,
            hidden_size=saved_config.get('HIDDEN_SIZE', TRAIN_CONFIG['HIDDEN_SIZE']),
            lstm_layers=saved_config.get('LSTM_LAYERS', 2),
            lstm_dropout=saved_config.get('LSTM_DROPOUT', 0.3),
            use_attention=saved_config.get('USE_ATTENTION', True),
            mlp_hidden=saved_config.get('MLP_HIDDEN', [128, 64]),
            mlp_dropout=saved_config.get('MLP_DROPOUT', 0.2),
            mode_a_alpha_range=saved_config.get('MODE_A_ALPHA_RANGE', (0.5, 1.5)),
            mode_a_beta_scale=saved_config.get('MODE_A_BETA_SCALE', 0.1),
            mode_d_delta_scale=saved_config.get('MODE_D_DELTA_SCALE', 0.1),
            mode_d_use_tanh_bound=saved_config.get('MODE_D_USE_TANH_BOUND', False),
            speed_feature_gain=saved_config.get('SPEED_FEATURE_GAIN', 1.8)
        )
        _load_adaptive_network_state(
            eval_model,
            model_path,
            map_location='cpu',
            strict=True,
            load_label=f"bundle model {model_path}",
        )
        eval_model.eval()

        diag_path = export_pretrain_diagnostics(
            model=eval_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_dir=model_dir,
            mode=mode_value,
        )
        if not diag_path or not os.path.exists(diag_path):
            return None

        by_source_diag_path = diag_path.replace('pretrain_diagnostics_', 'pretrain_diagnostics_by_source_')
        analysis_md_path = None
        if os.path.exists(by_source_diag_path):
            summarize_latest_compensation_ratio(model_dir, diag_path=by_source_diag_path)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_stem = os.path.splitext(os.path.basename(model_path))[0]
            analysis_md_path = os.path.join(model_dir, f"by_source_analysis_{model_stem}_{ts}.md")
            analyze_by_source_file(
                by_source_csv_path=by_source_diag_path,
                output_md_path=analysis_md_path,
                top_k=10
            )

        return {
            'diag_path': diag_path,
            'by_source_diag_path': by_source_diag_path if os.path.exists(by_source_diag_path) else None,
            'analysis_md_path': analysis_md_path if analysis_md_path and os.path.exists(analysis_md_path) else None,
        }
    except Exception as e:
        print(f"[BUNDLE] 指定模型诊断导出失败: {e}")
        return None


def bundle_model_related_files(model_path, config_path, model_dir, ensure_diagnostics=False):
    """
    将模型、配置、诊断、分析文件归档到同一目录。
    ensure_diagnostics=True 时若缺少诊断文件将自动补导出。
    """
    if not model_path or not os.path.exists(model_path):
        print(f"[BUNDLE] 模型文件不存在: {model_path}")
        return None
    if not config_path or not os.path.exists(config_path):
        print(f"[BUNDLE] 配置文件不存在: {config_path}")
        return None

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = os.path.join(model_dir, f"{model_stem}_{ts}")
    os.makedirs(bundle_dir, exist_ok=True)

    extra = None
    if ensure_diagnostics:
        extra = _export_diagnostics_for_specific_model(model_path, config_path, model_dir)

    files_to_copy = [model_path, config_path]

    if extra:
        files_to_copy.extend([
            extra.get('diag_path'),
            extra.get('by_source_diag_path'),
            extra.get('analysis_md_path')
        ])
    # 尝试拾取近期自动生成的诊断/分析/训练指标文件
    try:
        model_mtime = os.path.getmtime(model_path)
        recent_window_sec = 40 * 60
        candidate_diag_files = []
        candidate_analysis_files = []
        candidate_metrics_files = []
        for f in os.listdir(model_dir):
            p = os.path.join(model_dir, f)
            if not os.path.isfile(p):
                continue
            if f.startswith('pretrain_diagnostics_') or f.startswith('pretrain_diagnostics_by_source_'):
                candidate_diag_files.append(p)
            elif f.startswith(f'by_source_analysis_{model_stem}_') and f.endswith('.md'):
                candidate_analysis_files.append(p)
            elif f.startswith('training_metrics_') and f.endswith('.csv'):
                candidate_metrics_files.append(p)

        # 诊断CSV：按与模型时间的接近程度筛选
        candidate_diag_files.sort(key=lambda x: abs(os.path.getmtime(x) - model_mtime))
        for p in candidate_diag_files:
            if abs(os.path.getmtime(p) - model_mtime) <= recent_window_sec:
                files_to_copy.append(p)

        # 分析Markdown：同模型前缀的文档一并归档
        candidate_analysis_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        files_to_copy.extend(candidate_analysis_files)

        # 训练指标：优先最近的一个，避免将历史日志批量迁移
        if candidate_metrics_files:
            candidate_metrics_files.sort(key=lambda x: abs(os.path.getmtime(x) - model_mtime))
            nearest_metrics = candidate_metrics_files[0]
            if abs(os.path.getmtime(nearest_metrics) - model_mtime) <= recent_window_sec:
                files_to_copy.append(nearest_metrics)
    except Exception:
        pass

    def _move_into_bundle(src, dst_dir):
        if not src or not os.path.exists(src):
            return None
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.abspath(src) == os.path.abspath(dst):
            return dst
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        return dst

    copied_files = []
    copied_name_set = set()
    for src in files_to_copy:
        if not src or not os.path.exists(src):
            continue
        base_name = os.path.basename(src)
        if base_name in copied_name_set:
            continue
        moved_path = _move_into_bundle(src, bundle_dir)
        if moved_path:
            copied_files.append(moved_path)
            copied_name_set.add(base_name)

    print(f"[BUNDLE] 已归档到: {bundle_dir}")
    print(f"[BUNDLE] 文件数量: {len(copied_files)}")
    return {
        'bundle_dir': bundle_dir,
        'copied_files': copied_files,
    }