import os
from enum import Enum
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import Config_Para as cfg
from data_structures import State, Control
from vehicle_model import VehicleModel, PathTracker
from path_generator import PathGenerator
from LQR_ratio import LQR_car


# ==================== 配置与枚举 ====================
class ControlMode(Enum):
    A = 'A'  # Gain scheduling (Alpha * K + Beta)
    B = 'B'  # Gain scaling (Alpha * K)
    C = 'C'  # Gain bias (K + Beta)
    D = 'D'  # Direct control modification (u_LQR + u_net)


TRAIN_CONFIG = {
    'SEQ_LEN': 10,
    'BATCH_SIZE': 64,
    'EPOCHS': 1,
    'LR': 1e-3,
    'HIDDEN_SIZE': 64,
    'MODE': ControlMode.A,
    'DATA_ROOT': r"D:\Key_Tasks\RL_Control\Deep_RL Adaptive Control\LQR_Tune\Excellent_Data\Process_Data",
    'MODEL_DIR': r"D:\Key_Tasks\RL_Control\Deep_RL Adaptive Control\LQR_Tune\LQR_NN\models",
    'DATA_FILTER_ENABLED': False,          # 是否启用数据过滤
    'INCLUDE_DIR_KEYWORDS': [],            # 仅保留路径中包含任一关键词的csv_data目录（如['flatland']）
    'EXCLUDE_DIR_KEYWORDS': [],            # 排除路径中包含任一关键词的csv_data目录
    'INCLUDE_FILE_KEYWORDS': [],           # 仅保留文件名包含任一关键词的csv
    'EXCLUDE_FILE_KEYWORDS': [],           # 排除文件名包含任一关键词的csv

    # 新增训练配置
    'LSTM_LAYERS': 2,               # LSTM层数
    'LSTM_DROPOUT': 0.3,             # LSTM层间dropout（仅当layers>1时有效）
    'USE_ATTENTION': True,           # 是否使用时间注意力
    'MLP_HIDDEN': [128, 64],         # 融合MLP各层维度
    'MLP_DROPOUT': 0.2,              # MLP dropout率
    'WEIGHT_DECAY': 1e-4,            # 权重衰减
    'GRAD_CLIP_NORM': 1.0,           # 梯度裁剪阈值
    'HUBER_BETA': 0.03,              # SmoothL1(Huber) beta
    'SMOOTH_LOSS_WEIGHT': 0.2,       # 控制量平滑损失权重
    'STEER_LIMIT_LOSS_WEIGHT': 0.1,  # 转角越界惩罚权重
    'RATE_LOSS_WEIGHT': 0.05,        # 控制变化率惩罚权重
    'DATA_AUG_NOISE': 0.01,          # 数据增强噪声标准差（0表示关闭）
    'VAL_SPLIT': 0.2,                 # 验证集比例
    'USE_SPEED_STRATIFIED_SPLIT': True,  # 是否按速度区间分层划分（与Group Split结合）
    'SPEED_BINS_MPS': [1.5, 2.5, 3.5],   # 速度分层边界(m/s): [0,1.5), [1.5,2.5), [2.5,3.5), [3.5,+inf)
    'EARLY_STOP_PATIENCE': 10,        # 早停耐心值
    'USE_PITCH_FEATURE': True,        # 是否使用俯仰角及其历史序列作为时序输入
    'USE_DIFF_FEATURE': False,        # 是否在时间序列中加入差分特征（会改变输入维度）
}

if not os.path.exists(TRAIN_CONFIG['MODEL_DIR']):
    os.makedirs(TRAIN_CONFIG['MODEL_DIR'])


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

    print(f"Total samples loaded: {len(samples)}")
    return samples


class ControlDataset(Dataset):
    def __init__(self, samples, stats=None, augment=False):
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
            self.wheelbase[idx]
        )


# ==================== 网络模型（改进版） ====================
class AdaptiveNetwork(nn.Module):
    def __init__(self, mode: ControlMode, time_dim=5, scalar_dim=2, hidden_size=64,
                 lstm_layers=2, lstm_dropout=0.3, use_attention=True,
                 mlp_hidden=[128, 64], mlp_dropout=0.2):
        super().__init__()
        self.mode = mode
        self.use_attention = use_attention
        self.hidden_size = hidden_size

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

        # 标量特征网络
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
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
        scal_feat = self.scalar_net(scalar)  # (batch, 32)

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
            alpha_e = 0.5 + torch.sigmoid(alpha_e_raw)  # 范围[0.5, 1.5]
            beta_e = 0.1 * torch.tanh(beta_e_raw)       # 范围[-0.1, 0.1]
            alpha_th = 0.5 + torch.sigmoid(alpha_th_raw)
            beta_th = 0.1 * torch.tanh(beta_th_raw)
            return alpha_e, beta_e, alpha_th, beta_th

        elif self.mode == ControlMode.D:
            delta_add = mlp_out[:, 0] * 0.1  # 范围受tanh约束，但此处不限制，可加tanh
            # 也可加tanh限制，例如：
            # delta_add = torch.tanh(mlp_out[:, 0]) * 0.1
            return delta_add

        else:
            # 其他模式暂不实现详细改进，返回原始输出
            return mlp_out


# ==================== 训练流程（改进版） ====================
def train_network(resume_model_path=None):
    print(">>> 开始训练自适应网络（改进版）...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

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

    def speed_bin_id(speed_mps):
        # np.digitize 返回 [0..len(bins)]，右开区间行为更直观
        return int(np.digitize(speed_mps, TRAIN_CONFIG['SPEED_BINS_MPS'], right=False))

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
            b = speed_bin_id(group_speed_mean[g])
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
        counts = [0] * (len(TRAIN_CONFIG['SPEED_BINS_MPS']) + 1)
        for idx in index_list:
            v = float(samples[idx]['scalar'][0])
            counts[speed_bin_id(v)] += 1
        return counts

    train_bin_counts = bin_count_report(train_indices)
    val_bin_counts = bin_count_report(val_indices)
    print(f"Split -> train: {len(train_indices)}, val: {len(val_indices)}, groups: {len(group_keys)}")
    print(f"Speed bins (m/s edges={TRAIN_CONFIG['SPEED_BINS_MPS']}) train={train_bin_counts}, val={val_bin_counts}")

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    # 仅使用训练集统计量做归一化，避免验证信息泄漏
    train_dataset = ControlDataset(train_samples, augment=True)
    val_dataset = ControlDataset(val_samples, stats=train_dataset.stats, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['BATCH_SIZE'], shuffle=False)

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
        mlp_dropout=TRAIN_CONFIG['MLP_DROPOUT']
    ).to(device)

    if resume_model_path:
        try:
            checkpoint = torch.load(resume_model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"Resume training from checkpoint: {resume_model_path}")
        except Exception as e:
            print(f"Warning: failed to load resume checkpoint {resume_model_path}, fallback to fresh training. Error: {e}")

    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['LR'], weight_decay=TRAIN_CONFIG['WEIGHT_DECAY'])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['EPOCHS'])
    criterion = nn.SmoothL1Loss(beta=TRAIN_CONFIG['HUBER_BETA'])
    speed_bins_tensor = torch.tensor(TRAIN_CONFIG['SPEED_BINS_MPS'], dtype=torch.float32, device=device)
    num_speed_bins = len(TRAIN_CONFIG['SPEED_BINS_MPS']) + 1
    steer_min = float(cfg.STEERING_LIMIT_MIN)
    steer_max = float(cfg.STEERING_LIMIT_MAX)

    time_mean_device = train_dataset.stats['time_mean'].to(device)
    time_std_device = train_dataset.stats['time_std'].to(device)

    print(f"Training for {TRAIN_CONFIG['EPOCHS']} epochs...")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = None
    epoch_metrics = []
    k_e_base = 0.5
    k_th_base = 2.0

    for epoch in range(TRAIN_CONFIG['EPOCHS']):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_abs_sum = 0.0
        train_count = 0
        train_bin_abs_sum = [0.0] * num_speed_bins
        train_bin_count = [0] * num_speed_bins
        for batch in train_loader:
            norm_time, norm_scalar, raw_scalar, e, theta, delta_opt, _wheelbase = [x.to(device) for x in batch]

            if TRAIN_CONFIG['MODE'] == ControlMode.A:
                alpha_e, beta_e, alpha_th, beta_th = model(norm_time, norm_scalar)

                k_e_final = alpha_e * k_e_base + beta_e
                k_th_final = alpha_th * k_th_base + beta_th

                delta_pred = -k_e_final * e - k_th_final * theta

            elif TRAIN_CONFIG['MODE'] == ControlMode.D:
                delta_add = model(norm_time, norm_scalar)
                delta_base = -k_e_base * e - k_th_base * theta
                delta_pred = delta_base + delta_add
            else:
                delta_pred = -k_e_base * e - k_th_base * theta

            # 转角限幅一致性：用限幅后的预测与标签计算主损失
            delta_pred_clip = torch.clamp(delta_pred, min=steer_min, max=steer_max)
            delta_opt_clip = torch.clamp(delta_opt, min=steer_min, max=steer_max)
            loss_track = criterion(delta_pred_clip, delta_opt_clip)

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
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['GRAD_CLIP_NORM'])
            optimizer.step()

            total_train_loss += loss.item() * len(e)

            abs_err = torch.abs(delta_pred_clip.detach() - delta_opt_clip)
            train_abs_sum += abs_err.sum().item()
            train_count += abs_err.numel()

            speed_vals = raw_scalar[:, 0].contiguous()
            speed_bin_ids = torch.bucketize(speed_vals, speed_bins_tensor, right=False)
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
                norm_time, norm_scalar, raw_scalar, e, theta, delta_opt, _wheelbase = [x.to(device) for x in batch]

                if TRAIN_CONFIG['MODE'] == ControlMode.A:
                    alpha_e, beta_e, alpha_th, beta_th = model(norm_time, norm_scalar)
                    k_e_final = alpha_e * k_e_base + beta_e
                    k_th_final = alpha_th * k_th_base + beta_th
                    delta_pred = -k_e_final * e - k_th_final * theta
                elif TRAIN_CONFIG['MODE'] == ControlMode.D:
                    delta_add = model(norm_time, norm_scalar)
                    delta_base = -k_e_base * e - k_th_base * theta
                    delta_pred = delta_base + delta_add
                else:
                    delta_pred = -k_e_base * e - k_th_base * theta

                delta_pred_clip = torch.clamp(delta_pred, min=steer_min, max=steer_max)
                delta_opt_clip = torch.clamp(delta_opt, min=steer_min, max=steer_max)

                loss_track = criterion(delta_pred_clip, delta_opt_clip)
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
                )
                total_val_loss += loss.item() * len(e)

                abs_err = torch.abs(delta_pred_clip - delta_opt_clip)
                val_abs_sum += abs_err.sum().item()
                val_count += abs_err.numel()

                speed_vals = raw_scalar[:, 0].contiguous()
                speed_bin_ids = torch.bucketize(speed_vals, speed_bins_tensor, right=False)
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
            # 保存最佳模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"best_adaptive_net_{timestamp}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (val loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['EARLY_STOP_PATIENCE']:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 保存最终模型和配置（可选）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"adaptive_net_{timestamp}.pth")
    config_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"config_{timestamp}.pt")
    metrics_path = os.path.join(TRAIN_CONFIG['MODEL_DIR'], f"training_metrics_{timestamp}.csv")

    # 如果早停保存了最佳模型，则加载最佳模型再保存为最终？这里简单保存当前模型
    torch.save(model.state_dict(), final_model_path)
    serializable_config = dict(TRAIN_CONFIG)
    if isinstance(serializable_config.get('MODE'), ControlMode):
        serializable_config['MODE'] = serializable_config['MODE'].value

    torch.save({
        'stats': train_dataset.stats,
        'config': serializable_config
    }, config_path)
    pd.DataFrame(epoch_metrics).to_csv(metrics_path, index=False)

    print(f"最终模型已保存至: {final_model_path}")
    print(f"最佳模型（若早停触发）: {best_model_path}")
    print(f"训练指标日志已保存至: {metrics_path}")
    return final_model_path, config_path


# ==================== 仿真对比（与原始版本相同，仅需加载新模型） ====================
class AdaptiveLQRController:
    def __init__(self, model_path, config_path):
        self.device = torch.device('cpu')

        print(f"Loading Adaptive LQR Model from {model_path}")
        saved_data = torch.load(config_path, map_location=self.device)
        self.stats = saved_data['stats']
        self.config = saved_data['config']
        mode_value = self.config.get('MODE', ControlMode.A)
        if isinstance(mode_value, str):
            self.config['MODE'] = ControlMode(mode_value)

        # 确定时间维度
        time_dim = self.stats['time_mean'].shape[0]
        # 根据配置构建模型（需与训练时一致）
        self.model = AdaptiveNetwork(
            mode=self.config['MODE'],
            time_dim=time_dim,
            scalar_dim=2,
            hidden_size=self.config['HIDDEN_SIZE'],
            lstm_layers=self.config.get('LSTM_LAYERS', 2),
            lstm_dropout=self.config.get('LSTM_DROPOUT', 0.3),
            use_attention=self.config.get('USE_ATTENTION', True),
            mlp_hidden=self.config.get('MLP_HIDDEN', [128, 64]),
            mlp_dropout=self.config.get('MLP_DROPOUT', 0.2)
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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

    def update_history(self, e_y, e_psi, omega, pitch=0.0):
        self.history['lat_error'].append(e_y)
        self.history['heading_error'].append(e_psi)
        self.history['roll'].append(0.0)
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

    def get_control(self, state: State, e_y: float, e_psi: float, omega: float):
        self.update_history(e_y, e_psi, omega)

        self.lqr.update_car_state(state.x, state.y, state.psi, state.v)
        self.lqr.Update_A_B_matrix(cfg.VEHICLE_L)
        self.lqr.Update_Q_R_matrix(q11=100.0, q22=100.0, r00=10.0, r11=0.01, heading=1)
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

        norm_time = torch.FloatTensor((time_seq - t_mean) / t_std).unsqueeze(0)
        norm_scalar = torch.FloatTensor((scalar_feat - s_mean) / s_std).unsqueeze(0)

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


def run_comparison(model_path, config_path, path_type='sine'):
    print(f"\n>>> 开始对比仿真测试 (Adaptive LQR vs Pure LQR), path={path_type}...")

    v_ref = cfg.DEFAULT_V_REF
    dt = cfg.VEHICLE_DT
    path_type = str(path_type).lower().strip()
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

    veh_lqr = VehicleModel(state=State(x=0, y=0, psi=0, v=v_ref), dt=dt)
    lqr_pure = LQR_car(dt=dt)
    tracker_lqr = PathTracker()
    tracker_lqr.set_path(path)

    veh_apt = VehicleModel(state=State(x=0, y=0, psi=0, v=v_ref), dt=dt)
    ctrl_apt = AdaptiveLQRController(model_path, config_path)
    tracker_apt = PathTracker()
    tracker_apt.set_path(path)

    log_lqr = {'x': [], 'y': [], 'e_y': [], 'e_psi': [], 'delta': []}
    log_apt = {'x': [], 'y': [], 'e_y': [], 'e_psi': [], 'delta': []}

    steps = 400
    for _ in range(steps):
        ref_lqr = path[tracker_lqr.find_nearest_point(veh_lqr.state.x, veh_lqr.state.y)[0]]
        err_lqr = veh_lqr.calc_errors(ref_lqr)

        lqr_pure.update_car_state(veh_lqr.state.x, veh_lqr.state.y, veh_lqr.state.psi, veh_lqr.state.v)
        lqr_pure.Update_A_B_matrix(cfg.VEHICLE_L)
        lqr_pure.Update_Q_R_matrix(q11=100.0, q22=100.0, r00=10.0, r11=0.01, heading=1)
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

        cmd_apt, _ = ctrl_apt.get_control(veh_apt.state, err_apt.e_y, err_apt.e_psi, err_apt.e_psi_rate)

        veh_apt.update(Control(delta_target=cmd_apt))

        log_apt['x'].append(veh_apt.state.x)
        log_apt['y'].append(veh_apt.state.y)
        log_apt['e_y'].append(err_apt.e_y)
        log_apt['e_psi'].append(err_apt.e_psi)
        log_apt['delta'].append(cmd_apt)

    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    ref_x = [p.x for p in path[:steps]]
    ref_y = [p.y for p in path[:steps]]
    plt.plot(ref_x, ref_y, 'k--', label='Reference', alpha=0.5)
    plt.plot(log_lqr['x'], log_lqr['y'], 'b', label='Pure LQR')
    plt.plot(log_apt['x'], log_apt['y'], 'r', label='Adaptive LQR')
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(log_lqr['e_y'], 'b', label='Pure LQR')
    plt.plot(log_apt['e_y'], 'r', label='Adaptive LQR')
    plt.title("Lateral Error")
    plt.ylabel("Error (m)")
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(log_lqr['e_psi'], 'b', label='Pure LQR')
    plt.plot(log_apt['e_psi'], 'r', label='Adaptive LQR')
    plt.title("Heading Error")
    plt.ylabel("Error (rad)")
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(np.degrees(log_lqr['delta']), 'b', label='Pure LQR')
    plt.plot(np.degrees(log_apt['delta']), 'r', label='Adaptive LQR')
    plt.title("Steering Control")
    plt.ylabel("Angle (deg)")
    plt.xlabel("Step")
    plt.grid()

    plt.tight_layout()
    plt.show()

    mae_y_lqr = np.mean(np.abs(log_lqr['e_y']))
    mae_psi_lqr = np.mean(np.abs(log_lqr['e_psi']))

    mae_y_apt = np.mean(np.abs(log_apt['e_y']))
    mae_psi_apt = np.mean(np.abs(log_apt['e_psi']))

    print("\n========== 结果统计 ==========")
    print(f"{'Metric':<15} | {'Pure LQR':<12} | {'Adaptive LQR':<12} | {'Diff':<10}")
    print("-" * 55)
    print(f"{'MAE Lateral (m)':<15} | {mae_y_lqr:.4f}       | {mae_y_apt:.4f}           | {(mae_y_lqr - mae_y_apt):.4f}")
    print(f"{'MAE Heading (rad)':<15} | {mae_psi_lqr:.4f}       | {mae_psi_apt:.4f}           | {(mae_psi_lqr - mae_psi_apt):.4f}")


if __name__ == "__main__":
    # 训练
    model_path, config_path = train_network()
    if model_path:
        # 仿真对比
        run_comparison(model_path, config_path)