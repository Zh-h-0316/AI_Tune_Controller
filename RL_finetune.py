import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import Config_Para as cfg
from data_structures import State, Control
from vehicle_model import VehicleModel, PathTracker
from path_generator import PathGenerator


class RLValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def build_path_for_type(path_type, v_ref):
    path_type = str(path_type).lower().strip()
    if path_type == "straight":
        return PathGenerator.generate_straight_path(num_points=600, v_ref=v_ref, length=60.0)
    if path_type == "circle":
        return PathGenerator.generate_circle_path(num_points=600, v_ref=v_ref, radius=20.0)
    if path_type == "lane_change":
        return PathGenerator.generate_lane_change_path(num_points=600, v_ref=v_ref, lane_width=3.5)
    return PathGenerator.generate_sine_path(num_points=600, v_ref=v_ref, amplitude=1.0, frequency=0.3)


def suggest_rl_finetune(metrics, lateral_threshold=0.10):
    if not metrics:
        return True
    mae_lat = float(metrics.get("adaptive", {}).get("mae_lateral", 1.0))
    imp_lat = float(metrics.get("improvement", {}).get("mae_lateral", -1.0))
    return (mae_lat > float(lateral_threshold)) or (imp_lat <= 0.0)


def select_rl_trajectory(default_path="sine"):
    print("请选择RL微调使用的训练轨迹:")
    print("1. sine (正弦路径)")
    print("2. straight (直线路径)")
    print("3. circle (圆形路径)")
    print("4. lane_change (变道路径)")
    path_types = {"1": "sine", "2": "straight", "3": "circle", "4": "lane_change"}
    tip_default = default_path if default_path in path_types.values() else "sine"
    choice = input(f"输入选择 (1-4，默认 {tip_default}): ").strip()
    return path_types.get(choice, tip_default)


def fine_tune_with_rl(model_path, config_path, controller_cls, train_config, path_type="sine"):
    print("\n>>> 开始RL微调（PPO）...")
    ctrl = controller_cls(model_path, config_path)

    mode_value = ctrl.config.get("MODE", "A")
    mode_tag = mode_value.value if hasattr(mode_value, "value") else str(mode_value)
    if mode_tag not in ("A", "D"):
        print(f"当前模式 {mode_tag} 暂不支持RL微调（仅支持A/D）。")
        return None, None

    device = ctrl.device
    actor = ctrl.model
    actor.train()

    action_dim = 4 if mode_tag == "A" else 1
    log_std = nn.Parameter(torch.full((action_dim,), float(train_config.get("RL_INIT_LOG_STD", -1.2)), device=device))
    obs_dim = ctrl.seq_len * int(ctrl.stats["time_mean"].shape[0]) + 2
    critic = RLValueNet(obs_dim).to(device)

    optim_all = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()) + [log_std],
        lr=float(train_config.get("RL_LR", 3e-4))
    )

    gamma = float(train_config.get("RL_GAMMA", 0.99))
    gae_lambda = float(train_config.get("RL_GAE_LAMBDA", 0.95))
    clip_eps = float(train_config.get("RL_CLIP_EPS", 0.2))
    entropy_coef = float(train_config.get("RL_ENTROPY_COEF", 0.01))
    value_coef = float(train_config.get("RL_VALUE_COEF", 0.5))
    max_grad_norm = float(train_config.get("RL_MAX_GRAD_NORM", 1.0))

    episodes = int(train_config.get("RL_EPISODES", 20))
    steps_per_ep = int(train_config.get("RL_STEPS_PER_EPISODE", 300))
    ppo_epochs = int(train_config.get("RL_PPO_EPOCHS", 5))
    minibatch = int(train_config.get("RL_MINIBATCH_SIZE", 128))

    early_stop_patience = int(train_config.get("RL_EARLY_STOP_PATIENCE", 6))
    early_stop_min_delta = float(train_config.get("RL_EARLY_STOP_MIN_DELTA", 1e-3))

    def policy_mean(norm_time_batch, norm_scalar_batch):
        if mode_tag == "A":
            alpha_e, beta_e, alpha_th, beta_th = actor(norm_time_batch, norm_scalar_batch)
            return torch.stack([alpha_e, beta_e, alpha_th, beta_th], dim=1)
        delta_add = actor(norm_time_batch, norm_scalar_batch).view(-1, 1)
        return delta_add

    best_reward = -1e18
    best_actor_state = None
    best_log_std = None
    no_improve_rounds = 0

    for ep in range(1, episodes + 1):
        # 采样阶段使用eval模式，避免BatchNorm在batch=1时报错
        actor.eval()

        v_ref = cfg.DEFAULT_V_REF
        path = build_path_for_type(path_type, v_ref)
        veh = VehicleModel(state=State(x=0, y=0, psi=0, v=v_ref), dt=cfg.VEHICLE_DT)
        tracker = PathTracker()
        tracker.set_path(path)

        ctrl.history = {
            "lat_error": [0.0] * ctrl.seq_len,
            "heading_error": [0.0] * ctrl.seq_len,
            "roll": [0.0] * ctrl.seq_len,
            "omega": [0.0] * ctrl.seq_len
        }
        if ctrl.use_pitch_feature:
            ctrl.history["pitch"] = [0.0] * ctrl.seq_len

        t_mean = ctrl.stats["time_mean"].numpy().squeeze()
        t_std = ctrl.stats["time_std"].numpy().squeeze()
        s_mean = ctrl.stats["scalar_mean"].numpy().squeeze()
        s_std = ctrl.stats["scalar_std"].numpy().squeeze()

        buf_time = []
        buf_scalar = []
        buf_obs = []
        buf_action = []
        buf_logprob = []
        buf_value = []
        buf_reward = []
        buf_done = []

        total_reward = 0.0
        delta_prev = 0.0
        abs_lat_err_sum = 0.0

        for _step in range(steps_per_ep):
            ref = path[tracker.find_nearest_point(veh.state.x, veh.state.y)[0]]
            err = veh.calc_errors(ref)
            ctrl.update_history(err.e_y, err.e_psi, err.e_psi_rate)

            base_features = [
                ctrl.history["lat_error"],
                ctrl.history["heading_error"],
                ctrl.history["roll"]
            ]
            if ctrl.use_pitch_feature:
                base_features.append(ctrl.history["pitch"])
            base_features.append(ctrl.history["omega"])
            base_seq = np.column_stack(base_features).astype(np.float32)

            if ctrl.config.get("USE_DIFF_FEATURE", False):
                diff_e = np.diff(ctrl.history["lat_error"], prepend=ctrl.history["lat_error"][0])
                diff_theta = np.diff(ctrl.history["heading_error"], prepend=ctrl.history["heading_error"][0])
                diff_seq = np.column_stack([diff_e, diff_theta]).astype(np.float32)
                time_seq = np.concatenate([base_seq, diff_seq], axis=1)
            else:
                time_seq = base_seq

            scalar_feat = np.array([veh.state.v, cfg.VEHICLE_L], dtype=np.float32)
            n_time_np = (time_seq - t_mean) / t_std
            n_scalar_np = (scalar_feat - s_mean) / s_std

            n_time = torch.FloatTensor(n_time_np).unsqueeze(0).to(device)
            n_scalar = torch.FloatTensor(n_scalar_np).unsqueeze(0).to(device)
            obs_vec = torch.cat([n_time.view(1, -1), n_scalar], dim=1)

            with torch.no_grad():
                mean = policy_mean(n_time, n_scalar)
                std = log_std.exp().unsqueeze(0)
                dist = Normal(mean, std)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(dim=1)
                value = critic(obs_vec)

            ctrl.lqr.update_car_state(veh.state.x, veh.state.y, veh.state.psi, veh.state.v)
            ctrl.lqr.Update_A_B_matrix(cfg.VEHICLE_L)
            ctrl.lqr.Update_Q_R_matrix(q11=100.0, q22=100.0, r00=10.0, r11=0.01)
            k_lqr, _ = ctrl.lqr.Solve()
            k_psi = float(k_lqr[0, 0])
            k_y = float(k_lqr[0, 1])
            delta_lqr = -(k_psi * err.e_psi + k_y * err.e_y)

            if mode_tag == "A":
                a = action.squeeze(0)
                k_y_new = float(a[0]) * k_y + float(a[1])
                k_psi_new = float(a[2]) * k_psi + float(a[3])
                delta_cmd = -(k_psi_new * err.e_psi + k_y_new * err.e_y)
            else:
                delta_cmd = delta_lqr + float(action.squeeze(0)[0])

            delta_cmd = float(np.clip(delta_cmd, cfg.STEERING_LIMIT_MIN, cfg.STEERING_LIMIT_MAX))
            veh.update(Control(delta_target=delta_cmd))

            ref_next = path[tracker.find_nearest_point(veh.state.x, veh.state.y)[0]]
            err_next = veh.calc_errors(ref_next)
            reward = -(
                err_next.e_y ** 2
                + 0.5 * (err_next.e_psi ** 2)
                + 0.02 * (delta_cmd ** 2)
                + 0.02 * ((delta_cmd - delta_prev) ** 2)
            )
            delta_prev = delta_cmd
            done = (abs(err_next.e_y) > 3.0)

            total_reward += reward
            abs_lat_err_sum += abs(err_next.e_y)

            buf_time.append(torch.FloatTensor(n_time_np))
            buf_scalar.append(torch.FloatTensor(n_scalar_np))
            buf_obs.append(obs_vec.squeeze(0).detach().cpu())
            buf_action.append(action.squeeze(0).detach().cpu())
            buf_logprob.append(logprob.item())
            buf_value.append(value.item())
            buf_reward.append(float(reward))
            buf_done.append(float(done))

            if done:
                break

        if len(buf_reward) == 0:
            continue

        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(buf_reward))):
            mask = 1.0 - buf_done[t]
            delta = buf_reward[t] + gamma * next_value * mask - buf_value[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + buf_value[t])
            next_value = buf_value[t]

        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        old_logprob_t = torch.FloatTensor(buf_logprob).to(device)
        action_t = torch.stack(buf_action).to(device)
        time_t = torch.stack(buf_time).to(device)
        scalar_t = torch.stack(buf_scalar).to(device)
        obs_t = torch.stack(buf_obs).to(device)

        # PPO参数更新阶段切换为train模式
        actor.train()
        total_n = action_t.shape[0]
        for _ in range(ppo_epochs):
            idx = torch.randperm(total_n, device=device)
            for start in range(0, total_n, minibatch):
                mb = idx[start:start + minibatch]
                if mb.shape[0] < 2:
                    # BatchNorm要求每个通道至少2个样本
                    continue
                mb_time = time_t[mb]
                mb_scalar = scalar_t[mb]
                mb_obs = obs_t[mb]
                mb_act = action_t[mb]
                mb_old_logp = old_logprob_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]

                mean_new = policy_mean(mb_time, mb_scalar)
                std_new = log_std.exp().unsqueeze(0).expand_as(mean_new)
                dist_new = Normal(mean_new, std_new)
                logp_new = dist_new.log_prob(mb_act).sum(dim=1)
                entropy = dist_new.entropy().sum(dim=1).mean()

                ratio = torch.exp(logp_new - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = critic(mb_obs)
                value_loss = F.mse_loss(value_pred, mb_ret)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optim_all.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()) + [log_std],
                    max_grad_norm
                )
                optim_all.step()

        # 回到eval用于下一轮采样
        actor.eval()

        avg_abs_lat = abs_lat_err_sum / max(1, len(buf_reward))
        print(
            f"RL Episode {ep}/{episodes} | reward={total_reward:.3f} | "
            f"avg|e_y|={avg_abs_lat:.4f} | steps={len(buf_reward)}"
        )

        if total_reward > (best_reward + early_stop_min_delta):
            best_reward = total_reward
            best_actor_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            best_log_std = log_std.detach().cpu().clone()
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= early_stop_patience:
                print(
                    f"RL早停触发: 连续{no_improve_rounds}轮提升 < {early_stop_min_delta}，"
                    f"在episode {ep}提前结束。"
                )
                break

    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)
        log_std.data = best_log_std.to(device)

    actor.eval()
    rl_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_model = os.path.join(train_config["MODEL_DIR"], f"RL_adaptive_net_{rl_ts}.pth")
    out_config = os.path.join(train_config["MODEL_DIR"], f"RL_config_{rl_ts}.pt")

    torch.save(actor.state_dict(), out_model)
    rl_cfg = dict(ctrl.config)
    if hasattr(rl_cfg.get("MODE"), "value"):
        rl_cfg["MODE"] = rl_cfg["MODE"].value
    rl_cfg["RL_FINETUNED"] = True
    rl_cfg["RL_SOURCE_MODEL"] = os.path.basename(model_path)
    rl_cfg["RL_BEST_EPISODE_REWARD"] = float(best_reward)
    rl_cfg["RL_LOG_STD"] = [float(x) for x in log_std.detach().cpu().numpy().tolist()]
    rl_cfg["RL_TRAIN_PATH_TYPE"] = str(path_type)

    torch.save({"stats": ctrl.stats, "config": rl_cfg}, out_config)

    print(f"RL微调完成，模型保存至: {out_model}")
    print(f"RL微调配置保存至: {out_config}")

    # 生成部署用合并checkpoint
    try:
        import importlib.util
        base_dir = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(base_dir, "Adaptive Network.py")
        spec = importlib.util.spec_from_file_location("adaptive_network_merged", module_path)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        deploy_path = _mod.save_deployment_checkpoint(
            model=actor,
            stats=ctrl.stats,
            config=rl_cfg,
            model_dir=train_config["MODEL_DIR"],
            tag="RL_",
            copy_to_conversion=True
        )
        print(f"✅ RL部署合并文件已保存至: {deploy_path}")
    except Exception as e:
        print(f"Warning: RL部署合并文件生成失败: {e}")

    return out_model, out_config
