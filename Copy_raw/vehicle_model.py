"""
车辆模型 - 保留运动学模型和惯性环节，移除与MPC/PPO相关的线性化等未使用函数
"""
import numpy as np
from typing import List, Optional, Tuple
from data_structures import State, Control, ReferencePoint, ErrorState, HistoryState
from data_structures import normalize_angle, rotation_matrix, calculate_rate_filtered
import Config_Para as cfg

class VehicleModel:
    def __init__(self, state: Optional[State] = None, L: float = cfg.VEHICLE_L,
                 dt: float = cfg.VEHICLE_DT, v_ref: float = 2.0):
        if state:
            self.state = state
        else:
            self.state = State(v=v_ref)
        self.L = L
        self.dt = dt
        self.v_ref = v_ref
        self.tau = cfg.STEERING_TAU
        self.history: List[State] = []
        self.e_y_history: List[float] = []
        self.e_psi_history: List[float] = []
        self.delta_actual = 0.0

    def steering_inertia(self, delta_target: float) -> float:
        a = np.exp(-self.dt / self.tau)
        b = 1 - a
        self.delta_actual = a * self.delta_actual + b * delta_target
        return self.delta_actual


    def update(self, control: Control) -> State:
        delta_actual = self.steering_inertia(control.delta_target)

        # 添加控制噪声（高斯噪声，Limit为限幅值，近似3*sigma）
        noise_steer = np.random.normal(0, cfg.NOISE_CONTROL_LIMIT / 3.0)
        noise_steer = np.clip(noise_steer, -cfg.NOISE_CONTROL_LIMIT, cfg.NOISE_CONTROL_LIMIT)
        delta_with_noise = delta_actual + noise_steer

        delta_with_noise = np.clip(delta_with_noise, cfg.STEERING_LIMIT_MIN, cfg.STEERING_LIMIT_MAX)

        self.state.x += self.v_ref * np.cos(self.state.psi) * self.dt
        self.state.y += self.v_ref * np.sin(self.state.psi) * self.dt
        self.state.psi += self.v_ref / self.L * np.tan(delta_with_noise) * self.dt
        self.state.psi = normalize_angle(self.state.psi)
        self.state.v = self.v_ref
        self.state.delta_actual = delta_with_noise

        # 添加观测噪声（高斯噪声）
        n_pos_std = cfg.NOISE_POSITION_LIMIT / 3.0
        n_psi_std = cfg.NOISE_HEADING_LIMIT / 3.0
        n_vel_std = cfg.NOISE_VELOCITY_LIMIT / 3.0

        n_x = np.clip(np.random.normal(0, n_pos_std), -cfg.NOISE_POSITION_LIMIT, cfg.NOISE_POSITION_LIMIT)
        n_y = np.clip(np.random.normal(0, n_pos_std), -cfg.NOISE_POSITION_LIMIT, cfg.NOISE_POSITION_LIMIT)
        n_psi = np.clip(np.random.normal(0, n_psi_std), -cfg.NOISE_HEADING_LIMIT, cfg.NOISE_HEADING_LIMIT)
        n_v = np.clip(np.random.normal(0, n_vel_std), -cfg.NOISE_VELOCITY_LIMIT, cfg.NOISE_VELOCITY_LIMIT)

        self.state.x += n_x
        self.state.y += n_y
        self.state.psi += n_psi
        self.state.psi = normalize_angle(self.state.psi)
        self.state.v = max(0.5, self.v_ref + n_v)

        self.history.append(State(
            self.state.x, self.state.y, self.state.psi, self.state.v, self.state.delta_actual
        ))
        control.delta_actual = delta_with_noise
        return State(
            x=self.state.x,
            y=self.state.y,
            psi=self.state.psi,
            v=self.state.v,
            delta_actual=self.state.delta_actual
        )


    def calc_errors(self, ref_point: ReferencePoint) -> ErrorState:
        dx = self.state.x - ref_point.x
        dy = self.state.y - ref_point.y
        R = rotation_matrix(-ref_point.psi)
        path_coords = R @ np.array([dx, dy])
        e_y = path_coords[1]
        e_psi = normalize_angle(self.state.psi - ref_point.psi)

        self.e_y_history.append(e_y)
        self.e_psi_history.append(e_psi)
        max_history = cfg.HISTORY_LENGTH + cfg.FILTER_WINDOW
        if len(self.e_y_history) > max_history:
            self.e_y_history.pop(0)
            self.e_psi_history.pop(0)

        e_y_rate = calculate_rate_filtered(self.e_y_history, cfg.FILTER_WINDOW, self.dt)
        e_psi_rate = calculate_rate_filtered(self.e_psi_history, cfg.FILTER_WINDOW, self.dt)
        return ErrorState(e_y, e_psi, e_y_rate, e_psi_rate)

    def get_history_state(self) -> HistoryState:
        e_y_recent = self.e_y_history[-cfg.HISTORY_LENGTH:] if len(self.e_y_history) >= cfg.HISTORY_LENGTH else self.e_y_history
        e_psi_recent = self.e_psi_history[-cfg.HISTORY_LENGTH:] if len(self.e_psi_history) >= cfg.HISTORY_LENGTH else self.e_psi_history
        if len(e_y_recent) < cfg.HISTORY_LENGTH:
            e_y_recent = [0.0] * (cfg.HISTORY_LENGTH - len(e_y_recent)) + e_y_recent
            e_psi_recent = [0.0] * (cfg.HISTORY_LENGTH - len(e_psi_recent)) + e_psi_recent
        e_y_rate_filtered = calculate_rate_filtered(self.e_y_history, cfg.FILTER_WINDOW, self.dt)
        e_psi_rate_filtered = calculate_rate_filtered(self.e_psi_history, cfg.FILTER_WINDOW, self.dt)
        return HistoryState(
            e_y_history=e_y_recent,
            e_psi_history=e_psi_recent,
            e_y_rate_filtered=e_y_rate_filtered,
            e_psi_rate_filtered=e_psi_rate_filtered,
            v=self.state.v,
            L=self.L
        )

    def reset(self, state: Optional[State] = None, v_ref: Optional[float] = None) -> None:
        if state:
            self.state = state
        else:
            if v_ref is None:
                v_options = np.arange(cfg.VEHICLE_V_MIN, cfg.VEHICLE_V_MAX + cfg.VEHICLE_V_STEP, cfg.VEHICLE_V_STEP)
                self.v_ref = np.random.choice(v_options)
            else:
                self.v_ref = v_ref
            self.state = State(v=self.v_ref)
        self.history = []
        self.e_y_history = []
        self.e_psi_history = []
        self.delta_actual = 0.0


class PathTracker:
    def __init__(self, lookahead_distance: float = 5.0):
        self.lookahead_distance = lookahead_distance
        self.path: List[ReferencePoint] = []
        self.current_target_idx = 0

    def set_path(self, path: List[ReferencePoint]) -> None:
        self.path = path
        self.current_target_idx = 0

    def find_nearest_point(self, x: float, y: float) -> Tuple[int, float]:
        if not self.path:
            return 0, 0.0
        min_dist = float('inf')
        nearest_idx = 0
        for i, p in enumerate(self.path):
            dx = x - p.x
            dy = y - p.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist