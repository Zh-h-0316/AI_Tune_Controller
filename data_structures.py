"""
数据结构 - 包含所有必需的类和工具函数
"""
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class State:
    """车辆状态"""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 2.0
    delta_actual: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    omega: float = 0.0

@dataclass
class Control:
    """控制输入"""
    delta_target: float = 0.0
    delta_actual: float = 0.0

@dataclass
class ReferencePoint:
    """参考路径点"""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 2.0
    curvature: float = 0.0

@dataclass
class ErrorState:
    """误差状态"""
    e_y: float = 0.0
    e_psi: float = 0.0
    e_y_rate: float = 0.0
    e_psi_rate: float = 0.0

@dataclass
class HistoryState:
    """历史状态容器（用于数据驱动）"""
    e_y_history: List[float]
    e_psi_history: List[float]
    e_y_rate_filtered: float
    e_psi_rate_filtered: float
    v: float
    L: float

@dataclass
class Car_State:
    """LQR使用的车辆状态（兼容原C++代码）"""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 0.0
    rol: float = 0.0
    dpsi: float = 0.0
    err: float = 0.0

@dataclass
class Car_data:
    """LQR计算结果"""
    car_K0: float = 0.0
    car_K1: float = 0.0
    car_U0: float = 0.0

# ---------- 角度处理工具 ----------
def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return float(angle)

def rotation_matrix(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def moving_average_filter(data: List[float], window_size: int) -> float:
    if len(data) < window_size:
        return float(np.mean(data)) if data else 0.0
    return float(np.mean(data[-window_size:]))

def calculate_rate_filtered(data_history: List[float], window_size: int, dt: float) -> float:
    if len(data_history) < 2:
        return 0.0
    recent_data = data_history[-window_size:] if len(data_history) >= window_size else data_history
    t = np.arange(len(recent_data)) * dt
    A = np.vstack([t, np.ones(len(t))]).T
    slope, _ = np.linalg.lstsq(A, recent_data, rcond=None)[0]
    return float(slope)