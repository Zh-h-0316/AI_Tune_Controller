"""
路径生成器
"""
import numpy as np
from typing import List
from data_structures import ReferencePoint, normalize_angle

class PathGenerator:
    @staticmethod
    def generate_straight_path(length: float = 50.0, num_points: int = 500, v_ref: float = 2.0) -> List[ReferencePoint]:
        points = []
        for i in range(num_points):
            x = i * length / num_points
            points.append(ReferencePoint(x=x, y=0.0, psi=0.0, v=v_ref, curvature=0.0))
        return points

    @staticmethod
    def generate_sine_path(amplitude: float = 8.0, frequency: float = 0.05,
                          num_points: int = 500, v_ref: float = 2.0) -> List[ReferencePoint]:
        points = []
        for i in range(num_points):
            x = i * 0.1
            y = amplitude * np.sin(frequency * x)
            psi = np.arctan(amplitude * frequency * np.cos(frequency * x))
            psi = normalize_angle(psi)
            curvature = (amplitude * frequency**2 * np.sin(frequency * x)) / \
                       (1 + (amplitude * frequency * np.cos(frequency * x))**2)**(3/2)
            points.append(ReferencePoint(x=x, y=y, psi=psi, v=v_ref, curvature=curvature))
        return points

    @staticmethod
    def generate_circle_path(radius: float = 10.0, num_points: int = 500, v_ref: float = 2.0) -> List[ReferencePoint]:
        points = []
        for i in range(num_points):
            theta = 2 * np.pi * i / num_points
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            psi = theta + np.pi / 2
            psi = normalize_angle(psi)
            curvature = 1.0 / radius
            points.append(ReferencePoint(x=x, y=y, psi=psi, v=v_ref, curvature=curvature))
        start_idx = num_points // 4
        points = points[start_idx:] + points[:start_idx]
        return points

    @staticmethod
    def generate_lane_change_path(lane_width: float = 3.5, num_points: int = 500,
                                 v_ref: float = 2.0) -> List[ReferencePoint]:
        points = []
        for i in range(num_points):
            x = i * 0.1
            if x < 20:
                y, psi, curvature = 0.0, 0.0, 0.0
            elif x < 40:
                t = (x - 20) / 20
                y = lane_width * (10 * t**3 - 15 * t**4 + 6 * t**5)
                dy = lane_width * (30 * t**2 - 60 * t**3 + 30 * t**4) / 20
                d2y = lane_width * (60 * t - 180 * t**2 + 120 * t**3) / 400
                psi = np.arctan(dy)
                curvature = d2y / (1 + dy**2)**(3/2)
            else:
                y, psi, curvature = lane_width, 0.0, 0.0
            points.append(ReferencePoint(x=x, y=y, psi=psi, v=v_ref, curvature=curvature))
        return points

    @staticmethod
    def get_initial_state_for_path(path_type: str, **kwargs) -> tuple:
        if path_type == 'straight':
            return (0.0, 0.0, 0.0)
        elif path_type == 'sine':
            return (0.0, 0.0, 0.0)
        elif path_type == 'circle':
            radius = kwargs.get('radius', 10.0)
            return (radius, 0.0, np.pi/2)
        elif path_type == 'lane_change':
            return (0.0, 0.0, 0.0)
        else:
            return (0.0, 0.0, 0.0)