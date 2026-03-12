"""
配置文件
"""

# ==================== 车辆参数 ====================
VEHICLE_L = 2.36                     # 轴距 (m)
VEHICLE_V_MIN = 0.1                   # 最小速度 (m/s)
VEHICLE_V_MAX = 8.0                   # 最大速度 (m/s)
VEHICLE_V_STEP = 0.1                   # 速度步长 (m/s)
VEHICLE_DT = 0.1                       # 仿真时间步长 (s)

# ==================== 转向系统参数 ====================
STEERING_TAU = 0.1                     # 一阶惯性时间常数 (s)
STEERING_LIMIT_MIN = -0.5236            # 最小转角 -30° (rad)
STEERING_LIMIT_MAX = 0.5236             # 最大转角 +30° (rad)

# ==================== 历史状态参数 ====================
HISTORY_LENGTH = 20                     # 历史状态序列长度
FILTER_WINDOW = 10                       # 滤波窗口长度

# ==================== 路径参数 ====================
PATH_STRAIGHT_LENGTH = 50.0              # 直线路径长度 (m)
PATH_SINE_AMPLITUDE = 1.0                 # 正弦路径振幅 (m)
PATH_SINE_FREQUENCY = 0.5                 # 正弦路径频率 (rad/m)
PATH_CIRCLE_RADIUS = 1.0                  # 圆形路径半径 (m)
PATH_LANE_CHANGE_WIDTH = 0.5                # 换道路径宽度 (m)
PATH_NUM_POINTS = 500                       # 路径点数

# ==================== 默认仿真速度 ====================
DEFAULT_V_REF = 2.0                        # 默认参考速度 (m/s)

# ==================== 噪声参数 (设为限幅值，sigma = limit / 3) ====================
NOISE_CONTROL_LIMIT = 0.005                 # 控制噪声限幅 (0.035 rad ≈ 0.28°)
NOISE_POSITION_LIMIT = 0.005                 # 位置噪声限幅 (5mm)
NOISE_HEADING_LIMIT = 0.00375                # 航向噪声限幅 (0.2°)
NOISE_VELOCITY_LIMIT = 0.125                # 速度噪声限幅 (0.45km/h)

# ==================== 可视化参数 ====================
VIS_LINE_WIDTH = 1.5
VIS_ALPHA = 0.7
VIS_COLORS = ['r', 'g', 'b', 'm', 'c', 'orange']
VIS_FIG_SIZE = (12, 10)