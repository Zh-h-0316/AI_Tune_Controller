"""
LQR横向路径跟踪仿真主程序 - 单次运行
参数从 Config_Para 读取，轨迹图自适应铺满画面
"""
import numpy as np
import matplotlib.pyplot as plt

import Config_Para as cfg
from data_structures import State, Control
from path_generator import PathGenerator
from vehicle_model import VehicleModel, PathTracker
from LQR_ratio import LQR_car

# ==================== 仿真设置（可在此修改） ====================
PATH_TYPE = 'sine'          # 可选：'straight', 'sine', 'circle', 'lane_change'
V_REF = cfg.DEFAULT_V_REF   # 参考速度 (m/s)，从配置文件读取
# ==============================================================

def run_lqr_simulation(path_type=PATH_TYPE, v_ref=V_REF, plot=True):
    dt = cfg.VEHICLE_DT
    total_time = 20.0
    num_steps = int(total_time / dt)
    L = cfg.VEHICLE_L

    # 生成路径（参数全部从配置文件读取）
    if path_type == 'straight':
        path = PathGenerator.generate_straight_path(
            length=cfg.PATH_STRAIGHT_LENGTH,
            num_points=cfg.PATH_NUM_POINTS,
            v_ref=v_ref
        )
        init_state = State(x=0.0, y=0.0, psi=0.0, v=v_ref)
    elif path_type == 'sine':
        path = PathGenerator.generate_sine_path(
            amplitude=cfg.PATH_SINE_AMPLITUDE,
            frequency=cfg.PATH_SINE_FREQUENCY,
            num_points=cfg.PATH_NUM_POINTS,
            v_ref=v_ref
        )
        init_state = State(x=0.0, y=0.0, psi=0.0, v=v_ref)
    elif path_type == 'circle':
        path = PathGenerator.generate_circle_path(
            radius=cfg.PATH_CIRCLE_RADIUS,
            num_points=cfg.PATH_NUM_POINTS,
            v_ref=v_ref
        )
        init_state = State(x=cfg.PATH_CIRCLE_RADIUS, y=0.0, psi=np.pi/2, v=v_ref)
    elif path_type == 'lane_change':
        path = PathGenerator.generate_lane_change_path(
            lane_width=cfg.PATH_LANE_CHANGE_WIDTH,
            num_points=cfg.PATH_NUM_POINTS,
            v_ref=v_ref
        )
        init_state = State(x=0.0, y=0.0, psi=0.0, v=v_ref)
    else:
        raise ValueError(f"Unknown path type: {path_type}")

    # 初始化车辆和跟踪器
    vehicle = VehicleModel(state=init_state, L=L, dt=dt, v_ref=v_ref)
    tracker = PathTracker(lookahead_distance=2.0)
    tracker.set_path(path)

    # 初始化LQR
    lqr = LQR_car(max_num_iteration=700, tolerance=1e-6, dt=dt)
    lqr.update_car_state(init_state.x, init_state.y, init_state.psi, init_state.v)
    lqr.Update_A_B_matrix(L)
    lqr.Update_Q_R_matrix(
        q11=cfg.ANDROID_LQR_Q1,
        q22=cfg.ANDROID_LQR_Q2,
        r00=cfg.ANDROID_LQR_R,
        r11=cfg.ANDROID_LQR_R11,
    )

    # 数据记录
    time, states, errors, heading_errors, controls, actual_controls, ref_points = [], [], [], [], [], [], []

    for step in range(num_steps):
        t = step * dt
        nearest_idx, _ = tracker.find_nearest_point(vehicle.state.x, vehicle.state.y)
        ref = path[nearest_idx]

        err_state = vehicle.calc_errors(ref)
        e_y, e_psi = err_state.e_y, err_state.e_psi

        lqr.update_car_state(vehicle.state.x, vehicle.state.y, vehicle.state.psi, vehicle.state.v)
        lqr.Update_A_B_matrix(L)
        lqr.Update_Q_R_matrix(
            q11=cfg.ANDROID_LQR_Q1,
            q22=cfg.ANDROID_LQR_Q2,
            r00=cfg.ANDROID_LQR_R,
            r11=cfg.ANDROID_LQR_R11,
        )
        delta_target = lqr.CALC(np.array([[e_psi], [e_y]]))

        control = Control(delta_target=delta_target)
        updated_state = vehicle.update(control)

        time.append(t)
        states.append(updated_state)
        errors.append(e_y)
        heading_errors.append(e_psi)
        controls.append(delta_target)
        actual_controls.append(updated_state.delta_actual)
        ref_points.append(ref)


    results = {
        'time': np.array(time),
        'states': states,
        'errors': np.array(errors),
        'heading_errors': np.array(heading_errors),
        'controls': np.array(controls),
        'actual_controls': np.array(actual_controls),
        'ref_points': ref_points,
        'path_type': path_type,
        'v_ref': v_ref
    }

    if plot:
        plot_lqr_results(results)
    return results

def plot_lqr_results(results):
    time = results['time']
    states = results['states']
    errors = results['errors']
    heading_errors = results['heading_errors']
    controls = results['controls']
    actual_controls = results['actual_controls']
    path_type = results['path_type']
    v_ref = results['v_ref']

    traj_x = [s.x for s in states]
    traj_y = [s.y for s in states]
    ref_x = [p.x for p in results['ref_points']]
    ref_y = [p.y for p in results['ref_points']]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 轨迹图 - 紧凑显示，纵轴自动适应
    ax = axes[0,0]
    ax.plot(ref_x, ref_y, 'b--', lw=2, alpha=0.5, label='Reference')
    # Use markers to make points visible even if trajectory is weird
    ax.plot(traj_x, traj_y, 'r-', lw=1.5, label='Vehicle', alpha=0.8) 

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{path_type.title()} Path (v={v_ref:.1f} m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.margins(0.05)           # 设置5%边距，让图形铺满画面
    ax.autoscale_view()        # 自动调整范围

    # 2. 横向误差
    axes[0,1].plot(time, errors, 'g-', lw=1.5)
    axes[0,1].axhline(0, color='k', ls='--', alpha=0.5)
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Lateral Error (m)')
    axes[0,1].set_title('Lateral Error')
    axes[0,1].grid(True, alpha=0.3)

    # 3. 航向误差
    axes[0,2].plot(time, heading_errors*57.3, 'm-', lw=1.5)
    axes[0,2].axhline(0, color='k', ls='--', alpha=0.5)
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Heading Error (rad)')
    axes[0,2].set_title('Heading Error')
    axes[0,2].grid(True, alpha=0.3)

    # 4. 控制量
    axes[1,0].plot(time, np.rad2deg(controls), 'b-', alpha=0.7, label='Target')
    axes[1,0].plot(time, np.rad2deg(actual_controls), 'r--', alpha=0.7, label='Actual')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Steering Angle (deg)')
    axes[1,0].set_title('Control Input')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 5. 误差直方图
    axes[1,1].hist(errors, bins=30, color='g', alpha=0.7, edgecolor='black')
    axes[1,1].axvline(0, color='k', ls='--', alpha=0.5)
    axes[1,1].set_xlabel('Lateral Error (m)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Error Distribution')

    # 6. 统计信息
    axes[1,2].axis('off')
    mean_err = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))
    rms_err = np.sqrt(np.mean(errors**2))
    text = (f"Statistics:\n"
            f"Mean |error|: {mean_err:.4f} m\n"
            f"Max |error|: {max_err:.4f} m\n"
            f"RMS error: {rms_err:.4f} m\n"
            f"Steps: {len(time)}")
    axes[1,2].text(0.1, 0.5, text, transform=axes[1,2].transAxes,
                   fontsize=12, va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    print("\n=== LQR Simulation Results ===")
    print(f"Path type: {path_type}, v_ref: {v_ref:.2f} m/s")
    print(f"Mean |error|: {mean_err:.4f} m, Max |error|: {max_err:.4f} m, RMS: {rms_err:.4f} m")

if __name__ == "__main__":
    run_lqr_simulation('straight', v_ref=1.0)
    # for path in ['straight', 'sine', 'circle', 'lane_change']:
    #     run_lqr_simulation(path, v_ref=2.0)