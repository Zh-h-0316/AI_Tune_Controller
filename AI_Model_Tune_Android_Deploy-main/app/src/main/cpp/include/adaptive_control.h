/*
 * 自适应神经网络控制模块 - 接口声明
 *
 * 功能：整合RKNN模型推理与LQR控制算法，输出自适应补偿后的前轮转角。
 * 模型输入：10帧历史时序特征 + 2维标量特征
 * 模型输出：4维原始线性头；Mode A 解释为增益修正参数，Mode D 解释为直接控制补偿量
 * 控制流程：LQR计算基准控制量 → 网络推理 → 按模式做后处理 → 合成最终控制量
 *
 * Created by AI_Tune integration, 2026/03
 */

#ifndef ADAPTIVE_CONTROL_H
#define ADAPTIVE_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rknn_api.h"

// 用于存储自适应控制的中间调试变量
enum {
    ADAPTIVE_CONTROL_MODE_A = 0,
    ADAPTIVE_CONTROL_MODE_D = 1,
};

typedef struct {
    int control_mode;
    float raw_alpha_e;
    float raw_beta_e;
    float raw_alpha_th;
    float raw_beta_th;
    float alpha_e;
    float beta_e;
    float alpha_th;
    float beta_th;
    float lqr_k_e;
    float k_e_new;
    float lqr_k_th;
    float k_th_new;
    float raw_delta_add;
    float delta_add;
    float delta_lqr;
    float delta_final;
} AdaptiveDebugData;

/**
 * @brief 初始化自适应控制的RKNN模型.
 * @param model_path RKNN模型文件的路径.
 * @return 0: 成功, <0: 失败.
 */
int adaptive_control_init(const char* model_path);

/**
 * @brief 使用自适应神经网络和LQR计算最终的前轮转角.
 *
 * 内部维护10帧历史滑动窗口，调用RKNN推理获取原始输出，
 * 再按 Mode A / Mode D 解释输出并结合LQR基线控制量计算最终控制量。
 *
 * 时序特征(每帧5维): e_y, e_psi, roll, pitch, omega
 * 标量特征(2维): v, L
 * 输出(4维原始值 → sigmoid/tanh后处理): alpha_e, beta_e, alpha_theta, beta_theta
 *
 * @param e_y        横向误差 (m)
 * @param e_psi      航向误差 (rad)
 * @param roll       横滚角 (rad)
 * @param pitch      俯仰角 (rad)
 * @param omega      航向角速度 (rad/s)
 * @param v          车速绝对值 (m/s)
 * @param L          轴距 (m), 需与训练时单位一致
 * @param lqr_k_e    LQR横向误差增益 K[0,1]
 * @param lqr_k_th   LQR航向误差增益 K[0,0]
 * @param base_delta LQR基础前轮转角 (rad)
 * @param control_mode 0=Mode A, 1=Mode D
 * @return 经神经网络自适应补偿后的最终前轮转角 (rad)
 */
double adaptive_control_compute(
     double e_y, double e_psi, double roll, double pitch, double omega,
     double v, double L, double lqr_k_e, double lqr_k_th, double base_delta,
     int control_mode,
    AdaptiveDebugData* debug_data);

/**
 * @brief 释放自适应控制模块占用的资源.
 */
void adaptive_control_release(void);

/**
 * 检查模型是否已加载且历史窗口已填满，可以进行推理
 * @return 1=就绪, 0=未就绪
 */
int adaptive_control_is_ready(void);

#ifdef __cplusplus
}
#endif

#endif /* ADAPTIVE_CONTROL_H */
