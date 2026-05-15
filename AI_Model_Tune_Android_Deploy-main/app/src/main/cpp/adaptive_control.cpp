/*
 * 自适应神经网络控制模块 - 实现
 *
 * 整合RKNN模型推理与LQR控制，通过Mode A增益修正或Mode D直接补偿实现自适应控制。
 * - 管理RKNN模型生命周期（加载/释放）
 * - 维护10帧滑动历史窗口（每帧5维时序特征）
 * - 执行推理并对原始输出做sigmoid/tanh后处理
 * - 计算修正后的LQR增益并输出最终控制量
 *
 * Created by AI_Tune integration, 2026/03
 */

#include "adaptive_control.h"
#include "adaptive_net_rknn.h"
#include "android_log.h"
#include <cmath>
#include <cstring>

// ======================== Mode A 后处理参数 ========================
// 对应训练配置 TRAIN_CONFIG 中的 MODE_A_ALPHA_RANGE 和 MODE_A_BETA_SCALE
// 若训练侧修改了这些参数，部署时需同步更新此处
static const float MODE_A_ALPHA_MIN  = 0.3f;
static const float MODE_A_ALPHA_MAX  = 1.8f;
static const float MODE_A_ALPHA_SPAN = MODE_A_ALPHA_MAX - MODE_A_ALPHA_MIN;  // 1.5f
static const float MODE_A_BETA_SCALE = 0.25f;
static const float MODE_D_DELTA_SCALE = 0.30f;

// ======================== 模块内部状态 ========================
static adaptive_net_rknn_context* s_rknn_ctx = nullptr;
static int s_model_loaded = 0;

// 时序特征历史缓冲区 [TIME_STEPS × TIME_DIM] = [10 × 5] = 50 floats
static float s_history[ADAPTIVE_NET_TIME_STEPS * ADAPTIVE_NET_TIME_DIM];
static int   s_history_count = 0;   // 已累积的有效帧数

// ======================== 工具函数 ========================
static inline double fast_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// ======================== 公开接口实现 ========================

int adaptive_control_init(const char* model_path) {
    if (model_path == nullptr) {
        LOGE("adaptive_control_init: model_path is null");
        return -1;
    }

    // 先释放已有上下文（支持重复初始化）
    adaptive_control_release();

    int ret = adaptive_net_rknn_create(model_path, &s_rknn_ctx);
    if (ret != 0) {
        LOGE("RKNN model load failed (ret=%d), path=%s", ret, model_path);
        s_rknn_ctx = nullptr;
        return ret;
    }

    // 查询并打印模型信息
    adaptive_net_rknn_info info;
    if (adaptive_net_rknn_query(s_rknn_ctx, &info) == 0) {
        LOGI("RKNN model loaded: inputs=%d, outputs=%d, api=%s, driver=%s",
             info.input_count, info.output_count,
             info.api_version, info.driver_version);
    }

    s_model_loaded = 1;
    s_history_count = 0;
    memset(s_history, 0, sizeof(s_history));

    LOGI("Adaptive control initialized, model: %s", model_path);
    return 0;
}

void adaptive_control_release(void) {
    if (s_rknn_ctx != nullptr) {
        adaptive_net_rknn_destroy(s_rknn_ctx);
        s_rknn_ctx = nullptr;
    }
    s_model_loaded = 0;
    s_history_count = 0;
}

int adaptive_control_is_ready(void) {
    return s_model_loaded && (s_history_count >= ADAPTIVE_NET_TIME_STEPS);
}

double adaptive_control_compute(
    double e_y, double e_psi, double roll, double pitch, double omega,
    double v, double L,
    double lqr_k_e, double lqr_k_th, double base_delta,
    int control_mode,
    AdaptiveDebugData* debug_data)
{
    if (debug_data != nullptr) {
        memset(debug_data, 0, sizeof(*debug_data));
        debug_data->control_mode = control_mode;
        debug_data->delta_lqr = (float)base_delta;
        debug_data->delta_final = (float)base_delta;
    }

    // ---- 1. 更新历史缓冲区（滑动窗口） ----
    // 窗口已满时，整体前移一帧，丢弃最早帧
    if (s_history_count >= ADAPTIVE_NET_TIME_STEPS) {
        memmove(s_history,
                s_history + ADAPTIVE_NET_TIME_DIM,
                (ADAPTIVE_NET_TIME_STEPS - 1) * ADAPTIVE_NET_TIME_DIM * sizeof(float));
    }

    // 确定当前帧写入位置
    int write_idx = (s_history_count < ADAPTIVE_NET_TIME_STEPS)
                    ? s_history_count
                    : (ADAPTIVE_NET_TIME_STEPS - 1);

    int offset = write_idx * ADAPTIVE_NET_TIME_DIM;
    s_history[offset + 0] = (float)e_y;     // 横向误差 (m)
    s_history[offset + 1] = (float)e_psi;   // 航向误差 (rad)
    s_history[offset + 2] = (float)roll;    // 横滚角 (rad)
    s_history[offset + 3] = (float)pitch;   // 俯仰角 (rad)
    s_history[offset + 4] = (float)omega;   // 航向角速度 (rad/s)

    if (s_history_count < ADAPTIVE_NET_TIME_STEPS) {
        s_history_count++;
        // 窗口填充进度日志：当前已填充帧数 / 总帧数
        LOGI("History window filling: %d/%d | frame[%d]: e_y=%.4f e_psi=%.4f roll=%.4f pitch=%.4f omega=%.4f",
             s_history_count, ADAPTIVE_NET_TIME_STEPS, write_idx,
             s_history[offset + 0], s_history[offset + 1], s_history[offset + 2],
             s_history[offset + 3], s_history[offset + 4]);
        if (s_history_count == ADAPTIVE_NET_TIME_STEPS) {
            LOGI("History window FULL (%d/%d), NN inference enabled",
                 s_history_count, ADAPTIVE_NET_TIME_STEPS);
        }
    } else {
        // 窗口已满，滑动更新中
        LOGI("History window FULL (sliding) %d/%d | frame[%d]: e_y=%.4f e_psi=%.4f roll=%.4f pitch=%.4f omega=%.4f",
             s_history_count, ADAPTIVE_NET_TIME_STEPS, write_idx,
             s_history[offset + 0], s_history[offset + 1], s_history[offset + 2],
             s_history[offset + 3], s_history[offset + 4]);
    }

    // ---- 2. 若模型未就绪或历史窗口未满，返回LQR基础控制量 ----
    if (!s_model_loaded || s_history_count < ADAPTIVE_NET_TIME_STEPS) {
        return base_delta;
    }

    // ---- 3. 准备标量特征 [车速, 轴距] ----
    float scalar[ADAPTIVE_NET_SCALAR_DIM];
    scalar[0] = (float)v;
    scalar[1] = (float)L;

    // ---- 4. RKNN推理（内部自动做归一化） ----
    float output[ADAPTIVE_NET_OUTPUT_DIM];
    int ret = adaptive_net_rknn_infer_raw(s_rknn_ctx, s_history, scalar, output);
    if (ret != 0) {
        LOGE("RKNN inference failed (ret=%d)", ret);
        return base_delta;
    }

    double raw_alpha_e  = (double)output[0];
    double raw_beta_e   = (double)output[1];
    double raw_alpha_th = (double)output[2];
    double raw_beta_th  = (double)output[3];

    if (control_mode == ADAPTIVE_CONTROL_MODE_D) {
        double raw_delta_add = raw_alpha_e;
        double delta_add = raw_delta_add * MODE_D_DELTA_SCALE;
        double delta_final = base_delta + delta_add;

        if (debug_data != nullptr) {
            debug_data->raw_delta_add = (float)raw_delta_add;
            debug_data->delta_add = (float)delta_add;
            debug_data->delta_final = (float)delta_final;
        }

        LOGI("NN Mode D raw_delta=%.4f scale=%.3f delta_add=%.5f d_final=%.5f d_lqr=%.5f",
             raw_delta_add, MODE_D_DELTA_SCALE, delta_add, delta_final, base_delta);

        if(v <= 0.3 || abs(e_y) >= 0.03 || abs(e_psi) >= 0.04){
            if (debug_data != nullptr) {
                debug_data->delta_final = (float)base_delta;
            }
            LOGI("border change.");
            return base_delta;
        }
        if (debug_data != nullptr) {
            debug_data->delta_final = (float)delta_final;
        }
        return delta_final;
    }

    // ---- 5. Mode A 后处理：sigmoid/tanh 约束输出范围 ----
    // RKNN/ONNX导出的是原始线性层输出，需要与训练代码一致的激活函数
    // alpha ∈ [ALPHA_MIN, ALPHA_MAX],  beta ∈ [-BETA_SCALE, BETA_SCALE]
    double alpha_e  = MODE_A_ALPHA_MIN + MODE_A_ALPHA_SPAN * fast_sigmoid(raw_alpha_e);
    double beta_e   = MODE_A_BETA_SCALE * tanh(raw_beta_e);
    double alpha_th = MODE_A_ALPHA_MIN + MODE_A_ALPHA_SPAN * fast_sigmoid(raw_alpha_th);
    double beta_th  = MODE_A_BETA_SCALE * tanh(raw_beta_th);

    // ---- 6. 自适应增益修正 ----
    //   k_e'     = alpha_e  × k_e  + beta_e
    //   k_theta' = alpha_th × k_th + beta_th
    double k_e_new  = alpha_e  * lqr_k_e  + beta_e;
    double k_th_new = alpha_th * lqr_k_th + beta_th;

    // ---- 7. 填充调试数据 ----
    if (debug_data != nullptr) {
        debug_data->raw_alpha_e = (float)raw_alpha_e;
        debug_data->raw_beta_e = (float)raw_beta_e;
        debug_data->raw_alpha_th = (float)raw_alpha_th;
        debug_data->raw_beta_th = (float)raw_beta_th;
        debug_data->alpha_e = (float)alpha_e;
        debug_data->beta_e = (float)beta_e;
        debug_data->alpha_th = (float)alpha_th;
        debug_data->beta_th = (float)beta_th;
        debug_data->lqr_k_e = (float)lqr_k_e;
        debug_data->k_e_new = (float)k_e_new;
        debug_data->lqr_k_th = (float)lqr_k_th;
        debug_data->k_th_new = (float)k_th_new;
        debug_data->delta_final = (float)(-(k_th_new * e_psi + k_e_new * e_y));
    }

    // ---- 8. 计算自适应修正后的最终控制量 ----
    //   delta = -(k_theta' × e_psi + k_e' × e_y)
    double delta_nn = -(k_th_new * e_psi + k_e_new * e_y);

        LOGI("NN raw: [%.4f, %.4f, %.4f, %.4f] | a_e=%.3f b_e=%.4f a_th=%.3f b_th=%.4f | k_e:%.3f->%.3f k_th:%.3f->%.3f | d_nn=%.5f d_lqr=%.5f",
            raw_alpha_e, raw_beta_e, raw_alpha_th, raw_beta_th,
            alpha_e, beta_e, alpha_th, beta_th,
         lqr_k_e, k_e_new, lqr_k_th, k_th_new,
         delta_nn, base_delta);

    if(v <= 0.3 || abs(e_y) >= 0.03 || abs(e_psi) >= 0.04){
        if (debug_data != nullptr) {
            debug_data->delta_final = (float)base_delta;
        }
        LOGI("border change.");
        return base_delta;
    }
    if (debug_data != nullptr) {
        debug_data->delta_final = (float)delta_nn;
    }
    return delta_nn;
}
