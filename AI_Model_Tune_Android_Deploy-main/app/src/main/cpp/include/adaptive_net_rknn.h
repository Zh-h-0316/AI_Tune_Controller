#ifndef ADAPTIVE_NET_RKNN_H
#define ADAPTIVE_NET_RKNN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 模型输入输出固定维度定义。
 * 这里按当前模型的真实维度写死，便于调用方直接分配数组。
 */
#define ADAPTIVE_NET_TIME_STEPS 10
#define ADAPTIVE_NET_TIME_DIM 5
#define ADAPTIVE_NET_SCALAR_DIM 2
#define ADAPTIVE_NET_OUTPUT_DIM 4

typedef struct adaptive_net_rknn_context adaptive_net_rknn_context;

/*
 * 运行时查询信息。
 * 可以用来确认当前加载的模型是否符合预期。
 */
typedef struct adaptive_net_rknn_info {
    int32_t input_count;
    int32_t output_count;
    char api_version[256];
    char driver_version[256];
} adaptive_net_rknn_info;

/*
 * 创建 RKNN 上下文。
 *
 * 参数:
 * - model_path: .rknn 模型文件路径
 * - out_ctx: 输出上下文
 *
 * 返回:
 * - 0 表示成功
 * - 非 0 表示失败
 */
int adaptive_net_rknn_create(const char* model_path, adaptive_net_rknn_context** out_ctx);

/*
 * 销毁上下文，释放资源。
 */
void adaptive_net_rknn_destroy(adaptive_net_rknn_context* ctx);

/*
 * 查询 SDK 版本与输入输出数量。
 */
int adaptive_net_rknn_query(adaptive_net_rknn_context* ctx, adaptive_net_rknn_info* out_info);

/*
 * 将原始输入按训练统计量做归一化。
 *
 * raw_time:
 *   长度必须是 10 * 5 = 50
 *
 * raw_scalar:
 *   长度必须是 2
 *
 * normalized_time / normalized_scalar:
 *   输出缓冲区，长度分别为 50 / 2
 */
void adaptive_net_rknn_normalize(
    const float* raw_time,
    const float* raw_scalar,
    float* normalized_time,
    float* normalized_scalar
);

/*
 * 输入已经归一化好的数据，直接执行推理。
 *
 * time_features:
 *   长度必须是 50，布局按 [1, 10, 5] 展平
 *
 * scalar_features:
 *   长度必须是 2，布局按 [1, 2]
 *
 * output4:
 *   输出长度为 4
 */
int adaptive_net_rknn_infer_normalized(
    adaptive_net_rknn_context* ctx,
    const float* time_features,
    const float* scalar_features,
    float* output4
);

/*
 * 输入原始数据，内部自动做归一化后推理。
 */
int adaptive_net_rknn_infer_raw(
    adaptive_net_rknn_context* ctx,
    const float* raw_time,
    const float* raw_scalar,
    float* output4
);

#ifdef __cplusplus
}
#endif

#endif
