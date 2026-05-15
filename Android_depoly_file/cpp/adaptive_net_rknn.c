#include "adaptive_net_rknn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknn_api.h"

static const float k_time_mean[ADAPTIVE_NET_TIME_DIM] = {
    0.0009524399f,
    -0.00016289423f,
    -0.0012696810f,
    0.0128847540f,
    0.00014648598f
};

static const float k_time_std[ADAPTIVE_NET_TIME_DIM] = {
    0.00474151f,
    0.0024762452f,
    0.013740111f,
    0.01089334f,
    0.005303936f
};

static const float k_scalar_mean[ADAPTIVE_NET_SCALAR_DIM] = {
    1.0329387f,
    2.2524745f
};

static const float k_scalar_std[ADAPTIVE_NET_SCALAR_DIM] = {
    0.37117162f,
    0.12565596f
};

struct adaptive_net_rknn_context {
    rknn_context ctx;
    rknn_input_output_num io_num;
};

static int check_context(adaptive_net_rknn_context* ctx) {
    if (ctx == NULL) {
        return -1;
    }
    if (ctx->ctx <= 0) {
        return -2;
    }
    return 0;
}

int adaptive_net_rknn_create(const char* model_path, adaptive_net_rknn_context** out_ctx) {
    int ret = 0;
    adaptive_net_rknn_context* ctx = NULL;

    if (model_path == NULL || out_ctx == NULL) {
        return -1;
    }

    ctx = (adaptive_net_rknn_context*)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return -2;
    }

    ret = rknn_init(&ctx->ctx, (void*)model_path, 0, 0, NULL);
    if (ret != RKNN_SUCC) {
        free(ctx);
        return ret;
    }

    ret = rknn_query(ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num, sizeof(ctx->io_num));
    if (ret != RKNN_SUCC) {
        rknn_destroy(ctx->ctx);
        free(ctx);
        return ret;
    }

    *out_ctx = ctx;
    return 0;
}

void adaptive_net_rknn_destroy(adaptive_net_rknn_context* ctx) {
    if (ctx == NULL) {
        return;
    }
    if (ctx->ctx > 0) {
        rknn_destroy(ctx->ctx);
    }
    free(ctx);
}

int adaptive_net_rknn_query(adaptive_net_rknn_context* ctx, adaptive_net_rknn_info* out_info) {
    rknn_sdk_version version;
    int ret = 0;

    if (check_context(ctx) != 0 || out_info == NULL) {
        return -1;
    }

    memset(out_info, 0, sizeof(*out_info));
    memset(&version, 0, sizeof(version));

    ret = rknn_query(ctx->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret != RKNN_SUCC) {
        return ret;
    }

    out_info->input_count = (int32_t)ctx->io_num.n_input;
    out_info->output_count = (int32_t)ctx->io_num.n_output;
    snprintf(out_info->api_version, sizeof(out_info->api_version), "%s", version.api_version);
    snprintf(out_info->driver_version, sizeof(out_info->driver_version), "%s", version.drv_version);
    return 0;
}

void adaptive_net_rknn_normalize(
    const float* raw_time,
    const float* raw_scalar,
    float* normalized_time,
    float* normalized_scalar
) {
    int i = 0;

    for (i = 0; i < ADAPTIVE_NET_TIME_STEPS * ADAPTIVE_NET_TIME_DIM; ++i) {
        int feature_index = i % ADAPTIVE_NET_TIME_DIM;
        normalized_time[i] = (raw_time[i] - k_time_mean[feature_index]) / k_time_std[feature_index];
    }

    for (i = 0; i < ADAPTIVE_NET_SCALAR_DIM; ++i) {
        normalized_scalar[i] = (raw_scalar[i] - k_scalar_mean[i]) / k_scalar_std[i];
    }
}

int adaptive_net_rknn_infer_normalized(
    adaptive_net_rknn_context* ctx,
    const float* time_features,
    const float* scalar_features,
    float* output4
) {
    rknn_input inputs[ADAPTIVE_NET_SCALAR_DIM];
    rknn_output outputs[1];
    int ret = 0;

    if (check_context(ctx) != 0) {
        return -1;
    }
    if (time_features == NULL || scalar_features == NULL || output4 == NULL) {
        return -2;
    }

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].buf = (void*)time_features;
    inputs[0].size = ADAPTIVE_NET_TIME_STEPS * ADAPTIVE_NET_TIME_DIM * (uint32_t)sizeof(float);
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_UNDEFINED;

    inputs[1].index = 1;
    inputs[1].buf = (void*)scalar_features;
    inputs[1].size = ADAPTIVE_NET_SCALAR_DIM * (uint32_t)sizeof(float);
    inputs[1].pass_through = 0;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].fmt = RKNN_TENSOR_UNDEFINED;

    ret = rknn_inputs_set(ctx->ctx, 2, inputs);
    if (ret != RKNN_SUCC) {
        return ret;
    }

    ret = rknn_run(ctx->ctx, NULL);
    if (ret != RKNN_SUCC) {
        return ret;
    }

    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;
    outputs[0].index = 0;

    ret = rknn_outputs_get(ctx->ctx, 1, outputs, NULL);
    if (ret != RKNN_SUCC) {
        return ret;
    }

    memcpy(output4, outputs[0].buf, ADAPTIVE_NET_OUTPUT_DIM * sizeof(float));
    rknn_outputs_release(ctx->ctx, 1, outputs);
    return 0;
}

int adaptive_net_rknn_infer_raw(
    adaptive_net_rknn_context* ctx,
    const float* raw_time,
    const float* raw_scalar,
    float* output4
) {
    float normalized_time[ADAPTIVE_NET_TIME_STEPS * ADAPTIVE_NET_TIME_DIM];
    float normalized_scalar[ADAPTIVE_NET_SCALAR_DIM];

    if (raw_time == NULL || raw_scalar == NULL || output4 == NULL) {
        return -1;
    }

    adaptive_net_rknn_normalize(raw_time, raw_scalar, normalized_time, normalized_scalar);
    return adaptive_net_rknn_infer_normalized(ctx, normalized_time, normalized_scalar, output4);
}
