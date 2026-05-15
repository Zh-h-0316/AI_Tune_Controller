#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "adaptive_net_rknn.h"

static jlong ptr_to_handle(adaptive_net_rknn_context* ptr) {
    return (jlong)(intptr_t)ptr;
}

static adaptive_net_rknn_context* handle_to_ptr(jlong handle) {
    return (adaptive_net_rknn_context*)(intptr_t)handle;
}

JNIEXPORT jlong JNICALL
Java_com_example_modeldemo_RknnNative_nativeCreate(
    JNIEnv* env,
    jobject thiz,
    jstring model_path
) {
    const char* c_model_path = NULL;
    adaptive_net_rknn_context* ctx = NULL;
    int ret = 0;

    if (model_path == NULL) {
        return 0;
    }

    c_model_path = (*env)->GetStringUTFChars(env, model_path, NULL);
    if (c_model_path == NULL) {
        return 0;
    }

    ret = adaptive_net_rknn_create(c_model_path, &ctx);
    (*env)->ReleaseStringUTFChars(env, model_path, c_model_path);

    if (ret != 0) {
        return 0;
    }

    return ptr_to_handle(ctx);
}

JNIEXPORT void JNICALL
Java_com_example_modeldemo_RknnNative_nativeDestroy(
    JNIEnv* env,
    jobject thiz,
    jlong handle
) {
    adaptive_net_rknn_context* ctx = handle_to_ptr(handle);
    adaptive_net_rknn_destroy(ctx);
}

JNIEXPORT jstring JNICALL
Java_com_example_modeldemo_RknnNative_nativeDescribe(
    JNIEnv* env,
    jobject thiz,
    jlong handle
) {
    adaptive_net_rknn_context* ctx = handle_to_ptr(handle);
    adaptive_net_rknn_info info;
    char buffer[768];
    int ret = adaptive_net_rknn_query(ctx, &info);

    if (ret != 0) {
        snprintf(buffer, sizeof(buffer), "RKNN query failed: %d", ret);
        return (*env)->NewStringUTF(env, buffer);
    }

    snprintf(
        buffer,
        sizeof(buffer),
        "input_count=%d\noutput_count=%d\napi_version=%s\ndriver_version=%s",
        info.input_count,
        info.output_count,
        info.api_version,
        info.driver_version
    );
    return (*env)->NewStringUTF(env, buffer);
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_modeldemo_RknnNative_nativeInferRaw(
    JNIEnv* env,
    jobject thiz,
    jlong handle,
    jfloatArray time_array,
    jfloatArray scalar_array
) {
    adaptive_net_rknn_context* ctx = handle_to_ptr(handle);
    jsize time_len = (*env)->GetArrayLength(env, time_array);
    jsize scalar_len = (*env)->GetArrayLength(env, scalar_array);
    jfloatArray result = NULL;
    jfloat* time_ptr = NULL;
    jfloat* scalar_ptr = NULL;
    float output[ADAPTIVE_NET_OUTPUT_DIM];
    int ret = 0;

    if (time_len != ADAPTIVE_NET_TIME_STEPS * ADAPTIVE_NET_TIME_DIM) {
        return NULL;
    }
    if (scalar_len != ADAPTIVE_NET_SCALAR_DIM) {
        return NULL;
    }

    time_ptr = (*env)->GetFloatArrayElements(env, time_array, NULL);
    scalar_ptr = (*env)->GetFloatArrayElements(env, scalar_array, NULL);
    if (time_ptr == NULL || scalar_ptr == NULL) {
        if (time_ptr != NULL) {
            (*env)->ReleaseFloatArrayElements(env, time_array, time_ptr, JNI_ABORT);
        }
        if (scalar_ptr != NULL) {
            (*env)->ReleaseFloatArrayElements(env, scalar_array, scalar_ptr, JNI_ABORT);
        }
        return NULL;
    }

    ret = adaptive_net_rknn_infer_raw(ctx, time_ptr, scalar_ptr, output);

    (*env)->ReleaseFloatArrayElements(env, time_array, time_ptr, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, scalar_array, scalar_ptr, JNI_ABORT);

    if (ret != 0) {
        return NULL;
    }

    result = (*env)->NewFloatArray(env, ADAPTIVE_NET_OUTPUT_DIM);
    if (result == NULL) {
        return NULL;
    }

    (*env)->SetFloatArrayRegion(env, result, 0, ADAPTIVE_NET_OUTPUT_DIM, output);
    return result;
}
