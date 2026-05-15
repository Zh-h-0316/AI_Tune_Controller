cmake_minimum_required(VERSION 3.10)
project(rknn_bridge C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/librknnrt.so
)

add_library(rknn_bridge SHARED
    adaptive_net_rknn.c
    rknn_bridge_jni.c
)

find_library(log-lib log)

target_link_libraries(
    rknn_bridge
    rknnrt
    ${log-lib}
    dl
    m
)
