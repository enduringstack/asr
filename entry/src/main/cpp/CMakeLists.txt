cmake_minimum_required(VERSION 3.5.0)
project(ASR)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include
                    ${NATIVERENDER_ROOT_PATH}/feature_extractor
                    ${NATIVERENDER_ROOT_PATH}/asr_engine)

# Original entry library (kept for compatibility)
add_library(entry SHARED
    napi_init.cpp
    acoustic_scene/acoustic_scene_classifier.cpp
    feature_extractor/wenet_fft.cpp
)
target_link_libraries(entry PUBLIC
    libace_napi.z.so
    libnative_media_codecbase.so
    libnative_media_core.so
    libnative_media_avsource.so
    libnative_media_avdemuxer.so
    libnative_media_acodec.so
    libdl.so
)

# ASR Engine library
add_library(asr_engine SHARED
    asr_engine/asr_engine_napi.cpp
    asr_engine/cann_asr_session.cpp
    feature_extractor/wenet_fft.cpp
)

target_include_directories(asr_engine PRIVATE
    ${NATIVERENDER_ROOT_PATH}
)

# Link NAPI and system libraries
target_link_libraries(asr_engine PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
    libneural_network_core.so
)

# CANN Kit on HarmonyOS links NeuralNetworkRuntime from the OpenHarmony
# sysroot and hiai_foundation from the HMS native sysroot.
if(DEFINED HMOS_SDK_NATIVE)
    if(OHOS_ARCH STREQUAL "arm64-v8a")
        set(HMOS_LIB_ARCH "aarch64-linux-ohos")
    elseif(OHOS_ARCH STREQUAL "x86_64")
        set(HMOS_LIB_ARCH "x86_64-linux-ohos")
    else()
        set(HMOS_LIB_ARCH "")
    endif()

    if(HMOS_LIB_ARCH)
        set(HMS_NATIVE_LIB_DIR "${HMOS_SDK_NATIVE}/sysroot/usr/lib/${HMOS_LIB_ARCH}")
        target_link_directories(asr_engine PRIVATE ${HMS_NATIVE_LIB_DIR})
        target_link_libraries(asr_engine PUBLIC libhiai_foundation.so)
    endif()
endif()
