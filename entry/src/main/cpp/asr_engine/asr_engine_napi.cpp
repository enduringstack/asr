#include "napi/native_api.h"
#include "asr_engine/cann_asr_session.h"
#include <cstring>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

static std::unique_ptr<CannAsrSession> g_asr_session;
static std::mutex g_mutex;

struct ProcessChunkAsyncContext {
    napi_env env = nullptr;
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    std::vector<int16_t> pcm_data;
    std::string partial;
};

static std::string GetStringArg(napi_env env, napi_value value, const std::string& fallback) {
    napi_valuetype value_type;
    napi_typeof(env, value, &value_type);
    if (value_type != napi_string) {
        return fallback;
    }

    size_t len = 0;
    napi_get_value_string_utf8(env, value, nullptr, 0, &len);
    std::vector<char> buffer(len + 1, '\0');
    napi_get_value_string_utf8(env, value, buffer.data(), buffer.size(), &len);
    std::string result(buffer.data(), len);
    return result.empty() ? fallback : result;
}

static napi_value Initialize(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_asr_session && g_asr_session->isInitialized()) {
        napi_value result;
        napi_create_int32(env, 0, &result);
        return result;
    }

    g_asr_session = std::make_unique<CannAsrSession>();

    std::string model_dir = "/data/storage/el2/base/files/asr_models";
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc > 0 && args[0] != nullptr) {
        model_dir = GetStringArg(env, args[0], model_dir);
    }

    int ret = g_asr_session->initialize(model_dir);

    napi_value result;
    napi_create_int32(env, ret, &result);
    return result;
}

static napi_value InitializeFromFd(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_asr_session && g_asr_session->isInitialized()) {
        napi_value result;
        napi_create_int32(env, 0, &result);
        return result;
    }

    size_t argc = 5;
    napi_value args[5] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 4) {
        napi_value result;
        napi_create_int32(env, -100, &result);
        return result;
    }

    std::string model_dir = GetStringArg(env, args[0], "/data/storage/el2/base/files/asr_models");
    int32_t fd = -1;
    int64_t offset = 0;
    int64_t length = 0;
    napi_get_value_int32(env, args[1], &fd);
    napi_get_value_int64(env, args[2], &offset);
    napi_get_value_int64(env, args[3], &length);
    if (offset < 0 || length < 0) {
        napi_value result;
        napi_create_int32(env, -102, &result);
        return result;
    }

    g_asr_session = std::make_unique<CannAsrSession>();
    int ret = 0;
    if (argc >= 5 && args[4] != nullptr) {
        void* vocab_data = nullptr;
        size_t vocab_length = 0;
        napi_status status = napi_get_arraybuffer_info(env, args[4], &vocab_data, &vocab_length);
        if (status != napi_ok || vocab_data == nullptr || vocab_length == 0) {
            napi_value result;
            napi_create_int32(env, -103, &result);
            return result;
        }

        ret = g_asr_session->initializeFromFdWithVocabData(model_dir,
                                                           vocab_data,
                                                           vocab_length,
                                                           fd,
                                                           static_cast<size_t>(offset),
                                                           static_cast<size_t>(length));
    } else {
        ret = g_asr_session->initializeFromFd(model_dir,
                                              fd,
                                              static_cast<size_t>(offset),
                                              static_cast<size_t>(length));
    }

    napi_value result;
    napi_create_int32(env, ret, &result);
    return result;
}

// Process one audio chunk (PCM S16LE, 16kHz mono)
// Input: ArrayBuffer of PCM data
// Returns: string (partial recognition result)
static napi_value ProcessChunk(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_asr_session || !g_asr_session->isInitialized()) {
        napi_value result;
        napi_create_string_utf8(env, "", NAPI_AUTO_LENGTH, &result);
        return result;
    }

    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    // Get ArrayBuffer data
    void* data = nullptr;
    size_t data_len = 0;
    napi_get_arraybuffer_info(env, args[0], &data, &data_len);

    // Convert to int16_t vector
    int num_samples = data_len / sizeof(int16_t);
    std::vector<int16_t> pcm_data(num_samples);
    memcpy(pcm_data.data(), data, data_len);

    // Process
    std::string partial = g_asr_session->processChunk(pcm_data);

    napi_value result;
    napi_create_string_utf8(env, partial.c_str(), partial.size(), &result);
    return result;
}

static void ExecuteProcessChunk(napi_env env, void* data) {
    auto* context = static_cast<ProcessChunkAsyncContext*>(data);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_asr_session && g_asr_session->isInitialized()) {
        context->partial = g_asr_session->processChunk(context->pcm_data);
    }
}

static void CompleteProcessChunk(napi_env env, napi_status status, void* data) {
    auto* context = static_cast<ProcessChunkAsyncContext*>(data);
    napi_value result;
    napi_create_string_utf8(env, context->partial.c_str(), context->partial.size(), &result);
    napi_resolve_deferred(env, context->deferred, result);
    napi_delete_async_work(env, context->work);
    delete context;
}

static napi_value ProcessChunkAsync(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    napi_value promise;
    auto* context = new ProcessChunkAsyncContext();
    context->env = env;
    napi_create_promise(env, &context->deferred, &promise);

    if (argc < 1 || args[0] == nullptr) {
        napi_value result;
        napi_create_string_utf8(env, "", NAPI_AUTO_LENGTH, &result);
        napi_resolve_deferred(env, context->deferred, result);
        delete context;
        return promise;
    }

    void* data = nullptr;
    size_t data_len = 0;
    napi_status status = napi_get_arraybuffer_info(env, args[0], &data, &data_len);
    if (status != napi_ok || data == nullptr || data_len == 0) {
        napi_value result;
        napi_create_string_utf8(env, "", NAPI_AUTO_LENGTH, &result);
        napi_resolve_deferred(env, context->deferred, result);
        delete context;
        return promise;
    }

    int num_samples = data_len / sizeof(int16_t);
    context->pcm_data.resize(num_samples);
    memcpy(context->pcm_data.data(), data, num_samples * sizeof(int16_t));

    napi_value resource_name;
    napi_create_string_utf8(env, "ProcessAsrChunk", NAPI_AUTO_LENGTH, &resource_name);
    status = napi_create_async_work(env,
                                    nullptr,
                                    resource_name,
                                    ExecuteProcessChunk,
                                    CompleteProcessChunk,
                                    context,
                                    &context->work);
    if (status != napi_ok || context->work == nullptr ||
        napi_queue_async_work(env, context->work) != napi_ok) {
        napi_value result;
        napi_create_string_utf8(env, "", NAPI_AUTO_LENGTH, &result);
        napi_resolve_deferred(env, context->deferred, result);
        if (context->work != nullptr) {
            napi_delete_async_work(env, context->work);
        }
        delete context;
        return promise;
    }

    return promise;
}

// Finalize recognition (after all chunks processed)
// Returns: string (final recognition result)
static napi_value Finalize(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_asr_session || !g_asr_session->isInitialized()) {
        napi_value result;
        napi_create_string_utf8(env, "", NAPI_AUTO_LENGTH, &result);
        return result;
    }

    std::string final_result = g_asr_session->finalize();

    napi_value result;
    napi_create_string_utf8(env, final_result.c_str(), final_result.size(), &result);
    return result;
}

// Reset streaming state
static napi_value Reset(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_asr_session) {
        g_asr_session->reset();
    }

    napi_value result;
    napi_get_undefined(env, &result);
    return result;
}

// Get initialization status
static napi_value IsInitialized(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_mutex);

    bool init = g_asr_session && g_asr_session->isInitialized();

    napi_value result;
    napi_get_boolean(env, init, &result);
    return result;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        { "initialize", nullptr, Initialize, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "initializeFromFd", nullptr, InitializeFromFd, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "processChunk", nullptr, ProcessChunk, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "processChunkAsync", nullptr, ProcessChunkAsync, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "finalize", nullptr, Finalize, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "reset", nullptr, Reset, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "isInitialized", nullptr, IsInitialized, nullptr, nullptr, nullptr, napi_default, nullptr },
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module asrEngineModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "asr_engine",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterAsrEngineModule(void) {
    napi_module_register(&asrEngineModule);
}
