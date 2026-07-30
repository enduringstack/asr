#include "cann_asr_session.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <hilog/log.h>

#define ASR_LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, 0x0001, "ASRNative", __VA_ARGS__)
#define ASR_LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, 0x0001, "ASRNative", __VA_ARGS__)

namespace {
constexpr size_t INPUT_COUNT = 4;
constexpr size_t OUTPUT_COUNT = 3;

bool fileExists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

size_t getTensorSize(NN_Tensor* tensor) {
    size_t size = 0;
    if (tensor == nullptr || OH_NNTensor_GetSize(tensor, &size) != OH_NN_SUCCESS) {
        return 0;
    }
    return size;
}

std::vector<int32_t> getTensorShape(const NN_TensorDesc* desc) {
    int32_t* shape = nullptr;
    size_t shape_length = 0;
    if (desc == nullptr || OH_NNTensorDesc_GetShape(desc, &shape, &shape_length) != OH_NN_SUCCESS ||
        shape == nullptr) {
        return {};
    }
    return std::vector<int32_t>(shape, shape + shape_length);
}

int32_t dimFromEnd(const std::vector<int32_t>& shape, size_t offset_from_end) {
    if (shape.size() <= offset_from_end) {
        return -1;
    }
    return shape[shape.size() - 1 - offset_from_end];
}

float computePcmRms(const std::vector<int16_t>& pcm) {
    if (pcm.empty()) {
        return 0.0f;
    }
    double sum_sq = 0.0;
    for (int16_t sample : pcm) {
        double v = static_cast<double>(sample);
        sum_sq += v * v;
    }
    return static_cast<float>(std::sqrt(sum_sq / pcm.size()));
}

void computeFloatStats(const std::vector<float>& values,
                       float& min_value,
                       float& max_value,
                       float& mean_value) {
    if (values.empty()) {
        min_value = 0.0f;
        max_value = 0.0f;
        mean_value = 0.0f;
        return;
    }
    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    double sum = 0.0;
    for (float value : values) {
        sum += value;
    }
    min_value = *min_it;
    max_value = *max_it;
    mean_value = static_cast<float>(sum / values.size());
}

std::string describeTensorDesc(const char* role, size_t index, const NN_TensorDesc* desc) {
    const char* name = nullptr;
    (void)OH_NNTensorDesc_GetName(desc, &name);

    int32_t* shape = nullptr;
    size_t shape_length = 0;
    (void)OH_NNTensorDesc_GetShape(desc, &shape, &shape_length);

    size_t byte_size = 0;
    (void)OH_NNTensorDesc_GetByteSize(desc, &byte_size);

    OH_NN_DataType data_type = OH_NN_UNKNOWN;
    (void)OH_NNTensorDesc_GetDataType(desc, &data_type);

    std::ostringstream oss;
    oss << role << "[" << index << "] name=" << (name == nullptr ? "" : name)
        << " shape=[";
    for (size_t i = 0; i < shape_length; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "] bytes=" << byte_size << " dtype=" << static_cast<int>(data_type);
    return oss.str();
}
} // namespace

CannAsrSession::CannAsrSession()
    : beam_search_(5538) {}

CannAsrSession::~CannAsrSession() {
    releaseRuntime();
}

int CannAsrSession::initialize(const std::string& model_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    releaseRuntime();

    std::string vocab_path = model_dir + "/units.txt";
    int ret = loadVocab(vocab_path);
    if (ret != 0) {
        return ret;
    }

    std::string model_path = model_dir + "/" + MODEL_FILENAME;
    if (!fileExists(model_path)) {
        return -3;
    }

    streaming_state_.init(config_);
    beam_search_.reset();
    greedy_tokens_.clear();
    last_greedy_token_ = CTCBeamSearch::BLANK_ID;
    audio_buffer_.clear();
    feature_buffer_.clear();
    processed_windows_ = 0;

    ret = selectDevice();
    if (ret != 0) {
        releaseRuntime();
        return ret;
    }

    ret = loadModel(model_path);
    if (ret != 0) {
        releaseRuntime();
        return ret;
    }

    initialized_ = true;
    return 0;
}

int CannAsrSession::initializeFromFd(const std::string& model_dir,
                                     int model_fd,
                                     size_t model_offset,
                                     size_t model_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    releaseRuntime();

    std::string vocab_path = model_dir + "/units.txt";
    int ret = loadVocab(vocab_path);
    if (ret != 0) {
        return ret;
    }

    if (model_fd < 0 || model_length == 0) {
        return -4;
    }

    streaming_state_.init(config_);
    beam_search_.reset();
    greedy_tokens_.clear();
    last_greedy_token_ = CTCBeamSearch::BLANK_ID;
    audio_buffer_.clear();
    feature_buffer_.clear();
    processed_windows_ = 0;

    ret = selectDevice();
    if (ret != 0) {
        releaseRuntime();
        return ret;
    }

    ret = loadModelFromFd(model_fd, model_offset, model_length);
    if (ret != 0) {
        releaseRuntime();
        return ret;
    }

    initialized_ = true;
    return 0;
}

int CannAsrSession::initializeFromFdWithVocabData(const std::string& model_dir,
                                                  const void* vocab_data,
                                                  size_t vocab_length,
                                                  int model_fd,
                                                  size_t model_offset,
                                                  size_t model_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    releaseRuntime();

    if (vocab_data == nullptr || vocab_length == 0) {
        return -5;
    }

    std::string vocab_content(static_cast<const char*>(vocab_data), vocab_length);
    int ret = loadVocabFromString(vocab_content);
    if (ret != 0) {
        return ret;
    }

    if (model_fd < 0 || model_length == 0) {
        return -4;
    }

    streaming_state_.init(config_);
    beam_search_.reset();
    greedy_tokens_.clear();
    last_greedy_token_ = CTCBeamSearch::BLANK_ID;
    audio_buffer_.clear();
    feature_buffer_.clear();
    processed_windows_ = 0;

    ret = selectDevice();
    if (ret != 0) {
        releaseRuntime();
        return ret;
    }

    ret = loadModelFromFd(model_fd, model_offset, model_length);
    if (ret != 0) {
        ASR_LOGE("load model from raw fd failed: %{public}d, trying file fallback", ret);
        (void)mkdir(model_dir.c_str(), 0755);
        std::string model_path = model_dir + "/" + MODEL_FILENAME;
        int copy_ret = copyModelFromFdToFile(model_fd, model_offset, model_length, model_path);
        if (copy_ret != 0) {
            releaseRuntime();
            return copy_ret;
        }
        ret = loadModel(model_path);
    }
    if (ret != 0) {
        releaseRuntime();
        return ret;
    }

    initialized_ = true;
    return 0;
}

std::string CannAsrSession::processChunk(const std::vector<int16_t>& pcm_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return "";
    }

    audio_buffer_.insert(audio_buffer_.end(), pcm_data.begin(), pcm_data.end());
    return processAvailableAudio(false);
}

std::string CannAsrSession::finalize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return "";
    }

    std::string flushed = processAvailableAudio(true);
    if (!flushed.empty()) {
        return flushed;
    }
    return tokensToString(greedy_tokens_);
}

void CannAsrSession::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    audio_buffer_.clear();
    feature_buffer_.clear();
    streaming_state_.reset(config_);
    beam_search_.reset();
    greedy_tokens_.clear();
    last_greedy_token_ = CTCBeamSearch::BLANK_ID;
    processed_windows_ = 0;
}

int CannAsrSession::selectDevice() {
    const size_t* all_devices = nullptr;
    uint32_t device_count = 0;
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(&all_devices, &device_count);
    if (ret != OH_NN_SUCCESS || all_devices == nullptr || device_count == 0) {
        return -10;
    }

    device_id_ = all_devices[0];
    for (uint32_t i = 0; i < device_count; ++i) {
        const char* name = nullptr;
        if (OH_NNDevice_GetName(all_devices[i], &name) == OH_NN_SUCCESS && name != nullptr) {
            if (std::strstr(name, "HIAI_F") != nullptr || std::strstr(name, "NPU") != nullptr) {
                device_id_ = all_devices[i];
                break;
            }
        }
    }
    return 0;
}

int CannAsrSession::loadModel(const std::string& model_path) {
    OH_NNCompilation* compilation = OH_NNCompilation_ConstructWithOfflineModelFile(model_path.c_str());
    if (compilation == nullptr) {
        return -20;
    }

    OH_NN_ReturnCode ret = OH_NNCompilation_SetDevice(compilation, device_id_);
    if (ret != OH_NN_SUCCESS) {
        OH_NNCompilation_Destroy(&compilation);
        return -21;
    }

    (void)OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_HIGH);
    ret = OH_NNCompilation_Build(compilation);
    if (ret != OH_NN_SUCCESS) {
        ASR_LOGE("OH_NNCompilation_Build(file) failed: %{public}d", static_cast<int>(ret));
        OH_NNCompilation_Destroy(&compilation);
        return -22;
    }

    executor_ = OH_NNExecutor_Construct(compilation);
    OH_NNCompilation_Destroy(&compilation);
    if (executor_ == nullptr) {
        return -23;
    }

    return createIoTensors();
}

int CannAsrSession::loadModelFromFd(int model_fd, size_t model_offset, size_t model_length) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return -24;
    }

    size_t aligned_offset = model_offset & ~(static_cast<size_t>(page_size) - 1);
    size_t offset_delta = model_offset - aligned_offset;
    size_t map_length = model_length + offset_delta;
    void* mapped = mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE, model_fd,
                        static_cast<off_t>(aligned_offset));
    if (mapped == MAP_FAILED) {
        return -25;
    }

    const void* model_buffer = static_cast<const char*>(mapped) + offset_delta;
    int ret = loadModelFromBuffer(model_buffer, model_length);
    munmap(mapped, map_length);
    return ret;
}

int CannAsrSession::loadModelFromBuffer(const void* model_buffer, size_t model_length) {
    OH_NNCompilation* compilation =
        OH_NNCompilation_ConstructWithOfflineModelBuffer(model_buffer, model_length);
    if (compilation == nullptr) {
        return -26;
    }

    OH_NN_ReturnCode ret = OH_NNCompilation_SetDevice(compilation, device_id_);
    if (ret != OH_NN_SUCCESS) {
        OH_NNCompilation_Destroy(&compilation);
        return -27;
    }

    (void)OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_HIGH);
    ret = OH_NNCompilation_Build(compilation);
    if (ret != OH_NN_SUCCESS) {
        ASR_LOGE("OH_NNCompilation_Build(buffer) failed: %{public}d, model_length=%{public}zu",
                 static_cast<int>(ret),
                 model_length);
        OH_NNCompilation_Destroy(&compilation);
        return -28;
    }

    executor_ = OH_NNExecutor_Construct(compilation);
    OH_NNCompilation_Destroy(&compilation);
    if (executor_ == nullptr) {
        return -29;
    }

    return createIoTensors();
}

int CannAsrSession::copyModelFromFdToFile(int model_fd,
                                          size_t model_offset,
                                          size_t model_length,
                                          const std::string& model_path) {
    std::ofstream output(model_path, std::ios::binary | std::ios::trunc);
    if (!output.good()) {
        ASR_LOGE("open fallback model file failed");
        return -46;
    }

    constexpr size_t kBufferSize = 1024 * 1024;
    std::vector<char> buffer(kBufferSize);
    size_t copied = 0;
    while (copied < model_length) {
        size_t to_read = std::min(kBufferSize, model_length - copied);
        ssize_t n = pread(model_fd,
                          buffer.data(),
                          to_read,
                          static_cast<off_t>(model_offset + copied));
        if (n <= 0) {
            ASR_LOGE("read raw model fd failed at offset=%{public}zu", copied);
            return -47;
        }
        output.write(buffer.data(), n);
        if (!output.good()) {
            ASR_LOGE("write fallback model file failed");
            return -48;
        }
        copied += static_cast<size_t>(n);
    }
    output.close();
    ASR_LOGI("copied fallback model file bytes=%{public}zu", copied);
    return 0;
}

int CannAsrSession::createIoTensors() {
    destroyTensors();

    size_t input_count = 0;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputCount(executor_, &input_count);
    if (ret != OH_NN_SUCCESS || input_count != INPUT_COUNT) {
        return -30;
    }

    size_t output_count = 0;
    ret = OH_NNExecutor_GetOutputCount(executor_, &output_count);
    if (ret != OH_NN_SUCCESS || output_count != OUTPUT_COUNT) {
        return -31;
    }

    std::vector<std::vector<int32_t>> input_shapes;
    std::vector<std::vector<int32_t>> output_shapes;
    input_shapes.reserve(input_count);
    output_shapes.reserve(output_count);

    for (size_t i = 0; i < input_count; ++i) {
        NN_TensorDesc* desc = OH_NNExecutor_CreateInputTensorDesc(executor_, i);
        if (desc == nullptr) {
            destroyTensors();
            return -32;
        }
        input_shapes.push_back(getTensorShape(desc));
        std::string desc_text = describeTensorDesc("input", i, desc);
        ASR_LOGI("%{public}s", desc_text.c_str());
        NN_Tensor* tensor = OH_NNTensor_Create(device_id_, desc);
        OH_NNTensorDesc_Destroy(&desc);
        if (tensor == nullptr) {
            destroyTensors();
            return -33;
        }
        input_tensors_.push_back(tensor);
    }

    for (size_t i = 0; i < output_count; ++i) {
        NN_TensorDesc* desc = OH_NNExecutor_CreateOutputTensorDesc(executor_, i);
        if (desc == nullptr) {
            destroyTensors();
            return -34;
        }
        output_shapes.push_back(getTensorShape(desc));
        std::string desc_text = describeTensorDesc("output", i, desc);
        ASR_LOGI("%{public}s", desc_text.c_str());
        NN_Tensor* tensor = OH_NNTensor_Create(device_id_, desc);
        OH_NNTensorDesc_Destroy(&desc);
        if (tensor == nullptr) {
            destroyTensors();
            return -35;
        }
        output_tensors_.push_back(tensor);
    }

    int config_ret = refreshConfigFromTensorShapes(input_shapes, output_shapes);
    if (config_ret != 0) {
        destroyTensors();
        return config_ret;
    }

    streaming_state_.init(config_);
    beam_search_ = CTCBeamSearch(config_.vocab_size);
    ASR_LOGI("model config: blocks=%{public}d heads=%{public}d output=%{public}d kernel=%{public}d "
             "chunk=%{public}d left=%{public}d vocab=%{public}d window=%{public}d feature=%{public}d",
             config_.num_blocks,
             config_.attention_heads,
             config_.output_size,
             config_.cnn_module_kernel,
             config_.chunk_size,
             config_.num_left_chunks,
             config_.vocab_size,
             (config_.chunk_size - 1) * config_.subsampling_rate + config_.right_context + 1,
             config_.feature_size);

    return 0;
}

int CannAsrSession::refreshConfigFromTensorShapes(
    const std::vector<std::vector<int32_t>>& input_shapes,
    const std::vector<std::vector<int32_t>>& output_shapes) {
    if (input_shapes.size() < INPUT_COUNT || output_shapes.empty()) {
        return -36;
    }

    const auto& feature_shape = input_shapes[0];
    const auto& att_cache_shape = input_shapes[1];
    const auto& conv_cache_shape = input_shapes[2];
    const auto& mask_shape = input_shapes[3];
    const auto& log_probs_shape = output_shapes[0];

    int decoding_window = dimFromEnd(feature_shape, 1);
    int feature_size = dimFromEnd(feature_shape, 0);
    int num_blocks = dimFromEnd(att_cache_shape, 3);
    int attention_heads = dimFromEnd(att_cache_shape, 2);
    int cache_size = dimFromEnd(att_cache_shape, 1);
    int att_last_dim = dimFromEnd(att_cache_shape, 0);
    int conv_output_size = dimFromEnd(conv_cache_shape, 1);
    int conv_cache_width = dimFromEnd(conv_cache_shape, 0);
    int mask_width = dimFromEnd(mask_shape, 0);
    int chunk_size = dimFromEnd(log_probs_shape, 1);
    int vocab_size = dimFromEnd(log_probs_shape, 0);

    if (decoding_window <= 0 || feature_size <= 0 || num_blocks <= 0 ||
        attention_heads <= 0 || cache_size <= 0 || att_last_dim <= 0 ||
        conv_output_size <= 0 || conv_cache_width < 0 || mask_width <= 0 ||
        chunk_size <= 0 || vocab_size <= 0) {
        ASR_LOGE("invalid tensor shapes for model config");
        return -37;
    }

    if (cache_size % chunk_size != 0 || mask_width != cache_size + chunk_size) {
        ASR_LOGE("unsupported cache/mask shape cache=%{public}d chunk=%{public}d mask=%{public}d",
                 cache_size,
                 chunk_size,
                 mask_width);
        return -38;
    }

    int output_size_from_attention = attention_heads * (att_last_dim / 2);
    if (att_last_dim % 2 != 0 || output_size_from_attention != conv_output_size) {
        ASR_LOGE("inconsistent output size att_last=%{public}d heads=%{public}d conv=%{public}d",
                 att_last_dim,
                 attention_heads,
                 conv_output_size);
        return -39;
    }

    if (!vocab_.empty() && vocab_size != static_cast<int>(vocab_.size())) {
        ASR_LOGE("vocab size mismatch model=%{public}d units=%{public}zu",
                 vocab_size,
                 vocab_.size());
        return -44;
    }

    int inferred_right_context =
        decoding_window - (chunk_size - 1) * config_.subsampling_rate - 1;
    if (inferred_right_context < 0) {
        ASR_LOGE("invalid decoding window=%{public}d chunk=%{public}d subsampling=%{public}d",
                 decoding_window,
                 chunk_size,
                 config_.subsampling_rate);
        return -45;
    }

    config_.feature_size = feature_size;
    config_.num_blocks = num_blocks;
    config_.attention_heads = attention_heads;
    config_.output_size = conv_output_size;
    config_.cnn_module_kernel = conv_cache_width + 1;
    config_.chunk_size = chunk_size;
    config_.num_left_chunks = cache_size / chunk_size;
    config_.right_context = inferred_right_context;
    config_.vocab_size = vocab_size;

    return 0;
}

void CannAsrSession::releaseRuntime() {
    destroyTensors();
    if (executor_ != nullptr) {
        OH_NNExecutor_Destroy(&executor_);
        executor_ = nullptr;
    }
    initialized_ = false;
}

void CannAsrSession::destroyTensors() {
    for (NN_Tensor* tensor : input_tensors_) {
        if (tensor != nullptr) {
            OH_NNTensor_Destroy(&tensor);
        }
    }
    input_tensors_.clear();

    for (NN_Tensor* tensor : output_tensors_) {
        if (tensor != nullptr) {
            OH_NNTensor_Destroy(&tensor);
        }
    }
    output_tensors_.clear();
}

std::string CannAsrSession::processAvailableAudio(bool flush) {
    std::string latest;
    const int decoding_window =
        (config_.chunk_size - 1) * config_.subsampling_rate + config_.right_context + 1;
    const int samples_needed =
        (decoding_window - 1) * FbankExtractor::FRAME_SHIFT_SAMPLES +
        FbankExtractor::FRAME_LENGTH_SAMPLES;
    const int shift_samples =
        config_.chunk_size * config_.subsampling_rate * FbankExtractor::FRAME_SHIFT_SAMPLES;
    const int overlap_samples = samples_needed - shift_samples;

    while (fbank_extractor_.getNumFrames(static_cast<int>(audio_buffer_.size())) >= decoding_window) {
        std::vector<int16_t> chunk_audio(audio_buffer_.begin(), audio_buffer_.begin() + samples_needed);
        float pcm_rms = computePcmRms(chunk_audio);
        auto features = fbank_extractor_.extract(chunk_audio);
        if (processFeatureWindow(features, pcm_rms) != 0) {
            return latest;
        }
        latest = tokensToString(greedy_tokens_);

        int erase_count = std::min(shift_samples, static_cast<int>(audio_buffer_.size()));
        audio_buffer_.erase(audio_buffer_.begin(), audio_buffer_.begin() + erase_count);
    }

    if (flush && !audio_buffer_.empty() &&
        (processed_windows_ == 0 || static_cast<int>(audio_buffer_.size()) > overlap_samples)) {
        std::vector<int16_t> chunk_audio = audio_buffer_;
        chunk_audio.resize(samples_needed, 0);
        float pcm_rms = computePcmRms(chunk_audio);
        auto features = fbank_extractor_.extract(chunk_audio);
        if (processFeatureWindow(features, pcm_rms) == 0) {
            latest = tokensToString(greedy_tokens_);
        }
        audio_buffer_.clear();
    }

    return latest;
}

int CannAsrSession::processFeatureWindow(const std::vector<float>& features, float pcm_rms) {
    std::vector<float> ctc_probs;
    int ret = runStreamingModel(features, ctc_probs);
    if (ret != 0) {
        ASR_LOGE("runStreamingModel failed ret=%{public}d window=%{public}d", ret, processed_windows_);
        return ret;
    }

    const size_t tokens_before = greedy_tokens_.size();
    if (pcm_rms >= config_.min_decode_rms) {
        for (int t = 0; t < config_.chunk_size; t++) {
            std::vector<float> frame_probs(
                ctc_probs.begin() + t * config_.vocab_size,
                ctc_probs.begin() + (t + 1) * config_.vocab_size);
            beam_search_.processFrame(frame_probs);
            processGreedyFrame(frame_probs);
        }
    } else {
        last_greedy_token_ = CTCBeamSearch::BLANK_ID;
    }

    if (!ctc_probs.empty() && (processed_windows_ < 12 || processed_windows_ % 10 == 0)) {
        float feat_min = 0.0f;
        float feat_max = 0.0f;
        float feat_mean = 0.0f;
        computeFloatStats(features, feat_min, feat_max, feat_mean);

        int best_id = 0;
        int best_non_blank_id = -1;
        float best_score = -INFINITY;
        float best_non_blank_score = -INFINITY;
        for (int i = 0; i < config_.vocab_size; ++i) {
            float score = ctc_probs[i];
            if (score > best_score) {
                best_score = score;
                best_id = i;
            }
            if (i != CTCBeamSearch::BLANK_ID && score > best_non_blank_score) {
                best_non_blank_score = score;
                best_non_blank_id = i;
            }
        }
        std::string best_text = tokensToString(greedy_tokens_);
        const char* top_token = (best_id >= 0 && best_id < static_cast<int>(vocab_.size()))
                                    ? vocab_[best_id].c_str()
                                    : "";
        const char* top_non_blank_token =
            (best_non_blank_id >= 0 && best_non_blank_id < static_cast<int>(vocab_.size()))
                ? vocab_[best_non_blank_id].c_str()
                : "";
        ASR_LOGI("window=%{public}d pcm_rms=%{public}.1f feat=[%{public}.2f,%{public}.2f,%{public}.2f] "
                 "top=%{public}d/%{public}s/%{public}.2f top_nb=%{public}d/%{public}s/%{public}.2f "
                 "added=%{public}zu best_len=%{public}zu best=%{public}s",
                 processed_windows_,
                 pcm_rms,
                 feat_min,
                 feat_mean,
                 feat_max,
                 best_id,
                 top_token,
                 best_score,
                 best_non_blank_id,
                 top_non_blank_token,
                 best_non_blank_score,
                 greedy_tokens_.size() - tokens_before,
                 best_text.size(),
                 best_text.c_str());
    }

    ++processed_windows_;
    return 0;
}

void CannAsrSession::processGreedyFrame(const std::vector<float>& frame_probs) {
    if (frame_probs.empty()) {
        return;
    }
    int best_id = 0;
    float best_score = frame_probs[0];
    int count = std::min(config_.vocab_size, static_cast<int>(frame_probs.size()));
    for (int i = 1; i < count; ++i) {
        if (frame_probs[i] > best_score) {
            best_score = frame_probs[i];
            best_id = i;
        }
    }

    if (best_id == CTCBeamSearch::BLANK_ID) {
        last_greedy_token_ = CTCBeamSearch::BLANK_ID;
        return;
    }
    if (best_id != last_greedy_token_) {
        greedy_tokens_.push_back(best_id);
    }
    last_greedy_token_ = best_id;
}

int CannAsrSession::runStreamingModel(const std::vector<float>& features,
                                      std::vector<float>& ctc_probs) {
    if (executor_ == nullptr || input_tensors_.size() != INPUT_COUNT ||
        output_tensors_.size() != OUTPUT_COUNT) {
        return -40;
    }

    const size_t feature_count =
        static_cast<size_t>(((config_.chunk_size - 1) * config_.subsampling_rate +
                             config_.right_context + 1) *
                            config_.feature_size);
    std::vector<float> feature_input(feature_count, 0.0f);
    std::copy_n(features.begin(), std::min(features.size(), feature_input.size()),
                feature_input.begin());

    std::vector<float> attn_mask(
        static_cast<size_t>(config_.chunk_size * config_.num_left_chunks + config_.chunk_size), 0.0f);
    int valid_chunks = std::min(config_.num_left_chunks + 1, processed_windows_ + 1);
    int valid_frames = config_.chunk_size * valid_chunks;
    std::fill(attn_mask.end() - valid_frames, attn_mask.end(), 1.0f);

    if (!copyToTensor(0, feature_input.data(), feature_input.size() * sizeof(float)) ||
        !copyToTensor(1, streaming_state_.att_cache.data(),
                      streaming_state_.att_cache.size() * sizeof(float)) ||
        !copyToTensor(2, streaming_state_.cnn_cache.data(),
                      streaming_state_.cnn_cache.size() * sizeof(float)) ||
        !copyToTensor(3, attn_mask.data(), attn_mask.size() * sizeof(float))) {
        return -41;
    }

    OH_NN_ReturnCode ret = OH_NNExecutor_RunSync(executor_,
                                                 input_tensors_.data(),
                                                 input_tensors_.size(),
                                                 output_tensors_.data(),
                                                 output_tensors_.size());
    if (ret != OH_NN_SUCCESS) {
        return -42;
    }

    ctc_probs.assign(static_cast<size_t>(config_.chunk_size * config_.vocab_size), 0.0f);
    if (!copyFromTensor(0, ctc_probs.data(), ctc_probs.size() * sizeof(float)) ||
        !copyFromTensor(1, streaming_state_.att_cache.data(),
                        streaming_state_.att_cache.size() * sizeof(float)) ||
        !copyFromTensor(2, streaming_state_.cnn_cache.data(),
                        streaming_state_.cnn_cache.size() * sizeof(float))) {
        return -43;
    }

    return 0;
}

bool CannAsrSession::copyToTensor(size_t index, const void* data, size_t bytes) {
    if (index >= input_tensors_.size() || data == nullptr) {
        return false;
    }
    void* buffer = OH_NNTensor_GetDataBuffer(input_tensors_[index]);
    size_t tensor_size = getTensorSize(input_tensors_[index]);
    if (buffer == nullptr || tensor_size < bytes) {
        return false;
    }
    std::memcpy(buffer, data, bytes);
    return true;
}

bool CannAsrSession::copyFromTensor(size_t index, void* data, size_t bytes) {
    if (index >= output_tensors_.size() || data == nullptr) {
        return false;
    }
    void* buffer = OH_NNTensor_GetDataBuffer(output_tensors_[index]);
    size_t tensor_size = getTensorSize(output_tensors_[index]);
    if (buffer == nullptr || tensor_size < bytes) {
        return false;
    }
    std::memcpy(data, buffer, bytes);
    return true;
}

std::string CannAsrSession::tokensToString(const std::vector<int>& tokens) {
    std::string result;
    for (int id : tokens) {
        if (id < 0 || id >= static_cast<int>(vocab_.size())) {
            continue;
        }
        const std::string& token = vocab_[id];
        if (token == "<blank>" || token == "<unk>" || token == "<sos/eos>" || token == "▁") {
            continue;
        }
        result += token;
    }
    return result;
}

int CannAsrSession::loadVocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        return -1;
    }

    return loadVocabFromStream(file);
}

int CannAsrSession::loadVocabFromString(const std::string& vocab_content) {
    std::istringstream input(vocab_content);
    return loadVocabFromStream(input);
}

int CannAsrSession::loadVocabFromStream(std::istream& input) {
    vocab_.clear();
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        size_t space_pos = line.rfind(' ');
        if (space_pos != std::string::npos) {
            vocab_.push_back(line.substr(0, space_pos));
        } else {
            vocab_.push_back(line);
        }
    }

    if (vocab_.empty()) {
        return -2;
    }

    config_.vocab_size = static_cast<int>(vocab_.size());
    beam_search_ = CTCBeamSearch(config_.vocab_size);
    return 0;
}
