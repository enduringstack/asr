#include "acoustic_scene/acoustic_scene_classifier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "feature_extractor/wenet_fft.h"
#include "third_party/onnxruntime_c_api.h"

namespace {

constexpr int32_t kSampleRate = 16000;
constexpr int32_t kWindowSamples = 160000;
constexpr int32_t kWindowHopSamples = 80000;
constexpr int32_t kPreemphasizedSamples = kWindowSamples - 1;
constexpr int32_t kFftSize = 512;
constexpr int32_t kFftBins = kFftSize / 2 + 1;
constexpr int32_t kWinLength = 400;
constexpr int32_t kWindowOffset = (kFftSize - kWinLength) / 2;
constexpr int32_t kFrameShift = 160;
constexpr int32_t kPad = kFftSize / 2;
constexpr int32_t kFrames = 1000;
constexpr int32_t kClassCount = 6;
constexpr int32_t kMaxWindows = 5;

using Clock = std::chrono::steady_clock;

std::vector<float> ResampleLinear(const float* samples, size_t count, int32_t source_rate)
{
    if (samples == nullptr || count == 0 || source_rate <= 0) {
        return {};
    }
    if (source_rate == kSampleRate) {
        return std::vector<float>(samples, samples + count);
    }
    const double output_count_exact = static_cast<double>(count) * kSampleRate / source_rate;
    const size_t output_count = std::max<size_t>(1, static_cast<size_t>(std::llround(output_count_exact)));
    std::vector<float> output(output_count);
    const double source_step = static_cast<double>(source_rate) / kSampleRate;
    for (size_t index = 0; index < output_count; ++index) {
        const double source_position = std::min(
            static_cast<double>(count - 1), static_cast<double>(index) * source_step);
        const size_t left = static_cast<size_t>(source_position);
        const size_t right = std::min(left + 1, count - 1);
        const float fraction = static_cast<float>(source_position - left);
        output[index] = samples[left] * (1.0f - fraction) + samples[right] * fraction;
    }
    return output;
}

std::vector<size_t> WindowOffsets(size_t sample_count)
{
    if (sample_count <= static_cast<size_t>(kWindowSamples)) {
        return {0};
    }
    std::vector<size_t> offsets;
    for (size_t offset = 0; offset + kWindowSamples <= sample_count; offset += kWindowHopSamples) {
        offsets.push_back(offset);
    }
    const size_t final_offset = sample_count - kWindowSamples;
    if (offsets.empty() || offsets.back() != final_offset) {
        offsets.push_back(final_offset);
    }
    if (offsets.size() > kMaxWindows) {
        offsets.erase(offsets.begin(), offsets.end() - kMaxWindows);
    }
    return offsets;
}

size_t ReflectIndex(int64_t index, size_t length)
{
    if (length <= 1) {
        return 0;
    }
    while (index < 0 || index >= static_cast<int64_t>(length)) {
        if (index < 0) {
            index = -index;
        }
        if (index >= static_cast<int64_t>(length)) {
            index = static_cast<int64_t>(2 * length - 2) - index;
        }
    }
    return static_cast<size_t>(index);
}

std::vector<float> BuildHannWindow()
{
    std::vector<float> window(kWinLength);
    for (int32_t index = 0; index < kWinLength; ++index) {
        window[index] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * M_PI * index / static_cast<double>(kWinLength - 1)));
    }
    return window;
}

} // namespace

class AcousticSceneClassifier::Impl {
public:
    ~Impl()
    {
        reset();
    }

    bool initialize(const uint8_t* model_data, size_t model_size)
    {
        reset();
        if (model_data == nullptr || model_size == 0) {
            return fail("scene model is empty");
        }
        library_ = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
        if (library_ == nullptr) {
            return fail(std::string("unable to load libonnxruntime.so: ") + dlerror());
        }
        auto get_api_base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(library_, "OrtGetApiBase"));
        if (get_api_base == nullptr) {
            return fail("OrtGetApiBase is unavailable");
        }
        api_ = get_api_base()->GetApi(ORT_API_VERSION);
        if (api_ == nullptr) {
            return fail("ONNX Runtime API 1.16 is unavailable");
        }
        if (!check(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "acoustic_scene", &environment_))) {
            return false;
        }
        OrtSessionOptions* options = nullptr;
        if (!check(api_->CreateSessionOptions(&options))) {
            return false;
        }
        if (!check(api_->SetIntraOpNumThreads(options, 2)) ||
            !check(api_->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL))) {
            api_->ReleaseSessionOptions(options);
            return false;
        }
        // Keep the backing bytes alive for the whole session. Some ORT builds may retain
        // references to externally supplied model buffers after session construction.
        model_bytes_.assign(model_data, model_data + model_size);
        const bool created = check(api_->CreateSessionFromArray(
            environment_, model_bytes_.data(), model_bytes_.size(), options, &session_));
        api_->ReleaseSessionOptions(options);
        if (!created) {
            return false;
        }
        if (!check(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_))) {
            return false;
        }
        window_ = BuildHannWindow();
        bitrev_.resize(kFftSize);
        sintbl_.resize(kFftSize + kFftSize / 4);
        asr_frontend::make_bitrev(kFftSize, bitrev_.data());
        asr_frontend::make_sintbl(kFftSize, sintbl_.data());
        last_error_.clear();
        return true;
    }

    bool isInitialized() const
    {
        return session_ != nullptr;
    }

    AcousticScenePrediction classify(const float* samples, size_t sample_count, int32_t sample_rate)
    {
        if (!isInitialized()) {
            throw std::runtime_error("scene classifier is not initialized");
        }
        if (samples == nullptr || sample_count == 0 || sample_rate <= 0) {
            throw std::runtime_error("scene classifier received empty audio");
        }
        const auto started = Clock::now();
        std::vector<float> resampled = ResampleLinear(samples, sample_count, sample_rate);
        const std::vector<size_t> offsets = WindowOffsets(resampled.size());
        // Match the offline source-level protocol: aggregate model logits for
        // all temporal windows, then apply softmax exactly once. Averaging
        // already-normalized probabilities would produce a different mobile
        // decision from the calibrated Mac evaluator.
        std::vector<float> mean_logits(kClassCount, 0.0f);
        const int64_t input_shape[] = {1, kFftBins, kFrames};
        const char* input_names[] = {"power_spectrogram"};
        const char* output_names[] = {"logits"};
        for (const size_t offset : offsets) {
            std::vector<float> power(kFftBins * kFrames, 0.0f);
            computePower(resampled, offset, 0, &power);
            OrtValue* input = nullptr;
            if (!check(api_->CreateTensorWithDataAsOrtValue(
                memory_info_, power.data(), power.size() * sizeof(float), input_shape, 3,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input))) {
                throw std::runtime_error(last_error_);
            }
            OrtValue* output = nullptr;
            const OrtValue* inputs[] = {input};
            const bool ran = check(api_->Run(
                session_, nullptr, input_names, inputs, 1, output_names, 1, &output));
            api_->ReleaseValue(input);
            if (!ran) {
                if (output != nullptr) {
                    api_->ReleaseValue(output);
                }
                throw std::runtime_error(last_error_);
            }
            OrtTensorTypeAndShapeInfo* shape_info = nullptr;
            if (!check(api_->GetTensorTypeAndShape(output, &shape_info))) {
                api_->ReleaseValue(output);
                throw std::runtime_error(last_error_);
            }
            size_t element_count = 0;
            const bool shape_ok = check(
                api_->GetTensorShapeElementCount(shape_info, &element_count));
            api_->ReleaseTensorTypeAndShapeInfo(shape_info);
            if (!shape_ok || element_count != kClassCount) {
                api_->ReleaseValue(output);
                throw std::runtime_error("scene model returned an unexpected output shape");
            }
            void* output_data = nullptr;
            if (!check(api_->GetTensorMutableData(output, &output_data)) ||
                output_data == nullptr) {
                api_->ReleaseValue(output);
                throw std::runtime_error(last_error_);
            }
            const auto* row = static_cast<const float*>(output_data);
            for (int32_t index = 0; index < kClassCount; ++index) {
                if (!std::isfinite(row[index])) {
                    api_->ReleaseValue(output);
                    throw std::runtime_error("scene model returned non-finite logits");
                }
                mean_logits[index] += row[index];
            }
            api_->ReleaseValue(output);
        }
        for (float& value : mean_logits) {
            value /= static_cast<float>(offsets.size());
        }
        const float maximum = *std::max_element(mean_logits.begin(), mean_logits.end());
        std::vector<float> probabilities(kClassCount, 0.0f);
        float denominator = 0.0f;
        for (int32_t index = 0; index < kClassCount; ++index) {
            probabilities[index] = std::exp(mean_logits[index] - maximum);
            denominator += probabilities[index];
        }
        if (!std::isfinite(denominator) || denominator <= 0.0f) {
            throw std::runtime_error("scene model returned non-finite probabilities");
        }
        for (float& value : probabilities) {
            value /= denominator;
        }
        const auto top = std::max_element(probabilities.begin(), probabilities.end());
        AcousticScenePrediction prediction;
        prediction.class_index = static_cast<int32_t>(std::distance(probabilities.begin(), top));
        prediction.confidence = *top;
        prediction.window_count = static_cast<int32_t>(offsets.size());
        prediction.duration_seconds = static_cast<float>(resampled.size()) / kSampleRate;
        prediction.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - started).count();
        prediction.probabilities = std::move(probabilities);
        return prediction;
    }

    const std::string& lastError() const
    {
        return last_error_;
    }

private:
    void computePower(const std::vector<float>& samples, size_t offset, size_t batch,
                      std::vector<float>* output)
    {
        std::vector<float> window_samples(kWindowSamples, 0.0f);
        const size_t available = offset < samples.size() ? samples.size() - offset : 0;
        const size_t copied = std::min<size_t>(available, kWindowSamples);
        if (copied > 0) {
            std::copy_n(samples.data() + offset, copied, window_samples.data());
        }
        std::vector<float> emphasized(kPreemphasizedSamples);
        for (int32_t index = 0; index < kPreemphasizedSamples; ++index) {
            emphasized[index] = window_samples[index + 1] - 0.97f * window_samples[index];
        }

        std::vector<float> real(kFftSize, 0.0f);
        std::vector<float> imaginary(kFftSize, 0.0f);
        for (int32_t frame = 0; frame < kFrames; ++frame) {
            std::fill(real.begin(), real.end(), 0.0f);
            std::fill(imaginary.begin(), imaginary.end(), 0.0f);
            const int64_t frame_start = static_cast<int64_t>(frame * kFrameShift) - kPad;
            for (int32_t sample = 0; sample < kWinLength; ++sample) {
                const int64_t signal_index = frame_start + kWindowOffset + sample;
                real[kWindowOffset + sample] = emphasized[
                    ReflectIndex(signal_index, emphasized.size())] * window_[sample];
            }
            asr_frontend::fft(bitrev_.data(), sintbl_.data(), real.data(), imaginary.data(), kFftSize);
            for (int32_t bin = 0; bin < kFftBins; ++bin) {
                const float value = real[bin] * real[bin] + imaginary[bin] * imaginary[bin];
                const size_t target = (batch * kFftBins + bin) * kFrames + frame;
                (*output)[target] = value;
            }
        }
    }

    bool check(OrtStatus* status)
    {
        if (status == nullptr) {
            return true;
        }
        last_error_ = api_ != nullptr ? api_->GetErrorMessage(status) : "ONNX Runtime error";
        if (api_ != nullptr) {
            api_->ReleaseStatus(status);
        }
        return false;
    }

    bool fail(std::string message)
    {
        last_error_ = std::move(message);
        return false;
    }

    void reset()
    {
        if (api_ != nullptr) {
            if (memory_info_ != nullptr) {
                api_->ReleaseMemoryInfo(memory_info_);
            }
            if (session_ != nullptr) {
                api_->ReleaseSession(session_);
            }
            if (environment_ != nullptr) {
                api_->ReleaseEnv(environment_);
            }
        }
        memory_info_ = nullptr;
        session_ = nullptr;
        environment_ = nullptr;
        api_ = nullptr;
        model_bytes_.clear();
        if (library_ != nullptr) {
            dlclose(library_);
            library_ = nullptr;
        }
    }

    void* library_ = nullptr;
    const OrtApi* api_ = nullptr;
    OrtEnv* environment_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    std::vector<uint8_t> model_bytes_;
    std::vector<float> window_;
    std::vector<int> bitrev_;
    std::vector<float> sintbl_;
    std::string last_error_;
};

AcousticSceneClassifier::AcousticSceneClassifier() : impl_(std::make_unique<Impl>()) {}
AcousticSceneClassifier::~AcousticSceneClassifier() = default;

bool AcousticSceneClassifier::initialize(const uint8_t* model_data, size_t model_size)
{
    return impl_->initialize(model_data, model_size);
}

bool AcousticSceneClassifier::isInitialized() const
{
    return impl_->isInitialized();
}

AcousticScenePrediction AcousticSceneClassifier::classify(
    const float* samples, size_t sample_count, int32_t sample_rate)
{
    return impl_->classify(samples, sample_count, sample_rate);
}

const std::string& AcousticSceneClassifier::lastError() const
{
    return impl_->lastError();
}
