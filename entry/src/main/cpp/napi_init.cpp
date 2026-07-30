#include "napi/native_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "multimedia/player_framework/native_avbuffer.h"
#include "multimedia/player_framework/native_avcodec_audiocodec.h"
#include "multimedia/player_framework/native_avcodec_base.h"
#include "multimedia/player_framework/native_avdemuxer.h"
#include "multimedia/player_framework/native_avformat.h"
#include "multimedia/player_framework/native_avsource.h"

#define DR_MP3_IMPLEMENTATION
#include "third_party/dr_mp3.h"

namespace {

constexpr uint32_t TARGET_SAMPLE_RATE = 16000;
constexpr int64_t CODEC_QUERY_TIMEOUT_US = 100000;
constexpr int32_t FALLBACK_SAMPLE_FORMAT = SAMPLE_S16LE;

bool GetByteSpan(napi_env env, napi_value value, const uint8_t** data, size_t* length)
{
    bool is_array_buffer = false;
    napi_is_arraybuffer(env, value, &is_array_buffer);
    if (is_array_buffer) {
        void* raw_data = nullptr;
        size_t raw_length = 0;
        if (napi_get_arraybuffer_info(env, value, &raw_data, &raw_length) != napi_ok ||
            raw_data == nullptr || raw_length == 0) {
            return false;
        }
        *data = static_cast<const uint8_t*>(raw_data);
        *length = raw_length;
        return true;
    }

    bool is_typed_array = false;
    napi_is_typedarray(env, value, &is_typed_array);
    if (is_typed_array) {
        napi_typedarray_type type;
        size_t element_count = 0;
        void* raw_data = nullptr;
        napi_value array_buffer = nullptr;
        size_t byte_offset = 0;
        if (napi_get_typedarray_info(env, value, &type, &element_count, &raw_data, &array_buffer, &byte_offset) != napi_ok ||
            raw_data == nullptr || element_count == 0) {
            return false;
        }
        if (type != napi_uint8_array && type != napi_int8_array && type != napi_uint8_clamped_array) {
            return false;
        }
        *data = static_cast<const uint8_t*>(raw_data);
        *length = element_count;
        return true;
    }

    return false;
}

std::vector<float> DownmixAndResampleToMono16k(const float* interleaved,
                                               uint64_t frame_count,
                                               uint32_t channels,
                                               uint32_t source_sample_rate)
{
    if (interleaved == nullptr || frame_count == 0 || channels == 0 || source_sample_rate == 0) {
        return {};
    }

    std::vector<float> mono(static_cast<size_t>(frame_count));
    for (uint64_t i = 0; i < frame_count; ++i) {
        double sum = 0.0;
        const uint64_t base = i * channels;
        for (uint32_t ch = 0; ch < channels; ++ch) {
            sum += interleaved[base + ch];
        }
        mono[static_cast<size_t>(i)] = static_cast<float>(sum / static_cast<double>(channels));
    }

    if (source_sample_rate == TARGET_SAMPLE_RATE) {
        return mono;
    }

    const double ratio = static_cast<double>(source_sample_rate) / static_cast<double>(TARGET_SAMPLE_RATE);
    const auto output_count = static_cast<size_t>(
        std::max<double>(1.0, std::ceil(static_cast<double>(frame_count) / ratio)));
    std::vector<float> output(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        const double source_pos = static_cast<double>(i) * ratio;
        const size_t left = std::min(static_cast<size_t>(source_pos), mono.size() - 1);
        const size_t right = std::min(left + 1, mono.size() - 1);
        const float fraction = static_cast<float>(source_pos - static_cast<double>(left));
        output[i] = mono[left] * (1.0f - fraction) + mono[right] * fraction;
    }
    return output;
}

napi_value SetNumberProperty(napi_env env, napi_value object, const char* name, double value)
{
    napi_value property = nullptr;
    napi_create_double(env, value, &property);
    napi_set_named_property(env, object, name, property);
    return property;
}

napi_value CreateAudioDecodeResult(napi_env env,
                                   const std::vector<float>& samples,
                                   uint32_t original_sample_rate,
                                   uint32_t channels,
                                   double duration_seconds)
{
    void* sample_data = nullptr;
    napi_value sample_buffer = nullptr;
    const size_t sample_bytes = samples.size() * sizeof(float);
    napi_create_arraybuffer(env, sample_bytes, &sample_data, &sample_buffer);
    std::memcpy(sample_data, samples.data(), sample_bytes);

    napi_value result = nullptr;
    napi_create_object(env, &result);
    napi_set_named_property(env, result, "samples", sample_buffer);
    SetNumberProperty(env, result, "sampleRate", TARGET_SAMPLE_RATE);
    SetNumberProperty(env, result, "originalSampleRate", original_sample_rate);
    SetNumberProperty(env, result, "channels", channels);
    SetNumberProperty(env, result, "durationSeconds", duration_seconds);
    return result;
}

struct MemoryAudioSource {
    const uint8_t* data = nullptr;
    size_t size = 0;
};

int32_t ReadMemoryAudioSource(OH_AVBuffer* buffer, int32_t length, int64_t pos, void* user_data)
{
    auto* source = static_cast<MemoryAudioSource*>(user_data);
    if (source == nullptr || source->data == nullptr || buffer == nullptr || length <= 0 || pos < 0) {
        return -1;
    }
    if (static_cast<uint64_t>(pos) >= source->size) {
        return 0;
    }

    uint8_t* target = OH_AVBuffer_GetAddr(buffer);
    const int32_t capacity = OH_AVBuffer_GetCapacity(buffer);
    if (target == nullptr || capacity <= 0) {
        return -1;
    }

    const size_t available = source->size - static_cast<size_t>(pos);
    const int32_t bytes_to_copy = static_cast<int32_t>(
        std::min<size_t>(available, std::min<int32_t>(length, capacity)));
    if (bytes_to_copy <= 0) {
        return 0;
    }

    std::memcpy(target, source->data + pos, static_cast<size_t>(bytes_to_copy));
    OH_AVCodecBufferAttr attr = { 0, bytes_to_copy, 0, AVCODEC_BUFFER_FLAGS_NONE };
    OH_AVBuffer_SetBufferAttr(buffer, &attr);
    return bytes_to_copy;
}

struct PcmOutputFormat {
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    int32_t sample_format = FALLBACK_SAMPLE_FORMAT;
};

void UpdatePcmFormatFromAvFormat(OH_AVFormat* format, PcmOutputFormat* pcm_format)
{
    if (format == nullptr || pcm_format == nullptr) {
        return;
    }

    int32_t value = 0;
    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_AUD_SAMPLE_RATE, &value) && value > 0) {
        pcm_format->sample_rate = static_cast<uint32_t>(value);
    }
    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_AUD_CHANNEL_COUNT, &value) && value > 0) {
        pcm_format->channels = static_cast<uint32_t>(value);
    }
    if (OH_AVFormat_GetIntValue(format, OH_MD_KEY_AUDIO_SAMPLE_FORMAT, &value)) {
        pcm_format->sample_format = value;
    }
}

int32_t BytesPerSample(int32_t sample_format)
{
    switch (sample_format) {
        case SAMPLE_U8:
        case SAMPLE_U8P:
            return 1;
        case SAMPLE_S16LE:
        case SAMPLE_S16P:
            return 2;
        case SAMPLE_S24LE:
        case SAMPLE_S24P:
            return 3;
        case SAMPLE_S32LE:
        case SAMPLE_S32P:
        case SAMPLE_F32LE:
        case SAMPLE_F32P:
            return 4;
        default:
            return 0;
    }
}

bool IsPlanarSampleFormat(int32_t sample_format)
{
    return sample_format == SAMPLE_U8P ||
        sample_format == SAMPLE_S16P ||
        sample_format == SAMPLE_S24P ||
        sample_format == SAMPLE_S32P ||
        sample_format == SAMPLE_F32P;
}

float ReadPcmSampleAsFloat(const uint8_t* data, int32_t sample_format)
{
    switch (sample_format) {
        case SAMPLE_U8:
        case SAMPLE_U8P:
            return (static_cast<float>(*data) - 128.0f) / 128.0f;
        case SAMPLE_S16LE:
        case SAMPLE_S16P: {
            const int16_t value = static_cast<int16_t>(
                static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8));
            return static_cast<float>(value) / 32768.0f;
        }
        case SAMPLE_S24LE:
        case SAMPLE_S24P: {
            int32_t value = static_cast<int32_t>(data[0]) |
                (static_cast<int32_t>(data[1]) << 8) |
                (static_cast<int32_t>(data[2]) << 16);
            if ((value & 0x00800000) != 0) {
                value |= static_cast<int32_t>(0xFF000000);
            }
            return static_cast<float>(value) / 8388608.0f;
        }
        case SAMPLE_S32LE:
        case SAMPLE_S32P: {
            int32_t value = static_cast<int32_t>(data[0]) |
                (static_cast<int32_t>(data[1]) << 8) |
                (static_cast<int32_t>(data[2]) << 16) |
                (static_cast<int32_t>(data[3]) << 24);
            return static_cast<float>(static_cast<double>(value) / 2147483648.0);
        }
        case SAMPLE_F32LE:
        case SAMPLE_F32P: {
            float value = 0.0f;
            std::memcpy(&value, data, sizeof(float));
            return value;
        }
        default:
            return 0.0f;
    }
}

bool AppendDecodedPcm(OH_AVBuffer* buffer,
                      const OH_AVCodecBufferAttr& attr,
                      const PcmOutputFormat& pcm_format,
                      std::vector<float>* interleaved)
{
    if (buffer == nullptr || interleaved == nullptr || attr.size <= 0 ||
        pcm_format.channels == 0 || pcm_format.sample_rate == 0) {
        return true;
    }

    const int32_t bytes_per_sample = BytesPerSample(pcm_format.sample_format);
    if (bytes_per_sample <= 0) {
        return false;
    }

    uint8_t* base = OH_AVBuffer_GetAddr(buffer);
    const int32_t capacity = OH_AVBuffer_GetCapacity(buffer);
    if (base == nullptr || attr.offset < 0 || attr.size < 0 || attr.offset + attr.size > capacity) {
        return false;
    }

    const auto channels = static_cast<size_t>(pcm_format.channels);
    const size_t total_samples = static_cast<size_t>(attr.size) / static_cast<size_t>(bytes_per_sample);
    if (total_samples < channels) {
        return true;
    }

    const size_t frame_count = total_samples / channels;
    const uint8_t* payload = base + attr.offset;
    interleaved->reserve(interleaved->size() + frame_count * channels);

    if (IsPlanarSampleFormat(pcm_format.sample_format)) {
        const size_t plane_bytes = frame_count * static_cast<size_t>(bytes_per_sample);
        if (plane_bytes * channels > static_cast<size_t>(attr.size)) {
            return false;
        }
        for (size_t frame = 0; frame < frame_count; ++frame) {
            for (size_t channel = 0; channel < channels; ++channel) {
                const uint8_t* sample = payload + channel * plane_bytes + frame * bytes_per_sample;
                interleaved->push_back(ReadPcmSampleAsFloat(sample, pcm_format.sample_format));
            }
        }
        return true;
    }

    for (size_t sample_index = 0; sample_index < frame_count * channels; ++sample_index) {
        const uint8_t* sample = payload + sample_index * bytes_per_sample;
        interleaved->push_back(ReadPcmSampleAsFloat(sample, pcm_format.sample_format));
    }
    return true;
}

void DestroyFormat(OH_AVFormat* format)
{
    if (format != nullptr) {
        OH_AVFormat_Destroy(format);
    }
}

void DestroySource(OH_AVSource* source)
{
    if (source != nullptr) {
        OH_AVSource_Destroy(source);
    }
}

void DestroyDemuxer(OH_AVDemuxer* demuxer)
{
    if (demuxer != nullptr) {
        OH_AVDemuxer_Destroy(demuxer);
    }
}

void DestroyCodec(OH_AVCodec* codec)
{
    if (codec != nullptr) {
        OH_AudioCodec_Stop(codec);
        OH_AudioCodec_Destroy(codec);
    }
}

} // namespace

static napi_value Add(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value args[2] = {nullptr};

    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    napi_valuetype valuetype0;
    napi_typeof(env, args[0], &valuetype0);

    napi_valuetype valuetype1;
    napi_typeof(env, args[1], &valuetype1);

    double value0;
    napi_get_value_double(env, args[0], &value0);

    double value1;
    napi_get_value_double(env, args[1], &value1);

    napi_value sum;
    napi_create_double(env, value0 + value1, &sum);

    return sum;

}

static napi_value DecodeMp3ToMono16k(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1 || args[0] == nullptr) {
        napi_throw_error(env, nullptr, "audio file is invalid");
        return nullptr;
    }

    const uint8_t* mp3_data = nullptr;
    size_t mp3_size = 0;
    if (!GetByteSpan(env, args[0], &mp3_data, &mp3_size)) {
        napi_throw_error(env, nullptr, "audio file is invalid");
        return nullptr;
    }

    drmp3_config config;
    std::memset(&config, 0, sizeof(config));
    drmp3_uint64 frame_count = 0;
    float* decoded = drmp3_open_memory_and_read_pcm_frames_f32(
        mp3_data,
        mp3_size,
        &config,
        &frame_count,
        nullptr
    );
    if (decoded == nullptr || frame_count == 0 || config.channels == 0 || config.sampleRate == 0) {
        if (decoded != nullptr) {
            drmp3_free(decoded, nullptr);
        }
        napi_throw_error(env, nullptr, "audio file is invalid");
        return nullptr;
    }

    std::vector<float> samples = DownmixAndResampleToMono16k(
        decoded,
        static_cast<uint64_t>(frame_count),
        config.channels,
        config.sampleRate
    );
    drmp3_free(decoded, nullptr);

    if (samples.empty()) {
        napi_throw_error(env, nullptr, "audio file is invalid");
        return nullptr;
    }

    return CreateAudioDecodeResult(env, samples, config.sampleRate, config.channels,
        static_cast<double>(frame_count) / static_cast<double>(config.sampleRate));
}

static napi_value DecodeM4aToMono16k(napi_env env, napi_callback_info info)
{
    size_t argc = 3;
    napi_value args[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1 || args[0] == nullptr) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }

    const uint8_t* audio_data = nullptr;
    size_t audio_size = 0;
    if (!GetByteSpan(env, args[0], &audio_data, &audio_size)) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }

    bool has_start = false;
    bool has_end = false;
    int64_t start_us = 0;
    int64_t end_us = 0;
    if (argc >= 2 && args[1] != nullptr) {
        napi_valuetype type = napi_undefined;
        napi_typeof(env, args[1], &type);
        if (type == napi_number) {
            double seconds = 0.0;
            napi_get_value_double(env, args[1], &seconds);
            if (seconds > 0.0) {
                has_start = true;
                start_us = static_cast<int64_t>(seconds * 1000000.0);
            }
        }
    }
    if (argc >= 3 && args[2] != nullptr) {
        napi_valuetype type = napi_undefined;
        napi_typeof(env, args[2], &type);
        if (type == napi_number) {
            double seconds = 0.0;
            napi_get_value_double(env, args[2], &seconds);
            if (seconds > 0.0) {
                has_end = true;
                end_us = static_cast<int64_t>(seconds * 1000000.0);
            }
        }
    }
    if (has_start && has_end && end_us <= start_us) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }

    MemoryAudioSource memory_source { audio_data, audio_size };
    OH_AVDataSourceExt data_source { static_cast<int64_t>(audio_size), ReadMemoryAudioSource };
    std::unique_ptr<OH_AVSource, decltype(&DestroySource)> source(
        OH_AVSource_CreateWithDataSourceExt(&data_source, &memory_source),
        DestroySource
    );
    if (!source) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }

    std::unique_ptr<OH_AVFormat, decltype(&DestroyFormat)> source_format(
        OH_AVSource_GetSourceFormat(source.get()),
        DestroyFormat
    );
    int32_t track_count = 0;
    if (!source_format ||
        !OH_AVFormat_GetIntValue(source_format.get(), OH_MD_KEY_TRACK_COUNT, &track_count) ||
        track_count <= 0) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }

    uint32_t audio_track_index = 0;
    std::unique_ptr<OH_AVFormat, decltype(&DestroyFormat)> audio_format(nullptr, DestroyFormat);
    std::string mime;
    for (int32_t i = 0; i < track_count; ++i) {
        std::unique_ptr<OH_AVFormat, decltype(&DestroyFormat)> track_format(
            OH_AVSource_GetTrackFormat(source.get(), static_cast<uint32_t>(i)),
            DestroyFormat
        );
        if (!track_format) {
            continue;
        }

        int32_t track_type = -1;
        const char* track_mime = nullptr;
        const bool is_audio_type =
            OH_AVFormat_GetIntValue(track_format.get(), OH_MD_KEY_TRACK_TYPE, &track_type) &&
            track_type == MEDIA_TYPE_AUD;
        const bool has_audio_mime =
            OH_AVFormat_GetStringValue(track_format.get(), OH_MD_KEY_CODEC_MIME, &track_mime) &&
            track_mime != nullptr &&
            std::string(track_mime).rfind("audio/", 0) == 0;
        if (is_audio_type || has_audio_mime) {
            audio_track_index = static_cast<uint32_t>(i);
            if (track_mime != nullptr) {
                mime = track_mime;
            }
            audio_format.reset(track_format.release());
            break;
        }
    }

    if (!audio_format || mime.empty()) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败：没有音频轨道");
        return nullptr;
    }

    OH_AVFormat_SetIntValue(audio_format.get(), OH_MD_KEY_ENABLE_SYNC_MODE, 1);
    std::unique_ptr<OH_AVDemuxer, decltype(&DestroyDemuxer)> demuxer(
        OH_AVDemuxer_CreateWithSource(source.get()),
        DestroyDemuxer
    );
    if (!demuxer || OH_AVDemuxer_SelectTrackByID(demuxer.get(), audio_track_index) != AV_ERR_OK) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }
    if (has_start) {
        OH_AVDemuxer_SeekToTime(demuxer.get(), start_us / 1000, SEEK_MODE_CLOSEST_SYNC);
    }

    std::unique_ptr<OH_AVCodec, decltype(&DestroyCodec)> decoder(
        OH_AudioCodec_CreateByMime(mime.c_str(), false),
        DestroyCodec
    );
    if (!decoder ||
        OH_AudioCodec_Configure(decoder.get(), audio_format.get()) != AV_ERR_OK ||
        OH_AudioCodec_Prepare(decoder.get()) != AV_ERR_OK ||
        OH_AudioCodec_Start(decoder.get()) != AV_ERR_OK) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }

    PcmOutputFormat pcm_format;
    UpdatePcmFormatFromAvFormat(audio_format.get(), &pcm_format);
    {
        std::unique_ptr<OH_AVFormat, decltype(&DestroyFormat)> output_format(
            OH_AudioCodec_GetOutputDescription(decoder.get()),
            DestroyFormat
        );
        UpdatePcmFormatFromAvFormat(output_format.get(), &pcm_format);
    }
    if (pcm_format.sample_rate == 0 || pcm_format.channels == 0) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败");
        return nullptr;
    }

    bool input_eos = false;
    bool output_eos = false;
    bool fed_any_sample = false;
    uint32_t empty_input_retries = 0;
    uint32_t empty_output_retries = 0;
    std::vector<float> interleaved_pcm;

    while (!output_eos) {
        if (!input_eos) {
            uint32_t input_index = 0;
            OH_AVErrCode input_query = OH_AudioCodec_QueryInputBuffer(
                decoder.get(),
                &input_index,
                CODEC_QUERY_TIMEOUT_US
            );
            if (input_query == AV_ERR_OK) {
                OH_AVBuffer* input_buffer = OH_AudioCodec_GetInputBuffer(decoder.get(), input_index);
                if (input_buffer == nullptr) {
                    napi_throw_error(env, nullptr, "M4A 音频解码失败");
                    return nullptr;
                }

                OH_AVErrCode demux_result = OH_AVDemuxer_ReadSampleBuffer(
                    demuxer.get(),
                    audio_track_index,
                    input_buffer
                );
                OH_AVCodecBufferAttr attr = { 0, 0, 0, AVCODEC_BUFFER_FLAGS_NONE };
                OH_AVBuffer_GetBufferAttr(input_buffer, &attr);
                if (demux_result == AV_ERR_OK && attr.size > 0 &&
                    has_end && fed_any_sample && attr.pts > end_us) {
                    input_eos = true;
                    OH_AVCodecBufferAttr eos_attr = { attr.pts, 0, 0, AVCODEC_BUFFER_FLAGS_EOS };
                    OH_AVBuffer_SetBufferAttr(input_buffer, &eos_attr);
                    OH_AudioCodec_PushInputBuffer(decoder.get(), input_index);
                } else if (demux_result == AV_ERR_OK && attr.size > 0) {
                    fed_any_sample = true;
                    OH_AudioCodec_PushInputBuffer(decoder.get(), input_index);
                } else {
                    if (demux_result != AV_ERR_OK && !fed_any_sample) {
                        napi_throw_error(env, nullptr, "M4A 音频解码失败");
                        return nullptr;
                    }
                    input_eos = true;
                    OH_AVCodecBufferAttr eos_attr = { attr.pts, 0, 0, AVCODEC_BUFFER_FLAGS_EOS };
                    OH_AVBuffer_SetBufferAttr(input_buffer, &eos_attr);
                    OH_AudioCodec_PushInputBuffer(decoder.get(), input_index);
                }
                empty_input_retries = 0;
            } else if (input_query == AV_ERR_TRY_AGAIN_LATER) {
                ++empty_input_retries;
            } else {
                napi_throw_error(env, nullptr, "M4A 音频解码失败");
                return nullptr;
            }
        }

        uint32_t output_index = 0;
        OH_AVErrCode output_query = OH_AudioCodec_QueryOutputBuffer(
            decoder.get(),
            &output_index,
            CODEC_QUERY_TIMEOUT_US
        );
        if (output_query == AV_ERR_OK) {
            OH_AVBuffer* output_buffer = OH_AudioCodec_GetOutputBuffer(decoder.get(), output_index);
            if (output_buffer == nullptr) {
                napi_throw_error(env, nullptr, "M4A 音频解码失败");
                return nullptr;
            }

            OH_AVCodecBufferAttr attr = { 0, 0, 0, AVCODEC_BUFFER_FLAGS_NONE };
            OH_AVBuffer_GetBufferAttr(output_buffer, &attr);
            if (!AppendDecodedPcm(output_buffer, attr, pcm_format, &interleaved_pcm)) {
                OH_AudioCodec_FreeOutputBuffer(decoder.get(), output_index);
                napi_throw_error(env, nullptr, "M4A 音频解码失败：不支持的 PCM 输出格式");
                return nullptr;
            }
            output_eos = (attr.flags & AVCODEC_BUFFER_FLAGS_EOS) != 0;
            OH_AudioCodec_FreeOutputBuffer(decoder.get(), output_index);
            empty_output_retries = 0;
        } else if (output_query == AV_ERR_STREAM_CHANGED) {
            std::unique_ptr<OH_AVFormat, decltype(&DestroyFormat)> output_format(
                OH_AudioCodec_GetOutputDescription(decoder.get()),
                DestroyFormat
            );
            UpdatePcmFormatFromAvFormat(output_format.get(), &pcm_format);
            if (pcm_format.sample_rate == 0 || pcm_format.channels == 0) {
                napi_throw_error(env, nullptr, "M4A 音频解码失败");
                return nullptr;
            }
        } else if (output_query == AV_ERR_TRY_AGAIN_LATER) {
            ++empty_output_retries;
            if (input_eos && empty_output_retries > 100) {
                break;
            }
            if (!input_eos && empty_input_retries > 100 && empty_output_retries > 100) {
                napi_throw_error(env, nullptr, "M4A 音频解码失败");
                return nullptr;
            }
        } else {
            napi_throw_error(env, nullptr, "M4A 音频解码失败");
            return nullptr;
        }
    }

    if (interleaved_pcm.empty()) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败：没有可识别的音频数据");
        return nullptr;
    }

    const uint64_t frame_count = static_cast<uint64_t>(interleaved_pcm.size() / pcm_format.channels);
    std::vector<float> samples = DownmixAndResampleToMono16k(
        interleaved_pcm.data(),
        frame_count,
        pcm_format.channels,
        pcm_format.sample_rate
    );
    if (samples.empty()) {
        napi_throw_error(env, nullptr, "M4A 音频解码失败：没有可识别的音频数据");
        return nullptr;
    }

    return CreateAudioDecodeResult(env, samples, pcm_format.sample_rate, pcm_format.channels,
        static_cast<double>(frame_count) / static_cast<double>(pcm_format.sample_rate));
}

static napi_value ProbeM4aInfo(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1 || args[0] == nullptr) {
        napi_throw_error(env, nullptr, "M4A 音频解析失败");
        return nullptr;
    }

    const uint8_t* audio_data = nullptr;
    size_t audio_size = 0;
    if (!GetByteSpan(env, args[0], &audio_data, &audio_size)) {
        napi_throw_error(env, nullptr, "M4A 音频解析失败");
        return nullptr;
    }

    MemoryAudioSource memory_source { audio_data, audio_size };
    OH_AVDataSourceExt data_source { static_cast<int64_t>(audio_size), ReadMemoryAudioSource };
    std::unique_ptr<OH_AVSource, decltype(&DestroySource)> source(
        OH_AVSource_CreateWithDataSourceExt(&data_source, &memory_source),
        DestroySource
    );
    if (!source) {
        napi_throw_error(env, nullptr, "M4A 音频解析失败");
        return nullptr;
    }

    std::unique_ptr<OH_AVFormat, decltype(&DestroyFormat)> source_format(
        OH_AVSource_GetSourceFormat(source.get()),
        DestroyFormat
    );
    int32_t track_count = 0;
    int64_t duration_us = 0;
    if (!source_format ||
        !OH_AVFormat_GetIntValue(source_format.get(), OH_MD_KEY_TRACK_COUNT, &track_count) ||
        track_count <= 0) {
        napi_throw_error(env, nullptr, "M4A 音频解析失败");
        return nullptr;
    }
    OH_AVFormat_GetLongValue(source_format.get(), OH_MD_KEY_DURATION, &duration_us);

    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    for (int32_t i = 0; i < track_count; ++i) {
        std::unique_ptr<OH_AVFormat, decltype(&DestroyFormat)> track_format(
            OH_AVSource_GetTrackFormat(source.get(), static_cast<uint32_t>(i)),
            DestroyFormat
        );
        if (!track_format) {
            continue;
        }
        int32_t track_type = -1;
        const char* track_mime = nullptr;
        const bool is_audio_type =
            OH_AVFormat_GetIntValue(track_format.get(), OH_MD_KEY_TRACK_TYPE, &track_type) &&
            track_type == MEDIA_TYPE_AUD;
        const bool has_audio_mime =
            OH_AVFormat_GetStringValue(track_format.get(), OH_MD_KEY_CODEC_MIME, &track_mime) &&
            track_mime != nullptr &&
            std::string(track_mime).rfind("audio/", 0) == 0;
        if (!is_audio_type && !has_audio_mime) {
            continue;
        }
        int32_t value = 0;
        if (OH_AVFormat_GetIntValue(track_format.get(), OH_MD_KEY_AUD_SAMPLE_RATE, &value) && value > 0) {
            sample_rate = static_cast<uint32_t>(value);
        }
        if (OH_AVFormat_GetIntValue(track_format.get(), OH_MD_KEY_AUD_CHANNEL_COUNT, &value) && value > 0) {
            channels = static_cast<uint32_t>(value);
        }
        int64_t track_duration_us = 0;
        if (duration_us <= 0 && OH_AVFormat_GetLongValue(track_format.get(), OH_MD_KEY_DURATION, &track_duration_us)) {
            duration_us = track_duration_us;
        }
        break;
    }

    if (sample_rate == 0 || channels == 0) {
        napi_throw_error(env, nullptr, "M4A 音频解析失败：没有音频轨道");
        return nullptr;
    }

    napi_value result = nullptr;
    napi_create_object(env, &result);
    SetNumberProperty(env, result, "durationSeconds",
        duration_us > 0 ? static_cast<double>(duration_us) / 1000000.0 : 0.0);
    SetNumberProperty(env, result, "sampleRate", sample_rate);
    SetNumberProperty(env, result, "channels", channels);
    return result;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        { "add", nullptr, Add, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "decodeMp3ToMono16k", nullptr, DecodeMp3ToMono16k, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "decodeM4aToMono16k", nullptr, DecodeM4aToMono16k, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "probeM4aInfo", nullptr, ProbeM4aInfo, nullptr, nullptr, nullptr, napi_default, nullptr }
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&demoModule);
}
