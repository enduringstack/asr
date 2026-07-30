#ifndef FEATURE_EXTRACTOR_FBANK_H
#define FEATURE_EXTRACTOR_FBANK_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "feature_extractor/wenet_fft.h"

namespace asr_frontend {

enum class WindowType {
    kPovey = 0,
};

class Fbank {
public:
    Fbank(int num_bins,
          int sample_rate,
          int frame_length,
          int frame_shift,
          float low_freq = 20.0f,
          float log_floor = std::numeric_limits<float>::epsilon())
        : num_bins_(num_bins),
          sample_rate_(sample_rate),
          frame_length_(frame_length),
          frame_shift_(frame_shift),
          fft_points_(upperPowerOfTwo(frame_length)),
          low_freq_(low_freq),
          high_freq_(sample_rate / 2.0f),
          log_floor_(log_floor) {
        const int fft_points_4 = fft_points_ / 4;
        bitrev_.resize(fft_points_);
        sintbl_.resize(fft_points_ + fft_points_4);
        make_sintbl(fft_points_, sintbl_.data());
        make_bitrev(fft_points_, bitrev_.data());
        initMelFilters();
        initWindow(WindowType::kPovey);
    }

    int compute(const std::vector<float>& wave, std::vector<std::vector<float>>* feat) const {
        if (feat == nullptr) {
            return 0;
        }
        const int num_samples = static_cast<int>(wave.size());
        if (num_samples < frame_length_) {
            feat->clear();
            return 0;
        }

        const int num_frames = 1 + ((num_samples - frame_length_) / frame_shift_);
        feat->assign(num_frames, std::vector<float>(num_bins_, 0.0f));

        std::vector<float> fft_real(fft_points_, 0.0f);
        std::vector<float> fft_img(fft_points_, 0.0f);
        std::vector<float> power(fft_points_ / 2, 0.0f);

        for (int frame = 0; frame < num_frames; ++frame) {
            std::vector<float> data(wave.data() + frame * frame_shift_,
                                    wave.data() + frame * frame_shift_ + frame_length_);

            removeDcOffset(&data);
            preEmphasis(0.97f, &data);
            applyWindow(&data);

            std::fill(fft_real.begin(), fft_real.end(), 0.0f);
            std::fill(fft_img.begin(), fft_img.end(), 0.0f);
            std::memcpy(fft_real.data(), data.data(), sizeof(float) * frame_length_);

            fft(bitrev_.data(), sintbl_.data(), fft_real.data(), fft_img.data(), fft_points_);

            for (int i = 0; i < fft_points_ / 2; ++i) {
                power[i] = fft_real[i] * fft_real[i] + fft_img[i] * fft_img[i];
            }

            for (int bin = 0; bin < num_bins_; ++bin) {
                float mel_energy = 0.0f;
                const int start = bins_[bin].first;
                const std::vector<float>& weights = bins_[bin].second;
                for (size_t i = 0; i < weights.size(); ++i) {
                    mel_energy += weights[i] * power[start + static_cast<int>(i)];
                }
                if (mel_energy < log_floor_) {
                    mel_energy = log_floor_;
                }
                (*feat)[frame][bin] = std::log(mel_energy);
            }
        }

        return num_frames;
    }

private:
    static int upperPowerOfTwo(int n) {
        return static_cast<int>(std::pow(2.0, std::ceil(std::log(static_cast<double>(n)) / std::log(2.0))));
    }

    static float melScale(float freq) {
        return 1127.0f * std::log(1.0f + freq / 700.0f);
    }

    static void removeDcOffset(std::vector<float>* data) {
        if (data == nullptr || data->empty()) {
            return;
        }
        float mean = 0.0f;
        for (float value : *data) {
            mean += value;
        }
        mean /= static_cast<float>(data->size());
        for (float& value : *data) {
            value -= mean;
        }
    }

    static void preEmphasis(float coeff, std::vector<float>* data) {
        if (data == nullptr || data->empty() || coeff == 0.0f) {
            return;
        }
        for (int i = static_cast<int>(data->size()) - 1; i > 0; --i) {
            (*data)[i] -= coeff * (*data)[i - 1];
        }
        (*data)[0] -= coeff * (*data)[0];
    }

    void initMelFilters() {
        const int num_fft_bins = fft_points_ / 2;
        const float fft_bin_width = static_cast<float>(sample_rate_) / static_cast<float>(fft_points_);
        const float mel_low_freq = melScale(low_freq_);
        const float mel_high_freq = melScale(high_freq_);
        const float mel_freq_delta = (mel_high_freq - mel_low_freq) / static_cast<float>(num_bins_ + 1);

        bins_.resize(num_bins_);
        for (int bin = 0; bin < num_bins_; ++bin) {
            const float left_mel = mel_low_freq + static_cast<float>(bin) * mel_freq_delta;
            const float center_mel = mel_low_freq + static_cast<float>(bin + 1) * mel_freq_delta;
            const float right_mel = mel_low_freq + static_cast<float>(bin + 2) * mel_freq_delta;

            std::vector<float> dense_weights(num_fft_bins, 0.0f);
            int first_index = -1;
            int last_index = -1;
            for (int i = 0; i < num_fft_bins; ++i) {
                const float mel = melScale(fft_bin_width * static_cast<float>(i));
                if (mel > left_mel && mel < right_mel) {
                    float weight = 0.0f;
                    if (mel <= center_mel) {
                        weight = (mel - left_mel) / (center_mel - left_mel);
                    } else {
                        weight = (right_mel - mel) / (right_mel - center_mel);
                    }
                    dense_weights[i] = weight;
                    if (first_index == -1) {
                        first_index = i;
                    }
                    last_index = i;
                }
            }

            if (first_index == -1 || last_index < first_index) {
                bins_[bin].first = 0;
                bins_[bin].second.assign(1, 0.0f);
                continue;
            }

            bins_[bin].first = first_index;
            const int size = last_index + 1 - first_index;
            bins_[bin].second.resize(size);
            for (int i = 0; i < size; ++i) {
                bins_[bin].second[i] = dense_weights[first_index + i];
            }
        }
    }

    void initWindow(WindowType window_type) {
        window_.resize(frame_length_, 1.0f);
        if (window_type == WindowType::kPovey) {
            const double a = M_2PI / static_cast<double>(frame_length_ - 1);
            for (int i = 0; i < frame_length_; ++i) {
                window_[i] = static_cast<float>(std::pow(0.5 - 0.5 * std::cos(a * i), 0.85));
            }
        }
    }

    void applyWindow(std::vector<float>* data) const {
        if (data == nullptr || data->size() < window_.size()) {
            return;
        }
        for (size_t i = 0; i < window_.size(); ++i) {
            (*data)[i] *= window_[i];
        }
    }

    int num_bins_;
    int sample_rate_;
    int frame_length_;
    int frame_shift_;
    int fft_points_;
    float low_freq_;
    float high_freq_;
    float log_floor_;
    std::vector<std::pair<int, std::vector<float>>> bins_;
    std::vector<float> window_;
    std::vector<int> bitrev_;
    std::vector<float> sintbl_;
};

} // namespace asr_frontend

class FbankExtractor {
public:
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int NUM_MEL_BINS = 80;
    static constexpr int FRAME_LENGTH_MS = 25;
    static constexpr int FRAME_SHIFT_MS = 10;
    static constexpr int FRAME_LENGTH_SAMPLES = SAMPLE_RATE * FRAME_LENGTH_MS / 1000;
    static constexpr int FRAME_SHIFT_SAMPLES = SAMPLE_RATE * FRAME_SHIFT_MS / 1000;

    FbankExtractor()
        : fbank_(NUM_MEL_BINS,
                 SAMPLE_RATE,
                 FRAME_LENGTH_SAMPLES,
                 FRAME_SHIFT_SAMPLES,
                 20.0f,
                 std::numeric_limits<float>::epsilon()) {}

    std::vector<float> extract(const std::vector<int16_t>& pcm_samples) const {
        std::vector<float> wave(pcm_samples.size(), 0.0f);
        for (size_t i = 0; i < pcm_samples.size(); ++i) {
            wave[i] = static_cast<float>(pcm_samples[i]);
        }

        std::vector<std::vector<float>> matrix;
        int num_frames = fbank_.compute(wave, &matrix);
        std::vector<float> features(static_cast<size_t>(num_frames * NUM_MEL_BINS), 0.0f);
        for (int frame = 0; frame < num_frames; ++frame) {
            std::copy(matrix[frame].begin(),
                      matrix[frame].end(),
                      features.begin() + static_cast<size_t>(frame * NUM_MEL_BINS));
        }
        return features;
    }

    int getNumFrames(int num_samples) const {
        if (num_samples < FRAME_LENGTH_SAMPLES) {
            return 0;
        }
        return (num_samples - FRAME_LENGTH_SAMPLES) / FRAME_SHIFT_SAMPLES + 1;
    }

private:
    asr_frontend::Fbank fbank_;
};

#endif // FEATURE_EXTRACTOR_FBANK_H
