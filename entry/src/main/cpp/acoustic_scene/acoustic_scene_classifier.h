#ifndef ASR_ACOUSTIC_SCENE_CLASSIFIER_H
#define ASR_ACOUSTIC_SCENE_CLASSIFIER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct AcousticScenePrediction {
    int32_t class_index = -1;
    float confidence = 0.0f;
    int32_t window_count = 0;
    float duration_seconds = 0.0f;
    int64_t elapsed_ms = 0;
    std::vector<float> probabilities;
};

class AcousticSceneClassifier {
public:
    AcousticSceneClassifier();
    ~AcousticSceneClassifier();

    AcousticSceneClassifier(const AcousticSceneClassifier&) = delete;
    AcousticSceneClassifier& operator=(const AcousticSceneClassifier&) = delete;

    bool initialize(const uint8_t* model_data, size_t model_size);
    bool isInitialized() const;
    AcousticScenePrediction classify(const float* samples, size_t sample_count, int32_t sample_rate);
    const std::string& lastError() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // ASR_ACOUSTIC_SCENE_CLASSIFIER_H
