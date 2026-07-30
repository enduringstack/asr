#ifndef CANN_ASR_SESSION_H
#define CANN_ASR_SESSION_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <iosfwd>
#include <cstdint>

#include "neural_network_runtime/neural_network_core.h"
#include "feature_extractor/fbank.h"
#include "asr_engine/ctc_beam_search.h"

// ASR model configuration (from train.yaml)
struct AsrConfig {
    int output_size = 256;          // encoder output dimension
    int num_blocks = 12;            // encoder blocks
    int attention_heads = 4;
    int cnn_module_kernel = 8;
    int feature_size = 80;          // mel bins
    int vocab_size = 5235;
    int chunk_size = 16;            // frames per chunk
    int num_left_chunks = 4;        // left context chunks
    int subsampling_rate = 4;       // conv2d subsampling
    int right_context = 6;
    float min_decode_rms = 140.0f;
};

// Streaming state for wenet encoder
struct StreamingState {
    int offset = 0;
    std::vector<float> att_cache;   // [num_blocks * head * cache_size * d_k*2]
    std::vector<float> cnn_cache;   // [num_blocks * 1 * output_size * (kernel-1)]
    bool initialized = false;

    void init(const AsrConfig& cfg) {
        int cache_size = cfg.chunk_size * cfg.num_left_chunks;  // 64
        int d_k_2 = cfg.output_size / cfg.attention_heads * 2;  // 128
        att_cache.assign(cfg.num_blocks * cfg.attention_heads * cache_size * d_k_2, 0.0f);
        cnn_cache.assign(cfg.num_blocks * 1 * cfg.output_size * (cfg.cnn_module_kernel - 1), 0.0f);
        offset = cache_size;  // start with filled cache (16/4 mode)
        initialized = true;
    }

    void reset(const AsrConfig& cfg) {
        init(cfg);
    }
};

// CANN Kit ASR inference session
class CannAsrSession {
public:
    CannAsrSession();
    ~CannAsrSession();

    // Initialize: load .om models from rawfile directory
    // model_dir: path to directory containing encoder.om, ctc.om, decoder.om
    int initialize(const std::string& model_dir);
    int initializeFromFd(const std::string& model_dir, int model_fd, size_t model_offset, size_t model_length);
    int initializeFromFdWithVocabData(const std::string& model_dir,
                                      const void* vocab_data,
                                      size_t vocab_length,
                                      int model_fd,
                                      size_t model_offset,
                                      size_t model_length);

    // Process one audio chunk (100ms PCM S16LE)
    // Returns partial recognition result
    std::string processChunk(const std::vector<int16_t>& pcm_data);

    // Finalize: get final result after all chunks processed
    std::string finalize();

    // Reset streaming state
    void reset();

    // Check if initialized
    bool isInitialized() const { return initialized_; }

private:
    static constexpr const char* MODEL_FILENAME = "model-streaming-fixed-floatmask.om";

    AsrConfig config_;
    FbankExtractor fbank_extractor_;
    CTCBeamSearch beam_search_;
    StreamingState streaming_state_;
    bool initialized_ = false;
    int processed_windows_ = 0;
    std::vector<int> greedy_tokens_;
    int last_greedy_token_ = CTCBeamSearch::BLANK_ID;

    // Audio buffer for accumulating PCM data
    std::vector<int16_t> audio_buffer_;

    // Feature buffer
    std::vector<float> feature_buffer_;

    OH_NNExecutor* executor_ = nullptr;
    size_t device_id_ = 0;
    std::vector<NN_Tensor*> input_tensors_;
    std::vector<NN_Tensor*> output_tensors_;

    // Vocab table (token -> character mapping)
    std::vector<std::string> vocab_;

    int selectDevice();
    int loadModel(const std::string& model_path);
    int loadModelFromFd(int model_fd, size_t model_offset, size_t model_length);
    int loadModelFromBuffer(const void* model_buffer, size_t model_length);
    int copyModelFromFdToFile(int model_fd,
                              size_t model_offset,
                              size_t model_length,
                              const std::string& model_path);
    int createIoTensors();
    int refreshConfigFromTensorShapes(const std::vector<std::vector<int32_t>>& input_shapes,
                                      const std::vector<std::vector<int32_t>>& output_shapes);
    void releaseRuntime();
    void destroyTensors();
    int runStreamingModel(const std::vector<float>& features,
                          std::vector<float>& ctc_probs);
    std::string processAvailableAudio(bool flush);
    int processFeatureWindow(const std::vector<float>& features, float pcm_rms);
    void processGreedyFrame(const std::vector<float>& frame_probs);
    bool copyToTensor(size_t index, const void* data, size_t bytes);
    bool copyFromTensor(size_t index, void* data, size_t bytes);

    // Convert token IDs to string
    std::string tokensToString(const std::vector<int>& tokens);

    // Load vocab from units.txt
    int loadVocab(const std::string& vocab_path);
    int loadVocabFromString(const std::string& vocab_content);
    int loadVocabFromStream(std::istream& input);

    std::mutex mutex_;
};

#endif // CANN_ASR_SESSION_H
