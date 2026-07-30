#ifndef CTC_BEAM_SEARCH_H
#define CTC_BEAM_SEARCH_H

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>

// CTC Prefix Beam Search Decoder
// Simplified version for wenet streaming ASR

struct BeamEntry {
    std::vector<int> prefix;      // token indices
    float log_prob_blank;         // log prob ending with blank
    float log_prob_non_blank;     // log prob ending with non-blank
    float total_log_prob;         // total log prob

    BeamEntry() : log_prob_blank(-INFINITY), log_prob_non_blank(-INFINITY), total_log_prob(-INFINITY) {}
};

class CTCBeamSearch {
public:
    static constexpr int BLANK_ID = 0;
    static constexpr int BEAM_SIZE = 10;
    static constexpr int TOKEN_TOP_K = 32;

    CTCBeamSearch(int vocab_size) : vocab_size_(vocab_size) {}

    // Process one frame of CTC log-probabilities
    // log_probs: [vocab_size] log probabilities for current time step
    void processFrame(const std::vector<float>& log_probs) {
        std::vector<int> candidate_tokens = getTopTokens(log_probs);

        // New beams
        std::unordered_map<std::string, BeamEntry> new_beams;

        for (auto& [key, beam] : beams_) {
            // Extend with blank
            std::string blank_key = prefixToKey(beam.prefix);
            auto& new_beam = new_beams[blank_key];
            if (new_beam.total_log_prob == -INFINITY) {
                new_beam.prefix = beam.prefix;
                new_beam.log_prob_blank = -INFINITY;
                new_beam.log_prob_non_blank = -INFINITY;
            }
            float lp_blank = logAdd(beam.log_prob_blank, beam.log_prob_non_blank) + log_probs[BLANK_ID];
            new_beam.log_prob_blank = logAdd(new_beam.log_prob_blank, lp_blank);
            new_beam.total_log_prob = logAdd(new_beam.log_prob_blank, new_beam.log_prob_non_blank);

            // Extend with each non-blank token
            for (int c : candidate_tokens) {
                if (c == BLANK_ID) continue;

                std::vector<int> new_prefix = beam.prefix;
                if (new_prefix.empty() || new_prefix.back() != c) {
                    new_prefix.push_back(c);
                }

                std::string new_key = prefixToKey(new_prefix);
                auto& new_beam2 = new_beams[new_key];
                if (new_beam2.total_log_prob == -INFINITY) {
                    new_beam2.prefix = new_prefix;
                    new_beam2.log_prob_blank = -INFINITY;
                    new_beam2.log_prob_non_blank = -INFINITY;
                }

                if (new_prefix == beam.prefix) {
                    // Same prefix (repeated character) — only extend from blank
                    float lp = logAdd(beam.log_prob_blank + log_probs[c],
                                      new_beam2.log_prob_non_blank);
                    new_beam2.log_prob_non_blank = lp;
                } else {
                    // New character
                    float lp = logAdd(logAdd(beam.log_prob_blank, beam.log_prob_non_blank) + log_probs[c],
                                      new_beam2.log_prob_non_blank);
                    new_beam2.log_prob_non_blank = lp;
                }
                new_beam2.total_log_prob = logAdd(new_beam2.log_prob_blank, new_beam2.log_prob_non_blank);
            }
        }

        // Prune to beam size
        std::vector<std::pair<std::string, BeamEntry>> sorted_beams(new_beams.begin(), new_beams.end());
        std::sort(sorted_beams.begin(), sorted_beams.end(),
                  [](const auto& a, const auto& b) { return a.second.total_log_prob > b.second.total_log_prob; });

        beams_.clear();
        for (int i = 0; i < std::min((int)sorted_beams.size(), BEAM_SIZE); i++) {
            beams_[sorted_beams[i].first] = sorted_beams[i].second;
        }
    }

    // Get best hypothesis
    std::vector<int> getBestHypothesis() const {
        BeamEntry best;
        best.total_log_prob = -INFINITY;
        for (auto& [key, beam] : beams_) {
            if (beam.total_log_prob > best.total_log_prob) {
                best = beam;
            }
        }
        return best.prefix;
    }

    float getBestScore() const {
        float best = -INFINITY;
        for (auto& [key, beam] : beams_) {
            best = std::max(best, beam.total_log_prob);
        }
        return best;
    }

    void reset() {
        beams_.clear();
        BeamEntry initial;
        initial.log_prob_blank = 0.0f;  // log(1) = 0
        initial.total_log_prob = 0.0f;
        beams_[""] = initial;
    }

private:
    int vocab_size_;
    std::unordered_map<std::string, BeamEntry> beams_;

    std::string prefixToKey(const std::vector<int>& prefix) const {
        std::string key;
        for (int id : prefix) {
            key += std::to_string(id) + ",";
        }
        return key;
    }

    static float logAdd(float a, float b) {
        if (a == -INFINITY) return b;
        if (b == -INFINITY) return a;
        float max_val = std::max(a, b);
        return max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));
    }

    std::vector<int> getTopTokens(const std::vector<float>& log_probs) const {
        int count = std::min(vocab_size_, static_cast<int>(log_probs.size()));
        int top_k = std::min(TOKEN_TOP_K, count);
        std::vector<int> indices(count);
        for (int i = 0; i < count; ++i) {
            indices[i] = i;
        }
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                          [&](int a, int b) { return log_probs[a] > log_probs[b]; });
        indices.resize(top_k);
        if (std::find(indices.begin(), indices.end(), BLANK_ID) == indices.end()) {
            indices.push_back(BLANK_ID);
        }
        return indices;
    }
};

#endif // CTC_BEAM_SEARCH_H
