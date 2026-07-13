// generation_advanced.h
#pragma once
#include "model.h"
#include "tokenizer.h"
#include "value.h"
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <cmath>

namespace advanced_generation {

struct GenerationConfig {
    int max_length = 100;
    int min_length = 10;
    int num_beams = 1;
    float temperature = 0.8;
    float top_p = 0.9;
    int top_k = 40;
    float repetition_penalty = 1.0;
    float length_penalty = 1.0;
    int num_return_sequences = 1;
    bool use_cache = true;
    bool early_stopping = true;
    int no_repeat_ngram_size = 0;
    std::unordered_set<int> forbidden_tokens;
    std::unordered_set<int> forced_tokens;
};

class AdvancedGenerator {
public:
    AdvancedGenerator(MiniGPT& model, ByteLevelBPETokenizer& tokenizer);

    void set_config(const GenerationConfig& config);
    const GenerationConfig& get_config() const;
    std::vector<std::string> generate(const std::string& prompt, bool add_bos = false, bool add_eos = false);

private:
    MiniGPT& model;
    ByteLevelBPETokenizer& tokenizer;
    GenerationConfig config;

    std::vector<std::string> greedy_generate(const std::vector<int>& input_ids);
    std::vector<std::string> beam_search_generate(const std::vector<int>& input_ids);
    void apply_repetition_penalty(std::vector<Value::Ptr>& logits, 
                                   const std::vector<int>& tokens,
                                   float penalty);
};

} // namespace advanced_generation