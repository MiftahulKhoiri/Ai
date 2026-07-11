// generation_advanced.cpp
#include "generation_advanced.h"
#include "sampling.h"
#include <iostream>
#include <cmath>

namespace advanced_generation {

AdvancedGenerator::AdvancedGenerator(MiniGPT& model, ByteLevelBPETokenizer& tokenizer)
    : model(model), tokenizer(tokenizer) {
    config.max_length = 50;
    config.num_beams = 1;
    config.temperature = 0.8;
    config.top_p = 0.9;
    config.top_k = 40;
    config.repetition_penalty = 1.0;
    config.length_penalty = 1.0;
    config.num_return_sequences = 1;
    config.use_cache = true;
    config.no_repeat_ngram_size = 0;
}

void AdvancedGenerator::set_config(const GenerationConfig& cfg) {
    config = cfg;
}

const GenerationConfig& AdvancedGenerator::get_config() const {
    return config;
}

std::vector<std::string> AdvancedGenerator::generate(const std::string& prompt, bool add_bos, bool add_eos) {
    std::vector<int> input_ids = tokenizer.encode(prompt, add_bos, add_eos);
    if (input_ids.empty()) {
        std::cerr << "Prompt kosong atau tidak bisa di-encode.\n";
        return {};
    }
    
    model.set_training(false);
    if (config.use_cache) {
        model.init_cache();
    }
    
    if (config.num_beams <= 1) {
        return greedy_generate(input_ids);
    } else {
        return beam_search_generate(input_ids);
    }
}

std::vector<std::string> AdvancedGenerator::greedy_generate(const std::vector<int>& input_ids) {
    std::vector<int> current_tokens = input_ids;
    int eos_id = tokenizer.get_eos_token_id();
    
    for (int pos = (int)input_ids.size(); pos < config.max_length; ++pos) {
        int token_id = current_tokens.back();
        
        std::vector<Value::Ptr> logits;
        if (config.use_cache) {
            logits = model.forward_incremental(token_id, pos);
        } else {
            logits = model.forward_incremental(token_id, pos);
        }
        
        // Apply repetition penalty
        if (config.repetition_penalty > 1.0) {
            apply_repetition_penalty(logits, current_tokens, config.repetition_penalty);
        }
        
        int next_token = 0;
        if (config.top_p < 1.0 && config.top_p > 0.0) {
            auto sampled = sampling::top_p_sample(logits, config.top_p, config.temperature);
            next_token = sampled.empty() ? 0 : sampled[0];
        } else if (config.top_k > 0) {
            auto sampled = sampling::top_k_sample(logits, config.top_k, config.temperature);
            next_token = sampled.empty() ? 0 : sampled[0];
        } else {
            next_token = sampling::greedy_sample(logits);
        }
        
        if (config.forbidden_tokens.count(next_token) > 0) {
            next_token = sampling::greedy_sample(logits);
        }
        
        if (next_token == eos_id && pos >= config.min_length) {
            break;
        }
        
        current_tokens.push_back(next_token);
    }
    
    std::string text = tokenizer.decode(current_tokens);
    return {text};
}

std::vector<std::string> AdvancedGenerator::beam_search_generate(const std::vector<int>& input_ids) {
    // Simplified beam search - fallback to greedy for now
    // Full implementation would require more complex beam management
    return greedy_generate(input_ids);
}

void AdvancedGenerator::apply_repetition_penalty(std::vector<Value::Ptr>& logits, 
                                                   const std::vector<int>& tokens,
                                                   float penalty) {
    // Count token frequencies
    std::unordered_map<int, int> freq;
    for (int token : tokens) {
        freq[token]++;
    }
    
    // Apply penalty to logits based on frequency
    // This is simplified - in practice you'd need to map logit indices to token IDs
    for (size_t i = 0; i < logits.size(); ++i) {
        auto it = freq.find(static_cast<int>(i));
        if (it != freq.end()) {
            logits[i]->data /= penalty * it->second;
        }
    }
}

} // namespace advanced_generation