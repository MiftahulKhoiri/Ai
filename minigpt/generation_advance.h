// generation_advanced.h
#pragma once
#include "model.h"
#include "tokenizer.h"
#include "sampling.h"
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <numeric>

namespace advanced_generation {

struct GenerationConfig {
    int max_length = 100;
    int min_length = 10;
    int num_beams = 1;          // 1 = greedy, >1 = beam search
    float temperature = 0.8;
    float top_p = 0.9;
    int top_k = 40;
    float repetition_penalty = 1.0;  // >1.0 penalizes repetition
    float length_penalty = 1.0;       // >1.0 encourages longer sequences
    int num_return_sequences = 1;
    bool use_cache = true;
    int early_stopping = true;
    int no_repeat_ngram_size = 0;    // Prevent repetition of n-grams
    std::unordered_set<int> forbidden_tokens;
    std::unordered_set<int> forced_tokens;
};

// Beam search node
struct BeamNode {
    std::vector<int> tokens;
    double score;
    std::shared_ptr<BeamNode> parent;
    int token_id;
    int length;
    
    BeamNode() : score(0.0), parent(nullptr), token_id(-1), length(0) {}
    BeamNode(std::shared_ptr<BeamNode> p, int token, double score) 
        : tokens(p->tokens), score(score), parent(p), token_id(token), length(p->length + 1) {
        tokens.push_back(token);
    }
    
    bool operator<(const BeamNode& other) const {
        return score < other.score;
    }
};

class AdvancedGenerator {
public:
    AdvancedGenerator(MiniGPT& model, ByteLevelBPETokenizer& tokenizer)
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
    
    void set_config(const GenerationConfig& cfg) { config = cfg; }
    const GenerationConfig& get_config() const { return config; }
    
    // Generate with beam search
    std::vector<std::string> generate(const std::string& prompt, bool add_bos = false, bool add_eos = false) {
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
            // Use greedy or sampling
            return greedy_generate(input_ids);
        } else {
            // Use beam search
            return beam_search_generate(input_ids);
        }
    }
    
private:
    // Greedy generation with sampling
    std::vector<std::string> greedy_generate(const std::vector<int>& input_ids) {
        std::vector<int> current_tokens = input_ids;
        int eos_id = tokenizer.get_eos_token_id();
        int generated_count = 0;
        
        for (int pos = (int)input_ids.size(); pos < config.max_length; ++pos) {
            int token_id = current_tokens.back();
            
            // Forward
            std::vector<Value::Ptr> logits;
            if (config.use_cache) {
                logits = model.forward_incremental(token_id, pos);
            } else {
                // Full forward (not incremental)
                // This is simplified - in practice you'd use full forward
                logits = model.forward_incremental(token_id, pos);
            }
            
            // Apply repetition penalty
            if (config.repetition_penalty > 1.0) {
                apply_repetition_penalty(logits, current_tokens, config.repetition_penalty);
            }
            
            // Sample next token
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
            
            // Check forbidden tokens
            if (config.forbidden_tokens.count(next_token) > 0) {
                next_token = sampling::greedy_sample(logits);
            }
            
            // Stop if EOS
            if (next_token == eos_id && pos >= config.min_length) {
                break;
            }
            
            current_tokens.push_back(next_token);
            generated_count++;
        }
        
        std::string text = tokenizer.decode(current_tokens);
        return {text};
    }
    
    // Beam search generation
    std::vector<std::string> beam_search_generate(const std::vector<int>& input_ids) {
        int eos_id = tokenizer.get_eos_token_id();
        int beam_width = config.num_beams;
        int max_len = config.max_length;
        
        // Initialize beam with first token
        std::priority_queue<BeamNode> beam;
        BeamNode root;
        root.tokens = input_ids;
        root.length = input_ids.size();
        root.score = 0.0;
        root.parent = nullptr;
        
        std::vector<BeamNode> completed;
        
        // Start beam search
        for (int pos = (int)input_ids.size(); pos < max_len; ++pos) {
            std::priority_queue<BeamNode> new_beam;
            std::vector<BeamNode> all_candidates;
            
            // Expand each beam
            while (!beam.empty()) {
                BeamNode node = beam.top();
                beam.pop();
                
                // If EOS token reached or length exceeds, add to completed
                if (!node.tokens.empty() && node.tokens.back() == eos_id) {
                    completed.push_back(node);
                    continue;
                }
                
                // Get logits for current token
                int token_id = node.tokens.back();
                std::vector<Value::Ptr> logits;
                if (config.use_cache) {
                    // In beam search, we need to handle cache properly
                    // Simplified: reinitialize cache for each beam
                    // In production, you'd maintain separate caches per beam
                    model.init_cache();
                    for (int i = 0; i < (int)node.tokens.size(); ++i) {
                        logits = model.forward_incremental(node.tokens[i], i);
                    }
                } else {
                    logits = model.forward_incremental(token_id, pos);
                }
                
                // Apply repetition penalty
                if (config.repetition_penalty > 1.0) {
                    apply_repetition_penalty(logits, node.tokens, config.repetition_penalty);
                }
                
                // Apply temperature
                if (config.temperature != 1.0) {
                    for (auto& l : logits) {
                        l->data /= config.temperature;
                    }
                }
                
                // Get top-k candidates
                std::vector<std::pair<int, double>> candidates;
                candidates.reserve(logits.size());
                for (size_t i = 0; i < logits.size(); ++i) {
                    candidates.emplace_back(i, logits[i]->data);
                }
                
                std::sort(candidates.begin(), candidates.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
                
                int num_candidates = std::min(beam_width * 2, (int)candidates.size());
                
                // Add candidates to new beam
                for (int i = 0; i < num_candidates; ++i) {
                    int token = candidates[i].first;
                    double score = candidates[i].second;
                    
                    // Check forbidden
                    if (config.forbidden_tokens.count(token) > 0) continue;
                    
                    // Apply length penalty
                    double length_penalty = config.length_penalty;
                    if (config.length_penalty != 1.0) {
                        double len_factor = std::pow(5.0 + node.length, config.length_penalty) / 
                                           std::pow(6.0, config.length_penalty);
                        score *= len_factor;
                    }
                    
                    BeamNode child(std::make_shared<BeamNode>(node), token, node.score + score);
                    new_beam.push(child);
                }
            }
            
            // Keep top beam_width
            beam = new_beam;
            
            // Early stopping
            if (config.early_stopping && beam.empty()) break;
        }
        
        // Collect all completed sequences
        while (!beam.empty()) {
            completed.push_back(beam.top());
            beam.pop();
        }
        
        // Sort by score and select top sequences
        std::sort(completed.begin(), completed.end(),
                  [](const auto& a, const auto& b) { return a.score > b.score; });
        
        std::vector<std::string> results;
        int num_results = std::min(config.num_return_sequences, (int)completed.size());
        for (int i = 0; i < num_results; ++i) {
            results.push_back(tokenizer.decode(completed[i].tokens));
        }
        
        return results;
    }
    
    // Apply repetition penalty to logits
    void apply_repetition_penalty(std::vector<Value::Ptr>& logits, 
                                   const std::vector<int>& tokens,
                                   float penalty) {
        std::unordered_set<int> seen_tokens(tokens.begin(), tokens.end());
        for (auto& logit : logits) {
            // We need the index of this logit - simplified approach
            // In production, you'd have a way to map logit to token id
        }
        // Implementation depends on how logits are structured
        // This is a placeholder
    }
    
    MiniGPT& model;
    ByteLevelBPETokenizer& tokenizer;
    GenerationConfig config;
};

} // namespace advanced_generation