// generation.cpp
#include "generation.h"
#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

int argmax(const std::vector<Value::Ptr>& logits) {
    if (logits.empty()) return -1;
    
    int max_idx = 0;
    double max_val = logits[0]->data;
    
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i]->data > max_val) {
            max_val = logits[i]->data;
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

std::vector<int> sample_top_k(const std::vector<Value::Ptr>& logits, int k, float temperature) {
    if (logits.empty()) return {};
    if (k <= 0) k = static_cast<int>(logits.size());
    
    std::vector<std::pair<int, double>> scored;
    scored.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scored.push_back({static_cast<int>(i), logits[i]->data});
    }
    
    std::sort(scored.begin(), scored.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    if (k < (int)scored.size()) {
        scored.resize(k);
    }
    
    if (temperature != 1.0f && temperature > 0.0f) {
        for (auto& p : scored) {
            p.second = std::exp(p.second / temperature);
        }
    }
    
    double sum = 0.0;
    for (const auto& p : scored) {
        sum += p.second;
    }
    
    if (sum == 0.0 || std::isnan(sum)) {
        return {scored[0].first};
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    double r = dis(gen);
    double cumsum = 0.0;
    for (const auto& p : scored) {
        cumsum += p.second / sum;
        if (r <= cumsum) {
            return {p.first};
        }
    }
    
    return {scored.back().first};
}

void generate(MiniGPT& model, 
              ByteLevelBPETokenizer& tokenizer,
              const std::string& prompt, 
              int max_tokens,
              bool add_bos,
              bool add_eos) {
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated: ";

    std::vector<int> input_ids = tokenizer.encode(prompt, add_bos, add_eos);
    if (input_ids.empty()) {
        std::cerr << "Prompt kosong atau tidak bisa di-encode.\n";
        return;
    }

    model.set_training(false);
    model.init_cache();

    int last_token = input_ids.back();
    
    for (int pos = 0; pos < max_tokens; ++pos) {
        int token_id;
        
        if (pos < (int)input_ids.size()) {
            token_id = input_ids[pos];
        } else {
            token_id = last_token;
        }

        std::vector<Value::Ptr> logits = model.forward_incremental(token_id, pos);
        
        if (pos >= (int)input_ids.size()) {
            int next_token = argmax(logits);
            
            // Cek EOS
            int eos_id = tokenizer.get_eos_token_id();
            if (eos_id >= 0 && next_token == eos_id) {
                break;
            }
            
            std::cout << tokenizer.decode({next_token}) << std::flush;
            last_token = next_token;
        }
    }
    std::cout << "\n";
}