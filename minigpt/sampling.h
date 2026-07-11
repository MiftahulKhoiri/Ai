// sampling.h
#pragma once
#include "value.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace sampling {
    // Greedy decoding (argmax)
    inline int greedy_sample(const std::vector<Value::Ptr>& logits) {
        if (logits.empty()) return -1;
        int max_idx = 0;
        double max_val = logits[0]->data;
        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i]->data > max_val) {
                max_val = logits[i]->data;
                max_idx = i;
            }
        }
        return max_idx;
    }
    
    // Top-K sampling
    inline std::vector<int> top_k_sample(const std::vector<Value::Ptr>& logits, 
                                         int k, float temperature = 1.0f) {
        if (logits.empty()) return {};
        if (k <= 0) k = logits.size();
        
        std::vector<std::pair<int, double>> scored;
        scored.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            scored.emplace_back(i, logits[i]->data);
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
    
    // Top-P (Nucleus) sampling
    inline std::vector<int> top_p_sample(const std::vector<Value::Ptr>& logits,
                                         float p, float temperature = 1.0f) {
        if (logits.empty()) return {};
        if (p <= 0.0f) p = 1.0f;
        if (p >= 1.0f) return top_k_sample(logits, logits.size(), temperature);
        
        std::vector<std::pair<int, double>> scored;
        scored.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            scored.emplace_back(i, logits[i]->data);
        }
        
        if (temperature != 1.0f && temperature > 0.0f) {
            for (auto& pv : scored) {
                pv.second = std::exp(pv.second / temperature);
            }
        }
        
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        double sum = 0.0;
        for (const auto& pv : scored) {
            sum += pv.second;
        }
        
        if (sum == 0.0 || std::isnan(sum)) {
            return {scored[0].first};
        }
        
        // Normalize and find top-p
        double cumsum = 0.0;
        size_t idx = 0;
        for (; idx < scored.size(); ++idx) {
            cumsum += scored[idx].second / sum;
            if (cumsum >= p) break;
        }
        
        if (idx < scored.size()) {
            scored.resize(idx + 1);
        }
        
        // Re-normalize the selected tokens
        double new_sum = 0.0;
        for (const auto& pv : scored) {
            new_sum += pv.second;
        }
        
        if (new_sum == 0.0) {
            return {scored[0].first};
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        double r = dis(gen);
        cumsum = 0.0;
        for (const auto& pv : scored) {
            cumsum += pv.second / new_sum;
            if (r <= cumsum) {
                return {pv.first};
            }
        }
        return {scored.back().first};
    }
    
    // Batched sampling
    inline std::vector<int> batch_sample(const std::vector<std::vector<Value::Ptr>>& logits_batch,
                                         const std::string& method = "top_p",
                                         float temperature = 0.8f,
                                         int top_k = 40,
                                         float top_p = 0.9f) {
        std::vector<int> results;
        results.reserve(logits_batch.size());
        
        for (const auto& logits : logits_batch) {
            if (method == "greedy") {
                results.push_back(greedy_sample(logits));
            } else if (method == "top_k") {
                auto sampled = top_k_sample(logits, top_k, temperature);
                results.push_back(sampled.empty() ? 0 : sampled[0]);
            } else { // top_p
                auto sampled = top_p_sample(logits, top_p, temperature);
                results.push_back(sampled.empty() ? 0 : sampled[0]);
            }
        }
        return results;
    }
}