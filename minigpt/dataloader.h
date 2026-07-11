// dataloader.h
#pragma once
#include "tokenizer.h"
#include "value.h"
#include <vector>
#include <random>
#include <algorithm>
#include <memory>

class DataLoader {
public:
    DataLoader(ByteLevelBPETokenizer& tokenizer, int batch_size = 8, int seq_len = 128)
        : tokenizer(tokenizer), batch_size(batch_size), seq_len(seq_len), current_idx(0) {}
    
    // Load text and create sequences
    void load_text(const std::string& text, bool add_bos = true, bool add_eos = true) {
        std::vector<int> tokens = tokenizer.encode(text, add_bos, add_eos);
        sequences.clear();
        
        for (size_t i = 0; i + seq_len <= tokens.size(); i += seq_len / 2) {
            std::vector<int> seq(tokens.begin() + i, tokens.begin() + i + seq_len);
            sequences.push_back(seq);
        }
        
        // Shuffle indices
        indices.resize(sequences.size());
        std::iota(indices.begin(), indices.end(), 0);
        current_idx = 0;
        shuffle();
    }
    
    // Load from multiple texts
    void load_texts(const std::vector<std::string>& texts, bool add_bos = true, bool add_eos = true) {
        sequences.clear();
        for (const auto& text : texts) {
            std::vector<int> tokens = tokenizer.encode(text, add_bos, add_eos);
            for (size_t i = 0; i + seq_len <= tokens.size(); i += seq_len / 2) {
                std::vector<int> seq(tokens.begin() + i, tokens.begin() + i + seq_len);
                sequences.push_back(seq);
            }
        }
        
        indices.resize(sequences.size());
        std::iota(indices.begin(), indices.end(), 0);
        current_idx = 0;
        shuffle();
    }
    
    // Get next batch
    std::vector<std::pair<std::vector<int>, std::vector<int>>> next_batch() {
        if (current_idx >= (int)indices.size()) {
            current_idx = 0;
            shuffle();
        }
        
        int batch_end = std::min(current_idx + batch_size, (int)indices.size());
        std::vector<std::pair<std::vector<int>, std::vector<int>>> batch;
        batch.reserve(batch_end - current_idx);
        
        for (int i = current_idx; i < batch_end; ++i) {
            const auto& seq = sequences[indices[i]];
            std::vector<int> input_ids(seq.begin(), seq.end() - 1);
            std::vector<int> target_ids(seq.begin() + 1, seq.end());
            batch.emplace_back(std::move(input_ids), std::move(target_ids));
        }
        
        current_idx = batch_end;
        return batch;
    }
    
    // Shuffle data
    void shuffle() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
    }
    
    // Get number of batches
    size_t num_batches() const {
        return (sequences.size() + batch_size - 1) / batch_size;
    }
    
    // Reset to beginning
    void reset() {
        current_idx = 0;
        shuffle();
    }
    
    // Get total sequences
    size_t size() const { return sequences.size(); }
    
private:
    ByteLevelBPETokenizer& tokenizer;
    int batch_size;
    int seq_len;
    std::vector<std::vector<int>> sequences;
    std::vector<int> indices;
    int current_idx;
};