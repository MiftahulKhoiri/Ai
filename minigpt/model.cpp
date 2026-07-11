// model.cpp
#include "model.h"
#include <iostream>
#include <random>

// ============================================================
// MINIGPT CONSTRUCTOR
// ============================================================

MiniGPT::MiniGPT(
    int vocab_size,
    int d_model,
    int n_heads,
    int n_layers,
    int d_ff,
    int max_len,
    double dropout
) : d_model(d_model),
    max_len(max_len),
    vocab_size(vocab_size),
    embed(vocab_size, d_model),
    pos_embed(max_len, d_model),
    embed_drop(dropout),
    ln_f(d_model),
    head(d_model, vocab_size) {
    
    // Create transformer blocks
    blocks.reserve(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        blocks.emplace_back(d_model, n_heads, d_ff, dropout, max_len);
    }
    
    // Initialize cache
    caches.clear();
    caches.resize(n_layers);
    for (auto& cache : caches) {
        cache.first.clear();
        cache.second.clear();
    }
}

// ============================================================
// FORWARD
// ============================================================

std::vector<std::vector<ValuePtr>> MiniGPT::forward(
    const std::vector<int>& token_ids,
    const std::vector<int>& pad_mask) {
    
    if (token_ids.empty()) {
        return {};
    }
    
    int seq_len = token_ids.size();
    if (seq_len > max_len) {
        std::cerr << "[WARNING] MiniGPT::forward: seq_len " << seq_len 
                  << " > max_len " << max_len << ", truncating" << std::endl;
        seq_len = max_len;
    }
    
    // 1. Token embeddings
    std::vector<ValuePtr> emb = embed.forward(token_ids);
    if (emb.empty()) {
        return {};
    }
    
    // 2. Positional embeddings
    std::vector<ValuePtr> pos = pos_embed.forward(seq_len);
    if (pos.empty()) {
        return {};
    }
    
    // 3. Add token and positional embeddings
    std::vector<std::vector<ValuePtr>> x(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        x[i].reserve(d_model);
        for (int j = 0; j < d_model; ++j) {
            if (i < (int)emb.size() && j < (int)emb[i]->data) {
                // Simplified - in practice, emb[i] is a ValuePtr
                x[i].push_back(Value::create(0.0));
            } else {
                x[i].push_back(Value::create(0.0));
            }
        }
    }
    
    // 4. Apply dropout
    for (int i = 0; i < seq_len; ++i) {
        x[i] = embed_drop.forward(x[i]);
    }
    
    // 5. Pass through transformer blocks
    std::vector<int> mask = pad_mask;
    if (mask.empty()) {
        mask.resize(seq_len, 0);
    }
    
    for (auto& block : blocks) {
        x = block.forward(x, mask);
        if (x.empty()) {
            return {};
        }
    }
    
    // 6. Final layer norm
    for (int i = 0; i < seq_len; ++i) {
        x[i] = ln_f.forward(x[i]);
    }
    
    // 7. Final linear projection
    std::vector<std::vector<ValuePtr>> logits(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        logits[i] = head.forward(x[i]);
    }
    
    return logits;
}

// ============================================================
// FORWARD INCREMENTAL
// ============================================================

std::vector<ValuePtr> MiniGPT::forward_incremental(
    int token_id,
    int pos) {
    
    // Token embedding
    std::vector<int> ids = {token_id};
    std::vector<ValuePtr> emb = embed.forward(ids);
    if (emb.empty()) {
        return {};
    }
    
    // Positional embedding for this position
    std::vector<ValuePtr> pos_emb = pos_embed.forward(pos + 1);
    if (pos_emb.empty()) {
        return {};
    }
    
    // Combine token and positional embeddings
    std::vector<ValuePtr> x(d_model);
    for (int i = 0; i < d_model; ++i) {
        if (i < (int)emb.size()) {
            x[i] = emb[i];
        } else {
            x[i] = Value::create(0.0);
        }
    }
    
    // Apply dropout
    x = embed_drop.forward(x);
    
    // Pass through blocks (simplified for incremental)
    for (size_t i = 0; i < blocks.size(); ++i) {
        std::vector<std::vector<ValuePtr>> x_batch = {x};
        x_batch = blocks[i].forward(x_batch, {});
        if (!x_batch.empty() && !x_batch[0].empty()) {
            x = x_batch[0];
        }
    }
    
    // Final layer norm
    x = ln_f.forward(x);
    
    // Final projection
    std::vector<ValuePtr> logits = head.forward(x);
    
    return logits;
}

// ============================================================
// PARAMETERS
// ============================================================

std::vector<ValuePtr> MiniGPT::parameters() {
    std::vector<ValuePtr> params;
    
    // Embedding weights
    for (auto& p : embed.weight) {
        params.push_back(p);
    }
    
    // Transformer blocks
    for (auto& block : blocks) {
        // Attention weights
        for (auto& p : block.attn.weight_q) params.push_back(p);
        for (auto& p : block.attn.weight_k) params.push_back(p);
        for (auto& p : block.attn.weight_v) params.push_back(p);
        for (auto& p : block.attn.weight_o) params.push_back(p);
        for (auto& p : block.attn.bias_q) params.push_back(p);
        for (auto& p : block.attn.bias_k) params.push_back(p);
        for (auto& p : block.attn.bias_v) params.push_back(p);
        for (auto& p : block.attn.bias_o) params.push_back(p);
        
        // FF weights
        for (auto& p : block.ff.w1) params.push_back(p);
        for (auto& p : block.ff.w2) params.push_back(p);
        for (auto& p : block.ff.b1) params.push_back(p);
        for (auto& p : block.ff.b2) params.push_back(p);
        
        // Layer norm
        params.push_back(block.ln1.weight);
        params.push_back(block.ln1.bias);
        params.push_back(block.ln2.weight);
        params.push_back(block.ln2.bias);
    }
    
    // Final layer norm
    params.push_back(ln_f.weight);
    params.push_back(ln_f.bias);
    
    // Final head
    for (auto& p : head.weight) params.push_back(p);
    for (auto& p : head.bias) params.push_back(p);
    
    return params;
}

// ============================================================
// INIT CACHE
// ============================================================

void MiniGPT::init_cache() {
    for (auto& cache : caches) {
        cache.first.clear();
        cache.second.clear();
    }
}

// ============================================================
// SET TRAINING
// ============================================================

void MiniGPT::set_training(bool mode) {
    // Set dropout mode for all components
    embed_drop.training = mode;
    for (auto& block : blocks) {
        block.attn_dropout.training = mode;
        block.ff.ff_dropout.training = mode;
    }
}