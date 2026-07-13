// model.cpp
#include "model.h"
#include <iostream>
#include <random>

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

    blocks.reserve(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        blocks.emplace_back(d_model, n_heads, d_ff, dropout, max_len);
    }

    caches.clear();
    caches.resize(n_layers);
    for (auto& cache : caches) {
        cache.first.clear();
        cache.second.clear();
    }
}

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

    std::vector<std::vector<ValuePtr>> emb = embed.forward(token_ids);
    if (emb.empty()) {
        return {};
    }

    std::vector<ValuePtr> pos = pos_embed.forward(seq_len);
    if (pos.empty()) {
        return {};
    }

    std::vector<std::vector<ValuePtr>> x(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        x[i].reserve(d_model);
        for (int j = 0; j < d_model; ++j) {
            int pos_idx = i * d_model + j;
            ValuePtr tok_val = (i < (int)emb.size() && j < (int)emb[i].size())
                                ? emb[i][j] : Value::create(0.0);
            ValuePtr pos_val = (pos_idx < (int)pos.size())
                                ? pos[pos_idx] : Value::create(0.0);
            x[i].push_back(tok_val + pos_val);
        }
    }

    for (int i = 0; i < seq_len; ++i) {
        x[i] = embed_drop.forward(x[i]);
    }

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

    for (int i = 0; i < seq_len; ++i) {
        x[i] = ln_f.forward(x[i]);
    }

    std::vector<std::vector<ValuePtr>> logits(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        logits[i] = head.forward(x[i]);
    }

    return logits;
}

// FIX (server paralel): logika inti forward_incremental sekarang ada di
// overload yang menerima "external_cache" secara eksplisit. Versi lama
// (2 argumen) tinggal delegasi ke overload ini pakai this->caches --
// jadi tidak ada duplikasi logika, dan jalur single-threaded lama
// (generate() CLI) tetap berfungsi seperti sebelumnya tanpa perubahan
// perilaku.
std::vector<ValuePtr> MiniGPT::forward_incremental(int token_id, int pos) {
    if (caches.size() != blocks.size()) {
        caches.resize(blocks.size());
    }
    return forward_incremental(token_id, pos, caches);
}

std::vector<ValuePtr> MiniGPT::forward_incremental(
    int token_id,
    int pos,
    std::vector<LayerCache>& external_cache) {

    std::vector<int> ids = {token_id};
    std::vector<std::vector<ValuePtr>> emb = embed.forward(ids);
    if (emb.empty()) {
        return {};
    }

    std::vector<ValuePtr> pos_emb = pos_embed.forward(pos + 1);
    if (pos_emb.empty()) {
        return {};
    }

    int pos_offset = pos * d_model;

    std::vector<ValuePtr> x(d_model);
    for (int i = 0; i < d_model; ++i) {
        ValuePtr tok_val = (i < (int)emb[0].size()) ? emb[0][i] : Value::create(0.0);
        int p_idx = pos_offset + i;
        ValuePtr pos_val = (p_idx < (int)pos_emb.size()) ? pos_emb[p_idx] : Value::create(0.0);
        x[i] = tok_val + pos_val;
    }

    x = embed_drop.forward(x);

    if (external_cache.size() != blocks.size()) {
        external_cache.resize(blocks.size());
    }

    for (size_t i = 0; i < blocks.size(); ++i) {
        x = blocks[i].forward_incremental(x, external_cache[i].first, external_cache[i].second);
        if (x.empty()) {
            return {};
        }
    }

    x = ln_f.forward(x);

    std::vector<ValuePtr> logits = head.forward(x);

    return logits;
}

std::vector<ValuePtr> MiniGPT::parameters() {
    std::vector<ValuePtr> params;

    for (auto& p : embed.weight) {
        params.push_back(p);
    }

    for (auto& block : blocks) {
        for (auto& p : block.attn.weight_q) params.push_back(p);
        for (auto& p : block.attn.weight_k) params.push_back(p);
        for (auto& p : block.attn.weight_v) params.push_back(p);
        for (auto& p : block.attn.weight_o) params.push_back(p);
        for (auto& p : block.attn.bias_q) params.push_back(p);
        for (auto& p : block.attn.bias_k) params.push_back(p);
        for (auto& p : block.attn.bias_v) params.push_back(p);
        for (auto& p : block.attn.bias_o) params.push_back(p);

        for (auto& p : block.ff.w1) params.push_back(p);
        for (auto& p : block.ff.w2) params.push_back(p);
        for (auto& p : block.ff.b1) params.push_back(p);
        for (auto& p : block.ff.b2) params.push_back(p);

        params.push_back(block.ln1.weight);
        params.push_back(block.ln1.bias);
        params.push_back(block.ln2.weight);
        params.push_back(block.ln2.bias);
    }

    params.push_back(ln_f.weight);
    params.push_back(ln_f.bias);

    for (auto& p : head.weight) params.push_back(p);
    for (auto& p : head.bias) params.push_back(p);

    return params;
}

void MiniGPT::init_cache() {
    for (auto& cache : caches) {
        cache.first.clear();
        cache.second.clear();
    }
}

void MiniGPT::set_training(bool mode) {
    embed_drop.training = mode;
    for (auto& block : blocks) {
        block.attn_dropout.training = mode;
        block.ff.ff_dropout.training = mode;
    }
}