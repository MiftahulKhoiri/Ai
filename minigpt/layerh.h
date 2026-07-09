#pragma once
#include "value.h"
#include <vector>
#include <memory>
#include <random>

// ============================================================
// DROPOUT
// ============================================================
struct Dropout {
    double p;
    bool training;
    std::mt19937 rng;
    Dropout(double p = 0.1);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
};

// ============================================================
// LINEAR
// ============================================================
struct Linear {
    std::vector<std::vector<ValuePtr>> W;
    std::vector<ValuePtr> b;
    bool use_bias;

    Linear(int n_in, int n_out, bool bias = true);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
    std::vector<ValuePtr> parameters();
};

// ============================================================
// EMBEDDING
// ============================================================
struct Embedding {
    std::vector<std::vector<ValuePtr>> table;
    Embedding(int vocab_size, int d_model, double scale = 0.02);
    std::vector<ValuePtr> forward(int idx);
    std::vector<ValuePtr> parameters();
};

// ============================================================
// POSITIONAL EMBEDDING
// ============================================================
struct PositionalEmbedding {
    std::vector<std::vector<ValuePtr>> table;
    PositionalEmbedding(int max_len, int d_model, double scale = 0.02);
    std::vector<ValuePtr> forward(int pos);
    std::vector<ValuePtr> parameters();
};

// ============================================================
// LAYER NORMALIZATION
// ============================================================
struct LayerNorm {
    std::vector<ValuePtr> gamma, beta;
    double eps;
    LayerNorm(int dim, double eps = 1e-5);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
    std::vector<ValuePtr> parameters();
};

// ============================================================
// MULTI-HEAD SELF ATTENTION
// ============================================================
struct MultiHeadSelfAttention {
    int d_model, n_heads, d_head;
    Linear Wq, Wk, Wv, Wo;
    Dropout drop;

    MultiHeadSelfAttention(int d_model, int n_heads, double dropout = 0.1);
    std::vector<std::vector<ValuePtr>> forward(const std::vector<std::vector<ValuePtr>>& X,
                                               const std::vector<int>& pad_mask);
    std::vector<ValuePtr> forward_incremental(const std::vector<ValuePtr>& x,
                                             std::vector<std::vector<ValuePtr>>& K_cache,
                                             std::vector<std::vector<ValuePtr>>& V_cache);
    std::vector<ValuePtr> parameters();
    void set_training(bool mode);
};

// ============================================================
// FEED-FORWARD
// ============================================================
struct FeedForward {
    Linear fc1, fc2;
    Dropout drop;
    FeedForward(int d_model, int d_ff, double dropout = 0.1);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
    std::vector<ValuePtr> parameters();
    void set_training(bool mode);
};

// ============================================================
// TRANSFORMER BLOCK
// ============================================================
struct TransformerBlock {
    LayerNorm ln1, ln2;
    MultiHeadSelfAttention attn;
    FeedForward ff;

    TransformerBlock(int d_model, int n_heads, int d_ff, double dropout = 0.1);
    std::vector<std::vector<ValuePtr>> forward(const std::vector<std::vector<ValuePtr>>& X,
                                               const std::vector<int>& pad_mask);
    std::vector<ValuePtr> forward_incremental(const std::vector<ValuePtr>& x,
                                             std::vector<std::vector<ValuePtr>>& K_cache,
                                             std::vector<std::vector<ValuePtr>>& V_cache);
    std::vector<ValuePtr> parameters();
    void set_training(bool mode);
};