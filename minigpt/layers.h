// layers.h
#pragma once
#include "value.h"
#include "utils.h"
#include <vector>
#include <memory>

// ============================================================
// HELPER FUNCTIONS
// ============================================================

std::vector<ValuePtr> linear_forward(
    const std::vector<ValuePtr>& x,
    const std::vector<ValuePtr>& weight,
    const std::vector<ValuePtr>& bias);

double random_normal(double mean, double stddev);

// ============================================================
// EMBEDDING
// ============================================================

class Embedding {
public:
    std::vector<ValuePtr> weight;  // flat layout: vocab_size * d_model
    int vocab_size;
    int d_model;

    Embedding(int vocab_size, int d_model);
    std::vector<std::vector<ValuePtr>> forward(const std::vector<int>& ids);
};

// ============================================================
// POSITIONAL EMBEDDING
// ============================================================

class PositionalEmbedding {
public:
    std::vector<std::vector<ValuePtr>> pos_encoding;
    int max_len;
    int d_model;

    PositionalEmbedding(int max_len, int d_model);
    std::vector<ValuePtr> forward(int seq_len);
};

// ============================================================
// DROPOUT
// ============================================================

class Dropout {
public:
    double p;
    bool training;

    Dropout(double p = 0.1);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
};

// ============================================================
// LAYER NORM
// ============================================================

class LayerNorm {
public:
    int d_model;
    double eps;
    ValuePtr weight;
    ValuePtr bias;

    LayerNorm(int d_model, double eps = 1e-5);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
};

// ============================================================
// MULTI-HEAD SELF ATTENTION
// ============================================================

class MultiHeadSelfAttention {
public:
    int d_model;
    int n_heads;
    double dropout_p;
    int max_len;

    std::vector<ValuePtr> weight_q, weight_k, weight_v, weight_o;
    std::vector<ValuePtr> bias_q, bias_k, bias_v, bias_o;
    Dropout attn_dropout;

    MultiHeadSelfAttention(int d_model, int n_heads, double dropout = 0.1, int max_len = 1024);
    std::vector<std::vector<ValuePtr>> forward(
        const std::vector<std::vector<ValuePtr>>& x,
        const std::vector<int>& pad_mask = {});
};

// ============================================================
// FEED FORWARD
// ============================================================

class FeedForward {
public:
    int d_model;
    int d_ff;
    double dropout_p;

    std::vector<ValuePtr> w1, w2;
    std::vector<ValuePtr> b1, b2;
    Dropout ff_dropout;

    FeedForward(int d_model, int d_ff, double dropout = 0.1);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
};

// ============================================================
// TRANSFORMER BLOCK
// ============================================================

class TransformerBlock {
public:
    int d_model;
    int n_heads;
    int d_ff;
    double dropout_p;
    int max_len;

    MultiHeadSelfAttention attn;
    FeedForward ff;
    LayerNorm ln1, ln2;
    Dropout attn_dropout;

    TransformerBlock(int d_model, int n_heads, int d_ff, double dropout = 0.1, int max_len = 1024);
    std::vector<std::vector<ValuePtr>> forward(
        const std::vector<std::vector<ValuePtr>>& x,
        const std::vector<int>& pad_mask = {});
};

// ============================================================
// LINEAR (Final Projection)
// ============================================================

class Linear {
public:
    int in_features;
    int out_features;
    std::vector<ValuePtr> weight;
    std::vector<ValuePtr> bias;

    Linear(int in_features, int out_features);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x);
};