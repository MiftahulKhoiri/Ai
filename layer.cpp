#include "layers.h"
#include <cmath>
#include <algorithm>

// ============================================================
// DROPOUT
// ============================================================
Dropout::Dropout(double p) : p(p), training(true), rng(std::random_device{}()) {}

std::vector<ValuePtr> Dropout::forward(const std::vector<ValuePtr>& x) {
    if (!training || p <= 0) return x;
    double keep = 1.0 - p;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<ValuePtr> out;
    for (auto& v : x) {
        if (dist(rng) < keep)
            out.push_back(v / keep);  // operator/ (ValuePtr, double)
        else
            out.push_back(v * 0.0);
    }
    return out;
}

// ============================================================
// LINEAR
// ============================================================
Linear::Linear(int n_in, int n_out, bool bias) : use_bias(bias) {
    double scale = 1.0 / std::sqrt(n_in);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-scale, scale);
    W.resize(n_out, std::vector<ValuePtr>(n_in));
    for (int i = 0; i < n_out; ++i)
        for (int j = 0; j < n_in; ++j)
            W[i][j] = Value::create(dist(rng));
    if (bias)
        b.resize(n_out, Value::create(0.0));
}

std::vector<ValuePtr> Linear::forward(const std::vector<ValuePtr>& x) {
    std::vector<ValuePtr> out(W.size());
    for (size_t i = 0; i < W.size(); ++i) {
        ValuePtr s = Value::create(0.0);
        for (size_t j = 0; j < x.size(); ++j)
            s = s + W[i][j] * x[j];
        out[i] = s;
    }
    if (use_bias)
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = out[i] + b[i];
    return out;
}

std::vector<ValuePtr> Linear::parameters() {
    std::vector<ValuePtr> p;
    for (auto& row : W) for (auto& w : row) p.push_back(w);
    if (use_bias) for (auto& bi : b) p.push_back(bi);
    return p;
}

// ============================================================
// EMBEDDING
// ============================================================
Embedding::Embedding(int vocab_size, int d_model, double scale) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-scale, scale);
    table.resize(vocab_size, std::vector<ValuePtr>(d_model));
    for (int i = 0; i < vocab_size; ++i)
        for (int j = 0; j < d_model; ++j)
            table[i][j] = Value::create(dist(rng));
}

std::vector<ValuePtr> Embedding::forward(int idx) {
    std::vector<ValuePtr> out;
    for (auto& v : table[idx])
        out.push_back(v + 0.0);  // fresh node
    return out;
}

std::vector<ValuePtr> Embedding::parameters() {
    std::vector<ValuePtr> p;
    for (auto& row : table) for (auto& v : row) p.push_back(v);
    return p;
}

// ============================================================
// POSITIONAL EMBEDDING
// ============================================================
PositionalEmbedding::PositionalEmbedding(int max_len, int d_model, double scale) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-scale, scale);
    table.resize(max_len, std::vector<ValuePtr>(d_model));
    for (int i = 0; i < max_len; ++i)
        for (int j = 0; j < d_model; ++j)
            table[i][j] = Value::create(dist(rng));
}

std::vector<ValuePtr> PositionalEmbedding::forward(int pos) {
    std::vector<ValuePtr> out;
    for (auto& v : table[pos])
        out.push_back(v + 0.0);
    return out;
}

std::vector<ValuePtr> PositionalEmbedding::parameters() {
    std::vector<ValuePtr> p;
    for (auto& row : table) for (auto& v : row) p.push_back(v);
    return p;
}

// ============================================================
// LAYER NORMALIZATION
// ============================================================
LayerNorm::LayerNorm(int dim, double eps) : eps(eps) {
    for (int i = 0; i < dim; ++i) gamma.push_back(Value::create(1.0));
    for (int i = 0; i < dim; ++i) beta.push_back(Value::create(0.0));
}

std::vector<ValuePtr> LayerNorm::forward(const std::vector<ValuePtr>& x) {
    size_t n = x.size();
    ValuePtr mean = Value::create(0.0);
    for (auto& xi : x) mean = mean + xi;
    mean = mean / (double)n;
    ValuePtr var = Value::create(0.0);
    for (auto& xi : x) {
        auto diff = xi - mean;
        var = var + diff * diff;
    }
    var = var / (double)n;
    auto std = sqrt(var + eps);
    std::vector<ValuePtr> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = gamma[i] * ((x[i] - mean) / std) + beta[i];
    return out;
}

std::vector<ValuePtr> LayerNorm::parameters() {
    auto p = gamma;
    p.insert(p.end(), beta.begin(), beta.end());
    return p;
}

// ============================================================
// MULTI-HEAD SELF ATTENTION
// ============================================================
MultiHeadSelfAttention::MultiHeadSelfAttention(int d_model, int n_heads, double dropout)
    : d_model(d_model), n_heads(n_heads), d_head(d_model / n_heads),
      Wq(d_model, d_model, false), Wk(d_model, d_model, false),
      Wv(d_model, d_model, false), Wo(d_model, d_model, false), drop(dropout) {}

std::vector<std::vector<ValuePtr>> MultiHeadSelfAttention::forward(
        const std::vector<std::vector<ValuePtr>>& X,
        const std::vector<int>& pad_mask) {
    int seq_len = X.size();
    std::vector<std::vector<ValuePtr>> Q(seq_len), K(seq_len), V(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        Q[i] = Wq.forward(X[i]);
        K[i] = Wk.forward(X[i]);
        V[i] = Wv.forward(X[i]);
    }
    double scale = 1.0 / std::sqrt(d_head);
    std::vector<std::vector<ValuePtr>> outputs(seq_len, std::vector<ValuePtr>(d_model, Value::create(0.0)));
    for (int h = 0; h < n_heads; ++h) {
        int s = h * d_head, e = s + d_head;
        for (int i = 0; i < seq_len; ++i) {
            std::vector<int> valid_js;
            for (int j = 0; j <= i; ++j) if (pad_mask[j] == 1) valid_js.push_back(j);
            if (valid_js.empty()) valid_js.push_back(i);
            std::vector<ValuePtr> scores;
            for (int j : valid_js) {
                // extract head vectors
                std::vector<ValuePtr> qh(Q[i].begin()+s, Q[i].begin()+e);
                std::vector<ValuePtr> kh(K[j].begin()+s, K[j].begin()+e);
                ValuePtr dot_val = Value::create(0.0);
                for (int d = 0; d < d_head; ++d)
                    dot_val = dot_val + qh[d] * kh[d];
                scores.push_back(dot_val * scale);
            }
            auto weights = softmax(scores);
            std::vector<ValuePtr> head_out(d_head, Value::create(0.0));
            for (size_t w_idx = 0; w_idx < valid_js.size(); ++w_idx) {
                int j = valid_js[w_idx];
                for (int d = 0; d < d_head; ++d)
                    head_out[d] = head_out[d] + weights[w_idx] * V[j][s + d];
            }
            for (int d = 0; d < d_head; ++d)
                outputs[i][s + d] = head_out[d];
        }
    }
    std::vector<std::vector<ValuePtr>> out;
    for (int i = 0; i < seq_len; ++i)
        out.push_back(Wo.forward(outputs[i]));
    for (int i = 0; i < seq_len; ++i)
        out[i] = drop.forward(out[i]);
    return out;
}

std::vector<ValuePtr> MultiHeadSelfAttention::forward_incremental(
        const std::vector<ValuePtr>& x,
        std::vector<std::vector<ValuePtr>>& K_cache,
        std::vector<std::vector<ValuePtr>>& V_cache) {
    auto q = Wq.forward(x);
    auto k = Wk.forward(x);
    auto v = Wv.forward(x);
    K_cache.push_back(k);
    V_cache.push_back(v);
    double scale = 1.0 / std::sqrt(d_head);
    std::vector<ValuePtr> out(d_model, Value::create(0.0));
    for (int h = 0; h < n_heads; ++h) {
        int s = h * d_head, e = s + d_head;
        std::vector<ValuePtr> qh(q.begin()+s, q.begin()+e);
        std::vector<ValuePtr> scores;
        for (size_t pos = 0; pos < K_cache.size(); ++pos) {
            std::vector<ValuePtr> kh(K_cache[pos].begin()+s, K_cache[pos].begin()+e);
            ValuePtr dot_val = Value::create(0.0);
            for (int d = 0; d < d_head; ++d)
                dot_val = dot_val + qh[d] * kh[d];
            scores.push_back(dot_val * scale);
        }
        auto weights = softmax(scores);
        std::vector<ValuePtr> head_out(d_head, Value::create(0.0));
        for (size_t pos = 0; pos < V_cache.size(); ++pos) {
            for (int d = 0; d < d_head; ++d)
                head_out[d] = head_out[d] + weights[pos] * V_cache[pos][s + d];
        }
        for (int d = 0; d < d_head; ++d)
            out[s + d] = head_out[d];
    }
    auto out2 = Wo.forward(out);
    return drop.forward(out2);
}

std::vector<ValuePtr> MultiHeadSelfAttention::parameters() {
    auto p = Wq.parameters();
    auto p2 = Wk.parameters();
    auto p3 = Wv.parameters();
    auto p4 = Wo.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), p3.begin(), p3.end());
    p.insert(p.end(), p4.begin(), p4.end());
    return p;
}

void MultiHeadSelfAttention::set_training(bool mode) { drop.training = mode; }

// ============================================================
// FEED-FORWARD
// ============================================================
FeedForward::FeedForward(int d_model, int d_ff, double dropout)
    : fc1(d_model, d_ff), fc2(d_ff, d_model), drop(dropout) {}

std::vector<ValuePtr> FeedForward::forward(const std::vector<ValuePtr>& x) {
    auto h = fc1.forward(x);
    for (auto& v : h) v = gelu(v);
    h = drop.forward(h);
    return fc2.forward(h);
}

std::vector<ValuePtr> FeedForward::parameters() {
    auto p = fc1.parameters();
    auto p2 = fc2.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}

void FeedForward::set_training(bool mode) { drop.training = mode; }

// ============================================================
// TRANSFORMER BLOCK
// ============================================================
TransformerBlock::TransformerBlock(int d_model, int n_heads, int d_ff, double dropout)
    : ln1(d_model), ln2(d_model), attn(d_model, n_heads, dropout), ff(d_model, d_ff, dropout) {}

std::vector<std::vector<ValuePtr>> TransformerBlock::forward(
        const std::vector<std::vector<ValuePtr>>& X,
        const std::vector<int>& pad_mask) {
    int seq_len = X.size();
    std::vector<std::vector<ValuePtr>> ln1_out;
    for (auto& x : X) ln1_out.push_back(ln1.forward(x));
    auto attn_out = attn.forward(ln1_out, pad_mask);
    std::vector<std::vector<ValuePtr>> X2(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        X2[i].resize(X[i].size());
        for (size_t j = 0; j < X[i].size(); ++j)
            X2[i][j] = X[i][j] + attn_out[i][j];
    }
    std::vector<std::vector<ValuePtr>> ln2_out;
    for (auto& x : X2) ln2_out.push_back(ln2.forward(x));
    std::vector<std::vector<ValuePtr>> ff_out;
    for (auto& x : ln2_out) ff_out.push_back(ff.forward(x));
    std::vector<std::vector<ValuePtr>> out(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        out[i].resize(X[i].size());
        for (size_t j = 0; j < X[i].size(); ++j)
            out[i][j] = X2[i][j] + ff_out[i][j];
    }
    return out;
}

std::vector<ValuePtr> TransformerBlock::forward_incremental(
        const std::vector<ValuePtr>& x,
        std::vector<std::vector<ValuePtr>>& K_cache,
        std::vector<std::vector<ValuePtr>>& V_cache) {
    auto ln1_out = ln1.forward(x);
    auto attn_out = attn.forward_incremental(ln1_out, K_cache, V_cache);
    std::vector<ValuePtr> x2;
    for (size_t i = 0; i < x.size(); ++i)
        x2.push_back(x[i] + attn_out[i]);
    auto ln2_out = ln2.forward(x2);
    auto ff_out = ff.forward(ln2_out);
    std::vector<ValuePtr> out;
    for (size_t i = 0; i < x2.size(); ++i)
        out.push_back(x2[i] + ff_out[i]);
    return out;
}

std::vector<ValuePtr> TransformerBlock::parameters() {
    auto p = ln1.parameters();
    auto p2 = attn.parameters();
    auto p3 = ln2.parameters();
    auto p4 = ff.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), p3.begin(), p3.end());
    p.insert(p.end(), p4.begin(), p4.end());
    return p;
}

void TransformerBlock::set_training(bool mode) {
    attn.set_training(mode);
    ff.set_training(mode);
}