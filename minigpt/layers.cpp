// layers.cpp
#include "layers.h"
#include "utils.h"
#include "optim.h"
#include <cmath>
#include <iostream>
#include <random>

// ============================================================
// HELPER FUNCTIONS
// ============================================================

static std::random_device rd;
static std::mt19937 gen(rd());

double random_normal(double mean, double stddev) {
    std::normal_distribution<> dist(mean, stddev);
    return dist(gen);
}

// Linear forward
std::vector<ValuePtr> linear_forward(
    const std::vector<ValuePtr>& x,
    const std::vector<ValuePtr>& weight,
    const std::vector<ValuePtr>& bias) {

    if (x.empty() || weight.empty()) {
        std::cerr << "[ERROR] linear_forward: empty input" << std::endl;
        return {};
    }

    int in_features = x.size();
    int out_features = weight.size() / in_features;

    if (out_features == 0) {
        std::cerr << "[ERROR] linear_forward: out_features=0" << std::endl;
        return {};
    }

    std::vector<ValuePtr> result(out_features);

    for (int i = 0; i < out_features; ++i) {
        result[i] = Value::create(0.0);
        for (int j = 0; j < in_features; ++j) {
            int idx = i * in_features + j;
            if (idx < (int)weight.size() && j < (int)x.size()) {
                result[i] = result[i] + (weight[idx] * x[j]);
            } else {
                std::cerr << "[ERROR] linear_forward: idx=" << idx << " out of range" << std::endl;
                return {};
            }
        }
        if (!bias.empty() && i < (int)bias.size()) {
            result[i] = result[i] + bias[i];
        }
    }

    return result;
}

// ============================================================
// EMBEDDING
// ============================================================

Embedding::Embedding(int vocab_size, int d_model) 
    : vocab_size(vocab_size), d_model(d_model) {
    weight.reserve((size_t)vocab_size * d_model);
    for (int i = 0; i < vocab_size * d_model; ++i) {
        weight.push_back(Value::create(random_normal(0, 0.02)));
    }
}

std::vector<std::vector<ValuePtr>> Embedding::forward(const std::vector<int>& ids) {
    std::vector<std::vector<ValuePtr>> result;
    result.reserve(ids.size());

    for (int id : ids) {
        if (id < 0 || id >= vocab_size) {
            std::cerr << "[ERROR] Embedding: id " << id << " out of range (vocab size " << vocab_size << ")" << std::endl;
            result.push_back(std::vector<ValuePtr>(d_model, Value::create(0.0)));
            continue;
        }
        std::vector<ValuePtr> vec;
        vec.reserve(d_model);
        for (int j = 0; j < d_model; ++j) {
            vec.push_back(weight[(size_t)id * d_model + j]);
        }
        result.push_back(vec);
    }
    return result;
}

// ============================================================
// POSITIONAL EMBEDDING
// ============================================================

PositionalEmbedding::PositionalEmbedding(int max_len, int d_model) 
    : max_len(max_len), d_model(d_model) {
    pos_encoding.reserve(max_len);
    for (int pos = 0; pos < max_len; ++pos) {
        std::vector<ValuePtr> encoding;
        encoding.reserve(d_model);
        for (int i = 0; i < d_model; ++i) {
            double val;
            if (i % 2 == 0) {
                val = std::sin(pos / std::pow(10000.0, i / (double)d_model));
            } else {
                val = std::cos(pos / std::pow(10000.0, (i - 1) / (double)d_model));
            }
            encoding.push_back(Value::create(val));
        }
        pos_encoding.push_back(encoding);
    }
}

std::vector<ValuePtr> PositionalEmbedding::forward(int seq_len) {
    if (seq_len > max_len) {
        std::cerr << "[WARNING] PositionalEmbedding: seq_len " << seq_len 
                  << " > max_len " << max_len << ", truncating" << std::endl;
        seq_len = max_len;
    }

    std::vector<ValuePtr> result;
    if (seq_len <= 0 || seq_len > (int)pos_encoding.size()) {
        return result;
    }

    for (int i = 0; i < seq_len; ++i) {
        for (const auto& v : pos_encoding[i]) {
            result.push_back(v);
        }
    }
    return result;
}

// ============================================================
// DROPOUT
// ============================================================

Dropout::Dropout(double p) : p(p), training(true) {}

std::vector<ValuePtr> Dropout::forward(const std::vector<ValuePtr>& x) {
    if (!training || p <= 0.0) {
        return x;
    }

    if (p >= 1.0) {
        std::vector<ValuePtr> result;
        result.reserve(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result.push_back(Value::create(0.0));
        }
        return result;
    }

    std::vector<ValuePtr> result;
    result.reserve(x.size());

    double scale = 1.0 / (1.0 - p);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (size_t i = 0; i < x.size(); ++i) {
        if (dist(gen) < p) {
            result.push_back(Value::create(0.0));
        } else {
            result.push_back(x[i] * Value::create(scale));
        }
    }
    return result;
}

// ============================================================
// LAYER NORM
// ============================================================

LayerNorm::LayerNorm(int d_model, double eps) 
    : d_model(d_model), eps(eps) {
    weight = Value::create(1.0);
    bias = Value::create(0.0);
}

std::vector<ValuePtr> LayerNorm::forward(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};

    ValuePtr mean = Value::create(0.0);
    for (const auto& v : x) {
        mean = mean + v;
    }
    mean = mean / Value::create((double)x.size());

    ValuePtr var = Value::create(0.0);
    for (const auto& v : x) {
        var = var + ((v - mean) * (v - mean));
    }
    var = var / Value::create((double)x.size());

    std::vector<ValuePtr> result;
    result.reserve(x.size());
    ValuePtr stddev = sqrt(var + Value::create(eps));

    for (const auto& v : x) {
        ValuePtr normalized = (v - mean) / stddev;
        result.push_back(normalized * weight + bias);
    }
    return result;
}

// ============================================================
// MULTI-HEAD SELF ATTENTION
// ============================================================

MultiHeadSelfAttention::MultiHeadSelfAttention(int d_model, int n_heads, double dropout, int max_len)
    : d_model(d_model), n_heads(n_heads), dropout_p(dropout), max_len(max_len),
      attn_dropout(dropout) {

    if (d_model % n_heads != 0) {
        std::cerr << "[ERROR] MultiHeadSelfAttention: d_model " << d_model 
                  << " not divisible by n_heads " << n_heads << std::endl;
        n_heads = 1;
        while (d_model % n_heads != 0) n_heads++;
        this->n_heads = n_heads;
    }

    int d_k = d_model / n_heads;
    (void)d_k;

    for (int i = 0; i < d_model * d_model; ++i) {
        weight_q.push_back(Value::create(random_normal(0, 0.02)));
        weight_k.push_back(Value::create(random_normal(0, 0.02)));
        weight_v.push_back(Value::create(random_normal(0, 0.02)));
    }
    for (int i = 0; i < d_model * d_model; ++i) {
        weight_o.push_back(Value::create(random_normal(0, 0.02)));
    }

    for (int i = 0; i < d_model; ++i) {
        bias_q.push_back(Value::create(0.0));
        bias_k.push_back(Value::create(0.0));
        bias_v.push_back(Value::create(0.0));
        bias_o.push_back(Value::create(0.0));
    }
}

std::vector<std::vector<ValuePtr>> MultiHeadSelfAttention::forward(
    const std::vector<std::vector<ValuePtr>>& x,
    const std::vector<int>& pad_mask) {

    if (x.empty()) {
        return {};
    }

    int seq_len = x.size();
    if (seq_len == 0) return {};

    int d_model = x[0].size();
    if (d_model == 0) return {};

    int d_k = d_model / n_heads;
    if (d_k == 0) {
        std::cerr << "[ERROR] Attention: d_k=0" << std::endl;
        return {};
    }

    std::vector<std::vector<ValuePtr>> q(seq_len), k(seq_len), v(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        q[i] = linear_forward(x[i], weight_q, bias_q);
        k[i] = linear_forward(x[i], weight_k, bias_k);
        v[i] = linear_forward(x[i], weight_v, bias_v);
    }

    std::vector<std::vector<std::vector<ValuePtr>>> q_heads(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        q_heads[i].resize(n_heads);
        for (int h = 0; h < n_heads; ++h) {
            q_heads[i][h].reserve(d_k);
            for (int j = 0; j < d_k; ++j) {
                int idx = h * d_k + j;
                if (idx < (int)q[i].size()) {
                    q_heads[i][h].push_back(q[i][idx]);
                }
            }
        }
    }

    std::vector<std::vector<std::vector<ValuePtr>>> k_heads(n_heads);
    for (int h = 0; h < n_heads; ++h) {
        k_heads[h].resize(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            k_heads[h][i].reserve(d_k);
            for (int j = 0; j < d_k; ++j) {
                int idx = h * d_k + j;
                if (idx < (int)k[i].size()) {
                    k_heads[h][i].push_back(k[i][idx]);
                }
            }
        }
    }

    std::vector<std::vector<std::vector<ValuePtr>>> v_heads(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        v_heads[i].resize(n_heads);
        for (int h = 0; h < n_heads; ++h) {
            v_heads[i][h].reserve(d_k);
            for (int j = 0; j < d_k; ++j) {
                int idx = h * d_k + j;
                if (idx < (int)v[i].size()) {
                    v_heads[i][h].push_back(v[i][idx]);
                }
            }
        }
    }

    std::vector<std::vector<std::vector<ValuePtr>>> attn_outputs(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        attn_outputs[i].resize(n_heads);
        for (int h = 0; h < n_heads; ++h) {
            std::vector<ValuePtr> scores(seq_len);
            for (int j = 0; j < seq_len; ++j) {
                ValuePtr dot = Value::create(0.0);
                for (int kk = 0; kk < d_k; ++kk) {
                    if (kk < (int)q_heads[i][h].size() && kk < (int)k_heads[h][j].size()) {
                        dot = dot + (q_heads[i][h][kk] * k_heads[h][j][kk]);
                    }
                }
                scores[j] = dot;
            }

            double scale = 1.0 / std::sqrt((double)d_k);
            for (auto& score : scores) {
                score = score * Value::create(scale);
            }

            // Causal mask: query i tidak boleh attend ke key j > i
            for (int j = i + 1; j < seq_len; ++j) {
                scores[j] = scores[j] + Value::create(-1e9);
            }

            // Padding mask: mask posisi KEY yang padding
            if (!pad_mask.empty()) {
                for (int j = 0; j < seq_len; ++j) {
                    if (j < (int)pad_mask.size() && pad_mask[j] == 1) {
                        scores[j] = scores[j] + Value::create(-1e9);
                    }
                }
            }

            auto probs = softmax(scores);
            auto dropped_probs = attn_dropout.forward(probs);

            std::vector<ValuePtr> out(d_k);
            for (int j = 0; j < d_k; ++j) {
                out[j] = Value::create(0.0);
                for (int kk = 0; kk < seq_len; ++kk) {
                    if (kk < (int)dropped_probs.size() && j < (int)v_heads[kk][h].size()) {
                        out[j] = out[j] + (dropped_probs[kk] * v_heads[kk][h][j]);
                    }
                }
            }
            attn_outputs[i][h] = out;
        }
    }

    std::vector<std::vector<ValuePtr>> concat(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        concat[i].reserve(d_model);
        for (int h = 0; h < n_heads; ++h) {
            if (attn_outputs[i][h].size() == static_cast<size_t>(d_k)) {
                concat[i].insert(concat[i].end(), attn_outputs[i][h].begin(), attn_outputs[i][h].end());
            }
        }
    }

    std::vector<std::vector<ValuePtr>> output(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        if (!concat[i].empty()) {
            output[i] = linear_forward(concat[i], weight_o, bias_o);
        } else {
            output[i] = std::vector<ValuePtr>(d_model, Value::create(0.0));
        }
    }

    return output;
}

// BARU: attention untuk 1 token pakai KV-cache.
// k_cache/v_cache berisi K/V (flat, d_model per entri) dari semua posisi
// SEBELUM token ini. Token saat ini diproyeksikan, hasil K/V-nya
// ditambahkan ke cache (jadi cache mencakup posisi 0..pos setelah return),
// lalu Q token ini dibandingkan terhadap seluruh cache -- causal mask
// otomatis terpenuhi karena cache cuma berisi posisi <= pos.
std::vector<ValuePtr> MultiHeadSelfAttention::forward_incremental(
    const std::vector<ValuePtr>& x_token,
    std::vector<std::vector<ValuePtr>>& k_cache,
    std::vector<std::vector<ValuePtr>>& v_cache) {

    if (x_token.empty()) return {};

    int d_model_local = x_token.size();
    int d_k = d_model_local / n_heads;
    if (d_k == 0) {
        std::cerr << "[ERROR] Attention (incremental): d_k=0" << std::endl;
        return {};
    }

    std::vector<ValuePtr> q = linear_forward(x_token, weight_q, bias_q);
    std::vector<ValuePtr> k = linear_forward(x_token, weight_k, bias_k);
    std::vector<ValuePtr> v = linear_forward(x_token, weight_v, bias_v);

    if (q.empty() || k.empty() || v.empty()) return {};

    // Simpan K/V token ini ke cache SEBELUM dipakai, supaya token ini
    // juga bisa attend ke dirinya sendiri (posisi terakhir).
    k_cache.push_back(k);
    v_cache.push_back(v);

    int seq_len = (int)k_cache.size();  // jumlah posisi yang sudah ada, termasuk token ini

    std::vector<std::vector<ValuePtr>> q_heads(n_heads);
    for (int h = 0; h < n_heads; ++h) {
        q_heads[h].reserve(d_k);
        for (int j = 0; j < d_k; ++j) {
            int idx = h * d_k + j;
            if (idx < (int)q.size()) q_heads[h].push_back(q[idx]);
        }
    }

    double scale = 1.0 / std::sqrt((double)d_k);
    std::vector<ValuePtr> concat;
    concat.reserve(d_model_local);

    for (int h = 0; h < n_heads; ++h) {
        std::vector<ValuePtr> scores(seq_len);
        for (int t = 0; t < seq_len; ++t) {
            ValuePtr dot = Value::create(0.0);
            for (int kk = 0; kk < d_k; ++kk) {
                int idx = h * d_k + kk;
                if (idx < (int)k_cache[t].size()) {
                    dot = dot + (q_heads[h][kk] * k_cache[t][idx]);
                }
            }
            scores[t] = dot * Value::create(scale);
        }

        auto probs = softmax(scores);
        auto dropped_probs = attn_dropout.forward(probs);

        std::vector<ValuePtr> out(d_k);
        for (int j = 0; j < d_k; ++j) {
            out[j] = Value::create(0.0);
            for (int t = 0; t < seq_len; ++t) {
                int idx = h * d_k + j;
                if (t < (int)dropped_probs.size() && idx < (int)v_cache[t].size()) {
                    out[j] = out[j] + (dropped_probs[t] * v_cache[t][idx]);
                }
            }
        }
        concat.insert(concat.end(), out.begin(), out.end());
    }

    return linear_forward(concat, weight_o, bias_o);
}

// ============================================================
// FEED FORWARD
// ============================================================

FeedForward::FeedForward(int d_model, int d_ff, double dropout) 
    : d_model(d_model), d_ff(d_ff), dropout_p(dropout),
      ff_dropout(dropout) {

    for (int i = 0; i < d_model * d_ff; ++i) {
        w1.push_back(Value::create(random_normal(0, 0.02)));
    }
    for (int i = 0; i < d_ff * d_model; ++i) {
        w2.push_back(Value::create(random_normal(0, 0.02)));
    }

    for (int i = 0; i < d_ff; ++i) {
        b1.push_back(Value::create(0.0));
    }
    for (int i = 0; i < d_model; ++i) {
        b2.push_back(Value::create(0.0));
    }
}

std::vector<ValuePtr> FeedForward::forward(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};

    auto hidden = linear_forward(x, w1, b1);
    for (auto& v : hidden) {
        v = gelu(v);
    }

    auto dropped = ff_dropout.forward(hidden);
    auto output = linear_forward(dropped, w2, b2);

    return output;
}

// ============================================================
// TRANSFORMER BLOCK
// ============================================================

TransformerBlock::TransformerBlock(int d_model, int n_heads, int d_ff, double dropout, int max_len)
    : d_model(d_model), n_heads(n_heads), d_ff(d_ff), dropout_p(dropout), max_len(max_len),
      attn(d_model, n_heads, dropout, max_len),
      ff(d_model, d_ff, dropout),
      ln1(d_model),
      ln2(d_model),
      attn_dropout(dropout) {}

std::vector<std::vector<ValuePtr>> TransformerBlock::forward(
    const std::vector<std::vector<ValuePtr>>& x,
    const std::vector<int>& pad_mask) {

    if (x.empty()) return {};

    auto attn_out = attn.forward(x, pad_mask);
    if (attn_out.empty()) return {};

    std::vector<std::vector<ValuePtr>> attn_residual(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        attn_residual[i].reserve(x[i].size());
        for (size_t j = 0; j < x[i].size(); ++j) {
            if (j < attn_out[i].size()) {
                attn_residual[i].push_back(x[i][j] + attn_out[i][j]);
            } else {
                attn_residual[i].push_back(x[i][j]);
            }
        }
    }

    std::vector<std::vector<ValuePtr>> norm1(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        norm1[i] = ln1.forward(attn_residual[i]);
    }

    std::vector<std::vector<ValuePtr>> ff_out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ff_out[i] = ff.forward(norm1[i]);
    }

    std::vector<std::vector<ValuePtr>> ff_residual(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ff_residual[i].reserve(x[i].size());
        for (size_t j = 0; j < x[i].size(); ++j) {
            if (j < ff_out[i].size()) {
                ff_residual[i].push_back(attn_residual[i][j] + ff_out[i][j]);
            } else {
                ff_residual[i].push_back(attn_residual[i][j]);
            }
        }
    }

    std::vector<std::vector<ValuePtr>> output(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        output[i] = ln2.forward(ff_residual[i]);
    }

    return output;
}

// BARU: versi 1-token dari block, memakai KV-cache attention di atas.
std::vector<ValuePtr> TransformerBlock::forward_incremental(
    const std::vector<ValuePtr>& x_token,
    std::vector<std::vector<ValuePtr>>& k_cache,
    std::vector<std::vector<ValuePtr>>& v_cache) {

    if (x_token.empty()) return {};

    auto attn_out = attn.forward_incremental(x_token, k_cache, v_cache);
    if (attn_out.empty()) return {};

    std::vector<ValuePtr> attn_residual(x_token.size());
    for (size_t j = 0; j < x_token.size(); ++j) {
        attn_residual[j] = (j < attn_out.size()) ? (x_token[j] + attn_out[j]) : x_token[j];
    }

    auto norm1 = ln1.forward(attn_residual);
    auto ff_out = ff.forward(norm1);

    std::vector<ValuePtr> ff_residual(x_token.size());
    for (size_t j = 0; j < x_token.size(); ++j) {
        ff_residual[j] = (j < ff_out.size()) ? (attn_residual[j] + ff_out[j]) : attn_residual[j];
    }

    return ln2.forward(ff_residual);
}

// ============================================================
// LINEAR (Final Projection)
// ============================================================

Linear::Linear(int in_features, int out_features) 
    : in_features(in_features), out_features(out_features) {

    for (int i = 0; i < in_features * out_features; ++i) {
        weight.push_back(Value::create(random_normal(0, 0.02)));
    }

    for (int i = 0; i < out_features; ++i) {
        bias.push_back(Value::create(0.0));
    }
}

std::vector<ValuePtr> Linear::forward(const std::vector<ValuePtr>& x) {
    if (x.empty() || weight.empty()) {
        return {};
    }
    return linear_forward(x, weight, bias);
}