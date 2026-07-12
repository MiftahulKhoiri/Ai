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
            // Catatan: fallback ini membuat SATU Value dan menyalin shared_ptr-nya
            // d_model kali -> semua elemen menunjuk ke node yang sama. Untuk kasus
            // fallback (id invalid) dampaknya kecil, tapi perlu diwaspadai kalau
            // pola serupa dipakai di tempat lain untuk data yang ikut backward.
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
        // FIX: parameter lokal "n_heads" tadinya cuma menutupi (shadow) member
        // this->n_heads yang sudah di-set dari initializer list. Assignment di
        // atas hanya mengubah variabel lokal, bukan this->n_heads -- jadi kalau
        // d_model tidak habis dibagi n_heads awal, member yang dipakai di
        // forward() tetap nilai lama yang salah. Sinkronkan balik ke member:
        this->n_heads = n_heads;
    }

    int d_k = d_model / n_heads;
    (void)d_k; // Suppress unused variable warning

    // Initialize weights
    for (int i = 0; i < d_model * d_model; ++i) {
        weight_q.push_back(Value::create(random_normal(0, 0.02)));
        weight_k.push_back(Value::create(random_normal(0, 0.02)));
        weight_v.push_back(Value::create(random_normal(0, 0.02)));
    }
    for (int i = 0; i < d_model * d_model; ++i) {
        weight_o.push_back(Value::create(random_normal(0, 0.02)));
    }

    // Initialize biases
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

    // Project to Q, K, V
    std::vector<std::vector<ValuePtr>> q(seq_len), k(seq_len), v(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        q[i] = linear_forward(x[i], weight_q, bias_q);
        k[i] = linear_forward(x[i], weight_k, bias_k);
        v[i] = linear_forward(x[i], weight_v, bias_v);
    }

    // Reshape for multi-head
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

    // K heads: [n_heads, seq_len, d_k]
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

    // V heads: [seq_len, n_heads, d_k]
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

    // Compute attention
    std::vector<std::vector<std::vector<ValuePtr>>> attn_outputs(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        attn_outputs[i].resize(n_heads);
        for (int h = 0; h < n_heads; ++h) {
            // Compute scores
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

            // Scale
            double scale = 1.0 / std::sqrt((double)d_k);
            for (auto& score : scores) {
                score = score * Value::create(scale);
            }

            // FIX: causal mask -- query di posisi i tidak boleh attend ke
            // key di posisi j > i (mencegah model "curi lihat" token masa
            // depan saat training). Wajib untuk model autoregressive/GPT.
            for (int j = i + 1; j < seq_len; ++j) {
                scores[j] = scores[j] + Value::create(-1e9);
            }

            // Apply padding mask
            // FIX: cek pad_mask[j] (posisi KEY yang mau di-mask), bukan
            // pad_mask[i] (posisi query). Sebelumnya kode mengecek posisi
            // query, sehingga key padding tidak pernah ter-mask ketika
            // query-nya sendiri bukan padding.
            if (!pad_mask.empty()) {
                for (int j = 0; j < seq_len; ++j) {
                    if (j < (int)pad_mask.size() && pad_mask[j] == 1) {
                        scores[j] = scores[j] + Value::create(-1e9);
                    }
                }
            }

            // Softmax
            auto probs = softmax(scores);

            // Apply dropout
            auto dropped_probs = attn_dropout.forward(probs);

            // Apply to V
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

    // Concatenate heads
    std::vector<std::vector<ValuePtr>> concat(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        concat[i].reserve(d_model);
        for (int h = 0; h < n_heads; ++h) {
            if (attn_outputs[i][h].size() == static_cast<size_t>(d_k)) {
                concat[i].insert(concat[i].end(), attn_outputs[i][h].begin(), attn_outputs[i][h].end());
            }
        }
    }

    // Final projection
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

// ============================================================
// FEED FORWARD
// ============================================================

FeedForward::FeedForward(int d_model, int d_ff, double dropout) 
    : d_model(d_model), d_ff(d_ff), dropout_p(dropout),
      ff_dropout(dropout) {

    // Initialize weights
    for (int i = 0; i < d_model * d_ff; ++i) {
        w1.push_back(Value::create(random_normal(0, 0.02)));
    }
    for (int i = 0; i < d_ff * d_model; ++i) {
        w2.push_back(Value::create(random_normal(0, 0.02)));
    }

    // Initialize biases
    for (int i = 0; i < d_ff; ++i) {
        b1.push_back(Value::create(0.0));
    }
    for (int i = 0; i < d_model; ++i) {
        b2.push_back(Value::create(0.0));
    }
}

std::vector<ValuePtr> FeedForward::forward(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};

    // First linear + GELU
    auto hidden = linear_forward(x, w1, b1);
    for (auto& v : hidden) {
        v = gelu(v);
    }

    // Dropout
    auto dropped = ff_dropout.forward(hidden);

    // Second linear
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

    // Self-attention with residual
    auto attn_out = attn.forward(x, pad_mask);
    if (attn_out.empty()) return {};

    // Add residual and layer norm
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

    // Layer norm on residual
    std::vector<std::vector<ValuePtr>> norm1(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        norm1[i] = ln1.forward(attn_residual[i]);
    }

    // Feed forward with residual
    std::vector<std::vector<ValuePtr>> ff_out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ff_out[i] = ff.forward(norm1[i]);
    }

    // Add residual and layer norm
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

    // Final layer norm
    std::vector<std::vector<ValuePtr>> output(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        output[i] = ln2.forward(ff_residual[i]);
    }

    return output;
}

// ============================================================
// LINEAR (Final Projection)
// ============================================================

Linear::Linear(int in_features, int out_features) 
    : in_features(in_features), out_features(out_features) {

    // Initialize weights
    for (int i = 0; i < in_features * out_features; ++i) {
        weight.push_back(Value::create(random_normal(0, 0.02)));
    }

    // Initialize biases
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