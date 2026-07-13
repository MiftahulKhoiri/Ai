// attention.cpp
#include "attention.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

MultiHeadAttention::MultiHeadAttention(int n_heads, int d_model, int d_k, int d_v)
    : n_heads(n_heads), d_model(d_model) {
    if (d_k <= 0) this->d_k = d_model / n_heads;
    else          this->d_k = d_k;

    if (d_v <= 0) this->d_v = this->d_k;
    else          this->d_v = d_v;

    srand(static_cast<unsigned>(time(nullptr)));
    auto random_mat = [&](int rows, int cols) {
        std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                mat[i][j] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        return mat;
    };
    auto random_vec = [&](int size) {
        std::vector<float> vec(size);
        for (int i = 0; i < size; ++i)
            vec[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        return vec;
    };

    W_q = random_mat(d_model, n_heads * this->d_k);
    W_k = random_mat(d_model, n_heads * this->d_k);
    W_v = random_mat(d_model, n_heads * this->d_v);
    W_o = random_mat(n_heads * this->d_v, d_model);

    b_q = random_vec(n_heads * this->d_k);
    b_k = random_vec(n_heads * this->d_k);
    b_v = random_vec(n_heads * this->d_v);
    b_o = random_vec(d_model);

    keys_cache.resize(n_heads);
    values_cache.resize(n_heads);
}

std::vector<std::vector<float>> MultiHeadAttention::forward(
    const std::vector<std::vector<float>>& Q,
    const std::vector<std::vector<float>>& K,
    const std::vector<std::vector<float>>& V,
    const std::vector<std::vector<float>>& mask) {

    if (Q.empty()) return {};

    int seq_len = Q.size();
    int d_out_q = n_heads * d_k;
    int d_out_v = n_heads * d_v;

    std::vector<std::vector<float>> Q_proj = matmul(Q, W_q);
    std::vector<std::vector<float>> K_proj = matmul(K, W_k);
    std::vector<std::vector<float>> V_proj = matmul(V, W_v);

    if (!b_q.empty()) for (auto& row : Q_proj) for (int j = 0; j < d_out_q; ++j) row[j] += b_q[j];
    if (!b_k.empty()) for (auto& row : K_proj) for (int j = 0; j < d_out_q; ++j) row[j] += b_k[j];
    if (!b_v.empty()) for (auto& row : V_proj) for (int j = 0; j < d_out_v; ++j) row[j] += b_v[j];

    std::vector<std::vector<std::vector<float>>> head_outputs(n_heads);

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    for (int h = 0; h < n_heads; ++h) {
        int offset_k = h * d_k;
        int offset_v = h * d_v;

        std::vector<std::vector<float>> Q_head(seq_len, std::vector<float>(d_k));
        std::vector<std::vector<float>> K_head(seq_len, std::vector<float>(d_k));
        std::vector<std::vector<float>> V_head(seq_len, std::vector<float>(d_v));

        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_k; ++j) {
                Q_head[i][j] = Q_proj[i][offset_k + j];
                K_head[i][j] = K_proj[i][offset_k + j];
            }
            for (int j = 0; j < d_v; ++j) {
                V_head[i][j] = V_proj[i][offset_v + j];
            }
        }

        std::vector<std::vector<float>> K_head_t = transpose(K_head);
        std::vector<std::vector<float>> scores = matmul(Q_head, K_head_t);

        for (auto& row : scores)
            for (auto& val : row)
                val *= scale;

        if (!mask.empty()) {
            for (int i = 0; i < seq_len; ++i)
                for (int j = 0; j < seq_len; ++j)
                    scores[i][j] += mask[i][j];
        }

        std::vector<std::vector<float>> attn_weights = softmax(scores);
        std::vector<std::vector<float>> output_head = matmul(attn_weights, V_head);
        head_outputs[h] = output_head;
    }

    std::vector<std::vector<float>> concat_output(seq_len, std::vector<float>(d_out_v));
    for (int i = 0; i < seq_len; ++i) {
        int col = 0;
        for (int h = 0; h < n_heads; ++h) {
            for (int j = 0; j < d_v; ++j) {
                concat_output[i][col++] = head_outputs[h][i][j];
            }
        }
    }

    std::vector<std::vector<float>> output = matmul(concat_output, W_o);
    if (!b_o.empty())
        for (auto& row : output)
            for (int j = 0; j < d_model; ++j)
                row[j] += b_o[j];

    return output;
}

std::vector<float> MultiHeadAttention::forward(const std::vector<float>& new_token) {
    if (new_token.size() != static_cast<size_t>(d_model))
        throw std::invalid_argument("Ukuran token tidak sesuai dengan d_model");

    // FIX: pakai vecmul (arah benar: x=input, baris A=input, kolom A=output),
    // BUKAN matmul(A,x) (arah matrix-vector standar yang salah konvensi di
    // sini). Sebelumnya matmul(W_q, new_token) memperlakukan baris W_q
    // (ruang INPUT, d_model) seolah ruang output -- hasil proyeksi Q/K/V
    // di mode single-token beda total dari proyeksi di forward() full-
    // sequence, walau memakai bobot W_q/W_k/W_v yang identik persis.
    std::vector<float> q_proj = vecmul(new_token, W_q);   // [n_heads*d_k]
    std::vector<float> k_proj = vecmul(new_token, W_k);   // [n_heads*d_k]
    std::vector<float> v_proj = vecmul(new_token, W_v);   // [n_heads*d_v]

    if (!b_q.empty()) for (int i = 0; i < n_heads * d_k; ++i) q_proj[i] += b_q[i];
    if (!b_k.empty()) for (int i = 0; i < n_heads * d_k; ++i) k_proj[i] += b_k[i];
    if (!b_v.empty()) for (int i = 0; i < n_heads * d_v; ++i) v_proj[i] += b_v[i];

    std::vector<std::vector<float>> q_heads, k_heads, v_heads;
    split_heads(q_proj, n_heads, d_k, q_heads);
    split_heads(k_proj, n_heads, d_k, k_heads);
    split_heads(v_proj, n_heads, d_v, v_heads);

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    std::vector<std::vector<float>> head_outputs(n_heads);

    for (int h = 0; h < n_heads; ++h) {
        keys_cache[h].push_back(k_heads[h]);
        values_cache[h].push_back(v_heads[h]);

        int cache_len = keys_cache[h].size();
        const auto& qh = q_heads[h];

        std::vector<float> scores(cache_len, 0.0f);
        for (int t = 0; t < cache_len; ++t) {
            float dot = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                dot += qh[i] * keys_cache[h][t][i];
            }
            scores[t] = dot * scale;
        }

        float max_val = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        for (int t = 0; t < cache_len; ++t) {
            scores[t] = std::exp(scores[t] - max_val);
            sum_exp += scores[t];
        }
        for (int t = 0; t < cache_len; ++t) {
            scores[t] /= sum_exp;
        }

        std::vector<float> out_head(d_v, 0.0f);
        for (int t = 0; t < cache_len; ++t) {
            float w = scores[t];
            for (int i = 0; i < d_v; ++i) {
                out_head[i] += w * values_cache[h][t][i];
            }
        }
        head_outputs[h] = out_head;
    }

    std::vector<float> concat = concat_heads(head_outputs);
    // FIX: sama seperti di atas -- pakai vecmul, bukan matmul(W_o, concat)
    // yang memperlakukan baris W_o (ruang input, n_heads*d_v) seolah
    // ruang output.
    std::vector<float> output = vecmul(concat, W_o);   // [d_model]
    if (!b_o.empty())
        for (int i = 0; i < d_model; ++i) output[i] += b_o[i];

    return output;
}

void MultiHeadAttention::reset_cache() {
    for (auto& kc : keys_cache) kc.clear();
    for (auto& vc : values_cache) vc.clear();
}

void MultiHeadAttention::set_weights(
    const std::vector<std::vector<float>>& Wq,
    const std::vector<std::vector<float>>& Wk,
    const std::vector<std::vector<float>>& Wv,
    const std::vector<std::vector<float>>& Wo,
    const std::vector<float>& bq,
    const std::vector<float>& bk,
    const std::vector<float>& bv,
    const std::vector<float>& bo) {
    
    W_q = Wq; W_k = Wk; W_v = Wv; W_o = Wo;
    if (!bq.empty()) b_q = bq;
    if (!bk.empty()) b_k = bk;
    if (!bv.empty()) b_v = bv;
    if (!bo.empty()) b_o = bo;
}

std::vector<std::vector<float>> MultiHeadAttention::matmul(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B) {
    
    if (A.empty() || B.empty()) return {};
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();
    if (B.size() != static_cast<size_t>(n))
        throw std::invalid_argument("Dimensi matriks tidak cocok untuk perkalian");

    std::vector<std::vector<float>> C(m, std::vector<float>(p, 0.0f));
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < p; ++j)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<float> MultiHeadAttention::matmul(
    const std::vector<std::vector<float>>& A,
    const std::vector<float>& x) {
    
    if (A.empty() || x.empty()) return {};
    int m = A.size();
    int n = A[0].size();
    if (x.size() != static_cast<size_t>(n))
        throw std::invalid_argument("Dimensi tidak cocok untuk perkalian matriks-vektor");

    std::vector<float> y(m, 0.0f);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

// BARU: y[j] = Σᵢ x[i] · A[i][j] -- x adalah vektor INPUT, baris A ruang
// input, kolom A ruang output. Konvensi ini konsisten dengan forward()
// full-sequence (matmul(Q, W_q) memperlakukan baris W_q sebagai ruang
// input yang dikontraksikan dengan kolom Q).
std::vector<float> MultiHeadAttention::vecmul(
    const std::vector<float>& x,
    const std::vector<std::vector<float>>& A) {

    if (x.empty() || A.empty()) return {};
    int n = A.size();       // baris A = dimensi input, harus == x.size()
    int m = A[0].size();    // kolom A = dimensi output

    if (x.size() != static_cast<size_t>(n))
        throw std::invalid_argument("Dimensi tidak cocok untuk perkalian vektor-matriks");

    std::vector<float> y(m, 0.0f);
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        for (int j = 0; j < m; ++j)
            y[j] += xi * A[i][j];
    }
    return y;
}

std::vector<std::vector<float>> MultiHeadAttention::transpose(
    const std::vector<std::vector<float>>& M) {
    if (M.empty()) return {};
    int rows = M.size();
    int cols = M[0].size();
    std::vector<std::vector<float>> T(cols, std::vector<float>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            T[j][i] = M[i][j];
    return T;
}

std::vector<std::vector<float>> MultiHeadAttention::softmax(
    const std::vector<std::vector<float>>& M) {
    std::vector<std::vector<float>> result = M;
    for (auto& row : result) {
        float max_val = *std::max_element(row.begin(), row.end());
        float sum = 0.0f;
        for (auto& v : row) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (auto& v : row) v /= sum;
    }
    return result;
}

void MultiHeadAttention::split_heads(const std::vector<float>& vec,
                                     int n_heads, int d_head,
                                     std::vector<std::vector<float>>& out) {
    out.resize(n_heads, std::vector<float>(d_head));
    for (int h = 0; h < n_heads; ++h)
        for (int i = 0; i < d_head; ++i)
            out[h][i] = vec[h * d_head + i];
}

std::vector<float> MultiHeadAttention::concat_heads(
    const std::vector<std::vector<float>>& head_outputs) {
    int n_heads = head_outputs.size();
    if (n_heads == 0) return {};
    int d_v = head_outputs[0].size();
    std::vector<float> result(n_heads * d_v);
    for (int h = 0; h < n_heads; ++h)
        for (int i = 0; i < d_v; ++i)
            result[h * d_v + i] = head_outputs[h][i];
    return result;
}