// attention.h
#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include <cstddef>

class MultiHeadAttention {
public:
    MultiHeadAttention(int n_heads, int d_model, int d_k = -1, int d_v = -1);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& Q,
        const std::vector<std::vector<float>>& K,
        const std::vector<std::vector<float>>& V,
        const std::vector<std::vector<float>>& mask = {});

    std::vector<float> forward(const std::vector<float>& new_token);

    void reset_cache();

    void set_weights(const std::vector<std::vector<float>>& Wq,
                     const std::vector<std::vector<float>>& Wk,
                     const std::vector<std::vector<float>>& Wv,
                     const std::vector<std::vector<float>>& Wo,
                     const std::vector<float>& bq = {},
                     const std::vector<float>& bk = {},
                     const std::vector<float>& bv = {},
                     const std::vector<float>& bo = {});

private:
    int n_heads;
    int d_model;
    int d_k;
    int d_v;

    std::vector<std::vector<float>> W_q, W_k, W_v, W_o;
    std::vector<float> b_q, b_k, b_v, b_o;

    std::vector<std::vector<std::vector<float>>> keys_cache;
    std::vector<std::vector<std::vector<float>>> values_cache;

    std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A,
                                           const std::vector<std::vector<float>>& B);
    std::vector<float> matmul(const std::vector<std::vector<float>>& A,
                              const std::vector<float>& x);
    // BARU: perkalian vektor-matriks dengan arah yang BENAR untuk
    // proyeksi linear -- x diperlakukan sebagai vektor INPUT, baris A
    // adalah ruang input, kolom A adalah ruang output. y[j] = Σᵢ x[i]·A[i][j].
    // Ini konvensi yang sama dipakai forward() full-sequence via
    // matmul(Q, W_q) (matrix-matrix) -- dipakai supaya jalur single-token
    // (KV-cache) menghasilkan proyeksi yang identik dengan jalur
    // full-sequence untuk bobot yang sama.
    std::vector<float> vecmul(const std::vector<float>& x,
                              const std::vector<std::vector<float>>& A);
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& M);
    std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& M);
    void split_heads(const std::vector<float>& vec, int n_heads, int d_head,
                     std::vector<std::vector<float>>& out);
    std::vector<float> concat_heads(const std::vector<std::vector<float>>& head_outputs);
};

#endif