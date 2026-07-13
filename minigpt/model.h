// model.h
#pragma once

#include "layers.h"
#include <utility>
#include <vector>

struct MiniGPT {
    int d_model;
    int max_len;
    int vocab_size;

    Embedding embed;
    PositionalEmbedding pos_embed;
    Dropout embed_drop;

    std::vector<TransformerBlock> blocks;

    LayerNorm ln_f;
    Linear head;

    // Tipe cache satu layer: (K cache, V cache), sama seperti yang dipakai
    // TransformerBlock::forward_incremental.
    using LayerCache = std::pair<
        std::vector<std::vector<ValuePtr>>,
        std::vector<std::vector<ValuePtr>>
    >;

    // Cache internal -- dipakai jalur single-threaded lama (generate()
    // di generation.cpp / CLI main.cpp). JANGAN dipakai dari lebih dari
    // satu thread sekaligus, karena ini state yang di-share via member.
    std::vector<LayerCache> caches;

    MiniGPT(
        int vocab_size,
        int d_model = 16,
        int n_heads = 2,
        int n_layers = 2,
        int d_ff = 32,
        int max_len = 64,
        double dropout = 0.1
    );

    std::vector<std::vector<ValuePtr>> forward(
        const std::vector<int>& token_ids,
        const std::vector<int>& pad_mask = {}
    );

    void init_cache();

    // Versi lama: pakai cache internal (this->caches). Cocok untuk
    // pemakaian single-threaded (CLI generate()).
    std::vector<ValuePtr> forward_incremental(
        int token_id,
        int pos
    );

    // BARU: versi thread-safe -- cache disediakan PEMANGGIL (bukan
    // member model), jadi tiap request/thread bisa punya cache sendiri
    // tanpa saling bentrok. Dipakai server untuk generate paralel.
    std::vector<ValuePtr> forward_incremental(
        int token_id,
        int pos,
        std::vector<LayerCache>& external_cache
    );

    // Buat cache kosong baru berukuran sesuai jumlah layer -- dipanggil
    // sekali per request sebelum mulai generate_incremental.
    std::vector<LayerCache> make_cache() const {
        return std::vector<LayerCache>(blocks.size());
    }

    std::vector<ValuePtr> parameters();

    void set_training(bool mode);
};