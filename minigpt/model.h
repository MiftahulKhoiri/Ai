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

    // KV cache untuk setiap TransformerBlock
    std::vector<
        std::pair<
            std::vector<std::vector<ValuePtr>>, // Key cache
            std::vector<std::vector<ValuePtr>>  // Value cache
        >
    > caches;

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

    std::vector<ValuePtr> forward_incremental(
        int token_id,
        int pos
    );

    std::vector<ValuePtr> parameters();

    void set_training(bool mode);
};