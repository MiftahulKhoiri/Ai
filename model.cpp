#include "model.h"
#include <stdexcept>

MiniGPT::MiniGPT(int vocab_size, int d_model, int n_heads, int n_layers, int d_ff, int max_len, double dropout)
    : d_model(d_model), max_len(max_len), vocab_size(vocab_size),
      embed(vocab_size, d_model), pos_embed(max_len, d_model), embed_drop(dropout),
      ln_f(d_model), head(d_model, vocab_size) {
    for (int i = 0; i < n_layers; ++i)
        blocks.emplace_back(d_model, n_heads, d_ff, dropout);
}

std::vector<std::vector<ValuePtr>> MiniGPT::forward(const std::vector<int>& token_ids,
                                                    const std::vector<int>& pad_mask) {
    std::vector<std::vector<ValuePtr>> X;
    for (size_t pos = 0; pos < token_ids.size(); ++pos) {
        auto emb = embed.forward(token_ids[pos]);
        auto pe = pos_embed.forward(pos);
        std::vector<ValuePtr> combined;
        for (int d = 0; d < d_model; ++d)
            combined.push_back(emb[d] + pe[d]);
        X.push_back(combined);
    }
    for (auto& x : X) x = embed_drop.forward(x);
    for (auto& block : blocks)
        X = block.forward(X, pad_mask);
    std::vector<std::vector<ValuePtr>> ln_out;
    for (auto& x : X) ln_out.push_back(ln_f.forward(x));
    std::vector<std::vector<ValuePtr>> logits_seq;
    for (auto& x : ln_out) logits_seq.push_back(head.forward(x));
    return logits_seq;
}

void MiniGPT::init_cache() {
    caches.clear();
    for (size_t i = 0; i < blocks.size(); ++i)
        caches.push_back({{}, {}});
}

std::vector<ValuePtr> MiniGPT::forward_incremental(int token_id, int pos) {
    if (pos >= max_len) throw std::runtime_error("pos exceeds max_len");
    auto emb = embed.forward(token_id);
    auto pe = pos_embed.forward(pos);
    std::vector<ValuePtr> x;
    for (int d = 0; d < d_model; ++d) x.push_back(emb[d] + pe[d]);
    x = embed_drop.forward(x);
    for (size_t i = 0; i < blocks.size(); ++i) {
        x = blocks[i].forward_incremental(x, caches[i].first, caches[i].second);
    }
    auto ln_out = ln_f.forward(x);
    return head.forward(ln_out);
}

std::vector<ValuePtr> MiniGPT::parameters() {
    auto p = embed.parameters();
    auto p2 = pos_embed.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    for (auto& b : blocks) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto p_ln = ln_f.parameters();
    p.insert(p.end(), p_ln.begin(), p_ln.end());
    auto p_head = head.parameters();
    p.insert(p.end(), p_head.begin(), p_head.end());
    return p;
}

void MiniGPT::set_training(bool mode) {
    embed_drop.training = mode;
    for (auto& b : blocks) b.set_training(mode);
}