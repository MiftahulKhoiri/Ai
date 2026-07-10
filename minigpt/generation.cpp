#include "generation.h"
#include <algorithm>
#include <numeric>
#include <random>

int sample_from_logits(const std::vector<ValuePtr>& logits, double temperature, int top_k, double top_p) {
    std::vector<double> scaled;
    for (auto& v : logits)
        scaled.push_back(v->data / std::max(temperature, 1e-6));

    double m = *std::max_element(scaled.begin(), scaled.end());
    std::vector<double> exps;
    for (auto& s : scaled)
        exps.push_back(std::exp(s - m));
    double sum_e = std::accumulate(exps.begin(), exps.end(), 0.0);
    std::vector<double> probs(exps.size());
    for (size_t i = 0; i < exps.size(); ++i)
        probs[i] = exps[i] / sum_e;

    // top-k
    if (top_k > 0 && top_k < (int)probs.size()) {
        std::vector<size_t> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        for (int i = top_k; i < (int)idx.size(); ++i)
            probs[idx[i]] = 0.0;
    }

    // top-p (nucleus)
    if (top_p > 0.0 && top_p < 1.0) {
        std::vector<size_t> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        double cum = 0.0;
        for (size_t i = 0; i < idx.size(); ++i) {
            cum += probs[idx[i]];
            if (cum >= top_p) {
                for (size_t j = i+1; j < idx.size(); ++j)
                    probs[idx[j]] = 0.0;
                break;
            }
        }
    }

    double sum_p = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (sum_p <= 0)
        return std::max_element(scaled.begin(), scaled.end()) - scaled.begin();

    for (auto& p : probs) p /= sum_p;

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return probs.size() - 1;
}

std::string generate(MiniGPT& model, ByteLevelBPETokenizer& tok, const std::string& prompt,
                     int max_new_tokens, double temperature, int top_k, double top_p) {
    model.set_training(false);
    model.init_cache();
    auto ids = tok.encode(prompt, true, false);
    // ===== PERUBAHAN: tok.vocab → tok.get_vocab() =====
    int eos_id = tok.get_vocab().at("<eos>");

    std::vector<ValuePtr> logits;
    for (size_t pos = 0; pos < ids.size(); ++pos)
        logits = model.forward_incremental(ids[pos], pos);

    for (int _ = 0; _ < max_new_tokens; ++_) {
        int next_id = sample_from_logits(logits, temperature, top_k, top_p);
        ids.push_back(next_id);
        if (next_id == eos_id) break;
        int pos = ids.size() - 1;
        if (pos >= model.max_len) break;
        logits = model.forward_incremental(next_id, pos);
    }

    model.set_training(true);
    return tok.decode(ids);
}