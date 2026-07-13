// generation_advanced.cpp
#include "generation_advanced.h"
#include "sampling.h"
#include <iostream>
#include <cmath>

namespace advanced_generation {

AdvancedGenerator::AdvancedGenerator(MiniGPT& model, ByteLevelBPETokenizer& tokenizer)
    : model(model), tokenizer(tokenizer) {
    config.max_length = 50;
    config.num_beams = 1;
    config.temperature = 0.8;
    config.top_p = 0.9;
    config.top_k = 40;
    config.repetition_penalty = 1.0;
    config.length_penalty = 1.0;
    config.num_return_sequences = 1;
    config.use_cache = true;
    config.no_repeat_ngram_size = 0;
    // FIX (server paralel): cache dibuat di sini, milik instance ini
    // sendiri -- bukan lagi model.caches yang di-share antar request.
    cache = model.make_cache();
}

void AdvancedGenerator::set_config(const GenerationConfig& cfg) {
    config = cfg;
}

const GenerationConfig& AdvancedGenerator::get_config() const {
    return config;
}

std::vector<std::string> AdvancedGenerator::generate(const std::string& prompt, bool add_bos, bool add_eos) {
    std::vector<int> input_ids = tokenizer.encode(prompt, add_bos, add_eos);
    if (input_ids.empty()) {
        std::cerr << "Prompt kosong atau tidak bisa di-encode.\n";
        return {};
    }

    // CATATAN: model.set_training(false) TIDAK dipanggil di sini lagi.
    // Kalau dipanggil per-request, ini nulis field mutable yang di-share
    // semua thread (embed_drop.training, dst) -- race kalau paralel.
    // Untuk server, panggil model.set_training(false) SEKALI saat startup
    // (sebelum server mulai menerima request), bukan di sini.

    // FIX (server paralel): cache instance ini di-reset di awal tiap
    // pemanggilan generate(), bukan cache milik model yang di-share.
    cache.assign(model.blocks.size(), MiniGPT::LayerCache{});

    if (config.num_beams <= 1) {
        return greedy_generate(input_ids);
    } else {
        return beam_search_generate(input_ids);
    }
}

std::vector<std::string> AdvancedGenerator::greedy_generate(const std::vector<int>& input_ids) {
    std::vector<int> current_tokens = input_ids;
    int eos_id = tokenizer.get_eos_token_id();

    std::vector<Value::Ptr> logits;
    int pos = 0;
    for (; pos < (int)input_ids.size(); ++pos) {
        // FIX (server paralel): pakai overload forward_incremental yang
        // menerima cache eksternal (this->cache), bukan cache internal
        // model yang di-share antar request.
        logits = model.forward_incremental(input_ids[pos], pos, cache);
    }

    for (; pos < config.max_length; ++pos) {
        if (config.repetition_penalty > 1.0f) {
            apply_repetition_penalty(logits, current_tokens, config.repetition_penalty);
        }

        int next_token = 0;
        if (config.top_p < 1.0f && config.top_p > 0.0f) {
            auto sampled = sampling::top_p_sample(logits, config.top_p, config.temperature);
            next_token = sampled.empty() ? 0 : sampled[0];
        } else if (config.top_k > 0) {
            auto sampled = sampling::top_k_sample(logits, config.top_k, config.temperature);
            next_token = sampled.empty() ? 0 : sampled[0];
        } else {
            next_token = sampling::greedy_sample(logits);
        }

        if (config.forbidden_tokens.count(next_token) > 0) {
            next_token = sampling::greedy_sample(logits);
        }

        if (next_token == eos_id && (int)current_tokens.size() >= config.min_length) {
            break;
        }

        current_tokens.push_back(next_token);

        if (pos + 1 < config.max_length) {
            logits = model.forward_incremental(next_token, pos + 1, cache);
        }
    }

    std::string text = tokenizer.decode(current_tokens);
    return {text};
}

std::vector<std::string> AdvancedGenerator::beam_search_generate(const std::vector<int>& input_ids) {
    return greedy_generate(input_ids);
}

void AdvancedGenerator::apply_repetition_penalty(std::vector<Value::Ptr>& logits, 
                                                   const std::vector<int>& tokens,
                                                   float penalty) {
    std::unordered_map<int, int> freq;
    for (int token : tokens) {
        freq[token]++;
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        auto it = freq.find(static_cast<int>(i));
        if (it != freq.end()) {
            double score = logits[i]->data;
            if (score > 0) {
                score /= penalty;
            } else {
                score *= penalty;
            }
            logits[i]->data = score;
        }
    }
}

} // namespace advanced_generation