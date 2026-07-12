// generation.cpp
#include "generation.h"
#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

int argmax(const std::vector<Value::Ptr>& logits) {
    if (logits.empty()) return -1;

    int max_idx = 0;
    double max_val = logits[0]->data;

    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i]->data > max_val) {
            max_val = logits[i]->data;
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

std::vector<int> sample_top_k(const std::vector<Value::Ptr>& logits, int k, float temperature) {
    if (logits.empty()) return {};
    if (k <= 0) k = static_cast<int>(logits.size());

    std::vector<std::pair<int, double>> scored;
    scored.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scored.push_back({static_cast<int>(i), logits[i]->data});
    }

    std::sort(scored.begin(), scored.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (k < (int)scored.size()) {
        scored.resize(k);
    }

    // FIX: exp() harus selalu dijalankan untuk mengubah logit jadi bobot
    // softmax yang valid (non-negatif). Sebelumnya dilewati kalau
    // temperature == 1.0, sehingga logit mentah (bisa negatif) dipakai
    // langsung sebagai "probabilitas" -- normalisasi jadi tidak valid.
    if (temperature > 0.0f) {
        for (auto& p : scored) {
            p.second = std::exp(p.second / temperature);
        }
    } else {
        // temperature tidak valid -> fallback ke greedy (top-1 saja)
        return {scored[0].first};
    }

    double sum = 0.0;
    for (const auto& p : scored) {
        sum += p.second;
    }

    if (sum == 0.0 || std::isnan(sum)) {
        return {scored[0].first};
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double r = dis(gen);
    double cumsum = 0.0;
    for (const auto& p : scored) {
        cumsum += p.second / sum;
        if (r <= cumsum) {
            return {p.first};
        }
    }

    return {scored.back().first};
}

void generate(MiniGPT& model, 
              ByteLevelBPETokenizer& tokenizer,
              const std::string& prompt, 
              int max_tokens,
              bool add_bos,
              bool add_eos) {
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated: ";

    std::vector<int> input_ids = tokenizer.encode(prompt, add_bos, add_eos);
    if (input_ids.empty()) {
        std::cerr << "Prompt kosong atau tidak bisa di-encode.\n";
        return;
    }

    model.set_training(false);
    model.init_cache();

    // FIX: pisahkan fase "isi cache dari prompt" dan fase "generate token
    // baru". Sebelumnya token terakhir prompt di-feed dua kali: sekali di
    // akhir fase prompt (logits dibuang), lalu di-feed ULANG di awal fase
    // generate dengan posisi baru -- merusak KV-cache (posisi ganda untuk
    // konten yang sama) dan membuang logits yang sebenarnya sudah valid
    // untuk prediksi token pertama.

    // Fase 1: feed semua token prompt, simpan logits dari token TERAKHIR
    std::vector<Value::Ptr> logits;
    int pos = 0;
    for (; pos < (int)input_ids.size(); ++pos) {
        logits = model.forward_incremental(input_ids[pos], pos);
    }

    int eos_id = tokenizer.get_eos_token_id();

    // Fase 2: generate token baru satu per satu, mulai dari logits
    // hasil melihat seluruh prompt (tidak feed ulang token terakhir prompt)
    for (int step = 0; step < max_tokens; ++step) {
        int next_token = argmax(logits);

        if (eos_id >= 0 && next_token == eos_id) {
            break;
        }

        std::cout << tokenizer.decode({next_token}) << std::flush;

        // Feed token yang baru saja dihasilkan untuk posisi berikutnya
        logits = model.forward_incremental(next_token, pos);
        pos++;
    }
    std::cout << "\n";
}