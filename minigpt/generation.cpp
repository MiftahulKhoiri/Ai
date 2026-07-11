// generation.cpp
#include "generation.h"
#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Fungsi argmax: cari indeks dengan nilai terbesar
int argmax(const std::vector<Value::Ptr>& logits) {
    if (logits.empty()) return -1;
    
    int max_idx = 0;
    double max_val = logits[0]->data;
    
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i]->data > max_val) {
            max_val = logits[i]->data;
            max_idx = i;
        }
    }
    return max_idx;
}

// Fungsi sampling top-k dengan temperature
std::vector<int> sample_top_k(const std::vector<Value::Ptr>& logits, int k, float temperature) {
    if (logits.empty()) return {};
    if (k <= 0) k = logits.size();
    
    // Buat vector pasangan (indeks, nilai)
    std::vector<std::pair<int, double>> scored;
    scored.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scored.push_back({static_cast<int>(i), logits[i]->data});
    }
    
    // Urutkan berdasarkan nilai (terbesar ke terkecil)
    std::sort(scored.begin(), scored.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Ambil top-k
    if (k < (int)scored.size()) {
        scored.resize(k);
    }
    
    // Apply temperature
    if (temperature != 1.0 && temperature > 0.0) {
        for (auto& p : scored) {
            p.second = std::exp(p.second / temperature);
        }
    }
    
    // Normalisasi menjadi probabilitas
    double sum = 0.0;
    for (const auto& p : scored) {
        sum += p.second;
    }
    
    if (sum == 0.0 || std::isnan(sum)) {
        // Jika semua nilai 0, return indeks pertama
        return {scored[0].first};
    }
    
    // Sampling berdasarkan probabilitas
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

void generate(MiniGPT& model, ByteLevelBPETokenizer& tokenizer,
              const std::string& prompt, int max_tokens,
              bool add_bos, bool add_eos) {
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated: ";

    // Encode prompt
    std::vector<int> input_ids = tokenizer.encode(prompt, add_bos, add_eos);
    if (input_ids.empty()) {
        std::cerr << "Prompt kosong atau tidak bisa di-encode.\n";
        return;
    }

    // Set model ke mode evaluasi (non-training)
    model.set_training(false);

    // Inisialisasi cache untuk incremental generation
    model.init_cache();

    // Variabel untuk menyimpan token terakhir yang dihasilkan
    int last_token = input_ids.back();
    
    // Generate token by token
    for (int pos = 0; pos < max_tokens; ++pos) {
        int token_id;
        
        // Jika masih ada input prompt, gunakan token berikutnya
        if (pos < (int)input_ids.size()) {
            token_id = input_ids[pos];
        } else {
            // Gunakan token yang terakhir dihasilkan
            token_id = last_token;
        }

        // Forward incremental
        std::vector<Value::Ptr> logits = model.forward_incremental(token_id, pos);
        
        // Jika sudah melewati prompt, sampling dari logits
        if (pos >= (int)input_ids.size()) {
            // Ambil token berikutnya (gunakan argmax untuk deterministik)
            int next_token = argmax(logits);
            
            // Cek EOS - kita perlu cek apakah tokenizer punya EOS
            // (tanpa method get_eos_token_id, kita cek dari vocab)
            // Kita cari token "<eos>" di vocab
            auto& vocab = tokenizer.get_vocab();
            auto eos_it = vocab.find("<eos>");
            if (eos_it != vocab.end() && next_token == eos_it->second) {
                break;
            }
            
            // Cetak token
            std::cout << tokenizer.decode({next_token}) << std::flush;
            
            // Simpan token terakhir
            last_token = next_token;
        }
    }
    std::cout << "\n";
}