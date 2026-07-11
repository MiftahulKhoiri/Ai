// generation.cpp
#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <vector>

void generate(MiniGPT& model, Tokenizer& tokenizer,
              const std::string& prompt, int max_tokens) {
    std::cout << "Prompt: " << prompt << "\n";

    // Encode prompt
    std::vector<int> input_ids = tokenizer.encode(prompt);
    if (input_ids.empty()) {
        std::cerr << "Prompt kosong.\n";
        return;
    }

    // Set model ke mode evaluasi (non-training)
    model.set_training(false);

    // Inisialisasi cache untuk incremental generation
    model.init_cache();

    // Generate token by token
    for (int pos = 0; pos < max_tokens; ++pos) {
        // Jika masih ada input prompt, gunakan token berikutnya
        // Jika sudah habis, gunakan token terakhir yang digenerate
        int token_id;
        if (pos < (int)input_ids.size()) {
            token_id = input_ids[pos];
        } else {
            // Gunakan token yang terakhir dihasilkan (disimpan di suatu variabel)
            token_id = last_token; // kita perlu simpan
        }

        // Forward incremental
        std::vector<ValuePtr> logits = model.forward_incremental(token_id, pos);

        // Ambil argmax dari logits (logits adalah ValuePtr, ambil data dan cari indeks max)
        // Karena logits adalah vector<ValuePtr> untuk satu token? Sesuai signature:
        // std::vector<ValuePtr> forward_incremental(int token_id, int pos);
        // Ini mengembalikan logits untuk token saat ini (shape [vocab_size])
        // Cari indeks dengan nilai terbesar
        int next_token = argmax(logits); // fungsi sederhana

        // Cetak token
        std::cout << tokenizer.decode({next_token}) << std::flush;

        // Simpan token terakhir
        last_token = next_token;
        // Jika token EOS, berhenti
        if (next_token == tokenizer.eos_token_id()) break;
    }
    std::cout << "\n";
}