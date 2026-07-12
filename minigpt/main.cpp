// main.cpp
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <filesystem>   // FIX: untuk cek keberadaan file checkpoint

#include "model.h"
#include "tokenizer.h"
#include "train.h"        // FIX: sumber alias "Tokenizer", load_checkpoint(), train()
#include "generation.h"   // FIX: sumber definisi generate() yang sebenarnya (6 parameter)
#include "optim.h"
#include "utils.h"

// FIX: file_exists() sebelumnya dipanggil tapi tidak pernah didefinisikan
// di manapun. Pakai std::filesystem (C++17) untuk cek keberadaan file.
bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <train|generate>   Mode operasi (default: train)\n"
              << "  --data <path>             Path dataset (untuk training)\n"
              << "  --ckpt <path>             Path checkpoint (.bin) untuk load/save\n"
              << "  --vocab <path>            Path file vocab tokenizer (.json)\n"
              << "  --prompt <text>           Prompt untuk generasi\n"
              << "  --max-tokens <int>        Maksimum token yang digenerate (default: 50)\n"
              << "  --epochs <int>            Jumlah epoch (default: 10)\n"
              << "  --lr <float>              Learning rate (default: 1e-3)\n"
              << "  --wd <float>              Weight decay (default: 0.01)\n"
              << "  --warmup <int>            Warmup steps (default: 100)\n"
              << "  --help                    Tampilkan bantuan\n";
}

int main(int argc, char* argv[]) {
    // Default values
    std::string mode = "train";
    std::string data_path = "data.txt";
    std::string ckpt_path = "model.bin";
    std::string vocab_path = "vocab.json";   // FIX: satu file, bukan vocab.json + merges.txt
    std::string prompt = "";
    int max_tokens = 50;
    int epochs = 10;
    double lr = 1e-3;
    double wd = 0.01;
    int warmup_steps = 100;

    // Parsing argumen sederhana (manual)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i+1 < argc) mode = argv[++i];
        else if (arg == "--data" && i+1 < argc) data_path = argv[++i];
        else if (arg == "--ckpt" && i+1 < argc) ckpt_path = argv[++i];
        else if (arg == "--vocab" && i+1 < argc) vocab_path = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--epochs" && i+1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i+1 < argc) lr = std::stod(argv[++i]);
        else if (arg == "--wd" && i+1 < argc) wd = std::stod(argv[++i]);
        else if (arg == "--warmup" && i+1 < argc) warmup_steps = std::stoi(argv[++i]);
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Argumen tidak dikenal: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Inisialisasi tokenizer
    Tokenizer tokenizer;
    try {
        // FIX: load() cuma menerima SATU path (file JSON gabungan vocab +
        // merge_order), bukan dua file terpisah gaya GPT-2 asli.
        if (!tokenizer.load(vocab_path)) {
            std::cerr << "Gagal memuat tokenizer dari " << vocab_path << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Gagal memuat tokenizer: " << e.what() << "\n";
        return 1;
    }

    // Konfigurasi model — ukuran kecil untuk eksperimen cepat
    int vocab_size = tokenizer.vocab_size();

    // FIX: "d_model=128" dkk bukan named-argument C++ (itu sintaks
    // Python). Di C++ ini di-parse sebagai assignment ke variabel
    // "d_model" yang tidak pernah dideklarasikan -> gagal compile.
    // Parameter harus dikirim positional sesuai urutan konstruktor
    // MiniGPT di model.h: (vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout)
    MiniGPT model(vocab_size, 128, 4, 4, 512, 512, 0.1);

    // Jika checkpoint ada, muat
    if (file_exists(ckpt_path)) {
        std::cout << "Memuat checkpoint dari " << ckpt_path << "\n";
        load_checkpoint(model, ckpt_path);
    }

    if (mode == "train") {
        train(model, tokenizer, data_path, ckpt_path, epochs, lr, wd, warmup_steps);
    } else if (mode == "generate") {
        // FIX: pakai generate() asli dari generation.h (6 parameter,
        // 2 dengan default), bukan deklarasi lokal 4-parameter yang
        // tidak match simbol sebenarnya di generation.cpp.
        generate(model, tokenizer, prompt, max_tokens, /*add_bos=*/true, /*add_eos=*/false);
    } else {
        std::cerr << "Mode tidak valid: " << mode << "\n";
        return 1;
    }

    return 0;
}