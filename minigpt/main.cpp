// main.cpp
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <getopt.h>   // jika ada; jika tidak, kita pakai manual

#include "model.h"
#include "tokenizer.h"
#include "optim.h"
#include "utils.h"

// Deklarasi fungsi training (akan diimplementasikan di train.cpp)
void train(MiniGPT& model, Tokenizer& tokenizer, 
           const std::string& data_path, const std::string& ckpt_path,
           int epochs, double lr, double wd, int warmup_steps);

// Deklarasi fungsi generasi (bisa di generation.cpp, kita buat sederhana)
void generate(MiniGPT& model, Tokenizer& tokenizer,
              const std::string& prompt, int max_tokens);

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <train|generate>   Mode operasi (default: train)\n"
              << "  --data <path>             Path dataset (untuk training)\n"
              << "  --ckpt <path>             Path checkpoint (.bin) untuk load/save\n"
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

    // Inisialisasi tokenizer (pastikan file vocab dan merges ada)
    Tokenizer tokenizer;
    try {
        tokenizer.load("vocab.json", "merges.txt"); // sesuaikan nama file
    } catch (...) {
        std::cerr << "Gagal memuat tokenizer. Pastikan vocab.json dan merges.txt ada.\n";
        return 1;
    }

    // Konfigurasi model (sesuaikan dengan ukuran yang diinginkan)
    // Untuk GPT2 kecil, misal: vocab_size=50257, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_len=1024
    // Tapi untuk cepat, kita pakai ukuran kecil
    int vocab_size = tokenizer.vocab_size(); // asumsikan ada method vocab_size()
    MiniGPT model(vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512, max_len=512, dropout=0.1);

    // Jika checkpoint ada, muat
    if (file_exists(ckpt_path)) {
        std::cout << "Memuat checkpoint dari " << ckpt_path << "\n";
        load_checkpoint(model, ckpt_path); // kita implementasikan
    }

    if (mode == "train") {
        train(model, tokenizer, data_path, ckpt_path, epochs, lr, wd, warmup_steps);
    } else if (mode == "generate") {
        generate(model, tokenizer, prompt, max_tokens);
    } else {
        std::cerr << "Mode tidak valid: " << mode << "\n";
        return 1;
    }

    return 0;
}