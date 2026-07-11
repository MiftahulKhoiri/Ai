// main.cpp
#include <iostream>
#include <string>
#include <cstdlib>
#include <getopt.h>  // atau gunakan argparse buatan sendiri

#include "model.h"
#include "tokenizer.h"
#include "optim.h"
#include "utils.h"

// Deklarasi fungsi training (akan diimplementasikan di train.cpp)
void train(const Config& config, const std::string& data_path, const std::string& ckpt_path);
// Deklarasi fungsi generasi (mungkin di generation.cpp)
void generate(const Config& config, const std::string& ckpt_path, const std::string& prompt, int max_tokens);

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <train|generate>   Mode operasi (default: train)\n"
              << "  --config <path>           Path ke file config.json\n"
              << "  --ckpt <path>             Path checkpoint (.bin) untuk load/save\n"
              << "  --data <path>             Path dataset (untuk training)\n"
              << "  --prompt <text>           Prompt untuk generasi\n"
              << "  --max-tokens <int>        Maksimum token yang digenerate\n"
              << "  --help                    Tampilkan bantuan\n";
}

int main(int argc, char* argv[]) {
    // Default values
    std::string mode = "train";
    std::string config_path = "config.json";
    std::string ckpt_path = "model.bin";
    std::string data_path = "data.txt";
    std::string prompt = "";
    int max_tokens = 100;

    // Parsing argumen sederhana (gunakan getopt_long jika tersedia)
    // Di sini saya contohkan dengan manual agar portabel
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i+1 < argc) mode = argv[++i];
        else if (arg == "--config" && i+1 < argc) config_path = argv[++i];
        else if (arg == "--ckpt" && i+1 < argc) ckpt_path = argv[++i];
        else if (arg == "--data" && i+1 < argc) data_path = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Argumen tidak dikenal: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 1. Baca konfigurasi dari file JSON (Anda perlu implementasi load_config)
    Config config = load_config(config_path);

    // 2. Inisialisasi tokenizer (pastikan file vocab.json dan merges.txt ada)
    Tokenizer tokenizer;
    tokenizer.load("vocab.json", "merges.txt");

    // 3. Jalankan sesuai mode
    if (mode == "train") {
        train(config, data_path, ckpt_path);
    } else if (mode == "generate") {
        generate(config, ckpt_path, prompt, max_tokens);
    } else {
        std::cerr << "Mode tidak valid: " << mode << "\n";
        return 1;
    }

    return 0;
}