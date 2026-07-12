// main.cpp
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <filesystem>

#include "model.h"
#include "tokenizer.h"
#include "train.h"
#include "generation.h"
#include "config.h"
#include "optim.h"
#include "utils.h"

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
              << "  --config <path>           Path ModelConfig JSON (default: dipakai jika ada)\n"
              << "  --prompt <text>           Prompt untuk generasi\n"
              << "  --max-tokens <int>        Maksimum token yang digenerate (default: 50)\n"
              << "  --epochs <int>            Override epochs dari config\n"
              << "  --lr <float>              Override learning rate dari config\n"
              << "  --wd <float>              Override weight decay dari config\n"
              << "  --warmup <int>            Override warmup steps dari config\n"
              << "  --help                    Tampilkan bantuan\n";
}

int main(int argc, char* argv[]) {
    std::string mode = "train";
    std::string data_path = "data.txt";
    std::string ckpt_path = "model.bin";
    std::string vocab_path = "vocab.json";
    std::string config_path = "";
    std::string prompt = "";
    int max_tokens = 50;

    // FIX (unifikasi config): flag override ini kalau diisi menimpa
    // nilai dari ModelConfig (baik default maupun hasil --config).
    // Kalau tidak diisi user, config yang menentukan sepenuhnya.
    bool override_epochs = false, override_lr = false, override_wd = false, override_warmup = false;
    int epochs_ovr = 0, warmup_ovr = 0;
    double lr_ovr = 0.0, wd_ovr = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i+1 < argc) mode = argv[++i];
        else if (arg == "--data" && i+1 < argc) data_path = argv[++i];
        else if (arg == "--ckpt" && i+1 < argc) ckpt_path = argv[++i];
        else if (arg == "--vocab" && i+1 < argc) vocab_path = argv[++i];
        else if (arg == "--config" && i+1 < argc) config_path = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--epochs" && i+1 < argc) { epochs_ovr = std::stoi(argv[++i]); override_epochs = true; }
        else if (arg == "--lr" && i+1 < argc) { lr_ovr = std::stod(argv[++i]); override_lr = true; }
        else if (arg == "--wd" && i+1 < argc) { wd_ovr = std::stod(argv[++i]); override_wd = true; }
        else if (arg == "--warmup" && i+1 < argc) { warmup_ovr = std::stoi(argv[++i]); override_warmup = true; }
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Argumen tidak dikenal: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // FIX (unifikasi config): satu ModelConfig sebagai sumber kebenaran
    // tunggal. Kalau --config diberikan dan filenya ada, load dari situ;
    // kalau tidak, pakai default ModelConfig. CLI flag individual (kalau
    // diisi) meng-override field yang relevan di atasnya.
    ModelConfig config;
    if (!config_path.empty() && file_exists(config_path)) {
        config = ModelConfig::from_json(config_path);
        std::cout << "Config dimuat dari " << config_path << "\n";
    } else {
        std::cout << "Menggunakan ModelConfig default" << (config_path.empty() ? "" : " (file tidak ditemukan)") << "\n";
    }
    if (override_epochs) config.epochs = epochs_ovr;
    if (override_lr) config.learning_rate = lr_ovr;
    if (override_wd) config.weight_decay = wd_ovr;
    if (override_warmup) config.warmup_steps = warmup_ovr;

    Tokenizer tokenizer;
    try {
        if (!tokenizer.load(vocab_path)) {
            std::cerr << "Gagal memuat tokenizer dari " << vocab_path << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Gagal memuat tokenizer: " << e.what() << "\n";
        return 1;
    }

    // FIX (unifikasi config): vocab_size aktual dari tokenizer menang
    // atas apa pun yang tertulis di config file (config bisa saja stale
    // kalau tokenizer di-retrain dengan target vocab_size berbeda).
    config.vocab_size = tokenizer.vocab_size();

    // FIX (unifikasi config): model sekarang dibangun langsung dari
    // config, bukan angka hardcoded (128, 4, 4, 512, 512, 0.1) yang
    // bisa diam-diam beda dari config yang dipakai train().
    MiniGPT model(config.vocab_size, config.d_model, config.n_heads,
                  config.n_layers, config.d_ff, config.max_len, config.dropout);

    if (file_exists(ckpt_path)) {
        std::cout << "Memuat checkpoint dari " << ckpt_path << "\n";
        load_checkpoint(model, ckpt_path);
    }

    if (mode == "train") {
        train(model, tokenizer, data_path, ckpt_path, config);
        // Simpan config yang benar-benar dipakai, supaya reproducible
        // dan bisa dipakai lagi persis sama saat --ckpt ini di-resume.
        config.to_json(ckpt_path + ".config.json");
    } else if (mode == "generate") {
        generate(model, tokenizer, prompt, max_tokens, /*add_bos=*/true, /*add_eos=*/false);
    } else {
        std::cerr << "Mode tidak valid: " << mode << "\n";
        return 1;
    }

    return 0;
}