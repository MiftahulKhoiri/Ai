// train.cpp
#include "train.h"   // deklarasi fungsi train
#include "model.h"
#include "tokenizer.h"
#include "optim.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

// Fungsi pembantu: baca file teks, enkode, dan potong menjadi sequence
std::vector<std::vector<int>> load_dataset(const std::string& path, Tokenizer& tokenizer, int block_size) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Tidak bisa membuka file: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::vector<int> tokens = tokenizer.encode(text);
    std::vector<std::vector<int>> batches;
    for (size_t i = 0; i + block_size <= tokens.size(); i += block_size) {
        std::vector<int> seq(tokens.begin() + i, tokens.begin() + i + block_size);
        batches.push_back(seq);
    }
    return batches;
}

// Fungsi untuk menyimpan checkpoint (simpan semua parameter model)
void save_checkpoint(const MiniGPT& model, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    auto params = model.parameters();
    // Tulis jumlah parameter dulu
    uint32_t n = params.size();
    file.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (auto& p : params) {
        // Asumsikan ValuePtr memiliki method data() mengembalikan pointer ke double
        // dan size() mengembalikan jumlah elemen
        uint32_t size = p->size(); // sesuaikan
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(p->data()), size * sizeof(double));
    }
    file.close();
}

// Fungsi untuk memuat checkpoint
void load_checkpoint(MiniGPT& model, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Tidak bisa membuka checkpoint");
    auto params = model.parameters();
    uint32_t n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n != params.size()) throw std::runtime_error("Jumlah parameter tidak cocok");
    for (auto& p : params) {
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (size != p->size()) throw std::runtime_error("Ukuran parameter tidak cocok");
        file.read(reinterpret_cast<char*>(p->data()), size * sizeof(double));
    }
    file.close();
}

// Implementasi training
void train(MiniGPT& model, Tokenizer& tokenizer, 
           const std::string& data_path, const std::string& ckpt_path,
           int epochs, double lr, double wd, int warmup_steps) {
    std::cout << "Memulai training...\n";
    std::cout << "Model: d_model=" << model.d_model 
              << ", layers=" << model.blocks.size() 
              << ", heads=" << model.blocks[0].attn.n_heads << "\n";  // perhatikan akses

    // Siapkan optimizer
    auto params = model.parameters();
    AdamW optimizer(params, lr, 0.9, 0.999, 1e-8, wd, true);

    // Scheduler warmup cosine
    WarmupCosineScheduler scheduler(&optimizer, warmup_steps, epochs * 1000, lr, 1e-5);

    // Load dataset
    auto batches = load_dataset(data_path, tokenizer, model.max_len);
    std::cout << "Total batch: " << batches.size() << "\n";

    // Loop training
    int step = 0;
    int total_steps = epochs * batches.size();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle batch
        std::shuffle(batches.begin(), batches.end(), std::default_random_engine(std::random_device{}()));

        for (const auto& seq : batches) {
            // Input: semua token kecuali yang terakhir
            std::vector<int> input_ids(seq.begin(), seq.end() - 1);
            // Target: semua token kecuali yang pertama
            std::vector<int> target_ids(seq.begin() + 1, seq.end());

            // Forward
            std::vector<std::vector<ValuePtr>> logits = model.forward(input_ids, {}); // pad_mask kosong

            // Hitung loss
            ValuePtr loss = cross_entropy_loss(logits, target_ids, {}); // pad_mask kosong

            // Backward
            model.zero_grad(); // asumsikan ada method zero_grad
            loss->backward();

            // Clip grad (opsional)
            // clip_grad_norm(params, 1.0);

            // Step optimizer
            optimizer.step();
            scheduler.step();

            step++;
            if (step % 100 == 0) {
                std::cout << "Step " << step << ", loss = " << loss->data() << ", lr = " << optimizer.lr << "\n";
            }
            if (step % 1000 == 0) {
                save_checkpoint(model, ckpt_path);
                std::cout << "Checkpoint disimpan di " << ckpt_path << "\n";
            }
        }
        std::cout << "Epoch " << epoch+1 << " selesai.\n";
    }

    // Simpan final
    save_checkpoint(model, ckpt_path);
    std::cout << "Training selesai. Model terakhir disimpan di " << ckpt_path << "\n";
}