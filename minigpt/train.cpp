// train.cpp
#include "train.h"
#include "model.h"
#include "tokenizer.h"
#include "optim.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

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

void save_checkpoint(const MiniGPT& model, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    auto params = model.parameters();
    uint32_t n = params.size();
    file.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (auto& p : params) {
        file.write(reinterpret_cast<const char*>(&p->data), sizeof(double));
    }
    file.close();
}

void load_checkpoint(MiniGPT& model, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Tidak bisa membuka checkpoint");
    auto params = model.parameters();
    uint32_t n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n != params.size()) throw std::runtime_error("Jumlah parameter tidak cocok");
    for (auto& p : params) {
        file.read(reinterpret_cast<char*>(&p->data), sizeof(double));
    }
    file.close();
}

void train(MiniGPT& model, Tokenizer& tokenizer,
           const std::string& data_path, const std::string& ckpt_path,
           const ModelConfig& config) {
    std::cout << "Memulai training...\n";
    std::cout << "Model: d_model=" << model.d_model
              << ", layers=" << model.blocks.size()
              << ", heads=" << model.blocks[0].attn.n_heads << "\n";

    // FIX (unifikasi config): semua parameter optimizer sekarang dari
    // ModelConfig (beta1/beta2/eps/weight_decay), bukan angka hardcoded
    // terpisah (0.9, 0.999, 1e-8) yang sebelumnya bisa beda diam-diam
    // dari nilai yang dipakai kalau training dijalankan dari Python.
    auto params = model.parameters();
    AdamW optimizer(params, config.learning_rate, config.beta1, config.beta2,
                     config.eps, config.weight_decay, true);

    // Load dataset dulu, supaya total_steps yang sebenarnya diketahui
    // sebelum scheduler dibuat.
    auto batches = load_dataset(data_path, tokenizer, model.max_len);
    std::cout << "Total batch: " << batches.size() << "\n";

    int total_steps = config.epochs * static_cast<int>(batches.size());

    WarmupCosineScheduler scheduler(&optimizer, config.warmup_steps, total_steps,
                                     config.learning_rate, 1e-5);

    // Catatan desain: loop ini training SGD per-sequence (1 sequence =
    // 1 langkah optimizer), config.batch_size TIDAK dipakai di sini.
    // Kalau kamu butuh mini-batch beneran di jalur C++ murni ini (bukan
    // lewat training.py yang sudah batching), itu perlu perubahan
    // struktural terpisah (akumulasi loss beberapa sequence sebelum
    // backward(), seperti train_batch() di training.py).
    int step = 0;
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(batches.begin(), batches.end(), std::default_random_engine(std::random_device{}()));

        for (const auto& seq : batches) {
            std::vector<int> input_ids(seq.begin(), seq.end() - 1);
            std::vector<int> target_ids(seq.begin() + 1, seq.end());

            std::vector<std::vector<ValuePtr>> logits = model.forward(input_ids, {});
            ValuePtr loss = cross_entropy_loss(logits, target_ids, {});

            optimizer.zero_grad();
            loss->backward();

            optimizer.step();
            scheduler.step();

            step++;
            if (step % 100 == 0) {
                std::cout << "Step " << step << ", loss = " << loss->data << ", lr = " << optimizer.lr << "\n";
            }
            if (step % 1000 == 0) {
                save_checkpoint(model, ckpt_path);
                std::cout << "Checkpoint disimpan di " << ckpt_path << "\n";
            }
        }
        std::cout << "Epoch " << epoch+1 << " selesai.\n";
    }

    save_checkpoint(model, ckpt_path);
    std::cout << "Training selesai. Model terakhir disimpan di " << ckpt_path << "\n";
}