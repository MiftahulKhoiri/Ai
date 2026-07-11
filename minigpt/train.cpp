// train.cpp
#include "train.h"  // deklarasi fungsi train
#include "model.h"
#include "tokenizer.h"
#include "optim.h"
#include "tensor.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

// Fungsi pembantu: membaca dataset dan menghasilkan batch token
std::vector<std::vector<int>> load_dataset(const std::string& path, Tokenizer& tokenizer, int block_size) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Tidak bisa membuka file data: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Encode seluruh teks menjadi satu array token
    std::vector<int> tokens = tokenizer.encode(text);
    // Potong menjadi sequence-sepanjang block_size
    std::vector<std::vector<int>> batches;
    for (size_t i = 0; i + block_size <= tokens.size(); i += block_size) {
        std::vector<int> seq(tokens.begin() + i, tokens.begin() + i + block_size);
        batches.push_back(seq);
    }
    // Jika tersisa, bisa diabaikan atau ditambahkan dengan padding
    return batches;
}

// Fungsi training utama
void train(const Config& config, const std::string& data_path, const std::string& ckpt_path) {
    std::cout << "Memulai training...\n";
    std::cout << "Config: hidden=" << config.n_embd 
              << ", layers=" << config.n_layer 
              << ", heads=" << config.n_head << "\n";

    // 1. Buat model
    Model model(config);
    model.init_weights();  // asumsikan ada method inisialisasi

    // 2. Coba load checkpoint jika ada
    if (file_exists(ckpt_path)) {
        std::cout << "Memuat checkpoint dari " << ckpt_path << "\n";
        model.load(ckpt_path);
    }

    // 3. Buat optimizer (misal AdamW)
    AdamW optimizer(model.parameters(), config.learning_rate, config.beta1, config.beta2, config.weight_decay);

    // 4. Load dataset
    Tokenizer tokenizer;  // sebaiknya tokenizer sudah diinisialisasi di main dan diteruskan
    tokenizer.load("vocab.json", "merges.txt");
    auto batches = load_dataset(data_path, tokenizer, config.block_size);
    std::cout << "Total batch: " << batches.size() << "\n";

    // 5. Loop training
    int epoch = 0;
    int total_steps = 0;
    const int eval_interval = 100;
    const int save_interval = 1000;

    while (epoch < config.num_epochs) {
        // Shuffle batch
        std::shuffle(batches.begin(), batches.end(), std::default_random_engine(std::random_device{}()));

        for (const auto& seq : batches) {
            // Siapkan input dan target (shifted)
            // input = seq[0:-1], target = seq[1:]
            std::vector<int> input_tokens(seq.begin(), seq.end() - 1);
            std::vector<int> target_tokens(seq.begin() + 1, seq.end());

            Tensor input = tensor_from_vector(input_tokens);   // shape (block_size-1,)
            Tensor target = tensor_from_vector(target_tokens);

            // Forward
            Tensor logits = model.forward(input);
            Tensor loss = cross_entropy_loss(logits, target);

            // Backward
            model.zero_grad();
            loss.backward();

            // Update bobot
            optimizer.step();

            total_steps++;
            if (total_steps % eval_interval == 0) {
                std::cout << "Step " << total_steps << ", loss = " << loss.item() << "\n";
            }
            if (total_steps % save_interval == 0) {
                model.save(ckpt_path);
                std::cout << "Checkpoint disimpan di " << ckpt_path << "\n";
            }
        }
        epoch++;
        std::cout << "Epoch " << epoch << " selesai.\n";
    }

    // Simpan final
    model.save(ckpt_path);
    std::cout << "Training selesai. Model terakhir disimpan di " << ckpt_path << "\n";
}