// train.h
#pragma once
#include "model.h"
#include "tokenizer.h"
#include "config.h"
#include <string>
#include <vector>

using Tokenizer = ByteLevelBPETokenizer;

std::vector<std::vector<int>> load_dataset(const std::string& path,
                                            Tokenizer& tokenizer,
                                            int block_size);

void save_checkpoint(const MiniGPT& model, const std::string& path);
void load_checkpoint(MiniGPT& model, const std::string& path);

// FIX (unifikasi config): train() sekarang menerima satu ModelConfig,
// bukan epochs/lr/wd/warmup_steps sebagai parameter terpisah. Semua
// hyperparameter training berasal dari satu sumber yang sama dengan
// yang dipakai untuk membangun model di main.cpp -- menghindari drift
// antara config yang dipakai construct model vs config yang dipakai
// training (mis. lupa update salah satu saat eksperimen).
void train(MiniGPT& model,
           Tokenizer& tokenizer,
           const std::string& data_path,
           const std::string& ckpt_path,
           const ModelConfig& config);