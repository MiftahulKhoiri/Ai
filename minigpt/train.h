// train.h
#pragma once
#include "model.h"
#include "tokenizer.h"
#include <string>
#include <vector>

// FIX: alias ini penting -- tidak ada kelas C++ bernama "Tokenizer".
// Kelas yang benar-benar ada di tokenizer.h adalah ByteLevelBPETokenizer.
// "Tokenizer" cuma nama yang dipakai di sisi Python (bindings.cpp), bukan
// alias tipe C++. Taruh alias-nya di sini (header), supaya file lain yang
// include train.h otomatis dapat alias ini juga, tidak perlu diulang
// manual di tiap .cpp yang butuh.
using Tokenizer = ByteLevelBPETokenizer;

// Baca file teks, encode dengan tokenizer, potong jadi sequence
// non-overlapping sepanjang block_size.
std::vector<std::vector<int>> load_dataset(const std::string& path,
                                            Tokenizer& tokenizer,
                                            int block_size);

// Simpan seluruh parameter model (flat, satu double per Value) ke file
// biner di path yang diberikan.
void save_checkpoint(const MiniGPT& model, const std::string& path);

// Muat kembali parameter model dari file checkpoint biner. Melempar
// std::runtime_error kalau jumlah/ukuran parameter tidak cocok dengan
// model yang sedang dipakai.
void load_checkpoint(MiniGPT& model, const std::string& path);

// Jalankan training loop lengkap: load dataset, siapkan optimizer +
// scheduler, iterasi per-epoch dengan shuffle, forward/backward/step,
// simpan checkpoint berkala dan di akhir training.
void train(MiniGPT& model,
           Tokenizer& tokenizer,
           const std::string& data_path,
           const std::string& ckpt_path,
           int epochs,
           double lr,
           double wd,
           int warmup_steps);