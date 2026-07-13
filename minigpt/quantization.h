// quantization.h
#pragma once
#include "model.h"
#include "value.h"
#include "dataloader.h"
#include "optim.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>

namespace quantization {

// Quantize float to int8 with scale and zero point
inline void quantize_float_to_int8(const std::vector<double>& data,
                                   std::vector<int8_t>& quantized,
                                   double& scale,
                                   int8_t& zero_point) {
    if (data.empty()) return;

    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());

    scale = (max_val - min_val) / 255.0;
    if (scale == 0.0) scale = 1.0;

    // FIX: zero_point dihitung dalam ruang uint8 [0,255] dulu (standar
    // affine quantization), lalu digeser ke rentang int8 [-128,127].
    // Sebelumnya zero_point dihitung dari min_val tapi loop quantize di
    // bawah TIDAK memakainya (selalu pakai konstanta 128), sementara
    // dequantize MEMAKAI zero_point ini -- encode dan decode jadi pakai
    // skema berbeda, hasil round-trip salah untuk data yang tidak
    // simetris di sekitar nol (kasus umum untuk bobot neural net).
    double zp_double = std::round(-min_val / scale);
    zp_double = std::max(0.0, std::min(255.0, zp_double));
    int zero_point_u8 = static_cast<int>(zp_double);
    zero_point = static_cast<int8_t>(zero_point_u8 - 128);

    quantized.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        // FIX: pakai zero_point_u8 yang baru dihitung, bukan konstanta 128.
        double q = data[i] / scale + zero_point_u8;
        q = std::max(0.0, std::min(255.0, q));
        quantized[i] = static_cast<int8_t>(std::round(q) - 128);
    }
}

// Dequantize int8 to float
inline void dequantize_int8_to_float(const std::vector<int8_t>& quantized,
                                     std::vector<double>& data,
                                     double scale,
                                     int8_t zero_point) {
    data.resize(quantized.size());
    for (size_t i = 0; i < quantized.size(); ++i) {
        // FIX: konsisten dengan encode -- balikkan pergeseran int8->uint8
        // untuk quantized value maupun zero_point, baru selisihkan.
        int q_u8 = static_cast<int>(quantized[i]) + 128;
        int zp_u8 = static_cast<int>(zero_point) + 128;
        data[i] = static_cast<double>(q_u8 - zp_u8) * scale;
    }
}

// Quantize model parameters
inline void quantize_model(MiniGPT& model) {
    auto params = model.parameters();
    std::cout << "🔢 Quantizing model parameters..." << std::endl;
    (void)model;

    for (auto& p : params) {
        (void)p;
    }
    std::cout << "✅ Model quantization completed (simplified)" << std::endl;
}

// Prune weights below threshold
inline void prune_weights(MiniGPT& model, double threshold = 0.01) {
    auto params = model.parameters();
    int pruned_count = 0;
    int total_count = 0;
    (void)model;

    for (auto& p : params) {
        double val = p->data;
        if (std::abs(val) < threshold) {
            p->data = 0.0;
            pruned_count++;
        }
        total_count++;
    }

    if (total_count > 0) {
        double pruned_ratio = static_cast<double>(pruned_count) / total_count * 100.0;
        std::cout << "🧹 Pruned " << pruned_count << " / " << total_count 
                  << " weights (" << pruned_ratio << "%)" << std::endl;
    } else {
        std::cout << "🧹 No weights to prune" << std::endl;
    }
}

// Knowledge Distillation
struct DistillationConfig {
    float temperature = 3.0;
    float alpha = 0.7;  // Weight of distillation loss
    float beta = 0.3;   // Weight of student loss
    int epochs = 10;
};

inline void distill(MiniGPT& teacher, MiniGPT& student, 
                    DataLoader& dataloader, AdamW& optimizer,
                    const DistillationConfig& config = DistillationConfig()) {
    std::cout << "📚 Starting Knowledge Distillation for " << config.epochs << " epochs..." << std::endl;

    // FIX: teacher HARUS dalam mode eval (dropout mati) supaya "soft
    // target" yang diberikan ke student konsisten/deterministik, bukan
    // acak tiap forward pass. Student tetap dalam mode training.
    teacher.set_training(false);
    student.set_training(true);

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        float total_loss = 0.0;
        int batches = 0;
        dataloader.reset();

        while (batches < (int)dataloader.num_batches()) {
            auto batch = dataloader.next_batch();
            if (batch.empty()) break;

            for (size_t b = 0; b < batch.size(); ++b) {
                const auto& input_ids = batch[b].first;
                const auto& target_ids = batch[b].second;

                auto teacher_logits = teacher.forward(input_ids);
                auto student_logits = student.forward(input_ids);

                auto student_loss = cross_entropy_loss(student_logits, target_ids, {});

                // FIX: distillation loss (MSE antar logits) dibangun dari
                // operasi ValuePtr (+, -, *), BUKAN dihitung sebagai
                // double mentah lalu dibungkus Value::create(). Sebelumnya
                // distill_loss/combined adalah node baru yang lepas total
                // dari graph student_logits -- combined->backward() tidak
                // pernah mengubah grad parameter manapun, jadi
                // optimizer.step() pada dasarnya tidak melatih apa-apa.
                ValuePtr distill_loss = Value::create(0.0);
                int diff_count = 0;
                for (size_t pos = 0; pos < teacher_logits.size() && pos < student_logits.size(); ++pos) {
                    for (size_t i = 0; i < teacher_logits[pos].size() && i < student_logits[pos].size(); ++i) {
                        ValuePtr diff = teacher_logits[pos][i] - student_logits[pos][i];
                        distill_loss = distill_loss + (diff * diff);
                        diff_count++;
                    }
                }
                if (diff_count > 0) {
                    distill_loss = distill_loss / Value::create(static_cast<double>(diff_count));
                }

                // FIX: combined loss juga dibangun dari distill_loss dan
                // student_loss yang keduanya masih terhubung ke graph asli.
                ValuePtr combined = (distill_loss * Value::create(static_cast<double>(config.alpha)))
                                   + (student_loss * Value::create(static_cast<double>(config.beta)));

                optimizer.zero_grad();
                combined->backward();
                optimizer.step();

                total_loss += static_cast<float>(combined->data);
                batches++;
            }
        }

        if (batches > 0) {
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batches << std::endl;
        }
    }
    std::cout << "✅ Knowledge Distillation completed!" << std::endl;
}

// Model compression
inline void compress_model(MiniGPT& model, double compression_ratio = 0.5) {
    std::cout << "🔧 Compressing model to " << (compression_ratio * 100) << "% size..." << std::endl;
    (void)model;

    std::cout << "✅ Model compression completed (simplified)" << std::endl;
}

} // namespace quantization