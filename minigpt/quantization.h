// quantization.h
#pragma once
#include "model.h"
#include "value.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace quantization {

// Quantize float to int8 with scale and zero point
inline void quantize_float_to_int8(const std::vector<double>& data,
                                   std::vector<int8_t>& quantized,
                                   double& scale,
                                   int8_t& zero_point) {
    if (data.empty()) return;
    
    // Find min and max
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    
    // Calculate scale and zero point
    scale = (max_val - min_val) / 255.0;
    zero_point = static_cast<int8_t>(std::round(128.0 - min_val / scale));
    
    // Quantize
    quantized.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double q = data[i] / scale + 128.0;
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
        data[i] = (static_cast<double>(quantized[i]) + 128.0 - static_cast<double>(zero_point)) * scale;
    }
}

// Quantize model parameters
inline void quantize_model(MiniGPT& model) {
    auto params = model.parameters();
    for (auto& p : params) {
        // Store quantization parameters
        // In practice, you'd store scale and zero_point per parameter
        // This is a simplified version
    }
}

// Prune weights below threshold
inline void prune_weights(MiniGPT& model, double threshold = 0.01) {
    auto params = model.parameters();
    int pruned_count = 0;
    int total_count = 0;
    
    for (auto& p : params) {
        // This is simplified - in practice, you'd need to access
        // the actual weight tensors
        double val = p->data;
        if (std::abs(val) < threshold) {
            p->data = 0.0;
            pruned_count++;
        }
        total_count++;
    }
    
    double pruned_ratio = static_cast<double>(pruned_count) / total_count * 100.0;
    std::cout << "🧹 Pruned " << pruned_count << " / " << total_count 
              << " weights (" << pruned_ratio << "%)" << std::endl;
}

// Knowledge Distillation
struct DistillationConfig {
    float temperature = 3.0;
    float alpha = 0.7;  // Weight of distillation loss
    float beta = 0.3;   // Weight of student loss
};

inline void distill(MiniGPT& teacher, MiniGPT& student, 
                    DataLoader& dataloader, AdamW& optimizer,
                    const DistillationConfig& config = DistillationConfig()) {
    std::cout << "📚 Starting Knowledge Distillation..." << std::endl;
    
    for (int epoch = 0; epoch < 10; ++epoch) {
        float total_loss = 0.0;
        int batches = 0;
        dataloader.reset();
        
        while (batches < (int)dataloader.num_batches()) {
            auto batch = dataloader.next_batch();
            if (batch.empty()) break;
            
            for (const auto& [input_ids, target_ids] : batch) {
                // Teacher forward
                auto teacher_logits = teacher.forward(input_ids);
                
                // Student forward
                auto student_logits = student.forward(input_ids);
                
                // Compute student loss (cross entropy)
                auto student_loss = cross_entropy_loss(student_logits, target_ids, {});
                
                // Compute distillation loss (KL divergence)
                // Simplified: using MSE between logits
                double distill_loss_val = 0.0;
                for (size_t pos = 0; pos < teacher_logits.size() && pos < student_logits.size(); ++pos) {
                    for (size_t i = 0; i < teacher_logits[pos].size() && i < student_logits[pos].size(); ++i) {
                        double diff = teacher_logits[pos][i]->data - student_logits[pos][i]->data;
                        distill_loss_val += diff * diff;
                    }
                }
                auto distill_loss = Value::create(distill_loss_val);
                
                // Combined loss
                auto combined = Value::create(
                    config.alpha * distill_loss_val + config.beta * student_loss->data
                );
                
                // Backward
                optimizer.zero_grad();
                combined->backward();
                optimizer.step();
                
                total_loss += combined->data;
                batches++;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batches << std::endl;
    }
}

// Model compression
inline void compress_model(MiniGPT& model, double compression_ratio = 0.5) {
    std::cout << "🔧 Compressing model to " << (compression_ratio * 100) << "% size..." << std::endl;
    
    // This would involve:
    // 1. Pruning
    // 2. Quantization
    // 3. Low-rank factorization
    // Implement based on your model structure
}

} // namespace quantization