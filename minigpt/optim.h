#ifndef OPTIM_H
#define OPTIM_H

#include "value.h"
#include <vector>
#include <memory>

// Forward declarations
ValuePtr cross_entropy_loss(const std::vector<std::vector<ValuePtr>>& logits_seq,
                            const std::vector<int>& target_ids,
                            const std::vector<int>& pad_mask);

// Log softmax untuk stabilitas numerik yang lebih baik
std::vector<ValuePtr> log_softmax(const std::vector<ValuePtr>& x);

class AdamW {
public:
    AdamW(std::vector<ValuePtr> params, double lr = 1e-3, double betas1 = 0.9, 
          double betas2 = 0.999, double eps = 1e-8, double weight_decay = 0.01, 
          bool decoupled_wd = true);
    
    void step();
    void zero_grad();
    
    double lr;  // Learning rate (dapat diubah oleh scheduler)
    
private:
    std::vector<ValuePtr> params;
    double b1, b2, eps, wd;
    bool decoupled_wd;  // Gunakan decoupled weight decay (AdamW paper)
    std::vector<double> m, v;
    int t;
};

double clip_grad_norm(std::vector<ValuePtr>& params, double max_norm);

class WarmupCosineScheduler {
public:
    WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, 
                          double base_lr = 1e-3, double min_lr = 1e-5);
    
    double step();  // Mengembalikan learning rate yang baru
    
private:
    AdamW* opt;
    int warmup, total, step_num;
    double base_lr, min_lr;
};

#endif // OPTIM_H