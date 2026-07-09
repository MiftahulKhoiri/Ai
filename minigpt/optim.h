#pragma once
#include "value.h"
#include <vector>

// ============================================================
// CROSS ENTROPY LOSS
// ============================================================
ValuePtr cross_entropy_loss(const std::vector<std::vector<ValuePtr>>& logits_seq,
                            const std::vector<int>& target_ids,
                            const std::vector<int>& pad_mask);

// ============================================================
// ADAMW OPTIMIZER
// ============================================================
struct AdamW {
    std::vector<ValuePtr> params;
    double lr, b1, b2, eps, wd;
    std::vector<double> m, v;
    int t;

    AdamW(std::vector<ValuePtr> params, double lr = 0.01, double betas1 = 0.9, double betas2 = 0.999,
          double eps = 1e-8, double weight_decay = 0.01);
    void step();
    void zero_grad();
};

// ============================================================
// GRADIENT CLIPPING
// ============================================================
void clip_grad_norm(std::vector<ValuePtr>& params, double max_norm);

// ============================================================
// WARMUP COSINE SCHEDULER
// ============================================================
struct WarmupCosineScheduler {
    AdamW* opt;
    int warmup, total;
    double base_lr, min_lr;
    int step_num;

    WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, double base_lr, double min_lr = 1e-5);
    double step();
};