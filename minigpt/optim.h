#pragma once

#include <vector>
#include "value.h"

// ============================================================
// CROSS ENTROPY LOSS
// ============================================================
ValuePtr cross_entropy_loss(
    const std::vector<std::vector<ValuePtr>>& logits_seq,
    const std::vector<int>& target_ids,
    const std::vector<int>& pad_mask
);

// ============================================================
// ADAMW OPTIMIZER
// ============================================================
struct AdamW {
    std::vector<ValuePtr> params;
    double lr;
    double b1;
    double b2;
    double eps;
    double wd;

    std::vector<double> m;
    std::vector<double> v;

    int t;

    explicit AdamW(
        std::vector<ValuePtr> params,
        double lr = 0.01,
        double betas1 = 0.9,
        double betas2 = 0.999,
        double eps = 1e-8,
        double weight_decay = 0.01
    );

    void step();
    void zero_grad() noexcept;
};

// ============================================================
// GRADIENT CLIPPING
// ============================================================
double clip_grad_norm(
    std::vector<ValuePtr>& params,
    double max_norm
);

// ============================================================
// WARMUP COSINE SCHEDULER
// ============================================================
struct WarmupCosineScheduler {
    AdamW* opt;

    int warmup;
    int total;
    int step_num;

    double base_lr;
    double min_lr;

    WarmupCosineScheduler(
        AdamW* opt,
        int warmup_steps,
        int total_steps,
        double base_lr,
        double min_lr = 1e-5
    );

    double step();
};