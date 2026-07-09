#include "optim.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

ValuePtr cross_entropy_loss(const std::vector<std::vector<ValuePtr>>& logits_seq,
                            const std::vector<int>& target_ids,
                            const std::vector<int>& pad_mask) {
    std::vector<ValuePtr> losses;
    for (size_t i = 0; i < logits_seq.size(); ++i) {
        if (pad_mask[i] == 0) continue;
        auto probs = softmax(logits_seq[i]);
        losses.push_back(log(probs[target_ids[i]]) * -1.0);
    }
    if (losses.empty()) return Value::create(0.0);
    ValuePtr total = Value::create(0.0);
    for (auto& l : losses) total = total + l;
    return total / (double)losses.size();
}

AdamW::AdamW(std::vector<ValuePtr> params, double lr, double betas1, double betas2, double eps, double weight_decay)
    : params(params), lr(lr), b1(betas1), b2(betas2), eps(eps), wd(weight_decay),
      m(params.size(), 0.0), v(params.size(), 0.0), t(0) {}

void AdamW::step() {
    t++;
    double bc1 = 1 - std::pow(b1, t);
    double bc2 = 1 - std::pow(b2, t);
    for (size_t i = 0; i < params.size(); ++i) {
        double g = params[i]->grad;
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        double m_hat = m[i] / bc1;
        double v_hat = v[i] / bc2;
        params[i]->data -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * params[i]->data);
    }
}

void AdamW::zero_grad() {
    for (auto& p : params) p->grad = 0.0;
}

double clip_grad_norm(std::vector<ValuePtr>& params, double max_norm) {
    double total_sq = 0.0;
    for (auto& p : params) total_sq += p->grad * p->grad;
    double norm = std::sqrt(total_sq);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (auto& p : params) p->grad *= scale;
    }
    return norm;  // Kembalikan nilai norm
}

WarmupCosineScheduler::WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, double base_lr, double min_lr)
    : opt(opt), warmup(warmup_steps), total(total_steps), base_lr(base_lr), min_lr(min_lr), step_num(0) {}

double WarmupCosineScheduler::step() {
    step_num++;
    double lr;
    if (step_num < warmup)
        lr = base_lr * step_num / warmup;
    else {
        double progress = (double)(step_num - warmup) / (total - warmup);
        progress = std::min(1.0, progress);
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + std::cos(M_PI * progress));
    }
    opt->lr = lr;
    return lr;
}