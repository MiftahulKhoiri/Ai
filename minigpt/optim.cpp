#include "optim.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

ValuePtr cross_entropy_loss(const std::vector<std::vector<ValuePtr>>& logits_seq,
                            const std::vector<int>& target_ids,
                            const std::vector<int>& pad_mask) {
    // Validasi ukuran
    if (logits_seq.empty() || target_ids.empty() || pad_mask.empty()) {
        return Value::create(0.0);
    }
    
    if (logits_seq.size() != target_ids.size() || logits_seq.size() != pad_mask.size()) {
        throw std::invalid_argument(
            "Size mismatch: logits_seq=" + std::to_string(logits_seq.size()) +
            ", target_ids=" + std::to_string(target_ids.size()) +
            ", pad_mask=" + std::to_string(pad_mask.size())
        );
    }
    
    std::vector<ValuePtr> losses;
    losses.reserve(logits_seq.size());
    
    for (size_t i = 0; i < logits_seq.size(); ++i) {
        if (pad_mask[i] == 0) continue;
        
        // Validasi indeks target
        if (target_ids[i] < 0 || target_ids[i] >= static_cast<int>(logits_seq[i].size())) {
            throw std::out_of_range(
                "Target ID " + std::to_string(target_ids[i]) +
                " out of range [0, " + std::to_string(logits_seq[i].size()) + ") at position " + std::to_string(i)
            );
        }
        
        if (logits_seq[i].empty()) continue;
        
        // Gunakan log_softmax untuk stabilitas numerik yang lebih baik
        auto log_probs = log_softmax(logits_seq[i]);
        losses.push_back(log_probs[target_ids[i]] * -1.0);
    }
    
    if (losses.empty()) return Value::create(0.0);
    
    ValuePtr total = Value::create(0.0);
    for (const auto& l : losses) {
        total = total + l;
    }
    return total / static_cast<double>(losses.size());
}

// Implementasi log_softmax yang lebih stabil secara numerik
std::vector<ValuePtr> log_softmax(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};
    
    // Temukan nilai maksimum untuk stabilitas numerik
    double max_val = -std::numeric_limits<double>::infinity();
    for (const auto& val : x) {
        if (val->data > max_val) {
            max_val = val->data;
        }
    }
    
    // Hitung exp(x - max) dan sum
    std::vector<ValuePtr> exp_vals;
    exp_vals.reserve(x.size());
    ValuePtr sum_exp = Value::create(0.0);
    
    for (const auto& val : x) {
        auto shifted = val - max_val;
        auto exp_val = exp(shifted);
        exp_vals.push_back(exp_val);
        sum_exp = sum_exp + exp_val;
    }
    
    // Hitung log_softmax: x - max - log(sum_exp)
    auto log_sum = log(sum_exp);
    std::vector<ValuePtr> result;
    result.reserve(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        result.push_back(x[i] - max_val - log_sum);
    }
    
    return result;
}

AdamW::AdamW(std::vector<ValuePtr> params, double lr, double betas1, double betas2, 
             double eps, double weight_decay, bool decoupled_wd)
    : params(std::move(params)), lr(lr), b1(betas1), b2(betas2), eps(eps), wd(weight_decay),
      decoupled_wd(decoupled_wd), t(0) {
    if (this->params.empty()) {
        throw std::invalid_argument("Parameter list cannot be empty");
    }
    if (lr <= 0) {
        throw std::invalid_argument("Learning rate must be positive, got: " + std::to_string(lr));
    }
    if (eps <= 0) {
        throw std::invalid_argument("Epsilon must be positive, got: " + std::to_string(eps));
    }
    
    m.resize(this->params.size(), 0.0);
    v.resize(this->params.size(), 0.0);
}

void AdamW::step() {
    t++;
    double bc1 = 1.0 - std::pow(b1, t);
    double bc2 = 1.0 - std::pow(b2, t);
    
    for (size_t i = 0; i < params.size(); ++i) {
        double g = params[i]->grad;
        
        // Skip parameter dengan gradien NaN, Inf, atau 0
        if (std::isnan(g) || std::isinf(g) || g == 0.0) {
            continue;
        }
        
        // Update moments
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        
        double m_hat = m[i] / bc1;
        double v_hat = v[i] / bc2;
        
        // Decoupled weight decay (implementasi paper AdamW)
        if (decoupled_wd) {
            // AdamW: weight decay dipisahkan dari gradient update
            params[i]->data -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * params[i]->data);
        } else {
            // Adam dengan L2 regularization (original)
            params[i]->data -= lr * (m_hat / (std::sqrt(v_hat) + eps));
            // L2 regularization ditambahkan ke gradient (sudah termasuk di g)
        }
    }
}

void AdamW::zero_grad() {
    for (auto& p : params) {
        p->grad = 0.0;
    }
}

double clip_grad_norm(std::vector<ValuePtr>& params, double max_norm) {
    if (params.empty() || max_norm <= 0.0) {
        return 0.0;
    }
    
    double total_sq = 0.0;
    int valid_params = 0;
    
    for (const auto& p : params) {
        double g = p->grad;
        // Skip NaN dan Inf
        if (!std::isnan(g) && !std::isinf(g)) {
            total_sq += g * g;
            valid_params++;
        }
    }
    
    if (valid_params == 0) {
        return 0.0;
    }
    
    double norm = std::sqrt(total_sq);
    
    if (norm > max_norm && norm > 0.0) {
        double scale = max_norm / norm;
        for (auto& p : params) {
            double g = p->grad;
            if (!std::isnan(g) && !std::isinf(g)) {
                p->grad *= scale;
            }
        }
    }
    
    return norm;
}

WarmupCosineScheduler::WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, 
                                             double base_lr, double min_lr)
    : opt(opt), warmup(warmup_steps), total(total_steps), 
      base_lr(base_lr), min_lr(min_lr), step_num(0) {
    
    if (!opt) {
        throw std::invalid_argument("Optimizer pointer cannot be null");
    }
    if (total_steps <= 0) {
        throw std::invalid_argument("Total steps must be positive, got: " + std::to_string(total_steps));
    }
    if (warmup_steps < 0) {
        throw std::invalid_argument("Warmup steps cannot be negative, got: " + std::to_string(warmup_steps));
    }
    if (base_lr <= 0) {
        throw std::invalid_argument("Base learning rate must be positive, got: " + std::to_string(base_lr));
    }
    if (min_lr < 0) {
        throw std::invalid_argument("Minimum learning rate cannot be negative, got: " + std::to_string(min_lr));
    }
    
    // Pastikan warmup tidak melebihi total steps
    if (warmup >= total) {
        warmup = total - 1;
        if (warmup < 0) warmup = 0;
    }
}

double WarmupCosineScheduler::step() {
    step_num++;
    double lr;
    
    // Handle edge cases
    if (total <= 0) {
        lr = base_lr;
    } else if (warmup <= 0 || step_num > warmup) {
        // No warmup phase atau sudah melewati warmup
        if (total <= warmup) {
            // Hanya warmup, tidak ada cosine decay
            lr = base_lr;
        } else {
            // Cosine decay phase
            double progress = static_cast<double>(step_num - std::max(0, warmup)) / 
                            static_cast<double>(total - std::max(0, warmup));
            progress = std::max(0.0, std::min(1.0, progress));
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + std::cos(M_PI * progress));
        }
    } else {
        // Warmup phase (warmup > 0)
        lr = base_lr * static_cast<double>(step_num) / static_cast<double>(warmup);
    }
    
    // Pastikan learning rate valid
    if (std::isnan(lr) || std::isinf(lr)) {
        lr = min_lr;
    }
    
    opt->lr = lr;
    return lr;
}