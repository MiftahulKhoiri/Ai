// optim.cpp
#include "optim.h"
#include "layers.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// ============================================================
// SOFTMAX
// ============================================================
std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};
    
    // Cari max untuk stabilitas numerik
    double max_val = x[0]->data;
    for (const auto& v : x) {
        if (v->data > max_val) max_val = v->data;
    }
    
    // Hitung exp dan sum
    std::vector<ValuePtr> exps;
    exps.reserve(x.size());
    double sum = 0.0;
    for (const auto& v : x) {
        double exp_val = std::exp(v->data - max_val);
        auto exp_v = Value::create(exp_val);
        exps.push_back(exp_v);
        sum += exp_val;
    }
    
    // Normalisasi
    std::vector<ValuePtr> result;
    result.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        auto prob = exps[i] / Value::create(sum);
        result.push_back(prob);
    }
    
    // Backward untuk softmax
    for (size_t i = 0; i < x.size(); ++i) {
        result[i]->_backward = [result, x, i]() {
            double grad = result[i]->grad;
            for (size_t j = 0; j < x.size(); ++j) {
                double delta = (i == j) ? 1.0 : 0.0;
                double p_i = result[i]->data;
                double p_j = result[j]->data;
                x[j]->grad += grad * p_i * (delta - p_j);
            }
        };
    }
    
    return result;
}

// ============================================================
// LOG SOFTMAX
// ============================================================
std::vector<ValuePtr> log_softmax(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};
    
    // Cari max untuk stabilitas numerik
    double max_val = x[0]->data;
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i]->data > max_val) max_val = x[i]->data;
    }
    
    // Hitung exp dan sum
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += std::exp(x[i]->data - max_val);
    }
    
    double log_sum = std::log(sum);
    std::vector<ValuePtr> result;
    result.reserve(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        double log_val = x[i]->data - max_val - log_sum;
        auto v = Value::create(log_val);
        
        // Backward untuk log_softmax
        v->_backward = [v, x, i, max_val, sum]() {
            double grad = v->grad;
            for (size_t j = 0; j < x.size(); ++j) {
                double delta = (i == j) ? 1.0 : 0.0;
                double softmax_j = std::exp(x[j]->data - max_val) / sum;
                x[j]->grad += grad * (delta - softmax_j);
            }
        };
        
        result.push_back(v);
    }
    
    return result;
}

// ============================================================
// CROSS ENTROPY LOSS
// ============================================================
ValuePtr cross_entropy_loss(const std::vector<std::vector<ValuePtr>>& logits_seq,
                            const std::vector<int>& target_ids,
                            const std::vector<int>& pad_mask) {
    
    if (logits_seq.empty() || target_ids.empty()) {
        return Value::create(0.0);
    }
    
    std::vector<ValuePtr> losses;
    size_t seq_len = std::min(logits_seq.size(), target_ids.size());
    
    for (size_t pos = 0; pos < seq_len; ++pos) {
        // Skip padding
        if (pos < pad_mask.size() && pad_mask[pos] == 1) continue;
        
        int target = target_ids[pos];
        if (target < 0 || target >= (int)logits_seq[pos].size()) continue;
        
        // Log-softmax untuk stabilitas
        auto log_probs = log_softmax(logits_seq[pos]);
        
        // Negative log likelihood: -log(p_target)
        if (target < (int)log_probs.size()) {
            auto neg_log = Value::create(-log_probs[target]->data);
            
            // Backward: d(-log(p_target))/d(logits)
            neg_log->_backward = [neg_log, log_probs, target]() {
                double grad = neg_log->grad;
                log_probs[target]->grad += -grad;
            };
            
            losses.push_back(neg_log);
        }
    }
    
    if (losses.empty()) {
        return Value::create(0.0);
    }
    
    // Average loss
    double sum = 0.0;
    for (const auto& loss : losses) {
        sum += loss->data;
    }
    double avg = sum / losses.size();
    
    auto result = Value::create(avg);
    result->_backward = [result, losses]() {
        double grad = result->grad / losses.size();
        for (auto& loss : losses) {
            loss->grad += grad;
        }
    };
    
    return result;
}

// ============================================================
// ADAMW OPTIMIZER
// ============================================================
AdamW::AdamW(std::vector<ValuePtr> params, double lr, double betas1, 
             double betas2, double eps, double weight_decay, bool decoupled_wd)
    : lr(lr),
      params(std::move(params)),
      b1(betas1),
      b2(betas2),
      eps(eps),
      wd(weight_decay),
      decoupled_wd(decoupled_wd),
      m(this->params.size(), 0.0),
      v(this->params.size(), 0.0),
      t(0) {}

void AdamW::zero_grad() {
    for (auto& p : params) {
        p->grad = 0.0;
    }
}

void AdamW::step() {
    t++;
    double bias_correction1 = 1.0 - std::pow(b1, t);
    double bias_correction2 = 1.0 - std::pow(b2, t);
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
        double grad = p->grad;
        
        // Update momentum
        m[i] = b1 * m[i] + (1.0 - b1) * grad;
        v[i] = b2 * v[i] + (1.0 - b2) * grad * grad;
        
        // Bias correction
        double m_hat = m[i] / bias_correction1;
        double v_hat = v[i] / bias_correction2;
        
        // Update parameter
        double update = lr * m_hat / (std::sqrt(v_hat) + eps);
        
        // Decoupled weight decay (AdamW)
        if (decoupled_wd) {
            p->data -= lr * wd * p->data;
        }
        
        p->data -= update;
    }
}

// ============================================================
// CLIP GRAD NORM
// ============================================================
double clip_grad_norm(std::vector<ValuePtr>& params, double max_norm) {
    double total_norm = 0.0;
    for (const auto& p : params) {
        total_norm += p->grad * p->grad;
    }
    total_norm = std::sqrt(total_norm);
    
    if (total_norm > max_norm && max_norm > 0.0) {
        double scale = max_norm / total_norm;
        for (auto& p : params) {
            p->grad *= scale;
        }
        return total_norm * scale;
    }
    return total_norm;
}

// ============================================================
// WARMUP COSINE SCHEDULER
// ============================================================
WarmupCosineScheduler::WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, 
                                             double base_lr, double min_lr)
    : opt(opt), warmup(warmup_steps), total(total_steps), 
      step_num(0), base_lr(base_lr), min_lr(min_lr) {}

double WarmupCosineScheduler::step() {
    step_num++;
    double lr;
    
    if (step_num < warmup) {
        // Linear warmup
        lr = base_lr * static_cast<double>(step_num) / warmup;
    } else {
        // Cosine decay
        double progress = static_cast<double>(step_num - warmup) / (total - warmup);
        progress = std::min(1.0, progress);
        double cosine = 0.5 * (1.0 + std::cos(PI * progress));
        lr = min_lr + (base_lr - min_lr) * cosine;
    }
    
    opt->lr = lr;
    return lr;
}