// optim.cpp
#include "optim.h"
#include "layers.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <iostream>

// ============================================================
// SOFTMAX
// ============================================================
std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};

    double max_val = x[0]->data;
    for (const auto& v : x) {
        if (v->data > max_val) max_val = v->data;
    }

    std::vector<ValuePtr> exps;
    exps.reserve(x.size());
    double sum = 0.0;
    for (const auto& v : x) {
        double exp_val = std::exp(v->data - max_val);
        auto exp_v = Value::create(exp_val);
        exps.push_back(exp_v);
        sum += exp_val;
    }

    std::vector<ValuePtr> result;
    result.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        auto prob = exps[i] / Value::create(sum);
        result.push_back(prob);
    }

    // FIX: dua masalah sekaligus di sini.
    // (1) O(n^2) memori: capture vector "x"/"result" BY VALUE di setiap
    //     closure (versi lama) membuat tiap closure punya salinan
    //     sendiri -> pakai shared_ptr supaya semua closure berbagi 1
    //     salinan.
    // (2) REFERENCE CYCLE: kalau result_shared (berisi shared_ptr ke
    //     result[i] itu sendiri) di-capture oleh closure milik
    //     result[i], itu cycle (result[i] -> _backward -> capture
    //     result_shared -> berisi result[i] -> cycle). Makanya di sini
    //     kita simpan RAW POINTER (Value*) untuk elemen result, bukan
    //     shared_ptr, supaya tidak ada cycle.
    auto result_raw = std::make_shared<std::vector<Value*>>();
    result_raw->reserve(result.size());
    for (auto& r : result) result_raw->push_back(r.get());

    auto x_shared = std::make_shared<std::vector<ValuePtr>>(x);

    for (size_t i = 0; i < result.size(); ++i) {
        Value* self_ptr = result[i].get();
        self_ptr->_backward = [result_raw, x_shared, i]() {
            double grad = (*result_raw)[i]->grad;
            for (size_t j = 0; j < x_shared->size(); ++j) {
                double delta = (i == j) ? 1.0 : 0.0;
                double p_i = (*result_raw)[i]->data;
                double p_j = (*result_raw)[j]->data;
                (*x_shared)[j]->grad += grad * p_i * (delta - p_j);
            }
        };
        result[i]->_prev = *x_shared;
    }

    return result;
}

// ============================================================
// LOG SOFTMAX
// ============================================================
std::vector<ValuePtr> log_softmax(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};

    double max_val = x[0]->data;
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i]->data > max_val) max_val = x[i]->data;
    }

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += std::exp(x[i]->data - max_val);
    }

    double log_sum = std::log(sum);
    std::vector<ValuePtr> result;
    result.reserve(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        double log_val = x[i]->data - max_val - log_sum;
        result.push_back(Value::create(log_val));
    }

    // FIX: x dibungkus shared_ptr (dibagi semua closure, O(n) bukan
    // O(n^2)), dan closure milik result[i] capture RAW POINTER ke
    // dirinya sendiri (v_raw), BUKAN shared_ptr result[i] -> mencegah
    // reference cycle yang bikin node ini bocor permanen.
    auto x_shared = std::make_shared<std::vector<ValuePtr>>(x);

    for (size_t i = 0; i < result.size(); ++i) {
        Value* v_raw = result[i].get();
        v_raw->_backward = [v_raw, x_shared, i, max_val, sum]() {
            double grad = v_raw->grad;
            for (size_t j = 0; j < x_shared->size(); ++j) {
                double delta = (i == j) ? 1.0 : 0.0;
                double softmax_j = std::exp((*x_shared)[j]->data - max_val) / sum;
                (*x_shared)[j]->grad += grad * (delta - softmax_j);
            }
        };
        result[i]->_prev = *x_shared;
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
        if (pos < pad_mask.size() && pad_mask[pos] == 1) continue;

        int target = target_ids[pos];
        if (target < 0 || target >= (int)logits_seq[pos].size()) continue;

        auto log_probs = log_softmax(logits_seq[pos]);

        if (target < (int)log_probs.size()) {
            auto neg_log = Value::create(-log_probs[target]->data);
            neg_log->_prev = {log_probs[target]};

            // FIX: (1) hanya capture elemen yang benar-benar dipakai
            // (log_probs[target], satu ValuePtr) bukan seluruh vector
            // log_probs sepanjang vocab_size -> hindari O(vocab_size)
            // per posisi yang tidak perlu.
            // (2) capture RAW POINTER ke neg_log sendiri (bukan
            // shared_ptr neg_log) -> hindari reference cycle.
            ValuePtr target_log_prob = log_probs[target];
            Value* neg_log_ptr = neg_log.get();
            neg_log_ptr->_backward = [neg_log_ptr, target_log_prob]() {
                double grad = neg_log_ptr->grad;
                target_log_prob->grad += -grad;
            };
            losses.push_back(neg_log);
        }
    }

    if (losses.empty()) {
        return Value::create(0.0);
    }

    double sum = 0.0;
    for (const auto& loss : losses) {
        sum += loss->data;
    }
    double avg = sum / losses.size();

    auto losses_shared = std::make_shared<std::vector<ValuePtr>>(losses);

    auto result = Value::create(avg);
    result->_prev = *losses_shared;

    // FIX: raw pointer ke result sendiri, bukan shared_ptr result ->
    // hindari reference cycle (result -> _backward -> capture result).
    Value* result_ptr = result.get();
    result_ptr->_backward = [result_ptr, losses_shared]() {
        double grad = result_ptr->grad / losses_shared->size();
        for (auto& loss : *losses_shared) {
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

        m[i] = b1 * m[i] + (1.0 - b1) * grad;
        v[i] = b2 * v[i] + (1.0 - b2) * grad * grad;

        double m_hat = m[i] / bias_correction1;
        double v_hat = v[i] / bias_correction2;

        double update = lr * m_hat / (std::sqrt(v_hat) + eps);

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
        lr = base_lr * static_cast<double>(step_num) / warmup;
    } else {
        double progress = static_cast<double>(step_num - warmup) / (total - warmup);
        progress = std::min(1.0, progress);
        double cosine = 0.5 * (1.0 + std::cos(PI * progress));
        lr = min_lr + (base_lr - min_lr) * cosine;
    }

    opt->lr = lr;
    return lr;
}