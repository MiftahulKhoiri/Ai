#include "optim.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

// ===== HAPUS BARIS 10-12: PI sudah didefinisikan di optim.h =====
// namespace {
//     constexpr double PI = 3.14159265358979323846;
// }

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

        // Validasi nullptr dan indeks target
        if (logits_seq[i].empty()) continue;

        // Periksa nullptr untuk setiap elemen
        bool has_nullptr = false;
        for (const auto& val : logits_seq[i]) {
            if (!val) {
                has_nullptr = true;
                break;
            }
        }
        if (has_nullptr) continue;

        if (target_ids[i] < 0 || target_ids[i] >= static_cast<int>(logits_seq[i].size())) {
            throw std::out_of_range(
                "Target ID " + std::to_string(target_ids[i]) +
                " out of range [0, " + std::to_string(logits_seq[i].size()) + ") at position " + std::to_string(i)
            );
        }

        // Gunakan log_softmax untuk stabilitas numerik yang lebih baik
        auto log_probs = log_softmax(logits_seq[i]);

        // Periksa nullptr pada hasil dan target
        if (target_ids[i] < static_cast<int>(log_probs.size()) && log_probs[target_ids[i]]) {
            losses.push_back(log_probs[target_ids[i]] * -1.0);
        }
    }

    if (losses.empty()) return Value::create(0.0);

    ValuePtr total = Value::create(0.0);
    for (const auto& l : losses) {
        if (l) {  // Periksa nullptr
            total = total + l;
        }
    }
    return total / static_cast<double>(losses.size());
}

// Implementasi log_softmax yang lebih stabil secara numerik dan efisien
std::vector<ValuePtr> log_softmax(const std::vector<ValuePtr>& x) {
    if (x.empty()) return {};

    // Periksa nullptr dan temukan nilai maksimum untuk stabilitas numerik
    double max_val = -std::numeric_limits<double>::infinity();
    bool has_valid = false;

    for (const auto& val : x) {
        if (val && !std::isnan(val->data) && !std::isinf(val->data)) {
            if (val->data > max_val) {
                max_val = val->data;
            }
            has_valid = true;
        }
    }

    if (!has_valid) {
        // Jika semua nilai invalid, kembalikan vector dengan nilai 0
        return std::vector<ValuePtr>(x.size(), Value::create(0.0));
    }

    // Hitung sum(exp(x - max)) tanpa menyimpan exp_vals
    ValuePtr sum_exp = Value::create(0.0);

    for (const auto& val : x) {
        if (val && !std::isnan(val->data) && !std::isinf(val->data)) {
            auto shifted = val - max_val;
            auto exp_val = exp(shifted);
            if (exp_val) {  // Periksa nullptr
                sum_exp = sum_exp + exp_val;
            }
        }
    }

    // Hitung log(sum_exp)
    auto log_sum = log(sum_exp);
    if (!log_sum) {
        // Fallback jika log gagal
        return std::vector<ValuePtr>(x.size(), Value::create(0.0));
    }

    // Hitung log_softmax: x - max - log(sum_exp)
    std::vector<ValuePtr> result;
    result.reserve(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] && !std::isnan(x[i]->data) && !std::isinf(x[i]->data)) {
            result.push_back(x[i] - max_val - log_sum);
        } else {
            result.push_back(Value::create(-std::numeric_limits<double>::infinity()));
        }
    }

    return result;
}

// ===== PERBAIKAN 1: Urutkan init list sesuai deklarasi di header =====
// Urutan di header: params, b1, b2, eps, wd, decoupled_wd, m, v, t, lr
AdamW::AdamW(std::vector<ValuePtr> params, double lr, double betas1, double betas2, 
             double eps, double weight_decay, bool decoupled_wd)
    : params(std::move(params)),        // 1. params (vector<ValuePtr>)
      b1(betas1),                        // 2. b1 (double)
      b2(betas2),                        // 3. b2 (double)
      eps(eps),                          // 4. eps (double)
      wd(weight_decay),                  // 5. wd (double)
      decoupled_wd(decoupled_wd),        // 6. decoupled_wd (bool)
      m(this->params.size(), 0.0),       // 7. m (vector<double>)
      v(this->params.size(), 0.0),       // 8. v (vector<double>)
      t(0),                              // 9. t (int)
      lr(lr)                             // 10. lr (double) ← pindah ke sini
{
    // Validasi parameter
    if (this->params.empty()) {
        throw std::invalid_argument("Parameter list cannot be empty");
    }
    if (lr <= 0) {
        throw std::invalid_argument("Learning rate must be positive, got: " + std::to_string(lr));
    }
    if (eps <= 0) {
        throw std::invalid_argument("Epsilon must be positive, got: " + std::to_string(eps));
    }

    // Periksa nullptr pada parameter
    for (const auto& p : this->params) {
        if (!p) {
            throw std::invalid_argument("Parameter list contains null pointer");
        }
    }

    // m dan v sudah diinisialisasi di init list, tidak perlu resize lagi
    // m.resize(this->params.size(), 0.0);  ← HAPUS, sudah di init list
    // v.resize(this->params.size(), 0.0);  ← HAPUS, sudah di init list
}

void AdamW::step() {
    if (params.empty()) return;

    t++;
    double bc1 = 1.0 - std::pow(b1, t);
    double bc2 = 1.0 - std::pow(b2, t);

    // Hindari division by zero
    if (bc1 < 1e-10) bc1 = 1e-10;
    if (bc2 < 1e-10) bc2 = 1e-10;

    for (size_t i = 0; i < params.size(); ++i) {
        // Periksa nullptr
        if (!params[i]) continue;

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

        // Decoupled weight decay sesuai paper AdamW
        if (decoupled_wd && wd > 0.0) {
            // Langkah 1: Apply weight decay langsung ke parameter
            params[i]->data -= lr * wd * params[i]->data;
        }

        // Langkah 2: Apply Adam update
        double update = lr * m_hat / (std::sqrt(v_hat) + eps);
        params[i]->data -= update;

        // Jika bukan decoupled, weight decay sudah termasuk dalam gradien
        // (diharapkan sudah ditambahkan di luar optimizer)
    }
}

void AdamW::zero_grad() {
    for (auto& p : params) {
        if (p) {  // Periksa nullptr
            p->grad = 0.0;
        }
    }
}

double clip_grad_norm(std::vector<ValuePtr>& params, double max_norm) {
    if (params.empty() || max_norm <= 0.0) {
        return 0.0;
    }

    double total_sq = 0.0;
    int valid_params = 0;

    for (const auto& p : params) {
        if (!p) continue;  // Periksa nullptr

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
            if (!p) continue;  // Periksa nullptr

            double g = p->grad;
            if (!std::isnan(g) && !std::isinf(g)) {
                p->grad *= scale;
            }
        }
    }

    return norm;
}

// ===== PERBAIKAN 2: Urutkan init list sesuai deklarasi di header =====
// Urutan di header: opt, warmup, total, step_num, base_lr, min_lr
WarmupCosineScheduler::WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, 
                                             double base_lr, double min_lr)
    : opt(opt),              // 1. opt (AdamW*)
      warmup(warmup_steps),   // 2. warmup (int)
      total(total_steps),     // 3. total (int)
      step_num(0),            // 4. step_num (int) ← pindah ke sini
      base_lr(base_lr),       // 5. base_lr (double)
      min_lr(min_lr)          // 6. min_lr (double) ← pindah ke sini
{
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
    if (!opt) return 0.0;  // Periksa nullptr

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
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + std::cos(WarmupCosineScheduler::PI * progress));
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