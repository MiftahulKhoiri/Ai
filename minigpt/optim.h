// optim.h
#ifndef OPTIM_H
#define OPTIM_H

#include "value.h"
#include <vector>
#include <memory>
#include <cmath>

// Forward declarations
ValuePtr cross_entropy_loss(const std::vector<std::vector<ValuePtr>>& logits_seq,
                            const std::vector<int>& target_ids,
                            const std::vector<int>& pad_mask);

// Log softmax untuk stabilitas numerik yang lebih baik
std::vector<ValuePtr> log_softmax(const std::vector<ValuePtr>& x);

// Softmax
std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& x);

class AdamW {
public:
    AdamW(std::vector<ValuePtr> params, double lr = 1e-3, double betas1 = 0.9, 
          double betas2 = 0.999, double eps = 1e-8, double weight_decay = 0.01, 
          bool decoupled_wd = true);

    void step();
    void zero_grad();

    double lr;  // Learning rate (dapat diubah oleh scheduler)

    // Getter (untuk akses baca dari Python/pybind11)
    const std::vector<ValuePtr>& get_params() const { return params; }
    const std::vector<double>& get_m() const { return m; }
    const std::vector<double>& get_v() const { return v; }
    int get_t() const { return t; }

    // Setter (untuk deserialisasi dari Python)
    void set_params(const std::vector<ValuePtr>& p) { params = p; }
    void set_m(const std::vector<double>& m_) { m = m_; }
    void set_v(const std::vector<double>& v_) { v = v_; }
    void set_t(int t_) { t = t_; }

private:
    std::vector<ValuePtr> params;
    double b1, b2, eps, wd;
    bool decoupled_wd;
    std::vector<double> m, v;
    int t;
};

double clip_grad_norm(std::vector<ValuePtr>& params, double max_norm);

class WarmupCosineScheduler {
public:
    WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, 
                          double base_lr = 1e-3, double min_lr = 1e-5);

    double step();  // Mengembalikan learning rate yang baru

    // Getter dan Setter
    int get_step_num() const { return step_num; }
    void set_step_num(int s) { step_num = s; }

private:
    AdamW* opt;
    int warmup, total, step_num;
    double base_lr, min_lr;

    // Konstanta PI (bukan M_PI untuk menghindari konflik)
    static constexpr double PI = 3.14159265358979323846;
};

#endif // OPTIM_H