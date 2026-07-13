// linear.h
#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

#include "tensor.h"
#include "matrix.h"

namespace nn {

template<typename T>
class Linear {
public:

    Linear(
        size_t in_features,
        size_t out_features,
        bool use_bias = true
    );

    // Forward
    tnsr::Tensor<T> forward(const tnsr::Tensor<T>& input);

    // Backward
    tnsr::Tensor<T> backward(const tnsr::Tensor<T>& grad_output);

    // Optimizer step
    void step(T learning_rate);

    // Reset gradient
    void zero_grad();

    // Xavier initialization
    void reset_parameters();

    // Getter
    size_t input_size() const {
        return in_features_;
    }

    size_t output_size() const {
        return out_features_;
    }

    const linalg::Matrix<T>& weight() const {
        return weight_;
    }

    const std::vector<T>& bias() const {
        return bias_;
    }

private:

    size_t in_features_;
    size_t out_features_;
    bool use_bias_;

    // Weight matrix
    // Shape:
    // out_features x in_features
    linalg::Matrix<T> weight_;

    // Bias vector
    std::vector<T> bias_;

    // Gradient
    linalg::Matrix<T> grad_weight_;
    std::vector<T> grad_bias_;

    // Cache input
    tnsr::Tensor<T> input_cache_;

    std::mt19937 rng_;

private:

    T random_uniform(T min, T max);

};

// Explicit template instantiation
extern template class Linear<float>;
extern template class Linear<double>;

} // namespace nn