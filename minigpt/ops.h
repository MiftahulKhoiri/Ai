// ops.h
#pragma once
#include "tensor.h"
#include <vector>

namespace tnsr {
namespace ops {

// ============================================================
// BASIC OPERATIONS
// ============================================================
template<typename T>
Tensor<T> transpose(const Tensor<T>& a, int dim0 = 0, int dim1 = 1);

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b);

// ============================================================
// SCALAR OPERATIONS
// ============================================================
template<typename T>
Tensor<T> scalar_add(const Tensor<T>& a, T scalar);

template<typename T>
Tensor<T> scalar_mul(const Tensor<T>& a, T scalar);

// ============================================================
// REDUCTIONS
// ============================================================
template<typename T>
T sum(const Tensor<T>& a);

template<typename T>
T mean(const Tensor<T>& a);

template<typename T>
T max(const Tensor<T>& a);

template<typename T>
T min(const Tensor<T>& a);

// ============================================================
// ACTIVATION FUNCTIONS
// ============================================================
template<typename T>
Tensor<T> relu(const Tensor<T>& a);

template<typename T>
Tensor<T> sigmoid(const Tensor<T>& a);

template<typename T>
Tensor<T> tanh_activation(const Tensor<T>& a);

template<typename T>
Tensor<T> softmax(const Tensor<T>& a);

} // namespace ops
} // namespace tnsr