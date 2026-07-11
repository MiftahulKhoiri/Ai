// ops.cpp
#include "ops.h"
#include <cmath>
#include <algorithm>

namespace tnsr {
namespace ops {

// ============================================================
// TRANSPOSE
// ============================================================
template<typename T>
Tensor<T> transpose(const Tensor<T>& a, int dim0, int dim1) {
    // Jika tensor 2D, transpose standar
    if (a.shape.size() == 2) {
        size_t rows = a.shape[0];
        size_t cols = a.shape[1];
        Tensor<T> result({cols, rows});
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = a(i, j);
            }
        }
        return result;
    }
    
    // Untuk tensor multi-dimensi dengan dim tertentu
    // (dim0 dan dim1 digunakan untuk transpose dimensi tertentu)
    if (a.shape.size() > 2) {
        // Buat shape baru dengan menukar dim0 dan dim1
        std::vector<size_t> new_shape = a.shape;
        std::swap(new_shape[dim0], new_shape[dim1]);
        
        Tensor<T> result(new_shape);
        
        // Implementasi transpose untuk multi-dimensi
        // Ini adalah implementasi sederhana - untuk kasus umum
        std::vector<size_t> indices(a.shape.size(), 0);
        std::vector<size_t> new_indices = indices;
        
        size_t total = a.total_size();
        for (size_t idx = 0; idx < total; ++idx) {
            // Hitung indices dari linear index
            size_t temp = idx;
            for (int i = (int)indices.size() - 1; i >= 0; --i) {
                indices[i] = temp % a.shape[i];
                temp /= a.shape[i];
            }
            
            // Swap dim0 dan dim1 untuk new_indices
            new_indices = indices;
            std::swap(new_indices[dim0], new_indices[dim1]);
            
            // Copy data
            result(new_indices) = a(indices);
        }
        return result;
    }
    
    // Fallback: return copy
    return a;
}

// ============================================================
// MATMUL
// ============================================================
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::runtime_error("matmul only supports 2D tensors");
    }
    
    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];
    
    if (K != b.shape[0]) {
        throw std::runtime_error("matmul dimension mismatch");
    }
    
    Tensor<T> result({M, N});
    
    // Simple matrix multiplication (optimized with blocking)
    constexpr size_t BLOCK = 64;
    
    for (size_t i = 0; i < M; i += BLOCK) {
        for (size_t j = 0; j < N; j += BLOCK) {
            for (size_t k = 0; k < K; k += BLOCK) {
                size_t i_end = std::min(i + BLOCK, M);
                size_t j_end = std::min(j + BLOCK, N);
                size_t k_end = std::min(k + BLOCK, K);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        T aik = a(ii, kk);
                        for (size_t jj = j; jj < j_end; ++jj) {
                            result(ii, jj) += aik * b(kk, jj);
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

// ============================================================
// ADD
// ============================================================
template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("add: shape mismatch");
    }
    return a + b;
}

// ============================================================
// SUB
// ============================================================
template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("sub: shape mismatch");
    }
    return a - b;
}

// ============================================================
// MUL (element-wise)
// ============================================================
template<typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("mul: shape mismatch");
    }
    return a * b;
}

// ============================================================
// DIV (element-wise)
// ============================================================
template<typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("div: shape mismatch");
    }
    return a / b;
}

// ============================================================
// SCALAR OPERATIONS
// ============================================================
template<typename T>
Tensor<T> scalar_add(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) {
        result.data[i] = a.data[i] + scalar;
    }
    return result;
}

template<typename T>
Tensor<T> scalar_mul(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) {
        result.data[i] = a.data[i] * scalar;
    }
    return result;
}

// ============================================================
// REDUCTIONS
// ============================================================
template<typename T>
T sum(const Tensor<T>& a) {
    T total = 0;
    for (const auto& val : a.data) {
        total += val;
    }
    return total;
}

template<typename T>
T mean(const Tensor<T>& a) {
    if (a.data.empty()) return 0;
    return sum(a) / static_cast<T>(a.data.size());
}

template<typename T>
T max(const Tensor<T>& a) {
    if (a.data.empty()) return 0;
    return *std::max_element(a.data.begin(), a.data.end());
}

template<typename T>
T min(const Tensor<T>& a) {
    if (a.data.empty()) return 0;
    return *std::min_element(a.data.begin(), a.data.end());
}

// ============================================================
// ACTIVATION FUNCTIONS
// ============================================================
template<typename T>
Tensor<T> relu(const Tensor<T>& a) {
    Tensor<T> result(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) {
        result.data[i] = std::max(static_cast<T>(0), a.data[i]);
    }
    return result;
}

template<typename T>
Tensor<T> sigmoid(const Tensor<T>& a) {
    Tensor<T> result(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) {
        result.data[i] = 1.0 / (1.0 + std::exp(-a.data[i]));
    }
    return result;
}

template<typename T>
Tensor<T> tanh_activation(const Tensor<T>& a) {
    Tensor<T> result(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) {
        result.data[i] = std::tanh(a.data[i]);
    }
    return result;
}

template<typename T>
Tensor<T> softmax(const Tensor<T>& a) {
    if (a.shape.size() != 2) {
        throw std::runtime_error("softmax only supports 2D tensors");
    }
    
    Tensor<T> result(a.shape);
    size_t rows = a.shape[0];
    size_t cols = a.shape[1];
    
    for (size_t i = 0; i < rows; ++i) {
        // Find max for numerical stability
        T max_val = a(i, 0);
        for (size_t j = 1; j < cols; ++j) {
            if (a(i, j) > max_val) max_val = a(i, j);
        }
        
        // Compute exp and sum
        T sum_exp = 0;
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = std::exp(a(i, j) - max_val);
            sum_exp += result(i, j);
        }
        
        // Normalize
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) /= sum_exp;
        }
    }
    
    return result;
}

// ============================================================
// EXPLICIT INSTANTIATIONS
// ============================================================
#define INSTANTIATE(T) \
    template Tensor<T> transpose<T>(const Tensor<T>&, int, int); \
    template Tensor<T> matmul<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> add<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> sub<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> mul<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> div<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> scalar_add<T>(const Tensor<T>&, T); \
    template Tensor<T> scalar_mul<T>(const Tensor<T>&, T); \
    template T sum<T>(const Tensor<T>&); \
    template T mean<T>(const Tensor<T>&); \
    template T max<T>(const Tensor<T>&); \
    template T min<T>(const Tensor<T>&); \
    template Tensor<T> relu<T>(const Tensor<T>&); \
    template Tensor<T> sigmoid<T>(const Tensor<T>&); \
    template Tensor<T> tanh_activation<T>(const Tensor<T>&); \
    template Tensor<T> softmax<T>(const Tensor<T>&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace ops
} // namespace tnsr