// ops.h
#pragma once
#include "tensor.h"
#include "matrix.h"
#include "simd.h"
#include <vector>
#include <stdexcept>
#include <numeric>
#include <cstring>

namespace tnsr {
namespace ops {

// ============ ELEMENT-WISE (header untuk inlining) ============
template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape)
        throw std::invalid_argument("add: shape mismatch");
    Tensor<T> result(a.shape);
    const T* pa = a.data.data();
    const T* pb = b.data.data();
    T* pr = result.data.data();
    size_t n = a.data.size();

    if constexpr (std::is_same_v<T, float>) {
        constexpr size_t W = SIMD_FLOAT_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::add(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] + pb[i];
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr size_t W = SIMD_DOUBLE_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::add(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] + pb[i];
    } else {
        for (size_t i = 0; i < n; ++i) pr[i] = pa[i] + pb[i];
    }
    return result;
}

template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) throw std::invalid_argument("sub: shape mismatch");
    Tensor<T> result(a.shape);
    const T* pa = a.data.data();
    const T* pb = b.data.data();
    T* pr = result.data.data();
    size_t n = a.data.size();
    if constexpr (std::is_same_v<T, float>) {
        constexpr size_t W = SIMD_FLOAT_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::sub(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] - pb[i];
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr size_t W = SIMD_DOUBLE_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::sub(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] - pb[i];
    } else {
        for (size_t i = 0; i < n; ++i) pr[i] = pa[i] - pb[i];
    }
    return result;
}

template<typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) throw std::invalid_argument("mul: shape mismatch");
    Tensor<T> result(a.shape);
    const T* pa = a.data.data();
    const T* pb = b.data.data();
    T* pr = result.data.data();
    size_t n = a.data.size();
    if constexpr (std::is_same_v<T, float>) {
        constexpr size_t W = SIMD_FLOAT_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::mul(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] * pb[i];
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr size_t W = SIMD_DOUBLE_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::mul(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] * pb[i];
    } else {
        for (size_t i = 0; i < n; ++i) pr[i] = pa[i] * pb[i];
    }
    return result;
}

template<typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) throw std::invalid_argument("div: shape mismatch");
    Tensor<T> result(a.shape);
    const T* pa = a.data.data();
    const T* pb = b.data.data();
    T* pr = result.data.data();
    size_t n = a.data.size();
    if constexpr (std::is_same_v<T, float>) {
        constexpr size_t W = SIMD_FLOAT_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::div(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] / pb[i];
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr size_t W = SIMD_DOUBLE_WIDTH;
        size_t i = 0;
        for (; i + W <= n; i += W) {
            auto va = simd::load(pa + i, simd::unaligned);
            auto vb = simd::load(pb + i, simd::unaligned);
            simd::store(pr + i, simd::div(va, vb), simd::unaligned);
        }
        for (; i < n; ++i) pr[i] = pa[i] / pb[i];
    } else {
        for (size_t i = 0; i < n; ++i) pr[i] = pa[i] / pb[i];
    }
    return result;
}

// ============ MATMUL (2D) ============
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

// ============ TRANSPOSE (2D) ============
template<typename T>
Tensor<T> transpose(const Tensor<T>& a, int dim0, int dim1);

// ============ RESHAPE / VIEW ============
template<typename T>
Tensor<T> reshape(const Tensor<T>& a, const std::vector<size_t>& new_shape) {
    size_t total = std::accumulate(new_shape.begin(), new_shape.end(), 1UL, std::multiplies<>());
    if (total != a.data.size())
        throw std::invalid_argument("reshape: total size mismatch");
    Tensor<T> result(new_shape);
    std::copy(a.data.begin(), a.data.end(), result.data.begin());
    return result;
}

template<typename T>
Tensor<T> view(const Tensor<T>& a, const std::vector<size_t>& new_shape) {
    return reshape(a, new_shape);   // di sini asumsi contiguous, kita copy data
}

// ============ CONCAT / SPLIT / STACK ============
template<typename T>
Tensor<T> concat(const std::vector<Tensor<T>>& tensors, int axis);

template<typename T>
std::vector<Tensor<T>> split(const Tensor<T>& a, size_t sections, int axis);

template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, int axis);

// ============ SLICE ============
template<typename T>
Tensor<T> slice(const Tensor<T>& a, const std::vector<std::pair<size_t,size_t>>& ranges);

// ============ GATHER / SCATTER ============
template<typename T>
Tensor<T> gather(const Tensor<T>& a, int axis, const Tensor<int>& indices);

template<typename T>
Tensor<T> scatter(const Tensor<T>& a, int axis, const Tensor<int>& indices, const Tensor<T>& updates);

// -------------------------------------------------------------------
// Explicit instantiation declarations (ops.cpp akan menyediakan definisi)
// Untuk menghindari duplikasi saat template digunakan di banyak TU.
// -------------------------------------------------------------------
#define DECLARE_EXTERN(T) \
    extern template Tensor<T> add<T>(const Tensor<T>&, const Tensor<T>&); \
    extern template Tensor<T> sub<T>(const Tensor<T>&, const Tensor<T>&); \
    extern template Tensor<T> mul<T>(const Tensor<T>&, const Tensor<T>&); \
    extern template Tensor<T> div<T>(const Tensor<T>&, const Tensor<T>&); \
    extern template Tensor<T> matmul<T>(const Tensor<T>&, const Tensor<T>&); \
    extern template Tensor<T> transpose<T>(const Tensor<T>&, int, int); \
    extern template Tensor<T> reshape<T>(const Tensor<T>&, const std::vector<size_t>&); \
    extern template Tensor<T> view<T>(const Tensor<T>&, const std::vector<size_t>&); \
    extern template Tensor<T> concat<T>(const std::vector<Tensor<T>>&, int); \
    extern template std::vector<Tensor<T>> split<T>(const Tensor<T>&, size_t, int); \
    extern template Tensor<T> stack<T>(const std::vector<Tensor<T>>&, int); \
    extern template Tensor<T> slice<T>(const Tensor<T>&, const std::vector<std::pair<size_t,size_t>>&); \
    extern template Tensor<T> gather<T>(const Tensor<T>&, int, const Tensor<int>&); \
    extern template Tensor<T> scatter<T>(const Tensor<T>&, int, const Tensor<int>&, const Tensor<T>&);

DECLARE_EXTERN(float)
DECLARE_EXTERN(double)

#undef DECLARE_EXTERN

} // namespace ops
} // namespace tnsr