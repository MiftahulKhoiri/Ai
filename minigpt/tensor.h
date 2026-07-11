// tensor.h
#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <functional>
#include <cstdint>   // TAMBAHKAN
#include "simd.h"

namespace tnsr {

template<typename T>
class Tensor {
public:
  std::vector<T> data;
  std::vector<size_t> shape;
  std::vector<size_t> strides;

  Tensor() = default;
  Tensor(const std::vector<size_t>& shp) : shape(shp) {
    size_t total = std::accumulate(shp.begin(), shp.end(), 1UL, std::multiplies<>());
    data.resize(total);
    compute_strides();
  }

  template<typename Iter>
  Tensor(const std::vector<size_t>& shp, Iter begin, Iter end) : shape(shp) {
    data.assign(begin, end);
    if (data.size() != total_size()) throw std::runtime_error("Size mismatch");
    compute_strides();
  }

  size_t total_size() const {
    return std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<>());
  }

  // ===== TAMBAHKAN: Operator untuk 2D (matrix) =====
  T& operator()(size_t i, size_t j) {
    if (shape.size() != 2) {
      throw std::runtime_error("Tensor is not 2D");
    }
    return data[i * strides[0] + j * strides[1]];
  }
  
  const T& operator()(size_t i, size_t j) const {
    if (shape.size() != 2) {
      throw std::runtime_error("Tensor is not 2D");
    }
    return data[i * strides[0] + j * strides[1]];
  }

  // ===== TAMBAHKAN: Operator untuk 1D (vector) =====
  T& operator()(size_t i) {
    if (shape.size() != 1) {
      throw std::runtime_error("Tensor is not 1D");
    }
    return data[i * strides[0]];
  }
  
  const T& operator()(size_t i) const {
    if (shape.size() != 1) {
      throw std::runtime_error("Tensor is not 1D");
    }
    return data[i * strides[0]];
  }

  // Indexing multi-dimensi (untuk dimensi > 2)
  T& operator()(const std::vector<size_t>& indices) {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides[i];
    }
    return data[offset];
  }
  
  const T& operator()(const std::vector<size_t>& indices) const {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides[i];
    }
    return data[offset];
  }

  // ===== TAMBAHKAN: Operator += untuk Tensor =====
  Tensor& operator+=(const Tensor& other) {
    assert(shape == other.shape);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] += other.data[i];
    }
    return *this;
  }

  // ===== TAMBAHKAN: Operator *= untuk scalar =====
  Tensor& operator*=(const T& scalar) {
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] *= scalar;
    }
    return *this;
  }

  // Operator + (sudah ada)
  Tensor operator+(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    const T* src1 = data.data();
    const T* src2 = other.data.data();
    T* dst = result.data.data();
    size_t N = data.size();
    
    if constexpr (std::is_same_v<T, float>) {
      constexpr size_t W = SIMD_FLOAT_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::add(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] + src2[i];
    } else if constexpr (std::is_same_v<T, double>) {
      constexpr size_t W = SIMD_DOUBLE_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::add(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] + src2[i];
    } else {
      for (size_t i = 0; i < N; ++i) dst[i] = src1[i] + src2[i];
    }
    return result;
  }

  // Operator - 
  Tensor operator-(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    const T* src1 = data.data();
    const T* src2 = other.data.data();
    T* dst = result.data.data();
    size_t N = data.size();
    
    if constexpr (std::is_same_v<T, float>) {
      constexpr size_t W = SIMD_FLOAT_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::sub(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] - src2[i];
    } else if constexpr (std::is_same_v<T, double>) {
      constexpr size_t W = SIMD_DOUBLE_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::sub(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] - src2[i];
    } else {
      for (size_t i = 0; i < N; ++i) dst[i] = src1[i] - src2[i];
    }
    return result;
  }

  // Operator * (element-wise)
  Tensor operator*(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    const T* src1 = data.data();
    const T* src2 = other.data.data();
    T* dst = result.data.data();
    size_t N = data.size();
    
    if constexpr (std::is_same_v<T, float>) {
      constexpr size_t W = SIMD_FLOAT_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::mul(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] * src2[i];
    } else if constexpr (std::is_same_v<T, double>) {
      constexpr size_t W = SIMD_DOUBLE_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::mul(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] * src2[i];
    } else {
      for (size_t i = 0; i < N; ++i) dst[i] = src1[i] * src2[i];
    }
    return result;
  }

  // Operator / (element-wise)
  Tensor operator/(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    const T* src1 = data.data();
    const T* src2 = other.data.data();
    T* dst = result.data.data();
    size_t N = data.size();
    
    if constexpr (std::is_same_v<T, float>) {
      constexpr size_t W = SIMD_FLOAT_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::div(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] / src2[i];
    } else if constexpr (std::is_same_v<T, double>) {
      constexpr size_t W = SIMD_DOUBLE_WIDTH;
      size_t i = 0;
      for (; i + W <= N; i += W) {
        auto a = simd::load(src1 + i, simd::unaligned);
        auto b = simd::load(src2 + i, simd::unaligned);
        simd::store(dst + i, simd::div(a, b), simd::unaligned);
      }
      for (; i < N; ++i) dst[i] = src1[i] / src2[i];
    } else {
      for (size_t i = 0; i < N; ++i) dst[i] = src1[i] / src2[i];
    }
    return result;
  }

private:
  void compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
};

// Contoh eksplisit instansiasi (deklarasi)
extern template class Tensor<float>;
extern template class Tensor<double>;

} // namespace tnsr