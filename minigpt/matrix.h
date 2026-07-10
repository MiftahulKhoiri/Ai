// matrix.h
#pragma once
#include "tensor.h"
#include <cmath>
#include <stdexcept>

namespace linalg {

template<typename T>
class Matrix {
  tnsr::Tensor<T> tensor; // 2D tensor internal
public:
  size_t rows() const { return tensor.shape[0]; }
  size_t cols() const { return tensor.shape[1]; }

  Matrix(size_t r, size_t c) : tensor({r, c}) {}
  Matrix(size_t r, size_t c, const T& val) : tensor({r, c}) {
    std::fill(tensor.data.begin(), tensor.data.end(), val);
  }
  // Akses
  T& operator()(size_t i, size_t j) { return tensor.data[i * cols() + j]; }
  const T& operator()(size_t i, size_t j) const { return tensor.data[i * cols() + j]; }

  // Perkalian matriks dengan tiling & SIMD
  Matrix matmul(const Matrix& B) const {
    if (cols() != B.rows()) throw std::invalid_argument("Dimension mismatch");
    size_t M = rows(), N = B.cols(), K = cols();
    Matrix C(M, N, T(0));

    // Blocking factors
    constexpr size_t BM = 64, BN = 64, BK = 256;
    for (size_t i = 0; i < M; i += BM) {
      for (size_t j = 0; j < N; j += BN) {
        for (size_t k = 0; k < K; k += BK) {
          // micro-kernel
          size_t i_end = std::min(i + BM, M);
          size_t j_end = std::min(j + BN, N);
          size_t k_end = std::min(k + BK, K);
          for (size_t ii = i; ii < i_end; ++ii) {
            for (size_t kk = k; kk < k_end; ++kk) {
              T aik = (*this)(ii, kk);
              // SIMD pada j
              size_t jj = j;
              if constexpr (std::is_same_v<T, float>) {
                auto vaik = simd::set1(aik);
                for (; jj + SIMD_FLOAT_WIDTH <= j_end; jj += SIMD_FLOAT_WIDTH) {
                  auto vb = simd::load(&B(kk, jj), simd::unaligned);
                  auto vc = simd::load(&C(ii, jj), simd::unaligned);
                  vc = simd::fmadd(vaik, vb, vc);
                  simd::store(&C(ii, jj), vc, simd::unaligned);
                }
              } else if constexpr (std::is_same_v<T, double>) {
                auto vaik = simd::set1(aik);
                for (; jj + SIMD_DOUBLE_WIDTH <= j_end; jj += SIMD_DOUBLE_WIDTH) {
                  auto vb = simd::load(&B(kk, jj), simd::unaligned);
                  auto vc = simd::load(&C(ii, jj), simd::unaligned);
                  vc = simd::fmadd(vaik, vb, vc);
                  simd::store(&C(ii, jj), vc, simd::unaligned);
                }
              }
              // sisa elemen
              for (; jj < j_end; ++jj) {
                C(ii, jj) += aik * B(kk, jj);
              }
            }
          }
        }
      }
    }
    return C;
  }

  // Transpose
  Matrix transpose() const {
    Matrix res(cols(), rows());
    for (size_t i = 0; i < rows(); ++i)
      for (size_t j = 0; j < cols(); ++j)
        res(j, i) = (*this)(i, j);
    return res;
  }

  // Inverse via Gauss-Jordan (in-place)
  Matrix inverse() const {
    if (rows() != cols()) throw std::invalid_argument("Must be square");
    size_t n = rows();
    Matrix aug(n, 2*n, T(0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) aug(i, j) = (*this)(i, j);
      aug(i, n + i) = T(1);
    }
    for (size_t i = 0; i < n; ++i) {
      T pivot = aug(i, i);
      if (std::abs(pivot) < 1e-12) throw std::runtime_error("Singular matrix");
      for (size_t j = 0; j < 2*n; ++j) aug(i, j) /= pivot;
      for (size_t k = 0; k < n; ++k) {
        if (k == i) continue;
        T factor = aug(k, i);
        for (size_t j = 0; j < 2*n; ++j) aug(k, j) -= factor * aug(i, j);
      }
    }
    Matrix inv(n, n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        inv(i, j) = aug(i, n + j);
    return inv;
  }
};

extern template class Matrix<float>;
extern template class Matrix<double>;

} // namespace linalg