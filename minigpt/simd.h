// simd.h
#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifdef __AVX2__
  #include <immintrin.h>
  #define SIMD_FLOAT_WIDTH 8
  #define SIMD_DOUBLE_WIDTH 4
#elif defined(__SSE4_1__) || defined(__SSE3__)
  #include <smmintrin.h>
  #define SIMD_FLOAT_WIDTH 4
  #define SIMD_DOUBLE_WIDTH 2
#else
  #error "At least SSE4.1 is required"
#endif

namespace simd {

// Tag untuk alignment
struct aligned_t {} constexpr aligned{};
struct unaligned_t {} constexpr unaligned{};

// ============ float ============
#if SIMD_FLOAT_WIDTH == 8
using float_vec = __m256;
inline float_vec load(const float* ptr, aligned_t) { return _mm256_load_ps(ptr); }
inline float_vec load(const float* ptr, unaligned_t) { return _mm256_loadu_ps(ptr); }
inline void store(float* ptr, float_vec v, aligned_t) { _mm256_store_ps(ptr, v); }
inline void store(float* ptr, float_vec v, unaligned_t) { _mm256_storeu_ps(ptr, v); }
inline float_vec set1(float a) { return _mm256_set1_ps(a); }
inline float_vec add(float_vec a, float_vec b) { return _mm256_add_ps(a, b); }
inline float_vec sub(float_vec a, float_vec b) { return _mm256_sub_ps(a, b); }
inline float_vec mul(float_vec a, float_vec b) { return _mm256_mul_ps(a, b); }
inline float_vec div(float_vec a, float_vec b) { return _mm256_div_ps(a, b); }
inline float_vec fmadd(float_vec a, float_vec b, float_vec c) { return _mm256_fmadd_ps(a, b, c); }
#elif SIMD_FLOAT_WIDTH == 4
using float_vec = __m128;
inline float_vec load(const float* ptr, aligned_t) { return _mm_load_ps(ptr); }
inline float_vec load(const float* ptr, unaligned_t) { return _mm_loadu_ps(ptr); }
inline void store(float* ptr, float_vec v, aligned_t) { _mm_store_ps(ptr, v); }
inline void store(float* ptr, float_vec v, unaligned_t) { _mm_storeu_ps(ptr, v); }
inline float_vec set1(float a) { return _mm_set1_ps(a); }
inline float_vec add(float_vec a, float_vec b) { return _mm_add_ps(a, b); }
inline float_vec sub(float_vec a, float_vec b) { return _mm_sub_ps(a, b); }
inline float_vec mul(float_vec a, float_vec b) { return _mm_mul_ps(a, b); }
inline float_vec div(float_vec a, float_vec b) { return _mm_div_ps(a, b); }
inline float_vec fmadd(float_vec a, float_vec b, float_vec c) { return _mm_fmadd_ps(a, b, c); }
#endif

// ============ double ============
#if SIMD_DOUBLE_WIDTH == 4
using double_vec = __m256d;
inline double_vec load(const double* ptr, aligned_t) { return _mm256_load_pd(ptr); }
inline double_vec load(const double* ptr, unaligned_t) { return _mm256_loadu_pd(ptr); }
inline void store(double* ptr, double_vec v, aligned_t) { _mm256_store_pd(ptr, v); }
inline void store(double* ptr, double_vec v, unaligned_t) { _mm256_storeu_pd(ptr, v); }
inline double_vec set1(double a) { return _mm256_set1_pd(a); }
inline double_vec add(double_vec a, double_vec b) { return _mm256_add_pd(a, b); }
inline double_vec sub(double_vec a, double_vec b) { return _mm256_sub_pd(a, b); }
inline double_vec mul(double_vec a, double_vec b) { return _mm256_mul_pd(a, b); }
inline double_vec div(double_vec a, double_vec b) { return _mm256_div_pd(a, b); }
inline double_vec fmadd(double_vec a, double_vec b, double_vec c) { return _mm256_fmadd_pd(a, b, c); }
#elif SIMD_DOUBLE_WIDTH == 2
using double_vec = __m128d;
inline double_vec load(const double* ptr, aligned_t) { return _mm_load_pd(ptr); }
inline double_vec load(const double* ptr, unaligned_t) { return _mm_loadu_pd(ptr); }
inline void store(double* ptr, double_vec v, aligned_t) { _mm_store_pd(ptr, v); }
inline void store(double* ptr, double_vec v, unaligned_t) { _mm_storeu_pd(ptr, v); }
inline double_vec set1(double a) { return _mm_set1_pd(a); }
inline double_vec add(double_vec a, double_vec b) { return _mm_add_pd(a, b); }
inline double_vec sub(double_vec a, double_vec b) { return _mm_sub_pd(a, b); }
inline double_vec mul(double_vec a, double_vec b) { return _mm_mul_pd(a, b); }
inline double_vec div(double_vec a, double_vec b) { return _mm_div_pd(a, b); }
inline double_vec fmadd(double_vec a, double_vec b, double_vec c) { return _mm_fmadd_pd(a, b, c); }
#endif

// Operator convenience
inline float_vec operator+(float_vec a, float_vec b) { return add(a,b); }
inline float_vec operator-(float_vec a, float_vec b) { return sub(a,b); }
inline float_vec operator*(float_vec a, float_vec b) { return mul(a,b); }
inline float_vec operator/(float_vec a, float_vec b) { return div(a,b); }
inline double_vec operator+(double_vec a, double_vec b) { return add(a,b); }
inline double_vec operator-(double_vec a, double_vec b) { return sub(a,b); }
inline double_vec operator*(double_vec a, double_vec b) { return mul(a,b); }
inline double_vec operator/(double_vec a, double_vec b) { return div(a,b); }

} // namespace simd