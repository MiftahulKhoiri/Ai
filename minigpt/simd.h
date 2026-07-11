// simd.h
#pragma once
#include <cstdint>   // TAMBAHKAN
#include <cstdint>
#include <cstring>
#include <type_traits>

// Deteksi arsitektur
#if defined(__AVX2__)
    #define USE_AVX2 1
    #include <immintrin.h>
    #define SIMD_FLOAT_WIDTH 8
    #define SIMD_DOUBLE_WIDTH 4
#elif defined(__SSE4_1__) || defined(__SSE3__)
    #define USE_SSE 1
    #include <smmintrin.h>
    #define SIMD_FLOAT_WIDTH 4
    #define SIMD_DOUBLE_WIDTH 2
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define USE_NEON 1
    #include <arm_neon.h>
    #define SIMD_FLOAT_WIDTH 4
    #define SIMD_DOUBLE_WIDTH 2
#else
    #define USE_SCALAR 1
    #define SIMD_FLOAT_WIDTH 4
    #define SIMD_DOUBLE_WIDTH 2
    #warning "No SIMD support detected, using scalar fallback"
#endif

namespace simd {

// Tag untuk alignment
struct aligned_t {} constexpr aligned{};
struct unaligned_t {} constexpr unaligned{};

// ============================
// FLOAT VECTOR WRAPPER
// ============================
struct float_vec {
    #if defined(USE_AVX2)
        __m256 v;
        float_vec() = default;
        float_vec(__m256 v_) : v(v_) {}
    #elif defined(USE_SSE)
        __m128 v;
        float_vec() = default;
        float_vec(__m128 v_) : v(v_) {}
    #elif defined(USE_NEON)
        float32x4_t v;
        float_vec() = default;
        float_vec(float32x4_t v_) : v(v_) {}
    #else
        float data[4];
        float_vec() = default;
    #endif
};

// ============================
// DOUBLE VECTOR WRAPPER
// ============================
struct double_vec {
    #if defined(USE_AVX2)
        __m256d v;
        double_vec() = default;
        double_vec(__m256d v_) : v(v_) {}
    #elif defined(USE_SSE)
        __m128d v;
        double_vec() = default;
        double_vec(__m128d v_) : v(v_) {}
    #elif defined(USE_NEON)
        float64x2_t v;
        double_vec() = default;
        double_vec(float64x2_t v_) : v(v_) {}
    #else
        double data[2];
        double_vec() = default;
    #endif
};

// ============================
// FLOAT LOAD/STORE
// ============================
#if defined(USE_AVX2)
    inline float_vec load(const float* ptr, aligned_t) { return _mm256_load_ps(ptr); }
    inline float_vec load(const float* ptr, unaligned_t) { return _mm256_loadu_ps(ptr); }
    inline void store(float* ptr, float_vec v, aligned_t) { _mm256_store_ps(ptr, v.v); }
    inline void store(float* ptr, float_vec v, unaligned_t) { _mm256_storeu_ps(ptr, v.v); }
    inline float_vec set1(float a) { return _mm256_set1_ps(a); }
#elif defined(USE_SSE)
    inline float_vec load(const float* ptr, aligned_t) { return _mm_load_ps(ptr); }
    inline float_vec load(const float* ptr, unaligned_t) { return _mm_loadu_ps(ptr); }
    inline void store(float* ptr, float_vec v, aligned_t) { _mm_store_ps(ptr, v.v); }
    inline void store(float* ptr, float_vec v, unaligned_t) { _mm_storeu_ps(ptr, v.v); }
    inline float_vec set1(float a) { return _mm_set1_ps(a); }
#elif defined(USE_NEON)
    inline float_vec load(const float* ptr, aligned_t) { return vld1q_f32(ptr); }
    inline float_vec load(const float* ptr, unaligned_t) { return vld1q_f32(ptr); }
    inline void store(float* ptr, float_vec v, aligned_t) { vst1q_f32(ptr, v.v); }
    inline void store(float* ptr, float_vec v, unaligned_t) { vst1q_f32(ptr, v.v); }
    inline float_vec set1(float a) { return vdupq_n_f32(a); }
#else
    // Scalar fallback
    inline float_vec load(const float* ptr, aligned_t) {
        float_vec v;
        v.data[0] = ptr[0]; v.data[1] = ptr[1];
        v.data[2] = ptr[2]; v.data[3] = ptr[3];
        return v;
    }
    inline float_vec load(const float* ptr, unaligned_t) { return load(ptr, aligned); }
    inline void store(float* ptr, float_vec v, aligned_t) {
        ptr[0] = v.data[0]; ptr[1] = v.data[1];
        ptr[2] = v.data[2]; ptr[3] = v.data[3];
    }
    inline void store(float* ptr, float_vec v, unaligned_t) { store(ptr, v, aligned); }
    inline float_vec set1(float a) {
        float_vec v;
        v.data[0] = a; v.data[1] = a; v.data[2] = a; v.data[3] = a;
        return v;
    }
#endif

// ============================
// DOUBLE LOAD/STORE
// ============================
#if defined(USE_AVX2)
    inline double_vec load(const double* ptr, aligned_t) { return _mm256_load_pd(ptr); }
    inline double_vec load(const double* ptr, unaligned_t) { return _mm256_loadu_pd(ptr); }
    inline void store(double* ptr, double_vec v, aligned_t) { _mm256_store_pd(ptr, v.v); }
    inline void store(double* ptr, double_vec v, unaligned_t) { _mm256_storeu_pd(ptr, v.v); }
    inline double_vec set1(double a) { return _mm256_set1_pd(a); }
#elif defined(USE_SSE)
    inline double_vec load(const double* ptr, aligned_t) { return _mm_load_pd(ptr); }
    inline double_vec load(const double* ptr, unaligned_t) { return _mm_loadu_pd(ptr); }
    inline void store(double* ptr, double_vec v, aligned_t) { _mm_store_pd(ptr, v.v); }
    inline void store(double* ptr, double_vec v, unaligned_t) { _mm_storeu_pd(ptr, v.v); }
    inline double_vec set1(double a) { return _mm_set1_pd(a); }
#elif defined(USE_NEON)
    inline double_vec load(const double* ptr, aligned_t) { return vld1q_f64(ptr); }
    inline double_vec load(const double* ptr, unaligned_t) { return vld1q_f64(ptr); }
    inline void store(double* ptr, double_vec v, aligned_t) { vst1q_f64(ptr, v.v); }
    inline void store(double* ptr, double_vec v, unaligned_t) { vst1q_f64(ptr, v.v); }
    inline double_vec set1(double a) { return vdupq_n_f64(a); }
#else
    // Scalar fallback
    inline double_vec load(const double* ptr, aligned_t) {
        double_vec v;
        v.data[0] = ptr[0]; v.data[1] = ptr[1];
        return v;
    }
    inline double_vec load(const double* ptr, unaligned_t) { return load(ptr, aligned); }
    inline void store(double* ptr, double_vec v, aligned_t) {
        ptr[0] = v.data[0]; ptr[1] = v.data[1];
    }
    inline void store(double* ptr, double_vec v, unaligned_t) { store(ptr, v, aligned); }
    inline double_vec set1(double a) {
        double_vec v;
        v.data[0] = a; v.data[1] = a;
        return v;
    }
#endif

// ============================
// FLOAT OPERATIONS
// ============================
#if defined(USE_AVX2)
    inline float_vec add(float_vec a, float_vec b) { return _mm256_add_ps(a.v, b.v); }
    inline float_vec sub(float_vec a, float_vec b) { return _mm256_sub_ps(a.v, b.v); }
    inline float_vec mul(float_vec a, float_vec b) { return _mm256_mul_ps(a.v, b.v); }
    inline float_vec div(float_vec a, float_vec b) { return _mm256_div_ps(a.v, b.v); }
    inline float_vec fmadd(float_vec a, float_vec b, float_vec c) { return _mm256_fmadd_ps(a.v, b.v, c.v); }
#elif defined(USE_SSE)
    inline float_vec add(float_vec a, float_vec b) { return _mm_add_ps(a.v, b.v); }
    inline float_vec sub(float_vec a, float_vec b) { return _mm_sub_ps(a.v, b.v); }
    inline float_vec mul(float_vec a, float_vec b) { return _mm_mul_ps(a.v, b.v); }
    inline float_vec div(float_vec a, float_vec b) { return _mm_div_ps(a.v, b.v); }
    inline float_vec fmadd(float_vec a, float_vec b, float_vec c) { return _mm_fmadd_ps(a.v, b.v, c.v); }
#elif defined(USE_NEON)
    inline float_vec add(float_vec a, float_vec b) { return vaddq_f32(a.v, b.v); }
    inline float_vec sub(float_vec a, float_vec b) { return vsubq_f32(a.v, b.v); }
    inline float_vec mul(float_vec a, float_vec b) { return vmulq_f32(a.v, b.v); }
    inline float_vec div(float_vec a, float_vec b) { return vdivq_f32(a.v, b.v); }
    inline float_vec fmadd(float_vec a, float_vec b, float_vec c) { return vfmaq_f32(c.v, a.v, b.v); }
#else
    // Scalar fallback
    inline float_vec add(float_vec a, float_vec b) {
        float_vec r;
        for (int i = 0; i < 4; ++i) r.data[i] = a.data[i] + b.data[i];
        return r;
    }
    inline float_vec sub(float_vec a, float_vec b) {
        float_vec r;
        for (int i = 0; i < 4; ++i) r.data[i] = a.data[i] - b.data[i];
        return r;
    }
    inline float_vec mul(float_vec a, float_vec b) {
        float_vec r;
        for (int i = 0; i < 4; ++i) r.data[i] = a.data[i] * b.data[i];
        return r;
    }
    inline float_vec div(float_vec a, float_vec b) {
        float_vec r;
        for (int i = 0; i < 4; ++i) r.data[i] = a.data[i] / b.data[i];
        return r;
    }
    inline float_vec fmadd(float_vec a, float_vec b, float_vec c) {
        float_vec r;
        for (int i = 0; i < 4; ++i) r.data[i] = a.data[i] * b.data[i] + c.data[i];
        return r;
    }
#endif

// ============================
// DOUBLE OPERATIONS
// ============================
#if defined(USE_AVX2)
    inline double_vec add(double_vec a, double_vec b) { return _mm256_add_pd(a.v, b.v); }
    inline double_vec sub(double_vec a, double_vec b) { return _mm256_sub_pd(a.v, b.v); }
    inline double_vec mul(double_vec a, double_vec b) { return _mm256_mul_pd(a.v, b.v); }
    inline double_vec div(double_vec a, double_vec b) { return _mm256_div_pd(a.v, b.v); }
    inline double_vec fmadd(double_vec a, double_vec b, double_vec c) { return _mm256_fmadd_pd(a.v, b.v, c.v); }
#elif defined(USE_SSE)
    inline double_vec add(double_vec a, double_vec b) { return _mm_add_pd(a.v, b.v); }
    inline double_vec sub(double_vec a, double_vec b) { return _mm_sub_pd(a.v, b.v); }
    inline double_vec mul(double_vec a, double_vec b) { return _mm_mul_pd(a.v, b.v); }
    inline double_vec div(double_vec a, double_vec b) { return _mm_div_pd(a.v, b.v); }
    inline double_vec fmadd(double_vec a, double_vec b, double_vec c) { return _mm_fmadd_pd(a.v, b.v, c.v); }
#elif defined(USE_NEON)
    inline double_vec add(double_vec a, double_vec b) { return vaddq_f64(a.v, b.v); }
    inline double_vec sub(double_vec a, double_vec b) { return vsubq_f64(a.v, b.v); }
    inline double_vec mul(double_vec a, double_vec b) { return vmulq_f64(a.v, b.v); }
    inline double_vec div(double_vec a, double_vec b) { return vdivq_f64(a.v, b.v); }
    inline double_vec fmadd(double_vec a, double_vec b, double_vec c) { return vfmaq_f64(c.v, a.v, b.v); }
#else
    // Scalar fallback
    inline double_vec add(double_vec a, double_vec b) {
        double_vec r;
        r.data[0] = a.data[0] + b.data[0];
        r.data[1] = a.data[1] + b.data[1];
        return r;
    }
    inline double_vec sub(double_vec a, double_vec b) {
        double_vec r;
        r.data[0] = a.data[0] - b.data[0];
        r.data[1] = a.data[1] - b.data[1];
        return r;
    }
    inline double_vec mul(double_vec a, double_vec b) {
        double_vec r;
        r.data[0] = a.data[0] * b.data[0];
        r.data[1] = a.data[1] * b.data[1];
        return r;
    }
    inline double_vec div(double_vec a, double_vec b) {
        double_vec r;
        r.data[0] = a.data[0] / b.data[0];
        r.data[1] = a.data[1] / b.data[1];
        return r;
    }
    inline double_vec fmadd(double_vec a, double_vec b, double_vec c) {
        double_vec r;
        r.data[0] = a.data[0] * b.data[0] + c.data[0];
        r.data[1] = a.data[1] * b.data[1] + c.data[1];
        return r;
    }
#endif

// ============================
// OPERATOR OVERLOADS (sekarang menggunakan struct)
// ============================
inline float_vec operator+(float_vec a, float_vec b) { return add(a, b); }
inline float_vec operator-(float_vec a, float_vec b) { return sub(a, b); }
inline float_vec operator*(float_vec a, float_vec b) { return mul(a, b); }
inline float_vec operator/(float_vec a, float_vec b) { return div(a, b); }

inline double_vec operator+(double_vec a, double_vec b) { return add(a, b); }
inline double_vec operator-(double_vec a, double_vec b) { return sub(a, b); }
inline double_vec operator*(double_vec a, double_vec b) { return mul(a, b); }
inline double_vec operator/(double_vec a, double_vec b) { return div(a, b); }

} // namespace simd