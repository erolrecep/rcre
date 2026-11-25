/**
 * @file hal_internal.h
 * @brief Internal declarations for RCRE Hardware Abstraction Layer
 */

#ifndef RCRE_HAL_INTERNAL_H
#define RCRE_HAL_INTERNAL_H

#include "rcre.h"

/* Scalar Reference Micro-Kernels */
void rcre_scalar_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                          const float* A, size_t lda, 
                          const float* B, size_t ldb, 
                          float beta, float* C, size_t ldc);

void rcre_scalar_gemv_f32(size_t M, size_t K, float alpha,
                          const float* A, size_t lda,
                          const float* x, float beta, float* y);

float rcre_scalar_dot_f32(const float* x, const float* y, size_t n);

int32_t rcre_scalar_dot_i8(const int8_t* x, const int8_t* y, size_t n);

void rcre_scalar_gemm_i8(size_t M, size_t N, size_t K,
                         const int8_t* A, size_t lda,
                         const int8_t* B, size_t ldb,
                         int32_t* C, size_t ldc);

void rcre_scalar_gemm_bf16(size_t M, size_t N, size_t K, float alpha,
                           const uint16_t* A, size_t lda,
                           const uint16_t* B, size_t ldb,
                           float beta, float* C, size_t ldc);

void rcre_scalar_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_scalar_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_scalar_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_scalar_count_byte(const uint8_t* hay, size_t len, uint8_t needle);

bool rcre_scalar_utf8_validate(const uint8_t* data, size_t len);
size_t rcre_scalar_find_substr(const uint8_t* hay, size_t hay_len,
                               const uint8_t* needle, size_t needle_len);
void rcre_scalar_ascii_to_lower(const uint8_t* src, uint8_t* dst, size_t len);
void rcre_scalar_ascii_to_upper(const uint8_t* src, uint8_t* dst, size_t len);

/* H0 (2.1.0): F64 substrate + F32 completeness (scalar reference; every
 * ISA backend below provides matching cells, so no algorithm ever has to
 * branch on ISA — a missing cell is a build error, not a runtime check). */
void rcre_scalar_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                          const double* A, size_t lda, const double* B, size_t ldb,
                          double beta, double* C, size_t ldc);
double rcre_scalar_dot_f64(const double* x, const double* y, size_t n);
void rcre_scalar_axpy_f64(size_t n, double alpha, const double* x, double* y);
void rcre_scalar_gemv_f32(size_t M, size_t N, float alpha, const float* A, size_t lda,
                          const float* x, float beta, float* y);
void rcre_scalar_gemv_f64(size_t M, size_t N, double alpha, const double* A, size_t lda,
                          const double* x, double beta, double* y);
void rcre_scalar_scale_f32(size_t n, float alpha, float* x);
void rcre_scalar_scale_f64(size_t n, double alpha, double* x);
float  rcre_scalar_sum_f32(const float* x, size_t n);
double rcre_scalar_sum_f64(const double* x, size_t n);

/* ARM NEON 128-bit Micro-Kernels */
#if defined(__ARM_NEON) || defined(__aarch64__)
void rcre_neon_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

void rcre_neon_gemv_f32(size_t M, size_t K, float alpha,
                        const float* A, size_t lda,
                        const float* x, float beta, float* y);

float rcre_neon_dot_f32(const float* x, const float* y, size_t n);

int32_t rcre_neon_dot_i8(const int8_t* x, const int8_t* y, size_t n);

void rcre_neon_gemm_i8(size_t M, size_t N, size_t K,
                       const int8_t* A, size_t lda,
                       const int8_t* B, size_t ldb,
                       int32_t* C, size_t ldc);

void rcre_neon_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_neon_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_neon_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_neon_count_byte(const uint8_t* hay, size_t len, uint8_t needle);

bool rcre_neon_utf8_validate(const uint8_t* data, size_t len);
size_t rcre_neon_find_substr(const uint8_t* hay, size_t hay_len,
                             const uint8_t* needle, size_t needle_len);
void rcre_neon_ascii_to_lower(const uint8_t* src, uint8_t* dst, size_t len);
void rcre_neon_ascii_to_upper(const uint8_t* src, uint8_t* dst, size_t len);

/* H0 (2.1.0): F64 cells — NEON doubles are 128-bit / 4-wide */
void rcre_neon_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                        const double* A, size_t lda, const double* B, size_t ldb,
                        double beta, double* C, size_t ldc);
double rcre_neon_dot_f64(const double* x, const double* y, size_t n);
void rcre_neon_axpy_f64(size_t n, double alpha, const double* x, double* y);
void rcre_neon_gemv_f64(size_t M, size_t N, double alpha, const double* A, size_t lda,
                        const double* x, double beta, double* y);
void rcre_neon_scale_f64(size_t n, double alpha, double* x);
double rcre_neon_sum_f64(const double* x, size_t n);
#endif

/* x86 AVX2 256-bit Micro-Kernels (runtime-dispatched on Clang/GCC even without global -mavx2) */
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__clang__) || defined(__GNUC__))

/* H0 (2.1.0): F64 cells — AVX2 256-bit / 4-wide doubles */
void rcre_avx2_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                        const double* A, size_t lda, const double* B, size_t ldb,
                        double beta, double* C, size_t ldc);
double rcre_avx2_dot_f64(const double* x, const double* y, size_t n);
void rcre_avx2_axpy_f64(size_t n, double alpha, const double* x, double* y);
void rcre_avx2_gemv_f64(size_t M, size_t N, double alpha, const double* A, size_t lda,
                        const double* x, double beta, double* y);
void rcre_avx2_scale_f64(size_t n, double alpha, double* x);
double rcre_avx2_sum_f64(const double* x, size_t n);

/* H0 (2.1.0): F64 cells — AVX-512F 512-bit / 8-wide doubles (kernels_avx512.c;
 * per-function target attributes, called only when the probe says AVX512F) */
void rcre_avx512_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                          const double* A, size_t lda, const double* B, size_t ldb,
                          double beta, double* C, size_t ldc);
double rcre_avx512_dot_f64(const double* x, const double* y, size_t n);
void rcre_avx512_axpy_f64(size_t n, double alpha, const double* x, double* y);
void rcre_avx512_gemv_f64(size_t M, size_t N, double alpha, const double* A, size_t lda,
                          const double* x, double beta, double* y);
void rcre_avx512_scale_f64(size_t n, double alpha, double* x);
double rcre_avx512_sum_f64(const double* x, size_t n);

void rcre_avx2_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

void rcre_avx2_gemv_f32(size_t M, size_t K, float alpha,
                        const float* A, size_t lda,
                        const float* x, float beta, float* y);

float rcre_avx2_dot_f32(const float* x, const float* y, size_t n);

int32_t rcre_avx2_dot_i8(const int8_t* x, const int8_t* y, size_t n);

void rcre_avx2_gemm_i8(size_t M, size_t N, size_t K,
                       const int8_t* A, size_t lda,
                       const int8_t* B, size_t ldb,
                       int32_t* C, size_t ldc);

void rcre_avx2_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_avx2_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_avx2_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_avx2_count_byte(const uint8_t* hay, size_t len, uint8_t needle);

bool rcre_avx2_utf8_validate(const uint8_t* data, size_t len);
size_t rcre_avx2_find_substr(const uint8_t* hay, size_t hay_len,
                             const uint8_t* needle, size_t needle_len);
void rcre_avx2_ascii_to_lower(const uint8_t* src, uint8_t* dst, size_t len);
void rcre_avx2_ascii_to_upper(const uint8_t* src, uint8_t* dst, size_t len);

/* H0 (2.1.0): F64 cells — AVX2 doubles are 256-bit / 4-wide */
void rcre_avx2_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                        const double* A, size_t lda, const double* B, size_t ldb,
                        double beta, double* C, size_t ldc);
double rcre_avx2_dot_f64(const double* x, const double* y, size_t n);
void rcre_avx2_axpy_f64(size_t n, double alpha, const double* x, double* y);
void rcre_avx2_gemv_f64(size_t M, size_t N, double alpha, const double* A, size_t lda,
                        const double* x, double beta, double* y);
void rcre_avx2_scale_f64(size_t n, double alpha, double* x);
double rcre_avx2_sum_f64(const double* x, size_t n);
#endif

/* x86 AVX-512 512-bit Micro-Kernels */
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX512F__) || defined(__clang__) || defined(__GNUC__))
void rcre_avx512_gemm_f32(size_t M, size_t N, size_t K, float alpha,
                          const float* A, size_t lda,
                          const float* B, size_t ldb,
                          float beta, float* C, size_t ldc);

void rcre_avx512_gemv_f32(size_t M, size_t K, float alpha,
                          const float* A, size_t lda,
                          const float* x, float beta, float* y);

float rcre_avx512_dot_f32(const float* x, const float* y, size_t n);

void rcre_avx512_axpy_f32(size_t n, float alpha, const float* x, float* y);
#endif

/* WebAssembly 128-bit SIMD Micro-Kernels */
#if defined(__wasm_simd128__)
void rcre_wasm_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

void rcre_wasm_gemv_f32(size_t M, size_t K, float alpha,
                        const float* A, size_t lda,
                        const float* x, float beta, float* y);

float rcre_wasm_dot_f32(const float* x, const float* y, size_t n);

void rcre_wasm_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_wasm_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_wasm_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_wasm_count_byte(const uint8_t* hay, size_t len, uint8_t needle);

bool rcre_wasm_utf8_validate(const uint8_t* data, size_t len);
size_t rcre_wasm_find_substr(const uint8_t* hay, size_t hay_len,
                             const uint8_t* needle, size_t needle_len);
void rcre_wasm_ascii_to_lower(const uint8_t* src, uint8_t* dst, size_t len);
void rcre_wasm_ascii_to_upper(const uint8_t* src, uint8_t* dst, size_t len);
#endif

#endif /* RCRE_HAL_INTERNAL_H */
