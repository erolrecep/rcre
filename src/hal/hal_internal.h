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

float rcre_scalar_dot_f32(const float* x, const float* y, size_t n);

void rcre_scalar_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_scalar_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_scalar_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_scalar_count_byte(const uint8_t* hay, size_t len, uint8_t needle);

/* ARM NEON 128-bit Micro-Kernels */
#if defined(__ARM_NEON) || defined(__aarch64__)
void rcre_neon_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

float rcre_neon_dot_f32(const float* x, const float* y, size_t n);

void rcre_neon_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_neon_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_neon_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_neon_count_byte(const uint8_t* hay, size_t len, uint8_t needle);
#endif

/* x86 AVX2 256-bit Micro-Kernels (runtime-dispatched on Clang/GCC even without global -mavx2) */
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__clang__) || defined(__GNUC__))
void rcre_avx2_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

float rcre_avx2_dot_f32(const float* x, const float* y, size_t n);

void rcre_avx2_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_avx2_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_avx2_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_avx2_count_byte(const uint8_t* hay, size_t len, uint8_t needle);
#endif

/* WebAssembly 128-bit SIMD Micro-Kernels */
#if defined(__wasm_simd128__)
void rcre_wasm_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

float rcre_wasm_dot_f32(const float* x, const float* y, size_t n);

void rcre_wasm_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_wasm_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

size_t rcre_wasm_find_byte(const uint8_t* hay, size_t len, uint8_t needle);

size_t rcre_wasm_count_byte(const uint8_t* hay, size_t len, uint8_t needle);
#endif

#endif /* RCRE_HAL_INTERNAL_H */
