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

/* ARM NEON 128-bit Micro-Kernels */
#if defined(__ARM_NEON) || defined(__aarch64__)
void rcre_neon_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

float rcre_neon_dot_f32(const float* x, const float* y, size_t n);

void rcre_neon_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_neon_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);
#endif

/* x86 AVX2 256-bit Micro-Kernels */
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
void rcre_avx2_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc);

float rcre_avx2_dot_f32(const float* x, const float* y, size_t n);

void rcre_avx2_axpy_f32(size_t n, float alpha, const float* x, float* y);

void rcre_avx2_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);
#endif

#endif /* RCRE_HAL_INTERNAL_H */
