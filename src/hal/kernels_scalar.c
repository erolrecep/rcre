/**
 * @file kernels_scalar.c
 * @brief ISO C23 Reference Fallback Micro-Kernels
 * 
 * Satisfies HAL-REQ-02.
 */

#include "hal_internal.h"

float rcre_scalar_dot_f32(const float* x, const float* y, size_t n) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        acc0 += x[i + 0] * y[i + 0];
        acc1 += x[i + 1] * y[i + 1];
        acc2 += x[i + 2] * y[i + 2];
        acc3 += x[i + 3] * y[i + 3];
    }
    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

void rcre_scalar_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        y[i + 0] += alpha * x[i + 0];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void rcre_scalar_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                          const float* A, size_t lda, 
                          const float* B, size_t ldb, 
                          float beta, float* C, size_t ldc) {
    /* Scale C by beta first if beta != 1.0f */
    if (beta == 0.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    if (alpha == 0.0f) return;

    /* Cache-blocked GEMM: MB x NB x KB blocks */
    const size_t MB = 64;
    const size_t NB = 64;
    const size_t KB = 64;

    for (size_t ii = 0; ii < M; ii += MB) {
        size_t i_end = (ii + MB < M) ? ii + MB : M;

        for (size_t jj = 0; jj < N; jj += NB) {
            size_t j_end = (jj + NB < N) ? jj + NB : N;

            for (size_t kk = 0; kk < K; kk += KB) {
                size_t k_end = (kk + KB < K) ? kk + KB : K;

                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        float a_val = alpha * A[i * lda + k];
                        for (size_t j = jj; j < j_end; ++j) {
                            C[i * ldc + j] += a_val * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
}

void rcre_scalar_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    /* Exact Rec.601 15-bit fixed-point: 9798 * R + 19235 * G + 3735 * B (sum = 32768) */
    for (size_t i = 0; i < num_pixels; ++i) {
        uint32_t r = rgb[i * 3 + 0];
        uint32_t g = rgb[i * 3 + 1];
        uint32_t b = rgb[i * 3 + 2];
        gray[i] = (uint8_t)((9798 * r + 19235 * g + 3735 * b + 16384) >> 15);
    }
}
