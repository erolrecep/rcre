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

size_t rcre_scalar_find_byte(const uint8_t* hay, size_t len, uint8_t needle) {
    if (!hay || len == 0) {
        return RCRE_NPOS;
    }
    size_t i = 0;
    for (; i + 7 < len; i += 8) {
        if (hay[i + 0] == needle) return i + 0;
        if (hay[i + 1] == needle) return i + 1;
        if (hay[i + 2] == needle) return i + 2;
        if (hay[i + 3] == needle) return i + 3;
        if (hay[i + 4] == needle) return i + 4;
        if (hay[i + 5] == needle) return i + 5;
        if (hay[i + 6] == needle) return i + 6;
        if (hay[i + 7] == needle) return i + 7;
    }
    for (; i < len; ++i) {
        if (hay[i] == needle) return i;
    }
    return RCRE_NPOS;
}

size_t rcre_scalar_count_byte(const uint8_t* hay, size_t len, uint8_t needle) {
    if (!hay || len == 0) {
        return 0;
    }
    size_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        c0 += (size_t)(hay[i + 0] == needle);
        c1 += (size_t)(hay[i + 1] == needle);
        c2 += (size_t)(hay[i + 2] == needle);
        c3 += (size_t)(hay[i + 3] == needle);
    }
    size_t total = (c0 + c1) + (c2 + c3);
    for (; i < len; ++i) {
        total += (size_t)(hay[i] == needle);
    }
    return total;
}

void rcre_scalar_gemv_f32(size_t M, size_t K, float alpha,
                          const float* A, size_t lda,
                          const float* x, float beta, float* y) {
    for (size_t i = 0; i < M; ++i) {
        float acc = rcre_scalar_dot_f32(&A[i * lda], x, K);
        float prev = (beta == 0.0f) ? 0.0f : (y[i] * beta);
        y[i] = prev + alpha * acc;
    }
}

int32_t rcre_scalar_dot_i8(const int8_t* x, const int8_t* y, size_t n) {
    int32_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += (int32_t)x[i] * (int32_t)y[i];
    }
    return sum;
}

void rcre_scalar_gemm_i8(size_t M, size_t N, size_t K,
                         const int8_t* A, size_t lda,
                         const int8_t* B, size_t ldb,
                         int32_t* C, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * ldc + j] = 0;
        }
    }
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int32_t a_val = (int32_t)A[i * lda + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] += a_val * (int32_t)B[k * ldb + j];
            }
        }
    }
}

static inline float scalar_bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f = 0.0f;
    __builtin_memcpy(&f, &u, sizeof(float));
    return f;
}

void rcre_scalar_gemm_bf16(size_t M, size_t N, size_t K, float alpha,
                           const uint16_t* A, size_t lda,
                           const uint16_t* B, size_t ldb,
                           float beta, float* C, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * ldc + j] = (beta == 0.0f) ? 0.0f : (C[i * ldc + j] * beta);
        }
    }
    if (alpha == 0.0f) return;
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_val = alpha * scalar_bf16_to_f32(A[i * lda + k]);
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] += a_val * scalar_bf16_to_f32(B[k * ldb + j]);
            }
        }
    }
}

/* H0 (2.1.0): F64 substrate + F32 completeness — scalar reference cells     */
/* =========================================================================
 * These are the correctness baseline for every ISA backend: tests gate
 * NEON/AVX2/AVX-512 against exactly these routines (the e-means stream
 * culture scaled to the HAL). Determinism: single-threaded, fixed
 * accumulation order — bitwise reproducible by construction.
 * (gemv_f32 already lives above from main's v2.0 HAL extension.)
 * ========================================================================= */

void rcre_scalar_gemv_f64(size_t M, size_t N, double alpha,
                          const double* A, size_t lda, const double* x,
                          double beta, double* y) {
    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        const double* row = A + i * lda;
        for (size_t j = 0; j < N; ++j) acc += row[j] * x[j];
        y[i] = (beta == 0.0) ? alpha * acc : beta * y[i] + alpha * acc;
    }
}

void rcre_scalar_scale_f32(size_t n, float alpha, float* x) {
    for (size_t i = 0; i < n; ++i) x[i] *= alpha;
}

void rcre_scalar_scale_f64(size_t n, double alpha, double* x) {
    for (size_t i = 0; i < n; ++i) x[i] *= alpha;
}

float rcre_scalar_sum_f32(const float* x, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

double rcre_scalar_sum_f64(const double* x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

double rcre_scalar_dot_f64(const double* x, const double* y, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

void rcre_scalar_axpy_f64(size_t n, double alpha, const double* x, double* y) {
    for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

/* Cache-blocked F64 GEMM, 8x8 micro-kernel — same driver structure as the
 * F32 NEON path (MB/NB/KB blocking), so the blocking strategy is uniform
 * across the precision family and H2 only has to replace micro-kernels. */
#define RCRE_SCAL_GEMM_MB 128
#define RCRE_SCAL_GEMM_NB 128
#define RCRE_SCAL_GEMM_KB 128

static void scalar_gemm_microkernel_8x8(size_t K,
                                        const double* A, size_t lda,
                                        const double* B, size_t ldb,
                                        double* C, size_t ldc) {
    double c[8][8] = { {0.0} };
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < 8; ++j) {
            const double bk = B[k * ldb + j];
            for (size_t i = 0; i < 8; ++i) c[i][j] += A[i * lda + k] * bk;
        }
    }
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j) C[i * ldc + j] += c[i][j];
}

void rcre_scalar_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                          const double* A, size_t lda, const double* B, size_t ldb,
                          double beta, double* C, size_t ldc) {
    /* beta pre-scale (same convention as the F32 driver) */
    if (beta != 1.0) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i * ldc + j] = (beta == 0.0) ? 0.0 : C[i * ldc + j] * beta;
    }
    if (alpha == 0.0 || M == 0 || N == 0 || K == 0) return;

    for (size_t ii = 0; ii < M; ii += RCRE_SCAL_GEMM_MB) {
        size_t i_end = (ii + RCRE_SCAL_GEMM_MB < M) ? ii + RCRE_SCAL_GEMM_MB : M;
        for (size_t jj = 0; jj < N; jj += RCRE_SCAL_GEMM_NB) {
            size_t j_end = (jj + RCRE_SCAL_GEMM_NB < N) ? jj + RCRE_SCAL_GEMM_NB : N;
            for (size_t kk = 0; kk < K; kk += RCRE_SCAL_GEMM_KB) {
                size_t k_end = (kk + RCRE_SCAL_GEMM_KB < K) ? kk + RCRE_SCAL_GEMM_KB : K;
                size_t i = ii;
                for (; i + 7 < i_end; i += 8) {
                    size_t j = jj;
                    for (; j + 7 < j_end; j += 8)
                        scalar_gemm_microkernel_8x8(k_end - kk,
                                                    &A[i * lda + kk], lda,
                                                    &B[kk * ldb + j], ldb,
                                                    &C[i * ldc + j], ldc);
                    for (; j < j_end; ++j) {
                        for (size_t r = 0; r < 8; ++r) {
                            double acc = 0.0;
                            for (size_t k = kk; k < k_end; ++k)
                                acc += A[(i + r) * lda + k] * B[k * ldb + j];
                            C[(i + r) * ldc + j] += acc;
                        }
                    }
                }
                for (; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        double acc = 0.0;
                        for (size_t k = kk; k < k_end; ++k)
                            acc += A[i * lda + k] * B[k * ldb + j];
                        C[i * ldc + j] += acc;
                    }
                }
            }
        }
    }
    if (alpha != 1.0) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i * ldc + j] *= alpha;
    }
}
