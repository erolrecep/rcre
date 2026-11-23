/**
 * @file kernels_avx512.c
 * @brief x86 AVX-512 512-bit Vector Micro-Kernels
 *
 * Targets Intel Skylake-X/Sapphire Rapids and AMD Zen 4/5 servers.
 * Compiled safely with runtime CPUID gating in hal_dispatch.c.
 */

#include "hal_internal.h"

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX512F__) || defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>

#if !defined(__AVX512F__)
  #if defined(__clang__)
    #pragma clang attribute push(__attribute__((target("avx512f,fma"))), apply_to = function)
    #define RCRE_NEED_POP_AVX512 2
  #elif defined(__GNUC__)
    #pragma GCC push_options
    #pragma GCC target("avx512f,fma")
    #define RCRE_NEED_POP_AVX512 1
  #endif
#endif

float rcre_avx512_dot_f32(const float* x, const float* y, size_t n) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 31 < n; i += 32) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(&x[i]),      _mm512_loadu_ps(&y[i]),      sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(&x[i + 16]), _mm512_loadu_ps(&y[i + 16]), sum1);
    }
    __m512 sum = _mm512_add_ps(sum0, sum1);
    for (; i + 15 < n; i += 16) {
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(&x[i]), _mm512_loadu_ps(&y[i]), sum);
    }
    float total = _mm512_reduce_add_ps(sum);
    for (; i < n; ++i) {
        total += x[i] * y[i];
    }
    return total;
}

void rcre_avx512_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    __m512 va = _mm512_set1_ps(alpha);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vy = _mm512_loadu_ps(&y[i]);
        _mm512_storeu_ps(&y[i], _mm512_fmadd_ps(va, vx, vy));
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void rcre_avx512_gemv_f32(size_t M, size_t K, float alpha,
                            const float* A, size_t lda,
                            const float* x, float beta, float* y) {
    for (size_t i = 0; i < M; ++i) {
        float acc = rcre_avx512_dot_f32(&A[i * lda], x, K);
        float prev = (beta == 0.0f) ? 0.0f : (y[i] * beta);
        y[i] = prev + alpha * acc;
    }
}

static void avx512_gemm_microkernel_4x16(size_t K, float alpha,
                                         const float* A, size_t lda,
                                         const float* B, size_t ldb,
                                         float beta, float* C, size_t ldc) {
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m512 b0 = _mm512_loadu_ps(&B[k * ldb]);
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), b0, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), b0, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), b0, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), b0, c3);
    }

    __m512 valpha = _mm512_set1_ps(alpha);
    c0 = _mm512_mul_ps(c0, valpha);
    c1 = _mm512_mul_ps(c1, valpha);
    c2 = _mm512_mul_ps(c2, valpha);
    c3 = _mm512_mul_ps(c3, valpha);

    if (beta != 0.0f) {
        __m512 vbeta = _mm512_set1_ps(beta);
        c0 = _mm512_fmadd_ps(vbeta, _mm512_loadu_ps(&C[0 * ldc]), c0);
        c1 = _mm512_fmadd_ps(vbeta, _mm512_loadu_ps(&C[1 * ldc]), c1);
        c2 = _mm512_fmadd_ps(vbeta, _mm512_loadu_ps(&C[2 * ldc]), c2);
        c3 = _mm512_fmadd_ps(vbeta, _mm512_loadu_ps(&C[3 * ldc]), c3);
    }

    _mm512_storeu_ps(&C[0 * ldc], c0);
    _mm512_storeu_ps(&C[1 * ldc], c1);
    _mm512_storeu_ps(&C[2 * ldc], c2);
    _mm512_storeu_ps(&C[3 * ldc], c3);
}

void rcre_avx512_gemm_f32(size_t M, size_t N, size_t K, float alpha,
                          const float* A, size_t lda,
                          const float* B, size_t ldb,
                          float beta, float* C, size_t ldc) {
    const size_t MB = 64, NB = 64, KB = 64;
    if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] = (beta == 0.0f) ? 0.0f : (C[i * ldc + j] * beta);
            }
        }
    }
    if (alpha == 0.0f) return;

    for (size_t ii = 0; ii < M; ii += MB) {
        size_t i_end = (ii + MB < M) ? ii + MB : M;
        for (size_t jj = 0; jj < N; jj += NB) {
            size_t j_end = (jj + NB < N) ? jj + NB : N;
            for (size_t kk = 0; kk < K; kk += KB) {
                size_t k_end = (kk + KB < K) ? kk + KB : K;
                size_t sub_K = k_end - kk;
                size_t i = ii;
                for (; i + 3 < i_end; i += 4) {
                    size_t j = jj;
                    for (; j + 15 < j_end; j += 16) {
                        avx512_gemm_microkernel_4x16(sub_K, alpha,
                                                     &A[i * lda + kk], lda,
                                                     &B[kk * ldb + j], ldb,
                                                     1.0f, &C[i * ldc + j], ldc);
                    }
                    for (; j < j_end; ++j) {
                        for (size_t r = 0; r < 4; ++r) {
                            float acc = 0.0f;
                            for (size_t k = kk; k < k_end; ++k) {
                                acc += A[(i + r) * lda + k] * B[k * ldb + j];
                            }
                            C[(i + r) * ldc + j] += alpha * acc;
                        }
                    }
                }
                for (; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        float acc = 0.0f;
                        for (size_t k = kk; k < k_end; ++k) {
                            acc += A[i * lda + k] * B[k * ldb + j];
                        }
                        C[i * ldc + j] += alpha * acc;
                    }
                }
            }
        }
    }
}

#if defined(RCRE_NEED_POP_AVX512)
  #if RCRE_NEED_POP_AVX512 == 2
/* 8x8 micro-kernel: eight row accumulators (__m512d = 8 doubles each);
 * each k-step broadcasts A[i,k] against the 8-wide B columns. */
static void avx512_gemm64_microkernel_8x8(size_t K, double alpha,
                              const double* A, size_t lda,
                              const double* B, size_t ldb,
                              double* C, size_t ldc) {
    __m512d acc[8];
    for (size_t i = 0; i < 8; ++i) acc[i] = _mm512_load_pd(&C[i * ldc]);
    for (size_t k = 0; k < K; ++k) {
        __m512d b0 = _mm512_load_pd(&B[k * ldb + 0]);
        __m512d b1 = _mm512_load_pd(&B[k * ldb + 1]);
        __m512d b2 = _mm512_load_pd(&B[k * ldb + 2]);
        __m512d b3 = _mm512_load_pd(&B[k * ldb + 3]);
        __m512d b4 = _mm512_load_pd(&B[k * ldb + 4]);
        __m512d b5 = _mm512_load_pd(&B[k * ldb + 5]);
        __m512d b6 = _mm512_load_pd(&B[k * ldb + 6]);
        __m512d b7 = _mm512_load_pd(&B[k * ldb + 7]);
        for (size_t i = 0; i < 8; ++i) {
            __m512d a = _mm512_set1_pd(A[i * lda + k] * alpha);
            acc[i] = _mm512_fmadd_pd(a, b0, acc[i]);
            acc[i] = _mm512_fmadd_pd(a, b1, acc[i]);
            acc[i] = _mm512_fmadd_pd(a, b2, acc[i]);
            acc[i] = _mm512_fmadd_pd(a, b3, acc[i]);
            acc[i] = _mm512_fmadd_pd(a, b4, acc[i]);
            acc[i] = _mm512_fmadd_pd(a, b5, acc[i]);
            acc[i] = _mm512_fmadd_pd(a, b6, acc[i]);
            acc[i] = _mm512_fmadd_pd(a, b7, acc[i]);
        }
    }
    for (size_t i = 0; i < 8; ++i) _mm512_store_pd(&C[i * ldc], acc[i]);
}

void rcre_avx512_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                     const double* A, size_t lda, const double* B, size_t ldb,
                     double beta, double* C, size_t ldc) {
    if (beta != 1.0) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i * ldc + j] = (beta == 0.0) ? 0.0 : C[i * ldc + j] * beta;
    }
    if (alpha == 0.0 || M == 0 || N == 0 || K == 0) return;

    const size_t MB = 64, NB = 64, KB = 64;
    for (size_t ii = 0; ii < M; ii += MB) {
        size_t i_end = (ii + MB < M) ? ii + MB : M;
        for (size_t jj = 0; jj < N; jj += NB) {
            size_t j_end = (jj + NB < N) ? jj + NB : N;
            for (size_t kk = 0; kk < K; kk += KB) {
                size_t k_end = (kk + KB < K) ? kk + KB : K;
                size_t i = ii;
                for (; i + 7 < i_end; i += 8) {
                    size_t j = jj;
                    for (; j + 7 < j_end; j += 8)
                        avx512_gemm64_microkernel_8x8(k_end - kk, alpha,
                                                      &A[i * lda + kk], lda,
                                                      &B[kk * ldb + j], ldb,
                                                      &C[i * ldc + j], ldc);
                    for (; j < j_end; ++j)
                        for (size_t r = 0; r < 8; ++r) {
                            double acc = 0.0;
                            for (size_t k = kk; k < k_end; ++k)
                                acc += A[(i + r) * lda + k] * B[k * ldb + j];
                            C[(i + r) * ldc + j] += acc;
                        }
                }
                for (; i < i_end; ++i)
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

double rcre_avx512_dot_f64(const double* x, const double* y, size_t n) {
    __m512d acc = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm512_fmadd_pd(_mm512_load_pd(&x[i]), _mm512_load_pd(&y[i]), acc);
    double t[8];
    _mm512_store_pd(t, acc);
    double s = t[0];
    for (int k = 1; k < 8; ++k) s += t[k];
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}

void rcre_avx512_axpy_f64(size_t n, double alpha, const double* x, double* y) {
    __m512d va = _mm512_set1_pd(alpha);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_store_pd(&y[i], _mm512_fmadd_pd(va, _mm512_load_pd(&x[i]),
                                               _mm512_load_pd(&y[i])));
    for (; i < n; ++i) y[i] += alpha * x[i];
}

void rcre_avx512_gemv_f64(size_t M, size_t N, double alpha,
                     const double* A, size_t lda, const double* x,
                     double beta, double* y) {
    for (size_t i = 0; i < M; ++i) {
        __m512d acc = _mm512_setzero_pd();
        const double* row = A + i * lda;
        size_t j = 0;
        for (; j + 8 <= N; j += 8)
            acc = _mm512_fmadd_pd(_mm512_load_pd(&row[j]),
                                  _mm512_load_pd(&x[j]), acc);
        double t[8];
        _mm512_store_pd(t, acc);
        double s = t[0];
        for (int k = 1; k < 8; ++k) s += t[k];
        for (; j < N; ++j) s += row[j] * x[j];
        y[i] = (beta == 0.0) ? alpha * s : beta * y[i] + alpha * s;
    }
}

void rcre_avx512_scale_f64(size_t n, double alpha, double* x) {
    __m512d va = _mm512_set1_pd(alpha);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_store_pd(&x[i], _mm512_mul_pd(_mm512_load_pd(&x[i]), va));
    for (; i < n; ++i) x[i] *= alpha;
}

double rcre_avx512_sum_f64(const double* x, size_t n) {
    __m512d acc = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm512_add_pd(acc, _mm512_load_pd(&x[i]));
    double t[8];
    _mm512_store_pd(t, acc);
    double s = t[0];
    for (int k = 1; k < 8; ++k) s += t[k];
    for (; i < n; ++i) s += x[i];
    return s;
}

    #pragma clang attribute pop
  #elif RCRE_NEED_POP_AVX512 == 1
    #pragma GCC pop_options
  #endif
  #undef RCRE_NEED_POP_AVX512
#endif

#endif /* (__x86_64__ || _M_X64) && AVX512F-capable compiler */
