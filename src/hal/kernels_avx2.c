/**
 * @file kernels_avx2.c
 * @brief x86 AVX2 + FMA 256-bit Vector Micro-Kernels
 * 
 * Satisfies HAL-REQ-03.
 * Targets AMD Zen and Intel Haswell+ processors.
 */

#include "hal_internal.h"

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#include <string.h>

#if !defined(__AVX2__)
  #if defined(__clang__)
    #pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
  #elif defined(__GNUC__)
    #pragma GCC push_options
    #pragma GCC target("avx2,fma")
  #endif
#endif

float rcre_avx2_dot_f32(const float* x, const float* y, size_t n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(&x[i + 0]);
        __m256 y0 = _mm256_loadu_ps(&y[i + 0]);
        sum0 = _mm256_fmadd_ps(x0, y0, sum0);

        __m256 x1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 y1 = _mm256_loadu_ps(&y[i + 8]);
        sum1 = _mm256_fmadd_ps(x1, y1, sum1);
    }

    __m256 sum_all = _mm256_add_ps(sum0, sum1);

    /* Horizontal reduction of 8 floats in __m256 */
    __m128 vlow = _mm256_castps256_ps128(sum_all);
    __m128 vhigh = _mm256_extractf128_ps(sum_all, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(v128, v128);
    __m128 sums = _mm_add_ps(v128, shuf);
    shuf = _mm_shuffle_ps(sums, sums, 1);
    sums = _mm_add_ss(sums, shuf);
    float total = _mm_cvtss_f32(sums);

    for (; i < n; ++i) {
        total += x[i] * y[i];
    }

    return total;
}

void rcre_avx2_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    __m256 a_vec = _mm256_set1_ps(alpha);

    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(&x[i + 0]);
        __m256 y0 = _mm256_loadu_ps(&y[i + 0]);
        _mm256_storeu_ps(&y[i + 0], _mm256_fmadd_ps(a_vec, x0, y0));

        __m256 x1 = _mm256_loadu_ps(&x[i + 8]);
        __m256 y1 = _mm256_loadu_ps(&y[i + 8]);
        _mm256_storeu_ps(&y[i + 8], _mm256_fmadd_ps(a_vec, x1, y1));
    }

    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/* 4x16 register-blocked GEMM micro-kernel on AVX2 + FMA (8 YMM accumulators) */
static void avx2_gemm_microkernel_4x16(size_t K, float alpha,
                                       const float* A, size_t lda,
                                       const float* B, size_t ldb,
                                       float beta, float* C, size_t ldc) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    __m256 a_vec = _mm256_set1_ps(alpha);
    __m256 b_vec = _mm256_set1_ps(beta);

    #define STORE_ROW_AVX2_16(row, acc0, acc1) do { \
        __m256 old0 = (beta == 0.0f) ? _mm256_setzero_ps() : _mm256_loadu_ps(&C[(row) * ldc + 0]); \
        __m256 old1 = (beta == 0.0f) ? _mm256_setzero_ps() : _mm256_loadu_ps(&C[(row) * ldc + 8]); \
        __m256 out0 = _mm256_fmadd_ps(acc0, a_vec, _mm256_mul_ps(old0, b_vec)); \
        __m256 out1 = _mm256_fmadd_ps(acc1, a_vec, _mm256_mul_ps(old1, b_vec)); \
        _mm256_storeu_ps(&C[(row) * ldc + 0], out0); \
        _mm256_storeu_ps(&C[(row) * ldc + 8], out1); \
    } while(0)

    STORE_ROW_AVX2_16(0, c00, c01);
    STORE_ROW_AVX2_16(1, c10, c11);
    STORE_ROW_AVX2_16(2, c20, c21);
    STORE_ROW_AVX2_16(3, c30, c31);
    #undef STORE_ROW_AVX2_16
}

/* 4x8 register-blocked GEMM micro-kernel fallback for narrower column blocks */
static void avx2_gemm_microkernel_4x8(size_t K, float alpha,
                                      const float* A, size_t lda,
                                      const float* B, size_t ldb,
                                      float beta, float* C, size_t ldc) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb + 0]);

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b0, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b0, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b0, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b0, c3);
    }

    __m256 a_vec = _mm256_set1_ps(alpha);
    __m256 b_vec = _mm256_set1_ps(beta);

    #define STORE_ROW_AVX2_8(row, acc) do { \
        __m256 old = (beta == 0.0f) ? _mm256_setzero_ps() : _mm256_loadu_ps(&C[(row) * ldc + 0]); \
        __m256 out = _mm256_fmadd_ps(acc, a_vec, _mm256_mul_ps(old, b_vec)); \
        _mm256_storeu_ps(&C[(row) * ldc + 0], out); \
    } while(0)

    STORE_ROW_AVX2_8(0, c0);
    STORE_ROW_AVX2_8(1, c1);
    STORE_ROW_AVX2_8(2, c2);
    STORE_ROW_AVX2_8(3, c3);
    #undef STORE_ROW_AVX2_8
}

void rcre_avx2_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc) {
    /* Cache-blocked driver: MB x NB x KB blocks */
    const size_t MB = 64;
    const size_t NB = 64;
    const size_t KB = 64;

    /* Initialize C with beta scaling if needed */
    if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] = (beta == 0.0f) ? 0.0f : C[i * ldc + j] * beta;
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

                /* 4x16 and 4x8 tiled blocks */
                size_t i = ii;
                for (; i + 3 < i_end; i += 4) {
                    size_t j = jj;
                    for (; j + 15 < j_end; j += 16) {
                        avx2_gemm_microkernel_4x16(sub_K, alpha,
                                                   &A[i * lda + kk], lda,
                                                   &B[kk * ldb + j], ldb,
                                                   1.0f, /* C pre-scaled by beta */
                                                   &C[i * ldc + j], ldc);
                    }
                    for (; j + 7 < j_end; j += 8) {
                        avx2_gemm_microkernel_4x8(sub_K, alpha,
                                                  &A[i * lda + kk], lda,
                                                  &B[kk * ldb + j], ldb,
                                                  1.0f,
                                                  &C[i * ldc + j], ldc);
                    }
                    /* Remainder columns in 4-row block */
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

                /* Remainder rows */
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

void rcre_avx2_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    const __m128i shuf_r = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shuf_g = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shuf_b = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    const __m256i coeff_r = _mm256_set1_epi32(9798);
    const __m256i coeff_g = _mm256_set1_epi32(19235);
    const __m256i coeff_b = _mm256_set1_epi32(3735);
    const __m256i bias    = _mm256_set1_epi32(16384);

    size_t i = 0;
    for (; i + 7 < num_pixels; i += 8) {
        /* Read exactly 24 bytes (8 pixels) safely without buffer over-read */
        uint64_t lo0 = *(const uint64_t*)&rgb[i * 3 + 0];
        uint32_t hi0 = *(const uint32_t*)&rgb[i * 3 + 8];
        __m128i v0 = _mm_cvtsi64_si128((int64_t)lo0);
        v0 = _mm_insert_epi32(v0, (int32_t)hi0, 2);

        uint64_t lo1 = *(const uint64_t*)&rgb[i * 3 + 12];
        uint32_t hi1 = *(const uint32_t*)&rgb[i * 3 + 20];
        __m128i v1 = _mm_cvtsi64_si128((int64_t)lo1);
        v1 = _mm_insert_epi32(v1, (int32_t)hi1, 2);

        /* De-interleave R, G, B channels */
        __m128i r0 = _mm_shuffle_epi8(v0, shuf_r);
        __m128i r1 = _mm_shuffle_epi8(v1, shuf_r);
        __m128i r8 = _mm_unpacklo_epi32(r0, r1);

        __m128i g0 = _mm_shuffle_epi8(v0, shuf_g);
        __m128i g1 = _mm_shuffle_epi8(v1, shuf_g);
        __m128i g_chan8 = _mm_unpacklo_epi32(g0, g1);

        __m128i b0 = _mm_shuffle_epi8(v0, shuf_b);
        __m128i b1 = _mm_shuffle_epi8(v1, shuf_b);
        __m128i b8 = _mm_unpacklo_epi32(b0, b1);

        /* Zero-extend 8 uint8s into 8 uint32s in 256-bit registers */
        __m256i r32 = _mm256_cvtepu8_epi32(r8);
        __m256i g32 = _mm256_cvtepu8_epi32(g_chan8);
        __m256i b32 = _mm256_cvtepu8_epi32(b8);

        /* Fixed-point Rec.601 integer calculation */
        __m256i acc = bias;
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(r32, coeff_r));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(g32, coeff_g));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(b32, coeff_b));
        __m256i gray32 = _mm256_srli_epi32(acc, 15);

        /* Pack 8 32-bit integers down to 8 uint8 bytes */
        __m128i g_lo = _mm256_castsi256_si128(gray32);
        __m128i g_hi = _mm256_extracti128_si256(gray32, 1);
        __m128i g16  = _mm_packus_epi32(g_lo, g_hi);
        __m128i gray8 = _mm_packus_epi16(g16, g16);

        uint64_t res8 = (uint64_t)_mm_cvtsi128_si64(gray8);
        memcpy(&gray[i], &res8, 8);
    }

    /* Scalar remainder */
    for (; i < num_pixels; ++i) {
        uint32_t r = rgb[i * 3 + 0];
        uint32_t g = rgb[i * 3 + 1];
        uint32_t b = rgb[i * 3 + 2];
        gray[i] = (uint8_t)((9798 * r + 19235 * g + 3735 * b + 16384) >> 15);
    }
}

size_t rcre_avx2_find_byte(const uint8_t* hay, size_t len, uint8_t needle) {
    if (!hay || len == 0) {
        return RCRE_NPOS;
    }

    const __m256i vn = _mm256_set1_epi8((char)needle);
    size_t i = 0;

    for (; i + 63 < len; i += 64) {
        uint32_t m0 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)&hay[i +  0]), vn));
        uint32_t m1 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)&hay[i + 32]), vn));
        if ((m0 | m1) != 0) {
            if (m0 != 0) return i + (size_t)__builtin_ctz(m0);
            return i + 32 + (size_t)__builtin_ctz(m1);
        }
    }

    for (; i + 31 < len; i += 32) {
        uint32_t m0 = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)&hay[i]), vn));
        if (m0 != 0) {
            return i + (size_t)__builtin_ctz(m0);
        }
    }

    for (; i < len; ++i) {
        if (hay[i] == needle) return i;
    }
    return RCRE_NPOS;
}

size_t rcre_avx2_count_byte(const uint8_t* hay, size_t len, uint8_t needle) {
    if (!hay || len == 0) {
        return 0;
    }

    const __m256i vn   = _mm256_set1_epi8((char)needle);
    const __m256i zero = _mm256_setzero_si256();
    size_t total = 0;
    size_t i = 0;

    while (i + 63 < len) {
        __m256i acc0 = zero;
        __m256i acc1 = zero;

        size_t block_iters = (len - i) / 64;
        if (block_iters > 255) {
            block_iters = 255;
        }
        for (size_t b = 0; b < block_iters; ++b, i += 64) {
            acc0 = _mm256_sub_epi8(acc0, _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)&hay[i +  0]), vn));
            acc1 = _mm256_sub_epi8(acc1, _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)&hay[i + 32]), vn));
        }

        __m256i sad = _mm256_add_epi64(_mm256_sad_epu8(acc0, zero), _mm256_sad_epu8(acc1, zero));
        __m128i lo  = _mm256_castsi256_si128(sad);
        __m128i hi  = _mm256_extracti128_si256(sad, 1);
        __m128i sum = _mm_add_epi64(lo, hi);
        total += (size_t)_mm_cvtsi128_si64(sum) + (size_t)_mm_extract_epi64(sum, 1);
    }

    for (; i < len; ++i) {
        total += (size_t)(hay[i] == needle);
    }
    return total;
}

#if !defined(__AVX2__)
  #if defined(__clang__)
    #pragma clang attribute pop
  #elif defined(__GNUC__)
    #pragma GCC pop_options
  #endif
#endif

#endif /* (__x86_64__ || _M_X64) && AVX2-capable compiler */
