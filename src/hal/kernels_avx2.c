/**
 * @file kernels_avx2.c
 * @brief x86 AVX2 + FMA 256-bit Vector Micro-Kernels
 * 
 * Satisfies HAL-REQ-03.
 * Targets AMD Zen and Intel Haswell+ processors.
 */

#include "hal_internal.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#include <immintrin.h>

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

void rcre_avx2_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                        const float* A, size_t lda, 
                        const float* B, size_t ldb, 
                        float beta, float* C, size_t ldc) {
    /* Fallback to tiled scalar if AVX2 micro-kernel driver not specialized */
    rcre_scalar_gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void rcre_avx2_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    rcre_scalar_rgb_to_gray_u8(rgb, gray, num_pixels);
}

#endif /* (__x86_64__ || _M_X64) && __AVX2__ */
