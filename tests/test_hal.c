/**
 * @file test_hal.c
 * @brief Unit tests for RCRE Hardware Abstraction Layer & SIMD Micro-Kernels
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f

static void test_cpu_probe_and_backend(void) {
    printf("  -> Testing CPU feature probing and dynamic HAL backend...\n");
    rcre_hal_init();

    uint32_t feats = rcre_cpu_get_features();
    const char* brand = rcre_cpu_get_brand();
    assert(feats & RCRE_CPU_FEAT_SCALAR);
    assert(brand != NULL && brand[0] != '\0');

    printf("     Detected CPU: '%s'\n", brand);
    printf("     Active HAL Backend: '%s'\n", rcre_hal.backend_name);

#if defined(__ARM_NEON) || defined(__aarch64__)
    assert(feats & RCRE_CPU_FEAT_NEON);
    printf("     ✓ Confirmed ARM NEON hardware capability bit\n");
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    assert(feats & RCRE_CPU_FEAT_AVX2);
    printf("     ✓ Confirmed x86 AVX2 hardware capability bit\n");
#endif

#if defined(__wasm_simd128__)
    assert(feats & RCRE_CPU_FEAT_WASM_SIMD);
    printf("     ✓ Confirmed WebAssembly SIMD hardware capability bit\n");
#endif
}

static void test_hal_dot_f32(void) {
    printf("  -> Testing SIMD dot_f32 across various vector lengths...\n");
    const size_t test_sizes[] = {1, 3, 4, 7, 8, 15, 16, 23, 32, 63, 64, 127, 256, 1024};
    const size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (size_t s = 0; s < num_sizes; ++s) {
        size_t n = test_sizes[s];
        float* x = (float*)malloc(n * sizeof(float));
        float* y = (float*)malloc(n * sizeof(float));

        float scalar_truth = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            x[i] = (float)((int)(i % 17) - 8) * 0.25f;
            y[i] = (float)((int)(i % 13) - 6) * 0.5f;
            scalar_truth += x[i] * y[i];
        }

        float simd_result = rcre_hal.dot_f32(x, y, n);
        float diff = fabsf(simd_result - scalar_truth);
        float rel_err = diff / (fabsf(scalar_truth) + 1.0f);

        assert(rel_err < EPSILON);

        free(x);
        free(y);
    }
    printf("     ✓ dot_f32 verified across 14 vector sizes (rel_err < 1e-4)\n");
}

static void test_hal_axpy_f32(void) {
    printf("  -> Testing SIMD axpy_f32 (y = alpha * x + y)...\n");
    const size_t n = 250; /* Non-multiple of 16 */
    float* x = (float*)malloc(n * sizeof(float));
    float* y_simd = (float*)malloc(n * sizeof(float));
    float* y_ref = (float*)malloc(n * sizeof(float));

    float alpha = 2.5f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = (float)(i + 1) * 0.1f;
        y_simd[i] = (float)(i % 7) * 0.5f;
        y_ref[i] = y_simd[i];
    }

    /* Compute reference */
    for (size_t i = 0; i < n; ++i) {
        y_ref[i] += alpha * x[i];
    }

    rcre_hal.axpy_f32(n, alpha, x, y_simd);

    for (size_t i = 0; i < n; ++i) {
        assert(fabsf(y_simd[i] - y_ref[i]) < EPSILON);
    }

    free(x);
    free(y_simd);
    free(y_ref);
    printf("     ✓ axpy_f32 numerical parity verified\n");
}

static void test_hal_gemm_f32(void) {
    printf("  -> Testing cache-blocked SIMD gemm_f32...\n");
    /* Test non-multiple dimensions: M=37, N=43, K=29 */
    const size_t M = 37;
    const size_t N = 43;
    const size_t K = 29;

    const size_t lda = K + 4;
    const size_t ldb = N + 2;
    const size_t ldc = N + 3;

    float* A = (float*)malloc(M * lda * sizeof(float));
    float* B = (float*)malloc(K * ldb * sizeof(float));
    float* C_simd = (float*)malloc(M * ldc * sizeof(float));
    float* C_ref  = (float*)malloc(M * ldc * sizeof(float));

    for (size_t i = 0; i < M * lda; ++i) A[i] = (float)((int)(i % 11) - 5) * 0.2f;
    for (size_t i = 0; i < K * ldb; ++i) B[i] = (float)((int)(i % 7) - 3) * 0.3f;
    for (size_t i = 0; i < M * ldc; ++i) {
        C_simd[i] = (float)(i % 5) * 0.1f;
        C_ref[i]  = C_simd[i];
    }

    float alpha = 1.5f;
    float beta  = 0.75f;

    /* Compute reference C_ref = alpha * A * B + beta * C */
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C_ref[i * ldc + j] = alpha * sum + beta * C_ref[i * ldc + j];
        }
    }

    rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C_simd, ldc);

    float max_diff = 0.0f;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float diff = fabsf(C_simd[i * ldc + j] - C_ref[i * ldc + j]);
            if (diff > max_diff) max_diff = diff;
            assert(diff < 1e-3f);
        }
    }

    free(A);
    free(B);
    free(C_simd);
    free(C_ref);
    printf("     ✓ gemm_f32 (%zux%zux%zu) matched reference (max diff: %.6e)\n", M, N, K, max_diff);
}

static void test_hal_rgb_to_gray_u8(void) {
    printf("  -> Testing SIMD rgb_to_gray_u8 exact bit-parity...\n");
    const size_t num_pixels = 345; /* Non-multiple of 16 */
    uint8_t* rgb = (uint8_t*)malloc(num_pixels * 3);
    uint8_t* gray_simd = (uint8_t*)malloc(num_pixels);
    uint8_t* gray_ref  = (uint8_t*)malloc(num_pixels);

    for (size_t i = 0; i < num_pixels * 3; ++i) {
        rgb[i] = (uint8_t)(i * 37 + 13);
    }

    /* Scalar 15-bit fixed point reference */
    for (size_t i = 0; i < num_pixels; ++i) {
        uint32_t r = rgb[i * 3 + 0];
        uint32_t g = rgb[i * 3 + 1];
        uint32_t b = rgb[i * 3 + 2];
        gray_ref[i] = (uint8_t)((9798 * r + 19235 * g + 3735 * b + 16384) >> 15);
    }

    rcre_hal.rgb_to_gray_u8(rgb, gray_simd, num_pixels);

    for (size_t i = 0; i < num_pixels; ++i) {
        assert(gray_simd[i] == gray_ref[i]); /* Exact bit match */
    }

    free(rgb);
    free(gray_simd);
    free(gray_ref);
    printf("     ✓ rgb_to_gray_u8 exact 0-diff bit parity verified\n");
}

int main(void) {
    printf("🧪 Running RCRE Hardware Abstraction Layer Tests (HAL-REQ-01 to HAL-REQ-04)...\n");
    test_cpu_probe_and_backend();
    test_hal_dot_f32();
    test_hal_axpy_f32();
    test_hal_gemm_f32();
    test_hal_rgb_to_gray_u8();
    printf("✅ All Hardware Abstraction Layer Tests Passed Successfully!\n");
    return 0;
}
