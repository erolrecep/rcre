/**
 * @file test_hal.c
 * @brief Unit tests for RCRE Hardware Abstraction Layer & SIMD Micro-Kernels
 */

#include "rcre.h"
#include "test_framework.h"

#define EPSILON 1e-4f

static void test_cpu_probe_and_backend(void) {
    rcre_hal_init();

    uint32_t feats = rcre_cpu_get_features();
    const char* brand = rcre_cpu_get_brand();
    ASSERT_TRUE(feats & RCRE_CPU_FEAT_SCALAR);
    ASSERT_NOT_NULL(brand);
    ASSERT_TRUE(brand[0] != '\0');

#if defined(__ARM_NEON) || defined(__aarch64__)
    ASSERT_TRUE(feats & RCRE_CPU_FEAT_NEON);
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    ASSERT_TRUE(feats & RCRE_CPU_FEAT_AVX2);
#endif

#if defined(__wasm_simd128__)
    ASSERT_TRUE(feats & RCRE_CPU_FEAT_WASM_SIMD);
#endif
}

static void test_hal_dot_f32(void) {
    const size_t test_sizes[] = {1, 3, 4, 7, 8, 15, 16, 23, 32, 63, 64, 127, 256, 1024};
    const size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (size_t s = 0; s < num_sizes; ++s) {
        size_t n = test_sizes[s];
        float* x = (float*)malloc(n * sizeof(float));
        float* y = (float*)malloc(n * sizeof(float));
        ASSERT_NOT_NULL(x);
        ASSERT_NOT_NULL(y);

        float scalar_truth = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            x[i] = (float)((int)(i % 17) - 8) * 0.25f;
            y[i] = (float)((int)(i % 13) - 6) * 0.5f;
            scalar_truth += x[i] * y[i];
        }

        float simd_result = rcre_hal.dot_f32(x, y, n);
        float diff = fabsf(simd_result - scalar_truth);
        float tolerance = (fabsf(scalar_truth) + 1.0f) * EPSILON;
        ASSERT_TRUE(diff <= tolerance);

        free(x);
        free(y);
    }
}

static void test_hal_axpy_f32(void) {
    const size_t n = 250; /* Non-multiple of 16 */
    float* x = (float*)malloc(n * sizeof(float));
    float* y_simd = (float*)malloc(n * sizeof(float));
    float* y_ref = (float*)malloc(n * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(y_simd);
    ASSERT_NOT_NULL(y_ref);

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
    ASSERT_FLOAT_ARRAY_NEAR(y_ref, y_simd, n, EPSILON);

    free(x);
    free(y_simd);
    free(y_ref);
}

static void test_hal_gemm_f32(void) {
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
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(C_simd);
    ASSERT_NOT_NULL(C_ref);

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
    ASSERT_FLOAT_ARRAY_NEAR(C_ref, C_simd, M * ldc, 1e-3f);

    free(A);
    free(B);
    free(C_simd);
    free(C_ref);
}

static void test_hal_rgb_to_gray_u8(void) {
    const size_t num_pixels = 345; /* Non-multiple of 16 */
    uint8_t* rgb = (uint8_t*)malloc(num_pixels * 3);
    uint8_t* gray_simd = (uint8_t*)malloc(num_pixels);
    uint8_t* gray_ref  = (uint8_t*)malloc(num_pixels);
    ASSERT_NOT_NULL(rgb);
    ASSERT_NOT_NULL(gray_simd);
    ASSERT_NOT_NULL(gray_ref);

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
    ASSERT_MEM_EQ(gray_ref, gray_simd, num_pixels); /* Exact bit match */

    free(rgb);
    free(gray_simd);
    free(gray_ref);
}

static void test_hal_simd_byte_scanners(void) {
    /* Large buffer (> 255 * 64 = 16,320 bytes) to exercise uint8 lane overflow batching */
    const size_t N = 40019;
    uint8_t* buf = (uint8_t*)malloc(N);
    ASSERT_NOT_NULL(buf);
    memset(buf, '.', N);

    /* Test find_byte at every SIMD lane alignment & boundary */
    const size_t positions[] = {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 511, 16384, 40018};
    for (size_t p = 0; p < sizeof(positions) / sizeof(positions[0]); ++p) {
        size_t pos = positions[p];
        buf[pos] = '\n';
        /* Also test unaligned base pointer (+1, +3, +7) */
        for (size_t skew = 0; skew <= 7 && skew <= pos; ++skew) {
            size_t found = rcre_hal.find_byte(buf + skew, N - skew, '\n');
            ASSERT_UINT_EQ(pos - skew, found);
        }
        buf[pos] = '.';
    }
    ASSERT_UINT_EQ(RCRE_NPOS, rcre_hal.find_byte(buf, N, '\n'));

    /* Test dense count_byte (every byte matches -> tests >255 uint8 accumulator saturation safety) */
    ASSERT_UINT_EQ(N, rcre_hal.count_byte(buf, N, '.'));
    ASSERT_UINT_EQ(0, rcre_hal.count_byte(buf, N, '\n'));

    /* Sparse pseudo-random pattern parity vs scalar */
    size_t expected_nl = 0;
    for (size_t i = 0; i < N; ++i) {
        if ((i * 131 + 17) % 29 == 0) {
            buf[i] = '\n';
            expected_nl++;
        }
    }
    ASSERT_UINT_EQ(expected_nl, rcre_hal.count_byte(buf, N, '\n'));

    free(buf);
}

static void test_cpu_topology_and_pressure(void) {
    RCRE_CpuTopology topo = rcre_cpu_get_topology();
    ASSERT_TRUE(topo.logical_cores >= 1);
    ASSERT_TRUE(topo.perf_cores >= 1);
    ASSERT_TRUE(topo.effective_quota_cores >= 1);
    ASSERT_TRUE(topo.effective_quota_cores <= topo.logical_cores);

    int pressure = rcre_cpu_get_memory_pressure();
    ASSERT_TRUE(pressure >= 0 && pressure <= 2);
}

int main(void) {
    test_suite_begin("RCRE Hardware Abstraction Layer & SIMD (HAL-REQ-01 to 04)");
    TEST_RUN(test_cpu_probe_and_backend);
    TEST_RUN(test_hal_dot_f32);
    TEST_RUN(test_hal_axpy_f32);
    TEST_RUN(test_hal_gemm_f32);
    TEST_RUN(test_hal_rgb_to_gray_u8);
    TEST_RUN(test_hal_simd_byte_scanners);
    TEST_RUN(test_cpu_topology_and_pressure);
    return test_suite_report();
}
