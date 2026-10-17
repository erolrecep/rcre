/**
 * @file bench_hal.c
 * @brief Performance Benchmark: SIMD Micro-Kernels vs Scalar Reference
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Forward declare scalar routines for direct A/B comparison */
void rcre_scalar_gemm_f32(size_t M, size_t N, size_t K, float alpha, 
                          const float* A, size_t lda, 
                          const float* B, size_t ldb, 
                          float beta, float* C, size_t ldc);
void rcre_scalar_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t num_pixels);

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench_gemm(void) {
    printf("📊 --- GEMM Benchmark (512x512 Single-Precision Float) ---\n");
    const size_t N = 512;
    const size_t num_runs = 5;
    const double gflops_per_run = (2.0 * (double)N * (double)N * (double)N) * 1e-9;

    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    for (size_t i = 0; i < N * N; ++i) {
        A[i] = (float)((i % 13) - 6) * 0.1f;
        B[i] = (float)((i % 17) - 8) * 0.1f;
        C[i] = 0.0f;
    }

    /* 1. Scalar Reference */
    double t0 = get_time_sec();
    for (size_t r = 0; r < num_runs; ++r) {
        rcre_scalar_gemm_f32(N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }
    double t_scalar = (get_time_sec() - t0) / num_runs;
    double gflops_scalar = gflops_per_run / t_scalar;
    printf("   • Scalar Reference:   %7.2f ms | %6.2f GFLOPS\n", t_scalar * 1000.0, gflops_scalar);

    /* 2. Active HAL Backend */
    t0 = get_time_sec();
    for (size_t r = 0; r < num_runs; ++r) {
        rcre_hal.gemm_f32(N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }
    double t_simd = (get_time_sec() - t0) / num_runs;
    double gflops_simd = gflops_per_run / t_simd;
    double speedup = t_scalar / t_simd;
    printf("   • Active HAL (%s): %7.2f ms | %6.2f GFLOPS (Speedup: %.2fx)\n", 
           rcre_hal.backend_name, t_simd * 1000.0, gflops_simd, speedup);

    free(A);
    free(B);
    free(C);
}

static void bench_rgb_to_gray(void) {
    printf("📊 --- RGB to Grayscale Benchmark (1080p Video Frame 1920x1080) ---\n");
    const size_t W = 1920;
    const size_t H = 1080;
    const size_t num_pixels = W * H;
    const size_t num_runs = 50;

    uint8_t* rgb = (uint8_t*)malloc(num_pixels * 3);
    uint8_t* gray = (uint8_t*)malloc(num_pixels);

    for (size_t i = 0; i < num_pixels * 3; ++i) {
        rgb[i] = (uint8_t)(i * 33 + 7);
    }

    /* 1. Scalar Reference */
    double t0 = get_time_sec();
    for (size_t r = 0; r < num_runs; ++r) {
        rcre_scalar_rgb_to_gray_u8(rgb, gray, num_pixels);
    }
    double t_scalar = (get_time_sec() - t0) / num_runs;
    double mp_scalar = ((double)num_pixels * 1e-6) / t_scalar;
    printf("   • Scalar Reference:   %7.3f ms | %7.1f MP/s\n", t_scalar * 1000.0, mp_scalar);

    /* 2. Active HAL Backend */
    t0 = get_time_sec();
    for (size_t r = 0; r < num_runs; ++r) {
        rcre_hal.rgb_to_gray_u8(rgb, gray, num_pixels);
    }
    double t_simd = (get_time_sec() - t0) / num_runs;
    double mp_simd = ((double)num_pixels * 1e-6) / t_simd;
    double speedup = t_scalar / t_simd;
    printf("   • Active HAL (%s): %7.3f ms | %7.1f MP/s (Speedup: %.2fx)\n", 
           rcre_hal.backend_name, t_simd * 1000.0, mp_simd, speedup);

    free(rgb);
    free(gray);
}

int main(void) {
    rcre_hal_init();
    printf("======================================================================\n");
    printf("🚀 RCRE Hardware Acceleration & SIMD Micro-Kernel Benchmark\n");
    printf("   Host CPU: %s\n", rcre_cpu_get_brand());
    printf("   Active SIMD Architecture: %s\n", rcre_hal.backend_name);
    printf("======================================================================\n");

    bench_gemm();
    bench_rgb_to_gray();

    printf("======================================================================\n");
    printf("🏁 Benchmark Run Completed Successfully!\n");
    printf("======================================================================\n");
    return 0;
}
