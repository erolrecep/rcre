/**
 * @file bench_roofline.c
 * @brief RCRE Empirical Roofline Benchmark (GFLOP/s & Memory Bandwidth GB/s)
 */

#include "rcre.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double rf_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    rcre_hal_init();
    RCRE_CpuTopology topo = rcre_cpu_get_topology();

    printf("======================================================================\n");
    printf("📈 RCRE Roofline Benchmark (GFLOP/s, I8 GOP/s, Memory-Bound GEMV)\n");
    printf("   CPU: %s | Backend: %s | Cores: %u (%uP + %uE)\n",
           rcre_cpu_get_brand(), rcre_hal.backend_name,
           topo.logical_cores, topo.perf_cores, topo.eff_cores);
    printf("======================================================================\n");

    RCRE_Arena* arena = rcre_arena_create(128 * 1024 * 1024);
    RCRE_ThreadPool* pool = rcre_threadpool_create_ex(0, RCRE_QOS_PERF);
    if (!arena || !pool) return 1;

    /* 1. Compute-bound SGEMM (512 x 512) Single-Thread vs Multi-Thread */
    {
        const size_t N = 512;
        float* A = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
        float* B = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
        float* C = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
        for (size_t i = 0; i < N * N; ++i) {
            A[i] = (float)((i % 17) - 8) * 0.01f;
            B[i] = (float)((i % 13) - 6) * 0.01f;
        }
        double flops = 2.0 * (double)N * (double)N * (double)N;

        rcre_hal.gemm_f32(N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        double t0 = rf_now_sec();
        const int ITERS = 10;
        for (int i = 0; i < ITERS; ++i) rcre_hal.gemm_f32(N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        double dt_st = (rf_now_sec() - t0) / ITERS;

        rcre_hal_gemm_f32_mt(pool, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        t0 = rf_now_sec();
        for (int i = 0; i < ITERS; ++i) rcre_hal_gemm_f32_mt(pool, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        double dt_mt = (rf_now_sec() - t0) / ITERS;

        printf("  • SGEMM 512x512 (1T):  %6.2f ms | %6.2f GFLOP/s\n", dt_st * 1e3, (flops * 1e-9) / dt_st);
        printf("  • SGEMM 512x512 (MT):  %6.2f ms | %6.2f GFLOP/s (Speedup: %.2fx)\n",
               dt_mt * 1e3, (flops * 1e-9) / dt_mt, dt_st / dt_mt);
    }

    /* 2. Memory-bound GEMV (4096 x 4096 F32) */
    {
        rcre_arena_reset(arena);
        const size_t M = 4096, K = 4096;
        float* A = (float*)rcre_arena_alloc(arena, M * K * sizeof(float), 64);
        float* x = (float*)rcre_arena_alloc(arena, K * sizeof(float), 64);
        float* y = (float*)rcre_arena_alloc(arena, M * sizeof(float), 64);
        for (size_t i = 0; i < M * K; ++i) A[i] = 0.001f;
        for (size_t i = 0; i < K; ++i) x[i] = 1.0f;

        rcre_hal.gemv_f32(M, K, 1.0f, A, K, x, 0.0f, y);
        double t0 = rf_now_sec();
        const int ITERS = 20;
        for (int i = 0; i < ITERS; ++i) rcre_hal.gemv_f32(M, K, 1.0f, A, K, x, 0.0f, y);
        double dt = (rf_now_sec() - t0) / ITERS;
        double bytes = (double)(M * K + K + M) * sizeof(float);
        double flops = 2.0 * (double)M * (double)K;
        printf("  • GEMV  4096x4096:     %6.2f ms | %6.2f GB/s   | %6.2f GFLOP/s\n",
               dt * 1e3, (bytes * 1e-9) / dt, (flops * 1e-9) / dt);
    }

    /* 3. INT8 GEMM (512 x 512) */
    {
        rcre_arena_reset(arena);
        const size_t N = 512;
        int8_t* A = (int8_t*)rcre_arena_alloc(arena, N * N, 64);
        int8_t* B = (int8_t*)rcre_arena_alloc(arena, N * N, 64);
        int32_t* C = (int32_t*)rcre_arena_alloc(arena, N * N * sizeof(int32_t), 64);
        for (size_t i = 0; i < N * N; ++i) {
            A[i] = (int8_t)((i % 15) - 7);
            B[i] = (int8_t)((i % 11) - 5);
        }
        double ops = 2.0 * (double)N * (double)N * (double)N;
        double t0 = rf_now_sec();
        const int ITERS = 10;
        for (int i = 0; i < ITERS; ++i) rcre_hal.gemm_i8(N, N, N, A, N, B, N, C, N);
        double dt = (rf_now_sec() - t0) / ITERS;
        printf("  • IGEMM 512x512 (I8):  %6.2f ms | %6.2f GIOP/s\n", dt * 1e3, (ops * 1e-9) / dt);
    }

    /* 4. BF16 GEMM (512 x 512) */
    {
        rcre_arena_reset(arena);
        const size_t N = 512;
        uint16_t* A = (uint16_t*)rcre_arena_alloc(arena, N * N * sizeof(uint16_t), 64);
        uint16_t* B = (uint16_t*)rcre_arena_alloc(arena, N * N * sizeof(uint16_t), 64);
        float* C = (float*)rcre_arena_alloc(arena, N * N * sizeof(float), 64);
        for (size_t i = 0; i < N * N; ++i) {
            A[i] = 0x3F80U; /* 1.0 in BF16 */
            B[i] = 0x3F00U; /* 0.5 in BF16 */
        }
        double flops = 2.0 * (double)N * (double)N * (double)N;
        double t0 = rf_now_sec();
        const int ITERS = 10;
        for (int i = 0; i < ITERS; ++i) rcre_hal.gemm_bf16(N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        double dt = (rf_now_sec() - t0) / ITERS;
        printf("  • BGEMM 512x512 (BF16):%6.2f ms | %6.2f GFLOP/s\n", dt * 1e3, (flops * 1e-9) / dt);
    }

    rcre_threadpool_destroy(pool);
    rcre_arena_destroy(arena);
    printf("======================================================================\n");
    return 0;
}
