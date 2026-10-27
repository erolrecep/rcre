/**
 * @file bench_simd.c
 * @brief Benchmark 03: SIMD Vectorization & Arithmetic Intensity (HAL vs Scalar)
 */

#include "rcre_perf.h"

/* Scalar reference routines for A/B comparison */
static void scalar_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

static float scalar_dot_f32(const float* x, const float* y, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

static void scalar_gemm_f32(size_t M, size_t N, size_t K, float alpha,
                            const float* A, size_t lda,
                            const float* B, size_t ldb,
                            float beta, float* C, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

void rcre_bench_simd_vectorization(const RCRE_PerfConfig* cfg, RCRE_ProfContext* prof, RCRE_Arena* arena) {
    (void)cfg;
    const size_t vec_len = 8 * 1024 * 1024; /* 8M elements = 32 MB per vector */
    const size_t gemm_k = 512;               /* 512x512 GEMM */

    printf("\n===================================================================\n");
    printf(" 📊 [Benchmark 03] SIMD Vectorization & Arithmetic Intensity       \n");
    printf("===================================================================\n");
    printf(" CPU Brand Detected     : %s\n", rcre_cpu_get_brand());
    printf(" Active HAL Backend     : %s (Features: 0x%04x)\n", 
           rcre_hal.backend_name, rcre_cpu_get_features());
    printf(" Workload 1 & 2         : 8 Million Elements (32.00 MB / vector)\n");
    printf(" Workload 3 (GEMM)      : 512 x 512 Matrix Multiplication\n");
    printf("-------------------------------------------------------------------\n");

    RCRE_ArenaTemp checkpoint = rcre_arena_temp_begin(arena);

    /* Allocate vectors */
    float* x = (float*)rcre_arena_alloc(arena, vec_len * sizeof(float), 64);
    float* y = (float*)rcre_arena_alloc(arena, vec_len * sizeof(float), 64);
    float* y_ref = (float*)rcre_arena_alloc(arena, vec_len * sizeof(float), 64);
    if (!x || !y || !y_ref) {
        fprintf(stderr, "Error: Arena insufficient for 32MB vector allocations.\n");
        return;
    }

    for (size_t i = 0; i < vec_len; ++i) {
        x[i] = (float)(i % 17) * 0.1f;
        y[i] = (float)(i % 23) * 0.1f;
        y_ref[i] = y[i];
    }

    /* --------------------------------------------------------------------- */
    /* 1. AXPY Benchmark: y = alpha * x + y                                  */
    /* --------------------------------------------------------------------- */
    const float alpha = 2.5f;

    /* Scalar reference */
    uint64_t dur_axpy_scalar = 0;
    {
        RCRE_PROFILE_SCOPE(prof, "simd_axpy_scalar", z_axpy_s);
        scalar_axpy_f32(vec_len, alpha, x, y_ref);
        dur_axpy_scalar = RCRE_PROFILE_END(z_axpy_s);
    }
    double t_axpy_scalar_sec = (double)dur_axpy_scalar * 1e-9;
    double bw_axpy_scalar = (3.0 * (double)vec_len * sizeof(float) / 1e9) / t_axpy_scalar_sec;

    /* RCRE HAL SIMD dispatch */
    uint64_t dur_axpy_simd = 0;
    {
        RCRE_PROFILE_SCOPE(prof, "simd_axpy_hal", z_axpy_h);
        rcre_hal.axpy_f32(vec_len, alpha, x, y);
        dur_axpy_simd = RCRE_PROFILE_END(z_axpy_h);
    }
    double t_axpy_simd_sec = (double)dur_axpy_simd * 1e-9;
    double bw_axpy_simd = (3.0 * (double)vec_len * sizeof(float) / 1e9) / t_axpy_simd_sec;
    double speedup_axpy = t_axpy_scalar_sec / (t_axpy_simd_sec > 1e-9 ? t_axpy_simd_sec : 1e-9);

    printf("\n [1] Vector AXPY (y = alpha * x + y, 8M floats):\n");
    printf("     Scalar Reference : %7.2f ms | %6.2f GB/s\n", t_axpy_scalar_sec * 1000.0, bw_axpy_scalar);
    printf("     Active HAL SIMD  : %7.2f ms | %6.2f GB/s (Speedup: %.2fx)\n", 
           t_axpy_simd_sec * 1000.0, bw_axpy_simd, speedup_axpy);

    /* --------------------------------------------------------------------- */
    /* 2. Dot Product: sum = x . y                                           */
    /* --------------------------------------------------------------------- */
    float dot_scalar = 0.0f;
    uint64_t dur_dot_scalar = 0;
    {
        RCRE_PROFILE_SCOPE(prof, "simd_dot_scalar", z_dot_s);
        dot_scalar = scalar_dot_f32(x, y, vec_len);
        dur_dot_scalar = RCRE_PROFILE_END(z_dot_s);
    }
    rcre_perf_do_not_optimize(&dot_scalar);
    double t_dot_scalar_sec = (double)dur_dot_scalar * 1e-9;
    double bw_dot_scalar = (2.0 * (double)vec_len * sizeof(float) / 1e9) / t_dot_scalar_sec;

    float dot_simd = 0.0f;
    uint64_t dur_dot_simd = 0;
    {
        RCRE_PROFILE_SCOPE(prof, "simd_dot_hal", z_dot_h);
        dot_simd = rcre_hal.dot_f32(x, y, vec_len);
        dur_dot_simd = RCRE_PROFILE_END(z_dot_h);
    }
    rcre_perf_do_not_optimize(&dot_simd);
    double t_dot_simd_sec = (double)dur_dot_simd * 1e-9;
    double bw_dot_simd = (2.0 * (double)vec_len * sizeof(float) / 1e9) / t_dot_simd_sec;
    double speedup_dot = t_dot_scalar_sec / (t_dot_simd_sec > 1e-9 ? t_dot_simd_sec : 1e-9);

    printf("\n [2] Vector Dot Product (x . y, 8M floats):\n");
    printf("     Scalar Reference : %7.2f ms | %6.2f GB/s | Checksum: %.2f\n", 
           t_dot_scalar_sec * 1000.0, bw_dot_scalar, dot_scalar);
    printf("     Active HAL SIMD  : %7.2f ms | %6.2f GB/s (Speedup: %.2fx) | Checksum: %.2f\n", 
           t_dot_simd_sec * 1000.0, bw_dot_simd, speedup_dot, dot_simd);

    /* --------------------------------------------------------------------- */
    /* 3. GEMM Benchmark: C = alpha * A * B + beta * C (512x512)            */
    /* --------------------------------------------------------------------- */
    float* A = (float*)rcre_arena_alloc(arena, gemm_k * gemm_k * sizeof(float), 64);
    float* B = (float*)rcre_arena_alloc(arena, gemm_k * gemm_k * sizeof(float), 64);
    float* C = (float*)rcre_arena_alloc(arena, gemm_k * gemm_k * sizeof(float), 64);
    if (!A || !B || !C) {
        fprintf(stderr, "Error: Arena insufficient for GEMM allocations.\n");
        return;
    }

    for (size_t i = 0; i < gemm_k * gemm_k; ++i) {
        A[i] = (float)((i % 13) - 6) * 0.1f;
        B[i] = (float)((i % 17) - 8) * 0.1f;
        C[i] = 0.0f;
    }

    const double gflops_gemm = (2.0 * (double)gemm_k * (double)gemm_k * (double)gemm_k) * 1e-9;

    /* Scalar Reference */
    uint64_t dur_gemm_scalar = 0;
    {
        RCRE_PROFILE_SCOPE(prof, "simd_gemm_scalar", z_gemm_s);
        scalar_gemm_f32(gemm_k, gemm_k, gemm_k, 1.0f, A, gemm_k, B, gemm_k, 0.0f, C, gemm_k);
        dur_gemm_scalar = RCRE_PROFILE_END(z_gemm_s);
    }
    double t_gemm_scalar_sec = (double)dur_gemm_scalar * 1e-9;
    double gflops_scalar = gflops_gemm / t_gemm_scalar_sec;

    /* RCRE HAL SIMD GEMM */
    uint64_t dur_gemm_simd = 0;
    {
        RCRE_PROFILE_SCOPE(prof, "simd_gemm_hal", z_gemm_h);
        rcre_hal.gemm_f32(gemm_k, gemm_k, gemm_k, 1.0f, A, gemm_k, B, gemm_k, 0.0f, C, gemm_k);
        dur_gemm_simd = RCRE_PROFILE_END(z_gemm_h);
    }
    double t_gemm_simd_sec = (double)dur_gemm_simd * 1e-9;
    double gflops_simd = gflops_gemm / t_gemm_simd_sec;
    double speedup_gemm = t_gemm_scalar_sec / (t_gemm_simd_sec > 1e-9 ? t_gemm_simd_sec : 1e-9);

    printf("\n [3] GEMM Floating Point Throughput (512x512 Single-Precision):\n");
    printf("     Scalar Reference : %7.2f ms | %6.2f GFLOPS\n", t_gemm_scalar_sec * 1000.0, gflops_scalar);
    printf("     Active HAL SIMD  : %7.2f ms | %6.2f GFLOPS (Speedup: %.2fx)\n", 
           t_gemm_simd_sec * 1000.0, gflops_simd, speedup_gemm);

    printf("\n Architectural Synthesis:\n");
    printf("   • Compute-Bound Acceleration : GEMM achieves %.2fx higher compute throughput\n", speedup_gemm);
    printf("     via hardware register tiling and fused multiply-accumulate vector instructions.\n");

    if (prof) {
        rcre_prof_mark(prof, "GEMM_SIMD_GFLOPS", gflops_simd);
        rcre_prof_mark(prof, "GEMM_SCALAR_GFLOPS", gflops_scalar);
    }

    rcre_arena_temp_end(checkpoint);
}
