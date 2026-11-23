/**
 * @file bench_linalg.c
 * @brief H0 (2.1.0) numerical micro-benchmark: the F64 substrate.
 *
 * Reports dispatched (ISA) vs scalar timings for the HPC op matrix, plus
 * the compensated-reduction comparison at a size where naive accumulation
 * measurably drifts (n = 2^20, alternating sign-magnitude pairs). The
 * per-release bar (docs/benchmarks.md) is:
 *   - dgemm  >= 80% of OpenBLAS on the same core (H2 will close the gap;
 *     H0 reports the blocked-scalar baseline);
 *   - ddot/axpy/scale: vector/scalar speedup >= 3x (memory-bound ops);
 *   - Neumaier: accurate to floor at n = 2^20 (naive drifts ~1e-11).
 */

#include "rcre.h"
#include "hal_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench_gemm(void) {
    printf("\n📊 --- GEMM 512x512x512 (F32 vs F64) ---\n");
    const size_t N = 512, runs = 5;
    float *Af = malloc(N * N * sizeof(float)), *Bf = malloc(N * N * sizeof(float)), *Cf = malloc(N * N * sizeof(float));
    double *Ad = malloc(N * N * sizeof(double)), *Bd = malloc(N * N * sizeof(double)), *Cd = malloc(N * N * sizeof(double));
    for (size_t i = 0; i < N * N; ++i) {
        Af[i] = (float)((i % 13) - 6) * 0.1f;
        Bf[i] = (float)((i % 17) - 8) * 0.1f;
        Ad[i] = (double)((i % 13) - 6) * 0.1;
        Bd[i] = (double)((i % 17) - 8) * 0.1;
    }
    const double gflops = 2.0 * N * N * N * 1e-9;

    double t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_hal.gemm_f32(N, N, N, 1.0f, Af, N, Bf, N, 0.0f, Cf, N);
    printf("   • F32 dispatched (%-24s): %7.2f ms | %6.1f GFLOP/s\n",
           rcre_hal.backend_name, (get_time_sec() - t0) / runs * 1000.0,
           gflops / ((get_time_sec() - t0) / runs));

    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_scalar_gemm_f32(N, N, N, 1.0f, Af, N, Bf, N, 0.0f, Cf, N);
    double t_f32s = (get_time_sec() - t0) / runs;
    printf("   • F32 scalar reference:         %7.2f ms | %6.1f GFLOP/s\n",
           t_f32s * 1000.0, gflops / t_f32s);

    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_hal.gemm_f64(N, N, N, 1.0, Ad, N, Bd, N, 0.0, Cd, N);
    double t_f64v = (get_time_sec() - t0) / runs;
    printf("   • F64 dispatched (%-24s): %7.2f ms | %6.1f GFLOP/s\n",
           rcre_hal.backend_name, t_f64v * 1000.0, gflops / t_f64v);

    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_scalar_gemm_f64(N, N, N, 1.0, Ad, N, Bd, N, 0.0, Cd, N);
    double t_f64s = (get_time_sec() - t0) / runs;
    printf("   • F64 scalar reference:         %7.2f ms | %6.1f GFLOP/s\n",
           t_f64s * 1000.0, gflops / t_f64s);
    printf("   • F64 vector/scalar speedup:    %6.2fx\n", t_f64s / t_f64v);

    free(Af); free(Bf); free(Cf); free(Ad); free(Bd); free(Cd);
}

static void bench_blas1(void) {
    printf("\n\xF0\x9F\x93\x8A --- Level-1 2^20 F64 (GB/s, vector vs scalar) ---\n");
    const size_t n = 1 << 20, runs = 5;
    double *x = malloc(n * sizeof(double)), *y = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) { x[i] = (double)(i % 100) * 0.01; y[i] = (double)(i % 53) * 0.02; }

    double t0, t1v, t1s, acc = 0.0;

    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) acc = rcre_hal.dot_f64(x, y, n);
    t1v = (get_time_sec() - t0) / runs;
    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) acc = rcre_scalar_dot_f64(x, y, n);
    t1s = (get_time_sec() - t0) / runs;
    printf("   \xE2\x80\xA2 dot   vector: %7.3f ms | %6.2f GB/s\n", t1v * 1000.0, 2.0 * n * 8 / 1e9 / t1v);
    printf("   \xE2\x80\xA2 dot   scalar: %7.3f ms | %6.2f GB/s\n", t1s * 1000.0, 2.0 * n * 8 / 1e9 / t1s);
    (void)acc;

    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_hal.axpy_f64(n, 1.5, x, y);
    t1v = (get_time_sec() - t0) / runs;
    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_scalar_axpy_f64(n, 1.5, x, y);
    t1s = (get_time_sec() - t0) / runs;
    printf("   \xE2\x80\xA2 axpy  vector: %7.3f ms | %6.2f GB/s\n", t1v * 1000.0, 3.0 * n * 8 / 1e9 / t1v);
    printf("   \xE2\x80\xA2 axpy  scalar: %7.3f ms | %6.2f GB/s\n", t1s * 1000.0, 3.0 * n * 8 / 1e9 / t1s);

    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_hal.sum_f64(x, n);
    t1v = (get_time_sec() - t0) / runs;
    t0 = get_time_sec();
    for (size_t r = 0; r < runs; ++r) rcre_scalar_sum_f64(x, n);
    t1s = (get_time_sec() - t0) / runs;
    printf("   \xE2\x80\xA2 sum   vector: %7.3f ms | %6.2f GB/s\n", t1v * 1000.0, 1.0 * n * 8 / 1e9 / t1v);
    printf("   \xE2\x80\xA2 sum   scalar: %7.3f ms | %6.2f GB/s\n", t1s * 1000.0, 1.0 * n * 8 / 1e9 / t1s);

    free(x); free(y);
}

static void bench_reductions(void) {
    printf("\n📊 --- Reduction accuracy at n = 2^20 (naive drift vs compensated floor) ---\n");
    const size_t n = 1 << 20;
    double *x = malloc(n * sizeof(double));
    /* adversarial: x[0] = 1e20, rest 1.0. ulp(1e20) = 1024, so every
     * naive `sum += 1.0` is silently lost (1 < ulp/2); the true sum is
     * 1e20 + (n-1). Neumaier's compensation stream collects the lost
     * unit exactly; pairwise blocks of 1s sum exactly and only the
     * final merge with 1e20 rounds. Demonstrates: naive err ~ n,
     * compensated err ~ 1. */
    x[0] = 1e20;
    for (size_t i = 1; i < n; ++i) x[i] = 1.0;
    const double exact = 1e20 + (double)(n - 1);

    double naive = rcre_scalar_sum_f64(x, n);
    double neu   = rcre_sum_neumaier_f64(x, n);
    double pair  = rcre_sum_pairwise_f64(x, n);
    printf("   • exact:            %.17g\n", exact);
    printf("   • naive:            %.17g   (|err| = %.3e)\n", naive, fabs(naive - exact));
    printf("   • neumaier:         %.17g   (|err| = %.3e)\n", neu, fabs(neu - exact));
    printf("   • pairwise:         %.17g   (|err| = %.3e)\n", pair, fabs(pair - exact));
    free(x);
}

int main(void) {
    rcre_hal_init();
    printf("RCRE H0 numerical micro-benchmark — backend: %s (features 0x%x)\n",
           rcre_hal.backend_name, rcre_hal.cpu_features);
    bench_gemm();
    bench_blas1();
    bench_reductions();
    return 0;
}
