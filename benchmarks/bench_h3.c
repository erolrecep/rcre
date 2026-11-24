/* RCRE H3 benchmark — direct factorizations (2.2.0).
 *
 * Bars: LU, Cholesky, QR (factors + Q), syev (values / values+vec)
 * on F64 dense matrices, single-thread, -O3, -ffp-contract=off.
 *
 * Build: make bench_h3   Run: make bench_h3 (runs after build)
 */
#include "rcre.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double* mat(size_t n) {
    double* A = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            uint64_t z = 99 + (i * n + j) * 0x9E3779B97F4A7C15ull;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            A[i * n + j] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
        }
    return A;
}

/* SPD: A = M^T M + n*I */
static double* mat_spd(size_t n) {
    double* A = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            uint64_t z = 1234 + (i * n + j) * 0x9E3779B97F4A7C15ull;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            A[i * n + j] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
        }
    double* At = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            At[i * n + j] = A[j * n + i];
    rcre_linalg_gemm_f64(n, n, n, 1.0, At, n, A, n, 0.0, A, n);
    for (size_t i = 0; i < n; ++i) A[i * n + i] += (double)n;
    free(At);
    return A;
}

#define TIMEIT(label, n, flops, ...)                                         \
    do {                                                                     \
        double t0 = now_s();                                                 \
        __VA_ARGS__;                                                         \
        double t1 = now_s();                                                 \
        const double dt = t1 - t0;                                           \
        printf("  %-28s n=%-5zu %8.1f GFLOP/s ( %7.3f s)\n",                 \
               (label), (n), (flops) / dt / 1e9, dt);                        \
    } while (0)

int main(void) {
    rcre_hal_init();
    printf("=====================================================================\n");
    printf("📊 RCRE H3 — Direct Factorization Benchmark (F64)\n");
    printf("=====================================================================\n");
    /* lu/cholesky/syev scale to 4096; the unblocked 3-term Householder
     * QR update is memory-thrashing (documented), so QR bars stop at
     * 2048 to keep the bench under ~2 minutes. */
    const size_t all_sizes[4] = { 512, 1024, 2048, 4096 };
    const size_t qr_sizes[3]  = { 512, 1024, 2048 };
    for (int s = 0; s < 4; ++s) {
        const size_t n = all_sizes[s];
        const double fl = 2.0 / 3.0 * (double)n * n * n;
        double* A = mat(n);
        size_t* piv = (size_t*)malloc(n * sizeof(size_t));
        TIMEIT("lu_factor", n, fl, {
            (void)rcre_linalg_lu_factor_f64(A, n, n, piv); });
        free(A); free(piv);
        A = mat_spd(n);
        TIMEIT("cholesky", n, fl / 2.0, {
            (void)rcre_linalg_cholesky_f64(A, n, n, 1); });
        free(A);
    }
    for (int s = 0; s < 3; ++s) {
        const size_t n = qr_sizes[s];
        const double fl = 2.0 / 3.0 * (double)n * n * n;
        double* A = mat(n);
        double* tau = (double*)malloc(n * sizeof(double));
        double* Q = (double*)malloc(n * n * sizeof(double));
        TIMEIT("qr_factor", n, fl, {
            (void)rcre_linalg_qr_factor_f64(A, n, n, n, tau); });
        TIMEIT("qr_q_explicit", n, 4.0 * (double)n * n * n, {
            (void)rcre_linalg_qr_q_f64(A, n, n, n, tau, Q, n); });
        free(A); free(tau); free(Q);
    }
    for (int s = 0; s < 4; ++s) {
        const size_t n = all_sizes[s];
        const double fl = 2.0 / 3.0 * (double)n * n * n;
        double* A = mat_spd(n);
        double* w = (double*)malloc(n * sizeof(double));
        double* V = (double*)malloc(n * n * sizeof(double));
        TIMEIT("syev_values", n, fl, {
            (void)rcre_linalg_syev_f64(A, n, n, 0, w, NULL, 0); });
        free(A);
        A = mat_spd(n);
        TIMEIT("syev_values+vec", n, fl, {
            (void)rcre_linalg_syev_f64(A, n, n, 1, w, V, n); });
        free(A); free(w); free(V);
    }
    printf("---------------------------------------------------------------------\n");
    printf("Honest context: single-thread, no cache blocking inside the\n");
    printf("unblocked kernels; LAPACK (OpenBLAS) bars are the reference.\n");
    return 0;
}
