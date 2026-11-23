/**
 * @file bench_h2.c
 * @brief H2 throughput bars: blocked GEMM, general-size FFT, triangular solve.
 *
 * Honest scope: these are single-process CPU bars on this machine. The
 * GEMM comparison is unblocked (H0 HAL dispatch) vs blocked (H2) — both
 * RCRE; the FFT has no in-library reference, so the bar is absolute
 * (Mpts/s = n log2 n flops-normalized points per second) plus a
 * round-trip sanity check.
 */

#include "rcre.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_rand(double* x, size_t n, uint64_t seed) {
    for (size_t i = 0; i < n; ++i) {
        uint64_t z = seed + i * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        x[i] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
    }
}

int main(void) {
    rcre_hal_init();
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  RCRE H2 — Blocked GEMM / General FFT / TriSolve Bars       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    /* ---- GEMM: unblocked vs blocked ------------------------------------ */
    {
        size_t sizes[] = { 512, 1024, 2048 };
        for (size_t si = 0; si < 3; ++si) {
            const size_t n = sizes[si];
            double* A = (double*)malloc(n * n * sizeof(double));
            double* B = (double*)malloc(n * n * sizeof(double));
            double* C1 = (double*)malloc(n * n * sizeof(double));
            double* C2 = (double*)malloc(n * n * sizeof(double));
            fill_rand(A, n * n, 0xA1);
            fill_rand(B, n * n, 0xB2);
            /* warm */
            rcre_linalg_gemm_f64(n, n, n, 1.0, A, n, B, n, 0.0, C1, n);
            rcre_linalg_gemm_blocked_f64(n, n, n, 1.0, A, n, B, n, 0.0, C2, n);
            const double tu0 = now_sec();
            rcre_linalg_gemm_f64(n, n, n, 1.0, A, n, B, n, 0.0, C1, n);
            const double tu1 = now_sec();
            const double tb0 = now_sec();
            rcre_linalg_gemm_blocked_f64(n, n, n, 1.0, A, n, B, n, 0.0, C2, n);
            const double tb1 = now_sec();
            const double fl = 2.0 * (double)n * (double)n * (double)n;
            printf("  GEMM %4zu^3  unblocked %7.1f GF/s | blocked %7.1f GF/s  (%.2fx)\n",
                   n, fl / (tu1 - tu0) / 1e9, fl / (tb1 - tb0) / 1e9,
                   (tu1 - tu0) / (tb1 - tb0));
            free(A); free(B); free(C1); free(C2);
        }
    }

    /* ---- FFT: general sizes -------------------------------------------- */
    {
        struct { size_t n; const char* tag; } cases[] = {
            { 1024, "2^10" }, { 4096, "2^12" }, { 65536, "2^16" },
            { 262144, "2^18" }, { 1048576, "2^20" },
            { 1000000, "1e6 (2^6*5^6)" }, { 1048573, "2^20+1 (Bluestein)" },
            { 1000003, "prime (Bluestein)" }, { 831400, "smooth+11 (Bluestein tail)" },
        };
        for (size_t ci = 0; ci < 9; ++ci) {
            const size_t n = cases[ci].n;
            double* re = (double*)malloc(n * sizeof(double));
            double* im = (double*)malloc(n * sizeof(double));
            double* orre = (double*)malloc(n * sizeof(double));
            double* orim = (double*)malloc(n * sizeof(double));
            fill_rand(re, n, 0xF1);
            fill_rand(im, n, 0xF2);
            if (rcre_fft_1d_f64(re, im, orre, orim, n, 0) != 0) return 1;
            const double f0 = now_sec();
            if (rcre_fft_1d_f64(re, im, orre, orim, n, 0) != 0) return 1;
            const double f1 = now_sec();
            const double logn = log2((double)n);
            printf("  FFT %-9zu %-18s %8.1f Ms/s  (%.2f Mflop/s, ~5nlog n/pt)\n",
                   n, cases[ci].tag, (double)n / (f1 - f0) / 1e6,
                   5.0 * (double)n * logn / (f1 - f0) / 1e6);
            free(re); free(im); free(orre); free(orim);
        }
    }

    /* ---- Triangular solve ---------------------------------------------- */
    {
        const size_t ns[] = { 512, 2048, 8192 };
        for (size_t ti = 0; ti < 3; ++ti) {
            const size_t n = ns[ti];
            const size_t nrhs = 8;
            double* A = (double*)malloc(n * n * sizeof(double));
            double* X = (double*)malloc(n * nrhs * sizeof(double));
            fill_rand(A, n * n, 0x0C);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) A[i * n + j] = 0.0;
                A[i * n + i] += 2.0;
            }
            fill_rand(X, n * nrhs, 0x0D);
            const int rc_warm = rcre_linalg_trsm_f64(1, 0, 1, 1.0, A, n, X, nrhs, n, nrhs);
            (void)rc_warm;
            double t0 = now_sec();
            const int iters = n >= 8192 ? 20 : 100;
            for (int it = 0; it < iters; ++it) {
                const int rc = rcre_linalg_trsm_f64(1, 0, 1, 1.0, A, n, X, nrhs, n, nrhs);
                if (rc != 0) { fprintf(stderr, "trsm failed\n"); return 1; }
            }
            double t1 = now_sec();
            const double fl = (double)iters * (double)nrhs * (double)n * (double)n; /* ~n^2 per col */
            printf("  TRSM n=%-5zu nrhs=%zu  %7.1f MF/s (half the ~n^2 work counted)\n",
                   n, nrhs, fl / 2.0 / (t1 - t0) / 1e6);
            free(A); free(X);
        }
    }

    return 0;
}
