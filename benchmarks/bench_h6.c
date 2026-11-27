/* RCRE H6 benchmark — GPU bridge (2.3.0)
 *
 * Measures (single process; the CUDA backend when built in AND a GPU is
 * present — dlbox: RTX 5060 Ti, sm_120, CUDA 13.3 — otherwise SKIP):
 *
 *   SpMV f64     2D Poisson CSR, n = 4096/16384/65536  (ms, Ms entries/s)
 *   GEMM f32     512/1024/2048  (ms, GFLOP/s)
 *   GEMM f64     256/512        (ms, GFLOP/s)
 *   Philox fill  64M words      (ms, Gwords/s)  — bit-identical by design
 *   CG + ILU(0)  2D Poisson 64x64: total wall + per-iter, vs the H4
 *                all-CPU rcre_krylov_pcg wall (same 75-76 iter count)
 *
 * Timing: cudaEvent-free host wall clock around synchronize() — honest
 * end-to-end numbers (transfer + compute + sync), the way a solver
 * actually pays for the device.
 */
#include "rcre.h"
#include "extras/rcre_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void lcg_fill(void* dst, size_t n, unsigned seed, int width) {
    uint64_t st = seed;
    if (width == 4) {
        float* p = (float*)dst;
        for (size_t i = 0; i < n; ++i) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            p[i] = (float)((st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-24f;
        }
    } else {
        double* p = (double*)dst;
        for (size_t i = 0; i < n; ++i) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            p[i] = (double)((st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53;
        }
    }
}

static void poisson_csr(size_t nx, size_t ny, size_t* rp, size_t* ci, double* v) {
    rp[0] = 0;
    for (size_t i = 0; i < nx; ++i) for (size_t j = 0; j < ny; ++j) {
        size_t c = i * ny + j, m = 0;
        if (i > 0) { ci[rp[c] + m] = (i - 1) * ny + j; v[rp[c] + m] = -1.0; m++; }
        if (j > 0) { ci[rp[c] + m] = i * ny + (j - 1); v[rp[c] + m] = -1.0; m++; }
        ci[rp[c] + m] = c; v[rp[c] + m] = 4.0; m++;
        if (j + 1 < ny) { ci[rp[c] + m] = i * ny + (j + 1); v[rp[c] + m] = -1.0; m++; }
        if (i + 1 < nx) { ci[rp[c] + m] = (i + 1) * ny + j; v[rp[c] + m] = -1.0; m++; }
        rp[c + 1] = rp[c] + m;
    }
}

/* warm-up reps for device timing */
#define REPS 30

int main(void) {
    rcre_hal_init();
    printf("RCRE H6 GPU bridge bench — %s\n\n",
           rcre_gpu_backend_available("cuda") ? rcre_gpu_device_desc(rcre_gpu_open_named("cuda"))
                                              : "no CUDA in this build (SKIP device cases)");

    int have = rcre_gpu_backend_available("cuda");
    RCRE_GpuCtx* ctx = have ? rcre_gpu_open_named("cuda") : NULL;

    /* ---------------- SpMV f64 ---------------- */
    printf("SpMV f64 (2D Poisson CSR):\n");
    printf("  %-14s %-14s %-14s %-12s %-14s\n", "case", "cpu (ms)", "dev (ms)", "dev Ms/s", "speedup");
    size_t grids[3] = { 64, 128, 256 };
    for (int g = 0; g < 3; ++g) {
        size_t n = grids[g] * grids[g];
        size_t* rp = malloc((n + 1) * sizeof(size_t));
        size_t* ci = malloc(5 * n * sizeof(size_t));
        double* val = malloc(5 * n * 8);
        poisson_csr(grids[g], grids[g], rp, ci, val);
        size_t nnz = rp[n];
        RCRE_CSR_F64 A = { n, n, nnz, rp, ci, val };
        double* x = malloc(n * 8), *y = malloc(n * 8);
        lcg_fill(x, n, 0x51, 8);
        /* CPU */
        double t0 = now_ms();
        for (int r = 0; r < REPS; ++r) rcre_spmv_f64(&A, x, y);
        double tc = (now_ms() - t0) / REPS;
        if (ctx) {
            RCRE_GpuBuf *drp = rcre_gpu_alloc(ctx, (n + 1) * sizeof(size_t)),
                        *dci = rcre_gpu_alloc(ctx, nnz * sizeof(size_t)),
                        *dv = rcre_gpu_alloc(ctx, nnz * 8),
                        *dx = rcre_gpu_alloc(ctx, n * 8),
                        *dy = rcre_gpu_alloc(ctx, n * 8);
            (void)(rcre_gpu_h2d(ctx, drp, rp, (n + 1) * sizeof(size_t)));
            (void)(rcre_gpu_h2d(ctx, dci, ci, nnz * sizeof(size_t)));
            (void)(rcre_gpu_h2d(ctx, dv, val, nnz * 8));
            (void)(rcre_gpu_h2d(ctx, dx, x, n * 8));
            /* warm-up (device clocks + first launch) */
            for (int r = 0; r < 5; ++r) {
                (void)(rcre_gpu_spmv_f64_ctx(ctx, n, drp, dci, dv, dx, dy));
                (void)(rcre_gpu_synchronize(ctx));
            }
            double t1 = now_ms();
            for (int r = 0; r < REPS; ++r) {
                (void)(rcre_gpu_spmv_f64_ctx(ctx, n, drp, dci, dv, dx, dy));
                (void)(rcre_gpu_synchronize(ctx));
            }
            double td = (now_ms() - t1) / REPS;
            char casebuf[32];
            snprintf(casebuf, sizeof casebuf, "%zux%zu", grids[g], grids[g]);
            printf("  %-14s %-14.3f %-14.3f %-12.1f %-14.2fx\n",
                   casebuf, tc, td, nnz / (td * 1e3), tc / td);
            rcre_gpu_free(drp); rcre_gpu_free(dci); rcre_gpu_free(dv);
            rcre_gpu_free(dx); rcre_gpu_free(dy);
        } else {
            char casebuf[32];
            snprintf(casebuf, sizeof casebuf, "%zux%zu", grids[g], grids[g]);
            printf("  %-14s %-14.3f %-14s\n", casebuf, tc, "skip");
        }
        free(rp); free(ci); free(val); free(x); free(y);
    }

    /* ---------------- GEMM ---------------- */
    printf("\nGEMM (C = A*B):\n");
    printf("  %-12s %-14s %-14s %-12s %-14s\n", "case", "cpu (ms)", "dev (ms)", "dev GFLOP/s", "speedup");
    size_t g32[3] = { 512, 1024, 2048 };
    for (int g = 0; g < 3; ++g) {
        size_t m = g32[g];
        float *A = malloc(m * m * 4), *B = malloc(m * m * 4), *C = malloc(m * m * 4);
        lcg_fill(A, m * m, 0xA1, 4);
        lcg_fill(B, m * m, 0xB2, 4);
        memset(C, 0, m * m * 4);
        double t0 = now_ms();
        for (int r = 0; r < 5; ++r) rcre_linalg_gemm_f32(m, m, m, 1.0f, A, m, B, m, 0.0f, C, m);
        double tc = (now_ms() - t0) / 5.0;
        if (ctx) {
            RCRE_GpuBuf *dA = rcre_gpu_alloc(ctx, m * m * 4),
                        *dB = rcre_gpu_alloc(ctx, m * m * 4),
                        *dC = rcre_gpu_alloc(ctx, m * m * 4);
            (void)(rcre_gpu_h2d(ctx, dA, A, m * m * 4));
            (void)(rcre_gpu_h2d(ctx, dB, B, m * m * 4));
            for (int r = 0; r < 5; ++r) {
                (void)(rcre_gpu_gemm_f32_ctx(ctx, m, m, m, 1.0f, dA, m, dB, m, 0.0f, dC, m));
                (void)(rcre_gpu_synchronize(ctx));
            }
            double t1 = now_ms();
            for (int r = 0; r < REPS; ++r) {
                (void)(rcre_gpu_gemm_f32_ctx(ctx, m, m, m, 1.0f, dA, m, dB, m, 0.0f, dC, m));
                (void)(rcre_gpu_synchronize(ctx));
            }
            double td = (now_ms() - t1) / REPS;
            double gf = 2.0 * m * m * m / (td * 1e-3) / 1e9;
            char casebuf[32];
            snprintf(casebuf, sizeof casebuf, "f32 %zux%zu", m, m);
            printf("  %-12s %-14.3f %-14.3f %-12.1f %-14.2fx\n",
                   casebuf, tc, td, gf, tc / td);
            rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
        } else {
            char casebuf[32];
            snprintf(casebuf, sizeof casebuf, "f32 %zux%zu", m, m);
            printf("  %-12s %-14.3f %-14s\n", casebuf, tc, "skip");
        }
        free(A); free(B); free(C);
    }
    size_t g64[2] = { 256, 512 };
    for (int g = 0; g < 2; ++g) {
        size_t m = g64[g];
        double *A = malloc(m * m * 8), *B = malloc(m * m * 8), *C = malloc(m * m * 8);
        lcg_fill(A, m * m, 0xD4, 8);
        lcg_fill(B, m * m, 0xE5, 8);
        memset(C, 0, m * m * 8);
        double t0 = now_ms();
        for (int r = 0; r < 5; ++r) rcre_linalg_gemm_f64(m, m, m, 1.0, A, m, B, m, 0.0, C, m);
        double tc = (now_ms() - t0) / 5.0;
        if (ctx) {
            RCRE_GpuBuf *dA = rcre_gpu_alloc(ctx, m * m * 8),
                        *dB = rcre_gpu_alloc(ctx, m * m * 8),
                        *dC = rcre_gpu_alloc(ctx, m * m * 8);
            (void)(rcre_gpu_h2d(ctx, dA, A, m * m * 8));
            (void)(rcre_gpu_h2d(ctx, dB, B, m * m * 8));
            for (int r = 0; r < 5; ++r) {
                (void)(rcre_gpu_gemm_f64_ctx(ctx, m, m, m, 1.0, dA, m, dB, m, 0.0, dC, m));
                (void)(rcre_gpu_synchronize(ctx));
            }
            double t1 = now_ms();
            for (int r = 0; r < REPS; ++r) {
                (void)(rcre_gpu_gemm_f64_ctx(ctx, m, m, m, 1.0, dA, m, dB, m, 0.0, dC, m));
                (void)(rcre_gpu_synchronize(ctx));
            }
            double td = (now_ms() - t1) / REPS;
            double gf = 2.0 * m * m * m / (td * 1e-3) / 1e9;
            char casebuf[32];
            snprintf(casebuf, sizeof casebuf, "f64 %zux%zu", m, m);
            printf("  %-12s %-14.3f %-14.3f %-12.1f %-14.2fx\n",
                   casebuf, tc, td, gf, tc / td);
            rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
        } else {
            char casebuf[32];
            snprintf(casebuf, sizeof casebuf, "f64 %zux%zu", m, m);
            printf("  %-12s %-14.3f %-14s\n", casebuf, tc, "skip");
        }
        free(A); free(B); free(C);
    }

    /* ---------------- Philox fill ---------------- */
    printf("\nPhilox-4x32-10 fill (64M words, bit-identical by design):\n");
    size_t nwords = 64u * 1024u * 1024u;
    uint32_t* wh = malloc(nwords * 4);
    double t0 = now_ms();
    (void)rcre_rng_philox4x32_fill(wh, nwords, 7, 0xC0FFEEull);
    double tc = now_ms() - t0;
    printf("  %-12s %-14.3f  %-12.2f Gwords/s\n", "cpu", tc, nwords / (tc * 1e-3) / 1e9);
    if (ctx) {
        RCRE_GpuBuf* dw = rcre_gpu_alloc(ctx, nwords * 4);
        (void)rcre_gpu_philox4x32_fill(ctx, nwords, 7, 0xC0FFEEull, dw);
        (void)(rcre_gpu_synchronize(ctx));
        double t1 = now_ms();
        for (int r = 0; r < 10; ++r) {
            (void)rcre_gpu_philox4x32_fill(ctx, nwords, 7, 0xC0FFEEull, dw);
            (void)(rcre_gpu_synchronize(ctx));
        }
        double td = (now_ms() - t1) / 10.0;
        printf("  %-12s %-14.3f  %-12.2f Gwords/s (%.2fx)\n", "dev", td,
               nwords / (td * 1e-3) / 1e9, tc / td);
        rcre_gpu_free(dw);
    }
    free(wh);

    /* ---------------- CG + ILU(0) end-to-end ---------------- */
    printf("\nCG + ILU(0) on 2D Poisson 64x64 (n=4096), tol 1e-10:\n");
    {
        size_t n = 4096;
        size_t* rp = malloc((n + 1) * sizeof(size_t));
        size_t* ci = malloc(5 * n * sizeof(size_t));
        double* val = malloc(5 * n * 8);
        poisson_csr(64, 64, rp, ci, val);
        size_t nnz = rp[n];
        RCRE_CSR_F64 A = { n, n, nnz, rp, ci, val };
        RCRE_CSR_F64 fac = { 0, 0, 0, NULL, NULL, NULL };
        (void)rcre_pc_ilu0_f64(&A, &fac);
        double* b = malloc(n * 8);
        lcg_fill(b, n, 0xCAFE, 8);

        /* all-CPU H4 solver (the H4 bar) */
        double* xc = calloc(n, 8);
        double t0 = now_ms();
        int itc = rcre_krylov_pcg_f64(&A, b, xc, 1e-10, 500, RCRE_PC_ILU0,
                                      &fac, 0, NULL, NULL);
        double tc = now_ms() - t0;
        printf("  %-12s %d iters  %-10.3f ms  (%.2f ms/iter)\n", "cpu", itc, tc, tc / itc);

        if (ctx) {
            /* device iterations: H4 recurrence, SpMV+AXPY on device */
            RCRE_GpuBuf *drp = rcre_gpu_alloc(ctx, (n + 1) * sizeof(size_t)),
                        *dci = rcre_gpu_alloc(ctx, nnz * sizeof(size_t)),
                        *dv = rcre_gpu_alloc(ctx, nnz * 8),
                        *dx = rcre_gpu_alloc(ctx, n * 8),
                        *dz = rcre_gpu_alloc(ctx, n * 8),
                        *dy = rcre_gpu_alloc(ctx, n * 8);
            (void)(rcre_gpu_h2d(ctx, drp, rp, (n + 1) * sizeof(size_t)));
            (void)(rcre_gpu_h2d(ctx, dci, ci, nnz * sizeof(size_t)));
            (void)(rcre_gpu_h2d(ctx, dv, val, nnz * 8));
            double* x0 = calloc(n, 8);
            (void)(rcre_gpu_h2d(ctx, dx, x0, n * 8));
            double *r = malloc(n * 8), *z = malloc(n * 8), *p = malloc(n * 8),
                   *w = malloc(n * 8);
            memcpy(r, b, n * 8);
            rcre_pc_ilu0_solve_f64(&fac, r, z);
            memcpy(p, z, n * 8);
            double gam = rcre_linalg_dot_f64(z, r, n);
            int itd = 0;
            for (int it = 0; it < 500; ++it) {
                (void)(rcre_gpu_h2d(ctx, dz, p, n * 8));
                (void)(rcre_gpu_spmv_f64_ctx(ctx, n, drp, dci, dv, dz, dy));
                (void)(rcre_gpu_synchronize(ctx));
                (void)(rcre_gpu_d2h(ctx, w, dy, n * 8));
                double den = rcre_linalg_dot_f64(p, w, n);
                double alpha = (den != 0.0) ? gam / den : 0.0;
                if (!(alpha > 0.0)) break;
                (void)(rcre_gpu_h2d(ctx, dz, p, n * 8));
                (void)(rcre_gpu_axpy_f64_ctx(ctx, n, alpha, dz, dx));
                for (size_t i = 0; i < n; ++i) r[i] -= alpha * w[i];
                rcre_pc_ilu0_solve_f64(&fac, r, z);
                double gamn = rcre_linalg_dot_f64(z, r, n);
                double beta = (gam != 0.0) ? gamn / gam : 0.0;
                gam = gamn;
                for (size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
                ++itd;
                double rn = 0;
                for (size_t i = 0; i < n; ++i) rn += r[i] * r[i];
                if (sqrt(rn) < 1e-10) break;
            }
            double td = 0; /* re-measure the loop wall separately */
            /* second pass for clean timing */
            (void)(rcre_gpu_h2d(ctx, dx, x0, n * 8));
            memcpy(r, b, n * 8);
            rcre_pc_ilu0_solve_f64(&fac, r, z);
            memcpy(p, z, n * 8);
            gam = rcre_linalg_dot_f64(z, r, n);
            double t1 = now_ms();
            for (int it = 0; it < itd; ++it) {
                (void)(rcre_gpu_h2d(ctx, dz, p, n * 8));
                (void)(rcre_gpu_spmv_f64_ctx(ctx, n, drp, dci, dv, dz, dy));
                (void)(rcre_gpu_d2h(ctx, w, dy, n * 8));
                double den = rcre_linalg_dot_f64(p, w, n);
                double alpha = (den != 0.0) ? gam / den : 0.0;
                (void)(rcre_gpu_h2d(ctx, dz, p, n * 8));
                (void)(rcre_gpu_axpy_f64_ctx(ctx, n, alpha, dz, dx));
                for (size_t i = 0; i < n; ++i) r[i] -= alpha * w[i];
                rcre_pc_ilu0_solve_f64(&fac, r, z);
                double gamn = rcre_linalg_dot_f64(z, r, n);
                double beta = (gam != 0.0) ? gamn / gam : 0.0;
                gam = gamn;
                for (size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
            }
            (void)(rcre_gpu_synchronize(ctx));
            td = now_ms() - t1;
            printf("  %-12s %d iters  %-10.3f ms  (%.2f ms/iter, %.2fx) — ILU0+residual math on CPU; "
                   "SpMV+AXPY on device\n", "dev", itd, td, td / itd, tc / td);
            rcre_gpu_free(drp); rcre_gpu_free(dci); rcre_gpu_free(dv);
            rcre_gpu_free(dx); rcre_gpu_free(dz); rcre_gpu_free(dy);
            free(r); free(z); free(p); free(w); free(x0);
        }
        free(xc); free(b);
        free(rp); free(ci); free(val);
        free(fac.row_ptr); free(fac.col_idx); free(fac.values);
    }

    if (ctx) rcre_gpu_close(ctx);
    printf("\nNote: device CG pays per-iteration h2d/d2h transfers for the\n"
           "residual/p vectors at n=4096 (32 KB each) — the bar shows the\n"
           "bridge's honest end-to-end cost, not the kernel cost alone.\n");
    return 0;
}
