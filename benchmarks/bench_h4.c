/* RCRE H4 benchmark — iterative solvers + 2D-Poisson multigrid (2.2.0).
 *
 * Bars:
 *   - PCG + ILU(0) on 2D Poisson (SPD, smooth-dominated): O(n) iterations.
 *   - GMRES(32) + ILU(0) on 2D Poisson: robust nonsymmetric preconditioned
 *     path (matvec count + wall time).
 *   - V-cycle and FAS (fixed 2+2 red-black GS sweeps): per-cycle wall time.
 *
 * All bars are single-thread, -O3, -ffp-contract=off. The Poisson operator
 * is the 5-point h^2-scaled stencil with zero Dirichlet boundaries.
 *
 * Build: make bench_h4   Run: make bench_h4 (runs after build)
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

/* LCG random RHS (deterministic, seed = 0x987654321 * (n + 7)). */
static void random_rhs(size_t nn, double* f, size_t n) {
    uint64_t st = 0x987654321ull * (n + 7);
    for (size_t i = 0; i < nn; ++i) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        f[i] = (double)((st >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53) * 2.0 - 1.0;
    }
}

/* CSR 2D Poisson (rows ascending: c-n, c-1, c, c+1, c+n). */
static void poisson_csr(size_t n, RCRE_CSR_F64* A) {
    const size_t nn = n * n;
    A->rows = nn;
    A->cols = nn;
    A->row_ptr = (size_t*)malloc((nn + 1) * sizeof(size_t));
    A->col_idx = (size_t*)malloc(5 * nn * sizeof(size_t));
    A->values = (double*)malloc(5 * nn * sizeof(double));
    size_t nnz = 0;
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            const size_t c = j * n + i;
            A->row_ptr[c] = nnz;
            if (j > 0) { A->col_idx[nnz] = c - n; A->values[nnz] = -1.0; ++nnz; }
            if (i > 0) { A->col_idx[nnz] = c - 1; A->values[nnz] = -1.0; ++nnz; }
            A->col_idx[nnz] = c; A->values[nnz] = 4.0; ++nnz;
            if (i + 1 < n) { A->col_idx[nnz] = c + 1; A->values[nnz] = -1.0; ++nnz; }
            if (j + 1 < n) { A->col_idx[nnz] = c + n; A->values[nnz] = -1.0; ++nnz; }
        }
    }
    A->row_ptr[nn] = nnz;
    A->nnz = nnz;
}

int main(void) {
    rcre_hal_init();
    printf("RCRE H4 benchmark: Krylov solvers + 2D-Poisson multigrid (F64)\n");
    printf("single-thread, -O3, -ffp-contract=off\n\n");

    /* ---- PCG + ILU(0): O(n) iteration bar --------------------------------- */
    printf("PCG + ILU(0) on 2D Poisson (tol 1e-10, random RHS):\n");
    printf("  n^2        iters    ILU0 fac (ms)   solve (ms)   rel err\n");
    for (size_t n = 64; n <= 512; n *= 2) {
        const size_t nn = n * n;
        RCRE_CSR_F64 A;
        poisson_csr(n, &A);
        double* f = (double*)malloc(nn * sizeof(double));
        random_rhs(nn, f, n);
        double* x = (double*)calloc(nn, sizeof(double));
        RCRE_CSR_F64 fac;
        double t0 = now_s();
        if (rcre_pc_ilu0_f64(&A, &fac) != 0) { continue; }
        double t_fac = now_s() - t0;
        t0 = now_s();
        int it = rcre_krylov_pcg_f64(&A, f, x, 1e-10, 8 * n, RCRE_PC_ILU0, &fac,
                                     0, NULL, NULL);
        double t_solve = now_s() - t0;
        /* true relative residual */
        double* r = (double*)malloc(nn * sizeof(double));
        rcre_spmv_f64(&A, x, r);
        double e = 0.0, bf = 0.0;
        for (size_t i = 0; i < nn; ++i) {
            const double d = r[i] - f[i];
            e += d * d;
            bf += f[i] * f[i];
        }
        printf("  %5zu %8d %15.2f %13.2f   %.1e\n", n, it, t_fac * 1e3,
               t_solve * 1e3, sqrt(e) / sqrt(bf));
        free(r);
        free(f);
        free(x);
    }

    /* ---- GMRES(32) + ILU(0): robust nonsymmetric path --------------------- */
    printf("\nGMRES(32) + ILU(0) on 2D Poisson (tol 1e-8, random RHS):\n");
    printf("  n^2     matvecs    ILU0 fac (ms)   solve (ms)\n");
    for (size_t n = 64; n <= 256; n *= 2) {
        const size_t nn = n * n;
        RCRE_CSR_F64 A;
        poisson_csr(n, &A);
        double* f = (double*)malloc(nn * sizeof(double));
        random_rhs(nn, f, n);
        double* x = (double*)calloc(nn, sizeof(double));
        RCRE_CSR_F64 fac;
        double t0 = now_s();
        if (rcre_pc_ilu0_f64(&A, &fac) != 0) { continue; }
        double t_fac = now_s() - t0;
        t0 = now_s();
        int mv = rcre_krylov_gmres_f64(&A, f, x, 1e-8, 8 * n, 32, RCRE_PC_ILU0,
                                       &fac, NULL, NULL);
        double t_solve = now_s() - t0;
        printf("  %5zu %10d %15.2f %13.2f\n", n, mv, t_fac * 1e3, t_solve * 1e3);
        free(f);
        free(x);
    }

    /* ---- Multigrid: per-cycle wall time ------------------------------------ */
    printf("\nMultigrid on 2D Poisson (fixed 2+2 red-black GS, per V/FAS cycle):\n");
    printf("  n^2      V-cycle (ms)   FAS (ms)\n");
    for (size_t n = 32; n <= 256; n *= 2) {
        const size_t nn = n * n;
        double* f = (double*)malloc(nn * sizeof(double));
        random_rhs(nn, f, n);
        double* xv = (double*)calloc(nn, sizeof(double));
        double* xf = (double*)calloc(nn, sizeof(double));
        const int cycles = 20;
        double t0 = now_s();
        for (int k = 0; k < cycles; ++k) {
            if (rcre_mg_vcycle_poisson2d_f64(n, n, f, xv, 2, 2) != 1) break;
        }
        double t_vc = (now_s() - t0) / cycles;
        t0 = now_s();
        for (int k = 0; k < cycles; ++k) {
            if (rcre_mg_fas_poisson2d_f64(n, n, f, xf, 2, 2) != 1) break;
        }
        double t_fas = (now_s() - t0) / cycles;
        printf("  %5zu %15.3f %10.3f\n", n, t_vc * 1e3, t_fas * 1e3);
        free(f);
        free(xv);
        free(xf);
    }

    printf("\nDone.\n");
    return 0;
}
