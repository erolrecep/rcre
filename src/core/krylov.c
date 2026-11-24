/**
 * @file krylov.c
 * @brief RCRE H4 — sparse CSR + Krylov iterative solvers (2.2.0)
 *
 * The HPC iterative-solver core:
 *   - minimal F64/F32 CSR storage (public C view, caller-owned arrays)
 *   - SpMV (the Krylov workhorse; scalar kernels — SpMV is
 *     memory-bandwidth bound, the thread/SIMD work is H4b/H6)
 *   - preconditioners: Jacobi (diagonal) and ILU(0) (no fill, following
 *     the sparsity pattern — the sparse analogue of the H3 LU)
 *   - Krylov drivers: preconditioned CG (+ FCG mode), BiCGStab,
 *     restarted GMRES(m) with a Householder QR on the Hessenberg
 *
 * Design contract (matches H0-H3):
 *   - deterministic: fixed loop order, no threading, bitwise
 *     reproducible run-to-run (the slab-parallel SpMV port is H4b/H6);
 *   - F64 is the scientific substrate, F32 the workhorse — one
 *     type-generic implementation instantiated for both;
 *   - convergence is gated on the TRUE relative residual
 *     ||b - A x|| / ||b|| (accumulated in double even on F32);
 *   - callbacks are telemetry only (prof integration), they cannot
 *     abort the solve.
 */

#include "rcre.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* CSR from dense                                                            */
/* ========================================================================= */

RCRE_NODISCARD int rcre_csr_f64_from_dense(const double* A, size_t rows,
                                           size_t cols, double threshold,
                                           size_t** row_ptr,
                                           size_t** col_idx,
                                           double** values, size_t* nnz) {
    if (!A || !row_ptr || !col_idx || !values || !nnz) return -1;
    size_t total = 0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (fabs(A[i * cols + j]) > threshold) total++;
    size_t* rp = (size_t*)malloc((rows + 1) * sizeof(size_t));
    size_t* ci = (size_t*)malloc((total ? total : 1) * sizeof(size_t));
    double* va = (double*)malloc((total ? total : 1) * sizeof(double));
    if (!rp || !ci || !va) {
        free(rp); free(ci); free(va);
        *nnz = 0;
        return -1;
    }
    for (size_t i = 0; i <= rows; ++i) rp[i] = 0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (fabs(A[i * cols + j]) > threshold) rp[i + 1]++;
    for (size_t i = 1; i <= rows; ++i) rp[i] += rp[i - 1];
    /* running write cursors: each row fills from its own start (rp[i]) */
    size_t* cur = (size_t*)malloc((rows + 1) * sizeof(size_t));
    if (!cur) { free(rp); free(ci); free(va); *nnz = 0; return -1; }
    memcpy(cur, rp, (rows + 1) * sizeof(size_t));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            const double v = A[i * cols + j];
            if (fabs(v) > threshold) {
                const size_t at = cur[i]++;
                ci[at] = j;
                va[at] = v;
            }
        }
    free(cur);
    *row_ptr = rp;
    *col_idx = ci;
    *values = va;
    *nnz = total;
    return 0;
}

RCRE_NODISCARD int rcre_csr_f32_from_dense(const float* A, size_t rows,
                                           size_t cols, float threshold,
                                           size_t** row_ptr,
                                           size_t** col_idx,
                                           float** values, size_t* nnz) {
    if (!A || !row_ptr || !col_idx || !values || !nnz) return -1;
    size_t total = 0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (fabsf(A[i * cols + j]) > threshold) total++;
    size_t* rp = (size_t*)malloc((rows + 1) * sizeof(size_t));
    size_t* ci = (size_t*)malloc((total ? total : 1) * sizeof(size_t));
    float* va = (float*)malloc((total ? total : 1) * sizeof(float));
    if (!rp || !ci || !va) {
        free(rp); free(ci); free(va);
        *nnz = 0;
        return -1;
    }
    for (size_t i = 0; i <= rows; ++i) rp[i] = 0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (fabsf(A[i * cols + j]) > threshold) rp[i + 1]++;
    for (size_t i = 1; i <= rows; ++i) rp[i] += rp[i - 1];
    size_t* cur = (size_t*)malloc((rows + 1) * sizeof(size_t));
    if (!cur) { free(rp); free(ci); free(va); *nnz = 0; return -1; }
    memcpy(cur, rp, (rows + 1) * sizeof(size_t));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            const float v = A[i * cols + j];
            if (fabsf(v) > threshold) {
                const size_t at = cur[i]++;
                ci[at] = j;
                va[at] = v;
            }
        }
    free(cur);
    *row_ptr = rp;
    *col_idx = ci;
    *values = va;
    *nnz = total;
    return 0;
}

/* ========================================================================= */
/* SpMV                                                                      */
/* ========================================================================= */

void rcre_spmv_f64(const RCRE_CSR_F64* A, const double* x, double* y) {
    if (!A || !x || !y) return;
    for (size_t i = 0; i < A->rows; ++i) {
        double s = 0.0;
        for (size_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
            s += A->values[k] * x[A->col_idx[k]];
        y[i] = s;
    }
}

void rcre_spmv_f32(const RCRE_CSR_F32* A, const float* x, float* y) {
    if (!A || !x || !y) return;
    for (size_t i = 0; i < A->rows; ++i) {
        float s = 0.0f;
        for (size_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
            s += A->values[k] * x[A->col_idx[k]];
        y[i] = s;
    }
}

/* ========================================================================= */
/* Type-generic preconditioners + Krylov drivers                             */
/* =========================================================================
 * One implementation, two instantiations (F64 substrate, F32 workhorse).
 * SUF  = "f64" / "f32"  (name suffix)
 * CSR  = the RCRE_CSR_* struct type
 * REAL = double / float
 * ABSF = fabs / fabsf
 */

#define RCRE_KRYLOV_IMPL(SUF, CSR, REAL, ABSF)                               \
                                                                             \
/* --- helpers ------------------------------------------------------------- */ \
                                                                             \
static size_t krylov_find_col_##SUF(const CSR* A, size_t row, size_t col) {  \
    for (size_t k = A->row_ptr[row]; k < A->row_ptr[row + 1]; ++k) {         \
        if (A->col_idx[k] == col) return k;                                  \
        if (A->col_idx[k] > col) break; /* ascending */                      \
    }                                                                        \
    return (size_t)-1;                                                       \
}                                                                            \
                                                                             \
/* double-accumulated 2-norm: the convergence monitor on BOTH paths          */ \
static double krylov_norm_##SUF(const REAL* v, size_t n) {                   \
    double s = 0.0;                                                          \
    for (size_t i = 0; i < n; ++i) s += (double)v[i] * (double)v[i];         \
    return sqrt(s);                                                          \
}                                                                            \
                                                                             \
static REAL krylov_dot_##SUF(const REAL* a, const REAL* b, size_t n) {       \
    REAL s = (REAL)0;                                                        \
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];                         \
    return s;                                                                \
}                                                                            \
                                                                             \
static void krylov_axpy_##SUF(REAL* y, REAL a, const REAL* x, size_t n) {    \
    for (size_t i = 0; i < n; ++i) y[i] += a * x[i];                         \
}                                                                            \
                                                                             \
                                                                             \
static void krylov_apply_pc_##SUF(int pc, const void* pc_data,                \
                                  const REAL* r, REAL* z, size_t n) {        \
    switch (pc) {                                                            \
    case RCRE_PC_JACOBI: {                                                   \
        const REAL* inv = (const REAL*)pc_data;                              \
        for (size_t i = 0; i < n; ++i) z[i] = inv[i] * r[i];                 \
        break;                                                               \
    }                                                                        \
    case RCRE_PC_WEIGHTED_JACOBI: {                                          \
        /* pc_data = w[] with w[i] = omega/|A_ii| (pre-scaled by the user)  \
         * — same code path as plain Jacobi, different diagonal array.      \
         */                                                                  \
        const REAL* w = (const REAL*)pc_data;                                \
        for (size_t i = 0; i < n; ++i) z[i] = w[i] * r[i];                   \
        break;                                                               \
    }                                                                        \
    case RCRE_PC_ILU0: {                                                     \
        rcre_pc_ilu0_solve_##SUF((const CSR*)pc_data, r, z);                 \
        break;                                                               \
    }                                                                        \
    default:                                                                 \
        memcpy(z, r, n * sizeof(REAL));                                      \
        break;                                                               \
    }                                                                        \
}                                                                            \
                                                                             \
/* --- Jacobi --------------------------------------------------------------- */ \
                                                                             \
RCRE_NODISCARD int rcre_pc_jacobi_##SUF(const CSR* A, REAL* inv_diag) {      \
    if (!A || !inv_diag || A->rows != A->cols) return -1;                    \
    const size_t n = A->rows;                                                \
    for (size_t i = 0; i < n; ++i) {                                         \
        const size_t p = krylov_find_col_##SUF(A, i, i);                     \
        if (p == (size_t)-1 || A->values[p] == (REAL)0) return -1;           \
        inv_diag[i] = (REAL)1 / A->values[p];                                \
    }                                                                        \
    return 0;                                                                \
}                                                                            \
                                                                             \
/* --- ILU(0) --------------------------------------------------------------- */ \
/* Factors A = L*U on EXACTLY A's pattern: values[pos] is L[i,k] < i...      */ \
/* strictly below the diagonal and U[i,j] on/above it (unit L implicit).     */ \
/* O(nnz_below * |row_k| * |row_i|) with short-row linear searches — fine    */ \
/* for structured stencils; the big-n barrier is H4b (MG/AMG), not here.     */ \
                                                                             \
RCRE_NODISCARD int rcre_pc_ilu0_##SUF(const CSR* A, CSR* fac) {              \
    if (!A || !fac || A->rows != A->cols) return -1;                         \
    const size_t n = A->rows;                                                \
    size_t* rp = (size_t*)malloc((n + 1) * sizeof(size_t));                  \
    size_t* ci = (size_t*)malloc((A->nnz ? A->nnz : 1) * sizeof(size_t));    \
    REAL* va = (REAL*)malloc((A->nnz ? A->nnz : 1) * sizeof(REAL));          \
    if (!rp || !ci || !va) { free(rp); free(ci); free(va); return -1; }      \
    memcpy(rp, A->row_ptr, (n + 1) * sizeof(size_t));                        \
    memcpy(ci, A->col_idx, A->nnz * sizeof(size_t));                         \
    memcpy(va, A->values, A->nnz * sizeof(REAL));                            \
    /* publish the pattern BEFORE the sweep: the find_col searches read   \
     * fac's own row_ptr/col_idx during the factorization */                \
    fac->rows = n;                                                           \
    fac->cols = n;                                                           \
    fac->nnz = A->nnz;                                                       \
    fac->row_ptr = rp;                                                       \
    fac->col_idx = ci;                                                       \
    fac->values = va;                                                        \
    for (size_t k = 0; k < n; ++k) {                                         \
        const size_t pk = krylov_find_col_##SUF(fac, k, k);                  \
        if (pk == (size_t)-1 || va[pk] == (REAL)0) {                         \
            free(rp); free(ci); free(va); return -1;                         \
        }                                                                    \
        for (size_t i = k + 1; i < n; ++i) {                                 \
            const size_t pi = krylov_find_col_##SUF(fac, i, k);              \
            if (pi == (size_t)-1) continue;                                  \
            const REAL w = va[pi] / va[pk];                                  \
            if (w == (REAL)0) continue;                                      \
            va[pi] = w; /* L[i,k] */                                         \
            for (size_t q = A->row_ptr[k]; q < A->row_ptr[k + 1]; ++q) {     \
                const size_t j = A->col_idx[q];                              \
                if (j <= k) continue; /* only U[k,j] with j > k */           \
                const size_t pj = krylov_find_col_##SUF(fac, i, j);          \
                if (pj != (size_t)-1) va[pj] -= w * va[q];                   \
            }                                                                \
        }                                                                    \
    }                                                                        \
    return 0;                                                                \
}                                                                            \
                                                                             \
/* z = (L U)^-1 r, IN PLACE on z (forward then backward; both passes         */ \
/* touch only already- / not-yet-written entries, so one scratch buffer).    */ \
void rcre_pc_ilu0_solve_##SUF(const CSR* F, const REAL* r, REAL* z) {        \
    if (!F || !r || !z) return;                                              \
    const size_t n = F->rows;                                                \
    for (size_t i = 0; i < n; ++i) {                                         \
        REAL s = r[i];                                                       \
        for (size_t k = F->row_ptr[i]; k < F->row_ptr[i + 1]; ++k) {         \
            const size_t j = F->col_idx[k];                                  \
            if (j >= i) break; /* L part is strictly below */                \
            s -= F->values[k] * z[j];                                        \
        }                                                                    \
        z[i] = s;                                                            \
    }                                                                        \
    for (size_t i = n; i-- > 0;) {                                           \
        REAL s = z[i], diag = (REAL)0;                                       \
        for (size_t k = F->row_ptr[i]; k < F->row_ptr[i + 1]; ++k) {         \
            const size_t j = F->col_idx[k];                                  \
            if (j == i) { diag = F->values[k]; continue; }                   \
            if (j > i) s -= F->values[k] * z[j];                             \
        }                                                                    \
        z[i] = (diag != (REAL)0) ? s / diag : r[i]; /* defensive */          \
    }                                                                        \
}                                                                            \
                                                                             \
/* --- PCG / FCG ------------------------------------------------------------ */ \
                                                                             \
RCRE_NODISCARD int rcre_krylov_pcg_##SUF(const CSR* A, const REAL* b,        \
                                         REAL* x, double tol,                \
                                         size_t max_iter, int pc,            \
                                         const void* pc_data, int fcg,       \
                                         RCRE_KrylovCallback cb,             \
                                         void* user) {                       \
    if (!A || !b || !x) return -1;                                           \
    const size_t n = A->rows;                                                \
    if (n == 0) return 0;                                                    \
    if (pc == RCRE_PC_ILU0 && !pc_data) return -1;                           \
    if (pc == RCRE_PC_JACOBI && !pc_data) return -1;                         \
    if (pc == RCRE_PC_WEIGHTED_JACOBI && !pc_data) return -1;                \
    REAL* r = (REAL*)malloc(n * sizeof(REAL));                               \
    REAL* z = (REAL*)malloc(n * sizeof(REAL));                               \
    REAL* p = (REAL*)malloc(n * sizeof(REAL));                               \
    REAL* w = (REAL*)malloc(n * sizeof(REAL));                               \
    if (!r || !z || !p || !w) { free(r); free(z); free(p); free(w);          \
                                return -1; }                                 \
    /* r = b - A*x */                                                        \
    rcre_spmv_##SUF(A, x, w);                                                \
    for (size_t i = 0; i < n; ++i) r[i] = b[i] - w[i];                       \
    const double normb = krylov_norm_##SUF(b, n);                            \
    const double normref = normb > 0.0 ? normb : 1.0;                        \
    double rel = krylov_norm_##SUF(r, n) / normref;                          \
    if (cb) cb(0, rel, user);                                                \
    if (rel <= tol) { free(r); free(z); free(p); free(w); return 0; }        \
    /* first preconditioned step */                                          \
    krylov_apply_pc_##SUF(pc, pc_data, r, z, n);                             \
    memcpy(p, z, n * sizeof(REAL));                                          \
    REAL gam = krylov_dot_##SUF(fcg ? r : z, fcg ? p : r, n);                \
    int iters = 0;                                                           \
    int ok = 0;                                                              \
    while (iters < (int)max_iter) {                                          \
        ++iters;                                                             \
        rcre_spmv_##SUF(A, p, w); /* Ap */                                   \
        const REAL den = krylov_dot_##SUF(p, w, n);                          \
        const REAL alpha = (den != (REAL)0) ? gam / den : (REAL)0;           \
        if (alpha != alpha || alpha <= (REAL)0) break; /* breakdown */       \
        krylov_axpy_##SUF(x, alpha, p, n);                                   \
        for (size_t i = 0; i < n; ++i) r[i] -= alpha * w[i];                 \
        rel = krylov_norm_##SUF(r, n) / normref;                             \
        if (cb) cb((size_t)iters, rel, user);                                \
        if (rel <= tol) { ok = 1; break; }                                   \
        if (fcg) {                                                           \
            /* FCG: p is updated CG-style FROM THE RESIDUAL (M^-1 is        \
             * applied only to r0, saving one preconditioner apply per      \
             * iteration). Valid when M^-1 A is symmetric; with pc = NONE   \
             * FCG reduces exactly to plain CG. gamma = r^T r. */           \
            const REAL gamn = krylov_dot_##SUF(r, r, n);                     \
            const REAL beta = (gam != (REAL)0) ? gamn / gam : (REAL)0;       \
            for (size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];        \
            gam = gamn;                                                      \
        } else {                                                             \
            krylov_apply_pc_##SUF(pc, pc_data, r, z, n);                     \
            const REAL gamn = krylov_dot_##SUF(z, r, n);                     \
            const REAL beta = (gam != (REAL)0) ? gamn / gam : (REAL)0;       \
            for (size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];        \
            gam = gamn;                                                      \
        }                                                                    \
    }                                                                        \
    free(r); free(z); free(p); free(w);                                      \
    return ok ? iters : -1;                                                  \
}                                                                            \
                                                                             \
/* --- BiCGStab -------------------------------------------------------------- */ \
                                                                             \
/* Plain BiCGStab (original 1983 recurrence) on the LEFT-PRECONDITIONED    \
 * system B = M^-1 A with RHS f = M^-1 b (x is the state, exactly as in the  \
 * unpreconditioned case: the B-system solution IS x, and r = f - B x =      \
 * M^-1 (b - A x) is the standard left-preconditioned monitor, which equals  \
 * the true relative residual for pc = NONE). spmv_B(v) = M^-1 (A v) is      \
 * spmv followed by the preconditioner apply. The shadow start is v = r0     \
 * (van der Vorst's robust restart: measured here to converge on standard 1D\n * advection-diffusion matrices where the original zero-shadow form breaks  \
 * down or stagnates; see docs/hpc.md).                                  \
 * Convergence is matrix-dependent (BiCGStab's known fragility; see         \
 * docs/hpc.md — GMRES(m) is the robust nonsymmetric solver). Breakdowns    \
 * (zero recurrence pivots) return -1 with x at the last full update.       \
 * Returns the number of iterations or -1. */                                \
RCRE_NODISCARD int rcre_krylov_bicgstab_##SUF(const CSR* A, const REAL* b,   \
                                              REAL* x, double tol,           \
                                              size_t max_iter, int pc,       \
                                              const void* pc_data,           \
                                              RCRE_KrylovCallback cb,        \
                                              void* user) {                  \
    if (!A || !b || !x) return -1;                                           \
    const size_t n = A->rows;                                                \
    if (n == 0) return 0;                                                    \
    if ((pc == RCRE_PC_ILU0 || pc == RCRE_PC_JACOBI ||                       \
         pc == RCRE_PC_WEIGHTED_JACOBI) && !pc_data) return -1;              \
    REAL* f = (REAL*)malloc(n * sizeof(REAL));    /* f = M^-1 b */           \
    REAL* r = (REAL*)malloc(n * sizeof(REAL));    /* r = f - B x */          \
    REAL* r0 = (REAL*)malloc(n * sizeof(REAL));   /* shadow residual */      \
    REAL* v = (REAL*)malloc(n * sizeof(REAL));                                       \
    REAL* p = (REAL*)malloc(n * sizeof(REAL));                                       \
    REAL* w = (REAL*)malloc(n * sizeof(REAL));                                       \
    REAL* s = (REAL*)malloc(n * sizeof(REAL));                                       \
    REAL* t = (REAL*)malloc(n * sizeof(REAL));                                       \
    if (!f || !r || !r0 || !v || !p || !w || !s || !t) {                        \
        free(f); free(r); free(r0); free(v); free(p);                           \
        free(w); free(s); free(t);                                              \
        return -1;                                                             \
    }                                                                        \
    /* f = M^-1 b; r = f - B x0 = M^-1 (b - A x0) */                          \
    krylov_apply_pc_##SUF(pc, pc_data, b, f, n);                              \
    rcre_spmv_##SUF(A, x, w);                                                 \
    krylov_apply_pc_##SUF(pc, pc_data, w, w, n);  /* w = B x0 */              \
    for (size_t i = 0; i < n; ++i) r[i] = f[i] - w[i];                        \
    memcpy(r0, r, n * sizeof(REAL));                                          \
    memcpy(v, r, n * sizeof(REAL));   /* van der Vorst shadow start (v = r0) */ \
    memset(p, 0, n * sizeof(REAL));                                           \
    const double normf = krylov_norm_##SUF(f, n);                             \
    const double normref = normf > 0.0 ? normf : 1.0;                         \
    double rel = krylov_norm_##SUF(r, n) / normref;                           \
    if (cb) cb(0, rel, user);                                                 \
    if (rel <= tol) { /* x0 already good: leave x untouched, return 0 */      \
        free(f); free(r); free(r0); free(v); free(p);                         \
        free(w); free(s); free(t);                                            \
        return 0;                                                             \
    }                                                                        \
    REAL rho = (REAL)1, alpha = (REAL)1;                                      \
    int iters = 0, ok = 0;                                                    \
    while (iters < (int)max_iter) {                                           \
        ++iters;                                                              \
        const REAL rho_n = krylov_dot_##SUF(r, r0, n);                        \
        if (rho_n == (REAL)0) break; /* breakdown */                          \
        const REAL theta = rho_n / rho;                                       \
        for (size_t i = 0; i < n; ++i)                                        \
            p[i] = r[i] + theta * (v[i] - alpha * p[i]);                      \
        rcre_spmv_##SUF(A, p, w);                                             \
        krylov_apply_pc_##SUF(pc, pc_data, w, w, n);  /* w = B p */           \
        const REAL rw = krylov_dot_##SUF(w, r0, n);                           \
        if (rw == (REAL)0) break; /* breakdown */                             \
        alpha = rho_n / rw;                                                   \
        for (size_t i = 0; i < n; ++i) s[i] = r[i] - alpha * w[i];            \
        rcre_spmv_##SUF(A, s, t);                                             \
        krylov_apply_pc_##SUF(pc, pc_data, t, t, n);  /* t = B s */           \
        /* omega = (s, B s)/||B s||^2: least-squares value minimizing \
         * ||r - omega t||; with the van der Vorst shadow start this is \
         * the variant measured to be robust here. */                      \
        const REAL tt = krylov_dot_##SUF(s, t, n);                            \
        const REAL tw = krylov_dot_##SUF(t, t, n);                             \
        if (tw == (REAL)0) break; /* breakdown */                             \
        const REAL omega = tt / tw;                                           \
        for (size_t i = 0; i < n; ++i) x[i] += alpha * p[i] + omega * s[i];   \
        for (size_t i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];            \
        memcpy(v, s, n * sizeof(REAL));                                       \
        /* next step's rho = (r_new, r0) (van der Vorst) */                   \
        rho = krylov_dot_##SUF(r, r0, n);                                     \
        rel = krylov_norm_##SUF(r, n) / normref;                              \
        if (cb) cb((size_t)iters, rel, user);                                 \
        if (rel <= tol) { ok = 1; break; }                                    \
    }                                                                        \
    free(f); free(r); free(r0); free(v); free(p);                             \
    free(w); free(s); free(t);                                                \
    return ok ? iters : -1;                                                   \
}                                                                            \
                                                                             \
/* --- GMRES(m), restarted, Householder QR on the Hessenberg ----------------- */ \
                                                                             \
/* Householder QR of the active (m+1) x m Hessenberg block (leading dim \
 * LDA) applied to c (length m+1); on return R = H (upper triangle),    \
 * c = Q^T c (in place), y = R^-1 c (length m), *res = |c[m]| (the LS   \
 * residual of min ||c - H y||). LDA = the buffer's row stride (the     \
 * full allocation); only the first m+1 rows / m columns are active.    \
 */                                                                  \
static void krylov_hqr_##SUF(REAL* H, size_t m, size_t lda, REAL* c,    \
                             REAL* y, REAL* res) {                       \
    for (size_t k = 0; k < m; ++k) {                                     \
        /* x = column k of H, rows k..m (length m+1-k) */                \
        REAL ssq = (REAL)0;                                              \
        for (size_t i = k; i <= m; ++i) ssq += H[i * lda + k] * H[i * lda + k]; \
        const REAL nrm = (REAL)sqrt((double)ssq);                        \
        if (nrm == (REAL)0) continue;                                    \
        /* alpha = -sign(x0) * ||x||, sign(0) = +1. v is a SEPARATE       \
         * buffer: writing v[0] = x0 - alpha into H's own column would    \
         * corrupt the (k,k) entry that the j = k dot product reads. */   \
        const REAL alpha = (H[k * lda + k] < (REAL)0) ? nrm : (REAL)-nrm;\
        REAL* v = (REAL*)malloc((m + 1) * sizeof(REAL));                 \
        if (!v) { free(v); continue; } /* OOM: skip this reflector */    \
        for (size_t i = 0; i <= m - k; ++i) v[i] = H[(k + i) * lda + k]; \
        v[0] -= alpha; /* v = x - alpha*e1 */                            \
        REAL vv = (REAL)0;                                               \
        for (size_t i = 0; i <= m - k; ++i) vv += v[i] * v[i];           \
        if (vv == (REAL)0) continue;                                     \
        const REAL tau = (REAL)2 / vv;                                   \
        /* apply H2 = I - tau*v*v^T to H[k..m, k..m-1] (per column) */   \
        for (size_t j = k; j < m; ++j) {                                 \
            REAL d = (REAL)0;                                            \
            for (size_t i = k; i <= m; ++i) d += v[i - k] * H[i * lda + j]; \
            d *= tau;                                                    \
            for (size_t i = k; i <= m; ++i) H[i * lda + j] -= d * v[i - k]; \
        }                                                                \
        /* and to c[k..m] */                                             \
        {                                                                \
            REAL d = (REAL)0;                                            \
            for (size_t i = k; i <= m; ++i) d += v[i - k] * c[i];        \
            d *= tau;                                                    \
            for (size_t i = k; i <= m; ++i) c[i] -= d * v[i - k];        \
        }                                                                \
        free(v);                                                         \
    }                                                                    \
    /* y = R^-1 c (R = upper triangle of H, m x m) */                    \
    for (size_t i = m; i-- > 0;) {                                       \
        REAL s = c[i];                                                   \
        for (size_t j = i + 1; j < m; ++j) s -= H[i * lda + j] * y[j];   \
        const REAL diag = H[i * lda + i];                                \
        y[i] = (diag != (REAL)0) ? s / diag : (REAL)0;                   \
    }                                                                    \
    if (res) *res = (REAL)ABSF(c[m]);                                    \
}                                                                        \
                                                                         \
RCRE_NODISCARD int rcre_krylov_gmres_##SUF(const CSR* A, const REAL* b,      \
                                           REAL* x, double tol,              \
                                           size_t max_iter, size_t m,        \
                                           int pc, const void* pc_data,      \
                                           RCRE_KrylovCallback cb,           \
                                           void* user) {                     \
    if (!A || !b || !x) return -1;                                           \
    const size_t n = A->rows;                                                \
    if (n == 0) return 0;                                                    \
    if (m == 0 || m >= n) return -1;                                         \
    if ((pc == RCRE_PC_ILU0 || pc == RCRE_PC_JACOBI ||                       \
         pc == RCRE_PC_WEIGHTED_JACOBI) && !pc_data) return -1;              \
    REAL* r = (REAL*)malloc(n * sizeof(REAL));                               \
    REAL* w = (REAL*)malloc(n * sizeof(REAL));                               \
    REAL* z = (REAL*)malloc(n * sizeof(REAL));                               \
    REAL* V = (REAL*)malloc((size_t)(m + 1) * n * sizeof(REAL));             \
    REAL* H = (REAL*)calloc((size_t)(m + 1) * m, sizeof(REAL));              \
    REAL* c = (REAL*)malloc((m + 1) * sizeof(REAL));                         \
    REAL* y = (REAL*)malloc(m * sizeof(REAL));                               \
    if (!r || !w || !z || !V || !H || !c || !y) {                            \
        free(r); free(w); free(z); free(V); free(H); free(c); free(y);       \
        return -1;                                                           \
    }                                                                        \
    /* r = b - A x */                                                        \
    rcre_spmv_##SUF(A, x, w);                                                \
    for (size_t i = 0; i < n; ++i) r[i] = b[i] - w[i];                       \
    const double normb = krylov_norm_##SUF(b, n);                            \
    const double normref = normb > 0.0 ? normb : 1.0;                        \
    double rel = krylov_norm_##SUF(r, n) / normref;                          \
    if (cb) cb(0, rel, user);                                                \
    if (rel <= tol) { free(r); free(w); free(z); free(V); free(H);           \
        free(c); free(y); return 0; }                                        \
    int ok = 0;                                                              \
    size_t matvecs = 1; /* the initial residual matvec */                    \
    while (!ok) {                                                            \
        /* fresh Arnoldi basis from the current residual */                  \
        krylov_apply_pc_##SUF(pc, pc_data, r, z, n);                         \
        const REAL beta = (REAL)krylov_norm_##SUF(z, n);                     \
        if (beta == (REAL)0) { ok = 1; break; } /* M^-1 r = 0 => r = 0 */    \
        for (size_t i = 0; i < n; ++i) V[i] = z[i] / beta;                   \
        memset(H, 0, (size_t)(m + 1) * m * sizeof(REAL));                    \
        memset(c, 0, (m + 1) * sizeof(REAL));                                \
        c[0] = beta;                                                         \
        /* inner loop: at most m Arnoldi steps, bounded by the matvec       \
         * budget. max_iter counts ALL matvecs (initial + Arnoldi +          \
         * re-checks); the returned value is matvecs - 1 (x0 was applied     \
         * once before the solver "started"). */                             \
        size_t jcols = 0;                                                    \
        for (size_t j = 1; j <= m; ++j) {                                    \
            if (matvecs >= max_iter) break; /* next step would exceed */     \
            ++matvecs;                                                       \
            const REAL* vprev = V + (size_t)(j - 1) * n;                     \
            rcre_spmv_##SUF(A, vprev, w);                                    \
            krylov_apply_pc_##SUF(pc, pc_data, w, z, n);                     \
            /* modified Gram-Schmidt (2 passes) */                           \
            for (size_t i = 0; i < j; ++i) {                                 \
                const REAL* vi = V + i * n;                                  \
                H[i * m + (j - 1)] = krylov_dot_##SUF(vi, z, n);             \
                krylov_axpy_##SUF(z, -H[i * m + (j - 1)], vi, n);            \
            }                                                                \
            for (size_t i = 0; i < j; ++i) {                                 \
                const REAL* vi = V + i * n;                                  \
                const REAL h = krylov_dot_##SUF(vi, z, n);                   \
                H[i * m + (j - 1)] += h;                                     \
                krylov_axpy_##SUF(z, -h, vi, n);                             \
            }                                                                \
            const REAL hnext = (REAL)krylov_norm_##SUF(z, n);                \
            H[j * m + (j - 1)] = hnext;                                      \
            jcols = j;                                                       \
            rel = (double)hnext / normref;                                   \
            if (cb) cb(matvecs, rel, user);                                  \
            if (hnext <= (REAL)(tol * normref)) break; /* converged */       \
            if (j < m && hnext != (REAL)0) {                                 \
                REAL* vn = V + j * n;                                        \
                for (size_t i = 0; i < n; ++i) vn[i] = z[i] / hnext;         \
            }                                                                \
        }                                                                    \
        if (jcols == 0) break; /* budget exhausted before any step */        \
        /* least-squares solve on the active (jcols+1) x jcols block */      \
        REAL resls = (REAL)0;                                                \
        krylov_hqr_##SUF(H, jcols, m, c, y, &resls);                         \
        for (size_t j = 0; j < jcols; ++j) {                                 \
            const REAL* vj = V + j * n;                                      \
            krylov_axpy_##SUF(x, y[j], vj, n);                               \
        }                                                                    \
        /* re-check the TRUE residual (the Arnoldi residual is exact only    \
         * up to the re-orthogonalization quality); consumes 1 matvec */     \
        if (matvecs < max_iter) {                                            \
            ++matvecs;                                                       \
            rcre_spmv_##SUF(A, x, w);                                        \
            for (size_t i = 0; i < n; ++i) r[i] = b[i] - w[i];               \
            rel = krylov_norm_##SUF(r, n) / normref;                         \
            if (cb) cb(matvecs, rel, user);                                  \
        }                                                                    \
        if (rel <= tol) ok = 1;                                              \
    }                                                                        \
    free(r); free(w); free(z); free(V); free(H); free(c); free(y);           \
    return ok ? (int)(matvecs - 1) : -1;                                     \
}
RCRE_KRYLOV_IMPL(f64, RCRE_CSR_F64, double, fabs)
RCRE_KRYLOV_IMPL(f32, RCRE_CSR_F32, float, fabsf)
#undef RCRE_KRYLOV_IMPL
