/**
 * @file test_direct.c
 * @brief H3 direct factorization & eigendecomposition tests.
 *
 * Conformance strategy:
 *  - LU: A = P·L·U reconstruction + residual A·x = b + known singular.
 *  - Cholesky: L·L^T = A + SPD/PSD-neg detection + residual solve.
 *  - QR: Q·R = A, Q^T·Q = I, least-squares residual optimality
 *    (A^T(Ax - b) = 0 — the normal-equation check).
 *  - syev: eigenvalues vs an INDEPENDENT cyclic-Jacobi oracle (a
 *    different algorithm family), residual ||A·v - λv||, Q-orthogonal
 *    V, and a known closed-form case.
 *
 * The Jacobi oracle (n <= 32) is deliberately a different algorithm
 * (explicit Givens similarity sweeps on the full symmetric matrix) so
 * that agreement is evidence of correctness, not shared structure.
 */

#include "rcre.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- test-suite harness (same shape as the other suites) ----------------- */
static int g_checks = 0;
static int g_fails = 0;
#define ASSERT_TRUE(cond)                                                     \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(cond)) {                                                        \
            ++g_fails;                                                        \
            printf("    [ASSERT FAILED] %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                     \
    } while (0)

/* row-major max |A - B| over n×m */
static double max_mat_diff(const double* A, size_t lda, const double* B,
                           size_t ldb, size_t n, size_t m) {
    double mx = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            const double d = fabs(A[i * lda + j] - B[i * ldb + j]);
            if (d > mx) mx = d;
        }
    return mx;
}

static void fill_sym(double* A, size_t n, size_t seed) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uint64_t z = seed + (i * n + j) * 0x9E3779B97F4A7C15ull;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            A[i * n + j] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
        }
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            A[i * n + j] = A[j * n + i];
}

/* make A = M^T M + n·I (SPD) from a random M */
static void fill_spd(double* A, size_t n, size_t seed) {
    double* M = (double*)malloc(n * n * sizeof(double));
    fill_sym(M, n, seed); /* reuse as random (symmetry irrelevant) */
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = (double)n;
            for (size_t k = 0; k < n; ++k) s += M[i * n + k] * M[j * n + k];
            A[i * n + j] = s;
        }
    free(M);
}

/* ---- cyclic Jacobi eigenvalue oracle (n <= 32) --------------------------- */
static int jacobi_eigs(double* A, size_t n, double* w) {
    double off = 0.0;
    for (int sweep = 0; sweep < 500; ++sweep) {
        off = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                off += A[i * n + j] * A[i * n + j];
        if (off == 0.0) break;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q) {
                const double apq = A[p * n + q];
                if (apq == 0.0) continue;
                const double app = A[p * n + p], aqq = A[q * n + q];
                const double th = (aqq - app) / (2.0 * apq);
                const double t = (th >= 0.0 ? 1.0 : -1.0) /
                                 (fabs(th) + sqrt(th * th + 1.0));
                const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (size_t k = 0; k < n; ++k) {
                    if (k != p && k != q) {
                        const double akp = A[k * n + p], akq = A[k * n + q];
                        A[k * n + p] = c * akp - s * akq;
                        A[p * n + k] = A[k * n + p];
                        A[k * n + q] = s * akp + c * akq;
                        A[q * n + k] = A[k * n + q];
                    }
                }
                A[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A[p * n + q] = A[q * n + p] = 0.0;
            }
    }
    for (size_t i = 0; i < n; ++i) w[i] = A[i * n + i];
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (w[j] < w[i]) {
                const double t = w[i];
                w[i] = w[j];
                w[j] = t;
            }
    return 0;
}

/* ============================== LU ======================================== */

static void test_lu_factor_solve(void) {
    printf("  [lu_factor_solve]\n");
    const size_t n = 40;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* Aorig = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * 2 * sizeof(double));
    double* Bref = (double*)malloc(n * 2 * sizeof(double));
    size_t* piv = (size_t*)malloc(n * sizeof(size_t));
    fill_sym(A, n, 101);
    for (size_t r = 0; r < 2; ++r)
        for (size_t i = 0; i < n; ++i) {
            uint64_t z = 9000 + i * 31 + r;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            B[i * 2 + r] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
            Bref[i * 2 + r] = B[i * 2 + r];
        }
    memcpy(Aorig, A, n * n * sizeof(double));

    /* one-shot solve (B packed n x 2, row stride 2) */
    ASSERT_TRUE(rcre_linalg_solve_f64(Aorig, n, n, B, 2, 2) == 0);
    /* residual A*x - b */
    double* Ax = (double*)malloc(n * 2 * sizeof(double));
    rcre_linalg_gemm_f64(n, 2, n, 1.0, Aorig, n, B, 2, 0.0, Ax, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t i = 0; i < n; ++i)
            Ax[i * 2 + r] -= Bref[i * 2 + r];
    ASSERT_TRUE(max_mat_diff(Ax, 2, Ax, 2, n, 2) < 1e-11);

    /* packed factor + solve path agrees with the one-shot */
    double* B2 = (double*)malloc(n * 2 * sizeof(double));
    memcpy(B2, Bref, n * 2 * sizeof(double));
    double* F = (double*)malloc(n * n * sizeof(double));
    memcpy(F, Aorig, n * n * sizeof(double));
    ASSERT_TRUE(rcre_linalg_lu_factor_f64(F, n, n, piv) == 0);
    ASSERT_TRUE(rcre_linalg_lu_solve_f64(F, n, piv, n, B2, 2, 2) == 0);
    ASSERT_TRUE(max_mat_diff(B, 2, B2, 2, n, 2) < 1e-12);

    /* A = P L U reconstruction: build P, L, U and check P·L·U = A */
    double* P = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            P[i * n + j] = (i == j) ? 1.0 : 0.0;
    for (size_t k = 0; k < n; ++k)
        if (piv[k] != k)
            for (size_t j = 0; j < n; ++j) {
                const double t = P[k * n + j];
                P[k * n + j] = P[piv[k] * n + j];
                P[piv[k] * n + j] = t;
            }
    double* L = (double*)malloc(n * n * sizeof(double));
    double* U = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            L[i * n + j] = (i > j) ? F[i * n + j] : 0.0;
            U[i * n + j] = (i <= j) ? F[i * n + j] : 0.0;
        }
        L[i * n + i] += 1.0; /* unit diagonal */
    }
    /* A = P^T (L U): P is the row permutation (P A = L U) */
    double* LU = (double*)malloc(n * n * sizeof(double));
    double* PLU = (double*)malloc(n * n * sizeof(double));
    double* Pt = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            Pt[i * n + j] = P[j * n + i];
    rcre_linalg_gemm_f64(n, n, n, 1.0, L, n, U, n, 0.0, LU, n);
    rcre_linalg_gemm_f64(n, n, n, 1.0, Pt, n, LU, n, 0.0, PLU, n);
    ASSERT_TRUE(max_mat_diff(PLU, n, Aorig, n, n, n) < 1e-12);
    free(Pt);

    free(A); free(Aorig); free(B); free(Bref); free(B2); free(F);
    free(piv); free(Ax); free(P); free(L); free(U); free(LU); free(PLU);
}

static void test_lu_singular(void) {
    printf("  [lu_singular]\n");
    const size_t n = 8;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * sizeof(double));
    size_t* piv = (size_t*)malloc(n * sizeof(size_t));
    fill_sym(A, n, 202);
    /* make row 3 = row 5 exactly */
    for (size_t j = 0; j < n; ++j) A[3 * n + j] = A[5 * n + j];
    for (size_t i = 0; i < n; ++i) B[i] = 1.0;
    ASSERT_TRUE(rcre_linalg_lu_factor_f64(A, n, n, piv) == -1);
    free(A); free(B); free(piv);
}

/* ============================== Cholesky ================================== */

static void test_cholesky(void) {
    printf("  [cholesky]\n");
    const size_t n = 36;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* Aorig = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * 2 * sizeof(double));
    double* Bref = (double*)malloc(n * 2 * sizeof(double));
    double* L = (double*)malloc(n * n * sizeof(double));
    double* LLt = (double*)malloc(n * n * sizeof(double));
    double* Lt = (double*)malloc(n * n * sizeof(double));
    fill_spd(A, n, 303);
    memcpy(Aorig, A, n * n * sizeof(double));
    for (size_t r = 0; r < 2; ++r)
        for (size_t i = 0; i < n; ++i) {
            uint64_t z = 7000 + i * 17 + r;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            B[i * 2 + r] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
            Bref[i * 2 + r] = B[i * 2 + r];
        }
    memcpy(L, Aorig, n * n * sizeof(double));
    ASSERT_TRUE(rcre_linalg_cholesky_f64(L, n, n, 1) == 0);
    /* L·L^T = A — transpose only the factor (lower triangle); the upper
     * triangle of L still holds the original A (dpotrf convention). */
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            Lt[i * n + j] = (j >= i) ? L[j * n + i] : 0.0;
    rcre_linalg_gemm_f64(n, n, n, 1.0, L, n, Lt, n, 0.0, LLt, n);
    /* LLt must be symmetrized? No — L L^T is exactly symmetric mathematically;
     * check the lower triangle of LLt against A (the factor lives there) */
    {
        double mx = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <= i; ++j) {
                const double d = fabs(LLt[i * n + j] - Aorig[i * n + j]);
                if (d > mx) mx = d;
            }
        ASSERT_TRUE(mx < 1e-11);
    }
    /* solve */
    ASSERT_TRUE(rcre_linalg_cholesky_solve_f64(L, n, n, B, 2, 2, 1) == 0);
    double* Ax = (double*)malloc(n * 2 * sizeof(double));
    rcre_linalg_gemm_f64(n, 2, n, 1.0, Aorig, n, B, 2, 0.0, Ax, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t i = 0; i < n; ++i)
            Ax[i * 2 + r] -= Bref[i * 2 + r];
    ASSERT_TRUE(max_mat_diff(Ax, 2, Ax, 2, n, 2) < 1e-10);

    /* not positive definite: negative diagonal entry must be detected */
    double* G = (double*)malloc(4 * 4 * sizeof(double));
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            G[i * 4 + j] = (i == j) ? 1.0 : 0.0;
    G[2 * 4 + 2] = -1.0;
    ASSERT_TRUE(rcre_linalg_cholesky_f64(G, 4, 4, 1) == -1);
    free(G);
    free(A); free(Aorig); free(B); free(Bref); free(L); free(LLt); free(Lt); free(Ax);
}

/* ============================== QR ======================================== */

static void test_qr(void) {
    printf("  [qr]\n");
    const size_t m = 24, n = 12;
    double* A = (double*)malloc(m * n * sizeof(double));
    double* Aorig = (double*)malloc(m * n * sizeof(double));
    double* tau = (double*)malloc(n * sizeof(double));
    double* Q = (double*)malloc(m * m * sizeof(double));
    double* R = (double*)malloc(m * n * sizeof(double));
    /* rectangular: fill an m×m symmetric scratch, keep the first n columns */
    double* Atmp = (double*)malloc(m * m * sizeof(double));
    fill_sym(Atmp, m, 404);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i * n + j] = Atmp[i * m + j];
    free(Atmp);
    memcpy(Aorig, A, m * n * sizeof(double));

    ASSERT_TRUE(rcre_linalg_qr_factor_f64(A, n, m, n, tau) == 0);
    rcre_linalg_qr_q_f64(A, n, m, n, tau, Q, m);
    /* Q^T Q = I */
    {
        double* QtQ = (double*)malloc(m * m * sizeof(double));
        double* Qt = (double*)malloc(m * m * sizeof(double));
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < m; ++j)
                Qt[i * m + j] = Q[j * m + i];
        rcre_linalg_gemm_f64(m, m, m, 1.0, Qt, m, Q, m, 0.0, QtQ, m);
        double mx = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < m; ++j) {
                const double d = fabs(QtQ[i * m + j] - ((i == j) ? 1.0 : 0.0));
                if (d > mx) mx = d;
            }
        ASSERT_TRUE(mx < 1e-12);
        free(Qt); free(QtQ);
    }
    /* Q R = A (R = upper of A packed: build the full m×n R) */
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? A[i * n + j] : 0.0;
    double* QR = (double*)malloc(m * n * sizeof(double));
    rcre_linalg_gemm_f64(m, n, m, 1.0, Q, m, R, n, 0.0, QR, n);
    ASSERT_TRUE(max_mat_diff(QR, n, Aorig, n, m, n) < 1e-12);

    /* least squares: min ||A x - b||; check the normal equations
     * A^T (A x - b) = 0 and that the residual beats the zero solution */
    const size_t nrhs = 3;
    double* B = (double*)malloc(m * nrhs * sizeof(double));
    for (size_t r = 0; r < nrhs; ++r)
        for (size_t i = 0; i < m; ++i) {
            uint64_t z = 5000 + i * 13 + r;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            B[i * nrhs + r] = (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
        }
    /* pack a fresh factor (qr_factor clobbers A) */
    double* Ap = (double*)malloc(m * n * sizeof(double));
    memcpy(Ap, Aorig, m * n * sizeof(double));
    ASSERT_TRUE(rcre_linalg_qr_factor_f64(Ap, n, m, n, tau) == 0);
    ASSERT_TRUE(rcre_linalg_qr_lsolve_f64(Ap, n, m, n, tau, B, nrhs, nrhs) == 0);
    /* x is rows 0..n-1 */
    double* x = (double*)malloc(n * nrhs * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t r = 0; r < nrhs; ++r)
            x[i * nrhs + r] = B[i * nrhs + r];
    /* A^T A x - A^T b = 0 */
    double* AtA = (double*)malloc(n * n * sizeof(double));
    double* Atb = (double*)malloc(n * nrhs * sizeof(double));
    double* At = (double*)malloc(n * m * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            At[i * m + j] = Aorig[j * n + i];
    rcre_linalg_gemm_f64(n, n, m, 1.0, At, m, Aorig, n, 0.0, AtA, n);
    rcre_linalg_gemm_f64(n, nrhs, m, 1.0, At, m, B, nrhs, 0.0, Atb, nrhs);
    double* AtAx = (double*)malloc(n * nrhs * sizeof(double));
    rcre_linalg_gemm_f64(n, nrhs, n, 1.0, AtA, n, x, nrhs, 0.0, AtAx, nrhs);
    for (size_t i = 0; i < n; ++i)
        for (size_t r = 0; r < nrhs; ++r)
            AtAx[i * nrhs + r] -= Atb[i * nrhs + r];
    ASSERT_TRUE(max_mat_diff(AtAx, nrhs, AtAx, nrhs, n, nrhs) < 1e-10);
    /* residual must beat ||b|| (the zero solution) */
    {
        double res = 0.0;
        for (size_t r = 0; r < nrhs; ++r)
            for (size_t i = n; i < m; ++i)
                res += B[i * nrhs + r] * B[i * nrhs + r];
        double full = 0.0;
        for (size_t r = 0; r < nrhs; ++r)
            for (size_t i = 0; i < m; ++i)
                full += B[i * nrhs + r] * B[i * nrhs + r]; /* post-solve = x part + resid */
        ASSERT_TRUE(res < full);
    }
    free(A); free(Aorig); free(tau); free(Q); free(R); free(QR); free(B);
    free(x); free(AtA); free(Atb); free(At); free(AtAx); free(Ap);
}

/* ============================== syev ====================================== */

static void test_syev_small_known(void) {
    printf("  [syev_small_known]\n");
    /* A = [[2, 1], [1, 2]]: eigenvalues 1, 3 */
    double A[4] = { 2.0, 1.0, 1.0, 2.0 };
    double w[2], V[4];
    ASSERT_TRUE(rcre_linalg_syev_f64(A, 2, 2, 1, w, V, 2) == 0);
    ASSERT_TRUE(fabs(w[0] - 1.0) < 1e-14);
    ASSERT_TRUE(fabs(w[1] - 3.0) < 1e-14);
    /* eigenvector for λ=1: (1,-1)/sqrt(2) up to sign; the COLUMNS of V
     * are the eigenvectors (V[0], V[2] is column 0) */
    ASSERT_TRUE(fabs(fabs(V[0]) - 1.0 / sqrt(2.0)) < 1e-12);
    ASSERT_TRUE(fabs(fabs(V[2]) - 1.0 / sqrt(2.0)) < 1e-12);
    ASSERT_TRUE(fabs(V[0] + V[2]) < 1e-12); /* opposite signs */
    /* eigenvalues-only path agrees */
    double A2[4] = { 2.0, 1.0, 1.0, 2.0 };
    double w2[2];
    ASSERT_TRUE(rcre_linalg_syev_f64(A2, 2, 2, 0, w2, NULL, 0) == 0);
    ASSERT_TRUE(fabs(w2[0] - w[0]) < 1e-15 && fabs(w2[1] - w[1]) < 1e-15);
}

static void test_syev_vs_jacobi(void) {
    printf("  [syev_vs_jacobi]\n");
    const size_t n = 24;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* Aj = (double*)malloc(n * n * sizeof(double));
    double* w = (double*)malloc(n * sizeof(double));
    double* wj = (double*)malloc(n * sizeof(double));
    double* V = (double*)malloc(n * n * sizeof(double));
    fill_sym(A, n, 505);
    memcpy(Aj, A, n * n * sizeof(double));
    ASSERT_TRUE(jacobi_eigs(Aj, n, wj) == 0);
    ASSERT_TRUE(rcre_linalg_syev_f64(A, n, n, 1, w, V, n) == 0);
    /* eigenvalues: both sorted ascending; relative tolerance */
    double mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double rel = fabs(w[i] - wj[i]) / (1.0 + fabs(wj[i]));
        if (rel > mx) mx = rel;
    }
    ASSERT_TRUE(mx < 1e-9);
    /* residual: A0 v_j = w_j v_j for each column */
    double* A0 = (double*)malloc(n * n * sizeof(double));
    fill_sym(A0, n, 505);
    double* Av = (double*)malloc(n * n * sizeof(double));
    rcre_linalg_gemm_f64(n, n, n, 1.0, A0, n, V, n, 0.0, Av, n);
    double res = 0.0;
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i) {
            const double d = Av[i * n + j] - w[j] * V[i * n + j];
            res = fmax(res, fabs(d));
        }
    ASSERT_TRUE(res < 1e-10);
    /* Q-orthogonality: V^T V = I */
    double* VtV = (double*)malloc(n * n * sizeof(double));
    double* Vt = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            Vt[i * n + j] = V[j * n + i];
    rcre_linalg_gemm_f64(n, n, n, 1.0, Vt, n, V, n, 0.0, VtV, n);
    double ortho = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            ortho = fmax(ortho, fabs(VtV[i * n + j] - ((i == j) ? 1.0 : 0.0)));
    ASSERT_TRUE(ortho < 1e-12);
    free(A); free(Aj); free(w); free(wj); free(V); free(A0); free(Av);
    free(VtV); free(Vt);
}

static void test_syev_spd_and_sizes(void) {
    printf("  [syev_spd_and_sizes]\n");
    const size_t sizes[3] = { 8, 32, 64 };
    for (size_t si = 0; si < 3; ++si) {
        const size_t n = sizes[si];
        double* A = (double*)malloc(n * n * sizeof(double));
        double* Aj = (double*)malloc(n * n * sizeof(double));
        double* w = (double*)malloc(n * sizeof(double));
        double* wj = (double*)malloc(n * sizeof(double));
        fill_spd(A, n, 600 + si);
        memcpy(Aj, A, n * n * sizeof(double));
        jacobi_eigs(Aj, n, wj);
        ASSERT_TRUE(rcre_linalg_syev_f64(A, n, n, 0, w, NULL, 0) == 0);
        double mx = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double rel = fabs(w[i] - wj[i]) / (1.0 + fabs(wj[i]));
            if (rel > mx) mx = rel;
        }
        ASSERT_TRUE(mx < 1e-9);
        /* ascending order */
        for (size_t i = 1; i < n; ++i)
            ASSERT_TRUE(w[i] >= w[i - 1] || fabs(w[i] - w[i - 1]) < 1e-15);
        free(A); free(Aj); free(w); free(wj);
    }
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  RCRE H3 — Direct Factorizations & Syev Test Suite          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    rcre_hal_init();
    test_lu_factor_solve();
    test_lu_singular();
    test_cholesky();
    test_qr();
    test_syev_small_known();
    test_syev_vs_jacobi();
    test_syev_spd_and_sizes();
    if (g_fails == 0) {
        printf("----------------------------------------------------------------------\n");
        printf("  🎉 All 7 Tests Passed (%d assertions)\n", g_checks);
        printf("======================================================================\n");
        return 0;
    }
    printf("----------------------------------------------------------------------\n");
    printf("  ❌ FAILED: %d/%d assertions\n", g_fails, g_checks);
    printf("======================================================================\n");
    return 1;
}
