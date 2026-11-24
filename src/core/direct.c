/**
 * @file direct.c
 * @brief Direct factorizations and symmetric eigendecomposition, F64 (H3)
 *
 * LAPACK-class building blocks for the HPC scientific stack:
 *
 *  - LU with partial pivoting (dgetrf/dgetrs-class): factor + solve.
 *  - Cholesky (dpotrf/dpotrs-class) for symmetric positive-definite.
 *  - Householder QR (dgeqrf-class): factor, explicit Q, least squares.
 *  - Symmetric eigendecomposition (dsyev-class): Householder reduction
 *    to tridiagonal form + Wilkinson-shift implicit QL iteration on the
 *    tridiagonal, eigenvalues always, eigenvectors optional.
 *
 * Conventions: row-major throughout (RCRE), F64, deterministic plain
 * sequential (fixed loop order). Householder reflectors use the scaled
 * storage form H = I - tau*u*u^T with u[0] = 1 and
 *
 *     tau = 2 / (u^T u)              (the exact Householder coefficient
 *                                     for the unnormalized scaled vector;
 *                                     note: NOT 2/(1 + u^T u) — that
 *                                     variant is wrong for m = 1 and
 *                                     was caught by the Q-orthogonality
 *                                     audit during H3 development).
 *
 * The tridiagonal trailing update is the exact 3-term HAH form
 * A' = A - tau (u (A u)^T + (A u) u^T) + tau^2 (u^T A u) u u^T
 * (verified against an explicit full-matrix HAH oracle to ~1e-16).
 */

#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* LU with partial pivoting                                                  */
/* ========================================================================= */

int rcre_linalg_lu_factor_f64(double* A, size_t lda, size_t n, size_t* piv) {
    for (size_t k = 0; k < n; ++k) {
        size_t p = k;
        double best = fabs(A[k * lda + k]);
        for (size_t i = k + 1; i < n; ++i) {
            const double m = fabs(A[i * lda + k]);
            if (m > best) { best = m; p = i; }
        }
        piv[k] = p;
        if (p != k) {
            /* swap the ENTIRE row, including the L multipliers in
             * columns 0..k-1 — skipping them breaks the P consistency
             * (PA = LU) — caught by the n = 3 audit. */
            for (size_t j = 0; j < n; ++j) {
                const double t = A[k * lda + j];
                A[k * lda + j] = A[p * lda + j];
                A[p * lda + j] = t;
            }
        }
        if (best == 0.0) return -1;
        const double inv = 1.0 / A[k * lda + k];
        for (size_t i = k + 1; i < n; ++i) {
            const double m = A[i * lda + k] * inv;
            A[i * lda + k] = m;
            double* a_i = A + i * lda;
            const double* a_k = A + k * lda;
            for (size_t j = k + 1; j < n; ++j)
                a_i[j] -= m * a_k[j];
        }
    }
    return 0;
}

int rcre_linalg_lu_solve_f64(const double* A, size_t lda, const size_t* piv,
                             size_t n, double* B, size_t ldx, size_t nrhs) {
    for (size_t k = 0; k < n; ++k) {
        const size_t p = piv[k];
        if (p != k)
            for (size_t r = 0; r < nrhs; ++r) {
                const double t = B[k * ldx + r];
                B[k * ldx + r] = B[p * ldx + r];
                B[p * ldx + r] = t;
            }
    }
    for (size_t i = 1; i < n; ++i)
        for (size_t r = 0; r < nrhs; ++r) {
            double s = B[i * ldx + r];
            const double* a_i = A + i * lda;
            for (size_t k = 0; k < i; ++k)
                s -= a_i[k] * B[k * ldx + r];
            B[i * ldx + r] = s;
        }
    for (size_t i = n; i-- > 0;) {
        const size_t ii = i;
        for (size_t r = 0; r < nrhs; ++r) {
            double s = B[ii * ldx + r];
            const double* a_i = A + ii * lda;
            for (size_t k = ii + 1; k < n; ++k)
                s -= a_i[k] * B[k * ldx + r];
            B[ii * ldx + r] = s / a_i[ii];
        }
    }
    return 0;
}

int rcre_linalg_solve_f64(const double* A, size_t lda, size_t n,
                          double* B, size_t ldx, size_t nrhs) {
    if (lda != n) return -1; /* square packed input only */
    double* work = (double*)malloc(n * n * sizeof(double));
    size_t* piv = (size_t*)malloc(n * sizeof(size_t));
    memcpy(work, A, n * n * sizeof(double));
    int rc = rcre_linalg_lu_factor_f64(work, n, n, piv);
    if (rc == 0) rc = rcre_linalg_lu_solve_f64(work, n, piv, n, B, ldx, nrhs);
    free(work);
    free(piv);
    return rc;
}

/* ========================================================================= */
/* Cholesky (SPD)                                                            */
/* ========================================================================= */

int rcre_linalg_cholesky_f64(double* A, size_t lda, size_t n, int lower) {
    if (lower) {
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j; i < n; ++i) {
                double s = A[i * lda + j];
                for (size_t k = 0; k < j; ++k)
                    s -= A[i * lda + k] * A[j * lda + k];
                if (i == j) {
                    if (s <= 0.0) return -1;
                    A[i * lda + j] = sqrt(s);
                } else {
                    A[i * lda + j] = s / A[j * lda + j];
                }
            }
    } else {
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j; i < n; ++i) {
                double s = A[i * lda + j];
                for (size_t k = 0; k < j; ++k)
                    s -= A[k * lda + i] * A[k * lda + j];
                if (i == j) {
                    if (s <= 0.0) return -1;
                    A[i * lda + j] = sqrt(s);
                } else {
                    A[i * lda + j] = s / A[j * lda + j];
                }
            }
    }
    return 0;
}

/**
 * @brief Solve A X = B from a Cholesky factor (row-major, multi-RHS).
 * Lower-packed L: L y = B (forward) then L^T x = y (lower, transposed).
 * Upper-packed L^T: (L^T) y = B (upper) then L x = y (upper, transposed).
 * Uses the H2 trsm primitive.
 */
int rcre_linalg_cholesky_solve_f64(const double* F, size_t lda, size_t n,
                                   double* B, size_t ldx, size_t nrhs, int lower) {
    if (lower) {
        if (rcre_linalg_trsm_f64(0, 0, 0, 1.0, F, lda, B, ldx, n, nrhs) != 0)
            return -1;
        if (rcre_linalg_trsm_f64(0, 1, 0, 1.0, F, lda, B, ldx, n, nrhs) != 0)
            return -1;
    } else {
        if (rcre_linalg_trsm_f64(1, 0, 0, 1.0, F, lda, B, ldx, n, nrhs) != 0)
            return -1;
        if (rcre_linalg_trsm_f64(1, 1, 0, 1.0, F, lda, B, ldx, n, nrhs) != 0)
            return -1;
    }
    return 0;
}

/* ========================================================================= */
/* Householder QR                                                            */
/* ========================================================================= */

int rcre_linalg_qr_factor_f64(double* A, size_t lda, size_t m, size_t n,
                              double* tau) {
    const size_t kmax = (m < n) ? m : n;
    for (size_t k = 0; k < kmax; ++k) {
        double norm = 0.0;
        for (size_t i = k; i < m; ++i) {
            const double v = A[i * lda + k];
            norm += v * v;
        }
        norm = sqrt(norm);
        if (norm == 0.0) { tau[k] = 0.0; continue; }
        const double x0 = A[k * lda + k];
        double beta = (x0 >= 0.0) ? -norm : norm;
        if (x0 == 0.0) beta = -norm;
        const double v0 = x0 - beta;
        A[k * lda + k] = beta;
        for (size_t i = k + 1; i < m; ++i)
            A[i * lda + k] /= v0;
        /* tau = 2 / (u^T u), u[0] = 1 (scaled storage) */
        double uu = 1.0;
        for (size_t i = k + 1; i < m; ++i) {
            const double v = A[i * lda + k];
            uu += v * v;
        }
        tau[k] = 2.0 / uu;
        for (size_t j = k + 1; j < n; ++j) {
            double dot = A[k * lda + j]; /* u[0] = 1 term */
            for (size_t i = k + 1; i < m; ++i)
                dot += A[i * lda + k] * A[i * lda + j];
            dot *= tau[k];
            A[k * lda + j] -= dot;
            for (size_t i = k + 1; i < m; ++i)
                A[i * lda + j] -= dot * A[i * lda + k];
        }
    }
    return 0;
}

/**
 * @brief Explicit Q (m×m row-major) from a packed QR factorization.
 * The factorization computes R = H_{kmax-1} ... H_0 A (left application
 * in increasing k), so A = Q R with Q = H_0 H_1 ... H_{kmax-1} — the
 * REVERSE product. We therefore accumulate Q <- Q * H_k (right
 * multiplication, applied in increasing k): (Q H_k)[i,j] = Q[i,j] -
 * tau * u_j * (Q[i, k:] u), j >= k. NOTE the row loop is outer: each
 * row's dot product is computed before that row's entries change.
 */
void rcre_linalg_qr_q_f64(const double* A, size_t lda, size_t m, size_t n,
                          const double* tau, double* Q, size_t ldq) {
    const size_t kmax = (m < n) ? m : n;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j)
            Q[i * ldq + j] = (i == j) ? 1.0 : 0.0;
    for (size_t k = 0; k < kmax; ++k) {
        if (tau[k] == 0.0) continue;
        for (size_t i = 0; i < m; ++i) {
            double dot = Q[i * ldq + k]; /* u[0] = 1 term */
            for (size_t l = k + 1; l < m; ++l)
                dot += Q[i * ldq + l] * A[l * lda + k];
            dot *= tau[k];
            Q[i * ldq + k] -= dot;
            for (size_t j = k + 1; j < m; ++j)
                Q[i * ldq + j] -= dot * A[j * lda + k];
        }
    }
}

int rcre_linalg_qr_lsolve_f64(const double* A, size_t lda, size_t m, size_t n,
                              const double* tau, double* B, size_t ldx, size_t nrhs) {
    const size_t kmax = (m < n) ? m : n;
    for (size_t k = 0; k < kmax; ++k) {
        if (tau[k] == 0.0) continue;
        for (size_t r = 0; r < nrhs; ++r) {
            double dot = B[k * ldx + r];
            for (size_t i = k + 1; i < m; ++i)
                dot += A[i * lda + k] * B[i * ldx + r];
            dot *= tau[k];
            B[k * ldx + r] -= dot;
            for (size_t i = k + 1; i < m; ++i)
                B[i * ldx + r] -= dot * A[i * lda + k];
        }
    }
    for (size_t i = n; i-- > 0;) {
        const size_t ii = i;
        for (size_t r = 0; r < nrhs; ++r) {
            double s = B[ii * ldx + r];
            const double* a_i = A + ii * lda;
            for (size_t k = ii + 1; k < n; ++k)
                s -= a_i[k] * B[k * ldx + r];
            B[ii * ldx + r] = (a_i[ii] == 0.0) ? s : s / a_i[ii];
        }
    }
    return 0;
}

/* ========================================================================= */
/* Symmetric eigendecomposition                                              */
/* ========================================================================= */

/**
 * @brief Householder reduction to tridiagonal (internal).
 * A (n×n row-major, in) — the trailing blocks are overwritten by the
 * reduction; d (n, out) = diagonal, e (n-1, out) = first off-diagonal
 * (e[n-1] set to 0 if the array has length n). Q (n×n row-major, in/out)
 * accumulates the reflector product Q = H_{n-2} ... H_0 so that
 * Q * A0 * Q^T = tridiag(d, e); if Q is NULL the vectors are discarded.
 */
static int sytrd_f64(double* A, size_t lda, size_t n, double* d, double* e,
                     double* Q, size_t ldq) {
    if (Q != NULL) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                Q[i * ldq + j] = (i == j) ? 1.0 : 0.0;
    }
    for (size_t k = 0; k + 1 < n; ++k) {
        const size_t m = n - k - 1;
        double nn = 0.0;
        for (size_t i = k + 1; i < n; ++i) {
            const double v = A[i * lda + k];
            nn += v * v;
        }
        nn = sqrt(nn);
        const double x0 = A[(k + 1) * lda + k];
        d[k] = A[k * lda + k]; /* unchanged by the reduction */
        if (nn == 0.0) { e[k] = 0.0; continue; }
        double sigma = (x0 >= 0.0) ? -nn : nn;
        if (x0 == 0.0) sigma = -nn;
        const double v0 = x0 - sigma;
        /* store the scaled Householder vector in the sub-diagonal:
         * A[k+1, k] := u[0] = 1, A[k+2:, k] := x[i]/v0.
         * (FORGETTING to write A[k+1,k] = 1 leaves x0 there, corrupting
         * uu, W, and the Q accumulation — caught by the n = 2 audit.) */
        A[(k + 1) * lda + k] = 1.0;
        for (size_t i = k + 2; i < n; ++i)
            A[i * lda + k] /= v0;
        /* uu = u^T u over the WHOLE stored vector (u[0] = 1 is A[k+1,k]) */
        double uu = 0.0;
        for (size_t i = k + 1; i < n; ++i) {
            const double v = A[i * lda + k];
            uu += v * v;
        }
        const double tau = 2.0 / uu;
        /* e[k] = (H A H)[k, k+1] = A[k,k+1] - tau * (row_k · u).
         * u = (1, x[v0-scaled]): the first term uses the ORIGINAL
         * A[k,k+1] (u[0] = 1); the rest read the now-scaled column.
         * rowk = A[k,k+1] + sum_{l>k+1} A[k,l] * (A[l,k]/v0) */
        double rowk = A[k * lda + k + 1];
        for (size_t l = k + 2; l < n; ++l)
            rowk += A[k * lda + l] * A[l * lda + k];
        e[k] = A[k * lda + k + 1] - tau * rowk;
        /* trailing: W = (trailing A) u; A' = A - tau(Wu^T + uW^T)
         *          + tau^2 (u^T W) u u^T  (exact HAH) */
        double* W = (double*)malloc(m * sizeof(double));
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t l = 0; l < m; ++l)
                s += A[(k + 1 + j) * lda + (k + 1 + l)] * A[(k + 1 + l) * lda + k];
            W[j] = s;
        }
        double uW = 0.0;
        for (size_t i = 0; i < m; ++i)
            uW += A[(k + 1 + i) * lda + k] * W[i];
        for (size_t i = 0; i < m; ++i) {
            const double ui = A[(k + 1 + i) * lda + k];
            for (size_t j = i; j < m; ++j) {
                const double uj = A[(k + 1 + j) * lda + k];
                const double v = A[(k + 1 + i) * lda + (k + 1 + j)]
                               - tau * (W[i] * uj + ui * W[j])
                               + tau * tau * uW * ui * uj;
                A[(k + 1 + i) * lda + (k + 1 + j)] = v;
                A[(k + 1 + j) * lda + (k + 1 + i)] = v;
            }
        }
        free(W);
        /* Q <- Q * H_k (RIGHT multiplication, increasing k):
         * on return Q = H_0 H_1 ... H_{n-2} = L^T, so that with the
         * QL eigenvector V_T: A = (L^T V_T) D (L^T V_T)^T. Each row i:
         * q[i, k:m] <- q[i, k:m] - tau * u * (q[i, k:m] . u), u in the
         * stored column k (u[0] = 1 implicit). */
        if (Q != NULL) {
            for (size_t i = 0; i < n; ++i) {
                double dot = Q[i * ldq + k + 1]; /* u[0] = 1 term */
                for (size_t l = k + 2; l < n; ++l)
                    dot += Q[i * ldq + l] * A[l * lda + k];
                dot *= tau;
                Q[i * ldq + k + 1] -= dot;
                for (size_t j = k + 2; j < n; ++j)
                    Q[i * ldq + j] -= dot * A[j * lda + k];
            }
        }
    }
    d[n - 1] = A[(n - 1) * lda + (n - 1)];
    if (n > 0) e[n - 1] = 0.0;
    return 0;
}

/**
 * @brief Wilkinson-shift implicit QL iteration on a tridiagonal
 * (d, e in/out, length n; e[n-1] unused).
 *
 * Each sweep computes the exact implicit step T+ = Q^T T Q where
 * Q = G_l^T ... G_{na-2}^T is the Givens product from the QR of
 * M = T - mu*I (mu = the Wilkinson shift from the trailing 2x2).
 *
 * The Givens follow from the QR evolution of M. Key facts (verified
 * numerically): the z-value at step i is the sweep-START sub-diagonal
 * e[i] (rotations never touch rows below i+1), and the x-value
 * (the running M diagonal) has the closed form
 *   x_l     = d_l - mu
 *   x_{i+1} = -s_i*(c_{i-1}*e[i]) + c_i*(d[i+1] - mu),  c_{l-1} := 1
 * (the super-diagonal of M before G_i is c_{i-1}*e[i], the original).
 *
 * Q is formed densely per sweep (O(n^2)); the new tridiagonal is
 * computed directly as T+ = Q^T T Q using only Q's columns:
 *   d'_i = T_ii*(Q_ii^2 + Q_{i-1,i}^2 + Q_{i+1,i}^2)
 *          + 2*T_i,i+1*Q_ii*Q_{i+1,i} + 2*T_i-1,i*Q_ii*Q_{i-1,i}
 *   e'_i = T_ii*Q_ii*Q_i,i+1 + T_i+1,i+1*(Q_i+1,i*Q_i,i+1 +
 *          Q_i,i+1*Q_i+1,i+1) + T_i,i+1*(Q_ii*Q_i+1,i+1 +
 *          Q_i+1,i*Q_i,i+1) + T_i-1,i-1*Q_i-1,i*Q_i-1,i+1
 *          + T_i-1,i*(Q_i-1,i*Q_i,i+1 + Q_ii*Q_i-1,i+1)
 * (T tridiagonal; Q rows below l are untouched identity rows).
 * In exact arithmetic a Wilkinson-shift sweep deflates the bottom
 * element; the 30n+10 sweep budget per block is the standard bound.
 *
 * Eigenvectors: S (the running similarity, T = S^T A S) is updated
 * S <- S*Q per sweep (Q = G_l^T ... G_{na-2}^T with G_l^T leftmost,
 * applied to the columns in ASCENDING i). When all sweeps finish, T
 * is diagonal and S = S0*Q_1 ... Q_k is exactly the eigenvector
 * matrix in the original coordinates (A*S = S*diag(w)).
 *
 * -1 on non-convergence.
 */
static int steqr_f64(double* d, double* e, size_t n,
                     double* S, size_t ldv, int compute_vectors) {
    const double EPS = 2.220446049250313e-16;
    double* cA  = (double*)malloc(n * sizeof(double));
    double* sA  = (double*)malloc(n * sizeof(double));
    double* esnap = (double*)malloc((n > 1 ? n : 1) * sizeof(double));
    double* Q   = (double*)malloc((n ? n : 1) * (n ? n : 1) * sizeof(double));
    double* B   = (double*)malloc((n ? n : 1) * (n ? n : 1) * sizeof(double));
    double* TP  = (double*)malloc((n ? n : 1) * (n ? n : 1) * sizeof(double));
    double* nd  = (double*)malloc(n * sizeof(double));
    double* ne  = (double*)malloc((n > 1 ? n : 1) * sizeof(double));
    int ok = 1;
    size_t na = n; /* active size (bottom na elements) */
    const long budget = 30L * (long)n + 10;
    while (na > 1) {
        /* deflation scan from the bottom: e[idx] connects d[idx],
         * d[idx+1]; the first (bottom) candidate is e[na-2]. */
        size_t l = 0;
        int found = 0;
        for (size_t i = na - 1; i > 1; --i) {
            const size_t idx = i - 1;
            if (e[idx] == 0.0 ||
                fabs(e[idx]) <= EPS * (fabs(d[idx]) + fabs(d[idx + 1]))) {
                e[idx] = 0.0;
                l = idx + 1; /* active block = [idx+1 .. na-1] */
                found = 1;
                break;
            }
        }
        if (found && l == na - 1) break;
        long iters = 0;
        for (;;) {
            if (++iters > budget) { ok = 0; break; }
            /* exact Wilkinson shift: the trailing 2x2's eigenvalue
             * closest to d[na-1] (stable form, no cancellation). */
            const double delta = d[na - 1] - d[na - 2];
            const double ev = e[na - 2];
            double mu;
            if (delta != 0.0) {
                const double g = delta / 2.0 +
                                  (delta > 0.0 ? 1.0 : -1.0) * hypot(delta / 2.0, ev);
                mu = d[na - 1] + ev * ev / g;
            } else {
                mu = d[na - 1] + fabs(ev);
            }
            /* snapshot the sweep-start sub-diagonal (the z-values) */
            for (size_t i = l; i + 1 < na; ++i) esnap[i] = e[i];
            /* the Givens of the QR of M = T - mu I */
            double x = d[l] - mu;
            double c_prev = 1.0;
            for (size_t i = l; i + 1 < na; ++i) {
                const double z = esnap[i];
                const double rho = hypot(x, z);
                double c = 1.0, s = 0.0;
                if (rho != 0.0) { c = x / rho; s = z / rho; }
                cA[i] = c;
                sA[i] = s;
                /* next x: -s_i*(c_{i-1}*e[i]) + c_i*(d[i+1]-mu) */
                x = -s * c_prev * esnap[i] + c * (d[i + 1] - mu);
                c_prev = c;
            }
            /* Q = G_l^T ... G_{na-2}^T (G_l^T leftmost): start from
             * the identity, apply G_i^T from the left in DECREASING
             * i (G_{na-2}^T first, G_l^T last). */
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    Q[i * n + j] = (i == j) ? 1.0 : 0.0;
            for (long i = (long)na - 2; i >= (long)l; --i) {
                const double c = cA[i], s = sA[i];
                for (size_t j = 0; j < n; ++j) {
                    const double a = Q[i * n + j];
                    const double b = Q[(i + 1) * n + j];
                    Q[i * n + j] = c * a - s * b;
                    Q[(i + 1) * n + j] = s * a + c * b;
                }
            }
            /* T+ = Q^T T Q. Q's band grows with the number of Givens, so
             * the column-local forms are invalid; compute exactly:
             *   B  = Q^T T  (O(n^2): T tridiagonal, each B column is a
             *          3-term sum over Q's columns)
             *   T+ = B * Q  (dense gemm, O(n^3) per sweep)
             * The result is tridiagonal in exact arithmetic (the bulge
             * closes); take the diagonal and the symmetrized
             * super-diagonal, discard the roundoff-level fill. */
            /* Active-block restriction: Q's rows/columns outside
             * [l, na) are exact identity rows (the Givens only touch
             * rows l..na-1) and T is block-diagonal there (deflated
             * e = 0), so the products over k outside [l, na) vanish.
             * Cost per sweep: O(m^3) with m = na-l, shrinking as the
             * eigenvalues deflate. */
            const size_t m = na - l;
            for (size_t i = l; i < na; ++i)
                for (size_t j = l; j < na; ++j) {
                    double a = 0.0;
                    a += Q[j * n + i] * d[j];
                    if (j + 1 < na) a += Q[(j + 1) * n + i] * e[j];
                    if (j > l) a += Q[(j - 1) * n + i] * e[j - 1];
                    B[i * n + j] = a;
                }
            rcre_linalg_gemm_f64(m, m, m, 1.0, B + l * n, n, Q + l * n, n,
                                 0.0, TP + l * n, n);
            for (size_t i = l; i < na; ++i)
                nd[i] = TP[i * n + i];
            for (size_t i = l; i + 1 < na; ++i)
                ne[i] = 0.5 * (TP[i * n + (i + 1)] + TP[(i + 1) * n + i]);
            for (size_t i = l; i < na; ++i) d[i] = nd[i];
            for (size_t i = l; i + 1 < na; ++i) e[i] = ne[i];
            /* S <- S*Q (= S*G_l^T ... G_{na-2}^T with G_l^T leftmost:
             * evaluate left-to-right, apply G_i^T to the columns in
             * ASCENDING i: S <- S*G_i^T per step). At the end of all
             * sweeps T is diagonal, so S = S0*Q_1 ... Q_k is exactly
             * the eigenvector matrix in the original coordinates. */
            if (compute_vectors && S != NULL) {
                for (size_t i = l; i + 1 < na; ++i) {
                    const double c = cA[i], s = sA[i];
                    for (size_t r = 0; r < n; ++r) {
                        const double a = S[r * ldv + i];
                        const double b = S[r * ldv + i + 1];
                        S[r * ldv + i] = c * a + s * b;
                        S[r * ldv + i + 1] = -s * a + c * b;
                    }
                }
            }
            if (e[na - 2] == 0.0 ||
                fabs(e[na - 2]) <= EPS * (fabs(d[na - 2]) + fabs(d[na - 1]))) {
                e[na - 2] = 0.0;
                na -= 1;
                break;
            }
        }
        if (!ok) break;
    }
    free(cA); free(sA); free(esnap); free(Q); free(B); free(TP); free(nd); free(ne);
    return ok ? 0 : -1;
}

int rcre_linalg_syev_f64(double* A, size_t lda, size_t n, int compute_vectors,
                         double* w, double* V, size_t ldv) {
    if (n == 0) return 0;
    double* e = (double*)malloc((n > 1 ? n : 1) * sizeof(double));
    int rc;
    if (compute_vectors) {
        /* S starts as S0 = Q_trd (T = S0^T A S0) and is updated
         * S <- S*Q_k per QL sweep. When T is finally diagonal,
         * S = S0*Q_1 ... Q_k is the eigenvector matrix:
         * A*S = S*diag(w). */
        double* S = (double*)malloc(n * n * sizeof(double));
        rc = sytrd_f64(A, lda, n, w, e, S, n);
        if (rc == 0) rc = steqr_f64(w, e, n, S, n, 1);
        if (rc == 0 && V != NULL)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    V[i * ldv + j] = S[i * n + j];
        free(S);
    } else {
        rc = sytrd_f64(A, lda, n, w, e, NULL, 0);
        if (rc == 0) rc = steqr_f64(w, e, n, NULL, 0, 0);
    }
    free(e);
    if (rc != 0) return rc;
    /* ascending sort, transporting eigenvector columns */
    for (size_t i = 0; i + 1 < n; ++i) {
        size_t p = i;
        for (size_t j = i + 1; j < n; ++j)
            if (w[j] < w[p]) p = j;
        if (p != i) {
            const double t = w[i];
            w[i] = w[p];
            w[p] = t;
            if (compute_vectors && V != NULL)
                for (size_t r = 0; r < n; ++r) {
                    const double t2 = V[r * ldv + i];
                    V[r * ldv + i] = V[r * ldv + p];
                    V[r * ldv + p] = t2;
                }
        }
    }
    return 0;
}
