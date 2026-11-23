/**
 * @file linalg.c
 * @brief Public dispatched linalg entry points (H0, 2.1.0)
 *
 * Thin wrappers over the rcre_hal dispatch table: the algorithm layer
 * calls rcre_linalg_*_f64/f32 and gets whatever ISA the host probed
 * (scalar / NEON / AVX2 / AVX-512). No algorithm in the library (or in
 * user scientific code) needs to know or branch on the ISA — a cell
 * missing at the vector level falls back to the scalar default, so the
 * API surface is complete on every platform.
 *
 * Determinism: single-threaded, fixed accumulation order — bitwise
 * reproducible run-to-run (docs/hpc.md).
 */

#include "rcre.h"

void rcre_linalg_gemm_f32(size_t M, size_t N, size_t K, float alpha,
                          const float* A, size_t lda, const float* B, size_t ldb,
                          float beta, float* C, size_t ldc) {
    rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void rcre_linalg_gemm_f64(size_t M, size_t N, size_t K, double alpha,
                          const double* A, size_t lda, const double* B, size_t ldb,
                          double beta, double* C, size_t ldc) {
    rcre_hal.gemm_f64(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void rcre_linalg_gemv_f32(size_t M, size_t N, float alpha,
                          const float* A, size_t lda, const float* x,
                          float beta, float* y) {
    rcre_hal.gemv_f32(M, N, alpha, A, lda, x, beta, y);
}

void rcre_linalg_gemv_f64(size_t M, size_t N, double alpha,
                          const double* A, size_t lda, const double* x,
                          double beta, double* y) {
    rcre_hal.gemv_f64(M, N, alpha, A, lda, x, beta, y);
}

void rcre_linalg_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    rcre_hal.axpy_f32(n, alpha, x, y);
}

void rcre_linalg_axpy_f64(size_t n, double alpha, const double* x, double* y) {
    rcre_hal.axpy_f64(n, alpha, x, y);
}

void rcre_linalg_scale_f32(size_t n, float alpha, float* x) {
    rcre_hal.scale_f32(n, alpha, x);
}

void rcre_linalg_scale_f64(size_t n, double alpha, double* x) {
    rcre_hal.scale_f64(n, alpha, x);
}

float rcre_linalg_dot_f32(const float* x, const float* y, size_t n) {
    return rcre_hal.dot_f32(x, y, n);
}

double rcre_linalg_dot_f64(const double* x, const double* y, size_t n) {
    return rcre_hal.dot_f64(x, y, n);
}

float rcre_linalg_sum_f32(const float* x, size_t n) {
    return rcre_hal.sum_f32(x, n);
}

double rcre_linalg_sum_f64(const double* x, size_t n) {
    return rcre_hal.sum_f64(x, n);
}

/* ========================================================================= */
/* H2 (2.2.0) — Blocked GEMM + triangular solves                             */
/* ========================================================================= */

/*
 * One-sided (A-panel) blocked GEMM.
 *
 * Loop order (i, kk, j): for each A row-block and K-slab, sweep the
 * whole N range so that A(i0:i0+BM, kk0:kk0+BK) — BM*BK doubles — is
 * the resident micropanel while B/C tiles stream through. For BM =
 * BN = BK = 64 that panel is 32 KB: L1-resident on every modern core,
 * which is the entire point of BLIS's micro-kernel structure. The
 * micro-panel calls dispatch through rcre_hal.gemm_f64, so the vector
 * ISA still does the inner work.
 *
 * Determinism: fixed loop order — reproducible. The K accumulation
 * order differs from the unblocked path, so ulps may differ (float
 * determinism class, docs/hpc.md §2.2).
 */
void rcre_linalg_gemm_blocked_f64(size_t M, size_t N, size_t K, double alpha,
                                  const double* A, size_t lda, const double* B, size_t ldb,
                                  double beta, double* C, size_t ldc) {
    if (M == 0 || N == 0) return;
    /* K == 0: C = beta*C only */
    if (K == 0) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i * ldc + j] *= beta;
        return;
    }
    const size_t BM = 64, BN = 64, BK = 64;

    for (size_t i0 = 0; i0 < M; i0 += BM) {
        const size_t im = (i0 + BM < M) ? BM : M - i0;
        for (size_t k0 = 0; k0 < K; k0 += BK) {
            const size_t kk = (k0 + BK < K) ? BK : K - k0;
            const double* a_panel = A + i0 * lda + k0;
            /* alpha multiplies every panel (alpha*sum_k A_k B_k);
             * beta applies only to the first panel (preserves beta*C0). */
            const double panel_beta = (k0 == 0) ? beta : 1.0;
            for (size_t j0 = 0; j0 < N; j0 += BN) {
                const size_t jm = (j0 + BN < N) ? BN : N - j0;
                /* C(i0:i0+im, j0:j0+jm) += A_panel @ B(k0:k0+kk, j0:j0+jm) */
                rcre_hal.gemm_f64(im, jm, kk, alpha,
                                  a_panel, lda,
                                  B + k0 * ldb + j0, ldb,
                                  panel_beta,
                                  C + i0 * ldc + j0, ldc);
            }
        }
    }
}

/*
 * Triangular solve (H2). Row-major conventions:
 *   A row-major n x n, A(m, k) = A[m*lda + k].
 *   X row-major n x nrhs, X(m, r) = X[m*ldx + r].
 * Solves A*X = alpha*X (or A^T*X = alpha*X when trans).
 *
 * uplo: 0 = lower, 1 = upper. diag: 0 = unit, 1 = general.
 *
 * Lower, non-trans, general diag — forward substitution:
 *   for m: x[m] = alpha*(x[m] - sum_{k<m} A[m,k] x[k]) / A[m,m]
 * Upper, non-trans — backward substitution:
 *   for m from n-1: x[m] = alpha*(x[m] - sum_{k>m} A[m,k] x[k]) / A[m,m]
 * Transposed variants solve with A^T (swap the loop range; the sum
 * runs over the same stored triangle: (A^T x)[m] = sum_k A[k,m] x[k]).
 *
 * The inner products are plain sequential F64 (deterministic, fixed
 * order). No pivoting: a triangular matrix's only singularity is a
 * zero diagonal, which is detected and reported (-1).
 */
int rcre_linalg_trsm_f64(int uplo, int trans, int diag, double alpha,
                         const double* A, size_t lda,
                         double* X, size_t ldx, size_t n, size_t nrhs) {
    /* ROW-MAJOR X (RCRE convention): column r is X[m*ldx + r], m = 0..n-1.
     * (BLAS trsm is column-major; the ldx here is the row stride.) */
    for (size_t r = 0; r < nrhs; ++r) {
        const int lower = (uplo == 0);
        const int forward = lower != trans; /* lower or upper-trans => forward */
        if (forward) {
            for (size_t m = 0; m < n; ++m) {
                /* x[m] = (alpha*X0[m] - sum_{k<m} A[m,k] x[k]) / A[m,m]
                 * — alpha scales the RHS ONLY, never the substitution */
                double s = 0.0;
                const double* a_row = A + m * lda;
                if (lower) {
                    for (size_t k = 0; k < m; ++k) s -= a_row[k] * X[k * ldx + r];
                } else {
                    /* upper triangle, solving with A^T: (A^T x)[m] uses
                     * A[k,m] for k < m — the stored column m, rows < m */
                    for (size_t k = 0; k < m; ++k) s -= A[k * lda + m] * X[k * ldx + r];
                }
                const double d = diag ? (lower ? a_row[m] : A[m * lda + m]) : 1.0;
                if (d == 0.0) return -1;
                X[m * ldx + r] = (alpha * X[m * ldx + r] + s) / d;
            }
        } else {
            for (size_t mi = 0; mi < n; ++mi) {
                const size_t m = n - 1 - mi;
                /* x[m] = (alpha*X0[m] - sum_{k>m} A[m,k] x[k]) / A[m,m] */
                double s = 0.0;
                if (!lower) {
                    const double* a_row = A + m * lda;
                    for (size_t k = m + 1; k < n; ++k) s -= a_row[k] * X[k * ldx + r];
                } else {
                    /* lower triangle, solving with A^T: (A^T x)[m] uses
                     * A[k,m] for k > m — stored column m, rows > m */
                    for (size_t k = m + 1; k < n; ++k) s -= A[k * lda + m] * X[k * ldx + r];
                }
                const double d = diag ? (lower ? A[m * lda + m]
                                               : (A + m * lda)[m]) : 1.0;
                if (d == 0.0) return -1;
                X[m * ldx + r] = (alpha * X[m * ldx + r] + s) / d;
            }
        }
    }
    return 0;
}

int rcre_linalg_trsv_f64(int uplo, int trans, int diag, double alpha,
                         const double* A, size_t lda, double* x, size_t n) {
    return rcre_linalg_trsm_f64(uplo, trans, diag, alpha, A, lda, x, 1, n, 1);
}
