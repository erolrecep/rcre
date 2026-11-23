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
