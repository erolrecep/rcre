/**
 * @file gpu_stub.c
 * @brief Non-macOS GPU Compute Fallback Bridge (Extras Layer, ISO C23)
 */

#include "extras/rcre_gpu.h"

RCRE_NODISCARD bool rcre_gpu_is_available(void) {
    return false;
}

RCRE_NODISCARD const char* rcre_gpu_device_name(void) {
    return "CPU HAL Fallback";
}

RCRE_NODISCARD int rcre_gpu_gemm_f32(
    size_t M, size_t N, size_t K,
    float alpha,
    const float* A, size_t lda,
    const float* B, size_t ldb,
    float beta,
    float* C, size_t ldc
) {
    if (!A || !B || !C || M == 0 || N == 0 || K == 0) return RCRE_ERR_INVALID_ARG;
    rcre_hal_init();
    rcre_hal.gemm_f32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return RCRE_OK;
}
