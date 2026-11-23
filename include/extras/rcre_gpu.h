/**
 * @file rcre_gpu.h
 * @brief Optional Hardware GPU Offload Extras (Metal on macOS, WebGPU in browser, CPU fallback)
 *
 * Architectural boundary: Isolated in include/extras/ and src/extras/ so the
 * core C23 library in include/ and src/ remains strictly OS/vendor-header free.
 */

#ifndef RCRE_GPU_H
#define RCRE_GPU_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns true if a native GPU compute device (Metal) was initialized cleanly. */
RCRE_NODISCARD bool rcre_gpu_is_available(void);

/** Returns GPU device / fallback backend name (e.g. "Apple M5 Max (Metal)" or "CPU HAL Fallback"). */
RCRE_NODISCARD const char* rcre_gpu_device_name(void);

/**
 * @brief Executes C = alpha * A * B + beta * C on Metal GPU (or CPU HAL fallback).
 */
RCRE_NODISCARD int rcre_gpu_gemm_f32(
    size_t M, size_t N, size_t K,
    float alpha,
    const float* A, size_t lda,
    const float* B, size_t ldb,
    float beta,
    float* C, size_t ldc
);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_GPU_H */
