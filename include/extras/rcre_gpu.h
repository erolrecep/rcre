/**
 * @file rcre_gpu.h
 * @brief Optional Hardware GPU Compute Extras (H6, 2.3.0)
 *
 * Architectural boundary (the D1 bridge pattern, proven by io/Arrow,
 * io/DLPack and the zstd codec): isolated in include/extras/ and
 * src/extras/ so the core C23 library in include/ and src/ stays
 * strictly OS/vendor-header free. The core never links a GPU backend;
 * user programs link the backend TUs they want:
 *
 *   gpu_ctx.c        always   — context plumbing + CPU reference backend
 *                              (zero-dependency, the conformance oracle:
 *                              its ops delegate to the same core/HAL
 *                              functions the CPU solver paths use, so
 *                              results are bit-identical to CPU)
 *   gpu_cuda.c/.cu   opt-in   — NVIDIA CUDA (nvcc; RCRE_WITH_CUDA)
 *   gpu_metal.m      opt-in   — Apple Metal (legacy single-kernel SGEMM)
 *
 * H6 determinism bar: kernels run with FIXED grid/block configurations
 * (never auto-tuned off device queries), so a launch is deterministic
 * on a given device. Cross-device bit-identity is NOT claimed for FP
 * ops (addition order differs from CPU) — the gates are precision
 * bars, except Philox, which is bit-identical by construction
 * (word j = pure function of counter j/4 and key).
 */

#ifndef RCRE_GPU_H
#define RCRE_GPU_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Legacy single-kernel API (pre-H6; Metal on macOS, CPU HAL fallback)       */
/* ========================================================================= */

/** Returns true if a native GPU compute device (Metal) was initialized cleanly. */
RCRE_NODISCARD bool rcre_gpu_is_available(void);

/** Returns GPU device / fallback backend name (e.g. "Apple M5 Max (Metal)" or "CPU HAL Fallback"). */
RCRE_NODISCARD const char* rcre_gpu_device_name(void);

/**
 * @brief Executes C = alpha * A * B + beta * C on Metal GPU (or CPU HAL fallback).
 * @note Legacy managed-mode API. New code should use the context API
 *       below (rcre_gpu_open_named + device-buffer ops).
 */
RCRE_NODISCARD int rcre_gpu_gemm_f32(
    size_t M, size_t N, size_t K,
    float alpha,
    const float* A, size_t lda,
    const float* B, size_t ldb,
    float beta,
    float* C, size_t ldc
);

/* ========================================================================= */
/* H6 context API (2.3.0)                                                    */
/* ========================================================================= */

/** Opaque device context (one per backend). */
typedef struct RCRE_GpuCtx RCRE_GpuCtx;
/** Opaque device buffer. */
typedef struct RCRE_GpuBuf RCRE_GpuBuf;


/* ========================================================================= */
/* Backend vtable (internal bridge contract — user code never fills this)    */
/* ========================================================================= */

/** Implementation of one backend. Opaque to callers; complete here so
 *  backend TUs (always-compiled gpu_ctx.c, opt-in gpu_cuda.c) can
 *  register themselves. */
typedef struct RCRE_GpuBackend {
    const char* name;
    int (*available)(void);
    const char* (*device_desc)(void);
    RCRE_GpuBuf* (*alloc)(RCRE_GpuCtx* ctx, size_t bytes);
    void (*free_buf)(RCRE_GpuBuf* buf);
    int (*h2d)(RCRE_GpuCtx* ctx, RCRE_GpuBuf* dst, const void* src, size_t bytes);
    int (*d2h)(RCRE_GpuCtx* ctx, void* dst, RCRE_GpuBuf* src, size_t bytes);
    int (*sync)(RCRE_GpuCtx* ctx);
    int (*gemm_f32)(RCRE_GpuCtx*, size_t m, size_t n, size_t k, float alpha,
                    RCRE_GpuBuf*, size_t lda, RCRE_GpuBuf*, size_t ldb,
                    float beta, RCRE_GpuBuf*, size_t ldc);
    int (*gemm_f64)(RCRE_GpuCtx*, size_t m, size_t n, size_t k, double alpha,
                    RCRE_GpuBuf*, size_t lda, RCRE_GpuBuf*, size_t ldb,
                    double beta, RCRE_GpuBuf*, size_t ldc);
    int (*spmv_f32)(RCRE_GpuCtx*, size_t rows, RCRE_GpuBuf*, RCRE_GpuBuf*,
                    RCRE_GpuBuf*, RCRE_GpuBuf*, RCRE_GpuBuf*);
    int (*spmv_f64)(RCRE_GpuCtx*, size_t rows, RCRE_GpuBuf*, RCRE_GpuBuf*,
                    RCRE_GpuBuf*, RCRE_GpuBuf*, RCRE_GpuBuf*);
    int (*axpy_f32)(RCRE_GpuCtx*, size_t n, float a, RCRE_GpuBuf*, RCRE_GpuBuf*);
    int (*axpy_f64)(RCRE_GpuCtx*, size_t n, double a, RCRE_GpuBuf*, RCRE_GpuBuf*);
    int (*dot_f32)(RCRE_GpuCtx*, size_t n, RCRE_GpuBuf*, RCRE_GpuBuf*, RCRE_GpuBuf*);
    int (*dot_f64)(RCRE_GpuCtx*, size_t n, RCRE_GpuBuf*, RCRE_GpuBuf*, RCRE_GpuBuf*);
    int (*philox_fill)(RCRE_GpuCtx*, size_t n, uint64_t c0, uint64_t key,
                       RCRE_GpuBuf*);
} RCRE_GpuBackend;

/**
 * @brief Internal: backend self-registration (called from backend TUs' C
 *  constructors). User code never calls this.
 */
void rcre_gpu_register_backend(const char* name, RCRE_GpuBackend* be);

/** @brief 1 if the named backend ("cuda", "metal", "cpu") is built in AND
 *  a device is present, 0 otherwise. Never fails on unknown names (0). */
RCRE_NODISCARD int rcre_gpu_backend_available(const char* backend);

/** @brief Open a context for the named backend (or NULL). The "cpu"
 *  backend is always available. */
RCRE_NODISCARD RCRE_GpuCtx* rcre_gpu_open_named(const char* backend);

/** @brief Open the first available backend: cuda -> metal -> cpu. */
/* Internal layout (complete here so backend TUs can touch the fields;
 * user code treats both as opaque). */
struct RCRE_GpuCtx {
    struct RCRE_GpuBackend* be;
    void* priv;   /* backend-private state (device handle, ...) */
};
struct RCRE_GpuBuf {
    RCRE_GpuCtx* ctx;
    size_t       bytes;
    void*        native;   /* backend device pointer (host ptr for cpu) */
};

RCRE_NODISCARD RCRE_GpuCtx* rcre_gpu_open(void);

void rcre_gpu_close(RCRE_GpuCtx* ctx);

/** @brief Backend name ("cuda" | "metal" | "cpu"). */
RCRE_NODISCARD const char* rcre_gpu_backend_name(const RCRE_GpuCtx* ctx);
/** @brief Device description (e.g. "NVIDIA GeForce RTX 5060 Ti (sm_120)"). */
RCRE_NODISCARD const char* rcre_gpu_device_desc(const RCRE_GpuCtx* ctx);

/* ---- buffers ------------------------------------------------------------- */

/** @brief Allocate `bytes` on the device (NULL on failure). */
RCRE_NODISCARD RCRE_GpuBuf* rcre_gpu_alloc(RCRE_GpuCtx* ctx, size_t bytes);
void rcre_gpu_free(RCRE_GpuBuf* buf);
/** @brief Number of bytes in the buffer. */
RCRE_NODISCARD size_t rcre_gpu_bytes(const RCRE_GpuBuf* buf);

/** Host -> device copy (up to buf->bytes). */
RCRE_NODISCARD int rcre_gpu_h2d(RCRE_GpuCtx* ctx, RCRE_GpuBuf* dst,
                                const void* src, size_t bytes);
/** Device -> host copy (up to buf->bytes). */
RCRE_NODISCARD int rcre_gpu_d2h(RCRE_GpuCtx* ctx, void* dst,
                                RCRE_GpuBuf* src, size_t bytes);
/** @brief Flush pending work. No-op on the CPU backend. */
RCRE_NODISCARD int rcre_gpu_synchronize(RCRE_GpuCtx* ctx);

/* ---- numeric ops (device buffers; row-major) -----------------------------
 *
 * Each op validates its buffers against the given element counts
 * (bytes must match count * element size) and returns RCRE_OK on
 * success. Buffers of the same context are required.
 */

/** C = alpha*A*B + beta*C (f32). A: m x k (lda), B: k x n (ldb), C: m x n (ldc). */
RCRE_NODISCARD int rcre_gpu_gemm_f32_ctx(RCRE_GpuCtx* ctx,
                                         size_t m, size_t n, size_t k,
                                         float alpha,
                                         RCRE_GpuBuf* A, size_t lda,
                                         RCRE_GpuBuf* B, size_t ldb,
                                         float beta,
                                         RCRE_GpuBuf* C, size_t ldc);
/** C = alpha*A*B + beta*C (f64). */
RCRE_NODISCARD int rcre_gpu_gemm_f64_ctx(RCRE_GpuCtx* ctx,
                                         size_t m, size_t n, size_t k,
                                         double alpha,
                                         RCRE_GpuBuf* A, size_t lda,
                                         RCRE_GpuBuf* B, size_t ldb,
                                         double beta,
                                         RCRE_GpuBuf* C, size_t ldc);

/** y = A*x (CSR, f32). row_ptr: rows+1 size_t, col_idx/values: nnz. */
RCRE_NODISCARD int rcre_gpu_spmv_f32_ctx(RCRE_GpuCtx* ctx, size_t rows,
                                         RCRE_GpuBuf* row_ptr,
                                         RCRE_GpuBuf* col_idx,
                                         RCRE_GpuBuf* values,
                                         RCRE_GpuBuf* x, RCRE_GpuBuf* y);
/** y = A*x (CSR, f64). */
RCRE_NODISCARD int rcre_gpu_spmv_f64_ctx(RCRE_GpuCtx* ctx, size_t rows,
                                         RCRE_GpuBuf* row_ptr,
                                         RCRE_GpuBuf* col_idx,
                                         RCRE_GpuBuf* values,
                                         RCRE_GpuBuf* x, RCRE_GpuBuf* y);

/** y = a*x + y (f32). */
RCRE_NODISCARD int rcre_gpu_axpy_f32_ctx(RCRE_GpuCtx* ctx, size_t n,
                                         float a, RCRE_GpuBuf* x,
                                         RCRE_GpuBuf* y);
/** y = a*x + y (f64). */
RCRE_NODISCARD int rcre_gpu_axpy_f64_ctx(RCRE_GpuCtx* ctx, size_t n,
                                         double a, RCRE_GpuBuf* x,
                                         RCRE_GpuBuf* y);

/** *out = dot(x, y) (f32). out holds one float. */
RCRE_NODISCARD int rcre_gpu_dot_f32_ctx(RCRE_GpuCtx* ctx, size_t n,
                                        RCRE_GpuBuf* x, RCRE_GpuBuf* y,
                                        RCRE_GpuBuf* out);
/** *out = dot(x, y) (f64). out holds one double. */
RCRE_NODISCARD int rcre_gpu_dot_f64_ctx(RCRE_GpuCtx* ctx, size_t n,
                                        RCRE_GpuBuf* x, RCRE_GpuBuf* y,
                                        RCRE_GpuBuf* out);

/** out[j] = philox4x32(counter0 + j/4, key)[j%4] (n uint32 words).
 *  Bit-identical to rcre_rng_philox4x32_fill on every backend
 *  (pure function of the counter — the H6 RNG determinism bar). */
RCRE_NODISCARD int rcre_gpu_philox4x32_fill(RCRE_GpuCtx* ctx, size_t n,
                                            uint64_t counter0, uint64_t key,
                                            RCRE_GpuBuf* out);

/* ---- managed convenience (host pointers; hidden alloc/copy/compute) ------ */

/** C = alpha*A*B + beta*C from host pointers (f32). */
RCRE_NODISCARD int rcre_gpu_gemm_f32_host(RCRE_GpuCtx* ctx,
                                          size_t m, size_t n, size_t k,
                                          float alpha,
                                          const float* A, size_t lda,
                                          const float* B, size_t ldb,
                                          float beta,
                                          float* C, size_t ldc);
/** C = alpha*A*B + beta*C from host pointers (f64). */
RCRE_NODISCARD int rcre_gpu_gemm_f64_host(RCRE_GpuCtx* ctx,
                                          size_t m, size_t n, size_t k,
                                          double alpha,
                                          const double* A, size_t lda,
                                          const double* B, size_t ldb,
                                          double beta,
                                          double* C, size_t ldc);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_GPU_H */
