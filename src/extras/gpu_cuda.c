/* RCRE H6 — CUDA backend host side (2.3.0)
 *
 * Bridges the H6 context ABI (include/extras/rcre_gpu.h) to the CUDA
 * runtime. Compiled with the host compiler when the Makefile detects a
 * CUDA toolkit (RCRE_WITH_CUDA); the kernels live in gpu_kernels.cu
 * (nvcc). This TU registers itself from a C constructor — the Makefile
 * links the backend objects EXPLICITLY so the constructor is guaranteed
 * to run (the H5 archive-extraction lesson).
 *
 * Without a GPU or toolkit this TU is simply not compiled and
 * rcre_gpu_backend_available("cuda") reports 0 (infra-immune).
 */
#include "extras/rcre_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* kernel host wrappers (gpu_kernels.cu) */
int rcre_gpu_cuda_gemm_f32(const float* A, size_t lda, const float* B,
                           size_t ldb, float* C, size_t ldc, size_t m,
                           size_t n, size_t k, float alpha, float beta);
int rcre_gpu_cuda_gemm_f64(const double* A, size_t lda, const double* B,
                           size_t ldb, double* C, size_t ldc, size_t m,
                           size_t n, size_t k, double alpha, double beta);
int rcre_gpu_cuda_spmv_f32(const size_t* rp, const size_t* ci,
                           const float* val, const float* x, float* y,
                           size_t rows);
int rcre_gpu_cuda_spmv_f64(const size_t* rp, const size_t* ci,
                           const double* val, const double* x, double* y,
                           size_t rows);
int rcre_gpu_cuda_axpy_f32(float a, const float* x, float* y, size_t n);
int rcre_gpu_cuda_axpy_f64(double a, const double* x, double* y, size_t n);
int rcre_gpu_cuda_dot_f32(const float* x, const float* y, float* out, size_t n);
int rcre_gpu_cuda_dot_f64(const double* x, const double* y, double* out,
                          size_t n);
int rcre_gpu_cuda_philox_fill(uint32_t* out, size_t n, uint64_t c0,
                              uint64_t key);

/* ========================================================================= */
/* vtable                                                                    */
/* ========================================================================= */

static char g_desc[512] = "NVIDIA CUDA device";

static int cuda_available(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count > 0;
}

static const char* cuda_desc(void) {
    if (g_desc[0] == '\0') return "NVIDIA CUDA device";
    return g_desc;
}

static RCRE_GpuBuf* cuda_alloc(RCRE_GpuCtx* ctx, size_t bytes) {
    (void)ctx;
    void* p = NULL;
    if (!bytes) return NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}

static void cuda_free_buf(RCRE_GpuBuf* b) {
    if (b->native) cudaFree(b->native);
}

static int cuda_h2d(RCRE_GpuCtx* ctx, RCRE_GpuBuf* dst, const void* src,
                    size_t bytes) {
    (void)ctx;
    if (!bytes) return RCRE_OK;
    return cudaMemcpy(dst->native, src, bytes, cudaMemcpyHostToDevice)
               == cudaSuccess
               ? RCRE_OK
               : RCRE_ERR_IO;
}

static int cuda_d2h(RCRE_GpuCtx* ctx, void* dst, RCRE_GpuBuf* src, size_t bytes) {
    (void)ctx;
    if (!bytes) return RCRE_OK;
    return cudaMemcpy(dst, src->native, bytes, cudaMemcpyDeviceToHost)
               == cudaSuccess
               ? RCRE_OK
               : RCRE_ERR_IO;
}

static int cuda_sync(RCRE_GpuCtx* ctx) {
    (void)ctx;
    return cudaDeviceSynchronize() == cudaSuccess ? RCRE_OK : RCRE_ERR_IO;
}

static int cuda_gemm_f32(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                         float alpha, RCRE_GpuBuf* A, size_t lda,
                         RCRE_GpuBuf* B, size_t ldb, float beta,
                         RCRE_GpuBuf* C, size_t ldc) {
    (void)ctx;
    return rcre_gpu_cuda_gemm_f32((const float*)A->native, lda,
                                  (const float*)B->native, ldb,
                                  (float*)C->native, ldc, m, n, k, alpha,
                                  beta);
}

static int cuda_gemm_f64(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                         double alpha, RCRE_GpuBuf* A, size_t lda,
                         RCRE_GpuBuf* B, size_t ldb, double beta,
                         RCRE_GpuBuf* C, size_t ldc) {
    (void)ctx;
    return rcre_gpu_cuda_gemm_f64((const double*)A->native, lda,
                                  (const double*)B->native, ldb,
                                  (double*)C->native, ldc, m, n, k, alpha,
                                  beta);
}

static int cuda_spmv_f32(RCRE_GpuCtx* ctx, size_t rows, RCRE_GpuBuf* rp,
                         RCRE_GpuBuf* ci, RCRE_GpuBuf* val, RCRE_GpuBuf* x,
                         RCRE_GpuBuf* y) {
    (void)ctx;
    return rcre_gpu_cuda_spmv_f32((const size_t*)rp->native,
                                  (const size_t*)ci->native,
                                  (const float*)val->native,
                                  (const float*)x->native,
                                  (float*)y->native, rows);
}

static int cuda_spmv_f64(RCRE_GpuCtx* ctx, size_t rows, RCRE_GpuBuf* rp,
                         RCRE_GpuBuf* ci, RCRE_GpuBuf* val, RCRE_GpuBuf* x,
                         RCRE_GpuBuf* y) {
    (void)ctx;
    return rcre_gpu_cuda_spmv_f64((const size_t*)rp->native,
                                  (const size_t*)ci->native,
                                  (const double*)val->native,
                                  (const double*)x->native,
                                  (double*)y->native, rows);
}

static int cuda_axpy_f32(RCRE_GpuCtx* ctx, size_t n, float a, RCRE_GpuBuf* x,
                         RCRE_GpuBuf* y) {
    (void)ctx;
    return rcre_gpu_cuda_axpy_f32(a, (const float*)x->native,
                                  (float*)y->native, n);
}

static int cuda_axpy_f64(RCRE_GpuCtx* ctx, size_t n, double a, RCRE_GpuBuf* x,
                         RCRE_GpuBuf* y) {
    (void)ctx;
    return rcre_gpu_cuda_axpy_f64(a, (const double*)x->native,
                                  (double*)y->native, n);
}

static int cuda_dot_f32(RCRE_GpuCtx* ctx, size_t n, RCRE_GpuBuf* x,
                        RCRE_GpuBuf* y, RCRE_GpuBuf* out) {
    (void)ctx;
    return rcre_gpu_cuda_dot_f32((const float*)x->native,
                                 (const float*)y->native,
                                 (float*)out->native, n);
}

static int cuda_dot_f64(RCRE_GpuCtx* ctx, size_t n, RCRE_GpuBuf* x,
                        RCRE_GpuBuf* y, RCRE_GpuBuf* out) {
    (void)ctx;
    return rcre_gpu_cuda_dot_f64((const double*)x->native,
                                 (const double*)y->native,
                                 (double*)out->native, n);
}

static int cuda_philox_fill(RCRE_GpuCtx* ctx, size_t n, uint64_t c0,
                            uint64_t key, RCRE_GpuBuf* out) {
    (void)ctx;
    return rcre_gpu_cuda_philox_fill((uint32_t*)out->native, n, c0, key);
}

static RCRE_GpuBackend g_cuda_be = {
    "cuda", cuda_available, cuda_desc,
    cuda_alloc, cuda_free_buf, cuda_h2d, cuda_d2h, cuda_sync,
    cuda_gemm_f32, cuda_gemm_f64,
    cuda_spmv_f32, cuda_spmv_f64,
    cuda_axpy_f32, cuda_axpy_f64,
    cuda_dot_f32, cuda_dot_f64,
    cuda_philox_fill,
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void gpu_cuda_register(void) {
    rcre_gpu_register_backend("cuda", &g_cuda_be);
    /* capture the device description once (cheap: one properties call) */
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        snprintf(g_desc, sizeof g_desc, "%s (sm_%d%d, %zu MiB)", prop.name,
                 prop.major, prop.minor, (size_t)(prop.totalGlobalMem >> 20));
    }
}
