/* RCRE H6 — GPU bridge: context plumbing + CPU reference backend (2.3.0)
 *
 * The context vtable isolates every backend (cuda/metal/cpu) behind one
 * C ABI (include/extras/rcre_gpu.h). This file is ALWAYS compiled:
 *
 *   - it owns the backend registry (backends self-register from a C
 *     constructor — the H5 hook lesson: no weak symbols, and the Makefile
 *     links the backend objects explicitly so their constructors run);
 *   - it implements the "cpu" backend: the conformance oracle. Its ops
 *     delegate to the SAME core/HAL functions the CPU solver paths use
 *     (rcre_linalg_gemm_*, rcre_spmv_*, rcre_linalg_axpy_*,
 *     rcre_linalg_dot_*, rcre_rng_philox4x32_fill), so CPU-backend
 *     results are BIT-IDENTICAL to the CPU reference by construction —
 *     device backends are gated against this with precision bars,
 *     except Philox (bit-identical by construction on every backend).
 *
 * Zero-dependency: no vendor headers here.
 */
#include "extras/rcre_gpu.h"
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* vtable + registry                                                         */
/* ========================================================================= */

#define RCRE_GPU_MAX_BACKENDS 8
static RCRE_GpuBackend* g_backends[RCRE_GPU_MAX_BACKENDS];
static int g_nbackends = 0;

static RCRE_GpuBackend* backend_find(const char* name) {
    for (int i = 0; i < g_nbackends; ++i)
        if (strcmp(g_backends[i]->name, name) == 0) return g_backends[i];
    return NULL;
}

void rcre_gpu_register_backend(const char* name, RCRE_GpuBackend* be) {
    (void)name;
    if (g_nbackends < RCRE_GPU_MAX_BACKENDS && !backend_find(be->name))
        g_backends[g_nbackends++] = be;
}

/* ========================================================================= */
/* context API                                                               */
/* ========================================================================= */

int rcre_gpu_backend_available(const char* backend) {
    if (!backend) return 0;
    RCRE_GpuBackend* be = backend_find(backend);
    return (be && be->available()) ? 1 : 0;
}

RCRE_GpuCtx* rcre_gpu_open_named(const char* backend) {
    if (!backend) return NULL;
    RCRE_GpuBackend* be = backend_find(backend);
    if (!be || !be->available()) return NULL;
    RCRE_GpuCtx* ctx = (RCRE_GpuCtx*)calloc(1, sizeof(RCRE_GpuCtx));
    if (!ctx) return NULL;
    ctx->be = be;
    return ctx;
}

RCRE_GpuCtx* rcre_gpu_open(void) {
    static const char* order[2] = { "cuda", "metal" };
    for (int i = 0; i < 2; ++i) {
        RCRE_GpuCtx* c = rcre_gpu_open_named(order[i]);
        if (c) return c;
    }
    return rcre_gpu_open_named("cpu");
}

void rcre_gpu_close(RCRE_GpuCtx* ctx) {
    if (!ctx) return;
    free(ctx);
}

const char* rcre_gpu_backend_name(const RCRE_GpuCtx* ctx) {
    return ctx ? ctx->be->name : "none";
}

const char* rcre_gpu_device_desc(const RCRE_GpuCtx* ctx) {
    return (ctx && ctx->be->device_desc) ? ctx->be->device_desc() : "no device";
}

/* ========================================================================= */
/* buffer API (dispatch)                                                     */
/* ========================================================================= */

RCRE_GpuBuf* rcre_gpu_alloc(RCRE_GpuCtx* ctx, size_t bytes) {
    if (!ctx) return NULL;
    RCRE_GpuBuf* b = (RCRE_GpuBuf*)calloc(1, sizeof(RCRE_GpuBuf));
    if (!b) return NULL;
    b->ctx = ctx;
    b->bytes = bytes;
    b->native = ctx->be->alloc(ctx, bytes);
    if (!b->native) {
        free(b);
        return NULL;
    }
    return b;
}

void rcre_gpu_free(RCRE_GpuBuf* buf) {
    if (!buf) return;
    if (buf->native) buf->ctx->be->free_buf(buf);
    free(buf);
}

size_t rcre_gpu_bytes(const RCRE_GpuBuf* buf) {
    return buf ? buf->bytes : 0;
}

int rcre_gpu_h2d(RCRE_GpuCtx* ctx, RCRE_GpuBuf* dst, const void* src, size_t bytes) {
    if (!ctx || !dst || (src && !dst->native)) return RCRE_ERR_INVALID_ARG;
    if (bytes > dst->bytes) return RCRE_ERR_INVALID_ARG;
    return ctx->be->h2d(ctx, dst, src, bytes);
}

int rcre_gpu_d2h(RCRE_GpuCtx* ctx, void* dst, RCRE_GpuBuf* src, size_t bytes) {
    if (!ctx || !src || !dst) return RCRE_ERR_INVALID_ARG;
    if (bytes > src->bytes) return RCRE_ERR_INVALID_ARG;
    return ctx->be->d2h(ctx, dst, src, bytes);
}

int rcre_gpu_synchronize(RCRE_GpuCtx* ctx) {
    if (!ctx) return RCRE_ERR_INVALID_ARG;
    return ctx->be->sync(ctx);
}

/* buffer sanity: b must exist, belong to ctx, and hold at least count*es */
static int buf_ok(const RCRE_GpuCtx* ctx, const RCRE_GpuBuf* b,
                  size_t count, size_t es) {
    if (!b || b->ctx != ctx) return 0;
    return b->bytes >= count * es;
}

/* ========================================================================= */
/* op dispatch (gemm/spmv/axpy/dot/philox)                                   */
/* ========================================================================= */

int rcre_gpu_gemm_f32_ctx(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                          float alpha, RCRE_GpuBuf* A, size_t lda,
                          RCRE_GpuBuf* B, size_t ldb, float beta,
                          RCRE_GpuBuf* C, size_t ldc) {
    if (!ctx || !A || !B || !C || m == 0 || n == 0 || k == 0)
        return RCRE_ERR_INVALID_ARG;
    if (!buf_ok(ctx, A, m * k, 4) || !buf_ok(ctx, B, k * n, 4) ||
        !buf_ok(ctx, C, m * n, 4))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->gemm_f32(ctx, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

int rcre_gpu_gemm_f64_ctx(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                          double alpha, RCRE_GpuBuf* A, size_t lda,
                          RCRE_GpuBuf* B, size_t ldb, double beta,
                          RCRE_GpuBuf* C, size_t ldc) {
    if (!ctx || !A || !B || !C || m == 0 || n == 0 || k == 0)
        return RCRE_ERR_INVALID_ARG;
    if (!buf_ok(ctx, A, m * k, 8) || !buf_ok(ctx, B, k * n, 8) ||
        !buf_ok(ctx, C, m * n, 8))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->gemm_f64(ctx, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

int rcre_gpu_spmv_f32_ctx(RCRE_GpuCtx* ctx, size_t rows, RCRE_GpuBuf* rp,
                          RCRE_GpuBuf* ci, RCRE_GpuBuf* val,
                          RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    if (!ctx || !rp || !ci || !val || !x || !y) return RCRE_ERR_INVALID_ARG;
    /* nnz from the values buffer (col_idx same length) */
    size_t nnz = val->bytes / 4;
    if (!buf_ok(ctx, rp, rows + 1, sizeof(size_t)) ||
        !buf_ok(ctx, ci, nnz, 4) || !buf_ok(ctx, x, rows, 4) ||
        !buf_ok(ctx, y, rows, 4))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->spmv_f32(ctx, rows, rp, ci, val, x, y);
}

int rcre_gpu_spmv_f64_ctx(RCRE_GpuCtx* ctx, size_t rows, RCRE_GpuBuf* rp,
                          RCRE_GpuBuf* ci, RCRE_GpuBuf* val,
                          RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    if (!ctx || !rp || !ci || !val || !x || !y) return RCRE_ERR_INVALID_ARG;
    size_t nnz = val->bytes / 8;
    if (!buf_ok(ctx, rp, rows + 1, sizeof(size_t)) ||
        !buf_ok(ctx, ci, nnz, 8) || !buf_ok(ctx, x, rows, 8) ||
        !buf_ok(ctx, y, rows, 8))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->spmv_f64(ctx, rows, rp, ci, val, x, y);
}

int rcre_gpu_axpy_f32_ctx(RCRE_GpuCtx* ctx, size_t n, float a,
                          RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    if (!ctx || !x || !y) return RCRE_ERR_INVALID_ARG;
    if (!buf_ok(ctx, x, n, 4) || !buf_ok(ctx, y, n, 4))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->axpy_f32(ctx, n, a, x, y);
}

int rcre_gpu_axpy_f64_ctx(RCRE_GpuCtx* ctx, size_t n, double a,
                          RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    if (!ctx || !x || !y) return RCRE_ERR_INVALID_ARG;
    if (!buf_ok(ctx, x, n, 8) || !buf_ok(ctx, y, n, 8))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->axpy_f64(ctx, n, a, x, y);
}

int rcre_gpu_dot_f32_ctx(RCRE_GpuCtx* ctx, size_t n, RCRE_GpuBuf* x,
                         RCRE_GpuBuf* y, RCRE_GpuBuf* out) {
    if (!ctx || !x || !y || !out) return RCRE_ERR_INVALID_ARG;
    if (!buf_ok(ctx, x, n, 4) || !buf_ok(ctx, y, n, 4) ||
        !buf_ok(ctx, out, 1, 4))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->dot_f32(ctx, n, x, y, out);
}

int rcre_gpu_dot_f64_ctx(RCRE_GpuCtx* ctx, size_t n, RCRE_GpuBuf* x,
                         RCRE_GpuBuf* y, RCRE_GpuBuf* out) {
    if (!ctx || !x || !y || !out) return RCRE_ERR_INVALID_ARG;
    if (!buf_ok(ctx, x, n, 8) || !buf_ok(ctx, y, n, 8) ||
        !buf_ok(ctx, out, 1, 8))
        return RCRE_ERR_INVALID_ARG;
    return ctx->be->dot_f64(ctx, n, x, y, out);
}

int rcre_gpu_philox4x32_fill(RCRE_GpuCtx* ctx, size_t n, uint64_t counter0,
                             uint64_t key, RCRE_GpuBuf* out) {
    if (!ctx || !out) return RCRE_ERR_INVALID_ARG;
    if (!buf_ok(ctx, out, n, 4)) return RCRE_ERR_INVALID_ARG;
    return ctx->be->philox_fill(ctx, n, counter0, key, out);
}

/* ========================================================================= */
/* managed convenience                                                       */
/* ========================================================================= */

int rcre_gpu_gemm_f32_host(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                           float alpha, const float* A, size_t lda,
                           const float* B, size_t ldb, float beta,
                           float* C, size_t ldc) {
    if (!ctx || !A || !B || !C) return RCRE_ERR_INVALID_ARG;
    RCRE_GpuBuf* dA = rcre_gpu_alloc(ctx, m * k * 4);
    RCRE_GpuBuf* dB = rcre_gpu_alloc(ctx, k * n * 4);
    RCRE_GpuBuf* dC = rcre_gpu_alloc(ctx, m * n * 4);
    if (!dA || !dB || !dC) {
        rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    int rc = rcre_gpu_h2d(ctx, dA, A, m * k * 4);
    rc = (rc == RCRE_OK) ? rcre_gpu_h2d(ctx, dB, B, k * n * 4) : rc;
    rc = (rc == RCRE_OK) ? rcre_gpu_h2d(ctx, dC, C, m * n * 4) : rc;
    rc = (rc == RCRE_OK)
             ? rcre_gpu_gemm_f32_ctx(ctx, m, n, k, alpha, dA, lda, dB, ldb,
                                     beta, dC, ldc)
             : rc;
    rc = (rc == RCRE_OK) ? rcre_gpu_d2h(ctx, C, dC, m * n * 4) : rc;
    rc = (rc == RCRE_OK) ? rcre_gpu_synchronize(ctx) : rc;
    rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
    return rc;
}

int rcre_gpu_gemm_f64_host(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                           double alpha, const double* A, size_t lda,
                           const double* B, size_t ldb, double beta,
                           double* C, size_t ldc) {
    if (!ctx || !A || !B || !C) return RCRE_ERR_INVALID_ARG;
    RCRE_GpuBuf* dA = rcre_gpu_alloc(ctx, m * k * 8);
    RCRE_GpuBuf* dB = rcre_gpu_alloc(ctx, k * n * 8);
    RCRE_GpuBuf* dC = rcre_gpu_alloc(ctx, m * n * 8);
    if (!dA || !dB || !dC) {
        rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    int rc = rcre_gpu_h2d(ctx, dA, A, m * k * 8);
    rc = (rc == RCRE_OK) ? rcre_gpu_h2d(ctx, dB, B, k * n * 8) : rc;
    rc = (rc == RCRE_OK) ? rcre_gpu_h2d(ctx, dC, C, m * n * 8) : rc;
    rc = (rc == RCRE_OK)
             ? rcre_gpu_gemm_f64_ctx(ctx, m, n, k, alpha, dA, lda, dB, ldb,
                                     beta, dC, ldc)
             : rc;
    rc = (rc == RCRE_OK) ? rcre_gpu_d2h(ctx, C, dC, m * n * 8) : rc;
    rc = (rc == RCRE_OK) ? rcre_gpu_synchronize(ctx) : rc;
    rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
    return rc;
}

/* ========================================================================= */
/* CPU reference backend (the conformance oracle)                            */
/* ========================================================================= */
/*
 * Every op delegates to the core/HAL function the CPU solver paths use,
 * so this backend is bit-identical to the CPU reference BY CONSTRUCTION:
 *
 *   gemm  -> rcre_linalg_gemm_f32/f64 (HAL-dispatched, H0/H2)
 *   spmv  -> rcre_spmv_f32/f64        (H4)
 *   axpy  -> rcre_linalg_axpy_f32/f64 (H0)
 *   dot   -> rcre_linalg_dot_f32/f64  (H0)
 *   philox-> rcre_rng_philox4x32_fill (H6 core)
 */

static int cpu_available(void) { return 1; }

static const char* cpu_desc(void) {
    return "CPU reference (core/HAL delegation, deterministic)";
}

static RCRE_GpuBuf* cpu_alloc(RCRE_GpuCtx* ctx, size_t bytes) {
    (void)ctx;
    return bytes ? calloc(1, bytes) : NULL;
}

static void cpu_free_buf(RCRE_GpuBuf* b) { free(b->native); }

static int cpu_h2d(RCRE_GpuCtx* ctx, RCRE_GpuBuf* dst, const void* src,
                   size_t bytes) {
    (void)ctx;
    if (bytes && src) memcpy(dst->native, src, bytes);
    return RCRE_OK;
}

static int cpu_d2h(RCRE_GpuCtx* ctx, void* dst, RCRE_GpuBuf* src, size_t bytes) {
    (void)ctx;
    if (bytes) memcpy(dst, src->native, bytes);
    return RCRE_OK;
}

static int cpu_sync(RCRE_GpuCtx* ctx) { (void)ctx; return RCRE_OK; }

static int cpu_gemm_f32(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                        float alpha, RCRE_GpuBuf* A, size_t lda,
                        RCRE_GpuBuf* B, size_t ldb, float beta,
                        RCRE_GpuBuf* C, size_t ldc) {
    (void)ctx;
    rcre_hal_init();
    rcre_linalg_gemm_f32(m, n, k, alpha, (const float*)A->native, lda,
                         (const float*)B->native, ldb, beta,
                         (float*)C->native, ldc);
    return RCRE_OK;
}

static int cpu_gemm_f64(RCRE_GpuCtx* ctx, size_t m, size_t n, size_t k,
                        double alpha, RCRE_GpuBuf* A, size_t lda,
                        RCRE_GpuBuf* B, size_t ldb, double beta,
                        RCRE_GpuBuf* C, size_t ldc) {
    (void)ctx;
    rcre_hal_init();
    rcre_linalg_gemm_f64(m, n, k, alpha, (const double*)A->native, lda,
                         (const double*)B->native, ldb, beta,
                         (double*)C->native, ldc);
    return RCRE_OK;
}

static int cpu_spmv_f32(RCRE_GpuCtx* ctx, size_t rows, RCRE_GpuBuf* rp,
                        RCRE_GpuBuf* ci, RCRE_GpuBuf* val,
                        RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    (void)ctx;
    RCRE_CSR_F32 A = { rows, rows, 0, NULL, NULL, NULL };
    A.row_ptr = (size_t*)rp->native;
    A.col_idx = (size_t*)ci->native;
    A.values = (float*)val->native;
    A.nnz = val->bytes / 4;
    rcre_spmv_f32(&A, (const float*)x->native, (float*)y->native);
    return RCRE_OK;
}

static int cpu_spmv_f64(RCRE_GpuCtx* ctx, size_t rows, RCRE_GpuBuf* rp,
                        RCRE_GpuBuf* ci, RCRE_GpuBuf* val,
                        RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    (void)ctx;
    RCRE_CSR_F64 A = { rows, rows, 0, NULL, NULL, NULL };
    A.row_ptr = (size_t*)rp->native;
    A.col_idx = (size_t*)ci->native;
    A.values = (double*)val->native;
    A.nnz = val->bytes / 8;
    rcre_spmv_f64(&A, (const double*)x->native, (double*)y->native);
    return RCRE_OK;
}

static int cpu_axpy_f32(RCRE_GpuCtx* ctx, size_t n, float a,
                        RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    (void)ctx;
    rcre_linalg_axpy_f32(n, a, (const float*)x->native, (float*)y->native);
    return RCRE_OK;
}

static int cpu_axpy_f64(RCRE_GpuCtx* ctx, size_t n, double a,
                        RCRE_GpuBuf* x, RCRE_GpuBuf* y) {
    (void)ctx;
    rcre_linalg_axpy_f64(n, a, (const double*)x->native, (double*)y->native);
    return RCRE_OK;
}

static int cpu_dot_f32(RCRE_GpuCtx* ctx, size_t n, RCRE_GpuBuf* x,
                       RCRE_GpuBuf* y, RCRE_GpuBuf* out) {
    (void)ctx;
    float d = rcre_linalg_dot_f32((const float*)x->native,
                                  (const float*)y->native, n);
    *(float*)out->native = d;
    return RCRE_OK;
}

static int cpu_dot_f64(RCRE_GpuCtx* ctx, size_t n, RCRE_GpuBuf* x,
                       RCRE_GpuBuf* y, RCRE_GpuBuf* out) {
    (void)ctx;
    double d = rcre_linalg_dot_f64((const double*)x->native,
                                   (const double*)y->native, n);
    *(double*)out->native = d;
    return RCRE_OK;
}

static int cpu_philox_fill(RCRE_GpuCtx* ctx, size_t n, uint64_t c0,
                           uint64_t key, RCRE_GpuBuf* out) {
    (void)ctx;
    return rcre_rng_philox4x32_fill((uint32_t*)out->native, n, c0, key);
}

static RCRE_GpuBackend g_cpu_be = {
    "cpu", cpu_available, cpu_desc,
    cpu_alloc, cpu_free_buf, cpu_h2d, cpu_d2h, cpu_sync,
    cpu_gemm_f32, cpu_gemm_f64,
    cpu_spmv_f32, cpu_spmv_f64,
    cpu_axpy_f32, cpu_axpy_f64,
    cpu_dot_f32, cpu_dot_f64,
    cpu_philox_fill,
};

/* Register before main() runs (see the H5 hook lesson: this TU is
 * always compiled into the binaries that use the bridge, so its
 * constructor is guaranteed to run; no weak symbols involved). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void gpu_ctx_register_cpu(void) {
    rcre_gpu_register_backend("cpu", &g_cpu_be);
}
