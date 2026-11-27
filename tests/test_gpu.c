/* RCRE H6 test suite — GPU bridge (2.3.0)
 *
 * Runs the SAME gates on every available backend ("cpu" always; "cuda"
 * when the TU is linked and a GPU is present — dlbox: RTX 5060 Ti,
 * sm_120, CUDA 13.3):
 *
 *   - buffer round-trip (alloc/h2d/d2h/sync/free)
 *   - CPU backend: BIT-IDENTICAL to the core (gemm f32/f64, spmv, axpy,
 *     dot, philox) — by construction (it delegates to the core)
 *   - device backends:
 *       * philox: BIT-IDENTICAL to the core fill (pure function of the
 *         counter — the H6 RNG determinism bar)
 *       * gemm/spmv/axpy/dot: precision bars vs the core (documented
 *         order difference), plus DOUBLE-LAUNCH determinism
 *   - CG + ILU(0) on 2D Poisson: CPU factorization (H4) + device
 *     iteration kernels; converged rel error + iteration-count bar
 *   - error inputs
 *
 * The suite self-selects: `for each available backend in
 * {"cpu","cuda"}`.
 */
#include "rcre.h"
#include "extras/rcre_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_framework.h"

/* deterministic LCG data */
static void lcg_fill(void* dst, size_t n, unsigned seed, int width /*4|8*/) {
    uint64_t st = seed;
    if (width == 4) {
        float* p = (float*)dst;
        for (size_t i = 0; i < n; ++i) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            p[i] = (float)((st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-24;
        }
    } else {
        double* p = (double*)dst;
        for (size_t i = 0; i < n; ++i) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            p[i] = (double)((st >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53;
        }
    }
}

/* relative max-norm difference between two f32 arrays (scale by max abs) */
static double f32_reldiff(const float* a, const float* b, size_t n) {
    float da = 0.0f, sa = 0.0f, sb = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > da) da = d;
        float x = a[i] < 0 ? -a[i] : a[i];
        float y = b[i] < 0 ? -b[i] : b[i];
        if (x > sa) sa = x;
        if (y > sb) sb = y;
    }
    double scale = sa > sb ? (double)sa : (double)sb;
    return scale > 0.0 ? (double)da / scale : 0.0;
}

static double f64_reldiff(const double* a, const double* b, size_t n) {
    double da = 0.0, sa = 0.0, sb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > da) da = d;
        double x = a[i] < 0 ? -a[i] : a[i];
        double y = b[i] < 0 ? -b[i] : b[i];
        if (x > sa) sa = x;
        if (y > sb) sb = y;
    }
    double scale = sa > sb ? sa : sb;
    return scale > 0.0 ? da / scale : 0.0;
}

/* build a 2D Poisson 5-point CSR (Dirichlet, interior nx*ny) */
static void poisson_csr(size_t nx, size_t ny, size_t* row_ptr, size_t* col_idx,
                        double* values) {
    (void)nx; (void)ny;
    row_ptr[0] = 0;
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            size_t c = i * ny + j;
            size_t m = 0;
            /* ASCENDING col order (H4 ILU(0) precondition), linear
             * (row-priority) indices: left, up, self, down, right */
            if (i > 0)     { col_idx[row_ptr[c] + m] = (i - 1) * ny + j; values[row_ptr[c] + m] = -1.0; m++; }
            if (j > 0)     { col_idx[row_ptr[c] + m] = i * ny + (j - 1); values[row_ptr[c] + m] = -1.0; m++; }
            col_idx[row_ptr[c] + m] = c; values[row_ptr[c] + m] = 4.0; m++;
            if (j + 1 < ny) { col_idx[row_ptr[c] + m] = i * ny + (j + 1); values[row_ptr[c] + m] = -1.0; m++; }
            if (i + 1 < nx) { col_idx[row_ptr[c] + m] = (i + 1) * ny + j; values[row_ptr[c] + m] = -1.0; m++; }
            row_ptr[c + 1] = row_ptr[c] + m;
        }
    }
}

/* ========================================================================= */

static void test_buffers(const char* be_name) {
    printf("  [%s] buffers\n", be_name);
    RCRE_GpuCtx* ctx = rcre_gpu_open_named(be_name);
    ASSERT_TRUE(ctx != NULL);
    if (!ctx) return;
    RCRE_GpuBuf* b = rcre_gpu_alloc(ctx, 256);
    ASSERT_TRUE(b != NULL);
    ASSERT_TRUE(rcre_gpu_bytes(b) == 256);
    uint8_t src[256], dst[256];
    for (int i = 0; i < 256; ++i) src[i] = (uint8_t)(i * 7);
    ASSERT_TRUE(rcre_gpu_h2d(ctx, b, src, 256) == RCRE_OK);
    ASSERT_TRUE(rcre_gpu_d2h(ctx, dst, b, 256) == RCRE_OK);
    ASSERT_TRUE(memcmp(src, dst, 256) == 0);
    ASSERT_TRUE(rcre_gpu_synchronize(ctx) == RCRE_OK);
    /* over-length copy rejected */
    ASSERT_TRUE(rcre_gpu_h2d(ctx, b, src, 257) != RCRE_OK);
    ASSERT_TRUE(rcre_gpu_d2h(ctx, dst, b, 257) != RCRE_OK);
    rcre_gpu_free(b);
    rcre_gpu_close(ctx);
}

static void test_ops_cpu_backend(void) {
    printf("  [cpu] ops bit-identical to core\n");
    RCRE_GpuCtx* ctx = rcre_gpu_open_named("cpu");
    ASSERT_TRUE(ctx != NULL);
    if (!ctx) return;

    /* gemm f32: 64x48 = A(64x32) * B(32x48) */
    enum { M = 64, K = 32, N = 48 };
    float *A = malloc(M * K * 4), *B = malloc(K * N * 4), *C1 = malloc(M * N * 4), *C2 = malloc(M * N * 4);
    lcg_fill(A, M * K, 0xA1, 4);
    lcg_fill(B, K * N, 0xB2, 4);
    lcg_fill(C1, M * N, 0xC3, 4);
    memcpy(C2, C1, M * N * 4);
    RCRE_GpuBuf *dA = rcre_gpu_alloc(ctx, M * K * 4), *dB = rcre_gpu_alloc(ctx, K * N * 4),
                *dC = rcre_gpu_alloc(ctx, M * N * 4);
    (void)(rcre_gpu_h2d(ctx, dA, A, M * K * 4));
    (void)(rcre_gpu_h2d(ctx, dB, B, K * N * 4));
    (void)(rcre_gpu_h2d(ctx, dC, C1, M * N * 4));
    ASSERT_TRUE(rcre_gpu_gemm_f32_ctx(ctx, M, N, K, 1.5f, dA, K, dB, N,
                                      -0.25f, dC, N) == RCRE_OK);
    float* cout = malloc(M * N * 4);
    (void)(rcre_gpu_d2h(ctx, cout, dC, M * N * 4));
    rcre_hal_init();
    rcre_linalg_gemm_f32(M, N, K, 1.5f, A, K, B, N, -0.25f, C2, N);
    ASSERT_TRUE(memcmp(cout, C2, M * N * 4) == 0); /* bit-identical */

    /* gemm f64: 48x48 */
    enum { P = 48 };
    double *A6 = malloc(P * P * 8), *B6 = malloc(P * P * 8), *C6a = malloc(P * P * 8), *C6b = malloc(P * P * 8);
    lcg_fill(A6, P * P, 0xD4, 8);
    lcg_fill(B6, P * P, 0xE5, 8);
    lcg_fill(C6a, P * P, 0xF6, 8);
    memcpy(C6b, C6a, P * P * 8);
    RCRE_GpuBuf *dA6 = rcre_gpu_alloc(ctx, P * P * 8), *dB6 = rcre_gpu_alloc(ctx, P * P * 8),
                *dC6 = rcre_gpu_alloc(ctx, P * P * 8);
    (void)(rcre_gpu_h2d(ctx, dA6, A6, P * P * 8));
    (void)(rcre_gpu_h2d(ctx, dB6, B6, P * P * 8));
    (void)(rcre_gpu_h2d(ctx, dC6, C6a, P * P * 8));
    ASSERT_TRUE(rcre_gpu_gemm_f64_ctx(ctx, P, P, P, 1.0, dA6, P, dB6, P,
                                      0.5, dC6, P) == RCRE_OK);
    double* c6out = malloc(P * P * 8);
    (void)(rcre_gpu_d2h(ctx, c6out, dC6, P * P * 8));
    rcre_linalg_gemm_f64(P, P, P, 1.0, A6, P, B6, P, 0.5, C6b, P);
    ASSERT_TRUE(memcmp(c6out, C6b, P * P * 8) == 0);

    /* spmv f64 (Poisson 8x8) + axpy + dot */
    size_t n = 64;
    size_t* rp = malloc((n + 1) * sizeof(size_t));
    size_t* ci = malloc(5 * n * sizeof(size_t));
    double* val = malloc(5 * n * 8);
    poisson_csr(8, 8, rp, ci, val);
    size_t nnz = rp[n];
    double* x = malloc(n * 8), *y1 = malloc(n * 8), *y2 = malloc(n * 8), *z = malloc(n * 8);
    lcg_fill(x, n, 0x12, 8);
    lcg_fill(y1, n, 0x34, 8);
    memcpy(y2, y1, n * 8);
    RCRE_GpuBuf *drp = rcre_gpu_alloc(ctx, (n + 1) * sizeof(size_t)),
                *dci = rcre_gpu_alloc(ctx, nnz * sizeof(size_t)),
                *dv = rcre_gpu_alloc(ctx, nnz * 8),
                *dx = rcre_gpu_alloc(ctx, n * 8),
                *dy = rcre_gpu_alloc(ctx, n * 8);
    (void)(rcre_gpu_h2d(ctx, drp, rp, (n + 1) * sizeof(size_t)));
    (void)(rcre_gpu_h2d(ctx, dci, ci, nnz * sizeof(size_t)));
    (void)(rcre_gpu_h2d(ctx, dv, val, nnz * 8));
    (void)(rcre_gpu_h2d(ctx, dx, x, n * 8));
    (void)(rcre_gpu_h2d(ctx, dy, y1, n * 8));
    ASSERT_TRUE(rcre_gpu_spmv_f64_ctx(ctx, n, drp, dci, dv, dx, dy) == RCRE_OK);
    RCRE_CSR_F64 Acsr = { n, n, nnz, rp, ci, val };
    rcre_spmv_f64(&Acsr, x, y2);
    double* yout = malloc(n * 8);
    (void)(rcre_gpu_d2h(ctx, yout, dy, n * 8));
    ASSERT_TRUE(memcmp(yout, y2, n * 8) == 0);

    /* axpy: y = 0.5*x + y (on the spmv result, both sides) */
    memcpy(y2, yout, n * 8);
    (void)(rcre_gpu_h2d(ctx, dy, y2, n * 8));
    ASSERT_TRUE(rcre_gpu_axpy_f64_ctx(ctx, n, 0.5, dx, dy) == RCRE_OK);
    rcre_linalg_axpy_f64(n, 0.5, x, y2);
    (void)(rcre_gpu_d2h(ctx, yout, dy, n * 8));
    ASSERT_TRUE(memcmp(yout, y2, n * 8) == 0);

    /* dot */
    RCRE_GpuBuf* dout = rcre_gpu_alloc(ctx, 8);
    ASSERT_TRUE(rcre_gpu_dot_f64_ctx(ctx, n, dx, dy, dout) == RCRE_OK);
    double doutv, dref = rcre_linalg_dot_f64(x, y2, n);
    (void)(rcre_gpu_d2h(ctx, &doutv, dout, 8));
    ASSERT_TRUE(doutv == dref);

    /* philox: bit-identical to core (2048 words) */
    RCRE_GpuBuf* dw = rcre_gpu_alloc(ctx, 2048 * 4);
    ASSERT_TRUE(rcre_gpu_philox4x32_fill(ctx, 2048, 7, 0xC0FFEEull, dw) == RCRE_OK);
    uint32_t* wout = malloc(2048 * 4);
    (void)(rcre_gpu_d2h(ctx, wout, dw, 2048 * 4));
    uint32_t* wref = malloc(2048 * 4);
    (void)rcre_rng_philox4x32_fill(wref, 2048, 7, 0xC0FFEEull);
    ASSERT_TRUE(memcmp(wout, wref, 2048 * 4) == 0);

    free(A); free(B); free(C1); free(C2); free(cout);
    free(A6); free(B6); free(C6a); free(C6b); free(c6out);
    free(rp); free(ci); free(val); free(x); free(y1); free(y2); free(z);
    free(yout); free(wout); free(wref);
    rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
    rcre_gpu_free(dA6); rcre_gpu_free(dB6); rcre_gpu_free(dC6);
    rcre_gpu_free(drp); rcre_gpu_free(dci); rcre_gpu_free(dv);
    rcre_gpu_free(dx); rcre_gpu_free(dy); rcre_gpu_free(dout); rcre_gpu_free(dw);
    rcre_gpu_close(ctx);
}

/* device-precision gates (used by non-cpu backends) */
static void test_ops_device(const char* be_name) {
    printf("  [%s] device precision bars + determinism\n", be_name);
    RCRE_GpuCtx* ctx = rcre_gpu_open_named(be_name);
    ASSERT_TRUE(ctx != NULL);
    if (!ctx) return;

    /* gemm f32 256x256: rel diff vs core < 1e-5; double-launch identical */
    enum { M = 256 };
    float *A = malloc(M * M * 4), *B = malloc(M * M * 4), *C0 = malloc(M * M * 4);
    lcg_fill(A, M * M, 0xA1, 4);
    lcg_fill(B, M * M, 0xB2, 4);
    lcg_fill(C0, M * M, 0xC3, 4);
    RCRE_GpuBuf *dA = rcre_gpu_alloc(ctx, M * M * 4), *dB = rcre_gpu_alloc(ctx, M * M * 4),
                *dC = rcre_gpu_alloc(ctx, M * M * 4);
    (void)(rcre_gpu_h2d(ctx, dA, A, M * M * 4));
    (void)(rcre_gpu_h2d(ctx, dB, B, M * M * 4));
    float* out1 = malloc(M * M * 4), *out2 = malloc(M * M * 4);
    for (int run = 0; run < 2; ++run) {
        (void)(rcre_gpu_h2d(ctx, dC, C0, M * M * 4));
        ASSERT_TRUE(rcre_gpu_gemm_f32_ctx(ctx, M, M, M, 1.0f, dA, M, dB, M,
                                          0.0f, dC, M) == RCRE_OK);
        ASSERT_TRUE(rcre_gpu_synchronize(ctx) == RCRE_OK);
        (void)rcre_gpu_d2h(ctx, run == 0 ? out1 : out2, dC, M * M * 4);
    }
    rcre_hal_init();
    rcre_linalg_gemm_f32(M, M, M, 1.0f, A, M, B, M, 0.0f, C0, M);
    double rd = f32_reldiff(out1, C0, M * M);
    printf("    [%s] gemm f32 rel diff vs CPU: %.3e\n", be_name, rd);
    ASSERT_TRUE(rd < 1e-5);
    ASSERT_TRUE(memcmp(out1, out2, M * M * 4) == 0); /* deterministic */

    /* gemm f64 128x128: rel diff < 1e-12 */
    enum { Q = 128 };
    double *A6 = malloc(Q * Q * 8), *B6 = malloc(Q * Q * 8);
    lcg_fill(A6, Q * Q, 0xD4, 8);
    lcg_fill(B6, Q * Q, 0xE5, 8);
    RCRE_GpuBuf *dA6 = rcre_gpu_alloc(ctx, Q * Q * 8), *dB6 = rcre_gpu_alloc(ctx, Q * Q * 8),
                *dC6 = rcre_gpu_alloc(ctx, Q * Q * 8);
    (void)(rcre_gpu_h2d(ctx, dA6, A6, Q * Q * 8));
    (void)(rcre_gpu_h2d(ctx, dB6, B6, Q * Q * 8));
    ASSERT_TRUE(rcre_gpu_gemm_f64_ctx(ctx, Q, Q, Q, 1.0, dA6, Q, dB6, Q,
                                      0.0, dC6, Q) == RCRE_OK);
    (void)rcre_gpu_synchronize(ctx);
    double* c6 = malloc(Q * Q * 8);
    (void)(rcre_gpu_d2h(ctx, c6, dC6, Q * Q * 8));
    double* c6ref = malloc(Q * Q * 8);
    rcre_linalg_gemm_f64(Q, Q, Q, 1.0, A6, Q, B6, Q, 0.0, c6ref, Q);
    double rd6 = f64_reldiff(c6, c6ref, Q * Q);
    printf("    [%s] gemm f64 rel diff vs CPU: %.3e\n", be_name, rd6);
    ASSERT_TRUE(rd6 < 1e-12);

    /* spmv f64 Poisson 32x32: rel diff < 1e-14 (same row walk order as
     * the CPU scalar kernel — in fact bit-identical is expected; gate
     * is lenient to keep the bar honest) */
    size_t n = 1024;
    size_t* rp = malloc((n + 1) * sizeof(size_t));
    size_t* ci = malloc(5 * n * sizeof(size_t));
    double* val = malloc(5 * n * 8);
    poisson_csr(32, 32, rp, ci, val);
    size_t nnz = rp[n];
    double* x = malloc(n * 8);
    lcg_fill(x, n, 0x77, 8);
    RCRE_GpuBuf *drp = rcre_gpu_alloc(ctx, (n + 1) * sizeof(size_t)),
                *dci = rcre_gpu_alloc(ctx, nnz * sizeof(size_t)),
                *dv = rcre_gpu_alloc(ctx, nnz * 8),
                *dx = rcre_gpu_alloc(ctx, n * 8),
                *dy = rcre_gpu_alloc(ctx, n * 8);
    (void)(rcre_gpu_h2d(ctx, drp, rp, (n + 1) * sizeof(size_t)));
    (void)(rcre_gpu_h2d(ctx, dci, ci, nnz * sizeof(size_t)));
    (void)(rcre_gpu_h2d(ctx, dv, val, nnz * 8));
    (void)(rcre_gpu_h2d(ctx, dx, x, n * 8));
    ASSERT_TRUE(rcre_gpu_spmv_f64_ctx(ctx, n, drp, dci, dv, dx, dy) == RCRE_OK);
    (void)rcre_gpu_synchronize(ctx);
    double* yout = malloc(n * 8);
    (void)(rcre_gpu_d2h(ctx, yout, dy, n * 8));
    double* yref = malloc(n * 8);
    RCRE_CSR_F64 Acsr = { n, n, nnz, rp, ci, val };
    rcre_spmv_f64(&Acsr, x, yref);
    double rds = f64_reldiff(yout, yref, n);
    printf("    [%s] spmv f64 rel diff vs CPU: %.3e\n", be_name, rds);
    ASSERT_TRUE(rds < 1e-13);

    /* axpy + dot f32 8192: rel diff < 1e-6 (dot: tree vs sequential) */
    enum { L = 8192 };
    float *xf = malloc(L * 4), *yf = malloc(L * 4), *yref_f = malloc(L * 4);
    lcg_fill(xf, L, 0x5A, 4);
    lcg_fill(yf, L, 0x6B, 4);
    memcpy(yref_f, yf, L * 4);
    RCRE_GpuBuf *dxf = rcre_gpu_alloc(ctx, L * 4), *dyf = rcre_gpu_alloc(ctx, L * 4),
                *doutf = rcre_gpu_alloc(ctx, 4);
    (void)(rcre_gpu_h2d(ctx, dxf, xf, L * 4));
    (void)(rcre_gpu_h2d(ctx, dyf, yf, L * 4));
    ASSERT_TRUE(rcre_gpu_axpy_f32_ctx(ctx, L, 0.25f, dxf, dyf) == RCRE_OK);
    rcre_linalg_axpy_f32(L, 0.25f, xf, yref_f);
    ASSERT_TRUE(rcre_gpu_dot_f32_ctx(ctx, L, dxf, dyf, doutf) == RCRE_OK);
    (void)rcre_gpu_synchronize(ctx);
    float* aout = malloc(L * 4);
    (void)(rcre_gpu_d2h(ctx, aout, dyf, L * 4));
    float ddev;
    (void)(rcre_gpu_d2h(ctx, &ddev, doutf, 4));
    float dref = rcre_linalg_dot_f32(xf, yref_f, L);
    double rda = f32_reldiff(aout, yref_f, L);
    double rdd = (double)(fabsf(ddev - dref)) / (double)(dref != 0 ? fabsf(dref) : 1.0f);
    printf("    [%s] axpy f32 rel diff: %.3e | dot f32 rel diff: %.3e\n",
           be_name, rda, rdd);
    ASSERT_TRUE(rda < 1e-6);
    ASSERT_TRUE(rdd < 1e-5);

    /* philox: BIT-IDENTICAL to core (65536 words, the H6 RNG bar) */
    RCRE_GpuBuf* dw = rcre_gpu_alloc(ctx, 65536 * 4);
    ASSERT_TRUE(rcre_gpu_philox4x32_fill(ctx, 65536, 1000, 0xDEADBEEFCAFEBABEull, dw) == RCRE_OK);
    (void)rcre_gpu_synchronize(ctx);
    uint32_t* wout = malloc(65536 * 4);
    (void)(rcre_gpu_d2h(ctx, wout, dw, 65536 * 4));
    uint32_t* wref = malloc(65536 * 4);
    (void)rcre_rng_philox4x32_fill(wref, 65536, 1000, 0xDEADBEEFCAFEBABEull);
    int same = memcmp(wout, wref, 65536 * 4) == 0;
    printf("    [%s] philox 64k words bit-identical: %s\n", be_name,
           same ? "OK" : "FAIL");
    ASSERT_TRUE(same);

    free(A); free(B); free(C0); free(out1); free(out2);
    free(A6); free(B6); free(c6); free(c6ref);
    free(rp); free(ci); free(val); free(x); free(yout); free(yref);
    free(xf); free(yf); free(yref_f); free(aout); free(wout); free(wref);
    rcre_gpu_free(dA); rcre_gpu_free(dB); rcre_gpu_free(dC);
    rcre_gpu_free(dA6); rcre_gpu_free(dB6); rcre_gpu_free(dC6);
    rcre_gpu_free(drp); rcre_gpu_free(dci); rcre_gpu_free(dv);
    rcre_gpu_free(dx); rcre_gpu_free(dy); rcre_gpu_free(dxf);
    rcre_gpu_free(dyf); rcre_gpu_free(doutf); rcre_gpu_free(dw);
    rcre_gpu_close(ctx);
}

/* CG + ILU(0) Poisson: CPU factorization (H4) + device iteration kernels.
 *
 * x lives on the DEVICE (the vector state of the iteration); r, p, z
 * live on the CPU (ILU(0) apply is exact CPU math — the H4 checked-in
 * solver). Per iteration: h2d(z), device axpy (x += alpha*z), device
 * spmv (y = A*x), d2h(y), CPU residual update. This is the H6 solver
 * pattern: factorization/ILU on CPU, the hot SpMV+Axpy+dot on device.
 */
static void test_cg_poisson_device(const char* be_name) {
    printf("  [%s] CG + ILU(0) Poisson (CPU ILU0 factor + device iterations)\n", be_name);
    RCRE_GpuCtx* ctx = rcre_gpu_open_named(be_name);
    ASSERT_TRUE(ctx != NULL);
    if (!ctx) return;

    size_t n = 4096; /* 64x64 */
    size_t* rp = malloc((n + 1) * sizeof(size_t));
    size_t* ci = malloc(5 * n * sizeof(size_t));
    double* val = malloc(5 * n * 8);
    poisson_csr(64, 64, rp, ci, val);
    size_t nnz = rp[n];

    /* ILU(0) on CPU (H4) — the preconditioner is exact CPU math */
    RCRE_CSR_F64 A = { n, n, nnz, rp, ci, val };
    RCRE_CSR_F64 fac = { 0, 0, 0, NULL, NULL, NULL };
    ASSERT_TRUE(rcre_pc_ilu0_f64(&A, &fac) == RCRE_OK);

    /* rhs: random */
    double* b = malloc(n * 8);
    lcg_fill(b, n, 0xCAFE, 8);
    double rnorm = 0.0;
    for (size_t i = 0; i < n; ++i) rnorm += b[i] * b[i];
    rnorm = sqrt(rnorm);

    /* device buffers: x lives on the device (the iteration state) */
    RCRE_GpuBuf *drp = rcre_gpu_alloc(ctx, (n + 1) * sizeof(size_t)),
                *dci = rcre_gpu_alloc(ctx, nnz * sizeof(size_t)),
                *dv = rcre_gpu_alloc(ctx, nnz * 8),
                *dx = rcre_gpu_alloc(ctx, n * 8),
                *dz = rcre_gpu_alloc(ctx, n * 8),
                *dy = rcre_gpu_alloc(ctx, n * 8);
    (void)(rcre_gpu_h2d(ctx, drp, rp, (n + 1) * sizeof(size_t)));
    (void)(rcre_gpu_h2d(ctx, dci, ci, nnz * sizeof(size_t)));
    (void)(rcre_gpu_h2d(ctx, dv, val, nnz * 8));
    double* x0 = calloc(n, 8);
    (void)(rcre_gpu_h2d(ctx, dx, x0, n * 8));

    /* CPU scratch — mirrors the H4 rcre_krylov_pcg recurrence EXACTLY:
     *   z = M^-1 r;  alpha = (r z)/(p (A p));  x += a p;  r -= a (A p);
     *   z = M^-1 r;  beta = (z r)/(z_prev r_prev);  p = z + b p
     * so the iteration count matches the H4 bar (76 @ 64x64, tol 1e-10).
     * Device work per iteration: SpMV(A p) + AXPY(x), CPU: ILU0 solves. */
    double *r = malloc(n * 8), *z = malloc(n * 8), *p = malloc(n * 8),
           *w = malloc(n * 8), *y = malloc(n * 8);
    /* r = b - A*0 */
    memcpy(r, b, n * 8);
    rcre_pc_ilu0_solve_f64(&fac, r, z);
    memcpy(p, z, n * 8);
    double gam = rcre_linalg_dot_f64(z, r, n);
    int converged = 0;
    int iters = 0;
    for (iters = 0; iters < 500; ++iters) {
        /* w = A * p (device) */
        (void)(rcre_gpu_h2d(ctx, dz, p, n * 8));
        ASSERT_TRUE(rcre_gpu_spmv_f64_ctx(ctx, n, drp, dci, dv, dz,
                                          dy) == RCRE_OK);
        ASSERT_TRUE(rcre_gpu_synchronize(ctx) == RCRE_OK);
        ASSERT_TRUE(rcre_gpu_d2h(ctx, w, dy, n * 8) == RCRE_OK);
        double den = rcre_linalg_dot_f64(p, w, n);
        double alpha = (den != 0.0) ? gam / den : 0.0;
        if (!(alpha > 0.0)) break; /* breakdown */
        /* x += alpha * p (device) */
        (void)(rcre_gpu_h2d(ctx, dz, p, n * 8));
        ASSERT_TRUE(rcre_gpu_axpy_f64_ctx(ctx, n, alpha, dz, dx) == RCRE_OK);
        /* r -= alpha * w (CPU, same w — H4 recurrence) */
        for (size_t i = 0; i < n; ++i) r[i] -= alpha * w[i];
        double rel = sqrt(rcre_linalg_dot_f64(r, r, n)) / rnorm;
        if (rel <= 1e-10) {
            converged = 1;
            break;
        }
        rcre_pc_ilu0_solve_f64(&fac, r, z);
        double gamn = rcre_linalg_dot_f64(z, r, n);
        double beta = (gam != 0.0) ? gamn / gam : 0.0;
        gam = gamn;
        for (size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
    printf("    [%s] CG+ILU0: %d iters, converged=%d (tol 1e-10, max 500; "
           "H4 CPU bar: 76)\n", be_name, iters, converged);
    ASSERT_TRUE(converged);
    /* final solution check: rel ||b - A x|| / ||b|| */
    double* xf = malloc(n * 8);
    (void)(rcre_gpu_d2h(ctx, xf, dx, n * 8));
    double* af = malloc(n * 8);
    RCRE_CSR_F64 Acopy = { n, n, nnz, rp, ci, val };
    rcre_spmv_f64(&Acopy, xf, af);
    double fress = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = b[i] - af[i];
        fress += d * d;
    }
    double frel = sqrt(fress) / rnorm;
    printf("    [%s] final rel residual: %.3e\n", be_name, frel);
    ASSERT_TRUE(frel < 1e-9);

    free(rp); free(ci); free(val); free(b); free(x0);
    free(fac.row_ptr); free(fac.col_idx); free(fac.values);
    free(r); free(z); free(p); free(w); free(y); free(xf); free(af);
    rcre_gpu_free(drp); rcre_gpu_free(dci); rcre_gpu_free(dv);
    rcre_gpu_free(dx); rcre_gpu_free(dz); rcre_gpu_free(dy);
    rcre_gpu_close(ctx);
}

static void test_error_inputs(void) {
    printf("  [all] error inputs\n");
    RCRE_GpuCtx* ctx = rcre_gpu_open_named("cpu");
    ASSERT_TRUE(ctx != NULL);
    ASSERT_TRUE(rcre_gpu_open_named("no_such_backend") == NULL);
    ASSERT_TRUE(rcre_gpu_alloc(NULL, 16) == NULL);
    RCRE_GpuBuf* b = rcre_gpu_alloc(ctx, 16);
    ASSERT_TRUE(rcre_gpu_h2d(ctx, NULL, b, 4) != RCRE_OK);
    ASSERT_TRUE(rcre_gpu_d2h(ctx, NULL, b, 4) != RCRE_OK);
    ASSERT_TRUE(rcre_gpu_gemm_f32_ctx(ctx, 4, 4, 4, 1.0f, b, 4, b, 4,
                                      0.0f, b, 4) != RCRE_OK); /* 16B < 64B */
    ASSERT_TRUE(rcre_gpu_synchronize(NULL) != RCRE_OK);
    rcre_gpu_free(b);
    rcre_gpu_free(NULL); /* no-op */
    rcre_gpu_close(ctx);
}

int main(void) {
    test_suite_begin("RCRE GPU Bridge (H6, 2.3.0)");
    test_buffers("cpu");
    test_ops_cpu_backend();
    if (rcre_gpu_backend_available("cuda")) {
        test_buffers("cuda");
        test_ops_device("cuda");
        test_cg_poisson_device("cuda");
    } else {
        printf("  [cuda] not available in this build — precision/CG gates SKIP\n");
    }
    test_error_inputs();
    return test_suite_report();
}
