/* RCRE H9 HPC wave benchmark (2.6.0).
 *
 * Measured bars (docs/hpc.md 12): -O3, M5 Max.
 *
 *  - 3-D Poisson 256^3 (the game-plan full PDE run): wall time per
 *    pool size {NULL,1,2,4,8,18} + an FNV-1a hash of the solution —
 *    the six hashes must be IDENTICAL (the bit-reproducible run; the
 *    cross-machine half runs the same binary where the second device
 *    is registered — the RPi5 follow-up).
 *  - 3-D Poisson 64^3: repeat timing (warm-cache view).
 *  - Distributed ops on the loopback backend vs the core serial
 *    operators (4096x4096 Poisson CSR SpMV; 256x256 halo stencil) —
 *    the per-row cost of the distributed path vs the core.
 *  - FFT: the H9 power-of-two fast path (rcre_fft_pow2_f64) vs the
 *    H2 general path (rcre_fft_1d_f64) at 512, and the 256-DST line
 *    size (514 = 2 x 257, the Bluestein half).
 *
 * Accuracy gates live in tests/test_h9.c; this is the speed side.
 */
#include "rcre.h"
#include "extras/rcre_mpi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static double lcg01(uint64_t* st) {
    *st = *st * 6364136223846793005ull + 1442695040888963407ull;
    return (double)((*st >> 11) & ((1ull << 53) - 1)) / 9007199254740992.0;
}

static void blur3(double* b, size_t nx, size_t ny, size_t nz) {
    double* b2 = (double*)malloc(nx * ny * nz * sizeof(double));
    for (size_t j = 0; j < ny; ++j)
        for (size_t k = 0; k < nz; ++k)
            for (size_t i = 0; i < nx; ++i) {
                double s = 0.0;
                int cnt = 0;
                for (int dj = -1; dj <= 1; ++dj)
                    for (int dk = -1; dk <= 1; ++dk)
                        for (int di = -1; di <= 1; ++di) {
                            if (i < (size_t)-di || j < (size_t)-dj ||
                                k < (size_t)-dk)
                                continue;
                            size_t ii = i + di, jj = j + dj, kk = k + dk;
                            if (ii >= nx || jj >= ny || kk >= nz) continue;
                            s += b[(jj * nz + kk) * nx + ii];
                            ++cnt;
                        }
                b2[(j * nz + k) * nx + i] = s / cnt;
            }
    memcpy(b, b2, nx * ny * nz * sizeof(double));
    free(b2);
}

/* FNV-1a over the bytes of u (the cross-machine reproducibility tag) */
static uint64_t fnv1a(const double* u, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* p = (const uint8_t*)u;
    size_t nb = n * sizeof(double);
    for (size_t i = 0; i < nb; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void bench_poisson3d_256(void) {
    printf("== 3-D Poisson 256^3 (the game-plan full run) ==\n");
    const size_t nx = 256, ny = 256, nz = 256;
    const size_t n3 = nx * ny * nz;
    double* b = (double*)malloc(n3 * sizeof(double));
    double* u = (double*)malloc(n3 * sizeof(double));
    uint64_t st = 0x51A;
    for (size_t i = 0; i < n3; ++i) b[i] = lcg01(&st) * 2.0 - 1.0;
    blur3(b, nx, ny, nz);
    int pools[6] = {0, 1, 2, 4, 8, 18};
    uint64_t h0 = 0;
    int identical = 1;
    for (int p = 0; p < 6; ++p) {
        RCRE_ThreadPool* tp = pools[p] ? rcre_threadpool_create((size_t)pools[p])
                                       : NULL;
        double t0 = now_ms();
        rcre_pde_poisson3d_fft_f64(nx, ny, nz, b, u, tp);
        double ms = now_ms() - t0;
        uint64_t h = fnv1a(u, n3);
        if (p == 0)
            h0 = h;
        else if (h != h0)
            identical = 0;
        printf("  pool=%-2d: %8.1f ms   hash=%016lx\n", pools[p], ms,
               (unsigned long)h);
        if (tp) rcre_threadpool_destroy(tp);
    }
    printf("  bit-reproducible across pool sizes: %s\n",
           identical ? "YES (6/6 hashes identical)" : "NO");
    free(b);
    free(u);
}

static void bench_poisson3d_64(void) {
    printf("== 3-D Poisson 64^3 (repeat timing) ==\n");
    const size_t nx = 64, ny = 64, nz = 64;
    const size_t n3 = nx * ny * nz;
    double* b = (double*)malloc(n3 * sizeof(double));
    double* u = (double*)malloc(n3 * sizeof(double));
    uint64_t st = 0x6464;
    for (size_t i = 0; i < n3; ++i) b[i] = lcg01(&st) * 2.0 - 1.0;
    blur3(b, nx, ny, nz);
    RCRE_ThreadPool* tp = rcre_threadpool_create(18);
    for (int r = 0; r < 5; ++r) {
        double t0 = now_ms();
        rcre_pde_poisson3d_fft_f64(nx, ny, nz, b, u, tp);
        printf("  run %d (pool 18): %7.1f ms\n", r, now_ms() - t0);
    }
    rcre_threadpool_destroy(tp);
    double t0 = now_ms();
    rcre_pde_poisson3d_fft_f64(nx, ny, nz, b, u, NULL);
    printf("  serial:            %7.1f ms\n", now_ms() - t0);
    free(b);
    free(u);
}

static void poisson_csr(size_t n, size_t* rp, size_t* ci, double* val) {
    rp[0] = 0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            size_t c = i * n + j, m = 0;
            if (i > 0) {
                ci[rp[c] + m] = (i - 1) * n + j;
                val[rp[c] + m] = -1.0;
                m++;
            }
            if (j > 0) {
                ci[rp[c] + m] = i * n + (j - 1);
                val[rp[c] + m] = -1.0;
                m++;
            }
            ci[rp[c] + m] = c;
            val[rp[c] + m] = 4.0;
            m++;
            if (j + 1 < n) {
                ci[rp[c] + m] = i * n + (j + 1);
                val[rp[c] + m] = -1.0;
                m++;
            }
            if (i + 1 < n) {
                ci[rp[c] + m] = (i + 1) * n + j;
                val[rp[c] + m] = -1.0;
                m++;
            }
            rp[c + 1] = rp[c] + m;
        }
}

static void bench_dist_ops(void) {
    printf("== distributed ops (loopback) vs core ==\n");
    const size_t n = 4096;
    const size_t nn = n * n;
    size_t* rp = (size_t*)malloc((nn + 1) * sizeof(size_t));
    size_t* ci = (size_t*)malloc(nn * 5 * sizeof(size_t));
    double* val = (double*)malloc(nn * 5 * sizeof(double));
    double* x = (double*)malloc(nn * sizeof(double));
    double* y1 = (double*)malloc(nn * sizeof(double));
    double* y2 = (double*)malloc(nn * sizeof(double));
    uint64_t st = 0xD157;
    for (size_t i = 0; i < nn; ++i) x[i] = lcg01(&st) * 2.0 - 1.0;
    poisson_csr(n, rp, ci, val);
    RCRE_CSR_F64 A = {nn, (size_t)(nn * 5), (size_t)(nn * 5), rp, ci, val};
    double t0 = now_ms();
    rcre_spmv_f64(&A, x, y1);
    double ms_core = now_ms() - t0;
    RCRE_MpiCtx* ctx = NULL;
    (void)rcre_mpi_open(RCRE_MPI_BACKEND_LOOPBACK, &ctx);
    t0 = now_ms();
    rcre_dist_spmv_rows_f64(ctx, rp, ci, val, (int)nn, 0, (int)nn, x, y2);
    double ms_dist = now_ms() - t0;
    rcre_mpi_close(ctx);
    printf("  SpMV 4096^2 CSR:  core %7.1f ms | loopback-dist %7.1f ms | %s\n",
           ms_core, ms_dist,
           memcmp(y1, y2, nn * sizeof(double)) == 0 ? "bit-identical"
                                                    : "MISMATCH");
    free(rp);
    free(ci);
    free(val);
    free(x);
    free(y1);
    free(y2);

    /* halo stencil 256x256 */
    const size_t sx = 256, sy = 256, sn = sx * sy;
    double* u = (double*)malloc(sn * sizeof(double));
    double* v1 = (double*)malloc(sn * sizeof(double));
    double* v2 = (double*)malloc(sn * sizeof(double));
    for (size_t i = 0; i < sn; ++i) u[i] = lcg01(&st) * 2.0 - 1.0;
    t0 = now_ms();
    (void)rcre_pde_stencil5_f64(sx, sy, u, v1, NULL);
    double ms_c = now_ms() - t0;
    ctx = NULL;
    (void)rcre_mpi_open(RCRE_MPI_BACKEND_LOOPBACK, &ctx);
    t0 = now_ms();
    (void)rcre_dist_stencil5_f64(ctx, sx, sy, u, v2);
    double ms_d = now_ms() - t0;
    rcre_mpi_close(ctx);
    printf("  stencil5 256^2:   core %7.1f ms | loopback-dist %7.1f ms | %s\n",
           ms_c, ms_d,
           memcmp(v1, v2, sn * sizeof(double)) == 0 ? "bit-identical"
                                                    : "MISMATCH");
    free(u);
    free(v1);
    free(v2);
}

static void bench_fft(void) {
    printf("== FFT (H9 pow2 fast path vs H2 general) ==\n");
    const size_t N = 512;
    double* re = (double*)malloc(N * sizeof(double));
    double* im = (double*)malloc(N * sizeof(double));
    double* ore = (double*)malloc(N * sizeof(double));
    double* oim = (double*)malloc(N * sizeof(double));
    uint64_t st = 0xFA57;
    for (size_t i = 0; i < N; ++i) re[i] = lcg01(&st) * 2.0 - 1.0;
    int iters = 20000;
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i)
        (void)rcre_fft_pow2_f64((double*)re, (double*)im, N, 0);
    double ms_p = (now_ms() - t0) / iters;
    memcpy(im, oim, N * sizeof(double));
    t0 = now_ms();
    for (int i = 0; i < iters; ++i)
        (void)rcre_fft_1d_f64(re, NULL, ore, oim, N, 0);
    double ms_g = (now_ms() - t0) / iters;
    /* the H9 fast path vs the H2 general wrapper at 512 (both now take
     * the table-cached core; the wrapper cost is the delta) and the
     * 514 Bluestein half (the 256-DST line size) */
    printf("  fft-512 f64:      pow2-inplace %7.1f us | 1d-wrapper  "
           "%7.1f us\n",
           ms_p * 1000.0, ms_g * 1000.0);
    /* the 256-DST line size: 514 = 2 x 257 (Bluestein half) */
    const size_t M = 514;
    double* r5 = (double*)malloc(M * sizeof(double));
    double* i5 = (double*)malloc(M * sizeof(double));
    for (size_t i = 0; i < M; ++i) r5[i] = lcg01(&st) * 2.0 - 1.0;
    iters = 20000;
    t0 = now_ms();
    for (int i = 0; i < iters; ++i)
        (void)rcre_fft_1d_f64(r5, NULL, ore, oim, M, 0);
    printf("  fft-514 f64 (the 256-DST line, Bluestein):  %7.1f us\n",
           (now_ms() - t0) / iters * 1000.0);
    free(re);
    free(im);
    free(ore);
    free(oim);
    free(r5);
    free(i5);
}

int main(void) {
    rcre_hal_init();
    bench_fft();
    bench_dist_ops();
    bench_poisson3d_64();
    bench_poisson3d_256();
    return 0;
}
