/* RCRE H9 — 3-D structured-grid PDE (2.6.0).
 *
 *   - 7-point Dirichlet stencil (h=1), F64, slab-parallel.
 *   - 3-D Poisson spectral solver: the 7-point operator
 *         A = T_x (x) I (x) I + I (x) T_y (x) I + I (x) I (x) T_z,
 *         T = tridiag(-1, 2, -1)
 *     is diagonalized by the product of 1-D DST-I eigenvectors (the
 *     H7 2-D spectral solver, extended to 3 axes with a strided-line
 *     driver — no transposes).
 *
 * LAYOUT (the H4/H7 2-D layout, extended): point (i, j, k) with
 *   0 <= i < nx (x), 0 <= j < ny (y), 0 <= k < nz (z),
 *   flat index c = j*nz*nx + k*nx + i  (row-major, k fastest).
 *
 * Determinism: every parallel stage is an independent transform line
 * or an element-wise pass with a fixed per-element expression —
 * results are BIT-IDENTICAL across pool sizes (pool == NULL means
 * serial). Task scratch is per-line (the H7 slab contract).
 *
 * Eigenvalues (DST-I mode p = 1..n on an axis of length n):
 *   lam_p = 2 - 2*cos(pi*p/(n+1))
 *   lambda_{i,j,k} = lam_x[i] + lam_y[j] + lam_z[k]   (1-indexed modes)
 */
#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ========================================================================= */
/* 7-point stencil (h=1), F64                                                */
/* ========================================================================= */

typedef struct {
    const double* u;
    double* v;
    size_t nx, ny, nz;
} RCRE_Stencil7Ctx;

/*
 * (A u)_{i,j,k} = 6 u_{i,j,k}
 *                - u_{i-1,j,k} - u_{i+1,j,k}
 *                - u_{i,j-1,k} - u_{i,j+1,k}
 *                - u_{i,j,k-1} - u_{i,j,k+1}
 * Zero-boundary convention (the caller folds boundary data into b).
 * One task = one (j, k) plane-pair line over i (the fastest axis).
 */
static void stencil7_line(const RCRE_Stencil7Ctx* s, size_t line) {
    const size_t nx = s->nx, nz = s->nz;
    const size_t j = line / nz;
    const size_t k = line % nz;
    const double* up = s->u + (line * nx) + 0;
    double* vline = s->v + line * nx;
    for (size_t i = 0; i < nx; ++i) {
        const size_t c = line * nx + i;
        double acc = 6.0 * s->u[c];
        if (i > 0) acc -= s->u[c - 1];
        if (i + 1 < nx) acc -= s->u[c + 1];
        if (j > 0) acc -= s->u[c - nz * nx];
        if (j + 1 < s->ny) acc -= s->u[c + nz * nx];
        if (k > 0) acc -= s->u[c - nx];
        if (k + 1 < nz) acc -= s->u[c + nx];
        vline[i] = acc;
    }
    (void)up;
}

static void stencil7_task(void* arg, size_t line) {
    stencil7_line((const RCRE_Stencil7Ctx*)arg, line);
}

RCRE_NODISCARD int rcre_pde_stencil7_f64(size_t nx, size_t ny, size_t nz,
                                         const double* u, double* v,
                                         RCRE_ThreadPool* pool) {
    if (!u || !v || nx == 0 || ny == 0 || nz == 0) return -1;
    RCRE_Stencil7Ctx s = {u, v, nx, ny, nz};
    const size_t lines = ny * nz;
    if (pool)
        rcre_threadpool_parallel_for(pool, lines, stencil7_task, &s);
    else
        for (size_t l = 0; l < lines; ++l) stencil7_task(&s, l);
    return 0;
}

/* ========================================================================= */
/* strided-line 1-D DST-I stage (3 axes, in-place)                           */
/* ========================================================================= */
/*
 * For each of `lines` lines, transform the length-`n` line in place:
 *   axis 0 (x): line (j,k) id = j*nz+k, base = (j*nz+k)*nx,   stride 1
 *   axis 1 (y): line (i,k) id = i*nz+k, base = i + k*nx,     stride nz*nx
 *   axis 2 (z): line (i,j) id = i*ny+j, base = j*nz*nx + i,  stride nx
 * The even-extension + FFT is task-local (per line), so lines never
 * share scratch — bit-identity across pool sizes (the H7 slab
 * contract; the per-line expression is the H7 DST-I's exactly).
 */

typedef struct {
    double* data;
    size_t nx, ny, nz;
    int axis;
    int n;        /* line length */
    size_t stride;
    size_t l0, l1; /* slab: line range [l0, l1) */
    double* re;   /* slab-local, length 2(n+1) */
    double* im;
} RCRE_D3DSlab;

static size_t d3d_line_base(const RCRE_D3DSlab* t, size_t id) {
    const size_t nx = t->nx, ny = t->ny, nz = t->nz;
    if (t->axis == 0) return id * nx; /* (j,k): id = j*nz + k */
    if (t->axis == 1) {
        /* (i,k): id = i*nz + k; line over j: c = j*nz*nx + k*nx + i,
         * base = i + k*nx, stride = nz*nx */
        const size_t i = id / nz;
        const size_t k = id % nz;
        return i + k * nx;
    }
    const size_t i = id / ny; /* axis 2: (i,j): id = i*ny + j */
    const size_t j = id % ny;
    return j * nz * nx + i; /* line over k: base = j*nz*nx + i, stride nx */
}

/* The per-line expression (the H7 DST-I, exactly — see pde.c). */
static void d3d_line(const RCRE_D3DSlab* s, size_t id) {
    const int n = s->n;
    const int N = 2 * (n + 1);
    memset(s->re, 0, (size_t)N * sizeof(double));
    memset(s->im, 0, (size_t)N * sizeof(double)); /* the pow2 path reads
        * im as input (odd extension has zero imaginary part) — the
        * generic path used to hide this with its calloc'd temp */
    const size_t base = d3d_line_base(s, id);
    for (int j = 1; j <= n; ++j) {
        const double x = s->data[base + (size_t)(j - 1) * s->stride];
        s->re[j] = x;
        s->re[N - j] = -x;
    }
    if ((N & (N - 1)) == 0)
        (void)rcre_fft_pow2_f64(s->re, s->im, (size_t)N, 0); /* H9 fast path */
    else
        (void)rcre_fft_1d_f64(s->re, NULL, s->re, s->im, (size_t)N, 0);
    /* write back WITHOUT the -0.5: the per-axis 1/2 is folded into
     * the eigenvalue divide and the final scale (2^6 across the six
     * stages -> /64 in the final scale). The H7 2-D solver keeps its
     * -0.5 convention — independent code, both gated by residual
     * bars; the scaling location is a free degree of freedom
     * (docs/hpc.md 12.3). */
    for (int k2 = 1; k2 <= n; ++k2)
        s->data[base + (size_t)(k2 - 1) * s->stride] = -s->im[k2];
}

/*
 * One task = one slab of lines, with SLAB-LOCAL scratch (the H7 slab
 * contract at 3-D scale: 65536 per-line mallocs cost minutes; a fixed
 * slab count with per-slab scratch is the same determinism with
 * O(slabs) memory). Slab boundaries never change the per-line
 * arithmetic -> bit-identical across pool sizes and slab counts.
 */
static void d3d_slab_task(void* arg, size_t sid) {
    RCRE_D3DSlab* s = &((RCRE_D3DSlab*)arg)[sid];
    for (size_t l = s->l0; l < s->l1; ++l) d3d_line(s, l);
}

static int d3d_stage(double* data, size_t nx, size_t ny, size_t nz, int axis,
                     RCRE_ThreadPool* pool) {
    const int n = (axis == 0) ? (int)nx : (axis == 1) ? (int)ny : (int)nz;
    const size_t lines =
        (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
    const size_t per = (size_t)(2 * (n + 1)) * sizeof(double);

    /* serial path: one scratch, plain loop (the oracle) */
    if (!pool) {
        double* re = (double*)malloc(per);
        double* im = (double*)malloc(per);
        if (!re || !im) {
            free(re);
            free(im);
            return -1;
        }
        RCRE_D3DSlab s = {data, nx, ny, nz, axis, n,
                          (axis == 0) ? 1 : (axis == 1 ? nx * nz : nx), 0,
                          lines, re, im};
        for (size_t l = 0; l < lines; ++l) d3d_line(&s, l);
        free(re);
        free(im);
        return 0;
    }

    /* parallel path: slab partition. Slab count: one slab per line
     * for small grids, otherwise 4x the core count (granularity
     * heuristic; the count never affects the result) */
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc < 1) nproc = 1;
    size_t ns = (size_t)(4 * nproc);
    if (ns > lines) ns = lines;
    if (ns < 1) ns = 1;
    RCRE_D3DSlab* sl = (RCRE_D3DSlab*)calloc(ns, sizeof(RCRE_D3DSlab));
    if (!sl) return -1;
    int rc = 0;
    const size_t base = lines / ns, rem = lines % ns;
    for (size_t i = 0; i < ns; ++i) {
        sl[i].data = data;
        sl[i].nx = nx;
        sl[i].ny = ny;
        sl[i].nz = nz;
        sl[i].axis = axis;
        sl[i].n = n;
        sl[i].stride = (axis == 0) ? 1 : (axis == 1 ? nx * nz : nx);
        sl[i].l0 = i * base + (i < rem ? i : rem);
        sl[i].l1 = sl[i].l0 + base + (i < rem ? 1 : 0);
        sl[i].re = (double*)malloc(per);
        sl[i].im = (double*)malloc(per);
        if (!sl[i].re || !sl[i].im) rc = -1;
    }
    if (rc == 0) rcre_threadpool_parallel_for(pool, ns, d3d_slab_task, sl);
    for (size_t i = 0; i < ns; ++i) {
        free(sl[i].re);
        free(sl[i].im);
    }
    free(sl);
    return rc;
}

/* ========================================================================= */
/* 3-D Poisson — spectral (DST) solver                                       */
/* ========================================================================= */

RCRE_NODISCARD int rcre_pde_poisson3d_fft_f64(size_t nx, size_t ny, size_t nz,
                                              const double* b, double* u,
                                              RCRE_ThreadPool* pool) {
    if (!b || !u || nx < 2 || ny < 2 || nz < 2) return -1;
    rcre_hal_init();
    const size_t n3 = nx * ny * nz;
    double* work = (double*)malloc(n3 * sizeof(double));
    if (!work) return -1;
    memcpy(work, b, n3 * sizeof(double));
    int rc = 0;
    /* forward: x, y, z */
    rc = d3d_stage(work, nx, ny, nz, 0, pool);
    if (rc == 0) rc = d3d_stage(work, nx, ny, nz, 1, pool);
    if (rc == 0) rc = d3d_stage(work, nx, ny, nz, 2, pool);
    if (rc == 0) {
        /* eigenvalue divide: lambda_{i,j,k} = lx[i] + ly[j] + lz[k]
         * (precomputed per-axis tables; one division per cell, fixed
         * order per element -> bit-identical across pool sizes) */
        double* lx = (double*)malloc((nx + 1) * sizeof(double));
        double* ly = (double*)malloc((ny + 1) * sizeof(double));
        double* lz = (double*)malloc((nz + 1) * sizeof(double));
        if (!lx || !ly || !lz) rc = -1;
        for (size_t i = 1; i <= nx; ++i)
            lx[i] = 2.0 - 2.0 * cos(M_PI * (double)i / (double)(nx + 1));
        for (size_t j = 1; j <= ny; ++j)
            ly[j] = 2.0 - 2.0 * cos(M_PI * (double)j / (double)(ny + 1));
        for (size_t k = 1; k <= nz; ++k)
            lz[k] = 2.0 - 2.0 * cos(M_PI * (double)k / (double)(nz + 1));
        if (rc == 0) {
            for (size_t j = 0; j < ny; ++j)
                for (size_t k = 0; k < nz; ++k)
                    for (size_t i = 0; i < nx; ++i) {
                        const double lam = lx[i + 1] + ly[j + 1] + lz[k + 1];
                        work[(j * nz + k) * nx + i] /= lam;
                    }
        }
        free(lx);
        free(ly);
        free(lz);
    }
    /* inverse: z, y, x (DST-I is self-inverse up to the per-axis
     * 2/(n+1) scale, applied once at the end) */
    if (rc == 0) rc = d3d_stage(work, nx, ny, nz, 2, pool);
    if (rc == 0) rc = d3d_stage(work, nx, ny, nz, 1, pool);
    if (rc == 0) rc = d3d_stage(work, nx, ny, nz, 0, pool);
    if (rc == 0) {
        /* the no-half writeback makes each stage 2x the DST, so the
         * six stages contribute 2^6 = 64x; divide the scale by 64
         * (equivalently: the per-axis 2/(n+1) becomes 1/(n+1)/4 ...
         * kept in the single exact form below) */
        const double scale = (1.0 / 64.0) * (2.0 / (double)(nx + 1))
                           * (2.0 / (double)(ny + 1))
                           * (2.0 / (double)(nz + 1));
        for (size_t c = 0; c < n3; ++c) u[c] = scale * work[c];
    }
    free(work);
    return rc;
}
