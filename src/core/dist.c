/* RCRE H9 — first distributed operators on the 12-fn MPI shim (2.6.0).
 *
 * Two ops, written against the shim ABI (include/extras/rcre_mpi.h):
 *
 *   1. Distributed SpMV (row-block partition):
 *        rcre_dist_spmv_rows_f64  — each rank holds a contiguous CSR
 *           row block [r0, r1) and computes its y-block; x is full on
 *           every rank (bcast by the caller).
 *        rcre_dist_gather_rows_f64 — exchange blocks so every rank
 *           holds the full y.
 *   2. Halo-exchange 5-point stencil (row-slab partition, the H7
 *        layout c = j*nx + i):
 *        rcre_dist_stencil5_f64   — each rank owns a slab of rows,
 *           exchanges one halo row with each neighbor via isend/irecv,
 *           and computes its slab with the CORE row expression
 *           (identical operation order — the bit-identity contract).
 *
 * Determinism (the library-wide policy, applied across processes):
 *   - ctx == NULL is the serial oracle path: the op runs the exact
 *     core expression over the whole array (no communication).
 *   - loopback (size 1) runs the same code path with zero exchanges.
 *   - the per-row arithmetic is value- and order-identical to the
 *     serial op on the same input values, so the 1-process and
 *     2-process results are BIT-IDENTICAL (the test gate). The shim's
 *     ordered collectives (ascending-rank fold) make the allreduce
 *     path bit-stable too (docs/hpc.md 12.2).
 */
#include "rcre.h"
#include "extras/rcre_mpi.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
#define RCRE_DIST_MAX_PEERS 64

/* deterministic partitions (shared by both ops)                       */
/* ------------------------------------------------------------------ */
/*
 * Contiguous block partition of `total` items over `size` ranks:
 * rank r owns base + (r < rem ? 1 : 0) items starting at
 * r*base + min(r, rem).  size=1 -> [0, total).
 */
static void dist_partition(size_t total, int size, int rank, size_t* r0,
                           size_t* r1) {
    if (size < 1) size = 1;
    if (rank >= size) rank = size - 1;
    const size_t base = total / (size_t)size;
    const size_t rem = total % (size_t)size;
    const size_t start = (size_t)rank * base + ((size_t)rank < rem ? (size_t)rank : rem);
    const size_t len = base + (((size_t)rank) < rem ? 1 : 0);
    *r0 = start;
    *r1 = start + len;
}

/* ------------------------------------------------------------------ */
/* distributed SpMV (row-block partition)                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Each rank computes its owned y-block y_local[r1-r0] = A[r0:r1] x.
 *        x is FULL and valid on every rank (the caller bcasts it).
 *        ctx == NULL: serial oracle — compute all rows [0, rows) into
 *        y_local. The per-row accumulation is the core SpMV's exact
 *        left fold (bit-identity contract, docs/hpc.md 12.2).
 */
RCRE_NODISCARD int rcre_dist_spmv_rows_f64(RCRE_MpiCtx* ctx,
                                           const size_t* row_ptr,
                                           const size_t* col_idx,
                                           const double* values, int rows,
                                           int r0, int r1, const double* x,
                                           double* y_local) {
    (void)ctx; /* communication is the caller's job here (bcast x) */
    if (!row_ptr || !col_idx || !values || !x || !y_local || rows < 1 ||
        r0 < 0 || r1 > rows || r0 >= r1)
        return -1;
    /* the exact core row expression (krylov.c rcre_spmv_f64) */
    for (int i = r0; i < r1; ++i) {
        double s = 0.0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            s += values[k] * x[col_idx[k]];
        y_local[i - r0] = s;
    }
    return 0;
}

/**
 * @brief Exchange owned blocks so every rank holds the full y.
 *        y_full is n rows; y_local is the (r1-r0) owned block. Each
 *        rank isends its block to every other rank and irecvs the
 *        others (order: ascending rank, the deterministic contract).
 *        ctx == NULL / size 1: plain copy (no exchange).
 */
RCRE_NODISCARD int rcre_dist_gather_rows_f64(RCRE_MpiCtx* ctx, int n, int r0,
                                             int r1, const double* y_local,
                                             double* y_full) {
    if (!y_local || !y_full || n < 1 || r0 < 0 || r1 > n || r0 >= r1)
        return -1;
    if (!ctx || rcre_mpi_size(ctx) <= 1) {
        /* oracle path: the owned block IS the full vector
         * (loopback owns all rows) */
        if (r0 == 0 && r1 == n) {
            memcpy(y_full, y_local, (size_t)n * sizeof(double));
        } else {
            memcpy(&y_full[r0], y_local, (size_t)(r1 - r0) * sizeof(double));
        }
        return 0;
    }
    const int size = rcre_mpi_size(ctx);
    const int rank = rcre_mpi_rank(ctx);
    if (size - 1 > RCRE_DIST_MAX_PEERS) return -1;
    /* non-blocking, all-to-all: send my block to every other rank,
     * receive each sender's block at its deterministic partition
     * offset (the partition is a pure function of (n, size, rank)) */
    int send_req[RCRE_DIST_MAX_PEERS];
    int recv_req[RCRE_DIST_MAX_PEERS];
    int nsend = 0, nrecv = 0;
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        size_t sr0 = 0, sr1 = 0;
        dist_partition((size_t)n, size, r, &sr0, &sr1);
        send_req[nsend] = rcre_mpi_isend_f64(ctx, y_local, (size_t)(r1 - r0), r,
                                             1000 + rank);
        if (send_req[nsend] < 0) return -1;
        ++nsend;
        recv_req[nrecv] = rcre_mpi_irecv_f64(ctx, &y_full[sr0],
                                             (size_t)(sr1 - sr0), r, 1000 + r);
        if (recv_req[nrecv] < 0) return -1;
        ++nrecv;
    }
    for (int i = 0; i < nrecv; ++i)
        if (rcre_mpi_wait(ctx, recv_req[i]) != 0) return -1;
    for (int i = 0; i < nsend; ++i)
        if (rcre_mpi_wait(ctx, send_req[i]) != 0) return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* halo-exchange 5-point stencil (row-slab partition)                  */
/* ------------------------------------------------------------------ */

#define RCRE_DIST_TAG_HALO 2000

/**
 * @brief Distributed 5-point stencil (H7 layout, zero-boundary
 *        convention). ctx owns a row slab [j0, j1) of the ny rows;
 *        u_local is (j1-j0) x nx owned rows; v_local receives the
 *        same slab of the result. Halo rows are exchanged with the
 *        up/down neighbors via isend/irecv (one row each side).
 *
 *        ctx == NULL: serial oracle — the CORE row expression over
 *        all ny rows (identical to rcre_pde_stencil5_f64 with pool
 *        NULL). The owned-row arithmetic uses the exact core
 *        expression (operation order included) on the same values,
 *        so distributed and serial results are BIT-IDENTICAL.
 *
 *        Returns 0 on success, -1 on bad input, -2 on a communication
 *        failure.
 */
RCRE_NODISCARD int rcre_dist_stencil5_f64(RCRE_MpiCtx* ctx, size_t nx,
                                          size_t ny, const double* u_local,
                                          double* v_local) {
    if (!u_local || !v_local || nx == 0 || ny == 0) return -1;

    /* ---- serial oracle path (ctx == NULL): the core expression ---- */
    if (!ctx) {
        for (size_t j = 0; j < ny; ++j)
            for (size_t i = 0; i < nx; ++i) {
                size_t c = j * nx + i;
                double acc = 4.0 * u_local[c];
                if (j > 0) acc -= u_local[c - nx];
                if (j + 1 < ny) acc -= u_local[c + nx];
                if (i > 0) acc -= u_local[c - 1];
                if (i + 1 < nx) acc -= u_local[c + 1];
                v_local[c] = acc;
            }
        return 0;
    }

    const int size = rcre_mpi_size(ctx);
    const int rank = rcre_mpi_rank(ctx);
    size_t j0 = 0, j1 = 0;
    dist_partition(ny, size, rank, &j0, &j1);
    const size_t nrows = j1 - j0;

    /* halo buffers (one row each side) */
    double* halo_up = (double*)malloc((nx > 0) ? nx * sizeof(double) : 1);
    double* halo_dn = (double*)malloc((nx > 0) ? nx * sizeof(double) : 1);
    if (!halo_up || !halo_dn) {
        free(halo_up);
        free(halo_dn);
        return -1;
    }
    memset(halo_up, 0, nx * sizeof(double));
    memset(halo_dn, 0, nx * sizeof(double));

    const int up = rank - 1;   /* -1: none */
    const int dn = rank + 1;   /* size: none */

    /* post halo receives (I need: row j0-1 from up; row j1 from dn) */
    int req_up = -1, req_dn = -1;
    if (up >= 0)
        req_up = rcre_mpi_irecv_f64(ctx, halo_up, nx, up, RCRE_DIST_TAG_HALO);
    if (dn < size)
        req_dn = rcre_mpi_irecv_f64(ctx, halo_dn, nx, dn, RCRE_DIST_TAG_HALO);
    /* send my boundary rows (row j0 to up; row j1-1 to dn) — only
     * when I own at least one row (ny < size gives some ranks empty
     * slabs; an empty slab neither sends nor computes) */
    int sreq_up = -1, sreq_dn = -1;
    if (up >= 0 && nrows > 0)
        sreq_up = rcre_mpi_isend_f64(ctx, u_local, nx, up, RCRE_DIST_TAG_HALO);
    if (dn < size && nrows > 0)
        sreq_dn = rcre_mpi_isend_f64(
            ctx, u_local + (nrows - 1) * nx, nx, dn, RCRE_DIST_TAG_HALO);
    if ((up >= 0 && req_up < 0) || (dn < size && req_dn < 0) ||
        (up >= 0 && nrows > 0 && sreq_up < 0) ||
        (dn < size && nrows > 0 && sreq_dn < 0)) {
        free(halo_up);
        free(halo_dn);
        return -2;
    }
    if (req_up >= 0 && rcre_mpi_wait(ctx, req_up) != 0) {
        free(halo_up);
        free(halo_dn);
        return -2;
    }
    if (req_dn >= 0 && rcre_mpi_wait(ctx, req_dn) != 0) {
        free(halo_up);
        free(halo_dn);
        return -2;
    }
    if (sreq_up >= 0 && rcre_mpi_wait(ctx, sreq_up) != 0) {
        free(halo_up);
        free(halo_dn);
        return -2;
    }
    if (sreq_dn >= 0 && rcre_mpi_wait(ctx, sreq_dn) != 0) {
        free(halo_up);
        free(halo_dn);
        return -2;
    }
    free(halo_up);
    free(halo_dn);

    /* compute the owned rows with the CORE expression (same operation
     * order; the halo values are the true neighbor rows, so the
     * arithmetic is value-identical to serial) */
    for (size_t lj = 0; lj < nrows; ++lj) {
        const size_t j = j0 + lj; /* global row */
        const double* row = u_local + lj * nx;
        double* vrow = v_local + lj * nx;
        const double* uprow = (j > 0) ? ((j == j0) ? halo_up
                                                   : u_local + (lj - 1) * nx)
                                      : NULL;
        const double* dnrow =
            (j + 1 < ny) ? ((j + 1 == j1) ? halo_dn : u_local + (lj + 1) * nx)
                         : NULL;
        for (size_t i = 0; i < nx; ++i) {
            double acc = 4.0 * row[i];
            if (uprow) acc -= uprow[i];
            if (dnrow) acc -= dnrow[i];
            if (i > 0) acc -= row[i - 1];
            if (i + 1 < nx) acc -= row[i + 1];
            vrow[i] = acc;
        }
    }
    return 0;
}
