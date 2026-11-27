/**
 * @file test_mpi2.c
 * @brief H9 — the 2-process distributed gates (run under mpirun -np 2).
 *
 * Skips clean where the real-MPI backend is not registered (no MPI on
 * this machine: the gates run where MPI exists, the H6 CUDA pattern).
 * Under mpirun with the `mpi` backend linked:
 *
 *  - halo-exchange stencil on 2 ranks: each rank computes its slab via
 *    rcre_dist_stencil5_f64 (real isend/irecv halo exchange), and the
 *    assembled full-grid result must be BIT-IDENTICAL to the serial
 *    oracle (rcre_pde_stencil5_f64, pool NULL) — the game-plan bar
 *    "a first distributed operation, bit-identical to serial".
 *  - distributed SpMV on 2 ranks: block computation + all-to-all
 *    gather, bit-identical to rcre_spmv_f64.
 *  - ordered allreduce: the 2-rank allreduce of per-rank partial sums
 *    equals the serial ordered fold bit-for-bit.
 *
 * Usage: mpirun -np 2 bin/test_mpi2   (or: make test-mpi2)
 */
#include "rcre.h"
#include "extras/rcre_mpi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_fails = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        ++g_checks;                                                 \
        if (!(cond)) {                                              \
            ++g_fails;                                              \
            printf("[rank pending] FAIL %s:%d: %s\n", __FILE__,     \
                    __LINE__, msg);                                 \
        }                                                           \
    } while (0)

static double lcg01(uint64_t* st) {
    *st = *st * 6364136223846793005ull + 1442695040888963407ull;
    return (double)((*st >> 11) & ((1ull << 53) - 1)) / 9007199254740992.0;
}

/* barrier-synced print so rank output does not interleave */

int main(void) {
    RCRE_MpiCtx* ctx = NULL;
    int rc_open = rcre_mpi_open(RCRE_MPI_BACKEND_MPI, &ctx);
    if (rc_open != 0) {
        printf("test_mpi2: real-MPI backend not registered — skipping "
               "2-process gates clean (build with MPI: make check-mpi)\n");
        return 0;
    }
    const int size = rcre_mpi_size(ctx);
    const int rank = rcre_mpi_rank(ctx);
    if (size != 2) {
        if (rank == 0)
            printf("test_mpi2: expected 2 ranks (mpirun -np 2), got %d — "
                   "skipping\n",
                   size);
        rcre_mpi_close(ctx);
        return 0;
    }

    /* ---- halo stencil, 2 ranks, 33x33 grid ------------------------- */
    const size_t nx = 33, ny = 33, nn = nx * ny;
    double* u_full = (double*)malloc(nn * sizeof(double));
    double* v_ser = (double*)malloc(nn * sizeof(double));
    double* v_par = (double*)malloc(nn * sizeof(double));
    uint64_t st = 0xB0B1;
    for (size_t i = 0; i < nn; ++i) u_full[i] = lcg01(&st) * 2.0 - 1.0;
    rcre_pde_stencil5_f64(nx, ny, u_full, v_ser, NULL);

    /* local slab: the dist op owns rows [j0, j1) deterministically */
    const size_t base = ny / 2, rem = ny % 2;
    const size_t j0 = (size_t)rank * base + ((size_t)rank < rem ? (size_t)rank : rem);
    const size_t j1 = j0 + base + ((size_t)rank < rem ? 1 : 0);
    const size_t nrows = j1 - j0;
    double* u_local = (double*)malloc(nrows * nx * sizeof(double));
    double* v_local = (double*)malloc(nrows * nx * sizeof(double));
    memcpy(u_local, &u_full[j0 * nx], nrows * nx * sizeof(double));
    rcre_mpi_barrier(ctx);
    int rc = rcre_dist_stencil5_f64(ctx, nx, ny, u_local, v_local);
    CHECK(rc == 0, "dist stencil rc");
    /* assemble: rank 0 gathers both slabs (simple send/recv for np=2) */
    if (rank == 0) {
        memcpy(v_par, v_local, nrows * nx * sizeof(double));
        rcre_mpi_recv_f64(ctx, &v_par[base * nx], (nn - base * nx) / nx * nx,
                          1, 42);
    } else {
        rcre_mpi_send_f64(ctx, v_local, nrows * nx, 0, 42);
    }
    rcre_mpi_barrier(ctx);
    if (rank == 0) {
        CHECK(memcmp(v_ser, v_par, nn * sizeof(double)) == 0,
              "2-rank halo stencil == serial oracle (bit-identical)");
        printf("  [2-rank halo stencil] %s\n",
               memcmp(v_ser, v_par, nn * sizeof(double)) == 0
                   ? "bit-identical to serial"
                   : "MISMATCH");
    }
    free(u_full);
    free(v_ser);
    free(v_par);
    free(u_local);
    free(v_local);

    /* ---- distributed SpMV, 2 ranks --------------------------------- */
    const size_t n = 64;
    const size_t nnr = n * n;
    size_t* rp = (size_t*)malloc((nnr + 1) * sizeof(size_t));
    size_t* ci = (size_t*)malloc(nnr * 5 * sizeof(size_t));
    double* val = (double*)malloc(nnr * 5 * sizeof(double));
    double* x = (double*)malloc(nnr * sizeof(double));
    double* y_ser = (double*)malloc(nnr * sizeof(double));
    double* y_par = (double*)malloc(nnr * sizeof(double));
    st = 0x5EED;
    for (size_t i = 0; i < nnr; ++i) x[i] = lcg01(&st) * 2.0 - 1.0;
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
    RCRE_CSR_F64 A = {nnr, (size_t)(nnr * 5), (size_t)(nnr * 5), rp, ci, val};
    rcre_spmv_f64(&A, x, y_ser);
    /* x is full on every rank (built identically — no bcast needed;
     * the contract is caller-bcast, verified here by construction) */
    const int r0 = (int)(rank * (nnr / 2));
    const int r1 = (int)((rank == 0) ? (nnr / 2) : nnr);
    double* y_local = (double*)malloc((r1 - r0) * sizeof(double));
    rc = rcre_dist_spmv_rows_f64(ctx, rp, ci, val, (int)nnr, r0, r1, x,
                                 y_local);
    CHECK(rc == 0, "dist spmv rc");
    rc = rcre_dist_gather_rows_f64(ctx, (int)nnr, r0, r1, y_local, y_par);
    CHECK(rc == 0, "dist gather rc");
    if (rank == 0) {
        CHECK(memcmp(y_ser, y_par, nnr * sizeof(double)) == 0,
              "2-rank dist SpMV+gather == serial (bit-identical)");
        printf("  [2-rank dist SpMV] %s\n",
               memcmp(y_ser, y_par, nnr * sizeof(double)) == 0
                   ? "bit-identical to serial"
                   : "MISMATCH");
    }
    free(rp);
    free(ci);
    free(val);
    free(x);
    free(y_ser);
    free(y_par);
    free(y_local);

    /* ---- ordered allreduce ------------------------------------------ */
    /* rank r contributes r+1 to each element; the ordered fold (rank 0
     * first) is 1 + 2 = 3 — a pairwise tree would give the same value
     * here, but the contract is the ordered fold (docs 13.2) */
    double in[2], out[2];
    in[0] = (double)(rank + 1);
    in[1] = 100.0 * (double)(rank + 1);
    rcre_mpi_allreduce_sum_f64(ctx, in, out, 2);
    if (rank == 0) {
        CHECK(out[0] == 3.0 && out[1] == 300.0, "ordered allreduce fold");
        printf("  [ordered allreduce] %s\n",
               (out[0] == 3.0 && out[1] == 300.0) ? "fold ok" : "WRONG");
    }

    rcre_mpi_barrier(ctx);
    if (rank == 0) {
        if (g_fails) {
            printf("test_mpi2: %d/%d checks FAILED\n", g_fails, g_checks);
        } else {
            printf("test_mpi2: all %d checks passed\n", g_checks);
        }
    }
    rcre_mpi_close(ctx);
    return g_fails ? 1 : 0;
}
