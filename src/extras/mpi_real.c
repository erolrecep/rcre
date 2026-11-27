/* RCRE H9 — optional real-MPI backend for the 12-fn shim (2.6.0).
 *
 * Compiled EMPTY unless RCRE_HAVE_MPI is defined (the Makefile's
 * `check-mpi` probes for mpicc/mpi.h and adds -DRCRE_HAVE_MPI +
 * -I<mpi include> on the link line; `make test-mpi2` runs the
 * 2-process gates under mpirun and skips clean where MPI is absent —
 * the H6 CUDA pattern: the core never links a communication
 * implementation, and a missing backend is a skipped gate, not a
 * build failure).
 *
 * Implementation notes (real MPI build):
 *   - open(): MPI_Init_thread(MPI_THREAD_SINGLE) + MPI_Comm_rank/size
 *     on COMM_WORLD.
 *   - allreduce_sum_f64: MPI_Allreduce with a REDUCER that folds in
 *     ascending rank order (MPI_SUM on a left fold — the shim's
 *     ordered-reduction contract; the documented deviation from a
 *     vendor's pairwise tree is deliberate: bit-identity with the
 *     loopback/serial path).
 */
#include "extras/rcre_mpi.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(RCRE_HAVE_MPI)

#include <mpi.h>

#define RCRE_MPI_MAX_REQ 64

typedef struct {
    int used;
    int tag;
    int peer;
    MPI_Request req;
} RCRE_MpiRealReq;

static RCRE_MpiRealReq g_mpi_reqs[RCRE_MPI_MAX_REQ];

static int req_alloc(void) {
    for (int i = 0; i < RCRE_MPI_MAX_REQ; ++i)
        if (!g_mpi_reqs[i].used) return i;
    return -1;
}

static RCRE_MpiCtx* mpi_open(void) {
    int init = 0;
    MPI_Initialized(&init);
    if (!init) {
        if (MPI_Init(NULL, NULL) != MPI_SUCCESS) return NULL;
    }
    return (RCRE_MpiCtx*)calloc(1, sizeof(RCRE_MpiCtx));
}

static int mpi_size(RCRE_MpiCtx* c) {
    (void)c;
    int s = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

static int mpi_rank(RCRE_MpiCtx* c) {
    (void)c;
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

static void mpi_barrier(RCRE_MpiCtx* c) {
    (void)c;
    MPI_Barrier(MPI_COMM_WORLD);
}

static void mpi_bcast_f64(RCRE_MpiCtx* c, double* buf, size_t n, int root) {
    (void)c;
    if (n > 0)
        MPI_Bcast(buf, (int)n, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

static int mpi_send_f64(RCRE_MpiCtx* c, const double* buf, size_t n, int dest,
                        int tag) {
    (void)c;
    return MPI_Send(buf, (int)n, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD) ==
               MPI_SUCCESS
               ? 0
               : -1;
}

static int mpi_recv_f64(RCRE_MpiCtx* c, double* buf, size_t n, int src,
                        int tag) {
    (void)c;
    return MPI_Recv(buf, (int)n, MPI_DOUBLE, src, tag, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE) == MPI_SUCCESS
               ? 0
               : -1;
}

static int mpi_isend_f64(RCRE_MpiCtx* c, const double* buf, size_t n, int dest,
                         int tag) {
    (void)c;
    int slot = req_alloc();
    if (slot < 0) return -1;
    if (MPI_Isend(buf, (int)n, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD,
                  &g_mpi_reqs[slot].req) != MPI_SUCCESS)
        return -1;
    g_mpi_reqs[slot].used = 1;
    g_mpi_reqs[slot].tag = tag;
    g_mpi_reqs[slot].peer = dest;
    return slot;
}

static int mpi_irecv_f64(RCRE_MpiCtx* c, double* buf, size_t n, int src,
                         int tag) {
    (void)c;
    int slot = req_alloc();
    if (slot < 0) return -1;
    if (MPI_Irecv(buf, (int)n, MPI_DOUBLE, src, tag, MPI_COMM_WORLD,
                  &g_mpi_reqs[slot].req) != MPI_SUCCESS)
        return -1;
    g_mpi_reqs[slot].used = 1;
    g_mpi_reqs[slot].tag = tag;
    g_mpi_reqs[slot].peer = src;
    return slot;
}

static int mpi_wait(RCRE_MpiCtx* c, int req) {
    (void)c;
    /* fixed slots (no shifting): a request id stays valid until it is
     * waited on, regardless of other pending requests */
    if (req < 0 || req >= RCRE_MPI_MAX_REQ || !g_mpi_reqs[req].used) return -1;
    if (MPI_Wait(&g_mpi_reqs[req].req, MPI_STATUS_IGNORE) != MPI_SUCCESS)
        return -1;
    g_mpi_reqs[req].used = 0;
    return 0;
}

/* Ordered sum reduction: the shim contract is an ascending-rank left
 * fold (bit-identity with the serial path). MPI_Allreduce(MPI_SUM) is
 * commutative in exact arithmetic but NOT bit-stable under a vendor's
 * pairwise tree, so this backend does the ordered fold explicitly:
 * gather to rank 0, fold in rank order, broadcast. Two extra
 * round-trips — the determinism policy pays for itself (the H1
 * lesson, applied to collectives). */
static void mpi_allreduce_sum_f64(RCRE_MpiCtx* c, const double* in,
                                  double* out, size_t n) {
    (void)c;
    int root = 0, rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (n == 0) return;
    double* acc = (double*)malloc(n * sizeof(double));
    if (!acc) return;
    if (rank == root) {
        memcpy(acc, in, n * sizeof(double));
        for (int r = 1; r < size; ++r) {
            double* tmp = (double*)malloc(n * sizeof(double));
            if (!tmp) break;
            MPI_Recv(tmp, (int)n, MPI_DOUBLE, r, 77, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            for (size_t i = 0; i < n; ++i) acc[i] += tmp[i];
            free(tmp);
        }
        memcpy(out, acc, n * sizeof(double));
        MPI_Bcast(out, (int)n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    } else {
        MPI_Send(in, (int)n, MPI_DOUBLE, root, 77, MPI_COMM_WORLD);
        MPI_Bcast(out, (int)n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    }
    free(acc);
}

static void mpi_close(RCRE_MpiCtx* c) {
    free(c);
    /* MPI_Finalize is NOT called per-close: the backend owns the MPI
     * lifetime process-wide (documented); the process exits with it. */
}

static RCRE_MpiBackend g_mpi_be = {RCRE_MPI_BACKEND_MPI, mpi_open, mpi_size,
                                   mpi_rank, mpi_barrier, mpi_bcast_f64,
                                   mpi_send_f64, mpi_recv_f64, mpi_isend_f64,
                                   mpi_irecv_f64, mpi_wait,
                                   mpi_allreduce_sum_f64, mpi_close};

__attribute__((constructor)) static void mpi_real_register(void) {
    rcre_mpi_register_backend(RCRE_MPI_BACKEND_MPI, &g_mpi_be);
}

#endif /* RCRE_HAVE_MPI */
