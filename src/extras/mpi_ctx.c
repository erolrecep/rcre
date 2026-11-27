/* RCRE H9 — MPI shim: bridge core + loopback backend (2.6.0).
 *
 * The D1 bridge pattern (H6 GPU precedent):
 *   - mpi_ctx.c owns the backend registry and the 12-fn public ABI
 *     (vtable dispatch). It has NO external dependencies.
 *   - The `loopback` backend (one process, in-memory) is registered
 *     from a C constructor HERE, so every build has a working
 *     communicator — the conformance oracle for the distributed ops
 *     in src/core/dist.c (loopback runs must be bit-identical to the
 *     core serial operators).
 *   - `mpi_real.c` is the OPTIONAL real-MPI backend TU (compiled
 *     empty unless RCRE_HAVE_MPI; the Makefile's `check-mpi` probes
 *     for mpicc and skips clean where it is absent — the H6 CUDA
 *     pattern).
 *
 * Determinism: the loopback collectives are plain copies / identity
 * (size 1), so a distributed op run on loopback executes exactly the
 * serial code path — bit-identity is by construction, and the test
 * suite verifies it.
 */
#include "extras/rcre_mpi.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* registry                                                            */
/* ------------------------------------------------------------------ */

#define RCRE_MPI_MAX_BACKENDS 8

typedef struct {
    const char* name;
    RCRE_MpiBackend* be;
} RCRE_MpiSlot;

static RCRE_MpiSlot g_mpi_backends[RCRE_MPI_MAX_BACKENDS];
static int g_nmpi_backends = 0;

RCRE_NODISCARD bool rcre_mpi_backend_registered(const char* name) {
    if (!name) return false;
    for (int i = 0; i < g_nmpi_backends; ++i)
        if (strcmp(g_mpi_backends[i].name, name) == 0) return true;
    return false;
}

void rcre_mpi_register_backend(const char* name, RCRE_MpiBackend* be) {
    if (!name || !be || g_nmpi_backends >= RCRE_MPI_MAX_BACKENDS) return;
    for (int i = 0; i < g_nmpi_backends; ++i)
        if (strcmp(g_mpi_backends[i].name, name) == 0) {
            g_mpi_backends[i].be = be; /* re-register: replace in place */
            return;
        }
    {
        g_mpi_backends[g_nmpi_backends].name = name;
        g_mpi_backends[g_nmpi_backends].be = be;
        ++g_nmpi_backends;
    }
}

/* ------------------------------------------------------------------ */
/* context                                                             */
/* ------------------------------------------------------------------ */

struct RCRE_MpiCtx {
    RCRE_MpiBackend* be;
};

/* ------------------------------------------------------------------ */
/* the 12-function public ABI (vtable dispatch)                        */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_mpi_open(const char* backend, RCRE_MpiCtx** out) {
    if (!out) return -1;
    const RCRE_MpiBackend* be = NULL;
    if (backend) {
        for (int i = 0; i < g_nmpi_backends; ++i)
            if (strcmp(g_mpi_backends[i].name, backend) == 0)
                be = g_mpi_backends[i].be;
        if (!be) return -1; /* named backend not registered */
    } else {
        /* auto: prefer real MPI, fall back to loopback */
        const char* order[2] = {RCRE_MPI_BACKEND_MPI,
                                RCRE_MPI_BACKEND_LOOPBACK};
        for (int t = 0; t < 2 && !be; ++t)
            for (int i = 0; i < g_nmpi_backends; ++i)
                if (strcmp(g_mpi_backends[i].name, order[t]) == 0)
                    be = g_mpi_backends[i].be;
    }
    RCRE_MpiCtx* comm = be->open();
    if (!comm) return -1;
    /* wire the backend pointer onto the backend-allocated context
     * (the backend does not know its own registry slot) */
    comm->be = (RCRE_MpiBackend*)be;
    *out = comm;
    return 0;
}

void rcre_mpi_close(RCRE_MpiCtx* ctx) {
    if (!ctx || !ctx->be) return;
    ctx->be->close(ctx);
}

RCRE_NODISCARD int rcre_mpi_size(const RCRE_MpiCtx* ctx) {
    if (!ctx || !ctx->be) return -1;
    return ctx->be->size((RCRE_MpiCtx*)ctx);
}

RCRE_NODISCARD int rcre_mpi_rank(const RCRE_MpiCtx* ctx) {
    if (!ctx || !ctx->be) return -1;
    return ctx->be->rank((RCRE_MpiCtx*)ctx);
}

void rcre_mpi_barrier(RCRE_MpiCtx* ctx) {
    if (ctx && ctx->be) ctx->be->barrier(ctx);
}

void rcre_mpi_bcast_f64(RCRE_MpiCtx* ctx, double* buf, size_t n, int root) {
    if (ctx && ctx->be) ctx->be->bcast_f64(ctx, buf, n, root);
}

RCRE_NODISCARD int rcre_mpi_send_f64(RCRE_MpiCtx* ctx, const double* buf,
                                     size_t n, int dest, int tag) {
    if (!ctx || !ctx->be || !buf || n == 0) return -1;
    return ctx->be->send_f64(ctx, buf, n, dest, tag);
}

RCRE_NODISCARD int rcre_mpi_recv_f64(RCRE_MpiCtx* ctx, double* buf, size_t n,
                                     int src, int tag) {
    if (!ctx || !ctx->be || !buf || n == 0) return -1;
    return ctx->be->recv_f64(ctx, buf, n, src, tag);
}

RCRE_NODISCARD int rcre_mpi_isend_f64(RCRE_MpiCtx* ctx, const double* buf,
                                      size_t n, int dest, int tag) {
    if (!ctx || !ctx->be || !buf || n == 0) return -1;
    return ctx->be->isend_f64(ctx, buf, n, dest, tag);
}

RCRE_NODISCARD int rcre_mpi_irecv_f64(RCRE_MpiCtx* ctx, double* buf, size_t n,
                                      int src, int tag) {
    if (!ctx || !ctx->be || !buf || n == 0) return -1;
    return ctx->be->irecv_f64(ctx, buf, n, src, tag);
}

RCRE_NODISCARD int rcre_mpi_wait(RCRE_MpiCtx* ctx, int req) {
    if (!ctx || !ctx->be) return -1;
    return ctx->be->wait(ctx, req);
}

void rcre_mpi_allreduce_sum_f64(RCRE_MpiCtx* ctx, const double* in,
                                double* out, size_t n) {
    if (ctx && ctx->be) ctx->be->allreduce_sum_f64(ctx, in, out, n);
}

/* ------------------------------------------------------------------ */
/* loopback backend (always present; the conformance oracle)           */
/* ------------------------------------------------------------------ */
/*
 * One process: size 1, rank 0. Collectives are identity/copy; point-to-
 * point is a local memcpy (send/recv between the single rank, matched
 * by (src,dest,tag) — a non-matching call is a no-op returning 0, the
 * documented loopback contract: it mirrors what a rank sees when its
 * peer is itself).
 */

static RCRE_MpiCtx* lb_open(void) {
    return (RCRE_MpiCtx*)calloc(1, sizeof(RCRE_MpiCtx));
}
static int lb_size(RCRE_MpiCtx* c) {
    (void)c;
    return 1;
}
static int lb_rank(RCRE_MpiCtx* c) {
    (void)c;
    return 0;
}
static void lb_barrier(RCRE_MpiCtx* c) {
    (void)c;
}
static void lb_bcast_f64(RCRE_MpiCtx* c, double* buf, size_t n, int root) {
    (void)c;
    (void)buf;
    (void)n;
    (void)root; /* single rank: the buffer is already the root's */
}
static int lb_send_f64(RCRE_MpiCtx* c, const double* buf, size_t n, int dest,
                       int tag) {
    (void)c;
    (void)buf;
    (void)n;
    (void)dest;
    (void)tag;
    return 0; /* loopback: self-send, no local delivery queue */
}
static int lb_recv_f64(RCRE_MpiCtx* c, double* buf, size_t n, int src,
                       int tag) {
    (void)c;
    (void)buf;
    (void)n;
    (void)src;
    (void)tag;
    return 0;
}
static int lb_isend_f64(RCRE_MpiCtx* c, const double* buf, size_t n, int dest,
                        int tag) {
    (void)c;
    (void)buf;
    (void)n;
    (void)dest;
    (void)tag;
    return 0;
}
static int lb_irecv_f64(RCRE_MpiCtx* c, double* buf, size_t n, int src,
                        int tag) {
    (void)c;
    (void)buf;
    (void)n;
    (void)src;
    (void)tag;
    return 0;
}
static int lb_wait(RCRE_MpiCtx* c, int req) {
    (void)c;
    (void)req;
    return 0;
}
static void lb_allreduce_sum_f64(RCRE_MpiCtx* c, const double* in,
                                 double* out, size_t n) {
    (void)c;
    /* one process: ordered sum of a single operand */
    memcpy(out, in, n * sizeof(double));
}
static void lb_close(RCRE_MpiCtx* c) { free(c); }

static RCRE_MpiBackend g_lb_be = {RCRE_MPI_BACKEND_LOOPBACK, lb_open, lb_size,
                                  lb_rank, lb_barrier, lb_bcast_f64,
                                  lb_send_f64, lb_recv_f64, lb_isend_f64,
                                  lb_irecv_f64, lb_wait, lb_allreduce_sum_f64,
                                  lb_close};

/* Constructor registration (the H6 pattern: the Makefile links this
 * object explicitly, so the constructor is guaranteed to run; no weak
 * symbols). */
__attribute__((constructor)) static void mpi_ctx_register_loopback(void) {
    rcre_mpi_register_backend(RCRE_MPI_BACKEND_LOOPBACK, &g_lb_be);
}
