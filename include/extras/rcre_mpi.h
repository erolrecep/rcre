/**
 * @file rcre_mpi.h
 * @brief RCRE H9 — MPI shim: a 12-function C ABI over self-registering
 *        communication backends (the D1 bridge pattern, docs/hpc.md 12).
 *
 * Design (the H6 GPU-bridge precedent, applied to collective
 * communication):
 *
 *  - The core never links a communication implementation. This header
 *    defines an opaque RCRE_MpiCtx whose vtable is resolved at open
 *    time; backends self-register from a C constructor and the
 *    Makefile links the backend objects explicitly (no weak symbols —
 *    the H5 hook lesson).
 *  - `loopback` is ALWAYS available: a one-process in-memory backend
 *    registered by mpi_ctx.c. It is the conformance oracle — the
 *    distributed ops (src/core/dist.c) run against it and must be
 *    bit-identical to the core serial operators.
 *  - `mpi` (real MPI) is an OPTIONAL backend TU (src/extras/mpi_real.c)
 *    compiled empty unless RCRE_HAVE_MPI is defined (mpicc probe,
 *    `make check-mpi`); gates that need real processes skip clean
 *    where it is not linked (the H6 CUDA pattern).
 *
 * Determinism policy: collectives on this shim are ORDERED — bcast
 * copies from the root; allreduce-sum_f64 reduces in ascending rank
 * order (a left fold). A 2-process run is therefore bit-identical to
 * the 1-process loopback run of the same operation (pairwise-MPI
 * allreduce would not be — documented in docs/hpc.md 12.2).
 */
#ifndef RCRE_EXTRAS_MPI_H
#define RCRE_EXTRAS_MPI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if !defined(RCRE_NODISCARD)
#if defined(__has_c_attribute)
#if __has_c_attribute(nodiscard)
#define RCRE_NODISCARD [[nodiscard]]
#endif
#endif
#if !defined(RCRE_NODISCARD)
#if defined(__GNUC__) || defined(__clang__)
#define RCRE_NODISCARD __attribute__((warn_unused_result))
#else
#define RCRE_NODISCARD
#endif
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** Opaque communication context (one per communicator). */
typedef struct RCRE_MpiCtx RCRE_MpiCtx;

/** Backend registration (called from backend constructors). */
typedef struct RCRE_MpiBackend {
    const char* name; /* "loopback", "mpi", ... */
    /** Open a communicator; returns NULL on failure. */
    RCRE_MpiCtx* (*open)(void);
    /** Process count. */
    int (*size)(RCRE_MpiCtx* ctx);
    /** This rank (0-based). */
    int (*rank)(RCRE_MpiCtx* ctx);
    void (*barrier)(RCRE_MpiCtx* ctx);
    /** Broadcast n doubles from `root` (ordered: plain copy). */
    void (*bcast_f64)(RCRE_MpiCtx* ctx, double* buf, size_t n, int root);
    int (*send_f64)(RCRE_MpiCtx* ctx, const double* buf, size_t n, int dest,
                    int tag);
    int (*recv_f64)(RCRE_MpiCtx* ctx, double* buf, size_t n, int src,
                    int tag);
    /** Non-blocking; returns an opaque request id (>= 0) or -1. */
    int (*isend_f64)(RCRE_MpiCtx* ctx, const double* buf, size_t n, int dest,
                     int tag);
    int (*irecv_f64)(RCRE_MpiCtx* ctx, double* buf, size_t n, int src,
                     int tag);
    int (*wait)(RCRE_MpiCtx* ctx, int req);
    /** Ordered sum reduction (ascending-rank left fold) into `out`. */
    void (*allreduce_sum_f64)(RCRE_MpiCtx* ctx, const double* in, double* out,
                              size_t n);
    void (*close)(RCRE_MpiCtx* ctx);
} RCRE_MpiBackend;

RCRE_NODISCARD bool rcre_mpi_backend_registered(const char* name);
void rcre_mpi_register_backend(const char* name, RCRE_MpiBackend* be);

/* ------------------------------------------------------------------ */
/* The 12-function public ABI                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Open a communicator on the named backend ("loopback" or "mpi").
 *        NULL tries "mpi" then "loopback". 0 on success, -1 otherwise.
 */
RCRE_NODISCARD int rcre_mpi_open(const char* backend, RCRE_MpiCtx** out);

/** 2. Close the context (frees it). */
void rcre_mpi_close(RCRE_MpiCtx* ctx);

/** 3. Number of processes in the communicator. */
RCRE_NODISCARD int rcre_mpi_size(const RCRE_MpiCtx* ctx);

/** 4. This process's rank. */
RCRE_NODISCARD int rcre_mpi_rank(const RCRE_MpiCtx* ctx);

/** 5. Blocking barrier. */
void rcre_mpi_barrier(RCRE_MpiCtx* ctx);

/** 6. Broadcast n doubles from `root`. */
void rcre_mpi_bcast_f64(RCRE_MpiCtx* ctx, double* buf, size_t n, int root);

/** 7. Blocking send of n doubles. 0 on success, -1 on bad input. */
RCRE_NODISCARD int rcre_mpi_send_f64(RCRE_MpiCtx* ctx, const double* buf,
                                     size_t n, int dest, int tag);

/** 8. Blocking receive of n doubles. 0 on success, -1 on bad input. */
RCRE_NODISCARD int rcre_mpi_recv_f64(RCRE_MpiCtx* ctx, double* buf, size_t n,
                                     int src, int tag);

/** 9. Non-blocking send; request id >= 0 or -1. */
RCRE_NODISCARD int rcre_mpi_isend_f64(RCRE_MpiCtx* ctx, const double* buf,
                                      size_t n, int dest, int tag);

/** 10. Non-blocking receive; request id >= 0 or -1. */
RCRE_NODISCARD int rcre_mpi_irecv_f64(RCRE_MpiCtx* ctx, double* buf,
                                      size_t n, int src, int tag);

/** 11. Wait on a request. 0 on success, -1 on bad input. */
RCRE_NODISCARD int rcre_mpi_wait(RCRE_MpiCtx* ctx, int req);

/** 12. Ordered sum reduction (ascending rank) of n doubles into out. */
void rcre_mpi_allreduce_sum_f64(RCRE_MpiCtx* ctx, const double* in,
                                double* out, size_t n);

/* Backend name constants. */
#define RCRE_MPI_BACKEND_LOOPBACK "loopback"
#define RCRE_MPI_BACKEND_MPI "mpi"

#ifdef __cplusplus
}
#endif

#endif /* RCRE_EXTRAS_MPI_H */
