/**
 * @file e_means_internal.h
 * @brief e-means internal state machine (test-visible).
 *
 * The three-phase split (init / run-batch / update / finalize) mirrors the
 * reference's begin/accumulate/update/end_epoch/finalize surface. The
 * inject entry point is the strict-parity hook: the C state machine is
 * driven with (sums, counts) streams produced by the Python oracle
 * (ref/ref_emeans.py) and must reproduce its FP32 (c_j, N_j) trajectory
 * bit-for-bit (0 ULP). The full fit path produces its own statistics via
 * assignment instead of injection.
 *
 * Documented operation-order contract (docs/exactness.md):
 *  - batch sums s_j: FP32, unbuffered, rows in processing order (the
 *    full-run path groups by chunk/slab — fixed for a given (n, k, B);
 *    the injected path is exactly row order);
 *  - EMA: N *= decay (all k, FP32), then per touched j in index order:
 *    t = alpha*n, mass = N + t, mean = s/n, w = t/mass,
 *    c += w*(mean - c), c = mean when w == 1 (all FP32, no FMA);
 *  - alpha(t) computed in double (Eq. 5), cast to FP32 once at use;
 *    decay = (float)(1.0 - alpha_double);
 *  - revival threshold: total = sequential FP32 sum over j (index order),
 *    thr = (total * tau) / k in FP32; Gumbel keys: (double)logf(N_j) -
 *    log(-log(U_j)), U stream consumed for ALL k per check;
 *  - split: child written first (reads original donor), then donor;
 *    delta pattern [1,-1,1,-1,...]*1e-4; nextafter on coord 0 when
 *    donor/child collapse bit-wise;
 *  - finalization: FP64 pass sums/counts (documented improvement over the
 *    reference's FP32 pass sums), final division in FP32.
 */

#ifndef E_MEANS_INTERNAL_H
#define E_MEANS_INTERNAL_H

#include "rcre.h"
#include "ml_internal.h"  /* rcre_splitmix64 */

#define EM_INIT_USAGE_EXP    16               /* N_j(0) = 2^-16 (shipped reference) */
#define EM_SPLIT_EPSILON     1e-4f            /* donor-split delta (shipped code)   */
#define EM_UNUSED_TAU        0.1f             /* auto revival threshold             */
#define EM_NO_REVIVAL_MAX_D  2                /* revival auto-off at d <= 2         */
#define EM_AUTO_QUALITY_MAX_D 16              /* auto quality law at d <= 16        */
#define EM_CHECK_EVERY       5                /* revival check cadence (updates)    */
#define EM_ALPHA_MIN_FACTOR  0.1              /* cosine floor (Eq. 5)               */
#define EM_DMAT_BUDGET       (16u * 1024u * 1024u) /* chunk*k FP32 cap             */
#define EM_EXACT_DOTS_MAX    16384ul              /* k*d cap for the exact kernel   */
#define EM_MAX_STAT_BUFFERS  16               /* per-slab stats buffers             */
#define EM_MIN_SLAB_ROWS     64               /* don't split chunks below this     */
#define EM_MAX_EPOCHS        16               /* fixed recipe array                 */
#define EM_MAX_T             16.0f            /* nepochs cap (16 whole passes)      */
#define EM_MAX_N             ((size_t)1 << 32)/* uint32 permutation cap             */

/* ========================================================================= */
/* State                                                                     */
/* ========================================================================= */

typedef struct EMState {
    /* geometry */
    const float* X;
    size_t n, d, k;          /* n = active rows (subset-capped)              */
    size_t n_total;          /* full input rows (labels cover all)           */

    /* model (FP32) */
    float*  c;               /* [k][d] positions                             */
    float*  N;               /* [k]    masses                                */

    /* recipe */
    RCRE_EMMeansRecipe recipe;
    double alpha0_f64;       /* peak step in double (Eq. 5)                  */

    /* run bookkeeping */
    size_t update_idx;       /* t: 0-based global update index               */
    size_t epoch_idx;
    size_t epoch_cursor;
    bool   last_epoch;
    bool   capturing;        /* write labels during this batch               */
    bool   do_shuffle;

    /* active-row id array; per-epoch Fisher-Yates permutation of it */
    uint32_t* order;

    /* finalization (FP64) */
    double* pass_sums;       /* [k][d], active on last epoch                 */
    double* pass_counts;     /* [k]                                          */

    /* batch scratch */
    float*   cc;             /* [k] centroid norm-squares (per snapshot)     */
    float*   Dmat;           /* [chunk][k] gemm output (NULL: exact kernel)  */
    float*   gather;         /* [chunk][d] gathered rows (NULL: exact kernel) */
    size_t   chunk;          /* assignment chunk rows (adaptive)             */
    bool     use_exact_dots; /* exact per-row FP32 dots (k*d <= cap)         */
    float*   batch_s;        /* [k][d] FP32 batch sums                       */
    float*   batch_n;        /* [k]    FP32 batch counts                     */
    int32_t* labels;         /* [n_total] last-epoch labels (-1 default)     */

    /* per-slab stats (parallel path; serial uses slab 0) */
    float**    wsum;         /* [nbuf][k*d]                                  */
    uint32_t*  wcnt;         /* [nbuf][k]                                    */
    double*    wobj;         /* [nbuf] slab objectives                       */
    size_t     nbuf;         /* allocated slab buffers (16 or 1)             */

    /* RNG streams */
    uint64_t rng_shuf;       /* per-epoch Fisher-Yates                       */
    uint64_t rng_revive;     /* revival Gumbel U stream                      */
    uint64_t rng_init;       /* random-row init                              */

    /* reporting */
    double objective;        /* moving-pass F64 objective (informational)    */
    double obj_last;         /* last-epoch moving-pass objective             */
    size_t obj_last_rows;    /* rows of the last epoch                       */
    int    n_splits;         /* total revival splits (observability)         */

    /* injected revival U stream (strict parity; NULL = own RNG) */
    const double* inject_U;
    size_t        inject_U_pos;
    size_t        inject_U_len;

    /* parallel */
    RCRE_ThreadPool* pool;

    /* scratch owner (internal allocations: revival scratch, subset sort) */
    RCRE_Arena* arena;
} EMState;

/* ========================================================================= */
/* Recipe helpers (unit-tested directly)                                     */
/* ========================================================================= */

/** Round a positive double to the nearest power of two (ties -> higher). */
size_t em_round_pow2(double v);

/** Alpha schedule (Eq. 5): peak alpha0_f64, cosine to 0.1*alpha0. t 0-based. */
double em_scheduled_alpha(double alpha0_f64, size_t t, size_t total);

/** Fractional-epoch row count, reference ties-to-even rounding. */
size_t em_epoch_remainder(float nepochs, size_t n);

/** Resolve the full recipe from problem size + params. Mirrors the
 *  reference's _resolve_recipe (shipped-code constants). */
RCRE_NODISCARD int em_resolve_recipe(size_t n, size_t k, size_t d,
                                     const RCRE_EMMeansParams* params,
                                     RCRE_EMMeansRecipe* out);

/* ========================================================================= */
/* State machine                                                             */
/* ========================================================================= */

RCRE_NODISCARD int em_state_init(EMState* st, const RCRE_Tensor* X, size_t k,
                                 const RCRE_EMMeansParams* params,
                                 RCRE_Arena* arena, RCRE_ThreadPool* pool);

/** Apply one batch's statistics with the EMA update, finalization
 *  accumulation and revival check. `objective` is the batch's F64 sum of
 *  assigned distances (informational). Mirrors the reference's update().
 *  Strict-parity entry point (injected streams never touch the model). */
RCRE_NODISCARD int em_state_update(EMState* st, const float* sums,
                                   const float* counts, double objective);

/** Run one full batch of `m` row ids against the current centroid snapshot:
 *  chunked gemm + argmin, slab stats, merge, update. Used by the fit loop. */
RCRE_NODISCARD int em_state_run_batch(EMState* st, const uint32_t* rows,
                                      size_t m, bool capture);

/** Fisher-Yates permutation of the epoch's order array (per-epoch seed).
 *  No-op when shuffle is disabled. */
void em_state_shuffle_epoch(EMState* st);

/** Finalization: last-epoch means for non-empty clusters. */
RCRE_NODISCARD int em_state_finalize(EMState* st);

/** Recompute cc[] = ||c_j||^2 (call after any centroid move). */
void em_state_recompute_cc(EMState* st);

#endif /* E_MEANS_INTERNAL_H */
