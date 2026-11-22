/**
 * @file e_means.c
 * @brief e-means (Bessa et al. 2026) as an RCRE ML operator.
 *
 * Implements Algorithm 1 of "e-means: outperforming k-means (in one-tenth
 * the epochs)" (arXiv 2611.09501), reference implementation:
 * faiss/contrib/e_means (base.py, e_means.py). Scope of this port
 * (see docs/exactness.md for the full contract and deviations):
 *
 *   - Euclidean, FP32 model state, CPU, single process;
 *   - the shipped reference's constants are ground truth (the paper's
 *     Table 2 reports a different fit — the code comments say so);
 *   - finalization sums in FP64 (reference uses FP32; documented
 *     improvement — the final answer is the last-epoch mean);
 *   - RNG: RCRE splitmix64 streams (reference: MT19937) — trajectories
 *     are therefore not bit-comparable in default mode; strict parity is
 *     gated at the state-machine level with injected streams.
 *
 * Threading: batches are processed in chunks (chunk*k <= 16 MB of Dmat);
 * a chunk is partitioned into up to 16 contiguous slabs, each owning a
 * disjoint Dmat/gather row range and one FP32 stats buffer; slabs run on
 * the RCRE threadpool and their partials merge in slab (row) order. The
 * partition depends only on (m, k, B) — never on the pool size — so the
 * result is bit-identical across runs and pool sizes; serial vs parallel
 * differs only in FP32 addition grouping at slab boundaries (<= 1 ULP).
 */

#include "e_means_internal.h"
#include "ml_internal.h"  /* rcre_splitmix64 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ========================================================================= */
/* Shipped reference constants (faiss/contrib/e_means/base.py).              */
/* Batch laws B = C_B * k^bk * n^bn * T^bt; step law                          */
/* alpha0 = C_a * k^ak * B^aB * n^an * T^aT clamped to [1e-4, 1].             */
/* ========================================================================= */

typedef struct EMLaw { double cb, bk, bn, bt; } EMLaw;
static const EMLaw EM_LAW_THROUGHPUT = { 0.035134,   0.134749, 0.876473, 0.813344 };
static const EMLaw EM_LAW_QUALITY    = { 0.00439175, 0.134749, 0.876473, 0.813344 };
static const double EM_STEP_LAW[5] = { 0.8244, -0.4617, 0.9003, -0.4498, -0.1057 };

/* ========================================================================= */
/* Recipe helpers                                                            */
/* ========================================================================= */

size_t em_round_pow2(double v) {
    if (v <= 1.0) return 1;
    double low  = pow(2.0, floor(log2(v)));
    double high = 2.0 * low;
    return (size_t)(v / low < high / v ? low : high);
}

double em_scheduled_alpha(double alpha0_f64, size_t t, size_t total) {
    double f = (double)(t < total ? t : total) / (double)(total > 1 ? total : 1);
    double cosine = 0.5 * (1.0 + cos(M_PI * f));
    return alpha0_f64 * (EM_ALPHA_MIN_FACTOR + (1.0 - EM_ALPHA_MIN_FACTOR) * cosine);
}

size_t em_epoch_remainder(float nepochs, size_t n) {
    /* Python round() on ((T - floor(T)) * n): exact half-to-even on the
     * double value (the reference's _epoch_rows uses built-in round). */
    double whole = floor((double)nepochs);
    double v = ((double)nepochs - whole) * (double)n;
    double f = floor(v);
    double diff = v - f;
    if (diff > 0.5) return (size_t)(f + 1.0);
    if (diff < 0.5) return (size_t)f;
    return ((size_t)f % 2 == 0) ? (size_t)f : (size_t)f + 1; /* tie -> even */
}

RCRE_NODISCARD int em_resolve_recipe(size_t n, size_t k, size_t d,
                                     const RCRE_EMMeansParams* p,
                                     RCRE_EMMeansRecipe* out) {
    if (!out || n == 0 || k == 0) return RCRE_ERR_INVALID_ARG;
    if (p && p->alpha0 != 0.0f && !isfinite((double)p->alpha0))
        return RCRE_ERR_INVALID_ARG; /* explicit NaN alpha0 (0 = law) */

    float nepochs = (p && p->nepochs > 0.0f) ? p->nepochs : 5.0f;
    if (!(nepochs > 0.0f) || nepochs > EM_MAX_T || !isfinite((double)nepochs))
        return RCRE_ERR_INVALID_ARG;

    int      law     = p ? p->batch_law : RCRE_EM_BATCH_AUTO;
    bool     b_given = (p != NULL) && p->batch_size > 0;
    bool     a_given = (p != NULL) && p->alpha0 > 0.0f;
    uint64_t seed    = (p && p->seed != 0) ? p->seed : 1234ULL;

    /* Revival threshold: -1 = auto (0.1, off for d <= 2); else fixed. */
    float tau = (p == NULL) ? -1.0f : p->unused_threshold;
    if (tau < 0.0f) tau = (d <= EM_NO_REVIVAL_MAX_D) ? 0.0f : EM_UNUSED_TAU;
    if (tau < 0.0f || !isfinite((double)tau)) return RCRE_ERR_INVALID_ARG;

    /* Batch: manual or law. The law's "auto" resolves by dimension. */
    int resolved_law;
    size_t B = 0;
    if (b_given) {
        resolved_law = -1; /* manual */
        B = p->batch_size;
    } else {
        if (law == RCRE_EM_BATCH_AUTO)
            law = (d <= EM_AUTO_QUALITY_MAX_D) ? RCRE_EM_BATCH_QUALITY
                                               : RCRE_EM_BATCH_THROUGHPUT;
        if (law != RCRE_EM_BATCH_THROUGHPUT && law != RCRE_EM_BATCH_QUALITY)
            return RCRE_ERR_INVALID_ARG;
        resolved_law = law;
        const EMLaw* L = (law == RCRE_EM_BATCH_QUALITY) ? &EM_LAW_QUALITY
                                                        : &EM_LAW_THROUGHPUT;
        double raw_b = L->cb * pow((double)k, L->bk) * pow((double)n, L->bn)
                     * pow((double)nepochs, L->bt);
        B = em_round_pow2(raw_b);
        if (law == RCRE_EM_BATCH_THROUGHPUT) {
            size_t cap = em_round_pow2((double)n / 2.0);
            if (B > cap) B = cap;
        } else {
            size_t cap = n / 2;
            if (B > cap) B = cap;
        }
    }
    if (B > n) B = n;
    if (B < 1) B = 1;

    /* Step: manual or step law at the resolved batch. */
    double a0;
    if (!a_given) {
        a0 = EM_STEP_LAW[0] * pow((double)k, EM_STEP_LAW[1])
           * pow((double)B, EM_STEP_LAW[2])
           * pow((double)n, EM_STEP_LAW[3])
           * pow((double)nepochs, EM_STEP_LAW[4]);
        if (a0 < 1e-4) a0 = 1e-4;
        if (a0 > 1.0) a0 = 1.0;
    } else {
        a0 = (double)p->alpha0;
        if (a0 <= 0.0 || !isfinite(a0)) return RCRE_ERR_INVALID_ARG;
    }

    /* Epoch rows: whole passes, rounded fractional prefix, full closing
     * pass. T < 1 yields a single partial pass. */
    size_t whole = (size_t)floor((double)nepochs);
    size_t rem   = em_epoch_remainder(nepochs, n);
    size_t ne = 0;
    if (whole == 0) {
        out->epoch_rows[0] = rem;
        ne = 1;
    } else {
        for (size_t i = 0; i + 1 < whole; i++) out->epoch_rows[ne++] = n;
        if (rem > 0) out->epoch_rows[ne++] = rem;
        out->epoch_rows[ne++] = n;
    }
    if (ne > EM_MAX_EPOCHS) return RCRE_ERR_INVALID_ARG;

    size_t steps_per_pass = (n + B - 1) / B;
    size_t total_steps = 0;
    for (size_t i = 0; i < ne; i++)
        total_steps += (out->epoch_rows[i] + B - 1) / B;

    out->batch_size       = B;
    out->alpha0           = (float)a0;
    out->alpha0_full      = a0;
    out->nepochs          = nepochs;
    out->n_epochs         = ne;
    out->steps_per_pass   = steps_per_pass;
    out->total_steps      = total_steps;
    out->unused_threshold = tau;
    out->split_epsilon    = EM_SPLIT_EPSILON;
    out->init_usage       = ldexp(1.0f, -EM_INIT_USAGE_EXP);
    out->batch_law        = resolved_law;
    out->seed             = seed;
    return RCRE_OK;
}

/* ========================================================================= */
/* Model internals                                                           */
/* ========================================================================= */

void em_state_recompute_cc(EMState* st) {
    /* Exact sequential FP32 dot (index order, mul+add, no FMA — the build
     * uses -ffp-contract=off). Bit-identical to the oracle's cc_of; do NOT
     * route through rcre_hal.dot_f32 (NEON/FMA differ at the ULP level). */
    for (size_t j = 0; j < st->k; j++) {
        const float* cj = st->c + j * st->d;
        float acc = 0.0f;
        for (size_t dd = 0; dd < st->d; dd++) acc += cj[dd] * cj[dd];
        st->cc[j] = acc;
    }
}

void em_state_shuffle_epoch(EMState* st) {
    if (!st->do_shuffle) return;
    /* Reference's _epoch_seed: (seed * 1000003 + epoch) mod 2^32. The
     * stream is splitmix64 here (not MT19937) — different order, same
     * uniformity; documented (docs/exactness.md). */
    st->rng_shuf = (st->recipe.seed * 1000003ULL + (uint64_t)st->epoch_idx)
                   % 4294967296ULL;
    uint32_t* o = st->order;
    for (size_t i = st->n - 1; i > 0; i--) {
        uint64_t r = rcre_splitmix64(&st->rng_shuf);
        size_t j = (size_t)(r % (i + 1));
        uint32_t tmp = o[i]; o[i] = o[j]; o[j] = tmp;
    }
}

/* ========================================================================= */
/* Dead-cluster revival (Appendix H.1)                                       */
/* ========================================================================= */

/* One Gumbel uniform for cluster j: injected stream (strict parity) or the
 * run's own splitmix64 stream. Consumed for ALL k per check — the
 * reference draws a k-vector per check, so the stream position must match
 * even for non-donor entries. */
static double em_gumbel_u(EMState* st, size_t j) {
    (void)j;
    if (st->inject_U != NULL) {
        if (st->inject_U_pos >= st->inject_U_len) return 0.5;
        return st->inject_U[st->inject_U_pos++];
    }
    uint64_t r = rcre_splitmix64(&st->rng_revive);
    return (double)(r >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
}

/* Max-heap of (key, cluster) for the donor selection: build in O(k),
 * pop nsplit tops. Tie rule: larger key first, then smaller cluster index
 * (the reference argsorts F64 keys; exact ties have measure zero and the
 * parity tests assert a key margin). */
typedef struct EMHeapNode { double key; size_t j; } EMHeapNode;

static void em_heap_swap(EMHeapNode* h, size_t a, size_t b) {
    EMHeapNode t = h[a]; h[a] = h[b]; h[b] = t;
}
static int em_heap_gt(const EMHeapNode* a, const EMHeapNode* b) {
    if (a->key != b->key) return a->key > b->key;
    return a->j < b->j;
}
static void em_heap_sift_down(EMHeapNode* h, size_t i, size_t sz) {
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < sz && em_heap_gt(&h[l], &h[m])) m = l;
        if (r < sz && em_heap_gt(&h[r], &h[m])) m = r;
        if (m == i) return;
        em_heap_swap(h, i, m);
        i = m;
    }
}

static void em_revive(EMState* st) {
    if (st->recipe.unused_threshold <= 0.0f) return;

    size_t* targets = (size_t*)rcre_arena_alloc(st->arena, st->k * sizeof(size_t), 8);
    size_t* donors  = (size_t*)rcre_arena_alloc(st->arena, st->k * sizeof(size_t), 8);
    if (!targets || !donors) return; /* OOM: skip revival this check */

    /* Consume this check's U vector for ALL k FIRST — the reference draws
     * the full k-vector per check regardless of whether any split happens,
     * so the stream position must stay aligned with the oracle. */
    double* U = (double*)rcre_arena_alloc(st->arena, st->k * sizeof(double), 8);
    if (!U) return;
    for (size_t j = 0; j < st->k; j++) U[j] = em_gumbel_u(st, j);

    /* threshold: sequential FP32 sum (index order), then F32 mul/div —
     * the reference's _numpy_unused_threshold in scalar FP32. */
    float total = 0.0f;
    for (size_t j = 0; j < st->k; j++) total += st->N[j];
    float thr = (total * st->recipe.unused_threshold) / (float)st->k;

    size_t ntargets = 0, ndonors = 0;
    for (size_t j = 0; j < st->k; j++) {
        if (st->N[j] < thr) {
            targets[ntargets++] = j;
        } else if (st->N[j] > 0.0f) {
            donors[ndonors++] = j;
        }
    }
    size_t nsplit = ntargets < ndonors ? ntargets : ndonors;
    if (nsplit == 0) return;

    /* Gumbel keys over the donors (reference: uniform = max(U, tiny);
     * key = log(N) - log(-log(uniform)); the log(N) term is FP32 in the
     * reference because usage is an FP32 array). */
    EMHeapNode* heap = (EMHeapNode*)rcre_arena_alloc(
        st->arena, ndonors * sizeof(EMHeapNode), 8);
    if (!heap) return;

    const double tiny = 2.2250738585072014e-308; /* DBL_MIN */
    for (size_t i = 0; i < ndonors; i++) {
        size_t dj = donors[i];
        double u = U[dj] < tiny ? tiny : U[dj];
        double ln_n = (double)logf(st->N[dj]); /* FP32 log, then promote */
        heap[i].key = ln_n - log(-log(u));
        heap[i].j = dj;
    }
    for (size_t i = ndonors / 2; i-- > 0;) em_heap_sift_down(heap, i, ndonors);

    for (size_t i = 0; i < nsplit; i++) {
        EMHeapNode top = heap[0];
        heap[0] = heap[ndonors - 1];
        ndonors--;
        em_heap_sift_down(heap, 0, ndonors);
        size_t dj = top.j;
        size_t tj = targets[i];

        /* Eq. 8: alternating perturbation, delta = 1e-4 (shipped code).
         * The child (dead slot) reads the ORIGINAL donor row, so it is
         * written first; the donor then overwrites in place. */
        const float* cd_src = st->c + dj * st->d;
        float* cd = (float*)cd_src;
        float* ct = st->c + tj * st->d;
        for (size_t dd = 0; dd < st->d; dd++) {
            float off = EM_SPLIT_EPSILON * (((dd & 1u) == 0u) ? 1.0f : -1.0f);
            ct[dd] = cd_src[dd] * (1.0f + off);
        }
        for (size_t dd = 0; dd < st->d; dd++) {
            float off = EM_SPLIT_EPSILON * (((dd & 1u) == 0u) ? 1.0f : -1.0f);
            cd[dd] = cd_src[dd] * (1.0f - off);
        }

        /* _separate_splits: bit-identical children (zero centroids etc.)
         * get a nextafter nudge on coordinate 0. */
        bool collapsed = true;
        for (size_t dd = 0; dd < st->d; dd++)
            if (cd[dd] != ct[dd]) { collapsed = false; break; }
        if (collapsed) {
            float v = ct[0];
            float nb = nextafterf(v, INFINITY);
            if (!isfinite(nb)) nb = nextafterf(v, -INFINITY);
            ct[0] = nb;
        }

        /* Donor mass halves; each child gets half. */
        float half = st->N[dj] / 2.0f;
        st->N[dj] = half;
        st->N[tj] = half;
        st->n_splits++;
    }
}

/* ========================================================================= */
/* EMA update (reference _apply_statistics, ids = NULL path)                 */
/* ========================================================================= */

RCRE_NODISCARD int em_state_update(EMState* st, const float* sums,
                                   const float* counts, double objective) {
    double alpha_d = em_scheduled_alpha(st->alpha0_f64, st->update_idx,
                                        st->recipe.total_steps);
    float alpha = (float)alpha_d;
    float decay = (float)(1.0 - alpha_d);
    st->objective += objective;
    if (st->last_epoch) st->obj_last += objective;

    /* Last-epoch finalization accumulation (FP64 — documented improvement
     * over the reference's FP32 pass sums). */
    if (st->last_epoch) {
        for (size_t j = 0; j < st->k; j++)
            st->pass_counts[j] += (double)counts[j];
        for (size_t j = 0; j < st->k; j++) {
            const float* srow = sums + j * st->d;
            double* prow = st->pass_sums + j * st->d;
            for (size_t dd = 0; dd < st->d; dd++)
                prow[dd] += (double)srow[dd];
        }
    }

    /* EMA in the reference's FP32 operation order. */
    bool replace = (alpha_d == 1.0); /* reference: alpha == 1 corner */
    if (!replace)
        for (size_t j = 0; j < st->k; j++)
            st->N[j] = st->N[j] * decay;

    for (size_t j = 0; j < st->k; j++) {
        float n_j = counts[j];
        if (n_j <= 0.0f) continue;
        float t = alpha * n_j;
        /* Reference: retained = 0 when replacing (alpha == 1), so
         * mass = n_j exactly there, not N_j + n_j. */
        float mass = (replace ? 0.0f : st->N[j]) + t;
        float weight = t / mass;
        float* cj = st->c + j * st->d;
        const float* sj = sums + j * st->d;
        for (size_t dd = 0; dd < st->d; dd++) {
            float mean = sj[dd] / n_j;
            float pos = cj[dd] + weight * (mean - cj[dd]);
            if (weight == 1.0f) pos = mean; /* exact-override (alpha == 1) */
            cj[dd] = pos;
        }
        st->N[j] = mass;
    }

    /* Revival check cadence: after the 5th, 10th, ... update. */
    if ((st->update_idx + 1) % EM_CHECK_EVERY == 0) em_revive(st);

    em_state_recompute_cc(st);
    st->update_idx++;
    return RCRE_OK;
}

RCRE_NODISCARD int em_state_finalize(EMState* st) {
    for (size_t j = 0; j < st->k; j++) {
        if (st->pass_counts[j] <= 0.0) continue; /* keep position */
        float cnt = (float)st->pass_counts[j];
        const double* ps = st->pass_sums + j * st->d;
        float* cj = st->c + j * st->d;
        for (size_t dd = 0; dd < st->d; dd++)
            cj[dd] = (float)ps[dd] / cnt;
    }
    return RCRE_OK;
}

/* ========================================================================= */
/* State init                                                                */
/* ========================================================================= */

static int em_uint32_cmp(const void* a, const void* b) {
    uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    return (x > y) - (x < y);
}

RCRE_NODISCARD int em_state_init(EMState* st, const RCRE_Tensor* X, size_t k,
                                 const RCRE_EMMeansParams* p,
                                 RCRE_Arena* arena, RCRE_ThreadPool* pool) {
    memset(st, 0, sizeof(*st));
    if (!X || !X->data || !arena) return RCRE_ERR_INVALID_ARG;
    if (X->ndim != 2 || X->dtype != RCRE_DTYPE_F32) return RCRE_ERR_INVALID_ARG;
    size_t n_total = X->shape[0], d = X->shape[1];
    if (n_total == 0 || d == 0 || k == 0 || k > n_total || n_total >= EM_MAX_N)
        return RCRE_ERR_INVALID_ARG;
    st->X = (const float*)X->data;
    st->d = d;
    st->k = k;
    st->n_total = n_total;
    st->arena = arena;
    st->pool = pool;
    st->nbuf = pool ? EM_MAX_STAT_BUFFERS : 1;
    /* Exact assignment kernel for small k*d (bit-identical to the oracle,
     * row-independent); gemm kernel above the threshold. */
    st->use_exact_dots = (k * d) <= EM_EXACT_DOTS_MAX;

    /* Recipe on the (possibly capped) active size. */
    size_t cap = p ? p->max_points_per_centroid : 0;
    size_t n = n_total;
    if (cap > 0 && n_total > cap * k) n = cap * k;
    if (em_resolve_recipe(n, k, d, p, &st->recipe) != RCRE_OK)
        return RCRE_ERR_INVALID_ARG;
    st->alpha0_f64 = st->recipe.alpha0_full;
    st->n = n;
    st->do_shuffle = (p == NULL) || p->shuffle;

    /* Active row set: the full set, or a random sorted subset when the
     * max_points_per_centroid cap binds (reference parity: choice without
     * replacement + sort). */
    st->order = (uint32_t*)rcre_arena_alloc(arena, n_total * sizeof(uint32_t), 16);
    if (!st->order) return RCRE_ERR_OUT_OF_MEMORY;
    for (size_t i = 0; i < n_total; i++) st->order[i] = (uint32_t)i;
    if (n < n_total) {
        uint64_t rs = (p && p->seed) ? p->seed : 1234ULL;
        for (size_t i = 0; i < n; i++) {
            uint64_t r = rcre_splitmix64(&rs);
            size_t j = i + (size_t)(r % (n_total - i));
            uint32_t tmp = st->order[i]; st->order[i] = st->order[j];
            st->order[j] = tmp;
        }
        qsort(st->order, n, sizeof(uint32_t), em_uint32_cmp);
    }

    /* Scratch (arena, aligned). */
    st->c    = (float*)rcre_arena_alloc(arena, k * d * sizeof(float), 64);
    st->N    = (float*)rcre_arena_alloc(arena, k * sizeof(float), 64);
    st->cc   = (float*)rcre_arena_alloc(arena, k * sizeof(float), 64);
    st->batch_s = (float*)rcre_arena_alloc(arena, k * d * sizeof(float), 64);
    st->batch_n = (float*)rcre_arena_alloc(arena, k * sizeof(float), 64);
    st->labels  = (int32_t*)rcre_arena_alloc(arena, n_total * sizeof(int32_t), 64);
    st->pass_sums   = (double*)rcre_arena_alloc(arena, k * d * sizeof(double), 64);
    st->pass_counts = (double*)rcre_arena_alloc(arena, k * sizeof(double), 64);
    if (!st->c || !st->N || !st->cc || !st->batch_s || !st->batch_n ||
        !st->labels || !st->pass_sums || !st->pass_counts)
        return RCRE_ERR_OUT_OF_MEMORY;
    memset(st->labels, 0xFF, n_total * sizeof(int32_t)); /* -1 default */

    /* Assignment geometry. Exact kernel: no Dmat, the whole batch is one
     * chunk (row-independent dots). GEMM kernel: chunk*k under the Dmat
     * budget. */
    if (st->use_exact_dots) {
        st->chunk = st->recipe.batch_size;
        st->Dmat = NULL;
        st->gather = NULL;
    } else {
        size_t chunk_cap = EM_DMAT_BUDGET / (4u * (unsigned)k);
        if (chunk_cap < 1) chunk_cap = 1;
        st->chunk = st->recipe.batch_size < chunk_cap ? st->recipe.batch_size : chunk_cap;
        if (st->chunk < 1) st->chunk = 1;
        st->Dmat   = (float*)rcre_arena_alloc(arena, st->chunk * k * sizeof(float), 64);
        st->gather = (float*)rcre_arena_alloc(arena, st->chunk * d * sizeof(float), 64);
        if (!st->Dmat || !st->gather) return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Per-slab stats buffers. */
    st->wsum = (float**)rcre_arena_alloc(arena, st->nbuf * sizeof(float*), 8);
    st->wcnt = (uint32_t*)rcre_arena_alloc(arena, st->nbuf * k * sizeof(uint32_t), 64);
    st->wobj = (double*)rcre_arena_alloc(arena, st->nbuf * sizeof(double), 8);
    if (!st->wsum || !st->wcnt || !st->wobj) return RCRE_ERR_OUT_OF_MEMORY;
    for (size_t s = 0; s < st->nbuf; s++) {
        st->wsum[s] = (float*)rcre_arena_alloc(arena, k * d * sizeof(float), 64);
        if (!st->wsum[s]) return RCRE_ERR_OUT_OF_MEMORY;
    }

    /* Model init: provided centroids, all rows (n == k), or k uniform
     * distinct rows (partial Fisher-Yates on a private copy). */
    if (p && p->init_centroids) {
        const RCRE_Tensor* ic = p->init_centroids;
        if (!ic->data || ic->ndim != 2 || ic->dtype != RCRE_DTYPE_F32 ||
            ic->shape[0] != k || ic->shape[1] != d)
            return RCRE_ERR_SHAPE_MISMATCH;
        memcpy(st->c, ic->data, k * d * sizeof(float));
    } else if (n == k) {
        for (size_t i = 0; i < k; i++)
            memcpy(st->c + i * d, st->X + (size_t)st->order[i] * d, d * sizeof(float));
    } else {
        uint32_t* tmp = (uint32_t*)rcre_arena_alloc(arena, n * sizeof(uint32_t), 16);
        if (!tmp) return RCRE_ERR_OUT_OF_MEMORY;
        memcpy(tmp, st->order, n * sizeof(uint32_t));
        st->rng_init = st->recipe.seed;
        for (size_t i = 0; i < k; i++) {
            uint64_t r = rcre_splitmix64(&st->rng_init);
            size_t j = i + (size_t)(r % (n - i));
            uint32_t t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t;
        }
        for (size_t i = 0; i < k; i++)
            memcpy(st->c + i * d, st->X + (size_t)tmp[i] * d, d * sizeof(float));
    }
    for (size_t j = 0; j < k; j++)
        st->N[j] = ldexp(1.0f, -EM_INIT_USAGE_EXP); /* 2^-16 */

    st->rng_revive = st->recipe.seed;
    em_state_recompute_cc(st);
    return RCRE_OK;
}

/* ========================================================================= */
/* Assignment + batch statistics                                             */
/* ========================================================================= */

/* One slab: gather rows, gemm against the current centroid snapshot,
 * argmin (first index on ties), scatter stats into this slab's buffers.
 * Slabs of one chunk own disjoint row ranges of gather/Dmat, so they run
 * concurrently without races. */
typedef struct EMSlabTask {
    EMState*  st;
    const uint32_t* rows; /* chunk-local row ids (into st->X) */
    size_t cm;            /* chunk rows in this batch */
    size_t block;         /* rows per slab */
} EMSlabTask;

static void em_slab_task(void* arg, size_t task_id) {
    EMSlabTask* t = (EMSlabTask*)arg;
    EMState* st = t->st;
    size_t r0 = task_id * t->block;
    size_t r1 = r0 + t->block < t->cm ? r0 + t->block : t->cm;
    if (r0 >= r1) return;
    size_t nr = r1 - r0;

    float* ws = st->wsum[task_id];
    uint32_t* wc = st->wcnt + task_id * st->k;
    double* wo = &st->wobj[task_id];
    const float* cc = st->cc;

    if (st->use_exact_dots) {
        /* EXACT kernel: per-row TRUE squared distance to every centroid,
         * sequential FP32 in index order (diff*diff, mul+add, no FMA).
         * Row-independent — slab-safe by construction, bit-identical to
         * the oracle, and free of the cc - 2*x*c cancellation on
         * large-offset data. */
        for (size_t r = 0; r < nr; r++) {
            const float* x = st->X + (size_t)t->rows[r0 + r] * st->d;
            int32_t best = 0;
            float bv = 0.0f;
            {
                const float* c0 = st->c;
                for (size_t dd = 0; dd < st->d; dd++) {
                    float df = x[dd] - c0[dd];
                    bv += df * df;
                }
            }
            for (size_t j = 1; j < st->k; j++) {
                const float* cj = st->c + j * st->d;
                float acc = 0.0f;
                for (size_t dd = 0; dd < st->d; dd++) {
                    float df = x[dd] - cj[dd];
                    acc += df * df;
                }
                if (acc < bv) { bv = acc; best = (int32_t)j; }
            }
            uint32_t rid = t->rows[r0 + r];
            if (st->capturing) st->labels[rid] = best;
            wc[best]++;
            float* srow = ws + (size_t)best * st->d;
            for (size_t dd = 0; dd < st->d; dd++) srow[dd] += x[dd];
            *wo += (double)bv; /* true distance — no cancellation */
        }
        return;
    }

    /* GEMM kernel (large k*d): Dmat = chunk x C^T via the RCRE HAL, then
     * first-argmin of (cc - 2*D). The expanded score is monotone in the
     * true distance, so labels are correct while F32 precision holds;
     * the objective is ALWAYS the true distance to the winner (one extra
     * d-dot per row) so MSE stays exact on large-offset data. */
    float* g = st->gather + r0 * st->d;
    for (size_t r = 0; r < nr; r++)
        memcpy(g + r * st->d, st->X + (size_t)t->rows[r0 + r] * st->d,
               st->d * sizeof(float));
    float* D = st->Dmat + r0 * st->k;
    rcre_hal.gemm_f32(nr, st->k, st->d, 1.0f, g, st->d, st->c, st->d,
                      0.0f, D, st->k);
    for (size_t r = 0; r < nr; r++) {
        const float* row = D + r * st->k;
        int32_t best = 0;
        float bv = cc[0] - 2.0f * row[0];
        for (size_t j = 1; j < st->k; j++) {
            float v = cc[j] - 2.0f * row[j];
            if (v < bv) { bv = v; best = (int32_t)j; }
        }
        uint32_t rid = t->rows[r0 + r];
        if (st->capturing) st->labels[rid] = best;
        wc[best]++;
        const float* x = st->X + (size_t)rid * st->d;
        float* srow = ws + (size_t)best * st->d;
        for (size_t dd = 0; dd < st->d; dd++) srow[dd] += x[dd];
        /* exact objective: true distance to the winning centroid */
        const float* cw = st->c + (size_t)best * st->d;
        float dv = 0.0f;
        for (size_t dd = 0; dd < st->d; dd++) {
            float df = x[dd] - cw[dd];
            dv += df * df;
        }
        *wo += (double)dv;
    }
}

RCRE_NODISCARD int em_state_run_batch(EMState* st, const uint32_t* rows,
                                      size_t m, bool capture) {
    memset(st->batch_s, 0, st->k * st->d * sizeof(float));
    memset(st->batch_n, 0, st->k * sizeof(float));
    double obj = 0.0;
    st->capturing = capture;

    size_t off = 0;
    while (off < m) {
        size_t cm = m - off;
        if (cm > st->chunk) cm = st->chunk;

        /* Slab partition (depends only on cm, k, B — never pool size). */
        size_t nslab = 1;
        if (st->pool) {
            nslab = (cm + EM_MIN_SLAB_ROWS - 1) / EM_MIN_SLAB_ROWS;
            if (nslab > st->nbuf) nslab = st->nbuf;
            if (cm < EM_MIN_SLAB_ROWS) nslab = 1;
        }
        size_t block = (cm + nslab - 1) / nslab;

        for (size_t s = 0; s < nslab; s++) {
            memset(st->wsum[s], 0, st->k * st->d * sizeof(float));
            memset(st->wcnt + s * st->k, 0, st->k * sizeof(uint32_t));
            st->wobj[s] = 0.0;
        }

        if (st->pool && nslab > 1) {
            EMSlabTask task = { st, rows + off, cm, block };
            rcre_threadpool_parallel_for(st->pool, nslab, em_slab_task, &task);
        } else {
            for (size_t s = 0; s < nslab; s++) {
                EMSlabTask task = { st, rows + off, cm, block };
                em_slab_task(&task, s);
            }
        }

        /* Merge slab partials in slab (row) order into the batch stats. */
        for (size_t s = 0; s < nslab; s++) {
            const uint32_t* wc = st->wcnt + s * st->k;
            const float* ws = st->wsum[s];
            for (size_t j = 0; j < st->k; j++)
                st->batch_n[j] += (float)wc[j];
            for (size_t j = 0; j < st->k; j++) {
                const float* sr = ws + j * st->d;
                float* br = st->batch_s + j * st->d;
                for (size_t dd = 0; dd < st->d; dd++)
                    br[dd] += sr[dd];
            }
            obj += st->wobj[s];
        }
        off += cm;
    }

    return em_state_update(st, st->batch_s, st->batch_n, obj);
}

RCRE_EMMeansParams rcre_ml_emeans_params_default(void) {
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = 5.0f;
    p.unused_threshold = -1.0f; /* auto (0.1, off for d <= 2) */
    p.shuffle = true;
    p.seed = 1234ULL; /* reference default */
    p.capture_labels = true;
    return p;
}

/* ========================================================================= */
/* Public API                                                                */
/* ========================================================================= */

RCRE_NODISCARD int rcre_ml_emeans_recipe(const RCRE_Tensor* X, size_t k,
                                         const RCRE_EMMeansParams* p,
                                         RCRE_EMMeansRecipe* out) {
    /* Only n and d are consumed: a shape-only tensor (data == NULL) is a
     * valid input — planning the recipe needs no data. */
    if (!X || !out || k == 0) return RCRE_ERR_INVALID_ARG;
    if (X->ndim != 2 || X->dtype != RCRE_DTYPE_F32) return RCRE_ERR_INVALID_ARG;
    size_t n_total = X->shape[0], d = X->shape[1];
    if (n_total == 0 || d == 0 || k > n_total) return RCRE_ERR_INVALID_ARG;
    size_t cap = p ? p->max_points_per_centroid : 0;
    size_t n = (cap > 0 && n_total > cap * k) ? cap * k : n_total;
    return em_resolve_recipe(n, k, d, p, out);
}

RCRE_NODISCARD int rcre_ml_emeans_fit(const RCRE_Tensor* X, size_t k, float nepochs,
                                      RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                      RCRE_Arena* arena) {
    /* The convenience API takes an explicit T: validate it (the _ex path
     * treats nepochs <= 0 as "use the default of 5"). */
    if (!(nepochs > 0.0f) || nepochs > EM_MAX_T || !isfinite((double)nepochs))
        return RCRE_ERR_INVALID_ARG;
    RCRE_EMMeansParams p;
    memset(&p, 0, sizeof(p));
    p.nepochs = nepochs;
    p.shuffle = true;
    p.unused_threshold = -1.0f; /* auto */
    p.capture_labels = true;
    return rcre_ml_emeans_fit_ex(X, k, &p, centroids, labels, NULL, NULL,
                                 arena, NULL);
}

RCRE_NODISCARD int rcre_ml_emeans_fit_ex(const RCRE_Tensor* X, size_t k,
                                         const RCRE_EMMeansParams* params,
                                         RCRE_Tensor* centroids, RCRE_Tensor* labels,
                                         float* out_inertia, int* out_n_splits,
                                         RCRE_Arena* arena,
                                         RCRE_ThreadPool* pool) {
    if (!X || !X->data || !centroids || !centroids->data || !arena)
        return RCRE_ERR_INVALID_ARG;
    if (X->ndim != 2 || X->dtype != RCRE_DTYPE_F32 ||
        centroids->ndim != 2 || centroids->dtype != RCRE_DTYPE_F32 ||
        centroids->shape[1] != X->shape[1])
        return RCRE_ERR_SHAPE_MISMATCH;
    if (labels && labels->data && labels->shape[0] < X->shape[0])
        return RCRE_ERR_SHAPE_MISMATCH;

    EMState st;
    int rc = em_state_init(&st, X, k, params, arena, pool);
    if (rc != RCRE_OK) return rc;
    if (centroids->shape[0] != k) return RCRE_ERR_SHAPE_MISMATCH;

    bool capture = (params == NULL) || params->capture_labels;
    const uint32_t* base = st.order;

    for (size_t e = 0; e < st.recipe.n_epochs; e++) {
        size_t rows = st.recipe.epoch_rows[e];
        st.epoch_idx = e;
        st.epoch_cursor = 0;
        st.last_epoch = (e == st.recipe.n_epochs - 1);
        if (st.last_epoch) {
            memset(st.pass_sums, 0, st.k * st.d * sizeof(double));
            memset(st.pass_counts, 0, st.k * sizeof(double));
            st.obj_last = 0.0;
            st.obj_last_rows = rows;
        }
        em_state_shuffle_epoch(&st);
        size_t cursor = 0;
        while (cursor < rows) {
            size_t m = st.recipe.batch_size;
            if (m > rows - cursor) m = rows - cursor;
            rc = em_state_run_batch(&st, base + cursor, m,
                                    st.last_epoch && capture);
            if (rc != RCRE_OK) return rc;
            cursor += m;
        }
    }

    rc = em_state_finalize(&st);
    if (rc != RCRE_OK) return rc;

    memcpy(centroids->data, st.c, k * st.d * sizeof(float));

    if (labels && labels->data) {
        size_t n_total = X->shape[0];
        if (labels->dtype == RCRE_DTYPE_I32) {
            int32_t* l = (int32_t*)labels->data;
            for (size_t i = 0; i < n_total; i++) l[i] = st.labels[i];
        } else if (labels->dtype == RCRE_DTYPE_F32) {
            float* l = (float*)labels->data;
            for (size_t i = 0; i < n_total; i++) l[i] = (float)st.labels[i];
        }
    }

    if (out_inertia) {
        *out_inertia = (st.obj_last_rows > 0)
            ? (float)(st.obj_last / (double)st.obj_last_rows) : 0.0f;
    }
    if (out_n_splits) *out_n_splits = st.n_splits;
    return RCRE_OK;
}

/* ========================================================================= */
/* Exact assignment pass (post-training)                                     */
/* ========================================================================= */

RCRE_NODISCARD int rcre_ml_emeans_assign(const RCRE_Tensor* X,
                                         const RCRE_Tensor* centroids,
                                         RCRE_Tensor* labels, float* out_mse,
                                         RCRE_Arena* arena) {
    if (!X || !X->data || !centroids || !centroids->data || !labels ||
        !labels->data || !arena)
        return RCRE_ERR_INVALID_ARG;
    if (X->ndim != 2 || X->dtype != RCRE_DTYPE_F32 ||
        centroids->ndim != 2 || centroids->dtype != RCRE_DTYPE_F32 ||
        labels->ndim != 2 || X->shape[1] != centroids->shape[1])
        return RCRE_ERR_SHAPE_MISMATCH;
    size_t n = X->shape[0], d = X->shape[1], k = centroids->shape[0];
    if (n == 0 || d == 0 || k == 0) return RCRE_ERR_INVALID_ARG;

    const float* Xd = (const float*)X->data;
    const float* Cd = (const float*)centroids->data;

    float* cc = (float*)rcre_arena_alloc(arena, k * sizeof(float), 64);
    if (!cc) return RCRE_ERR_OUT_OF_MEMORY;
    /* Exact sequential FP32 dots (index order, mul+add, no FMA) — same
     * contract as the training exact kernel. */
    for (size_t j = 0; j < k; j++) {
        const float* cj = Cd + j * d;
        float acc = 0.0f;
        for (size_t dd = 0; dd < d; dd++) acc += cj[dd] * cj[dd];
        cc[j] = acc;
    }

    bool exact = (k * d) <= EM_EXACT_DOTS_MAX;
    float* D = NULL, *g = NULL;
    size_t chunk = n;
    if (!exact) {
        chunk = EM_DMAT_BUDGET / (4u * (unsigned)k);
        if (chunk < 1) chunk = 1;
        if (chunk > n) chunk = n;
        D = (float*)rcre_arena_alloc(arena, chunk * k * sizeof(float), 64);
        g = (float*)rcre_arena_alloc(arena, chunk * d * sizeof(float), 64);
        if (!D || !g) return RCRE_ERR_OUT_OF_MEMORY;
    }

    double mse = 0.0;
    size_t off = 0;
    while (off < n) {
        size_t cm = n - off;
        if (cm > chunk) cm = chunk;
        if (!exact) {
            for (size_t r = 0; r < cm; r++)
                memcpy(g + r * d, Xd + (off + r) * d, d * sizeof(float));
            rcre_hal.gemm_f32(cm, k, d, 1.0f, g, d, Cd, d, 0.0f, D, k);
        }
        for (size_t r = 0; r < cm; r++) {
            int32_t best = 0;
            float bv;
            if (exact) {
                /* true squared distance, sequential FP32 (no cancellation) */
                const float* x = Xd + (off + r) * d;
                float acc = 0.0f;
                for (size_t dd = 0; dd < d; dd++) {
                    float df = x[dd] - Cd[dd];
                    acc += df * df;
                }
                bv = acc;
                for (size_t j = 1; j < k; j++) {
                    const float* cj = Cd + j * d;
                    float a2 = 0.0f;
                    for (size_t dd = 0; dd < d; dd++) {
                        float df = x[dd] - cj[dd];
                        a2 += df * df;
                    }
                    if (a2 < bv) { bv = a2; best = (int32_t)j; }
                }
            } else {
                const float* row = D + r * k;
                bv = cc[0] - 2.0f * row[0];
                for (size_t j = 1; j < k; j++) {
                    float v = cc[j] - 2.0f * row[j];
                    if (v < bv) { bv = v; best = (int32_t)j; }
                }
                /* exact MSE: true distance to the winner */
                const float* x = Xd + (off + r) * d;
                const float* cw = Cd + (size_t)best * d;
                float dv = 0.0f;
                for (size_t dd = 0; dd < d; dd++) {
                    float df = x[dd] - cw[dd];
                    dv += df * df;
                }
                bv = dv;
            }
            mse += (double)bv;
            if (labels->dtype == RCRE_DTYPE_I32)
                ((int32_t*)labels->data)[off + r] = best;
            else
                ((float*)labels->data)[off + r] = (float)best;
        }
        off += cm;
    }
    if (out_mse) *out_mse = (float)(mse / (double)n);
    return RCRE_OK;
}
