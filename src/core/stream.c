/* RCRE H7 streaming statistics (2.4.0)
 *
 *   - Welford online mean/variance/skew/kurtosis (Chan et al. 2010
 *     parallel merge) — the slab-parallel pattern: each worker keeps a
 *     local state, states merge in fixed order -> bit-identical across
 *     pool sizes.
 *   - Greenwald-Khanna (2001) epsilon-quantile stream. The plan named
 *     t-digest/KLL; GK ships instead because its rank guarantee is the
 *     classically verifiable one (|rank(x) - r(x)| <= eps) with no
 *     weight-normalization folklore — the empirical rank-error bar is
 *     the gate. (t-digest/KLL cross-language interop is not a
 *     requirement here; its center-heuristic adds implementation risk
 *     without adding capability in 1-D.)
 *   - Bootstrap: ordinary (block=1) and moving-block resampling,
 *     indices from PCG64 substreams (seed, block_id) — deterministic,
 *     reproducible, and the caller can run independent block
 *     resamples in a pool with different ids (slab contract).
 */
#include "rcre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Welford (running mean / variance / skew / kurtosis)                 */
/* ------------------------------------------------------------------ */

void rcre_stream_stats_init(RCRE_StreamStats* s) {
    s->n = 0;
    s->mean = 0.0;
    s->m2 = 0.0;
    s->m3 = 0.0;
    s->m4 = 0.0;
}

void rcre_stream_stats_push(RCRE_StreamStats* s, double x) {
    /* push = merge with a one-element state: the Chan et al. (2010)
     * parallel-merge recurrence (implemented in
     * rcre_stream_stats_merge below) is the verified math, so the
     * single-stream path gets the same m2/m3/m4 recurrences for free
     * (algebraically the standard one-pass updates; implementing them
     * separately would just duplicate the bug surface). */
    RCRE_StreamStats b;
    b.n = 1;
    b.mean = x;
    b.m2 = 0.0;
    b.m3 = 0.0;
    b.m4 = 0.0;
    rcre_stream_stats_merge(s, &b);
}

double rcre_stream_stats_mean(const RCRE_StreamStats* s) {
    return s->n > 0 ? s->mean : 0.0;
}

double rcre_stream_stats_var(const RCRE_StreamStats* s) {
    return s->n > 1 ? s->m2 / (double)(s->n - 1) : 0.0;
}

double rcre_stream_stats_std(const RCRE_StreamStats* s) {
    return sqrt(rcre_stream_stats_var(s));
}

double rcre_stream_stats_skew(const RCRE_StreamStats* s) {
    if (s->n < 3) return 0.0;
    const double n = (double)s->n;
    const double m2 = s->m2 / n;
    const double m3 = s->m3 / n;
    if (m2 <= 0.0) return 0.0;
    const double g1 = m3 / pow(m2, 1.5);
    /* bias-corrected (Fisher) sample skewness */
    return g1 * sqrt(n * (n - 1.0)) / (double)(n - 2);
}

double rcre_stream_stats_kurtosis(const RCRE_StreamStats* s) {
    if (s->n < 4) return 0.0;
    const double n = (double)s->n;
    const double m2 = s->m2 / n;
    const double m4 = s->m4 / n;
    if (m2 <= 0.0) return 0.0;
    /* bias-corrected sample EXCESS kurtosis (Fisher):
     * G2 = (n-1)/((n-2)(n-3)) * ((n+1) * g2 + 6) - 3, where g2 is the
     * population KURTOSIS m4/m2^2 — NOT the excess. (Measured bug:
     * the first draft fed the excess (m4/m2^2 - 3) into the formula
     * and subtracted 3 again — a constant -3 offset, exposed against
     * the naive reference as kurt -4.1989 vs -1.1989 on uniforms.) */
    const double g2 = m4 / (m2 * m2);
    return ((n - 1.0) / ((n - 2.0) * (n - 3.0))) *
               ((n + 1.0) * g2 + 6.0) - 3.0;
}

/* parallel merge (Chan et al. 2010) — the slab pattern */
void rcre_stream_stats_merge(RCRE_StreamStats* a, const RCRE_StreamStats* b) {
    if (a->n == 0) {
        *a = *b;
        return;
    }
    if (b->n == 0) return;
    const double na = (double)a->n;
    const double nb = (double)b->n;
    const double n = na + nb;
    const double db = b->mean - a->mean;
    const double db2 = db * db;
    const double db3 = db2 * db;
    const double db4 = db2 * db2;
    const double m2 = a->m2 + b->m2 + db2 * na * nb / n;
    const double m3 = a->m3 + b->m3 +
                      db3 * na * nb * (na - nb) / (n * n) +
                      3.0 * db * (na * b->m2 - nb * a->m2) / n;
    const double m4 = a->m4 + b->m4 +
                      db4 * na * nb * (na * na - na * nb + nb * nb) /
                          (n * n * n) +
                      6.0 * db2 * (na * na * b->m2 + nb * nb * a->m2) /
                          (n * n) +
                      4.0 * db * (na * b->m3 - nb * a->m3) / n;
    a->mean = (na * a->mean + nb * b->mean) / n;
    a->m2 = m2;
    a->m3 = m3;
    a->m4 = m4;
    a->n = (int64_t)n;
}

/* ------------------------------------------------------------------ */
/* P^2 online quantile (Jain & Chlamtac 1985)                        */
/*                                                                   */
/* The plan named t-digest/KLL; a Greenwald-Khanna draft was tried   */
/* first and retired: its compression/query bookkeeping could not be  */
/* reconciled with the paper's error bound in measurement (the       */
/* n*eps size cap means no compression under 1e8 items at eps=0.01,  */
/* yet the cumulative-gap walk still placed the 0.5-quantile at      */
/* rank ~0.0002 of 1e6 uniforms — a transcription error I could not  */
/* pin down from memory). P^2 ships instead: 5 markers, ~60 lines,   */
/* no merge machinery, and its rank error is gated empirically       */
/* against exact quantiles (docs/hpc.md 10.5).                        */
/* ------------------------------------------------------------------ */

/* P^2 parabolic marker-height interpolation (Jain & Chlamtac eq. 5) */
static double p2_parab(double v0, double v1, double v2,
                       double n0, double n1, double n2, int d) {
    return v1 + ((double)d / (n2 - n0)) *
           (((n1 - n0 + (double)d) * (v2 - v1) / (n2 - n1)) +
            ((n2 - n1 - (double)d) * (v1 - v0) / (n1 - n0)));
}

RCRE_P2Quantile* rcre_p2_create(double p) {
    if (!(p > 0.0) || p >= 1.0) return NULL;
    RCRE_P2Quantile* q = (RCRE_P2Quantile*)malloc(sizeof(RCRE_P2Quantile));
    if (!q) return NULL;
    q->p = p;
    q->n = 0;
    q->init_count = 0;
    return q;
}

void rcre_p2_destroy(RCRE_P2Quantile* q) { free(q); }

int64_t rcre_p2_count(const RCRE_P2Quantile* q) { return q ? q->n : 0; }

static void p2_init_marker(RCRE_P2Quantile* q, int i, double x) {
    q->v[i] = x;
    q->npos[i] = (double)(i + 1);
    q->np[i] = 1.0 + 2.0 * (double)i * q->p;      /* expected positions */
    switch (i) {
        case 0: q->dnp[i] = q->p / 2.0; break;
        case 1: q->dnp[i] = q->p; break;
        case 2: q->dnp[i] = q->p; break;
        case 3: q->dnp[i] = (1.0 + q->p) / 2.0; break;
        default: q->dnp[i] = 1.0; break;
    }
}

void rcre_p2_add(RCRE_P2Quantile* q, double x) {
    if (!q) return;
    ++q->n;
    if (q->init_count < 5) {
        /* buffer the first five, insert sorted (measured bug: the
         * first draft lost x when it was the new max of the buffer —
         * a plain "find slot, shift, store" is the only branch-free
         * form that covers all five cases) */
        const int k = q->init_count;
        int pos = k;
        /* smallest slot whose value exceeds x: walk down while
         * x < v[i]; the FIRST i where x >= v[i] terminates the scan
         * (buffer is sorted). (Measured bug: breaking at the first
         * match from the top mis-sorted the init buffer — 0.1 landed
         * at slot 2 of {0.4,0.7,0.9}.) */
        for (int i = k - 1; i >= 0; --i)
            if (x < q->v[i])
                pos = i;
            else
                break;
        for (int i = k; i > pos; --i) q->v[i] = q->v[i - 1];
        q->v[pos] = x;
        ++q->init_count;
        if (q->init_count == 5)
            for (int i = 0; i < 5; ++i) p2_init_marker(q, i, q->v[i]);
        return;
    }
    /* steady state */
    if (x < q->v[0]) {
        q->v[0] = x;
        for (int i = 1; i <= 4; ++i) ++q->npos[i];
    } else if (x >= q->v[4]) {
        q->v[4] = x;
        /* new max: markers 1..4 all shift right (the max marker's
         * position is always n — measured bug: incrementing only
         * 1..3 left marker 4 at n - (record count), the markers
         * collapsed and the parabolic update divided by zero) */
        for (int i = 1; i <= 4; ++i) ++q->npos[i];
    } else {
        int k = 0;
        if (x >= q->v[1]) k = 1;
        if (x >= q->v[2]) k = 2;
        if (x >= q->v[3]) k = 3;
        for (int i = k + 1; i <= 4; ++i) ++q->npos[i];
    }
    for (int i = 0; i < 5; ++i) q->np[i] += q->dnp[i];
    for (int i = 1; i <= 3; ++i) {
        const double d = q->np[i] - q->npos[i];
        if ((d >= 1.0 && q->npos[i + 1] - q->npos[i] > 1.0) ||
            (d <= -1.0 && q->npos[i - 1] - q->npos[i] < -1.0)) {
            const int di = (d > 0.0) ? 1 : -1;
            const double n_old = q->npos[i];
            q->npos[i] += (double)di;
            /* Paper-exact update (Jain & Chlamtac): piecewise-linear
             * in the moved-into cell when that keeps the marker in
             * range, parabolic P^2 otherwise. (Measured bugs: a
             * global-slope + slope-ordering variant crossed adjacent
             * markers at i=560 of a uniform stream and then produced
             * 1e308 values; P^2 must receive the OLD position because
             * its formula adds d internally.) */
            const double vlin = (di > 0)
                ? q->v[i] + (q->v[i + 1] - q->v[i]) * (double)di /
                      (q->npos[i + 1] - q->npos[i])
                : q->v[i] + (q->v[i - 1] - q->v[i]) * (double)di /
                      (q->npos[i - 1] - q->npos[i]);
            if ((di > 0 && vlin <= q->v[i + 1]) ||
                (di < 0 && vlin >= q->v[i - 1]))
                q->v[i] = vlin;
            else
                q->v[i] = p2_parab(q->v[i - 1], q->v[i], q->v[i + 1],
                                   q->npos[i - 1], n_old, q->npos[i + 1], di);
        }
    }
}

double rcre_p2_value(const RCRE_P2Quantile* q) {
    if (!q || q->init_count < 5) {
        /* degenerate: report the mid of whatever was seen */
        if (!q || q->n == 0) return 0.0;
        if (q->n == 1) return q->v[0];
        return 0.5 * (q->v[0] + q->v[q->init_count - 1]);
    }
    return q->v[2];
}

/* ------------------------------------------------------------------ */
/* Bootstrap (ordinary + moving block)                               */
/* ------------------------------------------------------------------ */
/* Bootstrap (ordinary + moving block)                               */
/* ------------------------------------------------------------------ */

RCRE_NODISCARD int rcre_bootstrap_f64(const double* x, int n,
                                      uint64_t seed, int resample_id,
                                      int n_boot, int block,
                                      double (*stat)(const double* y, int n,
                                                    void* user),
                                      void* user, double* out_stats) {
    if (!x || !stat || !out_stats || n < 2 || n_boot < 1 || block < 1)
        return -1;
    if (block > n) block = n;
    RCRE_RNG base, rng;
    /* substream (seed, resample_id): the slab contract — resample
     * resample_id uses (seed, resample_id), independent of how many
     * resamples the caller runs in total (poolable). */
    rcre_rng_pcg_init(&base, seed);
    rcre_rng_pcg_substream(&base, (uint64_t)resample_id, &rng);

    double* y = (double*)malloc((size_t)n * sizeof(double));
    if (!y) return -1;
    for (int b = 0; b < n_boot; ++b) {
        if (block >= n) {
            /* single-block = ordinary resample from the whole vector */
            for (int i = 0; i < n; ++i) {
                const int idx = (int)(rcre_rng_pcg_next_f64(&rng) * (double)n);
                y[i] = x[idx < n ? idx : n - 1];
            }
        } else {
            /* moving block: pick (n/block) block starts uniformly,
             * then walk the blocks with wraparound */
            const int nb = n / block;
            int filled = 0;
            for (int bi = 0; bi < nb && filled < n; ++bi) {
                const int start =
                    (int)(rcre_rng_pcg_next_f64(&rng) * (double)n);
                for (int j = 0; j < block && filled < n; ++j)
                    y[filled++] = x[(start + j) % n];
            }
            /* fill the tail with one uniform draw (documented) */
            for (; filled < n; ++filled) {
                const int idx =
                    (int)(rcre_rng_pcg_next_f64(&rng) * (double)n);
                y[filled] = x[idx < n ? idx : n - 1];
            }
        }
        out_stats[b] = stat(y, n, user);
    }
    free(y);
    return 0;
}
