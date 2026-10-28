/**
 * @file xgb_core.c
 * @brief Dataset binning + histogram-based greedy tree fitting.
 *
 * The tree constructor implements the XGBoost "hist" method in its
 * canonical form:
 *   1. each feature is pre-quantized into <= max_bins bins (quantiles of a
 *      deterministic subsample);
 *   2. for a node, one pass over its rows accumulates (G, H) into
 *      per-feature, per-bin histogram cells (missing mass is derived per
 *      feature as node-total minus binned mass);
 *   3. the best split is the (feature, bin boundary) maximizing the
 *      L1+L2-regularized structural gain, with the missing mass routed to
 *      its default direction (the larger-Hessian side) at that boundary.
 * Complexity: O(nrows * nfeats) per node, O(n * F * depth) per tree.
 */
#include "xgb_internal.h"

/* ========================================================================= */
/* Quantile binning                                                          */
/* ========================================================================= */

static int cmp_float(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

/* bin = number of edges strictly less than x (binary search) */
static inline int bin_of(const float* edges, int nedges, float x) {
    int lo = 0, hi = nedges;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (edges[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

XGBDataset* xgb_dataset_create(const float* X, const float* y,
                               size_t n, size_t d,
                               const XGBConfig* cfg,
                               RCRE_Arena* arena) {
    if (!X || !y || n == 0 || d == 0 || !arena) return NULL;
    int B = cfg ? cfg->max_bins : XGB_MAX_BINS;
    if (B < 2) B = 2;
    /* bin values are stored in uint8 with 255 reserved for the missing
     * sentinel, so at most 255 distinct bins are usable */
    if (B > XGB_MAX_BINS - 1) B = XGB_MAX_BINS - 1;

    XGBDataset* ds = (XGBDataset*)rcre_arena_alloc(arena, sizeof(*ds), 64);
    if (!ds) return NULL;
    ds->n = n; ds->d = d;
    ds->X = (float*)rcre_arena_alloc(arena, n * d * sizeof(float), 16);
    ds->y = (float*)rcre_arena_alloc(arena, n * sizeof(float), 16);
    ds->bins = (uint8_t*)rcre_arena_alloc(arena, n * d, 16);
    ds->n_bins = (uint8_t*)rcre_arena_alloc(arena, d, 4);
    ds->edges = (float*)rcre_arena_alloc(arena, d * (XGB_MAX_BINS - 1) * sizeof(float), 16);
    if (!ds->X || !ds->y || !ds->bins || !ds->n_bins || !ds->edges) return NULL;

    memcpy(ds->X, X, n * d * sizeof(float));
    memcpy(ds->y, y, n * sizeof(float));

    float* vals = (float*)rcre_arena_alloc(arena, XGB_BIN_SAMPLE * sizeof(float), 16);
    if (!vals) return NULL;

    for (size_t f = 0; f < d; f++) {
        /* 1. collect finite values (deterministic stride subsample) */
        size_t m = 0;
        size_t stride = (n > XGB_BIN_SAMPLE) ? (n + XGB_BIN_SAMPLE - 1) / XGB_BIN_SAMPLE : 1;
        for (size_t i = 0; i < n; i += stride) {
            float v = ds->X[i * d + f];
            if (v == v) vals[m++] = v;   /* NaN check */
        }
        int nedges = 0;
        if (m >= 2) {
            qsort(vals, m, sizeof(float), cmp_float);
            if (vals[m - 1] > vals[0]) { /* all-equal -> 0 edges -> 1 bin */
            for (int b = 1; b < B; b++) {
                size_t pos = (size_t)((double)b * (double)m / (double)B);
                if (pos == 0) pos = 1;
                if (pos >= m) pos = m - 1;
                float e = vals[pos];
                /* keep strictly increasing edges (dedupe constant runs) */
                if (nedges == 0 || e > ds->edges[f * (XGB_MAX_BINS - 1) + (nedges - 1)])
                    ds->edges[f * (XGB_MAX_BINS - 1) + nedges++] = e;
            }
            }
        }
        ds->n_bins[f] = (uint8_t)(nedges + 1);
        /* 2. bin every sample */
        const float* ed = &ds->edges[f * (XGB_MAX_BINS - 1)];
        for (size_t i = 0; i < n; i++) {
            float v = ds->X[i * d + f];
            ds->bins[i * d + f] = (v != v) ? XGB_MISSING_BIN
                                           : (uint8_t)bin_of(ed, nedges, v);
        }
    }
    return ds;
}

size_t       xgb_dataset_rows(const XGBDataset* ds) { return ds->n; }
size_t       xgb_dataset_dim(const XGBDataset* ds)  { return ds->d; }
const float* xgb_dataset_x(const XGBDataset* ds)    { return ds->X; }
const float* xgb_dataset_y(const XGBDataset* ds)    { return ds->y; }

/* ========================================================================= */
/* Deterministic subsampling helpers                                         */
/* ========================================================================= */

void xgb_perm_prefix(int32_t* perm, int len, int take, uint64_t* rng) {
    if (take >= len) {
        for (int i = 0; i < len; i++) perm[i] = i;
        return;
    }
    for (int i = 0; i < take; i++) {
        uint64_t r = xgb_rand64(rng);
        int j = i + (int)(r % (uint64_t)(len - i));
        perm[i] = perm[j];
        perm[j] = perm[i];
    }
}

/* ========================================================================= */
/* Histogram tree fitting                                                    */
/* ========================================================================= */

typedef struct {
    int feature;
    int bin;          /* left side = bins 0..bin */
    float gain;
    bool default_left;
} BestSplit;

/* Build the node histogram and find the best split.
 * Returns true if a positive-gain split exists. */
static bool find_best_split(FitCtx* fc, const int32_t* rows, int nrows,
                            float* G_node, float* H_node, BestSplit* best) {
    *G_node = 0.0f;
    *H_node = 0.0f;

    fc->stamp_cur++;
    if (fc->stamp_cur == 0) {   /* counter wrap: full clear (essentially never) */
        memset(fc->hist_stamp, 0, (size_t)fc->nfeats * XGB_MAX_BINS * sizeof(int32_t));
        fc->stamp_cur = 1;
    }

    const XGBDataset* ds = fc->ds;
    const uint8_t* bins = ds->bins;
    const uint8_t* nb = ds->n_bins;
    const float* g = fc->g, * h = fc->h;
    const int32_t* feats = fc->feats;
    const int B = XGB_MAX_BINS;

    for (int r = 0; r < nrows; r++) {
        int i = rows[r];
        float gi = g[i], hi = h[i];
        *G_node += gi;
        *H_node += hi;
        for (int k = 0; k < fc->nfeats; k++) {
            int f = feats[k];
            int b = bins[(size_t)i * ds->d + f];
            if (b == XGB_MISSING_BIN) continue;
            size_t kk = (size_t)k * B + b;
            if (fc->hist_stamp[kk] != fc->stamp_cur) {
                fc->hist_stamp[kk] = fc->stamp_cur;
                fc->hist_g[kk] = gi;
                fc->hist_h[kk] = hi;
            } else {
                fc->hist_g[kk] += gi;
                fc->hist_h[kk] += hi;
            }
        }
    }

    float best_gain = 0.0f;
    int best_f = -1, best_b = -1;
    bool best_def = true;
    float mcg, mch;   /* current feature missing mass */

    for (int k = 0; k < fc->nfeats; k++) {
        int f = feats[k];
        int nbf = nb[f];
        if (nbf < 2) continue;              /* constant feature: no split */
        /* missing mass for this feature = node total - sum over bins */
        mcg = *G_node; mch = *H_node;
        for (int b = 0; b < nbf; b++) {
            size_t kk = (size_t)k * B + b;
            if (fc->hist_stamp[kk] == fc->stamp_cur) {
                mcg -= fc->hist_g[kk];
                mch -= fc->hist_h[kk];
            }
        }
        if (mch < 0.0f) mch = 0.0f; /* clamp float roundoff epsilon */

        float G_non_missing = *G_node - mcg;
        float H_non_missing = *H_node - mch;
        if (H_non_missing < 0.0f) H_non_missing = 0.0f;

        float run_g = 0.0f, run_h = 0.0f;
        for (int b = 0; b < nbf - 1; b++) {
            size_t kk = (size_t)k * B + b;
            if (fc->hist_stamp[kk] == fc->stamp_cur) {
                run_g += fc->hist_g[kk];
                run_h += fc->hist_h[kk];
            }
            float G_L_b = run_g, H_L_b = run_h;
            float G_R_b = G_non_missing - run_g, H_R_b = H_non_missing - run_h;

            /* Candidate 1: Missing values route to LEFT */
            float G_L1 = G_L_b + mcg, H_L1 = H_L_b + mch;
            float G_R1 = G_R_b,       H_R1 = H_R_b;
            float gain_l = -1.0f;
            if (H_L1 >= fc->cfg->min_child_weight && H_R1 >= fc->cfg->min_child_weight) {
                gain_l = (xgb_leaf_obj(*G_node, *H_node, fc->cfg->lambda, fc->cfg->alpha)
                          - (xgb_leaf_obj(G_L1, H_L1, fc->cfg->lambda, fc->cfg->alpha)
                             + xgb_leaf_obj(G_R1, H_R1, fc->cfg->lambda, fc->cfg->alpha)))
                         - fc->cfg->gamma;
            }

            /* Candidate 2: Missing values route to RIGHT */
            float G_L2 = G_L_b,       H_L2 = H_L_b;
            float G_R2 = G_R_b + mcg, H_R2 = H_R_b + mch;
            float gain_r = -1.0f;
            if (H_L2 >= fc->cfg->min_child_weight && H_R2 >= fc->cfg->min_child_weight) {
                gain_r = (xgb_leaf_obj(*G_node, *H_node, fc->cfg->lambda, fc->cfg->alpha)
                          - (xgb_leaf_obj(G_L2, H_L2, fc->cfg->lambda, fc->cfg->alpha)
                             + xgb_leaf_obj(G_R2, H_R2, fc->cfg->lambda, fc->cfg->alpha)))
                         - fc->cfg->gamma;
            }

            /* True XGBoost: select the direction that maximizes structural gain */
            float cand_gain = (gain_l >= gain_r) ? gain_l : gain_r;
            bool cand_def_left = (gain_l >= gain_r);

            if (cand_gain > best_gain) {
                best_gain = cand_gain;
                best_f = f;
                best_b = b;
                best_def = cand_def_left;
            }
        }
    }

    if (best_f < 0) return false;
    best->feature = best_f;
    best->bin = best_b;
    best->gain = best_gain;
    best->default_left = best_def;
    return true;
}

/* Optimal leaf weight under L1+L2 regularization */
static float leaf_weight(float G, float H, const XGBConfig* cfg) {
    float q = fabsf(G) - cfg->alpha;
    if (q <= 0.0f) return 0.0f;
    float den = H + cfg->lambda;
    if (den < XGB_HESS_FLOOR) den = XGB_HESS_FLOOR;
    return (G > 0.0f ? -1.0f : 1.0f) * q / den;
}

static int fit_node(FitCtx* fc, const int32_t* rows, int nrows, int depth,
                    XGBNode* out, size_t out_cap, size_t* out_idx) {
    const XGBDataset* ds = fc->ds;
    float G_node, H_node;
    BestSplit bs;

    if (nrows < 2 || depth >= fc->cfg->max_depth || *out_idx + 3 > out_cap) {
        if (*out_idx >= out_cap) return -1;
        G_node = 0.0f; H_node = 0.0f;
        for (int r = 0; r < nrows; r++) {
            G_node += fc->g[rows[r]];
            H_node += fc->h[rows[r]];
        }
        XGBNode* nd = &out[(*out_idx)++];
        nd->feature = 0;
        nd->threshold = 0.0f;
        nd->left = nd->right = 0;
        nd->leaf_value = leaf_weight(G_node, H_node, fc->cfg);
        nd->is_leaf = true;
        nd->default_left = true;
        return 0;
    }

    if (!find_best_split(fc, rows, nrows, &G_node, &H_node, &bs)) {
        if (*out_idx >= out_cap) return -1;
        XGBNode* nd = &out[(*out_idx)++];
        nd->feature = 0;
        nd->threshold = 0.0f;
        nd->left = nd->right = 0;
        nd->leaf_value = leaf_weight(G_node, H_node, fc->cfg);
        nd->is_leaf = true;
        nd->default_left = true;
        return 0;
    }

    if (*out_idx >= out_cap) return -1;
    XGBNode* nd = &out[(*out_idx)++];
    nd->feature = (uint32_t)bs.feature;
    nd->threshold = ds->edges[(size_t)bs.feature * (XGB_MAX_BINS - 1) + bs.bin];
    nd->left = (uint32_t)(*out_idx);
    nd->is_leaf = false;
    nd->default_left = bs.default_left;

    /* partition rows (children buffers live in the per-tree arena temp) */
    int32_t* rows_l = (int32_t*)rcre_arena_alloc(fc->arena, (size_t)nrows * sizeof(int32_t), 16);
    int32_t* rows_r = (int32_t*)rcre_arena_alloc(fc->arena, (size_t)nrows * sizeof(int32_t), 16);
    if (!rows_l || !rows_r) {
        /* arena pressure: degrade to a leaf instead of a corrupt split */
        nd->is_leaf = true;
        nd->threshold = 0.0f;
        nd->leaf_value = leaf_weight(G_node, H_node, fc->cfg);
        nd->default_left = true;
        return 0;
    }
    int nl = 0, nr = 0;
    const uint8_t* col = &ds->bins[bs.feature];
    for (int r = 0; r < nrows; r++) {
        int i = rows[r];
        int b = ds->bins[(size_t)i * ds->d + bs.feature];
        bool go_left = (b == XGB_MISSING_BIN) ? bs.default_left : (b <= bs.bin);
        (void)col;
        if (go_left) rows_l[nl++] = i;
        else         rows_r[nr++] = i;
    }

    int rc = fit_node(fc, rows_l, nl, depth + 1, out, out_cap, out_idx);
    if (rc == 0) {
        nd->right = (uint32_t)(*out_idx);
        rc = fit_node(fc, rows_r, nr, depth + 1, out, out_cap, out_idx);
    }
    return rc;
}

int xgb_fit_tree(FitCtx* fc, const int32_t* rows, int nrows,
                 XGBNode* out, size_t out_cap, size_t* out_idx) {
    return fit_node(fc, rows, nrows, 0, out, out_cap, out_idx);
}
